"""Compare exact outer-test patients on original-unit MoCA predictions.

EEG-only reproduces the legacy CBraMod segment mean, train-only scaling and
RidgeCV on all outer-train patients. M1–M3 retain their inner validation split
and validation-selected checkpoints. Matching the test set does not equalize
training recipes or establish the provenance of an imported historical CSV.
"""
import argparse
import copy
from itertools import combinations, product
import json
from pathlib import Path
import sys
import time
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from medley.features import load_features
from medley.metrics import regression_metrics
from medley.tasks import eeg_model, output
from multimodal.config import load_multimodal_config
from multimodal.data_adapter import MedleyDataAdapter
from multimodal.experiment_cohort import validate_predictions
from multimodal.experiment_evaluation import (_variant_signature, load_completed_run,
                                              load_legacy_predictions, load_series)
from multimodal.experiment_protocol import ExperimentProtocol, fingerprint, select_inner_validation
from multimodal.run_state import file_hash, read_json, write_json

MODELS = ("EEG-only", "M1", "M2", "M3")
FOLDS = range(1, 6)
METRICS = ("MAE", "RMSE", "Spearman")
APP = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = APP / "outputs" / "multimodal_comparisons"


def baseline(args, protocol, cohort, splits, directory):
    cfg = protocol.config.medley
    legacy = args.eeg_predictions
    identity = {
        "implementation": "legacy_segment_mean_ridge_v1",
        "source_sha256": file_hash(legacy) if legacy else None,
        "outer_sha256": protocol.outer.sha256,
        "cohort": cohort.cohort_fingerprint, "cache": cohort.cache_fingerprint,
        "encoder": cfg["encoder"], "eeg_alphas": cfg["evaluation"]["eeg_alphas"],
        "max_segments": cfg["evaluation"].get("max_segments"),
    }
    directory = directory / "eeg_baseline" / fingerprint(identity)
    need_features = not legacy and any(
        not (directory / f"seed{seed}_fold{fold}.csv").is_file() for seed, fold in splits
    )
    if legacy:
        series = load_legacy_predictions(legacy, label="EEG-only", model="eeg")
        if series.sources != {str(legacy.resolve()): identity["source_sha256"]}:
            raise ValueError("Historical EEG prediction file changed during ingestion")
        archived = series.frame
    elif need_features:
        features, _, subjects, missing = load_features(cfg, list(cohort.subjects))
        if missing or set(subjects) != set(cohort.subjects):
            raise ValueError("EEG feature cohort differs from the fixed split")
        indexes = {subject: index for index, subject in enumerate(subjects)}
        references = {row.subject: row for row in cohort.metadata}

    frames = []
    for (seed, fold), split in splits.items():
        path = directory / f"seed{seed}_fold{fold}.csv"
        if legacy:
            frame = archived.loc[(archived.seed == seed) & (archived.fold == fold)].copy()
        elif path.is_file():
            record = read_json(path.with_suffix(".json"))
            if record != {"baseline_identity": fingerprint(identity), "sha256": file_hash(path)}:
                raise ValueError("Saved EEG prediction checksum or identity mismatch")
            frame = pd.read_csv(path, dtype={"subject": str}, float_precision="round_trip")
        else:
            # RidgeCV selects regularization within the outer training set.
            train = [indexes[s] for s in split.train_subjects]
            test = [indexes[s] for s in split.test_subjects]
            y_train = [references[s].target for s in split.train_subjects]
            y_test = [references[s].target for s in split.test_subjects]
            predictions = output(cfg, eeg_model(cfg).fit(features[train], y_train), features[test])
            frame = pd.DataFrame({
                "seed": seed, "fold": fold, "subject": split.test_subjects, "model": "eeg",
                "subgroup": [references[s].subgroup for s in split.test_subjects],
                "y": y_test, "target": y_test, "yhat": predictions,
            })
            validate_predictions(frame, cohort, split, model="eeg")
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            with TemporaryDirectory(dir=directory) as temporary:
                staged = Path(temporary) / path.name
                frame.to_csv(staged, index=False)
                record = {"baseline_identity": fingerprint(identity), "sha256": file_hash(staged)}
                # Publish the digest first; a crash before the CSV simply retries the fold.
                write_json(path.with_suffix(".json"), record)
                staged.replace(path)
            print(f"EEG-only seed {seed} fold {fold} complete", flush=True)
        validate_predictions(frame, cohort, split, model="eeg")
        frames.append(frame.assign(model="EEG-only"))
    return pd.concat(frames, ignore_index=True), identity


class MissingFolds(ValueError):
    pass


def completed_paths(root, splits, *, full_epochs=False, models=MODELS[1:]):
    requested = set(product(models, splits))
    found = {}
    roots = (root / "multimodal_runs_30epochs", root / "multimodal_runs") if full_epochs else (root,)
    for directory in roots:
        candidates = {}
        for marker in directory.glob("*/executions/*/completed.json"):
            record = read_json(marker)
            key = (record["model"], (record["seed"], record["fold"]))
            if key not in requested or key in found:
                continue
            if full_epochs:
                history = read_json(marker.parent / record["attempt"] / "history.json")
                if len(history) != 30:
                    continue
            if key in candidates:
                raise ValueError(f"Multiple completed runs for {key}")
            candidates[key] = marker
        found.update(candidates)
    missing = sorted(requested - found.keys())
    if missing:
        raise MissingFolds(f"Missing completed model/seed/fold entries: {missing}")
    return found


def multimodal_predictions(paths, protocol, cohort, splits, *, full_epochs=False):
    frames, provenance = [], {}
    for model in MODELS[1:]:
        selected = [paths[model, key] for key in splits]
        if full_epochs:
            pieces = [load_completed_run(path, label=model) for path in selected]
            runs = {key: run for piece in pieces for key, run in piece.runs.items()}
            signatures = set()
            for run in runs.values():
                comparable = copy.deepcopy(run)
                training = comparable["manifest"]["resolved_config"]["settings"]["training"]
                if training["max_epochs"] != 30:
                    raise ValueError("Full-budget comparisons require max_epochs=30")
                # Patience does not change optimizer updates in a run that reached epoch 30.
                training["early_stopping_patience"] = 30
                signatures.add(_variant_signature(comparable))
            if len(signatures) != 1:
                raise ValueError(f"{model} has incompatible configurations across folds")
            frame = pd.concat([piece.frame for piece in pieces], ignore_index=True)
        else:
            series = load_series(selected, label=model)
            frame, runs = series.frame, series.runs
        provenance[model] = {}
        for key, run in runs.items():
            identity = run["manifest"]["identity"]
            for field, expected in (
                ("outer_sha256", protocol.outer.sha256),
                ("cohort_fingerprint", cohort.cohort_fingerprint),
                ("cache_fingerprint", cohort.cache_fingerprint),
                ("clinical_fingerprint", cohort.clinical_fingerprint),
            ):
                if identity[field] != expected:
                    raise ValueError(f"{model} seed/fold {key} has incompatible {field}")
            partition = select_inner_validation(
                splits[key], fraction=protocol.config.settings["training"]["validation_fraction"],
            )
            if run["manifest"]["subjects"] != {
                "train": list(partition.train_subjects),
                "validation": list(partition.validation_subjects),
                "test": list(partition.test_subjects),
            }:
                raise ValueError(f"{model} seed/fold {key} has a different inner partition")
            validate_predictions(frame, cohort, splits[key], model=run["completion"]["prediction_model"])
            provenance[model][f"{key[0]}/{key[1]}"] = run["completion"]["execution_id"]
        frames.append(frame.assign(model=model))
    return frames, provenance


def aligned_oof(frame, seeds, assignments, *, models=MODELS):
    """Require exact model/seed/fold/patient coverage and one target per patient."""
    expected = pd.MultiIndex.from_tuples(
        [(model, seed, fold, subject) for model in models
         for (seed, fold), patients in assignments.items() for subject in patients],
        names=["model", "seed", "fold", "subject"],
    )
    observed = pd.MultiIndex.from_frame(frame[list(expected.names)])
    if not observed.is_unique or not expected.difference(observed).empty or not observed.difference(expected).empty:
        raise ValueError("OOF coverage differs from the requested complete fixed splits")
    if set(frame.seed) != set(seeds) or frame.groupby("subject").target.nunique().ne(1).any():
        raise ValueError("OOF seed or patient target mismatch")
    return frame.sort_values(["model", "seed", "fold", "subject"])


def summarize(frame, seeds, *, models=MODELS):
    groups = [(f"seed_{seed}", frame[frame.seed == seed]) for seed in seeds]
    averaged = None
    if len(seeds) > 1:
        averaged = frame.groupby(["model", "subject"], as_index=False).agg(target=("target", "first"),
                                                                         yhat=("yhat", "mean"))
        groups.append(("seed_mean", averaged))
    metrics, paired = [], []
    for scope, group in groups:
        wide = group.pivot(index="subject", columns="model", values="yhat")
        y = group.groupby("subject").target.first().reindex(wide.index)
        scores = {model: regression_metrics(y, wide[model]) for model in models}
        metrics.extend({"scope": scope, "model": model, "n": len(y),
                        **{key: scores[model][key] for key in METRICS}} for model in models)
        paired.extend(
            {"scope": scope, "reference": left, "candidate": right, "n": len(y),
             **{f"delta_{key}": scores[right][key] - scores[left][key] for key in METRICS}}
            for left, right in combinations(models, 2)
        )
    return pd.DataFrame(metrics), pd.DataFrame(paired), averaged


def figures(paired, directory, primary_scope):
    from plot_multimodal import plt, save

    chosen = paired[paired.scope == primary_scope]
    labels = chosen.candidate + " vs " + chosen.reference
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for axis, metric in zip(axes, METRICS):
        axis.barh(labels, chosen[f"delta_{metric}"], color="#246884")
        axis.axvline(0, color="0.4", linewidth=1)
        axis.set_xlabel(f"Δ {metric}")
    fig.suptitle(f"Paired patient OOF · {primary_scope} · candidate minus reference")
    save(fig, directory, "paired_comparison")


def run(args):
    seeds = [0] if args.seeds == "0" else list(range(10))
    config = load_multimodal_config(
        APP / "configs/multimodal/clinical_only.json",
        ['clinical_text.sex_code_mapping={"0":"female","1":"male"}'],
    )
    protocol = ExperimentProtocol(config, MedleyDataAdapter(config))
    cohort = protocol.validate_cohort()
    if cohort.cache_spec_status != "validated":
        raise ValueError("Expected a verified CBraMod feature cache specification")
    splits = {(seed, fold): protocol.get_outer_split(seed=seed, fold=fold)
              for seed, fold in product(seeds, FOLDS)}
    paths = None
    if args.phase == "evaluate":
        results_root = APP / "outputs"
        if not args.full_epochs:
            results_root /= "multimodal_runs"
        deadline = time.monotonic() + args.wait_minutes * 60
        last_missing = None
        while True:
            try:
                paths = completed_paths(results_root, splits, full_epochs=args.full_epochs)
                break
            except MissingFolds as exc:
                if time.monotonic() >= deadline:
                    raise
                if str(exc) != last_missing:
                    print(str(exc), flush=True)
                    last_missing = str(exc)
                time.sleep(30)
    root = PRIVATE_ROOT
    models = MODELS
    if args.controlled_gpu:
        from moca_controlled_baselines import MODELS as CONTROLLED_MODELS, controlled_baselines
        controlled, baseline_identity = controlled_baselines(protocol, cohort, splits, root,
                                                              device=args.baseline_device, workers=args.baseline_workers)
        baselines = [controlled]
        models = (*CONTROLLED_MODELS, *MODELS[1:])
    else:
        eeg, baseline_identity = baseline(args, protocol, cohort, splits, root)
        baselines = [eeg]
        if args.all_baselines:
            from moca_legacy_baselines import LABELS, legacy_baselines
            legacy, legacy_identity = legacy_baselines(protocol, cohort, splits, root)
            baselines.append(legacy)
            baseline_identity = {"EEG-only": baseline_identity, "legacy_c0_c1": legacy_identity}
            models = ("EEG-only", *LABELS.values(), *MODELS[1:])
    multimodal, provenance = multimodal_predictions(paths, protocol, cohort, splits, full_epochs=args.full_epochs) if paths is not None else (None, None)
    if protocol.validate_cohort() != cohort:
        raise ValueError("Comparison data or feature cache changed during validation")
    if paths is None:
        return
    frame = aligned_oof(pd.concat([*baselines, *multimodal], ignore_index=True), seeds,
                        {key: split.test_subjects for key, split in splits.items()}, models=models)
    metrics, paired, averaged = summarize(frame, seeds, models=models)
    identity = {"seeds": seeds, "folds": list(FOLDS), "baseline": baseline_identity,
                "multimodal_executions": provenance}
    scope = f"seeds_{args.seeds}"
    if args.full_epochs:
        identity["full_epochs"] = 30
        scope += "_30epochs"
    if args.all_baselines:
        scope += "_all_models"
        identity["models"] = list(models)
    if args.controlled_gpu:
        scope += "_controlled_gpu"
    target = root / f"{scope}_{fingerprint(identity)[:16]}"
    if target.exists():
        expected = {"oof_predictions.csv.gz", "metrics.csv", "paired_metrics.csv", "provenance.json",
                    "paired_comparison.png", "paired_comparison.svg"}
        if averaged is not None:
            expected.add("seed_mean_predictions.csv.gz")
        if args.all_baselines:
            expected.update({"model_oof", "training_curves", "protocol_summary.md", "model_performance.png", "model_performance.svg"})
        if (not all((target / file).exists() for file in expected)
                or read_json(target / "provenance.json") != identity):
            raise ValueError("Incomplete or incompatible comparison output")
        print(target)
        return
    with TemporaryDirectory(dir=root) as temporary:
        staging = Path(temporary) / "comparison"
        staging.mkdir(mode=0o700)
        frame.to_csv(staging / "oof_predictions.csv.gz", index=False)
        metrics.to_csv(staging / "metrics.csv", index=False)
        paired.to_csv(staging / "paired_metrics.csv", index=False)
        if averaged is not None:
            averaged.to_csv(staging / "seed_mean_predictions.csv.gz", index=False)
        write_json(staging / "provenance.json", identity)
        primary = "seed_mean" if averaged is not None else "seed_0"
        figures(paired[paired.reference == "EEG-only"] if args.all_baselines else paired, staging, primary)
        if args.all_baselines:
            from moca_pipeline_artifacts import publish
            publish(staging, frame, metrics, paths, primary, controlled_gpu=args.controlled_gpu)
        staging.rename(target)
    print(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("baseline", "evaluate"))
    parser.add_argument("--seeds", choices=("0", "0-9"), default="0")
    parser.add_argument("--eeg-predictions", type=Path, help="Existing legacy EEG-only predictions.csv.gz")
    parser.add_argument("--all-baselines", action="store_true")
    parser.add_argument("--controlled-gpu", action="store_true")
    parser.add_argument("--baseline-device", default="cuda:0")
    parser.add_argument("--baseline-workers", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--full-epochs", action="store_true", help="Require 30 completed epochs in every Fold")
    parser.add_argument("--wait-minutes", type=int, default=0, help="Wait for missing Fold completions")
    args = parser.parse_args()
    if args.controlled_gpu:
        if args.eeg_predictions or not args.full_epochs:
            parser.error("--controlled-gpu requires --full-epochs and fresh controlled baseline provenance")
        args.all_baselines = True
    if args.wait_minutes < 0:
        parser.error("--wait-minutes must be nonnegative")
    try:
        # Legacy import also needs a private comparison output root.
        PRIVATE_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
        run(args)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"OOF comparison failed: {exc}\n")


if __name__ == "__main__":
    main()
