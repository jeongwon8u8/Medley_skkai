"""Run and report the preregistered Seed 0 M3 generalization ablations."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shlex
import signal
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event
import time

from compare_multimodal_oof import APP, regression_metrics
from multimodal.config import load_multimodal_config
from multimodal.data_adapter import MedleyDataAdapter
from multimodal.experiment_cohort import validate_predictions
from multimodal.experiment_evaluation import load_completed_run
from multimodal.experiment_protocol import ExperimentProtocol, fingerprint
from multimodal.experiment_runner import plan_experiments
from multimodal.run_state import read_json, write_json
from multimodal.training_resume import continuation_config

ARMS = {
    "A_fixed_attention4": ("M3", "sha256_rank_v1", "attention", 4, False),
    "B_stratified_attention4": ("M3", "sha256_moca_bands_v1", "attention", 4, False),
    "C_fixed_attention1": ("M3", "sha256_rank_v1", "attention", 1, True),
    "D_fixed_mean_soft": ("M3", "sha256_rank_v1", "mean", 1, True),
    "E_fixed_mean_late": ("M2", "sha256_rank_v1", "mean", 1, True),
}
CONTRASTS = ((next(iter(ARMS)), "B_stratified_attention4", "validation"),
             ("C_fixed_attention1", "D_fixed_mean_soft", "pooling"),
             ("D_fixed_mean_soft", "E_fixed_mean_late", "fusion"))
OUT = APP / "outputs" / "multimodal_ablation_m3_seed0"
FOLDS = range(1, 6)
ANCHOR = "A_fixed_attention4"
FOCUSED_ARMS = ("D_fixed_mean_soft", "E_fixed_mean_late")
FOCUSED_FOLDS = (4, 5)
FOCUSED_BUDGET_SECONDS = 120 * 60


def profile(name):
    if name == "de45":
        return FOCUSED_ARMS, FOCUSED_FOLDS, (ANCHOR,), (CONTRASTS[-1],), 44
    return tuple(ARMS), FOLDS, (), CONTRASTS, 113


def planned(arm, fold):
    model, inner, pooling, k, fixed = ARMS[arm]
    name = "fusion_b" if model == "M3" else "fusion_a"
    cfg = load_multimodal_config(APP / "configs" / "multimodal" / f"{name}.json", [
        "llm.model_id=" + os.environ.get("MEDLEY_LLM_PATH", "/workspace/models/Qwen3-4B"),
        "llm.revision=" + os.environ.get("MEDLEY_LLM_REVISION", "1cfa9a7208912126459214e8b04321603b3df60c"),
        "llm.device=" + os.environ.get("MEDLEY_LLM_DEVICE", "cuda:0"),
        'clinical_text.sex_code_mapping={"0":"female","1":"male"}',
        "training.max_epochs=30", "training.early_stopping_patience=30",
        f"ablation.enabled={str(fixed).lower()}",
        f"protocol.inner_policy={inner}", f"eeg.pooling.type={pooling}", f"eeg.pooling.num_queries={k}",
    ])
    if fixed:
        cfg.settings["eeg"]["pooling"]["fixed_projection"] = True
    return plan_experiments(cfg, models=[model], seeds=[0], folds=[fold], output_dir=OUT / arm)[0]


def check(series, plan, prepared):
    run = series.runs[0, plan.fold]
    manifest = run["manifest"]
    identity = manifest["identity"]
    if (continuation_config(manifest["resolved_config"]) != continuation_config(plan.config.to_dict())
            or identity["outer_sha256"] != prepared.outer_sha256):
        raise ValueError("Completed Fold differs from the requested scientific configuration")
    for key in ("cohort_fingerprint", "cache_fingerprint", "clinical_fingerprint"):
        if identity[key] != getattr(prepared.cohort, key):
            raise ValueError("Completed Fold has different patient or feature inputs")
    if manifest["subjects"] != {
        "train": list(prepared.partition.train_subjects),
        "validation": list(prepared.partition.validation_subjects),
        "test": list(prepared.partition.test_subjects),
    }:
        raise ValueError("Completed Fold has different patient partitions")
    frame = series.frame
    validate_predictions(frame, prepared.cohort, prepared.outer, model=run["completion"]["prediction_model"])
    return run


def select_completed(arm, plan, prepared):
    roots = [Path(plan.output_root)]
    if arm == "A_fixed_attention4":
        roots.append(APP / "outputs" / "multimodal_runs_30epochs")
    for root in roots:
        matches = []
        for marker in root.glob("*/executions/*/completed.json"):
            record = read_json(marker)
            if (record["model"], record["seed"], record["fold"]) != (plan.model, plan.seed, plan.fold):
                continue
            manifest = read_json(marker.parents[2] / "manifest.json")
            if continuation_config(manifest["resolved_config"]) != continuation_config(plan.config.to_dict()):
                continue
            attempt = marker.parent / record["attempt"]
            if len(read_json(attempt / "history.json")) != 30:
                continue
            series = load_completed_run(marker, label=arm)
            check(series, plan, prepared)
            matches.append(series)
        if len(matches) > 1:
            raise ValueError(f"{arm} fold {plan.fold} has multiple compatible completed runs")
        if matches:
            return matches[0]
    return None


def job(arm, fold, *, run=False):
    plan = planned(arm, fold)
    protocol = ExperimentProtocol(plan.config, MedleyDataAdapter(plan.config))
    prepared = protocol.prepare(seed=0, fold=fold)
    if prepared.cohort.n != 113 or prepared.cohort.cache_spec_status != "validated":
        raise ValueError("Expected the pinned 113-patient PD-EEG cohort and validated cache")
    existing = select_completed(arm, plan, prepared)
    if existing is not None or not run:
        return existing
    from multimodal.runner_execution import execute_plan

    # Concurrent launch of the same arm/Fold is rejected by RunStore ownership.
    result, = execute_plan([plan], existing="auto")
    series = load_completed_run(result.directory, label=arm)
    check(series, plan, prepared)
    return series


def report(profile_name="full"):
    import pandas as pd

    arms, folds, references, contrasts, expected_n = profile(profile_name)
    frames, identities = [], {}
    for arm in (*references, *arms):
        ids = []
        for fold in folds:
            series = job(arm, fold)
            if series is None:
                raise ValueError(f"Missing completed ablation: {arm}, fold {fold}")
            ids.append(series.runs[0, fold]["completion"]["execution_id"])
            frames.append(series.frame.assign(arm=arm))
        identities[arm] = ids
    frame = pd.concat(frames, ignore_index=True)
    canonical = set(frame.loc[frame.arm == next(iter(ARMS)), ["subject","seed","fold","y"]].itertuples(index=False, name=None))
    if len(canonical) != expected_n or frame.duplicated(["arm","subject"]).any() or any(
        set(g[["subject","seed","fold","y"]].itertuples(index=False, name=None)) != canonical
        for _, g in frame.groupby("arm")
    ):
        raise ValueError("Patient-level OOF coverage differs across ablations")
    metrics, deltas = [], []
    for (arm, fold), rows in frame.groupby(["arm", "fold"]):
        metrics.extend(subgroup_metrics(rows, arm=arm, fold=fold))
    for arm, rows in frame.groupby("arm"):
        metrics.extend(subgroup_metrics(rows, arm=arm, fold="matched_4_5" if profile_name == "de45" else "OOF"))
    table = pd.DataFrame(metrics)
    for reference, candidate, factor in contrasts:
        left = table[table.arm.eq(reference)].set_index(["fold","group"])
        right = table[table.arm.eq(candidate)].set_index(["fold","group"])
        for key in left.index:
            deltas.append({"reference":reference,"candidate":candidate,"factor":factor,
                           "fold":key[0],"group":key[1],"n":left.loc[key,"n"],
                           **{f"delta_{metric}":right.loc[key,metric]-left.loc[key,metric]
                              for metric in ("MAE","RMSE","Spearman","bias")}})
    provenance = {"design": "m3_diagnostic_de45_v1" if profile_name == "de45" else "m3_diagnostic_v1",
                  "seed": 0, "folds": list(folds), "executions": identities}
    deliverables = ["metrics.csv", "contrasts.csv", "oof_predictions.csv.gz"]
    if profile_name == "de45":
        deliverables.append("paired_predictions.csv")
    dest = OUT / ("report_" + fingerprint(provenance)[:16])
    if dest.exists():
        if (read_json(dest / "provenance.json") != provenance
                or not all((dest / name).is_file() for name in deliverables)):
            raise ValueError("Existing ablation report is incomplete or incompatible")
    else:
        with TemporaryDirectory(dir=OUT) as temporary:
            stage = Path(temporary) / "report"
            stage.mkdir(mode=0o700)
            table.to_csv(stage / "metrics.csv", index=False)
            pd.DataFrame(deltas).to_csv(stage / "contrasts.csv", index=False)
            frame.to_csv(stage / "oof_predictions.csv.gz", index=False)
            if profile_name == "de45":
                paired = frame.pivot(index=["seed", "fold", "subject", "y"], columns="arm", values="yhat").reset_index()
                d, e = FOCUSED_ARMS
                paired["delta_prediction_E_minus_D"] = paired[e] - paired[d]
                paired["delta_absolute_error_E_minus_D"] = (paired[e] - paired.y).abs() - (paired[d] - paired.y).abs()
                paired.to_csv(stage / "paired_predictions.csv", index=False)
            write_json(stage / "provenance.json", provenance)
            stage.rename(dest)
    print(dest)


def subgroup_metrics(rows, *, arm, fold):
    from medley.metrics import regression_metrics

    groups = (("all", rows), ("moca_le24", rows[rows.y <= 24]), ("moca_ge28", rows[rows.y >= 28]))
    for name, group in groups:
        values = regression_metrics(group.y, group.yhat) if len(group) else {}
        yield {"arm":arm,"fold":fold,"group":name,"n":len(group),
               **{metric:values.get(metric) for metric in ("MAE","RMSE","Spearman")},
               "bias": (group.yhat-group.y).mean() if len(group) else None}


def gpu_capacity(parallel):
    import re

    device = os.environ.get("MEDLEY_LLM_DEVICE", "cuda:0")
    match = re.fullmatch(r"cuda(?::([0-9]+))?", device)
    if not match:
        raise ValueError("This launcher requires a CUDA device")
    index = int(match[1] or 0)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        devices = visible.split(",")
        if index >= len(devices) or not devices[index] or devices[index] == "-1":
            raise ValueError("Selected CUDA device is not visible")
        selector = devices[index]
    else:
        selector = str(index)
    free = int(subprocess.check_output(
        ["nvidia-smi", "-i", selector, "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True,
    ).strip())
    # Historical resident usage was ~4.6 GiB. Budget 7 GiB per worker plus 2 GiB reserve.
    capacity = min(parallel, max(0, (free - 2048) // 7168))
    if capacity < 1:
        raise ValueError(f"GPU has only {free} MiB free; requires at least 9216 MiB")
    print(f"GPU free {free} MiB · using {capacity}/{parallel} requested workers", flush=True)
    return capacity


def launch(max_parallel, profile_name="full"):
    arms, folds, references, _, _ = profile(profile_name)
    if profile_name == "de45" and max_parallel != 1:
        raise ValueError("The de45 profile requires --max-parallel 1")
    import fcntl

    OUT.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (OUT / ".launcher.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another ablation launcher is running") from None
        for arm in references:
            for fold in folds:
                if job(arm, fold) is None:
                    raise ValueError(f"Required existing reference is absent: {arm}, fold {fold}")
        parallel = gpu_capacity(max_parallel)
        deadline = time.monotonic() + FOCUSED_BUDGET_SECONDS if profile_name == "de45" else float("inf")
        logs = OUT / "logs"
        logs.mkdir(exist_ok=True, mode=0o700)
        stopped = Event()
        exhausted = Event()

        def running():
            if time.monotonic() >= deadline:
                exhausted.set()
                stopped.set()
            return not stopped.is_set()

        def worker(arm):
            failures = []
            for fold in folds:
                if not running():
                    return failures
                cmd = [sys.executable, str(Path(__file__).resolve()), "fold", "--arm", arm, "--fold", str(fold)]
                if shutil.which("script"):
                    cmd = ["script", "-qefc", shlex.join(cmd), "/dev/null"]
                path = logs / f"{arm}_fold{fold}.log"
                print(f"{arm} fold {fold}: starting/rechecking · {path}", flush=True)
                with path.open("a") as log:
                    with subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                          start_new_session=True, env={**os.environ,"TERM":"xterm-256color"}) as child:
                        while child.poll() is None and running() and not stopped.wait(.5):
                            pass
                        if child.poll() is None:
                            child.terminate()
                            try:
                                child.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                child.kill()
                        status = child.wait()
                print(f"{arm} fold {fold}: {'completed/reused' if status==0 else f'failed ({status})'}", flush=True)
                if status:
                    failures.append((arm, fold))
            return failures

        with ThreadPoolExecutor(max_workers=parallel) as pool:
            try:
                failed = [item for failures in pool.map(worker, arms) for item in failures]
            except KeyboardInterrupt:
                stopped.set()
                raise
        if exhausted.is_set():
            raise ValueError("120-minute budget reached; completed runs and saved epochs can be reused")
        if failed:
            raise ValueError(f"Failed ablation folds {failed}; see {logs}")
        report(profile_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan","run","fold","report"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--fold", type=int, choices=FOLDS)
    parser.add_argument("--profile", choices=("full", "de45"), default="full")
    parser.add_argument("--max-parallel", type=int, choices=(1,2), default=1)
    args = parser.parse_args()
    try:
        if args.action == "plan":
            arms, folds, references, _, _ = profile(args.profile)
            for arm in (*references, *arms):
                for fold in folds:
                    print("reuse-only" if arm in references else "run-or-reuse", arm, planned(arm, fold).summary())
        elif args.action == "run":
            def terminate(_signum, _frame):
                raise KeyboardInterrupt
            signal.signal(signal.SIGTERM, terminate)
            launch(args.max_parallel, args.profile)
        elif args.action == "report":
            report(args.profile)
        else:
            if args.arm is None or args.fold is None:
                parser.error("fold requires --arm and --fold")
            job(args.arm, args.fold, run=True)
    except KeyboardInterrupt:
        parser.exit(130, "Ablation interrupted. Saved epochs can be resumed.\n")
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"Ablation failed ({getattr(exc, 'code', type(exc).__name__)}): {exc}\n")


if __name__ == "__main__":
    main()
