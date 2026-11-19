"""Bounded C1 -> existing trainer -> validation selection -> one test evaluation."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import copy
from pathlib import Path

from tqdm.contrib.logging import logging_redirect_tqdm

from .experiment_protocol import ExperimentProtocol, derive_seed
from .experiment_manifest import build_manifest
from .experiment_cohort import validate_predictions, audit_prediction_file
from .model_registry import RunnerError, check_readiness, create_runtime
from .run_state import RunStore, file_hash, read_json, write_json
from .training_control import train_select_restore
from .progress import progress_bar


@dataclass(frozen=True)
class ExecutionResult:
    directory: Path
    completion: dict
    metrics: dict
    reused: bool

    def public_metadata(self):
        return {"directory": str(self.directory), "execution_id": self.completion["execution_id"],
                "model": self.completion["model"], "seed": self.completion["seed"], "fold": self.completion["fold"],
                "status": self.completion["status"], "metrics": dict(self.metrics), "reused": self.reused,
                "training_resume": False}


def _loaders(adapter, prepared, config, *, seed, fold):
    import torch
    from torch.utils.data import DataLoader
    from .dataset import MedleySubjectDataset, MedleySubjectCollator
    generator = torch.Generator().manual_seed(derive_seed(seed, fold,
        purpose=f"dataloader:{config.settings['clinical_regression']['random_seed']}"))
    def loader(ids, shuffle=False):
        return DataLoader(MedleySubjectDataset(adapter, ids), batch_size=config.settings["training"]["batch_size"],
                          shuffle=shuffle, generator=generator, num_workers=0, drop_last=False,
                          collate_fn=MedleySubjectCollator())
    partition = prepared.partition
    return (loader(partition.train_subjects, config.settings["runner"]["shuffle_train"]),
            loader(partition.validation_subjects), loader(partition.test_subjects))


def _assert_inputs(protocol, expected):
    actual = protocol.validate_cohort()
    if (actual.cohort_fingerprint != expected.cohort_fingerprint or actual.cache_fingerprint != expected.cache_fingerprint
            or actual.clinical_fingerprint != expected.clinical_fingerprint):
        raise RunnerError("inputs_changed", "Cohort/targets/cache/clinical inputs changed during execution")


def _execute_one(plan, *, existing, adapter_factory, backbone_factory):
    from .experiment_runner import plan_experiments
    # Reject mutable/tampered plan fields, changed source, or changed working-directory semantics.
    expected = plan_experiments(plan.config, models=[plan.model], seeds=[plan.seed], folds=[plan.fold])[0]
    if expected != plan:
        raise RunnerError("plan_changed", "Planned configuration/source identity changed before execution")
    config = copy.deepcopy(plan.config)
    if plan.model != "M0" and backbone_factory is None and existing == "fail":
        check_readiness(config)
    with progress_bar("Data preparation", total=None, unit="stage", leave=False):
        if adapter_factory is None:
            from .data_adapter import MedleyDataAdapter
            adapter = MedleyDataAdapter(config)
        else:
            adapter = adapter_factory(copy.deepcopy(config))
        protocol = ExperimentProtocol(config, adapter)
        prepared = protocol.prepare(seed=plan.seed, fold=plan.fold)
        baseline_path = config.settings["runner"]["baseline_path"]
        baseline_model = config.settings["runner"]["baseline_model"]
        baseline_sha = None
        if plan.model == "M0":
            if not baseline_path or not baseline_model:
                raise RunnerError("baseline_source", "M0 requires runner.baseline_path and runner.baseline_model")
            baseline_sha = file_hash(baseline_path)
        manifest = build_manifest(protocol, seed=plan.seed, fold=plan.fold, model=plan.model,
                                  baseline_identifier=baseline_model if plan.model == "M0" else None)
        if manifest.to_dict()["identity"]["cohort_fingerprint"] != prepared.cohort.cohort_fingerprint:
            raise RunnerError("inputs_changed", "Protocol inputs changed between preparation and manifest capture")
    store = RunStore(manifest, baseline_sha256=baseline_sha)
    with store.claim(existing=existing) as cached:
        if cached is not None:
            import pandas as pd
            attempt = store.root / cached["attempt"]
            frame = pd.read_csv(attempt / "predictions.csv.gz", dtype={"subject": str}, keep_default_na=False)
            frame["subgroup"] = frame["subgroup"].replace("", None)
            validate_predictions(frame, prepared.cohort, prepared.outer, model=cached["prediction_model"],
                                 target_atol=config.settings["protocol"]["target_atol"])
            return ExecutionResult(store.root, cached, read_json(attempt / "metrics.json"), True)
        if plan.model == "M0":
            import pandas as pd
            from .clinical_trainer import ClinicalEvaluation
            from .clinical_checkpoint import export_predictions, PREDICTION_COLUMNS
            from medley.metrics import regression_metrics
            audit = audit_prediction_file(baseline_path, prepared.cohort, prepared.outer, model=baseline_model,
                                           target_atol=config.settings["protocol"]["target_atol"])
            frame = pd.read_csv(baseline_path, dtype={"subject": str}, keep_default_na=False)
            frame["subgroup"] = frame["subgroup"].replace("", None)
            if audit["source_sha256"] != baseline_sha or file_hash(baseline_path) != baseline_sha:
                raise RunnerError("baseline_changed", "Baseline prediction bytes changed during import")
            frame = frame.loc[(frame.model == baseline_model) & (frame.seed == plan.seed) & (frame.fold == plan.fold)]
            frame = frame.set_index("subject").loc[list(prepared.partition.test_subjects)].reset_index()
            frame = frame.loc[:, list(PREDICTION_COLUMNS)]
            all_metrics = regression_metrics(frame.y.to_numpy(), frame.yhat.to_numpy())
            metrics = {key: all_metrics[key] for key in ("n", "MAE", "RMSE")}
            export_predictions(ClinicalEvaluation(metrics, frame), store.attempt / "predictions.csv.gz")
            write_json(store.attempt / "baseline_audit.json", audit, exclusive=True)
            prediction_model, status = baseline_model, "imported"
        else:
            from .clinical_trainer import seed_training
            seed_training(derive_seed(plan.seed, plan.fold,
                purpose=f"training:{config.settings['clinical_regression']['random_seed']}"))
            with progress_bar("Model initialization", total=None, unit="stage", leave=False):
                runtime = create_runtime(plan.model, config, prepared, backbone_factory=backbone_factory)
            with progress_bar("Data loader setup", total=None, unit="stage", leave=False):
                train, validation, test = _loaders(adapter, prepared, config, seed=plan.seed, fold=plan.fold)
            write_json(store.attempt / "runtime_config.json", runtime.trainer.config.to_dict(), exclusive=True)
            outcome = train_select_restore(runtime, train, validation, store.attempt)
            write_json(store.attempt / "history.json", outcome["history"], exclusive=True)
            write_json(store.attempt / "selection.json", {"execution_id": store.execution_id,
                "c1_run_id": manifest.run_id, "model": plan.model, "seed": plan.seed, "fold": plan.fold,
                "best_epoch": outcome["best_epoch"], "validation_MAE": outcome["validation_MAE"],
                "checkpoint": str(outcome["checkpoint"].relative_to(store.attempt)),
                "restoration": "verified_exact_trainable_weights", "test_used_for_selection": False}, exclusive=True)
            _assert_inputs(protocol, prepared.cohort)
            # This is the only outer-test evaluate invocation in the training lifecycle.
            result = runtime.trainer.evaluate(test, role="test")
            validate_predictions(result.predictions, prepared.cohort, prepared.outer, model=runtime.prediction_model,
                                 target_atol=config.settings["protocol"]["target_atol"])
            if result.predictions.subject.tolist() != list(prepared.partition.test_subjects):
                raise RunnerError("prediction_order", "Test predictions do not preserve C1 source subject order")
            runtime.checkpoints.export_predictions(result, store.attempt / "predictions.csv.gz")
            metrics = result.metrics
            prediction_model, status = runtime.prediction_model, "completed"
        write_json(store.attempt / "metrics.json", metrics, exclusive=True)
        _assert_inputs(protocol, prepared.cohort)
        completion = store.complete(prediction_model=prediction_model, status=status)
        return ExecutionResult(store.root, completion, metrics, False)


def execute_plan(plans, *, max_runs=1, existing="fail", adapter_factory=None, backbone_factory=None):
    """Only explicit calls execute. Bounds checked before adapter or model construction.

    Factory injection is a deliberate test/embedding seam, not a CLI training-mode
    fallback; actual registered M1/M2/M3 classes and trainers remain mandatory.
    """
    from .experiment_runner import execution_guard
    execution_guard(plans, max_runs=max_runs)
    if existing not in ("fail", "resume", "restart"):
        raise RunnerError("existing_policy", "Existing-run policy must be fail, resume or restart")
    results = []
    with progress_bar("Experiments", total=len(plans), unit="run") as bar:
        with nullcontext() if bar.disable else logging_redirect_tqdm():
            for plan in plans:
                bar.set_postfix_str(f"model={plan.model} seed={plan.seed} fold={plan.fold}")
                results.append(_execute_one(plan, existing=existing, adapter_factory=adapter_factory,
                                            backbone_factory=backbone_factory))
                bar.update()
    return tuple(results)
