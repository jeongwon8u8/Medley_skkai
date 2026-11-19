"""Exit 0 for a verified reusable Fold, 3 when missing, 4 while it is active."""
import fcntl
import os
import sys

from compare_multimodal_oof import APP, MissingFolds, completed_paths, load_completed_run
from multimodal.config import load_multimodal_config
from multimodal.experiment_runner import plan_experiments
from multimodal.data_adapter import MedleyDataAdapter
from multimodal.experiment_protocol import ExperimentProtocol
from multimodal.training_resume import continuation_config

CONFIGS = {"M1": "clinical_only", "M2": "fusion_a", "M3": "fusion_b"}


def status(model, seed, fold):
    cfg = load_multimodal_config(APP / "configs" / "multimodal" / f"{CONFIGS[model]}.json", [
        "llm.model_id=" + os.environ.get("MEDLEY_LLM_PATH", "/workspace/models/Qwen3-4B"),
        "llm.revision=" + os.environ.get("MEDLEY_LLM_REVISION", "1cfa9a7208912126459214e8b04321603b3df60c"),
        "llm.device=" + os.environ.get("MEDLEY_LLM_DEVICE", "cuda:0"),
        'clinical_text.sex_code_mapping={"0":"female","1":"male"}',
        "training.max_epochs=30", "training.early_stopping_patience=30",
    ])
    plan, = plan_experiments(cfg, models=[model], seeds=[seed], folds=[fold])
    outputs = APP / "outputs"
    try:
        marker = completed_paths(outputs, {(seed, fold): None}, full_epochs=True, models=(model,))[model, (seed, fold)]
    except MissingFolds:
        from multimodal.run_state import read_json
        for root in ("multimodal_runs_30epochs", "multimodal_runs"):
            for manifest in (outputs / root).glob("*/manifest.json"):
                identity = read_json(manifest)["identity"]
                if (identity["model"], identity["seed"], identity["fold"]) != (model, seed, fold):
                    continue
                for lock in manifest.parent.glob("executions/*/.lock"):
                    with lock.open("rb") as stream:
                        try:
                            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            return 4
        return 3
    series = load_completed_run(marker, label=model)
    manifest = series.runs[seed, fold]["manifest"]
    if (manifest["identity"]["outer_sha256"] != plan.outer_sha256
            or continuation_config(manifest["resolved_config"]) != continuation_config(plan.config.to_dict())):
        raise ValueError("Completed Fold is incompatible with the requested split or configuration")
    protocol = ExperimentProtocol(plan.config, MedleyDataAdapter(plan.config))
    prepared = protocol.prepare(seed=seed, fold=fold)
    for fingerprint in ("cohort_fingerprint", "clinical_fingerprint", "cache_fingerprint"):
        if getattr(prepared.cohort, fingerprint) != manifest["identity"][fingerprint]:
            raise ValueError("Completed Fold has different current cohort, clinical or cache inputs")
    if manifest["subjects"] != {
        "train": list(prepared.partition.train_subjects),
        "validation": list(prepared.partition.validation_subjects),
        "test": list(prepared.partition.test_subjects),
    }:
        raise ValueError("Completed Fold has different current patient partitions")
    print(marker)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(status(sys.argv[1], int(sys.argv[2]), int(sys.argv[3])))
    except (OSError, KeyError, ValueError) as exc:
        print(f"Fold reuse check failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
