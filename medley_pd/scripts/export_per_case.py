"""Export verified test cases, optionally by replaying an inference checkpoint."""
import argparse
import gzip
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

from export_multimodal import APP, completed


def replay(marker, identity, attempt, *, backbone_factory=None, adapter_factory=None, case_path=None):
    from multimodal.config import load_multimodal_config
    from multimodal.data_adapter import MedleyDataAdapter
    from multimodal.experiment_cohort import validate_predictions
    from multimodal.experiment_protocol import ExperimentProtocol
    from multimodal.model_registry import create_runtime
    from multimodal.runner_execution import _assert_inputs, _loaders
    from multimodal.run_state import read_json
    from multimodal.training_control import verify_restoration

    config = load_multimodal_config(attempt / "runtime_config.json")
    adapter = (adapter_factory or MedleyDataAdapter)(config)
    seed, fold = marker["seed"], marker["fold"]
    protocol = ExperimentProtocol(config, adapter)
    prepared = protocol.prepare(seed=seed, fold=fold)
    if prepared.outer_sha256 != identity["outer_sha256"] or any(
        getattr(prepared.cohort, field) != identity[field]
        for field in ("cohort_fingerprint", "cache_fingerprint", "clinical_fingerprint")
    ):
        raise ValueError("Current inputs differ from the completed execution")
    runtime = create_runtime(marker["model"], config, prepared, backbone_factory=backbone_factory)
    selection = read_json(attempt / "selection.json")
    verify_restoration(runtime, attempt / selection["checkpoint"], best_epoch=selection["best_epoch"])
    _, _, test = _loaders(adapter, prepared, config, seed=seed, fold=fold)
    runtime.trainer.test_case_path = case_path
    result = runtime.trainer.evaluate(test, role="test")
    validate_predictions(result.predictions, prepared.cohort, prepared.outer,
                         model=marker["prediction_model"], target_atol=config.settings["protocol"]["target_atol"])
    if tuple(result.predictions.subject) != prepared.partition.test_subjects:
        raise ValueError("Replayed predictions do not preserve the completed test order")
    _assert_inputs(protocol, prepared.cohort)
    return result


def export_case(execution, *, re_evaluate=False, verified=None, replay_if_missing=False):
    from multimodal.run_state import file_hash, read_json, write_json

    marker, identity, _, attempt = verified if verified is not None else completed(execution)
    label = f"{marker['model']}_seed{marker['seed']}_fold{marker['fold']}"
    root = APP / "outputs" / "multimodal_per_case" / label / marker["execution_id"]
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if replay_if_missing and not re_evaluate:
        native = attempt / "per_case.csv"
        if native.is_file():
            return native
        for source in ("checkpoint_replay", "recorded_test"):
            existing = root / source
            if existing.is_dir():
                target = existing / "per_case.csv"
                if file_hash(target) != read_json(existing / "provenance.json")["csv_sha256"]:
                    raise ValueError("Existing per-case export failed its checksum")
                return target
        re_evaluate = True
    directory = root / ("checkpoint_replay" if re_evaluate else "recorded_test")
    target = directory / "per_case.csv"
    if directory.exists():
        if file_hash(target) != read_json(directory / "provenance.json")["csv_sha256"]:
            raise ValueError("Existing per-case export failed its checksum")
        return target
    with TemporaryDirectory(dir=root) as temporary:
        staging = Path(temporary) / "export"
        staging.mkdir(mode=0o700)
        csv = staging / target.name
        metadata = {"execution_id": marker["execution_id"], "source": directory.name,
                    "recorded_metrics": read_json(attempt / "metrics.json")}
        if re_evaluate:
            from multimodal.clinical_checkpoint import export_predictions

            result = replay(marker, identity, attempt, case_path=root / "replay_live" / target.name)
            export_predictions(result, csv)
            metadata["replayed_metrics"] = result.metrics
            metadata["checkpoint"] = read_json(attempt / "selection.json")["checkpoint"]
        else:
            with gzip.open(attempt / "predictions.csv.gz", "rb") as source, csv.open("xb") as output:
                shutil.copyfileobj(source, output)
        metadata["csv_sha256"] = file_hash(csv)
        write_json(staging / "provenance.json", metadata)
        staging.rename(directory)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executions", nargs="+", type=Path)
    parser.add_argument("--re-evaluate", action="store_true")
    parser.add_argument("--repair-missing", action="store_true")
    args = parser.parse_args()
    failures = []
    for execution in args.executions:
        try:
            print(export_case(execution, re_evaluate=args.re_evaluate, replay_if_missing=args.repair_missing), flush=True)
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            failures.append(execution)
            print(f"Per-case export failed for {execution.name} ({getattr(exc, 'code', type(exc).__name__)})",
                  file=sys.stderr, flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
