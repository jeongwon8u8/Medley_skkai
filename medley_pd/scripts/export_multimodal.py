"""Verify completed runs and export aggregate training/test plot data."""
import argparse
import csv
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

from multimodal.experiment_manifest import read_manifest
from multimodal.run_state import read_json, validate_completion


def completed(path):
    root = Path(path).expanduser().resolve()
    if root.name == "completed.json":
        root = root.parent
    marker = read_json(root / "completed.json")
    validate_completion(root, marker, verify_tensors=False)
    manifest = read_manifest(root.parent.parent / "manifest.json")
    return marker, manifest["identity"], manifest["resolved_config"], root / marker["attempt"]


def inspect(args):
    roots = [Path(args.execution)] if args.execution else [
        p.parent for p in Path(args.root).glob("*/executions/*/completed.json")
        if all(read_json(p).get(k) == getattr(args, k) for k in ("model", "seed", "fold"))
    ]
    if not roots and args.missing_ok:
        raise SystemExit(3)
    if len(roots) != 1:
        raise ValueError("Expected one completed result; supply its directory with --execution")
    verified = completed(roots[0])
    marker, _, _, attempt = verified
    if any(marker[k] != getattr(args, k) for k in ("model", "seed", "fold")):
        raise ValueError("Execution does not match model, seed and fold")
    result = {"model": args.model, "seed": args.seed, "fold": args.fold,
              "status": marker["status"], "integrity": "validated", "directory": str(attempt),
              "test": read_json(attempt / "metrics.json"),
              "selection": read_json(attempt / "selection.json")}
    from export_per_case import export_case
    result["per_case"] = str(export_case(roots[0], verified=verified, replay_if_missing=True))
    return result


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export(args):
    history, metrics, provenance, identities = [], [], [], set()
    comparable = ("seed", "fold", "target", "unit", "outer_sha256", "partition_fingerprint",
                  "cohort_fingerprint", "cache_fingerprint")
    for path in args.executions:
        marker, identity, config, attempt = completed(path)
        provenance.append(tuple(identity[key] for key in comparable))
        run = marker["execution_id"]
        if run in identities:
            raise ValueError("Duplicate execution")
        identities.add(run)
        model = identity["model"]
        category = {"M1": "Clinical-only", "M2": "Multimodal", "M3": "Multimodal"}.get(model)
        if model == "M0":
            baseline = identity["baseline_identifier"]
            category = "EEG-only" if baseline == "eeg" else f"Baseline ({baseline})"
        keys = {"model": model, "category": category, "seed": identity["seed"],
                "fold": identity["fold"], "execution_id": run}
        scores = read_json(attempt / "metrics.json")
        metrics.append({**keys, "test_n": scores["n"], "test_mae": scores["MAE"], "test_rmse": scores["RMSE"]})
        if model != "M0":
            records = read_json(attempt / "history.json")
            selection = read_json(attempt / "selection.json")
            training = config["settings"]["training"]
            max_epochs = training["max_epochs"]
            for row in records:
                history.append({
                    **keys, "epoch": row["epoch"], "train_loss": row["training"]["huber"],
                    "validation_mae": row["validation"]["MAE"], "validation_rmse": row["validation"]["RMSE"],
                    "optimizer_steps": row["training"]["optimizer_steps"], "bad_epochs": row["bad_epochs"],
                    "best_epoch": int(row["epoch"] == selection["best_epoch"]), "max_epochs": max_epochs,
                    "early_stop": int(row["bad_epochs"] >= training["early_stopping_patience"]),
                })
    if len(set(provenance)) != 1:
        raise ValueError("Comparison requires matching split, cohort, cache, seed and fold")
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = ["test_metrics.csv"] + (["history.csv"] if history else [])
    if any((output / name).exists() for name in names):
        raise ValueError("CSV output exists; choose a fresh output directory")
    write_csv(output / "test_metrics.csv", metrics)
    if history:
        write_csv(output / "history.csv", history)
    return {"csv": [str(output / name) for name in names]}


def figures(args):
    from plot_multimodal import plot

    result, = read_json(Path(args.result_file))
    path = result["directory"]
    verified = completed(path)
    marker, _, _, _ = verified
    if marker["execution_id"] != result["execution_id"] or marker["status"] != "completed":
        raise ValueError("Expected the exact completed training execution")
    from export_per_case import export_case
    per_case = export_case(path, verified=verified, replay_if_missing=True)
    base = Path(args.output).resolve()
    base.mkdir(parents=True, exist_ok=True)
    stem = f"{marker['model']}_seed{marker['seed']}_fold{marker['fold']}"
    files = {"history.csv", "test_metrics.csv", "learning_curves.png", "learning_curves.svg",
             "test_performance.png", "test_performance.svg"}
    for name in (stem, f"{stem}_{marker['execution_id']}"):
        target = base / name
        if target.is_dir() and all((target / file).is_file() for file in files):
            with (target / "test_metrics.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) == 1 and rows[0]["execution_id"] == marker["execution_id"]:
                return {"figures": str(target), "per_case": str(per_case)}
        if not target.exists() and not target.is_symlink():
            with TemporaryDirectory(dir=base) as temporary:
                staging = Path(temporary) / "figures"
                export(SimpleNamespace(executions=[path], output=staging))
                plot(staging, staging)
                try:
                    staging.rename(target)
                except OSError:
                    if target.exists():
                        continue
                    raise
            return {"figures": str(target), "per_case": str(per_case)}
    raise ValueError("Figure output already exists for this execution")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    lookup = modes.add_parser("inspect")
    lookup.add_argument("model", choices=("M1", "M2", "M3"))
    lookup.add_argument("seed", type=int)
    lookup.add_argument("fold", type=int)
    lookup.add_argument("--execution")
    lookup.add_argument("--missing-ok", action="store_true", help="Exit 3 when no completed run exists")
    lookup.add_argument("--root", default="outputs/multimodal_runs")
    exporter = modes.add_parser("export")
    exporter.add_argument("executions", nargs="+")
    exporter.add_argument("--output", required=True)
    renderer = modes.add_parser("figures")
    renderer.add_argument("result_file")
    renderer.add_argument("--output", default="outputs/multimodal_figures")
    args = parser.parse_args()
    try:
        result = {"inspect": inspect, "export": export, "figures": figures}[args.mode](args)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Unable to {args.mode} completed results ({type(exc).__name__}).\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
