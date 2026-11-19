"""Safe planning entrypoint; execution is explicit and bounded, never inferred."""
from __future__ import annotations

# Support both `python -m medley_pd.multimodal.experiment_runner` from repo root
# and application-style `PYTHONPATH=medley_pd python -m multimodal.experiment_runner`.
from pathlib import Path
import sys
if __package__ and __package__.startswith("medley_pd."):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import copy
import json

from .config import load_multimodal_config
from .experiment_plan import PlannedRun
from .experiment_protocol import ReadOnlyOuterSplits, fingerprint
from .model_registry import RunnerError, model_config, registration


def plan_experiments(config, *, models=None, seeds=None, folds=None, output_dir=None):
    config = copy.deepcopy(config)
    if output_dir is not None:
        config.settings["protocol"]["manifest_dir"] = str(Path(output_dir).expanduser().resolve())
    if models is None:
        models = [{"clinical_only": "M1", "fusion_a": "M2", "fusion_b": "M3"}[config.settings["architecture"]]]
    if isinstance(models, str) or not models or len(models) != len(set(models)):
        raise RunnerError("selection", "Models require a nonempty explicit sequence without duplicates")
    seeds = config.settings["evaluation"]["seeds"] if seeds is None else seeds
    folds = config.settings["evaluation"]["folds"] if folds is None else folds
    for values, key in ((seeds, "seeds"), (folds, "folds")):
        if (not values or len(values) != len(set(values)) or any(type(v) is not int for v in values)
                or not set(values) <= set(config.settings["evaluation"][key])):
            raise RunnerError("selection", "Seed/fold selection must use distinct configured identifiers")
    plans = []
    for model in models:
        effective = model_config(config, model)
        requirements = ["C1 exact cohort/cache/target validation before execution"]
        try:
            source = ReadOnlyOuterSplits(effective, None)
            outer_hash = source.sha256
        except ValueError:
            outer_hash = None
            requirements.append("Missing/invalid fixed split source; execution blocked")
        if model == "M0":
            requirements.append("Existing baseline_path and baseline_model; read-only import, no training")
        else:
            requirements.append("Local pinned LLM assets and compatible precision/device/backend; no download")
            if not effective.settings["llm"]["model_id"]:
                requirements.append("llm.model_id is missing")
            if not effective.settings["llm"]["revision"]:
                requirements.append("llm.revision is missing")
            if effective.settings["llm"]["quantization"] == "nf4":
                requirements.append("Genuine bitsandbytes NF4 + CUDA readiness and real-model VRAM preflight")
        root = str(Path(effective.settings["protocol"]["manifest_dir"]).expanduser().resolve())
        effective.settings["protocol"]["manifest_dir"] = root
        for seed in seeds:
            for fold in folds:
                identity = {"version": "medley_plan_v1", "model": model, "seed": seed, "fold": fold,
                            "config": effective.to_dict(), "outer_sha256": outer_hash, "output_root": root}
                plans.append(PlannedRun(model, seed, fold, copy.deepcopy(effective), fingerprint(identity),
                                        outer_hash, root, tuple(requirements)))
    return tuple(plans)


def execution_guard(plans, *, max_runs=1):
    if type(max_runs) is not int or max_runs < 1 or not plans or len(plans) > max_runs:
        raise RunnerError("run_limit", "Explicit run count exceeds max-runs; no model or data was loaded")
    keys = [(p.model, p.seed, p.fold, p.plan_id) for p in plans]
    if len(keys) != len(set(keys)):
        raise RunnerError("selection", "Duplicate planned executions are forbidden")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--model", nargs="+", choices=("M0", "M1", "M2", "M3"))
    parser.add_argument("--seed", nargs="+", type=int)
    parser.add_argument("--fold", nargs="+", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--ack-grid", action="store_true", help="Explicitly acknowledge a broader selected grid")
    parser.add_argument("--max-runs", type=int, default=1)
    parser.add_argument("--output-dir")
    parser.add_argument("--continue-from", help="Read compatible epoch state from a previous run root")
    parser.add_argument("--allow-fresh", action="store_true")
    parser.add_argument("--result-file", help="Write execution metadata to a new JSON file after success")
    existing = parser.add_mutually_exclusive_group()
    existing.add_argument("--resume-completed", action="store_true")
    existing.add_argument("--resume", action="store_true", help="Continue from the last saved epoch")
    existing.add_argument("--resume-or-start", action="store_true", help="Reuse, resume, or start a matching execution")
    existing.add_argument("--restart-incomplete", action="store_true")
    existing.add_argument("--fail-on-existing", action="store_true")
    parser.add_argument("--inspect-completed", metavar="EXECUTION_DIRECTORY")
    args = parser.parse_args(argv)
    try:
        if args.inspect_completed:
            from .run_state import inspect_completed
            print(json.dumps(inspect_completed(args.inspect_completed), sort_keys=True, indent=2))
            return 0
        if not args.config:
            parser.error("--config is required for planning/execution")
        config = load_multimodal_config(args.config, args.set)
        plans = plan_experiments(config, models=args.model, seeds=args.seed, folds=args.fold, output_dir=args.output_dir)
        if args.execute:
            if not args.model or (not args.ack_grid and (args.seed is None or args.fold is None)):
                raise RunnerError("execute_selection", "Execution requires --model and explicit --seed/--fold, or --ack-grid")
            execution_guard(plans, max_runs=args.max_runs)
        print(json.dumps({"planned_runs": len(plans), "models": sorted({p.model for p in plans}),
                          "execution_enabled": args.execute, "runs": [p.summary() for p in plans]}, sort_keys=True, indent=2))
        if args.execute:
            from .runner_execution import execute_plan
            policy = ("auto" if args.resume_or_start else "continue" if args.resume else "resume" if args.resume_completed
                      else "restart" if args.restart_incomplete else "fail")
            continuation = ({"continue_from": args.continue_from, "allow_fresh": args.allow_fresh}
                            if args.continue_from else {})
            results = execute_plan(plans, max_runs=args.max_runs, existing=policy, **continuation)
            payload = [result.public_metadata() for result in results]
            if args.result_file:
                from .run_state import write_json
                write_json(Path(args.result_file), payload, exclusive=True)
            print(json.dumps(payload, sort_keys=True, indent=2))
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        code = exc.code if isinstance(exc, RunnerError) else "configuration_or_execution"
        message = str(exc) if isinstance(exc, RunnerError) else "Invalid configuration or source inputs; inspect documented preflight"
        parser.exit(2, f"Runner error [{code}]: {message}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
