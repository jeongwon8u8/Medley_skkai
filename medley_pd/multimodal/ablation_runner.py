"""OFAT planning, explicit bounded C2 execution, and existing C3 comparison."""
from __future__ import annotations
from pathlib import Path
import sys
if __package__ and __package__.startswith("medley_pd."):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import argparse
import copy
from dataclasses import dataclass,field,replace
import json

from .ablation_registry import resolve_variant
from .config import load_multimodal_config
from .experiment_runner import plan_experiments,execution_guard
from .experiment_protocol import fingerprint
from .model_registry import RunnerError


def _equivalent_config(config):
    """Only operational/analysis fields and inactive Phase D opt-in are removed."""
    value=copy.deepcopy(config)
    settings=value["settings"]
    for key in ("analysis","experiment_name","evaluation"): settings.pop(key,None)
    settings.get("protocol",{}).pop("manifest_dir",None)
    ablation=settings.get("ablation",{})
    if (settings["eeg"]["pooling"]["type"]=="attention" and not settings["eeg"]["feature_adapter"]["enabled"]
            and not settings["fusion"]["numeric_skip"]):
        # New inactive D defaults cannot change the old architecture.
        settings.pop("ablation",None)
    else: ablation.pop("enabled",None)
    return value


def find_completed(plan,search_roots):
    """Read only explicit C2 registries; full byte/semantic checks, no tensor IO.

    A match remains conditional on revalidating current inputs before execution.
    Invalid relevant markers fail closed rather than scheduling replacement work.
    """
    from .experiment_manifest import read_manifest,runtime_metadata
    from .experiment_evaluation import load_completed_run
    found=[]
    current_runtime=runtime_metadata()
    for directory in search_roots:
        directory=Path(directory).expanduser().resolve()
        markers=[directory / "completed.json"] if (directory / "completed.json").exists() else sorted(directory.glob("medley-*/executions/*/completed.json"))
        for marker in markers:
            source=marker.parent.parent.parent / "manifest.json"
            manifest=read_manifest(source);info=manifest["identity"]
            if (info["model"],info["seed"],info["fold"])!=(plan.model,plan.seed,plan.fold): continue
            if _equivalent_config(manifest["resolved_config"])!=_equivalent_config(plan.config.to_dict()): continue
            if info["outer_sha256"]!=plan.outer_sha256: continue
            item=load_completed_run(marker)
            if any(info["runtime"][key]!=current_runtime[key] for key in ("python","libraries","application_source_fingerprint")):
                raise RunnerError("reuse_provenance","Completed configuration matches but code/environment equivalence is unverified")
            found.append(item)
    unique={next(iter(item.runs.values()))["source_root"]:item for item in found}
    if len(unique)>1:
        raise RunnerError("ambiguous_reuse","Several equivalent completed executions exist; select a single explicit source")
    return next(iter(unique.values()),None)


@dataclass(frozen=True)
class AblationPlan:
    variant: object=field(repr=False)
    reference_runs: tuple=field(repr=False)
    candidate_runs: tuple=field(repr=False)
    search_roots: tuple
    plan_id: str

    def summary(self):
        self.variant.assert_valid()
        entries=[]
        for role,runs in (("reference",self.reference_runs),("candidate",self.candidate_runs)):
            for run in runs:
                existing=find_completed(run,self.search_roots)
                entries.append({"role":role,**run.summary(),"existing_result":None if existing is None else next(iter(existing.runs.values()))["source_root"],
                    "reuse_status":"missing_requires_authorized_run" if existing is None else "completed_bytes_valid_current_inputs_pending"})
        return {"schema":"medley_ablation_plan_v1","plan_id":self.plan_id,"variant":self.variant.summary(),
                "estimated_runs":len(entries),"execution":False,"runs":entries}


def plan_ablation(variant,*,seeds,folds,output_dir,search_roots=()):
    variant.assert_valid()
    if seeds is None or folds is None or output_dir is None:
        raise RunnerError("ablation_selection","Provide explicit seed/fold selection and output directory")
    # One family/value, no Cartesian product of factors. C2 namespaces include exact config.
    reference=plan_experiments(variant.reference,models=[variant.model],seeds=seeds,folds=folds,output_dir=output_dir)
    candidate=plan_experiments(variant.candidate,models=[variant.model],seeds=seeds,folds=folds,output_dir=output_dir)
    roots=tuple(dict.fromkeys(str(Path(p).expanduser().resolve()) for p in
        (output_dir,variant.reference.settings["protocol"]["manifest_dir"],variant.candidate.settings["protocol"]["manifest_dir"],*search_roots)))
    identity={"variant":variant.variant_id,"reference":[r.plan_id for r in reference],"candidate":[r.plan_id for r in candidate],"search_roots":roots}
    return AblationPlan(variant,reference,candidate,roots,"ablation-plan-"+fingerprint(identity))


def _prepared(plan,adapter_factory):
    from .experiment_protocol import ExperimentProtocol
    from .data_adapter import MedleyDataAdapter
    adapter=MedleyDataAdapter(plan.config) if adapter_factory is None else adapter_factory(copy.deepcopy(plan.config))
    protocol=ExperimentProtocol(plan.config,adapter)
    return protocol.prepare(seed=plan.seed,fold=plan.fold)


def _same_population(a,b):
    return (a.cohort.cohort_fingerprint==b.cohort.cohort_fingerprint
            and a.partition==b.partition and a.outer_sha256==b.outer_sha256)


def execute_ablation(plan,*,execute=False,roles=None,max_runs=1,existing="resume",adapter_factory=None,backbone_factory=None):
    if not execute:
        return plan.summary()
    variant=plan.variant;variant.assert_valid()
    rebuilt=plan_ablation(variant,seeds=list(dict.fromkeys(r.seed for r in plan.reference_runs)),
        folds=list(dict.fromkeys(r.fold for r in plan.reference_runs)),output_dir=plan.reference_runs[0].output_root,search_roots=plan.search_roots)
    if rebuilt!=plan: raise RunnerError("ablation_plan_changed","Plan was changed after resolution")
    if not roles or isinstance(roles,str) or len(set(roles))!=len(roles) or not set(roles)<={"reference","candidate"}:
        raise RunnerError("ablation_selection","Explicit reference/candidate execution roles are required")
    selected=tuple(r for role,runs in (("reference",plan.reference_runs),("candidate",plan.candidate_runs)) if role in roles for r in runs)
    execution_guard(selected,max_runs=max_runs)  # before data/model reads
    # Both branches must use the identical C1 population and partition. No models yet.
    prepared={}
    for a,b in zip(plan.reference_runs,plan.candidate_runs):
        left,right=_prepared(a,adapter_factory),_prepared(b,adapter_factory)
        if not _same_population(left,right):
            raise RunnerError("ablation_cohort","Factor changes population/target/split; strict OFAT execution unavailable")
        prepared[a.plan_id]=left;prepared[b.plan_id]=right
    from .runner_execution import execute_plan,ExecutionResult
    from .run_state import read_json
    results=[]
    for run in selected:
        cached=find_completed(run,plan.search_roots)
        if cached is not None:
            cached.assert_unchanged();source=next(iter(cached.runs.values()));info=source["manifest"]["identity"];current=prepared[run.plan_id]
            if (info["cohort_fingerprint"]!=current.cohort.cohort_fingerprint or info["cache_fingerprint"]!=current.cohort.cache_fingerprint
                    or info["clinical_fingerprint"]!=current.cohort.clinical_fingerprint
                    or source["manifest"]["subjects"]!={"train":list(current.partition.train_subjects),"validation":list(current.partition.validation_subjects),"test":list(current.partition.test_subjects)}):
                raise RunnerError("reuse_inputs","Completed source differs from current C1 inputs")
            root=Path(source["source_root"]);completion=source["completion"]
            result=ExecutionResult(root,completion,read_json(root / completion["attempt"] / "metrics.json"),True)
        else:
            result=execute_plan((run,),max_runs=1,existing=existing,adapter_factory=adapter_factory,backbone_factory=backbone_factory)[0]
        results.append(result)
    return tuple(results)


def compare_ablation(variant,*,reference_paths,candidate_paths,analysis_config,output_dir=None):
    """OFAT-specific eligibility checks, then C3 owns every metric/CI/export."""
    from .experiment_evaluation import load_series
    from .evaluation_reports import analyze_predictions,export_analysis
    variant.assert_valid()
    sources={"candidate":load_series(candidate_paths,label="candidate"),"reference":load_series(reference_paths,label="reference")}
    for role,config in (("candidate",variant.candidate),("reference",variant.reference)):
        for run in sources[role].runs.values():
            if run["manifest"]["identity"]["model"]!=variant.model or _equivalent_config(run["manifest"]["resolved_config"])!=_equivalent_config(config.to_dict()):
                raise RunnerError("ablation_comparison","Completed source does not represent the declared variant")
    config=copy.deepcopy(analysis_config)
    options=config.settings["analysis"]
    options["pairs"]=[["candidate","reference"]]
    # Architectural changes are already attested by exact OFAT config verification.
    options["allowed_differences"]=[p for p in variant.changes if p.startswith("settings.eeg.") or p.startswith("settings.llm.") or p=="settings.clinical_set"]
    result=analyze_predictions(config,sources)
    result.manifest["ablation"]=variant.summary()
    result.manifest["parameter_counts"]={role:[run.get("trainable_parameters") for run in item.runs.values()] for role,item in sources.items()}
    result.manifest["computational_cost"]="not_measured"
    from .run_state import file_hash
    result.manifest["identity"]["ablation"]={"variant_id":variant.variant_id,
        "code":{name:file_hash(Path(__file__).with_name(name)) for name in ("ablation_registry.py","ablation_runner.py","numeric_branch.py","numeric_encoding.py")}}
    analysis_id="analysis-"+fingerprint(result.manifest["identity"])
    result.manifest["analysis_id"]=analysis_id
    summary=result.summary.replace(result.analysis_id,analysis_id)
    summary+=f"\nAblation family: {variant.family}; model: {variant.model}; variant: {variant.variant_id}.\n"
    summary+="Declared changes: "+json.dumps(variant.changes,sort_keys=True)+"\n"
    summary+="Measured trainable parameters per completed run: "+json.dumps(result.manifest["parameter_counts"],sort_keys=True)+"\n"
    summary+="Computational cost: not measured. Structural capacity may differ; synthetic comparisons do not establish clinical benefit.\n"
    result=replace(result,analysis_id=analysis_id,summary=summary)
    for table in result.tables.values():
        table["ablation_family"]=variant.family;table["variant_id"]=variant.variant_id
    if output_dir is not None: export_analysis(result,output_dir)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True);parser.add_argument("--set",action="append",default=[])
    parser.add_argument("--model",required=True,choices=("M1","M2","M3"))
    parser.add_argument("--family",required=True);parser.add_argument("--value",required=True)
    parser.add_argument("--seed",nargs="+",type=int,required=True);parser.add_argument("--fold",nargs="+",type=int,required=True)
    parser.add_argument("--output-dir",required=True);parser.add_argument("--search-root",action="append",default=[])
    mode=parser.add_mutually_exclusive_group();mode.add_argument("--execute",action="store_true");mode.add_argument("--dry-run",action="store_true")
    parser.add_argument("--role",nargs="+",choices=("reference","candidate"));parser.add_argument("--max-runs",type=int,default=1)
    parser.add_argument("--restart-incomplete",action="store_true")
    args=parser.parse_args(argv)
    try:
        config=load_multimodal_config(args.config,args.set)
        try: value=json.loads(args.value)
        except json.JSONDecodeError: value=args.value
        variant=resolve_variant(config,model=args.model,family=args.family,value=value)
        plan=plan_ablation(variant,seeds=args.seed,folds=args.fold,output_dir=args.output_dir,search_roots=args.search_root)
        outcome=execute_ablation(plan,execute=args.execute,roles=args.role,max_runs=args.max_runs,
                                 existing="restart" if args.restart_incomplete else "resume")
        print(json.dumps([r.public_metadata() for r in outcome] if args.execute else outcome,sort_keys=True,indent=2))
    except (ValueError,OSError,ImportError,RuntimeError) as exc:
        parser.exit(2,f"Ablation error [{getattr(exc,'code','ablation_inputs')}]: invalid configuration/provenance or execution prerequisites\n")
    return 0


if __name__=="__main__": raise SystemExit(main())
