"""Reproducible read-only analysis orchestration and private aggregate reports."""
from __future__ import annotations

from dataclasses import dataclass,field
import importlib.metadata
import math
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd

from .config import validate_config
from .experiment_protocol import fingerprint,canonical_json
from .experiment_manifest import REPOSITORY
from .experiment_evaluation import AnalysisError
from .prediction_matching import match_predictions
from .evaluation_metrics import summarize_series,paired_metrics
from .paired_bootstrap import paired_bootstrap
from .run_state import write_json,file_hash

DEFAULT_PAIRS=(("M1","M0"),("M2","M0"),("M3","M0"),("M2","M1"),("M3","M1"),("M3","M2"))
LIMITS=("Descriptive intervals condition on stored fitted predictions. Subject clustering preserves repeated patient records; "
        "overlapping training sets, fitted-model reuse, hyperparameter selection, dataset shift and small/nonrepresentative cohorts remain incompletely represented. "
        "No p-values or clinical-superiority inference are produced.")


def _settings(config,series):
    validate_config(config.settings)
    if not series or any(label!=item.label for label,item in series.items()):
        raise AnalysisError("sources","Provide explicit, consistently labeled prediction series")
    options=config.to_dict()["settings"]["analysis"]
    for item in series.values():
        item.assert_unchanged()
        for run in item.runs.values():
            source=run["manifest"]["resolved_config"]["settings"]
            if source["dataset"]!=config.settings["dataset"] or source["task"]!=config.settings["task"]:
                raise AnalysisError("target_scope","Source dataset/task differs from the analysis configuration")
            if source.get("analysis",{}).get("synthetic",False) and not options["synthetic"]:
                raise AnalysisError("synthetic_label","Authored synthetic sources require analysis.synthetic=true")
    scope={(seed,fold) for seed in config.settings["evaluation"]["seeds"] for fold in config.settings["evaluation"]["folds"]}
    pairs=options["pairs"] if options["pairs"] is not None else DEFAULT_PAIRS
    return options,scope,pairs


def analysis_plan(config,series):
    options,scope,pairs=_settings(config,series)
    comparisons=[]
    for candidate,reference in pairs:
        entry={"candidate":candidate,"reference":reference}
        if candidate not in series or reference not in series:
            comparisons.append({**entry,"status":"unavailable","reason":"model_series_missing"});continue
        try:
            matched=match_predictions(series[candidate],series[reference],matching=options["matching"],expected_runs=scope,
                synthetic=options["synthetic"],allowed_differences=options["allowed_differences"])
            comparisons.append({**entry,"status":"available_descriptive","matching":matched.public_metadata()})
        except AnalysisError as exc:
            comparisons.append({**entry,"status":"unavailable","reason":exc.code,"conditions":getattr(exc,"conditions",None)})
    return {"schema":"medley_analysis_plan_v1","models":sorted(series),"requested_runs":[list(k) for k in sorted(scope)],
            "data_label":"synthetic test data" if options["synthetic"] else "stored predictions; clinical provenance requires separate verification",
            "analysis_options":options,"comparisons":comparisons,"limits":LIMITS,"execution":False}


def _json_safe(value):
    if isinstance(value,dict): return {str(k):_json_safe(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [_json_safe(v) for v in value]
    if isinstance(value,np.generic): return _json_safe(value.item())
    if isinstance(value,float) and not math.isfinite(value): return None
    return value


@dataclass(frozen=True)
class AnalysisResult:
    analysis_id: str
    tables: dict=field(repr=False)
    manifest: dict=field(repr=False)
    summary: str
    sources: tuple=field(repr=False)


def analyze_predictions(config,series):
    options,scope,pairs=_settings(config,series)
    plan=analysis_plan(config,series)
    table_parts={name:[] for name in ("metrics_by_run","metrics_by_seed","metrics_by_fold","metrics_summary")}
    coverage={};unavailable=[]
    for label,item in sorted(series.items()):
        frame=item.frame
        present=set(frame[["seed","fold"]].itertuples(index=False,name=None)) & scope
        coverage[label]={"requested_runs":len(scope),"present_runs":[list(k) for k in sorted(present)],
                         "missing_runs":[list(k) for k in sorted(scope-present)]}
        try:
            tables=summarize_series(item,expected_runs=scope)
            for name,table in tables.items():
                table["coverage"]="complete_requested_scope" if present==scope else "partial_available_runs"
                table_parts[name].append(table)
        except AnalysisError as exc:
            unavailable.append({"label":label,"component":"metrics","reason":exc.code})
    comparisons=[];intervals=[];diagnostics=[]
    for entry in plan["comparisons"]:
        candidate,reference=entry["candidate"],entry["reference"]
        if entry["status"]=="unavailable":
            comparisons.append({"candidate":candidate,"reference":reference,"status":"unavailable","reason":entry["reason"]})
            intervals.append({"candidate":candidate,"reference":reference,"status":"unavailable","reason":entry["reason"]})
            continue
        matched=match_predictions(series[candidate],series[reference],matching=options["matching"],expected_runs=scope,
            synthetic=options["synthetic"],allowed_differences=options["allowed_differences"])
        point=paired_metrics(matched,aggregation=options["aggregation"])
        comparisons.append({**point,"status":"available_descriptive","population_label":matched.diagnostics["population_label"],
            "recorded_protocol_comparable":matched.conditions["recorded_protocol_comparable"],
            "fully_controlled_configuration":matched.conditions["fully_controlled_configuration"]})
        diagnostics.append({"candidate":candidate,"reference":reference,"diagnostics":matched.diagnostics,"conditions":matched.conditions})
        try:
            result=paired_bootstrap(matched,replicates=options["bootstrap_replicates"],confidence=options["confidence"],
                seed=options["bootstrap_seed"],min_subjects=options["min_subjects"],aggregation=options["aggregation"])
            intervals.append(result.metadata)
        except AnalysisError as exc:
            intervals.append({"candidate":candidate,"reference":reference,"status":"unavailable","reason":exc.code,
                              "aggregation":options["aggregation"]})
    if options["subgroups"]:
        unavailable.append({"component":"subgroups","reason":"subgroup_definition_provenance_not_present_in_C1_C2"})
    tables={name:pd.concat(parts,ignore_index=True) if parts else pd.DataFrame() for name,parts in table_parts.items()}
    tables.update(pairwise_comparison=pd.DataFrame(comparisons),bootstrap_intervals=pd.DataFrame(intervals))
    source_records={label:{"provenance":item.provenance,"files":dict(sorted(item.sources.items())),
        "runs":[{"seed":key[0],"fold":key[1],"run_id":run["manifest"]["run_id"],
                 "config_fingerprint":run["manifest"]["identity"]["config_fingerprint"]}
                for key,run in sorted(item.runs.items())]} for label,item in sorted(series.items())}
    code_files=("experiment_evaluation.py","prediction_matching.py","evaluation_metrics.py","paired_bootstrap.py","evaluation_reports.py","evaluation_cli.py")
    code={name:file_hash(Path(__file__).with_name(name)) for name in code_files if Path(__file__).with_name(name).exists()}
    runtime={"python":platform.python_version(),"libraries":{name:importlib.metadata.version(name) for name in ("numpy","pandas","scipy")}}
    identity={"version":"medley_analysis_identity_v1","target":"current_moca_original_units","unit":"subject",
              "options":options,"scope":[list(k) for k in sorted(scope)],"sources":source_records,"code":code,"runtime":runtime}
    analysis_id="analysis-"+fingerprint(identity)
    manifest={"schema":"medley_analysis_v1","analysis_id":analysis_id,"identity":identity,"data_label":plan["data_label"],
              "coverage":coverage,"comparisons":plan["comparisons"],"matching_diagnostics":diagnostics,"unavailable":unavailable,
              "direction":"candidate_minus_reference_negative_is_lower_error","bootstrap":"paired_subject_cluster_percentile_v1",
              "completion_verification":"C2_semantics_and_byte_hashes_no_model_or_tensor_loading","limits":LIMITS}
    summary=["# Matched multimodal evaluation", "",plan["data_label"],"",f"Analysis: {analysis_id}","",
             f"Target: current MoCA; aggregation: {options['aggregation']}; matching: {options['matching']}.",
             f"Bootstrap: subject clusters; draws={options['bootstrap_replicates']}; seed={options['bootstrap_seed']}; confidence={options['confidence']}.","",
             "Delta = candidate error minus reference error; negative values indicate lower error in the matched predictions.","",
             "| Candidate | Reference | Status | delta MAE | delta RMSE |","|---|---|---|---:|---:|"]
    for row in comparisons:
        def number(key): return format(row[key],".8g") if key in row else "unavailable"
        summary.append(f"| {row['candidate']} | {row['reference']} | {row['status']} | {number('delta_MAE')} | {number('delta_RMSE')} |")
    summary.extend(["", "Run means and pooled prediction rows are different estimands; denominators are recorded in CSV/JSON tables.","",LIMITS,"",
                    "Unavailable comparisons and interval reasons are recorded explicitly. No unexecuted performance is filled in."])
    for item in series.values(): item.assert_unchanged()
    return AnalysisResult(analysis_id,tables,_json_safe(manifest),"\n".join(summary)+"\n",tuple(series.values()))


def export_analysis(result,output_dir):
    root=Path(output_dir).expanduser().resolve();namespace=root / result.analysis_id
    for series in result.sources:
        series.assert_unchanged()
        protected={Path(run["c1_root"]).resolve() for run in series.runs.values()}
        for source in series.sources:
            if namespace==Path(source).resolve(): raise AnalysisError("output_path","Analysis cannot overwrite an input artifact")
        for run in series.runs.values():
            protected.add(Path(run["manifest"]["identity"]["outer_source"]).parent.resolve())
            for key in ("data_root","feature_root"):
                path=Path(run["manifest"]["resolved_config"]["medley"]["paths"][key])
                if path.is_absolute(): protected.add(path.resolve())
        if any(namespace==path or path in namespace.parents for path in protected):
            raise AnalysisError("output_path","Reports must be outside input experiment/data/cache/split namespaces")
    if namespace==REPOSITORY or REPOSITORY in namespace.parents:
        relative=namespace.relative_to(REPOSITORY)
        if (not relative.parts or relative.parts[0] not in ("outputs","checkpoints")
                or subprocess.run(["git","check-ignore","--quiet",str(namespace / "analysis_manifest.json")],cwd=REPOSITORY).returncode!=0):
            raise AnalysisError("output_path","In-repository analysis requires ignored outputs/ or checkpoints/")
    namespace.mkdir(parents=True,exist_ok=False,mode=0o700)
    write_json(namespace / "analysis_manifest.json",result.manifest,exclusive=True)
    options=result.manifest["identity"]["options"]
    if options["output_format"]=="json":
        write_json(namespace / "analysis_result.json",{name:_json_safe(table.to_dict(orient="records")) for name,table in result.tables.items()},exclusive=True)
    else:
        for name,table in sorted(result.tables.items()):
            path=namespace / (name+".csv")
            with path.open("x",encoding="utf-8",newline="") as stream:
                table.to_csv(stream,index=False,lineterminator="\n",float_format="%.17g")
            path.chmod(0o600)
        summary=namespace / "analysis_summary.md";summary.write_text(result.summary,encoding="utf-8");summary.chmod(0o600)
    for series in result.sources: series.assert_unchanged()
    hashes={path.name:file_hash(path) for path in sorted(namespace.iterdir()) if path.is_file()}
    write_json(namespace / "analysis_complete.json",{"schema":"medley_analysis_complete_v1","analysis_id":result.analysis_id,"artifacts":hashes},exclusive=True)
    return namespace
