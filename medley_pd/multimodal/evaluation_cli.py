"""Read-only matched analysis CLI; report creation requires --execute."""
from __future__ import annotations

from pathlib import Path
import sys
if __package__ and __package__.startswith("medley_pd."):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import argparse
import json

from .config import load_multimodal_config,validate_config
from .experiment_evaluation import load_completed_run,load_series,load_legacy_predictions,AnalysisError
from .evaluation_reports import analysis_plan,analyze_predictions,export_analysis


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--set",action="append",default=[])
    parser.add_argument("--input-manifests",nargs="+",default=[],help="LABEL=completed.json or LABEL=execution_directory; repeat labels for runs")
    parser.add_argument("--legacy-input",action="append",default=[],help="LABEL:original_model=prediction.csv.gz; provenance stays unverified")
    parser.add_argument("--models",nargs="+")
    parser.add_argument("--reference-model")
    matching=parser.add_mutually_exclusive_group()
    matching.add_argument("--strict-matching",action="store_true")
    matching.add_argument("--matched-subset",action="store_true")
    parser.add_argument("--aggregation",choices=("pooled_rows","mean_runs"))
    parser.add_argument("--bootstrap-replicates",type=int)
    parser.add_argument("--bootstrap-seed",type=int)
    parser.add_argument("--confidence",type=float)
    parser.add_argument("--output-dir")
    parser.add_argument("--synthetic-test-data",action="store_true")
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run",action="store_true")
    mode.add_argument("--execute",action="store_true")
    args=parser.parse_args(argv)
    try:
        config=load_multimodal_config(args.config,args.set)
        options=config.settings["analysis"]
        for argument,key in ((args.aggregation,"aggregation"),(args.bootstrap_replicates,"bootstrap_replicates"),
                             (args.bootstrap_seed,"bootstrap_seed"),(args.confidence,"confidence")):
            if argument is not None: options[key]=argument
        if args.matched_subset: options["matching"]="matched_subset"
        if args.strict_matching: options["matching"]="strict"
        if args.synthetic_test_data: options["synthetic"]=True
        validate_config(config.settings)
        paths={};series={}
        for source in args.input_manifests:
            if "=" in source: label,path=source.split("=",1)
            else:
                item=load_completed_run(source);label,path=item.label,source
            paths.setdefault(label,[]).append(path)
        for label,items in sorted(paths.items()): series[label]=load_series(items,label=label)
        for source in args.legacy_input:
            descriptor,path=source.split("=",1);label,model=descriptor.split(":",1)
            if label in series: raise AnalysisError("labels","Legacy and completed sources need distinct labels")
            series[label]=load_legacy_predictions(path,label=label,model=model)
        if args.models:
            if len(args.models)!=len(set(args.models)) or not set(args.models)<=series.keys():
                raise AnalysisError("models","Selected labels must exist once in the input collection")
            series={label:series[label] for label in args.models}
        if args.reference_model:
            if args.reference_model not in series: raise AnalysisError("reference","Requested reference label is absent")
            options["pairs"]=[[label,args.reference_model] for label in sorted(series) if label!=args.reference_model]
        validate_config(config.settings)
        if args.execute and not args.output_dir:
            raise AnalysisError("output_path","--execute requires an explicit --output-dir")
        plan=analysis_plan(config,series)
        print(json.dumps(plan,sort_keys=True,indent=2))
        if args.execute:
            result=analyze_predictions(config,series);directory=export_analysis(result,args.output_dir)
            print(json.dumps({"analysis_id":result.analysis_id,"directory":str(directory),"status":"reports_created"},sort_keys=True))
    except (ValueError,OSError,ImportError,RuntimeError) as exc:
        code=exc.code if isinstance(exc,AnalysisError) else "analysis_inputs"
        parser.exit(2,f"Analysis error [{code}]: invalid/incomplete/incompatible source or analysis configuration; no source artifact was modified\n")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
