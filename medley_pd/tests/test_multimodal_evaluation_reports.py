"""Prediction-only report/CLI integration against authored synthetic artifacts."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from test_multimodal_experiment_evaluation import completed_factory,adapter_files
from multimodal.experiment_evaluation import load_series,AnalysisError
from multimodal.evaluation_reports import analysis_plan,analyze_predictions,export_analysis
from multimodal.run_state import file_hash
from multimodal.evaluation_cli import main


@pytest.fixture
def collection(completed_factory):
    sources={};cfg=None
    for model,error in (("M0",2.),("M1",1.5),("M2",1.),("M3",.5)):
        paths=[]
        for seed in (0,1):
            root,cfg=completed_factory(model,seed=seed,error=error)
            paths.append(root)
        sources[model]=load_series(paths,label=model)
    cfg.settings["evaluation"].update(seeds=[0,1],folds=[1])
    return cfg,sources


def test_strict_plan_and_all_core_subset_comparisons(collection):
    cfg,sources=collection
    plan=analysis_plan(cfg,sources)
    assert len(plan["comparisons"])==6
    assert sum(r["status"]=="unavailable" for r in plan["comparisons"])==3
    cfg.settings["analysis"]["matching"]="matched_subset"
    result=analyze_predictions(cfg,sources)
    pairs=result.tables["pairwise_comparison"]
    assert len(pairs)==6 and set(pairs.status)=={"available_descriptive"}
    assert set(pairs.population_label)=={"matched subset"}
    assert (pairs.delta_MAE<0).all()
    assert "synthetic test data" in result.summary
    assert all(r["diagnostics"]["matched_rows"]==4 for r in result.manifest["matching_diagnostics"])
    assert all(r["n_unique_subjects"]==2 for r in result.tables["bootstrap_intervals"].to_dict("records"))


@pytest.mark.parametrize("output_format",["csv_json_markdown","json"])
def test_reproducible_export_integrity_and_overwrite(collection,tmp_path,output_format):
    cfg,sources=collection;cfg.settings["analysis"]["output_format"]=output_format
    before={p:d for source in sources.values() for p,d in source.sources.items()}
    result=analyze_predictions(cfg,sources)
    reversed_sources={label:load_series([run["source_root"] for _,run in reversed(list(item.runs.items()))],label=label)
                      for label,item in reversed(list(sources.items()))}
    again=analyze_predictions(cfg,reversed_sources)
    assert result.analysis_id==again.analysis_id
    for name in result.tables: pd.testing.assert_frame_equal(result.tables[name],again.tables[name])
    output=export_analysis(result,tmp_path / "reports")
    manifest=json.loads((output / "analysis_manifest.json").read_text())
    marker=json.loads((output / "analysis_complete.json").read_text())
    assert manifest["analysis_id"]==result.analysis_id
    assert all(file_hash(output / name)==digest for name,digest in marker["artifacts"].items())
    if output_format=="json":
        payload=json.loads((output / "analysis_result.json").read_text())
        assert len(payload["pairwise_comparison"])==6
    else:
        assert len(pd.read_csv(output / "metrics_by_run.csv"))==8
        assert "synthetic test data" in (output / "analysis_summary.md").read_text()
    with pytest.raises(FileExistsError): export_analysis(result,tmp_path / "reports")
    assert all(file_hash(path)==digest for path,digest in before.items())
    native=next(iter(sources["M1"].runs.values()))["c1_root"]
    with pytest.raises(AnalysisError): export_analysis(result,Path(native) / "forbidden")


def test_missing_scope_and_unsupported_interval_explicit(collection):
    cfg,sources=collection
    cfg.settings["evaluation"]["folds"]=[1,2]
    result=analyze_predictions(cfg,sources)
    assert set(result.tables["pairwise_comparison"].status)=={"unavailable"}
    assert set(result.tables["metrics_summary"].coverage)=={"partial_available_runs"}
    cfg.settings["evaluation"]["folds"]=[1]
    cfg.settings["analysis"].update(aggregation="mean_runs",subgroups=True)
    result=analyze_predictions(cfg,sources)
    assert "bootstrap_estimand" in set(result.tables["bootstrap_intervals"].reason)
    assert result.manifest["unavailable"][0]["component"]=="subgroups"
    cfg.settings["analysis"].update(aggregation="pooled_rows",min_subjects=5)
    result=analyze_predictions(cfg,sources)
    assert "insufficient_subjects" in set(result.tables["bootstrap_intervals"].reason)


def test_cli_default_no_writes_execute_and_no_model_imports(collection,tmp_path,capsys):
    cfg,sources=collection
    path=tmp_path / "analysis.json";path.write_text(json.dumps(cfg.to_dict()))
    inputs=[f"{label}={run['source_root']}" for label,item in sources.items() for run in item.runs.values()]
    output=tmp_path / "cli-reports"
    args=["--config",str(path),"--input-manifests",*inputs,"--output-dir",str(output)]
    assert main(args)==0 and not output.exists()
    assert '"execution": false' in capsys.readouterr().out
    assert main(args+["--execute"])==0
    assert len(list(output.glob("*/analysis_complete.json")))==1
    script="import sys; from multimodal.evaluation_cli import main; main(sys.argv[1:]); assert not any(n in sys.modules for n in ('torch','transformers','safetensors.torch'))"
    env={**os.environ,"PYTHONPATH":"src:medley_pd","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","CUDA_VISIBLE_DEVICES":""}
    done=subprocess.run([sys.executable,"-c",script,*args],env=env,capture_output=True,text=True)
    assert done.returncode==0,done.stderr
    with pytest.raises(SystemExit) as exc: main(["--config",str(path),"--input-manifests",*inputs,"--execute"])
    assert exc.value.code==2
