"""D3 bounded synthetic C1→C2→C3 acceptance; no network/large pretrained models."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from test_multimodal_data_adapter import files as adapter_files
from test_multimodal_experiment_runner import execution_inputs,tiny_factory,threads
from multimodal.ablation_registry import resolve_variant
from multimodal.ablation_runner import plan_ablation,execute_ablation,find_completed,compare_ablation,main
from multimodal.data_adapter import MedleyDataAdapter
from multimodal.run_state import file_hash


def setup(adapter_files,tmp_path,model="M2"):
    cfg,_,_=execution_inputs(adapter_files,tmp_path,model)
    cfg.settings["training"]["max_epochs"]=1
    cfg.settings["analysis"].update(synthetic=True,min_subjects=2,bootstrap_replicates=12)
    cfg.settings["evaluation"].update(seeds=[0],folds=[1])
    cfg.settings["eeg"]["pooling"]["num_queries"]=1
    return cfg


def test_d3_planning_cli_and_execution_guards(adapter_files,tmp_path):
    cfg=setup(adapter_files,tmp_path)
    variant=resolve_variant(cfg,model="M2",family="feature_adapter",value=True)
    output=tmp_path / "not-created"
    plan=plan_ablation(variant,seeds=[0],folds=[1],output_dir=output)
    assert plan==plan_ablation(variant,seeds=[0],folds=[1],output_dir=output)
    summary=execute_ablation(plan)
    assert not output.exists() and summary["estimated_runs"]==2 and not summary["execution"]
    def forbidden(cfg): raise AssertionError("guard allowed data/model read")
    for roles,limit in ((None,1),(["reference","candidate"],1),(["candidate","candidate"],2)):
        with pytest.raises(ValueError): execute_ablation(plan,execute=True,roles=roles,max_runs=limit,adapter_factory=forbidden)
    config=tmp_path / "config.json";config.write_text(json.dumps(cfg.to_dict()))
    args=["--config",str(config),"--model","M2","--family","feature_adapter","--value","true","--seed","0","--fold","1","--output-dir",str(output)]
    env={**os.environ,"PYTHONPATH":"src:medley_pd","CUDA_VISIBLE_DEVICES":"","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"}
    script="import sys; from multimodal.ablation_runner import main; main(sys.argv[1:]); assert not any(n in sys.modules for n in ('torch','transformers'))"
    done=subprocess.run([sys.executable,"-c",script,*args],env=env,capture_output=True,text=True)
    assert done.returncode==0,done.stderr
    assert not output.exists()
    with pytest.raises(SystemExit): main(args+["--execute"])


@pytest.mark.parametrize("model,family,value",[("M2","pooling","mean"),("M2","pooling","max"),
    ("M2","feature_adapter",True),("M3","soft_tokens",4),("M3","soft_tokens",8),
    ("M1","adaptation","lora"),("M2","clinical_set","C0"),("M2","numeric_skip",True),
    ("M2","encoder","scerebro")])
def test_d3_bounded_synthetic_variants_checkpoint_export_compare(model,family,value,adapter_files,tmp_path,tiny_factory):
    cfg=setup(adapter_files,tmp_path,model)
    if family=="encoder":
        # Authored fixture data only: a separate existing-width synthetic cache.
        cache=adapter_files[2];alternate=cache.parent / "scerebro";alternate.mkdir()
        for path in cache.glob("*.npz"):
            with np.load(path) as record:
                np.savez(alternate / path.name,feat=record["feat"][:,:3420],flag=record["flag"])
    variant=resolve_variant(cfg,model=model,family=family,value=value)
    plan=plan_ablation(variant,seeds=[0],folds=[1],output_dir=tmp_path / "runs")
    split=adapter_files[3];before=file_hash(split)
    results=execute_ablation(plan,execute=True,roles=["reference","candidate"],max_runs=2,
        adapter_factory=MedleyDataAdapter,backbone_factory=tiny_factory)
    assert len(results)==2 and all(not result.reused for result in results)
    assert results[0].directory!=results[1].directory and file_hash(split)==before
    for result in results:
        attempt=result.directory / result.completion["attempt"]
        frame=pd.read_csv(attempt / "predictions.csv.gz")
        assert list(frame)==["seed","fold","subject","subgroup","model","y","yhat","target"] and len(frame)==2
        assert result.completion["status"]=="completed"
    reference,candidate=results
    report=compare_ablation(variant,reference_paths=[reference.directory],candidate_paths=[candidate.directory],
        analysis_config=cfg,output_dir=tmp_path / "reports")
    pair=report.tables["pairwise_comparison"].iloc[0]
    assert pair.status=="available_descriptive",report.manifest["comparisons"]
    assert report.manifest["ablation"]["family"]==family and "synthetic test data" in report.summary
    counts=report.manifest["parameter_counts"]
    assert all(n>0 for values in counts.values() for n in values)
    assert len(list((tmp_path / "reports").glob("*/analysis_complete.json")))==1
    # Reuse never calls a model factory; both C1 inputs are checked again.
    def forbidden(cfg): raise AssertionError("completed source constructed a model")
    reuse_plan=plan_ablation(variant,seeds=[0],folds=[1],output_dir=tmp_path / "other-output",search_roots=[tmp_path / "runs"])
    assert find_completed(reuse_plan.reference_runs[0],reuse_plan.search_roots) is not None
    reused=execute_ablation(reuse_plan,execute=True,roles=["reference","candidate"],max_runs=2,
        adapter_factory=MedleyDataAdapter,backbone_factory=forbidden)
    assert all(r.reused for r in reused) and not (tmp_path / "other-output").exists()
    # Corrupt a relevant completion artifact: fail, do not silently retrain.
    marker=reference.directory / "completed.json";marker.write_text("{}")
    with pytest.raises(ValueError): find_completed(reuse_plan.reference_runs[0],reuse_plan.search_roots)


def test_d3_input_cohort_mismatch_stops_before_models(adapter_files,tmp_path,tiny_factory):
    from dataclasses import replace
    cfg=setup(adapter_files,tmp_path)
    variant=resolve_variant(cfg,model="M2",family="feature_adapter",value=True)
    plan=plan_ablation(variant,seeds=[0],folds=[1],output_dir=tmp_path / "runs")
    def factory(config):
        adapter=MedleyDataAdapter(config)
        if config.settings["eeg"]["feature_adapter"]["enabled"]:
            original=adapter.get_subject
            adapter.get_subject=lambda sid:replace(original(sid),moca=original(sid).moca-1)
        return adapter
    def forbidden(config): raise AssertionError("cohort failure loaded model")
    with pytest.raises(ValueError): execute_ablation(plan,execute=True,roles=["candidate"],adapter_factory=factory,backbone_factory=forbidden)
    assert not (tmp_path / "runs").exists()


def test_d3_reuse_rejects_code_environment_change(adapter_files,tmp_path,tiny_factory,monkeypatch):
    from multimodal import experiment_manifest
    cfg=setup(adapter_files,tmp_path)
    variant=resolve_variant(cfg,model="M2",family="numeric_skip",value=True)
    plan=plan_ablation(variant,seeds=[0],folds=[1],output_dir=tmp_path / "runs")
    execute_ablation(plan,execute=True,roles=["reference"],adapter_factory=MedleyDataAdapter,backbone_factory=tiny_factory)
    current=experiment_manifest.runtime_metadata
    def changed():
        value=current();value["application_source_fingerprint"]="different";return value
    monkeypatch.setattr(experiment_manifest,"runtime_metadata",changed)
    with pytest.raises(ValueError,match="code/environment"):
        find_completed(plan.reference_runs[0],plan.search_roots)


def test_d_review_numeric_provenance_detects_raw_changes_hidden_by_text(adapter_files,tmp_path):
    from dataclasses import replace
    from multimodal.experiment_protocol import ExperimentProtocol
    from multimodal.clinical_serializer import ClinicalTextSerializer
    cfg=setup(adapter_files,tmp_path);cfg.settings["clinical_text"]["numeric_significant_digits"]=1
    variant=resolve_variant(cfg,model="M2",family="numeric_skip",value=True)
    baseline=variant.reference;adapter=MedleyDataAdapter(baseline)
    before=ExperimentProtocol(baseline,adapter).prepare(seed=0,fold=1).cohort
    original=adapter.get_subject;serializer=ClinicalTextSerializer(baseline)
    sid=adapter.subject_ids[0]
    first=original(sid);changed=replace(first,clinical={**first.clinical,"age":float(first.clinical["age"])+.01})
    assert serializer.serialize(first.clinical)==serializer.serialize(changed.clinical)
    adapter.get_subject=lambda key:changed if key==sid else original(key)
    after=ExperimentProtocol(baseline,adapter).prepare(seed=0,fold=1).cohort
    assert before.cohort_fingerprint==after.cohort_fingerprint
    assert before.clinical_fingerprint!=after.clinical_fingerprint
    # OFF/ON record the same underlying input source when predictors are fixed.
    same=ExperimentProtocol(variant.candidate,MedleyDataAdapter(variant.candidate)).prepare(seed=0,fold=1).cohort
    assert before.clinical_fingerprint==same.clinical_fingerprint
