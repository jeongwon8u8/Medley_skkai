"""C3 prediction-only numerical fixtures; never construct/train a model."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from safetensors.numpy import save_file

from test_multimodal_data_adapter import files as adapter_files
from test_multimodal_experiment_protocol import setup_source
from multimodal.config import inference_config_digest
from multimodal.experiment_protocol import ExperimentProtocol,fingerprint
from multimodal.experiment_manifest import build_manifest
from multimodal.model_registry import model_config
from multimodal.run_state import RunStore,write_json,file_hash,_attempt_hashes,validate_completion
from multimodal.experiment_evaluation import (load_completed_run,load_series,load_legacy_predictions,
                                             validate_prediction_frame,AnalysisError)


@pytest.fixture
def completed_factory(adapter_files,tmp_path):
    """Authored C1/C2 artifact schemas, not actual trained model/performance evidence."""
    cfg,adapter,splitpath,ids=setup_source(adapter_files)
    cfg.settings["llm"].update(tuning="frozen",quantization="none",compute_dtype="float32",device="cpu")
    cfg.settings["training"].update(max_epochs=1,gradient_checkpointing=False)
    cfg.settings["analysis"].update(synthetic=True,bootstrap_replicates=32,min_subjects=2)
    cfg.settings["protocol"]["manifest_dir"]=str(tmp_path / "completed")
    def create(model="M1",seed=0,fold=1,error=1.,overrides=None):
        config=model_config(cfg,model)
        for path,value in (overrides or {}).items():
            node=config.settings;parts=path.split(".")
            for key in parts[:-1]: node=node[key]
            node[parts[-1]]=value
        protocol=ExperimentProtocol(config,adapter);prepared=protocol.prepare(seed=seed,fold=fold)
        manifest=build_manifest(protocol,seed=seed,fold=fold,model=model,
                                baseline_identifier="clinical:C1" if model=="M0" else None)
        store=RunStore(manifest,baseline_sha256=fingerprint("synthetic baseline bytes") if model=="M0" else None)
        with store.claim():
            rows=[]
            for sid in prepared.partition.test_subjects:
                record=adapter.get_subject(sid)
                rows.append(dict(seed=seed,fold=fold,subject=sid,subgroup=record.subgroup,
                    model={"M0":"clinical:C1","M1":"clinical_llm:C1","M2":"fusion_a_llm:C1","M3":"fusion_b_llm:C1"}[model],
                    y=record.moca,target=record.moca,yhat=record.moca+error))
            frame=pd.DataFrame(rows).loc[:,["seed","fold","subject","subgroup","model","y","yhat","target"]]
            frame.to_csv(store.attempt / "predictions.csv.gz",index=False)
            write_json(store.attempt / "metrics.json",dict(n=len(rows),MAE=abs(error),RMSE=abs(error)))
            if model=="M0":
                write_json(store.attempt / "baseline_audit.json",dict(source_sha256=store.identity["baseline_sha256"],training_provenance="not_verified"))
            else:
                checkpoint=store.attempt / "checkpoints" / "epoch-0001";checkpoint.mkdir(parents=True)
                save_file({"synthetic_fixture_weight":np.zeros((1,4),dtype=np.float32)},str(checkpoint / "trainable.safetensors"))
                runtime=config.to_dict();write_json(store.attempt / "runtime_config.json",runtime)
                write_json(checkpoint / "resolved_config.json",runtime)
                cpidentity=dict(format={"M1":"medley_m1_inference_v1","M2":"medley_m2_inference_v1","M3":"medley_m3_inference_v1"}[model],
                    config_digest=inference_config_digest(runtime),synthetic_frozen_digest=fingerprint("synthetic shared base"),
                    model_id=None,revision=None,mode="frozen",hidden_dim=4,
                    base_config_digest=fingerprint("synthetic tiny config"),tokenizer_digest=fingerprint("synthetic tokenizer"))
                write_json(checkpoint / "manifest.json",dict(identity=cpidentity,epochs_completed=1,optimizer_steps=1,
                    weights_sha256=file_hash(checkpoint / "trainable.safetensors")))
                write_json(store.attempt / "selection.json",dict(execution_id=store.execution_id,c1_run_id=manifest.run_id,
                    model=model,seed=seed,fold=fold,best_epoch=1,validation_MAE=1.,checkpoint="checkpoints/epoch-0001",
                    restoration="verified_exact_trainable_weights",test_used_for_selection=False))
                write_json(store.attempt / "history.json",[dict(epoch=1,training=dict(n=len(prepared.partition.train_subjects),huber=1.,optimizer_steps=1),
                    validation=dict(n=len(prepared.partition.validation_subjects),MAE=1.,RMSE=1.),improved=True,bad_epochs=0)])
            store.state("artifacts_validated_pending_completion")
            info=manifest.to_dict()["identity"]
            completion=dict(schema="medley_completion_v1",execution_identity=store.identity,execution_id=store.execution_id,
                model=model,seed=seed,fold=fold,prediction_model=rows[0]["model"],status="imported" if model=="M0" else "completed",
                manifest_sha256=file_hash(store.root.parent.parent / "manifest.json"),attempt=str(store.attempt.relative_to(store.root)),
                artifacts=_attempt_hashes(store.root,store.attempt))
            validate_completion(store.root,completion,verify_tensors=False)
            write_json(store.root / "completed.json",completion,exclusive=True)
        return store.root,config
    return create


def test_c31_completed_ingestion_readonly_and_repeated_runs(completed_factory):
    a,cfg=completed_factory(seed=0);b,_=completed_factory(seed=1)
    before=file_hash(a / "completed.json")
    series=load_series([b,a],label="variant_clinical")
    assert series.label=="variant_clinical" and len(series.frame)==4 and set(series.runs)=={(0,1),(1,1)}
    assert series.frame.subject.nunique()==2 and before==file_hash(a / "completed.json")
    series.assert_unchanged()
    with pytest.raises(AnalysisError): load_series([a,a])


@pytest.mark.parametrize("case",["missing_column","duplicate","nonfinite","target","subgroup","fold","subject"])
def test_c31_prediction_schema_rejections(completed_factory,case):
    root,_=completed_factory();frame=load_completed_run(root).frame
    if case=="missing_column": frame=frame.drop(columns="target")
    elif case=="duplicate": frame=pd.concat([frame,frame.iloc[:1]])
    elif case=="nonfinite": frame.loc[0,"yhat"]=float("nan")
    elif case=="target": frame.loc[0,"target"]+=1
    elif case=="subgroup": frame["subgroup"]=123
    elif case=="fold": frame["fold"]=True
    else: frame["subject"]=123
    with pytest.raises(ValueError): validate_prediction_frame(frame)


def test_c31_legacy_schema_and_invalid_completion(completed_factory,tmp_path):
    root,_=completed_factory("M0");loaded=load_completed_run(root)
    path=tmp_path / "legacy.csv";loaded.frame.to_csv(path,index=False)
    before=path.read_bytes()
    legacy=load_legacy_predictions(path,label="original_baseline",model="clinical:C1")
    assert legacy.runs=={} and len(legacy.frame)==2 and path.read_bytes()==before
    (root / "completed.json").unlink()
    with pytest.raises(ValueError): load_completed_run(root)


def test_c31_variant_and_source_integrity(completed_factory):
    a,_=completed_factory("M2",seed=0)
    b,_=completed_factory("M2",seed=1,overrides={"eeg.pooling.num_queries":7})
    with pytest.raises(AnalysisError): load_series([a,b],label="silently_mixed_variants")
    series=load_completed_run(a)
    path=next((a / "attempts").glob("*/predictions.csv.gz"));path.write_bytes(path.read_bytes()+b"corrupt")
    with pytest.raises(AnalysisError): series.assert_unchanged()
    with pytest.raises(ValueError): load_completed_run(a)


from multimodal.prediction_matching import match_predictions


def test_c32_strict_exact_matching_and_provenance_levels(completed_factory):
    a,_=completed_factory("M2");b,_=completed_factory("M3",error=.5)
    left,right=load_completed_run(a),load_completed_run(b)
    matched=match_predictions(left,right,synthetic=True)
    assert matched.diagnostics["matched_rows"]==2 and matched.conditions["recorded_protocol_comparable"]
    assert not matched.conditions["fully_controlled_configuration"]
    assert matched.frame.candidate_prediction.sub(matched.frame.target).tolist()==[1.,1.]
    with pytest.raises(AnalysisError): match_predictions(left,right,synthetic=False)


def test_c32_missing_runs_and_explicit_subset_counts(completed_factory):
    a,_=completed_factory("M1",seed=0);b,_=completed_factory("M2",seed=0)
    extra,_=completed_factory("M2",seed=1)
    left=load_series([a]);right=load_series([b,extra])
    with pytest.raises(AnalysisError): match_predictions(left,right,synthetic=True)
    result=match_predictions(left,right,matching="matched_subset",synthetic=True)
    assert result.diagnostics["population_label"]=="matched subset"
    assert result.diagnostics["candidate_rows"]==2 and result.diagnostics["reference_rows"]==4
    assert result.diagnostics["matched_rows"]==2 and result.diagnostics["excluded_reference_rows"]==2
    with pytest.raises(AnalysisError): match_predictions(left,load_series([b]),expected_runs={(0,1),(1,1)},synthetic=True)


@pytest.mark.parametrize("change",["target","subgroup","fold"])
def test_c32_exact_outcomes_reject_bad_metadata(completed_factory,change):
    from multimodal.experiment_evaluation import PredictionSeries
    a,_=completed_factory("M1");b,_=completed_factory("M2")
    left,right=load_completed_run(a),load_completed_run(b)
    frame=right.frame
    if change=="target": frame.loc[0,["y","target"]]+=1
    elif change=="subgroup": frame.loc[0,"subgroup"]="OTHER"
    else: frame["fold"]=2
    changed=PredictionSeries(right.label,frame,right.runs,right.sources)
    with pytest.raises(AnalysisError): match_predictions(left,changed,synthetic=True)


def test_c32_m0_provenance_cannot_be_promoted(completed_factory):
    a,_=completed_factory("M1");b,_=completed_factory("M0")
    left,right=load_completed_run(a),load_completed_run(b)
    with pytest.raises(AnalysisError): match_predictions(left,right,synthetic=True)
    matched=match_predictions(left,right,matching="matched_subset",synthetic=True)
    assert matched.conditions["checks"]["population"] and not matched.conditions["checks"]["training_protocol"]
    assert not matched.conditions["recorded_protocol_comparable"]


def test_c32_declared_configuration_ablation_stays_distinct(completed_factory):
    a,_=completed_factory("M2",overrides={"eeg.pooling.num_queries":1})
    b,_=completed_factory("M2",overrides={"eeg.pooling.num_queries":7})
    left,right=load_completed_run(a,label="K1"),load_completed_run(b,label="K7")
    with pytest.raises(AnalysisError): match_predictions(left,right,synthetic=True)
    result=match_predictions(left,right,synthetic=True,allowed_differences=["settings.eeg.pooling.num_queries"])
    assert result.conditions["allowed_differences"]==["settings.eeg.pooling.num_queries"]
    assert not result.conditions["fully_controlled_configuration"]


from multimodal.evaluation_metrics import core_metrics,summarize_series,paired_metrics


def test_c33_manual_mae_rmse_and_existing_medley_definition():
    from medley.metrics import regression_metrics
    y=np.array([20.,21.,22.]);p=np.array([21.,19.,22.])
    metrics=core_metrics(y,p)
    assert metrics["MAE"]==1. and metrics["RMSE"]==pytest.approx(np.sqrt(5/3))
    existing=regression_metrics(y,p)
    assert metrics=={k:existing[k] for k in ("n","MAE","RMSE")}
    for bad in ([float("nan"),21.,22.],[[20.,21.,22.]],np.array([True,False,True])):
        with pytest.raises(AnalysisError): core_metrics(bad,p)


def test_c33_aggregation_estimands_and_repeated_subjects():
    from multimodal.experiment_evaluation import PredictionSeries
    frame=pd.DataFrame([dict(seed=s,fold=f,subject=sid,subgroup=None,model="synthetic",y=20.,target=20.,yhat=20.+error)
        for s,f,sid,error in [(0,1,"a",1.),(0,1,"b",1.),(0,2,"c",4.),(1,1,"a",2.)]])
    series=PredictionSeries("custom_variant",frame,{}, {})
    tables=summarize_series(series)
    summary=tables["metrics_summary"].set_index("estimand")
    assert summary.loc["pooled_prediction_rows","MAE"]==2.
    assert summary.loc["pooled_prediction_rows","RMSE"]==pytest.approx(np.sqrt(22/4))
    assert summary.loc["unweighted_mean_run_metrics","MAE"]==pytest.approx(7/3)
    assert summary.loc["unweighted_mean_run_metrics","RMSE"]==pytest.approx(7/3)
    assert summary.loc["pooled_prediction_rows","n_prediction_rows"]==4
    assert summary.loc["pooled_prediction_rows","n_unique_subjects"]==3
    assert len(tables["metrics_by_run"])==3 and len(tables["metrics_by_seed"])==2 and len(tables["metrics_by_fold"])==2


def test_c33_pairwise_candidate_minus_reference(completed_factory):
    a,_=completed_factory("M2",error=.5);b,_=completed_factory("M1",error=2.)
    pair=match_predictions(load_completed_run(a),load_completed_run(b),synthetic=True)
    for policy in ("pooled_rows","mean_runs"):
        result=paired_metrics(pair,aggregation=policy)
        assert result["delta_MAE"]==-1.5 and result["delta_RMSE"]==-1.5
    reverse=match_predictions(load_completed_run(b),load_completed_run(a),synthetic=True)
    assert paired_metrics(reverse)["delta_MAE"]==1.5


def test_c3_review_intrinsic_settings_not_fully_controlled(completed_factory):
    from multimodal.experiment_evaluation import PredictionSeries
    from multimodal.prediction_matching import comparison_conditions
    root,_=completed_factory("M2")
    a=load_completed_run(root,label="projector_small")
    runs=copy.deepcopy(a.runs)
    run=next(iter(runs.values()))
    run["manifest"]["resolved_config"]["settings"]["fusion_a"]["projection_hidden_dim"]=987
    b=PredictionSeries("projector_large",a.frame,runs,a.sources)
    conditions=comparison_conditions(a,b,set(a.runs),synthetic=True)
    assert conditions["recorded_protocol_comparable"]
    assert not conditions["fully_controlled_configuration"]
    assert not conditions["intrinsic_model_settings_equal"]
