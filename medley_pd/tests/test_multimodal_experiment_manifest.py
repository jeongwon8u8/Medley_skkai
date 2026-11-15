"""C1 canonical identities, protected manifests and compatibility tests."""
import copy
import json
from pathlib import Path

import pytest

from test_multimodal_experiment_protocol import setup_source
from test_multimodal_data_adapter import files as adapter_files
from multimodal.experiment_protocol import ExperimentProtocol,ProtocolError,fingerprint,canonical_json
from multimodal.experiment_manifest import build_manifest,save_manifest,read_manifest,validate_manifest,REPOSITORY
from multimodal.config import CONFIG_DIR,ConfigError,load_multimodal_config,save_resolved_config


def protocol(adapter_files,tmp_path):
    cfg,adapter,path,ids=setup_source(adapter_files)
    cfg.settings["protocol"]["manifest_dir"]=str(tmp_path / "private_runs")
    return ExperimentProtocol(cfg,adapter)


def test_c14_stable_private_manifest_and_public_whitelist(adapter_files,tmp_path):
    run=protocol(adapter_files,tmp_path)
    a,b=[build_manifest(run,seed=0,fold=1) for _ in range(2)]
    assert a.to_json()==b.to_json() and a.run_id==b.run_id
    public=a.public_metadata();text=canonical_json(public)
    assert public["counts"]==dict(train=6,validation=2,test=2)
    assert all(sid not in text for sid in run.outer.subjects)
    assert "subjects" not in public and "resolved_config" not in public and str(tmp_path) not in text
    payload=a.to_dict()
    assert "target" not in payload["subjects"] and "clinical" not in payload
    assert payload["identity"]["extraction_history"]=="not_verified"
    assert payload["identity"]["llm_revision_status"]=="unspecified"
    path=save_manifest(a)
    assert path.stat().st_mode & 0o777==0o600
    assert path.parent.stat().st_mode & 0o777==0o700
    assert read_manifest(path)==payload
    with pytest.raises(FileExistsError): save_manifest(a)
    payload["subjects"]["train"].clear()
    assert a.to_dict()["subjects"]["train"]


@pytest.mark.parametrize("setting,value",[("clinical_set","C0"),("eeg.pooling.num_queries",7),("training.lr_new_modules",.002),
    ("training.validation_fraction",.3)])
def test_c14_material_config_changes_identity(adapter_files,tmp_path,setting,value):
    run=protocol(adapter_files,tmp_path)
    original=build_manifest(run,seed=0,fold=1)
    cfg=copy.deepcopy(run.config);parts=setting.split(".");node=cfg.settings
    for key in parts[:-1]: node=node[key]
    node[parts[-1]]=value
    if setting=="clinical_set":
        # Matching adapter configuration is required; extra predictor policy is explicit.
        cfg.settings["clinical_text"]["extra_approved_policy"]="ignore"
    other=ExperimentProtocol(cfg,run.adapter)
    changed=build_manifest(other,seed=0,fold=1)
    assert original.run_id!=changed.run_id
    if setting!="training.validation_fraction":
        assert original.to_dict()["subjects"]==changed.to_dict()["subjects"]


def test_c14_m0_requires_baseline_name_and_never_certifies_historical_protocol(adapter_files,tmp_path):
    run=protocol(adapter_files,tmp_path)
    with pytest.raises(ProtocolError): build_manifest(run,seed=0,fold=1,model="M0")
    manifest=build_manifest(run,seed=0,fold=1,model="M0",baseline_identifier="clinical:C1")
    identity=manifest.to_dict()["identity"]
    assert identity["architecture"]=="legacy_numerical"
    assert identity["baseline_training_provenance"]=="not_verified"
    assert identity["inner_policy"]["baseline_scope"]=="proposed_only_not_historical_baseline_verification"
    with pytest.raises(ProtocolError): build_manifest(run,seed=0,fold=1,model="M2")


@pytest.mark.parametrize("case",["ids","counts","config","run_id","namespace","extra_field"])
def test_c14_corruption_rejected(adapter_files,tmp_path,case):
    run=protocol(adapter_files,tmp_path);payload=build_manifest(run,seed=0,fold=1).to_dict()
    if case=="ids": payload["subjects"]["train"][0]=payload["subjects"]["test"][0]
    elif case=="counts": payload["identity"]["counts"]["train"]+=1
    elif case=="config": payload["resolved_config"]["settings"]["clinical_set"]="C0"
    elif case=="run_id": payload["run_id"]="forged"
    elif case=="namespace": payload["output_namespace"]=str(tmp_path / "elsewhere")
    else: payload["clinical_records"]=["not allowed"]
    with pytest.raises(ProtocolError): validate_manifest(payload)


def test_c14_private_output_policy_and_source_mutation(adapter_files,tmp_path):
    run=protocol(adapter_files,tmp_path)
    for root in (REPOSITORY / "docs",run.outer.path.parent):
        run.config.settings["protocol"]["manifest_dir"]=str(root)
        candidate=ExperimentProtocol(run.config,run.adapter)
        manifest=build_manifest(candidate,seed=0,fold=1)
        with pytest.raises(ProtocolError): save_manifest(manifest)
    run.outer.path.write_bytes(run.outer.path.read_bytes()+b"\n")
    with pytest.raises(ProtocolError): build_manifest(run,seed=0,fold=1)


def test_c14_config_inheritance_legacy_and_invalid_options(tmp_path):
    child=tmp_path / "child.json"
    child.write_text(json.dumps({"extends":str(CONFIG_DIR / "fusion_b.json"),"protocol":{"strict_comparability":False}}))
    cfg=load_multimodal_config(child,["training.validation_fraction=0.25","protocol.manifest_dir=outputs/c1"])
    assert cfg.settings["protocol"]["strict_comparability"] is False and cfg.settings["training"]["validation_fraction"]==.25
    path=tmp_path / "snapshot.json";save_resolved_config(cfg,path)
    assert load_multimodal_config(path).to_dict()==cfg.to_dict()
    legacy=cfg.to_dict();del legacy["settings"]["protocol"];path.write_text(json.dumps(legacy))
    assert load_multimodal_config(path).settings["protocol"]["strict_comparability"] is True
    for option in ["protocol.inner_policy=random","protocol.cohort_policy=intersection","protocol.strict_comparability=1",
                   "protocol.manifest_dir=null","protocol.target_atol=-1"]:
        with pytest.raises(ConfigError): load_multimodal_config(child,[option])
    for preset in ("base","clinical_only","fusion_a","fusion_b"):
        assert load_multimodal_config(CONFIG_DIR / f"{preset}.json").settings["protocol"]["inner_policy"]=="sha256_rank_v1"


@pytest.mark.parametrize("change",["model","seed","fold","clinical_set","eeg","llm","inner_seed","inner_fraction","labels"])
def test_review_semantic_manifest_inconsistencies_rejected_even_with_new_id(adapter_files,tmp_path,change):
    run=protocol(adapter_files,tmp_path);payload=build_manifest(run,seed=0,fold=1).to_dict()
    identity=payload["identity"]
    if change=="model": identity["model"]="M2"
    elif change=="seed": identity["seed"]=999
    elif change=="fold": identity["fold"]=99
    elif change=="clinical_set": identity["clinical_set"]="C0"
    elif change=="eeg": identity["eeg"]["feature_dim"]=5
    elif change=="llm": identity["llm"]["tuning"]="lora"
    elif change=="inner_seed": identity["inner_policy"]["derived_seed"]+=1
    elif change=="inner_fraction": identity["inner_policy"]["fraction"]=.8
    else: identity["labels"]=[20]
    payload["run_id"]="medley-"+fingerprint(identity)
    payload["output_namespace"]=str(Path(identity["output_root"]) / payload["run_id"])
    with pytest.raises(ProtocolError): validate_manifest(payload)


def test_manifest_relative_paths_bound_before_cwd_changes(adapter_files,tmp_path,monkeypatch):
    cfg,adapter,path,ids=setup_source(adapter_files)
    monkeypatch.chdir(tmp_path)
    cfg.settings["protocol"]["manifest_dir"]="private_runs"
    run=ExperimentProtocol(cfg,adapter)
    before=build_manifest(run,seed=0,fold=1)
    another=tmp_path / "other_cwd";another.mkdir();monkeypatch.chdir(another)
    after=build_manifest(run,seed=0,fold=1)
    assert before.to_json()==after.to_json()
    assert Path(after.to_dict()["output_namespace"]).parent==tmp_path / "private_runs"
