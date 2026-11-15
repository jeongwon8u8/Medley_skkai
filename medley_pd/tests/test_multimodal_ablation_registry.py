"""D1 pure config gates; no models, source writes or variant execution."""
import copy
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
from multimodal.config import CONFIG_DIR,load_multimodal_config,inference_config_digest
from multimodal.ablation_registry import resolve_variant


def config(): return load_multimodal_config(CONFIG_DIR / "fusion_b.json")


@pytest.mark.parametrize("model,family,value",[("M2","pooling","mean"),("M2","pooling","max"),
    ("M2","feature_adapter",True),("M3","soft_tokens",1),("M3","soft_tokens",8),
    ("M1","adaptation","frozen"),("M2","adaptation","lora"),("M2","numeric_skip",True),
    ("M1","clinical_set","C0"),("M3","clinical_set","C2"),("M3","encoder","scerebro")])
def test_d1_stable_ofat_and_no_mutation(model,family,value):
    cfg=config()
    if family=="pooling": cfg.settings["eeg"]["pooling"]["num_queries"]=1
    before=copy.deepcopy(cfg.to_dict())
    a=resolve_variant(cfg,model=model,family=family,value=value)
    b=resolve_variant(cfg,model=model,family=family,value=value)
    assert a.summary()==b.summary() and a.variant_id==b.variant_id
    a.assert_valid();assert cfg.to_dict()==before and a.changes
    if family=="encoder":
        assert "settings.eeg.feature_dim" in a.changes and "medley.encoder.name" in a.changes
    if family=="adaptation": assert a.candidate.settings["llm"]["quantization"]=="none"
    other=copy.deepcopy(a.candidate);other.settings["training"]["max_epochs"]+=1
    with pytest.raises(ValueError): resolve_variant(cfg,model=model,family=family,value=value,candidate=other)


@pytest.mark.parametrize("model,family,value",[("M0","clinical_set","C0"),("M1","pooling","mean"),
    ("M2","soft_tokens",8),("M3","numeric_skip",True),("M3","soft_tokens",7),("M2","unknown",True),
    ("M3","pooling","mean"),("M1","clinical_set","C1")])
def test_d1_invalid_or_confounded(model,family,value):
    with pytest.raises(ValueError): resolve_variant(config(),model=model,family=family,value=value)


def test_d1_inheritance_and_distinct_fingerprints(tmp_path):
    path=tmp_path / "inherited.json"
    path.write_text(json.dumps({"extends":str(CONFIG_DIR / "fusion_b.json"),"evaluation":{"seeds":[0],"folds":[1]}}))
    cfg=load_multimodal_config(path)
    a=resolve_variant(cfg,model="M3",family="soft_tokens",value=1)
    b=resolve_variant(cfg,model="M3",family="soft_tokens",value=8)
    assert a.variant_id!=b.variant_id and a.metadata["seed_scope"]==[0]
    digest=inference_config_digest(cfg.to_dict())
    legacy=cfg.to_dict();legacy["settings"].pop("ablation")
    assert inference_config_digest(legacy)==digest
