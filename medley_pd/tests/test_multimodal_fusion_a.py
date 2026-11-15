"""B1 independent representations, followed by B2 external fusion checks."""
import copy
import json

import pytest
import torch

from test_multimodal_fusion_b import backbone, inputs
from multimodal.representations import (MeanQueryAggregation, RepresentationProjection,
                                        EEGRepresentationEncoder, ClinicalRepresentationEncoder)
from multimodal.clinical_regressor import clinical_prediction_state
from multimodal.config import CONFIG_DIR, ConfigError, load_multimodal_config, save_resolved_config


@pytest.fixture(scope="module", autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def a_backbone(mode="frozen", side="right", family="gpt2", overrides=()):
    return backbone(mode, side, family, overrides=["architecture=fusion_a", "fusion.type=representation", *overrides])


@pytest.mark.parametrize("encoder,width", [("cbramod",3800),("scerebro",3420)])
@pytest.mark.parametrize("k", [1,4,7])
def test_b1_eeg_shapes_padding_flags_mean_and_gradients(encoder,width,k):
    cfg = load_multimodal_config(CONFIG_DIR / "fusion_a.json", [f"eeg.encoder={encoder}",f"eeg.feature_dim={width}",
        f"eeg.pooling.num_queries={k}"])
    branch = EEGRepresentationEncoder(cfg)
    data = inputs(width=width)
    eeg, mask, flags = data["eeg_features"].requires_grad_(True),data["eeg_mask"],data["artifact_flags"]
    before = eeg.detach().clone()
    pooled = branch.pooling(eeg,mask,artifact_flags=flags)
    assert pooled.shape == (3,k,256)
    torch.testing.assert_close(branch.aggregation(pooled), pooled.mean(1),rtol=0,atol=0)
    out = branch(eeg_features=eeg,eeg_mask=mask,artifact_flags=flags)
    assert out.shape == (3,256) and torch.isfinite(out).all()
    (out*torch.linspace(.1,1,256)).sum().backward()
    for p in branch.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
    assert torch.equal(eeg.grad[~mask],torch.zeros_like(eeg.grad[~mask]))
    assert eeg.grad[mask].abs().sum()>0 and torch.equal(eeg.detach(),before)
    altered=eeg.detach().clone();altered[~mask]=float("nan")
    torch.testing.assert_close(branch(eeg_features=altered,eeg_mask=mask,artifact_flags=~flags),out,rtol=0,atol=0)
    torch.testing.assert_close(branch(eeg_features=eeg[:1],eeg_mask=mask[:1],artifact_flags=flags[:1]),out[:1],atol=2e-6,rtol=2e-5)
    invalid=mask.clone();invalid[0]=False
    with pytest.raises(ValueError): branch(eeg_features=eeg,eeg_mask=invalid)


@pytest.mark.parametrize("side", ["left","right"])
@pytest.mark.parametrize("mode", ["frozen","lora"])
@pytest.mark.parametrize("family", ["gpt2","qwen2","gemma"])
def test_b1_clinical_suffix_text_only_projection_and_lora(side,mode,family):
    llm = a_backbone(mode,side,family)
    branch=ClinicalRepresentationEncoder(llm).eval()
    clinical=inputs()["clinical"]
    before=copy.deepcopy(clinical)
    texts=branch.serializer.serialize_batch(clinical)
    tokens=branch.tokenizer(texts)
    assert texts[-1]=="" and all("MoCA" not in text for text in texts)
    suffix=llm.tokenizer.encode("Predicted MoCA:",add_special_tokens=False)
    assert len(suffix)==3
    for row in range(3):
        actual=tokens.input_ids[row,tokens.attention_mask[row].bool()].tolist()
        assert actual[-3:]==suffix and actual.count(suffix[0])==1
    calls=[];states=[]
    hook=llm.model.register_forward_pre_hook(lambda m,a,kw:calls.append(kw),with_kwargs=True)
    post=llm.model.register_forward_hook(lambda m,a,o:states.append(o.hidden_states[-1]))
    hidden=branch.encode_hidden(clinical)
    hook.remove();post.remove()
    assert len(calls)==1 and calls[0]["inputs_embeds"] is None
    assert torch.equal(calls[0]["input_ids"],tokens.input_ids)
    torch.testing.assert_close(hidden,states[0][torch.arange(3),tokens.prediction_positions],rtol=0,atol=0)
    assert hidden.shape==(3,llm.hidden_dim) and llm.hidden_dim!=256
    out=branch(clinical)
    assert out.shape==(3,256) and torch.isfinite(out).all()
    out.square().sum().backward()
    assert branch.projection.linear.weight.grad.abs().sum()>0
    base=[p for n,p in llm.model.named_parameters() if "lora_" not in n]
    assert all(not p.requires_grad and p.grad is None for p in base)
    if mode=="lora":
        adapters=[p for n,p in llm.model.named_parameters() if "lora_" in n]
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapters)
        assert any(p.grad.abs().sum()>0 for p in adapters)
    assert clinical==before
    invalid=tokens.prediction_positions.clone();invalid[0]=0
    with pytest.raises(ValueError): clinical_prediction_state(llm,input_ids=tokens.input_ids,
        attention_mask=tokens.attention_mask,prediction_positions=invalid)


@pytest.mark.parametrize("key", ["moca","subject_id","subgroup","visit_id","ctt","updrs_total"])
def test_b1_clinical_leakage_rejected(key):
    branch=ClinicalRepresentationEncoder(a_backbone())
    clinical=inputs()["clinical"];clinical[0][key]=19.125
    with pytest.raises(ValueError): branch(clinical)


@pytest.mark.parametrize("case", ["max","empty","integer","nonfinite","projection_width","projection_device"])
def test_b1_invalid_components(case):
    if case=="max":
        with pytest.raises(ValueError): MeanQueryAggregation("max")
    elif case.startswith("projection"):
        value=torch.ones(2,3,device="meta" if case=="projection_device" else "cpu")
        with pytest.raises(ValueError): RepresentationProjection(4,5)(value)
    else:
        value={"empty":torch.ones(2,0,3),"integer":torch.ones(2,4,3,dtype=torch.long),
               "nonfinite":torch.full((2,4,3),float("nan"))}[case]
        with pytest.raises(ValueError): MeanQueryAggregation()(value)


@pytest.mark.parametrize("dtype", ["bfloat16","float16"])
def test_b1_clinical_hidden_cast_preserves_gradients(dtype):
    branch=ClinicalRepresentationEncoder(a_backbone(overrides=[f"llm.compute_dtype={dtype}"]))
    assert branch.encode_hidden(inputs()["clinical"]).dtype==getattr(torch,dtype)
    out=branch(inputs()["clinical"])
    assert out.dtype==torch.float32
    out.sum().backward()
    assert torch.isfinite(branch.projection.linear.weight.grad).all()


def test_b1_configuration_roundtrip_and_unsupported_options(tmp_path):
    child=tmp_path / "child.json"
    child.write_text(json.dumps({"extends":str(CONFIG_DIR / "fusion_a.json"),"fusion_a":{"eeg_projection_dim":32}}))
    cfg=load_multimodal_config(child,["fusion_a.clinical_projection_dim=48","fusion.fusion_dim=64","fusion.dropout=0.2"])
    assert cfg.settings["fusion_a"]["eeg_projection_dim"]==32
    assert cfg.settings["fusion_a"]["clinical_projection_dim"]==48
    snapshot=tmp_path / "snapshot.json";save_resolved_config(cfg,snapshot)
    assert load_multimodal_config(snapshot).to_dict()==cfg.to_dict()
    for option in ["fusion_a.query_aggregation=max","fusion_a.layers=3","fusion_a.layer_norm=false",
        "fusion_a.activation=relu","fusion.numeric_skip=true","fusion_a.eeg_projection_dim=0",
        "fusion_a.clinical_projection_dim=true","fusion_a.lr_pooling=-1"]:
        with pytest.raises(ConfigError): load_multimodal_config(child,[option])


from multimodal.fusion_a import EEGClinicalRepresentationFusionRegressor,RepresentationFusionHead
from multimodal.clinical_regressor import moca_huber_loss


def make_model(mode="frozen",side="right",overrides=()):
    return EEGClinicalRepresentationFusionRegressor(a_backbone(mode,side,overrides=overrides))


@pytest.mark.parametrize("mode", ["frozen","lora"])
@pytest.mark.parametrize("side", ["left","right"])
@pytest.mark.parametrize("batch", [1,3])
def test_b2_external_fusion_shapes_order_mlp_gradient_and_no_mutation(mode,side,batch):
    model=make_model(mode,side).eval()
    data=inputs()
    if batch==1: data={k:v[1:2] for k,v in data.items()}
    before=copy.deepcopy(data)
    calls=[]
    hook=model.llm.register_forward_pre_hook(lambda m,a,kw:calls.append(kw),with_kwargs=True)
    reps=model.extract_representations(**data)
    hook.remove()
    assert reps.eeg.shape==reps.clinical.shape==(batch,256)
    assert len(calls)==1 and calls[0]["inputs_embeds"] is None
    text=model.tokenizer(model.serializer.serialize_batch(data["clinical"]))
    assert torch.equal(calls[0]["input_ids"],text.input_ids)
    combined=model.head.concatenate(reps.eeg,reps.clinical)
    assert combined.shape==(batch,512)
    torch.testing.assert_close(combined[:,:256],reps.eeg)
    torch.testing.assert_close(combined[:,256:],reps.clinical)
    assert model.head.norm.normalized_shape==(512,)
    assert model.head.mlp[0].weight.shape==(256,512) and model.head.mlp[3].weight.shape==(1,256)
    assert isinstance(model.head.mlp[1],torch.nn.GELU) and model.head.mlp[2].p==.1
    pred=model(**data)
    assert pred.shape==(batch,)
    moca_huber_loss(pred,torch.arange(batch).float()+19.125).backward()
    for name,p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(),name
            if "lora_" not in name: assert p.grad.abs().sum()>0,name
        else: assert p.grad is None
    if mode=="lora": assert any(p.grad.abs().sum()>0 for n,p in model.named_parameters() if "lora_" in n)
    for key in data:
        if isinstance(data[key],torch.Tensor): assert torch.equal(data[key],before[key])
        else: assert data[key]==before[key]
    with torch.no_grad(): model.head.mlp[3].weight.zero_();model.head.mlp[3].bias.fill_(42.)
    torch.testing.assert_close(model(**data),torch.full((batch,),42.),rtol=0,atol=0)


@pytest.mark.parametrize("case", ["batch","width","rank","nonfinite","device","integer"])
def test_b2_fusion_input_rejection(case):
    head=RepresentationFusionHead(4,5,6)
    a,b=torch.ones(2,4),torch.ones(2,5)
    if case=="batch": b=b[:1]
    elif case=="width": b=b[:,:4]
    elif case=="rank": a=a[None]
    elif case=="nonfinite": a[0,0]=float("inf")
    elif case=="device": a=a.to("meta")
    elif case=="integer": a=a.long()
    with pytest.raises(ValueError): head(a,b)


def test_b2_both_independent_modalities_affect_prediction():
    model=make_model(overrides=["fusion.dropout=0"]).eval()
    data=inputs()
    reps=model.extract_representations(**data)
    changed_eeg=copy.deepcopy(data);changed_eeg["eeg_features"][data["eeg_mask"]]+=2
    changed_clinical=copy.deepcopy(data);changed_clinical["clinical"][0]["age"]=35
    eeg_reps=model.extract_representations(**changed_eeg)
    clinical_reps=model.extract_representations(**changed_clinical)
    torch.testing.assert_close(eeg_reps.clinical,reps.clinical,rtol=0,atol=0)
    torch.testing.assert_close(clinical_reps.eeg,reps.eeg,rtol=0,atol=0)
    assert not torch.allclose(eeg_reps.eeg,reps.eeg)
    assert not torch.allclose(clinical_reps.clinical[0],reps.clinical[0],atol=1e-8,rtol=0)
    # Controlled first hidden unit reads one coordinate from each normalized branch.
    with torch.no_grad():
        model.head.mlp[0].weight.zero_();model.head.mlp[0].bias.zero_()
        model.head.mlp[0].weight[0,0]=1;model.head.mlp[0].weight[0,256]=1
        model.head.mlp[3].weight.zero_();model.head.mlp[3].bias.zero_();model.head.mlp[3].weight[0,0]=1
    base=model(**data)
    assert not torch.allclose(model(**changed_eeg),base,atol=1e-8,rtol=0)
    assert not torch.allclose(model(**changed_clinical),base,atol=1e-8,rtol=0)


def test_b2_configured_dimensions_and_model_errors():
    model=make_model(overrides=["eeg.pooling.hidden_dim=32","eeg.pooling.num_heads=8",
        "fusion_a.eeg_projection_dim=17","fusion_a.clinical_projection_dim=23","fusion.fusion_dim=31","fusion.dropout=0.2"])
    reps=model.extract_representations(**inputs())
    assert reps.eeg.shape==(3,17) and reps.clinical.shape==(3,23)
    assert model.head.norm.normalized_shape==(40,) and model.head.mlp[0].weight.shape==(31,40)
    assert model.head.mlp[2].p==.2
    assert model(**inputs()).shape==(3,)
    data=inputs();data["clinical"]=data["clinical"][:1]
    with pytest.raises(ValueError): model(**data)
    with pytest.raises(ValueError): EEGClinicalRepresentationFusionRegressor(backbone())
    with pytest.raises(NotImplementedError): make_model(overrides=["eeg.feature_adapter.enabled=true"])
    with pytest.raises(TypeError): model(**inputs(),moca=torch.ones(3))


@pytest.mark.parametrize("side", ["left","right"])
def test_b2_eos_as_pad_and_padding_batch_equivalence(side):
    llm=a_backbone(side=side);llm.tokenizer.pad_token=llm.tokenizer.eos_token
    model=EEGClinicalRepresentationFusionRegressor(llm).eval()
    data=inputs();pred=model(**data)
    for row in range(3):
        single={k:v[row:row+1] for k,v in data.items()}
        torch.testing.assert_close(model(**single)[0],pred[row],rtol=2e-5,atol=2e-6)
    text=model.tokenizer(model.serializer.serialize_batch(data["clinical"]))
    assert all(text.input_ids[i,p]!=llm.tokenizer.eos_token_id for i,p in enumerate(text.prediction_positions))


@pytest.mark.parametrize("feature_set",["C0","C1","C2"])
def test_b4_clinical_sets_and_effective_model_snapshot(feature_set,tmp_path):
    model=make_model(overrides=[f"clinical_set={feature_set}"])
    data=inputs();fields=model.backbone.resolved_config.clinical_variables
    data["clinical"]=[{key:row.get(key) for key in fields} for row in data["clinical"]]
    assert model(**data).shape==(3,)
    saved=tmp_path / "resolved.json";save_resolved_config(model.backbone.resolved_config,saved)
    assert load_multimodal_config(saved).to_dict()==model.backbone.resolved_config.to_dict()


def test_b4_head_and_projection_explicit_dtype_conversion_and_rejection():
    head=RepresentationFusionHead(4,5,6,dropout=0)
    a=torch.randn(2,4,dtype=torch.bfloat16,requires_grad=True)
    b=torch.randn(2,5,dtype=torch.float64,requires_grad=True)
    pred=head(a,b)
    assert pred.dtype==torch.float32
    pred.sum().backward()
    assert torch.isfinite(a.grad).all() and a.grad.abs().sum()>0
    assert torch.isfinite(b.grad).all() and b.grad.abs().sum()>0
    with pytest.raises(ValueError): head(torch.full((2,4),1e100,dtype=torch.float64),b)
    with pytest.raises(ValueError): RepresentationProjection(4,5)(torch.full((2,4),1e100,dtype=torch.float64))
    for args in [(True,5,6,.1),(4,0,6,.1),(4,5,6,float("nan")),(4,5,6,1.)]:
        with pytest.raises(ValueError): RepresentationFusionHead(*args)


def test_b4_rejects_unfrozen_backbone_and_new_modules_stay_fp32():
    llm=a_backbone();llm.embedding_layer.weight.requires_grad_(True)
    with pytest.raises(ValueError): EEGClinicalRepresentationFusionRegressor(llm)
    llm=a_backbone();previous=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        model=EEGClinicalRepresentationFusionRegressor(llm)
    finally: torch.set_default_dtype(previous)
    assert all(p.dtype==torch.float32 for n,p in model.named_parameters() if ".llm." not in n)
    assert model(**inputs()).shape==(3,)
