"""D2 independently verified architecture alternatives; tiny local models only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
import torch
from multimodal.eeg_pooling import MaskedReductionPooling


@pytest.fixture(scope="module",autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("kind,expected",[("mean",[-3.,-4.]),("max",[-2.,-3.])])
def test_d2_masked_reductions_exact_padding_gradient(kind,expected):
    pool=MaskedReductionPooling(2,hidden_dim=2,kind=kind)
    with torch.no_grad(): pool.input_projection.weight.copy_(torch.eye(2));pool.input_projection.bias.zero_()
    x=torch.tensor([[[-2.,-5.],[float('nan'),float('inf')],[-4.,-3.]]],requires_grad=True)
    mask=torch.tensor([[True,False,True]])
    out=pool(x,mask,artifact_flags=mask)
    torch.testing.assert_close(out,torch.tensor([[expected]]))
    expanded=torch.cat([x.detach(),torch.full((1,2,2),float('nan'))],dim=1)
    torch.testing.assert_close(pool(expanded,torch.cat([mask,torch.zeros(1,2,dtype=torch.bool)],1)),out)
    torch.testing.assert_close(pool(x,mask,artifact_flags=~mask),out)
    out.sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad[0,1].eq(0).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in pool.parameters())
    with pytest.raises(ValueError): pool(x,torch.zeros_like(mask))


@pytest.mark.parametrize("model",["M2","M3"])
@pytest.mark.parametrize("kind",["mean","max"])
def test_d2_reduction_model_integration(model,kind):
    from test_multimodal_fusion_b import backbone,inputs
    from multimodal.fusion_a import EEGClinicalRepresentationFusionRegressor
    from multimodal.fusion_b import EEGClinicalSoftTokenRegressor
    arch,fusion=("fusion_a","representation") if model=="M2" else ("fusion_b","soft_token")
    llm=backbone(overrides=[f"architecture={arch}",f"fusion.type={fusion}","ablation.enabled=true",
        f"eeg.pooling.type={kind}","eeg.pooling.num_queries=1"])
    cls=EEGClinicalRepresentationFusionRegressor if model=="M2" else EEGClinicalSoftTokenRegressor
    instance=cls(llm);prediction=instance(**inputs())
    assert prediction.shape==(3,) and torch.isfinite(prediction).all()
    prediction.sum().backward()
    assert instance.pooling.input_projection.weight.grad is not None


def test_d2_adapter_identity_initialization_mask_and_gradients():
    from multimodal.eeg_pooling import EEGFeatureAdapter
    off=EEGFeatureAdapter(3);on=EEGFeatureAdapter(3,enabled=True)
    x=torch.randn(2,3,3,requires_grad=True);mask=torch.tensor([[True,False,True],[True,True,False]])
    assert off(x) is x and not list(off.parameters())
    torch.testing.assert_close(on(x),x)
    bad=x.detach().clone();bad[~mask]=float('nan')
    bad.requires_grad_()
    initial = on(bad, eeg_mask=mask)
    torch.testing.assert_close(initial, x.masked_fill(~mask.unsqueeze(-1), 0.), rtol=0, atol=0)
    initial.square().sum().backward()
    assert all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for p in on.parameters())
    assert torch.count_nonzero(on.up.weight.grad) > 0
    assert torch.count_nonzero(bad.grad[~mask]) == 0
    with torch.no_grad():
        on.up.bias.fill_(1.)
    updated = on(bad, eeg_mask=mask)
    torch.testing.assert_close(updated[mask], x[mask] + 1.)
    assert torch.count_nonzero(updated[~mask]) == 0
    with pytest.raises(ValueError): on(torch.ones(1,2,4))


@pytest.mark.parametrize("model",["M2","M3"])
def test_d2_adapter_model_optimizer_and_checkpoint(model,tmp_path):
    from test_multimodal_fusion_b import backbone,inputs
    from multimodal.fusion_a import EEGClinicalRepresentationFusionRegressor
    from multimodal.fusion_b import EEGClinicalSoftTokenRegressor
    from multimodal import fusion_a_trainer,fusion_b_trainer,fusion_a_checkpoint,fusion_b_checkpoint
    arch,fusion=("fusion_a","representation") if model=="M2" else ("fusion_b","soft_token")
    options=[f"architecture={arch}",f"fusion.type={fusion}","ablation.enabled=true","eeg.feature_adapter.enabled=true"]
    instance=(EEGClinicalRepresentationFusionRegressor if model=="M2" else EEGClinicalSoftTokenRegressor)(backbone(overrides=options))
    optimizer=(fusion_a_trainer if model=="M2" else fusion_b_trainer).build_optimizer(instance)
    params=[id(p) for g in optimizer.param_groups for p in g["params"]]
    assert len(params)==len(set(params)) and set(params)=={id(p) for p in instance.parameters() if p.requires_grad}
    batch=inputs();batch["eeg_features"][~batch["eeg_mask"]]=float('nan')
    instance(**batch).square().mean().backward()
    adapter=instance.eeg_encoder.feature_adapter if model=="M2" else instance.feature_adapter
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters())
    optimizer.step()
    cp=fusion_a_checkpoint if model=="M2" else fusion_b_checkpoint
    cp.save_inference_checkpoint(instance,tmp_path / model)
    cp.load_inference_checkpoint(instance,tmp_path / model)
    disabled=backbone(overrides=[f"architecture={arch}",f"fusion.type={fusion}","ablation.enabled=true"])
    other=(EEGClinicalRepresentationFusionRegressor if model=="M2" else EEGClinicalSoftTokenRegressor)(disabled)
    with pytest.raises(ValueError): cp.load_inference_checkpoint(other,tmp_path / model)


@pytest.mark.parametrize("feature_set,width",[("C0",5),("C1",11),("C2",15)])
def test_d2_numeric_encoding_approved_missingness_coding(feature_set,width):
    from test_multimodal_fusion_a import a_backbone
    from multimodal.numeric_branch import NumericClinicalBranch
    branch=NumericClinicalBranch(a_backbone(overrides=["ablation.enabled=true","fusion.numeric_skip=true",f"clinical_set={feature_set}"]).resolved_config)
    record={name:None for name in branch.serializer.fields}
    encoded=branch.encode([record]);assert encoded.shape==(1,width) and encoded[0,1]==1
    record["sex"]="F";assert branch.encode([record])[0,2:5].tolist()==[1.,0.,0.]
    branch([record]).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters())
    for key in ("moca","ctt","updrs_total","subject_id","subgroup"):
        with pytest.raises(ValueError): branch([{**record,key:25}])
    with pytest.raises(ValueError): branch([{**record,"sex":99}])
    with pytest.raises(ValueError): branch([{**record,"age":float('nan')}])


def test_d2_numeric_m2_optimizer_checkpoint_and_off_unchanged(tmp_path):
    from test_multimodal_fusion_a import a_backbone
    from test_multimodal_fusion_b import inputs
    from multimodal.fusion_a import EEGClinicalRepresentationFusionRegressor as Model
    from multimodal.fusion_a_trainer import build_optimizer
    from multimodal.fusion_a_checkpoint import save_inference_checkpoint,load_inference_checkpoint
    torch.manual_seed(7);legacy=Model(a_backbone()).eval()
    torch.manual_seed(7);off=Model(a_backbone(overrides=["ablation.enabled=true"])).eval()
    off.load_state_dict(legacy.state_dict(),strict=True)
    torch.testing.assert_close(legacy(**inputs()),off(**inputs()),rtol=0,atol=0)
    on=Model(a_backbone(overrides=["ablation.enabled=true","fusion.numeric_skip=true"]))
    optimizer=build_optimizer(on);prediction=on(**inputs());prediction.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in on.numeric_branch.parameters())
    groups=[id(p) for g in optimizer.param_groups for p in g["params"]]
    assert len(groups)==len(set(groups)) and set(groups)=={id(p) for p in on.parameters() if p.requires_grad}
    optimizer.step();save_inference_checkpoint(on,tmp_path / "numeric")
    load_inference_checkpoint(on,tmp_path / "numeric")
    with pytest.raises(ValueError): load_inference_checkpoint(off,tmp_path / "numeric")


@pytest.mark.parametrize("k",[1,4,8])
@pytest.mark.parametrize("mode",["frozen","lora"])
def test_d2_soft_tokens_exact_suffix_mask_gradients(k,mode,tmp_path):
    from test_multimodal_fusion_b import backbone,inputs
    from multimodal.fusion_b import EEGClinicalSoftTokenRegressor
    from multimodal.fusion_b_trainer import build_optimizer
    from multimodal.fusion_b_checkpoint import save_inference_checkpoint,load_inference_checkpoint
    model=EEGClinicalSoftTokenRegressor(backbone(mode=mode,overrides=[f"eeg.pooling.num_queries={k}","ablation.enabled=true"]))
    batch=inputs();text=model.tokenizer(model.serializer.serialize_batch(batch["clinical"]))
    fused=model.prepare_inputs(**batch)
    assert model.pooling.queries.shape[0]==k and fused.inputs_embeds.shape[1]==text.input_ids.shape[1]+k
    assert torch.equal(fused.prediction_positions,text.prediction_positions+k)
    assert torch.equal(fused.attention_mask[:,:k],torch.ones(3,k,dtype=torch.long))
    assert torch.equal(fused.attention_mask[:,k:],text.attention_mask)
    optimizer=build_optimizer(model);model(**batch).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.pooling.parameters())
    lora=[p for n,p in model.llm.named_parameters() if "lora_" in n]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in lora)
    optimizer.step();save_inference_checkpoint(model,tmp_path / f"k{k}")
    load_inference_checkpoint(model,tmp_path / f"k{k}")


def test_d_review_typed_attention_factory_rejects_reduction():
    from multimodal.config import CONFIG_DIR,load_multimodal_config
    from multimodal.eeg_pooling import LearnedQueryAttentionPooling
    cfg=load_multimodal_config(CONFIG_DIR / "fusion_b.json",["ablation.enabled=true","eeg.pooling.num_queries=1","eeg.pooling.type=mean"])
    with pytest.raises(ValueError): LearnedQueryAttentionPooling.from_config(cfg)


def test_d_review_numeric_dimension_prevents_fully_controlled_claim():
    import copy
    from multimodal.experiment_evaluation import PredictionSeries
    from multimodal.prediction_matching import comparison_conditions
    # Direct metadata fixture: no experiment result or medical performance claim.
    from test_multimodal_fusion_a import a_backbone
    config=a_backbone(overrides=["ablation.enabled=true","fusion.numeric_skip=true"]).resolved_config.to_dict()
    identity=dict(cohort_fingerprint="same",source_subject_fingerprint="same",source_cohort_n=2,target="current_moca_original_units",unit="subject",
        outer_sha256="same",assignment_fingerprint="same",cache_fingerprint="same",clinical_fingerprint="same",model="M2",architecture="fusion_a",
        runtime=dict(python="same",libraries={},application_source_fingerprint="same"),llm_revision_status="unspecified")
    run=dict(manifest=dict(identity=identity,resolved_config=config,subjects={}),checkpoint_identity=dict(synthetic_frozen_digest="same",tokenizer_digest="same"))
    other=copy.deepcopy(run);other["manifest"]["resolved_config"]["settings"]["ablation"]["numeric_projection_dim"]+=1
    import pandas as pd
    a=PredictionSeries("a",pd.DataFrame(),{(0,1):run},{})
    b=PredictionSeries("b",pd.DataFrame(),{(0,1):other},{})
    conditions=comparison_conditions(a,b,{(0,1)},synthetic=True)
    assert conditions["recorded_protocol_comparable"] and not conditions["fully_controlled_configuration"]
