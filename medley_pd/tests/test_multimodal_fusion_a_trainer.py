"""B3 M2 subject-level training, partition integrity and safe persistence."""
import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from test_multimodal_fusion_a import make_model,a_backbone
from test_multimodal_fusion_b_trainer import loader,IDS,SPLIT
from test_multimodal_data_adapter import files as adapter_files
from multimodal.fusion_a import EEGClinicalRepresentationFusionRegressor
from multimodal.fusion_a_trainer import FusionATrainer,build_optimizer,seed_training
from multimodal.fusion_a_checkpoint import save_inference_checkpoint,load_inference_checkpoint,export_predictions
from multimodal.clinical_regressor import moca_huber_loss
from multimodal.config import ResolvedConfig
from multimodal.dataset import MedleySubjectDataset,MedleySubjectCollator
from multimodal.data_adapter import MedleyDataAdapter,SubjectSplit


@pytest.fixture(scope="module",autouse=True)
def threads():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def trainer(mode="frozen",overrides=(),train=IDS[:3],validation=IDS[5:7]):
    model=make_model(mode,overrides=["eeg.pooling.hidden_dim=16","fusion_a.eeg_projection_dim=16",
        "fusion_a.clinical_projection_dim=16","fusion.fusion_dim=16","fusion.dropout=0",*overrides])
    return FusionATrainer(model,split=SPLIT,train_subjects=train,validation_subjects=validation)


@pytest.mark.parametrize("mode",["frozen","lora"])
@pytest.mark.parametrize("size",[1,3])
def test_b3_train_gradients_evaluate_export_checkpoint_smoke(mode,size,tmp_path):
    run=trainer(mode)
    before={n:p.detach().clone() for n,p in run.model.named_parameters()}
    gradients={}
    handles=[p.register_hook(lambda g,n=n:gradients.setdefault(n,[]).append(g.detach().clone()))
             for n,p in run.model.named_parameters() if p.requires_grad]
    result=run.train_epoch(loader(IDS[:3],size))
    for handle in handles: handle.remove()
    assert result["optimizer_steps"]==1 and result["n"]==3 and np.isfinite(result["huber"])
    assert run.optimizer_steps==run.epochs_completed==1
    for name,p in run.model.named_parameters():
        if p.requires_grad:
            assert name in gradients and all(torch.isfinite(g).all() for g in gradients[name])
            if "lora_" not in name: assert any(g.abs().sum()>0 for g in gradients[name]),name
        else: assert p.grad is None and torch.equal(before[name],p)
    if mode=="lora": assert any(g.abs().sum()>0 for n,gs in gradients.items() if "lora_" in n for g in gs)
    assert all(p.grad is None for p in run.model.parameters())
    for prefix in ("eeg_encoder.pooling.","eeg_encoder.projection.","clinical_encoder.projection.","head."):
        assert any(not torch.equal(before[n],p) for n,p in run.model.named_parameters() if n.startswith(prefix))
    snapshot={n:p.detach().clone() for n,p in run.model.named_parameters()}
    state=copy.deepcopy(run.optimizer.state_dict())
    validation=run.evaluate(loader(IDS[5:7],2))
    assert run.model.training and run.optimizer_steps==1
    assert all(torch.equal(snapshot[n],p) for n,p in run.model.named_parameters())
    assert state["param_groups"]==run.optimizer.state_dict()["param_groups"]
    for key,values in state["state"].items():
        for name,value in values.items(): torch.testing.assert_close(value,run.optimizer.state_dict()["state"][key][name],rtol=0,atol=0)
    frame=validation.predictions
    assert tuple(frame.columns)==("seed","fold","subject","subgroup","model","y","yhat","target")
    assert frame.subject.tolist()==list(IDS[5:7]) and frame.model.tolist()==["fusion_a_llm:C1"]*2
    assert frame.seed.tolist()==[0]*2 and frame.fold.tolist()==[1]*2
    e=frame.yhat-frame.y
    assert validation.metrics==pytest.approx(dict(n=2,MAE=abs(e).mean(),RMSE=np.sqrt((e**2).mean())))
    path=export_predictions(validation,tmp_path / "m2.csv.gz")
    assert pd.read_csv(path).subject.tolist()==list(IDS[5:7])
    with pytest.raises(FileExistsError): export_predictions(validation,path)
    base=tmp_path / "baseline.csv";base.write_text("M0/M1/M3 protected")
    with pytest.raises(FileExistsError): export_predictions(validation,base)
    assert base.read_text()=="M0/M1/M3 protected"
    checkpoint=save_inference_checkpoint(run.model,tmp_path / "m2",epochs_completed=1,optimizer_steps=1)
    from safetensors.torch import load_file
    weights=load_file(str(checkpoint / "trainable.safetensors"))
    assert weights.keys()=={n for n,p in run.model.named_parameters() if p.requires_grad}
    assert all(".llm." not in n or "lora_" in n for n in weights)
    fresh=trainer(mode);fresh.model.requires_grad_(False)
    metadata=load_inference_checkpoint(fresh.model,checkpoint)
    assert metadata["kind"]=="inference_only_not_training_resume" and metadata["git_revision"]
    assert metadata["optimizer_steps"]==metadata["epochs_completed"]==1
    np.testing.assert_allclose(fresh.evaluate(loader(IDS[5:7],2)).predictions.yhat,frame.yhat,rtol=1e-6,atol=1e-7)
    second=save_inference_checkpoint(fresh.model,tmp_path / "frozen")
    load_inference_checkpoint(trainer(mode).model,second)
    with pytest.raises(FileExistsError): save_inference_checkpoint(run.model,checkpoint)


@pytest.mark.parametrize("mode",["frozen","lora"])
def test_b3_optimizer_configured_groups_complete_unique(mode):
    run=trainer(mode,overrides=["fusion_a.lr_pooling=0.001","fusion_a.lr_eeg_projection=0.002",
        "fusion_a.lr_clinical_projection=0.003","fusion_a.lr_fusion=0.004","training.lr_lora=0.00005"])
    groups=run.optimizer.param_groups
    assert [g["name"] for g in groups]==["eeg_pooling","eeg_projection","clinical_projection","fusion"]+(["lora"] if mode=="lora" else [])
    assert [g["lr"] for g in groups]==[.001,.002,.003,.004]+([.00005] if mode=="lora" else [])
    params=[id(p) for g in groups for p in g["params"]]
    assert len(params)==len(set(params)) and set(params)=={id(p) for p in run.model.parameters() if p.requires_grad}


@pytest.mark.parametrize("mode",["frozen","lora"])
@pytest.mark.parametrize("size,acc,n,steps",[(1,3,3,1),(2,2,3,1),(1,2,3,2),(2,2,5,2)])
def test_b3_accumulation_actual_subject_counts_and_partial_groups(mode,size,acc,n,steps):
    options=[f"training.gradient_accumulation={acc}","training.gradient_clip_norm=100000"]
    ids=IDS[:n]
    run,reference=[trainer(mode,overrides=options,train=ids,validation=()) for _ in range(2)]
    samples=[loader(ids).dataset[i] for i in range(n)]
    expected=[]
    for start in range(0,n,size*acc):
        batch=MedleySubjectCollator()(samples[start:start+size*acc])
        _,x,y=reference._batch(batch,ids,set())
        reference.model.train();reference.optimizer.zero_grad(set_to_none=True)
        moca_huber_loss(reference._forward(x),y).backward()
        expected.append({n:p.grad.clone() for n,p in reference.model.named_parameters() if p.grad is not None})
        reference.optimizer.step();reference.optimizer.zero_grad(set_to_none=True)
    actual=[];step=run.optimizer.step
    def capture(*args,**kwargs):
        actual.append({n:p.grad.clone() for n,p in run.model.named_parameters() if p.grad is not None})
        return step(*args,**kwargs)
    run.optimizer.step=capture
    result=run.train_epoch(loader(ids,size))
    assert result["optimizer_steps"]==run.optimizer_steps==steps and len(actual)==len(expected)
    for a,b in zip(actual,expected):
        assert a.keys()==b.keys()
        for name in a: torch.testing.assert_close(a[name],b[name],rtol=2e-4,atol=2e-6,msg=name)
    for (name,a),b in zip(run.model.named_parameters(),reference.model.parameters()):
        torch.testing.assert_close(a,b,rtol=2e-5,atol=5e-6,msg=name)
    assert all(p.grad is None for p in run.model.parameters())
    np.testing.assert_allclose(run.evaluate(loader(ids,2),role="train").predictions.yhat,
        reference.evaluate(loader(ids,2),role="train").predictions.yhat,rtol=2e-5,atol=5e-6)


@pytest.mark.parametrize("case",["base","projection_frozen","omitted","duplicate","adapters_frozen"])
def test_b3_optimizer_bad_parameter_scope(case):
    model=trainer("lora" if case=="adapters_frozen" else "frozen").model
    if case=="base": model.backbone.embedding_layer.weight.requires_grad_(True)
    elif case=="projection_frozen": model.clinical_encoder.projection.requires_grad_(False)
    elif case=="omitted": model.extra=torch.nn.Parameter(torch.ones(1))
    elif case=="duplicate": model.head.extra=model.pooling.queries
    else:
        for name,p in model.llm.named_parameters():
            if "lora_" in name: p.requires_grad_(False)
    with pytest.raises(ValueError): build_optimizer(model)


@pytest.mark.parametrize("case",["test_train","test_validation","train_val_overlap","outer_overlap","seed"])
def test_b3_partition_integrity(case):
    split,train,val=SPLIT,IDS[:3],IDS[5:7]
    if case=="test_train": train=IDS[7:]
    elif case=="test_validation": val=IDS[7:]
    elif case=="train_val_overlap": val=IDS[2:4]
    elif case=="outer_overlap": split=SubjectSplit(0,1,IDS[:7],IDS[6:])
    elif case=="seed": split=SubjectSplit(99,1,IDS[:7],IDS[7:])
    with pytest.raises(ValueError): FusionATrainer(trainer().model,split=split,train_subjects=train,validation_subjects=val)


@pytest.mark.parametrize("case",["duplicate","missing","test","bad_labels","bad_eeg","target_leakage"])
def test_b3_loader_guards_and_gradient_cleanup(case):
    run=trainer();batches=list(loader(IDS[:3]))
    if case=="duplicate": batches=[batches[0],batches[0]]
    elif case=="missing": batches=batches[:1]
    elif case=="test": batches=list(loader(IDS[7:]))
    elif case=="bad_labels": batches[0]["moca"]=torch.ones(1,1)
    elif case=="bad_eeg": batches[0]["eeg_mask"]=batches[0]["eeg_mask"].float()
    elif case=="target_leakage": batches[0]["clinical"][0]["moca"]=20
    with pytest.raises(ValueError): run.train_epoch(batches)
    assert run.epochs_completed==0 and all(p.grad is None for p in run.model.parameters())


def test_b3_evaluation_roles_coverage_no_leakage_or_mutation():
    run=trainer()
    batch=list(loader(IDS[5:7],2));before=copy.deepcopy(batch)
    expected=run.evaluate(batch)
    for current,old in zip(batch,before):
        for key in current:
            if isinstance(current[key],torch.Tensor): assert torch.equal(current[key],old[key])
            else: assert current[key]==old[key]
    batch[0]["moca"].fill_(29.875);batch[0]["subgroups"]=["OTHER"]*2
    actual=run.evaluate(batch)
    np.testing.assert_array_equal(actual.predictions.yhat,expected.predictions.yhat)
    assert not np.array_equal(actual.predictions.y,expected.predictions.y)
    with pytest.raises(ValueError): run.evaluate(loader(IDS[7:]))
    assert run.evaluate(loader(IDS[7:]),role="test").metrics["n"]==1
    with pytest.raises(ValueError): run.evaluate(list(loader(IDS[5:6])))
    repeated=list(loader(IDS[5:7]))
    with pytest.raises(ValueError): run.evaluate([repeated[0],repeated[0]])
    assert run.model.training
    run.model.eval();run.evaluate(loader(IDS[5:7]));assert not run.model.training
    empty=trainer(validation=())
    with pytest.raises(ValueError): empty.evaluate(loader(IDS[5:7]))


@pytest.mark.parametrize("case",["k","eeg_width","clinical_width","hidden_width","mode","tokenizer","base",
    "snapshot","corrupt","tensor","missing_git","bad_git","negative_steps","architecture"])
def test_b3_checkpoint_incompatibility_before_mutation(case,tmp_path):
    from safetensors.torch import load_file,save_file
    source=trainer();path=save_inference_checkpoint(source.model,tmp_path / "m2")
    target=trainer("lora" if case=="mode" else "frozen",overrides={
        "k":["eeg.pooling.num_queries=7"],"eeg_width":["fusion_a.eeg_projection_dim=17"],
        "clinical_width":["fusion_a.clinical_projection_dim=17"],"hidden_width":["fusion.fusion_dim=17"]}.get(case,[]))
    if case=="tokenizer": target.model.backbone.tokenizer.padding_side="left"
    elif case=="base":
        with torch.no_grad(): target.model.backbone.embedding_layer.weight.add_(1)
    elif case=="snapshot": (path / "resolved_config.json").write_text('{}')
    elif case=="corrupt":
        file=path / "trainable.safetensors";file.write_bytes(file.read_bytes()+b"corrupt")
    elif case=="tensor":
        file=path / "trainable.safetensors";weights=load_file(str(file));weights["head.mlp.3.weight"]=torch.ones(1,15)
        save_file(weights,str(file));manifest=json.loads((path / "manifest.json").read_text())
        manifest["weights_sha256"]=hashlib.sha256(file.read_bytes()).hexdigest()
        (path / "manifest.json").write_text(json.dumps(manifest))
    elif case in ("missing_git","bad_git","negative_steps"):
        manifest=json.loads((path / "manifest.json").read_text())
        if case=="missing_git": del manifest["git_revision"]
        elif case=="bad_git": manifest["git_revision"]=[]
        else: manifest["optimizer_steps"]=-1
        (path / "manifest.json").write_text(json.dumps(manifest))
    elif case=="architecture":
        from test_multimodal_fusion_b_trainer import trainer as m3_trainer
        from multimodal.fusion_b_checkpoint import save_inference_checkpoint as save_m3
        path=save_m3(m3_trainer().model,tmp_path / "m3")
    before={n:p.detach().clone() for n,p in target.model.named_parameters()}
    with pytest.raises(ValueError): load_inference_checkpoint(target.model,path)
    assert all(torch.equal(before[n],p) for n,p in target.model.named_parameters())


@pytest.mark.parametrize("mode",["frozen","lora"])
def test_b3_actual_disk_adapter_split_and_end_to_end(mode,adapter_files,tmp_path):
    cfg,table,cache,split_file,ids=adapter_files
    split_bytes=split_file.read_bytes()
    prepared=a_backbone(mode,overrides=["eeg.pooling.hidden_dim=16","fusion_a.eeg_projection_dim=16",
        "fusion_a.clinical_projection_dim=16","fusion.fusion_dim=16","evaluation.seeds=[0,1]",
        'clinical_text.sex_code_mapping={"0":"female","1":"male"}'])
    prepared.resolved_config=ResolvedConfig(copy.deepcopy(prepared.resolved_config.settings),copy.deepcopy(cfg.medley),"deferred")
    adapter=MedleyDataAdapter(prepared.resolved_config);split=adapter.get_split(0,1)
    assert set(split.train_subjects).isdisjoint(split.test_subjects)
    train,val=split.train_subjects[:-2],split.train_subjects[-2:]
    def batches(subjects):
        return DataLoader(MedleySubjectDataset(adapter,subjects),batch_size=2,shuffle=False,collate_fn=MedleySubjectCollator())
    run=FusionATrainer(EEGClinicalRepresentationFusionRegressor(prepared),split=split,train_subjects=train,validation_subjects=val)
    assert run.train_epoch(batches(train))["n"]==len(train)
    assert run.evaluate(batches(val)).metrics["n"]==len(val)
    result=run.evaluate(batches(split.test_subjects),role="test")
    assert result.predictions.subject.tolist()==list(split.test_subjects)
    assert all(np.isfinite(list(result.metrics.values())))
    export_predictions(result,tmp_path / "disk_m2.csv.gz")
    checkpoint=save_inference_checkpoint(run.model,tmp_path / "disk_m2")
    with torch.no_grad(): run.model.head.mlp[3].bias.add_(10)
    load_inference_checkpoint(run.model,checkpoint)
    np.testing.assert_allclose(run.evaluate(batches(split.test_subjects),role="test").predictions.yhat,
                               result.predictions.yhat,rtol=1e-6,atol=1e-7)
    assert split_file.read_bytes()==split_bytes


@pytest.mark.parametrize("mode",["frozen","lora"])
def test_b3_pre_phase_b_m3_checkpoint_still_loads(mode,tmp_path):
    from test_multimodal_fusion_b_trainer import trainer as m3_trainer
    from multimodal.fusion_b_checkpoint import save_inference_checkpoint as save_m3,load_inference_checkpoint as load_m3
    from multimodal.clinical_checkpoint import _digest
    source,target=m3_trainer(mode),m3_trainer(mode)
    with torch.no_grad(): source.model.head.bias.add_(2)
    path=save_m3(source.model,tmp_path / "old_m3")
    cfg=json.loads((path / "resolved_config.json").read_text());del cfg["settings"]["fusion_a"]
    (path / "resolved_config.json").write_text(json.dumps(cfg))
    manifest=json.loads((path / "manifest.json").read_text());manifest["identity"]["config_digest"]=_digest(cfg)
    (path / "manifest.json").write_text(json.dumps(manifest))
    load_m3(target.model,path)
    torch.testing.assert_close(target.model.head.bias,source.model.head.bias,rtol=0,atol=0)


def test_b3_fixed_seed_dropout_shuffle_and_no_resume_claim(tmp_path):
    a,b=[trainer(overrides=["fusion.dropout=0.1"]) for _ in range(2)]
    seed_training(42);first=a.train_epoch(loader(IDS[:3],shuffle=True,seed=12))
    seed_training(42);second=b.train_epoch(loader(IDS[:3],shuffle=True,seed=12))
    assert first==second
    for p,q in zip(a.model.parameters(),b.model.parameters()): torch.testing.assert_close(p,q,rtol=0,atol=0)
    from multimodal.llm_backbone import BackboneError
    with pytest.raises(BackboneError): trainer("qlora")
    with pytest.raises(ValueError,match="BF16"): trainer(overrides=["clinical_regression.mixed_precision=bfloat16"])
    run=trainer(overrides=["training.gradient_accumulation=1"])
    batches=list(loader(IDS[:3]))
    with pytest.raises(ValueError): run.train_epoch([batches[0],batches[0]])
    assert run.optimizer_steps==1 and run.epochs_completed==0
    assert all(p.grad is None for p in run.model.parameters())
