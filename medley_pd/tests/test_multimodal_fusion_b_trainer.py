"""A3 genuine subject-level CPU training; no pretrained downloads or patient data."""
import copy
import json
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from test_multimodal_fusion_b import backbone
from multimodal.fusion_b import EEGClinicalSoftTokenRegressor
from multimodal.fusion_b_trainer import FusionBTrainer, build_optimizer, seed_training
from multimodal.clinical_regressor import moca_huber_loss
from multimodal.data_adapter import SubjectRecord, SubjectSplit
from multimodal.dataset import MedleySubjectDataset, MedleySubjectCollator
from multimodal.config import CONFIG_DIR, ConfigError, load_multimodal_config, save_resolved_config


@pytest.fixture(scope="module", autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


IDS = tuple(f"synthetic-a3-{i}" for i in range(8))
SPLIT = SubjectSplit(0, 1, IDS[:7], IDS[7:])


def loader(ids, batch_size=1, *, shuffle=False, seed=0):
    records = {}
    for sid in ids:
        i = IDS.index(sid)
        clinical = dict(age=60.+i, sex=None, updrs_part_iii=24.+i, disease_duration=6.5, ledd=None)
        if i == 1:
            clinical.update(disease_duration=None, updrs_part_iii=None)
        if i == 2:
            clinical = dict.fromkeys(clinical)
        eeg = np.random.default_rng(i).normal(size=(i+1, 3800)).astype(np.float32)
        records[sid] = SubjectRecord(sid, eeg, np.arange(i+1) % 2 == 0,
                                    MappingProxyType(clinical), 19.125+i, "SYNTHETIC")
    adapter = SimpleNamespace(subject_ids=tuple(ids), clinical_variables=tuple(clinical), get_subject=records.__getitem__)
    return DataLoader(MedleySubjectDataset(adapter, tuple(ids)), batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), collate_fn=MedleySubjectCollator())


def trainer(mode="frozen", overrides=(), train=IDS[:3], validation=IDS[5:7]):
    model = EEGClinicalSoftTokenRegressor(backbone(mode, overrides=["eeg.pooling.hidden_dim=16", *overrides]))
    return FusionBTrainer(model, split=SPLIT, train_subjects=train, validation_subjects=validation)


@pytest.mark.parametrize("mode", ["frozen", "lora"])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_step_actual_huber_gradient_paths_and_parameter_updates(mode, batch_size):
    run = trainer(mode)
    before = {n: p.detach().clone() for n, p in run.model.named_parameters()}
    gradients = {}
    handles = [p.register_hook(lambda g, n=n: gradients.setdefault(n, []).append(g.detach().clone()))
               for n, p in run.model.named_parameters() if p.requires_grad]
    result = run.train_epoch(loader(IDS[:3], batch_size))
    for handle in handles:
        handle.remove()
    assert result["n"] == 3 and result["optimizer_steps"] == 1 and np.isfinite(result["huber"])
    for n, p in run.model.named_parameters():
        if p.requires_grad:
            assert n in gradients and all(torch.isfinite(g).all() for g in gradients[n])
            if "lora_" not in n:
                assert any(g.abs().sum() > 0 for g in gradients[n]), n
        else:
            assert p.grad is None and torch.equal(before[n], p)
    for prefix in ("pooling.", "projector.", "head."):
        assert any(not torch.equal(before[n], p) for n, p in run.model.named_parameters() if n.startswith(prefix))
    if mode == "lora":
        assert any(g.abs().sum() > 0 for n, gs in gradients.items() if "lora_" in n for g in gs)
    assert all(p.grad is None for p in run.model.parameters())


@pytest.mark.parametrize("mode", ["frozen", "lora"])
def test_groups_complete_unique_configurable(mode):
    run = trainer(mode, overrides=["fusion_b_training.lr_pooling=0.001", "fusion_b_training.lr_projector=0.002",
                                   "fusion_b_training.lr_head=0.003", "training.lr_lora=0.00004"])
    groups = run.optimizer.param_groups
    assert [g["name"] for g in groups] == ["eeg_pooling", "eeg_projector", "regression_head"] + (["lora"] if mode == "lora" else [])
    assert [g["lr"] for g in groups] == [.001, .002, .003] + ([.00004] if mode == "lora" else [])
    params = [id(p) for g in groups for p in g["params"]]
    assert len(params) == len(set(params))
    assert set(params) == {id(p) for p in run.model.parameters() if p.requires_grad}
    assert all(g["weight_decay"] == .01 for g in groups)


@pytest.mark.parametrize("case", ["base", "head_frozen", "omission", "duplicate", "adapters_frozen"])
def test_optimizer_rejects_incorrect_trainability(case):
    model = trainer("lora" if case == "adapters_frozen" else "frozen").model
    if case == "base": model.backbone.embedding_layer.weight.requires_grad_(True)
    elif case == "head_frozen": model.head.requires_grad_(False)
    elif case == "omission": model.extra = torch.nn.Parameter(torch.ones(1))
    elif case == "duplicate": model.projector.extra = model.pooling.queries
    else:
        for n, p in model.llm.named_parameters():
            if "lora_" in n: p.requires_grad_(False)
    with pytest.raises(ValueError):
        build_optimizer(model)


@pytest.mark.parametrize("mode", ["frozen", "lora"])
@pytest.mark.parametrize("size,acc,n,steps", [(1,3,3,1), (2,2,3,1), (1,2,3,2), (2,2,5,2)])
def test_subject_weighted_accumulation_equals_effective_batches(mode, size, acc, n, steps):
    ids = IDS[:n]
    options = [f"training.gradient_accumulation={acc}", "training.gradient_clip_norm=100000"]
    run, reference = (trainer(mode, overrides=options, train=ids, validation=()) for _ in range(2))
    reference_gradients = []
    samples = [loader(ids).dataset[i] for i in range(n)]
    for start in range(0, n, size*acc):
        batch = MedleySubjectCollator()(samples[start:start+size*acc])
        _, inputs, targets = reference._batch(batch, ids, set())
        reference.model.train()
        reference.optimizer.zero_grad(set_to_none=True)
        moca_huber_loss(reference._forward(inputs), targets).backward()
        reference_gradients.append({name:p.grad.clone() for name,p in reference.model.named_parameters() if p.grad is not None})
        reference.optimizer.step()
        reference.optimizer.zero_grad(set_to_none=True)
    actual_gradients = []
    original_step = run.optimizer.step
    def capture_step(*args, **kwargs):
        actual_gradients.append({name:p.grad.clone() for name,p in run.model.named_parameters() if p.grad is not None})
        return original_step(*args, **kwargs)
    run.optimizer.step = capture_step
    result = run.train_epoch(loader(ids, size))
    assert len(actual_gradients) == len(reference_gradients)
    for actual, expected in zip(actual_gradients, reference_gradients):
        assert actual.keys() == expected.keys()
        for name in actual:
            torch.testing.assert_close(actual[name], expected[name], rtol=2e-4, atol=2e-6, msg=name)
    assert result["optimizer_steps"] == steps and run.optimizer_steps == steps
    for (name, a), b in zip(run.model.named_parameters(), reference.model.parameters()):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=5e-6, msg=name)
    assert all(p.grad is None for p in run.model.parameters())
    a = run.evaluate(loader(ids,2), role="train").predictions.yhat
    b = reference.evaluate(loader(ids,2), role="train").predictions.yhat
    np.testing.assert_allclose(a,b,rtol=2e-5,atol=5e-6)


@pytest.mark.parametrize("mode", ["frozen", "lora"])
def test_validation_metrics_alignment_export_and_no_mutation(mode, tmp_path):
    from multimodal.fusion_b_checkpoint import export_predictions
    run = trainer(mode)
    run.train_epoch(loader(IDS[:3]))
    before = {n: p.detach().clone() for n, p in run.model.named_parameters()}
    state = copy.deepcopy(run.optimizer.state_dict())
    steps = run.optimizer_steps
    source = list(loader(IDS[5:7], 2))
    original = copy.deepcopy(source)
    result = run.evaluate(source)
    assert run.model.training and run.optimizer_steps == steps
    assert all(torch.equal(before[n], p) for n, p in run.model.named_parameters())
    assert state["param_groups"] == run.optimizer.state_dict()["param_groups"]
    for k, values in state["state"].items():
        for name, value in values.items():
            torch.testing.assert_close(value, run.optimizer.state_dict()["state"][k][name], rtol=0, atol=0)
    frame = result.predictions
    assert tuple(frame.columns) == ("seed", "fold", "subject", "subgroup", "model", "y", "yhat", "target")
    assert frame.subject.tolist() == list(IDS[5:7]) and frame.model.tolist() == ["fusion_b_llm:C1"]*2
    assert frame.seed.tolist() == [0]*2 and frame.fold.tolist() == [1]*2
    e = frame.yhat - frame.y
    assert result.metrics == pytest.approx(dict(n=2, MAE=abs(e).mean(), RMSE=np.sqrt((e**2).mean())))
    for batch, old in zip(source, original):
        for key in batch:
            if isinstance(batch[key], torch.Tensor): assert torch.equal(batch[key], old[key])
            else: assert batch[key] == old[key]
    output = export_predictions(result, tmp_path / "m3.csv.gz")
    assert pd.read_csv(output).subject.tolist() == list(IDS[5:7])
    baseline = tmp_path / "baseline.csv"
    baseline.write_text("existing baseline")
    with pytest.raises(FileExistsError): export_predictions(result, baseline)
    assert baseline.read_text() == "existing baseline"
    with pytest.raises(FileExistsError): export_predictions(result, output)
    for batch in source: batch["moca"].fill_(29.875)
    np.testing.assert_array_equal(run.evaluate(source).predictions.yhat, frame.yhat)


@pytest.mark.parametrize("mode", ["frozen", "lora"])
def test_end_to_end_checkpoint_smoke_complete_scope_and_frozen_restore(mode, tmp_path):
    from multimodal.fusion_b_checkpoint import save_inference_checkpoint, load_inference_checkpoint
    from safetensors.torch import load_file
    run = trainer(mode)
    run.train_epoch(loader(IDS[:3], 2))
    expected = run.evaluate(loader(IDS[5:7], 2)).predictions.yhat.to_numpy()
    path = save_inference_checkpoint(run.model, tmp_path / "m3", epochs_completed=run.epochs_completed,
                                    optimizer_steps=run.optimizer_steps)
    weights = load_file(str(path / "trainable.safetensors"))
    assert set(weights) == {n for n, p in run.model.named_parameters() if p.requires_grad}
    assert all(not n.startswith("llm.") or "lora_" in n for n in weights)
    fresh = trainer(mode)
    fresh.model.requires_grad_(False)
    info = load_inference_checkpoint(fresh.model, path)
    assert info["epochs_completed"] == 1 and info["optimizer_steps"] == 1
    assert info["kind"] == "inference_only_not_training_resume" and info["git_revision"]
    np.testing.assert_allclose(fresh.evaluate(loader(IDS[5:7], 2)).predictions.yhat, expected, rtol=1e-6, atol=1e-7)
    second = save_inference_checkpoint(fresh.model, tmp_path / "frozen")
    other = trainer(mode)
    load_inference_checkpoint(other.model, second)
    np.testing.assert_allclose(other.evaluate(loader(IDS[5:7], 2)).predictions.yhat, expected, rtol=1e-6, atol=1e-7)
    with pytest.raises(FileExistsError): save_inference_checkpoint(run.model, path)


@pytest.mark.parametrize("mismatch", ["k", "width", "mode", "tokenizer", "base", "config", "corrupt", "tensor"])
def test_checkpoint_mismatch_rejected_before_mutation(mismatch, tmp_path):
    from multimodal.fusion_b_checkpoint import save_inference_checkpoint, load_inference_checkpoint
    from safetensors.torch import load_file, save_file
    import hashlib
    source = trainer()
    path = save_inference_checkpoint(source.model, tmp_path / "m3")
    target = trainer("lora" if mismatch == "mode" else "frozen", overrides={
        "k": ["eeg.pooling.num_queries=7"], "width": ["eeg.pooling.hidden_dim=32"]}.get(mismatch, []))
    if mismatch == "tokenizer": target.model.backbone.tokenizer.padding_side = "left"
    elif mismatch == "base":
        with torch.no_grad(): target.model.backbone.embedding_layer.weight.add_(1)
    elif mismatch == "config":
        (path / "resolved_config.json").write_text('{}')
    elif mismatch == "corrupt":
        p = path / "trainable.safetensors"; p.write_bytes(p.read_bytes()+b"corrupt")
    elif mismatch == "tensor":
        p = path / "trainable.safetensors"
        weights = load_file(str(p)); weights["head.weight"] = torch.zeros(1, 15)
        save_file(weights, str(p))
        manifest = json.loads((path / "manifest.json").read_text())
        manifest["weights_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
        (path / "manifest.json").write_text(json.dumps(manifest))
    before = {n:p.detach().clone() for n,p in target.model.named_parameters()}
    with pytest.raises(ValueError): load_inference_checkpoint(target.model, path)
    assert all(torch.equal(before[n], p) for n,p in target.model.named_parameters())


@pytest.mark.parametrize("case", ["test_train", "test_validation", "overlap", "split_overlap", "duplicate_ids", "seed"])
def test_partition_integrity(case):
    model = trainer().model
    train, validation, split = IDS[:3], IDS[5:7], SPLIT
    if case == "test_train": train = IDS[7:]
    elif case == "test_validation": validation = IDS[7:]
    elif case == "overlap": validation = IDS[2:4]
    elif case == "split_overlap": split = SubjectSplit(0,1,IDS[:7],IDS[6:])
    elif case == "duplicate_ids": train = IDS[:1]*2
    elif case == "seed": split = SubjectSplit(99,1,IDS[:7],IDS[7:])
    with pytest.raises(ValueError): FusionBTrainer(model, split=split, train_subjects=train, validation_subjects=validation)


@pytest.mark.parametrize("case", ["duplicate", "missing", "wrong_loader", "test", "bad_labels", "wrong_batch", "clinical_target"])
def test_training_loader_integrity_and_error_cleanup(case):
    run = trainer()
    batches = list(loader(IDS[:3]))
    if case == "duplicate": batches = [batches[0],batches[0]]
    elif case == "missing": batches = batches[:1]
    elif case == "wrong_loader": batches = loader(IDS[:2])
    elif case == "test": batches = list(loader(IDS[7:]))
    elif case == "bad_labels": batches[0]["moca"] = torch.tensor([[19.]])
    elif case == "wrong_batch": batches[0]["eeg_features"] = torch.ones(2,1,3800)
    elif case == "clinical_target": batches[0]["clinical"][0]["moca"] = 29.
    with pytest.raises(ValueError): run.train_epoch(batches)
    assert run.epochs_completed == 0 and all(p.grad is None for p in run.model.parameters())


def test_evaluation_coverage_roles_modes_and_no_selection():
    run = trainer()
    with pytest.raises(ValueError): run.evaluate(loader(IDS[7:]))
    assert run.evaluate(loader(IDS[7:]), role="test").metrics["n"] == 1
    with pytest.raises(ValueError): run.evaluate(loader(IDS[5:6]))
    repeated = list(loader(IDS[5:7]))
    with pytest.raises(ValueError): run.evaluate([repeated[0], repeated[0]])
    assert run.model.training
    run.model.eval()
    run.evaluate(loader(IDS[5:7]))
    assert not run.model.training
    with pytest.raises(ValueError): run.evaluate(loader(IDS[5:7]), role="best")
    empty = trainer(validation=())
    with pytest.raises(ValueError): empty.evaluate(loader(IDS[5:7]))


def test_trainer_padding_artifacts_serialization_shape_loss_and_no_mutation():
    run = trainer()
    batches = list(loader(IDS[:3],3))
    batch = batches[0]
    before = copy.deepcopy(batch)
    run.model.eval()
    ids, inputs, targets = run._batch(batch, IDS[:3], set())
    pred = run._forward(inputs)
    assert pred.shape == targets.shape == (3,)
    expected_loss = torch.nn.functional.huber_loss(pred, targets, delta=1.)
    torch.testing.assert_close(moca_huber_loss(pred,targets), expected_loss)
    text = run.model.serializer.serialize_batch(inputs["clinical"])
    assert all("MoCA" not in t and "19.125" not in t and "synthetic" not in t for t in text)
    assert text[2] == ""  # All missing values remain eligible.
    altered = copy.deepcopy(inputs)
    altered["eeg_features"][~altered["eeg_mask"]] = float("nan")
    altered["artifact_flags"] = ~altered["artifact_flags"]
    torch.testing.assert_close(run._forward(altered), pred, rtol=0, atol=0)
    for key in batch:
        if isinstance(batch[key], torch.Tensor): assert torch.equal(batch[key], before[key])
        else: assert batch[key] == before[key]
    with pytest.raises(ValueError): moca_huber_loss(pred, targets[:,None])


def test_config_defaults_inheritance_overrides_snapshot_and_seeded_shuffle(tmp_path):
    child = tmp_path / "child.json"
    child.write_text(json.dumps(dict(extends=str(CONFIG_DIR / "fusion_b.json"),
                                    fusion_b_training=dict(lr_pooling=.004))))
    cfg = load_multimodal_config(child, ["fusion_b_training.lr_head=0.003"])
    assert cfg.settings["fusion_b_training"] == dict(lr_pooling=.004,lr_projector=None,lr_head=.003,lr_feature_adapter=None)
    snapshot = tmp_path / "snapshot.json"
    save_resolved_config(cfg,snapshot)
    assert load_multimodal_config(snapshot).to_dict() == cfg.to_dict()
    legacy = cfg.to_dict(); del legacy["settings"]["fusion_b_training"]
    snapshot.write_text(json.dumps(legacy))
    assert all(v is None for v in load_multimodal_config(snapshot).settings["fusion_b_training"].values())
    for value in ("0", "-1", "true", ".nan", "oops"):
        with pytest.raises(ConfigError): load_multimodal_config(child,[f"fusion_b_training.lr_head={value}"])
    a,b = trainer(),trainer()
    info = a.train_epoch(loader(IDS[:3],shuffle=True,seed=12))
    assert b.train_epoch(loader(IDS[:3],shuffle=True,seed=12)) == info
    for p,q in zip(a.model.parameters(),b.model.parameters()): torch.testing.assert_close(p,q,rtol=0,atol=0)
    seed_training(23); x=(np.random.rand(),torch.rand(1))
    seed_training(23); y=(np.random.rand(),torch.rand(1))
    assert x[0] == y[0] and torch.equal(x[1],y[1])


def test_precision_and_real_qlora_readiness():
    from multimodal.llm_backbone import BackboneError
    with pytest.raises(ValueError, match="BF16"):
        trainer(overrides=["clinical_regression.mixed_precision=bfloat16"])
    with pytest.raises(BackboneError): trainer("qlora")


def test_completed_steps_survive_later_failure():
    run = trainer(overrides=["training.gradient_accumulation=1"])
    batches = list(loader(IDS[:3]))
    with pytest.raises(ValueError): run.train_epoch([batches[0],batches[0]])
    assert run.optimizer_steps == 1 and run.epochs_completed == 0
    assert all(p.grad is None for p in run.model.parameters())


@pytest.mark.parametrize("mutation", ["missing_git", "bad_git", "negative_steps", "missing_manifest", "pickle"])
def test_invalid_checkpoint_manifest_never_mutates(mutation, tmp_path):
    from multimodal.fusion_b_checkpoint import save_inference_checkpoint, load_inference_checkpoint
    source, target = trainer(), trainer()
    with torch.no_grad(): source.model.head.bias.add_(5)
    path = save_inference_checkpoint(source.model, tmp_path / "m3")
    manifest = path / "manifest.json"
    metadata = json.loads(manifest.read_text())
    if mutation == "missing_git": del metadata["git_revision"]
    elif mutation == "bad_git": metadata["git_revision"] = []
    elif mutation == "negative_steps": metadata["optimizer_steps"] = -1
    manifest.write_text(json.dumps(metadata))
    if mutation == "missing_manifest": manifest.unlink()
    elif mutation == "pickle": path = tmp_path / "untrusted.pkl"; path.write_bytes(b"not pickle-loaded")
    before = {n:p.detach().clone() for n,p in target.model.named_parameters()}
    with pytest.raises(ValueError): load_inference_checkpoint(target.model,path)
    assert all(torch.equal(before[n],p) for n,p in target.model.named_parameters())


def test_pre_a3_m1_checkpoint_compatibility(tmp_path):
    from test_multimodal_clinical_trainer import trainer as m1_trainer
    from multimodal.clinical_checkpoint import save_inference_checkpoint, load_inference_checkpoint, _digest
    source, target = m1_trainer(), m1_trainer()
    with torch.no_grad(): source.model.head.bias.add_(2)
    path = save_inference_checkpoint(source.model,tmp_path / "old_m1")
    cfg = json.loads((path / "resolved_config.json").read_text())
    del cfg["settings"]["fusion_b_training"]
    (path / "resolved_config.json").write_text(json.dumps(cfg))
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["identity"]["config_digest"] = _digest(cfg)
    (path / "manifest.json").write_text(json.dumps(manifest))
    load_inference_checkpoint(target.model,path)
    torch.testing.assert_close(target.model.head.bias,source.model.head.bias,rtol=0,atol=0)


from test_multimodal_data_adapter import files as adapter_files


@pytest.mark.parametrize("mode", ["frozen", "lora"])
def test_disk_adapter_fixed_split_dataset_training_export_reload(mode, adapter_files, tmp_path):
    from multimodal.config import ResolvedConfig
    from multimodal.data_adapter import MedleyDataAdapter
    from multimodal.fusion_b_checkpoint import save_inference_checkpoint,load_inference_checkpoint,export_predictions
    cfg, table, cache, split_path, ids = adapter_files
    original_split = split_path.read_bytes()
    prepared = backbone(mode,overrides=["eeg.pooling.hidden_dim=16", "evaluation.seeds=[0,1]",
        'clinical_text.sex_code_mapping={"0":"female","1":"male"}'])
    prepared.resolved_config = ResolvedConfig(copy.deepcopy(prepared.resolved_config.settings),
                                              copy.deepcopy(cfg.medley),"deferred")
    adapter = MedleyDataAdapter(prepared.resolved_config)
    split = adapter.get_split(0,1)
    assert set(split.train_subjects).isdisjoint(split.test_subjects)
    train, val = split.train_subjects[:-2],split.train_subjects[-2:]
    def batches(subjects):
        return DataLoader(MedleySubjectDataset(adapter,subjects),batch_size=2,shuffle=False,
                          collate_fn=MedleySubjectCollator())
    run = FusionBTrainer(EEGClinicalSoftTokenRegressor(prepared),split=split,
                        train_subjects=train,validation_subjects=val)
    assert run.train_epoch(batches(train))["n"] == len(train)
    assert run.evaluate(batches(val)).metrics["n"] == len(val)
    result = run.evaluate(batches(split.test_subjects),role="test")
    assert result.predictions.subject.tolist() == list(split.test_subjects)
    export_predictions(result,tmp_path / "disk_m3.csv.gz")
    checkpoint = save_inference_checkpoint(run.model,tmp_path / "disk_m3")
    with torch.no_grad(): run.model.head.bias.add_(10)
    load_inference_checkpoint(run.model,checkpoint)
    np.testing.assert_allclose(run.evaluate(batches(split.test_subjects),role="test").predictions.yhat,
                               result.predictions.yhat,rtol=1e-6,atol=1e-7)
    assert split_path.read_bytes() == original_split


def test_private_artifact_directories_ignored_and_source_paths_rejected(tmp_path):
    import subprocess
    from multimodal.clinical_checkpoint import REPOSITORY
    from multimodal.fusion_b_checkpoint import save_inference_checkpoint,export_predictions
    run = trainer()
    result = run.evaluate(loader(IDS[5:7]))
    for path in (REPOSITORY / "outputs" / "a3.csv.gz", REPOSITORY / "checkpoints" / "a3" / "manifest.json"):
        assert subprocess.run(["git","check-ignore","--quiet",str(path)],cwd=REPOSITORY).returncode == 0
    with pytest.raises(ValueError): export_predictions(result,REPOSITORY / "source.csv.gz")
    with pytest.raises(ValueError): save_inference_checkpoint(run.model,REPOSITORY / "source_checkpoint")
