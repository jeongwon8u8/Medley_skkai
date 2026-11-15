"""M1 optimization, partition guards, private export and safetensors roundtrip."""
import copy
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from test_multimodal_clinical_regressor import make_model, forward
from multimodal.clinical_regressor import moca_huber_loss
from multimodal.clinical_tokenization import ClinicalTokenizer
from multimodal.clinical_trainer import ClinicalTrainer, seed_training
from multimodal.clinical_checkpoint import (export_predictions, load_inference_checkpoint,
    save_inference_checkpoint, REPOSITORY)
from multimodal.config import CONFIG_DIR, load_multimodal_config, save_resolved_config
from multimodal.data_adapter import SubjectRecord, SubjectSplit
from multimodal.dataset import MedleySubjectDataset, MedleySubjectCollator


@pytest.fixture(scope="module", autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


IDS = tuple(f"synthetic-{i}" for i in range(7))
SPLIT = SubjectSplit(0, 1, IDS[:6], IDS[6:])


def loader(ids, batch_size=1):
    records = {}
    for sid in ids:
        i = IDS.index(sid)
        clinical = dict(age=60. + i, sex=None, updrs_part_iii=24. + i, disease_duration=6.5, ledd=None)
        if i == 1:
            clinical.update(disease_duration=None, updrs_part_iii=None)
        if i == 2:
            clinical = dict.fromkeys(clinical)
        records[sid] = SubjectRecord(sid, np.full((i + 1, 4), 9999., dtype=np.float32),
            np.zeros(i + 1, dtype=bool), MappingProxyType(clinical), 19.125 + i, "SYNTHETIC_PD")
    adapter = SimpleNamespace(subject_ids=tuple(ids), clinical_variables=("age", "sex", "updrs_part_iii", "disease_duration", "ledd"),
                              get_subject=records.__getitem__)
    return DataLoader(MedleySubjectDataset(adapter, tuple(ids)), batch_size=batch_size, shuffle=False,
                      collate_fn=MedleySubjectCollator())


def trainer(mode="frozen", overrides=(), train=IDS[:3], validation=IDS[3:6]):
    return ClinicalTrainer(make_model(mode, overrides=overrides), split=SPLIT,
                           train_subjects=train, validation_subjects=validation)


@pytest.mark.parametrize("mode", ["frozen", "lora"])
def test_synthetic_training_evaluation_export_checkpoint_smoke(mode, tmp_path):
    run = trainer(mode)
    before = {n: p.detach().clone() for n, p in run.model.named_parameters()}
    info = run.train_epoch(loader(IDS[:3]))
    assert info["n"] == 3 and info["optimizer_steps"] == 1 and np.isfinite(info["huber"])
    assert not torch.equal(before["head.weight"], run.model.head.weight)
    assert all(torch.equal(before[n], p) for n, p in run.model.named_parameters() if not p.requires_grad)
    if mode == "lora":
        assert any(not torch.equal(before[n], p) for n, p in run.model.named_parameters() if "lora_" in n)
    snapshot = {n: p.detach().clone() for n, p in run.model.named_parameters()}
    result = run.evaluate(loader(IDS[3:6], 2))
    assert run.model.training
    assert result.metrics["n"] == 3
    assert all(torch.equal(snapshot[n], p) for n, p in run.model.named_parameters())
    assert result.predictions.subject.tolist() == list(IDS[3:6])
    assert np.isfinite(result.predictions.yhat).all()
    error = result.predictions.yhat - result.predictions.y
    assert result.metrics["MAE"] == pytest.approx(np.abs(error).mean())
    assert result.metrics["RMSE"] == pytest.approx(np.sqrt((error**2).mean()))
    assert (result.predictions.target == result.predictions.y).all()  # Medley target is numeric original-unit value.
    out = export_predictions(result, tmp_path / "predictions.csv.gz")
    restored = pd.read_csv(out)
    assert restored.subject.tolist() == list(IDS[3:6])
    assert restored.model.tolist() == ["clinical_llm:C1"] * 3
    assert restored.seed.tolist() == [0] * 3 and restored.fold.tolist() == [1] * 3
    with pytest.raises(FileExistsError):
        export_predictions(result, out)
    second = export_predictions(result, tmp_path / "other.csv.gz")
    assert out.read_bytes() == second.read_bytes()
    checkpoint = save_inference_checkpoint(run.model, tmp_path / "checkpoint", epochs_completed=1, optimizer_steps=1)
    target = trainer(mode)
    metadata = load_inference_checkpoint(target.model, checkpoint)
    assert metadata["kind"] == "inference_only_not_training_resume"
    loaded = target.evaluate(loader(IDS[3:6], 2))
    np.testing.assert_array_equal(result.predictions.yhat, loaded.predictions.yhat)
    from safetensors.torch import load_file
    weights = load_file(str(checkpoint / "trainable.safetensors"))
    assert all(n.startswith("head.") or "lora_" in n for n in weights)
    with pytest.raises(FileExistsError):
        save_inference_checkpoint(run.model, checkpoint)


@pytest.mark.parametrize("mode", ["frozen", "lora"])
def test_optimizer_groups(mode):
    run = trainer(mode)
    groups = run.optimizer.param_groups
    assert groups[0]["lr"] == 1e-4 and groups[0]["name"] == "regression_head"
    assert len(groups) == (1 if mode == "frozen" else 2)
    if mode == "lora":
        assert groups[1]["lr"] == 2e-5 and groups[1]["name"] == "lora"
    actual = {id(p) for g in groups for p in g["params"]}
    assert actual == {id(p) for p in run.model.parameters() if p.requires_grad}
    assert all(g["weight_decay"] == .01 for g in groups)


@pytest.mark.parametrize("mode", ["frozen", "lora"])
@pytest.mark.parametrize("batch_size,accumulation,n,expected_steps", [(1, 3, 3, 1), (2, 2, 3, 1), (1, 2, 3, 2), (2, 2, 5, 2)])
def test_accumulation_matches_explicit_sample_weighted_groups(batch_size, accumulation, n, expected_steps, mode):
    ids = IDS[:n]
    overrides = [f"training.gradient_accumulation={accumulation}", "training.gradient_clip_norm=100000"]
    run = trainer(mode, overrides=overrides, train=ids, validation=())
    reference = trainer(mode, overrides=overrides, train=ids, validation=())
    batches = list(loader(ids, batch_size))
    for offset in range(0, len(batches), accumulation):
        group = batches[offset:offset + accumulation]
        clinical = [c for batch in group for c in batch["clinical"]]
        targets = torch.cat([batch["moca"] for batch in group])
        tokens = reference.tokenizer(reference.serializer.serialize_batch(clinical))
        reference.model.train()
        reference.optimizer.zero_grad(set_to_none=True)
        moca_huber_loss(forward(reference.model, tokens), targets).backward()
        reference.optimizer.step()
        reference.optimizer.zero_grad(set_to_none=True)
    result = run.train_epoch(loader(ids, batch_size))
    assert result["optimizer_steps"] == expected_steps
    for (name, a), b in zip(run.model.named_parameters(), reference.model.parameters()):
        if a.requires_grad:
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6, msg=name)
    assert all(p.grad is None for p in run.model.parameters())


@pytest.mark.parametrize("train,validation", [(IDS[6:], ()), (IDS[:2], IDS[1:3]), (IDS[:2], IDS[6:]), (IDS[:1] * 2, ())])
def test_split_construction_guard(train, validation):
    with pytest.raises(ValueError):
        trainer(train=train, validation=validation)


def test_outer_test_cannot_train_or_validate_and_coverage_is_strict():
    run = trainer()
    before = run.model.head.weight.detach().clone()
    with pytest.raises(ValueError):
        run.train_epoch(loader(IDS[6:]))
    assert torch.equal(before, run.model.head.weight)
    with pytest.raises(ValueError):
        run.evaluate(loader(IDS[6:]))
    assert run.evaluate(loader(IDS[6:]), role="test").metrics["n"] == 1
    batches = list(loader(IDS[:3]))
    with pytest.raises(ValueError, match="Duplicate"):
        run.train_epoch([batches[0], batches[0]])
    with pytest.raises(ValueError, match="omitted"):
        run.evaluate(list(loader(IDS[3:5])))
    with pytest.raises(ValueError, match="omitted"):
        run.train_epoch(batches[:1])


def test_eeg_not_read_label_leakage_and_contamination_guard():
    run = trainer()
    batches = list(loader(IDS[:3]))
    for batch in batches:
        for key in ("eeg_features", "eeg_mask", "artifact_flags"):
            del batch[key]
    assert run.train_epoch(batches)["n"] == 3
    source = list(loader(IDS[3:6]))
    result = run.evaluate(source)
    for batch in source:
        batch["moca"].fill_(29.875)
    changed = run.evaluate(source)
    np.testing.assert_array_equal(changed.predictions.yhat, result.predictions.yhat)
    assert not np.array_equal(changed.predictions.y, result.predictions.y)
    batches[0]["clinical"][0]["moca"] = 29.875
    with pytest.raises(ValueError):
        run.train_epoch(batches)
    assert all(p.grad is None for p in run.model.parameters())


def test_determinism_and_configuration_legacy_roundtrip(tmp_path):
    seed_training(11)
    first = trainer()
    info = first.train_epoch(loader(IDS[:3]))
    seed_training(11)
    second = trainer()
    assert second.train_epoch(loader(IDS[:3])) == info
    torch.testing.assert_close(first.model.head.weight, second.model.head.weight, rtol=0, atol=0)
    for name in ("base", "clinical_only", "fusion_a", "fusion_b"):
        cfg = load_multimodal_config(CONFIG_DIR / f"{name}.json")
        path = tmp_path / f"{name}.json"
        save_resolved_config(cfg, path)
        assert load_multimodal_config(path).to_dict() == cfg.to_dict()
        legacy = cfg.to_dict()
        del legacy["settings"]["clinical_regression"]
        path.write_text(json.dumps(legacy))
        assert load_multimodal_config(path).settings["clinical_regression"]["head_type"] == "linear"


def test_private_paths_and_corrupted_incompatible_checkpoints(tmp_path):
    run = trainer()
    result = run.evaluate(loader(IDS[3:6]))
    with pytest.raises(ValueError):
        export_predictions(result, REPOSITORY / "baseline_predictions.csv.gz")
    with pytest.raises(ValueError):
        save_inference_checkpoint(run.model, REPOSITORY / "bad_checkpoint")
    bad_result = copy.deepcopy(result)
    bad_result.predictions.loc[1, "subject"] = bad_result.predictions.loc[0, "subject"]
    with pytest.raises(ValueError):
        export_predictions(bad_result, tmp_path / "duplicate.csv")
    checkpoint = save_inference_checkpoint(run.model, tmp_path / "checkpoint")
    other = trainer("lora")
    before = other.model.head.weight.detach().clone()
    with pytest.raises(ValueError, match="Incompatible"):
        load_inference_checkpoint(other.model, checkpoint)
    assert torch.equal(before, other.model.head.weight)
    raw = checkpoint / "trainable.safetensors"
    raw.write_bytes(raw.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="corrupted"):
        load_inference_checkpoint(run.model, checkpoint)
    with pytest.raises(ValueError):
        load_inference_checkpoint(run.model, tmp_path / "untrusted.pkl")


def test_cpu_rejects_cuda_mixed_precision():
    with pytest.raises(ValueError, match="BF16"):
        trainer(overrides=["clinical_regression.mixed_precision=bfloat16"])


def test_fixed_split_must_match_configured_seed_fold():
    with pytest.raises(ValueError, match="configured selection"):
        ClinicalTrainer(make_model(), split=SubjectSplit(99, 1, IDS[:6], IDS[6:]), train_subjects=IDS[:3])


def test_m1_inherits_cli_overrides_and_snapshots(tmp_path):
    path = tmp_path / "child.json"
    path.write_text(json.dumps({"extends": str(CONFIG_DIR / "clinical_only.json"),
                               "clinical_regression": {"random_seed": 23}}))
    cfg = load_multimodal_config(path, ["clinical_regression.random_seed=42", "training.lr_new_modules=0.002"])
    assert cfg.settings["clinical_regression"]["random_seed"] == 42
    assert cfg.settings["training"]["lr_new_modules"] == .002
    saved = tmp_path / "snapshot.json"
    save_resolved_config(cfg, saved)
    assert load_multimodal_config(saved).to_dict() == cfg.to_dict()


@pytest.mark.parametrize("mode", ["frozen", "lora"])
def test_checkpoint_restores_fully_frozen_inference_model(mode, tmp_path):
    source = trainer(mode)
    source.train_epoch(loader(IDS[:3]))
    expected = source.evaluate(loader(IDS[3:6])).predictions.yhat.to_numpy()
    checkpoint = save_inference_checkpoint(source.model, tmp_path / "checkpoint")
    target = trainer(mode)
    target.model.requires_grad_(False)
    load_inference_checkpoint(target.model, checkpoint)
    actual = target.evaluate(loader(IDS[3:6])).predictions.yhat.to_numpy()
    np.testing.assert_array_equal(actual, expected)
    assert all(not p.requires_grad for p in target.model.parameters())
    # Saving an inference-frozen model must also preserve the head and adapters.
    second = save_inference_checkpoint(target.model, tmp_path / "frozen-checkpoint")
    fresh = trainer(mode)
    load_inference_checkpoint(fresh.model, second)
    np.testing.assert_array_equal(fresh.evaluate(loader(IDS[3:6])).predictions.yhat, expected)


def test_completed_optimizer_steps_recorded_if_later_batch_fails():
    run = trainer(overrides=["training.gradient_accumulation=1"])
    batches = list(loader(IDS[:3]))
    with pytest.raises(ValueError, match="Duplicate"):
        run.train_epoch([batches[0], batches[0]])
    assert run.optimizer_steps == 1 and run.epochs_completed == 0
    assert all(p.grad is None for p in run.model.parameters())
