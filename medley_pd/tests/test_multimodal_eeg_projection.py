"""Independent projection and Task 3 -> pooling -> projection smoke tests."""
import json
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
import torch

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from multimodal.config import CONFIG_DIR, ConfigError, load_multimodal_config, save_resolved_config
from multimodal.eeg_pooling import EEGFeatureAdapter, LearnedQueryAttentionPooling, seed_eeg_modules
from multimodal.eeg_projection import EEGToLLMProjector


@pytest.fixture(scope="module", autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(*overrides):
    return load_multimodal_config(CONFIG_DIR / "fusion_b.json", overrides)


@pytest.mark.parametrize("batch,k,hidden", [(1, 1, 8), (3, 4, 32), (2, 7, 19)])
@pytest.mark.parametrize("normalize", [False, True])
def test_projection_shapes_gradient_and_actual_width(batch, k, hidden, normalize):
    projector = EEGToLLMProjector(256, hidden, layer_norm=normalize)
    x = torch.randn(batch, k, 256, requires_grad=True)
    out = projector(x)
    assert out.shape == (batch, k, hidden) and torch.isfinite(out).all()
    assert out.device == x.device
    assert isinstance(projector.norm, torch.nn.LayerNorm if normalize else torch.nn.Identity)
    expected = projector.norm(projector.linear(x))
    torch.testing.assert_close(out, expected)
    (out * torch.randn_like(out)).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    for name, parameter in projector.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_config_output_dim_matches_metadata_without_loading_llm():
    cfg = config()
    with pytest.raises(ValueError, match="actual backbone.hidden_dim"):
        EEGToLLMProjector.from_config(cfg)
    # Stand-in for the Task 4 metadata property only, never a fake LLM forward.
    metadata = SimpleNamespace(hidden_dim=37)
    assert EEGToLLMProjector.from_config(cfg, hidden_dim=metadata.hidden_dim).output_dim == 37
    explicit = config("eeg.projector.output_dim=37")
    assert EEGToLLMProjector.from_config(explicit).output_dim == 37
    with pytest.raises(ValueError, match="disagrees"):
        EEGToLLMProjector.from_config(explicit, hidden_dim=32)


@pytest.mark.parametrize("input_dim,output_dim,normalize", [(0, 16, True), (256, 0, True), (True, 16, True),
    (256, False, True), (256, 1.5, True), (256, 16, "yes")])
def test_invalid_constructor(input_dim, output_dim, normalize):
    with pytest.raises(ValueError):
        EEGToLLMProjector(input_dim, output_dim, layer_norm=normalize)


@pytest.mark.parametrize("x", [torch.empty(0, 4, 8), torch.empty(1, 0, 8), torch.ones(1, 4, 7),
    torch.ones(4, 8), torch.ones(1, 4, 8, dtype=torch.long), torch.full((1, 4, 8), float("inf")),
    torch.ones(1, 4, 8, dtype=torch.float64)])
def test_invalid_projection_inputs(x):
    with pytest.raises(ValueError):
        EEGToLLMProjector(8, 16)(x)


def test_bfloat16_modules_and_differentiable_llm_dtype_cast():
    pool = LearnedQueryAttentionPooling(6, hidden_dim=8, num_heads=2).to(dtype=torch.bfloat16)
    projector = EEGToLLMProjector(8, 16).to(dtype=torch.bfloat16)
    x = torch.randn(2, 3, 6, dtype=torch.bfloat16, requires_grad=True)
    result = projector(pool(x, torch.ones(2, 3, dtype=torch.bool)))
    assert result.dtype == torch.bfloat16 and torch.isfinite(result).all()
    # A2 must explicitly match real LLM embedding dtype; cast preserves autograd.
    (result.to(torch.float32) * torch.randn(2, 4, 16)).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert projector.linear.weight.grad.abs().sum() > 0


def test_cpu_bfloat16_autocast_keeps_gradients():
    pool = LearnedQueryAttentionPooling(6, hidden_dim=8, num_heads=2)
    projector = EEGToLLMProjector(8, 16)
    x = torch.randn(2, 3, 6, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = projector(pool(x, torch.ones(2, 3, dtype=torch.bool)))
    assert result.dtype == torch.bfloat16
    (result.float() * torch.randn(2, 4, 16)).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("encoder,width", [("cbramod", 3800), ("scerebro", 3420)])
def test_task3_dataset_collator_pooling_projection_backward_smoke(encoder, width):
    from multimodal.data_adapter import SubjectRecord
    from multimodal.dataset import MedleySubjectDataset, MedleySubjectCollator

    cfg = config(f"eeg.encoder={encoder}", f"eeg.feature_dim={width}", "eeg.projector.output_dim=32")
    seed_eeg_modules(cfg)
    ids = ("synthetic-a", "synthetic-b", "synthetic-c")
    records = {sid: SubjectRecord(sid, np.random.default_rng(i).normal(size=(i+1, width)).astype(np.float32),
        np.ones(i+1, dtype=bool), MappingProxyType(dict(age=60., sex=None, updrs_part_iii=None,
            disease_duration=None, ledd=None)), 22., None) for i, sid in enumerate(ids)}
    adapter = SimpleNamespace(subject_ids=ids, clinical_variables=cfg.clinical_variables, get_subject=records.__getitem__)
    data = MedleySubjectDataset(adapter, ids)
    batch = MedleySubjectCollator()([data[i] for i in range(len(data))])
    assert batch["subject_ids"] == list(ids)
    assert batch["eeg_mask"].sum(dim=1).tolist() == [1, 2, 3]
    assert batch["artifact_flags"][batch["eeg_mask"]].all()  # All real segments flagged and retained.
    batch["eeg_features"].requires_grad_(True)
    feature_adapter = EEGFeatureAdapter.from_config(cfg)
    pool = LearnedQueryAttentionPooling.from_config(cfg)
    projector = EEGToLLMProjector.from_config(cfg)
    pooled = pool(feature_adapter(batch["eeg_features"]), batch["eeg_mask"], artifact_flags=batch["artifact_flags"])
    tokens = projector(pooled)
    assert pooled.shape == (3, 4, 256) and tokens.shape == (3, 4, 32)
    (tokens * torch.randn_like(tokens)).mean().backward()
    for module in (pool, projector):
        for name, p in module.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
    grad = batch["eeg_features"].grad
    assert torch.equal(grad[~batch["eeg_mask"]], torch.zeros_like(grad[~batch["eeg_mask"]]))
    assert grad[batch["eeg_mask"]].abs().sum() > 0


@pytest.mark.parametrize("name", ["base", "clinical_only", "fusion_a", "fusion_b"])
def test_all_json_old_snapshots_and_resolved_roundtrip(name, tmp_path):
    cfg = load_multimodal_config(CONFIG_DIR / f"{name}.json")
    assert cfg.settings["eeg"]["seed"] == 0
    assert cfg.settings["eeg"]["projector"]["output_dim"] is None
    snapshot = tmp_path / f"{name}.json"
    save_resolved_config(cfg, snapshot)
    assert load_multimodal_config(snapshot).to_dict() == cfg.to_dict()
    legacy = cfg.to_dict()
    del legacy["settings"]["eeg"]["seed"]
    del legacy["settings"]["eeg"]["projector"]["output_dim"]
    snapshot.write_text(json.dumps(legacy))
    assert load_multimodal_config(snapshot).to_dict() == cfg.to_dict()


def test_inheritance_overrides_seed_and_projector_reproducibility(tmp_path):
    child = tmp_path / "child.json"
    child.write_text(json.dumps({"extends": str(CONFIG_DIR / "fusion_b.json"),
                                "eeg": {"seed": 13, "projector": {"output_dim": 48}}}))
    cfg = load_multimodal_config(child, ["eeg.pooling.num_queries=7", "eeg.pooling.hidden_dim=32",
        "eeg.pooling.num_heads=8", "eeg.projector.output_dim=24", "eeg.projector.layer_norm=false"])
    assert cfg.settings["eeg"]["seed"] == 13
    seed_eeg_modules(cfg)
    first_pool, first_projection = LearnedQueryAttentionPooling.from_config(cfg), EEGToLLMProjector.from_config(cfg)
    seed_eeg_modules(cfg)
    second_pool, second_projection = LearnedQueryAttentionPooling.from_config(cfg), EEGToLLMProjector.from_config(cfg)
    x, mask = torch.randn(2, 3, 3800), torch.ones(2, 3, dtype=torch.bool)
    first = first_projection(first_pool(x, mask))
    second = second_projection(second_pool(x, mask))
    assert first.shape == (2, 7, 24)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("override", ["eeg.seed=-1", "eeg.seed=true", "eeg.seed=4294967296",
    "eeg.projector.output_dim=0", "eeg.projector.output_dim=true", "eeg.projector.output_dim=3.5",
    "eeg.pooling.type=mean", "eeg.projector.type=mlp", "eeg.pooling.num_heads=0",
    "eeg.pooling.hidden_dim=7", "eeg.pooling.num_queries=0"])
def test_invalid_or_unimplemented_config_options(override):
    with pytest.raises(ConfigError):
        config(override)


def test_projection_rejects_device_mismatch_without_moving_inputs():
    projector = EEGToLLMProjector(8, 16).to("meta")
    x = torch.ones(1, 4, 8)
    with pytest.raises(ValueError, match="same device"):
        projector(x)
    assert x.device.type == "cpu"


def test_modules_do_not_import_or_load_an_llm():
    code = '''
import sys
sys.path.insert(0, sys.argv[1])
from multimodal.config import CONFIG_DIR, load_multimodal_config
from multimodal.eeg_pooling import EEGFeatureAdapter, LearnedQueryAttentionPooling
from multimodal.eeg_projection import EEGToLLMProjector
cfg = load_multimodal_config(CONFIG_DIR / 'fusion_b.json')
EEGFeatureAdapter.from_config(cfg)
LearnedQueryAttentionPooling.from_config(cfg)
EEGToLLMProjector.from_config(cfg, hidden_dim=16)
assert not any(k.split('.')[0] in {'transformers','peft','tokenizers'} for k in sys.modules)
'''
    subprocess.run([sys.executable, "-c", code, str(APP)], check=True, capture_output=True, text=True)
