"""Synthetic cached-EEG attention contracts: no model/data downloads or CUDA."""
from pathlib import Path
import sys

import pytest
import torch

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from multimodal.config import CONFIG_DIR, load_multimodal_config
from multimodal.eeg_pooling import EEGFeatureAdapter, LearnedQueryAttentionPooling, seed_eeg_modules


@pytest.fixture(scope="module", autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(encoder="cbramod", *overrides):
    return load_multimodal_config(CONFIG_DIR / "fusion_b.json", [f"eeg.encoder={encoder}",
        f"eeg.feature_dim={3800 if encoder == 'cbramod' else 3420}", *overrides])


@pytest.mark.parametrize("encoder,width", [("cbramod", 3800), ("scerebro", 3420)])
@pytest.mark.parametrize("batch,length,k", [(1, 1, 4), (3, 5, 4), (2, 2, 1), (1, 2, 7)])
def test_encoder_width_shapes_and_queries(encoder, width, batch, length, k):
    cfg = config(encoder, f"eeg.pooling.num_queries={k}")
    pool = LearnedQueryAttentionPooling.from_config(cfg)
    features = torch.randn(batch, length, width)
    mask = torch.ones(batch, length, dtype=torch.bool)
    if batch > 1 and length > 1:
        mask[0, 1:] = False
    out = pool(features, mask)
    assert out.shape == (batch, k, 256)
    assert torch.isfinite(out).all() and out.device == features.device
    assert isinstance(pool.queries, torch.nn.Parameter) and pool.queries.requires_grad


def test_padding_polarity_values_length_and_per_subject_equivalence():
    torch.manual_seed(51)
    pool = LearnedQueryAttentionPooling(6, hidden_dim=8, num_heads=2).eval()
    x = torch.randn(2, 4, 6)
    mask = torch.tensor([[True, False, True, False], [False, True, True, True]])
    original = pool(x, mask)
    changed = x.clone()
    changed[~mask] = torch.finfo(x.dtype).max
    torch.testing.assert_close(pool(changed, mask), original, rtol=1e-6, atol=1e-6)
    for value in (float("nan"), float("inf"), float("-inf")):
        changed[~mask] = value
        torch.testing.assert_close(pool(changed, mask), original, rtol=1e-6, atol=1e-6)
    extended = torch.cat([x, torch.full((2, 3, 6), float("nan"))], dim=1)
    extended_mask = torch.cat([mask, torch.zeros(2, 3, dtype=torch.bool)], dim=1)
    torch.testing.assert_close(pool(extended, extended_mask), original, rtol=1e-6, atol=1e-6)
    for row in range(2):
        real = x[row, mask[row]].unsqueeze(0)
        solo = pool(real, torch.ones(real.shape[:2], dtype=torch.bool))
        torch.testing.assert_close(solo[0], original[row], rtol=1e-6, atol=1e-6)
    # A real feature change must still affect attention output.
    changed = x.clone()
    changed[mask] += 3
    assert not torch.allclose(pool(changed, mask), original)


def test_no_padded_gradients_all_pool_parameters_receive_finite_gradients():
    torch.manual_seed(8)
    pool = LearnedQueryAttentionPooling(6, num_queries=4, hidden_dim=8, num_heads=2)
    x = torch.randn(2, 5, 6, requires_grad=True)
    mask = torch.tensor([[True, False, True, False, False], [True, True, True, True, False]])
    # Also verify backward is safe when unused padding contains NaN.
    with torch.no_grad():
        x[~mask] = float("nan")
    out = pool(x, mask)
    (out * torch.randn_like(out)).sum().backward()
    assert torch.equal(x.grad[~mask], torch.zeros_like(x.grad[~mask]))
    assert torch.isfinite(x.grad).all() and x.grad[mask].abs().sum() > 0
    for name, parameter in pool.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_artifacts_are_not_padding_and_input_is_not_mutated():
    pool = LearnedQueryAttentionPooling(6, hidden_dim=8, num_heads=2)
    x = torch.randn(2, 4, 6)
    mask = torch.ones(2, 4, dtype=torch.bool)
    mask[0, 2:] = False
    before = x.clone()
    result = pool(x, mask)
    torch.testing.assert_close(pool(x, mask, artifact_flags=torch.ones_like(mask)), result, rtol=0, atol=0)
    torch.testing.assert_close(pool(x, mask, artifact_flags=torch.zeros_like(mask)), result, rtol=0, atol=0)
    assert torch.equal(before, x)


@pytest.mark.parametrize("kwargs", [{"input_dim": 0}, {"input_dim": True}, {"input_dim": 3.5},
    {"num_queries": 0}, {"num_queries": -1}, {"hidden_dim": 0}, {"num_heads": 0},
    {"num_heads": True}, {"hidden_dim": 7, "num_heads": 2}])
def test_invalid_dimensions(kwargs):
    with pytest.raises(ValueError):
        LearnedQueryAttentionPooling(**{**dict(input_dim=6, hidden_dim=8, num_heads=2), **kwargs})


@pytest.mark.parametrize("case", ["empty_batch", "empty_time", "wrong_width", "rank", "integer", "float_mask",
    "mask_shape", "all_masked", "one_all_masked", "nonfinite_real", "artifact_dtype", "artifact_shape", "dtype"])
def test_invalid_inputs(case):
    pool = LearnedQueryAttentionPooling(6, hidden_dim=8, num_heads=2)
    x, mask, kwargs = torch.randn(2, 3, 6), torch.ones(2, 3, dtype=torch.bool), {}
    if case == "empty_batch": x, mask = x[:0], mask[:0]
    elif case == "empty_time": x, mask = x[:, :0], mask[:, :0]
    elif case == "wrong_width": x = x[:, :, :5]
    elif case == "rank": x = x[0]
    elif case == "integer": x = x.long()
    elif case == "float_mask": mask = mask.float()
    elif case == "mask_shape": mask = mask[:, :2]
    elif case == "all_masked": mask[:] = False
    elif case == "one_all_masked": mask[0] = False
    elif case == "nonfinite_real": x[0, 0, 0] = float("nan")
    elif case == "artifact_dtype": kwargs["artifact_flags"] = mask.float()
    elif case == "artifact_shape": kwargs["artifact_flags"] = mask[:, :2]
    elif case == "dtype": x = x.double()
    with pytest.raises(ValueError):
        pool(x, mask, **kwargs)


def test_adapter_off_is_exact_identity_and_on_is_explicitly_deferred():
    cfg = config()
    adapter = EEGFeatureAdapter.from_config(cfg)
    x = torch.randn(1, 2, 3800, requires_grad=True)
    assert adapter(x) is x
    assert list(adapter.parameters()) == []
    adapter(x).sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))
    cfg.settings["eeg"]["feature_adapter"]["enabled"] = True
    for factory in (EEGFeatureAdapter.from_config, LearnedQueryAttentionPooling.from_config):
        with pytest.raises(NotImplementedError, match="Phase D"):
            factory(cfg)


@pytest.mark.parametrize("encoder,count", [("cbramod", 984328), ("scerebro", 885908)])
def test_adapter_from_config_bottleneck_parameter_counts(encoder, count, caplog):
    cfg = config(encoder, "ablation.enabled=true", "eeg.feature_adapter.enabled=true")
    with caplog.at_level("INFO", logger="multimodal.eeg_pooling"):
        adapter = EEGFeatureAdapter.from_config(cfg)
    assert sum(p.numel() for p in adapter.parameters() if p.requires_grad) == count
    assert f"trainable_parameters={count}" in caplog.text
    cfg.settings["eeg"]["feature_adapter"].update(hidden_dim=12, dropout=0.25)
    adapter = EEGFeatureAdapter.from_config(cfg)
    assert adapter.down.out_features == adapter.up.in_features == 12
    assert adapter.dropout.p == 0.25


def test_configured_seed_reproduces_initialization_without_implicit_resets():
    cfg = config("cbramod", "eeg.seed=13", "eeg.pooling.hidden_dim=16")
    seed_eeg_modules(cfg)
    a = LearnedQueryAttentionPooling.from_config(cfg)
    b = LearnedQueryAttentionPooling.from_config(cfg)
    assert not torch.equal(a.queries, b.queries)
    seed_eeg_modules(cfg)
    restored = LearnedQueryAttentionPooling.from_config(cfg)
    assert all(torch.equal(p, restored.state_dict()[n]) for n, p in a.state_dict().items())
    seed_eeg_modules(config("cbramod", "eeg.seed=14", "eeg.pooling.hidden_dim=16"))
    different = LearnedQueryAttentionPooling.from_config(cfg)
    assert not torch.equal(a.queries, different.queries)


def test_no_implicit_device_transfer():
    pool = LearnedQueryAttentionPooling(6, hidden_dim=8, num_heads=2)
    x, mask = torch.randn(1, 2, 6), torch.ones(1, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="same device"):
        pool(x.to("meta"), mask)
    with pytest.raises(ValueError, match="feature device"):
        pool(x, mask.to("meta"))
    with pytest.raises(ValueError, match="metadata"):
        pool(x, mask, artifact_flags=mask.to("meta"))
