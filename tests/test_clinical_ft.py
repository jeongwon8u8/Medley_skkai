"""Synthetic leakage, checkpoint and reusable representation tests."""
import numpy as np
import pandas as pd
import torch

from codl.clinical.ft_clinical import (
    ClinicalFT, DEFAULTS, fit_member, infer_member, load_member, restore_prep, tensors,
)
from codl.clinical.run_pd_ft import inner_indices


def synthetic_fit():
    rng = np.random.default_rng(3)
    train_x = np.column_stack([rng.normal(60, 4, 20), rng.integers(0, 2, 20)])
    train_x[0, 0] = np.nan
    val_x = np.array([[1000., 0.], [1100., 1.]])
    train_y = rng.normal(24, 2, 20)
    val_y = np.array([100., 102.])
    settings = {**DEFAULTS, 'nn_epochs': 5, 'nn_patience': 3, 'min_delta': 1e-5}
    torch.set_num_threads(1)
    payload, log = fit_member(train_x, train_y, val_x, val_y, ['age', 'sex'], settings, 0)
    return payload, log, train_x, train_y, val_x


def test_preprocessing_ignores_validation_and_target_scale_is_train_only():
    payload, _, tx, ty, _ = synthetic_fit()
    np.testing.assert_allclose(payload['prep']['mu'][0], np.nanmean(tx[:, 0]))
    np.testing.assert_allclose(payload['prep']['sd'][0], np.nanstd(tx[:, 0]))
    assert payload['prep']['cat'] == [False, True]
    np.testing.assert_allclose(payload['y_mean'], ty.mean())
    np.testing.assert_allclose(payload['y_std'], ty.std())


def test_checkpoint_roundtrip_and_vector_path(tmp_path):
    payload, log, _, _, vx = synthetic_fit()
    path = tmp_path / 'synthetic.pt'
    torch.save(payload, path)
    p1, hidden = infer_member(path, vx)
    model, prep, saved = load_member(path)
    x, missing = tensors(prep, vx)
    vector = model.encode(x, missing)
    assert hidden.shape == (2, 32)
    assert vector['tokens'].shape == (2, 2, 8)
    assert vector['flat'].shape == (2, 16)
    reconstructed = model.net.mlp[3:](vector['clinical_vector']).squeeze(-1)
    torch.testing.assert_close(reconstructed, model(x, missing))
    np.testing.assert_allclose(p1, reconstructed.detach().numpy() * saved['y_std'] + saved['y_mean'])
    assert saved['selected_epoch'] == [r['epoch'] for r in log if r['selected']][-1]
    vector['clinical_vector'].sum().backward()
    assert model.net.tok.W.grad is not None


def test_missing_numeric_and_sex_tokenization():
    payload, _, _, _, _ = synthetic_fit()
    prep = restore_prep(payload['prep'])
    model = ClinicalFT(prep.cat, payload['settings'])
    x, missing = tensors(prep, [[np.nan, 0], [65, 1], [70, np.nan]])
    tokens = model.encode(x, missing)['tokens']
    tok = model.net.tok
    torch.testing.assert_close(tokens[0, 0], tok.miss[0])
    torch.testing.assert_close(tokens[0, 1], tok.cat_emb[1, 0] + tok.b[1])
    torch.testing.assert_close(tokens[1, 1], tok.cat_emb[1, 1] + tok.b[1])
    torch.testing.assert_close(tokens[1, 0], x[1, 0] * tok.W[0] + tok.b[0])
    torch.testing.assert_close(tokens[2, 1], tok.miss[1])


def test_fresh_seeded_training_is_reproducible_and_split_is_disjoint():
    first, _, _, _, _ = synthetic_fit()
    second, _, _, _, _ = synthetic_fit()
    assert first['initial_state_sha256'] == second['initial_state_sha256']
    assert first['selected_state_sha256'] == second['selected_state_sha256']
    for a, b in zip(first['state_dict'].values(), second['state_dict'].values()):
        assert a.data_ptr() != b.data_ptr()
    for n in [90, 91]:
        tr, va = inner_indices(n, 0, .2)
        assert len(va) == 18 and not set(tr) & set(va)
        assert set(tr) | set(va) == set(range(n))


def test_prediction_csv_preserves_float32_values_after_float64_cast(tmp_path):
    predictions = np.array([23.123456, 19.234567], dtype=np.float32).astype(np.float64)
    path = tmp_path / 'predictions.csv'
    pd.DataFrame({'yhat': predictions}).to_csv(path, index=False)
    np.testing.assert_allclose(pd.read_csv(path).yhat, predictions, rtol=0, atol=1e-12)
