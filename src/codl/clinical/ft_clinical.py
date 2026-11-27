"""Team FT tokenizer/head with strictly inner-train-only fitting.

Uses fusion_nn's modules directly; only the training orchestration is new.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'medley_pd'))
# Import registry first, matching Medley's normal circular-import order.
from medley import fusion  # noqa: F401,E402
from medley.fusion_nn import ClinicalPrep, DEFAULTS, _modules  # noqa: E402


class ClinicalFT(nn.Module):
    """Unchanged team Net with an explicit differentiable vector interface."""

    def __init__(self, cat_mask, settings):
        super().__init__()
        _, Net = _modules()
        self.net = Net(0, cat_mask, settings['nn_token_dim'],
                       settings['nn_hidden'], settings['nn_dropout'])

    def encode(self, x, missing):
        tokens = self.net.tok(x, missing)
        flat = tokens.flatten(1)
        # Pre-dropout GELU hidden vector, 32 dimensions with team defaults.
        hidden = self.net.mlp[:3](flat)
        return {'tokens': tokens, 'flat': flat, 'clinical_vector': hidden}

    def forward(self, x, missing):
        return self.net(None, x, missing)


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def prep_state(prep):
    return {'cat': prep.cat.tolist(), 'mu': prep.mu.tolist(), 'sd': prep.sd.tolist()}


def restore_prep(state):
    prep = ClinicalPrep()
    prep.cat = np.asarray(state['cat'], dtype=bool)
    prep.mu = np.asarray(state['mu'], dtype=float)
    prep.sd = np.asarray(state['sd'], dtype=float)
    return prep


def tensors(prep, values):
    values = np.asarray(values, dtype=float)
    if np.isinf(values).any():
        raise ValueError('Infinite clinical value')
    cats = values[:, prep.cat]
    observed = cats[~np.isnan(cats)]
    if not np.isin(observed, [0, 1]).all():
        raise ValueError('Sex must be encoded as 0/1 or missing')
    x, missing = prep.transform(values)
    return torch.from_numpy(x), torch.from_numpy(missing)


def fit_member(train_x, train_y, val_x, val_y, columns, settings, seed):
    """Test inputs are deliberately absent from this API."""
    started = time.perf_counter()
    torch.manual_seed(seed)
    prep = ClinicalPrep().fit(train_x)
    if prep.cat.tolist() != [c == 'sex' for c in columns]:
        raise ValueError('Team categorical inference must identify only sex')
    if np.isnan(np.asarray(train_x, float)).all(axis=0).any():
        raise ValueError('All-missing inner-training feature')
    tx, tm = tensors(prep, train_x)
    vx, vm = tensors(prep, val_x)
    ym = float(np.mean(train_y))
    ys = float(np.std(train_y))
    ys = ys if ys > 1e-8 else 1.0
    ty = torch.as_tensor((np.asarray(train_y) - ym) / ys, dtype=torch.float32)
    vy = torch.as_tensor((np.asarray(val_y) - ym) / ys, dtype=torch.float32)
    val_raw = torch.tensor(np.array(val_y, copy=True), dtype=torch.float32)
    model = ClinicalFT(prep.cat, settings)
    initial_hash = state_hash(model)
    opt = torch.optim.AdamW(model.parameters(), lr=settings['nn_lr'], weight_decay=settings['nn_wd'])
    loss_fn = nn.HuberLoss(delta=1.0)
    best, wait, best_epoch, best_state = np.inf, 0, 0, None
    log = []
    for epoch in range(1, settings['nn_epochs'] + 1):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(tx, tm), ty)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vp = model(vx, vm)
            vl = float(loss_fn(vp, vy))
            mae = float(torch.mean(torch.abs(vp * ys + ym - val_raw)))
        if not np.isfinite(vl):
            raise ValueError('Nonfinite validation loss')
        improved = vl < best - settings['min_delta']
        log.append({'epoch': epoch, 'train_huber': float(loss.detach()), 'val_huber': vl,
                    'val_MAE': mae, 'selected': improved})
        if improved:
            best, best_epoch, wait = vl, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= settings['nn_patience']:
            break
    model.load_state_dict(best_state)
    model.eval()
    payload = {'state_dict': best_state, 'prep': prep_state(prep), 'y_mean': ym, 'y_std': ys,
               'columns': list(columns), 'settings': dict(settings), 'seed': int(seed),
               'selected_epoch': best_epoch, 'best_val_huber': best,
               'epochs_ran': len(log), 'initial_state_sha256': initial_hash,
               'selected_state_sha256': state_hash(model), 'seconds': time.perf_counter() - started}
    return payload, log


def load_member(path):
    """Load own state-dict checkpoint; use model.encode for downstream GNNs."""
    payload = torch.load(path, map_location='cpu', weights_only=True)
    prep = restore_prep(payload['prep'])
    model = ClinicalFT(prep.cat, payload['settings'])
    model.load_state_dict(payload['state_dict'])
    model.eval()
    return model, prep, payload


def infer_member(path, values):
    model, prep, payload = load_member(path)
    x, missing = tensors(prep, values)
    with torch.no_grad():
        pred = model(x, missing).numpy() * payload['y_std'] + payload['y_mean']
        hidden = model.encode(x, missing)['clinical_vector'].numpy()
    return pred, hidden
