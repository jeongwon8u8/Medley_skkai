"""Verify FT artifacts, saved checkpoints, inner partitions and OOF metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from codl.clinical.ft_clinical import ClinicalFT, load_member, tensors, state_hash
from codl.clinical.run_pd_ft import (
    comparison_table, inner_indices, load_reference, sha, validate_oof, write_json,
)
from medley.participants import clinical_sets


def audit(out):
    cfg, d, splits, ridge, _ = load_reference()
    manifest = json.loads((out / 'metadata.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'completed':
        raise ValueError('Training did not complete')
    for path, value in manifest['code_hashes'].items():
        assert sha(ROOT / path) == value, 'Implementation changed since training'
    settings, folds = manifest['settings'], manifest['folds']
    torch.set_num_threads(settings['threads'])
    pred = pd.read_csv(out / 'predictions.csv.gz', dtype={'subject': str})
    selected = pd.read_csv(out / 'selected_epochs.csv')
    assert len(selected) == len(folds) * 2 * settings['nn_members']
    assert not selected.duplicated(['model', 'fold', 'member']).any()
    subjects = splits.loc[splits.fold.isin(folds), 'subject'].tolist()
    validate_oof(pred, splits, subjects, ['clinical_ft:C0', 'clinical_ft:C1'])
    checks, metrics = [], []
    for fold in folds:
        test_ids = set(splits.loc[splits.fold == fold, 'subject'])
        outer = d.loc[~d.subject.isin(test_ids)].reset_index(drop=True)
        test = d.loc[d.subject.isin(test_ids)].reset_index(drop=True)
        for name, cols in clinical_sets(cfg).items():
            folder = out / name / f'fold{fold}'
            vectors = np.load(folder / 'clinical_vectors.npz', allow_pickle=False)
            np.testing.assert_array_equal(vectors['subjects'], test.subject)
            assert vectors['vectors'].shape == (len(test), 5, 32)
            reproduced = []
            for member in range(settings['nn_members']):
                record = selected[(selected.model == f'clinical_ft:{name}') &
                                  (selected.fold == fold) & (selected.member == member)].iloc[0]
                path = folder / f'member{member}.pt'
                assert sha(path) == record.checkpoint_sha256
                model, prep, payload = load_member(path)
                assert payload['settings'] == settings and payload['columns'] == cols
                assert prep.cat.tolist() == [c == 'sex' for c in cols]
                expected_seed = settings['seed'] * 1000 + member
                assert payload['seed'] == expected_seed
                tr, va = inner_indices(len(outer), expected_seed, settings['nn_val_frac'])
                partition = payload['partition']
                assert partition['train'] == outer.loc[tr, 'subject'].tolist()
                assert partition['validation'] == outer.loc[va, 'subject'].tolist()
                assert partition['test'] == test.subject.tolist()
                assert not set(partition['train']) & set(partition['validation'])
                assert not (set(partition['train']) | set(partition['validation'])) & test_ids
                train_x = outer.loc[tr, cols].to_numpy(float)
                numeric = ~prep.cat
                means = np.nanmean(train_x[:, numeric], axis=0)
                sd = np.nanstd(train_x[:, numeric], axis=0)
                sd = np.where(sd < 1e-8, 1., sd)
                np.testing.assert_allclose(prep.mu[numeric], means, rtol=0, atol=1e-12)
                np.testing.assert_allclose(prep.sd[numeric], sd, rtol=0, atol=1e-12)
                train_y = outer.loc[tr, 'target'].to_numpy()
                np.testing.assert_allclose(payload['y_mean'], train_y.mean(), rtol=0, atol=1e-12)
                np.testing.assert_allclose(payload['y_std'], train_y.std(), rtol=0, atol=1e-12)
                torch.manual_seed(expected_seed)
                fresh = ClinicalFT(prep.cat, settings)
                assert state_hash(fresh) == payload['initial_state_sha256']
                assert state_hash(model) == payload['selected_state_sha256']
                history = pd.read_csv(folder / f'member{member}_history.csv')
                assert history.epoch.tolist() == list(range(1, len(history) + 1))
                assert np.isfinite(history[['train_huber', 'val_huber', 'val_MAE']]).all().all()
                best, wait, best_epoch = np.inf, 0, None
                for row in history.itertuples():
                    improved = row.val_huber < best - settings['min_delta']
                    assert bool(row.selected) == improved
                    if improved:
                        best, wait, best_epoch = row.val_huber, 0, row.epoch
                    else:
                        wait += 1
                    if row.epoch < len(history):
                        assert wait < settings['nn_patience']
                assert len(history) == settings['nn_epochs'] or wait == settings['nn_patience']
                assert best_epoch == payload['selected_epoch'] == record.selected_epoch
                np.testing.assert_allclose(best, payload['best_val_huber'], rtol=0, atol=1e-12)
                vx, vm = tensors(prep, outer.loc[va, cols].to_numpy(float))
                vy = torch.as_tensor((outer.loc[va, 'target'].to_numpy() - payload['y_mean']) /
                                     payload['y_std'], dtype=torch.float32)
                x, missing = tensors(prep, test[cols].to_numpy(float))
                with torch.no_grad():
                    val_loss = float(torch.nn.functional.huber_loss(model(vx, vm), vy))
                    yp = model(x, missing).numpy() * payload['y_std'] + payload['y_mean']
                    hidden = model.encode(x, missing)['clinical_vector'].numpy()
                np.testing.assert_allclose(val_loss, best, rtol=0, atol=1e-7)
                np.testing.assert_array_equal(yp, vectors['member_predictions'][:, member])
                np.testing.assert_array_equal(hidden, vectors['vectors'][:, member, :])
                reproduced.append(yp)
                checks.append({'model': f'clinical_ft:{name}', 'fold': fold, 'member': member,
                               'inner_train_n': len(tr), 'validation_n': len(va), 'test_n': len(test),
                               'selected_epoch': best_epoch, 'checkpoint_preprocessing_partition_verified': True})
            saved = pred[(pred.model == f'clinical_ft:{name}') & (pred.fold == fold)].set_index('subject')
            np.testing.assert_array_equal(saved.loc[test.subject, 'y'], test.target)
            np.testing.assert_array_equal(saved.loc[test.subject, 'target'], test.target)
            np.testing.assert_allclose(saved.loc[test.subject, 'yhat'], np.stack(reproduced, axis=1).mean(axis=1),
                                       rtol=0, atol=1e-6)
    summary = pd.read_csv(out / 'summary.csv').set_index('model')
    per = pd.read_csv(out / 'metrics_per_seed.csv').set_index('model')
    for name, group in pred.groupby('model'):
        row = {'model': name, 'n': len(group), 'MAE': mean_absolute_error(group.y, group.yhat),
               'RMSE': float(np.sqrt(mean_squared_error(group.y, group.yhat))),
               'R2': r2_score(group.y, group.yhat), 'Spearman': float(spearmanr(group.y, group.yhat).statistic)}
        for key in ('MAE', 'RMSE', 'R2', 'Spearman'):
            np.testing.assert_allclose(row[key], summary.loc[name, 'avg_' + key], rtol=0, atol=1e-12)
            np.testing.assert_allclose(row[key], per.loc[name, key], rtol=0, atol=1e-12)
        metrics.append(row)
    pd.DataFrame(metrics).to_csv(out / 'verified_oof_metrics.csv', index=False)
    if folds == [1, 2, 3, 4, 5]:
        comparison = comparison_table(ridge, pred)
        saved = pd.read_csv(out / 'ridge_ft_comparison.csv')
        for key in ('MAE', 'RMSE', 'R2', 'Spearman', 'delta_MAE_vs_Ridge'):
            np.testing.assert_allclose(comparison[key], saved[key], rtol=0, atol=1e-12)
        print(comparison[['model', 'n', 'MAE', 'RMSE', 'R2', 'Spearman', 'delta_MAE_vs_Ridge']].to_string(index=False))
    write_json(out / 'audit.json', {'status': 'passed', 'n_oof_per_model': len(subjects),
               'folds': folds, 'members_verified': len(checks), 'checks': checks,
               'test_predictions_reproduced': True, 'vectors_reproduced': True,
               'train_only_preprocessing': True, 'validation_checkpoint_selection_verified': True,
               'one_oof_per_subject_per_model': True, 'metrics_independently_recalculated': True})
    print(f'PASS: {len(checks)} member checkpoints; {len(subjects)} OOF patients/model; partitions, statistics, logs, vectors and metrics.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    audit(parser.parse_args().output.resolve())
