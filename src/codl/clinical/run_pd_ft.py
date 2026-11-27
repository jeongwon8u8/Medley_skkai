"""Run team FT clinical-only C0/C1 with fixed outer folds and private artifacts."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from codl.clinical.ft_clinical import DEFAULTS, fit_member, infer_member
from codl.clinical.run_pd_ridge import SPLIT_SHA
import numpy as np
import pandas as pd
import torch
from medley.evaluate import summarize
from medley.metrics import regression_metrics
from medley.participants import clinical_sets, task_cohort

RIDGE = ROOT / 'outputs/clinical_ridge/clinical_ridge_c0_c1_seed0_20261126'


def sha(path, normalize=False):
    data = Path(path).read_bytes()
    return hashlib.sha256(data.replace(b'\r\n', b'\n') if normalize else data).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def load_reference():
    """Use the completed Ridge run's config, loader, source and split identity."""
    meta = json.loads((RIDGE / 'metadata.json').read_text(encoding='utf-8'))
    cfg = copy.deepcopy(meta['config'])
    source = Path(cfg['paths']['data_root']) / 'ds007526/participants.tsv'
    split_path = ROOT / 'medley_pd/splits/pd_eeg__moca.csv'
    if sha(source) != meta['participants_sha256'] or sha(split_path, True) != SPLIT_SHA:
        raise ValueError('Data or fixed split differs from Ridge')
    d = task_cohort(cfg).assign(target=lambda x: x.moca.astype(float))
    splits = pd.read_csv(split_path, dtype={'subject': str}).query('seed == 0')
    if len(d) != 113 or d.subject.nunique() != 113 or len(splits) != 113 or splits.subject.nunique() != 113:
        raise ValueError('Expected 113 unique patients')
    if set(d.subject) != set(splits.subject):
        raise ValueError('Cohort differs from pinned split')
    ridge = pd.read_csv(RIDGE / 'predictions.csv.gz', dtype={'subject': str})
    for name in clinical_sets(cfg):
        r = ridge.loc[ridge.model == f'clinical:{name}']
        if len(r) != 113 or r.subject.nunique() != 113 or set(r.seed) != {0}:
            raise ValueError('Invalid Ridge OOF reference')
        merged = r.merge(splits, on=['subject', 'seed'], suffixes=('', '_fixed'), validate='one_to_one')
        if len(merged) != 113 or not (merged.fold == merged.fold_fixed).all():
            raise ValueError('Ridge fold mismatch')
        np.testing.assert_array_equal(r.set_index('subject').loc[d.subject, 'y'], d.target)
    return cfg, d, splits, ridge, meta


def inner_indices(n, seed, fraction):
    perm = np.random.RandomState(seed).permutation(n)
    nv = max(2, int(round(n * fraction)))
    return perm[nv:], perm[:nv]


def validate_oof(pred, splits, subjects, models):
    if set(pred.model) != set(models) or set(pred.seed) != {0}:
        raise ValueError('Model/seed mismatch')
    if pred.duplicated(['subject', 'model', 'seed']).any():
        raise ValueError('Duplicate OOF predictions')
    if not np.isfinite(pred[['target', 'y', 'yhat']].to_numpy()).all():
        raise ValueError('Nonfinite OOF value')
    for model in models:
        group = pred[pred.model == model]
        if len(group) != len(subjects) or set(group.subject) != set(subjects):
            raise ValueError('Missing or extra OOF patient')
    joined = pred.merge(splits, on=['subject', 'seed'], suffixes=('', '_fixed'), validate='many_to_one')
    if len(joined) != len(pred) or not (joined.fold == joined.fold_fixed).all():
        raise ValueError('OOF fold mismatch')


def comparison_table(ridge, pred):
    rows = []
    for name in ('C0', 'C1'):
        r = ridge[ridge.model == 'clinical:' + name]
        f = pred[pred.model == 'clinical_ft:' + name]
        joined = r.merge(f, on=['subject', 'seed', 'fold'], suffixes=('_ridge', '_ft'), validate='one_to_one')
        if len(joined) != 113:
            raise ValueError('Paired comparison requires 113 matched patients')
        np.testing.assert_array_equal(joined.y_ridge, joined.y_ft)
        rm = regression_metrics(r.y, r.yhat)
        fm = regression_metrics(f.y, f.yhat)
        rows.extend([{'model': 'Ridge ' + name, **rm, 'delta_MAE_vs_Ridge': 0.0},
                     {'model': 'FT-only ' + name, **fm, 'delta_MAE_vs_Ridge': fm['MAE'] - rm['MAE']}])
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=Path(__file__).with_name('pd_ft_seed0.json'))
    ap.add_argument('--output-root', type=Path, default=ROOT / 'outputs/clinical_ft')
    ap.add_argument('--folds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    args = ap.parse_args()
    settings = json.loads(args.config.read_text(encoding='utf-8'))
    folds = sorted(args.folds)
    if len(folds) != len(set(folds)) or not set(folds) <= set(range(1, 6)):
        raise ValueError('Invalid folds')
    if settings['seed'] != 0 or settings['device'] != 'cpu':
        raise ValueError('This recorded protocol is seed 0 on CPU')
    for key in DEFAULTS:
        if key != 'nn_pca' and settings[key] != DEFAULTS[key]:
            raise ValueError('Team settings changed; define a separate protocol')
    cfg, d, splits, ridge, ridge_meta = load_reference()
    out = args.output_root.resolve() / settings['experiment_name']
    out.mkdir(parents=True, exist_ok=False)
    cfg['paths']['result_root'] = str(out)
    cfg['experiment']['experiment_name'] = settings['experiment_name']
    torch.set_num_threads(settings['threads'])
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    code_paths = ['src/codl/clinical/ft_clinical.py', 'src/codl/clinical/run_pd_ft.py',
                  'medley_pd/medley/fusion_nn.py', 'medley_pd/medley/participants.py',
                  'medley_pd/medley/metrics.py', 'medley_pd/medley/evaluate.py']
    manifest = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
                'settings': settings, 'config': cfg, 'folds': folds,
                'split_sha256_lf': SPLIT_SHA, 'participants_sha256': ridge_meta['participants_sha256'],
                'ridge_predictions_sha256': sha(RIDGE / 'predictions.csv.gz'),
                'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'code_hashes': {p: sha(ROOT / p) for p in code_paths},
                'python': sys.version, 'torch': str(torch.__version__),
                'selection': 'Per-member validation Huber on inner-train standardized MoCA; min_delta=1e-5',
                'refit': False, 'members': 5, 'outer_test_used_during_training': False}
    write_json(out / 'metadata.json', manifest)
    write_json(out / 'config_used.json', {'training': settings, 'medley': cfg})
    predictions, all_members = [], []
    for fold in folds:
        test_ids = set(splits.loc[splits.fold == fold, 'subject'])
        outer = d.loc[~d.subject.isin(test_ids)].reset_index(drop=True)
        test = d.loc[d.subject.isin(test_ids)].reset_index(drop=True)
        for name, cols in clinical_sets(cfg).items():
            folder = out / name / f'fold{fold}'
            folder.mkdir(parents=True)
            member_pred, member_vectors = [], []
            for member in range(settings['nn_members']):
                seed = settings['seed'] * 1000 + member
                tr, va = inner_indices(len(outer), seed, settings['nn_val_frac'])
                payload, log = fit_member(
                    outer.loc[tr, cols].to_numpy(float), outer.loc[tr, 'target'].to_numpy(),
                    outer.loc[va, cols].to_numpy(float), outer.loc[va, 'target'].to_numpy(),
                    cols, settings, seed)
                payload['partition'] = {'train': outer.loc[tr, 'subject'].tolist(),
                                        'validation': outer.loc[va, 'subject'].tolist(),
                                        'test': test.subject.tolist()}
                checkpoint = folder / f'member{member}.pt'
                torch.save(payload, checkpoint)
                pd.DataFrame(log).to_csv(folder / f'member{member}_history.csv', index=False)
                # Test features are first passed to a model AFTER checkpoint selection.
                yp, vector = infer_member(checkpoint, test[cols].to_numpy(float))
                member_pred.append(yp)
                member_vectors.append(vector)
                record = {'model': 'clinical_ft:' + name, 'fold': fold, 'member': member,
                          'init_seed': seed, 'n_train': len(tr), 'n_validation': len(va), 'n_test': len(test),
                          **{k: payload[k] for k in ('selected_epoch', 'epochs_ran', 'best_val_huber',
                                                     'seconds', 'initial_state_sha256', 'selected_state_sha256')},
                          'checkpoint_sha256': sha(checkpoint)}
                all_members.append(record)
            # Preserve the exact float32 ensemble values through CSV roundtrips.
            yp = np.stack(member_pred, axis=1).mean(axis=1).astype(np.float64)
            frame = test[['subject', 'subgroup', 'target']].assign(
                seed=0, fold=fold, model='clinical_ft:' + name, y=test.target, yhat=yp)
            frame.to_csv(folder / 'predictions.csv.gz', index=False)
            # Member latent axes are not assumed aligned: preserve [patient, member, hidden].
            np.savez_compressed(folder / 'clinical_vectors.npz', subjects=test.subject.to_numpy(dtype=str),
                                vectors=np.stack(member_vectors, axis=1),
                                member_predictions=np.stack(member_pred, axis=1))
            predictions.append(frame)
            pd.DataFrame(all_members).to_csv(out / 'selected_epochs.csv', index=False)
            print(f'{name} fold {fold}: {len(test)} OOF predictions; 5 members saved', flush=True)
    pred = pd.concat(predictions, ignore_index=True)
    subjects = splits.loc[splits.fold.isin(folds), 'subject'].tolist()
    validate_oof(pred, splits, subjects, ['clinical_ft:C0', 'clinical_ft:C1'])
    pred.to_csv(out / 'predictions.csv.gz', index=False)
    summarize(pred, cfg, str(out))
    if folds == [1, 2, 3, 4, 5]:
        comparison = comparison_table(ridge, pred)
        comparison.to_csv(out / 'ridge_ft_comparison.csv', index=False)
        print(comparison[['model', 'n', 'MAE', 'RMSE', 'R2', 'Spearman', 'delta_MAE_vs_Ridge']].to_string(index=False))
    manifest.update(status='completed', n_subjects_per_model=len(subjects),
                    elapsed_seconds=time.perf_counter() - started,
                    finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(out / 'metadata.json', manifest)
    print(f'Results: {out}', flush=True)


if __name__ == '__main__':
    main()
