"""Clinical-only PD MoCA runner reusing Medley models and fixed outer folds.

Defaults to validation only. Add --run to fit and save private predictions.
No EEG features are loaded or synthesized.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from io import StringIO
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'medley_pd'))

import numpy as np
import pandas as pd
from configs.config_loader import build_cfg, load_json
from medley.evaluate import summarize
from medley.participants import clinical_sets, task_cohort
from medley.tasks import clin_model

SPLIT_SHA = '5ef85c2056831d71cc04b603b06001739e5d69671a3a57a646597aebb2af6602'


def predict(cfg, cohort, splits):
    """Fit the unchanged Medley clinical pipeline on each outer training set."""
    rows = []
    for fold in sorted(splits.fold.unique()):
        test_ids = set(splits.loc[splits.fold == fold, 'subject'])
        test = cohort.subject.isin(test_ids).to_numpy()
        for name, columns in clinical_sets(cfg).items():
            x = cohort[columns].to_numpy(dtype=float)
            model = clin_model(cfg).fit(x[~test], cohort.target.to_numpy()[~test])
            rows.append(cohort.loc[test, ['subject', 'subgroup', 'target']].assign(
                seed=0, fold=int(fold), model=f'clinical:{name}',
                y=cohort.target.to_numpy()[test], yhat=model.predict(x[test])))
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--data-root', type=Path, default=ROOT / 'data')
    ap.add_argument('--output-root', type=Path, default=ROOT / 'outputs' / 'clinical_ridge')
    ap.add_argument('--run', action='store_true')
    args = ap.parse_args()
    cfg = build_cfg(load_json(str(args.config.resolve())))
    cfg['paths']['data_root'] = str(args.data_root.resolve())
    cfg['paths']['result_root'] = str(args.output_root.resolve())
    if cfg['task']['name'] != 'moca' or cfg['task']['transform'] != 'none':
        raise ValueError('Only original-unit MoCA regression is supported')
    if cfg['evaluation']['seeds'] != [0] or cfg['evaluation']['n_folds'] != 5:
        raise ValueError('Expected seed 0 and 5 folds')
    if cfg['models'] != {'eeg': False, 'clinical': True, 'baselines': [], 'fusion': []}:
        raise ValueError('Only clinical models are supported')
    if clinical_sets(cfg) != {'C0': ['age', 'sex'], 'C1': ['age', 'sex', 'updrs_part_iii', 'disease_duration', 'ledd']}:
        raise ValueError('C0/C1 definitions differ')
    split_path = ROOT / 'medley_pd/splits/pd_eeg__moca.csv'
    split_sha = hashlib.sha256(split_path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    if split_sha != SPLIT_SHA:
        raise ValueError('Fixed split changed')
    splits = pd.read_csv(split_path, dtype={'subject': str}).query('seed == 0')
    with redirect_stdout(StringIO()):
        cohort = task_cohort(cfg)
    if (len(cohort) != 113 or cohort.subject.nunique() != 113 or
            len(splits) != 113 or splits.subject.nunique() != 113 or
            set(cohort.subject) != set(splits.subject)):
        raise ValueError('Cohort must exactly match the fixed 113 patients')
    if splits.groupby('fold').size().to_dict() != {1: 23, 2: 23, 3: 23, 4: 22, 5: 22}:
        raise ValueError('Unexpected fold sizes')
    cohort = cohort.assign(target=cohort.moca.astype(float))
    if not np.isfinite(cohort.target).all():
        raise ValueError('Nonfinite target')
    for name, cols in clinical_sets(cfg).items():
        x = cohort[cols].to_numpy(dtype=float)
        if np.isinf(x).any():
            raise ValueError('Infinite clinical value')
        for fold in range(1, 6):
            tr = ~cohort.subject.isin(splits.loc[splits.fold == fold, 'subject'])
            if np.isnan(x[tr]).all(axis=0).any():
                raise ValueError('All-missing training feature')
        print(f'{name}: {len(cols)} variables; missing cells={int(np.isnan(x).sum())}')
    out = args.output_root.resolve() / cfg['experiment']['experiment_name']
    if out.exists():
        raise FileExistsError('Choose a new output directory; overwrite is disabled')
    print('PASS: 113 matching patients; seed 0; folds 23/23/23/22/22; no EEG required.')
    if not args.run:
        print('Validation only. Add --run to train.')
        return
    pred = predict(cfg, cohort, splits)
    if len(pred) != 226 or pred.duplicated(['model', 'subject', 'seed']).any():
        raise ValueError('Invalid OOF coverage')
    out.mkdir(parents=True, exist_ok=False)
    pred.to_csv(out / 'predictions.csv.gz', index=False)
    summary, _, _ = summarize(pred, cfg, str(out))
    import sklearn
    source = args.data_root / 'ds007526' / 'participants.tsv'
    metadata = {'status': 'completed', 'code_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'split_sha256_lf': split_sha, 'participants_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'dataset': 'ds007526', 'version': '1.0.2', 'n': 113, 'seed': 0,
        'source_url': 'https://raw.githubusercontent.com/OpenNeuroDatasets/ds007526/1.0.2/participants.tsv',
        'python': sys.version, 'sklearn': sklearn.__version__, 'config': cfg,
        'note': 'Clinical-only; no EEG cache. Historical numerical equivalence on real data not established.'}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(summary[['model', 'avg_MAE', 'avg_Spearman', 'avg_R2']].to_string(index=False))
    print(f'Results: {out}')


if __name__ == '__main__':
    main()
