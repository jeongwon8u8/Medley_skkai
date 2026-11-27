"""Synthetic checks of the cache-free clinical runner against legacy fitting."""
import copy

import numpy as np
import pandas as pd

from codl.clinical.run_pd_ridge import predict
from medley.config import load_config
from medley.evaluate import _fit_predict_all


def fixture():
    rng = np.random.default_rng(12)
    cfg = load_config('pd_eeg', 'cbramod', 'moca')
    cfg['models'] = {'eeg': False, 'clinical': True, 'baselines': [], 'fusion': []}
    cfg['task']['clinical_sets'] = {'C0': ['age', 'sex'], 'C1': ['age', 'sex', 'updrs_part_iii', 'disease_duration', 'ledd']}
    d = pd.DataFrame({'subject': [f'synthetic-{i}' for i in range(30)],
        'subgroup': 'synthetic', 'target': rng.normal(24, 3, 30),
        'age': rng.normal(65, 6, 30), 'sex': rng.integers(0, 2, 30),
        'updrs_part_iii': rng.normal(20, 4, 30),
        'disease_duration': rng.uniform(1, 10, 30), 'ledd': rng.uniform(100, 500, 30)})
    d.loc[[0, 7, 14], 'ledd'] = np.nan
    splits = pd.DataFrame({'subject': d.subject, 'seed': 0, 'fold': np.arange(30) % 5 + 1})
    return cfg, d, splits


def test_matches_legacy_clinical_predictions():
    cfg, d, splits = fixture()
    result = predict(cfg, d, splits)
    assert len(result) == 60
    assert not result.duplicated(['model', 'subject']).any()
    for fold in range(1, 6):
        te = splits.fold.to_numpy() == fold
        matrices = {name: d[cols].to_numpy() for name, cols in cfg['task']['clinical_sets'].items()}
        expected = _fit_predict_all(cfg, np.empty((sum(~te), 0)), np.empty((sum(te), 0)),
            {k: x[~te] for k, x in matrices.items()}, {k: x[te] for k, x in matrices.items()},
            np.zeros(sum(~te)), np.zeros(sum(te)), d.target.to_numpy()[~te])
        for model, values in expected.items():
            got = result[(result.fold == fold) & (result.model == model)]
            np.testing.assert_allclose(got.yhat, values, rtol=0, atol=1e-12)


def test_heldout_targets_do_not_affect_their_predictions():
    cfg, d, splits = fixture()
    baseline = predict(cfg, d, splits)
    changed = copy.deepcopy(d)
    changed.loc[splits.fold == 1, 'target'] += 1000
    alternate = predict(cfg, changed, splits)
    np.testing.assert_allclose(baseline.loc[baseline.fold == 1, 'yhat'],
                               alternate.loc[alternate.fold == 1, 'yhat'], rtol=0, atol=1e-12)
