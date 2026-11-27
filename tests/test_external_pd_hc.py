import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from codl.eeg.external_pd_hc import (
    EXPECTED, ExternalPDHC, load_checkpoint, person_metrics,
    save_checkpoint, split_people, validate_people,
)
from codl.eeg.external_graphs import build_recording
from codl.eeg.train_external_pd_hc import fit


def tables():
    # Entirely synthetic IDs; no study participants in this test.
    people = pd.DataFrame([
        {'person_key': f'fake-{ds}-{label}-{i}', 'original_id': f'original-{label}-{i}',
         'dataset': ds, 'label_pd': label, 'moca': np.nan, 'country': 'Chile'}
        for ds, counts in EXPECTED.items() for label, n in counts.items() for i in range(n)
    ])
    excluded = pd.DataFrame({'person_key': [f'holdout-{i}' for i in range(14)],
                             'original_id': [f'holdout-original-{i}' for i in range(14)],
                             'country': ['Chile']*14})
    return people, excluded


def test_roster_guards_and_person_split():
    people, excluded = tables()
    validate_people(people, excluded)
    first = split_people(people)
    pd.testing.assert_frame_equal(first, split_people(people.sample(frac=1, random_state=4)))
    assert len(first) == 163 and first.person_key.nunique() == 163
    assert first.groupby(['dataset', 'label_pd']).split.nunique().eq(2).all()
    bad = people.copy()
    bad.loc[bad.dataset.eq('brainlat_cl') & bad.label_pd.eq(1), 'moca'] = 20
    with pytest.raises(ValueError, match='MoCA'): validate_people(bad, excluded)
    bad = people.copy(); bad.loc[0, 'person_key'] = excluded.person_key.iloc[0]
    with pytest.raises(ValueError, match='Held-out'): validate_people(bad, excluded)
    with pytest.raises(ValueError, match='14'): validate_people(people, excluded.iloc[:-1])


def test_brainlat_identity_requires_country_and_source_id():
    people, excluded = tables()
    source = people.loc[people.dataset.eq('brainlat_cl'), 'original_id'].iloc[0]
    excluded.loc[0, ['country', 'original_id']] = ['Argentina', source]
    validate_people(people, excluded)  # Same number in another country is not an ID match.
    excluded.loc[0, 'country'] = 'Chile'
    with pytest.raises(ValueError, match='original BrainLat ID'): validate_people(people, excluded)


def test_person_not_window_metrics():
    rows = pd.DataFrame({'person_key': ['a']*9+['b', 'c', 'd'],
                         'label_pd': [0]*9+[0, 1, 1],
                         'probability': [.1]*9+[.6, .7, .8]})
    person, metrics = person_metrics(rows)
    assert len(person) == 4 and metrics['balanced_accuracy'] == .75
    assert metrics['auroc'] == 1


def test_classifier_train_step_and_separate_checkpoint(tmp_path):
    settings = json.loads(Path('src/codl/eeg/external_pd_hc_seed0.json').read_text())
    torch.set_num_threads(2); torch.manual_seed(0)
    model = ExternalPDHC(settings)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['lr'])
    x = torch.randn(4, 19, 200)*20
    lap = -torch.ones(4, 19, 19)/19
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x, lap), torch.tensor([0., 1., 0., 1.]))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step(); model.eval()
    with torch.no_grad():
        before = model(x, lap); vectors = model.encode(x, lap)
    path = tmp_path/'synthetic_only.pt'
    save_checkpoint(path, model, settings, {'status': 'synthetic_test_only'})
    loaded, payload = load_checkpoint(path)
    assert not any(k.startswith('head.') for k in payload['encoder_state_dict'])
    assert vectors.shape == (4, 256)
    with torch.no_grad():
        torch.testing.assert_close(loaded(x, lap), before, rtol=0, atol=0)
        torch.testing.assert_close(loaded.encode(x, lap), vectors, rtol=0, atol=0)
    with pytest.raises(FileExistsError): save_checkpoint(path, model, settings, {})


def test_cache_settings_and_content_validation(tmp_path):
    import copy
    import mne
    settings = json.loads(Path('src/codl/eeg/external_pd_hc_seed0.json').read_text())
    cfg = {'dataset': {'line_freq': 50},
           'preprocess': {'l_freq': .3, 'h_freq': 45., 'notch': None,
                          'rereference': None, 'flag_uv': 100., 'min_good_channels': 20},
           'encoder': {k: settings[k] for k in ('fs', 'seg_sec', 'amp_div', 'channels')}}
    cfg['encoder']['name'] = 'synthetic'
    raw = mne.io.RawArray(np.random.default_rng(0).normal(size=(19, 2400))*1e-5,
                         mne.create_info(settings['channels'], 200, 'eeg'), verbose=False)
    path = tmp_path/'synthetic_raw.fif'; raw.save(path, overwrite=False, verbose=False)
    first, meta = build_recording(path, [], cfg, settings, tmp_path/'cache', {'synthetic': True})
    stamp = first.stat().st_mtime_ns
    same, _ = build_recording(path, [], cfg, settings, tmp_path/'cache', {'synthetic': True})
    assert same == first and same.stat().st_mtime_ns == stamp
    changed = copy.deepcopy(cfg); changed['preprocess']['h_freq'] = 40.
    second, _ = build_recording(path, [], changed, settings, tmp_path/'cache', {'synthetic': True})
    assert second != first
    changed['encoder']['seg_sec'] = 2
    with pytest.raises(ValueError, match='19,200'):
        build_recording(path, [], changed, settings, tmp_path/'cache', {'synthetic': True})
    first.write_bytes(first.read_bytes()+b'corruption')
    with pytest.raises(ValueError, match='mismatch'):
        build_recording(path, [], cfg, settings, tmp_path/'cache', {'synthetic': True})


def test_external_training_engine_synthetic_only(tmp_path):
    settings = json.loads(Path('src/codl/eeg/external_pd_hc_seed0.json').read_text())
    settings.update(epochs=2, patience=1, windows_per_patient_per_epoch=2)
    people = pd.DataFrame({'person_key':[f'fake{i}' for i in range(8)],
                           'label_pd':[0,1]*4, 'split':['train']*4+['validation']*4})
    generator=torch.Generator().manual_seed(42)
    signals={p:(torch.randn(3,19,200,generator=generator)*20,-torch.ones(3,19,19)/19)
             for p in people.person_key}
    out=tmp_path/'synthetic_training'
    result=fit(signals,people,settings,out,{'scope':'synthetic test; no research participants'})
    assert result['n_train']==4 and result['n_validation']==4
    assert result['reload_and_features']=='exact_match'
    history=pd.read_csv(out/'history.csv')
    assert result['selected_epoch']==int(history.loc[history.validation_person_bce.idxmin(),'epoch'])
    predictions=pd.read_csv(out/'validation_predictions.csv')
    assert set(predictions.person_key)==set(people[people.split.eq('validation')].person_key)
    assert predictions.n_windows.eq(3).all()
    assert (out/'encoder.pt').exists() and (out/'classifier.pt').exists()
