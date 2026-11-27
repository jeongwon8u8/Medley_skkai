"""External PD/HC model utilities; no PD-EEG cohort or MoCA dependency.

Paper-based modified DirectedGNN, randomly initialized. This module does not
assert that a private roster is complete or that cross-cohort deduplication
has been established. Those checks must precede actual training.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn

from codl.eeg.directed_gnn import DirectedGNN


EXPECTED = {
    'turku': {0: 20, 1: 20},
    'ds003490': {0: 25, 1: 25},
    'ds002778': {0: 16, 1: 15},
    'brainlat_cl': {0: 27, 1: 15},
}


class ExternalPDHC(nn.Module):
    """EEG-only logit and reusable 256-dimensional EEG encoder."""

    def __init__(self, settings):
        super().__init__()
        self.encoder = DirectedGNN(settings)
        # Remove the MoCA head completely, including its parameters.
        self.encoder.head = nn.Identity()
        self.classifier = nn.Linear(settings['head_hidden'], 1)

    def encode(self, x, lap):
        return self.encoder.encode(x, lap)

    def forward(self, x, lap):
        return self.classifier(self.encoder.drop(self.encode(x, lap))).squeeze(-1)


def validate_people(people, exclusions):
    """Validate a reviewed person table, not a dataset-prefix dedup heuristic.

    person_key must come from the team's original-ID/recruitment-source review.
    The caller must retain that review and the original pool file separately.
    exclusions is the FULL 14-person BrainLat MoCA holdout list, not a list
    constructed by filtering only the pretraining pool.
    """
    required = {'person_key', 'dataset', 'original_id', 'label_pd', 'moca', 'country'}
    if not required <= set(people):
        raise ValueError(f'Missing person columns: {sorted(required - set(people))}')
    if people[['person_key', 'dataset', 'original_id']].isna().any().any():
        raise ValueError('Missing identity')
    if people.person_key.astype(str).str.strip().eq('').any():
        raise ValueError('Empty identity')
    if people.person_key.duplicated().any():
        raise ValueError('Repeated person; group all sessions before splitting')
    if people.duplicated(['dataset', 'original_id']).any():
        raise ValueError('Repeated source identity')
    if not people.label_pd.isin([0, 1]).all():
        raise ValueError('label_pd must be HC=0, PD=1')
    if set(people.dataset) != set(EXPECTED):
        raise ValueError('Unexpected or missing cohort; PD-EEG/Iowa are forbidden')
    if not {'person_key', 'original_id', 'country'} <= set(exclusions):
        raise ValueError('Full BrainLat exclusion roster required')
    if len(exclusions) != 14 or exclusions.isna().any().any():
        raise ValueError('Require 14 identified BrainLat MoCA holdouts')
    if exclusions.person_key.nunique() != 14 or exclusions.duplicated(['country', 'original_id']).any():
        raise ValueError('Duplicate holdout identity')
    bl = people[people.dataset.eq('brainlat_cl')]
    if set(people.person_key) & set(exclusions.person_key):
        raise ValueError('Held-out person included')
    if set(zip(bl.country, bl.original_id)) & set(zip(exclusions.country, exclusions.original_id)):
        raise ValueError('Held-out original BrainLat ID included')
    if not bl.country.eq('Chile').all():
        raise ValueError('BrainLat must be Chile only')
    if bl.loc[bl.label_pd.eq(1), 'moca'].notna().any():
        raise ValueError('BrainLat PD with MoCA must be excluded')
    actual = {ds: {int(k): int(v) for k, v in g.label_pd.value_counts().items()}
              for ds, g in people.groupby('dataset')}
    if actual != EXPECTED:
        raise ValueError(f'Pool count mismatch: {actual}')
    return actual


def split_people(people, seed=0, validation_fraction=.2):
    """Person-level cohort x label stratification; metadata never model input."""
    if people.person_key.duplicated().any():
        raise ValueError('Split input must have one row per person')
    people = people.sort_values('person_key').reset_index(drop=True).copy()
    strata = people.dataset.astype(str) + ':' + people.label_pd.astype(str)
    train, validation = train_test_split(
        np.arange(len(people)), random_state=seed,
        test_size=validation_fraction, stratify=strata,
    )
    people['split'] = 'train'
    people.loc[validation, 'split'] = 'validation'
    assert not set(people.iloc[train].person_key) & set(people.iloc[validation].person_key)
    return people


def person_metrics(window_predictions):
    """Average sigmoid probabilities over every retained window per person."""
    d = window_predictions
    if not len(d) or not np.isfinite(d.probability).all():
        raise ValueError('Empty/nonfinite predictions')
    if not d.probability.between(0, 1).all():
        raise ValueError('Invalid probability')
    if not d.label_pd.isin([0, 1]).all():
        raise ValueError('Invalid label')
    if d.groupby('person_key').label_pd.nunique().max() != 1:
        raise ValueError('Inconsistent person labels')
    p = d.groupby('person_key', sort=True).agg(
        label_pd=('label_pd', 'first'), probability=('probability', 'mean'),
        n_windows=('probability', 'size'),
    ).reset_index()
    if p.label_pd.nunique() != 2:
        raise ValueError('AUROC requires both classes')
    return p, {
        'n_people': len(p),
        'auroc': float(roc_auc_score(p.label_pd, p.probability)),
        'balanced_accuracy': float(balanced_accuracy_score(p.label_pd, p.probability >= .5)),
        'threshold': .5,
    }


def save_checkpoint(path, model, settings, provenance):
    """Single container with separate encoder and classifier state dictionaries."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    torch.save({
        'format': 'external_pd_hc_v1', 'settings': settings,
        'encoder_state_dict': model.encoder.state_dict(),
        'classifier_state_dict': model.classifier.state_dict(),
        'provenance': provenance,
    }, path)


def load_checkpoint(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['format'] != 'external_pd_hc_v1':
        raise ValueError('Not an external PD/HC checkpoint')
    model = ExternalPDHC(payload['settings'])
    model.encoder.load_state_dict(payload['encoder_state_dict'], strict=True)
    model.classifier.load_state_dict(payload['classifier_state_dict'], strict=True)
    return model.eval(), payload
