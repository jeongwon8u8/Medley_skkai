"""Patient-grouped splitting helpers; never split EEG windows independently."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def grouped_stratified_folds(labels, subject_ids, n_splits: int = 5, seed: int = 42):
    """Yield non-overlapping subject-level train/test indices.

    Requires one consistent label per subject and enough subjects per class.
    Callers must verify cohort, temporal visit policy, and class balance.
    """
    y = np.asarray(labels)
    groups = np.asarray(subject_ids)
    if y.ndim != 1 or groups.ndim != 1 or len(y) != len(groups):
        raise ValueError("labels and subject_ids must be aligned one-dimensional arrays")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if len(y) == 0:
        raise ValueError("empty cohort")
    for subject in np.unique(groups):
        if np.unique(y[groups == subject]).size != 1:
            raise ValueError("inconsistent labels within subject; define time-aware task")
    unique_subject_labels = np.array([y[groups == g][0] for g in np.unique(groups)])
    _, per_class = np.unique(unique_subject_labels, return_counts=True)
    if per_class.size < 2 or per_class.min() < n_splits:
        raise ValueError("not enough subjects from each class for stratified folds")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_index, test_index in splitter.split(np.zeros(len(y)), y, groups=groups):
        if set(groups[train_index]) & set(groups[test_index]):
            raise AssertionError("subject leakage detected")
        yield train_index, test_index
