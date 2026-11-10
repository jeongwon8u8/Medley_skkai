import math
import numpy as np
import pytest
from codl.common.metrics import binary_classification_metrics
from codl.common.splits import grouped_stratified_folds
from codl.robustness.metrics import swap_metrics


def test_binary_metric_happy_path():
    result = binary_classification_metrics([0, 1, 0, 1], [.1, .9, .2, .8])
    assert result["auroc"] == 1.0
    assert result["auprc"] == 1.0
    assert result["f1"] == 1.0


def test_single_class_raises():
    with pytest.raises(ValueError):
        binary_classification_metrics([1, 1], [.4, .7])


def test_grouped_folds_no_subject_overlap():
    subjects = np.repeat(np.arange(12), 2)
    labels = np.repeat(np.array([0, 1] * 6), 2)
    for train, test in grouped_stratified_folds(labels, subjects, n_splits=3):
        assert not set(subjects[train]) & set(subjects[test])


def test_subject_label_mismatch_raises():
    with pytest.raises(ValueError, match="inconsistent"):
        list(grouped_stratified_folds([0, 1, 0, 1], ["a", "a", "b", "b"], n_splits=2))


def test_swap_metrics_definition():
    result = swap_metrics([0, 1, 1, 0], [.1, .9, .8, .2], [.9, .8, .1, .2])
    assert result["flip_rate"] == .5
    assert result["negative_flip_rate"] == .5
    assert result["baseline_correct_count"] == 4


def test_no_baseline_correct_cases():
    result = swap_metrics([1, 1], [.1, .2], [.2, .3])
    assert math.isnan(result["negative_flip_rate"])
