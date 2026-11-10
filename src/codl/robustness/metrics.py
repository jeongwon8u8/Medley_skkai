"""Robustness metric definitions for fixed EEG and shifted clinical inputs."""
from __future__ import annotations

import numpy as np


def swap_metrics(labels, baseline_prob, perturbed_prob, threshold: float = 0.5) -> dict[str, float]:
    """Returns flip rates and probability shifts on the same subjects.

    Negative flip rate denominator = count of baseline-correct cases.
    This is not a causal measure of the clinical modality's contribution.
    """
    y = np.asarray(labels)
    p0 = np.asarray(baseline_prob, dtype=float)
    p1 = np.asarray(perturbed_prob, dtype=float)
    if any(a.ndim != 1 for a in (y, p0, p1)) or not (len(y) == len(p0) == len(p1)) or not len(y):
        raise ValueError("nonempty aligned 1D arrays required")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("labels must be binary")
    if not (np.isfinite(p0).all() and np.isfinite(p1).all()) or np.any((p0 < 0) | (p0 > 1)) or np.any((p1 < 0) | (p1 > 1)):
        raise ValueError("probabilities must be finite within [0,1]")
    if not (0 < threshold < 1):
        raise ValueError("threshold must be strictly between 0 and 1")
    b = (p0 >= threshold).astype(int)
    a = (p1 >= threshold).astype(int)
    correct = b == y
    return {
        "mean_abs_probability_shift": float(np.mean(np.abs(p1 - p0))),
        "flip_rate": float(np.mean(b != a)),
        "negative_flip_rate": float(np.mean(a[correct] != y[correct])) if correct.any() else float('nan'),
        "baseline_correct_count": int(correct.sum()),
        "n_subjects": int(len(y)),
    }
