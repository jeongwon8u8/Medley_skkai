"""Small classification evaluation helpers for subject-level arrays."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score


def binary_classification_metrics(labels, probabilities, threshold: float = 0.5) -> dict[str, float]:
    """Compute AUROC, AUPRC and F1 from patient-level probabilities.

    The threshold must be chosen without peeking at a held-out test set.
    """
    y = np.asarray(labels)
    scores = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or scores.ndim != 1 or len(y) != len(scores) or len(y) == 0:
        raise ValueError("nonempty aligned 1D arrays required")
    if not set(np.unique(y)).issubset({0, 1}) or len(np.unique(y)) != 2:
        raise ValueError("binary labels with both classes are required")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("probabilities must be finite and within [0,1]")
    if not (0 < threshold < 1):
        raise ValueError("threshold must be strictly between 0 and 1")
    return {
        "auroc": float(roc_auc_score(y, scores)),
        "auprc": float(average_precision_score(y, scores)),
        "f1": float(f1_score(y, scores >= threshold, zero_division=0)),
    }
