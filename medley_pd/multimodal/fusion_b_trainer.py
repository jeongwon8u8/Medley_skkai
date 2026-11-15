"""M3 explicit-epoch training using Task 6's subject-level execution engine.

No split generation, early stopping, baseline fitting or experiment runner.
"""
from __future__ import annotations

import torch

from .clinical_trainer import ClinicalTrainer, ClinicalEvaluation, seed_training
from .config import FUSION_B_TRAINING_DEFAULTS, validate_config
from .fusion_b import EEGClinicalSoftTokenRegressor
from .training_utils import build_role_optimizer, prepare_eeg_inputs

FusionBEvaluation = ClinicalEvaluation


def build_optimizer(model: EEGClinicalSoftTokenRegressor) -> torch.optim.AdamW:
    """Deterministic role groups; reject any duplicated, omitted or base weight."""
    if not isinstance(model, EEGClinicalSoftTokenRegressor):
        raise ValueError("M3 optimizer requires Architecture B")
    config = model.backbone.resolved_config
    validate_config(config.settings)
    rates = {**FUSION_B_TRAINING_DEFAULTS, **config.settings.get("fusion_b_training", {})}
    return build_role_optimizer(model, [
        ("eeg_pooling", model.pooling, rates["lr_pooling"]),
        ("eeg_projector", model.projector, rates["lr_projector"]),
        ("regression_head", model.head, rates["lr_head"]),
        ("feature_adapter", model.feature_adapter, rates["lr_feature_adapter"]),
    ])


class FusionBTrainer(ClinicalTrainer):
    """Reuse partition guards, subject-weighted accumulation and evaluation.

    Constructor: model, split=Task2 SubjectSplit, train_subjects=explicit IDs,
    validation_subjects=explicit withheld outer-train IDs (default empty).
    train_epoch(loader) and evaluate(loader, role='validation'|'test'|'train')
    operate on actual Task 3 subject batches. No automatic validation selection.
    Config comes from the supplied prepared backbone, preventing divergence.
    """

    def _build_optimizer(self):
        return build_optimizer(self.model)

    def _model_identifier(self):
        return f"fusion_b_llm:{self.serializer.feature_set}"

    def _batch(self, batch, expected, seen):
        ids, targets = self._batch_targets(batch, expected, seen)
        inputs = prepare_eeg_inputs(batch, n_subjects=len(ids), device=self.device,
                                    dtype=self.model.pooling.input_projection.weight.dtype)
        return ids, inputs, targets

    def _forward(self, inputs):
        return self.model(**inputs)
