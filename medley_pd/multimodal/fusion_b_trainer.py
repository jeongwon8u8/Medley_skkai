"""M3 explicit-epoch training using Task 6's subject-level execution engine.

No split generation, early stopping, baseline fitting or experiment runner.
"""
from __future__ import annotations

import torch

from .clinical_trainer import ClinicalTrainer, ClinicalEvaluation, seed_training
from .config import FUSION_B_TRAINING_DEFAULTS, validate_config
from .fusion_b import EEGClinicalSoftTokenRegressor
from .llm_backbone import parameter_statistics

FusionBEvaluation = ClinicalEvaluation


def build_optimizer(model: EEGClinicalSoftTokenRegressor) -> torch.optim.AdamW:
    """Deterministic role groups; reject any duplicated, omitted or base weight."""
    if not isinstance(model, EEGClinicalSoftTokenRegressor):
        raise ValueError("M3 optimizer requires Architecture B")
    config = model.backbone.resolved_config
    validate_config(config.settings)
    settings = config.settings["training"]
    rates = {**FUSION_B_TRAINING_DEFAULTS, **config.settings.get("fusion_b_training", {})}
    stats = parameter_statistics(model.llm)
    if not stats["base_frozen"]:
        raise ValueError("Pretrained base parameters must remain frozen")
    groups = []
    for name, module, key in (("eeg_pooling", model.pooling, "lr_pooling"),
                              ("eeg_projector", model.projector, "lr_projector"),
                              ("regression_head", model.head, "lr_head"),
                              ("feature_adapter", model.feature_adapter, "lr_feature_adapter")):
        parameters = list(module.parameters())
        if any(not p.requires_grad for p in parameters):
            raise ValueError("All new-module parameters must be trainable")
        if parameters:
            groups.append({"name": name, "params": parameters,
                           "lr": settings["lr_new_modules"] if rates[key] is None else rates[key]})
    adapters = []
    for name, parameter in model.llm.named_parameters():
        if parameter.requires_grad:
            if "lora_" not in name or model.backbone.mode == "frozen":
                raise ValueError("Unexpected trainable pretrained parameter")
            adapters.append(parameter)
    if model.backbone.mode in ("lora", "qlora") and not adapters:
        raise ValueError("Adaptation mode requires trainable Task 4 adapters")
    if adapters:
        groups.append({"name": "lora", "params": adapters, "lr": settings["lr_lora"]})
    actual = [id(p) for group in groups for p in group["params"]]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Every trainable parameter must occur in exactly one optimizer group")
    return torch.optim.AdamW(groups, weight_decay=settings["weight_decay"])


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
        eeg, mask, flags = (batch[key] for key in ("eeg_features", "eeg_mask", "artifact_flags"))
        if (not isinstance(eeg, torch.Tensor) or eeg.ndim != 3 or eeg.shape[0] != len(ids)
                or not eeg.is_floating_point() or not isinstance(mask, torch.Tensor)
                or mask.shape != eeg.shape[:2] or mask.dtype != torch.bool
                or not isinstance(flags, torch.Tensor) or flags.shape != mask.shape or flags.dtype != torch.bool):
            raise ValueError("Expected aligned Task 3 EEG features, boolean existence mask and artifact flags")
        dtype = self.model.pooling.input_projection.weight.dtype
        inputs = dict(eeg_features=eeg.to(device=self.device, dtype=dtype),
                      eeg_mask=mask.to(self.device), artifact_flags=flags.to(self.device),
                      clinical=batch["clinical"])
        return ids, inputs, targets

    def _forward(self, inputs):
        return self.model(**inputs)
