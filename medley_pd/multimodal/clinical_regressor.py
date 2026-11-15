"""Clinical-only current-MoCA regression using the existing causal backbone."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .llm_backbone import LLMBackbone, select_prediction_state, parameter_statistics


class ClinicalLLMRegressor(nn.Module):
    """[B,L] IDs/mask + explicit [B] suffix positions -> raw float32 [B]."""

    def __init__(self, backbone: LLMBackbone):
        super().__init__()
        if backbone.resolved_config.settings["architecture"] != "clinical_only":
            raise ValueError("M1 requires architecture=clinical_only")
        stats = parameter_statistics(backbone.model)
        if not stats["base_frozen"] or (backbone.mode == "frozen" and stats["trainable_parameters"]):
            raise ValueError("Backbone trainability violates the Task 4 adaptation contract")
        if backbone.mode in ("lora", "qlora") and not stats["trainable_lora_parameters"]:
            raise ValueError("Adapter mode requires trainable Task 4 LoRA parameters")
        self.backbone = backbone
        self.llm = backbone.model  # Register the supplied model, without reloading.
        self.head = nn.Linear(backbone.hidden_dim, 1, device=backbone.embedding_layer.weight.device,
                              dtype=torch.float32)

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                prediction_positions: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone.forward_hidden(input_ids=input_ids, attention_mask=attention_mask)
        selected = select_prediction_state(hidden, attention_mask, prediction_positions)
        # M1's tokenization policy has no trailing EOS: prediction is the last
        # real token. Explicit selection must never choose a clinical/pad token.
        last = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(input_ids)
        last = last.masked_fill(~attention_mask.bool(), -1).max(dim=1).values
        if not torch.equal(prediction_positions, last):
            raise ValueError("M1 prediction position must be the final actual suffix token")
        predictions = self.head(selected.to(self.head.weight.dtype)).squeeze(-1).float()
        if not bool(torch.isfinite(predictions).all()):
            raise ValueError("Nonfinite clinical predictions")
        return predictions


def moca_huber_loss(predictions: torch.Tensor, targets: torch.Tensor, *, delta: float = 1.0,
                    reduction: str = "mean") -> torch.Tensor:
    if (not isinstance(predictions, torch.Tensor) or not isinstance(targets, torch.Tensor)
            or predictions.ndim != 1 or targets.shape != predictions.shape or predictions.numel() == 0
            or not predictions.is_floating_point() or not targets.is_floating_point()
            or targets.device != predictions.device or not bool(torch.isfinite(predictions).all())
            or not bool(torch.isfinite(targets).all()) or not bool(((targets >= 0) & (targets <= 30)).all())):
        raise ValueError("MoCA predictions/targets require aligned finite floating [B] tensors; targets in [0,30]")
    if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta) or delta <= 0:
        raise ValueError("Huber delta must be positive and finite")
    if reduction not in ("mean", "sum", "none"):
        raise ValueError("Invalid Huber reduction")
    return F.huber_loss(predictions.float(), targets.float(), delta=delta, reduction=reduction)
