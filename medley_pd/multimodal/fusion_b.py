"""Architecture B: differentiable EEG prefixes inside one shared causal LLM.

Sequence assembly is independent of EEG pooling and model forward. No trainer,
optimizer, split selection, checkpoint/export or pretrained loading lives here.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .clinical_serializer import ClinicalTextSerializer
from .clinical_tokenization import ClinicalTokens, ClinicalTokenizer
from .config import validate_config
from .eeg_pooling import EEGFeatureAdapter, LearnedQueryAttentionPooling
from .eeg_projection import EEGToLLMProjector
from .llm_backbone import LLMBackbone, parameter_statistics, select_prediction_state


@dataclass(frozen=True)
class FusedInputs:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    prediction_positions: torch.Tensor


def assemble_soft_token_inputs(backbone: LLMBackbone, eeg_tokens: torch.Tensor,
                               text_tokens: ClinicalTokens) -> FusedInputs:
    """[B,K,H] EEG prefix + actual text embeddings -> [B,K+L,H].

    No LLM forward occurs here. EEG is explicitly cast to the actual embedding
    output dtype, preserving gradients; no tensor is moved between devices.
    Text padding is retained, including holes after an EEG prefix with left-pad.
    """
    device = backbone.embedding_layer.weight.device
    if (not isinstance(eeg_tokens, torch.Tensor) or eeg_tokens.ndim != 3
            or min(eeg_tokens.shape) == 0 or eeg_tokens.shape[-1] != backbone.hidden_dim
            or not eeg_tokens.is_floating_point() or eeg_tokens.device != device
            or not bool(torch.isfinite(eeg_tokens).all())):
        raise ValueError("EEG soft tokens require finite floating [B,K,H] on the actual LLM embedding device")
    if not isinstance(text_tokens, ClinicalTokens):
        raise ValueError("Use Task 6 ClinicalTokens for clinical text and suffix")
    ids, mask, positions = text_tokens.input_ids, text_tokens.attention_mask, text_tokens.prediction_positions
    if (not isinstance(ids, torch.Tensor) or ids.ndim != 2 or min(ids.shape) == 0
            or ids.dtype != torch.long or ids.device != device or ids.shape[0] != eeg_tokens.shape[0]
            or not bool(((ids >= 0) & (ids < backbone.embedding_layer.num_embeddings)).all())):
        raise ValueError("Text IDs must be valid int64 [B,L] aligned with EEG on the embedding device")
    if (not isinstance(mask, torch.Tensor) or mask.shape != ids.shape or mask.device != device
            or not bool(((mask == 0) | (mask == 1)).all()) or not bool(mask.bool().any(dim=1).all())):
        raise ValueError("Text attention mask must be binary [B,L] with actual suffix tokens")
    settings = backbone.resolved_config.settings["llm"]
    model_limit = getattr(backbone.model.config, "max_position_embeddings", None) or settings["max_sequence_tokens"]
    if eeg_tokens.shape[1] + ids.shape[1] > min(settings["max_sequence_tokens"], model_limit):
        raise ValueError("EEG prefix plus complete clinical/suffix text exceeds context budget; no truncation")
    text_embeddings = backbone.embedding_layer(ids)
    # Reuse Task 4's full shape/device/bounds/unmasked-position checks.
    if not isinstance(positions, torch.Tensor):
        raise ValueError("Explicit text prediction positions are required")
    select_prediction_state(text_embeddings, mask, positions)
    last = torch.arange(ids.shape[1], device=device).expand_as(ids).masked_fill(~mask.bool(), -1).max(dim=1).values
    if not torch.equal(positions, last):
        raise ValueError("Text prediction position must be the final actual suffix token")
    combined = torch.cat([eeg_tokens.to(dtype=text_embeddings.dtype), text_embeddings], dim=1)
    if not bool(torch.isfinite(combined).all()):
        raise ValueError("Nonfinite fused embeddings after explicit dtype conversion")
    k = eeg_tokens.shape[1]
    prefix_mask = torch.ones((ids.shape[0], k), dtype=torch.long, device=device)
    combined_mask = torch.cat([prefix_mask, mask.long()], dim=1)
    return FusedInputs(combined, combined_mask, positions + k)


class EEGClinicalSoftTokenRegressor(nn.Module):
    """Task 3 EEG tensors and clinical dictionaries -> raw float32 MoCA [B].

    Supply a separately prepared Task 4 backbone for M3 (no fitted M1 head is
    accepted). New EEG modules/head are placed on its device in FP32; the LLM
    itself is never moved or cast. Explicitly injected A1 modules retain dtype
    and must already be on that device and match the resolved configuration.
    """

    def __init__(self, backbone: LLMBackbone, *, pooling: LearnedQueryAttentionPooling | None = None,
                 projector: EEGToLLMProjector | None = None, feature_adapter: EEGFeatureAdapter | None = None):
        super().__init__()
        config = backbone.resolved_config
        validate_config(config.settings)
        if config.settings["architecture"] != "fusion_b" or config.settings["fusion"]["type"] != "soft_token":
            raise ValueError("Architecture B requires fusion_b with soft_token fusion")
        stats = parameter_statistics(backbone.model)
        if not stats["base_frozen"] or (backbone.mode == "frozen" and stats["trainable_parameters"]):
            raise ValueError("Backbone trainability violates the Task 4 adaptation contract")
        if backbone.mode in ("lora", "qlora") and not stats["trainable_lora_parameters"]:
            raise ValueError("Adapter mode requires prepared Task 4 LoRA parameters")
        eeg = config.settings["eeg"]
        if eeg["feature_adapter"]["enabled"]:
            raise NotImplementedError("EEG feature adapter ON is deferred to Phase D")
        if eeg["projector"].get("output_dim") not in (None, backbone.hidden_dim):
            raise ValueError("Configured EEG projection width disagrees with actual LLM hidden_dim")
        device = backbone.embedding_layer.weight.device
        self.backbone = backbone
        self.llm = backbone.model
        self.feature_adapter = (EEGFeatureAdapter.from_config(config) if feature_adapter is None else feature_adapter)
        self.pooling = (LearnedQueryAttentionPooling.from_config(config).to(device=device, dtype=torch.float32)
                        if pooling is None else pooling)
        self.projector = (EEGToLLMProjector.from_config(config, hidden_dim=backbone.hidden_dim).to(device=device, dtype=torch.float32)
                          if projector is None else projector)
        if not isinstance(self.feature_adapter, EEGFeatureAdapter) or self.feature_adapter.input_dim != eeg["feature_dim"]:
            raise ValueError("Use an A1 identity feature adapter matching the cached feature width")
        if (not isinstance(self.pooling, LearnedQueryAttentionPooling)
                or self.pooling.input_dim != eeg["feature_dim"]
                or self.pooling.num_queries != eeg["pooling"]["num_queries"]
                or self.pooling.hidden_dim != eeg["pooling"]["hidden_dim"]
                or self.pooling.attention.num_heads != eeg["pooling"]["num_heads"]):
            raise ValueError("Injected A1 pooling must match the resolved EEG configuration")
        if (not isinstance(self.projector, EEGToLLMProjector)
                or self.projector.input_dim != self.pooling.hidden_dim or self.projector.output_dim != backbone.hidden_dim
                or isinstance(self.projector.norm, nn.LayerNorm) != eeg["projector"]["layer_norm"]):
            raise ValueError("Injected A1 projector must match pooling, normalization and actual LLM width")
        for module in (self.feature_adapter, self.pooling, self.projector):
            if any(p.device != device for p in module.parameters()):
                raise ValueError("EEG modules must be on the actual LLM device; move explicitly")
        self.serializer = ClinicalTextSerializer(config)
        self.tokenizer = ClinicalTokenizer(backbone)
        # A new head for M3; no fitted M1 parameters are copied or reused.
        self.head = nn.Linear(backbone.hidden_dim, 1, dtype=torch.float32, device=device)

    def prepare_inputs(self, *, eeg_features: torch.Tensor, eeg_mask: torch.Tensor,
                       clinical, artifact_flags: torch.Tensor | None = None) -> FusedInputs:
        """Serialize/tokenize exactly once, pool/project EEG, then assemble.

        This method performs no LLM forward and never reads labels/subject IDs.
        EEG/masks must already match the EEG module device/dtype. Only newly
        constructed Task 6 token tensors are explicitly moved to embedding device.
        """
        texts = self.serializer.serialize_batch(clinical)
        text_tokens = self.tokenizer(texts).to(self.backbone.embedding_layer.weight.device)
        pooled = self.pooling(self.feature_adapter(eeg_features), eeg_mask, artifact_flags=artifact_flags)
        return assemble_soft_token_inputs(self.backbone, self.projector(pooled), text_tokens)

    def forward(self, *, eeg_features: torch.Tensor, eeg_mask: torch.Tensor,
                clinical, artifact_flags: torch.Tensor | None = None) -> torch.Tensor:
        fused = self.prepare_inputs(eeg_features=eeg_features, eeg_mask=eeg_mask,
                                    clinical=clinical, artifact_flags=artifact_flags)
        hidden = self.backbone.forward_hidden(inputs_embeds=fused.inputs_embeds, attention_mask=fused.attention_mask)
        selected = select_prediction_state(hidden, fused.attention_mask, fused.prediction_positions)
        predictions = self.head(selected.to(dtype=self.head.weight.dtype)).squeeze(-1).float()
        if not bool(torch.isfinite(predictions).all()):
            raise ValueError("Nonfinite Architecture B predictions")
        return predictions
