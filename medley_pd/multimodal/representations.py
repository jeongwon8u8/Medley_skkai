"""Independent EEG and text-only clinical representations for external fusion."""
from __future__ import annotations

import torch
from torch import nn

from .clinical_regressor import clinical_prediction_state
from .clinical_serializer import ClinicalTextSerializer
from .clinical_tokenization import ClinicalTokenizer
from .config import FUSION_A_DEFAULTS, validate_config
from .eeg_pooling import EEGFeatureAdapter, LearnedQueryAttentionPooling
from .llm_backbone import parameter_statistics


class MeanQueryAggregation(nn.Module):
    """All K learned queries are valid; original window masks do not apply here."""

    def __init__(self, kind="mean"):
        super().__init__()
        if kind != "mean":
            raise ValueError("Only mean learned-query aggregation is implemented")

    def forward(self, queries):
        if (not isinstance(queries, torch.Tensor) or queries.ndim != 3 or min(queries.shape) == 0
                or not queries.is_floating_point() or not bool(torch.isfinite(queries).all())):
            raise ValueError("Query aggregation requires finite floating [B,K,D]")
        return queries.mean(dim=1)


class RepresentationProjection(nn.Module):
    """Separate Linear projection; cast dtype differentiably, never move devices."""

    def __init__(self, input_dim, output_dim, *, device=None):
        super().__init__()
        if any(type(d) is not int or d <= 0 for d in (input_dim, output_dim)):
            raise ValueError("Projection dimensions must be positive integers")
        self.linear = nn.Linear(input_dim, output_dim, device=device, dtype=torch.float32)

    def forward(self, representation):
        weight = self.linear.weight
        if (not isinstance(representation, torch.Tensor) or representation.ndim != 2
                or representation.shape[0] == 0 or representation.shape[1] != weight.shape[1]
                or not representation.is_floating_point() or representation.device != weight.device
                or not bool(torch.isfinite(representation).all())):
            raise ValueError("Projection requires aligned finite floating [B,D] on its device")
        output = self.linear(representation.to(dtype=weight.dtype))
        if not bool(torch.isfinite(output).all()):
            raise ValueError("Nonfinite projected representation")
        return output


class EEGRepresentationEncoder(nn.Module):
    """Cached windows -> A1 pooling -> mean of queries -> independent projection."""

    def __init__(self, config, *, device=None):
        super().__init__()
        validate_config(config.settings)
        options = {**FUSION_A_DEFAULTS, **config.settings.get("fusion_a", {})}
        self.feature_adapter = EEGFeatureAdapter.from_config(config)
        self.pooling = LearnedQueryAttentionPooling.from_config(config).to(device=device, dtype=torch.float32)
        self.aggregation = MeanQueryAggregation(options["query_aggregation"])
        self.projection = RepresentationProjection(config.settings["eeg"]["pooling"]["hidden_dim"],
                                                   options["eeg_projection_dim"], device=device)

    def forward(self, *, eeg_features, eeg_mask, artifact_flags=None):
        pooled = self.pooling(self.feature_adapter(eeg_features), eeg_mask, artifact_flags=artifact_flags)
        return self.projection(self.aggregation(pooled))


class ClinicalRepresentationEncoder(nn.Module):
    """Approved clinical dictionaries -> exact Task 6 text-only state -> projection.

    No EEG argument or inputs_embeds sequence construction exists in this branch.
    The supplied Task 4 backbone is used directly; caller supplies separate model
    instances for independent M1/M2/M3 training.
    """

    def __init__(self, backbone):
        super().__init__()
        validate_config(backbone.resolved_config.settings)
        stats = parameter_statistics(backbone.model)
        if not stats["base_frozen"] or (backbone.mode == "frozen" and stats["trainable_parameters"]):
            raise ValueError("Backbone trainability violates the Task 4 contract")
        if backbone.mode in ("lora", "qlora") and not stats["trainable_lora_parameters"]:
            raise ValueError("Adapter mode requires prepared Task 4 LoRA parameters")
        self.backbone = backbone
        self.llm = backbone.model
        self.serializer = ClinicalTextSerializer(backbone.resolved_config)
        self.tokenizer = ClinicalTokenizer(backbone)
        options = {**FUSION_A_DEFAULTS, **backbone.resolved_config.settings.get("fusion_a", {})}
        self.projection = RepresentationProjection(backbone.hidden_dim, options["clinical_projection_dim"],
                                                   device=backbone.embedding_layer.weight.device)

    def encode_hidden(self, clinical):
        tokens = self.tokenizer(self.serializer.serialize_batch(clinical)).to(self.backbone.embedding_layer.weight.device)
        return clinical_prediction_state(self.backbone, input_ids=tokens.input_ids,
                                         attention_mask=tokens.attention_mask,
                                         prediction_positions=tokens.prediction_positions)

    def forward(self, clinical):
        return self.projection(self.encode_hidden(clinical))
