"""Architecture A / M2: independently encoded modalities fused outside the LLM."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import FUSION_A_DEFAULTS, validate_config
from .representations import EEGRepresentationEncoder, ClinicalRepresentationEncoder


@dataclass(frozen=True)
class FusionARepresentations:
    eeg: torch.Tensor
    clinical: torch.Tensor


class RepresentationFusionHead(nn.Module):
    """concat([EEG, clinical]) -> LayerNorm -> Linear/GELU/Dropout/Linear."""

    def __init__(self, eeg_dim=256, clinical_dim=256, hidden_dim=256, dropout=.1, *, device=None):
        super().__init__()
        if any(type(d) is not int or d <= 0 for d in (eeg_dim, clinical_dim, hidden_dim)):
            raise ValueError("Fusion dimensions must be positive integers")
        if type(dropout) not in (int,float) or not 0 <= dropout < 1:
            raise ValueError("Fusion dropout must be finite in [0,1)")
        self.eeg_dim, self.clinical_dim = eeg_dim, clinical_dim
        self.norm = nn.LayerNorm(eeg_dim + clinical_dim, device=device, dtype=torch.float32)
        self.mlp = nn.Sequential(nn.Linear(eeg_dim+clinical_dim,hidden_dim,device=device,dtype=torch.float32),
                                 nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim,1,device=device,dtype=torch.float32))

    def concatenate(self, eeg, clinical):
        for value, width in ((eeg,self.eeg_dim),(clinical,self.clinical_dim)):
            if (not isinstance(value,torch.Tensor) or value.ndim!=2 or value.shape[0]==0
                    or value.shape[1]!=width or not value.is_floating_point()
                    or value.device!=self.norm.weight.device or not bool(torch.isfinite(value).all())):
                raise ValueError("Fusion requires finite floating [B,D] branch vectors on the fusion device")
        if eeg.shape[0]!=clinical.shape[0]:
            raise ValueError("EEG and clinical batch sizes must match exactly")
        # Coherent new-module dtype; casts preserve autograd and never move devices.
        combined=torch.cat([eeg.to(dtype=self.norm.weight.dtype),clinical.to(dtype=self.norm.weight.dtype)],dim=-1)
        if not bool(torch.isfinite(combined).all()):
            raise ValueError("Nonfinite fusion representation after dtype conversion")
        return combined

    def forward(self, eeg, clinical):
        prediction=self.mlp(self.norm(self.concatenate(eeg,clinical))).squeeze(-1).float()
        if not bool(torch.isfinite(prediction).all()):
            raise ValueError("Nonfinite Architecture A predictions")
        return prediction


class EEGClinicalRepresentationFusionRegressor(nn.Module):
    """Task 3 EEG tensors + clinical dictionaries -> raw float32 MoCA [B].

    No EEG soft tokens, M1 fitted head, or M3 forward is used. The supplied Task 4
    backbone is not cloned. Use independently prepared backbones for comparisons.
    """

    def __init__(self, backbone):
        super().__init__()
        config=backbone.resolved_config
        validate_config(config.settings)
        if config.settings["architecture"]!="fusion_a" or config.settings["fusion"]["type"]!="representation":
            raise ValueError("Architecture A requires fusion_a with representation fusion")
        self.backbone=backbone
        device=backbone.embedding_layer.weight.device
        self.eeg_encoder=EEGRepresentationEncoder(config,device=device)
        self.clinical_encoder=ClinicalRepresentationEncoder(backbone)
        options={**FUSION_A_DEFAULTS,**config.settings.get("fusion_a",{})}
        self.head=RepresentationFusionHead(options["eeg_projection_dim"],options["clinical_projection_dim"],
                                           config.settings["fusion"]["fusion_dim"],
                                           config.settings["fusion"]["dropout"],device=device)

    @property
    def llm(self):
        return self.clinical_encoder.llm

    @property
    def pooling(self):
        return self.eeg_encoder.pooling

    @property
    def serializer(self):
        return self.clinical_encoder.serializer

    @property
    def tokenizer(self):
        return self.clinical_encoder.tokenizer

    def extract_representations(self, *, eeg_features, eeg_mask, clinical, artifact_flags=None):
        eeg=self.eeg_encoder(eeg_features=eeg_features,eeg_mask=eeg_mask,artifact_flags=artifact_flags)
        clinical_vector=self.clinical_encoder(clinical)
        if eeg.shape[0]!=clinical_vector.shape[0]:
            raise ValueError("EEG and clinical subject counts must match exactly")
        return FusionARepresentations(eeg,clinical_vector)

    def forward(self, *, eeg_features, eeg_mask, clinical, artifact_flags=None):
        vectors=self.extract_representations(eeg_features=eeg_features,eeg_mask=eeg_mask,
                                             clinical=clinical,artifact_flags=artifact_flags)
        return self.head(vectors.eeg,vectors.clinical)
