"""Exact standalone team classes; no LLM dependencies.
Source: origin/medley-pd-code; imported class bodies unchanged.
"""

import torch
from torch import nn

SOURCE_COMMIT = '14871e54e61b4f3d177399405bb31519dcbba063'

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
