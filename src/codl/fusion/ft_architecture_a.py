"""Frozen EEG vectors + fresh FT tokenizer, two projections and team A head."""
import torch
from torch import nn
from codl.clinical.ft_clinical import _modules
from codl.fusion.team_representation import RepresentationProjection, RepresentationFusionHead


class FTArchitectureA(nn.Module):
    def __init__(self, cat_mask):
        super().__init__()
        _, Net = _modules()
        self.tokenizer = Net(0, cat_mask, 8, 32, .2).tok
        self.eeg_projection = RepresentationProjection(256, 256)
        self.clinical_projection = RepresentationProjection(len(cat_mask)*8, 256)
        self.head = RepresentationFusionHead(256, 256, 256, .1)

    def representations(self, eeg, clinical, missing):
        return (self.eeg_projection(eeg),
                self.clinical_projection(self.tokenizer(clinical, missing).flatten(1)))

    def forward(self, eeg, clinical, missing):
        return self.head(*self.representations(eeg, clinical, missing))

    def verify_gradients(self, eeg, clinical, missing):
        a, b = self.representations(eeg, clinical, missing)
        combined = self.head.concatenate(a, b)
        assert a.shape == b.shape == (len(eeg), 256)
        assert combined.shape == (len(eeg), 512)
        norms = {}
        for name in ('tokenizer', 'eeg_projection', 'clinical_projection', 'head'):
            gradients = [p.grad for p in getattr(self, name).parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            norms[name] = sum(float(g.abs().sum()) for g in gradients)
            assert norms[name] > 0, name
        return {'eeg_shape': list(a.shape), 'clinical_shape': list(b.shape),
                'fusion_shape': list(combined.shape), 'gradient_l1': norms}
