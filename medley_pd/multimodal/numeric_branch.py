"""M2-only approved predictors, raw values plus missingness, no fitted statistics."""
from __future__ import annotations
import torch
from torch import nn
from .clinical_serializer import ClinicalTextSerializer
from .numeric_encoding import encode_numeric_record


class NumericClinicalBranch(nn.Module):
    """Numeric fields: [raw_value_or_zero,missing]; sex: [female,male,missing].

    Zero is a masked encoding, not imputed clinical measurement. Scaling is
    explicitly 'none'; no cohort/test statistics exist. Linear projection has
    independent trainable weights and is concatenated before M2's fusion MLP.
    """
    def __init__(self,config,*,device=None):
        super().__init__()
        if (config.settings["architecture"]!="fusion_a" or not config.settings["ablation"]["enabled"]
                or not config.settings["fusion"]["numeric_skip"]):
            raise ValueError("Numeric branch supports explicitly enabled M2 only")
        self.serializer=ClinicalTextSerializer(config)
        self.input_dim=2*(len(self.serializer.fields)-1)+3
        self.output_dim=config.settings["ablation"]["numeric_projection_dim"]
        self.projection=nn.Linear(self.input_dim,self.output_dim,device=device,dtype=torch.float32)

    def encode(self,records):
        self.serializer.serialize_batch(records)  # same leakage/missingness/coding validation
        if not records: raise ValueError("Numeric branch requires subjects")
        rows=[encode_numeric_record(self.serializer,record) for record in records]
        encoded=torch.tensor(rows,device=self.projection.weight.device,dtype=self.projection.weight.dtype)
        if not bool(torch.isfinite(encoded).all()): raise ValueError("Numeric values exceed finite encoding range")
        return encoded

    def forward(self,records):
        output=self.projection(self.encode(records))
        if not bool(torch.isfinite(output).all()): raise ValueError("Nonfinite numeric representation")
        return output
