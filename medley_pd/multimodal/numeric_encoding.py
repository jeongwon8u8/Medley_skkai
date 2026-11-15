"""Pure predictor-only raw encoding shared by Phase D provenance and M2 branch."""
from __future__ import annotations
import math
from .clinical_serializer import _numeric


def encode_numeric_record(serializer,record):
    serializer.serialize(record)  # exact Task 5 allowlist/missingness/sex validation
    row=[]
    for name in serializer.fields:
        value=record[name]
        if name=="sex":
            sex=None if value is None else serializer._sex(value)
            row.extend([float(sex=="female"),float(sex=="male"),float(sex is None)])
        else:
            row.extend([0. if value is None else float(_numeric(value)),float(value is None)])
    if not all(math.isfinite(v) for v in row):
        raise ValueError("Clinical values exceed finite raw encoding range")
    return row
