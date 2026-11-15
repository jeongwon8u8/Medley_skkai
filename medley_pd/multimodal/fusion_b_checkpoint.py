"""M3 tensor-only inference checkpoint; base LLM reconstructed separately.

No optimizer or RNG persistence: this is not full training resumption.
"""
from __future__ import annotations

from .clinical_checkpoint import _identity, export_predictions
from .fusion_b import EEGClinicalSoftTokenRegressor
from .inference_checkpoint import save_checkpoint, load_checkpoint


def _parameters(model):
    if not isinstance(model, EEGClinicalSoftTokenRegressor):
        raise ValueError("M3 checkpoint requires Architecture B")
    return {name: parameter for name, parameter in model.named_parameters()
            if name.startswith(("pooling.", "projector.", "head.", "feature_adapter."))
            or name.startswith("llm.") and "lora_" in name}


def _model_identity(model):
    _parameters(model)
    return {**_identity(model), "format": "medley_m3_inference_v1"}


def save_inference_checkpoint(model, path, *, epochs_completed=0, optimizer_steps=0):
    return save_checkpoint(model, path, parameter_selector=_parameters, identity_builder=_model_identity,
                           label="M3", epochs_completed=epochs_completed, optimizer_steps=optimizer_steps)


def load_inference_checkpoint(model, path):
    return load_checkpoint(model, path, parameter_selector=_parameters, identity_builder=_model_identity, label="M3")
