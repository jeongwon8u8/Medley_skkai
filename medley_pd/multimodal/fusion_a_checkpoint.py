"""M2 inference weights only; reconstruct the exact Task 4 base separately."""
from .clinical_checkpoint import _identity, export_predictions
from .fusion_a import EEGClinicalRepresentationFusionRegressor
from .inference_checkpoint import save_checkpoint, load_checkpoint


def _parameters(model):
    if not isinstance(model,EEGClinicalRepresentationFusionRegressor):
        raise ValueError("M2 checkpoint requires Architecture A")
    return {name:p for name,p in model.named_parameters()
            if name.startswith(("eeg_encoder.","clinical_encoder.projection.","head.","numeric_branch."))
            or name.startswith("clinical_encoder.llm.") and "lora_" in name}


def _model_identity(model):
    _parameters(model)
    return {**_identity(model),"format":"medley_m2_inference_v1"}


def save_inference_checkpoint(model,path,*,epochs_completed=0,optimizer_steps=0):
    return save_checkpoint(model,path,parameter_selector=_parameters,identity_builder=_model_identity,
                           label="M2",epochs_completed=epochs_completed,optimizer_steps=optimizer_steps)


def load_inference_checkpoint(model,path):
    return load_checkpoint(model,path,parameter_selector=_parameters,identity_builder=_model_identity,label="M2")
