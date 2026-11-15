"""Small shared optimizer and Task 3 placement helpers; no training orchestration."""
from __future__ import annotations

import torch

from .config import validate_config
from .llm_backbone import parameter_statistics


def build_role_optimizer(model, roles):
    """roles: deterministic (name, module, optional LR) triples; null inherits LR."""
    validate_config(model.backbone.resolved_config.settings)
    settings=model.backbone.resolved_config.settings["training"]
    if not parameter_statistics(model.llm)["base_frozen"]:
        raise ValueError("Pretrained base parameters must remain frozen")
    groups=[]
    for name,module,rate in roles:
        parameters=list(module.parameters())
        if any(not p.requires_grad for p in parameters):
            raise ValueError("All new-module parameters must be trainable")
        if parameters:
            groups.append({"name":name,"params":parameters,
                           "lr":settings["lr_new_modules"] if rate is None else rate})
    adapters=[]
    for name,parameter in model.llm.named_parameters():
        if parameter.requires_grad:
            if "lora_" not in name or model.backbone.mode=="frozen":
                raise ValueError("Unexpected trainable pretrained parameter")
            adapters.append(parameter)
    if model.backbone.mode in ("lora","qlora") and not adapters:
        raise ValueError("Adaptation mode requires trainable Task 4 adapters")
    if adapters:
        groups.append({"name":"lora","params":adapters,"lr":settings["lr_lora"]})
    actual=[id(p) for group in groups for p in group["params"]]
    expected={id(p) for p in model.parameters() if p.requires_grad}
    if len(actual)!=len(set(actual)) or set(actual)!=expected:
        raise ValueError("Every trainable parameter must occur in exactly one optimizer group")
    return torch.optim.AdamW(groups,weight_decay=settings["weight_decay"])


def prepare_eeg_inputs(batch, *, n_subjects, device, dtype):
    """Validate subject axes and explicitly place EEG; never read targets/IDs."""
    eeg,mask,flags=(batch[key] for key in ("eeg_features","eeg_mask","artifact_flags"))
    if (not isinstance(eeg,torch.Tensor) or eeg.ndim!=3 or eeg.shape[0]!=n_subjects
            or not eeg.is_floating_point() or not isinstance(mask,torch.Tensor)
            or mask.shape!=eeg.shape[:2] or mask.dtype!=torch.bool
            or not isinstance(flags,torch.Tensor) or flags.shape!=mask.shape or flags.dtype!=torch.bool):
        raise ValueError("Expected aligned Task 3 EEG features, boolean existence mask and artifact flags")
    return dict(eeg_features=eeg.to(device=device,dtype=dtype),eeg_mask=mask.to(device),
                artifact_flags=flags.to(device),clinical=batch["clinical"])
