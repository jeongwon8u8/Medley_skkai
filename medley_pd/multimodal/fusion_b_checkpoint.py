"""M3 tensor-only inference checkpoint; base LLM reconstructed separately.

No optimizer or RNG persistence: this is not full training resumption.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from safetensors.torch import load_file, save_file
import torch

from .clinical_checkpoint import _digest, _identity, _private_path, export_predictions
from .fusion_b import EEGClinicalSoftTokenRegressor


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
    """Exclusive directory creation; complete role weights even if inference-frozen."""
    if any(type(v) is not int or v < 0 for v in (epochs_completed, optimizer_steps)):
        raise ValueError("Checkpoint progress counters must be nonnegative integers")
    identity = _model_identity(model)
    tensors = {n: p.detach().cpu().contiguous().clone() for n, p in _parameters(model).items()}
    if not tensors or any(not bool(torch.isfinite(p).all()) for p in tensors.values()):
        raise ValueError("M3 checkpoint weights must be finite")
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
                                  check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    path = _private_path(path)
    path.mkdir(parents=True, exist_ok=False)
    save_file(tensors, str(path / "trainable.safetensors"))
    config = model.backbone.resolved_config.to_dict()
    (path / "resolved_config.json").write_text(json.dumps(config, sort_keys=True, indent=2), encoding="utf-8")
    metadata = dict(identity=identity, weights_sha256=hashlib.sha256((path / "trainable.safetensors").read_bytes()).hexdigest(),
                    epochs_completed=epochs_completed, optimizer_steps=optimizer_steps, git_revision=revision)
    (path / "manifest.json").write_text(json.dumps(metadata, sort_keys=True, indent=2), encoding="utf-8")
    return path


def load_inference_checkpoint(model, path):
    """Validate all metadata and tensors BEFORE any parameter mutation."""
    path = Path(path).expanduser().resolve()
    try:
        metadata = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        config = json.loads((path / "resolved_config.json").read_text(encoding="utf-8"))
        identity = _model_identity(model)
        git_revision = metadata["git_revision"]
        if git_revision is not None and (not isinstance(git_revision, str) or not git_revision.strip()):
            raise ValueError("Invalid M3 Git provenance metadata")
        if (metadata["identity"] != identity or _digest(config) != identity["config_digest"]
                or hashlib.sha256((path / "trainable.safetensors").read_bytes()).hexdigest() != metadata["weights_sha256"]
                or any(type(metadata[key]) is not int or metadata[key] < 0 for key in ("epochs_completed", "optimizer_steps"))):
            raise ValueError("Incompatible or corrupted M3 inference checkpoint metadata")
        tensors = load_file(str(path / "trainable.safetensors"), device="cpu")
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Missing or invalid M3 checkpoint files") from exc
    parameters = _parameters(model)
    if tensors.keys() != parameters.keys() or any(
        tensors[n].shape != p.shape or tensors[n].dtype != p.dtype or not bool(torch.isfinite(tensors[n]).all())
        for n, p in parameters.items()
    ):
        raise ValueError("Incompatible M3 trainable tensor structure")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(tensors[name].to(parameter.device))
    return {key: metadata[key] for key in ("epochs_completed", "optimizer_steps", "git_revision")} | {
        "kind": "inference_only_not_training_resume"}
