"""Explicit private exports and lightweight inference checkpoints, never pickle.

No optimizer/scheduler/RNG state is saved: these are NOT training-resume files.
Reconstruct the identical Task 4 backbone and tokenizer before loading.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file, save_file
import torch

from .clinical_regressor import ClinicalLLMRegressor
from .clinical_trainer import ClinicalEvaluation


REPOSITORY = Path(__file__).resolve().parents[2]
PREDICTION_COLUMNS = ("seed", "fold", "subject", "subgroup", "model", "y", "yhat", "target")


def _private_path(path) -> Path:
    path = Path(path).expanduser().resolve()
    if path == REPOSITORY or REPOSITORY in path.parents:
        relative = path.relative_to(REPOSITORY)
        if not relative.parts or relative.parts[0] not in ("outputs", "checkpoints"):
            raise ValueError("In-repository artifacts must use ignored outputs/ or checkpoints/")
    return path


def export_predictions(result: ClinicalEvaluation, path) -> Path:
    """Explicit export only; exclusive create, stable CSV/gzip, no stdout logs."""
    path = _private_path(path)
    if not (path.name.endswith(".csv") or path.name.endswith(".csv.gz")):
        raise ValueError("Prediction export requires .csv or .csv.gz")
    frame = result.predictions
    if (tuple(frame.columns) != PREDICTION_COLUMNS or len(frame) == 0
            or frame.duplicated(["seed", "subject", "model"]).any()
            or not np.isfinite(frame[["y", "yhat", "target"]].to_numpy(dtype=float)).all()
            or not np.array_equal(frame.y.to_numpy(), frame.target.to_numpy())):
        raise ValueError("Invalid or duplicate subject-level prediction records")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")
    with path.open("xb") as stream:
        if path.name.endswith(".gz"):
            with gzip.GzipFile(fileobj=stream, mode="wb", filename="", mtime=0) as zipped:
                zipped.write(text)
        else:
            stream.write(text)
    return path


def _checkpoint_parameters(model):
    # Inference callers may freeze the head/adapters too. Select the documented
    # checkpoint scope by role, never by temporary optimization flags.
    parameters = {n: p for n, p in model.named_parameters()
                  if n.startswith("head.") or n.startswith("llm.") and "lora_" in n}
    if not {"head.weight", "head.bias"} <= parameters.keys():
        raise ValueError("M1 checkpoint requires the complete linear regression head")
    return parameters


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _config_digest(config):
    # Architecture-specific optional defaults must not invalidate older M1/M3 files.
    settings = config["settings"]
    excluded = {"protocol"}
    if settings["architecture"] != "fusion_a":
        excluded.add("fusion_a")
    if settings["architecture"] == "clinical_only":
        excluded.add("fusion_b_training")
    return _digest({**config, "settings": {k: v for k, v in settings.items() if k not in excluded}})


def _identity(model):
    backbone = model.backbone
    config = backbone.resolved_config.to_dict()
    llm = config["settings"]["llm"]
    if llm["model_id"] is not None and llm["revision"] is None:
        raise ValueError("Pin a model revision before saving a recoverable pretrained checkpoint")
    synthetic_hash = None
    if llm["model_id"] is None:
        frozen = [(n, p) for n, p in model.llm.named_parameters() if "lora_" not in n]
        if sum(p.numel() for _, p in frozen) > 1_000_000:
            raise ValueError("Large injected backbones require an explicit model ID and revision")
        h = hashlib.sha256()
        for name, p in frozen:
            h.update(name.encode())
            h.update(p.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
        synthetic_hash = h.hexdigest()
    tokenizer = backbone.tokenizer
    tokenizer_identity = {"vocabulary": tokenizer.get_vocab(), "padding_side": tokenizer.padding_side,
                          "special_ids": [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id],
                          "backend": tokenizer.backend_tokenizer.to_str()}
    return {"format": "medley_m1_inference_v1", "config_digest": _config_digest(config),
            "base_config_digest": _digest(backbone.model.config.to_dict()),
            "tokenizer_digest": _digest(tokenizer_identity), "synthetic_frozen_digest": synthetic_hash,
            "model_id": llm["model_id"], "revision": llm["revision"], "mode": backbone.mode,
            "hidden_dim": backbone.hidden_dim, "kind": "inference_only_not_training_resume"}


def save_inference_checkpoint(model: ClinicalLLMRegressor, path, *, epochs_completed=0, optimizer_steps=0) -> Path:
    """Save head + adapters only. Caller preserves/reloads the base separately."""
    if any(type(v) is not int or v < 0 for v in (epochs_completed, optimizer_steps)):
        raise ValueError("Checkpoint progress counters must be nonnegative integers")
    identity = _identity(model)
    tensors = {n: p.detach().cpu().contiguous().clone() for n, p in _checkpoint_parameters(model).items()}
    if not tensors or any(not bool(torch.isfinite(p).all()) for p in tensors.values()):
        raise ValueError("Checkpoint trainable weights must be finite")
    path = _private_path(path)
    path.mkdir(parents=True, exist_ok=False)
    save_file(tensors, str(path / "trainable.safetensors"))
    config = model.backbone.resolved_config.to_dict()
    (path / "resolved_config.json").write_text(json.dumps(config, sort_keys=True, indent=2), encoding="utf-8")
    weights_hash = hashlib.sha256((path / "trainable.safetensors").read_bytes()).hexdigest()
    metadata = {"identity": identity, "weights_sha256": weights_hash,
                "epochs_completed": epochs_completed, "optimizer_steps": optimizer_steps}
    # Manifest written last: incomplete directories cannot be loaded.
    (path / "manifest.json").write_text(json.dumps(metadata, sort_keys=True, indent=2), encoding="utf-8")
    return path


def load_inference_checkpoint(model: ClinicalLLMRegressor, path) -> dict:
    """Load tensor-only weights into an explicitly reconstructed compatible M1.

    No dynamic imports, pickle, downloads, optimizer or configuration replacement.
    Metadata/hashes detect mismatch/corruption, not authenticity of untrusted files.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError("Checkpoint must be an existing inference checkpoint directory")
    try:
        metadata = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        saved_config = json.loads((path / "resolved_config.json").read_text(encoding="utf-8"))
        identity = _identity(model)
        saved_identity = dict(metadata["identity"])
        if saved_identity["config_digest"] not in (_digest(saved_config), _config_digest(saved_config)):
            raise ValueError("Incompatible or corrupted M1 inference checkpoint metadata")
        saved_identity["config_digest"] = _config_digest(saved_config)
        if (saved_identity != identity or _config_digest(saved_config) != identity["config_digest"]
                or hashlib.sha256((path / "trainable.safetensors").read_bytes()).hexdigest() != metadata["weights_sha256"]):
            raise ValueError("Incompatible or corrupted M1 inference checkpoint metadata")
        tensors = load_file(str(path / "trainable.safetensors"), device="cpu")
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Missing or invalid M1 checkpoint files") from exc
    parameters = _checkpoint_parameters(model)
    if tensors.keys() != parameters.keys() or any(
        tensors[n].shape != p.shape or tensors[n].dtype != p.dtype or not bool(torch.isfinite(tensors[n]).all())
        for n, p in parameters.items()
    ):
        raise ValueError("Incompatible M1 trainable tensor structure")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(tensors[name].to(parameter.device))
    return {"epochs_completed": metadata["epochs_completed"], "optimizer_steps": metadata["optimizer_steps"],
            "kind": "inference_only_not_training_resume"}
