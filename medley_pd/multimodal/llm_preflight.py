"""Safe environment/metadata preflight; loading and network access are opt-in."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
from typing import Any

import torch

from .config import ConfigError, ResolvedConfig, load_multimodal_config
from .llm_backbone import (BackboneError, LLMBackbone, inspect_model_config,
                           inspect_pretrained_config, load_llm_backbone)


def _public_report(value: Any) -> Any:
    """Redact recognizable credential syntax, including inside local paths.

    Does not read authentication variables or change actual cache/model paths.
    Arbitrary secrets disguised as ordinary text cannot be classified reliably.
    """
    if isinstance(value, dict):
        return {key: _public_report(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_public_report(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16}",
                       "[REDACTED]", value)
        value = re.sub(r"(?i)(https?:/{1,2})[^/\s@]+@", r"\1[REDACTED]@", value)
        return re.sub(r"(?i)((?:access_token|auth_token|api_key|password|secret|token)[=:])[^/&\s]+",
                      r"\1[REDACTED]", value)
    return value


def collect_environment(cache_dir: str | None = None) -> dict[str, Any]:
    """Collect package/hardware facts; never enumerate environment or auth data."""
    versions = {}
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "safetensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    if cache_dir is None:
        try:
            from huggingface_hub.constants import HF_HUB_CACHE
            cache_dir = HF_HUB_CACHE
        except ImportError:
            cache_dir = str(Path.home() / ".cache" / "huggingface" / "hub")
    cache = Path(cache_dir).expanduser().resolve()
    existing = cache
    while not existing.exists():
        existing = existing.parent
    disk = shutil.disk_usage(existing)
    gpu = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        gpu = {"name": props.name, "total_vram_bytes": total, "free_vram_bytes": free,
               "compute_capability": list(torch.cuda.get_device_capability(0)),
               "bf16_supported": torch.cuda.is_bf16_supported()}
    return _public_report({"python": platform.python_version(), "versions": versions,
            "cuda_available": torch.cuda.is_available(), "torch_cuda_runtime": torch.version.cuda,
            "gpu": gpu, "model_cache": str(cache), "disk_free_bytes": disk.free,
            "disk_total_bytes": disk.total, "dependency_status": "installed_versions_not_import_or_quantization_proof"})


def estimate_parameter_storage(parameter_count: int = 4_000_000_000) -> dict[str, Any]:
    """Theoretical weight-only lower bounds, NOT a model-fit measurement."""
    if type(parameter_count) is not int or parameter_count <= 0:
        raise BackboneError("parameter_count must be a positive integer")
    return {"kind": "theoretical_weight_storage_estimate_not_peak_memory", "parameter_count": parameter_count,
            "float32_bytes": parameter_count * 4, "float16_bfloat16_bytes": parameter_count * 2,
            "nf4_packed_lower_bound_bytes": (parameter_count + 1) // 2,
            "excludes": ["quantization_metadata_and_unquantized_layers", "LoRA_and_optimizer_state",
                         "activations_and_attention", "CUDA_workspace_and_allocator"],
            "fit_in_24gb_verified": False}


def probe_backbone(backbone: LLMBackbone) -> dict[str, Any]:
    """Explicit tiny synthetic forward/backward probe, with no task loss/trainer.

    Intended for a freshly loaded model; clears its gradients before probing.
    No EEG tokens, clinical text or patient inputs are constructed.
    """
    model = backbone.model
    model.zero_grad(set_to_none=True)
    embeddings = backbone.embedding_layer
    device = embeddings.weight.device
    ids = (torch.arange(3, device=device) % embeddings.num_embeddings).reshape(1, 3)
    mask = torch.ones_like(ids)
    by_ids = backbone.forward_hidden(input_ids=ids, attention_mask=mask)
    external = embeddings(ids).detach().requires_grad_(True)
    hidden = backbone.forward_hidden(inputs_embeds=external, attention_mask=mask)
    from .llm_backbone import select_prediction_state
    selected = select_prediction_state(hidden, mask)
    selected[:, 0].sum().backward()
    external_ok = external.grad is not None and bool(torch.isfinite(external.grad).all()) and bool(external.grad.abs().sum() > 0)
    base_clean = all(p.grad is None for n, p in model.named_parameters() if "lora_" not in n)
    gradients = [p.grad for n, p in model.named_parameters() if "lora_" in n and p.grad is not None]
    adapter_finite = bool(gradients) and all(bool(torch.isfinite(grad).all()) for grad in gradients)
    adapter_grads = adapter_finite and any(bool(grad.abs().sum() > 0) for grad in gradients)
    return {"input_ids_hidden_shape": list(by_ids.shape), "inputs_embeds_hidden_shape": list(hidden.shape),
            "external_embedding_gradients_finite_nonzero": external_ok,
            "base_gradients_absent": base_clean, "lora_nonzero_gradient_observed": adapter_grads,
            "lora_gradients_finite": adapter_finite if backbone.mode != "frozen" else None,
            "probe": "synthetic_not_MoCA_training"}


def run_preflight(config: ResolvedConfig, *, inspect_metadata: bool = False,
                  load_model: bool = False, allow_download: bool = False,
                  parameter_count: int = 4_000_000_000) -> dict[str, Any]:
    """Default is environment-only: no AutoConfig/tokenizer/weight access."""
    llm = config.settings["llm"]
    env = collect_environment(llm["cache_dir"])
    report = {"environment": env,
              "requested": {key: llm[key] for key in ("model_id", "revision", "tuning", "quantization", "compute_dtype",
                                                        "local_files_only", "trust_remote_code", "device", "max_sequence_tokens")},
              "storage": estimate_parameter_storage(parameter_count),
              "storage_count_source": "caller_supplied_or_hypothetical_4B_not_inferred_model_size",
              "model_metadata": "not_inspected", "pretrained_load": "not_run",
              "quantized_gpu_forward": "not_run", "peak_allocated_bytes": None, "peak_reserved_bytes": None}
    report["nf4_prerequisite_flags"] = {
        "cuda_available": env["cuda_available"], "bitsandbytes_installed": env["versions"]["bitsandbytes"] is not None,
        "actual_backend_and_forward_verified": False,
    }
    if inspect_metadata:
        report["model_metadata"] = inspect_model_config(inspect_pretrained_config(config, allow_download=allow_download))
    if load_model:
        from .llm_backbone import resolve_device
        device = resolve_device(llm["device"])
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        backbone = load_llm_backbone(config, allow_download=allow_download)
        report["pretrained_load"] = "passed"
        report["backbone"] = backbone.metadata()
        report["probe"] = probe_backbone(backbone)
        if llm["quantization"] == "nf4" and device.type == "cuda":
            report["quantized_gpu_forward"] = "passed"
        if device.type == "cuda":
            report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    return _public_report(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--inspect-model", action="store_true", help="Explicitly read model configuration")
    parser.add_argument("--load-model", action="store_true", help="Explicitly load/probe a real model; may consume VRAM")
    parser.add_argument("--allow-download", action="store_true", help="Explicit network/large-weight opt-in; also requires local_files_only=false")
    parser.add_argument("--parameter-count", type=int, default=4_000_000_000)
    args = parser.parse_args(argv)
    if args.allow_download and not (args.inspect_model or args.load_model):
        parser.error("--allow-download requires --inspect-model or --load-model")
    try:
        config = load_multimodal_config(args.config, args.set)
        result = run_preflight(config, inspect_metadata=args.inspect_model, load_model=args.load_model,
                               allow_download=args.allow_download, parameter_count=args.parameter_count)
    except (BackboneError, ConfigError) as exc:
        parser.exit(2, f"Preflight error: {_public_report(str(exc))}\n")
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
