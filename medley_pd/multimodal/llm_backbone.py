"""HF causal-LM backbone and explicit Frozen/LoRA/QLoRA preparation.

Downloads are disabled by default. This module contains no EEG/clinical model,
regression head, optimizer or training loop. Frozen forwards retain autograd.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import importlib
import inspect
from typing import Any

import torch
from torch import nn

from .config import ResolvedConfig, require_model_id, validate_config


class BackboneError(ValueError):
    """Unsupported model, adaptation policy, input or runtime requirement."""


def _dependency(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except (ImportError, OSError, RuntimeError):
        raise BackboneError(f"Cannot import {name}; install compatible Task 4 dependencies in the project environment") from None


def inspect_model_config(model_config: Any) -> dict[str, Any]:
    """Inspect metadata only; never instantiate weights or infer from model ID."""
    multimodal = any(getattr(model_config, key, None) is not None
                     for key in ("vision_config", "audio_config", "text_config"))
    if getattr(model_config, "is_encoder_decoder", False) or multimodal:
        raise BackboneError("Requires a text-only decoder causal LM; multimodal/encoder-decoder configs need a separate explicit adapter")
    return {"model_type": getattr(model_config, "model_type", None),
            "architectures": getattr(model_config, "architectures", None),
            "max_position_embeddings": getattr(model_config, "max_position_embeddings", None),
            "compatibility": "metadata_only_not_forward_verified"}


def resolve_device(name: str) -> torch.device:
    """Resolve explicit single-device placement without automatic offloading."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)
    if device.type == "cuda" and (not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count()):
        raise BackboneError("Requested CUDA device is unavailable; no precision/mode fallback is performed")
    return device


def require_quantization_backend() -> Any:
    """Check CUDA and bnb imports; this does NOT prove a quantized forward works."""
    if not torch.cuda.is_available():
        raise BackboneError("NF4/QLoRA requires CUDA in this initial backend; CPU-only tests must use Frozen/LoRA")
    bnb = _dependency("bitsandbytes")
    if not hasattr(bnb.nn, "Linear4bit"):
        raise BackboneError("Installed bitsandbytes has no Linear4bit backend")
    lib = getattr(getattr(bnb, "cextension", None), "lib", None)
    if lib is None:
        raise BackboneError("bitsandbytes native library is unavailable; verify its CUDA wheel/runtime compatibility")
    if getattr(lib, "compiled_with_cuda", True) is False:
        raise BackboneError("bitsandbytes was built without the required CUDA backend")
    return bnb


def quantization_config(settings: dict[str, Any]) -> Any:
    """Construct actual HF NF4 configuration after explicit backend checks."""
    require_quantization_backend()
    transformers = _dependency("transformers")
    return transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_compute_dtype=getattr(torch, settings["compute_dtype"]),
                                           bnb_4bit_use_double_quant=True)


def _tokenizer_setup(tokenizer: Any, settings: dict[str, Any]) -> None:
    if settings["padding_side"] is not None:
        tokenizer.padding_side = settings["padding_side"]
    if tokenizer.padding_side not in ("left", "right"):
        raise BackboneError("Tokenizer padding_side must be left or right")
    if tokenizer.pad_token_id is None:
        if settings["pad_token_policy"] != "eos" or tokenizer.eos_token_id is None or tokenizer.eos_token is None:
            raise BackboneError("Tokenizer has no pad token; explicitly select pad_token_policy=eos only if reusing EOS is intended")
        tokenizer.pad_token = tokenizer.eos_token
    # Never add tokens or resize a quantized embedding matrix implicitly.


def resolve_lora_targets(model: nn.Module, requested: list[str] | None) -> list[str]:
    """Match actual Linear/Conv1D modules, excluding the output head by identity."""
    Conv1D = _dependency("transformers.pytorch_utils").Conv1D
    output = model.get_output_embeddings()
    candidates = {name: module for name, module in model.named_modules()
                  if name and isinstance(module, (nn.Linear, Conv1D)) and module is not output}
    if requested is None:
        result = list(candidates)
    else:
        result = []
        for item in requested:
            matches = [name for name in candidates if name == item or name.endswith("." + item)]
            if not matches:
                raise BackboneError("A configured LoRA target matches no supported non-output Linear/Conv1D module")
            result.extend(matches)
        result = list(dict.fromkeys(result))
    if not result:
        raise BackboneError("No supported LoRA target modules found in actual model architecture")
    return result


def parameter_statistics(model: nn.Module) -> dict[str, Any]:
    """HF/PEFT parameter counts, not a memory estimate or proof of gradients."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if callable(getattr(model, "get_nb_trainable_parameters", None)):
        trainable, total = model.get_nb_trainable_parameters()
    elif callable(getattr(model, "num_parameters", None)):
        total = model.num_parameters()
    else:
        total = sum(p.numel() for p in model.parameters())
    base_frozen = all(not p.requires_grad for name, p in model.named_parameters() if "lora_" not in name)
    return {"total_parameters": total, "trainable_parameters": trainable,
            "trainable_percent": 100 * trainable / total if total else 0.0,
            "base_frozen": base_frozen,
            "trainable_lora_parameters": sum(p.numel() for n, p in model.named_parameters() if "lora_" in n and p.requires_grad)}


def _valid_mask(mask: torch.Tensor, shape: tuple[int, int], device: torch.device) -> torch.Tensor:
    if (not isinstance(mask, torch.Tensor) or mask.ndim != 2 or tuple(mask.shape) != shape
            or mask.device != device or not bool(((mask == 0) | (mask == 1)).all())):
        raise BackboneError("attention_mask must be binary [B,L] on the input device")
    valid = mask.bool()
    if not bool(valid.any(dim=1).all()):
        raise BackboneError("Each sequence requires at least one unmasked position")
    return valid


def select_prediction_state(hidden: torch.Tensor, attention_mask: torch.Tensor,
                            positions: torch.Tensor | None = None) -> torch.Tensor:
    """Select last actual unmasked position (left/right padding), or explicit IDs."""
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or min(hidden.shape) == 0 or not hidden.is_floating_point():
        raise BackboneError("hidden states must be nonempty [B,L,H]")
    valid = _valid_mask(attention_mask, tuple(hidden.shape[:2]), hidden.device)
    if positions is None:
        positions = torch.arange(hidden.shape[1], device=hidden.device).expand(hidden.shape[:2]).masked_fill(~valid, -1).max(dim=1).values
    if (not isinstance(positions, torch.Tensor) or positions.dtype != torch.long
            or positions.device != hidden.device or tuple(positions.shape) != (hidden.shape[0],)
            or not bool(((positions >= 0) & (positions < hidden.shape[1])).all())):
        raise BackboneError("prediction positions must be int64 [B] within sequence bounds")
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    if not bool(valid[rows, positions].all()):
        raise BackboneError("Prediction positions must refer to real unmasked tokens")
    return hidden[rows, positions]


@dataclass
class LLMBackbone:
    """Loaded causal model/tokenizer plus effective, snapshot-compatible config."""

    model: nn.Module
    tokenizer: Any
    resolved_config: ResolvedConfig
    target_modules: tuple[str, ...] = ()
    requested_revision: str | None = None

    @property
    def hidden_dim(self) -> int:
        return int(self.embedding_layer.weight.shape[1])

    @property
    def embedding_layer(self) -> nn.Module:
        return self.model.get_input_embeddings()

    @property
    def mode(self) -> str:
        return self.resolved_config.settings["llm"]["tuning"]

    def metadata(self) -> dict[str, Any]:
        settings = self.resolved_config.settings["llm"]
        return {"model_id": settings["model_id"], "requested_revision": self.requested_revision,
                "resolved_revision": getattr(self.model.config, "_commit_hash", None),
                "model_type": self.model.config.model_type, "hidden_dim": self.hidden_dim,
                "mode": self.mode, "device": str(self.embedding_layer.weight.device),
                "embedding_dtype": str(self.embedding_layer.weight.dtype),
                "padding_side": self.tokenizer.padding_side, "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id, "has_chat_template": bool(getattr(self.tokenizer, "chat_template", None)),
                "target_modules": list(self.target_modules), **parameter_statistics(self.model)}

    def forward_hidden(self, *, input_ids: torch.Tensor | None = None,
                       inputs_embeds: torch.Tensor | None = None,
                       attention_mask: torch.Tensor) -> torch.Tensor:
        """Final hidden states [B,L,H]; autograd remains enabled in Frozen mode."""
        if (input_ids is None) == (inputs_embeds is None):
            raise BackboneError("Provide exactly one of input_ids and inputs_embeds")
        source = input_ids if input_ids is not None else inputs_embeds
        if not isinstance(source, torch.Tensor) or source.ndim != (2 if input_ids is not None else 3) or min(source.shape) == 0:
            raise BackboneError("Expected input_ids [B,L] or inputs_embeds [B,L,H]")
        if source.device != self.embedding_layer.weight.device:
            raise BackboneError("Inputs must be on the embedding device; move batches explicitly")
        if input_ids is not None:
            if input_ids.dtype != torch.long or not bool(((input_ids >= 0) & (input_ids < self.embedding_layer.num_embeddings)).all()):
                raise BackboneError("input_ids must be valid int64 vocabulary indices")
        elif inputs_embeds.shape[-1] != self.hidden_dim or not inputs_embeds.is_floating_point() or not bool(torch.isfinite(inputs_embeds).all()):
            raise BackboneError("inputs_embeds must be finite floating [B,L,H] with the model hidden dimension")
        budget = self.resolved_config.settings["llm"]["max_sequence_tokens"]
        model_limit = getattr(self.model.config, "max_position_embeddings", None) or budget
        if source.shape[1] > min(budget, model_limit):
            raise BackboneError("Sequence exceeds configured/model context budget; no truncation is performed")
        valid = _valid_mask(attention_mask, tuple(source.shape[:2]), source.device)
        # Explicit position IDs keep real tokens at positions 0..N-1 under
        # either padding side. They do not define any prediction suffix.
        positions = valid.long().cumsum(dim=1).sub(1).clamp_min(0)
        output = self.model(input_ids=input_ids, inputs_embeds=inputs_embeds,
                            attention_mask=valid.long(), position_ids=positions,
                            output_hidden_states=True, return_dict=True, use_cache=False)
        states = getattr(output, "hidden_states", None)
        if not states or tuple(states[-1].shape) != (*source.shape[:2], self.hidden_dim):
            raise BackboneError("Model did not return final hidden states [B,L,H]")
        if not bool(torch.isfinite(states[-1]).all()):
            raise BackboneError("Model returned nonfinite final hidden states")
        return states[-1]


def prepare_llm_backbone(config: ResolvedConfig, model: nn.Module, tokenizer: Any) -> LLMBackbone:
    """Prepare an explicitly supplied local model, including tiny synthetic ones.

    This function never downloads or creates weights. QLoRA requires an actually
    4-bit-loaded model; a full-precision injected model cannot be called QLoRA.
    """
    validate_config(config.settings)
    effective = copy.deepcopy(config)
    settings = effective.settings["llm"]
    inspect_model_config(model.config)
    if not callable(getattr(model, "can_generate", None)) or not model.can_generate():
        raise BackboneError("Model must support causal language modeling")
    if model.get_input_embeddings() is None:
        raise BackboneError("Model must expose a token embedding layer")
    for key in ("inputs_embeds", "attention_mask", "position_ids"):
        signature = inspect.signature(model.forward).parameters
        if key not in signature and not any(p.kind == p.VAR_KEYWORD for p in signature.values()):
            raise BackboneError(f"Model forward does not expose {key}")
    _tokenizer_setup(tokenizer, settings)
    vocabulary = tokenizer.get_vocab()
    if not vocabulary or max(vocabulary.values()) >= model.get_input_embeddings().num_embeddings:
        raise BackboneError("Tokenizer vocabulary exceeds model embeddings; implicit resizing is prohibited")
    mode = settings["tuning"]
    if hasattr(model, "peft_config"):
        raise BackboneError("Inject an unadapted base model, not an already PEFT-wrapped model")
    if settings["quantization"] == "nf4":
        bnb = require_quantization_backend()
        quant = getattr(model.config, "quantization_config", {})
        if hasattr(quant, "to_dict"):
            quant = quant.to_dict()
        if not getattr(model, "is_loaded_in_4bit", False) or quant.get("bnb_4bit_quant_type") != "nf4":
            raise BackboneError("NF4 mode requires a genuinely 4-bit NF4-loaded model; full-precision LoRA is not QLoRA")
        if not any(isinstance(module, bnb.nn.Linear4bit) and getattr(module.weight, "quant_state", None) is not None
                   for module in model.modules()):
            raise BackboneError("Model has no genuinely quantized Linear4bit weights")
    elif getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        raise BackboneError("Loaded model quantization disagrees with configuration")
    model.requires_grad_(False)
    targets: list[str] = []
    checkpointing = effective.settings["training"]["gradient_checkpointing"]
    if mode == "qlora":
        peft = _dependency("peft")
        model = peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=checkpointing,
                                                    gradient_checkpointing_kwargs={"use_reentrant": False})
    if mode in ("lora", "qlora"):
        peft = _dependency("peft")
        targets = resolve_lora_targets(model, settings["target_modules"])
        options = peft.LoraConfig(task_type="CAUSAL_LM", r=settings["lora_r"], lora_alpha=settings["lora_alpha"],
                                  lora_dropout=settings["lora_dropout"], target_modules=targets, bias="none")
        model = peft.get_peft_model(model, options)
        stats = parameter_statistics(model)
        if not stats["base_frozen"] or stats["trainable_lora_parameters"] == 0:
            raise BackboneError("PEFT attachment did not leave only trainable LoRA parameters")
    if checkpointing and mode != "qlora":
        if not getattr(model, "supports_gradient_checkpointing", False):
            raise BackboneError("Selected model does not support requested gradient checkpointing")
        modern = getattr(model, "activation_checkpointing_enable", None)
        if callable(modern):
            modern(activation_checkpointing_kwargs={"use_reentrant": False})
        else:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    settings["padding_side"] = tokenizer.padding_side
    settings["target_modules"] = targets or None
    settings["device"] = str(model.get_input_embeddings().weight.device)
    requested_revision = settings["revision"]
    resolved_revision = getattr(model.config, "_commit_hash", None)
    if resolved_revision:
        settings["revision"] = resolved_revision
    return LLMBackbone(model, tokenizer, effective, tuple(targets), requested_revision)


def inspect_pretrained_config(config: ResolvedConfig, *, allow_download: bool = False) -> Any:
    """Read model metadata only, with an explicit network gate."""
    validate_config(config.settings)
    model_id = require_model_id(config)
    settings = config.settings["llm"]
    if not settings["local_files_only"] and not allow_download:
        raise BackboneError("Network access requires allow_download=True; local_files_only is the default")
    transformers = _dependency("transformers")
    try:
        model_config = transformers.AutoConfig.from_pretrained(model_id, revision=settings["revision"],
                       cache_dir=settings["cache_dir"], local_files_only=settings["local_files_only"],
                       trust_remote_code=settings["trust_remote_code"])
    except (OSError, ValueError, ImportError):
        raise BackboneError("Cannot access model config: check local cache, revision, architecture support and gated-repository authorization") from None
    inspect_model_config(model_config)
    return model_config


def load_llm_backbone(config: ResolvedConfig, *, allow_download: bool = False) -> LLMBackbone:
    """Explicit model loading; local-only by default, no fallback or auto-offload."""
    model_config = inspect_pretrained_config(config, allow_download=allow_download)
    settings = config.settings["llm"]
    transformers = _dependency("transformers")
    device = resolve_device(settings["device"])
    if settings["compute_dtype"] == "bfloat16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise BackboneError("BF16 was requested but the selected GPU does not support it")
    kwargs = dict(revision=settings["revision"], cache_dir=settings["cache_dir"],
                  local_files_only=settings["local_files_only"], trust_remote_code=settings["trust_remote_code"])
    model_kwargs = dict(config=model_config, dtype=getattr(torch, settings["compute_dtype"]))
    if settings["quantization"] == "nf4":
        if device.type != "cuda":
            raise BackboneError("NF4 requires a CUDA device in this backend")
        model_kwargs.update(quantization_config=quantization_config(settings), device_map={"": str(device)})
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(require_model_id(config), **kwargs)
        model = transformers.AutoModelForCausalLM.from_pretrained(require_model_id(config), **kwargs, **model_kwargs)
    except torch.OutOfMemoryError:
        raise BackboneError("Insufficient memory while loading; no automatic mode or precision fallback") from None
    except (OSError, ValueError, ImportError, RuntimeError):
        raise BackboneError("Cannot load tokenizer/causal LM: verify access, text-only architecture and dependency compatibility") from None
    if settings["quantization"] == "none":
        model = model.to(device)
    return prepare_llm_backbone(config, model, tokenizer)
