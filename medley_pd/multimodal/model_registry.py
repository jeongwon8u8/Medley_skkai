"""Lazy M0–M3 registry; scientific models/trainers keep their existing APIs."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import importlib
import importlib.util
import re

from .config import ResolvedConfig, validate_config, _validate_medley


class RunnerError(ValueError):
    """Sanitized error codes; never include patient values or arbitrary exceptions."""

    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class Registration:
    model: str
    architecture: str
    fusion: str
    module: str | None
    model_class: str | None
    trainer_module: str | None
    trainer_class: str | None
    checkpoint_module: str | None
    prediction_prefix: str | None


REGISTRY = {
    "M0": Registration("M0", "legacy_numerical", "none", None, None, None, None, None, None),
    "M1": Registration("M1", "clinical_only", "none", "clinical_regressor", "ClinicalLLMRegressor",
                       "clinical_trainer", "ClinicalTrainer", "clinical_checkpoint", "clinical_llm"),
    "M2": Registration("M2", "fusion_a", "representation", "fusion_a", "EEGClinicalRepresentationFusionRegressor",
                       "fusion_a_trainer", "FusionATrainer", "fusion_a_checkpoint", "fusion_a_llm"),
    "M3": Registration("M3", "fusion_b", "soft_token", "fusion_b", "EEGClinicalSoftTokenRegressor",
                       "fusion_b_trainer", "FusionBTrainer", "fusion_b_checkpoint", "fusion_b_llm"),
}


def registration(model):
    if not isinstance(model, str) or model not in REGISTRY:
        raise RunnerError("model", "Supported model identifiers are M0, M1, M2 and M3")
    return REGISTRY[model]


def model_config(config, model):
    entry = registration(model)
    settings = copy.deepcopy(config.to_dict()["settings"])
    if model != "M0":
        settings["architecture"], settings["fusion"]["type"] = entry.architecture, entry.fusion
    validate_config(settings)
    _validate_medley(settings, config.medley)
    return ResolvedConfig(settings, copy.deepcopy(config.medley), config.split_validation)


def check_readiness(config):
    """Local-assets-only real execution. Tests may explicitly inject a tiny factory."""
    llm = config.settings["llm"]
    if not llm["model_id"]:
        raise RunnerError("model_id", "Set llm.model_id and an immutable revision before real execution")
    if not llm["revision"] or re.fullmatch(r"[0-9a-fA-F]{40}", llm["revision"]) is None:
        raise RunnerError("revision", "Pin llm.revision to an immutable 40-hex commit before execution")
    if not llm["local_files_only"]:
        raise RunnerError("local_assets", "Runner requires llm.local_files_only=true; provision assets separately")
    for name in ("torch", "transformers", "safetensors", *(('peft',) if llm["tuning"] != "frozen" else ()),
                 *(('bitsandbytes',) if llm["quantization"] == "nf4" else ())):
        if importlib.util.find_spec(name) is None:
            raise RunnerError("dependency", f"Required dependency {name} is unavailable; complete environment preflight")


def _module(name):
    return importlib.import_module(f"{__package__}.{name}")


@dataclass
class ModelRuntime:
    model: object
    trainer: object
    checkpoints: object
    prediction_model: str

    def save(self, path):
        return self.checkpoints.save_inference_checkpoint(self.model, path,
            epochs_completed=self.trainer.epochs_completed, optimizer_steps=self.trainer.optimizer_steps)

    def restore(self, path):
        return self.checkpoints.load_inference_checkpoint(self.model, path)


def validate_runtime_config(requested, resolved):
    """Task 4 runtime resolution may not change scientific/runner configuration."""
    expected, actual = copy.deepcopy(requested), copy.deepcopy(resolved)
    validate_config(actual["settings"])
    _validate_medley(actual["settings"], actual["medley"])
    before, after = requested["settings"]["llm"], resolved["settings"]["llm"]
    if (before["revision"] is not None and before["revision"] != after["revision"]
            or before["padding_side"] is not None and before["padding_side"] != after["padding_side"]):
        raise RunnerError("runtime_config", "Runtime backbone revision or padding policy differs from request")
    if before["target_modules"] and before["tuning"] != "frozen":
        targets = after["target_modules"] or []
        matches = lambda name, target: name == target or name.endswith("." + target)
        if (not targets or any(not any(matches(name, target) for target in before["target_modules"]) for name in targets)
                or any(not any(matches(name, target) for name in targets) for target in before["target_modules"])):
            raise RunnerError("runtime_config", "Resolved adapter targets differ from requested modules")
    for document in (expected, actual):
        for key in ("padding_side", "target_modules", "device", "revision"):
            document["settings"]["llm"].pop(key, None)
    if expected != actual:
        raise RunnerError("runtime_config", "Backbone factory changed the intended experiment configuration")


def create_runtime(model, config, prepared, *, backbone_factory=None):
    entry = registration(model)
    if model == "M0":
        raise RunnerError("baseline_import_only", "M0 is a read-only prediction import; no model factory exists")
    if backbone_factory is None:
        check_readiness(config)
        backbone = _module("llm_backbone").load_llm_backbone(config, allow_download=False)
    else:
        backbone = backbone_factory(copy.deepcopy(config))
    validate_runtime_config(config.to_dict(), backbone.resolved_config.to_dict())
    instance = getattr(_module(entry.module), entry.model_class)(backbone)
    trainer = getattr(_module(entry.trainer_module), entry.trainer_class)(instance, split=prepared.outer,
        train_subjects=prepared.partition.train_subjects, validation_subjects=prepared.partition.validation_subjects)
    return ModelRuntime(instance, trainer, _module(entry.checkpoint_module),
                        f"{entry.prediction_prefix}:{config.settings['clinical_set']}")
