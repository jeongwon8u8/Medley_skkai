"""Strict, JSON-based configuration for planned multimodal components.

Precedence: explicit ``extends`` ancestors < selected file < dotted overrides.
Medley owns dataset/task/encoder/preprocessing/path and fixed-split defaults.
This module never imports torch, transformers, features, or split generators.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable

import yaml

from medley.config import load_config as load_medley_config

JsonDict = dict[str, Any]
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "multimodal"
# Optional Task 4 fields are filled when reading older configs/snapshots.
LLM_RUNTIME_DEFAULTS = {
    "cache_dir": None, "local_files_only": True, "trust_remote_code": False,
    "remote_code_reason": None, "target_modules": None, "max_sequence_tokens": 512,
    "padding_side": None, "pad_token_policy": "require", "device": "auto",
}
# Versioned Task 5 defaults also apply to pre-Task-5 JSON and snapshots.
CLINICAL_TEXT_DEFAULTS = {
    "template_version": "medley_neutral_v1",
    "numeric_significant_digits": 15,
    "extra_approved_policy": "reject",
    "sex_code_mapping": None,
}
# Minimal M1 options; other optimization/loss/token budgets already exist.
CLINICAL_REGRESSION_DEFAULTS = {
    "prediction_suffix": "Predicted MoCA:", "head_type": "linear",
    "mixed_precision": "none", "random_seed": 0,
}
# Null group overrides inherit the existing new-module rate.
FUSION_B_TRAINING_DEFAULTS = dict.fromkeys(("lr_pooling", "lr_projector", "lr_head", "lr_feature_adapter"))
# Field names are a schema, not a second set of experiment/preprocessing values.
_SCHEMA: JsonDict = {
    **dict.fromkeys(("schema_version", "architecture", "experiment_name", "dataset", "task", "clinical_set")),
    "llm": dict.fromkeys(("model_id", "revision", "tuning", "quantization", "compute_dtype",
                          "lora_r", "lora_alpha", "lora_dropout", "max_text_tokens")),
    "eeg": {
        **dict.fromkeys(("encoder", "input_type", "feature_dim", "seed")),
        "feature_adapter": dict.fromkeys(("enabled", "hidden_dim", "dropout")),
        "pooling": dict.fromkeys(("type", "num_queries", "hidden_dim", "num_heads")),
        "projector": dict.fromkeys(("type", "layer_norm", "output_dim")),
    },
    "fusion": dict.fromkeys(("type", "token_position", "numeric_skip", "uncertainty", "fusion_dim", "dropout")),
    "prediction": dict.fromkeys(("type", "hidden_dim", "dropout", "clip_range")),
    "loss": dict.fromkeys(("primary", "huber_delta", "auxiliary")),
    "training": dict.fromkeys(("batch_size", "gradient_accumulation", "max_epochs", "early_stopping_patience",
                               "lr_lora", "lr_new_modules", "weight_decay", "gradient_checkpointing",
                               "validation_fraction", "optimizer", "gradient_clip_norm")),
    "evaluation": dict.fromkeys(("use_existing_splits", "seeds", "n_folds", "folds", "save_per_case")),
}
_SCHEMA["llm"].update(dict.fromkeys(LLM_RUNTIME_DEFAULTS))
_SCHEMA["clinical_text"] = dict.fromkeys(CLINICAL_TEXT_DEFAULTS)
_SCHEMA["clinical_regression"] = dict.fromkeys(CLINICAL_REGRESSION_DEFAULTS)
_SCHEMA["fusion_b_training"] = dict.fromkeys(FUSION_B_TRAINING_DEFAULTS)


def _with_optional_defaults(settings: Any) -> Any:
    result = copy.deepcopy(settings)
    if isinstance(result, dict) and isinstance(result.get("llm"), dict):
        result["llm"] = {**copy.deepcopy(LLM_RUNTIME_DEFAULTS), **result["llm"]}
    if isinstance(result, dict):
        if "clinical_text" not in result:
            result["clinical_text"] = copy.deepcopy(CLINICAL_TEXT_DEFAULTS)
        elif isinstance(result["clinical_text"], dict):
            result["clinical_text"] = {**copy.deepcopy(CLINICAL_TEXT_DEFAULTS), **result["clinical_text"]}
        if "clinical_regression" not in result:
            result["clinical_regression"] = copy.deepcopy(CLINICAL_REGRESSION_DEFAULTS)
        elif isinstance(result["clinical_regression"], dict):
            result["clinical_regression"] = {**copy.deepcopy(CLINICAL_REGRESSION_DEFAULTS), **result["clinical_regression"]}
        if "fusion_b_training" not in result:
            result["fusion_b_training"] = copy.deepcopy(FUSION_B_TRAINING_DEFAULTS)
        elif isinstance(result["fusion_b_training"], dict):
            result["fusion_b_training"] = {**FUSION_B_TRAINING_DEFAULTS, **result["fusion_b_training"]}
        # A1 optional fields preserve earlier JSON and resolved snapshots.
        if isinstance(result.get("eeg"), dict):
            result["eeg"].setdefault("seed", 0)
            if isinstance(result["eeg"].get("projector"), dict):
                result["eeg"]["projector"].setdefault("output_dim", None)
    return result


class ConfigError(ValueError):
    """An invalid configuration, override, or available split manifest."""


def recursive_merge(base: JsonDict, patch: JsonDict) -> JsonDict:
    """Return an independent recursively merged mapping; replace lists/scalars."""
    result = copy.deepcopy(base)
    for key, value in patch.items():
        result[key] = (recursive_merge(result[key], value)
                       if isinstance(result.get(key), dict) and isinstance(value, dict)
                       else copy.deepcopy(value))
    return result


def _pairs(pairs: list[tuple[str, Any]]) -> JsonDict:
    result: JsonDict = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read(path: Path) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ConfigError(f"Cannot read JSON configuration {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"{path}: configuration must be a JSON object")
    return value


def _inherit(path: Path, stack: tuple[Path, ...] = ()) -> JsonDict:
    path = path.resolve()
    if path in stack or len(stack) >= 32:
        raise ConfigError(f"Cyclic or excessively deep configuration inheritance: {path}")
    value = _read(path)
    if "extends" not in value:
        return value
    parent = value.pop("extends")
    if not isinstance(parent, str) or not parent.strip():
        raise ConfigError("extends must be a nonempty JSON file path")
    return recursive_merge(_inherit(path.parent / parent, (*stack, path)), value)


def _shape(value: Any, template: Any, path: str = "") -> None:
    if not isinstance(template, dict):
        return
    if not isinstance(value, dict):
        raise ConfigError(f"{path or 'configuration'} must be an object")
    unknown = value.keys() - template.keys()
    missing = template.keys() - value.keys()
    if unknown:
        raise ConfigError(f"Unknown configuration key: {path + '.' if path else ''}{sorted(unknown)[0]}")
    if missing:
        raise ConfigError(f"Missing required setting: {path + '.' if path else ''}{sorted(missing)[0]}")
    for key in template:
        _shape(value[key], template[key], f"{path}.{key}".lstrip("."))


def _get(config: JsonDict, path: str) -> Any:
    value: Any = config
    for key in path.split("."):
        value = value[key]
    return value


def _number(config: JsonDict, path: str, *, integer: bool = False,
            minimum: float = 0, strict: bool = True, maximum: float | None = None) -> None:
    value = _get(config, path)
    valid_type = type(value) is int if integer else type(value) in (int, float)
    if (not valid_type or not math.isfinite(value)
            or (value <= minimum if strict else value < minimum)
            or (maximum is not None and value >= maximum)):
        bounds = f"{'>' if strict else '>='} {minimum}" + (f" and < {maximum}" if maximum is not None else "")
        raise ConfigError(f"{path} must be a finite {'integer' if integer else 'number'} {bounds}")


def _choice(config: JsonDict, path: str, choices: Iterable[Any]) -> None:
    value = _get(config, path)
    if not any(type(value) is type(option) and value == option for option in choices):
        raise ConfigError(f"{path} must be one of {tuple(choices)}")


def validate_config(settings: JsonDict) -> None:
    """Validate a complete settings mapping without accessing models/data/GPU."""
    settings = _with_optional_defaults(settings)
    _shape(settings, _SCHEMA)
    for path, choices in {
        "schema_version": (1,), "architecture": ("clinical_only", "fusion_a", "fusion_b"),
        "dataset": ("pd_eeg", "iowa_rest"), "task": ("moca",),
        "clinical_set": ("C0", "C1", "C2"), "llm.tuning": ("frozen", "lora", "qlora"),
        "llm.quantization": ("none", "nf4"),
        "llm.compute_dtype": ("float32", "float16", "bfloat16"),
        "eeg.encoder": ("cbramod", "scerebro"), "eeg.input_type": ("cached_features",),
        "eeg.pooling.type": ("attention",), "eeg.projector.type": ("linear",),
        "fusion.type": ("none", "representation", "soft_token"),
        "fusion.token_position": ("prefix",), "fusion.numeric_skip": (False,),
        "fusion.uncertainty": (False,), "prediction.type": ("regression",),
        "loss.primary": ("huber",), "loss.auxiliary": (False,),
        "training.optimizer": ("adamw",), "evaluation.use_existing_splits": (True,),
    }.items():
        _choice(settings, path, choices)
    _choice(settings, "clinical_text.template_version", ("medley_neutral_v1",))
    _choice(settings, "clinical_text.extra_approved_policy", ("reject", "ignore"))
    _number(settings, "clinical_text.numeric_significant_digits", integer=True, maximum=35)
    mapping = settings["clinical_text"]["sex_code_mapping"]
    if mapping is not None and (
        not isinstance(mapping, dict) or not mapping
        or any(not isinstance(k, str) or re.fullmatch(r"0|-?[1-9][0-9]*", k) is None
               or not isinstance(v, str) or v not in ("female", "male") for k, v in mapping.items())
    ):
        raise ConfigError("clinical_text.sex_code_mapping must be null or verified integer-code strings mapped to female/male")
    _choice(settings, "clinical_regression.prediction_suffix", ("Predicted MoCA:",))
    _choice(settings, "clinical_regression.head_type", ("linear",))
    _choice(settings, "clinical_regression.mixed_precision", ("none", "bfloat16"))
    _number(settings, "clinical_regression.random_seed", integer=True, strict=False, maximum=2**32)
    for key in FUSION_B_TRAINING_DEFAULTS:
        if settings["fusion_b_training"][key] is not None:
            _number(settings, f"fusion_b_training.{key}")
    name = settings["experiment_name"]
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", name):
        raise ConfigError("experiment_name must be 1–128 ASCII letters/digits/underscores/hyphens, starting with a letter or digit")
    for path in ("llm.model_id", "llm.revision"):
        value = _get(settings, path)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ConfigError(f"{path} must be null or a nonempty string")
    for path in ("llm.lora_r", "llm.lora_alpha", "llm.max_text_tokens", "eeg.feature_dim",
                 "eeg.feature_adapter.hidden_dim", "eeg.pooling.num_queries", "eeg.pooling.hidden_dim",
                 "eeg.pooling.num_heads", "fusion.fusion_dim", "prediction.hidden_dim",
                 "training.batch_size", "training.gradient_accumulation", "training.max_epochs",
                 "training.early_stopping_patience", "evaluation.n_folds"):
        _number(settings, path, integer=True)
    for path in ("llm.lora_dropout", "eeg.feature_adapter.dropout", "fusion.dropout", "prediction.dropout"):
        _number(settings, path, strict=False, maximum=1)
    for path in ("loss.huber_delta", "training.lr_lora", "training.lr_new_modules", "training.gradient_clip_norm"):
        _number(settings, path)
    _number(settings, "training.weight_decay", strict=False)
    _number(settings, "training.validation_fraction", maximum=1)
    for path in ("eeg.feature_adapter.enabled", "eeg.projector.layer_norm",
                 "training.gradient_checkpointing", "evaluation.save_per_case"):
        if type(_get(settings, path)) is not bool:
            raise ConfigError(f"{path} must be a boolean")
    expected = {"clinical_only": "none", "fusion_a": "representation", "fusion_b": "soft_token"}
    if settings["fusion"]["type"] != expected[settings["architecture"]]:
        raise ConfigError("architecture and fusion.type are incompatible")
    eeg = settings["eeg"]
    _number(settings, "eeg.seed", integer=True, strict=False, maximum=2**32)
    if eeg["projector"]["output_dim"] is not None:
        _number(settings, "eeg.projector.output_dim", integer=True)
    if eeg["feature_dim"] != {"cbramod": 3800, "scerebro": 3420}[eeg["encoder"]]:
        raise ConfigError("eeg.feature_dim must match encoder: cbramod=3800, scerebro=3420")
    if eeg["pooling"]["hidden_dim"] % eeg["pooling"]["num_heads"]:
        raise ConfigError("eeg.pooling.hidden_dim must be divisible by num_heads")
    tuning, quant = settings["llm"]["tuning"], settings["llm"]["quantization"]
    if (tuning == "qlora" and quant != "nf4") or (tuning == "lora" and quant != "none"):
        raise ConfigError("llm.tuning/quantization: qlora requires nf4; lora requires none; frozen permits either")
    clip = settings["prediction"]["clip_range"]
    if clip is not None and (not isinstance(clip, list) or len(clip) != 2
                            or any(type(v) not in (int, float) or not math.isfinite(v) for v in clip)
                            or not 0 <= clip[0] < clip[1] <= 30):
        raise ConfigError("prediction.clip_range must be null or [low, high] with 0 <= low < high <= 30 (inference only)")
    for key in ("seeds", "folds"):
        values = settings["evaluation"][key]
        minimum = 0 if key == "seeds" else 1
        if (not isinstance(values, list) or not values
                or any(type(v) is not int or v < minimum for v in values)
                or len(set(values)) != len(values)):
            raise ConfigError(f"evaluation.{key} must be a nonempty list of distinct integers >= {minimum}")
    if max(settings["evaluation"]["folds"]) > settings["evaluation"]["n_folds"]:
        raise ConfigError("evaluation.folds exceeds evaluation.n_folds")
    _no_credentials(settings)
    llm = settings["llm"]
    for key in ("local_files_only", "trust_remote_code"):
        if type(llm[key]) is not bool:
            raise ConfigError(f"llm.{key} must be a boolean")
    for key in ("cache_dir", "remote_code_reason"):
        if llm[key] is not None and (not isinstance(llm[key], str) or not llm[key].strip()):
            raise ConfigError(f"llm.{key} must be null or a nonempty string")
    if llm["trust_remote_code"] and (not llm["remote_code_reason"] or not llm["revision"]):
        raise ConfigError("trust_remote_code requires remote_code_reason and an explicit revision")
    _choice(settings, "llm.padding_side", (None, "left", "right"))
    _choice(settings, "llm.pad_token_policy", ("require", "eos"))
    _number(settings, "llm.max_sequence_tokens", integer=True)
    if llm["max_sequence_tokens"] < llm["max_text_tokens"]:
        raise ConfigError("llm.max_sequence_tokens must cover max_text_tokens")
    if not isinstance(llm["device"], str) or not re.fullmatch(r"auto|cpu|cuda(?::[0-9]+)?", llm["device"]):
        raise ConfigError("llm.device must be auto, cpu or cuda[:index]")
    targets = llm["target_modules"]
    if targets is not None and (not isinstance(targets, list) or not targets
                               or any(not isinstance(x, str) or not x.strip() for x in targets)
                               or len(set(targets)) != len(targets)):
        raise ConfigError("llm.target_modules must be null or distinct nonempty module names")


def apply_overrides(settings: JsonDict, overrides: Iterable[str]) -> JsonDict:
    """Apply existing-style YAML-typed key=value overrides to known paths only."""
    result = copy.deepcopy(settings)
    for override in overrides:
        if not isinstance(override, str) or "=" not in override:
            raise ConfigError("Overrides must use key=value")
        path, raw = override.split("=", 1)
        parts = path.strip().split(".")
        current = result
        for key in parts[:-1]:
            if not isinstance(current.get(key), dict):
                raise ConfigError(f"Unknown override path: {path}")
            current = current[key]
        if parts[-1] not in current:
            raise ConfigError(f"Unknown override key: {path}")
        try:
            current[parts[-1]] = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            try:
                current[parts[-1]] = yaml.safe_load(raw)
            except yaml.YAMLError as exc:
                raise ConfigError(f"Invalid override value for {path}") from exc
    return result


def _no_credentials(value: Any, path: str = "configuration") -> None:
    """Reject credential fields and credential-bearing URLs before persistence."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in {"token", "password", "secret", "api_key", "access_token", "auth_token", "credentials"}:
                raise ConfigError(f"Credentials must not be stored in {path}.{key}; use environment authentication")
            _no_credentials(item, f"{path}.{key}")
    elif isinstance(value, list):
        for item in value:
            _no_credentials(item, path)
    elif isinstance(value, str) and (re.search(r"https?://[^/\s]*@|[?&](token|api_key|password)=", value, re.I)
                                    or value.startswith("hf_")):
        raise ConfigError(f"Credential-like value in {path}; use environment authentication")


def _validate_medley(settings: JsonDict, medley: JsonDict) -> None:
    """Check integration against Medley's actual task, encoder and protocol."""
    try:
        for key in ("dataset", "task", "encoder", "preprocess", "paths", "evaluation", "models"):
            if not isinstance(medley.get(key), dict):
                raise ConfigError(f"Resolved Medley configuration requires object: {key}")
        if (medley["dataset"]["name"] != settings["dataset"] or medley["task"]["name"] != settings["task"]
                or medley["encoder"]["name"] != settings["eeg"]["encoder"]):
            raise ConfigError("Resolved Medley dataset/task/encoder does not match settings")
        task, ds, enc, ev = (medley[k] for k in ("task", "dataset", "encoder", "evaluation"))
        if task["target"] != "moca" or task.get("type", "regression") != "regression" or task.get("transform") != "none":
            raise ConfigError("Only original-unit current MoCA regression is supported")
        sets = (ds.get("clinical_sets") or {}).get(task["name"]) or task["clinical_sets"]
        selected = settings["clinical_set"]
        if selected not in sets:
            raise ConfigError(f"clinical_set {selected} is unavailable for dataset {ds['name']}")
        variables = sets[selected]
        banned = set(task.get("exclude", [])) | {"moca", "ctt", "updrs_total", task["target"], "subject", "group"}
        if set(variables) & banned or not set(variables) <= ds["columns"].keys():
            raise ConfigError("Selected clinical set contains prohibited or unmapped variables")
        expected_dim = len(enc["channels"]) * (200 if enc["name"] == "cbramod" else enc["arch"]["embed_dim"])
        if (enc.get("pool", "channel") != "channel" or expected_dim != settings["eeg"]["feature_dim"]):
            raise ConfigError("Existing Medley encoder settings disagree with cached feature dimension")
        evaluation = settings["evaluation"]
        if evaluation["n_folds"] != ev["n_folds"]:
            raise ConfigError("evaluation.n_folds must preserve the existing Medley protocol; select evaluation.folds instead")
        if not set(evaluation["seeds"]) <= set(ev["seeds"]):
            raise ConfigError("evaluation.seeds must select existing Medley seeds")
        if not isinstance(medley["paths"]["split_root"], str):
            raise ConfigError("Medley paths.split_root must be a string")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ConfigError("Incomplete or invalid resolved Medley configuration") from exc
    _no_credentials(medley)


def validate_split_selection(settings: JsonDict, medley: JsonDict) -> str:
    """Read available CSV metadata only; return 'deferred' if absent, never write."""
    path = Path(medley["paths"]["split_root"]) / f"{settings['dataset']}__{settings['task']}.csv"
    if not path.exists():
        return "deferred"
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not {"subject", "seed", "fold"} <= set(reader.fieldnames or []):
                raise ConfigError("Fixed split CSV requires subject, seed, fold columns")
            rows = list(reader)
        if not rows:
            raise ConfigError("Fixed split CSV is empty")
        pairs: set[tuple[str, int]] = set()
        coverage: dict[int, set[int]] = {}
        cohorts: dict[int, set[str]] = {}
        for row in rows:
            seed, fold = int(row["seed"]), int(row["fold"])
            subject = row["subject"]
            if not subject or (subject, seed) in pairs or seed < 0 or not 1 <= fold <= settings["evaluation"]["n_folds"]:
                raise ConfigError("Fixed split CSV has duplicate, empty or out-of-range assignments")
            pairs.add((subject, seed))
            coverage.setdefault(seed, set()).add(fold)
            cohorts.setdefault(seed, set()).add(subject)
        if any(cohort != next(iter(cohorts.values())) for cohort in cohorts.values()):
            raise ConfigError("Fixed split CSV has inconsistent subject coverage across seeds")
        for seed in settings["evaluation"]["seeds"]:
            if not set(settings["evaluation"]["folds"]) <= coverage.get(seed, set()):
                raise ConfigError(f"Selected seed/folds unavailable in fixed split metadata (seed {seed})")
    except (OSError, ValueError, TypeError, csv.Error) as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(f"Cannot validate fixed split metadata at {path}") from exc
    return "validated"


@dataclass(frozen=True)
class ResolvedConfig:
    """Validated settings plus resolved Medley provenance; no executable modules.

    Mappings are independent copies, not deeply immutable. Revalidate on saving.
    """

    settings: JsonDict
    medley: JsonDict
    split_validation: str

    @property
    def effective_batch_size(self) -> int:
        """Configured single-GPU batch size including gradient accumulation."""
        return self.settings["training"]["batch_size"] * self.settings["training"]["gradient_accumulation"]

    @property
    def clinical_variables(self) -> list[str]:
        """Return permitted variables from the existing Medley task/dataset."""
        task, ds = self.medley["task"], self.medley["dataset"]
        sets = (ds.get("clinical_sets") or {}).get(task["name"]) or task["clinical_sets"]
        return list(sets[self.settings["clinical_set"]])

    def to_dict(self) -> JsonDict:
        """Return a standalone snapshot, independent of inheritance source paths."""
        return copy.deepcopy({"settings": _with_optional_defaults(self.settings), "medley": self.medley})


def load_multimodal_config(path: str | Path, overrides: Iterable[str] = ()) -> ResolvedConfig:
    """Resolve JSON inheritance/snapshots, apply overrides and validate read-only."""
    document = _inherit(Path(path))
    snapshot = "settings" in document or "medley" in document
    if snapshot:
        if document.keys() != {"settings", "medley"}:
            raise ConfigError("Snapshot must contain exactly settings and medley")
        settings = document["settings"]
    else:
        settings = document
    settings = _with_optional_defaults(settings)
    # Validate shape before override traversal, so invalid containers fail clearly.
    _shape(settings, _SCHEMA)
    settings = apply_overrides(settings, overrides)
    validate_config(settings)
    if snapshot:
        medley = document["medley"]
        if not isinstance(medley, dict):
            raise ConfigError("Snapshot medley must be an object")
    else:
        try:
            medley = load_medley_config(settings["dataset"], settings["eeg"]["encoder"], settings["task"])
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise ConfigError(f"Cannot resolve existing Medley configuration: {exc}") from exc
    _validate_medley(settings, medley)
    status = validate_split_selection(settings, medley)
    return ResolvedConfig(copy.deepcopy(settings), copy.deepcopy(medley), status)


def require_model_id(config: ResolvedConfig) -> str:
    """Future model loaders must call this before loading; no network action."""
    model_id = config.settings["llm"]["model_id"]
    if not isinstance(model_id, str) or not model_id.strip():
        raise ConfigError("A concrete llm.model_id is required before model loading")
    return model_id


def save_resolved_config(config: ResolvedConfig, output_path: str | Path, *, overwrite: bool = False) -> Path:
    """Save sorted JSON; exclusive create by default, explicit overwrite opt-in.

    Parent directories must exist. Only JSON destinations are accepted. Secrets
    belong in environment authentication, never configuration fields.
    """
    validate_config(config.settings)
    _validate_medley(config.settings, config.medley)
    path = Path(output_path).expanduser()
    if path.suffix.lower() != ".json" or path.is_symlink():
        raise ConfigError("Snapshot output must be a .json file and not a symbolic link")
    # Snapshot writing must never overwrite source configs or fixed manifests.
    resolved = path.resolve()
    protected = Path(__file__).resolve().parents[1] / "configs"
    if resolved.is_relative_to(protected) or resolved.is_relative_to(Path(config.medley["paths"]["split_root"]).resolve()):
        raise ConfigError("Snapshot output must be outside existing config and split directories")
    payload = json.dumps(config.to_dict(), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if overwrite else os.O_EXCL) | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
    except OSError as exc:
        raise ConfigError(f"Cannot save snapshot at {path}; parent must exist and overwrite requires explicit permission: {exc.strerror}") from exc
    return path


def main(argv: list[str] | None = None) -> int:
    """Validate configuration, print resolved JSON, optionally save a snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly permit replacing the snapshot output")
    args = parser.parse_args(argv)
    if args.overwrite and args.output is None:
        parser.error("--overwrite requires --output")
    try:
        config = load_multimodal_config(args.config, args.set)
        if args.output is not None:
            save_resolved_config(config, args.output, overwrite=args.overwrite)
    except ConfigError as exc:
        parser.exit(2, f"Configuration error: {exc}\n")
    print(json.dumps(config.to_dict(), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
