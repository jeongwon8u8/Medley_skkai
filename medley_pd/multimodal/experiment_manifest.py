"""Canonical private C1 run manifests; no labels/clinical records/embeddings.

Hashes bind observed content and metadata, not authenticity or extraction truth.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess

from .config import _no_credentials, _pairs, validate_config, _validate_medley
from .experiment_protocol import (ProtocolError, canonical_json, fingerprint, subject_ids,
                                  derive_seed, select_inner_validation)
from .data_adapter import SubjectSplit

REPOSITORY = Path(__file__).resolve().parents[2]
MODEL_ARCHITECTURES = {"M1": "clinical_only", "M2": "fusion_a", "M3": "fusion_b"}
IDENTITY_KEYS = set("""version model architecture baseline_identifier seed fold target unit
outer_sha256 assignment_fingerprint outer_source source_cohort_n model_eligible_n
matched_cohort_policy counts partition_fingerprint cohort_fingerprint cache_fingerprint
clinical_fingerprint cache_spec_status provenance_scope extraction_history baseline_training_provenance
inner_policy eeg clinical_set llm llm_revision_status config_fingerprint runtime output_root
source_subject_fingerprint""".split())


def _baselines(config):
    models, dataset, task = (config.medley[key] for key in ("models", "dataset", "task"))
    clinical_sets = (dataset.get("clinical_sets") or {}).get(task["name"]) or task["clinical_sets"]
    names = set(models.get("baselines") or ()) | {"eeg"}
    names |= {f"clinical:{name}" for name in clinical_sets}
    names |= {f"fusion_{name}:{clinical}" for name in (models.get("fusion") or ()) for clinical in clinical_sets}
    return names


def _revision_status(revision):
    if revision is None:
        return "unspecified"
    return "immutable_sha_supplied_not_verified" if re.fullmatch(r"[0-9a-fA-F]{40}", revision) else "mutable_reference_supplied_not_verified"


def runtime_metadata():
    libraries = {}
    for name in ("numpy", "pandas", "torch", "transformers", "peft", "safetensors"):
        try:
            libraries[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            libraries[name] = None
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPOSITORY,
                                  check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPOSITORY,
                                    check=True, capture_output=True, text=True).stdout)
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    sources = []
    for directory in ("multimodal", "medley"):
        for source in sorted((REPOSITORY / "medley_pd" / directory).rglob("*.py")):
            sources.append([str(source.relative_to(REPOSITORY)), fingerprint(source.read_text(encoding="utf-8"))])
    return {"python": platform.python_version(), "libraries": libraries,
            "application_source_fingerprint": fingerprint(sources),
            "git_revision": revision, "working_tree_dirty": dirty,
            "deterministic_algorithms": "not_enabled_by_protocol",
            "cuda_execution": "not_verified_by_protocol"}


@dataclass(frozen=True)
class ExperimentManifest:
    _payload: dict = field(repr=False)
    _protected_roots: tuple[Path, ...] = field(repr=False, default=())

    @property
    def run_id(self):
        return self._payload["run_id"]

    def to_dict(self):
        return copy.deepcopy(self._payload)

    def to_json(self):
        return canonical_json(self._payload) + "\n"

    def public_metadata(self):
        """Whitelist: never expose IDs, target values, local paths or config records."""
        identity = self._payload["identity"]
        return {"schema": self._payload["schema"], "run_id": self.run_id,
                **{key: copy.deepcopy(identity[key]) for key in
                   ("model", "architecture", "seed", "fold", "counts", "cohort_fingerprint",
                    "cache_fingerprint", "config_fingerprint", "outer_sha256", "provenance_scope")}}


def build_manifest(protocol, *, seed, fold, model=None, baseline_identifier=None):
    """Prepare/revalidate inputs and build a deterministic experiment identity.

    M0 manifests describe an audit/planned comparison; historical baseline inner
    selection/training is explicitly NOT certified by creating this manifest.
    """
    config = protocol.config
    validate_config(config.settings)
    _validate_medley(config.settings, config.medley)
    prepared = protocol.prepare(seed=seed, fold=fold)
    architecture = config.settings["architecture"]
    inferred = {v: k for k, v in MODEL_ARCHITECTURES.items()}[architecture]
    model = inferred if model is None else model
    if not isinstance(model, str):
        raise ProtocolError("model_identity", "Model identifier must be a supported string")
    if model == "M0":
        if not isinstance(baseline_identifier, str) or baseline_identifier not in _baselines(config):
            raise ProtocolError("baseline_identity", "M0 requires an explicit existing Medley baseline identifier")
        architecture = "legacy_numerical"
    elif model not in MODEL_ARCHITECTURES or MODEL_ARCHITECTURES[model] != architecture or baseline_identifier is not None:
        raise ProtocolError("model_identity", "Model identifier and configured architecture disagree")
    partition, cohort = prepared.partition, prepared.cohort
    subjects = {"train": list(partition.train_subjects), "validation": list(partition.validation_subjects),
                "test": list(partition.test_subjects)}
    snapshot = config.to_dict()
    root = protocol.manifest_root
    identity = {
        "version": "medley_c1_identity_v1", "model": model, "architecture": architecture,
        "baseline_identifier": baseline_identifier, "seed": seed, "fold": fold,
        "target": "current_moca_original_units", "unit": "subject",
        "outer_sha256": prepared.outer_sha256, "assignment_fingerprint": prepared.assignment_fingerprint,
        "outer_source": str(protocol.outer.path), "source_cohort_n": len(protocol.outer.subjects),
        "model_eligible_n": cohort.n, "source_subject_fingerprint": fingerprint(cohort.subjects), "matched_cohort_policy": "exact_source_no_intersection",
        "counts": partition.counts, "partition_fingerprint": fingerprint(subjects),
        "cohort_fingerprint": cohort.cohort_fingerprint, "cache_fingerprint": cohort.cache_fingerprint,
        "clinical_fingerprint": cohort.clinical_fingerprint, "cache_spec_status": cohort.cache_spec_status,
        "provenance_scope": cohort.provenance_scope,
        "extraction_history": "not_verified", "baseline_training_provenance": "not_verified" if model == "M0" else "not_applicable",
        "inner_policy": {"method": partition.policy, "fraction": partition.validation_fraction,
                         "rounding": partition.rounding, "derived_seed": partition.derived_seed,
                         "target_stratification": False,
                         "baseline_scope": "proposed_only_not_historical_baseline_verification" if model == "M0" else "planned_training"},
        "eeg": {"encoder": config.settings["eeg"]["encoder"], "feature_dim": config.settings["eeg"]["feature_dim"]},
        "clinical_set": config.settings["clinical_set"],
        "llm": {k: config.settings["llm"][k] for k in ("model_id", "revision", "tuning")},
        "llm_revision_status": _revision_status(config.settings["llm"]["revision"]),
        "config_fingerprint": fingerprint(snapshot), "runtime": runtime_metadata(), "output_root": str(root),
    }
    run_id = "medley-" + fingerprint(identity)
    payload = {"schema": "medley_experiment_manifest_v1", "run_id": run_id, "identity": identity,
               "subjects": subjects, "resolved_config": snapshot, "output_namespace": str(root / run_id)}
    _no_credentials(payload)
    protected = protocol.source_roots
    manifest = ExperimentManifest(payload, protected)
    validate_manifest(manifest.to_dict())
    protocol.outer.assert_unchanged()
    return manifest


def validate_manifest(payload):
    """Check self-consistency only. Revalidating source provenance requires protocol.prepare()."""
    try:
        if (set(payload) != {"schema", "run_id", "identity", "subjects", "resolved_config", "output_namespace"}
                or payload["schema"] != "medley_experiment_manifest_v1"):
            raise ValueError()
        identity, subjects = payload["identity"], payload["subjects"]
        if set(subjects) != {"train", "validation", "test"}:
            raise ValueError()
        partitions = [subject_ids(subjects[k]) for k in ("train", "validation", "test")]
        if any(set(a) & set(b) for i, a in enumerate(partitions) for b in partitions[i+1:]):
            raise ValueError()
        config = payload["resolved_config"]
        validate_config(config["settings"])
        _validate_medley(config["settings"], config["medley"])
        if (payload["run_id"] != "medley-" + fingerprint(identity)
                or identity["partition_fingerprint"] != fingerprint(subjects)
                or identity["config_fingerprint"] != fingerprint(config)
                or identity["counts"] != {k: len(v) for k, v in subjects.items()}
                or sum(identity["counts"].values()) != identity["source_cohort_n"]
                or identity["source_cohort_n"] != identity["model_eligible_n"]
                or Path(payload["output_namespace"]) != Path(identity["output_root"]) / payload["run_id"]):
            raise ValueError()
        if set(identity) != IDENTITY_KEYS:
            raise ValueError()
        settings = config["settings"]
        model = identity["model"]
        if (type(identity["seed"]) is not int or identity["seed"] not in settings["evaluation"]["seeds"]
                or type(identity["fold"]) is not int or identity["fold"] not in settings["evaluation"]["folds"]
                or any(type(v) is not int or v <= 0 for v in identity["counts"].values())):
            raise ValueError()
        if model == "M0":
            from .config import ResolvedConfig
            if identity["architecture"] != "legacy_numerical" or identity["baseline_identifier"] not in _baselines(ResolvedConfig(settings, config["medley"], "deferred")):
                raise ValueError()
        elif (model not in MODEL_ARCHITECTURES or identity["architecture"] != MODEL_ARCHITECTURES[model]
              or identity["architecture"] != settings["architecture"] or identity["baseline_identifier"] is not None):
            raise ValueError()
        expected_inner = {
            "method": settings["protocol"]["inner_policy"], "fraction": settings["training"]["validation_fraction"],
            "rounding": "decimal_ceil_no_clamping", "derived_seed": derive_seed(identity["seed"], identity["fold"]),
            "target_stratification": False,
            "baseline_scope": "proposed_only_not_historical_baseline_verification" if model == "M0" else "planned_training",
        }
        reconstructed = SubjectSplit(identity["seed"], identity["fold"], partitions[0] + partitions[1], partitions[2])
        expected_partition = select_inner_validation(reconstructed, fraction=settings["training"]["validation_fraction"])
        if (identity["inner_policy"] != expected_inner
                or set(expected_partition.validation_subjects) != set(partitions[1])
                or identity["source_subject_fingerprint"] != fingerprint(sorted(s for part in partitions for s in part))
                or identity["clinical_set"] != settings["clinical_set"]
                or identity["eeg"] != {k: settings["eeg"][k] for k in ("encoder", "feature_dim")}
                or identity["llm"] != {k: settings["llm"][k] for k in ("model_id", "revision", "tuning")}
                or identity["llm_revision_status"] != _revision_status(settings["llm"]["revision"])
                or identity["target"] != "current_moca_original_units" or identity["unit"] != "subject"
                or identity["version"] != "medley_c1_identity_v1"
                or identity["matched_cohort_policy"] != "exact_source_no_intersection"
                or identity["provenance_scope"] != "validated_decoded_records_not_extraction_history"
                or identity["extraction_history"] != "not_verified"
                or identity["baseline_training_provenance"] != ("not_verified" if model == "M0" else "not_applicable")
                or not Path(identity["outer_source"]).is_absolute() or not Path(identity["output_root"]).is_absolute()):
            raise ValueError()
        for key in ("outer_sha256", "assignment_fingerprint", "partition_fingerprint", "cohort_fingerprint",
                    "cache_fingerprint", "clinical_fingerprint", "config_fingerprint", "source_subject_fingerprint"):
            if not isinstance(identity[key], str) or re.fullmatch(r"[0-9a-f]{64}", identity[key]) is None:
                raise ValueError()
        runtime = identity["runtime"]
        if (set(runtime) != {"python", "libraries", "application_source_fingerprint", "git_revision", "working_tree_dirty",
                             "deterministic_algorithms", "cuda_execution"}
                or set(runtime["libraries"]) != {"numpy", "pandas", "torch", "transformers", "peft", "safetensors"}
                or runtime["deterministic_algorithms"] != "not_enabled_by_protocol"
                or runtime["cuda_execution"] != "not_verified_by_protocol"):
            raise ValueError()
        _no_credentials(payload)
    except (ValueError, TypeError, KeyError, AttributeError):
        raise ProtocolError("manifest", "Invalid or inconsistent experiment manifest") from None
    return payload


def save_manifest(manifest):
    """Exclusive private namespace and 0600 JSON, never source/config/data paths."""
    if not isinstance(manifest, ExperimentManifest):
        raise ProtocolError("manifest", "An explicitly built experiment manifest is required")
    payload = validate_manifest(manifest.to_dict())
    path = Path(payload["output_namespace"]).resolve()
    if any(path == root or root in path.parents for root in manifest._protected_roots):
        raise ProtocolError("manifest_path", "Manifest namespace must be outside data/cache/split source roots")
    if path == REPOSITORY or REPOSITORY in path.parents:
        ignored = subprocess.run(["git", "check-ignore", "--quiet", str(path / "manifest.json")], cwd=REPOSITORY).returncode == 0
        if not ignored:
            raise ProtocolError("manifest_path", "In-repository detailed manifests require ignored private directories")
    path.mkdir(parents=True, exist_ok=False, mode=0o700)
    descriptor = os.open(path / "manifest.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(manifest.to_json())
    return path / "manifest.json"


def read_manifest(path):
    """JSON only, no imports/pickle/downloads; does not certify present-day source equality."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_pairs)
    except (OSError, ValueError, UnicodeError):
        raise ProtocolError("manifest", "Missing or invalid experiment manifest JSON") from None
    return copy.deepcopy(validate_manifest(payload))
