"""Read-only bridge from existing Medley files to current-MoCA subject records.

No feature extraction, temporal averaging, batching, split generation or model
execution occurs here. Subject-level values must remain in private storage.
"""
from __future__ import annotations

import copy
from contextlib import redirect_stdout
import csv
from dataclasses import dataclass, field
from io import StringIO
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from types import MappingProxyType
from zipfile import BadZipFile

import numpy as np
import pandas as pd

from medley import features, participants
from medley.config import preprocess_spec, spec_hash
from .config import ConfigError, ResolvedConfig, validate_config, validate_split_selection

# Safety allowlist only: C0/C1/C2 composition still comes from Medley's YAML.
APPROVED_PREDICTORS = frozenset({
    "age", "sex", "updrs_part_iii", "disease_duration", "ledd", "pigd_score", "td_score",
})


class DataAdapterError(ValueError):
    """A data contract violation, with a stable code and no patient values."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class SubjectRecord:
    """One patient; arrays preserve cache dtype/order and all flagged segments.

    Private fields are excluded from repr to avoid accidental clinical logging.
    Arrays are independent copies; frozen does not make their contents immutable.
    """

    subject_id: str = field(repr=False)
    eeg_features: np.ndarray = field(repr=False)
    artifact_flags: np.ndarray = field(repr=False)
    clinical: Mapping[str, float | None] = field(repr=False)
    moca: float = field(repr=False)
    subgroup: str | None = field(repr=False)


@dataclass(frozen=True)
class SubjectExclusion:
    """Explicit participant eligibility outcome; never automatically written."""

    subject_id: str = field(repr=False)
    reason: str


@dataclass(frozen=True)
class SubjectSplit:
    """Fixed outer train/test assignment; no validation subdivision."""

    seed: int
    fold: int
    train_subjects: tuple[str, ...] = field(repr=False)
    test_subjects: tuple[str, ...] = field(repr=False)


def validate_subject_partition(train: Sequence[str], test: Sequence[str]) -> None:
    """Reject overlapping/duplicate subject IDs and empty train/test sides."""
    if set(train) & set(test):
        raise DataAdapterError("split_overlap", "Train/test subject overlap is prohibited")
    if len(set(train)) != len(train) or len(set(test)) != len(test):
        raise DataAdapterError("duplicate_split_subject", "Duplicate subject in train/test partition")
    if len(train) == 0 or len(test) == 0:
        raise DataAdapterError("empty_partition", "Fixed split must contain both train and test subjects")


def _subject_id(value: Any) -> str:
    if (not isinstance(value, str) or value != value.strip()
            or value.lower() in {"", "nan", "none", "<na>", "null"}
            or value in {".", ".."} or any(c in value for c in ("/", "\\", "\x00"))):
        raise DataAdapterError("invalid_subject_id", "Subject ID is missing, unsafe or not a canonical string")
    return value


def _label_reason(value: Any) -> str | None:
    if pd.isna(value):
        return "missing_moca"
    if isinstance(value, (bool, np.bool_)):
        return "invalid_moca"
    try:
        score = float(value)
    except (ValueError, TypeError):
        return "invalid_moca"
    return None if np.isfinite(score) and 0 <= score <= 30 else "invalid_moca"


def _clinical_fields(config: ResolvedConfig) -> tuple[str, ...]:
    """Use existing clinical-set selection plus strict predictor/source checks."""
    cfg = config.medley
    try:
        chosen = participants.clinical_sets(cfg)[config.settings["clinical_set"]]
    except (ValueError, KeyError) as exc:
        raise DataAdapterError("clinical_set", "Unsupported or prohibited clinical feature set") from exc
    if (not chosen or len(set(chosen)) != len(chosen)
            or not set(chosen) <= APPROVED_PREDICTORS):
        raise DataAdapterError("clinical_leakage", "Clinical set must contain distinct explicitly approved predictors only")
    columns = cfg["dataset"]["columns"]
    target = cfg["task"]["target"]
    banned = {target, "moca", "ctt", "updrs_total"} | set(cfg["task"].get("exclude", []))
    banned_sources = {str(columns[name]).casefold() for name in banned if name in columns}
    selected_sources: set[str] = set()
    for name in chosen:
        if name not in columns:
            raise DataAdapterError("clinical_column", "Required clinical column mapping is missing")
        source = columns[name]
        if not isinstance(source, str) or not source.strip():
            raise DataAdapterError("clinical_column", "Clinical source column must be a nonempty string")
        normalized = re.sub(r"[^a-z0-9]", "", source.casefold())
        if (source.casefold() in banned_sources
                or any(alias in normalized for alias in
                       ("moca", "ctt", "montrealcognitiveassessment", "colortrailstest"))
                or ("updrs" in normalized and "total" in normalized)):
            raise DataAdapterError("clinical_leakage", "Clinical column alias refers to a prohibited cognitive assessment")
        if source.casefold() in selected_sources:
            raise DataAdapterError("clinical_column", "Approved clinical predictors must not share a source column")
        selected_sources.add(source.casefold())
    return tuple(chosen)


class MedleyDataAdapter:
    """Read participants once, retrieve complete sequences lazily, reuse splits.

    Relative paths follow Medley's cwd-based rules, then are bound at adapter
    construction. Configuration and returned records never alias caller arrays.
    """

    def __init__(self, config: ResolvedConfig):
        validate_config(config.settings)
        self._config = copy.deepcopy(config)
        self._cfg = self._config.medley
        settings = self._config.settings
        if (self._cfg["dataset"]["name"] != settings["dataset"]
                or self._cfg["encoder"]["name"] != settings["eeg"]["encoder"]):
            raise DataAdapterError("configuration_mismatch", "Resolved dataset/encoder differs from multimodal settings")
        task = self._cfg["task"]
        if (task["name"] != "moca" or task["target"] != "moca"
                or task.get("type", "regression") != "regression"
                or task.get("transform") != "none" or not task.get("patients_only", True)):
            raise DataAdapterError("target_configuration", "Only patient-level current-MoCA regression in original units is supported")
        encoder = self._cfg["encoder"]
        actual_width = len(encoder["channels"]) * (200 if encoder["name"] == "cbramod" else encoder["arch"]["embed_dim"])
        if encoder.get("pool", "channel") != "channel" or actual_width != settings["eeg"]["feature_dim"]:
            raise DataAdapterError("feature_dimension", "Encoder configuration disagrees with cached feature dimension")
        self._variables = _clinical_fields(self._config)
        for name in ("data_root", "feature_root", "split_root"):
            self._cfg["paths"][name] = str(Path(self._cfg["paths"][name]).resolve())
        self.cache_spec_status = self._check_cache_spec()
        notice_buffer = StringIO()
        try:
            # Legacy sex normalization may print source values. Retain only an
            # aggregate notice, never patient content in terminal/log messages.
            with redirect_stdout(notice_buffer):
                table = participants.load_participants(self._cfg)
        except FileNotFoundError:
            raise DataAdapterError("missing_participants", "Participant metadata file is missing; configure an existing data_root") from None
        except KeyError:
            raise DataAdapterError("participant_schema", "Participant metadata is missing a configured clinical, target or subgroup column") from None
        self.metadata_notices = (("Medley emitted a normalization notice; verify clinical coding privately.",)
                                 if notice_buffer.getvalue() else ())
        required = {"subject", "is_patient", "moca"} | set(self._variables)
        if not required <= set(table.columns):
            raise DataAdapterError("participant_schema", "Participant table lacks required clinical/target fields")
        ids = [_subject_id(value) for value in table.subject]
        if len(ids) != len(set(ids)):
            raise DataAdapterError("duplicate_subject", "Duplicate participant subject records are not supported")
        exclusions = []
        for _, row in table.iterrows():
            if not isinstance(row.is_patient, (bool, np.bool_)):
                raise DataAdapterError("participant_schema", "Patient eligibility must be a boolean")
            reason = "not_patient" if not row.is_patient else _label_reason(row.moca)
            if reason:
                exclusions.append(SubjectExclusion(row.subject, reason))
        self.exclusions = tuple(exclusions)
        self._excluded = {entry.subject_id: entry.reason for entry in exclusions}
        # Reuse the existing cohort definition, then tighten finite/range checks.
        cohort = participants.task_cohort(self._cfg, table)
        cohort = cohort[~cohort.subject.isin(self._excluded)].copy()
        self._table = table.set_index("subject", drop=False)
        self._cohort = cohort.set_index("subject", drop=False)

    @property
    def subject_ids(self) -> tuple[str, ...]:
        """Eligible patient IDs in participant-table order; caches load on demand."""
        return tuple(self._cohort.index)

    @property
    def clinical_variables(self) -> tuple[str, ...]:
        """The selected, checked Medley clinical feature names."""
        return self._variables

    def _check_cache_spec(self) -> str:
        path = Path(features.feature_dir(self._cfg)) / "spec.json"
        if not path.exists():
            return "absent"
        try:
            spec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise DataAdapterError("cache_spec", "Cannot read feature cache spec.json") from exc
        if not isinstance(spec, dict) or spec_hash(spec) != spec_hash(preprocess_spec(self._cfg)):
            raise DataAdapterError("cache_spec", "Feature cache spec.json differs from configured Medley preprocessing/encoder")
        return "validated"

    def get_subject(self, subject_id: str) -> SubjectRecord:
        """Return a validated patient record, never averaging or removing windows."""
        subject_id = _subject_id(subject_id)
        if subject_id not in self._table.index:
            raise DataAdapterError("unknown_subject", "Unknown subject ID in participant metadata")
        if subject_id in self._excluded:
            reason = self._excluded[subject_id]
            raise DataAdapterError(reason, f"Subject is ineligible for current MoCA regression: {reason}")
        row = self._cohort.loc[subject_id]
        clinical: dict[str, float | None] = {}
        for name in self._variables:
            value = row[name]
            if pd.isna(value):
                clinical[name] = None
            else:
                try:
                    numeric = float(value)
                except (ValueError, TypeError) as exc:
                    raise DataAdapterError("clinical_value", f"Clinical field {name} is not numeric or missing") from exc
                if not np.isfinite(numeric):
                    raise DataAdapterError("clinical_value", f"Clinical field {name} must be finite or explicitly missing")
                clinical[name] = numeric
        path = Path(features.feature_dir(self._cfg)) / f"{subject_id}.npz"
        try:
            sequence, flags = features.load_segments(self._cfg, subject_id)
            # Medley's loader casts flags to bool. Validate the stored values too,
            # so NaN/strings cannot be silently converted to valid artifact flags.
            with np.load(path, allow_pickle=False) as archive:
                raw_flags = archive["flag"]
                if raw_flags.dtype.kind not in "bifu" or not np.isin(raw_flags, [0, 1]).all():
                    raise DataAdapterError("artifact_flags", "Stored artifact flags must be boolean or numeric 0/1")
        except FileNotFoundError:
            raise DataAdapterError("missing_cache", "EEG feature cache is missing for the requested subject/encoder") from None
        except (OSError, KeyError, ValueError, TypeError, IndexError, EOFError, BadZipFile) as exc:
            if isinstance(exc, DataAdapterError):
                raise
            raise DataAdapterError("malformed_cache", "Cannot read EEG NPZ cache with feat and flag arrays") from None
        if not isinstance(sequence, np.ndarray) or sequence.ndim != 2 or sequence.shape[0] == 0:
            raise DataAdapterError("feature_shape", "EEG features must be a nonempty [T, D_e] array")
        if sequence.shape[1] != self._config.settings["eeg"]["feature_dim"]:
            raise DataAdapterError("feature_dimension", "Cached EEG feature width differs from configured encoder")
        if sequence.dtype.kind != "f" or not np.isfinite(sequence).all():
            raise DataAdapterError("feature_values", "EEG features must contain finite real floating-point values")
        if not isinstance(flags, np.ndarray) or flags.ndim != 1 or len(flags) != len(sequence):
            raise DataAdapterError("artifact_length", "EEG feature/artifact flag lengths must agree on T")
        subgroup = row.get("subgroup")
        subgroup = None if pd.isna(subgroup) else str(subgroup)
        return SubjectRecord(subject_id, sequence.copy(), flags.astype(bool, copy=True),
                             MappingProxyType(clinical), float(row.moca), subgroup)

    def get_split(self, seed: int, fold: int) -> SubjectSplit:
        """Read saved outer split IDs; no writes, generation or inner subdivision.

        Fixed subjects absent/ineligible in metadata cause errors, not cohort
        shrinkage. Extra eligible subjects outside the manifest are not inserted.
        Cache availability remains a get_subject concern; no window is a subject.
        """
        evaluation = self._config.settings["evaluation"]
        if type(seed) is not int or seed not in evaluation["seeds"]:
            raise DataAdapterError("split_selection", "Requested seed is outside configured selection")
        if type(fold) is not int or fold not in evaluation["folds"]:
            raise DataAdapterError("split_selection", "Requested fold is outside configured selection")
        try:
            if validate_split_selection(self._config.settings, self._cfg) == "deferred":
                raise DataAdapterError("missing_splits", "Fixed split file is missing; generation is prohibited")
        except ConfigError as exc:
            raise DataAdapterError("split_schema", "Invalid existing fixed split metadata") from exc
        path = Path(self._cfg["paths"]["split_root"]) / f"{self._config.settings['dataset']}__moca.csv"
        try:
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                header = reader.fieldnames or []
                if len(header) != len(set(header)) or not {"subject", "seed", "fold"} <= set(header):
                    raise DataAdapterError("split_schema", "Fixed split CSV headers must be unique and include subject, seed, fold")
                rows = list(reader)
                if any(None in row or any(value is None for value in row.values()) for row in rows):
                    raise DataAdapterError("split_schema", "Fixed split CSV rows must match the header width")
        except (OSError, csv.Error) as exc:
            raise DataAdapterError("missing_splits", "Cannot read existing fixed split file") from exc
        # Recheck the actual read so malformed/changed assignments cannot bypass
        # validation between reads. IDs remain private and are not in messages.
        selected = []
        seen: set[str] = set()
        try:
            for row in rows:
                if int(row["seed"]) == seed:
                    sid = _subject_id(row["subject"])
                    assigned = int(row["fold"])
                    if sid in seen:
                        raise DataAdapterError("split_overlap", "Repeated subject assignment could cause train/test overlap")
                    if not 1 <= assigned <= evaluation["n_folds"]:
                        raise DataAdapterError("split_schema", "Fixed fold assignment is outside protocol")
                    if sid not in self._table.index:
                        raise DataAdapterError("split_subject", "Fixed split references subject absent from participant metadata")
                    if sid not in self._cohort.index:
                        raise DataAdapterError("split_ineligible", "Fixed split contains ineligible subject; inspect exclusions without regenerating splits")
                    seen.add(sid)
                    selected.append((sid, assigned))
        except (ValueError, TypeError, KeyError) as exc:
            if isinstance(exc, DataAdapterError):
                raise
            raise DataAdapterError("split_schema", "Malformed fixed split assignment") from exc
        train = tuple(sid for sid, assigned in selected if assigned != fold)
        test = tuple(sid for sid, assigned in selected if assigned == fold)
        validate_subject_partition(train, test)
        return SubjectSplit(seed, fold, train, test)
