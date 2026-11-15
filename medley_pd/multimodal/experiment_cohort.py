"""Explicit cohort snapshots and read-only prediction alignment, never intersection."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from numbers import Real, Integral
from pathlib import Path

import numpy as np
import pandas as pd

from .clinical_serializer import ClinicalTextSerializer
from .numeric_encoding import encode_numeric_record
from .data_adapter import SubjectRecord, SubjectSplit, validate_subject_partition
from .experiment_protocol import ProtocolError, fingerprint, subject_ids


@dataclass(frozen=True)
class SubjectMetadata:
    subject: str = field(repr=False)
    target: float = field(repr=False)
    subgroup: str | None = field(repr=False)


@dataclass(frozen=True)
class CohortSnapshot:
    subjects: tuple[str, ...] = field(repr=False)
    metadata: tuple[SubjectMetadata, ...] = field(repr=False)
    cohort_fingerprint: str
    cache_fingerprint: str
    clinical_fingerprint: str
    cache_spec_status: str
    # Content hashes verify observed inputs, not extraction history or clinical truth.
    provenance_scope: str = "validated_decoded_records_not_extraction_history"

    @property
    def n(self):
        return len(self.subjects)


@dataclass(frozen=True)
class CohortMatch:
    matched: bool
    models: tuple[str, ...]
    counts: tuple[int, ...]
    reason: str
    scope: str = "subject_target_subgroup_only_not_training_equivalence"


def _array_digest(value):
    array = np.ascontiguousarray(value)
    return fingerprint({"shape": list(array.shape), "dtype": array.dtype.str,
                        "sha256": hashlib.sha256(array.tobytes()).hexdigest()})


def validate_cohort(config, adapter, source_subjects):
    """Validate ALL source subjects; any missing/extra/invalid record is an error.

    Selected clinical fields must exist; None/all-missing remains valid per Task 5.
    Returned metadata is private and omitted from repr. No labels enter a manifest.
    """
    expected = tuple(sorted(subject_ids(source_subjects)))
    available = subject_ids(adapter.subject_ids)
    if set(expected) != set(available):
        raise ProtocolError("cohort_membership", "Source and model-eligible cohorts differ; no implicit filtering",
                            details={"source_n": len(expected), "eligible_n": len(available),
                                     "missing_n": len(set(expected)-set(available)),
                                     "extra_n": len(set(available)-set(expected))})
    serializer = ClinicalTextSerializer(config)
    metadata, caches, clinical = [], [], []
    for sid in expected:
        try:
            record = adapter.get_subject(sid)
            if not isinstance(record, SubjectRecord) or record.subject_id != sid:
                raise ValueError()
            eeg, flags = record.eeg_features, record.artifact_flags
            if (not isinstance(eeg, np.ndarray) or eeg.ndim != 2 or eeg.shape[0] == 0
                    or eeg.shape[1] != config.settings["eeg"]["feature_dim"] or eeg.dtype.kind != "f"
                    or not np.isfinite(eeg).all() or not isinstance(flags, np.ndarray)
                    or flags.dtype != np.bool_ or flags.shape != (eeg.shape[0],)):
                raise ValueError()
            target = record.moca
            if (isinstance(target, (bool, np.bool_)) or not isinstance(target, Real)
                    or not np.isfinite(target) or not 0 <= target <= 30
                    or record.subgroup is not None and not isinstance(record.subgroup, str)):
                raise ValueError()
            text = serializer.serialize(record.clinical)
        except (ValueError, TypeError, KeyError, OSError):
            raise ProtocolError("cohort_record", "A source subject failed EEG, clinical, target or metadata validation") from None
        metadata.append(SubjectMetadata(sid, float(target), record.subgroup))
        caches.append([sid, _array_digest(eeg), _array_digest(flags)])
        # Phase D attests raw approved predictors for BOTH skip OFF and ON.
        # Text rounding alone cannot pin the additional numeric branch's inputs.
        clinical_input={"text":text,"raw_numeric":encode_numeric_record(serializer,record.clinical)} if config.settings.get("ablation",{}).get("enabled",False) else text
        clinical.append([sid, fingerprint(clinical_input)])
    identity = [[m.subject, m.target, m.subgroup] for m in metadata]
    return CohortSnapshot(expected, tuple(metadata), fingerprint(identity), fingerprint(caches),
                          fingerprint(clinical), getattr(adapter, "cache_spec_status", "unverified"))


def compare_cohorts(snapshots, *, strict=True):
    """No intersection: compare explicit named snapshots; strict mismatch raises."""
    if not snapshots or any(not isinstance(s, CohortSnapshot) for s in snapshots.values()):
        raise ProtocolError("comparison", "Explicit validated cohort snapshots are required")
    if any(not isinstance(name, str) or not name.strip() for name in snapshots):
        raise ProtocolError("comparison", "Model identifiers must be nonempty strings")
    names = tuple(sorted(snapshots))
    first = snapshots[names[0]]
    matched = all(s.subjects == first.subjects and s.metadata == first.metadata for s in snapshots.values())
    result = CohortMatch(matched, names, tuple(snapshots[n].n for n in names),
                         "matched" if matched else "subject_target_or_subgroup_mismatch")
    if strict and not matched:
        error = ProtocolError("comparison_mismatch", "Model cohorts/targets/subgroups differ; direct comparability refused")
        error.report = result
        raise error
    return result


def validate_predictions(frame, cohort, outer, *, model, target_atol=1e-5):
    """Audit one explicit seed/fold/model from a legacy long prediction table.

    Does NOT certify baseline training/cache/preprocessing provenance. Other
    selected runs are not combined or scored. Duplicate subject rows for this
    model/seed across different folds are also rejected.
    """
    required = {"seed", "fold", "subject", "subgroup", "model", "y", "yhat", "target"}
    if (not isinstance(frame, pd.DataFrame) or not required <= set(frame.columns)
            or frame.columns.duplicated().any() or not isinstance(cohort, CohortSnapshot)
            or type(target_atol) not in (int, float) or not np.isfinite(target_atol) or target_atol < 0):
        raise ProtocolError("predictions", "Invalid prediction schema or tolerance")
    if (not isinstance(outer, SubjectSplit) or type(outer.seed) is not int or outer.seed < 0
            or type(outer.fold) is not int or outer.fold < 1 or not isinstance(model, str) or not model.strip()):
        raise ProtocolError("prediction_metadata", "An explicit outer split and model identifier are required")
    validate_subject_partition(subject_ids(outer.train_subjects), subject_ids(outer.test_subjects))
    if set(outer.train_subjects + outer.test_subjects) != set(cohort.subjects):
        raise ProtocolError("prediction_population", "Outer split and cohort subject populations differ")
    for seed, fold, name in frame[["seed", "fold", "model"]].itertuples(index=False, name=None):
        if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0
                or isinstance(fold, (bool, np.bool_)) or not isinstance(fold, Integral) or fold < 1
                or not isinstance(name, str) or not name.strip()):
            raise ProtocolError("prediction_metadata", "Prediction seed/fold require non-boolean integers and model strings")
    selected_seed = frame.loc[(frame.model == model) & (frame.seed == outer.seed)]
    if selected_seed.duplicated("subject").any():
        raise ProtocolError("prediction_duplicate", "Duplicate subject predictions within model/seed")
    run = selected_seed.loc[selected_seed.fold == outer.fold]
    expected = set(outer.test_subjects)
    if (len(run) != len(expected) or set(run.subject) != expected
            or not expected <= set(cohort.subjects)):
        raise ProtocolError("prediction_population", "Prediction subjects differ from the exact outer test cohort")
    reference = {r.subject: r for r in cohort.metadata}
    for row in run.itertuples(index=False):
        values = [row.y, row.yhat, row.target]
        if (any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in values)
                or not np.isfinite(values).all()):
            raise ProtocolError("prediction_values", "Predictions and targets must be finite numbers")
        target = reference[row.subject].target
        subgroup = None if pd.isna(row.subgroup) else row.subgroup
        if (not np.isclose(row.y, target, atol=target_atol, rtol=0)
                or not np.isclose(row.target, target, atol=target_atol, rtol=0)):
            raise ProtocolError("prediction_target", "Prediction subject-to-target mapping differs")
        if subgroup != reference[row.subject].subgroup:
            raise ProtocolError("prediction_subgroup", "Prediction subgroup mapping differs")
    return {"model": model, "seed": outer.seed, "fold": outer.fold, "n": len(run),
            "population_aligned": True, "target_atol": target_atol, "training_provenance": "not_verified",
            "cohort_fingerprint": cohort.cohort_fingerprint}


def audit_prediction_file(path, cohort, outer, *, model, target_atol=1e-5):
    """Read-only CSV/gzip audit with source content hash, no prediction logging."""
    path = Path(path).resolve()
    try:
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        frame = pd.read_csv(path, dtype={"subject": str}, keep_default_na=False)
        # Match Task 6 CSV's empty subgroup -> None, without coercing subject IDs.
        frame["subgroup"] = frame["subgroup"].replace("", None)
        result = validate_predictions(frame, cohort, outer, model=model, target_atol=target_atol)
        if hashlib.sha256(path.read_bytes()).hexdigest() != before:
            raise ProtocolError("prediction_changed", "Prediction source changed during audit")
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        if isinstance(exc, ProtocolError):
            raise
        raise ProtocolError("prediction_file", "Missing or invalid prediction source") from None
    return {**result, "source_sha256": before}
