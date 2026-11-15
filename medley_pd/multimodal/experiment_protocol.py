"""Read-only subject protocol; no outer split generation, fitting or orchestration."""
from __future__ import annotations

import copy
import csv
from dataclasses import dataclass, field
import hashlib
from io import StringIO
import json
from pathlib import Path

from .config import validate_config, _validate_medley
from .data_adapter import SubjectSplit, validate_subject_partition, _subject_id


class ProtocolError(ValueError):
    """Privacy-preserving error code; never embeds IDs, targets or clinical values."""

    def __init__(self, code, message, *, details=None):
        self.code = code
        self.details = {} if details is None else dict(details)
        super().__init__(message)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def subject_ids(values):
    if isinstance(values, (str, bytes)):
        raise ProtocolError("subject_ids", "Subject IDs require an explicit sequence")
    ids = tuple(values)
    for sid in ids:
        _subject_id(sid)
    if not ids or len(ids) != len(set(ids)):
        raise ProtocolError("subject_ids", "Subject IDs must be nonempty and unique")
    return ids


class ReadOnlyOuterSplits:
    """Pin source bytes and cross-check every Task 2 split against the source.

    Configuration paths follow Medley's cwd-based semantics, bound at creation.
    Source content is checked before/after adapter access. No legacy get_splits
    call is used, because that function can generate/write missing outer splits.
    """

    def __init__(self, config, adapter):
        validate_config(config.settings)
        _validate_medley(config.settings, config.medley)
        self.config = copy.deepcopy(config)
        self.adapter = adapter
        self.path = (Path(config.medley["paths"]["split_root"]) /
                     f"{config.settings['dataset']}__{config.settings['task']}.csv").resolve()
        try:
            raw = self.path.read_bytes()
            reader = csv.DictReader(StringIO(raw.decode("utf-8")))
            header = reader.fieldnames or []
            if len(header) != len(set(header)) or not {"subject", "seed", "fold"} <= set(header):
                raise ValueError()
            records = list(reader)
            if not records:
                raise ValueError()
            rows, seen, cohorts = [], set(), {}
            for record in records:
                if None in record or any(v is None for v in record.values()):
                    raise ValueError()
                sid = _subject_id(record["subject"])
                seed, fold = int(record["seed"]), int(record["fold"])
                if seed < 0 or not 1 <= fold <= config.settings["evaluation"]["n_folds"] or (sid, seed) in seen:
                    raise ValueError()
                seen.add((sid, seed))
                rows.append((sid, seed, fold))
                cohorts.setdefault(seed, set()).add(sid)
            if any(ids != next(iter(cohorts.values())) for ids in cohorts.values()):
                raise ValueError()
            for seed in config.settings["evaluation"]["seeds"]:
                available = {f for _, s, f in rows if s == seed}
                if not set(config.settings["evaluation"]["folds"]) <= available:
                    raise ValueError()
        except (OSError, UnicodeError, ValueError, KeyError, TypeError, csv.Error):
            raise ProtocolError("outer_source", "Missing or malformed fixed outer split source") from None
        self._rows = tuple(rows)
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.subjects = tuple(sorted(next(iter(cohorts.values()))))
        self.assignment_fingerprint = fingerprint(sorted(rows))

    def assert_unchanged(self):
        try:
            unchanged = hashlib.sha256(self.path.read_bytes()).hexdigest() == self.sha256
        except OSError:
            unchanged = False
        if not unchanged:
            raise ProtocolError("outer_changed", "Fixed outer split source changed or disappeared")

    def get_outer_split(self, *, seed, fold):
        self.assert_unchanged()
        selection = self.config.settings["evaluation"]
        if type(seed) is not int or seed not in selection["seeds"] or type(fold) is not int or fold not in selection["folds"]:
            raise ProtocolError("selection", "Seed/fold must select actual configured source identifiers")
        selected = [(sid, f) for sid, s, f in self._rows if s == seed]
        expected = SubjectSplit(seed, fold, tuple(s for s, f in selected if f != fold),
                                tuple(s for s, f in selected if f == fold))
        validate_subject_partition(expected.train_subjects, expected.test_subjects)
        actual = self.adapter.get_split(seed, fold)
        self.assert_unchanged()
        if actual != expected:
            raise ProtocolError("adapter_split", "Adapter split does not exactly match pinned source assignments/order")
        available = set(subject_ids(self.adapter.subject_ids))
        if not set(self.subjects) <= available:
            raise ProtocolError("absent_subject", "Source split subjects are absent from adapter cohort")
        return expected


def derive_seed(seed, fold, *, purpose="inner_validation"):
    """Domain-separated SHA-256, first 32 bits big-endian; no Python hash/RNG."""
    if (type(seed) is not int or seed < 0 or type(fold) is not int or fold < 1
            or not isinstance(purpose, str) or not purpose.strip()):
        raise ProtocolError("seed", "Seed/fold and seed purpose must be explicit valid identifiers")
    digest = fingerprint({"version": "medley_seed_v1", "seed": seed, "fold": fold, "purpose": purpose})
    return int(digest[:8], 16)


@dataclass(frozen=True)
class InnerPartition:
    seed: int
    fold: int
    train_subjects: tuple[str, ...] = field(repr=False)
    validation_subjects: tuple[str, ...] = field(repr=False)
    test_subjects: tuple[str, ...] = field(repr=False)
    derived_seed: int
    validation_fraction: float
    policy: str = "sha256_rank_v1"
    rounding: str = "decimal_ceil_no_clamping"

    @property
    def counts(self):
        return {"train": len(self.train_subjects), "validation": len(self.validation_subjects),
                "test": len(self.test_subjects)}


def select_inner_validation(outer, *, fraction=.2):
    """Rank outer-train IDs by a seeded SHA-256; choose ceil(N*f), no strata.

    Output order preserves the original outer source, selection does not depend
    on model architecture, feature set, targets, globals or input iteration order.
    """
    from fractions import Fraction
    import math

    if not isinstance(outer, SubjectSplit):
        raise ProtocolError("inner_outer", "An explicit Task 2 outer SubjectSplit is required")
    train = subject_ids(outer.train_subjects)
    test = subject_ids(outer.test_subjects)
    validate_subject_partition(train, test)
    if type(fraction) not in (float, int) or not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ProtocolError("fraction", "Validation fraction must be finite and strictly between zero and one")
    proportion = Fraction(str(fraction)) * len(train)
    n_validation = (proportion.numerator + proportion.denominator - 1) // proportion.denominator
    if not 1 <= n_validation < len(train):
        raise ProtocolError("small_cohort", "Selected validation policy cannot leave nonempty train and validation")
    seed = derive_seed(outer.seed, outer.fold)
    ranked = sorted(train, key=lambda sid: (fingerprint({"version": "medley_inner_rank_v1",
                                                         "derived_seed": seed, "subject": sid}), sid))
    chosen = set(ranked[:n_validation])
    return InnerPartition(outer.seed, outer.fold, tuple(s for s in train if s not in chosen),
                          tuple(s for s in train if s in chosen), test, seed, float(fraction))


class ExperimentProtocol:
    """Shared M0–M3 protocol access, no models, training, split writes or scores.

    Exact-source policy requires all source subjects, no implicit intersection.
    prepare() revalidates records and source before producing a private snapshot.
    """

    def __init__(self, config, adapter):
        self.outer = ReadOnlyOuterSplits(config, adapter)
        self.config = copy.deepcopy(config)
        self.adapter = adapter
        self.manifest_root = Path(config.settings["protocol"]["manifest_dir"]).expanduser().resolve()
        self.source_roots = tuple(Path(config.medley["paths"][key]).resolve()
                                  for key in ("split_root", "data_root", "feature_root"))

    def get_outer_split(self, *, seed, fold):
        return self.outer.get_outer_split(seed=seed, fold=fold)

    def validate_cohort(self):
        from .experiment_cohort import validate_cohort
        self.outer.assert_unchanged()
        snapshot = validate_cohort(self.config, self.adapter, self.outer.subjects)
        self.outer.assert_unchanged()
        return snapshot

    def prepare(self, *, seed, fold):
        outer = self.get_outer_split(seed=seed, fold=fold)
        cohort = self.validate_cohort()
        partition = select_inner_validation(outer, fraction=self.config.settings["training"]["validation_fraction"])
        return PreparedProtocol(outer, partition, cohort, self.outer.sha256, self.outer.assignment_fingerprint)

    def compare_cohorts(self, snapshots):
        from .experiment_cohort import compare_cohorts
        return compare_cohorts(snapshots, strict=self.config.settings["protocol"]["strict_comparability"])


@dataclass(frozen=True)
class PreparedProtocol:
    outer: SubjectSplit = field(repr=False)
    partition: InnerPartition
    cohort: object = field(repr=False)
    outer_sha256: str
    assignment_fingerprint: str


def seed_run(seed, fold, *, purpose="training"):
    """Explicit global Python/NumPy/torch (including CUDA) seed; no deterministic-mode switch."""
    from .clinical_trainer import seed_training
    derived = derive_seed(seed, fold, purpose=purpose)
    seed_training(derived)
    return derived


def dataloader_generator(seed, fold):
    """Independent CPU torch generator for later C2 loaders; does not reset globals."""
    import torch
    return torch.Generator().manual_seed(derive_seed(seed, fold, purpose="dataloader"))
