"""CPU subject samples and right-padded EEG batches over the Task 2 adapter.

One item is one patient. EEG existence masks and artifact flags are distinct.
There is no sampling, preprocessing, text serialization or model execution here.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real
import operator
from typing import Any, TypedDict

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .data_adapter import APPROVED_PREDICTORS, MedleyDataAdapter, SubjectRecord


class SubjectSample(TypedDict):
    """One subject with float32 CPU features/label and Boolean artifact flags."""

    subject_id: str
    eeg_features: torch.Tensor
    artifact_flags: torch.Tensor
    clinical: dict[str, float | None]
    moca: torch.Tensor
    subgroup: str | None


class SubjectBatch(TypedDict):
    """Patient-ordered CPU batch; eeg_mask=True denotes an existing segment."""

    subject_ids: list[str]
    eeg_features: torch.Tensor
    eeg_mask: torch.Tensor
    artifact_flags: torch.Tensor
    clinical: list[dict[str, float | None]]
    moca: torch.Tensor
    subgroups: list[str | None]


class SubjectDataError(ValueError):
    """Malformed sample/batch input; messages do not include patient values."""


def _check_id(value: Any) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SubjectDataError("subject_id must be a nonempty canonical string")


def _clinical_copy(value: Any) -> dict[str, float | None]:
    """Defend the existing Task 2 allowlist without duplicating source mappings."""
    if not isinstance(value, Mapping) or not value or not set(value) <= APPROVED_PREDICTORS:
        raise SubjectDataError("clinical must contain only Task 2 allowlisted predictors; targets are prohibited")
    for item in value.values():
        if item is not None and (isinstance(item, (bool, np.bool_))
                                 or not isinstance(item, Real) or not math.isfinite(item)):
            raise SubjectDataError("clinical values must preserve Task 2 finite numeric-or-None representation")
    return dict(value)


def _check_subgroup(value: Any) -> None:
    if value is not None and not isinstance(value, str):
        raise SubjectDataError("subgroup must be a string or None")


def _check_tensor(value: Any, name: str, dtype: torch.dtype, ndim: int) -> None:
    if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
            or value.device.type != "cpu" or value.dtype != dtype or value.ndim != ndim):
        raise SubjectDataError(f"{name} must be a strided CPU {dtype} tensor with rank {ndim}")


def _validate_sample(sample: Any) -> None:
    required = {"subject_id", "eeg_features", "artifact_flags", "clinical", "moca", "subgroup"}
    if not isinstance(sample, Mapping) or set(sample) != required:
        raise SubjectDataError("sample must contain exactly subject_id, eeg_features, artifact_flags, clinical, moca, subgroup")
    _check_id(sample["subject_id"])
    _check_subgroup(sample["subgroup"])
    _clinical_copy(sample["clinical"])
    eeg, flags, target = sample["eeg_features"], sample["artifact_flags"], sample["moca"]
    _check_tensor(eeg, "eeg_features", torch.float32, 2)
    _check_tensor(flags, "artifact_flags", torch.bool, 1)
    _check_tensor(target, "moca", torch.float32, 0)
    if eeg.shape[0] == 0 or eeg.shape[1] == 0:
        raise SubjectDataError("eeg_features must have nonempty [T_i, D_e] dimensions")
    if flags.shape[0] != eeg.shape[0]:
        raise SubjectDataError("artifact_flags length must match the EEG segment count")
    if not bool(torch.isfinite(eeg).all()):
        raise SubjectDataError("eeg_features must be finite")
    if not bool(torch.isfinite(target)) or not 0 <= target.item() <= 30:
        raise SubjectDataError("moca must be a finite float32 scalar in [0, 30]")


class MedleySubjectDataset(Dataset[SubjectSample]):
    """Lazy one-patient Dataset over an explicit ordered eligible-ID sequence.

    Membership in adapter.subject_ids inherits Task 2's canonical ID/eligibility
    validation. No cache is read until indexing. Adapter errors propagate intact.
    """

    def __init__(self, adapter: MedleyDataAdapter, subject_ids: Sequence[str]):
        if isinstance(subject_ids, (str, bytes)) or not isinstance(subject_ids, Sequence) or len(subject_ids) == 0:
            raise SubjectDataError("subject_ids must be a nonempty sequence of canonical subject IDs")
        ids = tuple(subject_ids)
        for sid in ids:
            _check_id(sid)
        if len(ids) != len(set(ids)):
            raise SubjectDataError("Duplicate subject IDs in Dataset are prohibited")
        if not set(ids) <= set(adapter.subject_ids):
            raise SubjectDataError("Unknown or ineligible subject ID; use Task 2 canonical eligible IDs")
        variables = tuple(adapter.clinical_variables)
        if not variables or len(variables) != len(set(variables)) or not set(variables) <= APPROVED_PREDICTORS:
            raise SubjectDataError("Adapter clinical_variables violate the Task 2 allowlist")
        self._adapter = adapter
        self._subject_ids = ids
        self._clinical_variables = frozenset(variables)

    @property
    def subject_ids(self) -> tuple[str, ...]:
        """Explicit IDs in supplied order; never inferred from EEG windows."""
        return self._subject_ids

    def __len__(self) -> int:
        return len(self._subject_ids)

    def __getitem__(self, index: int) -> SubjectSample:
        """Return independent CPU tensors and clinical dict for one subject.

        Python negative indexing is supported; slices and Boolean indices are not.
        Finite float64 EEG may round on conversion to float32, never overflow.
        """
        if isinstance(index, (bool, np.bool_)):
            raise TypeError("Dataset index must be an integer, not a boolean")
        position = operator.index(index)
        sid = self._subject_ids[position]
        record = self._adapter.get_subject(sid)
        if not isinstance(record, SubjectRecord):
            raise SubjectDataError("Adapter must return a complete Task 2 SubjectRecord")
        if record.subject_id != sid:
            raise SubjectDataError("SubjectRecord identity does not match the requested subject")
        eeg, flags = record.eeg_features, record.artifact_flags
        if np.ma.isMaskedArray(eeg) or np.ma.isMaskedArray(flags):
            raise SubjectDataError("SubjectRecord EEG and artifact_flags must not be masked arrays")
        if (not isinstance(eeg, np.ndarray) or eeg.ndim != 2 or min(eeg.shape) == 0
                or eeg.dtype.kind != "f" or not np.isfinite(eeg).all()):
            raise SubjectDataError("SubjectRecord EEG must be a finite floating nonempty [T_i, D_e] array")
        if np.abs(eeg).max() > np.finfo(np.float32).max:
            raise SubjectDataError("SubjectRecord EEG cannot be represented as finite float32")
        if (not isinstance(flags, np.ndarray) or flags.ndim != 1
                or flags.dtype != np.bool_ or len(flags) != len(eeg)):
            raise SubjectDataError("SubjectRecord artifact_flags must be Boolean [T_i] aligned with EEG")
        if (isinstance(record.moca, (bool, np.bool_)) or not isinstance(record.moca, Real)
                or not math.isfinite(record.moca) or not 0 <= record.moca <= 30):
            raise SubjectDataError("SubjectRecord moca must be a finite numeric scalar in [0, 30]")
        clinical = _clinical_copy(record.clinical)
        if set(clinical) != self._clinical_variables:
            raise SubjectDataError("SubjectRecord clinical keys differ from the adapter's selected feature set")
        _check_subgroup(record.subgroup)
        # Own contiguous NumPy copies handle read-only/negative-stride caches and
        # prevent caller mutation from changing Task 2 records or future samples.
        return SubjectSample(
            subject_id=sid,
            eeg_features=torch.from_numpy(np.array(eeg, dtype=np.float32, order="C", copy=True)),
            artifact_flags=torch.from_numpy(np.array(flags, order="C", copy=True)),
            clinical=clinical,
            moca=torch.tensor(record.moca, dtype=torch.float32, device="cpu"),
            subgroup=record.subgroup,
        )


class MedleySubjectCollator:
    """Right-pad CPU samples, preserving patient order and all real segments.

    EEG padding is zero; flag padding is False and meaningless where eeg_mask
    is False. eeg_mask is sequence existence, never artifact cleanliness or an
    LLM attention mask. No lengths are truncated, averaged, sorted or sampled.
    """

    def __call__(self, samples: Sequence[SubjectSample]) -> SubjectBatch:
        if not isinstance(samples, Sequence) or len(samples) == 0:
            raise SubjectDataError("Collator requires a nonempty sequence of subject samples")
        for sample in samples:
            _validate_sample(sample)
        ids = [sample["subject_id"] for sample in samples]
        if len(ids) != len(set(ids)):
            raise SubjectDataError("Duplicate subject IDs in a batch are prohibited")
        if len({sample["eeg_features"].shape[1] for sample in samples}) != 1:
            raise SubjectDataError("Batch contains incompatible EEG feature widths")
        if any(set(sample["clinical"]) != set(samples[0]["clinical"]) for sample in samples):
            raise SubjectDataError("Batch clinical keys must share the selected feature set")
        lengths = torch.tensor([sample["eeg_features"].shape[0] for sample in samples], dtype=torch.long, device="cpu")
        eeg = pad_sequence([sample["eeg_features"] for sample in samples], batch_first=True, padding_value=0.0)
        flags = pad_sequence([sample["artifact_flags"] for sample in samples], batch_first=True, padding_value=0)
        mask = torch.arange(eeg.shape[1], device="cpu")[None, :] < lengths[:, None]
        return SubjectBatch(
            subject_ids=ids,
            eeg_features=eeg,
            eeg_mask=mask,
            artifact_flags=flags,
            clinical=[dict(sample["clinical"]) for sample in samples],
            moca=torch.stack([sample["moca"] for sample in samples]),
            subgroups=[sample["subgroup"] for sample in samples],
        )
