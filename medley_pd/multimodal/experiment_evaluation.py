"""Prediction-only ingestion; no model construction, tensors or training imports."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd

from .data_adapter import _subject_id
from .experiment_protocol import fingerprint
from .experiment_manifest import read_manifest
from .run_state import read_json, file_hash as _file_hash, validate_completion

COLUMNS = ("seed", "fold", "subject", "subgroup", "model", "y", "yhat", "target")
KEYS = ["seed", "fold", "subject"]


class AnalysisError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def file_hash(path):
    try:
        return _file_hash(path)
    except OSError:
        raise AnalysisError("source_unavailable", "Required source artifact is missing or unreadable") from None


def validate_prediction_frame(frame):
    if (not isinstance(frame, pd.DataFrame) or frame.empty or frame.columns.duplicated().any()
            or set(frame.columns) != set(COLUMNS)):
        raise AnalysisError("schema", "Predictions require the exact existing eight-column schema")
    result = frame.loc[:, list(COLUMNS)].copy(deep=True)
    for row in result.itertuples(index=False):
        _subject_id(row.subject)
        if (isinstance(row.seed, (bool, np.bool_)) or not isinstance(row.seed, Integral) or row.seed < 0
                or isinstance(row.fold, (bool, np.bool_)) or not isinstance(row.fold, Integral) or row.fold < 1
                or not isinstance(row.model, str) or not row.model.strip()
                or row.subgroup is not None and not isinstance(row.subgroup, str)
                or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in (row.y,row.yhat,row.target))
                or not np.isfinite([row.y,row.yhat,row.target]).all() or not 0 <= row.target <= 30 or row.y != row.target):
            raise AnalysisError("values", "Invalid subject/seed/fold/model/subgroup or finite original-unit target/prediction values")
    if result.duplicated(["model", *KEYS]).any() or result.duplicated(["model", "seed", "subject"]).any():
        raise AnalysisError("duplicates", "One subject must have one prediction/fold per model and seed")
    for _, group in result.groupby("subject", sort=False):
        if group.target.nunique(dropna=False) != 1 or group.subgroup.nunique(dropna=False) != 1:
            raise AnalysisError("subject_metadata", "Target or subgroup changed across repeated subject predictions")
    return result


def _read_frame(path):
    try:
        frame = pd.read_csv(path, dtype={"subject": str, "model": str}, keep_default_na=False, float_precision="round_trip")
        if "subgroup" in frame:
            frame["subgroup"] = frame["subgroup"].replace("", None)
        return validate_prediction_frame(frame)
    except (OSError, ValueError, TypeError, KeyError):
        raise AnalysisError("predictions", "Missing or invalid prediction file; no values were coerced or imputed") from None


@dataclass(frozen=True)
class PredictionSeries:
    label: str
    _frame: pd.DataFrame = field(repr=False)
    runs: dict = field(repr=False)
    sources: dict = field(repr=False)
    provenance: str = "completed_C2_byte_hashes_no_tensor_loading"

    @property
    def frame(self):
        return self._frame.copy(deep=True)

    def assert_unchanged(self):
        for path, digest in self.sources.items():
            if file_hash(path) != digest:
                raise AnalysisError("source_changed", "Input artifacts changed after ingestion")


def load_completed_run(path, *, label=None):
    root = Path(path).expanduser().resolve()
    if root.name == "completed.json": root = root.parent
    marker = root / "completed.json"
    before = file_hash(marker)
    completion = read_json(marker)
    validate_completion(root, completion, verify_tensors=False)
    manifest_path = root.parent.parent / "manifest.json"
    manifest = read_manifest(manifest_path)
    attempt = root / completion["attempt"]
    frame = _read_frame(attempt / "predictions.csv.gz")
    info = manifest["identity"]
    label = info["model"] if label is None else label
    if not isinstance(label, str) or not label.strip():
        raise AnalysisError("label", "Analysis labels must be explicit nonempty strings")
    if (frame.subject.tolist() != manifest["subjects"]["test"] or not (frame.seed == info["seed"]).all()
            or not (frame.fold == info["fold"]).all() or not (frame.model == completion["prediction_model"]).all()):
        raise AnalysisError("coverage", "Predictions do not exactly cover their completed source partition")
    checkpoint_identity = None
    parameter_count = None
    if info["model"] != "M0":
        selection = read_json(attempt / "selection.json")
        checkpoint_metadata = read_json(attempt / selection["checkpoint"] / "manifest.json")
        checkpoint_identity = checkpoint_metadata["identity"]
        parameter_count = checkpoint_metadata.get("trainable_parameters")
        if parameter_count is not None and (type(parameter_count) is not int or parameter_count < 0):
            raise AnalysisError("parameter_count", "Invalid measured trainable parameter metadata")
    run = {"manifest": manifest, "completion": completion, "checkpoint_identity": checkpoint_identity,
           "source_root": str(root), "c1_root": str(root.parent.parent), "trainable_parameters": parameter_count}
    sources = {str(marker): before, str(manifest_path): file_hash(manifest_path)}
    sources.update({str(root / name): digest for name,digest in completion["artifacts"].items()})
    result = PredictionSeries(label,frame,{(info["seed"],info["fold"]):run},sources)
    result.assert_unchanged()
    return result


def _variant_signature(run):
    config = copy.deepcopy(run["manifest"]["resolved_config"])
    settings = config["settings"]
    for key in ("analysis", "experiment_name", "evaluation"):
        settings.pop(key,None)
    settings.get("protocol",{}).pop("manifest_dir",None)
    settings.get("runner",{}).pop("baseline_path",None)
    config["medley"].pop("paths",None)
    return fingerprint({"model":run["completion"]["prediction_model"],"configuration":config})


def load_series(paths, *, label=None):
    pieces = [load_completed_run(path,label=label) for path in paths]
    if not pieces or len({piece.label for piece in pieces}) != 1:
        raise AnalysisError("series", "A series needs completed runs with one explicit experiment label")
    runs,sources = {},{}
    for piece in pieces:
        if runs.keys() & piece.runs.keys():
            raise AnalysisError("duplicate_run", "Duplicate seed/fold executions cannot be combined")
        runs.update(piece.runs);sources.update(piece.sources)
    if len({_variant_signature(run) for run in runs.values()}) != 1:
        raise AnalysisError("variant", "Different experimental configurations require different analysis labels")
    frame = validate_prediction_frame(pd.concat([piece.frame for piece in pieces],ignore_index=True)).sort_values(KEYS).reset_index(drop=True)
    return PredictionSeries(pieces[0].label,frame,runs,sources)


def load_legacy_predictions(path, *, label, model):
    """Original M0 schema only; row matching never certifies historical provenance."""
    path = Path(path).expanduser().resolve()
    before = file_hash(path)
    frame = _read_frame(path)
    frame = frame.loc[frame.model == model].copy()
    if frame.empty:
        raise AnalysisError("legacy_model", "Requested original baseline model is absent")
    result = PredictionSeries(label,frame,{}, {str(path):before},"legacy_rows_only_unverified_provenance")
    result.assert_unchanged()
    return result
