"""Private single-owner attempts and atomic, integrity-checked run completion.

No optimizer/RNG persistence and no mid-epoch resume. Failed attempts are retained.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from numbers import Integral, Real
import uuid

from .config import _pairs
from .experiment_protocol import canonical_json, fingerprint
from .experiment_manifest import REPOSITORY, read_manifest
from .model_registry import RunnerError, registration, validate_runtime_config


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_pairs)
    except (OSError, ValueError, TypeError):
        raise RunnerError("run_artifact", "Missing or invalid run metadata") from None


def write_json(path, value, *, exclusive=False):
    """0600 temp + fsync; exclusive atomic link for immutable published records."""
    path = Path(path)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(canonical_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _artifact(root, name):
    relative = Path(name)
    path = root / relative
    if (relative.is_absolute() or ".." in relative.parts or path.is_symlink()
            or not path.is_file() or not path.resolve().is_relative_to(root.resolve())
            or any(parent.is_symlink() for parent in path.parents if parent != root.parent)):
        raise RunnerError("run_artifact", "Artifact path is missing, redirected or outside its run")
    return path


def _attempt_hashes(root, attempt):
    return {str(path.relative_to(root)): file_hash(_artifact(root, str(path.relative_to(root))))
            for path in sorted(attempt.rglob("*")) if path.is_file()}


def validate_completion(root, completion, *, expected_identity=None, verify_tensors=True):
    """Validate durable artifacts; C3 may explicitly verify tensor bytes by hash only.

    Default still loads/checks finite trainable tensors for C2; verify_tensors=False
    retains all completion/selection/config/checksum checks without tensor/model IO.
    """
    root = Path(root).resolve()
    try:
        identity = completion["execution_identity"]
        if (set(identity) != {"version", "c1_run_id", "baseline_sha256"} or identity["version"] != "medley_execution_v1"
                or completion["schema"] != "medley_completion_v1" or completion["execution_id"] != fingerprint(identity)
                or root.name != completion["execution_id"]
                or expected_identity is not None and identity != expected_identity):
            raise ValueError()
        manifest_path = root.parent.parent / "manifest.json"
        manifest = read_manifest(manifest_path)
        if (file_hash(manifest_path) != completion["manifest_sha256"]
                or manifest["run_id"] != identity["c1_run_id"]):
            raise ValueError()
        info = manifest["identity"]
        if (type(completion["seed"]) is not int or type(completion["fold"]) is not int
                or completion["model"] != info["model"] or completion["seed"] != info["seed"]
                or completion["fold"] != info["fold"]):
            raise ValueError()
        attempt_name = completion["attempt"]
        if len(Path(attempt_name).parts) != 2 or Path(attempt_name).parts[0] != "attempts":
            raise ValueError()
        attempt = root / attempt_name
        if completion["artifacts"] != _attempt_hashes(root, attempt) or not completion["artifacts"]:
            raise ValueError()
        for name in ("metrics.json", "predictions.csv.gz", "state.json"):
            if str(Path(attempt_name) / name) not in completion["artifacts"]:
                raise ValueError()
        import numpy as np
        import pandas as pd
        from medley.metrics import regression_metrics
        frame = pd.read_csv(attempt / "predictions.csv.gz", dtype={"subject": str}, keep_default_na=False)
        if (tuple(frame.columns) != ("seed", "fold", "subject", "subgroup", "model", "y", "yhat", "target")
                or frame.subject.tolist() != manifest["subjects"]["test"] or frame.subject.duplicated().any()
                or not (frame.seed == info["seed"]).all() or not (frame.fold == info["fold"]).all()
                or not (frame.model == completion["prediction_model"]).all()
                or not np.isfinite(frame[["y", "target", "yhat"]].to_numpy(dtype=float)).all()
                or not np.array_equal(frame.y, frame.target)):
            raise ValueError()
        for row in frame.itertuples(index=False):
            if (isinstance(row.seed, (bool, np.bool_)) or not isinstance(row.seed, Integral)
                    or isinstance(row.fold, (bool, np.bool_)) or not isinstance(row.fold, Integral)
                    or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                           for value in (row.y, row.target, row.yhat))):
                raise ValueError()
        state = read_json(attempt / "state.json")
        if (state["status"] != "artifacts_validated_pending_completion"
                or state["execution_id"] != completion["execution_id"] or state["training_resume"] is not False):
            raise ValueError()
        metrics = read_json(attempt / "metrics.json")
        if (type(metrics["n"]) is not int or any(type(metrics[key]) not in (int, float) or not math.isfinite(metrics[key])
                                               or metrics[key] < 0 for key in ("MAE", "RMSE"))):
            raise ValueError()
        calculated = regression_metrics(frame.y.to_numpy(), frame.yhat.to_numpy())
        if metrics["n"] != len(frame) or any(not math.isclose(metrics[k], calculated[k], rel_tol=1e-10, abs_tol=1e-10)
                                               for k in ("MAE", "RMSE")):
            raise ValueError()
        if info["model"] == "M0":
            if (completion["status"] != "imported" or completion["prediction_model"] != info["baseline_identifier"]
                    or identity["baseline_sha256"] is None):
                raise ValueError()
            audit = read_json(attempt / "baseline_audit.json")
            if audit["source_sha256"] != identity["baseline_sha256"] or audit["training_provenance"] != "not_verified":
                raise ValueError()
        else:
            if completion["status"] != "completed" or identity["baseline_sha256"] is not None:
                raise ValueError()
            entry = registration(info["model"])
            if completion["prediction_model"] != f"{entry.prediction_prefix}:{info['clinical_set']}":
                raise ValueError()
            selection = read_json(attempt / "selection.json")
            checkpoint = _artifact(root, str(Path(attempt_name) / selection["checkpoint"] / "manifest.json")).parent
            runtime = read_json(attempt / "runtime_config.json")
            cp = read_json(checkpoint / "manifest.json")
            from .config import inference_config_digest as _config_digest
            expected_format = {"M1": "medley_m1_inference_v1", "M2": "medley_m2_inference_v1", "M3": "medley_m3_inference_v1"}[info["model"]]
            if (type(cp["epochs_completed"]) is not int or type(cp["optimizer_steps"]) is not int
                    or selection["execution_id"] != completion["execution_id"]
                    or selection["best_epoch"] != cp["epochs_completed"]
                    or cp["identity"]["format"] != expected_format
                    or cp["identity"]["config_digest"] != _config_digest(runtime)
                    or file_hash(checkpoint / "trainable.safetensors") != cp["weights_sha256"]):
                raise ValueError()
            validate_runtime_config(manifest["resolved_config"], runtime)
            if read_json(checkpoint / "resolved_config.json") != runtime:
                raise ValueError()
            expected_selection = {"execution_id", "c1_run_id", "model", "seed", "fold", "best_epoch", "validation_MAE",
                                  "checkpoint", "restoration", "test_used_for_selection"}
            if (set(selection) != expected_selection or selection["c1_run_id"] != manifest["run_id"]
                    or selection["model"] != info["model"] or type(selection["seed"]) is not int
                    or type(selection["fold"]) is not int or selection["seed"] != info["seed"] or selection["fold"] != info["fold"]
                    or type(selection["best_epoch"]) is not int or selection["best_epoch"] < 1
                    or selection["restoration"] != "verified_exact_trainable_weights"
                    or selection["test_used_for_selection"] is not False
                    or selection["checkpoint"] != f"checkpoints/epoch-{selection['best_epoch']:04d}"):
                raise ValueError()
            history = read_json(attempt / "history.json")
            from .training_control import EarlyStopping
            settings = runtime["settings"]
            stopping = EarlyStopping(patience=settings["training"]["early_stopping_patience"],
                                     min_delta=settings["runner"]["min_delta"])
            maximum = settings["training"]["max_epochs"]
            if not history or len(history) > maximum:
                raise ValueError()
            n_train = len(manifest["subjects"]["train"])
            microbatches = (n_train + settings["training"]["batch_size"] - 1) // settings["training"]["batch_size"]
            accumulation = settings["training"]["gradient_accumulation"]
            steps = (microbatches + accumulation - 1) // accumulation
            for epoch, row in enumerate(history, 1):
                improved, stop = stopping.observe(row["validation"]["MAE"])
                if (type(row["epoch"]) is not int or row["epoch"] != epoch or type(row["improved"]) is not bool
                        or row["improved"] != improved or type(row["bad_epochs"]) is not int or row["bad_epochs"] != stopping.bad_epochs
                        or row["training"]["n"] != n_train or row["training"]["optimizer_steps"] != steps
                        or row["validation"]["n"] != len(manifest["subjects"]["validation"])
                        or not math.isfinite(row["training"]["huber"]) or row["training"]["huber"] < 0
                        or not math.isfinite(row["validation"]["RMSE"]) or row["validation"]["RMSE"] < 0
                        or stop and epoch != len(history)):
                    raise ValueError()
            if (not stop and len(history) != maximum or stopping.best_epoch != selection["best_epoch"]
                    or stopping.best != selection["validation_MAE"] or cp["optimizer_steps"] != steps * stopping.best_epoch):
                raise ValueError()
            if verify_tensors:
                from safetensors.torch import load_file
                import torch
                tensors = load_file(str(checkpoint / "trainable.safetensors"), device="cpu")
                if not tensors or any(not bool(torch.isfinite(t).all()) for t in tensors.values()):
                    raise ValueError()
    except (KeyError, TypeError, ValueError, OSError, IndexError, AttributeError):
        raise RunnerError("completion_invalid", "Completed-run identity or required artifacts failed integrity validation") from None
    return completion


def inspect_completed(path):
    root = Path(path).expanduser().resolve()
    completion = validate_completion(root, read_json(root / "completed.json"))
    return {"execution_id": completion["execution_id"], "model": completion["model"], "seed": completion["seed"],
            "fold": completion["fold"], "status": completion["status"], "integrity": "validated",
            "training_resume": False, "directory": str(root)}


def _validate_namespace(manifest):
    path = Path(manifest.to_dict()["output_namespace"])
    if path != path.resolve() or (path / "manifest.json").is_symlink():
        raise RunnerError("run_path", "Run namespace must not redirect through symbolic links")
    if any(path == root or root in path.parents for root in manifest._protected_roots):
        raise RunnerError("run_path", "Run namespace must stay outside data/cache/split sources")
    if path == REPOSITORY or REPOSITORY in path.parents:
        relative = path.relative_to(REPOSITORY)
        if (not relative.parts or relative.parts[0] not in ("outputs", "checkpoints")
                or subprocess.run(["git", "check-ignore", "--quiet", str(path / "manifest.json")], cwd=REPOSITORY).returncode != 0):
            raise RunnerError("run_path", "In-repository runner outputs require ignored outputs/ or checkpoints/")
    return path


class RunStore:
    def __init__(self, manifest, *, baseline_sha256=None):
        self.manifest = manifest
        namespace = _validate_namespace(manifest)
        # Atomically publish the C1 schema under a short single-owner bootstrap
        # lock. A crash before publication leaves no false/partial manifest.
        namespace.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(namespace / ".manifest.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RunnerError("run_busy", "Another process is publishing this C1 run manifest") from None
            manifest_path = namespace / "manifest.json"
            if manifest_path.exists():
                if read_manifest(manifest_path) != manifest.to_dict():
                    raise RunnerError("run_identity", "Existing C1 namespace has incompatible manifest identity")
            else:
                write_json(manifest_path, manifest.to_dict(), exclusive=True)
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
        self.identity = {"version": "medley_execution_v1", "c1_run_id": manifest.run_id,
                         "baseline_sha256": baseline_sha256}
        self.execution_id = fingerprint(self.identity)
        self.root = namespace / "executions" / self.execution_id
        self.attempt = None
        self._owned = False

    @contextmanager
    def claim(self, *, existing="fail"):
        if existing not in ("fail", "resume", "restart"):
            raise RunnerError("existing_policy", "Existing-run policy must be fail, resume or restart")
        if self._owned:
            raise RunnerError("run_busy", "This execution already owns an active attempt")
        _validate_namespace(self.manifest)
        if self.root != self.root.resolve() or (self.root / "attempts").is_symlink():
            raise RunnerError("run_path", "Execution/attempt paths must not redirect through symbolic links")
        self.root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            self.root.mkdir(mode=0o700)
            existed = False
        except FileExistsError:
            existed = True
            if existing == "fail" or self.root.is_symlink():
                raise RunnerError("run_exists", "Execution namespace already exists; explicit validated reuse/restart is required") from None
        descriptor = os.open(self.root / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RunnerError("run_busy", "Another process owns this execution") from None
            self._owned = True
            if (self.root / "completed.json").exists():
                if existing != "resume":
                    raise RunnerError("run_exists", "Completed runs cannot be restarted or overwritten")
                yield validate_completion(self.root, read_json(self.root / "completed.json"), expected_identity=self.identity)
                return
            if existed and existing != "restart":
                raise RunnerError("run_incomplete", "No valid completion marker; use explicit restart-incomplete")
            self.attempt = self.root / "attempts" / uuid.uuid4().hex
            self.attempt.mkdir(parents=True, mode=0o700)
            self.state("running")
            try:
                yield None
                if not (self.root / "completed.json").exists():
                    self.state("incomplete")
            except BaseException as exc:
                if not (self.root / "completed.json").exists():
                    self.state("failed", error_code="runner_error" if isinstance(exc, RunnerError) else "execution_failed")
                raise
        finally:
            self._owned = False
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def state(self, status, **metadata):
        if not self._owned or self.attempt is None:
            raise RunnerError("run_ownership", "State writes require an owned active attempt")
        if (self.root / "completed.json").exists():
            raise RunnerError("run_exists", "Published completed artifacts are immutable")
        write_json(self.attempt / "state.json", {"status": status, "execution_id": self.execution_id,
                                               "training_resume": False, **metadata})

    def complete(self, *, prediction_model, status="completed"):
        if not self._owned or self.attempt is None:
            raise RunnerError("run_ownership", "Completion requires an owned active attempt")
        self.state("artifacts_validated_pending_completion")
        identity = self.manifest.to_dict()["identity"]
        completion = {"schema": "medley_completion_v1", "execution_identity": self.identity,
                      "execution_id": self.execution_id, "model": identity["model"], "seed": identity["seed"],
                      "fold": identity["fold"], "prediction_model": prediction_model, "status": status,
                      "manifest_sha256": file_hash(self.root.parent.parent / "manifest.json"),
                      "attempt": str(self.attempt.relative_to(self.root)), "artifacts": _attempt_hashes(self.root, self.attempt)}
        validate_completion(self.root, completion, expected_identity=self.identity)
        write_json(self.root / "completed.json", completion, exclusive=True)
        return completion
