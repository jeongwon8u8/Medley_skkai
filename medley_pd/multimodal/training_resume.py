"""Private epoch-boundary training state, separate from inference checkpoints."""
import copy
import os
from pathlib import Path
import pickle
import random
import re
import shutil
import tempfile

import numpy as np
import torch

from .model_registry import RunnerError
from .run_state import file_hash, read_json, validate_completion
from .experiment_manifest import read_manifest
from .experiment_protocol import fingerprint


def continuation_config(document):
    config = copy.deepcopy(document)
    config["settings"]["training"].pop("early_stopping_patience")
    config["settings"]["protocol"].pop("manifest_dir")
    return config


def find_continuation(source_root, manifest, *, exclude_execution=None):
    """Find a compatible legacy epoch under a separately identified source run."""
    current = manifest.to_dict()
    config = continuation_config(current["resolved_config"])
    identity = current["identity"]
    ignored = {"runtime", "config_fingerprint", "output_root"}
    comparable = {key: value for key, value in identity.items() if key not in ignored}
    candidates = []
    for path in Path(source_root).resolve().glob("*/manifest.json"):
        source = read_manifest(path)
        if any(source["identity"][field] != identity[field] for field in ("model", "seed", "fold")):
            continue
        if (continuation_config(source["resolved_config"]) != config
                or {k: v for k, v in source["identity"].items() if k not in ignored} != comparable
                or source["subjects"] != current["subjects"]):
            continue
        for run in (path.parent / "executions").glob("*"):
            if run.name == exclude_execution:
                continue
            marker = run / "completed.json"
            if marker.is_file():
                completion = read_json(marker)
                validate_completion(run, completion, verify_tensors=False)
                attempts = [run / completion["attempt"]]
            else:
                attempts = list((run / "attempts").glob("*"))
            for attempt in attempts:
                for snapshot in (attempt / "resume").glob("epoch-*.pt"):
                    candidates.append((snapshot, run.name, read_json(attempt / "runtime_config.json")))
    if not candidates:
        return None
    candidates.sort(key=lambda item: int(item[0].name.split("-")[1]), reverse=True)
    if len(candidates) > 1 and candidates[0][0].name.split("-")[1] == candidates[1][0].name.split("-")[1]:
        raise RunnerError("resume_source", "Multiple compatible source states at the same latest epoch")
    return candidates[0]


class EpochJournal:
    def __init__(self, runtime, loader, directory, execution_id, *, live_metadata=None):
        self.runtime = runtime
        self.trainer = runtime.trainer
        self.generator = loader.generator
        self.live_metadata = live_metadata
        self.directory = Path(directory)
        self.identity = (execution_id, fingerprint(self.trainer.config.to_dict()))

    def _parameters(self):
        return {name: p for name, p in self.runtime.model.named_parameters() if p.requires_grad}

    def save(self, history, best_epoch):
        epoch = len(history)
        numpy_state = np.random.get_state()
        payload = {
            "version": 1, "identity": self.identity, "history": history, "best_epoch": best_epoch,
            "optimizer_steps": self.trainer.optimizer_steps,
            "weights": {name: p.detach().cpu() for name, p in self._parameters().items()},
            "optimizer": self.trainer.optimizer.state_dict(),
            "python": random.getstate(),
            "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if self.trainer.device.type == "cuda" else None,
            "loader": self.generator.get_state(),
        }
        directory = self.directory / "resume"
        directory.mkdir(exist_ok=True, mode=0o700)
        fd, name = tempfile.mkstemp(dir=directory)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "wb") as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            path = directory / f"epoch-{epoch:04d}-{file_hash(temporary)}.pt"
            os.replace(temporary, path)
            fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            for previous in directory.glob("epoch-*.pt"):
                if previous != path:
                    previous.unlink()
        finally:
            temporary.unlink(missing_ok=True)

        if self.live_metadata is not None:
            from .live_artifacts import publish_curves

            publish_curves(self.directory / "live", history, best_epoch,
                           training=self.trainer.config.settings["training"], **self.live_metadata)

    def restore(self, snapshot, source=None):
        snapshot = Path(snapshot)
        match = re.fullmatch(r"epoch-(\d+)-([a-f0-9]{64})\.pt", snapshot.name)
        try:
            if (snapshot.resolve() != snapshot or not match or file_hash(snapshot) != match[2]):
                raise ValueError()
            state = torch.load(snapshot, map_location="cpu", weights_only=True)
            history = state["history"]
            expected = self.identity
            if source is not None:
                execution_id, old_config = source
                if continuation_config(old_config) != continuation_config(self.trainer.config.to_dict()):
                    raise ValueError()
                expected = (execution_id, fingerprint(old_config))
            if (state["version"] != 1 or state["identity"] != expected
                    or len(history) != int(match[1]) or not 1 <= state["best_epoch"] <= len(history)):
                raise ValueError()
            name = f"epoch-{state['best_epoch']:04d}"
            best = snapshot.parents[1] / "checkpoints" / name
            if best.resolve() != best or not best.is_dir() or any(p.is_symlink() for p in best.iterdir()):
                raise ValueError()
            # Validate the inference checkpoint against the reconstructed backbone.
            destination = self.directory / "checkpoints" / name
            if source is None:
                metadata = self.runtime.restore(best)
                shutil.copytree(best, destination)
            else:
                settings = self.trainer.config.settings
                old_settings = old_config["settings"]
                fields = (("training", "early_stopping_patience"), ("protocol", "manifest_dir"))
                original = [settings[group][key] for group, key in fields]
                try:
                    for group, key in fields:
                        settings[group][key] = old_settings[group][key]
                    metadata = self.runtime.restore(best)
                finally:
                    for (group, key), value in zip(fields, original):
                        settings[group][key] = value
                self.runtime.checkpoints.save_inference_checkpoint(
                    self.runtime.model, destination,
                    epochs_completed=metadata["epochs_completed"], optimizer_steps=metadata["optimizer_steps"],
                )
            if metadata["epochs_completed"] != state["best_epoch"]:
                raise ValueError()
            parameters = self._parameters()
            if state["weights"].keys() != parameters.keys():
                raise ValueError()
            with torch.no_grad():
                for name, p in parameters.items():
                    value = state["weights"][name]
                    if value.shape != p.shape or value.dtype != p.dtype or not torch.isfinite(value).all():
                        raise ValueError()
                    p.copy_(value)
            self.trainer.optimizer.load_state_dict(state["optimizer"])
            self.trainer.epochs_completed = len(history)
            self.trainer.optimizer_steps = state["optimizer_steps"]
            random.setstate(state["python"])
            name, keys, *rest = state["numpy"]
            np.random.set_state((name, np.asarray(keys, dtype=np.uint32), *rest))
            torch.random.set_rng_state(state["torch"])
            if state["cuda"] is not None:
                if len(state["cuda"]) != torch.cuda.device_count():
                    raise ValueError()
                torch.cuda.set_rng_state_all(state["cuda"])
            self.generator.set_state(state["loader"])
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, EOFError, pickle.UnpicklingError) as exc:
            raise RunnerError("resume_state", "Epoch state or best checkpoint is invalid or incompatible") from exc
        return history, state["best_epoch"]
