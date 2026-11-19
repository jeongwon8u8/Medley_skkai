"""GPU Ridge and stacking on the exact M1–M3 inner train/validation/test splits."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

import numpy as np
import pandas as pd

from medley.config import logspace
from medley.evaluate import prepare
from multimodal.experiment_cohort import validate_predictions
from multimodal.experiment_protocol import fingerprint, select_inner_validation
from multimodal.run_state import file_hash, read_json, write_json

from moca_legacy_baselines import LABELS

MODELS = ("EEG-only", *LABELS.values())
_CUDA_SOLVE_LOCK = Lock()


class RidgePath:
    """Ridge with intercept and train-only median imputation / scaling."""

    def __init__(self, x, y, alphas):
        import torch

        self.median = torch.nanquantile(x, 0.5, dim=0).nan_to_num(0.)
        clean = torch.where(x.isnan(), self.median, x)
        self.center = clean.mean(dim=0)
        scale = clean.std(dim=0, correction=0)
        self.scale = torch.where(scale > 0, scale, 1.)
        self.x = (clean - self.center) / self.scale
        self.intercept = y.mean()
        kernel = self.x @ self.x.T
        identity = torch.eye(len(x), device=x.device, dtype=x.dtype)
        systems = kernel.unsqueeze(0) + alphas[:, None, None] * identity
        targets = (y - self.intercept).expand(len(alphas), -1).unsqueeze(-1)
        if x.is_cuda:
            # PyTorch lazily initializes its CUDA linear algebra backend.
            # Only the solve is serialized; each Fold retains its CUDA stream.
            with _CUDA_SOLVE_LOCK:
                solution = torch.linalg.solve(systems, targets)
        else:
            solution = torch.linalg.solve(systems, targets)
        self.weights = solution.squeeze(-1) @ self.x

    def predict(self, x):
        import torch

        clean = torch.where(x.isnan(), self.median, x)
        return (clean - self.center) / self.scale @ self.weights.T + self.intercept


def select_alpha(path, x, y):
    return int((path.predict(x) - y[:, None]).abs().mean(dim=0).argmin().item())


def fit_fold(features, clinical, target, train, validation, test, eeg_alphas, clinical_alphas, folds):
    import torch
    from sklearn.model_selection import KFold

    eeg = RidgePath(features[train], target[train], eeg_alphas)
    eeg_index = select_alpha(eeg, features[validation], target[validation])
    validation_eeg = eeg.predict(features[validation])[:, eeg_index]
    oof = torch.empty_like(target[train])
    for inner_train, inner_test in KFold(folds, shuffle=True, random_state=42).split(train):
        inner = RidgePath(features[train[inner_train]], target[train[inner_train]],
                          eeg_alphas[eeg_index:eeg_index + 1])
        oof[inner_test] = inner.predict(features[train[inner_test]]).squeeze(-1)

    chosen = {"EEG-only": float(eeg_alphas[eeg_index])}
    fitted = {}
    for name, values in clinical.items():
        c_label, s_label = LABELS[f"clinical:{name}"], LABELS[f"fusion_stack:{name}"]
        linear = RidgePath(values[train], target[train], clinical_alphas)
        index = select_alpha(linear, values[validation], target[validation])
        chosen[c_label] = float(clinical_alphas[index])
        stack = RidgePath(torch.column_stack((oof, values[train])), target[train], clinical_alphas)
        index_stack = select_alpha(stack, torch.column_stack((validation_eeg, values[validation])),
                                   target[validation])
        chosen[s_label] = float(clinical_alphas[index_stack])
        fitted[name] = linear, index, stack, index_stack

    # Run test inputs only after all five hyperparameters are selected.
    test_eeg = eeg.predict(features[test])[:, eeg_index]
    predictions = {"EEG-only": test_eeg}
    for name, values in clinical.items():
        linear, index, stack, index_stack = fitted[name]
        predictions[LABELS[f"clinical:{name}"]] = linear.predict(values[test])[:, index]
        predictions[LABELS[f"fusion_stack:{name}"]] = stack.predict(
            torch.column_stack((test_eeg, values[test])),
        )[:, index_stack]
    return {name: values.cpu().numpy() for name, values in predictions.items()}, chosen


def controlled_baselines(protocol, cohort, splits, root, *, device, workers=1):
    cfg = protocol.config.medley
    fraction = protocol.config.settings["training"]["validation_fraction"]
    settings = cfg["evaluation"]
    identity = {
        "implementation": "torch_cuda_ridge_validation_mae_v1",
        "outer_sha256": protocol.outer.sha256, "cohort": cohort.cohort_fingerprint,
        "cache": cohort.cache_fingerprint, "clinical": cohort.clinical_fingerprint,
        "validation_fraction": fraction, "eeg_alphas": settings["eeg_alphas"],
        "clinical_alphas": settings["clinical_alphas"], "stack_inner_folds": settings["stack_inner_folds"],
        "max_segments": settings.get("max_segments"), "device": device,
    }
    key = fingerprint(identity)
    directory = Path(root) / "controlled_gpu_baselines" / key
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    paths = {index: directory / f"seed{index[0]}_fold{index[1]}.csv" for index in splits}
    missing = sum(not path.is_file() for path in paths.values())
    if missing:
        import torch

        gpu = torch.device(device)
        if gpu.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("Controlled baseline fitting requires CUDA")
        active_workers = min(workers, missing)
        print(f"Controlled Ridge/Stack device: {torch.cuda.get_device_name(gpu)} · "
              f"{active_workers} concurrent Fold streams", flush=True)
        with redirect_stdout(StringIO()):
            frame, features, columns, _ = prepare(cfg)
        if set(frame.subject) != set(cohort.subjects) or len(frame) != cohort.n:
            raise ValueError("Controlled baseline cohort differs from the fixed patient split")
        indexes = {subject: i for i, subject in enumerate(frame.subject)}

        def tensor(values):
            return torch.tensor(values, dtype=torch.float64, device=gpu)

        features, target = tensor(features), tensor(frame.y.to_numpy())
        clinical = {name: tensor(columns[name]) for name in ("C0", "C1")}
        eeg_alphas, clinical_alphas = tensor(logspace(settings["eeg_alphas"])), tensor(logspace(settings["clinical_alphas"]))

        input_stream = torch.cuda.current_stream(gpu)

    def process(item):
        (seed, fold), path = item
        split = splits[seed, fold]
        saved = path.with_suffix(".json")
        partition = select_inner_validation(split, fraction=fraction)
        fresh = not path.is_file()
        if fresh:
            train, val, test = [np.asarray([indexes[s] for s in subjects])
                                for subjects in (partition.train_subjects, partition.validation_subjects, partition.test_subjects)]
            stream = torch.cuda.Stream(device=gpu)
            stream.wait_stream(input_stream)
            with torch.cuda.device(gpu), torch.cuda.stream(stream):
                predictions, chosen = fit_fold(features, clinical, target, train, val, test, eeg_alphas,
                                               clinical_alphas, settings["stack_inner_folds"])
            rows = frame.iloc[test][["subject", "subgroup", "y", "target"]]
            result = pd.concat([rows.assign(seed=seed, fold=fold, model=model, yhat=values)
                                for model, values in predictions.items()], ignore_index=True)
        else:
            metadata = read_json(saved)
            if metadata["identity"] != key or metadata["sha256"] != file_hash(path):
                raise ValueError("Controlled baseline Fold checksum or identity mismatch")
            result = pd.read_csv(path, dtype={"subject": str})
        if set(result.model) != set(MODELS):
            raise ValueError("Controlled Fold is missing a baseline model")
        for model in MODELS:
            validate_predictions(result, cohort, split, model=model)
        if fresh:
            with TemporaryDirectory(dir=directory) as temporary:
                staging = Path(temporary) / path.name
                result.to_csv(staging, index=False)
                write_json(saved, {"identity": key, "sha256": file_hash(staging), "alpha": chosen,
                                  "counts": partition.counts, "partition_fingerprint": fingerprint([
                                      partition.train_subjects, partition.validation_subjects, partition.test_subjects])})
                staging.replace(path)
            print(f"Controlled GPU baselines seed {seed} fold {fold} complete", flush=True)
        return result

    if missing and active_workers > 1:
        with ThreadPoolExecutor(max_workers=active_workers) as pool:
            frames = list(pool.map(process, paths.items()))
    else:
        frames = list(map(process, paths.items()))
    return pd.concat(frames, ignore_index=True), identity
