"""Reproduce the legacy C0/C1 clinical and Stack baselines on pinned outer folds."""
from contextlib import redirect_stdout
import copy
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from medley.evaluate import _fit_predict_all, prepare
from multimodal.experiment_cohort import validate_predictions
from multimodal.experiment_protocol import fingerprint
from multimodal.run_state import file_hash, read_json, write_json

LABELS = {
    "clinical:C0": "Clinical-only C0",
    "clinical:C1": "Clinical-only C1",
    "fusion_stack:C0": "Stack C0",
    "fusion_stack:C1": "Stack C1",
}


def legacy_baselines(protocol, cohort, splits, root):
    cfg = copy.deepcopy(protocol.config.medley)
    cfg["models"] = {"eeg": False, "clinical": True, "fusion": ["stack"], "baselines": []}
    identity = {
        "implementation": "medley_cv_predict_c0_c1_stack_v1",
        "outer_sha256": protocol.outer.sha256, "cohort": cohort.cohort_fingerprint,
        "cache": cohort.cache_fingerprint, "clinical": cohort.clinical_fingerprint,
        "config": {key: cfg[key] for key in ("dataset", "encoder", "preprocess", "task", "evaluation", "models")},
    }
    key = fingerprint(identity)
    directory = Path(root) / "legacy_baselines" / key
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    paths = {index: directory / f"seed{index[0]}_fold{index[1]}.csv" for index in splits}
    if any(not path.is_file() for path in paths.values()):
        # The legacy diagnostic may show patient IDs on missing features. The
        # exact coverage check below rejects the drop without printing those IDs.
        with redirect_stdout(StringIO()):
            cohort_frame, features, clinical, _ = prepare(cfg)
        if set(cohort_frame.subject) != set(cohort.subjects) or len(cohort_frame) != cohort.n:
            raise ValueError("Legacy CV cohort differs from the fixed patient split")
        clinical = {name: clinical[name] for name in ("C0", "C1")}
    frames = []
    for (seed, fold), path in paths.items():
        split = splits[seed, fold]
        fresh = not path.is_file()
        if fresh:
            test = cohort_frame.subject.isin(split.test_subjects).to_numpy()
            train = ~test
            predicted = _fit_predict_all(
                cfg, features[train], features[test],
                {name: values[train] for name, values in clinical.items()},
                {name: values[test] for name, values in clinical.items()},
                cohort_frame.noise.to_numpy()[train], cohort_frame.noise.to_numpy()[test],
                cohort_frame.y.to_numpy()[train], {"seed": seed},
            )
            subset = cohort_frame.loc[test, ["subject", "subgroup", "y", "target"]]
            frame = pd.concat([subset.assign(seed=seed, fold=fold, model=model, yhat=values)
                               for model, values in predicted.items()], ignore_index=True)
        else:
            if read_json(path.with_suffix(".json")) != {"baseline_identity": key, "sha256": file_hash(path)}:
                raise ValueError("Legacy baseline Fold checksum or identity mismatch")
            frame = pd.read_csv(path, dtype={"subject": str})
        if set(frame.model) != set(LABELS):
            raise ValueError("Legacy Fold is missing a baseline model")
        for model in LABELS:
            validate_predictions(frame, cohort, split, model=model)
        if fresh:
            with TemporaryDirectory(dir=directory) as temporary:
                staged = Path(temporary) / path.name
                frame.to_csv(staged, index=False)
                write_json(path.with_suffix(".json"), {"baseline_identity": key, "sha256": file_hash(staged)})
                staged.replace(path)
        frames.append(frame.assign(model=frame.model.map(LABELS)))
    return pd.concat(frames, ignore_index=True), identity
