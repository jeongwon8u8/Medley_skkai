from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_multimodal_oof import MODELS, aligned_oof, completed_paths, figures, summarize


def test_coverage_blocks_missing_folds_and_duplicate_patients(tmp_path):
    with pytest.raises(ValueError, match="Missing completed model/seed/fold"):
        completed_paths(tmp_path, {(0, 1): ("a",), (0, 2): ("b",)})
    assignments = {(0, 1): ("a",), (0, 2): ("b",)}
    frame = pd.DataFrame([
        {"model": model, "seed": seed, "fold": fold, "subject": subject, "target": 20}
        for model in MODELS for (seed, fold), patients in assignments.items() for subject in patients
    ])
    assert len(aligned_oof(frame, [0], assignments)) == len(frame)
    with pytest.raises(ValueError, match="coverage"):
        aligned_oof(frame.iloc[1:], [0], assignments)
    with pytest.raises(ValueError, match="coverage"):
        aligned_oof(pd.concat([frame, frame.iloc[:1]]), [0], assignments)


def test_paired_metrics_use_patients_and_average_predictions_before_scoring(tmp_path):
    assignments = {(seed, 1): ("a", "b", "c") for seed in (0, 1)}
    frame = pd.DataFrame([
        {"model": model, "seed": seed, "fold": 1, "subject": subject,
         "target": target, "yhat": target + bias + (1 if seed == 0 else -1)}
        for (seed, _), subjects in assignments.items()
        for subject, target in zip(subjects, (10, 20, 30))
        for bias, model in enumerate(MODELS)
    ])
    frame = aligned_oof(frame, [0, 1], assignments)
    metrics, paired, mean_predictions = summarize(frame, [0, 1])
    assert metrics.query("scope == 'seed_0' and model == 'EEG-only'").MAE.item() == 1
    assert metrics.query("scope == 'seed_mean' and model == 'EEG-only'").MAE.item() == 0
    row = paired.query("scope == 'seed_mean' and reference == 'EEG-only' and candidate == 'M2'")
    assert (row.delta_MAE.item(), row.delta_RMSE.item(), row.delta_Spearman.item()) == (2, 2, 0)
    assert len(mean_predictions) == 3 * len(MODELS)
    figures(paired, tmp_path, "seed_mean")
    assert (tmp_path / "paired_comparison.png").is_file()
    assert (tmp_path / "paired_comparison.svg").is_file()


def test_baseline_fit_then_reuses_audited_fold(adapter_files, tmp_path, monkeypatch):
    import compare_multimodal_oof as comparison
    from types import SimpleNamespace
    from test_multimodal_experiment_runner import execution_inputs

    _, _, protocol = execution_inputs(adapter_files, tmp_path)
    cohort = protocol.validate_cohort()
    splits = {(0, 1): protocol.get_outer_split(seed=0, fold=1)}
    args = SimpleNamespace(eeg_predictions=None)
    first, identity = comparison.baseline(args, protocol, cohort, splits, tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed EEG-only fold was refitted")

    monkeypatch.setattr(comparison, "eeg_model", forbidden)
    second, second_identity = comparison.baseline(args, protocol, cohort, splits, tmp_path)
    assert identity == second_identity
    pd.testing.assert_frame_equal(first.fillna(""), second.fillna(""), check_dtype=False)


from test_multimodal_data_adapter import files as adapter_files


def test_full_budget_selection_rejects_early_stops_and_uses_legacy_completed_folds(tmp_path):
    import json

    def publish(root, model, epochs):
        marker = tmp_path / root / model / "executions" / model / "completed.json"
        attempt = marker.parent / "attempts" / "case"
        attempt.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"model": model, "seed": 0, "fold": 1, "attempt": "attempts/case"}))
        (attempt / "history.json").write_text(json.dumps([{}] * epochs))
        return marker

    m1 = publish("multimodal_runs_30epochs", "M1", 30)
    m2 = publish("multimodal_runs", "M2", 30)
    publish("multimodal_runs", "M3", 18)
    with pytest.raises(ValueError, match="M3"):
        completed_paths(tmp_path, {(0, 1): ("a",)}, full_epochs=True)
    m3 = publish("multimodal_runs_30epochs", "M3", 30)
    assert set(completed_paths(tmp_path, {(0, 1): ("a",)}, full_epochs=True).values()) == {m1, m2, m3}


def test_legacy_c0_c1_and_stack_reuse_same_outer_test_patients(adapter_files, tmp_path, monkeypatch):
    from test_multimodal_experiment_runner import execution_inputs
    import moca_legacy_baselines as legacy

    _, _, protocol = execution_inputs(adapter_files, tmp_path)
    cohort = protocol.validate_cohort()
    splits = {(0, 1): protocol.get_outer_split(seed=0, fold=1)}
    result, identity = legacy.legacy_baselines(protocol, cohort, splits, tmp_path)
    assert set(result.model) == set(legacy.LABELS.values())
    assert all(set(rows.subject) == set(splits[0, 1].test_subjects) for _, rows in result.groupby("model"))

    def forbidden(*args, **kwargs):
        raise AssertionError("Verified baseline was refitted")

    monkeypatch.setattr(legacy, "_fit_predict_all", forbidden)
    cached, restored_identity = legacy.legacy_baselines(protocol, cohort, splits, tmp_path)
    assert identity == restored_identity
    pd.testing.assert_frame_equal(result.fillna(""), cached.fillna(""), check_dtype=False)


def test_all_model_reports_include_oof_and_curves(tmp_path):
    import json
    from moca_legacy_baselines import LABELS
    from moca_pipeline_artifacts import publish

    models = ("EEG-only", *LABELS.values(), "M1", "M2", "M3")
    assignments = {(0, 1): ("a", "b")}
    rows = [{"model": model, "seed": 0, "fold": 1, "subject": subject, "target": value,
             "yhat": value + number} for number, model in enumerate(models)
            for subject, value in zip(assignments[0, 1], (20, 24))]
    frame = aligned_oof(pd.DataFrame(rows), [0], assignments, models=models)
    metrics, pairs, _ = summarize(frame, [0], models=models)
    assert len(metrics) == 8 and len(pairs) == 28
    paths = {}
    for model in ("M1", "M2", "M3"):
        run = tmp_path / model
        attempt = run / "attempts" / "first"
        attempt.mkdir(parents=True)
        marker = run / "completed.json"
        marker.write_text(json.dumps({"attempt": "attempts/first"}))
        (attempt / "history.json").write_text(json.dumps([
            {"epoch": 1, "training": {"huber": 1}, "validation": {"MAE": 2}},
        ]))
        (attempt / "selection.json").write_text('{"best_epoch":1}')
        paths[model, (0, 1)] = marker
    publish(tmp_path, frame, metrics, paths, "seed_0")
    assert len(list((tmp_path / "model_oof").glob("*.csv"))) == 8
    assert len(list((tmp_path / "training_curves").glob("*.svg"))) == 3
    assert (tmp_path / "model_performance.png").is_file()
