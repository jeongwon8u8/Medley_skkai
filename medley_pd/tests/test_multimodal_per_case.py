import gzip
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import export_per_case


def test_recorded_per_case_is_plain_csv_and_idempotent(tmp_path, monkeypatch):
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    recorded = b"subject,y,yhat\nsynthetic,20,21\n"
    (attempt / "predictions.csv.gz").write_bytes(gzip.compress(recorded))
    (attempt / "metrics.json").write_text('{"n":1,"MAE":1,"RMSE":1}')
    marker = {"model": "M1", "seed": 0, "fold": 1, "execution_id": "synthetic"}
    monkeypatch.setattr(export_per_case, "APP", tmp_path)
    monkeypatch.setattr(export_per_case, "completed", lambda _: (marker, {}, {}, attempt))
    path = export_per_case.export_case(attempt)
    assert path.read_bytes() == recorded
    assert export_per_case.export_case(attempt) == path
    assert not (attempt / "per_case.csv").exists()


def test_live_csv_replaces_partial_batches(tmp_path):
    import csv
    from multimodal.live_artifacts import write_cases

    path = tmp_path / "per_case.csv"
    first = dict(seed=0, fold=1, subject="a", subgroup="", model="M1", y=20, yhat=21, target=20)
    second = dict(first, subject="b")
    for rows in ([], [first], [first, second]):
        write_cases(path, rows)
        with path.open() as stream:
            assert [row["subject"] for row in csv.DictReader(stream)] == [row["subject"] for row in rows]


def test_missing_completed_case_replays_and_keeps_original(tmp_path, monkeypatch):
    import pandas as pd
    from types import SimpleNamespace

    attempt = tmp_path / "attempt"
    attempt.mkdir()
    (attempt / "metrics.json").write_text('{"n":1,"MAE":1,"RMSE":1}')
    (attempt / "selection.json").write_text('{"checkpoint":"checkpoints/epoch-0001"}')
    marker = {"model": "M1", "seed": 0, "fold": 1, "execution_id": "synthetic"}
    monkeypatch.setattr(export_per_case, "APP", tmp_path)
    monkeypatch.setattr(export_per_case, "completed", lambda _: (marker, {}, {}, attempt))
    calls = []

    def replay(*args, case_path):
        calls.append(case_path)
        from multimodal.live_artifacts import PREDICTION_COLUMNS, write_cases
        rows = [dict(zip(PREDICTION_COLUMNS, (0, 1, "test", None, "M1", 20, 21, 20)))]
        write_cases(case_path, rows)
        return SimpleNamespace(metrics={"n": 1, "MAE": 1, "RMSE": 1}, predictions=pd.DataFrame(rows))

    monkeypatch.setattr(export_per_case, "replay", replay)
    output = export_per_case.export_case(attempt, replay_if_missing=True)
    assert output.is_file() and calls[0].is_file()
    assert export_per_case.export_case(attempt, replay_if_missing=True) == output
    assert len(calls) == 1
    assert not (attempt / "per_case.csv").exists()
