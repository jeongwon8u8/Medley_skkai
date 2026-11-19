"""Private, atomically refreshed epoch figures and test case CSVs."""
import csv
import os
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory


PREDICTION_COLUMNS = ("seed", "fold", "subject", "subgroup", "model", "y", "yhat", "target")


def write_cases(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with NamedTemporaryFile(mode="w", newline="", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            writer = csv.DictWriter(stream, fieldnames=PREDICTION_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def publish_curves(directory, history, best_epoch, *, model, seed, fold, execution_id, training):
    from scripts.plot_multimodal import curves

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    category = "Clinical-only" if model == "M1" else "Multimodal"
    rows = [{
        "model": model, "category": category, "seed": seed, "fold": fold, "execution_id": execution_id,
        "epoch": row["epoch"], "train_loss": row["training"]["huber"],
        "validation_mae": row["validation"]["MAE"], "validation_rmse": row["validation"]["RMSE"],
        "optimizer_steps": row["training"]["optimizer_steps"], "bad_epochs": row["bad_epochs"],
        "best_epoch": str(int(row["epoch"] == best_epoch)),
        "early_stop": str(int(row["bad_epochs"] >= training["early_stopping_patience"])),
        "max_epochs": training["max_epochs"],
    } for row in history]
    with TemporaryDirectory(dir=directory) as temporary:
        staging = Path(temporary)
        with (staging / "history.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        curves(rows, staging)
        for path in staging.iterdir():
            path.replace(directory / path.name)
