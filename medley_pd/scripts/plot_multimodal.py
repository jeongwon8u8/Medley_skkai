"""Recreate publication figures from exported aggregate CSV files alone."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def records(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def label(row):
    return f"{row['model']} {row['category']}"


def labels(rows):
    names = [label(row) for row in rows]
    return [name if names.count(name) == 1 else f"{name} ({row['execution_id'][:8]})"
            for name, row in zip(names, rows)]


def save(fig, directory, name):
    for extension in ("png", "svg"):
        path = directory / f"{name}.{extension}"
        with path.open("xb") as stream:
            fig.savefig(stream, format=extension, dpi=300, bbox_inches="tight")
    plt.close(fig)


def curves(rows, output):
    groups = {}
    for row in rows:
        groups.setdefault(row["execution_id"], []).append(row)
    names = labels([group[0] for group in groups.values()])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for name, group in zip(names, groups.values()):
        group.sort(key=lambda row: int(row["epoch"]))
        epochs = [int(row["epoch"]) for row in group]
        for ax, field, ylabel in zip(axes, ("train_loss", "validation_mae"),
                                     ("Train Huber loss", "Validation MAE (MoCA points)")):
            line, = ax.plot(epochs, [float(row[field]) for row in group], label=name)
            for row in group:
                if row["best_epoch"] == "1":
                    ax.scatter(int(row["epoch"]), float(row[field]), marker="*", s=100,
                               color=line.get_color(), zorder=3)
                if row["early_stop"] == "1":
                    ax.axvline(int(row["epoch"]), color=line.get_color(), linestyle="--", alpha=.65)
            ax.set(xlabel="Epoch", ylabel=ylabel)
            ax.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle("Learning curves · ★ best validation epoch · dashed line: early stop")
    save(fig, output, "learning_curves")


def comparison(rows, output):
    names = labels(rows)
    fig, axes = plt.subplots(1, 2, figsize=(9, max(3, 1.8 + .5 * len(rows))), layout="constrained")
    for ax, field, title in zip(axes, ("test_mae", "test_rmse"), ("Test MAE", "Test RMSE")):
        bars = ax.barh(names, [float(row[field]) for row in rows], color="#246884")
        ax.bar_label(bars, fmt="%.3f", padding=4)
        ax.set(title=title, xlabel="MoCA points")
        ax.margins(x=.2)
    fig.suptitle(f"Test performance · seed {rows[0]['seed']} · fold {rows[0]['fold']}")
    save(fig, output, "test_performance")


def plot(csv_directory, output):
    output.mkdir(parents=True, exist_ok=True)
    history = csv_directory / "history.csv"
    metrics = csv_directory / "test_metrics.csv"
    if not (history.is_file() or metrics.is_file()):
        raise ValueError("Expected history.csv or test_metrics.csv")
    if history.is_file():
        curves(records(history), output)
    if metrics.is_file():
        comparison(records(metrics), output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.csv_directory
    plot(args.csv_directory, output)


if __name__ == "__main__":
    main()
