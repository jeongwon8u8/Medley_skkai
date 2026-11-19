"""Final per-model CSVs, aggregate curves, and performance figures."""
from pathlib import Path

import pandas as pd

from multimodal.run_state import read_json
from plot_multimodal import plt, save


def publish(directory, frame, metrics, paths, primary, *, controlled_gpu=False):
    oof = directory / "model_oof"
    oof.mkdir(mode=0o700)
    for model, rows in frame.groupby("model", sort=False):
        rows.to_csv(oof / f"{model.replace(' ', '_')}.csv", index=False)
    selected = metrics[metrics.scope == primary]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), layout="constrained")
    for axis, metric in zip(axes, ("MAE", "RMSE", "Spearman")):
        bars = axis.barh(selected.model, selected[metric], color="#246884")
        axis.bar_label(bars, fmt="%.3f", padding=4)
        axis.set_title(metric)
        axis.margins(x=.22)
    fig.suptitle(f"PD-EEG · patient OOF · {primary}")
    save(fig, directory, "model_performance")

    curves = directory / "training_curves"
    curves.mkdir(mode=0o700)
    for model in ("M1", "M2", "M3"):
        rows = []
        for (name, (seed, fold)), marker in paths.items():
            if name != model:
                continue
            attempt = marker.parent / read_json(marker)["attempt"]
            history = read_json(attempt / "history.json")
            best = read_json(attempt / "selection.json")["best_epoch"]
            rows.extend({"seed": seed, "fold": fold, "epoch": row["epoch"],
                         "train_loss": row["training"]["huber"],
                         "validation_mae": row["validation"]["MAE"],
                         "best_epoch": row["epoch"] == best} for row in history)
        history = pd.DataFrame(rows)
        history.to_csv(curves / f"{model}_history.csv", index=False)
        avg = history.groupby("epoch")[["train_loss", "validation_mae"]].agg(["mean", "std"])
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
        for axis, field, label in zip(axes, ("train_loss", "validation_mae"), ("Train Huber", "Validation MAE")):
            mean, sd = avg[field]["mean"], avg[field]["std"].fillna(0)
            axis.plot(avg.index, mean)
            axis.fill_between(avg.index, mean - sd, mean + sd, alpha=.2)
            axis.set(xlabel="Epoch", ylabel=label)
        fig.suptitle(f"{model} · fold mean ± SD")
        save(fig, curves, model)
    protocol = Path(__file__).with_name("moca_pipeline_protocol.md").read_text()
    if controlled_gpu:
        protocol += Path(__file__).with_name("moca_controlled_protocol.md").read_text()
    scores = "\n".join(f"- {row.model} · MAE {row.MAE:.4f} · RMSE {row.RMSE:.4f} · Spearman {row.Spearman:.4f}"
                       for row in selected.itertuples())
    (directory / "protocol_summary.md").write_text(f"{protocol}\n\n## Results ({primary})\n\n{scores}\n")
