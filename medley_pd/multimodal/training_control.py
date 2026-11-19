"""Validation-only orchestration around existing explicit-epoch trainers."""
from __future__ import annotations

import math
from numbers import Real

from .model_registry import RunnerError
from .progress import progress_bar


def train_epoch_with_validation(runtime, train_loader, validation_loader):
    """No test loader is accepted by the training/selection boundary."""
    training = runtime.trainer.train_epoch(train_loader)
    validation = runtime.trainer.evaluate(validation_loader, role="validation")
    metrics = validation.metrics
    if (metrics["n"] != len(runtime.trainer.validation_subjects)
            or any(isinstance(metrics[key], bool) or not isinstance(metrics[key], Real)
                   or not math.isfinite(metrics[key]) or metrics[key] < 0 for key in ("MAE", "RMSE"))):
        raise RunnerError("validation_metrics", "Validation MAE/RMSE must be finite nonnegative subject-level values")
    return {"training": dict(training), "validation": dict(metrics)}


class EarlyStopping:
    """One-based epochs. Equality is not improvement; first finite score is best."""

    def __init__(self, *, patience=5, min_delta=0.):
        if (type(patience) is not int or patience < 1 or type(min_delta) not in (int, float)
                or not math.isfinite(min_delta) or min_delta < 0):
            raise RunnerError("early_stopping", "Patience must be positive and min_delta finite/nonnegative")
        self.patience, self.min_delta = patience, min_delta
        self.best, self.best_epoch, self.bad_epochs, self.epochs = None, None, 0, 0

    def observe(self, value):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise RunnerError("validation_metrics", "Invalid validation MAE; run cannot select a checkpoint")
        self.epochs += 1
        improved = self.best is None or value < self.best - self.min_delta
        if improved:
            self.best, self.best_epoch, self.bad_epochs = float(value), self.epochs, 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience


def verify_restoration(runtime, checkpoint, *, best_epoch):
    """Tensor equality verifies the actual restored modules, without test access."""
    from safetensors.torch import load_file
    import torch
    metadata = runtime.restore(checkpoint)
    if metadata.get("epochs_completed") != best_epoch:
        raise RunnerError("checkpoint_restore", "Restored checkpoint progress differs from selected best epoch")
    tensors = load_file(str(checkpoint / "trainable.safetensors"), device="cpu")
    parameters = dict(runtime.model.named_parameters())
    if not tensors or any(name not in parameters or not torch.equal(parameters[name].detach().cpu(), value)
                          for name, value in tensors.items()):
        raise RunnerError("checkpoint_restore", "Selected checkpoint did not restore the intended model weights")
    return metadata


def train_select_restore(runtime, train_loader, validation_loader, directory, *, journal=None, snapshot=None, source=None):
    """Only validation drives selection. Test is intentionally absent from this API."""
    from pathlib import Path
    settings = runtime.trainer.config.settings
    stopping = EarlyStopping(patience=settings["training"]["early_stopping_patience"],
                             min_delta=settings["runner"]["min_delta"])
    history, best_checkpoint = [], None
    max_epochs = settings["training"]["max_epochs"]
    stop = False
    if snapshot is not None:
        history, saved_best = journal.restore(snapshot, source=source)
        for epoch, row in enumerate(history, 1):
            if stop:
                raise RunnerError("resume_state", "Saved history extends beyond early stopping")
            improved, stop = stopping.observe(row["validation"]["MAE"])
            if (row["epoch"], row["improved"], row["bad_epochs"]) != (epoch, improved, stopping.bad_epochs):
                raise RunnerError("resume_state", "Saved history differs from validation selection")
        if stopping.best_epoch != saved_best or len(history) > max_epochs:
            raise RunnerError("resume_state", "Saved history differs from the configured epoch budget")
        best_checkpoint = Path(directory) / "checkpoints" / f"epoch-{saved_best:04d}"
        if source is not None:
            journal.save(history, saved_best)
    with progress_bar("Epochs", total=max_epochs, unit="epoch") as bar:
        bar.update(len(history))
        for epoch in range(len(history) + 1, max_epochs + 1):
            if stop:
                break
            bar.set_description_str(f"Epochs ({epoch}/{max_epochs})")
            record = train_epoch_with_validation(runtime, train_loader, validation_loader)
            improved, stop = stopping.observe(record["validation"]["MAE"])
            history.append({"epoch": epoch, **record, "improved": improved, "bad_epochs": stopping.bad_epochs})
            if improved:
                best_checkpoint = Path(directory) / "checkpoints" / f"epoch-{epoch:04d}"
                runtime.save(best_checkpoint)
            if journal is not None:
                journal.save(history, stopping.best_epoch)
            if not bar.disable:
                rates = "/".join(f"{group['lr']:.3g}" for group in runtime.trainer.optimizer.param_groups)
                bar.set_postfix(train_loss=f"{record['training']['huber']:.4f}", lr=rates,
                                optimizer_steps=runtime.trainer.optimizer_steps, refresh=False)
            bar.update()
            if stop:
                bar.set_description_str("Epochs (early stop)", refresh=False)
                break
    if best_checkpoint is None:
        raise RunnerError("checkpoint_selection", "No valid validation checkpoint was selected")
    verify_restoration(runtime, best_checkpoint, best_epoch=stopping.best_epoch)
    return {"best_epoch": stopping.best_epoch, "validation_MAE": stopping.best,
            "epochs_completed": len(history), "history": history, "checkpoint": best_checkpoint}
