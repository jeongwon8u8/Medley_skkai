"""Small explicit-epoch M1 trainer; no split creation or experiment runner."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import random

import numpy as np
import pandas as pd
import torch

from medley.metrics import regression_metrics
from .clinical_regressor import ClinicalLLMRegressor, moca_huber_loss
from .clinical_serializer import ClinicalTextSerializer
from .clinical_tokenization import ClinicalTokenizer
from .config import CLINICAL_REGRESSION_DEFAULTS
from .data_adapter import SubjectSplit, validate_subject_partition


def seed_training(seed: int) -> None:
    """Call BEFORE model/DataLoader construction; not a promise of GPU bit equality."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer in [0,2**32)")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_optimizer(model: ClinicalLLMRegressor) -> torch.optim.AdamW:
    """Only head and Task 4 LoRA weights; configured decay includes bias/norm."""
    settings = model.backbone.resolved_config.settings["training"]
    head = list(model.head.parameters())
    adapters = []
    for name, parameter in model.llm.named_parameters():
        if parameter.requires_grad:
            if "lora_" not in name or model.backbone.mode == "frozen":
                raise ValueError("Unexpected trainable pretrained parameter")
            adapters.append(parameter)
    if not all(p.requires_grad for p in head):
        raise ValueError("Regression head must remain trainable")
    groups = [{"params": head, "lr": settings["lr_new_modules"], "name": "regression_head"}]
    if adapters:
        groups.append({"params": adapters, "lr": settings["lr_lora"], "name": "lora"})
    return torch.optim.AdamW(groups, weight_decay=settings["weight_decay"])


@dataclass(frozen=True)
class ClinicalEvaluation:
    metrics: dict[str, float | int]
    predictions: pd.DataFrame  # Private patient-level values: never logged automatically.


class ClinicalTrainer:
    """One explicit training epoch / evaluation pass at a time.

    Pass adapter.get_split(seed, fold) and explicit inner train/validation IDs.
    Validation may use only withheld OUTER TRAIN IDs. No inner split is inferred.
    Test evaluation is explicit, never used for model/checkpoint selection here.
    """

    def __init__(self, model: ClinicalLLMRegressor, *, split: SubjectSplit,
                 train_subjects, validation_subjects=()):
        if not isinstance(split, SubjectSplit):
            raise ValueError("A Task 2 fixed SubjectSplit is required")
        validate_subject_partition(split.train_subjects, split.test_subjects)
        if (type(split.seed) is not int or split.seed < 0 or type(split.fold) is not int or split.fold < 1):
            raise ValueError("Invalid fixed split metadata")
        self.train_subjects = self._ids(train_subjects)
        self.validation_subjects = self._ids(validation_subjects, allow_empty=True)
        if (not set(self.train_subjects) <= set(split.train_subjects)
                or not set(self.validation_subjects) <= set(split.train_subjects)
                or set(self.train_subjects) & set(self.validation_subjects)):
            raise ValueError("Training/validation must be disjoint subsets of outer train; test subjects are forbidden")
        self.split = split
        self.model = model
        self.config = model.backbone.resolved_config
        selection = self.config.settings["evaluation"]
        if split.seed not in selection["seeds"] or split.fold not in selection["folds"]:
            raise ValueError("Fixed split seed/fold is outside the configured selection")
        self._ids(split.train_subjects)
        self._ids(split.test_subjects)
        self.serializer = ClinicalTextSerializer(self.config)
        self.tokenizer = ClinicalTokenizer(model.backbone)
        self.optimizer = build_optimizer(model)
        self.device = model.head.weight.device
        options = {**CLINICAL_REGRESSION_DEFAULTS, **self.config.settings.get("clinical_regression", {})}
        self.precision = options["mixed_precision"]
        if self.precision == "bfloat16" and (self.device.type != "cuda" or not torch.cuda.is_bf16_supported()):
            raise ValueError("BF16 autocast requires supported CUDA; choose none for CPU synthetic runs")
        self.epochs_completed = 0
        self.optimizer_steps = 0

    @staticmethod
    def _ids(ids, allow_empty=False):
        if isinstance(ids, (str, bytes)):
            raise ValueError("Subject IDs must be an explicit sequence")
        ids = tuple(ids)
        if ((not ids and not allow_empty) or len(ids) != len(set(ids))
                or any(not isinstance(s, str) or not s.strip() for s in ids)):
            raise ValueError("Subject IDs must be unique nonempty strings")
        return ids

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16) if self.precision == "bfloat16" else nullcontext()

    def _loader_check(self, loader, expected):
        if not expected:
            raise ValueError("No subjects explicitly assigned to this evaluation role")
        ids = getattr(getattr(loader, "dataset", None), "subject_ids", None)
        if ids is not None and (len(ids) != len(expected) or set(ids) != set(expected)):
            raise ValueError("Loader Dataset subjects do not match the explicit partition")

    def _batch(self, batch, expected, seen):
        ids = self._ids(batch["subject_ids"])
        if not set(ids) <= set(expected) or seen & set(ids):
            raise ValueError("Duplicate subject or subject outside the explicit partition")
        clinical = batch["clinical"]
        labels = batch["moca"]
        if (len(clinical) != len(ids) or not isinstance(labels, torch.Tensor)
                or tuple(labels.shape) != (len(ids),) or not labels.is_floating_point()
                or not bool(torch.isfinite(labels).all()) or not bool(((labels >= 0) & (labels <= 30)).all())):
            raise ValueError("Batch requires one finite original-unit MoCA target per subject")
        seen.update(ids)
        texts = self.serializer.serialize_batch(clinical)
        tokens = self.tokenizer(texts).to(self.device)
        # No EEG features/masks/artifact flags are read or passed to M1.
        return ids, tokens, labels.to(self.device, dtype=torch.float32)

    def _forward(self, tokens):
        return self.model(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask,
                          prediction_positions=tokens.prediction_positions)

    def train_epoch(self, loader) -> dict[str, float | int]:
        self._loader_check(loader, self.train_subjects)
        settings = self.config.settings["training"]
        accumulation = settings["gradient_accumulation"]
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        seen, group_samples, group_batches, total_loss, total_samples, steps = set(), 0, 0, 0., 0, 0

        def step(sample_count):
            parameters = [p for group in self.optimizer.param_groups for p in group["params"]]
            for p in parameters:
                if p.grad is not None:
                    p.grad.div_(sample_count)
            torch.nn.utils.clip_grad_norm_(parameters, settings["gradient_clip_norm"], error_if_nonfinite=True)
            self.optimizer.step()
            self.optimizer_steps += 1
            self.optimizer.zero_grad(set_to_none=True)

        try:
            for batch in loader:
                ids, tokens, targets = self._batch(batch, self.train_subjects, seen)
                with self._autocast():
                    predictions = self._forward(tokens)
                    loss = moca_huber_loss(predictions, targets, delta=self.config.settings["loss"]["huber_delta"],
                                           reduction="sum")
                loss.backward()
                n = len(ids)
                total_loss += float(loss.detach())
                total_samples += n
                group_samples += n
                group_batches += 1
                if group_batches == accumulation:
                    step(group_samples)
                    steps += 1
                    group_samples, group_batches = 0, 0
            if seen != set(self.train_subjects):
                raise ValueError("Training loader omitted explicitly assigned subjects")
            if group_batches:
                step(group_samples)
                steps += 1
        except Exception:
            self.optimizer.zero_grad(set_to_none=True)
            raise
        self.epochs_completed += 1
        return {"n": total_samples, "huber": total_loss / total_samples, "optimizer_steps": steps}

    def evaluate(self, loader, *, role="validation") -> ClinicalEvaluation:
        choices = {"validation": self.validation_subjects, "test": self.split.test_subjects,
                   "train": self.train_subjects}
        if role not in choices:
            raise ValueError("Evaluation role must be validation, test or train")
        expected = choices[role]
        self._loader_check(loader, expected)
        was_training = self.model.training
        self.model.eval()
        rows, seen = [], set()
        try:
            with torch.no_grad():
                for batch in loader:
                    ids, tokens, targets = self._batch(batch, expected, seen)
                    subgroups = batch["subgroups"]
                    if len(subgroups) != len(ids) or any(s is not None and not isinstance(s, str) for s in subgroups):
                        raise ValueError("Subgroup metadata must align with subjects")
                    with self._autocast():
                        predictions = self._forward(tokens)
                    for sid, subgroup, y, p in zip(ids, subgroups, targets.cpu().tolist(), predictions.cpu().tolist()):
                        rows.append({"seed": self.split.seed, "fold": self.split.fold, "subject": sid,
                                     "subgroup": subgroup, "model": f"clinical_llm:{self.serializer.feature_set}",
                                     "y": y, "yhat": p, "target": y})
        finally:
            self.model.train(was_training)
        if seen != set(expected):
            raise ValueError("Evaluation loader omitted explicitly assigned subjects")
        frame = pd.DataFrame(rows)
        metrics = regression_metrics(frame.y.to_numpy(), frame.yhat.to_numpy())
        return ClinicalEvaluation({k: metrics[k] for k in ("n", "MAE", "RMSE")}, frame)
