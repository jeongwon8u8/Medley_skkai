"""M1 plain-text token policy: no chat template, new tokens, EOS or truncation."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .config import CLINICAL_REGRESSION_DEFAULTS
from .llm_backbone import LLMBackbone


@dataclass(frozen=True)
class ClinicalTokens:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prediction_positions: torch.Tensor

    def to(self, device: torch.device) -> ClinicalTokens:
        return ClinicalTokens(*(value.to(device) for value in
                                (self.input_ids, self.attention_mask, self.prediction_positions)))


class ClinicalTokenizer:
    """Append '\n\nPredicted MoCA:'; select the final actual suffix token.

    Tokenize the entire prompt together, without automatic BOS/EOS insertion.
    Empty clinical text is retained and receives the suffix alone. No text is
    truncated; both clinical-only and total token budgets are enforced.
    """

    def __init__(self, backbone: LLMBackbone):
        self.tokenizer = backbone.tokenizer
        settings = backbone.resolved_config.settings
        self.suffix = settings.get("clinical_regression", CLINICAL_REGRESSION_DEFAULTS)["prediction_suffix"]
        self.text_budget = settings["llm"]["max_text_tokens"]
        self.budget = settings["llm"]["max_sequence_tokens"]
        limit = getattr(backbone.model.config, "max_position_embeddings", None)
        if limit is not None:
            self.budget = min(self.budget, limit)
        self.side = self.tokenizer.padding_side
        self.pad_id = self.tokenizer.pad_token_id
        if self.side not in ("left", "right") or self.pad_id is None:
            raise ValueError("Use the Task 4 configured tokenizer with an explicit padding policy")
        suffix_ids = self._encode(self.suffix)
        if not suffix_ids or len(suffix_ids) > self.budget:
            raise ValueError("Complete prediction suffix cannot fit the token budget")

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False, truncation=False)

    def __call__(self, texts: Sequence[str]) -> ClinicalTokens:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence) or not texts:
            raise ValueError("Expected a nonempty sequence of clinical texts")
        encoded = []
        for text in texts:
            if not isinstance(text, str):
                raise ValueError("Clinical texts must be strings")
            if len(self._encode(text)) > self.text_budget:
                raise ValueError("Clinical text exceeds max_text_tokens; no truncation is permitted")
            prompt = text + "\n\n" + self.suffix if text else self.suffix
            ids = self._encode(prompt)
            if not ids or len(ids) > self.budget:
                raise ValueError("Clinical text plus complete suffix exceeds context budget")
            encoded.append(ids)
        width = max(map(len, encoded))
        ids = torch.full((len(encoded), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        positions = []
        for row, values in enumerate(encoded):
            start = width - len(values) if self.side == "left" else 0
            ids[row, start:start + len(values)] = torch.tensor(values, dtype=torch.long)
            mask[row, start:start + len(values)] = 1
            positions.append(start + len(values) - 1)
        return ClinicalTokens(ids, mask, torch.tensor(positions, dtype=torch.long))
