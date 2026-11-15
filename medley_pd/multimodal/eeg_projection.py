"""Trainable pooled-EEG to actual LLM-width projection, independent of an LLM."""
from __future__ import annotations

import torch
from torch import nn

from .config import ResolvedConfig, validate_config
from .eeg_pooling import _floating_input, _positive_integer


class EEGToLLMProjector(nn.Module):
    """[B,K,D_p] -> Linear(D_p,H) -> optional LayerNorm(H) -> [B,K,H].

    Existing eeg.projector.layer_norm is honored (default true). No hidden-size
    guess, device move, detach, pretrained loading or clinical fusion occurs.
    """

    def __init__(self, input_dim: int, output_dim: int, *, layer_norm: bool = True):
        super().__init__()
        _positive_integer("input_dim", input_dim)
        _positive_integer("output_dim", output_dim)
        if type(layer_norm) is not bool:
            raise ValueError("Projector layer_norm must be boolean")
        self.input_dim, self.output_dim = input_dim, output_dim
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim) if layer_norm else nn.Identity()

    @classmethod
    def from_config(cls, config: ResolvedConfig, *, hidden_dim: int | None = None) -> EEGToLLMProjector:
        """Pass backbone.hidden_dim, or set an explicit validated output_dim.

        If both are present they must agree. A1 never loads/forwards an LLM.
        """
        validate_config(config.settings)
        eeg = config.settings["eeg"]
        output_dim = eeg["projector"].get("output_dim")
        if hidden_dim is not None:
            _positive_integer("LLM hidden_dim", hidden_dim)
            if output_dim is not None and output_dim != hidden_dim:
                raise ValueError("Configured projector output_dim disagrees with actual LLM hidden_dim")
            output_dim = hidden_dim
        if output_dim is None:
            raise ValueError("Provide actual backbone.hidden_dim or explicitly configure eeg.projector.output_dim")
        return cls(eeg["pooling"]["hidden_dim"], output_dim, layer_norm=eeg["projector"]["layer_norm"])

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        _floating_input(pooled, self.input_dim, self.linear.weight)
        if not bool(torch.isfinite(pooled).all()):
            raise ValueError("Pooled EEG representations must be finite")
        projected = self.norm(self.linear(pooled))
        if not bool(torch.isfinite(projected).all()):
            raise ValueError("Nonfinite EEG projection output")
        return projected
