"""Independent cached-feature identity adapter and learned-query EEG pooling.

No encoder execution, clinical/label access, temporal truncation or LLM fusion.
"""
from __future__ import annotations

import torch
from torch import nn

from .config import ResolvedConfig, validate_config


def _positive_integer(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _floating_input(value: torch.Tensor, width: int, weight: torch.Tensor) -> None:
    if (not isinstance(value, torch.Tensor) or value.ndim != 3 or min(value.shape) == 0
            or value.shape[-1] != width or not value.is_floating_point()):
        raise ValueError("Expected nonempty floating features [B,T,D] with the configured width")
    if value.device != weight.device:
        raise ValueError("Features and module parameters must be on the same device; no implicit transfer")
    if value.dtype != weight.dtype and not torch.is_autocast_enabled(value.device.type):
        raise ValueError("Feature/module dtypes differ; explicitly cast or use supported autocast")


def seed_eeg_modules(config: ResolvedConfig) -> int:
    """Explicitly seed torch BEFORE constructing adapter/pool/projector.

    Initialization consumes the caller's RNG in construction order. Never reset
    the seed implicitly inside each component. GPU bitwise equality is not claimed.
    """
    validate_config(config.settings)
    seed = config.settings["eeg"].get("seed", 0)
    torch.manual_seed(seed)
    return seed


class EEGFeatureAdapter(nn.Identity):
    """OFF is exact identity, including tensor identity and gradient connection.

    ON is deliberately unsupported until Phase D; existing reserved hidden_dim
    and dropout options do not silently activate an MLP.
    """

    def __init__(self, input_dim: int, *, enabled: bool = False):
        super().__init__()
        _positive_integer("input_dim", input_dim)
        if type(enabled) is not bool:
            raise ValueError("Feature adapter enabled must be boolean")
        if enabled:
            raise NotImplementedError("EEG feature adapter ON is deferred to Phase D; use enabled=false")
        self.input_dim = self.output_dim = input_dim

    @classmethod
    def from_config(cls, config: ResolvedConfig) -> EEGFeatureAdapter:
        validate_config(config.settings)
        eeg = config.settings["eeg"]
        return cls(eeg["feature_dim"], enabled=eeg["feature_adapter"]["enabled"])


class LearnedQueryAttentionPooling(nn.Module):
    """[B,T,D_e] + boolean existence mask [B,T] -> [B,K,D_p].

    Task 3 True=real becomes PyTorch key_padding_mask True=ignore via inversion.
    Artifact flags are optional validated metadata, never an exclusion/weight.
    No positional encoding: sequence permutations preserving mask are invariant.
    """

    def __init__(self, input_dim: int, *, num_queries: int = 4, hidden_dim: int = 256,
                 num_heads: int = 4):
        super().__init__()
        for name, value in (("input_dim", input_dim), ("num_queries", num_queries),
                            ("hidden_dim", hidden_dim), ("num_heads", num_heads)):
            _positive_integer(name, value)
        if hidden_dim % num_heads:
            raise ValueError("Pooling hidden_dim must be divisible by num_heads")
        self.input_dim, self.num_queries, self.hidden_dim = input_dim, num_queries, hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.queries = nn.Parameter(torch.empty(num_queries, hidden_dim))
        nn.init.normal_(self.queries, std=hidden_dim ** -0.5)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0., batch_first=True)

    @classmethod
    def from_config(cls, config: ResolvedConfig) -> LearnedQueryAttentionPooling:
        validate_config(config.settings)
        eeg = config.settings["eeg"]
        pooling = eeg["pooling"]
        # Construction stays modular. Do not accidentally bypass an unsupported
        # adapter request simply because the caller builds pooling on its own.
        if eeg["feature_adapter"]["enabled"]:
            raise NotImplementedError("EEG feature adapter ON is deferred to Phase D")
        return cls(eeg["feature_dim"], num_queries=pooling["num_queries"],
                   hidden_dim=pooling["hidden_dim"], num_heads=pooling["num_heads"])

    def forward(self, eeg_features: torch.Tensor, eeg_mask: torch.Tensor, *,
                artifact_flags: torch.Tensor | None = None) -> torch.Tensor:
        _floating_input(eeg_features, self.input_dim, self.input_projection.weight)
        if (not isinstance(eeg_mask, torch.Tensor) or eeg_mask.dtype != torch.bool
                or eeg_mask.shape != eeg_features.shape[:2] or eeg_mask.device != eeg_features.device):
            raise ValueError("eeg_mask must be boolean [B,T] on the feature device; True means real segment")
        if not bool(eeg_mask.any(dim=1).all()):
            raise ValueError("Every subject must have at least one real EEG segment; all-masked rows are invalid")
        if artifact_flags is not None and (
            not isinstance(artifact_flags, torch.Tensor) or artifact_flags.dtype != torch.bool
            or artifact_flags.shape != eeg_mask.shape or artifact_flags.device != eeg_mask.device
        ):
            raise ValueError("artifact_flags must be separate boolean [B,T] metadata on the feature device")
        # Sanitize BEFORE any learned operation: masked NaN/Inf/huge values must
        # neither influence attention nor poison backward as 0*Inf/NaN.
        clean = eeg_features.masked_fill(~eeg_mask.unsqueeze(-1), 0.)
        if not bool(torch.isfinite(clean).all()):
            raise ValueError("Real EEG features must be finite")
        projected = self.input_projection(clean)
        queries = self.queries.unsqueeze(0).expand(clean.shape[0], -1, -1)
        pooled, _ = self.attention(queries, projected, projected,
                                   key_padding_mask=~eeg_mask, need_weights=False)
        if not bool(torch.isfinite(pooled).all()):
            raise ValueError("Nonfinite attention pooling output")
        return pooled
