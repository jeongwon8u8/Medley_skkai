"""Independent cached-feature identity adapter and learned-query EEG pooling.

No encoder execution, clinical/label access, temporal truncation or LLM fusion.
"""
from __future__ import annotations

import logging

import torch
from torch import nn

from .config import ResolvedConfig, validate_config

logger = logging.getLogger(__name__)


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

    ON adds a zero-initialized bottleneck residual to frozen cached features.
    """

    def __init__(self, input_dim: int, *, enabled: bool = False,
                 hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        _positive_integer("input_dim", input_dim)
        if type(enabled) is not bool:
            raise ValueError("Feature adapter enabled must be boolean")
        self.enabled = enabled
        self.input_dim = self.output_dim = input_dim
        if enabled:
            _positive_integer("hidden_dim", hidden_dim)
            self.norm = nn.LayerNorm(input_dim)
            self.down = nn.Linear(input_dim, hidden_dim)
            self.activation = nn.GELU()
            self.dropout = nn.Dropout(dropout)
            self.up = nn.Linear(hidden_dim, input_dim)
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)
        logger.info("EEG feature adapter enabled=%s trainable_parameters=%d",
                    enabled, sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, value: torch.Tensor, *, eeg_mask: torch.Tensor | None = None) -> torch.Tensor:
        if not self.enabled:
            return value
        _floating_input(value, self.input_dim, self.down.weight)
        padding = None
        if eeg_mask is not None:
            if (not isinstance(eeg_mask, torch.Tensor) or eeg_mask.dtype != torch.bool
                    or eeg_mask.shape != value.shape[:2] or eeg_mask.device != value.device):
                raise ValueError("Adapter mask must be boolean [B,T] on input device")
            padding = ~eeg_mask.unsqueeze(-1)
            value = value.masked_fill(padding, 0.)
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Nonfinite real adapter input")
        result = value + self.up(self.dropout(self.activation(self.down(self.norm(value)))))
        if padding is not None:
            result = result.masked_fill(padding, 0.)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("Nonfinite adapter output")
        return result

    @classmethod
    def from_config(cls, config: ResolvedConfig) -> EEGFeatureAdapter:
        validate_config(config.settings)
        eeg = config.settings["eeg"]
        if eeg["feature_adapter"]["enabled"] and not config.settings.get("ablation",{}).get("enabled",False):
            raise NotImplementedError("Feature adapter ON requires explicit Phase D opt-in")
        return cls(eeg["feature_dim"], **eeg["feature_adapter"])


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
        if pooling["type"] != "attention":
            raise ValueError("Typed attention factory requires attention; use build_eeg_pooling")
        # Construction stays modular. Do not accidentally bypass an unsupported
        # adapter request simply because the caller builds pooling on its own.
        if eeg["feature_adapter"]["enabled"] and not config.settings.get("ablation",{}).get("enabled",False):
            raise NotImplementedError("Feature adapter ON requires explicit Phase D opt-in")
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


class MaskedReductionPooling(nn.Module):
    """Masked raw-feature mean/max -> Linear(D_e,D_p) -> one EEG vector.

    This has one query-equivalent output, never replicated K tokens. The learned
    projection preserves downstream widths; attention capacity is removed.
    """
    def __init__(self,input_dim,*,hidden_dim=256,kind="mean"):
        super().__init__()
        _positive_integer("input_dim",input_dim);_positive_integer("hidden_dim",hidden_dim)
        if kind not in ("mean","max"): raise ValueError("Unsupported masked reduction")
        self.input_dim,self.hidden_dim,self.num_queries,self.kind=input_dim,hidden_dim,1,kind
        self.input_projection=nn.Linear(input_dim,hidden_dim)

    def forward(self,eeg_features,eeg_mask,*,artifact_flags=None):
        _floating_input(eeg_features,self.input_dim,self.input_projection.weight)
        if (not isinstance(eeg_mask,torch.Tensor) or eeg_mask.dtype!=torch.bool
                or eeg_mask.shape!=eeg_features.shape[:2] or eeg_mask.device!=eeg_features.device
                or not bool(eeg_mask.any(dim=1).all())):
            raise ValueError("Mask requires real segments for every subject")
        if artifact_flags is not None and (not isinstance(artifact_flags,torch.Tensor)
                or artifact_flags.dtype!=torch.bool or artifact_flags.shape!=eeg_mask.shape
                or artifact_flags.device!=eeg_mask.device):
            raise ValueError("Artifact flags are separate boolean metadata")
        clean=eeg_features.masked_fill(~eeg_mask.unsqueeze(-1),0.)
        if not bool(torch.isfinite(clean).all()): raise ValueError("Real EEG features must be finite")
        if self.kind=="mean": reduced=clean.sum(1)/eeg_mask.sum(1,keepdim=True)
        else: reduced=clean.masked_fill(~eeg_mask.unsqueeze(-1),float("-inf")).amax(1)
        output=self.input_projection(reduced).unsqueeze(1)
        if not bool(torch.isfinite(output).all()): raise ValueError("Nonfinite reduction output")
        return output


def build_eeg_pooling(config):
    validate_config(config.settings)
    eeg=config.settings["eeg"];options=eeg["pooling"]
    if options["type"]=="attention": return LearnedQueryAttentionPooling.from_config(config)
    return MaskedReductionPooling(eeg["feature_dim"],hidden_dim=options["hidden_dim"],kind=options["type"])
