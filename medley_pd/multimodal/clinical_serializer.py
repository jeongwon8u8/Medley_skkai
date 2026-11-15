"""Versioned, deterministic predictor-only English text; no model or tokenizer.

Numeric sex codes require an explicitly verified mapping. Neutral measurement
labels deliberately make no unverified unit or phenotype interpretation.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from types import MappingProxyType

import numpy as np

from .config import CLINICAL_TEXT_DEFAULTS, ResolvedConfig, validate_config
from .data_adapter import APPROVED_PREDICTORS


FIELD_ORDER = ("age", "sex", "updrs_part_iii", "disease_duration", "ledd", "pigd_score", "td_score")
FEATURE_SETS = MappingProxyType({"C0": FIELD_ORDER[:2], "C1": FIELD_ORDER[:5], "C2": FIELD_ORDER})
TEMPLATES = MappingProxyType({
    "age": "The recorded age value is {}.",
    "sex": "The patient's recorded sex is {}.",
    "updrs_part_iii": "The recorded UPDRS Part III score is {}.",
    "disease_duration": "The recorded disease duration value is {}.",
    "ledd": "The recorded LEDD value is {}.",
    "pigd_score": "The recorded PIGD score is {}.",
    "td_score": "The recorded TD score is {}.",
})


class ClinicalSerializationError(ValueError):
    """Invalid predictor structure/value; messages never include input values."""


def _numeric(value: object) -> Decimal:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ClinicalSerializationError("Clinical measurements must be numeric scalars or None")
    # NumPy's own shortest string avoids widening float32 artifacts to float64.
    number = Decimal(str(value))
    if not number.is_finite():
        raise ClinicalSerializationError("Nonfinite clinical measurements are invalid; use None for missing")
    return number


def _format_number(value: object, digits: int) -> str:
    number = _numeric(value)
    if not isinstance(value, (int, np.integer)):
        # Do not inherit the caller's traps/exponent limits or rounding policy.
        with localcontext(Context(prec=digits, rounding=ROUND_HALF_EVEN,
                                  Emin=-999999, Emax=999999, traps=[])):
            number = +number
    if number.is_zero():
        return "0"
    # Fixed notation is deterministic even for very small/large measurements.
    text = format(number, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


class ClinicalTextSerializer:
    """Consume only Task 2 record.clinical or Task 3 batch['clinical'].

    Every selected key must be present (None denotes missing). Extra approved
    keys require explicit ignore policy; all other keys always fail closed.
    All-missing records yield an intentional empty string, retaining batch slots.
    Constructor copies options, so later config mutations cannot change output.
    """

    def __init__(self, config: ResolvedConfig):
        if not isinstance(config, ResolvedConfig):
            raise TypeError("ClinicalTextSerializer requires a Task 1 ResolvedConfig")
        validate_config(config.settings)
        self.feature_set = config.settings["clinical_set"]
        self.fields = FEATURE_SETS[self.feature_set]
        if tuple(config.clinical_variables) != self.fields or not set(self.fields) <= APPROVED_PREDICTORS:
            raise ClinicalSerializationError("Configured Medley clinical set does not match the versioned template fields")
        options = {**CLINICAL_TEXT_DEFAULTS, **config.settings.get("clinical_text", {})}
        self.template_version = options["template_version"]
        self._digits = options["numeric_significant_digits"]
        self._extra_policy = options["extra_approved_policy"]
        self._sex_mapping = dict(options["sex_code_mapping"] or {})

    def _sex(self, value: object) -> str:
        if isinstance(value, str):
            labels = {"female": "female", "f": "female", "male": "male", "m": "male"}
            if value.strip().lower() in labels:
                return labels[value.strip().lower()]
            raise ClinicalSerializationError("Unsupported categorical sex string")
        number = _numeric(value)
        if number != number.to_integral_value():
            raise ClinicalSerializationError("Sex codes must be integral and explicitly mapped")
        code = format(number, "f").split(".")[0]
        if number.is_zero():
            code = "0"
        if code not in self._sex_mapping:
            raise ClinicalSerializationError("Ambiguous or unsupported sex code; configure a verified sex_code_mapping")
        return self._sex_mapping[code]

    def serialize(self, clinical: Mapping[str, object]) -> str:
        """Render predictors in fixed order; never stringify records or metadata."""
        if not isinstance(clinical, Mapping):
            raise ClinicalSerializationError("Expected a clinical predictor mapping only")
        # Exact allowlist is stricter than alias heuristics: no alias can bypass it.
        if any(not isinstance(key, str) or key not in APPROVED_PREDICTORS for key in clinical):
            raise ClinicalSerializationError("Unknown or prohibited clinical key; only canonical predictors are accepted")
        if not set(self.fields) <= clinical.keys():
            raise ClinicalSerializationError("Selected feature-set keys are missing; represent missing measurements with None")
        if self._extra_policy == "reject" and clinical.keys() - set(self.fields):
            raise ClinicalSerializationError("Extra approved predictors require explicit extra_approved_policy=ignore")
        formatted = {}
        # Validate even ignored extras; invalid input is never hidden by selection.
        for key, value in clinical.items():
            if value is not None:
                formatted[key] = self._sex(value) if key == "sex" else _format_number(value, self._digits)
        return " ".join(TEMPLATES[key].format(formatted[key]) for key in self.fields if key in formatted)

    def serialize_batch(self, clinical_records: Sequence[Mapping[str, object]]) -> list[str]:
        """Preserve batch order/length, including empty text; errors return no list."""
        if isinstance(clinical_records, (str, bytes)) or not isinstance(clinical_records, Sequence):
            raise ClinicalSerializationError("Expected a sequence of clinical predictor mappings")
        return [self.serialize(clinical) for clinical in clinical_records]
