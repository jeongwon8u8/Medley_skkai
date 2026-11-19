"""Plan type shared by module CLI and execution-time revalidation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .experiment_protocol import fingerprint


@dataclass(frozen=True)
class PlannedRun:
    model: str
    seed: int
    fold: int
    config: object = field(repr=False)
    plan_id: str
    outer_sha256: str | None
    output_root: str
    prerequisites: tuple[str, ...]

    def summary(self):
        return {"model": self.model, "seed": self.seed, "fold": self.fold, "plan_id": self.plan_id,
                "config_fingerprint": fingerprint(self.config.to_dict()), "outer_sha256": self.outer_sha256,
                "output": str(Path(self.output_root) / "<C1-run-id>" / "executions" / "<execution-id>"),
                "status": "planned_not_executed", "cohort_cache_identity": "unverified_no_patient_or_cache_reads",
                "llm": {k: self.config.settings["llm"][k] for k in ("model_id", "revision", "tuning")},
                "prerequisites": list(self.prerequisites)}
