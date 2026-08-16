"""Trial value objects and result extraction."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TrialSpec:
    """One hyperparameter combination."""

    learning_rate: float
    rank: int
    num_layers: int
    max_seq_length: int
    grad_accumulation_steps: int
    iters: int


@dataclass(frozen=True)
class TrialResult:
    """Outcome of one trial run."""

    trial_id: str
    validation_loss: float | None
    adapter_path: str
    status: str
    parameters: TrialSpec
    benchmark_metrics: dict[str, float] = field(default_factory=dict)


VALIDATION_LOSS_PATTERN: re.Pattern[str] = re.compile(
    r"Val loss\s+([0-9]+(?:\.[0-9]+)?)"
)


def loss_from_output(output: str) -> float | None:
    """Return the lowest validation loss reported in the trainer output."""
    values = [
        float(match.group(1)) for match in VALIDATION_LOSS_PATTERN.finditer(output)
    ]
    return min(values) if values else None


def load_metrics(path: Path) -> dict[str, float]:
    """Load optional benchmark metrics from a hook-produced JSON file."""
    if not path.exists():
        return {}
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError(f"benchmark metrics must be an object: {path}")
    return {
        str(key): float(value)
        for key, value in record.items()
        if isinstance(value, (int, float))
    }


__all__ = ["TrialResult", "TrialSpec", "load_metrics", "loss_from_output"]
