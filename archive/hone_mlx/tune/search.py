"""Hyperparameter search-space expansion and result selection."""

from __future__ import annotations

import itertools
from typing import Any

from hone.tune.spec import TrialResult, TrialSpec

REQUIRED_KEYS: tuple[str, ...] = (
    "learning_rate",
    "rank",
    "num_layers",
    "max_seq_length",
    "grad_accumulation_steps",
    "iters",
)


def expand_search(
    search_space: dict[str, Any],
    *,
    max_trials: int | None = None,
) -> list[TrialSpec]:
    """Build a deterministic Cartesian search, optionally bounded by a budget."""
    values: list[list[float | int]] = []
    for key in REQUIRED_KEYS:
        options = search_space.get(key)
        if not isinstance(options, list) or not options:
            raise ValueError(f"search space must define non-empty {key}")
        values.append(options)
    specs = [
        TrialSpec(
            learning_rate=float(combination[0]),
            rank=int(combination[1]),
            num_layers=int(combination[2]),
            max_seq_length=int(combination[3]),
            grad_accumulation_steps=int(combination[4]),
            iters=int(combination[5]),
        )
        for combination in itertools.product(*values)
    ]
    if max_trials is not None:
        if max_trials < 1:
            raise ValueError("max_trials must be positive")
        specs = specs[:max_trials]
    return specs


def objective_value(result: TrialResult, objective: str) -> float | None:
    """Return the objective value for a result, or None when missing."""
    if objective == "validation_loss":
        return result.validation_loss
    return result.benchmark_metrics.get(objective)


def select_best(
    results: list[TrialResult],
    *,
    objective: str,
) -> TrialResult:
    """Select the best result for an objective (min for validation_loss, else max)."""
    successful = [
        result
        for result in results
        if result.status == "completed"
        and objective_value(result, objective) is not None
    ]
    if not successful:
        raise ValueError(f"no successful trial produced objective {objective}")
    if objective == "validation_loss":
        return min(
            successful,
            key=lambda result: objective_value(result, objective) or float("inf"),
        )
    return max(
        successful,
        key=lambda result: objective_value(result, objective) or float("-inf"),
    )


__all__ = ["REQUIRED_KEYS", "expand_search", "objective_value", "select_best"]
