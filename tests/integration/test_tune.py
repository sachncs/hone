"""Tests for hone.cli.tune: build_trials, parse_validation_loss, objective_value."""

from __future__ import annotations

import pytest

from hone.cli.tune import (
    REQUIRED_KEYS,
    TrialResult,
    TrialSpec,
    build_trials,
    objective_value,
    parse_validation_loss,
)


def test_build_trials_is_deterministic_and_bounded() -> None:
    trials = build_trials(
        {
            "learning_rate": [0.00001, 0.00002],
            "rank": [4, 8],
            "num_layers": [8],
            "max_seq_length": [2048],
            "grad_accumulation_steps": [8],
            "iters": [100, 200],
        },
        max_trials=3,
    )
    assert len(trials) == 3
    assert isinstance(trials[0], TrialSpec)
    assert trials[0].learning_rate == 0.00001


def test_build_trials_rejects_missing_required_key() -> None:
    space = {
        "learning_rate": [0.00001],
        "rank": [4],
        "num_layers": [8],
        "max_seq_length": [2048],
        "grad_accumulation_steps": [8],
    }
    with pytest.raises(ValueError, match="iters"):
        build_trials(space, max_trials=None)


def test_parse_validation_loss_returns_best_checkpoint_loss() -> None:
    output = "Iter 100: Val loss 2.4\nIter 200: Val loss 2.1\n"
    assert parse_validation_loss(output) == 2.1


def test_parse_validation_loss_returns_none_when_missing() -> None:
    assert parse_validation_loss("training failed") is None


def test_parse_validation_loss_handles_empty_string() -> None:
    assert parse_validation_loss("") is None


def test_objective_value_returns_validation_loss_for_default_objective() -> None:
    spec = TrialSpec(0.00001, 4, 8, 2048, 8, 100)
    result = TrialResult(
        trial_id="t1",
        validation_loss=2.0,
        adapter_path="/tmp/a",
        status="completed",
        parameters=spec,
        benchmark_metrics={"acc": 0.9},
    )
    assert objective_value(result, "validation_loss") == 2.0


def test_objective_value_returns_benchmark_metric() -> None:
    spec = TrialSpec(0.00001, 4, 8, 2048, 8, 100)
    result = TrialResult(
        trial_id="t1",
        validation_loss=None,
        adapter_path="/tmp/a",
        status="completed",
        parameters=spec,
        benchmark_metrics={"livecodebench_pass_at_1": 0.42},
    )
    assert objective_value(result, "livecodebench_pass_at_1") == 0.42


def test_objective_value_returns_none_for_unknown_metric() -> None:
    spec = TrialSpec(0.00001, 4, 8, 2048, 8, 100)
    result = TrialResult(
        trial_id="t1",
        validation_loss=None,
        adapter_path="/tmp/a",
        status="completed",
        parameters=spec,
        benchmark_metrics={"acc": 0.9},
    )
    assert objective_value(result, "unknown") is None


def test_required_keys_constant() -> None:
    assert REQUIRED_KEYS == (
        "learning_rate",
        "rank",
        "num_layers",
        "max_seq_length",
        "grad_accumulation_steps",
        "iters",
    )
