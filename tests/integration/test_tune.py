"""Tests for hone.cli.tune: expand, loss, score."""

from __future__ import annotations

import pytest

from hone.cli.tune import (
    REQUIRED_KEYS,
    TrialResult,
    TrialSpec,
    expand,
    loss,
    score,
)


def test_expand_is_deterministic_and_bounded() -> None:
    specs = expand(
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
    assert len(specs) == 3
    assert isinstance(specs[0], TrialSpec)
    assert specs[0].learning_rate == 0.00001


def test_expand_rejects_missing_required_key() -> None:
    space = {
        "learning_rate": [0.00001],
        "rank": [4],
        "num_layers": [8],
        "max_seq_length": [2048],
        "grad_accumulation_steps": [8],
    }
    with pytest.raises(ValueError, match="iters"):
        expand(space, max_trials=None)


def test_loss_returns_best_checkpoint_loss() -> None:
    output = "Iter 100: Val loss 2.4\nIter 200: Val loss 2.1\n"
    assert loss(output) == 2.1


def test_loss_returns_none_when_missing() -> None:
    assert loss("training failed") is None


def test_loss_handles_empty_string() -> None:
    assert loss("") is None


def test_score_returns_validation_loss_for_default_objective() -> None:
    spec = TrialSpec(0.00001, 4, 8, 2048, 8, 100)
    result = TrialResult(
        trial_id="t1",
        validation_loss=2.0,
        adapter_path="/tmp/a",
        status="completed",
        parameters=spec,
        benchmark_metrics={"acc": 0.9},
    )
    assert score(result, "validation_loss") == 2.0


def test_score_returns_benchmark_metric() -> None:
    spec = TrialSpec(0.00001, 4, 8, 2048, 8, 100)
    result = TrialResult(
        trial_id="t1",
        validation_loss=None,
        adapter_path="/tmp/a",
        status="completed",
        parameters=spec,
        benchmark_metrics={"livecodebench_pass_at_1": 0.42},
    )
    assert score(result, "livecodebench_pass_at_1") == 0.42


def test_score_returns_none_for_unknown_metric() -> None:
    spec = TrialSpec(0.00001, 4, 8, 2048, 8, 100)
    result = TrialResult(
        trial_id="t1",
        validation_loss=None,
        adapter_path="/tmp/a",
        status="completed",
        parameters=spec,
        benchmark_metrics={"acc": 0.9},
    )
    assert score(result, "unknown") is None


def test_required_keys_constant() -> None:
    assert REQUIRED_KEYS == (
        "learning_rate",
        "rank",
        "num_layers",
        "max_seq_length",
        "grad_accumulation_steps",
        "iters",
    )
