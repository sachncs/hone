"""Tests for hone.tune and hone.cli.tune: search, loss, selection."""

from __future__ import annotations

import pytest

from hone.tune import (
    TrialResult,
    TrialSpec,
    expand_search,
    loss_from_output,
    select_best,
)
from hone.tune.search import REQUIRED_KEYS, objective_value


def test_expand_is_deterministic_and_bounded() -> None:
    specs = expand_search(
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
        expand_search(space, max_trials=None)


def test_loss_returns_best_checkpoint_loss() -> None:
    output = "Iter 100: Val loss 2.4\nIter 200: Val loss 2.1\n"
    assert loss_from_output(output) == 2.1


def test_loss_returns_none_when_missing() -> None:
    assert loss_from_output("training failed") is None


def test_loss_handles_empty_string() -> None:
    assert loss_from_output("") is None


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


def test_select_best_minimises_validation_loss() -> None:
    spec_a = TrialSpec(0.00001, 4, 8, 2048, 8, 100)
    spec_b = TrialSpec(0.00002, 4, 8, 2048, 8, 100)
    results = [
        TrialResult(
            trial_id="a",
            validation_loss=2.5,
            adapter_path="/tmp/a",
            status="completed",
            parameters=spec_a,
        ),
        TrialResult(
            trial_id="b",
            validation_loss=2.1,
            adapter_path="/tmp/b",
            status="completed",
            parameters=spec_b,
        ),
    ]
    assert select_best(results, objective="validation_loss").trial_id == "b"


def test_select_best_maximises_benchmark_metric() -> None:
    spec_a = TrialSpec(0.00001, 4, 8, 2048, 8, 100)
    spec_b = TrialSpec(0.00002, 4, 8, 2048, 8, 100)
    results = [
        TrialResult(
            trial_id="a",
            validation_loss=None,
            adapter_path="/tmp/a",
            status="completed",
            parameters=spec_a,
            benchmark_metrics={"acc": 0.5},
        ),
        TrialResult(
            trial_id="b",
            validation_loss=None,
            adapter_path="/tmp/b",
            status="completed",
            parameters=spec_b,
            benchmark_metrics={"acc": 0.7},
        ),
    ]
    assert select_best(results, objective="acc").trial_id == "b"


def test_select_best_raises_when_no_results_have_objective() -> None:
    spec = TrialSpec(0.00001, 4, 8, 2048, 8, 100)
    results = [
        TrialResult(
            trial_id="a",
            validation_loss=None,
            adapter_path="/tmp/a",
            status="failed",
            parameters=spec,
        ),
    ]
    with pytest.raises(ValueError, match="no successful trial"):
        select_best(results, objective="validation_loss")


def test_required_keys_constant() -> None:
    assert REQUIRED_KEYS == (
        "learning_rate",
        "rank",
        "num_layers",
        "max_seq_length",
        "grad_accumulation_steps",
        "iters",
    )
