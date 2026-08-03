"""Tests for reproducible hyperparameter search behavior."""

from scripts.tune_mlx import TrialSpec, build_trials, parse_validation_loss


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


def test_parse_validation_loss_returns_best_checkpoint_loss() -> None:
    output = "Iter 100: Val loss 2.4\nIter 200: Val loss 2.1\n"

    assert parse_validation_loss(output) == 2.1
    assert parse_validation_loss("training failed") is None
