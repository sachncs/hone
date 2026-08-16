"""Behavior tests for hone.train.divergence."""

from __future__ import annotations

import pytest

from hone.train.divergence import DivergenceDetector


def test_clean_log_is_not_divergent() -> None:
    log = "Iter 1: Train loss 2.0\nIter 2: Train loss 1.8\n"
    assert not DivergenceDetector().divergent(log)


def test_few_nans_below_threshold_is_not_divergent() -> None:
    log = "Iter 1: Train loss nan\nIter 2: Train loss 1.8\n"
    assert not DivergenceDetector().divergent(log)


def test_nans_at_threshold_is_divergent() -> None:
    log = "\n".join(f"Iter {i}: Train loss nan" for i in (10, 20, 30))
    assert DivergenceDetector().divergent(log)


def test_nans_above_threshold_is_divergent() -> None:
    log = "\n".join(f"Iter {i}: Train loss nan" for i in (10, 20, 30, 40))
    assert DivergenceDetector().divergent(log)


def test_custom_threshold() -> None:
    detector = DivergenceDetector(threshold=5)
    log = "\n".join(f"Iter {i}: Train loss nan" for i in (10, 20, 30, 40))
    assert not detector.divergent(log)


def test_zero_threshold_rejected() -> None:
    with pytest.raises(ValueError, match="threshold must be positive"):
        DivergenceDetector(threshold=0)
