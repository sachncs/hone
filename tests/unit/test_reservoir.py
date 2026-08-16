"""Behavior tests for hone.prepare.reservoir.Reservoir."""

from __future__ import annotations

import pytest

from hone.prepare.reservoir import Reservoir


def test_reservoir_rejects_zero_cap() -> None:
    with pytest.raises(ValueError, match="cap must be positive"):
        Reservoir[int](cap=0, seed=42)


def test_reservoir_fills_until_cap() -> None:
    reservoir: Reservoir[int] = Reservoir(cap=3, seed=42)
    for value in (1, 2, 3):
        reservoir.observe(value)
    assert reservoir.sample() == [1, 2, 3]


def test_reservoir_replaces_uniformly_after_cap() -> None:
    reservoir: Reservoir[int] = Reservoir(cap=2, seed=42)
    for value in range(100):
        reservoir.observe(value)
    sample = reservoir.sample()
    assert len(sample) == 2
    assert all(isinstance(item, int) for item in sample)


def test_reservoir_same_seed_same_sample() -> None:
    a: Reservoir[int] = Reservoir(cap=5, seed=7)
    b: Reservoir[int] = Reservoir(cap=5, seed=7)
    for value in range(50):
        a.observe(value)
        b.observe(value)
    assert a.sample() == b.sample()


def test_reservoir_different_seed_different_sample() -> None:
    a: Reservoir[int] = Reservoir(cap=5, seed=7)
    b: Reservoir[int] = Reservoir(cap=5, seed=8)
    for value in range(50):
        a.observe(value)
        b.observe(value)
    assert a.sample() != b.sample()


def test_reservoir_extend_accepts_iterable() -> None:
    reservoir: Reservoir[int] = Reservoir(cap=5, seed=1)
    reservoir.extend(range(10))
    assert reservoir.seen == 10
    assert len(reservoir) == 5


def test_reservoir_shuffle_mutates_in_place() -> None:
    reservoir: Reservoir[int] = Reservoir(cap=5, seed=1)
    reservoir.extend(range(10))
    before = reservoir.sample()
    shuffled = reservoir.shuffle()
    assert len(shuffled) == 5
    assert set(shuffled) == set(before)
