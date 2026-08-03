"""Behavior tests for hone.split: Splitter and MIN_VALID."""

from __future__ import annotations

import pytest

from hone.model import Example, Message, Role
from hone.split import MIN_VALID, Splitter


def make_examples(count: int) -> list[Example]:
    return [
        Example(
            messages=(
                Message(role=Role.user, content=f"q{i}"),
                Message(role=Role.assistant, content=f"a{i}"),
            ),
            metadata={"row_id": i},
        )
        for i in range(count)
    ]


def test_splitter_rejects_ratio_zero() -> None:
    with pytest.raises(ValueError, match="ratio must be between"):
        Splitter(0.0, seed=42)


def test_splitter_rejects_ratio_one() -> None:
    with pytest.raises(ValueError, match="ratio must be between"):
        Splitter(1.0, seed=42)


def test_splitter_rejects_negative_ratio() -> None:
    with pytest.raises(ValueError, match="ratio must be between"):
        Splitter(-0.1, seed=42)


def test_splitter_rejects_ratio_above_one() -> None:
    with pytest.raises(ValueError, match="ratio must be between"):
        Splitter(1.5, seed=42)


def test_splitter_rejects_single_example() -> None:
    with pytest.raises(ValueError, match="at least two examples"):
        Splitter(0.25, seed=42).split(make_examples(1))


def test_splitter_rejects_empty_examples() -> None:
    with pytest.raises(ValueError, match="at least two examples"):
        Splitter(0.25, seed=42).split([])


def test_splitter_same_seed_produces_same_split() -> None:
    examples = make_examples(20)
    train_a, valid_a = Splitter(0.25, seed=42).split(examples)
    train_b, valid_b = Splitter(0.25, seed=42).split(examples)
    assert train_a == train_b
    assert valid_a == valid_b


def test_splitter_different_seeds_produce_different_splits() -> None:
    examples = make_examples(50)
    train_a, _ = Splitter(0.25, seed=42).split(examples)
    train_b, _ = Splitter(0.25, seed=43).split(examples)
    assert [e.metadata["row_id"] for e in train_a] != [
        e.metadata["row_id"] for e in train_b
    ]


def test_splitter_validation_has_at_least_min_valid() -> None:
    examples = make_examples(20)
    _, valid = Splitter(0.05, seed=42).split(examples)
    assert len(valid) >= MIN_VALID


def test_splitter_returns_disjoint_sets() -> None:
    examples = make_examples(20)
    train, valid = Splitter(0.25, seed=42).split(examples)
    train_ids = {id(e) for e in train}
    valid_ids = {id(e) for e in valid}
    assert train_ids.isdisjoint(valid_ids)


def test_splitter_preserves_all_elements() -> None:
    examples = make_examples(20)
    train, valid = Splitter(0.25, seed=42).split(examples)
    assert len(train) + len(valid) == len(examples)
    recovered = train + valid
    assert sorted(id(e) for e in recovered) == sorted(id(e) for e in examples)
