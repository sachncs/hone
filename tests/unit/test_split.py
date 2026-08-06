"""Behavior tests for hone.split: Splitter and MIN_VALID."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hone.model import Example, Message, Role
from hone.split import MIN_VALID, Splitter, split_file


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


def make_jsonl(path: Path, count: int) -> None:
    """Write count distinct chat records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for i in range(count):
            record = {
                "messages": [
                    {"role": "user", "content": f"q{i}"},
                    {"role": "assistant", "content": f"a{i}"},
                ]
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


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


def test_split_file_returns_counts(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    make_jsonl(source, 40)
    train_count, valid_count = split_file(
        source, tmp_path / "train.jsonl", tmp_path / "valid.jsonl", 0.05, seed=42
    )
    assert (train_count, valid_count) == (38, 2)


def test_split_file_preserves_all_lines_byte_identical(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    make_jsonl(source, 40)
    split_file(source, tmp_path / "train.jsonl", tmp_path / "valid.jsonl", 0.1, seed=42)
    recovered = read_lines(tmp_path / "train.jsonl") + read_lines(
        tmp_path / "valid.jsonl"
    )
    assert sorted(recovered) == sorted(read_lines(source))


def test_split_file_partitions_are_disjoint(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    make_jsonl(source, 40)
    split_file(source, tmp_path / "train.jsonl", tmp_path / "valid.jsonl", 0.1, seed=42)
    train = set(read_lines(tmp_path / "train.jsonl"))
    valid = set(read_lines(tmp_path / "valid.jsonl"))
    assert train.isdisjoint(valid)


def test_split_file_same_seed_produces_same_split(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    make_jsonl(source, 40)
    split_file(
        source, tmp_path / "a" / "train.jsonl", tmp_path / "a" / "valid.jsonl", 0.1, 42
    )
    split_file(
        source, tmp_path / "b" / "train.jsonl", tmp_path / "b" / "valid.jsonl", 0.1, 42
    )
    assert read_lines(tmp_path / "a" / "train.jsonl") == read_lines(
        tmp_path / "b" / "train.jsonl"
    )
    assert read_lines(tmp_path / "a" / "valid.jsonl") == read_lines(
        tmp_path / "b" / "valid.jsonl"
    )


def test_split_file_different_seeds_produce_different_splits(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    make_jsonl(source, 50)
    split_file(
        source, tmp_path / "a" / "train.jsonl", tmp_path / "a" / "valid.jsonl", 0.1, 42
    )
    split_file(
        source, tmp_path / "b" / "train.jsonl", tmp_path / "b" / "valid.jsonl", 0.1, 43
    )
    assert read_lines(tmp_path / "a" / "valid.jsonl") != read_lines(
        tmp_path / "b" / "valid.jsonl"
    )


def test_split_file_valid_has_at_least_min_valid(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    make_jsonl(source, 2)
    train_count, valid_count = split_file(
        source, tmp_path / "train.jsonl", tmp_path / "valid.jsonl", 0.05, seed=42
    )
    assert valid_count >= MIN_VALID
    assert (train_count, valid_count) == (1, 1)


def test_split_file_ratio_near_one_keeps_train_nonempty(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    make_jsonl(source, 5)
    train_count, valid_count = split_file(
        source, tmp_path / "train.jsonl", tmp_path / "valid.jsonl", 0.99, seed=42
    )
    assert train_count >= 1
    assert train_count + valid_count == 5


def test_split_file_rejects_ratio_zero(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    make_jsonl(source, 10)
    with pytest.raises(ValueError, match="ratio must be between"):
        split_file(source, tmp_path / "train.jsonl", tmp_path / "valid.jsonl", 0.0, 42)


def test_split_file_rejects_ratio_one(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    make_jsonl(source, 10)
    with pytest.raises(ValueError, match="ratio must be between"):
        split_file(source, tmp_path / "train.jsonl", tmp_path / "valid.jsonl", 1.0, 42)


def test_split_file_rejects_negative_ratio(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    make_jsonl(source, 10)
    with pytest.raises(ValueError, match="ratio must be between"):
        split_file(source, tmp_path / "train.jsonl", tmp_path / "valid.jsonl", -0.1, 42)


def test_split_file_rejects_ratio_above_one(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    make_jsonl(source, 10)
    with pytest.raises(ValueError, match="ratio must be between"):
        split_file(source, tmp_path / "train.jsonl", tmp_path / "valid.jsonl", 1.5, 42)


def test_split_file_rejects_single_line(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    make_jsonl(source, 1)
    with pytest.raises(ValueError, match="at least two JSON lines"):
        split_file(source, tmp_path / "train.jsonl", tmp_path / "valid.jsonl", 0.1, 42)


def test_split_file_rejects_malformed_line(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    make_jsonl(source, 3)
    with source.open("a", encoding="utf-8") as handle:
        handle.write("this is not json\n")
    with pytest.raises(ValueError, match=r":4:"):
        split_file(source, tmp_path / "train.jsonl", tmp_path / "valid.jsonl", 0.1, 42)


def test_split_file_leaves_no_tmp_files(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    make_jsonl(source, 40)
    split_file(source, tmp_path / "train.jsonl", tmp_path / "valid.jsonl", 0.1, 42)
    assert list(tmp_path.glob("*.tmp")) == []
