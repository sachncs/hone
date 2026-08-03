"""End-to-end pipeline tests for hone."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hone import (
    Example,
    Message,
    Normalizer,
    Reader,
    Role,
    Splitter,
    Writer,
)
from hone.cli import app

runner = CliRunner()


def make_example(row_id: int = 0) -> Example:
    return Example(
        messages=(
            Message(role=Role.user, content=f"q-{row_id}"),
            Message(role=Role.assistant, content=f"a-{row_id}"),
        ),
        metadata={"row_id": row_id, "score": float(row_id)},
    )


def test_end_to_end_local_jsonl_to_trainable(tmp_path: Path) -> None:
    """Walk a JSONL file through normalize → split → write → read."""
    input_path = tmp_path / "raw.jsonl"
    output_dir = tmp_path / "out"
    with input_path.open("w", encoding="utf-8") as handle:
        for index in range(20):
            row = {
                "messages": [
                    {"role": "user", "content": f"q-{index}"},
                    {"role": "assistant", "content": f"a-{index}"},
                ],
                "row_id": index,
                "score": float(index),
            }
            handle.write(json.dumps(row) + "\n")

    result = runner.invoke(
        app, ["prepare", "file", "--input", str(input_path), "--output", str(output_dir), "--ratio", "0.1"]
    )
    assert result.exit_code == 0

    examples = list(Reader().read(output_dir / "train.jsonl"))
    valid = list(Reader().read(output_dir / "valid.jsonl"))
    assert len(examples) + len(valid) == 20
    assert all(isinstance(e, Example) for e in examples)
    assert all(isinstance(e, Example) for e in valid)


def test_splitter_after_normalize_roundtrips(tmp_path: Path) -> None:
    """Normalize records, split, and verify each partition round-trips."""
    records = [
        {"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}], "row_id": i}
        for i in range(30)
    ]
    examples = [Normalizer().normalize(r) for r in records]
    train, valid = Splitter(0.2, seed=42).split(examples)
    assert train and valid

    train_path = tmp_path / "train.jsonl"
    valid_path = tmp_path / "valid.jsonl"
    Writer().write(train_path, train)
    Writer().write(valid_path, valid)
    assert list(Reader().read(train_path)) == train
    assert list(Reader().read(valid_path)) == valid


def test_writer_after_normalize_roundtrips(tmp_path: Path) -> None:
    """Normalize then write then read produces identical Example objects."""
    records = [
        {"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]}
        for i in range(5)
    ]
    original = [Normalizer().normalize(r) for r in records]
    path = tmp_path / "out.jsonl"
    Writer().write(path, original)
    loaded = list(Reader().read(path))
    assert loaded == original
