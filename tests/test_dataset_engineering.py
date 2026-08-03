"""Tests for deterministic dataset persistence and splitting."""

from pathlib import Path

from finetune.jsonl import JsonlDatasetReader, JsonlDatasetWriter
from finetune.models import ChatMessage, MessageRole, TrainingExample
from finetune.splitting import DatasetSplitter


def build_examples() -> list[TrainingExample]:
    return [
        TrainingExample(
            messages=(
                ChatMessage(MessageRole.USER, f"question {index}"),
                ChatMessage(MessageRole.ASSISTANT, f"answer {index}"),
            ),
            metadata={"row_id": index},
        )
        for index in range(4)
    ]


def test_splitter_is_deterministic_and_disjoint() -> None:
    examples = build_examples()
    first_split = DatasetSplitter(0.25, seed=42).split(examples)
    second_split = DatasetSplitter(0.25, seed=42).split(examples)

    assert first_split == second_split
    train_ids = {example.metadata["row_id"] for example in first_split[0]}
    validation_ids = {example.metadata["row_id"] for example in first_split[1]}
    assert train_ids.isdisjoint(validation_ids)
    assert len(first_split[0]) + len(first_split[1]) == len(examples)


def test_jsonl_writer_and_reader_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "train.jsonl"
    examples = build_examples()

    assert JsonlDatasetWriter().write(path, examples) == len(examples)
    assert list(JsonlDatasetReader().read(path)) == examples
