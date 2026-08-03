"""Behavioral tests for the JSONL preparation contracts."""

import json

import pytest

from finetune.models import ChatMessage, MessageRole, TrainingExample
from finetune.normalization import (
    SWETrainingExampleNormalizer,
    TrainingExampleNormalizer,
)

NORMALIZER = TrainingExampleNormalizer()
SWE_NORMALIZER = SWETrainingExampleNormalizer()


def test_normalize_converts_prompt_completion_rows() -> None:
    record = NORMALIZER.normalize({"prompt": " Solve it ", "completion": " answer "})

    serialized = {"messages": [message.to_record() for message in record.messages]}

    assert serialized == {
        "messages": [
            {"role": "user", "content": "Solve it"},
            {"role": "assistant", "content": "answer"},
        ]
    }


def test_normalize_rejects_non_object_messages() -> None:
    with pytest.raises(ValueError, match="messages must contain objects"):
        NORMALIZER.normalize({"messages": [{"role": "user", "content": "ok"}, "bad"]})


def test_normalize_rejects_examples_without_assistant_response() -> None:
    with pytest.raises(ValueError, match="training example"):
        NORMALIZER.normalize({"messages": [{"role": "user", "content": "question"}]})


def test_row_to_messages_includes_repository_context_and_patch() -> None:
    record = SWE_NORMALIZER.normalize(
        {
            "instance_id": "example__1",
            "problem_statement": " Fix the bug. ",
            "repo": "example/project",
            "version": "abc123",
            "patch": "diff --git a/a b/a",
        }
    ).to_record()

    assert record["instance_id"] == "example__1"
    messages = record["messages"]
    assert isinstance(messages, list)
    assert "Repository: example/project" in messages[0]["content"]
    assert messages[-1] == {
        "role": "assistant",
        "content": "diff --git a/a b/a",
    }


def test_generated_records_are_json_serializable() -> None:
    record = SWE_NORMALIZER.normalize(
        {"problem_statement": "Issue", "patch": "Patch"}
    ).to_record()

    assert json.loads(json.dumps(record)) == record


def test_training_example_is_immutable_and_counts_characters() -> None:
    example = TrainingExample(
        messages=(
            ChatMessage(MessageRole.USER, "question"),
            ChatMessage(MessageRole.ASSISTANT, "answer"),
        ),
        metadata={"instance_id": "example"},
    )

    assert example.character_count == len("question") + len("answer")
    with pytest.raises(AttributeError):
        example.messages = ()
