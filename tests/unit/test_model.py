"""Behavior tests for hone.model: Example, Message, Role."""

from __future__ import annotations

import dataclasses

import pytest

from hone.model import Example, Message, Meta, Role


def test_role_accepts_system_user_assistant() -> None:
    assert Role("system") is Role.system
    assert Role("user") is Role.user
    assert Role("assistant") is Role.assistant


def test_role_rejects_unknown_value() -> None:
    with pytest.raises(ValueError):
        Role("tool")


def test_role_serializes_to_string() -> None:
    assert str(Role.user) == "user"
    assert str(Role.assistant) == "assistant"


def test_message_rejects_empty_content() -> None:
    with pytest.raises(ValueError, match="content cannot be empty"):
        Message(role=Role.user, content="")


def test_message_rejects_whitespace_only_content() -> None:
    with pytest.raises(ValueError, match="content cannot be empty"):
        Message(role=Role.user, content="   \t\n")


def test_message_strips_surrounding_whitespace() -> None:
    message = Message(role=Role.user, content="  hi  ")
    assert message.content == "hi"


def test_message_preserves_internal_whitespace() -> None:
    message = Message(role=Role.user, content="a  b\nc")
    assert message.content == "a  b\nc"


def test_message_is_immutable() -> None:
    message = Message(role=Role.user, content="hi")
    with pytest.raises(dataclasses.FrozenInstanceError):
        message.content = "x"  # type: ignore[misc]


def test_example_requires_at_least_two_messages() -> None:
    single = Message(role=Role.user, content="q")
    with pytest.raises(ValueError, match="at least two messages"):
        Example(messages=(single,), metadata={})


def test_example_requires_assistant_last() -> None:
    user = Message(role=Role.user, content="q")
    with pytest.raises(ValueError, match="must end with an assistant message"):
        Example(messages=(user, user), metadata={})


def test_example_accepts_system_user_assistant() -> None:
    system = Message(role=Role.system, content="be helpful")
    user = Message(role=Role.user, content="q")
    assistant = Message(role=Role.assistant, content="a")
    example = Example(messages=(system, user, assistant), metadata={})
    assert len(example.messages) == 3
    assert example.messages[0].role is Role.system


def test_example_character_count_sums_message_lengths() -> None:
    user = Message(role=Role.user, content="hello")
    assistant = Message(role=Role.assistant, content="world!")
    example = Example(messages=(user, assistant), metadata={})
    assert example.character_count == len("hello") + len("world!")


def test_example_metadata_is_defensively_copied() -> None:
    user = Message(role=Role.user, content="q")
    assistant = Message(role=Role.assistant, content="a")
    original: Meta = {"k": 1}
    example = Example(messages=(user, assistant), metadata=original)
    original["k"] = 999
    assert example.metadata["k"] == 1


def test_example_to_record_roundtrip() -> None:
    user = Message(role=Role.user, content="q")
    assistant = Message(role=Role.assistant, content="a")
    example = Example(
        messages=(user, assistant),
        metadata={"instance_id": "ex1", "score": 0.5},
    )
    record = example.to_record()
    assert record["messages"] == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]
    assert record["instance_id"] == "ex1"
    assert record["score"] == 0.5
