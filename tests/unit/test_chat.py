"""Behavior tests for hone.chat: chat-format validation."""

from __future__ import annotations

import pytest

from hone.chat import parse_messages, parse_messages_from_record
from hone.errors import ValidationError
from hone.model import Role


def test_parse_messages_returns_tuple_of_messages() -> None:
    messages = parse_messages(
        [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
    )
    assert [m.role for m in messages] == [Role.user, Role.assistant]
    assert [m.content for m in messages] == ["Q", "A"]


def test_parse_messages_rejects_non_list() -> None:
    with pytest.raises(ValidationError, match="expected a list"):
        parse_messages("not a list")


def test_parse_messages_rejects_non_object_item() -> None:
    with pytest.raises(ValidationError, match="must be an object"):
        parse_messages(
            [
                {"role": "user", "content": "ok"},
                "bad",
            ]
        )


def test_parse_messages_rejects_missing_role() -> None:
    with pytest.raises(ValidationError, match="missing 'role'"):
        parse_messages([{"content": "ok"}])


def test_parse_messages_rejects_missing_content() -> None:
    with pytest.raises(ValidationError, match="missing 'content'"):
        parse_messages([{"role": "user"}])


def test_parse_messages_rejects_empty_content() -> None:
    with pytest.raises(ValidationError, match="content is empty"):
        parse_messages([{"role": "user", "content": "  "}])


def test_parse_messages_rejects_unknown_role() -> None:
    with pytest.raises(ValidationError, match=r"role"):
        parse_messages([{"role": "tool", "content": "x"}])


def test_parse_messages_includes_location_in_error() -> None:
    with pytest.raises(ValidationError, match=r"data.jsonl:42"):
        parse_messages("not a list", location="data.jsonl:42")


def test_parse_messages_from_record_extracts_messages_field() -> None:
    messages = parse_messages_from_record(
        {"messages": [{"role": "user", "content": "Q"}]},
        location="record",
    )
    assert messages[0].role is Role.user


def test_parse_messages_from_record_rejects_missing_messages() -> None:
    with pytest.raises(ValidationError, match="missing 'messages'"):
        parse_messages_from_record({}, location="record")
