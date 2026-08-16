"""Behavior tests for hone.prepare.mappers."""

from __future__ import annotations

import pytest

from hone.errors import ValidationError
from hone.prepare.mappers import as_codeforces_text, as_sft


def test_as_sft_promotes_messages_field() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
    }
    assert as_sft(row) == {"messages": row["messages"]}


def test_as_sft_promotes_human_and_gpt_aliases() -> None:
    row = {
        "messages": [
            {"role": "Human", "content": "Q"},
            {"role": "GPT", "content": "A"},
        ]
    }
    out = as_sft(row)
    assert out is not None
    assert out["messages"][0]["role"] == "user"
    assert out["messages"][1]["role"] == "assistant"


def test_as_sft_rejects_short_messages_list() -> None:
    row = {"messages": [{"role": "user", "content": "only one"}]}
    assert as_sft(row) is None


def test_as_sft_rejects_messages_with_non_assistant_last() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "Q"},
            {"role": "user", "content": "second user"},
        ]
    }
    assert as_sft(row) is None


def test_as_sft_builds_from_prompt_completion() -> None:
    row = {"prompt": "Q", "completion": "A"}
    out = as_sft(row)
    assert out == {
        "messages": [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
    }


def test_as_sft_falls_back_through_aliases() -> None:
    assert as_sft({"instruction": "Q", "response": "A"}) is not None
    assert as_sft({"input": "Q", "output": "A"}) is not None
    assert as_sft({"question": "Q", "answer": "A"}) is not None
    assert as_sft({"problem": "Q", "solution": "A"}) is not None


def test_as_sft_drops_empty_prompt_or_completion() -> None:
    assert as_sft({"prompt": "   ", "completion": "A"}) is None
    assert as_sft({"prompt": "Q", "completion": ""}) is None
    assert as_sft({"prompt": "Q"}) is None


def test_as_sft_rejects_dict_or_list_completion() -> None:
    assert as_sft({"prompt": "Q", "completion": {"key": "v"}}) is None
    assert as_sft({"prompt": "Q", "completion": ["a", "b"]}) is None


def test_as_sft_rejects_messages_with_unknown_role() -> None:
    row = {
        "messages": [
            {"role": "tool", "content": "x"},
            {"role": "assistant", "content": "a"},
        ]
    }
    assert as_sft(row) is None


def test_as_sft_rejects_messages_with_empty_content() -> None:
    row = {
        "messages": [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "a"},
        ]
    }
    assert as_sft(row) is None


def test_as_sft_returns_none_for_unknown_shape() -> None:
    assert as_sft({"foo": "bar"}) is None


def test_as_codeforces_text_concatenates_present_sections() -> None:
    row = {
        "title": "T",
        "description": "D",
        "input_format": "I",
        "output_format": "O",
        "editorial": "E",
    }
    out = as_codeforces_text(row)
    assert "## TITLE\nT" in out["text"]
    assert "## EDITORIAL\nE" in out["text"]


def test_as_codeforces_text_omits_blank_sections() -> None:
    row = {"title": "T", "description": "", "editorial": "E"}
    out = as_codeforces_text(row)
    assert "DESCRIPTION" not in out["text"]
    assert "EDITORIAL" in out["text"]


def test_as_codeforces_text_raises_when_all_sections_blank() -> None:
    with pytest.raises(ValidationError, match="no serializable problem text"):
        as_codeforces_text({"title": "", "description": ""})
