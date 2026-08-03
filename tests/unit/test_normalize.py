"""Behavior tests for hone.normalize: Normalizer and SweNormalizer."""

from __future__ import annotations

import pytest

from hone.model import Role
from hone.normalize import Normalizer, SweNormalizer


def test_normalize_prompt_completion_creates_user_assistant() -> None:
    example = Normalizer().normalize({"prompt": "Q", "completion": "A"})
    assert example.messages[0].role is Role.user
    assert example.messages[0].content == "Q"
    assert example.messages[1].role is Role.assistant
    assert example.messages[1].content == "A"


def test_normalize_messages_creates_message_list() -> None:
    example = Normalizer().normalize(
        {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}
    )
    assert [m.role for m in example.messages] == [Role.user, Role.assistant]


def test_normalize_messages_rejects_non_object() -> None:
    with pytest.raises(ValueError, match=r"messages\[\d+\] must be an object"):
        Normalizer().normalize(
            {"messages": [{"role": "user", "content": "ok"}, "bad"]}
        )


def test_normalize_messages_rejects_missing_role() -> None:
    with pytest.raises(ValueError, match="messages\\[0\\] is missing 'role'"):
        Normalizer().normalize(
            {"messages": [{"content": "ok"}, {"role": "assistant", "content": "a"}]}
        )


def test_normalize_messages_rejects_missing_content() -> None:
    with pytest.raises(ValueError, match="messages\\[0\\] is missing 'content'"):
        Normalizer().normalize(
            {"messages": [{"role": "user"}, {"role": "assistant", "content": "a"}]}
        )


def test_normalize_accepts_system_user_assistant() -> None:
    example = Normalizer().normalize(
        {
            "messages": [
                {"role": "system", "content": "you are helpful"},
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
            ]
        }
    )
    assert [m.role for m in example.messages] == [Role.system, Role.user, Role.assistant]


def test_normalize_rejects_missing_prompt_and_messages() -> None:
    with pytest.raises(ValueError, match="expected 'messages' list or 'prompt'/'completion'"):
        Normalizer().normalize({"foo": "bar"})


def test_normalize_strips_surrounding_whitespace() -> None:
    example = Normalizer().normalize({"prompt": "  Q  ", "completion": "  A  "})
    assert example.messages[0].content == "Q"
    assert example.messages[1].content == "A"


def test_swe_rejects_missing_problem_statement() -> None:
    with pytest.raises(ValueError, match="missing problem_statement"):
        SweNormalizer().normalize({"instance_id": "i", "patch": "p"})


def test_swe_rejects_missing_patch() -> None:
    with pytest.raises(ValueError, match="missing patch"):
        SweNormalizer().normalize({"instance_id": "i", "problem_statement": "p"})


def test_swe_includes_repo_and_version_in_prompt() -> None:
    example = SweNormalizer().normalize(
        {
            "instance_id": "i",
            "problem_statement": "fix bug",
            "repo": "org/repo",
            "version": "abc",
            "patch": "diff",
        }
    )
    prompt = example.messages[0].content
    assert "Repository: org/repo" in prompt
    assert "Version: abc" in prompt
    assert "Issue:" in prompt


def test_swe_handles_missing_repo() -> None:
    example = SweNormalizer().normalize({"problem_statement": "p", "patch": "diff"})
    prompt = example.messages[0].content
    assert "Repository: " in prompt
    assert "Version: " in prompt


def test_swe_metadata_contains_instance_id() -> None:
    example = SweNormalizer().normalize(
        {"instance_id": "i_42", "problem_statement": "p", "patch": "diff"}
    )
    assert example.metadata.get("instance_id") == "i_42"
