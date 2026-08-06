"""Tests for the generate subcommands and unfence helper."""

from __future__ import annotations

from hone.cli.generate import unfence


def test_unfence_removes_double_newline_fenced_block() -> None:
    text = "```python\nx = 1\n```"
    assert unfence(text) == "x = 1"


def test_unfence_returns_text_without_fences_unchanged() -> None:
    assert unfence("x = 1") == "x = 1"


def test_unfence_handles_no_closing_fence() -> None:
    text = "```\nx = 1"
    assert unfence(text) == "x = 1"


def test_unfence_strips_surrounding_whitespace() -> None:
    text = "  ```\nx = 1\n```  "
    assert unfence(text) == "x = 1"


def test_unfence_preserves_internal_backticks_in_code() -> None:
    text = "```\nresult = `echo hi`\n```"
    assert unfence(text) == "result = `echo hi`"
