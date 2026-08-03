"""Tests for the generate subcommands and strip_fences helper."""

from __future__ import annotations

import pytest

from hone.cli.generate import strip_fences


def test_strip_fences_removes_double_newline_fenced_block() -> None:
    text = "```python\nx = 1\n```"
    assert strip_fences(text) == "x = 1"


def test_strip_fences_returns_text_without_fences_unchanged() -> None:
    assert strip_fences("x = 1") == "x = 1"


def test_strip_fences_handles_no_closing_fence() -> None:
    text = "```\nx = 1"
    assert strip_fences(text) == "x = 1"


def test_strip_fences_strips_surrounding_whitespace() -> None:
    text = "  ```\nx = 1\n```  "
    assert strip_fences(text) == "x = 1"


def test_strip_fences_preserves_internal_backticks_in_code() -> None:
    text = "```\nresult = `echo hi`\n```"
    assert strip_fences(text) == "result = `echo hi`"
