"""Behavior tests for hone.generate.prompt."""

from __future__ import annotations

from hone.generate import format_chat_prompt, lcb_user_prompt, unfence


def test_unfence_strips_opening_and_closing_fences() -> None:
    text = "```python\ndef f(): pass\n```"
    assert unfence(text) == "def f(): pass"


def test_unfence_returns_text_unchanged_when_no_fences() -> None:
    assert unfence("def f(): pass") == "def f(): pass"


def test_unfence_strips_surrounding_whitespace() -> None:
    assert unfence("  \ndef f(): pass\n  ") == "def f(): pass"


def test_unfence_handles_opening_fence_without_closing() -> None:
    text = "```\ndef f(): pass"
    assert unfence(text) == "def f(): pass"


def test_format_chat_prompt_calls_tokenizer_with_correct_flags() -> None:
    captured: dict[str, object] = {}

    class _Tokenizer:
        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            *,
            add_generation_prompt: bool,
            enable_thinking: bool,
        ) -> str:
            captured["messages"] = list(messages)
            captured["add_generation_prompt"] = add_generation_prompt
            captured["enable_thinking"] = enable_thinking
            return "<rendered>"

    rendered = format_chat_prompt(_Tokenizer(), user_content="hello")
    assert rendered == "<rendered>"
    assert captured["messages"] == [{"role": "user", "content": "hello"}]
    assert captured["add_generation_prompt"] is True
    assert captured["enable_thinking"] is False


def test_format_chat_prompt_passes_through_enable_thinking() -> None:
    captured: dict[str, object] = {}

    class _Tokenizer:
        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            *,
            add_generation_prompt: bool,
            enable_thinking: bool,
        ) -> str:
            captured["enable_thinking"] = enable_thinking
            return ""

    format_chat_prompt(_Tokenizer(), user_content="x", enable_thinking=True)
    assert captured["enable_thinking"] is True


def test_lcb_user_prompt_includes_question() -> None:
    prompt = lcb_user_prompt("Solve fizzbuzz")
    assert "Solve fizzbuzz" in prompt
    assert "markdown fences" in prompt


def test_lcb_user_prompt_stringifies_non_string() -> None:
    prompt = lcb_user_prompt({"q": "v"})
    assert "Solve this competitive-programming problem" in prompt
