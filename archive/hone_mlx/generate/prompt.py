"""Prompt-formatting helpers for inference.

Two responsibilities:

* :func:`unfence` — strip optional markdown fences from a
  generated code snippet. The CLI / eval layers can rely on this
  to keep the raw text returned by the model parseable.
* :func:`format_chat_prompt` — apply a tokenizer's chat template
  to a one-message list. Centralized so the CLI never calls
  ``tokenizer.apply_chat_template`` directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol


class TokenizerLike(Protocol):
    """Minimal tokenizer protocol — just enough for prompt templating."""

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str: ...


def unfence(text: str) -> str:
    """Remove optional markdown fences from generated source code."""
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def format_chat_prompt(
    tokenizer: TokenizerLike,
    *,
    user_content: str,
    enable_thinking: bool = False,
) -> str:
    """Apply the tokenizer's chat template to a single user message."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def lcb_user_prompt(question_content: Any) -> str:
    """Render a LiveCodeBench question into the standard user prompt."""
    return (
        "Solve this competitive-programming problem. "
        "Return only the complete solution code, with no "
        "markdown fences.\n\n" + str(question_content)
    )


__all__ = ["TokenizerLike", "format_chat_prompt", "lcb_user_prompt", "unfence"]
