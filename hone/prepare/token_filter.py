"""Tokenizer-based length filter for prepared records.

The :class:`TokenFilter` wraps a HuggingFace tokenizer and decides
whether a record is short enough to keep. The filter is opt-in:
passing ``max_tokens=0`` skips loading the tokenizer entirely.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol


class TokenizerLike(Protocol):
    """Minimal tokenizer protocol — just enough for token counting."""

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


class TokenFilter:
    """Drop records whose joined text exceeds ``max_tokens`` tokens."""

    def __init__(self, *, max_tokens: int, tokenizer: TokenizerLike | None) -> None:
        if max_tokens < 0:
            raise ValueError(f"max_tokens must be non-negative, got {max_tokens}")
        self._max_tokens = max_tokens
        self._tokenizer = tokenizer

    @property
    def enabled(self) -> bool:
        """Whether the filter is active."""
        return self._max_tokens > 0 and self._tokenizer is not None

    def too_long(self, record: Mapping[str, Any]) -> bool:
        """Return True when ``record`` should be skipped by the filter."""
        if not self.enabled or self._tokenizer is None:
            return False
        tokens = self._tokenizer.encode(_record_text(record), add_special_tokens=False)
        return len(tokens) > self._max_tokens


def _record_text(record: Mapping[str, Any]) -> str:
    """Extract the textual payload of a chat- or text-format record."""
    messages = record.get("messages")
    if isinstance(messages, list):
        return "\n".join(
            str(message.get("content", ""))
            for message in messages
            if isinstance(message, Mapping)
        )
    text = record.get("text")
    return str(text) if text is not None else ""


def load_tokenizer(model_id: str, *, trust_remote_code: bool = True) -> TokenizerLike:
    """Lazy-load the tokenizer used by the length filter.

    Uses ``transformers.AutoTokenizer`` (already a transitive
    dependency via ``huggingface-hub``) so the prepare layer has
    no MLX dependency of its own. Raises :class:`PrepareError`
    with an actionable message if the install is missing
    ``transformers``.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        from hone.prepare.service import DataError

        raise DataError(
            "transformers is required for token-length filtering; "
            "install with `uv pip install '.[dev]'` or pass max_tokens=0"
        ) from error

    tokenizer: TokenizerLike = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )
    return tokenizer


__all__ = ["TokenFilter", "TokenizerLike", "load_tokenizer"]
