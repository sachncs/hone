"""Behavior tests for hone.prepare.token_filter.TokenFilter."""

from __future__ import annotations

from hone.prepare.token_filter import TokenFilter


class _FixedTokenizer:
    """Tokenizer stub whose encode yields ``range(len(text))``."""

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return list(range(len(text)))


def test_filter_disabled_when_max_tokens_zero() -> None:
    filter_ = TokenFilter(max_tokens=0, tokenizer=_FixedTokenizer())
    assert not filter_.enabled
    assert not filter_.too_long({"text": "x" * 10_000})


def test_filter_disabled_when_tokenizer_none() -> None:
    filter_ = TokenFilter(max_tokens=10, tokenizer=None)
    assert not filter_.enabled
    assert not filter_.too_long({"text": "x" * 10_000})


def test_filter_enabled_when_max_tokens_positive_and_tokenizer_present() -> None:
    filter_ = TokenFilter(max_tokens=100, tokenizer=_FixedTokenizer())
    assert filter_.enabled


def test_filter_rejects_oversize_record() -> None:
    filter_ = TokenFilter(max_tokens=10, tokenizer=_FixedTokenizer())
    assert filter_.too_long({"text": "x" * 100})


def test_filter_accepts_undersize_record() -> None:
    filter_ = TokenFilter(max_tokens=100, tokenizer=_FixedTokenizer())
    assert not filter_.too_long({"text": "x" * 50})


def test_filter_concatenates_messages_text() -> None:
    filter_ = TokenFilter(max_tokens=10, tokenizer=_FixedTokenizer())
    record = {
        "messages": [
            {"role": "user", "content": "x" * 50},
            {"role": "assistant", "content": "y" * 50},
        ]
    }
    assert filter_.too_long(record)


def test_filter_rejects_negative_max_tokens() -> None:
    import pytest

    with pytest.raises(ValueError, match="non-negative"):
        TokenFilter(max_tokens=-1, tokenizer=None)
