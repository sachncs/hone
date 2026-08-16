"""Row mappers that convert raw HF rows into chat-format records.

Two mappers today:

* :func:`as_sft` — turns an HF row into a chat-format record
  (``{"messages": [...]}``). Falls back through common column
  aliases (``prompt``/``input``/``instruction``, ``completion``/
  ``response``/``output``) when a ``messages`` field is missing.
* :func:`as_codeforces_text` — collapses the five canonical
  Codeforces problem fields into a single ``{"text": ...}``
  record for plain-text pre-training.

Both return ``None`` for rows that cannot be mapped; the prepare
service counts and skips them.
"""

from __future__ import annotations

from collections.abc import Mapping

_PROMPT_KEYS: tuple[str, ...] = (
    "prompt",
    "input",
    "instruction",
    "question",
    "problem",
)
_ANSWER_KEYS: tuple[str, ...] = (
    "completion",
    "response",
    "output",
    "answer",
    "solution",
)
_ROLE_MAP: dict[str, str] = {
    "human": "user",
    "user": "user",
    "system": "system",
    "assistant": "assistant",
    "gpt": "assistant",
    "bot": "assistant",
}


def as_sft(row: Mapping[str, object]) -> dict[str, object] | None:
    """Map an HF row to chat-format or ``None`` if not mappable."""
    messages = row.get("messages")
    if isinstance(messages, list):
        mapped = _messages_from_list(messages)
        if mapped is None:
            return None
        if len(mapped) >= 2 and mapped[-1]["role"] == "assistant":
            return {"messages": mapped}
        return None
    prompt, answer = _prompt_completion(row)
    if prompt is None or answer is None or isinstance(answer, (dict, list)):
        return None
    prompt_text = str(prompt).strip()
    answer_text = str(answer).strip()
    if not prompt_text or not answer_text:
        return None
    return {
        "messages": [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": answer_text},
        ]
    }


def as_codeforces_text(row: Mapping[str, object]) -> dict[str, object]:
    """Concatenate the five canonical Codeforces fields into ``{"text": ...}``.

    Raises :class:`ValueError` when no serializable problem text is
    present so the prepare service can count and skip.
    """
    sections = [
        ("TITLE", row.get("title")),
        ("DESCRIPTION", row.get("description")),
        ("INPUT FORMAT", row.get("input_format")),
        ("OUTPUT FORMAT", row.get("output_format")),
        ("EDITORIAL", row.get("editorial")),
    ]
    text = "\n\n".join(
        f"## {name}\n{value}".strip()
        for name, value in sections
        if value and str(value).strip()
    )
    if not text.strip():
        raise ValueError("row has no serializable problem text")
    return {"text": text}


def _messages_from_list(raw_messages: list[object]) -> list[dict[str, str]] | None:
    """Translate a row's ``messages`` list into the canonical chat schema."""
    out: list[dict[str, str]] = []
    for message in raw_messages:
        if not isinstance(message, Mapping):
            return None
        raw_role = message.get("role")
        content = message.get("content")
        if raw_role is None or content is None:
            return None
        role = _ROLE_MAP.get(str(raw_role).lower())
        if role is None:
            return None
        text = str(content).strip()
        if not text:
            return None
        out.append({"role": role, "content": text})
    return out


def _prompt_completion(
    row: Mapping[str, object],
) -> tuple[object | None, object | None]:
    """Return ``(prompt, answer)`` from the first non-empty alias."""
    prompt = next((row.get(key) for key in _PROMPT_KEYS if row.get(key)), None)
    answer = next((row.get(key) for key in _ANSWER_KEYS if row.get(key)), None)
    return prompt, answer


__all__ = ["as_codeforces_text", "as_sft"]
