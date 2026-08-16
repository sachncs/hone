"""Shared chat-format parsing used by the persistence and normalization layers.

Both ``hone.jsonl.Reader`` and ``hone.normalize.Normalizer`` need to
turn a raw ``messages`` list (a list of ``{"role", "content"}``
dicts) into a tuple of validated ``Message`` objects. The previous
implementation duplicated that logic across the two modules; this
module is the single source of truth.

The function raises :class:`hone.errors.ValidationError` with a
``location`` attribute (``"<path>:<line>"`` or just ``"messages[N]"``)
so callers can attach a positional context without each having to
rebuild the error message.
"""

from __future__ import annotations

from collections.abc import Mapping

from hone.errors import ValidationError
from hone.model import Message, Role


def parse_messages(
    raw_messages: object,
    *,
    location: str = "messages",
) -> tuple[Message, ...]:
    """Validate a chat-format ``messages`` list into a tuple of ``Message``.

    Args:
        raw_messages: The candidate value; must be a list of dicts each
            carrying ``role`` and ``content``.
        location: A short prefix used in error messages so callers can
            pinpoint the offending record (e.g. ``"data.jsonl:42"`` or
            ``"messages"``).

    Raises:
        ValidationError: If the input is the wrong shape, a message is
            missing ``role``/``content``, the role is unknown, or the
            content is empty.
    """
    if not isinstance(raw_messages, list):
        kind = type(raw_messages).__name__
        raise ValidationError(f"{location}: expected a list, got {kind}")
    messages: list[Message] = []
    for index, raw_message in enumerate(raw_messages):
        messages.append(_parse_one(raw_message, index, location))
    return tuple(messages)


def _parse_one(raw_message: object, index: int, location: str) -> Message:
    """Parse a single message dict at ``index`` inside ``location``."""
    if not isinstance(raw_message, Mapping):
        raise ValidationError(
            f"{location}[{index}] must be an object, got {type(raw_message).__name__}"
        )
    if "role" not in raw_message:
        raise ValidationError(f"{location}[{index}] is missing 'role'")
    if "content" not in raw_message:
        raise ValidationError(f"{location}[{index}] is missing 'content'")
    try:
        role = Role(str(raw_message["role"]))
    except ValueError as error:
        raise ValidationError(f"{location}[{index}].role: {error}") from error
    content = str(raw_message["content"])
    if not content.strip():
        raise ValidationError(f"{location}[{index}].content is empty")
    return Message(role=role, content=content)


def parse_messages_from_record(
    record: Mapping[str, object],
    *,
    location: str = "record",
) -> tuple[Message, ...]:
    """Extract and validate the ``messages`` field of a record mapping."""
    raw_messages = record.get("messages")
    if raw_messages is None:
        raise ValidationError(f"{location}: missing 'messages'")
    return parse_messages(raw_messages, location=f"{location}.messages")


__all__ = ["parse_messages", "parse_messages_from_record"]
