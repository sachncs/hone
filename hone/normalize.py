"""Source-record normalizers that produce Example objects."""

from __future__ import annotations

from collections.abc import Mapping

from hone.model import Example, Message, Role


class Normalizer:
    """Normalize chat or prompt/completion records into one data contract."""

    def normalize(self, record: Mapping[str, object]) -> Example:
        """Convert a source record or raise a descriptive validation error."""
        raw_messages = record.get("messages")
        if isinstance(raw_messages, list):
            messages = self._messages(raw_messages)
            return Example(messages=messages, metadata={})
        if "prompt" in record and "completion" in record:
            return Example(
                messages=(
                    Message(Role.user, str(record["prompt"])),
                    Message(Role.assistant, str(record["completion"])),
                ),
                metadata={},
            )
        raise ValueError("expected 'messages' list or 'prompt'/'completion' pair")

    def _messages(self, raw_messages: list[object]) -> tuple[Message, ...]:
        messages: list[Message] = []
        for index, raw_message in enumerate(raw_messages):
            if not isinstance(raw_message, Mapping):
                raise ValueError(f"messages[{index}] must be an object")
            if "role" not in raw_message:
                raise ValueError(f"messages[{index}] is missing 'role'")
            if "content" not in raw_message:
                raise ValueError(f"messages[{index}] is missing 'content'")
            try:
                role = Role(str(raw_message["role"]))
            except ValueError as error:
                raise ValueError(
                    f"messages[{index}].role: {error}"
                ) from error
            messages.append(
                Message(role=role, content=str(raw_message["content"]))
            )
        return tuple(messages)
