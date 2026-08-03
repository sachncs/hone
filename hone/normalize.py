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
            return Example(messages=self.chat(raw_messages), metadata={})
        if "prompt" in record and "completion" in record:
            return Example(
                messages=(
                    Message(Role.user, str(record["prompt"])),
                    Message(Role.assistant, str(record["completion"])),
                ),
                metadata={},
            )
        raise ValueError("expected 'messages' list or 'prompt'/'completion' pair")

    @staticmethod
    def chat(raw_messages: list[object]) -> tuple[Message, ...]:
        """Validate a chat-style messages list into a tuple of Message.

        Public per AGENTS.md no-semi-private rule. Treat as the
        chat-format implementation of normalize; prefer normalize
        for new call sites.
        """
        messages: list[Message] = []
        for index, raw in enumerate(raw_messages):
            if not isinstance(raw, Mapping):
                raise ValueError(f"messages[{index}] must be an object")
            if "role" not in raw:
                raise ValueError(f"messages[{index}] is missing 'role'")
            if "content" not in raw:
                raise ValueError(f"messages[{index}] is missing 'content'")
            try:
                role = Role(str(raw["role"]))
            except ValueError as error:
                raise ValueError(f"messages[{index}].role: {error}") from error
            messages.append(Message(role=role, content=str(raw["content"])))
        return tuple(messages)
