"""Domain models for supervised fine-tuning records."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias

from hone.types import JsonScalar

Scalar: TypeAlias = JsonScalar
Meta: TypeAlias = dict[str, JsonScalar]


class Role(StrEnum):
    """Roles accepted by chat-style fine-tuning records."""

    system = "system"
    user = "user"
    assistant = "assistant"


@dataclass(frozen=True, slots=True)
class Message:
    """A non-empty message in a chat conversation."""

    role: Role
    content: str

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("message content cannot be empty")
        object.__setattr__(self, "content", self.content.strip())

    def to_record(self) -> dict[str, str]:
        """Serialize the message for JSONL and tokenizer APIs."""
        return {"role": str(self.role), "content": self.content}


@dataclass(frozen=True, slots=True)
class Example:
    """A validated supervised fine-tuning example."""

    messages: tuple[Message, ...]
    metadata: Meta = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.messages) < 2:
            raise ValueError(
                "a training example needs at least two messages, "
                f"got {len(self.messages)}"
            )
        if self.messages[-1].role is not Role.assistant:
            raise ValueError(
                "a training example must end with an assistant message, "
                f"got role {self.messages[-1].role!r}"
            )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def character_count(self) -> int:
        """Return the total un-tokenized conversation length."""
        return sum(len(message.content) for message in self.messages)

    def to_record(self) -> dict[str, object]:
        """Serialize the example while preserving optional metadata."""
        record: dict[str, object] = {
            "messages": [message.to_record() for message in self.messages]
        }
        record.update(self.metadata)
        return record
