"""Immutable domain models for supervised fine-tuning records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

JsonValue: TypeAlias = str | int | float | bool | None
Metadata: TypeAlias = dict[str, JsonValue]


class MessageRole(StrEnum):
    """Roles accepted by the model chat template."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """A non-empty message in a model conversation."""

    role: MessageRole
    content: str

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("message content cannot be empty")
        object.__setattr__(self, "content", self.content.strip())

    def to_record(self) -> dict[str, str]:
        """Serialize the message for JSONL and tokenizer APIs."""
        return {"role": self.role.value, "content": self.content}


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """A validated supervised fine-tuning example."""

    messages: tuple[ChatMessage, ...]
    metadata: Metadata

    def __post_init__(self) -> None:
        if len(self.messages) < 2:
            raise ValueError("a training example needs at least two messages")
        if self.messages[-1].role is not MessageRole.ASSISTANT:
            raise ValueError("a training example must end with an assistant message")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_record(self) -> dict[str, object]:
        """Serialize the example while preserving optional metadata."""
        record: dict[str, object] = {
            "messages": [message.to_record() for message in self.messages]
        }
        record.update(self.metadata)
        return record

    @property
    def character_count(self) -> int:
        """Return the total un-tokenized conversation length."""
        return sum(len(message.content) for message in self.messages)
