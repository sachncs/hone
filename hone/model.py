"""Domain models for supervised fine-tuning records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


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
