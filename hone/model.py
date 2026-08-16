"""Domain models for supervised fine-tuning records.

These dataclasses are the single source of truth for what a
fine-tuning example looks like in memory. The persistence layer
(JSONL) and the normalization layer (chat/prompt-completion sources)
both convert into and out of ``Example``; no other module should
need to know about JSONL structure or raw record shapes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from hone.errors import ValidationError

# A scalar value safe to round-trip through JSON: str/int/float/bool/None.
type JsonScalar = str | int | float | bool | None

# A bag of scalar metadata associated with an example. The keys are
# arbitrary (instance_id, source URL, token count, ...); values are
# restricted to JSON scalars so the record round-trips losslessly.
type Meta = Mapping[str, JsonScalar]


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
            raise ValidationError("message content cannot be empty")
        # Strip on construction; ``frozen=True`` requires object.__setattr__.
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
            raise ValidationError(
                "a training example needs at least two messages, "
                f"got {len(self.messages)}"
            )
        if self.messages[-1].role is not Role.assistant:
            raise ValidationError(
                "a training example must end with an assistant message, "
                f"got role {self.messages[-1].role!r}"
            )
        # Defensive copy: callers may mutate their input dict after
        # construction; we hold an immutable Mapping so post-construction
        # mutations don't leak into the example.
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
        # Re-apply metadata; this widens to dict[str, object] at the boundary.
        record.update(self.metadata)
        return record


__all__ = ["Example", "JsonScalar", "Message", "Meta", "Role"]
