"""Domain models for supervised fine-tuning records."""

from __future__ import annotations

from enum import StrEnum


class Role(StrEnum):
    """Roles accepted by chat-style fine-tuning records."""

    system = "system"
    user = "user"
    assistant = "assistant"
