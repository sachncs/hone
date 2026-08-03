"""Adapters from common source-record shapes into domain models."""

from __future__ import annotations

from collections.abc import Mapping

from finetune.models import ChatMessage, MessageRole, TrainingExample


class TrainingExampleNormalizer:
    """Normalize chat or prompt/completion records into one data contract."""

    def normalize(self, record: Mapping[str, object]) -> TrainingExample:
        """Convert a source record or raise a descriptive validation error."""
        raw_messages = record.get("messages")
        if isinstance(raw_messages, list):
            messages = self.__normalize_messages(raw_messages)
        elif "prompt" in record and "completion" in record:
            messages = (
                ChatMessage(MessageRole.USER, str(record["prompt"])),
                ChatMessage(MessageRole.ASSISTANT, str(record["completion"])),
            )
        else:
            raise ValueError("expected messages or prompt/completion")
        return TrainingExample(messages=messages, metadata={})

    def __normalize_messages(self, raw_messages: list[object]) -> tuple[ChatMessage, ...]:
        messages: list[ChatMessage] = []
        for raw_message in raw_messages:
            if not isinstance(raw_message, Mapping):
                raise ValueError("messages must contain objects")
            try:
                role = MessageRole(str(raw_message["role"]))
                content = str(raw_message["content"])
            except KeyError as error:
                raise ValueError(f"message is missing {error.args[0]}") from error
            messages.append(ChatMessage(role=role, content=content))
        return tuple(messages)


class SWETrainingExampleNormalizer:
    """Convert official SWE-bench rows into patch-generation examples."""

    def normalize(self, record: Mapping[str, object]) -> TrainingExample:
        """Build a prompt containing repository context and the issue text."""
        statement = str(record.get("problem_statement", "")).strip()
        if not statement:
            raise ValueError("missing problem_statement")
        patch = str(record.get("patch", "")).strip()
        if not patch:
            raise ValueError("missing patch")
        prompt = (
            "You are repairing a real software repository. Return only a unified "
            "diff patch; do not explain the answer.\n\n"
            f"Repository: {str(record.get('repo', '')).strip()}\n"
            f"Version: {str(record.get('version', '')).strip()}\n\n"
            f"Issue:\n{statement}"
        )
        metadata = {
            "instance_id": str(record["instance_id"])
            if record.get("instance_id") is not None
            else None
        }
        return TrainingExample(
            messages=(
                ChatMessage(MessageRole.USER, prompt),
                ChatMessage(MessageRole.ASSISTANT, patch),
            ),
            metadata=metadata,
        )
