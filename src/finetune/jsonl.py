"""Streaming JSONL persistence for validated training examples."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from finetune.models import ChatMessage, MessageRole, TrainingExample


class JsonlDatasetReader:
    """Read training examples from a UTF-8 JSONL file."""

    def read(self, path: Path) -> Iterator[TrainingExample]:
        """Yield validated examples and report the failing line number."""
        with path.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    yield self.__parse_record(record)
                except (TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error

    def __parse_record(self, record: object) -> TrainingExample:
        if not isinstance(record, dict):
            raise ValueError("each JSONL record must be an object")
        raw_messages = record.get("messages")
        if not isinstance(raw_messages, list):
            raise ValueError("record must contain a messages list")
        messages: list[ChatMessage] = []
        for raw_message in raw_messages:
            if not isinstance(raw_message, dict):
                raise ValueError("messages must contain objects")
            try:
                role = MessageRole(str(raw_message["role"]))
                content = str(raw_message["content"])
            except KeyError as error:
                raise ValueError(f"message is missing {error.args[0]}") from error
            messages.append(ChatMessage(role=role, content=content))
        metadata = {
            key: value
            for key, value in record.items()
            if key != "messages" and isinstance(value, (str, int, float, bool, type(None)))
        }
        return TrainingExample(messages=tuple(messages), metadata=metadata)


class JsonlDatasetWriter:
    """Write validated training examples to deterministic UTF-8 JSONL."""

    def write(self, path: Path, examples: Iterable[TrainingExample]) -> int:
        """Write examples and return the number of records written."""
        path.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with path.open("w", encoding="utf-8") as output_file:
            for example in examples:
                output_file.write(
                    json.dumps(example.to_record(), ensure_ascii=False, sort_keys=True)
                    + "\n"
                )
                written += 1
        return written
