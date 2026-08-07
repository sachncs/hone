"""Streaming JSONL persistence for validated training examples."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from hone.model import Example, Message, Role
from hone.types import JsonScalar


class Reader:
    """Read training examples from a UTF-8 JSONL file."""

    def read(self, path: Path) -> Iterator[Example]:
        """Yield validated examples; raise on malformed lines.

        Each error message is prefixed with `<path>:<line>: ...` so
        callers can locate the offending record quickly.
        """
        with path.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
                yield self.record(path, line_number, record)

    def record(self, path: Path, line_number: int, record: object) -> Example:
        """Validate one parsed JSON record into an Example.

        Public per AGENTS.md no-semi-private rule. Exposed for
        callers that already hold a parsed record and want to
        reuse the same validation logic as read().
        """
        if not isinstance(record, dict):
            raise ValueError(
                f"{path}:{line_number}: each JSONL record must be an object"
            )
        raw_messages = record.get("messages")
        if not isinstance(raw_messages, list):
            raise ValueError(
                f"{path}:{line_number}: record must contain a 'messages' list"
            )
        messages: list[Message] = []
        for index, raw_message in enumerate(raw_messages):
            if not isinstance(raw_message, dict):
                raise ValueError(
                    f"{path}:{line_number}: messages[{index}] must be an object"
                )
            if "role" not in raw_message:
                raise ValueError(
                    f"{path}:{line_number}: messages[{index}] is missing 'role'"
                )
            if "content" not in raw_message:
                raise ValueError(
                    f"{path}:{line_number}: messages[{index}] is missing 'content'"
                )
            try:
                role = Role(str(raw_message["role"]))
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: messages[{index}].role: {error}"
                ) from error
            content = str(raw_message["content"])
            if not content.strip():
                raise ValueError(
                    f"{path}:{line_number}: messages[{index}].content is empty"
                )
            messages.append(Message(role=role, content=content))
        metadata: dict[str, JsonScalar] = {
            key: value
            for key, value in record.items()
            if key != "messages"
            and isinstance(value, (str, int, float, bool, type(None)))
        }
        return Example(messages=tuple(messages), metadata=metadata)


class Writer:
    """Write validated training examples to deterministic UTF-8 JSONL."""

    def write(self, path: Path, examples: Iterable[Example]) -> int:
        """Write examples and return the number of records written.

        Creates parent directories if missing. UTF-8 encoded.
        Keys sorted alphabetically for deterministic output.
        """
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
