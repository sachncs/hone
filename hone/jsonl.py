"""Streaming JSONL persistence for validated training examples.

Two responsibilities, two classes:

* :class:`Reader` — stream a JSONL file and yield validated
  :class:`~hone.model.Example` records, raising
  :class:`~hone.errors.ValidationError` with a ``<path>:<line>``
  prefix on any malformed input.
* :class:`Writer` — write a stream of :class:`~hone.model.Example`
  records to a UTF-8 JSONL file with sorted keys for deterministic
  output.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from hone.chat import parse_messages
from hone.errors import ValidationError
from hone.model import Example, JsonScalar

_Location = str


class Reader:
    """Read training examples from a UTF-8 JSONL file."""

    def read(self, path: Path) -> Iterator[Example]:
        """Yield validated examples; raise on malformed lines.

        Each error message is prefixed with ``<path>:<line>: ...``
        so callers can locate the offending record quickly.
        """
        with path.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValidationError(f"{path}:{line_number}: {error}") from error
                yield self._record(path, line_number, record)

    def record(self, path: Path, line_number: int, record: object) -> Example:
        """Validate one parsed JSON record into an Example.

        Public per the no-semi-private rule: callers that already
        hold a parsed record can reuse the same validation logic
        used by :meth:`read`.
        """
        return self._record(path, line_number, record)

    def _record(self, source: Path, line_number: int, record: object) -> Example:
        location: _Location = f"{source}:{line_number}"
        if not isinstance(record, dict):
            raise ValidationError(f"{location}: each JSONL record must be an object")
        raw_messages = record.get("messages")
        if not isinstance(raw_messages, list):
            raise ValidationError(f"{location}: record must contain a 'messages' list")
        messages = parse_messages(raw_messages, location=location)
        metadata: dict[str, JsonScalar] = {
            key: value
            for key, value in record.items()
            if key != "messages"
            and isinstance(value, (str, int, float, bool, type(None)))
        }
        return Example(messages=messages, metadata=metadata)


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


__all__ = ["Reader", "Writer"]
