"""Source-record normalizers that produce :class:`~hone.model.Example`.

Two concrete normalizers today:

* :class:`Normalizer` — accepts chat-format or prompt/completion
  records and emits a uniform ``Example``.
* :class:`SweNormalizer` — converts SWE-bench rows into
  patch-generation examples.

Both raise :class:`~hone.errors.ValidationError` for malformed
input so the CLI layer can catch a single exception class.
"""

from __future__ import annotations

from collections.abc import Mapping

from hone.chat import parse_messages
from hone.errors import ValidationError
from hone.model import Example, JsonScalar, Message, Role


class Normalizer:
    """Normalize chat or prompt/completion records into one data contract."""

    def normalize(self, record: Mapping[str, object]) -> Example:
        """Convert a source record or raise :class:`ValidationError`."""
        if "messages" in record:
            messages = parse_messages(record["messages"], location="messages")
            return Example(messages=messages, metadata={})
        if "prompt" in record and "completion" in record:
            return self._from_prompt_completion(record)
        raise ValidationError("expected 'messages' list or 'prompt'/'completion' pair")

    @staticmethod
    def _from_prompt_completion(record: Mapping[str, object]) -> Example:
        """Build a 2-message Example from a prompt/completion pair."""
        prompt = str(record["prompt"])
        completion = str(record["completion"])
        return Example(
            messages=(
                Message(Role.user, prompt),
                Message(Role.assistant, completion),
            ),
            metadata={},
        )


class SweNormalizer:
    """Convert official SWE-bench rows into patch-generation examples."""

    def normalize(self, record: Mapping[str, object]) -> Example:
        """Build a prompt containing repository context and the issue text."""
        statement = str(record.get("problem_statement", "")).strip()
        if not statement:
            raise ValidationError("missing problem_statement")
        patch = str(record.get("patch", "")).strip()
        if not patch:
            raise ValidationError("missing patch")
        prompt = (
            "You are repairing a real software repository. Return only a unified "
            "diff patch; do not explain the answer.\n\n"
            f"Repository: {str(record.get('repo', '')).strip()}\n"
            f"Version: {str(record.get('version', '')).strip()}\n\n"
            f"Issue:\n{statement}"
        )
        instance_id = record.get("instance_id")
        metadata: dict[str, JsonScalar] = {}
        if instance_id is not None:
            metadata["instance_id"] = str(instance_id)
        return Example(
            messages=(
                Message(Role.user, prompt),
                Message(Role.assistant, patch),
            ),
            metadata=metadata,
        )


__all__ = ["Normalizer", "SweNormalizer"]
