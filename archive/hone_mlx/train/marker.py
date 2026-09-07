"""Sidecar marker bookkeeping for prepared JSONLs.

When :func:`hone.prepare.prepare_stream` runs with
``--max-tokens N`` set, the resulting JSONL must be rebuilt if a
later run requests a different ``N``. The sidecar marker records
the values used so the orchestrator can detect a stale JSONL
without re-reading it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from hone.errors import PipelineError


@dataclass(frozen=True)
class StageMarker:
    """Sidecar metadata for one prepared JSONL."""

    data_path: Path
    max_tokens: int
    max_samples: int

    @property
    def path(self) -> Path:
        """Where the sidecar JSON lives on disk."""
        return self.data_path.parent / (
            f".prepared-max-tokens-{self.max_tokens}-samples-{self.max_samples}.json"
        )

    def exists(self) -> bool:
        """Whether the sidecar JSON is present."""
        return self.path.is_file()

    def read(self) -> dict[str, object]:
        """Read the sidecar JSON; raise if it is missing or malformed."""
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise PipelineError(f"stage marker missing on disk: {self.path}") from error
        except json.JSONDecodeError as error:
            raise PipelineError(
                f"stage marker at {self.path} is not valid JSON: {error}"
            ) from error
        if not isinstance(payload, dict):
            kind = type(payload).__name__
            raise PipelineError(
                f"stage marker at {self.path} must be an object, got {kind}"
            )
        return payload

    def write(self) -> None:
        """Persist the sidecar JSON to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "max_tokens": self.max_tokens,
                    "max_samples": self.max_samples,
                    "data": str(self.data_path),
                }
            ),
            encoding="utf-8",
        )


def needs_rebuild(data_path: Path, max_tokens: int, max_samples: int) -> bool:
    """Return True if ``data_path`` must be re-prepared.

    A rebuild is required when:

    * the JSONL is missing or empty, OR
    * any length filter is requested and no marker exists for
      the (max_tokens, max_samples) pair.
    """
    if not data_path.is_file() or data_path.stat().st_size == 0:
        return True
    if max_tokens <= 0 and max_samples <= 0:
        return False
    return not StageMarker(data_path, max_tokens, max_samples).exists()


__all__ = ["StageMarker", "needs_rebuild"]
