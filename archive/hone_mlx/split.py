"""Deterministic dataset splitting with explicit reproducibility controls.

Two operations, one module:

* :class:`Splitter` — in-memory split of an ``Example`` sequence into
  disjoint train and validation partitions.
* :func:`partition_file` — streaming equivalent that holds the
  validation reservoir in memory and writes train and valid files
  byte-identical to the source. The implementation is exposed via
  the :class:`Partitioner` class so it can be tested without
  reaching into module-level helpers.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path

from hone.errors import ValidationError
from hone.model import Example

MIN_VALID: int = 1


class Splitter:
    """Split examples into disjoint train and validation partitions."""

    def __init__(self, ratio: float, seed: int) -> None:
        if not 0 < ratio < 1:
            raise ValidationError(
                f"ratio must be between 0 and 1 (exclusive), got {ratio}"
            )
        self.ratio = ratio
        self.seed = seed

    def split(self, examples: Sequence[Example]) -> tuple[list[Example], list[Example]]:
        """Return shuffled (train, valid) partitions.

        Preconditions:
        - ``examples`` has at least 2 entries.

        Postconditions:
        - train and valid are disjoint.
        - ``len(train) + len(valid) == len(examples)``.
        - ``len(valid) >= MIN_VALID`` when ``len(examples) >= 2``.
        - Order is deterministic for a given seed.
        """
        if len(examples) < 2:
            raise ValidationError(
                f"at least two examples are required, got {len(examples)}"
            )
        shuffled = list(examples)
        random.Random(self.seed).shuffle(shuffled)
        valid_count = max(MIN_VALID, round(len(shuffled) * self.ratio))
        valid = shuffled[:valid_count]
        train = shuffled[valid_count:]
        return train, valid


class Partitioner:
    """Streaming JSONL → disjoint train/valid partitioner.

    Reservoir-samples the validation subset with a seeded RNG so the
    split is deterministic for a given input order and memory stays
    bounded by the validation size rather than the dataset size.
    """

    def __init__(self, ratio: float, seed: int) -> None:
        if not 0 < ratio < 1:
            raise ValidationError(
                f"ratio must be between 0 and 1 (exclusive), got {ratio}"
            )
        self.ratio = ratio
        self.seed = seed

    def run(
        self,
        source: Path,
        train_path: Path,
        valid_path: Path,
    ) -> tuple[int, int]:
        """Partition ``source`` into ``train_path`` and ``valid_path``.

        Returns ``(train_count, valid_count)``.

        ``valid_path`` is promoted before ``train_path`` so a crash
        between the renames leaves the holdout on disk instead of
        dropping it; the worst case is a benign train/valid overlap,
        never data loss.
        """
        total = self._count(source)
        valid_count = max(MIN_VALID, round(total * self.ratio))
        if valid_count >= total:
            valid_count = total - 1

        reservoir: set[int] = self._reservoir(source, total, valid_count)
        self._write(source, train_path, valid_path, reservoir)
        return total - valid_count, valid_count

    @staticmethod
    def _count(source: Path) -> int:
        """Count JSON lines in ``source``, raising on malformed lines."""
        total = 0
        with source.open(encoding="utf-8", newline="") as input_file:
            for line_number, line in enumerate(input_file, 1):
                try:
                    json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValidationError(f"{source}:{line_number}: {error}") from error
                total += 1
        if total < 2:
            raise ValidationError(f"at least two JSON lines are required, got {total}")
        return total

    def _reservoir(self, source: Path, total: int, valid_count: int) -> set[int]:
        """Reservoir-sample ``valid_count`` line numbers out of ``source``."""
        rng = random.Random(self.seed)
        reservoir: list[int] = []
        with source.open(encoding="utf-8", newline="") as input_file:
            for line_number, _ in enumerate(input_file, 1):
                if len(reservoir) < valid_count:
                    reservoir.append(line_number)
                else:
                    index = rng.randrange(line_number)
                    if index < valid_count:
                        reservoir[index] = line_number
        if len(reservoir) != valid_count:
            raise ValidationError(
                f"reservoir under-filled: wanted {valid_count}, "
                f"got {len(reservoir)} of {total}"
            )
        return set(reservoir)

    @staticmethod
    def _write(
        source: Path,
        train_path: Path,
        valid_path: Path,
        valid_lines: set[int],
    ) -> None:
        """Write ``source`` lines to ``train_path`` / ``valid_path`` disjoint."""
        train_tmp = train_path.with_suffix(".jsonl.tmp")
        valid_tmp = valid_path.with_suffix(".jsonl.tmp")
        train_path.parent.mkdir(parents=True, exist_ok=True)
        valid_path.parent.mkdir(parents=True, exist_ok=True)
        with (
            source.open(encoding="utf-8", newline="") as input_file,
            train_tmp.open("w", encoding="utf-8", newline="") as train_output,
            valid_tmp.open("w", encoding="utf-8", newline="") as valid_output,
        ):
            for line_number, line in enumerate(input_file, 1):
                target = valid_output if line_number in valid_lines else train_output
                target.write(line)
        valid_tmp.replace(valid_path)
        train_tmp.replace(train_path)


def partition_file(
    source: Path,
    train_path: Path,
    valid_path: Path,
    ratio: float,
    seed: int,
) -> tuple[int, int]:
    """Streaming partition helper; convenience wrapper over :class:`Partitioner`."""
    return Partitioner(ratio, seed).run(source, train_path, valid_path)


# Backwards-compat alias preserved for the public API; deprecated.
partition = partition_file


__all__ = ["MIN_VALID", "Partitioner", "Splitter", "partition_file"]
