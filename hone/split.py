"""Deterministic dataset splitting with explicit reproducibility controls."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path

from hone.model import Example

MIN_VALID: int = 1


class Splitter:
    """Split examples into disjoint train and validation partitions."""

    def __init__(self, ratio: float, seed: int) -> None:
        if not 0 < ratio < 1:
            raise ValueError(f"ratio must be between 0 and 1 (exclusive), got {ratio}")
        self.ratio = ratio
        self.seed = seed

    def split(self, examples: Sequence[Example]) -> tuple[list[Example], list[Example]]:
        """Return shuffled (train, valid) partitions.

        Preconditions:
        - examples has at least 2 entries.

        Postconditions:
        - train and valid are disjoint.
        - len(train) + len(valid) == len(examples).
        - len(valid) >= MIN_VALID when len(examples) >= 2 and ratio > 0.
        - Order is deterministic for a given seed.
        """
        if len(examples) < 2:
            raise ValueError(f"at least two examples are required, got {len(examples)}")
        shuffled = list(examples)
        random.Random(self.seed).shuffle(shuffled)
        valid_count = max(MIN_VALID, round(len(shuffled) * self.ratio))
        valid = shuffled[:valid_count]
        train = shuffled[valid_count:]
        return train, valid


def partition(
    file: Path,
    train_path: Path,
    valid_path: Path,
    ratio: float,
    seed: int,
) -> tuple[int, int]:
    """Stream a JSONL file into disjoint train and valid partitions.

    Reservoir-samples the validation subset with a seeded RNG, so the
    split is deterministic for a given input order and memory stays
    bounded by the validation size rather than the dataset size.

    Preconditions:
    - Every line of file is valid JSON (a malformed or truncated
      line raises ValueError with a ``path:line`` prefix).
    - file has at least 2 lines.
    - ratio is in (0, 1) exclusive.

    Postconditions:
    - train_path and valid_path hold raw file lines, byte-identical.
    - train and valid are disjoint and together hold every file line.
    - len(valid) >= MIN_VALID and len(train) >= 1.
    - The split is deterministic for a given file order and seed;
      regenerating the file (e.g. upstream dataset drift) can change
      which lines land in valid.

    valid_path is promoted before train_path so a crash between the
    renames leaves the holdout on disk instead of dropping it; the
    worst case is a benign train/valid overlap, never data loss.

    Returns (train_count, valid_count).
    """
    if not 0 < ratio < 1:
        raise ValueError(f"ratio must be between 0 and 1 (exclusive), got {ratio}")

    total = count(file)
    if total < 2:
        raise ValueError(f"at least two JSON lines are required, got {total}")
    valid_count = max(MIN_VALID, round(total * ratio))
    if valid_count >= total:
        valid_count = total - 1

    rng = random.Random(seed)
    reservoir: list[int] = []
    with file.open(encoding="utf-8", newline="") as input_file:
        for line_number, _ in enumerate(input_file, 1):
            if len(reservoir) < valid_count:
                reservoir.append(line_number)
            else:
                index = rng.randrange(line_number)
                if index < valid_count:
                    reservoir[index] = line_number

    write(file, train_path, valid_path, set(reservoir))
    return total - valid_count, valid_count


def count(lines: Path) -> int:
    """Return the number of JSON lines in a JSONL file.

    Treat as internal: exposed publicly per the no-semi-private rule.
    Raises ValueError with a ``path:line`` prefix on the first
    malformed line so corrupt or truncated files fail close to their
    source instead of crashing downstream JSONL consumers.
    """
    total = 0
    with lines.open(encoding="utf-8", newline="") as input_file:
        for line_number, line in enumerate(input_file, 1):
            try:
                json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{lines}:{line_number}: {error}") from error
            total += 1
    return total


def write(
    jsonl: Path,
    path: Path,
    valid: Path,
    valid_lines: set[int],
) -> None:
    """Write a JSONL file's lines into disjoint train and valid files.

    Treat as internal: exposed publicly per the no-semi-private rule.
    Lines whose 1-based line number is in valid_lines go to the valid
    file and the rest go to the train file at path; raw lines are
    preserved byte-identical. The valid file is promoted before the
    train file so a crash between the renames leaves the holdout on
    disk instead of dropping it.
    """
    train = path.with_suffix(".jsonl.tmp")
    valid_path_tmp = valid.with_suffix(".jsonl.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    valid.parent.mkdir(parents=True, exist_ok=True)
    with (
        jsonl.open(encoding="utf-8", newline="") as input_file,
        train.open("w", encoding="utf-8", newline="") as train_output,
        valid_path_tmp.open("w", encoding="utf-8", newline="") as valid_output,
    ):
        for line_number, line in enumerate(input_file, 1):
            (valid_output if line_number in valid_lines else train_output).write(line)
    valid_path_tmp.replace(valid)
    train.replace(path)
