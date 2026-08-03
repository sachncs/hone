"""Deterministic dataset splitting with explicit reproducibility controls."""

from __future__ import annotations

import random
from collections.abc import Sequence

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
