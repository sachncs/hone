"""Deterministic dataset splitting with explicit reproducibility controls."""

from __future__ import annotations

import random
from collections.abc import Sequence

from finetune.models import TrainingExample


class DatasetSplitter:
    """Split examples into disjoint train and validation partitions."""

    def __init__(self, validation_ratio: float, seed: int) -> None:
        if not 0 < validation_ratio < 1:
            raise ValueError("validation ratio must be between 0 and 1")
        self.validation_ratio = validation_ratio
        self.seed = seed

    def split(
        self, examples: Sequence[TrainingExample]
    ) -> tuple[list[TrainingExample], list[TrainingExample]]:
        """Return shuffled ``(train, validation)`` partitions."""
        if len(examples) < 2:
            raise ValueError("at least two examples are required")
        shuffled = list(examples)
        random.Random(self.seed).shuffle(shuffled)
        validation_count = max(1, round(len(shuffled) * self.validation_ratio))
        validation = shuffled[:validation_count]
        train = shuffled[validation_count:]
        return train, validation
