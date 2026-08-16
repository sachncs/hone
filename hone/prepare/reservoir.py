"""Deterministic reservoir sampling.

Vitter's Algorithm R exposed as a small object: feed items one at
a time, get back a uniformly-random sample of size ``cap`` after
``n`` items have been seen (for any ``n >= cap``).

Use :meth:`Reservoir.observe` for each row, then :meth:`sample` to
retrieve the kept rows. Order inside the sample is randomized by
:meth:`shuffle` so callers do not have to.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator


class Reservoir[T]:
    """Uniform-random reservoir sampler with bounded memory."""

    def __init__(self, cap: int, *, seed: int) -> None:
        if cap < 1:
            raise ValueError(f"reservoir cap must be positive, got {cap}")
        self._cap = cap
        self._rng = random.Random(seed)
        self._items: list[T] = []
        self._seen = 0

    def observe(self, item: T) -> None:
        """Incorporate one item into the reservoir."""
        self._seen += 1
        if len(self._items) < self._cap:
            self._items.append(item)
        else:
            index = self._rng.randrange(self._seen)
            if index < self._cap:
                self._items[index] = item

    def extend(self, items: Iterable[T]) -> None:
        """Observe every item from an iterable in order."""
        for item in items:
            self.observe(item)

    def sample(self) -> list[T]:
        """Return the current reservoir contents (insertion order)."""
        return list(self._items)

    def shuffle(self) -> list[T]:
        """Return the reservoir contents, shuffled in place."""
        self._rng.shuffle(self._items)
        return self.sample()

    @property
    def seen(self) -> int:
        """Number of items observed so far."""
        return self._seen

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)


__all__ = ["Reservoir"]
