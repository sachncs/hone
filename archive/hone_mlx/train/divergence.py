"""Divergence detection for staged training.

A clean run produces zero ``Train loss nan`` lines; long-tail
records truncated to an empty loss target yield them within the
first few iterations and the optimizer state never recovers.
Three or more NaN samples is treated as a hard stop.
"""

from __future__ import annotations

import re

_NAN_THRESHOLD: int = 3
_NAN_PATTERN: re.Pattern[str] = re.compile(r"Train loss nan")


class DivergenceDetector:
    """Decide whether a stage's captured log shows runaway NaN losses."""

    def __init__(self, *, threshold: int = _NAN_THRESHOLD) -> None:
        if threshold < 1:
            raise ValueError(f"threshold must be positive, got {threshold}")
        self._threshold = threshold

    def divergent(self, log_text: str) -> bool:
        """Return True when NaN samples meet or exceed the threshold."""
        return len(_NAN_PATTERN.findall(log_text)) >= self._threshold


__all__ = ["DivergenceDetector"]
