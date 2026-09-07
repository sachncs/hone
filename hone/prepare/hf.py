"""HuggingFace adapter.

A thin wrapper over the ``datasets`` library that returns an
``Iterable[dict]`` for a given ``(repo, config, split)`` so the
prepare service does not have to import ``datasets`` directly.

The functions are deliberately narrow: callers pass strings, get
back a stream, and never see a ``Dataset`` object.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any


class HubStream:
    """Streaming iterable over one HF dataset split."""

    def __init__(
        self,
        repo: str,
        *,
        config: str | None = None,
        split: str = "train",
    ) -> None:
        self._repo = repo
        self._config = config
        self._split = split

    def __iter__(self) -> Iterator[dict[str, Any]]:
        from datasets import load_dataset

        if self._config is None:
            dataset = load_dataset(self._repo, split=self._split, streaming=True)
        else:
            dataset = load_dataset(
                self._repo,
                name=self._config,
                split=self._split,
                streaming=True,
            )
        yield from dataset


def load_split(
    repo: str,
    *,
    config: str | None = None,
    split: str = "train",
) -> Iterable[dict[str, Any]]:
    """Materialize one HF split fully (non-streaming)."""
    from datasets import load_dataset

    if config is None:
        dataset: Iterable[dict[str, Any]] = load_dataset(repo, split=split)
        return dataset
    dataset = load_dataset(repo, name=config, split=split)
    return dataset


__all__ = ["HubStream", "load_split"]
