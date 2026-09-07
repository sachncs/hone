"""Training-stage value objects and selection logic.

A :class:`Stage` is one row of the full training sequence: which
HF dataset to pull, which configs to use, where to write the
prepared JSONL, and where to write the resulting LoRA adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Stage:
    """One row of the multi-stage training sequence."""

    repo: str
    """HuggingFace repo ID to pull the raw rows from."""

    configs: str
    """Comma-separated HF configs to iterate."""

    data_path: Path
    """Where the prepared JSONL should land."""

    adapter_path: Path
    """Where the trained LoRA adapter should land."""


def full_sequence() -> tuple[Stage, ...]:
    """Return the canonical 5-stage training sequence.

    The order and choice of datasets is part of hone's public
    contract — ``./train.sh`` runs them in this exact order with
    each stage resuming the previous stage's adapter.
    """
    return (
        Stage(
            repo="ianncity/KIMI-K2.5-1000000x",
            configs="General-Distillation,PHD-Science,General-Math,MultilingualSTEM",
            data_path=Path("data/full/kimi/train.jsonl"),
            adapter_path=Path("artifacts/full/01-kimi"),
        ),
        Stage(
            repo="Modotte/CodeX-7M-Non-Thinking",
            configs="default",
            data_path=Path("data/full/codex/train.jsonl"),
            adapter_path=Path("artifacts/full/02-codex"),
        ),
        Stage(
            repo="inclusionAI/Ling-Coder-SFT",
            configs="default",
            data_path=Path("data/full/ling/train.jsonl"),
            adapter_path=Path("artifacts/full/03-ling"),
        ),
        Stage(
            repo="open-r1/codeforces",
            configs="default",
            data_path=Path("data/full/codeforces/train.jsonl"),
            adapter_path=Path("artifacts/full/04-codeforces"),
        ),
        Stage(
            repo="microsoft/rStar-Coder",
            configs="seed_sft",
            data_path=Path("data/full/rstar/train.jsonl"),
            adapter_path=Path("artifacts/full/05-rstar"),
        ),
    )


def select_stages(selector: str) -> list[Stage]:
    """Resolve a ``--stages`` CLI selector against the full sequence.

    Accepts:

    * empty string or ``"all"`` — every stage, in order.
    * comma-separated 1-based indices (``"02,03,04"``).
    * comma-separated stage directory names (``"02-codex,03-ling"``).

    Returns the deduplicated subset, preserving full-sequence
    order. Raises :class:`ValueError` for unknown indices / names
    or empty selections.
    """
    sequence = full_sequence()
    text = selector.strip().lower()
    if text in {"", "all"}:
        return list(sequence)

    seen: set[Path] = set()
    selected: list[Stage] = []
    for token in selector.split(","):
        key = token.strip()
        if not key:
            continue
        if key.isdigit():
            index = int(key) - 1
            if index < 0 or index >= len(sequence):
                raise ValueError(
                    f"stages index out of range: {key} (valid 1..{len(sequence)})"
                )
            stage = sequence[index]
        else:
            matches = [stage for stage in sequence if stage.adapter_path.name == key]
            if not matches:
                raise ValueError(
                    f"stages name not found: {key}; "
                    f"valid names: {[s.adapter_path.name for s in sequence]}"
                )
            stage = matches[0]
        if stage.adapter_path in seen:
            continue
        seen.add(stage.adapter_path)
        selected.append(stage)
    if not selected:
        raise ValueError("stages produced an empty selection")
    return selected


__all__ = ["Stage", "full_sequence", "select_stages"]
