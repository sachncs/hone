#!/usr/bin/env python3
"""Build the 60K-row 4-source combined dataset for the next experiment.

Pulls subsets of:
  * Modotte/CodeX-7M-Non-Thinking       — 25K rows
  * nvidia/Nemotron-SFT-Competitive-Programming-v2 — 15K rows
  * nvidia/Nemotron-SFT-SWE-v2         — 15K rows
  * inclusionAI/Ling-Coder-SFT          — 5K rows

Writes a shuffled train.jsonl + valid.jsonl (5% split) under
data/soup/nemotron-combined/.

The CodeX + Ling-Coder inputs are regenerated from HuggingFace
inside this script (via :func:`prepare_stream` and
:func:`prepare_ling_coder`) so the script is self-contained and
runs on a fresh clone without depending on previously-prepared
local files.

Usage:
    uv run python scripts/prep_nemotron_combined.py

This script is one-shot: it doesn't depend on the current
training run finishing. Run it now so the next SFT has its
data ready when the GPU frees up.
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Add repo root to path so 'hone.prepare' is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hone.prepare import (
    NEMOTRON_COMPETITIVE_PROGRAMMING,
    NEMOTRON_SWE,
    NemotronConfig,
    PrepareRequest,
    prepare_ling_coder,
    prepare_stream,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "soup"
COMBINED_DIR = DATA_DIR / "nemotron-combined"

# Per-source row caps. Total: 60K.
CAPS: dict[str, int] = {
    "codex": 25_000,
    "nemotron-cp": 15_000,
    "nemotron-swe": 15_000,
    "lingcoder": 5_000,
}


def _stratified_subsets() -> dict[str, list[dict]]:
    """Build per-source subsets.

    CodeX and Ling-Coder are streamed from HuggingFace into a
    short-lived working directory and then read back as JSONL;
    the two Nemotron configs stream directly via
    :func:`stream_nemotron`. The whole pipeline is self-contained
    so the script runs on a fresh clone without depending on
    previously-prepared local files.
    """
    subsets: dict[str, list[dict]] = {}
    rng = random.Random(42)

    # CodeX: stream from HF into a working directory and read back.
    codex_dir = DATA_DIR / "codex"
    if not (codex_dir / "train.jsonl").exists() or not (codex_dir / "valid.jsonl").exists():
        codex_dir.mkdir(parents=True, exist_ok=True)
        logging.info("codex: regenerating from Modotte/CodeX-7M-Non-Thinking")
        prepare_stream(
            request=PrepareRequest(output=codex_dir, seed=42),
            repo="Modotte/CodeX-7M-Non-Thinking",
            configs="default",
            mode="sft",
            max_samples=CAPS["codex"] * 2,
        )
    codex_rows = (codex_dir / "all.jsonl").read_text(encoding="utf-8").strip().split("\n")
    rng.shuffle(codex_rows)
    subsets["codex"] = [json.loads(line) for line in codex_rows[: CAPS["codex"]]]
    logging.info("codex: kept %d rows", len(subsets["codex"]))

    # Ling-Coder: regenerate from HF if the cached file is missing.
    ling_dir = DATA_DIR / "lingcoder"
    if not (ling_dir / "train.jsonl").exists() or not (ling_dir / "valid.jsonl").exists():
        ling_dir.mkdir(parents=True, exist_ok=True)
        logging.info("lingcoder: regenerating from inclusionAI/Ling-Coder-SFT")
        prepare_ling_coder(
            request=PrepareRequest(output=ling_dir, seed=42),
            max_samples=CAPS["lingcoder"] * 2,
        )
    ling_rows = (ling_dir / "train.jsonl").read_text(encoding="utf-8").strip().split("\n")
    rng.shuffle(ling_rows)
    subsets["lingcoder"] = [json.loads(line) for line in ling_rows[: CAPS["lingcoder"]]]
    logging.info("lingcoder: kept %d rows", len(subsets["lingcoder"]))

    # Nemotron-CP: stream + reservoir.
    from hone.prepare.nemotron import stream_nemotron

    nemotron_cp = list(
        stream_nemotron(NEMOTRON_COMPETITIVE_PROGRAMMING, max_rows=CAPS["nemotron-cp"], seed=42)
    )
    subsets["nemotron-cp"] = nemotron_cp
    logging.info("nemotron-cp: kept %d rows", len(nemotron_cp))

    # Nemotron-SWE: stream + reservoir.
    nemotron_swe = list(stream_nemotron(NEMOTRON_SWE, max_rows=CAPS["nemotron-swe"], seed=42))
    subsets["nemotron-swe"] = nemotron_swe
    logging.info("nemotron-swe: kept %d rows", len(nemotron_swe))

    return subsets


def _combine_and_split(subsets: dict[str, list[dict]]) -> tuple[Path, Path]:
    """Concatenate, shuffle, write train.jsonl + valid.jsonl at 5% split."""
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    combined: list[dict] = []
    for rows in subsets.values():
        combined.extend(rows)
    random.Random(42).shuffle(combined)
    split = int(len(combined) * 0.95)
    train_path = COMBINED_DIR / "train.jsonl"
    valid_path = COMBINED_DIR / "valid.jsonl"
    for path, items in ((train_path, combined[:split]), (valid_path, combined[split:])):
        with path.open("w", encoding="utf-8") as handle:
            for row in items:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        logging.info("wrote %d rows to %s", len(items), path)
    return train_path, valid_path


def main() -> int:
    logging.info("targeting 60K-row stratified subset across 4 sources")
    subsets = _stratified_subsets()
    train_path, valid_path = _combine_and_split(subsets)
    logging.info("done; train=%s valid=%s", train_path, valid_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
