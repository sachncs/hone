#!/usr/bin/env python3
"""Create SWE-style SFT data from an official SWE-bench training split.

Only the train split is accepted by default. Evaluation patches must never be
used for training because they directly reveal the answer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from finetune.jsonl import JsonlDatasetWriter
from finetune.logging import configure_logging
from finetune.models import TrainingExample
from finetune.normalization import SWETrainingExampleNormalizer
from finetune.splitting import DatasetSplitter


def row_to_messages(row: dict[str, object]) -> dict[str, object]:
    """Convert one SWE-bench row into the repository's chat contract."""
    example: TrainingExample = SWETrainingExampleNormalizer().normalize(row)
    serialized: dict[str, object] = example.to_record()
    return serialized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="SWE-bench/SWE-bench")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/swe"))
    parser.add_argument("--valid-ratio", type=float, default=0.05)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--max-chars",
        type=int,
        default=14000,
        help="approximate prompt+patch cap for the 4096-token config",
    )
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    logger = configure_logging()
    if arguments.split != "train":
        raise SystemExit("refusing non-train split; evaluation patches would leak")
    if not 0 < arguments.valid_ratio < 1:
        raise SystemExit("--valid-ratio must be between 0 and 1")
    if arguments.max_samples is not None and arguments.max_samples < 2:
        raise SystemExit("--max-samples must be at least 2")
    if arguments.max_chars < 1:
        raise SystemExit("--max-chars must be positive")
    rows = load_dataset(arguments.dataset, split=arguments.split)
    converted: list[TrainingExample] = []
    normalizer = SWETrainingExampleNormalizer()
    for row in rows:
        try:
            item = normalizer.normalize(row)
            if item.character_count > arguments.max_chars:
                continue
            converted.append(item)
        except ValueError:
            continue
        if arguments.max_samples and len(converted) >= arguments.max_samples:
            break
    if len(converted) < 2:
        raise SystemExit("not enough valid SWE examples")
    train, validation = DatasetSplitter(arguments.valid_ratio, arguments.seed).split(
        converted
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    writer = JsonlDatasetWriter()
    writer.write(arguments.output_dir / "train.jsonl", train)
    writer.write(arguments.output_dir / "valid.jsonl", validation)
    logger.info("wrote %d train and %d validation rows", len(train), len(validation))


if __name__ == "__main__":
    main()
