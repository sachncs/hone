"""Command-line entry points for local dataset preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from finetune.jsonl import JsonlDatasetWriter
from finetune.logging import configure_logging
from finetune.normalization import TrainingExampleNormalizer
from finetune.splitting import DatasetSplitter


def prepare_data_main(arguments: list[str] | None = None) -> None:
    """Validate and split chat or prompt/completion JSONL data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--valid-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--verbose", action="store_true")
    parsed = parser.parse_args(arguments)
    logger = configure_logging(parsed.verbose)
    if parsed.max_samples is not None and parsed.max_samples < 2:
        raise SystemExit("--max-samples must be at least 2")

    normalizer = TrainingExampleNormalizer()
    examples = []
    with parsed.input.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            try:
                source_record = json.loads(line)
                if not isinstance(source_record, dict):
                    raise ValueError("each JSONL record must be an object")
                examples.append(normalizer.normalize(source_record))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise SystemExit(f"{parsed.input}:{line_number}: {error}") from error
    if parsed.max_samples is not None:
        examples = examples[: parsed.max_samples]

    train, validation = DatasetSplitter(parsed.valid_ratio, parsed.seed).split(examples)
    writer = JsonlDatasetWriter()
    writer.write(parsed.output_dir / "train.jsonl", train)
    writer.write(parsed.output_dir / "valid.jsonl", validation)
    logger.info(
        "wrote %d train and %d validation examples", len(train), len(validation)
    )
