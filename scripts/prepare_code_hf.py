#!/usr/bin/env python3
"""Stream labeled competitive-programming solutions into MLX chat JSONL."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

from finetune.logging import configure_logging

type JsonObject = dict[str, object]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="teven/code_contests")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/code"))
    parser.add_argument("--language", default="PYTHON")
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--scan-limit", type=int, default=250000)
    parser.add_argument("--valid-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    logger = configure_logging()
    if arguments.max_samples < 2:
        raise SystemExit("--max-samples must be at least 2")
    if arguments.scan_limit < 1:
        raise SystemExit("--scan-limit must be positive")
    if not 0 < arguments.valid_ratio < 1:
        raise SystemExit("--valid-ratio must be between 0 and 1")

    stream = load_dataset(arguments.dataset, split=arguments.split, streaming=True)
    rng = random.Random(arguments.seed)
    reservoir: list[JsonObject] = []
    seen = 0
    for row in stream:
        if (
            row.get("language")
            and str(row.get("language", "")).upper() != arguments.language.upper()
        ):
            continue
        question = str(row.get("description", row.get("question", ""))).strip()
        solution = str(row.get("solution", row.get("answer", ""))).strip()
        if len(question) < 80 or len(solution) < 20:
            continue
        item: JsonObject = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Solve this competitive-programming problem in Python. "
                        "Return only the complete program.\n\n" + question
                    ),
                },
                {"role": "assistant", "content": solution},
            ]
        }
        seen += 1
        if len(reservoir) < arguments.max_samples:
            reservoir.append(item)
        else:
            index = rng.randrange(seen)
            if index < arguments.max_samples:
                reservoir[index] = item
        if seen >= arguments.scan_limit:
            break

    if len(reservoir) < 2:
        raise SystemExit("fewer than two usable examples found")
    rng.shuffle(reservoir)
    valid_count = max(1, round(len(reservoir) * arguments.valid_ratio))
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in (
        ("train.jsonl", reservoir[valid_count:]),
        ("valid.jsonl", reservoir[:valid_count]),
    ):
        with (arguments.output_dir / name).open("w", encoding="utf-8") as output_file:
            for value in values:
                output_file.write(json.dumps(value, ensure_ascii=False) + "\n")
    logger.info("selected %d of %d usable rows", len(reservoir), seen)
    logger.info(
        "wrote %d train and %d validation rows",
        len(reservoir) - valid_count,
        valid_count,
    )


if __name__ == "__main__":
    main()
