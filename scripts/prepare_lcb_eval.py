#!/usr/bin/env python3
"""Download LiveCodeBench prompts for evaluation only.

This intentionally writes to data/eval and never to a training directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from finetune.logging import configure_logging


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="release_v2")
    ap.add_argument("--output", type=Path, default=Path("data/eval/lcb.jsonl"))
    args = ap.parse_args()
    logger = configure_logging()
    ds = load_dataset("livecodebench/code_generation_lite", version_tag=args.version)
    split = ds["test"] if isinstance(ds, dict) else ds
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in split:
            f.write(
                json.dumps(
                    {
                        "question_id": row["question_id"],
                        "question_content": row["question_content"],
                        "contest_date": str(row.get("contest_date", "")),
                        "difficulty": row.get("difficulty", ""),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    logger.info("wrote %d evaluation prompts to %s", len(split), args.output)


if __name__ == "__main__":
    main()
