#!/usr/bin/env python3
"""Materialize every row of a Hugging Face training config as MLX JSONL.

No sampling, max-samples, validation split, or truncation is performed here.
The caller is responsible for sufficient disk space.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from finetune.logging import configure_logging

type JsonObject = dict[str, object]

ROLE_MAP = {
    "human": "user",
    "user": "user",
    "system": "system",
    "assistant": "assistant",
    "gpt": "assistant",
    "bot": "assistant",
}


def normalize_messages(value: object) -> list[JsonObject] | None:
    if not isinstance(value, list):
        return None
    out: list[JsonObject] = []
    for item in value:
        if not isinstance(item, dict):
            return None
        role = ROLE_MAP.get(str(item.get("role", "")).lower())
        content = item.get("content")
        if role is None or content is None:
            return None
        content = str(content).strip()
        if content:
            out.append({"role": role, "content": content})
    if len(out) >= 2 and out[-1]["role"] == "assistant":
        return out
    return None


def as_sft(row: JsonObject) -> JsonObject | None:
    messages = normalize_messages(row.get("messages"))
    if messages:
        return {"messages": messages}
    prompt = next(
        (
            row.get(k)
            for k in ("prompt", "question", "instruction", "problem")
            if row.get(k)
        ),
        None,
    )
    answer = next(
        (
            row.get(k)
            for k in ("completion", "response", "answer", "solution", "output")
            if row.get(k)
        ),
        None,
    )
    if (
        prompt is not None
        and answer is not None
        and not isinstance(answer, (dict, list))
    ):
        return {
            "messages": [
                {"role": "user", "content": str(prompt).strip()},
                {"role": "assistant", "content": str(answer).strip()},
            ]
        }
    return None


def codeforces_text(row: JsonObject) -> JsonObject:
    fields = [
        ("TITLE", row.get("title")),
        ("DESCRIPTION", row.get("description")),
        ("INPUT FORMAT", row.get("input_format")),
        ("OUTPUT FORMAT", row.get("output_format")),
        ("EDITORIAL", row.get("editorial")),
    ]
    text = "\n\n".join(f"## {name}\n{value}" for name, value in fields if value)
    if not text:
        raise ValueError("row has no serializable problem text")
    return {"text": text}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--configs", required=True, help="comma-separated HF configs")
    ap.add_argument("--split", default="train")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--mode", choices=("sft", "codeforces-text"), default="sft")
    args = ap.parse_args()
    logger = configure_logging()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with args.output.open("w", encoding="utf-8") as out:
        for config in args.configs.split(","):
            ds = load_dataset(args.repo, name=config, split=args.split, streaming=True)
            for row in ds:
                item = (
                    codeforces_text(row)
                    if args.mode == "codeforces-text"
                    else as_sft(row)
                )
                if item is None:
                    skipped += 1
                    continue
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
                written += 1
                if written % 100000 == 0:
                    logger.info("written=%d skipped=%d", written, skipped)
    logger.info(
        "complete: written=%d skipped=%d output=%s", written, skipped, args.output
    )


if __name__ == "__main__":
    main()
