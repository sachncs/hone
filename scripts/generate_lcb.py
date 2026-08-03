#!/usr/bin/env python3
"""Generate LiveCodeBench custom-evaluator JSON using an MLX model/adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from finetune.logging import configure_logging


def clean_generated_code(text: str) -> str:
    """Remove optional markdown fences from generated source code."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/eval/lcb.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("artifacts/lcb_outputs.json"))
    ap.add_argument("--model", default="mlx-community/MiniCPM5-1B-4bit")
    ap.add_argument("--adapter-path")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()
    logger = configure_logging()
    loaded_model = load(args.model, adapter_path=args.adapter_path, return_config=False)
    model, tokenizer = loaded_model[0], loaded_model[1]
    outputs = []
    for line in args.input.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": (
                        "Solve this competitive-programming problem. Return only the "
                        "complete solution code, with no markdown fences.\n\n"
                        + row["question_content"]
                    ),
                }
            ],
            add_generation_prompt=True,
            enable_thinking=False,
        )
        codes = []
        for _ in range(args.samples):
            text = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=args.max_tokens,
                sampler=make_sampler(temp=args.temperature),
                verbose=False,
            )
            codes.append(clean_generated_code(text))
        outputs.append({"question_id": row["question_id"], "code_list": codes})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("wrote %d questions to %s", len(outputs), args.output)


if __name__ == "__main__":
    main()
