#!/usr/bin/env python3
"""Generate with MiniCPM5 using the correct no-thinking coding template."""

from __future__ import annotations

import argparse
import sys

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--model", default="mlx-community/MiniCPM5-1B-4bit")
    ap.add_argument("--adapter-path")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()
    loaded_model = load(
        args.model,
        adapter_path=args.adapter_path,
        return_config=False,
    )
    model, tokenizer = loaded_model[0], loaded_model[1]
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    sys.stdout.write(
        generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            sampler=make_sampler(temp=args.temperature),
            verbose=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
