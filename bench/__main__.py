"""``python -m bench`` driver.

Usage:
    python -m bench --model openbmb/MiniCPM5-1B-MLX
    python -m bench --model openbmb/MiniCPM5-1B-MLX --adapter artifacts/soup-codex-full
    python -m bench --model openbmb/MiniCPM5-1B-MLX --benchmarks humaneval
    python -m bench --model openbmb/MiniCPM5-1B-MLX --pass-at-10 --output results/base.json

Reports go to stdout (markdown) and optionally to a JSON file.
"""  # noqa: E501

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from bench import format_report
from bench.humaneval import run_humaneval
from bench.mbpp import run_mbpp
from bench.model import MlxCoder
from bench.report import build_report


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bench",
        description="Run HumanEval + MBPP on an MLX model + optional LoRA.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HF model id or local path (e.g. openbmb/MiniCPM5-1B-MLX).",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Path to a LoRA adapter directory.",
    )
    parser.add_argument(
        "--benchmarks",
        default="humaneval,mbpp",
        help="Comma-separated subset of {humaneval,mbpp}.",
    )
    parser.add_argument(
        "--pass-at-10",
        action="store_true",
        help="Also compute pass@10 (10 samples per problem at temp=0.8).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max generation length per completion.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Per-problem subprocess timeout in seconds.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap on the number of problems per benchmark (for smoke runs).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file for the structured report.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG-level logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    selected = {name.strip() for name in args.benchmarks.split(",")}

    coder = MlxCoder(model_id=args.model, adapter_path=args.adapter)

    humaneval_run = None
    if "humaneval" in selected:
        problems: list | None = None
        if args.limit is not None:
            from bench.humaneval import load_problems as load_humaneval

            problems = load_humaneval()[: args.limit]
        humaneval_run = run_humaneval(
            coder,
            max_tokens=args.max_tokens,
            pass_at_10=args.pass_at_10,
            timeout=args.timeout,
            problems=problems,
        )

    mbpp_run = None
    if "mbpp" in selected:
        problems = None
        if args.limit is not None:
            from bench.mbpp import load_problems as load_mbpp

            problems = load_mbpp()[: args.limit]
        mbpp_run = run_mbpp(
            coder,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            problems=problems,
        )

    report = build_report(
        model_id=args.model,
        adapter=args.adapter,
        humaneval=humaneval_run,
        mbpp=mbpp_run,
    )

    sys.stdout.write(format_report(report))
    sys.stdout.flush()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report.to_json(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
