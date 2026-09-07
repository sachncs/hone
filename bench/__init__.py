"""MLX-friendly coding benchmarks.

A thin harness that loads an MLX model (+ optional LoRA adapter)
and scores it on HumanEval and MBPP using the standard prompt
formats. Reports pass@1 and pass@10.

Why a custom harness instead of lm-evaluation-harness:

* lm-eval's HFLM requires transformers + torch, which we
  deliberately don't have on this Apple-Silicon repo.
* The MLX path needs a custom LM wrapper anyway; building a thin
  harness is less code than bridging MLX into lm-eval's LM
  protocol.
* Tests run in a subprocess with a 5-second timeout so a model
  that emits an infinite loop can't hang the harness.
"""

from bench.humaneval import run_humaneval
from bench.mbpp import run_mbpp
from bench.report import BenchmarkReport, format_report

__all__ = [
    "BenchmarkReport",
    "format_report",
    "run_humaneval",
    "run_mbpp",
]
