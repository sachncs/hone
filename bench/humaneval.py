"""HumanEval pass@1 / pass@10 for MLX adapters.

HumanEval is the OpenAI 164-problem coding benchmark. Each problem
gives a function signature + docstring; the model must complete
the body. lm-evaluation-harness uses the prompt as-is and runs
the candidate through ``code_eval``'s test runner.

Our harness follows the same protocol: emit the prompt's
function body, prepend it back to the signature + docstring,
and run against the canonical test cases. pass@1 uses greedy
decoding (temp=0); pass@10 uses temp=0.8 with 10 samples.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from bench.model import GenerationParams, MlxCoder
from bench.sandbox import TestOutcome, aggregate, evaluate

log = logging.getLogger(__name__)

HUMANEVAL_PROMPT_TEMPLATE = (
    "Complete the following Python function. Return only the function body.\n\n"
    "{prompt}"
)


@dataclass(frozen=True)
class HumanEvalProblem:
    """One HumanEval problem."""

    task_id: str
    prompt: str
    """Function signature + docstring; the model must complete the body."""
    test: str
    """Canonical ``check(...)`` call that exercises the function."""
    entry_point: str
    """Function name (e.g. ``add``)."""


@dataclass(frozen=True)
class HumanEvalRun:
    """Result of evaluating one model on HumanEval."""

    pass_at_1: float
    pass_at_10: float | None
    pass_at_10_run: bool
    """True iff pass@10 was actually computed (not just left as None)."""
    total: int
    seconds: float


def load_problems() -> list[HumanEvalProblem]:
    """Load the canonical 164 HumanEval problems from HuggingFace."""
    from datasets import load_dataset

    dataset = load_dataset("openai/openai_humaneval", split="test")
    return [
        HumanEvalProblem(
            task_id=row["task_id"],
            prompt=row["prompt"],
            test=row["test"],
            entry_point=row["entry_point"],
        )
        for row in dataset
    ]


def _wrap_completion(prompt: str, completion: str) -> str:
    """Prepend the prompt back to the model completion to form a full program.

    HumanEval's prompt is a Python snippet that ends mid-function;
    the model's output is the function body. Re-attaching the
    prompt lets ``code_eval`` run the test against the function.
    """
    body = completion.strip()
    if body.startswith("```"):
        # Strip markdown fences if the model emitted any
        lines = body.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        body = "\n".join(lines).strip()
    return prompt + body


def run_humaneval(
    model: MlxCoder,
    *,
    max_tokens: int = 512,
    pass_at_10: bool = False,
    timeout: float = 5.0,
    problems: Sequence[HumanEvalProblem] | None = None,
) -> HumanEvalRun:
    """Run HumanEval against the loaded model; return pass@1 and optional pass@10.

    ``pass_at_10=True`` costs ~10x more generations. The harness
    reports both numbers in the same object so a single call covers
    the most common reporting cases.
    """
    if problems is None:
        problems = load_problems()
    started = time.monotonic()

    prompts = [
        HUMANEVAL_PROMPT_TEMPLATE.format(prompt=problem.prompt) for problem in problems
    ]

    # pass@1: greedy, one sample per problem.
    log.info("HumanEval pass@1: %d problems, greedy", len(prompts))
    one_samples = model.generate(
        prompts, GenerationParams(max_tokens=max_tokens, temperature=0.0, n_samples=1)
    )

    # pass@10: optional, 10 samples at temp=0.8 per problem.
    ten_samples: list[list[str]] | None = None
    if pass_at_10:
        log.info("HumanEval pass@10: %d problems x 10 samples", len(prompts))
        ten_samples = model.generate(
            prompts, GenerationParams(max_tokens=max_tokens, temperature=0.8, n_samples=10)
        )

    outcomes: list[TestOutcome] = []
    for problem, completions in zip(problems, one_samples):
        candidate = _wrap_completion(problem.prompt, completions[0])
        outcome = evaluate(candidate, problem.test, timeout=timeout)
        outcomes.append(outcome)

    pass_at_1 = aggregate(outcomes)["pass_rate"]

    pass_at_10_score: float | None = None
    if ten_samples is not None:
        ten_pass: list[bool] = []
        for problem, samples in zip(problems, ten_samples):
            any_pass = False
            for completion in samples:
                candidate = _wrap_completion(problem.prompt, completion)
                if evaluate(candidate, problem.test, timeout=timeout).passed:
                    any_pass = True
                    break
            ten_pass.append(any_pass)
        if ten_pass:
            pass_at_10_score = sum(ten_pass) / len(ten_pass)

    elapsed = time.monotonic() - started
    log.info(
        "HumanEval done: pass@1=%.3f pass@10=%s in %.1fs",
        pass_at_1,
        f"{pass_at_10_score:.3f}" if pass_at_10_score is not None else "n/a",
        elapsed,
    )
    return HumanEvalRun(
        pass_at_1=pass_at_1,
        pass_at_10=pass_at_10_score,
        pass_at_10_run=pass_at_10,
        total=len(outcomes),
        seconds=elapsed,
    )


__all__ = ["HumanEvalProblem", "HumanEvalRun", "load_problems", "run_humaneval"]
# Allow the loader to be a path-based CLI later.
_ = (os, Path)
