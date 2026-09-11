"""MBPP pass@1 for MLX adapters.

MBPP is Google-research's 500-problem Python benchmark (the
'full' split). Each problem is a short natural-language spec
+ 3 visible tests. The model must write a function that passes
the tests.

The harness uses a 3-shot prompt with worked examples drawn from
the prompt split, then a single greedy completion per problem,
then runs the visible tests in a sandboxed subprocess.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

from bench.model import GenerationParams, MlxCoder
from bench.sandbox import TestOutcome, aggregate, evaluate

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MBPPProblem:
    """One MBPP problem."""

    task_id: int
    text: str
    """Natural-language specification."""
    test_list: list[str]
    """3 visible tests; we run all 3 (MBPP's standard ``code_eval``)."""
    code: str
    """Reference solution; not used at eval time, only for documentation."""


@dataclass(frozen=True)
class MBPPRun:
    """Result of evaluating one model on MBPP."""

    pass_at_1: float
    total: int
    seconds: float


def load_problems() -> list[MBPPProblem]:
    """Load the MBPP 'full' split (500 problems)."""
    from datasets import load_dataset

    dataset = load_dataset("google-research-datasets/mbpp", name="full", split="test")
    return [
        MBPPProblem(
            task_id=row["task_id"],
            text=row["text"],
            test_list=list(row["test_list"]),
            code=row["code"],
        )
        for row in dataset
    ]


def _load_fewshot() -> list[MBPPProblem]:
    """Load 3 worked examples from the prompt split for few-shot prompting."""
    from datasets import load_dataset

    dataset = load_dataset("google-research-datasets/mbpp", name="full", split="prompt")
    return [
        MBPPProblem(
            task_id=row["task_id"],
            text=row["text"],
            test_list=list(row["test_list"]),
            code=row["code"],
        )
        for row in dataset.select(range(3))
    ]


def _format_fewshot(problem: MBPPProblem) -> str:
    """One worked example: problem spec -> code."""
    tests = problem.test_list[:3]
    return (
        f"Task: {problem.text}\n"
        "Your code should pass these tests:\n\n"
        + "\n".join(tests)
        + "\n[BEGIN]\n"
        + problem.code
        + "\n[DONE]\n"
    )


def _format_prompt(problem: MBPPProblem, fewshot: Sequence[MBPPProblem]) -> str:
    """3-shot MBPP prompt for one problem.

    Format matches lm-eval-harness's mbpp.yaml: ``You are an
    expert Python programmer, ... Your code should pass these
    tests: [BEGIN]``. The model is expected to emit code between
    ``[BEGIN]`` and ``[DONE]``.
    """
    tests = problem.test_list[:3]
    head = (
        "You are an expert Python programmer. "
        "Solve each task by writing a Python function that passes the tests.\n\n"
    )
    fewshot_block = "".join(_format_fewshot(p) for p in fewshot)
    return (
        head + fewshot_block + "\nNew task:\n"
        f"Task: {problem.text}\n"
        "Your code should pass these tests:\n\n" + "\n".join(tests) + "\n[BEGIN]\n"
    )


def _extract_python_block(text: str) -> str:
    """Pull a clean Python candidate out of a model completion.

    Three passes, in order:

    1. If the model used markdown fences, take the first fenced
       block (skipping a leading ``python`` tag).
    2. Slice from the first ``def`` line through the first
       ``[DONE]`` marker (MBPP's stop token) or end-of-text.
       This matters because MBPP's test asserts call the function
       by name — the candidate must define ``def`` and must not
       leak ``[DONE]`` into the script.
    3. Fall back to the raw text.
    """
    text = text.strip()
    if "```" in text:
        lines = text.splitlines()
        inside = False
        out: list[str] = []
        for line in lines:
            if line.strip().startswith("```") and not inside:
                inside = True
                continue
            if line.strip() == "```" and inside:
                break
            if inside:
                out.append(line)
        candidate = "\n".join(out).strip()
        if candidate:
            return candidate
    # Slice from first ``def`` through first ``[DONE]`` (MBPP stop).
    lines = text.splitlines()
    start = None
    end = len(lines)
    for index, line in enumerate(lines):
        if start is None and line.lstrip().startswith("def "):
            start = index
        if start is not None and line.strip() == "[DONE]":
            end = index
            break
    if start is not None:
        return "\n".join(lines[start:end])
    return text


def run_mbpp(
    model: MlxCoder,
    *,
    max_tokens: int = 512,
    timeout: float = 3.0,
    problems: Sequence[MBPPProblem] | None = None,
) -> MBPPRun:
    """Run MBPP against the loaded model; return pass@1."""
    if problems is None:
        problems = load_problems()
    started = time.monotonic()

    fewshot = _load_fewshot()
    prompts = [_format_prompt(problem, fewshot) for problem in problems]
    log.info("MBPP pass@1: %d problems (3-shot), greedy", len(prompts))
    completions = model.generate(
        prompts, GenerationParams(max_tokens=max_tokens, temperature=0.0, n_samples=1)
    )

    outcomes: list[TestOutcome] = []
    for problem, completion_set in zip(problems, completions, strict=True):
        candidate = _extract_python_block(completion_set[0])
        test_block = "\n".join(problem.test_list)
        outcome = evaluate(candidate, test_block, timeout=timeout)
        outcomes.append(outcome)

    elapsed = time.monotonic() - started
    pass_at_1 = aggregate(outcomes)["pass_rate"]
    log.info("MBPP done: pass@1=%.3f in %.1fs", pass_at_1, elapsed)
    return MBPPRun(pass_at_1=pass_at_1, total=len(outcomes), seconds=elapsed)


__all__ = ["MBPPProblem", "MBPPRun", "load_problems", "run_mbpp"]
