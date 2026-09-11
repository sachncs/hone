"""Tests for the bench/ harness.

The sandbox is unit-testable without loading an MLX model; the
loaders are smoke-tested with a 2-problem cap. We do NOT run
the full MLX benchmark in CI; that takes too long and requires
the 600 MB MiniCPM5-1B-MLX checkpoint.
"""

from __future__ import annotations

from bench.humaneval import HumanEvalProblem, _wrap_completion
from bench.mbpp import MBPPProblem, _extract_python_block
from bench.sandbox import TestOutcome, aggregate, evaluate

# ---------------------------------------------------------------------------
# sandbox
# ---------------------------------------------------------------------------


def test_sandbox_passes_simple_assert() -> None:
    outcome = evaluate("x = 1", "assert x == 1")
    assert outcome.passed is True
    assert outcome.timed_out is False
    assert outcome.stderr == ""


def test_sandbox_fails_assertion_mismatch() -> None:
    outcome = evaluate("x = 2", "assert x == 1")
    assert outcome.passed is False
    assert outcome.timed_out is False
    assert "AssertionError" in outcome.stderr or "assert" in outcome.stderr


def test_sandbox_handles_runtime_error() -> None:
    outcome = evaluate("raise ValueError('boom')", "assert True")
    assert outcome.passed is False
    assert "ValueError" in outcome.stderr


def test_sandbox_times_out_on_infinite_loop() -> None:
    outcome = evaluate("while True: pass", "assert True", timeout=1.0)
    assert outcome.passed is False
    assert outcome.timed_out is True


def test_sandbox_aggregate_computes_pass_rate() -> None:
    outcomes = [
        TestOutcome(passed=True, stderr="", timed_out=False),
        TestOutcome(passed=False, stderr="x", timed_out=False),
        TestOutcome(passed=False, stderr="y", timed_out=True),
    ]
    summary = aggregate(outcomes)
    assert summary["total"] == 3
    assert summary["passed"] == 1
    assert summary["pass_rate"] == 1 / 3
    assert summary["timed_out"] == 1


# ---------------------------------------------------------------------------
# HumanEval prompt wrapping
# ---------------------------------------------------------------------------


def test_wrap_completion_strips_markdown_fences() -> None:
    prompt = "def add(a, b):\n    "
    completion = "```python\nreturn a + b\n```"
    out = _wrap_completion(prompt, completion)
    assert "return a + b" in out
    assert "```" not in out


def test_wrap_completion_handles_raw_completion() -> None:
    prompt = "def add(a, b):\n    "
    completion = "return a + b"
    out = _wrap_completion(prompt, completion)
    assert out == prompt + completion


# ---------------------------------------------------------------------------
# MBPP code extraction
# ---------------------------------------------------------------------------


def test_extract_python_block_returns_first_fence() -> None:
    text = "Here is the code:\n```python\ndef f(x):\n    return x + 1\n```\nDone."
    out = _extract_python_block(text)
    assert out == "def f(x):\n    return x + 1"


def test_extract_python_block_falls_back_to_raw() -> None:
    text = "def f(x):\n    return x + 1"
    assert _extract_python_block(text) == text


# ---------------------------------------------------------------------------
# BenchmarkReport JSON shape
# ---------------------------------------------------------------------------


def test_report_distinguishes_unselected_from_pass_at_10_not_run() -> None:
    """A not-selected benchmark serialises as None; a selected benchmark
    that did not run pass@10 still carries pass_at_10_run=False.
    """
    from bench.humaneval import HumanEvalRun
    from bench.report import build_report

    selected_no_pass10 = HumanEvalRun(
        pass_at_1=0.25,
        pass_at_10=None,
        pass_at_10_run=False,
        total=10,
        seconds=12.0,
    )
    selected_pass10 = HumanEvalRun(
        pass_at_1=0.25,
        pass_at_10=0.42,
        pass_at_10_run=True,
        total=10,
        seconds=120.0,
    )

    no_humaneval = build_report(
        model_id="m", adapter=None, humaneval=None, mbpp=None
    ).to_json()
    assert no_humaneval["humaneval"] is None
    assert no_humaneval["mbpp"] is None

    humaneval_only = build_report(
        model_id="m", adapter=None, humaneval=selected_no_pass10, mbpp=None
    ).to_json()
    assert humaneval_only["humaneval"]["pass_at_10"] is None
    assert humaneval_only["humaneval"]["pass_at_10_run"] is False

    with_pass10 = build_report(
        model_id="m", adapter=None, humaneval=selected_pass10, mbpp=None
    ).to_json()
    assert with_pass10["humaneval"]["pass_at_10_run"] is True
    assert with_pass10["humaneval"]["pass_at_10"] == 0.42


# ---------------------------------------------------------------------------
# loader smoke tests
# ---------------------------------------------------------------------------


def test_humaneval_loader_returns_problems() -> None:
    """Smoke test: confirm HumanEval loads and yields HumanEvalProblem instances.

    Skipped automatically if the dataset isn't downloadable
    (offline CI).
    """
    from bench.humaneval import load_problems

    problems = load_problems()
    if not problems:
        return  # offline CI
    assert isinstance(problems[0], HumanEvalProblem)
    assert problems[0].task_id
    assert "def " in problems[0].prompt
    assert "check" in problems[0].test


def test_mbpp_loader_returns_problems() -> None:
    """Smoke test: confirm MBPP loads and yields MBPPProblem instances.

    Skipped automatically if the dataset isn't downloadable.
    """
    from bench.mbpp import load_problems

    problems = load_problems()
    if not problems:
        return  # offline CI
    assert isinstance(problems[0], MBPPProblem)
    assert problems[0].text
    assert len(problems[0].test_list) >= 3
