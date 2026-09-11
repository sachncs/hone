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


def test_sandbox_rejects_memory_hog() -> None:
    """A candidate that allocates a huge list is killed by RLIMIT_AS."""
    candidate = "a = []\nwhile True: a.append(' ' * (10**7))"
    outcome = evaluate(candidate, "assert True", timeout=5.0)
    assert outcome.passed is False
    assert outcome.stderr != "" or outcome.timed_out is True


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


# ---------------------------------------------------------------------------
# MlxCoder smoke (mocked mlx_lm)
# ---------------------------------------------------------------------------


def test_mlx_coder_generate_returns_n_samples_per_prompt() -> None:
    """MlxCoder.generate is exercised against a mock mlx_lm.

    The harness is the only caller of mlx_lm.generate; a regression
    in the upstream API (e.g. the sampler= kwarg rename that has
    happened in past releases) would only surface when someone
    runs the harness on a Mac. This mock keeps the regression
    gate inside CI.
    """
    from bench.model import GenerationParams, MlxCoder

    class _FakeModel:
        def __init__(self) -> None:
            self.calls = 0

    class _FakeTokenizer:
        def __call__(self, *args: object, **kwargs: object) -> object:
            return None

    def _fake_load(
        model_id: str, adapter_path: str | None = None
    ) -> tuple[_FakeModel, _FakeTokenizer]:
        return (_FakeModel(), _FakeTokenizer())

    def _fake_generate(
        model: _FakeModel,
        tokenizer: _FakeTokenizer,
        *,
        prompt: str,
        max_tokens: int,
        sampler: object,
        verbose: bool = False,
    ) -> str:
        model.calls += 1
        return f"{prompt}->out{model.calls}"

    def _fake_make_sampler(*, temp: float) -> object:
        return ("sampler", temp)

    import sys

    fake_module = type(sys)("mlx_lm")
    fake_module.load = _fake_load  # type: ignore[attr-defined]
    fake_module.generate = _fake_generate  # type: ignore[attr-defined]
    sampler_module = type(sys)("mlx_lm.sample_utils")
    sampler_module.make_sampler = _fake_make_sampler  # type: ignore[attr-defined]
    fake_module.sample_utils = sampler_module  # type: ignore[attr-defined]

    monkey = __import__("pytest").MonkeyPatch()
    try:
        monkey.setitem(sys.modules, "mlx_lm", fake_module)
        monkey.setitem(sys.modules, "mlx_lm.sample_utils", sampler_module)
        coder = MlxCoder(model_id="fake/model", adapter_path=None)
        completions = coder.generate(
            ["p1", "p2"],
            GenerationParams(max_tokens=8, temperature=0.0, n_samples=3),
        )
    finally:
        monkey.undo()

    assert len(completions) == 2
    for prompt, samples in zip(["p1", "p2"], completions, strict=True):
        assert len(samples) == 3
        for sample in samples:
            assert sample.startswith(prompt + "->out")
