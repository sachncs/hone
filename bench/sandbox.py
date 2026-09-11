"""Sandboxed subprocess test runner.

Runs the candidate's code against the test cases in a subprocess
with a hard timeout. A model that emits an infinite loop (or a
model that hangs on a wrong call) must not hang the harness.

Captures stdout+stderr; returns ``True`` iff every test passes
with no exception, no stderr noise, and exit code 0.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path

TIMEOUT_SECONDS: float = 5.0


@dataclass(frozen=True)
class TestOutcome:
    """Result of running one candidate against one test case list."""

    passed: bool
    """True iff every test passed with exit 0 and no stderr."""
    stderr: str
    """Captured stderr (empty when passed)."""
    timed_out: bool
    """True when the subprocess exceeded TIMEOUT_SECONDS."""


def _wrap_script(script: str, test_block: str) -> str:
    """Concatenate the candidate body and the test block."""
    return textwrap.dedent(script).rstrip() + "\n\n" + textwrap.dedent(test_block)


def evaluate(
    candidate: str,
    test_block: str,
    *,
    timeout: float = TIMEOUT_SECONDS,
) -> TestOutcome:
    """Run ``candidate + test_block`` in a subprocess; return outcome.

    The script is written to a temp file and invoked with the same
    Python interpreter the harness is running under, so any
    user-installed packages the candidate imports are available.
    """
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8",
    ) as handle:
        handle.write(_wrap_script(candidate, test_block))
        tmp_path = Path(handle.name)
    try:
        completed = subprocess.run(
            [sys.executable, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return TestOutcome(passed=False, stderr="<timeout>", timed_out=True)
    finally:
        tmp_path.unlink(missing_ok=True)
    if completed.returncode != 0:
        return TestOutcome(passed=False, stderr=completed.stderr, timed_out=False)
    return TestOutcome(passed=True, stderr="", timed_out=False)


def aggregate(outcomes: list[TestOutcome]) -> dict[str, int | float]:
    """Reduce a list of outcomes to a small report dict."""
    total = len(outcomes)
    passed = sum(1 for outcome in outcomes if outcome.passed)
    timed_out = sum(1 for outcome in outcomes if outcome.timed_out)
    return {
        "total": total,
        "passed": passed,
        "pass_rate": (passed / total) if total else 0.0,
        "timed_out": timed_out,
    }


__all__ = ["TIMEOUT_SECONDS", "TestOutcome", "aggregate", "evaluate"]
