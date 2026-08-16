"""Human-readable benchmark report.

Takes the dataclass outputs from each benchmark and produces
a single markdown report string + a JSON dict for programmatic
consumption.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from bench.humaneval import HumanEvalRun
from bench.mbpp import MBPPRun


@dataclass(frozen=True)
class BenchmarkReport:
    """Combined HumanEval + MBPP results for one model."""

    model_id: str
    adapter: str | None
    humaneval: dict[str, Any]
    mbpp: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable dict for storage."""
        return {
            "model_id": self.model_id,
            "adapter": self.adapter,
            "humaneval": self.humaneval,
            "mbpp": self.mbpp,
        }


def build_report(
    *,
    model_id: str,
    adapter: str | None,
    humaneval: HumanEvalRun | None,
    mbpp: MBPPRun | None,
) -> BenchmarkReport:
    """Combine runs into a single report."""
    return BenchmarkReport(
        model_id=model_id,
        adapter=adapter,
        humaneval=asdict(humaneval) if humaneval is not None else {},
        mbpp=asdict(mbpp) if mbpp is not None else {},
    )


def format_report(report: BenchmarkReport) -> str:
    """Render a BenchmarkReport as markdown."""
    lines: list[str] = []
    lines.append(f"# Benchmark report — {report.model_id}")
    if report.adapter:
        lines.append("")
        lines.append(f"adapter: `{report.adapter}`")
    if report.humaneval:
        lines.append("")
        lines.append("## HumanEval")
        for key, value in report.humaneval.items():
            if isinstance(value, float):
                lines.append(f"- **{key}**: {value:.3f}")
            else:
                lines.append(f"- **{key}**: {value}")
    if report.mbpp:
        lines.append("")
        lines.append("## MBPP")
        for key, value in report.mbpp.items():
            if isinstance(value, float):
                lines.append(f"- **{key}**: {value:.3f}")
            else:
                lines.append(f"- **{key}**: {value}")
    return "\n".join(lines) + "\n"


__all__ = ["BenchmarkReport", "build_report", "format_report"]
