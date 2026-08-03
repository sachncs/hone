"""CLI subcommand group: data preparation (filled in Phase 3)."""

from __future__ import annotations

import typer

app: typer.Typer = typer.Typer(help="Prepare datasets.", no_args_is_help=True)


@app.command("file")
def file(
    input: str = typer.Option(..., "--input", help="Path to input JSONL."),
    output: str = typer.Option(..., "--output", help="Directory for train/valid JSONL files."),
    ratio: float = typer.Option(0.05, "--ratio", help="Validation split ratio (exclusive 0..1)."),
    seed: int = typer.Option(42, "--seed", help="Deterministic seed."),
    max_samples: int | None = typer.Option(None, "--max-samples", help="Cap on examples read."),
) -> None:
    """Normalize and split a local JSONL file."""
    raise NotImplementedError("hone prepare file is implemented in T3.2")


@app.command("code")
def code(
    dataset: str = typer.Option("teven/code_contests", "--dataset", help="HuggingFace dataset ID."),
    split: str = typer.Option("train", "--split", help="HF dataset split."),
    output: str = typer.Option("data/processed/code", "--output", help="Output directory."),
    language: str = typer.Option("PYTHON", "--language", help="Programming language filter."),
    max_samples: int = typer.Option(20000, "--max-samples", help="Reservoir cap."),
    scan_limit: int = typer.Option(250000, "--scan-limit", help="Stop scanning after this many rows."),
    ratio: float = typer.Option(0.02, "--ratio", help="Validation ratio."),
    seed: int = typer.Option(42, "--seed", help="Deterministic seed."),
) -> None:
    """Reservoir-sample competitive-programming rows and split."""
    raise NotImplementedError("hone prepare code is implemented in T3.3")


@app.command("swe")
def swe(
    dataset: str = typer.Option("SWE-bench/SWE-bench", "--dataset", help="HuggingFace dataset ID."),
    split: str = typer.Option("train", "--split", help="HF dataset split."),
    output: str = typer.Option("data/processed/swe", "--output", help="Output directory."),
    ratio: float = typer.Option(0.05, "--ratio", help="Validation ratio."),
    max_samples: int | None = typer.Option(None, "--max-samples", help="Cap."),
    max_chars: int = typer.Option(14000, "--max-chars", help="Drop rows exceeding this prompt+patch length."),
    seed: int = typer.Option(42, "--seed", help="Deterministic seed."),
) -> None:
    """Build SWE-bench SFT rows. Refuses non-train splits."""
    raise NotImplementedError("hone prepare swe is implemented in T3.4")


@app.command("all")
def all(  # noqa: A001
    repo: str = typer.Option(..., "--repo", help="HuggingFace repo ID."),
    configs: str = typer.Option(..., "--configs", help="Comma-separated HF configs."),
    split: str = typer.Option("train", "--split"),
    output: str = typer.Option(..., "--output", help="Output JSONL path."),
    mode: str = typer.Option("sft", "--mode", help="sft or codeforces-text."),
) -> None:
    """Materialize every row of an HF config as MLX JSONL."""
    raise NotImplementedError("hone prepare all is implemented in T3.5")


@app.command("evaluate")
def evaluate(
    version: str = typer.Option("release_v2", "--version", help="LiveCodeBench release tag."),
    output: str = typer.Option("data/eval/lcb.jsonl", "--output", help="Output JSONL path."),
) -> None:
    """Download LiveCodeBench prompts for evaluation only."""
    raise NotImplementedError("hone prepare evaluate is implemented in T3.6")
