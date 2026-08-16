"""CLI subcommand group: data preparation.

Thin transport layer over :mod:`hone.prepare.service`. Each command
parses typer options, raises ``typer.BadParameter`` for invalid
input, and delegates to the corresponding service function. The
service raises :class:`hone.errors.DataError`; the CLI converts
that to ``typer.BadParameter`` so users see the same UX as before.
"""

from __future__ import annotations

from pathlib import Path

import typer

from hone.errors import DataError, HoneError
from hone.log import setup
from hone.prepare import (
    PrepareRequest,
    prepare_eval_prompts,
    prepare_local_file,
    prepare_reservoir_sample,
    prepare_stream,
    prepare_swe,
)

app: typer.Typer = typer.Typer(help="Prepare datasets.", no_args_is_help=True)


def _reraise(error: HoneError) -> typer.BadParameter:
    """Convert a domain exception into a CLI-friendly error."""
    raise typer.BadParameter(str(error)) from error


@app.command("file")
def file(
    input: str = typer.Option(..., "--input", help="Path to input JSONL."),
    output: str = typer.Option(
        ..., "--output", help="Directory for train/valid JSONL files."
    ),
    ratio: float = typer.Option(
        0.05, "--ratio", help="Validation split ratio (exclusive 0..1)."
    ),
    seed: int = typer.Option(42, "--seed", help="Deterministic seed."),
    max_samples: int | None = typer.Option(
        None, "--max-samples", help="Cap on examples read."
    ),
) -> None:
    """Normalize and split a local JSONL file."""
    logger = setup(verbose=False)
    request = PrepareRequest(output=Path(output), seed=seed, logger=logger)
    try:
        prepare_local_file(
            input_path=Path(input),
            request=request,
            ratio=ratio,
            max_samples=max_samples,
        )
    except DataError as error:
        _reraise(error)


@app.command("code")
def code(
    dataset: str = typer.Option(
        "teven/code_contests", "--dataset", help="HuggingFace dataset ID."
    ),
    split: str = typer.Option("train", "--split", help="HF dataset split."),
    output: str = typer.Option(
        "data/processed/code", "--output", help="Output directory."
    ),
    language: str = typer.Option(
        "PYTHON", "--language", help="Programming language filter."
    ),
    max_samples: int = typer.Option(20000, "--max-samples", help="Reservoir cap."),
    scan_limit: int = typer.Option(
        250000, "--scan-limit", help="Stop scanning after this many rows."
    ),
    ratio: float = typer.Option(0.02, "--ratio", help="Validation ratio."),
    seed: int = typer.Option(42, "--seed", help="Deterministic seed."),
) -> None:
    """Reservoir-sample competitive-programming rows and split."""
    logger = setup(verbose=False)
    request = PrepareRequest(output=Path(output), seed=seed, logger=logger)
    try:
        prepare_reservoir_sample(
            request=request,
            dataset=dataset,
            split=split,
            language=language,
            max_samples=max_samples,
            scan_limit=scan_limit,
            ratio=ratio,
        )
    except DataError as error:
        _reraise(error)


@app.command("swe")
def swe(
    dataset: str = typer.Option(
        "SWE-bench/SWE-bench", "--dataset", help="HuggingFace dataset ID."
    ),
    split: str = typer.Option("train", "--split", help="HF dataset split."),
    output: str = typer.Option(
        "data/processed/swe", "--output", help="Output directory."
    ),
    ratio: float = typer.Option(0.05, "--ratio", help="Validation ratio."),
    max_samples: int | None = typer.Option(None, "--max-samples", help="Cap."),
    max_chars: int = typer.Option(
        14000, "--max-chars", help="Drop rows exceeding this prompt+patch length."
    ),
    seed: int = typer.Option(42, "--seed", help="Deterministic seed."),
) -> None:
    """Build SWE-bench SFT rows. Refuses non-train splits."""
    logger = setup(verbose=False)
    request = PrepareRequest(output=Path(output), seed=seed, logger=logger)
    try:
        prepare_swe(
            request=request,
            dataset=dataset,
            split=split,
            ratio=ratio,
            max_samples=max_samples,
            max_chars=max_chars,
        )
    except DataError as error:
        _reraise(error)


@app.command("all")
def all_cmd(
    repo: str = typer.Option(..., "--repo", help="HuggingFace repo ID."),
    configs: str = typer.Option(..., "--configs", help="Comma-separated HF configs."),
    split: str = typer.Option("train", "--split"),
    output: str = typer.Option(..., "--output", help="Output JSONL path."),
    mode: str = typer.Option("sft", "--mode", help="sft or codeforces-text."),
    max_tokens: int = typer.Option(
        0,
        "--max-tokens",
        help="Drop records whose token count exceeds this value; 0 disables.",
    ),
    max_samples: int = typer.Option(
        0,
        "--max-samples",
        help="Stop after this many rows are written; 0 means full pass. "
        "Useful for capping multi-million-row HF streams to a tractable subset "
        "for short fine-tune runs.",
    ),
    tokenizer_model: str = typer.Option(
        "openbmb/MiniCPM5-1B",
        "--tokenizer-model",
        help="HF model id used for token-count filtering.",
    ),
) -> None:
    """Materialize every row of an HF config as MLX JSONL."""
    logger = setup(verbose=False)
    request = PrepareRequest(output=Path(output), seed=42, logger=logger)
    try:
        prepare_stream(
            request=request,
            repo=repo,
            configs=configs,
            split=split,
            mode=mode,
            max_tokens=max_tokens,
            max_samples=max_samples,
            tokenizer_model=tokenizer_model,
        )
    except DataError as error:
        _reraise(error)


@app.command("evaluate")
def evaluate(
    version: str = typer.Option(
        "release_v2", "--version", help="LiveCodeBench release tag."
    ),
    output: str = typer.Option(
        "data/eval/lcb.jsonl", "--output", help="Output JSONL path."
    ),
) -> None:
    """Download LiveCodeBench prompts for evaluation only."""
    logger = setup(verbose=False)
    prepare_eval_prompts(output=Path(output), version=version, logger=logger)


__all__ = ["app"]
