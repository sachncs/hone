"""CLI subcommand group: generation (filled in Phase 3)."""

from __future__ import annotations

import typer

app: typer.Typer = typer.Typer(help="Generate text from a trained adapter.", no_args_is_help=True)


@app.command("prompt")
def prompt(
    prompt: str = typer.Argument(..., help="Prompt text."),
    model: str = typer.Option("mlx-community/MiniCPM5-1B-4bit", "--model"),
    adapter: str | None = typer.Option(None, "--adapter", help="Adapter path."),
    max_tokens: int = typer.Option(1024, "--max-tokens"),
    temperature: float = typer.Option(0.2, "--temperature"),
) -> None:
    """Single-prompt generation."""
    raise NotImplementedError("hone generate prompt is implemented in T3.11")


@app.command("file")
def file(
    input: str = typer.Option(..., "--input", help="JSONL of prompts."),
    output: str = typer.Option("artifacts/lcb_outputs.json", "--output"),
    model: str = typer.Option("mlx-community/MiniCPM5-1B-4bit", "--model"),
    adapter: str | None = typer.Option(None, "--adapter"),
    samples: int = typer.Option(1, "--samples"),
    max_tokens: int = typer.Option(1536, "--max-tokens"),
    temperature: float = typer.Option(0.2, "--temperature"),
) -> None:
    """Bulk generation from a JSONL prompts file."""
    raise NotImplementedError("hone generate file is implemented in T3.12")
