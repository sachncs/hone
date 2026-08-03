"""CLI subcommand group: evaluation (filled in Phase 3)."""

from __future__ import annotations

import typer

app: typer.Typer = typer.Typer(help="Evaluate a trained adapter.", no_args_is_help=True)


@app.command("run")
def run(
    version: str = typer.Option("release_v2", "--version", help="LiveCodeBench release tag."),
    samples: int = typer.Option(1, "--samples"),
    adapter: str | None = typer.Option(None, "--adapter"),
    lcb_dir: str = typer.Option("../LiveCodeBench", "--lcb-dir"),
) -> None:
    """Run the LiveCodeBench evaluator on a trained adapter."""
    raise NotImplementedError("hone evaluate run is implemented in T3.14")
