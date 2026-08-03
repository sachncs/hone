"""CLI subcommand group: hyperparameter search (filled in Phase 3)."""

from __future__ import annotations

import typer

app: typer.Typer = typer.Typer(help="Search hyperparameters.", no_args_is_help=True)


@app.command("run")
def run(
    config: str = typer.Option(..., "--config", help="Base MLX LoRA config."),
    space: str = typer.Option(..., "--space", help="Search-space YAML."),
    output: str = typer.Option(..., "--output", help="Trial output directory."),
    device: str = typer.Option("gpu", "--device"),
    max_trials: int | None = typer.Option(None, "--max-trials"),
    objective: str = typer.Option("validation_loss", "--objective"),
    benchmark_command: str | None = typer.Option(None, "--benchmark-command"),
) -> None:
    """Run hyperparameter trials and select the best."""
    raise NotImplementedError("hone tune run is implemented in T3.13")
