"""CLI subcommand group: training (filled in Phase 3)."""

from __future__ import annotations

import typer

app: typer.Typer = typer.Typer(help="Train adapters.", no_args_is_help=True)


@app.command("code")
def code(
    config: str = typer.Option("configs/code.yaml", "--config", help="Path to MLX LoRA config."),
    device: str = typer.Option("gpu", "--device", help="gpu or cpu."),
    backend: str = typer.Option("mlx", "--backend", help="mlx or cuda."),
) -> None:
    """Train a coding adapter."""
    raise NotImplementedError("hone train code is implemented in T3.7")


@app.command("swe")
def swe(
    config: str = typer.Option("configs/swe.yaml", "--config", help="Path to MLX LoRA config."),
    device: str = typer.Option("gpu", "--device", help="gpu or cpu."),
    backend: str = typer.Option("mlx", "--backend", help="mlx or cuda."),
) -> None:
    """Train an SWE adapter."""
    raise NotImplementedError("hone train swe is implemented in T3.8")


@app.command("all")
def all(  # noqa: A001
    model: str = typer.Option("mlx-community/MiniCPM5-1B-4bit", "--model"),
    layers: int = typer.Option(8, "--layers"),
    accum: int = typer.Option(16, "--accum"),
    seq_len: int = typer.Option(4096, "--seq-len"),
    save_every: int = typer.Option(100000, "--save-every"),
) -> None:
    """Run the full training sequence across every dataset."""
    raise NotImplementedError("hone train all is implemented in T3.9")
