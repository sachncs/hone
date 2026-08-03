"""CLI subcommand group: training."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import typer

from hone.log import setup

app: typer.Typer = typer.Typer(help="Train adapters.", no_args_is_help=True)


def _run_mlx(config: Path, device: str) -> int:
    """Invoke python -m hone.run with HONE_DEVICE set; return exit code."""
    logger = setup(verbose=False)
    environment = os.environ.copy()
    environment["HONE_DEVICE"] = device
    logger.info("running python -m hone.run --config %s (HONE_DEVICE=%s)", config, device)
    completed = subprocess.run(
        [sys.executable, "-m", "hone.run", "--config", str(config)],
        env=environment,
        check=False,
    )
    return completed.returncode


def _run_cuda(config: Path) -> int:
    """Invoke the Unsloth/CUDA training path; placeholder for Phase 3.10."""
    raise NotImplementedError("CUDA backend (Unsloth) is implemented in T3.10")


@app.command("code")
def code(
    config: str = typer.Option("configs/code.yaml", "--config", help="Path to MLX LoRA config."),
    device: str = typer.Option("gpu", "--device", help="gpu or cpu."),
    backend: str = typer.Option("mlx", "--backend", help="mlx or cuda."),
) -> None:
    """Train a coding adapter."""
    if backend not in {"mlx", "cuda"}:
        raise typer.BadParameter("--backend must be 'mlx' or 'cuda'")
    config_path = Path(config)
    if not config_path.is_file():
        raise typer.BadParameter(f"config not found: {config_path}")
    if backend == "mlx":
        exit_code = _run_mlx(config_path, device)
    else:
        exit_code = _run_cuda(config_path)
    raise typer.Exit(code=exit_code)


@app.command("swe")
def swe(
    config: str = typer.Option("configs/swe.yaml", "--config", help="Path to MLX LoRA config."),
    device: str = typer.Option("gpu", "--device", help="gpu or cpu."),
    backend: str = typer.Option("mlx", "--backend", help="mlx or cuda."),
) -> None:
    """Train an SWE adapter."""
    if backend not in {"mlx", "cuda"}:
        raise typer.BadParameter("--backend must be 'mlx' or 'cuda'")
    config_path = Path(config)
    if not config_path.is_file():
        raise typer.BadParameter(f"config not found: {config_path}")
    if backend == "mlx":
        exit_code = _run_mlx(config_path, device)
    else:
        exit_code = _run_cuda(config_path)
    raise typer.Exit(code=exit_code)


@app.command("all")
def all_cmd(
    model: str = typer.Option("mlx-community/MiniCPM5-1B-4bit", "--model"),
    layers: int = typer.Option(8, "--layers"),
    accum: int = typer.Option(16, "--accum"),
    seq_len: int = typer.Option(4096, "--seq-len"),
    save_every: int = typer.Option(100000, "--save-every"),
) -> None:
    """Run the full training sequence across every dataset."""
    from hone.cli import prepare

    logger = setup(verbose=False)
    logger.info("starting full-sequence training with model=%s", model)
    artifacts_root = Path("artifacts/full")
    artifacts_root.mkdir(parents=True, exist_ok=True)
    data_root = Path("data/full")
    data_root.mkdir(parents=True, exist_ok=True)

    sequence = [
        (
            "ianncity/KIMI-K2.5-1000000x",
            "General-Distillation,PHD-Science,General-Math,MultilingualSTEM",
            data_root / "kimi" / "train.jsonl",
            artifacts_root / "01-kimi",
        ),
        (
            "Modotte/CodeX-7M-Non-Thinking",
            "default",
            data_root / "codex" / "train.jsonl",
            artifacts_root / "02-codex",
        ),
        (
            "inclusionAI/Ling-Coder-SFT",
            "default",
            data_root / "ling" / "train.jsonl",
            artifacts_root / "03-ling",
        ),
        (
            "open-r1/codeforces",
            "default",
            data_root / "codeforces" / "train.jsonl",
            artifacts_root / "04-codeforces",
        ),
    ]
    previous_adapter: Path | None = None
    for repo, configs, data_path, adapter in sequence:
        if not data_path.exists():
            logger.info("preparing %s", data_path)
            mode = "codeforces-text" if "codeforces" in repo else "sft"
            prepare.all_cmd(repo=repo, configs=configs, output=str(data_path), mode=mode)
        if not data_path.is_file():
            raise typer.BadParameter(f"dataset missing after prepare: {data_path}")

        iters = max(1, sum(1 for _ in data_path.open(encoding="utf-8")) - 1)
        args = [
            "--model", model,
            "--train",
            "--data", str(data_path.parent),
            "--adapter-path", str(adapter),
            "--iters", str(iters),
            "--batch-size", "1",
            "--grad-accumulation-steps", str(accum),
            "--num-layers", str(layers),
            "--max-seq-length", str(seq_len),
            "--save-every", str(save_every),
            "--seed", "42",
        ]
        if "codeforces" not in repo:
            args.append("--mask-prompt")
        if previous_adapter is not None:
            args.extend(["--resume-adapter-file", str(previous_adapter / "adapters.safetensors")])

        environment = os.environ.copy()
        environment["HONE_DEVICE"] = "gpu"
        logger.info("training %s iters=%d", adapter.name, iters)
        completed = subprocess.run(
            [sys.executable, "-m", "hone.run", *args],
            env=environment,
            check=False,
        )
        if completed.returncode != 0:
            raise typer.Exit(code=completed.returncode)
        previous_adapter = adapter


__all__ = ["app"]
