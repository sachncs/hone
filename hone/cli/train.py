"""CLI subcommand group: training."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import typer

from hone.log import setup
from hone.split import partition

app: typer.Typer = typer.Typer(help="Train adapters.", no_args_is_help=True)


def invoke_mlx(config: Path, device: str) -> int:
    """Invoke python -m hone.run with HONE_DEVICE set; return exit code."""
    logger = setup(verbose=False)
    environment = os.environ.copy()
    environment["HONE_DEVICE"] = device
    logger.info(
        "running python -m hone.run --config %s (HONE_DEVICE=%s)", config, device
    )
    completed = subprocess.run(
        [sys.executable, "-m", "hone.run", "--config", str(config)],
        env=environment,
        check=False,
    )
    return completed.returncode


def invoke_cuda(config: Path) -> int:
    """Invoke the Unsloth/CUDA training path; placeholder for Phase 3.10."""
    raise NotImplementedError("CUDA backend (Unsloth) is implemented in T3.10")


@app.command("code")
def code(
    config: str = typer.Option(
        "configs/code.yaml", "--config", help="Path to MLX LoRA config."
    ),
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
        exit_code = invoke_mlx(config_path, device)
    else:
        exit_code = invoke_cuda(config_path)
    raise typer.Exit(code=exit_code)


@app.command("swe")
def swe(
    config: str = typer.Option(
        "configs/swe.yaml", "--config", help="Path to MLX LoRA config."
    ),
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
        exit_code = invoke_mlx(config_path, device)
    else:
        exit_code = invoke_cuda(config_path)
    raise typer.Exit(code=exit_code)


@app.command("all")
def all_cmd(
    model: str = typer.Option("openbmb/MiniCPM5-1B", "--model"),
    layers: int = typer.Option(16, "--layers"),
    accum: int = typer.Option(32, "--accum"),
    seq_len: int = typer.Option(8192, "--seq-len"),
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

    previous_adapter: Path | None = None
    for repo, configs, data_path, adapter in FULL_SEQUENCE:
        if not data_path.is_file() or data_path.stat().st_size == 0:
            logger.info("preparing %s", data_path)
            mode = "codeforces-text" if "codeforces" in repo else "sft"
            prepare.all_cmd(
                repo=repo,
                configs=configs,
                split="train",
                output=str(data_path),
                mode=mode,
            )
        if not data_path.is_file() or data_path.stat().st_size == 0:
            raise typer.BadParameter(f"dataset missing after prepare: {data_path}")

        valid_path = data_path.parent / "valid.jsonl"
        fresh_valid = valid_path.is_file() and valid_path.stat().st_size > 0
        if not fresh_valid or data_path.stat().st_mtime > valid_path.stat().st_mtime:
            train_count, valid_count = partition(
                data_path, data_path, valid_path, ratio=0.05, seed=42
            )
            logger.info(
                "split %d train and %d valid from %s",
                train_count,
                valid_count,
                data_path.parent,
            )
        else:
            train_count = sum(1 for _ in data_path.open(encoding="utf-8"))
        iters = max(1, train_count)
        args = [
            "--model",
            model,
            "--train",
            "--data",
            str(data_path.parent),
            "--adapter-path",
            str(adapter),
            "--iters",
            str(iters),
            "--batch-size",
            "1",
            "--grad-accumulation-steps",
            str(accum),
            "--num-layers",
            str(layers),
            "--max-seq-length",
            str(seq_len),
            "--save-every",
            str(save_every),
            "--seed",
            "42",
        ]
        if "codeforces" not in repo:
            args.append("--mask-prompt")
        if previous_adapter is not None:
            args.extend(
                [
                    "--resume-adapter-file",
                    str(previous_adapter / "adapters.safetensors"),
                ]
            )

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


FULL_SEQUENCE: list[tuple[str, str, Path, Path]] = [
    (
        "ianncity/KIMI-K2.5-1000000x",
        "General-Distillation,PHD-Science,General-Math,MultilingualSTEM",
        Path("data/full/kimi/train.jsonl"),
        Path("artifacts/full/01-kimi"),
    ),
    (
        "Modotte/CodeX-7M-Non-Thinking",
        "default",
        Path("data/full/codex/train.jsonl"),
        Path("artifacts/full/02-codex"),
    ),
    (
        "inclusionAI/Ling-Coder-SFT",
        "default",
        Path("data/full/ling/train.jsonl"),
        Path("artifacts/full/03-ling"),
    ),
    (
        "open-r1/codeforces",
        "default",
        Path("data/full/codeforces/train.jsonl"),
        Path("artifacts/full/04-codeforces"),
    ),
    (
        "microsoft/rStar-Coder",
        "seed_sft",
        Path("data/full/rstar/train.jsonl"),
        Path("artifacts/full/05-rstar"),
    ),
]


__all__ = ["FULL_SEQUENCE", "app"]
