"""CLI subcommand group: training."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import typer

from hone.log import setup
from hone.split import partition

app: typer.Typer = typer.Typer(help="Train adapters.", no_args_is_help=True)


NAN_THRESHOLD = 3


def _stage_marker(data_path: Path, max_tokens: int, max_samples: int) -> Path:
    """Sidecar written after a `prepare` step that filtered by `max_tokens`.

    Re-runs of `train all` reuse the JSONL only when the requested
    `max_tokens` matches the recorded one. Otherwise the data is
    rebuilt so a forgotten `--max-tokens` cannot silently re-introduce
    long-tail records that mask-prompt truncates to NaN.
    """
    name = f".prepared-max-tokens-{max_tokens}-samples-{max_samples}.json"
    return data_path.parent / name


def _needs_prepare(data_path: Path, max_tokens: int, max_samples: int) -> bool:
    if not data_path.is_file() or data_path.stat().st_size == 0:
        return True
    if max_tokens <= 0 and max_samples <= 0:
        return False
    return not _stage_marker(data_path, max_tokens, max_samples).is_file()


def _is_divergent(log_text: str) -> bool:
    """Return True if the staged training shows signs of runaway loss.

    A clean run produces zero `Train loss nan` lines; long-tail records
    truncated to an empty loss target yield them within the first few
    iterations and the optimizer state never recovers. Three or more
    NaN samples is treated as a hard stop.
    """
    return len(re.findall(r"Train loss nan", log_text)) >= NAN_THRESHOLD


def _select_stages(stages: str) -> list[tuple[str, str, Path, Path]]:
    """Filter FULL_SEQUENCE by user-supplied selection.

    Accepts `all`, 1-based indices (`02`, `02,03,04`), or stage
    directory names (`02-codex`). Whichever is supplied, an empty or
    `all` value yields the full sequence unchanged.
    """
    if stages.strip().lower() in {"", "all"}:
        return list(FULL_SEQUENCE)
    selected: list[tuple[str, str, Path, Path]] = []
    for token in stages.split(","):
        key = token.strip()
        if not key:
            continue
        if key.isdigit():
            index = int(key) - 1
            if index < 0 or index >= len(FULL_SEQUENCE):
                raise typer.BadParameter(
                    f"--stages index out of range: {key} "
                    f"(valid 1..{len(FULL_SEQUENCE)})"
                )
            selected.append(FULL_SEQUENCE[index])
            continue
        matched = [entry for entry in FULL_SEQUENCE if entry[3].name == key]
        if not matched:
            raise typer.BadParameter(
                f"--stages name not found: {key}; "
                f"valid names: {[e[3].name for e in FULL_SEQUENCE]}"
            )
        selected.extend(matched)
    seen: set[Path] = set()
    deduped: list[tuple[str, str, Path, Path]] = []
    for entry in selected:
        adapter = entry[3]
        if adapter in seen:
            continue
        seen.add(adapter)
        deduped.append(entry)
    if not deduped:
        raise typer.BadParameter("--stages produced an empty selection")
    return deduped


def _invoke_mlx(config: Path, device: str) -> int:
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
        exit_code = _invoke_mlx(config_path, device)
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
        exit_code = _invoke_mlx(config_path, device)
    else:
        exit_code = invoke_cuda(config_path)
    raise typer.Exit(code=exit_code)


def _run_stage_capture(
    cmd: list[str],
    env: dict[str, str],
    log_path: Path,
) -> tuple[int, str]:
    """Tee stage output to stdout and `log_path`, return (rc, log_text).

    Live output to stdout lets an outer `tee` (e.g. `train.sh`) keep
    capturing the unified run log; the per-stage log is what the
    divergence watchdog scans.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    captured: list[str] = []
    with log_path.open("w", encoding="utf-8") as handle, subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1,
    ) as process:
        stdout = process.stdout
        if stdout is None:
            return process.wait(), ""
        for line in stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            handle.write(line)
            handle.flush()
            captured.append(line)
        return_code = process.wait()
    return return_code, "".join(captured)


@app.command("all")
def all_cmd(
    model: str = typer.Option("openbmb/MiniCPM5-1B", "--model"),
    layers: int = typer.Option(16, "--layers"),
    accum: int = typer.Option(32, "--accum"),
    seq_len: int = typer.Option(
        4096,
        "--seq-len",
        help="Max sequence length; 4096 is the recommended default for the M3 Pro "
        "(18 GB unified memory). 8192 quadruples attention cost and risks OOM "
        "with mixed-length batches. Pair with 'hone prepare all --max-tokens N' "
        "where N <= seq_len to drop long-tail records upstream.",
    ),
    save_every: int = typer.Option(100000, "--save-every"),
    max_tokens: int = typer.Option(
        4096,
        "--max-tokens",
        min=0,
        help="Drop records longer than this during prepare; 0 disables. "
        "Must be <= --seq-len. A sidecar marker (.prepared-max-tokens-N.json) "
        "tracks the filter value used so a re-run with a different value "
        "forces a rebuild.",
    ),
    max_samples: int = typer.Option(
        0,
        "--max-samples",
        min=0,
        help="Cap on rows written by each stage's `prepare` step. 0 means "
        "process the entire HF stream. Useful for bounding runtime on "
        "multi-million-row datasets (CodeX-7M, KIMI-K2.5); the sidecar "
        "marker is keyed on this value too.",
    ),
    learning_rate: float = typer.Option(
        5e-6,
        "--lr",
        help="LoRA learning rate. Conservative default for masked-prompt "
        "SFT on MiniCPM-1B; raise to 1e-5 only after a clean loss curve.",
    ),
    max_iters: int | None = typer.Option(
        None,
        "--max-iters",
        help="Cap iterations per stage (useful for smoke tests). Defaults "
        "to one full pass over the training JSONL.",
    ),
    grad_checkpoint: bool = typer.Option(
        False,
        "--grad-checkpoint/--no-grad-checkpoint",
        help="Enable gradient checkpointing during training. Trades ~30% "
        "of throughput for substantially lower peak memory; recommended "
        "on the M3 Pro's 18 GB unified memory when --seq-len > 2048.",
    ),
    stages: str = typer.Option(
        "all",
        "--stages",
        help="Subset of FULL_SEQUENCE to run. Use 'all', comma-separated "
        "1-based indices ('02,03,04'), or stage names ('02-codex'). "
        "Code-only: --stages 02-codex,03-ling,04-codeforces,05-rstar.",
    ),
) -> None:
    """Run the full training sequence across every dataset.

    Defaults assume data is pre-filtered via 'hone prepare all --max-tokens 4096'.
    The wrapper enforces this: passing --max-tokens writes a sidecar marker so a
    later rerun with a different value rebuilds the JSONL automatically.

    A divergence watchdog scans each stage's output for repeated NaN losses
    and aborts the run, removing the poisoned adapter so subsequent stages
    do not resume from it.
    """
    from hone.cli import prepare

    logger = setup(verbose=False)
    logger.info("starting full-sequence training with model=%s", model)
    artifacts_root = Path("artifacts/full")
    artifacts_root.mkdir(parents=True, exist_ok=True)
    data_root = Path("data/full")
    data_root.mkdir(parents=True, exist_ok=True)
    if max_tokens > seq_len:
        raise typer.BadParameter(
            f"--max-tokens ({max_tokens}) must be <= --seq-len ({seq_len})"
        )

    selected = _select_stages(stages)
    logger.info(
        "stages selected: %s",
        [entry[3].name for entry in selected],
    )

    previous_adapter: Path | None = None
    for repo, configs, data_path, adapter in selected:
        marker = _stage_marker(data_path, max_tokens, max_samples)
        if _needs_prepare(data_path, max_tokens, max_samples):
            if data_path.is_file():
                logger.info(
                    "preparing %s (rebuild; marker %s missing or stale)",
                    data_path,
                    marker.name,
                )
            else:
                logger.info("preparing %s", data_path)
            mode = "codeforces-text" if "codeforces" in repo else "sft"
            prepare.all_cmd(
                repo=repo,
                configs=configs,
                split="train",
                output=str(data_path),
                mode=mode,
                max_tokens=max_tokens,
                max_samples=max_samples,
                tokenizer_model=model,
            )
        if not data_path.is_file() or data_path.stat().st_size == 0:
            raise typer.BadParameter(f"dataset missing after prepare: {data_path}")
        marker.write_text(
            json.dumps(
                {
                    "max_tokens": max_tokens,
                    "max_samples": max_samples,
                    "data": str(data_path),
                }
            ),
            encoding="utf-8",
        )

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
        if max_iters is not None:
            iters = min(iters, max_iters)
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
            "--learning-rate",
            str(learning_rate),
            "--save-every",
            str(save_every),
            "--seed",
            "42",
        ]
        if "codeforces" not in repo:
            args.append("--mask-prompt")
        if grad_checkpoint:
            args.append("--grad-checkpoint")
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
        stage_log = Path("artifacts/logs") / f"stage-{adapter.name}.log"
        return_code, log_text = _run_stage_capture(
            [sys.executable, "-m", "hone.run", *args], environment, stage_log
        )
        if _is_divergent(log_text):
            shutil.rmtree(adapter, ignore_errors=True)
            logger.error(
                "aborting: stage=%s produced NaN losses in %s; "
                "adapter %s removed so downstream stages do not resume "
                "from poisoned weights. Inspect %s and rerun.",
                adapter.name,
                stage_log,
                adapter,
                data_path,
            )
            raise typer.Exit(code=2)
        if return_code != 0:
            raise typer.Exit(code=return_code)
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


__all__ = [
    "FULL_SEQUENCE",
    "_is_divergent",
    "_needs_prepare",
    "_select_stages",
    "_stage_marker",
    "app",
]
