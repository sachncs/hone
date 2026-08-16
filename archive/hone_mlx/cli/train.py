"""CLI subcommand group: training.

Thin transport layer over :mod:`hone.train`. The ``code`` and
``swe`` commands delegate to :class:`hone.backends.SubprocessLauncher`;
the ``all`` command delegates to
:class:`hone.train.TrainOrchestrator`.

Two CLI affordances the application layer does NOT know about:

* :func:`stage_marker` — exposed for tests and for direct callers
  that want to inspect / write sidecar markers without going
  through the orchestrator.
* :func:`select_stages` — re-exported from the application layer
  so existing tests that import it via ``hone.cli.train`` keep
  working.
"""

from __future__ import annotations

import os
from pathlib import Path

import typer

from hone.backends import SubprocessLauncher
from hone.errors import PipelineError
from hone.log import setup
from hone.train import StageMarker, StageRunner, TrainOrchestrator
from hone.train.divergence import DivergenceDetector
from hone.train.orchestrator import OrchestratorOptions, PrepareFn
from hone.train.stages import full_sequence as _full_sequence

app: typer.Typer = typer.Typer(help="Train adapters.", no_args_is_help=True)


# ---------------------------------------------------------------------------
# Public helpers — kept for test compatibility
# ---------------------------------------------------------------------------


def stage_marker(data_path: Path, max_tokens: int, max_samples: int) -> Path:
    """Return the sidecar marker path for a prepared JSONL."""
    return StageMarker(data_path, max_tokens, max_samples).path


def _invoke_mlx(config: Path, device: str) -> int:
    """Invoke ``python -m hone.run`` with ``HONE_DEVICE`` set; return rc."""
    logger = setup(verbose=False)
    environment = os.environ.copy()
    environment["HONE_DEVICE"] = device
    logger.info(
        "running python -m hone.run --config %s (HONE_DEVICE=%s)", config, device
    )
    return SubprocessLauncher().run(["--config", str(config)], environment)


def _cuda_placeholder() -> int:
    """Stub for the Unsloth/CUDA backend."""
    raise NotImplementedError("CUDA backend (Unsloth) is implemented in T3.10")


def _select_stages(selector: str) -> list:
    """CLI-facing wrapper that converts :class:`ValueError` for typer."""
    from hone.train.stages import select_stages as _select

    try:
        return list(_select(selector))
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error


def _needs_prepare(data_path: Path, max_tokens: int, max_samples: int) -> bool:
    """CLI-facing wrapper that re-exports :func:`needs_rebuild`."""
    from hone.train.marker import needs_rebuild

    return needs_rebuild(data_path, max_tokens, max_samples)


def _is_divergent(log_text: str) -> bool:
    """CLI-facing wrapper that re-exports :class:`DivergenceDetector`."""
    return DivergenceDetector().divergent(log_text)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def _dispatch_backend(config_path: Path, device: str, backend: str) -> int:
    """Resolve the chosen backend; raise on unknown or unsupported."""
    if backend not in {"mlx", "cuda"}:
        raise typer.BadParameter("--backend must be 'mlx' or 'cuda'")
    if not config_path.is_file():
        raise typer.BadParameter(f"config not found: {config_path}")
    return _invoke_mlx(config_path, device) if backend == "mlx" else _cuda_placeholder()


@app.command("code")
def code(
    config: str = typer.Option(
        "configs/code.yaml", "--config", help="Path to MLX LoRA config."
    ),
    device: str = typer.Option("gpu", "--device", help="gpu or cpu."),
    backend: str = typer.Option("mlx", "--backend", help="mlx or cuda."),
) -> None:
    """Train a coding adapter."""
    config_path = Path(config)
    exit_code = _dispatch_backend(config_path, device, backend)
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
    config_path = Path(config)
    exit_code = _dispatch_backend(config_path, device, backend)
    raise typer.Exit(code=exit_code)


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
        help="Subset of the full sequence to run. Use 'all', comma-separated "
        "1-based indices ('02,03,04'), or stage names ('02-codex'). "
        "Code-only: --stages 02-codex,03-ling,04-codeforces,05-rstar.",
    ),
) -> None:
    """Run the full training sequence across every dataset."""
    if max_tokens > seq_len:
        raise typer.BadParameter(
            f"--max-tokens ({max_tokens}) must be <= --seq-len ({seq_len})"
        )
    options = OrchestratorOptions(
        model=model,
        layers=layers,
        accum=accum,
        seq_len=seq_len,
        save_every=save_every,
        max_tokens=max_tokens,
        max_samples=max_samples,
        learning_rate=learning_rate,
        max_iters=max_iters,
        grad_checkpoint=grad_checkpoint,
    )
    runner = StageRunner(log_path=options.log_dir / "_placeholder.log")
    prepare_fn = _make_prepare_fn()
    orchestrator = TrainOrchestrator(
        options=options,
        runner=runner,
        prepare_fn=prepare_fn,
        logger=setup(verbose=False),
    )
    try:
        results = orchestrator.run(stages)
    except PipelineError as error:
        raise typer.BadParameter(str(error)) from error

    if any(result.divergent for result in results):
        raise typer.Exit(code=2)
    if results and results[-1].return_code != 0:
        raise typer.Exit(code=results[-1].return_code)


def _make_prepare_fn() -> PrepareFn:
    """Build a prepare callback that delegates to the prepare service."""
    from hone.cli import prepare as prepare_module

    def prepare_fn(**kwargs: object) -> None:
        prepare_module.all_cmd(**kwargs)  # type: ignore[arg-type]

    return prepare_fn


# ---------------------------------------------------------------------------
# Public re-exports for tests
# ---------------------------------------------------------------------------


FULL_SEQUENCE = _full_sequence()


__all__ = [
    "FULL_SEQUENCE",
    "_is_divergent",
    "_needs_prepare",
    "_select_stages",
    "app",
    "stage_marker",
]
