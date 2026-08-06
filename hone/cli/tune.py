"""CLI subcommand group: hyperparameter search."""

from __future__ import annotations

import itertools
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import typer
import yaml

from hone.log import setup

app: typer.Typer = typer.Typer(help="Search hyperparameters.", no_args_is_help=True)


REQUIRED_KEYS: tuple[str, ...] = (
    "learning_rate",
    "rank",
    "num_layers",
    "max_seq_length",
    "grad_accumulation_steps",
    "iters",
)


@dataclass(frozen=True, slots=True)
class TrialSpec:
    learning_rate: float
    rank: int
    num_layers: int
    max_seq_length: int
    grad_accumulation_steps: int
    iters: int


@dataclass(frozen=True, slots=True)
class TrialResult:
    trial_id: str
    validation_loss: float | None
    adapter_path: str
    status: str
    parameters: TrialSpec
    benchmark_metrics: dict[str, float]


VALIDATION_LOSS_PATTERN = re.compile(r"Val loss\s+([0-9]+(?:\.[0-9]+)?)")


def expand(search_space: dict[str, object], max_trials: int | None) -> list[TrialSpec]:
    """Build a deterministic Cartesian search, optionally bounded by a budget."""
    values: list[list[float | int]] = []
    for key in REQUIRED_KEYS:
        options = search_space.get(key)
        if not isinstance(options, list) or not options:
            raise ValueError(f"search space must define non-empty {key}")
        values.append(options)
    specs = [
        TrialSpec(
            learning_rate=float(combination[0]),
            rank=int(combination[1]),
            num_layers=int(combination[2]),
            max_seq_length=int(combination[3]),
            grad_accumulation_steps=int(combination[4]),
            iters=int(combination[5]),
        )
        for combination in itertools.product(*values)
    ]
    if max_trials is not None:
        if max_trials < 1:
            raise ValueError("max_trials must be positive")
        specs = specs[:max_trials]
    return specs


def loss(output: str) -> float | None:
    """Extract the lowest validation loss reported by MLX training."""
    values = [
        float(match.group(1)) for match in VALIDATION_LOSS_PATTERN.finditer(output)
    ]
    return min(values) if values else None


def metrics(path: Path) -> dict[str, float]:
    """Load optional benchmark metrics from a hook-produced JSON file."""
    if not path.exists():
        return {}
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError(f"benchmark metrics must be an object: {path}")
    return {
        str(key): float(value)
        for key, value in record.items()
        if isinstance(value, (int, float))
    }


def materialize(
    path: Path,
    base_config: dict[str, object],
    trial: TrialSpec,
    adapter_path: Path,
) -> None:
    """Materialize one MLX YAML configuration."""
    config = dict(base_config)
    config.update(
        {
            "learning_rate": trial.learning_rate,
            "num_layers": trial.num_layers,
            "max_seq_length": trial.max_seq_length,
            "grad_accumulation_steps": trial.grad_accumulation_steps,
            "iters": trial.iters,
            "adapter_path": str(adapter_path),
            "lora_parameters": {
                "keys": ["self_attn.q_proj", "self_attn.v_proj"],
                "rank": trial.rank,
                "scale": trial.rank * 2,
                "dropout": 0.05,
            },
        }
    )
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def score(result: TrialResult, objective: str) -> float | None:
    """Return the objective value for a result."""
    if objective == "validation_loss":
        return result.validation_loss
    return result.benchmark_metrics.get(objective)


def execute(
    trial_id: str,
    trial: TrialSpec,
    base_config: dict[str, object],
    output_dir: Path,
    mlx_binary: Path,
    device: str,
    benchmark_command: str | None,
    logger: logging.Logger,
) -> TrialResult:
    """Train and evaluate one isolated trial."""
    trial_dir = output_dir / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = trial_dir / "adapter"
    config_path = trial_dir / "config.yaml"
    log_path = trial_dir / "train.log"
    metrics_path = trial_dir / "metrics.json"
    materialize(config_path, base_config, trial, adapter_path)
    command = [
        sys.executable,
        str(mlx_binary),
        "--config",
        str(config_path),
    ]
    logger.info("starting %s: %s", trial_id, " ".join(command))
    environment = os.environ.copy()
    environment["HONE_DEVICE"] = device
    completed = subprocess.run(
        command,
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    output = completed.stdout + completed.stderr
    log_path.write_text(output, encoding="utf-8")
    validation_loss = loss(output)
    trial_metrics: dict[str, float] = {}
    status = "completed" if completed.returncode == 0 else "failed"
    if completed.returncode == 0 and benchmark_command:
        environment.update(
            {
                "ADAPTER_PATH": str(adapter_path),
                "TRIAL_DIR": str(trial_dir),
                "METRICS_PATH": str(metrics_path),
            }
        )
        benchmark = subprocess.run(
            benchmark_command,
            cwd=Path.cwd(),
            shell=True,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        (trial_dir / "benchmark.log").write_text(
            benchmark.stdout + benchmark.stderr, encoding="utf-8"
        )
        if benchmark.returncode != 0:
            status = "benchmark_failed"
        trial_metrics = metrics(metrics_path)
    if status == "completed" and validation_loss is None:
        status = "missing_validation_loss"
    return TrialResult(
        trial_id=trial_id,
        validation_loss=validation_loss,
        adapter_path=str(adapter_path),
        status=status,
        parameters=trial,
        benchmark_metrics=trial_metrics,
    )


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
    logger = setup(verbose=False)
    config_path = Path(config)
    space_path = Path(space)
    output_dir = Path(output)
    if not config_path.is_file():
        raise typer.BadParameter(f"config not found: {config_path}")
    if not space_path.is_file():
        raise typer.BadParameter(f"search space not found: {space_path}")

    base_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    search_space = yaml.safe_load(space_path.read_text(encoding="utf-8"))
    if not isinstance(base_config, dict) or not isinstance(search_space, dict):
        raise typer.BadParameter("configuration and search space must be YAML mappings")

    specs = expand(search_space, max_trials)
    output_dir.mkdir(parents=True, exist_ok=True)

    launcher_path = Path(__file__).resolve().parent.parent / "run.py"
    if not launcher_path.is_file():
        raise typer.BadParameter(f"launcher not found: {launcher_path}")

    results: list[TrialResult] = []
    for index, trial in enumerate(specs, 1):
        trial_id = f"trial-{index:03d}"
        results.append(
            execute(
                trial_id=trial_id,
                trial=trial,
                base_config=base_config,
                output_dir=output_dir,
                mlx_binary=launcher_path,
                device=device,
                benchmark_command=benchmark_command,
                logger=logger,
            )
        )

    (output_dir / "results.json").write_text(
        json.dumps([asdict(result) for result in results], indent=2, sort_keys=True),
        encoding="utf-8",
    )

    successful = [
        result
        for result in results
        if result.status == "completed" and score(result, objective) is not None
    ]
    if not successful:
        raise typer.BadParameter(f"no successful trial produced objective {objective}")
    if objective == "validation_loss":
        best = min(
            successful,
            key=lambda result: score(result, objective) or float("inf"),
        )
    else:
        best = max(
            successful,
            key=lambda result: score(result, objective) or float("-inf"),
        )
    (output_dir / "best.json").write_text(
        json.dumps(asdict(best), indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info(
        "best trial=%s objective=%s value=%s",
        best.trial_id,
        objective,
        score(best, objective),
    )


__all__ = ["app"]
