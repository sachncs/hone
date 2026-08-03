#!/usr/bin/env python3
"""Run reproducible MLX LoRA trials and select the best validation checkpoint."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from finetune.logging import configure_logging

type JsonObject = dict[str, object]
VALIDATION_LOSS_PATTERN = re.compile(r"Val loss\s+([0-9]+(?:\.[0-9]+)?)")


@dataclass(frozen=True, slots=True)
class TrialSpec:
    """One complete, reproducible training configuration."""

    learning_rate: float
    rank: int
    num_layers: int
    max_seq_length: int
    grad_accumulation_steps: int
    iters: int


@dataclass(frozen=True, slots=True)
class TrialResult:
    """Persisted result for one trial."""

    trial_id: str
    validation_loss: float | None
    adapter_path: str
    status: str
    parameters: TrialSpec
    benchmark_metrics: dict[str, float]


def objective_value(result: TrialResult, objective: str) -> float | None:
    """Return an objective value, with validation loss treated as minimization."""
    if objective == "validation_loss":
        return result.validation_loss
    return result.benchmark_metrics.get(objective)


def build_trials(search_space: JsonObject, max_trials: int | None) -> list[TrialSpec]:
    """Build a deterministic Cartesian search, optionally bounded by a budget."""
    parameter_names = (
        "learning_rate",
        "rank",
        "num_layers",
        "max_seq_length",
        "grad_accumulation_steps",
        "iters",
    )
    values: list[list[float | int]] = []
    for parameter_name in parameter_names:
        parameter_values = search_space.get(parameter_name)
        if not isinstance(parameter_values, list) or not parameter_values:
            raise ValueError(f"search space must define non-empty {parameter_name}")
        values.append(parameter_values)
    trials = [
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
        trials = trials[:max_trials]
    return trials


def parse_validation_loss(output: str) -> float | None:
    """Extract the lowest validation loss reported by MLX training."""
    losses = [
        float(match.group(1)) for match in VALIDATION_LOSS_PATTERN.finditer(output)
    ]
    return min(losses) if losses else None


def load_metrics(path: Path) -> dict[str, float]:
    """Load optional benchmark metrics from a hook-produced JSON file."""
    if not path.exists():
        return {}
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError(f"benchmark metrics must be an object: {path}")
    return {
        str(key): float(value)
        for key, value in record.items()
        if isinstance(value, int | float)
    }


def write_trial_config(
    path: Path,
    base_config: JsonObject,
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


def run_trial(
    trial_id: str,
    trial: TrialSpec,
    base_config: JsonObject,
    output_dir: Path,
    mlx_binary: str,
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
    write_trial_config(config_path, base_config, trial, adapter_path)
    command = [mlx_binary, "--config", str(config_path)]
    if mlx_binary.endswith("run_mlx_lora.py"):
        command.insert(0, sys.executable)
    logger.info("starting %s: %s", trial_id, " ".join(command))
    environment = os.environ.copy()
    environment["FINETUNE_DEVICE"] = device
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
    validation_loss = parse_validation_loss(output)
    benchmark_metrics: dict[str, float] = {}
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
            text=True,
            capture_output=True,
            env=environment,
        )
        (trial_dir / "benchmark.log").write_text(
            benchmark.stdout + benchmark.stderr, encoding="utf-8"
        )
        if benchmark.returncode != 0:
            status = "benchmark_failed"
        benchmark_metrics = load_metrics(metrics_path)
    if status == "completed" and validation_loss is None:
        status = "missing_validation_loss"
    result = TrialResult(
        trial_id=trial_id,
        validation_loss=validation_loss,
        adapter_path=str(adapter_path),
        status=status,
        parameters=trial,
        benchmark_metrics=benchmark_metrics,
    )
    logger.info(
        "%s status=%s validation_loss=%s metrics=%s",
        trial_id,
        status,
        validation_loss,
        benchmark_metrics,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--search-space", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mlx-binary", default=".venv/bin/mlx_lm.lora")
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default=os.environ.get("FINETUNE_DEVICE", "gpu"),
    )
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--benchmark-command")
    parser.add_argument(
        "--objective",
        default="validation_loss",
        help="validation_loss (lower) or a benchmark metric (higher)",
    )
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args()
    if arguments.mlx_binary == ".venv/bin/mlx_lm.lora":
        arguments.mlx_binary = str(Path.cwd() / "scripts/run_mlx_lora.py")
    logger = configure_logging(arguments.verbose)
    base_config = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
    search_space = yaml.safe_load(arguments.search_space.read_text(encoding="utf-8"))
    if not isinstance(base_config, dict) or not isinstance(search_space, dict):
        raise SystemExit("configuration and search space must be YAML objects")
    trials = build_trials(search_space, arguments.max_trials)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        run_trial(
            f"trial-{index:03d}",
            trial,
            base_config,
            arguments.output_dir,
            arguments.mlx_binary,
            arguments.device,
            arguments.benchmark_command,
            logger,
        )
        for index, trial in enumerate(trials, 1)
    ]
    result_records = [asdict(result) for result in results]
    (arguments.output_dir / "results.json").write_text(
        json.dumps(result_records, indent=2, sort_keys=True), encoding="utf-8"
    )
    successful = [
        result
        for result in results
        if result.status == "completed"
        and objective_value(result, arguments.objective) is not None
    ]
    if not successful:
        raise SystemExit(
            f"no successful trial produced objective {arguments.objective}"
        )
    if arguments.objective == "validation_loss":
        best = min(
            successful,
            key=lambda result: (
                objective_value(result, arguments.objective) or float("inf")
            ),
        )
    else:
        best = max(
            successful,
            key=lambda result: (
                objective_value(result, arguments.objective) or float("-inf")
            ),
        )
    (arguments.output_dir / "best.json").write_text(
        json.dumps(asdict(best), indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info(
        "best trial=%s objective=%s value=%s",
        best.trial_id,
        arguments.objective,
        objective_value(best, arguments.objective),
    )


if __name__ == "__main__":
    main()
