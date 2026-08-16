"""Per-trial training + benchmarking runner.

One :meth:`TrialRunner.run` call trains one :class:`TrialSpec` and
optionally invokes an external benchmark command. The runner is
isolated from the orchestration loop so it can be tested with a
fake launcher.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import yaml

from hone.tune.spec import TrialResult, TrialSpec, load_metrics, loss_from_output


class Launcher(Protocol):
    """Same port as :class:`hone.backends.Launcher`; re-declared for locality."""

    def run(self, args: list[str], env: dict[str, str]) -> int: ...


@dataclass(frozen=True)
class TrialPaths:
    """Per-trial filesystem layout."""

    trial_dir: Path
    config_path: Path
    adapter_path: Path
    log_path: Path
    metrics_path: Path


class TrialRunner:
    """Train + benchmark one trial in isolation."""

    def __init__(
        self,
        *,
        launcher: Launcher,
        launcher_args: list[str],
        device: str,
        benchmark_command: str | None,
        logger: logging.Logger,
    ) -> None:
        self._launcher = launcher
        self._launcher_args = launcher_args
        self._device = device
        self._benchmark_command = benchmark_command
        self._logger = logger

    def run(
        self,
        *,
        trial_id: str,
        trial: TrialSpec,
        base_config: dict[str, object],
        output_dir: Path,
    ) -> TrialResult:
        """Materialize the config, train, and benchmark the trial."""
        paths = _materialize_paths(output_dir, trial_id)
        _write_config(paths.config_path, base_config, trial, paths.adapter_path)
        return self._train_and_benchmark(trial_id, trial, paths)

    def _train_and_benchmark(
        self,
        trial_id: str,
        trial: TrialSpec,
        paths: TrialPaths,
    ) -> TrialResult:
        command = [
            sys.executable,
            *self._launcher_args,
            "--config",
            str(paths.config_path),
        ]
        self._logger.info("starting %s: %s", trial_id, " ".join(command))
        env = os.environ.copy()
        env["HONE_DEVICE"] = self._device
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        output = completed.stdout + completed.stderr
        paths.log_path.write_text(output, encoding="utf-8")
        validation_loss = loss_from_output(output)
        status = "completed" if completed.returncode == 0 else "failed"
        trial_metrics: dict[str, float] = {}
        if completed.returncode == 0 and self._benchmark_command is not None:
            status, trial_metrics = self._run_benchmark(paths)
        if status == "completed" and validation_loss is None:
            status = "missing_validation_loss"
        return TrialResult(
            trial_id=trial_id,
            validation_loss=validation_loss,
            adapter_path=str(paths.adapter_path),
            status=status,
            parameters=trial,
            benchmark_metrics=trial_metrics,
        )

    def _run_benchmark(self, paths: TrialPaths) -> tuple[str, dict[str, float]]:
        env = os.environ.copy()
        env.update(
            {
                "ADAPTER_PATH": str(paths.adapter_path),
                "TRIAL_DIR": str(paths.trial_dir),
                "METRICS_PATH": str(paths.metrics_path),
            }
        )
        benchmark = subprocess.run(
            self._benchmark_command or "",
            cwd=Path.cwd(),
            shell=True,
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        (paths.trial_dir / "benchmark.log").write_text(
            benchmark.stdout + benchmark.stderr, encoding="utf-8"
        )
        status = "benchmark_failed" if benchmark.returncode != 0 else "completed"
        return status, load_metrics(paths.metrics_path)


def _materialize_paths(output_dir: Path, trial_id: str) -> TrialPaths:
    """Compute per-trial paths and create the trial directory."""
    trial_dir = output_dir / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    return TrialPaths(
        trial_dir=trial_dir,
        config_path=trial_dir / "config.yaml",
        adapter_path=trial_dir / "adapter",
        log_path=trial_dir / "train.log",
        metrics_path=trial_dir / "metrics.json",
    )


def _write_config(
    path: Path,
    base_config: dict[str, object],
    trial: TrialSpec,
    adapter_path: Path,
) -> None:
    """Render one trial's MLX YAML config to disk."""
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


__all__ = ["Launcher", "TrialPaths", "TrialRunner"]
