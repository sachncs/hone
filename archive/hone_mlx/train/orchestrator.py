"""Training-sequence orchestrator.

Walks the selected :class:`~hone.train.stages.Stage` list, ensuring
each stage's JSONL is prepared (re-building when the sidecar
marker is stale), splitting train/valid, invoking the trainer via
:class:`~hone.train.runner.StageRunner`, and aborting on divergence
with the poisoned adapter removed.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from hone.errors import PipelineError
from hone.split import partition_file
from hone.train.divergence import DivergenceDetector
from hone.train.marker import StageMarker, needs_rebuild
from hone.train.runner import StageRunner
from hone.train.stages import Stage, select_stages


@dataclass(frozen=True)
class OrchestratorOptions:
    """Tunables for :class:`TrainOrchestrator`."""

    model: str = "openbmb/MiniCPM5-1B"
    layers: int = 16
    accum: int = 32
    seq_len: int = 4096
    save_every: int = 100_000
    max_tokens: int = 4096
    max_samples: int = 0
    learning_rate: float = 5e-6
    max_iters: int | None = None
    grad_checkpoint: bool = False
    log_dir: Path = Path("artifacts/logs")
    artifacts_root: Path = Path("artifacts/full")
    data_root: Path = Path("data/full")


@dataclass(frozen=True)
class StageResult:
    """Outcome of one stage; surfaces exit code and divergence flag."""

    stage: Stage
    return_code: int
    divergent: bool


class TrainOrchestrator:
    """Drive the full training sequence end-to-end."""

    def __init__(
        self,
        *,
        options: OrchestratorOptions,
        runner: StageRunner,
        detector: DivergenceDetector | None = None,
        logger: logging.Logger | None = None,
        prepare_fn: PrepareFn | None = None,
    ) -> None:
        self._options = options
        self._runner = runner
        self._detector = detector or DivergenceDetector()
        self._logger = logger or logging.getLogger("hone.train")
        self._prepare_fn = prepare_fn

    def run(self, selector: str = "all") -> list[StageResult]:
        """Run every selected stage; abort on the first divergence."""
        try:
            stages = select_stages(selector)
        except ValueError as error:
            raise PipelineError(str(error)) from error
        self._logger.info(
            "stages selected: %s", [stage.adapter_path.name for stage in stages]
        )

        self._options.artifacts_root.mkdir(parents=True, exist_ok=True)
        self._options.data_root.mkdir(parents=True, exist_ok=True)
        self._options.log_dir.mkdir(parents=True, exist_ok=True)

        previous_adapter: Path | None = None
        results: list[StageResult] = []
        for stage in stages:
            self._ensure_prepared(stage)
            self._ensure_split(stage)
            iters = self._iteration_count(stage)
            self._logger.info("training %s iters=%d", stage.adapter_path.name, iters)
            args = self._train_args(stage, iters, previous_adapter)
            env = os.environ.copy()
            env["HONE_DEVICE"] = "gpu"
            log_path = self._options.log_dir / f"stage-{stage.adapter_path.name}.log"
            self._runner.run.__self__._log_path = log_path  # type: ignore[attr-defined]
            return_code = self._runner.run(args, env)
            divergent = self._detector.divergent(self._runner.captured_text)
            results.append(
                StageResult(stage=stage, return_code=return_code, divergent=divergent)
            )
            if divergent:
                shutil.rmtree(stage.adapter_path, ignore_errors=True)
                self._logger.error(
                    "aborting: stage=%s produced NaN losses in %s; "
                    "adapter %s removed so downstream stages do not resume "
                    "from poisoned weights. Inspect %s and rerun.",
                    stage.adapter_path.name,
                    log_path,
                    stage.adapter_path,
                    stage.data_path,
                )
                return results
            if return_code != 0:
                return results
            previous_adapter = stage.adapter_path
        return results

    # ------------------------------------------------------------------
    # Stage preparation
    # ------------------------------------------------------------------

    def _ensure_prepared(self, stage: Stage) -> None:
        marker = StageMarker(
            stage.data_path, self._options.max_tokens, self._options.max_samples
        )
        if not needs_rebuild(
            stage.data_path, self._options.max_tokens, self._options.max_samples
        ):
            return
        if stage.data_path.is_file():
            self._logger.info(
                "preparing %s (rebuild; marker %s missing or stale)",
                stage.data_path,
                marker.path.name,
            )
        else:
            self._logger.info("preparing %s", stage.data_path)
        if self._prepare_fn is None:
            raise PipelineError(
                "no prepare function registered; orchestrator cannot rebuild datasets"
            )
        mode = "codeforces-text" if "codeforces" in stage.repo else "sft"
        self._prepare_fn(
            repo=stage.repo,
            configs=stage.configs,
            split="train",
            output=str(stage.data_path),
            mode=mode,
            max_tokens=self._options.max_tokens,
            max_samples=self._options.max_samples,
            tokenizer_model=self._options.model,
        )
        if not stage.data_path.is_file() or stage.data_path.stat().st_size == 0:
            raise PipelineError(f"dataset missing after prepare: {stage.data_path}")
        marker.write()

    def _ensure_split(self, stage: Stage) -> None:
        valid_path = stage.data_path.parent / "valid.jsonl"
        fresh_valid = valid_path.is_file() and valid_path.stat().st_size > 0
        if (
            fresh_valid
            and stage.data_path.stat().st_mtime <= valid_path.stat().st_mtime
        ):
            return
        train_count, valid_count = partition_file(
            stage.data_path, stage.data_path, valid_path, ratio=0.05, seed=42
        )
        self._logger.info(
            "split %d train and %d valid from %s",
            train_count,
            valid_count,
            stage.data_path.parent,
        )

    def _iteration_count(self, stage: Stage) -> int:
        with stage.data_path.open(encoding="utf-8") as handle:
            train_lines = sum(1 for _ in handle)
        iters = max(1, train_lines)
        if self._options.max_iters is not None:
            iters = min(iters, self._options.max_iters)
        return iters

    # ------------------------------------------------------------------
    # Argument construction
    # ------------------------------------------------------------------

    def _train_args(
        self,
        stage: Stage,
        iters: int,
        previous_adapter: Path | None,
    ) -> list[str]:
        args = [
            "--model",
            self._options.model,
            "--train",
            "--data",
            str(stage.data_path.parent),
            "--adapter-path",
            str(stage.adapter_path),
            "--iters",
            str(iters),
            "--batch-size",
            "1",
            "--grad-accumulation-steps",
            str(self._options.accum),
            "--num-layers",
            str(self._options.layers),
            "--max-seq-length",
            str(self._options.seq_len),
            "--learning-rate",
            str(self._options.learning_rate),
            "--save-every",
            str(self._options.save_every),
            "--seed",
            "42",
        ]
        if "codeforces" not in stage.repo:
            args.append("--mask-prompt")
        if self._options.grad_checkpoint:
            args.append("--grad-checkpoint")
        if previous_adapter is not None:
            args.extend(
                [
                    "--resume-adapter-file",
                    str(previous_adapter / "adapters.safetensors"),
                ]
            )
        return args


# Type alias for the prepare function dependency.
PrepareFn = Callable[..., None]


__all__ = ["OrchestratorOptions", "PrepareFn", "StageResult", "TrainOrchestrator"]


__all__ = ["OrchestratorOptions", "StageResult", "TrainOrchestrator"]
