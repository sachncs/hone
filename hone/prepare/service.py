"""Prepare service — the application-layer orchestrator.

Five public functions, one per ``hone prepare`` subcommand:

* :func:`prepare_local_file` — normalize and split a local JSONL.
* :func:`prepare_reservoir_sample` — reservoir-sample an HF stream.
* :func:`prepare_swe` — build SWE-bench SFT rows.
* :func:`prepare_stream` — materialize every row of an HF config
  with optional token-length filtering.
* :func:`prepare_eval_prompts` — download LiveCodeBench prompts.

Each function returns a :class:`PrepareResult` so the CLI can log
counts and exit codes without inspecting private state. The CLI
itself is a thin wrapper around these functions.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hone.errors import DataError, ValidationError
from hone.jsonl import Writer
from hone.normalize import Normalizer, SweNormalizer
from hone.prepare.hf import HubStream, load_split
from hone.prepare.mappers import as_codeforces_text, as_sft
from hone.prepare.reservoir import Reservoir
from hone.prepare.token_filter import TokenFilter, load_tokenizer

Mapper = Callable[[dict[str, Any]], dict[str, Any] | None]


@dataclass(frozen=True)
class PrepareResult:
    """Summary of a prepare run, suitable for logging and tests."""

    written: int = 0
    skipped: int = 0
    filtered_long: int = 0
    train_count: int | None = None
    valid_count: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PrepareRequest:
    """Common knobs every prepare entry point accepts."""

    output: Path
    seed: int = 42
    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger("hone.prepare"),
    )


# ---------------------------------------------------------------------------
# prepare_local_file
# ---------------------------------------------------------------------------


def prepare_local_file(
    *,
    input_path: Path,
    request: PrepareRequest,
    ratio: float,
    max_samples: int | None = None,
) -> PrepareResult:
    """Normalize and split a local JSONL file.

    Args:
        input_path: The source JSONL.
        request: Output path, seed, optional logger.
        ratio: Validation split ratio (exclusive 0..1).
        max_samples: Optional cap on examples read; defaults to all.
    """
    if max_samples is not None and max_samples < 2:
        raise DataError("--max-samples must be at least 2")
    if not 0 < ratio < 1:
        raise DataError(f"--ratio must be between 0 and 1, got {ratio}")

    logger = request.logger
    normalizer = Normalizer()
    examples: list = []
    with input_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise DataError(f"{input_path}:{line_number}: {error}") from error
            if not isinstance(record, dict):
                raise DataError(
                    f"{input_path}:{line_number}: each record must be an object"
                )
            try:
                examples.append(normalizer.normalize(record))
            except ValidationError as error:
                raise DataError(f"{input_path}:{line_number}: {error}") from error
    if max_samples is not None:
        examples = examples[:max_samples]

    train, valid = _splitter(ratio, request.seed).split(examples)
    train_count = Writer().write(request.output / "train.jsonl", train)
    valid_count = Writer().write(request.output / "valid.jsonl", valid)
    logger.info(
        "wrote %d train and %d validation examples to %s",
        train_count,
        valid_count,
        request.output,
    )
    return PrepareResult(
        written=train_count + valid_count,
        train_count=train_count,
        valid_count=valid_count,
    )


# ---------------------------------------------------------------------------
# prepare_reservoir_sample
# ---------------------------------------------------------------------------


def prepare_reservoir_sample(
    *,
    request: PrepareRequest,
    dataset: str = "teven/code_contests",
    split: str = "train",
    language: str = "PYTHON",
    max_samples: int = 20000,
    scan_limit: int = 250000,
    ratio: float = 0.02,
) -> PrepareResult:
    """Reservoir-sample competitive-programming rows and split them."""
    if max_samples < 2:
        raise DataError("--max-samples must be at least 2")
    if scan_limit < 1:
        raise DataError("--scan-limit must be positive")
    if not 0 < ratio < 1:
        raise DataError(f"--ratio must be between 0 and 1, got {ratio}")

    logger = request.logger
    reservoir = Reservoir[dict[str, Any]](max_samples, seed=request.seed)
    seen = 0
    for row in HubStream(dataset, split=split):
        language_value = row.get("language")
        if language_value and str(language_value).upper() != language.upper():
            continue
        question = str(row.get("description", row.get("question", ""))).strip()
        solution = str(row.get("solution", row.get("answer", ""))).strip()
        if len(question) < 80 or len(solution) < 20:
            continue
        record = _wrap_problem(question, solution)
        reservoir.observe(record)
        seen += 1
        if seen >= scan_limit:
            break
    if len(reservoir) < 2:
        raise DataError("fewer than two usable examples found")

    rows = reservoir.shuffle()
    valid_count = max(1, round(len(rows) * ratio))
    request.output.mkdir(parents=True, exist_ok=True)
    for name, values in (
        ("train.jsonl", rows[valid_count:]),
        ("valid.jsonl", rows[:valid_count]),
    ):
        with (request.output / name).open("w", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    train_count = len(rows) - valid_count
    logger.info(
        "selected %d of %d usable rows; wrote %d train and %d valid",
        len(reservoir),
        seen,
        train_count,
        valid_count,
    )
    return PrepareResult(
        written=len(rows),
        train_count=train_count,
        valid_count=valid_count,
        extras={"seen": seen},
    )


def _wrap_problem(question: str, solution: str) -> dict[str, Any]:
    """Wrap a (question, solution) pair into a chat-format record."""
    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Solve this competitive-programming problem in Python. "
                    "Return only the complete program.\n\n" + question
                ),
            },
            {"role": "assistant", "content": solution},
        ]
    }


# ---------------------------------------------------------------------------
# prepare_swe
# ---------------------------------------------------------------------------


def prepare_swe(
    *,
    request: PrepareRequest,
    dataset: str = "SWE-bench/SWE-bench",
    split: str = "train",
    ratio: float = 0.05,
    max_samples: int | None = None,
    max_chars: int = 14000,
) -> PrepareResult:
    """Build SWE-bench SFT rows. Refuses non-train splits."""
    if split != "train":
        raise DataError("refusing non-train split; evaluation patches would leak")
    if not 0 < ratio < 1:
        raise DataError(f"--ratio must be between 0 and 1, got {ratio}")
    if max_samples is not None and max_samples < 2:
        raise DataError("--max-samples must be at least 2")
    if max_chars < 1:
        raise DataError("--max-chars must be positive")

    logger = request.logger
    normalizer = SweNormalizer()
    converted: list = []
    for row in load_split(dataset, split=split):
        try:
            example = normalizer.normalize(row)
        except ValidationError:
            continue
        if example.character_count > max_chars:
            continue
        converted.append(example)
        if max_samples is not None and len(converted) >= max_samples:
            break
    if len(converted) < 2:
        raise DataError("not enough valid SWE examples")

    train, valid = _splitter(ratio, request.seed).split(converted)
    train_count = Writer().write(request.output / "train.jsonl", train)
    valid_count = Writer().write(request.output / "valid.jsonl", valid)
    logger.info("wrote %d train and %d validation rows", train_count, valid_count)
    return PrepareResult(
        written=train_count + valid_count,
        train_count=train_count,
        valid_count=valid_count,
    )


# ---------------------------------------------------------------------------
# prepare_stream
# ---------------------------------------------------------------------------


def prepare_stream(
    *,
    request: PrepareRequest,
    repo: str,
    configs: Iterable[str],
    split: str = "train",
    mode: str = "sft",
    max_tokens: int = 0,
    max_samples: int = 0,
    tokenizer_model: str = "openbmb/MiniCPM5-1B",
) -> PrepareResult:
    """Materialize every row of an HF config as MLX JSONL.

    Args:
        request: Output path, seed, optional logger.
        repo: HuggingFace repo ID.
        configs: Comma-separated list of HF configs to iterate.
        split: HF split to use.
        mode: ``"sft"`` for chat-format rows; ``"codeforces-text"`` for plain-text rows.
        max_tokens: Drop records longer than this many tokens; ``0`` disables.
        max_samples: Stop after this many rows are written; ``0`` means full pass.
        tokenizer_model: HF model id used for token-count filtering.
    """
    if mode not in {"sft", "codeforces-text"}:
        raise DataError(f"--mode must be 'sft' or 'codeforces-text', got {mode!r}")
    if max_tokens < 0:
        raise DataError(f"--max-tokens must be non-negative, got {max_tokens}")

    logger = request.logger
    mapper = as_sft if mode == "sft" else _safe_codeforces
    tokenizer = load_tokenizer(tokenizer_model) if max_tokens > 0 else None
    filter_ = TokenFilter(max_tokens=max_tokens, tokenizer=tokenizer)

    request.output.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = filtered_long = 0
    started = time.monotonic()
    last_log = started
    with request.output.open("w", encoding="utf-8") as handle:
        for config in _iter_configs(configs):
            stream = HubStream(repo, config=config, split=split)
            for row in stream:
                try:
                    record = mapper(row)
                except ValidationError:
                    record = None
                if record is None:
                    skipped += 1
                    continue
                if filter_.too_long(record):
                    filtered_long += 1
                    continue
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                if written % 100_000 == 0:
                    logger.info(
                        "written=%d skipped=%d filtered_long=%d",
                        written,
                        skipped,
                        filtered_long,
                    )
                if max_samples > 0 and written >= max_samples:
                    logger.info("max-samples reached (%d), stopping early", max_samples)
                    break
                if written % 5000 == 0 and time.monotonic() - last_log > 30.0:
                    elapsed = time.monotonic() - started
                    logger.info(
                        "heartbeat w=%d s=%d fl=%d t=%.0fs",
                        written,
                        skipped,
                        filtered_long,
                        elapsed,
                    )
                    last_log = time.monotonic()
            else:
                continue
            break
    elapsed = time.monotonic() - started
    logger.info(
        "complete: written=%d skipped=%d filtered_long=%d output=%s elapsed=%.1fs",
        written,
        skipped,
        filtered_long,
        request.output,
        elapsed,
    )
    return PrepareResult(
        written=written,
        skipped=skipped,
        filtered_long=filtered_long,
        extras={"elapsed_seconds": elapsed},
    )


def _safe_codeforces(row: dict[str, Any]) -> dict[str, Any] | None:
    """Wrap :func:`as_codeforces_text` and convert ``ValidationError`` to ``None``."""
    try:
        return as_codeforces_text(row)
    except ValidationError:
        return None


def _iter_configs(configs: Iterable[str] | str) -> Iterator[str]:
    """Accept either a comma-separated string or an iterable of configs."""
    if isinstance(configs, str):
        yield from (token.strip() for token in configs.split(",") if token.strip())
        return
    yield from configs


# ---------------------------------------------------------------------------
# prepare_eval_prompts
# ---------------------------------------------------------------------------


def prepare_eval_prompts(
    *,
    output: Path,
    version: str = "release_v2",
    logger: logging.Logger | None = None,
) -> PrepareResult:
    """Download LiveCodeBench prompts for evaluation only."""
    from datasets import load_dataset

    log = logger or logging.getLogger("hone.prepare")
    dataset = load_dataset("livecodebench/code_generation_lite", version_tag=version)
    split_data = dataset["test"] if isinstance(dataset, dict) else dataset
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in split_data:
            handle.write(
                json.dumps(
                    {
                        "question_id": row["question_id"],
                        "question_content": row["question_content"],
                        "contest_date": str(row.get("contest_date", "")),
                        "difficulty": row.get("difficulty", ""),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
    log.info("wrote %d evaluation prompts to %s", written, output)
    return PrepareResult(written=written)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _splitter(ratio: float, seed: int) -> Any:
    """In-memory splitter used by the local-file and SWE prepare paths."""
    from hone.split import Splitter

    return Splitter(ratio, seed)


__all__ = [
    "Mapper",
    "PrepareRequest",
    "PrepareResult",
    "prepare_eval_prompts",
    "prepare_local_file",
    "prepare_reservoir_sample",
    "prepare_stream",
    "prepare_swe",
]
