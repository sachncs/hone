"""Soup-ready JSONL preparation service.

Each function takes a :class:`PrepareRequest` (output path, seed,
logger) plus its own knobs and returns a :class:`PrepareResult`
(written / skipped / filtered_long counts). The functions do not
depend on any CLI, MLX, or fine-tuning layer; they only emit JSONL
files that Soup's trainer reads directly.

All errors are subclasses of :class:`PrepareError` so the caller
can catch one exception type.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from hone.prepare.hf import HubStream, load_split
from hone.prepare.mappers import as_codeforces_text, as_sft
from hone.prepare.reservoir import Reservoir
from hone.prepare.token_filter import TokenFilter, load_tokenizer


class PrepareError(Exception):
    """Base error for the prepare service.

    All other prepare-layer errors inherit from this so callers
    can catch a single exception type at the application boundary.
    """


class ValidationError(PrepareError):
    """A record failed validation; ``location`` describes where."""


class DataError(PrepareError):
    """A dataset path is missing, malformed, or empty."""


class Role(StrEnum):
    """Chat roles accepted by every chat-format record we emit."""

    system = "system"
    user = "user"
    assistant = "assistant"


_ROLE_ALIASES: dict[str, Role] = {
    "human": Role.user,
    "user": Role.user,
    "system": Role.system,
    "assistant": Role.assistant,
    "gpt": Role.assistant,
    "bot": Role.assistant,
}


@dataclass(frozen=True)
class PrepareResult:
    """Summary of one prepare run; safe to log directly."""

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
# JSONL writing + chat validation
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Write JSONL records with sorted keys; return count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            written += 1
    return written


def _parse_chat(raw_messages: object, *, location: str) -> list[dict[str, str]]:
    """Validate a ``messages`` list into the canonical chat schema.

    Empty content, missing fields, and unknown roles all raise
    :class:`ValidationError` with a positional ``location``.
    """
    if not isinstance(raw_messages, list):
        raise ValidationError(
            f"{location}: expected a list, got {type(raw_messages).__name__}"
        )
    out: list[dict[str, str]] = []
    for index, raw_message in enumerate(raw_messages):
        if not isinstance(raw_message, dict):
            raise ValidationError(
                f"{location}[{index}]: expected an object, "
                f"got {type(raw_message).__name__}"
            )
        raw_role = raw_message.get("role")
        content = raw_message.get("content")
        if raw_role is None:
            raise ValidationError(f"{location}[{index}]: missing 'role'")
        if content is None:
            raise ValidationError(f"{location}[{index}]: missing 'content'")
        role = _ROLE_ALIASES.get(str(raw_role).lower())
        if role is None:
            raise ValidationError(
                f"{location}[{index}].role: unknown role {raw_role!r}"
            )
        text = str(content).strip()
        if not text:
            raise ValidationError(f"{location}[{index}].content is empty")
        out.append({"role": str(role), "content": text})
    return out


def _normalize_chat_record(
    record: dict[str, object], *, location: str
) -> dict[str, object]:
    """Normalize one record to chat format or raise :class:`ValidationError`."""
    raw_messages = record.get("messages")
    if isinstance(raw_messages, list):
        return {"messages": _parse_chat(raw_messages, location=location)}
    if "prompt" in record and "completion" in record:
        prompt_text = str(record["prompt"]).strip()
        completion_text = str(record["completion"]).strip()
        if not prompt_text or not completion_text:
            raise ValidationError(
                f"{location}: prompt or completion is empty after stripping"
            )
        return {
            "messages": [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": completion_text},
            ]
        }
    raise ValidationError(
        f"{location}: expected 'messages' list or 'prompt'/'completion' pair"
    )


def _split_records(
    records: list[dict[str, object]], ratio: float, seed: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Deterministic shuffled train/valid split; ratio must be in (0, 1)."""
    if not 0 < ratio < 1:
        raise DataError(f"--ratio must be between 0 and 1, got {ratio}")
    if len(records) < 2:
        raise DataError("at least two records are required")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    valid_count = max(1, round(len(shuffled) * ratio))
    return shuffled[valid_count:], shuffled[:valid_count]


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

    ``max_samples`` caps how many records are read (after parsing,
    before splitting); default is "read everything".
    """
    if max_samples is not None and max_samples < 2:
        raise DataError("--max-samples must be at least 2")
    records: list[dict[str, object]] = []
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
                records.append(
                    _normalize_chat_record(
                        record, location=f"{input_path}:{line_number}"
                    )
                )
            except ValidationError as error:
                raise DataError(f"{input_path}:{line_number}: {error}") from error
    if max_samples is not None:
        records = records[:max_samples]

    train, valid = _split_records(records, ratio, request.seed)
    train_count = _write_jsonl(request.output / "train.jsonl", train)
    valid_count = _write_jsonl(request.output / "valid.jsonl", valid)
    request.logger.info(
        "wrote %d train and %d validation records to %s",
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
    """Reservoir-sample competitive-programming rows and split them.

    Scans up to ``scan_limit`` HF rows, keeps a uniform random sample
    of ``max_samples`` rows in the language of choice, then splits
    1 - ratio / ratio between train and valid.
    """
    if max_samples < 2:
        raise DataError("--max-samples must be at least 2")
    if scan_limit < 1:
        raise DataError("--scan-limit must be positive")
    if not 0 < ratio < 1:
        raise DataError(f"--ratio must be between 0 and 1, got {ratio}")

    reservoir: Reservoir[dict[str, object]] = Reservoir(max_samples, seed=request.seed)
    seen = 0
    for row in HubStream(dataset, split=split):
        language_value = row.get("language")
        if language_value and str(language_value).upper() != language.upper():
            continue
        question = str(row.get("description", row.get("question", ""))).strip()
        solution = str(row.get("solution", row.get("answer", ""))).strip()
        if len(question) < 80 or len(solution) < 20:
            continue
        reservoir.observe(
            {
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
        )
        seen += 1
        if seen >= scan_limit:
            break
    if len(reservoir) < 2:
        raise DataError("fewer than two usable records found")

    rows = reservoir.shuffle()
    valid_count = max(1, round(len(rows) * ratio))
    train_count = len(rows) - valid_count
    request.output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(request.output / "train.jsonl", rows[valid_count:])
    _write_jsonl(request.output / "valid.jsonl", rows[:valid_count])
    request.logger.info(
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
    """Build SWE-bench SFT rows; refuses non-train splits.

    Each row becomes ``{"messages": [user, assistant]}`` where the
    user message names repo + version + issue, and the assistant
    message is the unified-diff patch.
    """
    if split != "train":
        raise DataError("refusing non-train split; evaluation patches would leak")
    if not 0 < ratio < 1:
        raise DataError(f"--ratio must be between 0 and 1, got {ratio}")
    if max_samples is not None and max_samples < 2:
        raise DataError("--max-samples must be at least 2")
    if max_chars < 1:
        raise DataError("--max-chars must be positive")

    converted: list[dict[str, object]] = []
    for row in load_split(dataset, split=split):
        try:
            record = _normalize_swe_row(row)
        except ValidationError:
            continue
        messages = record["messages"]
        if not isinstance(messages, list):
            continue
        if sum(len(m["content"]) for m in messages if isinstance(m, dict)) > max_chars:
            continue
        converted.append(record)
        if max_samples is not None and len(converted) >= max_samples:
            break
    if len(converted) < 2:
        raise DataError("not enough valid SWE rows")

    train, valid = _split_records(converted, ratio, request.seed)
    train_count = _write_jsonl(request.output / "train.jsonl", train)
    valid_count = _write_jsonl(request.output / "valid.jsonl", valid)
    request.logger.info(
        "wrote %d train and %d validation SWE rows", train_count, valid_count
    )
    return PrepareResult(
        written=train_count + valid_count,
        train_count=train_count,
        valid_count=valid_count,
    )


def _normalize_swe_row(row: dict[str, object]) -> dict[str, object]:
    """Convert one SWE-bench row into a prompt + patch pair."""
    statement = str(row.get("problem_statement", "")).strip()
    if not statement:
        raise ValidationError("missing problem_statement")
    patch = str(row.get("patch", "")).strip()
    if not patch:
        raise ValidationError("missing patch")
    prompt = (
        "You are repairing a real software repository. Return only a unified "
        "diff patch; do not explain the answer.\n\n"
        f"Repository: {str(row.get('repo', '')).strip()}\n"
        f"Version: {str(row.get('version', '')).strip()}\n\n"
        f"Issue:\n{statement}"
    )
    metadata: dict[str, object] = {}
    instance_id = row.get("instance_id")
    if instance_id is not None:
        metadata["instance_id"] = str(instance_id)
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": patch},
        ],
        **metadata,
    }


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
    """Materialize every row of an HF config as chat-format JSONL.

    ``mode='sft'`` produces ``{"messages": [...]}`` records; set
    ``mode='codeforces-text'`` to emit ``{"text": ...}`` for plain-text
    pre-training instead.
    """
    if mode not in {"sft", "codeforces-text"}:
        raise DataError(f"--mode must be 'sft' or 'codeforces-text', got {mode!r}")
    if max_tokens < 0:
        raise DataError(f"--max-tokens must be non-negative, got {max_tokens}")

    request.logger.info("streaming repo=%s split=%s mode=%s", repo, split, mode)
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
                    request.logger.info(
                        "written=%d skipped=%d filtered_long=%d",
                        written,
                        skipped,
                        filtered_long,
                    )
                if max_samples > 0 and written >= max_samples:
                    request.logger.info(
                        "max-samples reached (%d), stopping early", max_samples
                    )
                    break
                if written % 5000 == 0 and time.monotonic() - last_log > 30.0:
                    elapsed = time.monotonic() - started
                    request.logger.info(
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
    request.logger.info(
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


def _safe_codeforces(row: dict[str, object]) -> dict[str, object] | None:
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
# prepare_lingsard
# ---------------------------------------------------------------------------


def prepare_lingsard(
    *,
    request: PrepareRequest,
    dataset: str = "inclusionAI/Ling-Coder-SFT",
    split: str = "train",
    max_samples: int = 30_000,
    max_chars: int = 8_000,
    ratio: float = 0.05,
    language_filter: str = "Python",
) -> PrepareResult:
    """Stream + filter + split the inclusionAI/Ling-Coder-SFT dataset.

    Distinct from :func:`prepare_stream`: Ling-Coder rows are
    already in ``{"messages": [...]}`` chat format, so we skip
    row mapping. We also filter by ``languages`` field (most
    rows are Python-only; a minority are multi-language) and
    drop rows whose combined message length exceeds ``max_chars``
    so the trainer never sees a long-tail record.
    """
    from hone.prepare.hf import HubStream

    if max_samples < 2:
        raise DataError("--max-samples must be at least 2")
    if max_chars < 1:
        raise DataError("--max-chars must be positive")
    if not 0 < ratio < 1:
        raise DataError(f"--ratio must be between 0 and 1, got {ratio}")

    request.logger.info(
        "streaming dataset=%s split=%s language=%s", dataset, split, language_filter
    )

    converted: list[dict[str, object]] = []
    seen = 0
    for row in HubStream(dataset, split=split):
        seen += 1
        languages = row.get("languages") or []
        if (
            isinstance(languages, list)
            and language_filter
            and language_filter not in languages
        ):
            continue
        raw_messages = row.get("messages")
        if not isinstance(raw_messages, list):
            continue
        try:
            messages = _parse_chat(raw_messages, location=f"{dataset}:{seen}")
        except ValidationError:
            continue
        if sum(len(m["content"]) for m in messages) > max_chars:
            continue
        record: dict[str, object] = {"messages": messages}
        converted.append(record)
        if len(converted) >= max_samples:
            break

    if len(converted) < 2:
        raise DataError(
            f"not enough usable Ling-Coder rows ({len(converted)}); "
            f"raise --max-tokens / --scan-limit or relax --max-chars"
        )

    train, valid = _split_records(converted, ratio, request.seed)
    train_count = _write_jsonl(request.output / "train.jsonl", train)
    valid_count = _write_jsonl(request.output / "valid.jsonl", valid)
    request.logger.info(
        "scanned %d rows; wrote %d train and %d valid Ling-Coder rows",
        seen,
        train_count,
        valid_count,
    )
    return PrepareResult(
        written=train_count + valid_count,
        train_count=train_count,
        valid_count=valid_count,
        extras={"scanned": seen},
    )


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


__all__ = [
    "DataError",
    "PrepareError",
    "PrepareRequest",
    "PrepareResult",
    "Role",
    "ValidationError",
    "prepare_eval_prompts",
    "prepare_lingsard",
    "prepare_local_file",
    "prepare_reservoir_sample",
    "prepare_stream",
    "prepare_swe",
]
