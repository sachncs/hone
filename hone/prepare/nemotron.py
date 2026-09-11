"""Streaming + filtering for the two Nemotron SFT datasets.

Both datasets ship as ``messages`` JSONL with optional
``reasoning_content`` per message (chain-of-thought). The datasets
have a schema mismatch with the auto-generated HF metadata
(``messages`` is declared with 2 fields but the data has 3),
so :func:`datasets.load_dataset` raises ``CastError`` on the
competitive-programming split.

This module bypasses the dataset cast by reading the jsonl
files directly via :mod:`huggingface_hub.hf_hub_download`. It
streams rows one at a time (no full download needed for the
subset we use), filters + normalizes, and writes to a local
JSONL the Soup trainer can consume.

Public surface:

* :func:`stream_nemotron_competitive_programming` — Python-only
  rows from nvidia/Nemotron-SFT-Competitive-Programming-v2's
  ``competitive_programming_python_*`` jsonl files.
* :func:`stream_nemotron_swe` — rows from
  nvidia/Nemotron-SFT-SWE-v2's ``swe`` + ``agentless`` jsonl files.

Both yield :class:`dict` rows already in ``{messages: [...]}``
chat format; the Soup trainer's :func:`prepare_stream`
post-processor can drop them straight into ``train.jsonl``.
"""

from __future__ import annotations

import json
import logging
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class NemotronConfig:
    """Which Nemotron files to pull and how to filter them."""

    repo_id: str
    filenames: tuple[str, ...]
    """JSONL files inside the repo's ``data/`` directory."""
    name: str
    """Short label used in output filenames."""
    max_chars: int = 8000
    """Drop rows whose combined message length exceeds this."""


def _download(repo_id: str, filename: str, *, cache_dir: Path) -> Path:
    """Download a single jsonl file from a HF dataset repo."""
    from huggingface_hub import hf_hub_download

    log.info("downloading %s/%s", repo_id, filename)
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            cache_dir=str(cache_dir),
        )
    )


def _validate_messages(row: dict[str, object]) -> dict[str, object] | None:
    """Validate a Nemotron row; return a chat-format dict or ``None``.

    Soup's chat template doesn't have a slot for chain-of-thought,
    so the ``reasoning_content`` field is silently dropped. The
    returned record carries only ``messages`` (role + content per
    turn); no ``metadata`` is preserved.
    """
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None
    cleaned: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        content = message.get("content")
        if role is None or content is None:
            return None
        text = str(content).strip()
        if not text:
            return None
        cleaned.append({"role": str(role), "content": text})
    if not cleaned:
        return None
    return {"messages": cleaned}


def _stream_jsonl(path: Path) -> Iterator[dict[str, object]]:
    """Yield non-empty jsonl lines from ``path``."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def stream_nemotron(
    config: NemotronConfig,
    *,
    max_rows: int,
    seed: int = 42,
    cache_dir: Path | None = None,
) -> Iterator[dict[str, object]]:
    """Stream + filter + reservoir-sample up to ``max_rows`` from a Nemotron repo.

    Reads each file once, normalizes every row through
    :func:`_validate_messages`, drops rows whose combined
    message length exceeds ``config.max_chars``, and yields the
    survivor. Sampling is a deterministic uniform pass with
    random seed so two calls with the same seed produce the same
    subset (modulo file-order which HF doesn't guarantee).

    For 60K rows from a ~330K-row repo this reads ~5 GB and
    produces ~150 MB of jsonl. The caller writes the output to
    a local path.
    """
    import random

    cache = cache_dir or Path(tempfile.gettempdir()) / "hone-nemotron-cache"
    cache.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    # Reservoir sampling across an arbitrary number of rows: keep
    # every row until the cap, then replace with probability
    # ``cap / seen``. For 60K from ~330K the cap/seen ratio is small
    # so the subset stays diverse.
    pool: list[dict[str, object]] = []
    seen = 0
    for filename in config.filenames:
        try:
            path = _download(config.repo_id, filename, cache_dir=cache)
        except Exception as error:  # network or 404 — keep going
            log.warning("skipping %s/%s: %s", config.repo_id, filename, error)
            continue
        for raw in _stream_jsonl(path):
            cleaned = _validate_messages(raw)
            if cleaned is None:
                continue
            messages_obj: object = cleaned["messages"]
            if not isinstance(messages_obj, list):
                continue
            total_chars = sum(
                len(str(m["content"])) for m in messages_obj if isinstance(m, dict)
            )
            if total_chars > config.max_chars:
                continue
            seen += 1
            if len(pool) < max_rows:
                pool.append(cleaned)
            else:
                index = rng.randrange(seen)
                if index < max_rows:
                    pool[index] = cleaned
    rng.shuffle(pool)
    log.info(
        "%s: scanned=%d kept=%d (cap=%d)",
        config.name,
        seen,
        len(pool),
        max_rows,
    )
    yield from pool[:max_rows]


def materialize(
    configs_and_caps: list[tuple[NemotronConfig, int]],
    output_dir: Path,
    *,
    seed: int = 42,
) -> dict[str, int]:
    """Stream + write Nemotron subsets to ``output_dir``.

    Each (config, cap) becomes two files:
    ``<config.name>-train.jsonl`` (cap rows) and
    ``<config.name>-valid.jsonl`` (cap // 20 rows, min 1).
    Returns a dict of file -> row count for logging.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    totals: dict[str, int] = {}
    for config, cap in configs_and_caps:
        rows = list(stream_nemotron(config, max_rows=cap, seed=seed))
        train_rows = rows[: max(1, cap - max(1, cap // 20))]
        valid_rows = rows[max(1, cap - max(1, cap // 20)) :]
        train_path = output_dir / f"{config.name}-train.jsonl"
        valid_path = output_dir / f"{config.name}-valid.jsonl"
        for path, items in (
            (train_path, train_rows),
            (valid_path, valid_rows),
        ):
            with path.open("w", encoding="utf-8") as handle:
                for row in items:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            totals[str(path)] = len(items)
    return totals


# Canonical Nemotron configs reused across the repo.
NEMOTRON_COMPETITIVE_PROGRAMMING = NemotronConfig(
    repo_id="nvidia/Nemotron-SFT-Competitive-Programming-v2",
    filenames=(
        "data/competitive_programming_python_00.jsonl",
        "data/competitive_programming_python_01.jsonl",
    ),
    name="nemotron-cp",
    max_chars=8000,
)


NEMOTRON_SWE = NemotronConfig(
    repo_id="nvidia/Nemotron-SFT-SWE-v2",
    filenames=(
        "data/swe.jsonl",
        "data/agentless.jsonl",
    ),
    name="nemotron-swe",
    max_chars=12000,
)


__all__ = [
    "NEMOTRON_COMPETITIVE_PROGRAMMING",
    "NEMOTRON_SWE",
    "NemotronConfig",
    "materialize",
    "stream_nemotron",
]
