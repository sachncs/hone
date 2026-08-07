"""CLI subcommand group: data preparation."""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer

from hone.jsonl import Writer
from hone.log import setup
from hone.normalize import Normalizer
from hone.split import Splitter

app: typer.Typer = typer.Typer(help="Prepare datasets.", no_args_is_help=True)


@app.command("file")
def file(
    input: str = typer.Option(..., "--input", help="Path to input JSONL."),
    output: str = typer.Option(
        ..., "--output", help="Directory for train/valid JSONL files."
    ),
    ratio: float = typer.Option(
        0.05, "--ratio", help="Validation split ratio (exclusive 0..1)."
    ),
    seed: int = typer.Option(42, "--seed", help="Deterministic seed."),
    max_samples: int | None = typer.Option(
        None, "--max-samples", help="Cap on examples read."
    ),
) -> None:
    """Normalize and split a local JSONL file."""
    logger = setup(verbose=False)
    if max_samples is not None and max_samples < 2:
        raise typer.BadParameter("--max-samples must be at least 2")

    input_path = Path(input)
    output_path = Path(output)
    if not input_path.is_file():
        raise typer.BadParameter(f"input file not found: {input_path}")

    normalizer = Normalizer()
    examples = []
    with input_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise typer.BadParameter(
                    f"{input_path}:{line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise typer.BadParameter(
                    f"{input_path}:{line_number}: each record must be an object"
                )
            try:
                examples.append(normalizer.normalize(record))
            except ValueError as error:
                raise typer.BadParameter(
                    f"{input_path}:{line_number}: {error}"
                ) from error
    if max_samples is not None:
        examples = examples[:max_samples]

    train, valid = Splitter(ratio, seed).split(examples)
    writer = Writer()
    train_count = writer.write(output_path / "train.jsonl", train)
    valid_count = writer.write(output_path / "valid.jsonl", valid)
    logger.info(
        "wrote %d train and %d validation examples to %s",
        train_count,
        valid_count,
        output_path,
    )


@app.command("code")
def code(
    dataset: str = typer.Option(
        "teven/code_contests", "--dataset", help="HuggingFace dataset ID."
    ),
    split: str = typer.Option("train", "--split", help="HF dataset split."),
    output: str = typer.Option(
        "data/processed/code", "--output", help="Output directory."
    ),
    language: str = typer.Option(
        "PYTHON", "--language", help="Programming language filter."
    ),
    max_samples: int = typer.Option(20000, "--max-samples", help="Reservoir cap."),
    scan_limit: int = typer.Option(
        250000, "--scan-limit", help="Stop scanning after this many rows."
    ),
    ratio: float = typer.Option(0.02, "--ratio", help="Validation ratio."),
    seed: int = typer.Option(42, "--seed", help="Deterministic seed."),
) -> None:
    """Reservoir-sample competitive-programming rows and split."""
    from datasets import load_dataset

    if max_samples < 2:
        raise typer.BadParameter("--max-samples must be at least 2")
    if scan_limit < 1:
        raise typer.BadParameter("--scan-limit must be positive")
    if not 0 < ratio < 1:
        raise typer.BadParameter("--ratio must be between 0 and 1")

    logger = setup(verbose=False)
    stream = load_dataset(dataset, split=split, streaming=True)
    rng = __import__("random").Random(seed)
    reservoir: list[dict[str, object]] = []
    seen = 0
    for row in stream:
        language_value = row.get("language")
        if language_value and str(language_value).upper() != language.upper():
            continue
        question = str(row.get("description", row.get("question", ""))).strip()
        solution = str(row.get("solution", row.get("answer", ""))).strip()
        if len(question) < 80 or len(solution) < 20:
            continue
        record: dict[str, object] = {
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
        seen += 1
        if len(reservoir) < max_samples:
            reservoir.append(record)
        else:
            index = rng.randrange(seen)
            if index < max_samples:
                reservoir[index] = record
        if seen >= scan_limit:
            break
    if len(reservoir) < 2:
        raise typer.BadParameter("fewer than two usable examples found")
    rng.shuffle(reservoir)
    valid_count = max(1, round(len(reservoir) * ratio))
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    for name, values in (
        ("train.jsonl", reservoir[valid_count:]),
        ("valid.jsonl", reservoir[:valid_count]),
    ):
        with (output_path / name).open("w", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    logger.info(
        "selected %d of %d usable rows; wrote %d train and %d valid",
        len(reservoir),
        seen,
        len(reservoir) - valid_count,
        valid_count,
    )


@app.command("swe")
def swe(
    dataset: str = typer.Option(
        "SWE-bench/SWE-bench", "--dataset", help="HuggingFace dataset ID."
    ),
    split: str = typer.Option("train", "--split", help="HF dataset split."),
    output: str = typer.Option(
        "data/processed/swe", "--output", help="Output directory."
    ),
    ratio: float = typer.Option(0.05, "--ratio", help="Validation ratio."),
    max_samples: int | None = typer.Option(None, "--max-samples", help="Cap."),
    max_chars: int = typer.Option(
        14000, "--max-chars", help="Drop rows exceeding this prompt+patch length."
    ),
    seed: int = typer.Option(42, "--seed", help="Deterministic seed."),
) -> None:
    """Build SWE-bench SFT rows. Refuses non-train splits."""
    if split != "train":
        raise typer.BadParameter(
            "refusing non-train split; evaluation patches would leak"
        )
    if not 0 < ratio < 1:
        raise typer.BadParameter("--ratio must be between 0 and 1")
    if max_samples is not None and max_samples < 2:
        raise typer.BadParameter("--max-samples must be at least 2")
    if max_chars < 1:
        raise typer.BadParameter("--max-chars must be positive")

    from datasets import load_dataset

    from hone.normalize import SweNormalizer
    from hone.split import Splitter as Split

    logger = setup(verbose=False)
    rows = load_dataset(dataset, split=split)
    converted = []
    normalizer = SweNormalizer()
    for row in rows:
        try:
            example = normalizer.normalize(row)
        except ValueError:
            continue
        if example.character_count > max_chars:
            continue
        converted.append(example)
        if max_samples and len(converted) >= max_samples:
            break
    if len(converted) < 2:
        raise typer.BadParameter("not enough valid SWE examples")

    train, valid = Split(ratio, seed).split(converted)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    writer = Writer()
    train_count = writer.write(output_path / "train.jsonl", train)
    valid_count = writer.write(output_path / "valid.jsonl", valid)
    logger.info("wrote %d train and %d validation rows", train_count, valid_count)


@app.command("all")
def all_cmd(
    repo: str = typer.Option(..., "--repo", help="HuggingFace repo ID."),
    configs: str = typer.Option(..., "--configs", help="Comma-separated HF configs."),
    split: str = typer.Option("train", "--split"),
    output: str = typer.Option(..., "--output", help="Output JSONL path."),
    mode: str = typer.Option("sft", "--mode", help="sft or codeforces-text."),
    max_tokens: int = typer.Option(
        0,
        "--max-tokens",
        help="Drop records whose token count exceeds this value; 0 disables.",
    ),
    max_samples: int = typer.Option(
        0,
        "--max-samples",
        help="Stop after this many rows are written; 0 means full pass. "
        "Useful for capping multi-million-row HF streams to a tractable subset "
        "for short fine-tune runs.",
    ),
    tokenizer_model: str = typer.Option(
        "openbmb/MiniCPM5-1B",
        "--tokenizer-model",
        help="HF model id used for token-count filtering.",
    ),
) -> None:
    """Materialize every row of an HF config as MLX JSONL."""
    if mode not in {"sft", "codeforces-text"}:
        raise typer.BadParameter("--mode must be 'sft' or 'codeforces-text'")
    if max_tokens < 0:
        raise typer.BadParameter("--max-tokens must be non-negative")

    from datasets import load_dataset

    from hone.log import setup as setup_log

    logger = setup_log(verbose=False)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    filtered_long = 0
    tokenizer = None
    if max_tokens > 0:
        from mlx_lm.tokenizer_utils import AutoTokenizer

        logger.info("loading tokenizer %s for length filtering", tokenizer_model)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model, trust_remote_code=True
        )

    def as_sft(row: dict[str, object]) -> dict[str, object] | None:
        role_map = {
            "human": "user",
            "user": "user",
            "system": "system",
            "assistant": "assistant",
            "gpt": "assistant",
            "bot": "assistant",
        }
        raw_messages = row.get("messages")
        if isinstance(raw_messages, list):
            out = []
            for message in raw_messages:
                if not isinstance(message, dict):
                    return None
                role = role_map.get(str(message.get("role", "")).lower())
                content = message.get("content")
                if role is None or content is None:
                    return None
                content = str(content).strip()
                if content:
                    out.append({"role": role, "content": content})
            if len(out) >= 2 and out[-1]["role"] == "assistant":
                return {"messages": out}
            return None
        prompt = next(
            (
                row.get(k)
                for k in (
                    "prompt",
                    "input",
                    "instruction",
                    "question",
                    "problem",
                )
                if row.get(k)
            ),
            None,
        )
        answer = next(
            (
                row.get(k)
                for k in (
                    "completion",
                    "response",
                    "output",
                    "answer",
                    "solution",
                )
                if row.get(k)
            ),
            None,
        )
        if prompt is None or answer is None or isinstance(answer, (dict, list)):
            return None
        prompt_text = str(prompt).strip()
        answer_text = str(answer).strip()
        if not prompt_text or not answer_text:
            return None
        return {
            "messages": [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": answer_text},
            ]
        }

    def codeforces_text(row: dict[str, object]) -> dict[str, object]:
        fields = [
            ("TITLE", row.get("title")),
            ("DESCRIPTION", row.get("description")),
            ("INPUT FORMAT", row.get("input_format")),
            ("OUTPUT FORMAT", row.get("output_format")),
            ("EDITORIAL", row.get("editorial")),
        ]
        text = "\n\n".join(
            f"## {name}\n{value}".strip()
            for name, value in fields
            if value and str(value).strip()
        )
        if not text.strip():
            raise ValueError("row has no serializable problem text")
        return {"text": text}

    def token_count(record: dict[str, object]) -> int:
        if "messages" in record:
            messages = record["messages"]
            messages_list = messages if isinstance(messages, list) else []
            text = "\n".join(
                str(message["content"])
                for message in messages_list
                if isinstance(message, dict)
            )
        elif "text" in record:
            text = str(record["text"])
        else:
            return 0
        assert tokenizer is not None
        return len(tokenizer.encode(text, add_special_tokens=False))

    with output_path.open("w", encoding="utf-8") as handle:
        started = time.monotonic()
        last_log = started
        for config in configs.split(","):
            dataset = load_dataset(repo, name=config, split=split, streaming=True)
            for row in dataset:
                try:
                    record = (
                        codeforces_text(row)
                        if mode == "codeforces-text"
                        else as_sft(row)
                    )
                except ValueError:
                    record = None
                if record is None:
                    skipped += 1
                    continue
                if tokenizer is not None and token_count(record) > max_tokens:
                    filtered_long += 1
                    continue
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                if written % 100000 == 0:
                    logger.info(
                        "written=%d skipped=%d filtered_long=%d",
                        written,
                        skipped,
                        filtered_long,
                    )
                if max_samples > 0 and written >= max_samples:
                    logger.info(
                        "max-samples reached (%d), stopping early", max_samples
                    )
                    break
                if written % 5000 == 0 and time.monotonic() - last_log > 30.0:
                    logger.info(
                        "heartbeat: written=%d skipped=%d filtered_long=%d "
                        "elapsed=%.0fs",
                        written,
                        skipped,
                        filtered_long,
                        time.monotonic() - started,
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
        output_path,
        elapsed,
    )


@app.command("evaluate")
def evaluate(
    version: str = typer.Option(
        "release_v2", "--version", help="LiveCodeBench release tag."
    ),
    output: str = typer.Option(
        "data/eval/lcb.jsonl", "--output", help="Output JSONL path."
    ),
) -> None:
    """Download LiveCodeBench prompts for evaluation only."""
    from datasets import load_dataset

    from hone.log import setup as setup_log

    logger = setup_log(verbose=False)
    dataset = load_dataset("livecodebench/code_generation_lite", version_tag=version)
    split_data = dataset["test"] if isinstance(dataset, dict) else dataset
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
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
    logger.info("wrote %d evaluation prompts to %s", len(split_data), output_path)


__all__ = ["app"]
