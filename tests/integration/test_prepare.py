"""Integration tests for hone.cli.prepare using typer.testing.CliRunner."""

from __future__ import annotations

import importlib.machinery
import json
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import patch

from typer.testing import CliRunner

from hone.cli import app

runner = CliRunner()


def jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def fake_datasets(value: Any) -> Any:
    """Return a synthetic `datasets` module whose `load_dataset` returns `value`.

    Each call to `load_dataset` returns a fresh iterator so multiple
    invocations of the same prepare subcommand see the same rows.
    A list value is iterated fresh per call; a dict value (used by
    `hone prepare evaluate`, which indexes the result by split name)
    is returned as-is.
    """
    rows: list[Any] = list(value) if not isinstance(value, list) else value
    module: Any = types.ModuleType("datasets")
    module.__spec__ = importlib.machinery.ModuleSpec("datasets", loader=None)

    def load(*args: Any, **kwargs: Any) -> Any:
        return value if isinstance(value, dict) else iter(rows)

    module.load_dataset = load
    sys.modules["datasets"] = module
    return module


def restore_datasets() -> None:
    sys.modules.pop("datasets", None)


def chat_row(q: str = "Q", a: str = "A", row_id: int = 0) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": f"{q}-{row_id}"},
            {"role": "assistant", "content": f"{a}-{row_id}"},
        ],
        "row_id": row_id,
    }


def test_prepare_file_creates_train_and_valid(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.jsonl"
    output_dir = tmp_path / "out"
    rows = [chat_row(row_id=i) for i in range(40)]
    jsonl(input_path, rows)

    result = runner.invoke(
        app,
        ["prepare", "file", "--input", str(input_path), "--output", str(output_dir)],
    )
    assert result.exit_code == 0
    assert (output_dir / "train.jsonl").is_file()
    assert (output_dir / "valid.jsonl").is_file()
    train_rows = [
        json.loads(line)
        for line in (output_dir / "train.jsonl").read_text().splitlines()
        if line.strip()
    ]
    valid_rows = [
        json.loads(line)
        for line in (output_dir / "valid.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(train_rows) + len(valid_rows) == 40


def test_prepare_file_respects_max_samples(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.jsonl"
    output_dir = tmp_path / "out"
    jsonl(input_path, [chat_row(row_id=i) for i in range(50)])

    result = runner.invoke(
        app,
        [
            "prepare",
            "file",
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            "--max-samples",
            "10",
        ],
    )
    assert result.exit_code == 0
    train = (output_dir / "train.jsonl").read_text().splitlines()
    valid = (output_dir / "valid.jsonl").read_text().splitlines()
    assert len(train) + len(valid) == 10


def test_prepare_file_respects_seed(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.jsonl"
    jsonl(input_path, [chat_row(row_id=i) for i in range(30)])

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    runner.invoke(
        app,
        [
            "prepare",
            "file",
            "--input",
            str(input_path),
            "--output",
            str(out_a),
            "--seed",
            "42",
        ],
    )
    runner.invoke(
        app,
        [
            "prepare",
            "file",
            "--input",
            str(input_path),
            "--output",
            str(out_b),
            "--seed",
            "42",
        ],
    )
    assert (out_a / "train.jsonl").read_text() == (out_b / "train.jsonl").read_text()
    assert (out_a / "valid.jsonl").read_text() == (out_b / "valid.jsonl").read_text()


def test_prepare_file_deterministic_with_same_seed(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.jsonl"
    jsonl(input_path, [chat_row(row_id=i) for i in range(30)])

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    runner.invoke(
        app,
        [
            "prepare",
            "file",
            "--input",
            str(input_path),
            "--output",
            str(out_a),
            "--seed",
            "1",
        ],
    )
    runner.invoke(
        app,
        [
            "prepare",
            "file",
            "--input",
            str(input_path),
            "--output",
            str(out_b),
            "--seed",
            "2",
        ],
    )
    assert (out_a / "train.jsonl").read_text() != (out_b / "train.jsonl").read_text()


def test_prepare_swe_refuses_test_split() -> None:
    result = runner.invoke(
        app, ["prepare", "swe", "--dataset", "SWE-bench/SWE-bench", "--split", "test"]
    )
    assert result.exit_code != 0
    assert (
        "evaluation patches would leak" in result.output
        or "evaluation patches would leak" in (result.stderr or "")
    )


def test_prepare_swe_filters_by_max_chars(tmp_path: Path) -> None:
    rows = [
        {"instance_id": "i1", "problem_statement": "p", "patch": "x"},
    ]
    fake_datasets(value=iter(rows))
    try:
        result = runner.invoke(
            app,
            [
                "prepare",
                "swe",
                "--dataset",
                "SWE-bench/SWE-bench",
                "--max-chars",
                "1",
                "--output",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code != 0
        assert (
            "not enough valid SWE examples" in result.output
            or "not enough valid SWE examples" in (result.stderr or "")
        )
    finally:
        restore_datasets()


def test_prepare_code_reservoir_sampling_is_deterministic(tmp_path: Path) -> None:
    fake_row = {
        "language": "PYTHON",
        "description": "x" * 200,
        "solution": "def f(): pass" + "  x" * 30,
    }
    fake_datasets(value=iter([fake_row] * 100))
    try:
        result_a = runner.invoke(
            app,
            [
                "prepare",
                "code",
                "--dataset",
                "hf/dummy",
                "--output",
                str(tmp_path / "a"),
                "--max-samples",
                "10",
                "--scan-limit",
                "100",
                "--ratio",
                "0.2",
                "--seed",
                "42",
            ],
        )
        result_b = runner.invoke(
            app,
            [
                "prepare",
                "code",
                "--dataset",
                "hf/dummy",
                "--output",
                str(tmp_path / "b"),
                "--max-samples",
                "10",
                "--scan-limit",
                "100",
                "--ratio",
                "0.2",
                "--seed",
                "42",
            ],
        )
        assert result_a.exit_code == 0
        assert result_b.exit_code == 0
        assert (tmp_path / "a" / "train.jsonl").read_text() == (
            tmp_path / "b" / "train.jsonl"
        ).read_text()
    finally:
        restore_datasets()


def test_prepare_code_filters_by_language(tmp_path: Path) -> None:
    rows = [
        {
            "language": "PYTHON",
            "description": "x" * 200,
            "solution": "def f(): pass" + "  y" * 30,
        },
        {
            "language": "CPP",
            "description": "x" * 200,
            "solution": "int main() {}" + "  y" * 30,
        },
        {
            "language": "PYTHON",
            "description": "x" * 200,
            "solution": "def g(): pass" + "  y" * 30,
        },
    ]
    fake_datasets(value=iter(rows))
    try:
        result = runner.invoke(
            app,
            [
                "prepare",
                "code",
                "--dataset",
                "hf/dummy",
                "--output",
                str(tmp_path / "out"),
                "--max-samples",
                "10",
                "--scan-limit",
                "100",
                "--ratio",
                "0.5",
                "--seed",
                "42",
            ],
        )
        assert result.exit_code == 0
        written = (tmp_path / "out" / "train.jsonl").read_text().splitlines() + (
            tmp_path / "out" / "valid.jsonl"
        ).read_text().splitlines()
        assert len(written) == 2
    finally:
        restore_datasets()


def test_prepare_all_writes_valid_messages(tmp_path: Path) -> None:
    rows = [
        {
            "messages": [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
            ]
        },
    ]
    fake_datasets(value=iter(rows))
    try:
        result = runner.invoke(
            app,
            [
                "prepare",
                "all",
                "--repo",
                "hf/dummy",
                "--configs",
                "default",
                "--output",
                str(tmp_path / "out.jsonl"),
            ],
        )
        assert result.exit_code == 0
        written = [
            json.loads(line)
            for line in (tmp_path / "out.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert written == rows
    finally:
        restore_datasets()


def test_prepare_all_drops_records_with_empty_prompt_or_completion(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "prompt": "real question",
            "completion": "real answer",
        },
        {
            "prompt": "   ",
            "completion": "answer only",
        },
        {
            "prompt": "prompt only",
            "completion": "",
        },
        {
            "messages": [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": "ok"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "ok"},
                {"role": "assistant", "content": "  \t  "},
            ]
        },
    ]
    fake_datasets(value=iter(rows))
    try:
        result = runner.invoke(
            app,
            [
                "prepare",
                "all",
                "--repo",
                "hf/dummy",
                "--configs",
                "default",
                "--output",
                str(tmp_path / "out.jsonl"),
            ],
        )
        assert result.exit_code == 0
        written = [
            json.loads(line)
            for line in (tmp_path / "out.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert written == [
            {
                "messages": [
                    {"role": "user", "content": "real question"},
                    {"role": "assistant", "content": "real answer"},
                ]
            }
        ]
        assert "skipped=4" in result.output
    finally:
        restore_datasets()


def test_prepare_all_filters_records_over_max_tokens(tmp_path: Path) -> None:
    rows = [
        {
            "messages": [
                {"role": "user", "content": "short"},
                {"role": "assistant", "content": "ok"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "x" * 200},
                {"role": "assistant", "content": "y" * 200},
            ]
        },
    ]

    class FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            return list(range(len(text)))

    fake_datasets(value=iter(rows))
    try:
        with patch(
            "mlx_lm.tokenizer_utils.AutoTokenizer.from_pretrained",
            return_value=FakeTokenizer(),
        ):
            result = runner.invoke(
                app,
                [
                    "prepare",
                    "all",
                    "--repo",
                    "hf/dummy",
                    "--configs",
                    "default",
                    "--output",
                    str(tmp_path / "out.jsonl"),
                    "--max-tokens",
                    "100",
                    "--tokenizer-model",
                    "fake/model",
                ],
            )
        assert result.exit_code == 0
        written = [
            json.loads(line)
            for line in (tmp_path / "out.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert written == [rows[0]]
        assert "filtered_long=1" in result.output
    finally:
        restore_datasets()


def test_prepare_all_rejects_negative_max_tokens(tmp_path: Path) -> None:
    fake_datasets(value=iter([]))
    try:
        result = runner.invoke(
            app,
            [
                "prepare",
                "all",
                "--repo",
                "hf/dummy",
                "--configs",
                "default",
                "--output",
                str(tmp_path / "out.jsonl"),
                "--max-tokens",
                "-5",
            ],
        )
        assert result.exit_code != 0
        assert "max-tokens" in result.output or "max-tokens" in (result.stderr or "")
    finally:
        restore_datasets()


def test_prepare_evaluate_writes_lcb_prompts(tmp_path: Path) -> None:
    fake_split = [
        {
            "question_id": "q1",
            "question_content": "solve x",
            "contest_date": "2024",
            "difficulty": "easy",
        },
        {
            "question_id": "q2",
            "question_content": "solve y",
            "contest_date": "2024",
            "difficulty": "hard",
        },
    ]
    fake_dataset = {"test": fake_split}
    fake_datasets(value=fake_dataset)
    try:
        result = runner.invoke(
            app,
            ["prepare", "evaluate", "--output", str(tmp_path / "lcb.jsonl")],
        )
        assert result.exit_code == 0
        written = [
            json.loads(line)
            for line in (tmp_path / "lcb.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert written[0]["question_id"] == "q1"
        assert written[1]["question_id"] == "q2"
    finally:
        restore_datasets()
