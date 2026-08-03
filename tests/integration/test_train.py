"""Integration tests for hone.cli.train: dispatch verification.

These tests assert that the CLI correctly builds the subprocess
invocation, sets HONE_DEVICE in the environment, and surfaces the
subprocess exit code. The actual training step is mocked so the
tests do not depend on the mlx_lm trainer behavior on CPU devices.

Full training runs are exercised in tests/mlx/test_run.py on
Apple Silicon with a real model.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from hone.cli import app
from hone.cli import train as train_module

pytestmark = pytest.mark.mlx

runner = CliRunner()


def test_train_code_refuses_nonexistent_config(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "train",
            "code",
            "--config",
            str(tmp_path / "missing.yaml"),
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code != 0
    assert "config not found" in result.output or "config not found" in (
        result.stderr or ""
    )


def test_train_swe_refuses_nonexistent_config(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "train",
            "swe",
            "--config",
            str(tmp_path / "missing.yaml"),
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code != 0
    assert "config not found" in result.output or "config not found" in (
        result.stderr or ""
    )


def test_train_code_refuses_unknown_backend(tmp_path: Path) -> None:
    config = tmp_path / "code.yaml"
    config.write_text("model: m\ntrain: true\ndata: d\n", encoding="utf-8")
    result = runner.invoke(
        app,
        ["train", "code", "--config", str(config), "--backend", "tpu"],
    )
    assert result.exit_code != 0
    assert "backend" in result.output.lower()


captured_subprocess: dict[str, object] = {}


def capture_subprocess_call(
    *args: Any, **kwargs: Any
) -> subprocess.CompletedProcess[str]:
    """Module-level fake for subprocess.run that records args and env."""
    command: Sequence[str] = kwargs.get("args", args[0] if args else [])
    env: dict[str, str] = kwargs.get("env", {})
    captured_subprocess["command"] = command
    captured_subprocess["env"] = env
    return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


def test_train_code_sets_hone_device_and_invokes_subprocess(
    tmp_path: Path,
) -> None:
    config = tmp_path / "code.yaml"
    config.write_text("model: m\ntrain: true\ndata: d\n", encoding="utf-8")
    captured_subprocess.clear()
    with patch.object(
        train_module.subprocess, "run", side_effect=capture_subprocess_call
    ):
        result = runner.invoke(
            app,
            ["train", "code", "--config", str(config), "--device", "cpu"],
        )
    assert result.exit_code == 0
    command = captured_subprocess["command"]
    env = captured_subprocess["env"]
    assert isinstance(command, list)
    assert command[0] == sys.executable
    assert command[1:3] == ["-m", "hone.run"]
    assert "--config" in command
    assert str(config) in command
    assert isinstance(env, dict)
    assert env.get("HONE_DEVICE") == "cpu"


def test_train_swe_sets_hone_device_and_invokes_subprocess(
    tmp_path: Path,
) -> None:
    config = tmp_path / "swe.yaml"
    config.write_text("model: m\ntrain: true\ndata: d\n", encoding="utf-8")
    captured_subprocess.clear()
    with patch.object(
        train_module.subprocess, "run", side_effect=capture_subprocess_call
    ):
        result = runner.invoke(
            app,
            ["train", "swe", "--config", str(config), "--device", "gpu"],
        )
    assert result.exit_code == 0
    env = captured_subprocess["env"]
    assert isinstance(env, dict)
    assert env.get("HONE_DEVICE") == "gpu"


def test_full_sequence_has_expected_datasets() -> None:
    """The full sequence must include all 5 documented datasets in order."""
    from hone.cli.train import FULL_SEQUENCE

    repos = [entry[0] for entry in FULL_SEQUENCE]
    assert repos == [
        "ianncity/KIMI-K2.5-1000000x",
        "Modotte/CodeX-7M-Non-Thinking",
        "inclusionAI/Ling-Coder-SFT",
        "open-r1/codeforces",
        "microsoft/rStar-Coder",
    ]
    adapters = [entry[3] for entry in FULL_SEQUENCE]
    assert adapters == [
        Path("artifacts/full/01-kimi"),
        Path("artifacts/full/02-codex"),
        Path("artifacts/full/03-ling"),
        Path("artifacts/full/04-codeforces"),
        Path("artifacts/full/05-rstar"),
    ]


def test_full_sequence_adapter_directories_are_unique() -> None:
    """Each stage must write to a distinct adapter directory."""
    from hone.cli.train import FULL_SEQUENCE

    adapters = [entry[3] for entry in FULL_SEQUENCE]
    assert len(set(adapters)) == len(adapters)


def test_train_all_help_runs() -> None:
    """The all subcommand must be registered and parseable."""
    result = runner.invoke(app, ["train", "all", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.stdout
