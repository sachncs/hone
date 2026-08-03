"""Integration tests for hone.cli.train: dispatch verification.

These tests assert that the CLI correctly builds the subprocess
invocation, sets HONE_DEVICE in the environment, and surfaces the
subprocess exit code. The actual training step is mocked so the
tests do not depend on the mlx_lm trainer behavior on CPU devices.

Full training runs are exercised in tests/mlx/test_run.py on
Apple Silicon with a real model.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from hone.cli import app, train as train_module

pytestmark = pytest.mark.mlx

runner = CliRunner()


def test_train_code_refuses_nonexistent_config(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["train", "code", "--config", str(tmp_path / "missing.yaml"), "--device", "cpu"],
    )
    assert result.exit_code != 0
    assert "config not found" in result.output or "config not found" in (result.stderr or "")


def test_train_swe_refuses_nonexistent_config(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["train", "swe", "--config", str(tmp_path / "missing.yaml"), "--device", "cpu"],
    )
    assert result.exit_code != 0
    assert "config not found" in result.output or "config not found" in (result.stderr or "")


def test_train_code_refuses_unknown_backend(tmp_path: Path) -> None:
    config = tmp_path / "code.yaml"
    config.write_text("model: m\ntrain: true\ndata: d\n", encoding="utf-8")
    result = runner.invoke(
        app,
        ["train", "code", "--config", str(config), "--backend", "tpu"],
    )
    assert result.exit_code != 0
    assert "backend" in result.output.lower()


def test_train_code_sets_hone_device_and_invokes_subprocess(
    tmp_path: Path,
) -> None:
    config = tmp_path / "code.yaml"
    config.write_text("model: m\ntrain: true\ndata: d\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["command"] = kwargs.get("args", args[0] if args else None)
        captured["env"] = kwargs.get("env", {})
        return subprocess.CompletedProcess(captured["command"], 0, stdout="", stderr="")

    with patch.object(train_module.subprocess, "run", side_effect=fake_run):
        result = runner.invoke(
            app,
            ["train", "code", "--config", str(config), "--device", "cpu"],
        )
    assert result.exit_code == 0
    command = captured["command"]
    assert isinstance(command, list)
    assert command[0] == sys.executable
    assert command[1:3] == ["-m", "hone.run"]
    assert "--config" in command
    assert str(config) in command
    env = captured["env"]
    assert isinstance(env, dict)
    assert env.get("HONE_DEVICE") == "cpu"


def test_train_swe_sets_hone_device_and_invokes_subprocess(
    tmp_path: Path,
) -> None:
    config = tmp_path / "swe.yaml"
    config.write_text("model: m\ntrain: true\ndata: d\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["env"] = kwargs.get("env", {})
        return subprocess.CompletedProcess(
            kwargs.get("args", args[0] if args else None), 0, stdout="", stderr=""
        )

    with patch.object(train_module.subprocess, "run", side_effect=fake_run):
        result = runner.invoke(
            app,
            ["train", "swe", "--config", str(config), "--device", "gpu"],
        )
    assert result.exit_code == 0
    assert captured["env"].get("HONE_DEVICE") == "gpu"  # type: ignore[union-attr]
