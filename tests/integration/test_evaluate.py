"""Tests for hone.cli.evaluate."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from hone.cli import app

runner = CliRunner()


def test_evaluate_refuses_missing_lcb_dir(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["evaluate", "run", "--lcb-dir", str(tmp_path / "no-such-lcb")],
    )
    assert result.exit_code != 0
    output = result.output + (result.stderr or "")
    assert "LiveCodeBench" in output or "lcb" in output.lower()


def test_evaluate_invokes_lcb_runner(tmp_path: Path) -> None:
    lcb_dir = tmp_path / "LCB"
    lcb_dir.mkdir()

    captured: dict[str, object] = {}

    class FakeCompleted:
        returncode = 0

    def fake_run(*args: object, **kwargs: object) -> FakeCompleted:
        captured["args"] = kwargs.get("args", args[0] if args else None)
        captured["cwd"] = kwargs.get("cwd")
        return FakeCompleted()

    with (
        patch("hone.cli.evaluate.prepare_evaluate"),
        patch("hone.cli.evaluate.generate_file"),
        patch("hone.cli.evaluate.subprocess.run", side_effect=fake_run),
    ):
        result = runner.invoke(app, ["evaluate", "run", "--lcb-dir", str(lcb_dir)])

    assert result.exit_code == 0
    args = captured.get("args")
    assert isinstance(args, list)
    assert "-m" in args
    assert "lcb_runner.runner.custom_evaluator" in " ".join(args)
    assert captured.get("cwd") == lcb_dir
