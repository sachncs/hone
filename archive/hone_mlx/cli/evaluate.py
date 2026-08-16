"""CLI subcommand group: evaluation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import typer

from hone.cli.generate import file as generate_file
from hone.cli.prepare import evaluate as prepare_evaluate
from hone.log import setup

app: typer.Typer = typer.Typer(help="Evaluate a trained adapter.", no_args_is_help=True)


@app.command("run")
def run(
    version: str = typer.Option(
        "release_v2", "--version", help="LiveCodeBench release tag."
    ),
    samples: int = typer.Option(1, "--samples"),
    adapter: str | None = typer.Option(None, "--adapter"),
    lcb_dir: str = typer.Option("../LiveCodeBench", "--lcb-dir"),
) -> None:
    """Run the LiveCodeBench evaluator on a trained adapter."""
    logger = setup(verbose=False)
    lcb_path = Path(lcb_dir)
    if not lcb_path.is_dir():
        raise typer.BadParameter(
            f"LiveCodeBench directory not found: {lcb_path}. "
            "Clone https://github.com/LiveCodeBench/LiveCodeBench into it first."
        )

    prepare_evaluate(version=version)
    if adapter is not None:
        generate_file(
            input="data/eval/lcb.jsonl",
            output="artifacts/lcb_outputs.json",
            adapter=adapter,
            samples=samples,
        )
    else:
        generate_file(
            input="data/eval/lcb.jsonl",
            output="artifacts/lcb_outputs.json",
            samples=samples,
        )

    runner_command = [
        sys.executable,
        "-m",
        "lcb_runner.runner.custom_evaluator",
        "--custom_output_file",
        str(Path("artifacts") / "lcb_outputs.json"),
    ]
    logger.info("invoking official LCB evaluator at %s", lcb_path)
    environment = os.environ.copy()
    completed = subprocess.run(
        runner_command,
        cwd=lcb_path,
        check=False,
        env=environment,
    )
    raise typer.Exit(code=completed.returncode)


__all__ = ["app"]
