"""CLI subcommand group: hyperparameter search.

Thin transport layer over :mod:`hone.tune`. The command parses
typer options, expands the search space, runs every trial via
:class:`~hone.tune.TrialRunner`, and selects the best result for
the chosen objective.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import typer
import yaml

from hone.backends import SubprocessLauncher
from hone.errors import HoneError
from hone.log import setup
from hone.tune import TrialResult, TrialRunner, expand_search, select_best

app: typer.Typer = typer.Typer(help="Search hyperparameters.", no_args_is_help=True)


# Backwards-compat re-exports for tests that import the dataclasses
# directly from this module.
TrialSpec = __import__("hone.tune.spec", fromlist=["TrialSpec"]).TrialSpec


def _reraise(error: HoneError | ValueError) -> typer.BadParameter:
    """Convert a domain exception into a CLI-friendly error."""
    raise typer.BadParameter(str(error)) from error


@app.command("run")
def run(
    config: str = typer.Option(..., "--config", help="Base MLX LoRA config."),
    space: str = typer.Option(..., "--space", help="Search-space YAML."),
    output: str = typer.Option(..., "--output", help="Trial output directory."),
    device: str = typer.Option("gpu", "--device"),
    max_trials: int | None = typer.Option(None, "--max-trials"),
    objective: str = typer.Option("validation_loss", "--objective"),
    benchmark_command: str | None = typer.Option(None, "--benchmark-command"),
) -> None:
    """Run hyperparameter trials and select the best."""
    logger = setup(verbose=False)
    config_path = Path(config)
    space_path = Path(space)
    output_dir = Path(output)
    if not config_path.is_file():
        raise typer.BadParameter(f"config not found: {config_path}")
    if not space_path.is_file():
        raise typer.BadParameter(f"search space not found: {space_path}")

    base_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    search_space = yaml.safe_load(space_path.read_text(encoding="utf-8"))
    if not isinstance(base_config, dict) or not isinstance(search_space, dict):
        raise typer.BadParameter("configuration and search space must be YAML mappings")

    try:
        specs = expand_search(search_space, max_trials=max_trials)
    except ValueError as error:
        _reraise(error)
    output_dir.mkdir(parents=True, exist_ok=True)

    launcher_path = Path(__file__).resolve().parent.parent / "run.py"
    if not launcher_path.is_file():
        raise typer.BadParameter(f"launcher not found: {launcher_path}")

    runner = TrialRunner(
        launcher=SubprocessLauncher(),
        launcher_args=[str(launcher_path)],
        device=device,
        benchmark_command=benchmark_command,
        logger=logger,
    )

    results: list[TrialResult] = []
    for index, trial in enumerate(specs, 1):
        results.append(
            runner.run(
                trial_id=f"trial-{index:03d}",
                trial=trial,
                base_config=base_config,
                output_dir=output_dir,
            )
        )

    (output_dir / "results.json").write_text(
        json.dumps([asdict(result) for result in results], indent=2, sort_keys=True),
        encoding="utf-8",
    )

    try:
        best = select_best(results, objective=objective)
    except ValueError as error:
        _reraise(error)
    (output_dir / "best.json").write_text(
        json.dumps(asdict(best), indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info(
        "best trial=%s objective=%s value=%s",
        best.trial_id,
        objective,
        _objective_value(best, objective),
    )


def _objective_value(result: TrialResult, objective: str) -> float | None:
    """Return the objective value for logging."""
    from hone.tune.search import objective_value

    return objective_value(result, objective)


__all__ = ["TrialSpec", "app"]
