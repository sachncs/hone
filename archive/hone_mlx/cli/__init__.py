"""CLI dispatcher built on typer.

Five top-level subcommands: prepare, train, generate, tune, evaluate.
Each subcommand module exposes its own typer.Typer as `app`; the
dispatcher attaches them.

main() returns an int exit code so tests can assert on it.
"""

from __future__ import annotations

from typing import NoReturn

import typer

from hone.cli import evaluate, generate, prepare, train, tune

app: typer.Typer = typer.Typer(
    name="hone",
    help="hone: a model- and dataset-agnostic supervised fine-tuning pipeline.",
    no_args_is_help=True,
)

app.add_typer(prepare.app, name="prepare")
app.add_typer(train.app, name="train")
app.add_typer(generate.app, name="generate")
app.add_typer(tune.app, name="tune")
app.add_typer(evaluate.app, name="evaluate")


def main() -> int:
    """Console-script entry point. Returns a process exit code."""
    try:
        app()
        return 0
    except SystemExit as exit_event:
        return 0 if exit_event.code is None else int(exit_event.code)


def entry() -> NoReturn:
    """python -m hone entry point. Calls sys.exit with the CLI exit code."""
    raise SystemExit(main())
