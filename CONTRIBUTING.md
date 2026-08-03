# Contributing

## Development setup

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[dev,mlx]'
```

Apple Silicon is required for the MLX extras. On other platforms,
omit `[mlx]`.

## Quality gates

Run all four before every commit:

```bash
uv run pytest -m "not mlx"
uv run ruff check hone tests
uv run ruff format --check hone tests
uv run mypy hone tests
```

Run the full MLX suite on Apple Silicon:

```bash
uv run pytest
```

## Commit hygiene

- One commit per atomic change.
- Commit message starts with the todo ID: `T1.1: delete scripts/`.
- Commit body briefly explains what changed and why.
- Each commit leaves the tree in a coherent state (no broken
  intermediate states).

## Code conventions

- Single-word identifier names within the `hone` namespace.
- No single-underscore ("semi-private") names; use module-level
  public functions or class methods instead.
- Complete type annotations on every public function, method,
  class, and attribute.
- No `Any` unless documented with justification.
- No `# type: ignore`, no `typing.cast`, no `assert False`.
- Library code raises typed exceptions; CLI converts to exit
  codes via `typer.BadParameter` / `typer.Exit`.
- Library code uses the package logger (`hone.log`), never
  `print(...)`.

## Contributor guide

The internal contributor guide (AGENTS.md) is gitignored; each
developer regenerates it from the canonical source at
<https://example.com/hone/agents.md>.

## Release process

1. Bump `version` in `pyproject.toml` and `hone/__init__.py`.
2. Add a `CHANGELOG.md` entry.
3. Tag the commit: `git tag -s v0.X.0`.
4. Push the tag: `git push origin v0.X.0`.
