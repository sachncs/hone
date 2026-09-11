# Contributing

## Development setup

The prepare library is platform-independent (Apple Silicon is
*not* required). The Soup driver + the `bench/` MLX harness do
require Apple Silicon; install those extras only on a Mac.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[dev]'
```

To install the Soup driver (Apple Silicon only):

```bash
uv pip install "soup-cli[mlx]==0.73.2" "transformers>=4.57,<5" "huggingface-hub<1.0,>=0.34"
```

The same pins are used by `setup.sh` so the two install paths
converge on the same dependency state.

## Quality gates

Run all four before every commit (mirrors `.github/workflows/ci.yml`):

```bash
uv run pytest
uv run ruff check bench hone tests
uv run ruff format --check bench hone tests
uv run mypy bench hone tests
```

The `pytest -m "not mlx"` selector from the 0.2.x days no longer
exists: no test in `tests/` carries a `pytest.mark.mlx` marker, so
the unqualified `pytest` command is the one CI runs.

## Commit hygiene

- One commit per atomic change.
- Commit message starts with `fix:`, `feat:`, `docs:`, `refactor:`,
  `test:`, or `style:` (Conventional Commits).
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

## Release process

1. Bump `version` in `pyproject.toml` (the only source of truth — `hone/__init__.py` no longer exists).
2. Add a `CHANGELOG.md` entry.
3. Tag the commit: `git tag -s v0.X.0`.
4. Push the tag: `git push origin v0.X.0`.
5. Draft a GitHub Release with the CHANGELOG entry as the body
   (`gh release create v0.X.0 --notes-file CHANGELOG.md`).
   Publishing the release makes `pip install hone==0.X.0` reproducible.
