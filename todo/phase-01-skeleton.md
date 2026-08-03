# Phase 1 — Package skeleton

## Goal

Replace the existing `src/finetune/` and `scripts/` layout with a flat
`hone/` package at the repository root. Establish the public-API surface
as placeholder exports, install the package, and verify the console
script runs (even if it errors because CLI is stubbed).

## Standing acceptance criteria

See `todo/README.md`. These apply to every todo in this phase.

## Todos

### T1.1 — Delete `scripts/` directory

Remove `scripts/` entirely. Every file currently in `scripts/` will be
replaced by a `hone <subcommand>` invocation in Phase 3. No compat shim.

Acceptance:
- `ls scripts/` returns "No such file or directory".
- No reference to `scripts/` remains in any committed file.
- All previous `scripts/*.py` logic is provably re-implemented in
  `hone/cli/*.py` (Phase 3). At Phase 1 completion this is forward-
  looking; the commit message must note that the CLI re-implementation
  is pending.

### T1.2 — Delete `build/` directory

Already gitignored. Just remove the directory if it exists.

Acceptance:
- `ls build/` returns "No such file or directory".

### T1.3 — Delete `dist/` directory

Already gitignored. Just remove the directory if it exists.

Acceptance:
- `ls dist/` returns "No such file or directory".

### T1.4 — Delete `uv.lock`

This package is a library; the lockfile belongs to applications, not
libraries. Lockfile is gitignored; remove the existing file.

Acceptance:
- `ls uv.lock` returns "No such file or directory".
- `uv pip install -e .` still succeeds without `uv.lock` (it will
  regenerate one for local development but it stays untracked).

### T1.5 — Delete `src/` directory

Flat layout. No `src/`.

Acceptance:
- `ls src/` returns "No such file or directory".

### T1.6 — Create directory `hone/`

Acceptance:
- `hone/` exists at the repo root.
- `hone/` contains an empty `__init__.py` (covered in T1.8).

### T1.7 — Create directory `hone/cli/`

Acceptance:
- `hone/cli/` exists.
- `hone/cli/__init__.py` exists (empty placeholder, covered in T1.18).

### T1.8 — Create `hone/__init__.py`

Module docstring + `__version__ = "0.2.0"`. Public API exports are
added in Phase 2 (T2.13).

Acceptance:
- `python -c "import hone; print(hone.__version__)"` prints `0.2.0`.
- `hone/__init__.py` has a one-line module docstring.
- No re-exports of unfinished placeholders.

### T1.9 — Create `hone/__main__.py`

Enables `python -m hone`. Body: `from hone.cli import main; main()`.

Acceptance:
- `python -m hone --help` runs (will fail with AttributeError because
  `hone.cli` is empty; this is acceptable for Phase 1).
- `hone/__main__.py` has a one-line module docstring.

### T1.10 — Create `hone/types.py`

Define `JsonObject` and `JsonScalar` type aliases. Replaces the three
duplicate `type JsonObject = dict[str, object]` definitions currently
in `scripts/prepare_code_hf.py`, `scripts/prepare_full_hf.py`, and
`scripts/tune_mlx.py`.

Acceptance:
- `from hone.types import JsonObject, JsonScalar` works.
- `JsonObject: TypeAlias = dict[str, object]`.
- `JsonScalar: TypeAlias = str | int | float | bool | None`.
- No imports from `mlx` or other heavy deps.
- Module docstring explains the purpose.

### T1.11 — Create empty `hone/model.py`

Placeholder. Real implementation in Phase 2 (T2.1–T2.5).

Acceptance:
- File exists.
- File has a one-line module docstring stating "filled in Phase 2".
- No functional code yet.

### T1.12 — Create empty `hone/normalize.py`

Acceptance: same as T1.11.

### T1.13 — Create empty `hone/split.py`

Acceptance: same as T1.11.

### T1.14 — Create empty `hone/jsonl.py`

Acceptance: same as T1.11.

### T1.15 — Create empty `hone/log.py`

Acceptance: same as T1.11.

### T1.16 — Create empty `hone/config.py`

Acceptance: same as T1.11.

### T1.17 — Create empty `hone/run.py`

Acceptance: same as T1.11.

### T1.18 — Create empty `hone/cli/__init__.py`

Acceptance: same as T1.11, but in `hone/cli/`.

### T1.19 — Create empty `hone/cli/prepare.py`

Acceptance: same as T1.11.

### T1.20 — Create empty `hone/cli/train.py`

Acceptance: same as T1.11.

### T1.21 — Create empty `hone/cli/generate.py`

Acceptance: same as T1.11.

### T1.22 — Create empty `hone/cli/tune.py`

Acceptance: same as T1.11.

### T1.23 — Create empty `hone/cli/evaluate.py`

Acceptance: same as T1.11.

### T1.24 — Rewrite `pyproject.toml`

Flat layout. Package name `hone`. Console script `hone`. Add typer dep.
Drop `src` references. Drop `scripts/`. Add `hypothesis` to `dev` extra.

Required content:

```toml
[project]
name = "hone"
version = "0.2.0"
description = "Model- and dataset-agnostic supervised fine-tuning pipeline."
requires-python = ">=3.12,<3.14"
license = { text = "MIT" }
authors = [{ name = "hone contributors" }]
readme = "README.md"
keywords = ["finetune", "lora", "mlx", "training"]
classifiers = [
  "Development Status :: 4 - Beta",
  "Intended Audience :: Developers",
  "License :: OSI Approved :: MIT License",
  "Programming Language :: Python :: 3.12",
  "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
  "pyyaml>=6.0",
  "typer>=0.12",
]

[project.urls]
Homepage = "https://github.com/example/hone"
Repository = "https://github.com/example/hone"
Issues = "https://github.com/example/hone/issues"

[project.scripts]
hone = "hone.cli:main"

[project.optional-dependencies]
dev = ["mypy>=1.13", "pytest>=8.3", "ruff>=0.8", "hypothesis>=6.100"]
mlx = ["mlx-lm[train]>=0.29.0", "datasets>=3.0"]
cuda = ["torch>=2.5", "datasets>=3.0", "transformers>=4.57.3,<5", "trl>=0.18", "peft>=0.14", "unsloth"]

[build-system]
requires = ["setuptools>=70"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["hone", "hone.cli"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
pythonpath = ["."]
markers = [
  "mlx: requires Apple Silicon Metal backend",
  "slow: takes more than 1 second",
]

[tool.ruff]
target-version = "py312"
line-length = 88
include = ["hone", "tests"]
extend-exclude = ["build", "dist"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "PL", "RUF"]

[tool.ruff.lint.isort]
known-first-party = ["hone"]

[tool.mypy]
explicit_package_bases = true
python_version = "3.12"
check_untyped_defs = true
disallow_untyped_defs = true
ignore_missing_imports = true
no_implicit_optional = true
warn_return_any = true
warn_unused_ignores = true
```

Acceptance:
- `uv pip install -e '.[dev]'` succeeds.
- `hone` console script is on `PATH` after install.
- `uv pip install -e '.[mlx]'` would install mlx deps.
- No `[tool.setuptools.packages.find]` references; explicit `packages =`.
- `pythonpath = ["."]` (flat layout).
- Typer is a hard dependency (used by `hone.cli`).

### T1.25 — Update `LICENSE` copyright

Replace "MiniCPM5 coding pipeline contributors" with "hone contributors".

Acceptance:
- `LICENSE` no longer mentions "MiniCPM5".
- License body (MIT terms) unchanged.

### T1.26 — Verify `uv pip install -e '.[dev]'` succeeds

Acceptance:
- Command exits 0.
- `python -c "import hone"` works.
- `which hone` resolves to the venv-installed script.

### T1.27 — Verify `hone --help` runs

Acceptance:
- `hone --help` exits with a clean error (because CLI is stubbed) or
  with help text. Either is acceptable at Phase 1 completion; the CLI
  implementation is in Phase 3.
- The error message does not contain "ModuleNotFoundError" or
  "ImportError" — those indicate a missing dependency or path issue.

## Success criteria for Phase 1

All 27 todos complete with their acceptance criteria met.

Standing quality gates all pass:

```bash
uv run pytest           # may be empty
uv run ruff check hone tests
uv run ruff format --check hone tests
uv run mypy hone tests
```

`python -c "import hone; print(hone.__version__)"` prints `0.2.0`.

`hone --help` does not crash with `ImportError` or `ModuleNotFoundError`.

`AGENTS.md` is not tracked by git (gitignored).

`scripts/` and `src/` are gone.

`uv.lock` is not tracked by git.
