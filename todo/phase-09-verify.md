# Phase 9 — Cleanup & verification

## Goal

Final cleanup. Run every quality gate. Smoke-test every CLI
subcommand by hand. Verify CI workflows parse.

## Standing acceptance criteria

See `todo/README.md`.

## Todos

### T9.1 — Delete `build/` if regenerated

Acceptance:
- `ls build/` returns "No such file or directory".

### T9.2 — Delete `dist/` if regenerated

Acceptance:
- `ls dist/` returns "No such file or directory".

### T9.3 — Delete `__pycache__/` directories

Acceptance:
- `find . -name __pycache__ -type d | grep -v .venv | wc -l`
  returns 0 outside `.venv/`.

### T9.4 — Delete `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `.benchmarks/`

Acceptance:
- None of these directories exist at the repo root.

### T9.5 — Run `uv pip install -e '.[dev]'` and verify success

Acceptance:
- Command exits 0.

### T9.6 — Run `uv run pytest -m "not mlx"` and verify all pass

Acceptance:
- Exits 0.
- All non-mlx tests pass.

### T9.7 — Run `uv run ruff check hone tests` and verify clean

Acceptance:
- Exits 0.

### T9.8 — Run `uv run ruff format --check hone tests` and verify clean

Acceptance:
- Exits 0.

### T9.9 — Run `uv run mypy hone tests` and verify clean

Acceptance:
- Exits 0.

### T9.10 — Manual smoke: `hone --help`

Acceptance:
- Lists all 5 subcommands: prepare, train, generate, tune,
  evaluate.

### T9.11 — Manual smoke: `hone prepare file --help`

Acceptance:
- Lists `input` and `output` as required.

### T9.12 — Manual smoke: `hone prepare code --help`

Acceptance:
- Lists `dataset`, `split`, `output` flags.

### T9.13 — Manual smoke: `hone prepare swe --help`

Acceptance:
- Lists `dataset`, `split`, `output` flags.

### T9.14 — Manual smoke: `hone prepare all --help`

Acceptance:
- Lists `repo`, `configs`, `output` flags.

### T9.15 — Manual smoke: `hone prepare evaluate --help`

Acceptance:
- Lists `version` and `output` flags.

### T9.16 — Manual smoke: `hone train code --help`

Acceptance:
- Lists `config` and `device` flags.

### T9.17 — Manual smoke: `hone generate prompt --help`

Acceptance:
- Lists `prompt`, `model`, `adapter`, `max_tokens`,
  `temperature`.

### T9.18 — Manual smoke: `hone generate file --help`

Acceptance:
- Lists `input`, `output`, `model`, `adapter`, `samples`,
  `max_tokens`, `temperature`.

### T9.19 — Manual smoke: `hone tune --help`

Acceptance:
- Lists `config`, `space`, `output`, `device`, `max_trials`,
  `objective`.

### T9.20 — Manual smoke: `hone evaluate --help`

Acceptance:
- Lists `version`, `samples`, `adapter`, `lcb_dir`.

### T9.21 — Manual smoke: `python -m hone --help`

Acceptance:
- Same output as T9.10.

### T9.22 — Manual smoke: `python -m hone.run --help` (Apple Silicon only)

Acceptance:
- Exits 0 and shows mlx_lm.lora's help text.

## Success criteria for Phase 9 (and the entire refactor)

All 22 todos complete with their acceptance criteria met.

All standing quality gates pass.

Every CLI subcommand has been smoke-tested by hand.

The repository is open-source ready:

- `LICENSE` is generic (not MiniCPM5-specific).
- `README.md` is concise and links to `docs/`.
- `CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md` exist.
- `pyproject.toml` is complete (name, version, description,
  console script, deps, extras, classifiers, URLs).
- `pyproject.toml` uses flat layout (`hone/`, not `src/hone/`).
- `tests/` contains ~100–120 tests covering behavior, edge cases,
  errors, integration, and property-based properties.
- `.github/workflows/` splits Linux and Apple CI.
- `.gitignore` excludes generated artifacts, caches, lockfiles,
  and `AGENTS.md`.
- No `scripts/` directory remains.
- No `src/` directory remains.
- No double-underscore methods remain in `hone/`.
- No `SystemExit` calls remain in `hone/` library code.
- No `print(...)` calls remain in `hone/` library code.
- Every public API has complete type annotations.
- Every public function has a Google-style docstring.
- The `hone` package is installable via `uv pip install -e .`.
- The `hone` console script is on `PATH` after install.
- `python -m hone` and `python -m hone.run` both work.
