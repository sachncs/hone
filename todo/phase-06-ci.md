# Phase 6 — CI split

## Goal

Split CI into Linux (no-mlx) and Apple (full) workflows. Linux CI
runs `pytest -m "not mlx"` only. Apple CI runs the full suite.
Both must pass for the repo to be considered healthy.

## Standing acceptance criteria

See `todo/README.md`.

## Todos

### T6.1 — Add `pytest.importorskip("mlx.core")` to mlx tests

`tests/mlx/test_run.py` imports `mlx.core` at module level. On
Linux, mlx is unavailable; the skip must prevent `ImportError`.

Acceptance:
- First non-import line of `tests/mlx/test_run.py` is
  `pytest.importorskip("mlx.core")`.
- All tests in this file skip (not fail) on Linux.

### T6.2 — Mark mlx tests with `@pytest.mark.mlx`

Acceptance:
- Every test in `tests/mlx/test_run.py` carries
  `@pytest.mark.mlx`.
- `pytest -m "not mlx"` excludes every test in that file.

### T6.3 — Update `.github/workflows/quality.yml` for Linux

Run `uv run pytest -m "not mlx"` on Ubuntu-latest.

Acceptance:
- Workflow file uses `ubuntu-latest`.
- The pytest invocation includes `-m "not mlx"`.
- Workflow runs `uv sync --extra dev`.
- Workflow runs `ruff check`, `ruff format --check`, `mypy`.

### T6.4 — Create `.github/workflows/apple.yml`

Run the full test suite on macos-latest.

Acceptance:
- Workflow file uses `macos-latest`.
- The pytest invocation is `uv run pytest` (no marker filter).
- Workflow runs `uv sync --extra dev --extra mlx`.
- Workflow runs `ruff check`, `ruff format --check`, `mypy`.

### T6.5 — Verify both workflow files parse as valid YAML

Acceptance:
- `python -c "import yaml; yaml.safe_load(open('.github/workflows/quality.yml'))"`
  exits 0.
- `python -c "import yaml; yaml.safe_load(open('.github/workflows/apple.yml'))"`
  exits 0.

## Success criteria for Phase 6

All 5 todos complete with their acceptance criteria met.

`pytest -m "not mlx"` is the canonical non-mlx command.

`pytest` (full) runs on Apple CI only.

No test ever fails on Linux due to a missing mlx import.
