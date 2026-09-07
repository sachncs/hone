# Phase 7 — Documentation

## Goal

Rewrite `README.md` for the new package. Add `docs/` split for
navigability. Add `CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md`.

## Standing acceptance criteria

See `todo/README.md`.

## Todos

### T7.1 — Rewrite `README.md`

Keep under 200 lines. Sections:
- Title + one-line tagline.
- "Why hone?" (one short paragraph).
- Quickstart (5 commands or fewer).
- Table of contents linking to `docs/`.
- License, contributing, security pointers.
- No `scripts/...` references.
- No MiniCPM5-specific framing.

Acceptance:
- `README.md` is ≤ 200 lines.
- `grep -E "scripts/" README.md` returns no matches.
- All CLI invocations match `hone <subcommand>` syntax.
- `README.md` references `docs/install.md`, `docs/quickstart.md`,
  `docs/data.md`, `docs/train.md`, `docs/eval.md`,
  `docs/architecture.md`.

### T7.2 — Create `docs/install.md`

Covers installation (uv, pip, conda), Python version, optional
extras (`[mlx]`, `[cuda]`, `[dev]`), and Apple Silicon requirements.

Acceptance:
- File exists at `docs/install.md`.
- Mentions Python 3.12+.
- Documents `uv pip install -e '.[dev,mlx]'` and the CUDA
  alternative.

### T7.3 — Create `docs/quickstart.md`

End-to-end example: install → prepare data → train → generate.

Acceptance:
- File exists at `docs/quickstart.md`.
- Walks through at least: `hone prepare file`, `hone train code`,
  `hone generate prompt`.
- Uses example data (e.g., a tiny inline JSONL snippet).

### T7.4 — Create `docs/data.md`

Documents the data contract:
- JSONL schema (`{"messages": [...]}`).
- `prompt`/`completion` alternative.
- SWE-bench format.
- Reservoir sampling parameters.
- Split ratio behavior.

Acceptance:
- File exists at `docs/data.md`.
- Shows example records for each format.
- Documents `hone prepare file`, `hone prepare code`,
  `hone prepare swe`, `hone prepare all`, `hone prepare evaluate`.

### T7.5 — Create `docs/train.md`

Documents the training commands:
- `hone train code`, `hone train swe`, `hone train all`.
- Config files and their keys (or pointer to
  `mlx_lm.lora`'s docs).
- Tuning: `hone tune`.
- Backend selection (`--backend cuda`).

Acceptance:
- File exists at `docs/train.md`.
- Covers all three training leaf actions and the tuning command.

### T7.6 — Create `docs/eval.md`

Documents evaluation:
- `hone evaluate` for LiveCodeBench.
- SWE-bench: explain the harness requirement; this is not a
  generation-only pipeline.
- Benchmark hooks for `hone tune`.

Acceptance:
- File exists at `docs/eval.md`.
- Documents `hone evaluate`.
- States explicitly that plain-text generation is not a valid
  substitute for SWE-bench's official evaluator.

### T7.7 — Create `docs/architecture.md`

Package layout, public API, design choices, naming conventions.

Acceptance:
- File exists at `docs/architecture.md`.
- Includes the directory tree of `hone/`.
- Documents the public API surface (the `from hone import ...`
  set).
- Explains the single-word naming convention.

### T7.8 — Create `CHANGELOG.md`

Document the v0.2.0 restructure.

Acceptance:
- File exists at `CHANGELOG.md`.
- Top entry is `## [0.2.0] - <date>` with sections: Added,
  Changed, Removed.
- Lists the rename of the package and CLI.
- Lists the deletion of `scripts/`.
- Lists the CI split.

### T7.9 — Create `CONTRIBUTING.md`

Dev setup, quality gates, release process.

Acceptance:
- File exists at `CONTRIBUTING.md`.
- Lists the four quality gates:
  `pytest -m "not mlx"`, `ruff check`, `ruff format --check`,
  `mypy`.
- Describes the commit-per-todo convention (each commit references
  a todo ID).
- Notes the AGENTS.md file (gitignored) is regenerated per
  developer.

### T7.10 — Create `SECURITY.md`

Placeholder security policy.

Acceptance:
- File exists at `SECURITY.md`.
- States the supported versions (only the latest).
- Provides a contact channel (GitHub Issues for now).

## Success criteria for Phase 7

All 10 todos complete with their acceptance criteria met.

`README.md` is ≤ 200 lines and contains no `scripts/` references.

`docs/` contains six files, each ≤ 200 lines.

`CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md` exist and are
non-empty.

Every public symbol listed in `hone/__init__.py` is mentioned in
either `README.md` or `docs/architecture.md`.
