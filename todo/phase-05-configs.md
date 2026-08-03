# Phase 5 — Configs

## Goal

Rename config files to model-agnostic, hardware-agnostic names.
Delete the dead `m3pro.yaml`. Update all references.

## Standing acceptance criteria

See `todo/README.md`.

## Todos

### T5.1 — Rename `configs/m3pro-code.yaml` → `configs/code.yaml`

Acceptance:
- `configs/code.yaml` exists with identical content.
- `configs/m3pro-code.yaml` no longer exists.

### T5.2 — Rename `configs/m3pro-swe.yaml` → `configs/swe.yaml`

Acceptance:
- `configs/swe.yaml` exists with identical content.
- `configs/m3pro-swe.yaml` no longer exists.

### T5.3 — Delete `configs/m3pro.yaml`

This file is referenced by no script in `scripts/` (which is now
deleted) and no code in `hone/`. It is dead code.

Acceptance:
- `configs/m3pro.yaml` no longer exists.

### T5.4 — Rename `configs/tuning-code.yaml` → `configs/tune-code.yaml`

Acceptance:
- `configs/tune-code.yaml` exists with identical content.

### T5.5 — Rename `configs/tuning-swe.yaml` → `configs/tune-swe.yaml`

Acceptance:
- `configs/tune-swe.yaml` exists with identical content.

### T5.6 — Update internal references to old config paths

Update any code in `hone/` that referenced the old config paths.

Acceptance:
- `grep -r "m3pro-" hone/ tests/` returns no matches.
- `grep -r "tuning-" hone/ tests/` returns no matches.
- `hone/cli/train.py` and `hone/cli/tune.py` reference the new
  config names.

## Success criteria for Phase 5

All 6 todos complete with their acceptance criteria met.

Standing quality gates all pass.

`configs/` contains exactly four files: `code.yaml`, `swe.yaml`,
`tune-code.yaml`, `tune-swe.yaml`.

No committed code references the old config paths.
