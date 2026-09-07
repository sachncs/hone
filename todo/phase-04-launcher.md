# Phase 4 — MLX launcher (`hone/run.py`)

## Goal

Move the device-selection launcher logic from
`scripts/run_mlx_lora.py` into `hone/run.py` inside the package.
Apply the renamed single-word public API, rename the env var to
`HONE_DEVICE`, and remove the misleading `sys.exit(run_lora())`
wrapper.

## Standing acceptance criteria

See `todo/README.md`.

## Todos

### T4.1 — Move logic from `scripts/run_mlx_lora.py` → `hone/run.py`

The `scripts/` directory was deleted in T1.1. If any references
remain, relocate them now.

Acceptance:
- `hone/run.py` contains the launcher logic.
- No code references `scripts/run_mlx_lora.py`.

### T4.2 — Rename `configure_device` → `setup`

Acceptance:
- `from hone.run import setup` works.
- No `configure_device` identifier remains in `hone/`.
- Public function with complete type annotations.

### T4.3 — Rename `metal_available` → `has_metal`

Acceptance:
- `from hone.run import has_metal` works.
- Defensive `getattr(mx.metal, "is_available", lambda: False)()`
  fallback preserved.

### T4.4 — Rename `gpu_device_info` → `gpu_info`

Acceptance:
- `from hone.run import gpu_info` works.
- Cross-version fallback (`mx.device_info` vs `mx.metal.device_info`)
  preserved.

### T4.5 — Rename `requested_device` → `read_device`

Acceptance:
- `from hone.run import read_device` works.
- Reads `HONE_DEVICE` env var (after T4.7).
- Lowercase comparison, defaults to `"gpu"`.

### T4.6 — Rename `SUPPORTED_DEVICES` → `DEVICES`

Acceptance:
- `from hone.run import DEVICES` works.
- `DEVICES == frozenset({"cpu", "gpu"})`.

### T4.7 — Rename env var `FINETUNE_DEVICE` → `HONE_DEVICE`

Acceptance:
- `read_device()` reads `os.environ.get("HONE_DEVICE", "gpu")`.
- The error message mentions `HONE_DEVICE` explicitly.
- No reference to `FINETUNE_DEVICE` remains in `hone/`.

### T4.8 — Remove `sys.exit(run_lora())`

The wrapper is misleading because `mlx_lm.lora.main()` returns
`None`. The console-script `mlx_lm.lora` already calls `sys.exit`
in its own `__main__` block. Calling `run_lora()` directly is
sufficient.

Acceptance:
- `hone/run.py::main` calls `run_lora()` and returns.
- No `sys.exit(...)` in `hone/run.py`.
- No change in observable behavior — when invoked via
  `python -m hone.run`, the process still exits with the mlx
  script's exit code (which is 0 on success because mlx_lm.lora
  sets `sys.exit` implicitly).

### T4.9 — Verify `python -m hone.run --help` works

Acceptance:
- Exits 0.
- Prints `mlx_lm.lora`'s `--help` text.

### T4.10 — Move `tests/test_device_launcher.py` → `tests/mlx/test_run.py`

Update all test names and imports to the renamed identifiers.

Acceptance:
- `tests/test_device_launcher.py` deleted.
- `tests/mlx/test_run.py` exists with `pytest.importorskip("mlx.core")`
  at the top.
- All tests marked `@pytest.mark.mlx`.
- Test names reference `read_device`, `has_metal`, `setup` (not
  `requested_device`, `metal_available`, `configure_device`).

## Success criteria for Phase 4

All 10 todos complete with their acceptance criteria met.

Standing quality gates all pass:

```bash
uv run pytest -m "not mlx"
uv run pytest                  # on Apple Silicon
uv run ruff check hone tests
uv run ruff format --check hone tests
uv run mypy hone tests
```

`python -m hone.run --help` works and shows the mlx_lm.lora CLI.

The launcher logs the active device at the top of every training run.

GPU training refuses to start when Metal is unavailable, with a
clear error message mentioning `HONE_DEVICE=cpu` as the fallback.
