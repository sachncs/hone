# Phase 3 — CLI with typer

## Goal

Build the `typer`-based CLI in `hone/cli/`. Five top-level subcommands
(`prepare`, `train`, `generate`, `tune`, `evaluate`) with leaf actions
matching the renamed single-word convention. Each leaf action is a
single-word function name; the parent module name provides the verb.

## Standing acceptance criteria

See `todo/README.md`.

## Todos

### T3.1 — Implement `hone/cli/__init__.py` dispatcher

Five `typer.Typer` sub-apps, registered on a top-level `app`. `main()`
calls `app()`. `main()` returns `int` (exit code) for testability.

Acceptance:
- `hone --help` lists all 5 subcommands.
- `hone prepare --help` lists 5 leaf actions.
- `hone train --help` lists 3 leaf actions.
- `hone generate --help` lists 2 leaf actions.
- `hone tune --help` and `hone evaluate --help` show flags.
- `main()` returns 0 on success and non-zero on failure.
- `python -m hone` invokes `main()` via `hone/__main__.py`.

### T3.2 — Implement `hone/cli/prepare.py::file`

`file(input: Path, output: Path, ratio: float = 0.05, seed: int = 42,
max_samples: int | None = None)` — local JSONL → train/valid splits.

Acceptance:
- Reads JSONL from `input`.
- Normalizes each line via `Normalizer`.
- Splits via `Splitter(ratio, seed)`.
- Writes `train.jsonl` and `valid.jsonl` under `output`.
- Logs counts at INFO level.
- Raises (does not exit) on invalid input; CLI converts to non-zero
  exit code.

### T3.3 — Implement `hone/cli/prepare.py::code`

`code(dataset: str, split: str, output: Path, language: str = "PYTHON",
max_samples: int = 20000, scan_limit: int = 250000, ratio: float = 0.02,
seed: int = 42)` — HF competitive programming dataset → JSONL.

Acceptance:
- Streams the HF dataset, filters by language.
- Reservoir samples up to `max_samples`.
- Splits and writes train/valid JSONL files.
- Deterministic with same seed.

### T3.4 — Implement `hone/cli/prepare.py::swe`

`swe(dataset: str, split: str, output: Path, ratio: float = 0.05,
max_samples: int | None = None, max_chars: int = 14000,
seed: int = 42)` — HF SWE-bench dataset → JSONL.

Acceptance:
- Refuses non-`train` split with `ValueError` mentioning "evaluation
  patches would leak".
- Filters rows exceeding `max_chars`.
- Skips rows that fail normalization.
- Splits and writes train/valid JSONL files.

### T3.5 — Implement `hone/cli/prepare.py::all`

`all(repo: str, configs: str, split: str, output: Path, mode: str =
"sft")` — HF full corpus dump (no sampling, no validation split).

Acceptance:
- `mode="sft"` filters rows into SFT chat format.
- `mode="codeforces-text"` writes continued-pretraining text rows.
- Streams without loading the full dataset into memory.
- Logs progress every 100,000 rows.

### T3.6 — Implement `hone/cli/prepare.py::evaluate`

`evaluate(version: str = "release_v2", output: Path =
Path("data/eval/lcb.jsonl"))` — LCB eval prompts.

Acceptance:
- Downloads `livecodebench/code_generation_lite` for the given
  `version`.
- Writes the four documented fields: `question_id`,
  `question_content`, `contest_date`, `difficulty`.
- Writes to `data/eval/` (not to a training directory).

### T3.7 — Implement `hone/cli/train.py::code`

`code(config: Path, device: str = "gpu")` — mlx code training.

Acceptance:
- Sets `HONE_DEVICE` env var.
- Invokes `python -m hone.run --config <config>` via subprocess
  (clean isolation; easier to test).
- Surfaces the subprocess exit code to the CLI exit code.

### T3.8 — Implement `hone/cli/train.py::swe`

`swe(config: Path, device: str = "gpu")` — mlx SWE training.

Acceptance: same as T3.7.

### T3.9 — Implement `hone/cli/train.py::all`

`all(model: str = "openbmb/MiniCPM5-1B", layers: int = 8,
accum: int = 16, seq_len: int = 4096, save_every: int = 100000)` —
full sequence mlx training (was `scripts/train_full_sequence.sh`).

Acceptance:
- Calls `hone prepare all` for each of KIMI, CodeX, Ling-Coder,
  Codeforces.
- Trains each in sequence, resuming from the previous adapter.
- Surfaces subprocess exit codes.

### T3.10 — Add `--backend cuda` flag to `hone/cli/train.py`

When `--backend cuda`, dispatch to the Unsloth fallback (was
`scripts/train_unsloth.py`) instead of the mlx launcher.

Acceptance:
- `--backend cuda` invokes `hone.run.unsloth(...)` (the unsloth
  recipe lives in `hone/cli/train.py` or `hone/run.py`).
- Default backend is `mlx`.

### T3.11 — Implement `hone/cli/generate.py::prompt`

`prompt(prompt: str, model: str = "openbmb/MiniCPM5-1B",
adapter: Path | None = None, max_tokens: int = 1024, temperature:
float = 0.2)` — single-prompt generation.

Acceptance:
- Loads `model` with optional `adapter`.
- Applies the chat template with `enable_thinking=False`.
- Generates and prints the result to stdout.
- Exit code 0 on success.

### T3.12 — Implement `hone/cli/generate.py::file`

`file(input: Path, output: Path = Path("artifacts/lcb_outputs.json"),
model: str = "openbmb/MiniCPM5-1B", adapter: Path | None =
None, samples: int = 1, max_tokens: int = 1536, temperature: float =
0.2)` — bulk generation from JSONL prompts.

Acceptance:
- Reads prompts from `input`.
- For each prompt, generates `samples` completions.
- Strips markdown fences via the local `clean` helper (named `clean`
  per single-word convention).
- Writes a JSON array of `{question_id, code_list}` to `output`.

### T3.13 — Implement `hone/cli/tune.py::run`

`run(config: Path, space: Path, output: Path, device: str = "gpu",
max_trials: int | None = None, objective: str = "validation_loss",
benchmark_command: str | None = None)` — hyperparameter search
(was `scripts/tune_mlx.py`).

Acceptance:
- Builds Cartesian product of the search space.
- Runs each trial via subprocess (`python -m hone.run --config <path>`).
- Selects the best trial by objective (validation_loss minimizes,
  benchmark metric maximizes).
- Writes `results.json` and `best.json`.
- Surfaces failures via exit code.

### T3.14 — Implement `hone/cli/evaluate.py::run`

`run(version: str = "release_v2", samples: int = 1,
adapter: Path | None = None, lcb_dir: Path = Path("../LiveCodeBench"))`
— LCB evaluator (was `scripts/eval_lcb.sh`).

Acceptance:
- Runs `hone prepare evaluate` for the given version.
- Runs `hone generate file` with the adapter if provided.
- Invokes the official LCB evaluator via subprocess.
- Surfaces the evaluator exit code.

### T3.15 — Verify `hone --help` lists all 5 subcommands

Acceptance:
- Output includes `prepare`, `train`, `generate`, `tune`,
  `evaluate`.

### T3.16 — Verify `hone prepare --help` lists 5 leaf actions

Acceptance:
- Output includes `file`, `code`, `swe`, `all`, `evaluate`.

### T3.17 — Verify `hone train --help` lists 3 leaf actions

Acceptance:
- Output includes `code`, `swe`, `all`.

### T3.18 — Verify `hone generate --help` lists 2 leaf actions

Acceptance:
- Output includes `prompt`, `file`.

### T3.19 — Verify `hone tune --help` and `hone evaluate --help`

Acceptance:
- Both show their respective flags without errors.

## Success criteria for Phase 3

All 19 todos complete with their acceptance criteria met.

Standing quality gates all pass:

```bash
uv run pytest
uv run ruff check hone tests
uv run ruff format --check hone tests
uv run mypy hone tests
```

Every CLI subcommand and leaf action is invocable and shows help text.

`hone` is no longer a Python module reference but a working CLI.

Library code raises typed exceptions; the CLI converts them to exit
codes via typer's built-in error handling.
