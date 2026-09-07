# Changelog

All notable changes to **hone** are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-08-17

### Changed

- **Soup-first cut.** The training pipeline (`hone.train`,
  `hone.tune`, `hone.generate`, `hone.cli.*`, `hone.run`,
  `hone.backends`, `hone.model`, `hone.jsonl`, `hone.normalize`,
  `hone.split`, `hone.chat`, `hone.errors`, `hone.config`,
  `hone.log`, and the `hone` CLI binary) is moved to
  [`archive/hone_mlx/`](archive/hone_mlx/) as a frozen alternative
  for operators who need the hand-rolled MLX/Unsloth pipeline.
  See [`docs/ARCHIVE.md`](docs/ARCHIVE.md) for what it is and how
  to revive it.
- **`hone` is now a JSONL prep library.** The remaining
  `hone.prepare` subpackage is the only thing exposed at the top
  level. It has zero CLI, zero MLX deps, and zero Unsloth deps —
  just `datasets` plus Python stdlib.
- Repository description changed from
  "Apple Silicon supervised fine-tuning pipeline" to
  "JSONL data preparation for Soup fine-tuning."
- Repository keywords narrowed to `soup, finetune, data-prep, jsonl, huggingface`.
- README rewritten around Soup as the primary path; the MLX
  alternative is documented once and moved to ARCHIVE.md.
- `docs/{install,quickstart,data,train,eval,architecture}.md`
  removed; they described the deleted MLX pipeline. Replaced with
  [`docs/ARCHIVE.md`](docs/ARCHIVE.md) (frozen-driver reference) and
  the existing [`docs/SOTA-EXPECTATIONS.md`](docs/SOTA-EXPECTATIONS.md).
- `pyproject.toml` slimmed: `typer`, `pyyaml`, `mlx-lm`,
  `transformers`, `trl`, `peft`, `unsloth`, `torch`, `hypothesis`,
  `click` removed from runtime + dev dependencies. `[soup]` is the
  new optional extra; `[mlx]` and `[cuda]` removed.
- `setup.sh` simplified: installs `hone[dev,soup]`, runs the new
  8-test `tests/test_prepare.py` suite, runs `train-soup.sh smoke`
  when `data/full/codex/train.jsonl` exists.

### Removed

- `hone.cli` package and the `hone` CLI binary.
- `hone.train`, `hone.tune`, `hone.generate` subpackages.
- `configs/{code,swe,smoke,smoke-kimi,tune-code,tune-swe}.yaml`.
- `train.sh`.
- `tests/{unit,integration,mlx,property}/` — 218 tests covering
  the deleted MLX pipeline. Replaced by `tests/test_prepare.py`
  (8 tests) covering the prepare layer's actual behavior.

### Added

- `tests/test_prepare.py` — 8 end-to-end tests for the prepare
  service: local file split, prompt/completion acceptance, ratio
  validation, malformed-JSON rejection, empty-row detection,
  exception hierarchy, and the chat role enum.
- [`docs/ARCHIVE.md`](docs/ARCHIVE.md) — reference for the
  frozen MLX driver; lists every archived module, its purpose,
  and how to revive it on a development branch.

## [0.2.0] - 2026-08-03

The last release of the original `hone` MLX/Unsloth driver.
Summarized here for history; everything from this release lives
in [`archive/hone_mlx/`](archive/hone_mlx/) and is no longer
maintained.

### Added

- `hone` package: end-to-end supervised fine-tuning pipeline
  with five `typer` subcommands (`prepare`, `train`, `generate`,
  `tune`, `evaluate`).
- `python -m hone.run` entry point for the MLX device launcher.
- MLX device launcher reading `HONE_DEVICE` with explicit Metal
  verification and graceful refusal when Metal is unavailable.
- CUDA/Unsloth path on NVIDIA hosts via the `[cuda]` extra.
- Property-based tests with `hypothesis` for Writer/Reader,
  Splitter, and Normalizer invariants.
- Apple Silicon + Linux CI workflows under
  `.github/workflows/ci.yml`.

### Changed

- Renamed package from `finetune-pipeline` to `hone`.
- Default model reference updated from
  `mlx-community/MiniCPM5-1B-4bit` (post-training quantized) to
  `openbmb/MiniCPM5-1B` (upstream base model).
