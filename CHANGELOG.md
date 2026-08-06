# Changelog

All notable changes to **hone** are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `hone.split.partition`: streaming, seeded, disjoint train/valid
  split of a JSONL file with memory bounded by the validation size.
- `click` is now listed in the `[dev]` extra so the test suite
  imports it without a manual install step.

### Changed
- Renamed public API to comply with the single-word identifier
  rule: `spit` → `partition`, `_FORMAT` → `FORMAT`,
  `_run_mlx`/`_run_cuda` → `invoke_mlx`/`invoke_cuda`,
  `read_device` → `device`, `has_metal` → `metal`,
  `gpu_info` → `gpu`, `setup_device` → `select`,
  `build_trials` → `expand`, `parse_validation_loss` → `loss`,
  `load_metrics` → `metrics`, `write_trial_config` → `materialize`,
  `objective_value` → `score`, `run_trial` → `execute`,
  `strip_fences` → `unfence`.
- Forbidden local names (`item`, `valid_tmp`, `handler`,
  `utils`) replaced throughout the library and CLI.
- Test helper functions renamed to single-word identifiers
  (`make_examples` → `examples`, `make_jsonl` → `jsonl`,
  `read_lines` → `lines`, `write_jsonl` → `jsonl`,
  `write_chat_jsonl` → `chat_jsonl`, `make_chat_row` → `chat_row`,
  `make_example` → `example`, `fake_datasets_module` →
  `fake_datasets`, `unfake_datasets_module` → `restore_datasets`,
  `capture_subprocess_call` → `make_capture`,
  `argument_after` → `value_after`,
  `messages_strategy` → `messages`, `example_strategy` → `example`,
  `chat_record_strategy` → `chat_record`,
  `message_strategy` → `message`).

### Fixed
- `hone train all` now creates a deterministic 5% validation split
  (`valid.jsonl`) per stage before training, fixing the
  `Validation set not found or empty` failure from mlx_lm. `--iters`
  is the post-split train line count.
- `hone.split.partition` previously called `write()` with mismatched
  keyword arguments (`train_path`, `valid_path`) that did not match
  the helper's positional signature.

## [0.2.0] - 2026-08-03

### Added
- `hone` package: model- and dataset-agnostic supervised
  fine-tuning pipeline.
- Public API: `Example`, `Message`, `Role`, `Scalar`, `Meta`,
  `Normalizer`, `SweNormalizer`, `Splitter`, `Reader`, `Writer`,
  `REQUIRED_KEYS`, `MIN_VALID`, `LOGGER`, `setup`, `get`,
  `load`, `save`, `validate`.
- Five CLI subcommands (`prepare`, `train`, `generate`, `tune`,
  `evaluate`) implemented with `typer` and a single binary.
- `python -m hone.run` entry point for the MLX device launcher.
- MLX device launcher (`HONE_DEVICE` env var) that explicitly
  verifies Metal availability, refuses to start with `gpu` when
  Metal is unavailable, and logs the active accelerator.
- Apple Silicon CI workflow plus a Linux no-MLX CI workflow,
  consolidated under `.github/workflows/ci.yml`.
- Property-based tests with `hypothesis` for round-trip and
  invariant properties (Writer/Reader, Splitter, Normalizer).
- `docs/` split: install, quickstart, data, train, eval,
  architecture.
- `CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md`,
  `.github/dependabot.yml`.

### Changed
- Renamed package from `finetune-pipeline` to `hone`.
- Renamed `TrainingExample` → `Example`, `ChatMessage` →
  `Message`, `DatasetSplitter` → `Splitter`,
  `JsonlDatasetReader` → `Reader`, `JsonlDatasetWriter` →
  `Writer`.
- Renamed `FINETUNE_DEVICE` → `HONE_DEVICE`.
- Renamed `scripts/` orchestration to `hone <subcommand>`.
- Hardware-specific config names (`m3pro-code.yaml`,
  `m3pro-swe.yaml`, `m3pro.yaml`, `tuning-code.yaml`,
  `tuning-swe.yaml`) replaced with generic names (`code.yaml`,
  `swe.yaml`, `tune-code.yaml`, `tune-swe.yaml`). The dead
  `m3pro.yaml` was deleted.
- Moved from `src/finetune/` to flat `hone/` at repo root.
- Adopted `typer` for CLI parsing.
- Default model reference updated from
  `mlx-community/MiniCPM5-1B-4bit` (post-training quantized)
  to `openbmb/MiniCPM5-1B` (upstream base model). The
  `mlx-community/...` form is used only for inference.

### Removed
- `src/finetune/` package layout (replaced by flat `hone/`).
- `scripts/` directory (replaced by CLI subcommands).
- `configs/m3pro.yaml` (dead code).
- Double-underscore methods (`__normalize_messages`,
  `__parse_record`) — replaced by public methods.
- `SystemExit` calls from library code (CLI converts exceptions
  to exit codes via `typer.BadParameter` and `typer.Exit`).
- `print(...)` calls from library code (replaced by
  `hone.log`).

## [0.1.0]

Initial release as `finetune-pipeline`. Apple Silicon M3 Pro,
18 GB unified memory, MLX-LM with LoRA, hardware-named configs.
