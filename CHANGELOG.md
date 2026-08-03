# Changelog

## [0.2.0] - 2026-08-03

### Added
- `hone` package: model- and dataset-agnostic supervised
  fine-tuning pipeline.
- Public API: `Example`, `Message`, `Role`, `Scalar`, `Meta`,
  `Normalizer`, `SweNormalizer`, `Splitter`, `Reader`, `Writer`.
- Five CLI subcommands (`prepare`, `train`, `generate`, `tune`,
  `evaluate`) implemented with `typer`.
- `python -m hone.run` entry point for the MLX device launcher.
- MLX device launcher (`HONE_DEVICE` env var) that explicitly
  verifies Metal availability and logs the active accelerator.
- Apple Silicon CI workflow (`.github/workflows/apple.yml`).
- Property-based tests with `hypothesis` for round-trip and
  invariant properties.
- `docs/` split: install, quickstart, data, train, eval,
  architecture.
- `CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md`.

### Changed
- Renamed package from `finetune-pipeline` to `hone`.
- Renamed `TrainingExample` → `Example`, `ChatMessage` →
  `Message`, `DatasetSplitter` → `Splitter`, `JsonlDatasetReader`
  → `Reader`, `JsonlDatasetWriter` → `Writer`.
- Renamed `FINETUNE_DEVICE` → `HONE_DEVICE`.
- Renamed `scripts/` orchestration to `hone <subcommand>`.
- Hardware-specific config names (`m3pro-*`) replaced with
  generic names (`code.yaml`, `swe.yaml`).
- Moved from `src/finetune/` to flat `hone/` at repo root.
- Adopted `typer` for CLI parsing.

### Removed
- `src/finetune/` package layout (replaced by flat `hone/`).
- `scripts/` directory (replaced by CLI subcommands).
- `configs/m3pro.yaml` (dead code).
- Double-underscore methods (`__normalize_messages`,
  `__parse_record`) — replaced by public methods.
- `SystemExit` calls from library code (CLI converts exceptions
  to exit codes).
- `print(...)` calls from library code (replaced by logger).

## [0.1.0]

Initial release as `finetune-pipeline`.
