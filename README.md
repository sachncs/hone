<p align="center">
  <h1 align="center">hone</h1>
  <p align="center">Apple Silicon supervised fine-tuning pipeline — plug-and-play LoRA training on the M-series Mac.</p>
  <p align="center">
    <a href="#installation"><img src="https://img.shields.io/badge/python-3.12%7C3.13-blue" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
    <a href="https://github.com/sachncs/finetune/actions"><img src="https://img.shields.io/github/actions/workflow/status/sachncs/finetune/ci.yml?branch=main" alt="CI"></a>
    <a href="#installation"><img src="https://img.shields.io/badge/hone-0.2.0-blue" alt="Version"></a>
    <a href="https://github.com/sachncs/finetune/stargazers"><img src="https://img.shields.io/github/stars/sachncs/finetune" alt="Stars"></a>
  </p>
</p>

**hone** is a single-word, plug-and-play supervised fine-tuning
pipeline **for Apple Silicon**. Built and tested on macOS with the
M-series unified-memory architecture, it uses
[MLX-LM](https://github.com/ml-explore/mlx-lm) with LoRA out of
the box and falls back to a CUDA/Unsloth path on NVIDIA hosts.

This is a fine-tuning pipeline for Apple Silicon. The model used
for both training and validation is `openbmb/MiniCPM5-1B`. The
data contract is a plain JSONL of chat messages or
`prompt`/`completion` pairs; the CLI is a single `hone` binary with
five top-level subcommands (`prepare`, `train`, `generate`,
`tune`, `evaluate`).

The pipeline is fully typed, fully documented (Google-style
docstrings on every public symbol), and ships with explicit device
selection (`HONE_DEVICE`) plus fail-fast validation when Metal is
unavailable on Apple Silicon.

> **Author and maintainer**: Sachin

| Concern | Library |
|---|---|
| Backend (Apple Silicon) | [MLX-LM](https://github.com/ml-explore/mlx-lm) with LoRA |
| Backend (NVIDIA) | [Unsloth](https://github.com/unslothai/unsloth) with LoRA (optional `[cuda]`) |
| Soup driver | [Soup](https://github.com/MakazhanAlpamys/Soup) (optional `[soup]`) — one-YAML training via `soup train --backend mlx` |
| Data format | JSONL (`messages` or `prompt`/`completion`) |
| CLI | [Typer](https://github.com/tiangolo/typer) |
| Property tests | [Hypothesis](https://github.com/HypothesisWorks/hypothesis) |
| Lint / format | ruff |
| Type check | mypy (strict) |
| Tests | pytest |

## Features

- **Built for Apple Silicon** — `uv pip install -e '.[dev,mlx]'` and `hone train code` runs on the M-series GPU; the launcher detects Metal at startup and refuses to fall back to CPU silently.
- **Unified-memory aware** — Conservative defaults for 18 GB M3 Pro: 4-bit base model, batch size 1, gradient checkpointing, gradient accumulation instead of memory-heavy batch.
- **Explicit GPU device selection** — `HONE_DEVICE=gpu|cpu` env var; launcher refuses to start without Metal on Apple Silicon.
- **Deterministic dataset preparation** — seeded splitter (`Splitter(ratio, seed)`) with provably disjoint partitions.
- **Reservoir sampling for large corpora** — `hone prepare code` streams HuggingFace datasets with deterministic seed.
- **HuggingFace-aware** — `hone prepare swe` rejects non-train splits to prevent eval-patch leakage.
- **Multi-stage training** — `hone train all` orchestrates the full KIMI → CodeX → Ling-Coder → Codeforces sequence with adapter resume.
- **Hyperparameter search** — `hone tune` runs Cartesian-product trials and selects by validation loss or benchmark metric.
- **LiveCodeBench evaluation** — `hone evaluate` invokes the official LCB evaluator on a trained adapter.
- **Soup integration** — one-YAML SFT pipeline (configs/soup-sft-codex-*.yaml + train-soup.sh) trains MiniCPM5-1B-MLX on CodeX with `soup train --backend mlx` and ships the LoRA adapter. ~7.4 GB peak on M3 Pro 18 GB.
- **Production-grade logging** — every entry point logs via `hone.log`; CLI surfaces exit codes via typer.
- **No half-private names** — every identifier is public per AGENTS.md; library code raises typed exceptions; CLI converts to exit codes.

## Apple Silicon

hone is targeted at, and primarily developed on, Apple Silicon
(macOS on M1/M2/M3/M4-series chips with unified memory). The
MLX backend runs on the Metal GPU; the launcher reads `HONE_DEVICE`
and refuses to start with `gpu` when Metal is unavailable. The
default model and the recommended hyperparameters are tuned for an
18 GB M3 Pro and should be adjusted for other memory budgets.

For NVIDIA hosts, install the `[cuda]` extra and run with
`--backend cuda`. The CUDA path is a thin wrapper around the
OpenBMB MiniCPM5 Unsloth recipe; it is not the default and is not
tested in CI.

## Installation

### From source (Apple Silicon)

```bash
git clone https://github.com/sachncs/finetune.git
cd finetune
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[dev,mlx]'
```

Verify:

```bash
python -c "import hone; print(hone.__version__)"   # 0.2.0
hone --help
python -m hone.run --help   # delegates to mlx_lm.lora
```

### From source (NVIDIA)

```bash
git clone https://github.com/sachncs/finetune.git
cd finetune
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[dev,cuda]'
hone train code --backend cuda
```

| Extra | Includes |
|---|---|
| `dev` | mypy, pytest, ruff, hypothesis |
| `mlx` | mlx-lm, datasets (Apple Silicon only) |
| `cuda` | torch, transformers, trl, peft, unsloth (NVIDIA) |

Core deps only: `pyyaml`, `typer`.

## Quick Start

### Prepare data

```bash
hone prepare file --input data/raw.jsonl --output data/processed/code
```

Reads JSONL, normalizes each line to a chat record, splits 95/5 with
seed 42, writes `train.jsonl` and `valid.jsonl` under the output
directory.

For HF-config corpora that mix very long sequences (e.g. KIMI's
general-distillation + STEM configs reach 10000+ tokens per row), pass
`--max-tokens` to drop the long tail before training:

```bash
hone prepare all --repo ianncity/KIMI-K2.5-1000000x \
    --configs General-Distillation,PHD-Science,General-Math,MultilingualSTEM \
    --max-tokens 4096 \
    --output data/full/kimi/train.jsonl
```

Records longer than the cap are skipped at prepare time, before they
can produce empty loss targets after `--max-seq-length` truncation.

### Train (Apple Silicon, MLX)

```bash
hone train code
```

Reads `configs/code.yaml`, invokes `python -m hone.run` with
`HONE_DEVICE=gpu`, and writes the adapter to `artifacts/code-lora/`.

Before launching the multi-day full sequence, validate that a small
kimi subset trains without NaN losses:

```bash
./train.sh --layers 4 --seq-len 2048   # smoke training run
```

Or run the dedicated `configs/smoke-kimi.yaml` against an already-prepared
`data/full/kimi/train.jsonl` to confirm the data pipeline produced
clean records.

### Generate

```bash
hone generate prompt --adapter artifacts/code-lora \
    "Write a Python solution for two sum."
```

## Soup (one-YAML training pipeline)

hone ships a parallel pipeline that uses
[Soup](https://github.com/MakazhanAlpamys/Soup) (a single-config LLM
fine-tuner from the community) to train `openbmb/MiniCPM5-1B-MLX` on
the prepared CodeX corpus. Use it when you want Soup's data validation,
experiment tracking, and `soup ship` regression gate alongside hone's
existing data preparation. **Honest expectation: this gives you a
strong small-model coding SFT, not a leaderboard-topping model — see
[`docs/SOTA-EXPECTATIONS.md`](docs/SOTA-EXPECTATIONS.md) for the
measurable targets.**

```bash
uv pip install "soup-cli[mlx]"     # one-time: add Soup alongside hone
./train-soup.sh smoke              # 32 iters on 40 rows, ~30 s, validates the pipeline
./train-soup.sh full               # 1 epoch over 95k CodeX rows, ~70 min on M3 Pro 18 GB
./train-soup.sh gen "Write a Python function to compute factorial."  # prompt
./train-soup.sh export             # fuse LoRA into the base for deployment
./train-soup.sh ship               # `soup ship` regression gate (needs PyTorch)
```

| Concern | `hone` (this repo) | Soup (this section) |
|---|---|---|
| Config schema | hone YAML (`configs/code.yaml`) | Soup YAML (`configs/soup-sft-codex-*.yaml`) |
| Backend on M-series | MLX via `python -m hone.run` | MLX via `soup train --backend mlx` |
| Adapter resume | yes (multi-stage `train.sh`) | yes (`resume_from_checkpoint`) |
| Data format | chatml only | chatml / alpaca / sharegpt / DPO / KTO |
| Inference | `hone generate prompt` | `mlx_lm.generate` / `mlx_lm fuse` for deployment |
| Eval | `hone evaluate run` (LiveCodeBench) | `soup ship` (7-suite regression gate) |
| Smoke pattern | `./train.sh --layers 4 --seq-len 2048` | `./train-soup.sh smoke` |

The smoke run measures **~7.4 GB peak** on M3 Pro 18 GB at seq 2048
batch 1 with grad-checkpointing, so a full run on the same hardware
has ~10 GB of headroom and is safe to launch in the background.
A small `soup_mlx_compat.py` shim is installed at
`.venv/lib/python3.12/site-packages/` so MLX inference loads the
Llama tokenizer when transformers 4.57 cannot resolve
`TokenizersBackend` without PyTorch.

## Subcommands

```text
hone prepare      data preparation
  file           normalize and split a local JSONL
  code           reservoir-sample competitive-programming rows
  swe            build SWE-bench SFT rows (refuses non-train splits)
  all            materialize every row of an HF config
  evaluate       download LiveCodeBench prompts

hone train        train adapters
  code           coding adapter via mlx or cuda
  swe            SWE adapter via mlx or cuda
  all            full sequence across every dataset

hone generate     inference
  prompt         single-prompt generation
  file           bulk generation from a JSONL prompts file

hone tune         hyperparameter search
  run            Cartesian product trial runner

hone evaluate     evaluation
  run            LiveCodeBench evaluator (delegates to LCB)
```

## Configuration

Settings live in `hone.config` and are loaded from YAML files.
Every config value is read at process start and missing required
keys raise immediately.

Required keys (validated by `hone.config.validate`):

| Key | Description |
|---|---|
| `model` | HuggingFace model identifier (e.g. `openbmb/MiniCPM5-1B`) |
| `train` | Boolean; must be `true` for training |
| `data` | Directory containing `train.jsonl` and `valid.jsonl` |

Full config schema is the upstream `mlx_lm.lora` contract — see
[`docs/train.md`](docs/train.md) for an example.

## Environment

| Variable | Default | Effect |
|---|---|---|
| `HONE_DEVICE` | `gpu` | `gpu` or `cpu`; case-insensitive; validated at startup |

## Documentation

- [`docs/install.md`](docs/install.md) — installation, environment, extras
- [`docs/quickstart.md`](docs/quickstart.md) — first training run end-to-end
- [`docs/data.md`](docs/data.md) — data contract and preparation commands
- [`docs/train.md`](docs/train.md) — training commands and configs
- [`docs/eval.md`](docs/eval.md) — LiveCodeBench and SWE-bench evaluation
- [`docs/architecture.md`](docs/architecture.md) — package layout, public API, naming convention

## Project Structure

```
hone/
├── hone/                  # The library
│   ├── __init__.py        # Public API + __version__
│   ├── __main__.py        # python -m hone entry
│   ├── model.py           # Example, Message, Role, Scalar, Meta
│   ├── normalize.py       # Normalizer, SweNormalizer
│   ├── split.py           # Splitter + MIN_VALID
│   ├── jsonl.py           # Reader, Writer
│   ├── log.py             # setup, get, LOGGER
│   ├── config.py          # load, save, validate + REQUIRED_KEYS
│   ├── run.py             # MLX device launcher (HONE_DEVICE)
│   ├── types.py           # JsonScalar, JsonObject
│   └── cli/
│       ├── __init__.py    # typer dispatcher + main()
│       ├── prepare.py     # file, code, swe, all, evaluate
│       ├── train.py       # code, swe, all
│       ├── generate.py    # prompt, file, unfence
│       ├── tune.py        # run + TrialSpec, TrialResult
│       └── evaluate.py    # run
├── tests/
│   ├── unit/              # 52 behavior tests
│   ├── integration/       # 29 CLI integration tests
│   ├── property/          # 5 hypothesis property tests
│   └── mlx/               # 11 MLX-device tests (mark-gated)
├── configs/
│   ├── code.yaml
│   ├── swe.yaml
│   ├── tune-code.yaml
│   ├── tune-swe.yaml
│   ├── smoke.yaml
│   ├── smoke-kimi.yaml
│   ├── soup-sft-codex-smoke.yaml   # Soup MLX smoke (32 iters / 30 s)
│   └── soup-sft-codex-full.yaml    # Soup MLX full SFT (1 epoch / ~70 min)
├── docs/                  # install, quickstart, data, train, eval, architecture, SOTA-EXPECTATIONS
├── todo/                  # per-phase acceptance criteria
├── train-soup.sh          # Soup driver (smoke/full/gen/export/ship)
├── soup_mlx_compat.py     # AutoTokenizer Llama-fallback for MLX without PyTorch
├── .github/workflows/ci.yml
├── pyproject.toml
├── CHANGELOG.md
├── CONTRIBUTING.md
├── SECURITY.md
└── LICENSE
```

## Development

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[dev,mlx]'
```

Linting and formatting:

```bash
uv run pytest -m "not mlx"
uv run ruff check hone tests
uv run ruff format --check hone tests
uv run mypy hone tests
```

## Testing

```bash
uv run pytest                        # full suite (Apple Silicon)
uv run pytest -m "not mlx"           # Linux-compatible
uv run pytest tests/property/         # hypothesis property tests
uv run pytest tests/mlx/              # MLX-device tests (skipped on Linux)
```

Before a multi-day run, smoke-test the kimi data path:

```bash
HONE_DEVICE=gpu uv run hone train code --config configs/smoke-kimi.yaml --device gpu
```

Expect finite train loss and stable val loss within 50 iters; abort
the multi-day run if the smoke run produces `nan` or `0.000` losses.

The current collection size is reported by `pytest --collect-only`.
The suite covers every public API: behavior, edge cases, invalid
inputs, error paths, integration via `typer.testing.CliRunner`, and
property-based invariants (Writer/Reader round-trip, Splitter
disjoint + element-preservation, Normalizer validity).

## Tech Stack

| Category | Technology |
|---|---|
| Target platform | Apple Silicon (macOS, M-series) |
| Language | Python 3.12+ |
| Backend (Apple) | MLX-LM with LoRA |
| Backend (NVIDIA) | Unsloth with LoRA |
| CLI | Typer |
| Property tests | Hypothesis |
| Lint / format | ruff |
| Type check | mypy (strict) |
| Tests | pytest |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All contributions are
welcome; the maintainer (Sachin) reviews every PR.

## Security

Vulnerability reporting, supported versions, and the disclosure
timeline live in [SECURITY.md](SECURITY.md).

## Author

**Sachin** — author and maintainer. See
[github.com/sachncs](https://github.com/sachncs).

## License

[MIT](LICENSE) © 2026 Sachin
