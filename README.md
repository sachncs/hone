<p align="center">
  <h1 align="center">hone</h1>
  <p align="center">JSONL data preparation for Soup fine-tuning — one config, one command.</p>
  <p align="center">
    <a href="#installation"><img src="https://img.shields.io/badge/python-3.12%7C3.13-blue" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
    <a href="https://github.com/sachncs/hone/actions"><img src="https://img.shields.io/github/actions/workflow/status/sachncs/hone/ci.yml?branch=main" alt="CI"></a>
    <a href="#installation"><img src="https://img.shields.io/badge/hone-0.3.0-blue" alt="Version"></a>
    <a href="https://github.com/sachncs/hone/stargazers"><img src="https://img.shields.io/github/stars/sachncs/hone" alt="Stars"></a>
  </p>
</p>

**hone** prepares the JSONL files that
[Soup](https://github.com/MakazhanAlpamys/Soup) consumes for
supervised fine-tuning. The pipeline:

1. streams a HuggingFace dataset (or reads a local JSONL),
2. normalizes every row to chat format
   (`{"messages": [{"role", "content"}, ...]}`) or text format
   (`{"text": "..."}`),
3. drops rows over a configurable token-length cap so the
   trainer never sees an empty loss target after truncation,
4. reservoir-samples competitive-programming corpora, and
5. writes `train.jsonl` + `valid.jsonl` that Soup loads directly.

The training step itself is delegated to Soup. The original
`hone` MLX/Unsloth driver is preserved as a frozen alternative
in [`archive/`](archive/) — see
[`docs/ARCHIVE.md`](docs/ARCHIVE.md) for what it is and how to
restore it.

> **Author and maintainer**: Sachin

| Concern | Library |
|---|---|
| Data format | JSONL (`messages` or `prompt`/`completion`) |
| HF adapter | [`datasets`](https://huggingface.co/docs/datasets) |
| Trainer | [Soup](https://github.com/MakazhanAlpamys/Soup) — `pip install "soup-cli[mlx]"` |
| Optional alt trainer | frozen `archive/hone_mlx/` (MLX + Unsloth, see [`docs/ARCHIVE.md`](docs/ARCHIVE.md)) |
| Lint / format | ruff |
| Type check | mypy |
| Tests | pytest |

## Quick Start

### 1. Install

```bash
git clone https://github.com/sachncs/hone.git
cd hone
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[dev,soup]'
```

`setup.sh` automates this and runs the Soup smoke against an
existing `data/full/codex/train.jsonl`:

```bash
./setup.sh
```

### 2. Prepare a JSONL

Pick the function that matches your source data:

| Source | Function |
|---|---|
| Local JSONL | `hone.prepare.prepare_local_file` |
| HF competitive-programming corpus | `hone.prepare.prepare_reservoir_sample` |
| HF SWE-bench | `hone.prepare.prepare_swe` |
| Any HF dataset (chat or codeforces-text) | `hone.prepare.prepare_stream` |
| LiveCodeBench prompts (for eval) | `hone.prepare.prepare_eval_prompts` |

Example — stream CodeX, drop rows over 4096 tokens, write
`all.jsonl` under the directory:

```python
from pathlib import Path
import logging
from hone.prepare import prepare_stream, PrepareRequest

logging.basicConfig(level=logging.INFO)
request = PrepareRequest(output=Path("data/full/codex"), seed=42)
prepare_stream(
    request=request,
    repo="Modotte/CodeX-7M-Non-Thinking",
    configs="default",
    mode="sft",
    max_tokens=4096,
    tokenizer_model="openbmb/MiniCPM5-1B",
)
```

### 3. Train with Soup

```bash
uv run soup train --config configs/soup-sft-codex-full.yaml
```

Or use the wrapper:

```bash
./train-soup.sh smoke   # 32 iters / ~30 s, validates the pipeline
./train-soup.sh full    # 1 epoch over the 95k-row CodeX corpus, ~70 min on M3 Pro 18 GB
./train-soup.sh gen "Write a Python hello world"
./train-soup.sh export  # fuse LoRA into the base
./train-soup.sh ship    # `soup ship` regression gate (needs PyTorch)
```

The smoke run measures **~7.4 GB peak** on the M3 Pro 18 GB at
seq 2048 batch 1 with grad-checkpointing, so the full run has
~10 GB of headroom.

See [`configs/soup-sft-codex-smoke.yaml`](configs/soup-sft-codex-smoke.yaml)
and [`configs/soup-sft-codex-full.yaml`](configs/soup-sft-codex-full.yaml)
for the two training recipes.

## Python API

The package is small on purpose. Public surface:

```python
from hone.prepare import (
    prepare_local_file,    # normalize + split a local JSONL
    prepare_reservoir_sample,  # uniform-random sample of an HF stream
    prepare_swe,           # SWE-bench -> patch-completion JSONL
    prepare_stream,        # full HF config -> chat or text records
    prepare_eval_prompts,  # LiveCodeBench prompt dump

    PrepareRequest,        # common knobs (output, seed, logger)
    PrepareResult,         # summary (written / skipped / filtered_long)

    PrepareError, DataError, ValidationError,  # exception hierarchy
    Role,                  # chat-role enum
)
```

## How good will the model be?

A 1B-parameter LoRA on an 18 GB MacBook Pro cannot reach absolute
SOTA for coding — that lives at 30B+ params and multi-day multi-GPU
runs. What this pipeline produces is a **strong small-model
coding SFT** that:

- beats the base `openbmb/MiniCPM5-1B-Instruct` on HumanEval / MBPP /
  LiveCodeBench v5 (realistic under-2B targets documented in
  [`docs/SOTA-EXPECTATIONS.md`](docs/SOTA-EXPECTATIONS.md)),
- runs interactively on the M-series GPU where frontier models
  cannot,
- composes cleanly with `soup ship` for regression detection
  and `mlx_lm fuse` for deployment.

## Configuration

Soup owns the training config (`configs/soup-sft-codex-*.yaml`).
The prepare service takes:

| Argument | Effect |
|---|---|
| `output` | Directory to write `train.jsonl` (+ `valid.jsonl`); created if missing |
| `seed` | RNG seed for splits and reservoir sampling |
| `ratio` | Validation split ratio (exclusive 0..1) |
| `max_tokens` | Drop rows whose token count exceeds this; `0` disables |
| `max_samples` | Stop after this many rows are written; `0` means full pass |
| `tokenizer_model` | HF model id used for token-count filtering |

## Documentation

- [`docs/SOTA-EXPECTATIONS.md`](docs/SOTA-EXPECTATIONS.md) — what
  "strong small-model SFT" means, with measurable targets and
  honest failure modes.
- [`docs/ARCHIVE.md`](docs/ARCHIVE.md) — the frozen MLX / Unsloth
  driver and how to revive it.

## Project Structure

```
hone/
├── hone/
│   └── prepare/                # JSONL prep library
│       ├── __init__.py         # public surface
│       ├── service.py          # prepare_local_file, prepare_stream, ...
│       ├── mappers.py          # as_sft, as_codeforces_text
│       ├── reservoir.py        # Vitter-style uniform sampler
│       ├── token_filter.py     # tokenizer-based length filter
│       └── hf.py               # HubStream / load_split
├── archive/
│   └── hone_mlx/               # frozen MLX / Unsloth driver
├── tests/
│   └── test_prepare.py         # end-to-end tests for the prepare layer
├── configs/
│   ├── soup-sft-codex-smoke.yaml
│   └── soup-sft-codex-full.yaml
├── docs/
│   ├── SOTA-EXPECTATIONS.md
│   └── ARCHIVE.md
├── train-soup.sh               # Soup driver (smoke/full/gen/export/ship)
├── setup.sh                    # bootstrap: install + tests + soup smoke
├── soup_mlx_compat.py          # AutoTokenizer Llama-fallback shim
├── pyproject.toml
├── README.md
├── CHANGELOG.md
├── LICENSE
└── .gitignore
```

## License

[MIT](LICENSE) © 2026 Sachin
