# hone

A model- and dataset-agnostic supervised fine-tuning pipeline.

## Why hone?

The name is short on purpose: a single word, easy to type, easy to
remember, and evocative of *refining* — the work that fine-tuning does.
In this repo, `Example` means a training example; in your imports,
`from hone import Example` is unambiguous because the package
namespace disambiguates it.

## Quickstart

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[dev,mlx]'

hone prepare file --input data/raw.jsonl --output data/processed/code
hone train code
hone generate prompt "Write a Python solution for two sum."
```

## Documentation

- [install.md](docs/install.md) — installation, environment, extras
- [quickstart.md](docs/quickstart.md) — first training run end-to-end
- [data.md](docs/data.md) — data contract and preparation commands
- [train.md](docs/train.md) — training commands and configs
- [eval.md](docs/eval.md) — LiveCodeBench and SWE-bench evaluation
- [architecture.md](docs/architecture.md) — package layout, public API

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

## License

MIT — see [LICENSE](LICENSE).
