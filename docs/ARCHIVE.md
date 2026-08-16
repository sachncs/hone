# Archive — frozen hone MLX / Unsloth driver

The original `hone` package was an end-to-end supervised fine-tuning
pipeline with five `typer` subcommands (`prepare`, `train`, `generate`,
`tune`, `evaluate`), driven by `python -m hone.run` against `mlx_lm`'s
LoRA trainer on Apple Silicon (with a CUDA/Unsloth fallback on
NVIDIA).

It is **no longer the recommended path**. The Soup driver is now
the default; it handles training, evaluation, and deployment as
one YAML file with much less glue code.

Everything that made the original MLX driver work is preserved in
`archive/hone_mlx/` for reference or revival:

```
archive/hone_mlx/
├── __init__.py        # public API (Example, Message, Role, Splitter, ...)
├── __main__.py        # python -m hone entry
├── backends.py        # Backend / Launcher Protocols + MlxBackend
├── chat.py            # chat-format validator (parse_messages)
├── cli/               # typer commands (prepare, train, generate, ...)
├── config.py          # YAML loader / validator
├── errors.py          # domain exception hierarchy
├── generate/          # inference helpers (unfence, format_chat_prompt)
├── jsonl.py           # Reader, Writer
├── log.py             # logging setup
├── model.py           # Example, Message, Role, Meta, JsonScalar
├── normalize.py       # Normalizer, SweNormalizer
├── run.py             # python -m hone.run launcher (delegates to mlx_lm.lora)
├── split.py           # Splitter + Partitioner
├── train/             # training orchestration (Stage, StageMarker, ...)
├── tune/              # hyperparameter search (TrialRunner, ...)
├── hone.egg-info/     # old setuptools metadata
```

## What it is

This is a frozen snapshot. The code is the result of the August 2026
refactor (218 tests, ruff + mypy clean) — none of it has been
modified since. It is preserved for three reasons:

1. **Reference.** Operators learning the prepare / split / train
   patterns can read a working end-to-end pipeline instead of
   guessing from `hone.prepare` alone.
2. **Escape hatch.** A team that needs a custom training loop
   with multi-stage adapter resume (`hone.train.TrainOrchestrator`)
   or hyperparameter search (`hone.tune.TrialRunner`) can copy
   the archived modules back into a working tree.
3. **Benchmark.** A team that wants to compare Soup's
   `soup train --backend mlx` to a hand-rolled MLX pipeline can use
   this as the baseline.

## What it is NOT

- **Maintained.** There is no CI for it; there will be no new
  features. Bug reports against it will be closed with "use Soup".
- **Documented.** `docs/{install,quickstart,data,train,eval,
  architecture}.md` are gone. The docstrings inside the modules
  are the only documentation.
- **Installable.** `pyproject.toml` no longer declares
  `hone.cli` as a package; the `[mlx]` and `[cuda]` extras are
  gone. Installing the modern `hone` package does not pull these
  modules in.

## Restoring (for development)

If you want to revive the MLX driver for experimentation:

```bash
cp -r archive/hone_mlx/* hone/
# Add `cli` back to [tool.setuptools].packages in pyproject.toml
# Add `[mlx]` and `[cuda]` extras back to pyproject.toml
# Add `hone = "hone.cli:main"` to [project.scripts]
uv pip install -e '.[dev,mlx]'
hone train code
```

Be aware that none of this is tested against the current `main`
branch, and Soup is the path that gets new features.
