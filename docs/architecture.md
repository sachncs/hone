# Architecture

## Layout

```
hone/
  __init__.py        public API + __version__
  __main__.py        python -m hone entry point
  model.py           Example, Message, Role, Scalar, Meta
  normalize.py       Normalizer, SweNormalizer
  split.py           Splitter + MIN_VALID constant
  jsonl.py           Reader, Writer
  log.py             setup, get, LOGGER
  config.py          load, save, validate + REQUIRED_KEYS
  run.py             MLX device launcher (HONE_DEVICE)
  types.py           JsonObject, JsonScalar
  cli/
    __init__.py      typer dispatcher + main()
    prepare.py       file, code, swe, all, evaluate
    train.py         code, swe, all
    generate.py      prompt, file + strip_fences
    tune.py          run + TrialSpec, TrialResult
    evaluate.py      run

tests/
  unit/              behavior tests (no mlx needed)
  integration/       CLI tests via typer.testing.CliRunner
  property/          hypothesis round-trip + invariants
  mlx/               Apple-Silicon only
```

## Public API

```python
from hone import (
    Example, Message, Role, Scalar, Meta,
    Normalizer, SweNormalizer,
    Splitter, Reader, Writer,
    JsonObject, JsonScalar,
    REQUIRED_KEYS, MIN_VALID, LOGGER,
    setup, get, load, save, validate,
    __version__,
)
```

## Naming convention

Every identifier in the package is a single word. The
`hone.model` and `hone.cli.*` namespaces disambiguate terms
that would otherwise be too generic (`Example`, `Message`,
`Reader`, `Writer`, `Splitter`).

Single-underscore ("semi-private") identifiers are prohibited
by AGENTS.md. Module-level helpers are public; their docstrings
state "treat as internal" when appropriate.

## Device selection

Every training entry point sets `HONE_DEVICE` in the subprocess
environment and invokes `python -m hone.run`. The launcher
reads the variable, validates it, and refuses to start with
`gpu` when Metal is unavailable. The selected device is logged
at the top of every training run.

## Determinism

- `Splitter(ratio, seed)` produces an identical split for a
  given seed.
- `hone prepare code` uses `random.Random(seed)` for its
  reservoir sampler.
- All training randomness is controlled by the YAML config's
  `seed` field.
