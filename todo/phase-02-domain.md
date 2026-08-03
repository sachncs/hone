# Phase 2 — Domain layer

## Goal

Implement the domain layer (`hone/model.py`, `hone/normalize.py`,
`hone/split.py`, `hone/jsonl.py`, `hone/log.py`, `hone/config.py`)
with the renamed single-word public API. Eliminate all double-underscore
methods. Replace `JsonObject` duplications with the shared `hone.types`
aliases. Move library code away from `SystemExit` toward typed
exceptions.

## Standing acceptance criteria

See `todo/README.md`.

## Todos

### T2.1 — Implement `Role` enum in `hone/model.py`

StrEnum with lowercase members: `system`, `user`, `assistant`.

Acceptance:
- `from hone.model import Role` works.
- `Role.system`, `Role.user`, `Role.assistant` exist.
- `Role("system")` returns `Role.system`.
- `Role("invalid")` raises `ValueError`.
- `str(Role.user) == "user"`.
- Module docstring describes the type.

### T2.2 — Implement `Message` dataclass

Frozen, slots. Validates non-empty content, strips whitespace. Single
attribute `content` plus inherited `role`.

Acceptance:
- `Message(role=Role.user, content="hi")` works.
- `Message(role=Role.user, content="  ")` raises `ValueError`.
- `Message(role=Role.user, content=" hi ")` stores `"hi"` (stripped).
- `Message` is immutable (frozen): `message.content = "x"` raises
  `FrozenInstanceError` or `AttributeError`.
- `to_record()` returns `{"role": "user", "content": "hi"}`.
- Public methods only — no `__helper` style methods.

### T2.3 — Implement `Example` dataclass

Frozen, slots. Holds `messages: tuple[Message, ...]` and
`metadata: Meta`. Validates:
- At least 2 messages.
- Last message role is `assistant`.

Defensive copy of metadata to prevent external mutation.

Acceptance:
- `Example(messages=(u, a), metadata={...})` works.
- `Example(messages=(u,), metadata={...})` raises `ValueError`.
- `Example(messages=(u, u), metadata={...})` raises `ValueError`
  (last must be assistant).
- `Example(messages=(a, u), metadata={...})` raises `ValueError`
  (last must be assistant).
- Mutating the original metadata dict after construction does not
  change `Example.metadata`.
- `character_count` returns `sum(len(m.content) for m in messages)`.
- `to_record()` returns `{"messages": [...]}` with metadata merged.
- Public methods only — no `__helper` style methods.
- `to_record()` round-trips through `Reader.read` (covered in T8.8).

### T2.4 — Add `Scalar` type alias in `hone/model.py`

`Scalar: TypeAlias = str | int | float | bool | None`.

Acceptance:
- `from hone.model import Scalar` works.
- Module docstring explains the type.

### T2.5 — Add `Meta` type alias in `hone/model.py`

`Meta: TypeAlias = dict[str, Scalar]`.

Acceptance:
- `from hone.model import Meta` works.

### T2.6 — Implement `Normalizer` in `hone/normalize.py`

Public class. `normalize(record: Mapping[str, object]) -> Example`
method. Accepts:
- `{"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}`
- `{"prompt": "Q", "completion": "A"}`

Raises `ValueError` (not `SystemExit`) with field-path-prefixed
messages: `"messages[2].role: ..."`, not bare `"..."`.

Acceptance:
- `Normalizer().normalize({"messages": [...]})` returns an `Example`.
- `Normalizer().normalize({"prompt": "...", "completion": "..."})`
  returns an `Example` with `messages=(user, assistant)`.
- Empty messages list raises `ValueError` mentioning "messages".
- Non-list messages raises `ValueError`.
- Message without `role` raises `ValueError` mentioning "role".
- Message without `content` raises `ValueError` mentioning "content".
- Non-string role raises `ValueError`.
- Non-string content raises `ValueError`.
- The previously-private `__normalize_messages` is gone (no
  double-underscore method exists).
- Public surface area is `normalize` only (no `messages` helper if
  not needed by callers — KISS).

### T2.7 — Implement `SweNormalizer` in `hone/normalize.py`

Public class. `normalize(record)` returns an `Example` whose user
message contains repo, version, and problem statement; assistant
message is the patch.

Acceptance:
- Missing `problem_statement` raises `ValueError`.
- Missing `patch` raises `ValueError`.
- The prompt includes `Repository: <repo>` and `Version: <version>`.
- Metadata contains `instance_id` when provided.
- Empty `problem_statement` raises `ValueError`.
- Empty `patch` raises `ValueError`.
- Missing `repo` defaults to empty string in prompt.
- Missing `version` defaults to empty string in prompt.
- Library raises `ValueError`, not `SystemExit`.

### T2.8 — Implement `Splitter` in `hone/split.py`

Public class. Constructor takes `ratio: float` and `seed: int`.
`split(examples)` returns `(train, valid)`.

Acceptance:
- `Splitter(ratio=0.1, seed=42).split(examples)` returns disjoint
  `(train, valid)` lists.
- `ratio=0.0` raises `ValueError`.
- `ratio=1.0` raises `ValueError`.
- `ratio=-0.1` raises `ValueError`.
- `ratio=1.1` raises `ValueError`.
- Empty examples raises `ValueError`.
- Single example raises `ValueError`.
- Same seed produces identical split (determinism property).
- Different seeds produce different splits (in expectation; tested
  with seed 42 vs 43 on 10+ examples).
- `len(train) + len(valid) == len(examples)`.
- `set(train).isdisjoint(set(valid))`.
- The validation set has at least `MIN_VALID = 1` examples when
  `len(examples) >= 2` and `ratio > 0`.
- Module-level constant `MIN_VALID = 1` exists.
- The original example objects are preserved by reference (not
  deep-copied).

### T2.9 — Implement `Reader` in `hone/jsonl.py`

Public class. `read(path: Path) -> Iterator[Example]` yields parsed
`Example` objects. Raises `ValueError` (not `JSONDecodeError`) with
the line number prefix when a line is malformed.

Acceptance:
- `Reader().read(path)` yields valid `Example` objects.
- Blank lines are skipped silently.
- Malformed JSON line raises `ValueError` whose message contains
  the line number (e.g., `path:5: ...`).
- Non-dict JSONL record raises `ValueError` mentioning line number.
- Record without `messages` raises `ValueError` mentioning line
  number.
- Record with `messages` not a list raises `ValueError` mentioning
  line number.
- Message not a dict raises `ValueError` mentioning line number
  and index.
- Metadata fields are preserved (strings, numbers, bools, nulls).
- Non-scalar metadata values are filtered out (defensive).
- No double-underscore methods (the previous `__parse_record` is
  gone; if a helper is needed, it is named `record`).

### T2.10 — Implement `Writer` in `hone/jsonl.py`

Public class. `write(path: Path, examples: Iterable[Example]) -> int`
writes one JSON record per line, sorted keys, UTF-8, no `ensure_ascii`.

Acceptance:
- `Writer().write(path, examples)` writes one line per example.
- Returns the count of written records (int).
- Creates parent directories if missing.
- File is UTF-8 encoded.
- Unicode characters are not escaped (no `\uXXXX` for non-ASCII).
- Keys are sorted alphabetically in each line.
- A round-trip `write → read` produces equivalent `Example` objects
  (covered by T8.8 property test).

### T2.11 — Implement `setup` and `get` in `hone/log.py`

Public functions. `setup(verbose: bool = False) -> logging.Logger`
returns the configured package logger (idempotent). `get(name: str)
-> logging.Logger` returns a child logger.

Acceptance:
- `setup()` returns the package logger.
- Calling `setup()` twice does not duplicate handlers (idempotent).
- `setup(verbose=True)` sets DEBUG level.
- `setup(verbose=False)` sets INFO level.
- `get("test")` returns a logger with name `hone.test`.
- No `print(...)` used internally; only the logging module.

### T2.12 — Implement `load`, `validate`, `save` in `hone/config.py`

Public functions for YAML config I/O.

`load(path: Path) -> JsonObject`: read YAML, return as dict.
`validate(config: JsonObject) -> None`: raise `ValueError` if any
required key (`model`, `train`, `data`) is missing.
`save(path: Path, config: JsonObject) -> None`: write YAML.

Acceptance:
- `load(path)` returns a dict matching the YAML content.
- `load(missing_path)` raises `FileNotFoundError`.
- `validate({})` raises `ValueError` mentioning each missing key.
- `validate({"model": "x"})` raises `ValueError` (missing `train`,
  `data`).
- `validate({"model": "x", "train": True, "data": "y"})` returns
  None (no exception).
- `save(path, config)` writes valid YAML that round-trips through
  `load(path)`.
- Parent directories of `path` are created if missing.

### T2.13 — Update `hone/__init__.py` exports

Export the public API.

Acceptance:
- `from hone import Example, Message, Role, Scalar, Meta` works.
- `from hone import Normalizer, SweNormalizer, Splitter, Reader, Writer`
  works.
- `from hone import JsonObject, JsonScalar` works.
- `from hone import __version__` works and equals `"0.2.0"`.
- `__all__` lists all exported names.
- Underscore-prefixed names are not exported.

### T2.14 — Rename `tests/test_data_contracts.py`

Split into `tests/unit/test_model.py` and
`tests/unit/test_normalize.py` with updated import paths.

Acceptance:
- `tests/test_data_contracts.py` deleted.
- `tests/unit/test_model.py` exists and imports from `hone.model`.
- `tests/unit/test_normalize.py` exists and imports from
  `hone.normalize`.
- Test names use the new public-API names (`Message`, `Example`,
  `Normalizer`, `SweNormalizer`).

### T2.15 — Rename `tests/test_dataset_engineering.py`

Split into `tests/unit/test_split.py` and `tests/unit/test_jsonl.py`.

Acceptance:
- `tests/test_dataset_engineering.py` deleted.
- `tests/unit/test_split.py` imports `Splitter` from `hone.split`.
- `tests/unit/test_jsonl.py` imports `Reader`, `Writer` from
  `hone.jsonl`.

### T2.16 — Rename `tests/test_tuning.py`

Move to `tests/unit/test_tune.py` with updated imports.

Acceptance:
- `tests/test_tuning.py` deleted.
- `tests/unit/test_tune.py` imports the moved tuning logic from
  `hone.cli.tune` (forward to Phase 3 if the move target does not
  exist yet — but T3.13 creates the module, so the test should
  be updated once both phases complete).

### T2.17 — Run `pytest` and verify all existing tests pass

Acceptance:
- `uv run pytest` exits 0.
- Total test count equals the count of renamed tests (no test was
  silently dropped).

### T2.18 — Run `ruff check hone tests` and verify clean

Acceptance:
- Exits 0.
- No `E`, `F`, `I`, `UP`, `B`, `SIM`, `PL`, `RUF` violations.

### T2.19 — Run `ruff format --check hone tests` and verify clean

Acceptance:
- Exits 0.

### T2.20 — Run `mypy hone tests` and verify clean

Acceptance:
- Exits 0.
- No `[no-untyped-def]`, `[call-overload]`, or `[no-any-return]`
  violations in the new code.

## Success criteria for Phase 2

All 20 todos complete with their acceptance criteria met.

The standing quality gates all pass:

```bash
uv run pytest
uv run ruff check hone tests
uv run ruff format --check hone tests
uv run mypy hone tests
```

`from hone import Example, Message, Normalizer, Splitter` works.

No double-underscore methods exist anywhere under `hone/`.

No `SystemExit` calls remain in `hone/*.py` library code.

The three duplicate `type JsonObject = ...` definitions in `scripts/`
are gone (Phase 1 deletion covers `scripts/`; this is a confirmation).
