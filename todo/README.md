# Refactor plan: `finetune-pipeline` → `hone`

This folder holds the atomic refactor plan, derived from a deep audit of the
existing repository against AGENTS.md (which is intentionally not tracked in
this repo; see `.gitignore`).

Every todo is independently verifiable. Each todo becomes a single commit
during execution. No todo is skipped, merged, or lowered in scope.

## Phase index

| Phase | File | Goal |
|---|---|---|
| 1 | [phase-01-skeleton.md](phase-01-skeleton.md) | Replace `src/finetune` and `scripts/` with a flat `hone/` package skeleton and updated `pyproject.toml`. |
| 2 | [phase-02-domain.md](phase-02-domain.md) | Implement the domain layer (`model`, `normalize`, `split`, `jsonl`, `log`, `config`, `types`) with renamed single-word public API and no double-underscore methods. |
| 3 | [phase-03-cli.md](phase-03-cli.md) | Build the `typer`-based CLI with sub-apps for `prepare`, `train`, `generate`, `tune`, `evaluate`. |
| 4 | [phase-04-launcher.md](phase-04-launcher.md) | Move the MLX device launcher into `hone/run.py`, rename identifiers, drop the misleading `sys.exit(run_lora())`, rename the env var to `HONE_DEVICE`. |
| 5 | [phase-05-configs.md](phase-05-configs.md) | Rename config files to model-agnostic names; delete dead `m3pro.yaml`. |
| 6 | [phase-06-ci.md](phase-06-ci.md) | Split CI into Linux (no-mlx) and Apple (full) jobs. |
| 7 | [phase-07-docs.md](phase-07-docs.md) | Rewrite `README.md`, add `docs/` split, `CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md`. |
| 8 | [phase-08-tests.md](phase-08-tests.md) | Add comprehensive behavior, integration, and property-based tests. |
| 9 | [phase-09-verify.md](phase-09-verify.md) | Cleanup, full quality gates, manual smoke tests. |

## Acceptance criteria (every todo)

These are the standing rules. They apply to every todo, not just to specific
phases. They are derived from AGENTS.md and are non-negotiable.

### Code quality

- Every public function, method, class, and attribute has complete type
  annotations. No `Any` unless documented with a justification.
- No `# type: ignore`, no `typing.cast`, no `assert False` for control flow.
- No `eval`, `exec`, `os.system`, or runtime monkey patching.
- No `print(...)` for user-facing output; use the package logger (`hone.log`).
- No single-underscore "semi-private" identifiers. Every name is either
  fully public or `__name__`-mangled via dataclass / `__post_init__` only.
- No double-underscore "name-mangled" methods on classes (no `__helper`).
- No forbidden names from AGENTS.md: `tmp`, `temp`, `foo`, `bar`, `baz`,
  `obj`, `var`, `item`, `misc`, `thing`, `manager`, `helper`, `utils`,
  `processor`, `handler`, `service`, `engine`. Renames are explicit.
- Single-word identifier names per package convention.
- Every magic number replaced with a named module-level constant.
- Functions fit within ~40 lines; long functions are decomposed.
- No duplication of logic. Duplicated type aliases, env-var handling, or
  JSONL writing belong in shared modules (`hone/types.py`, `hone/jsonl.py`).
- No global mutable state outside of dataclass slots.
- All errors raise specific exception types with actionable messages; no
  bare `raise SystemExit` from library code.
- All side-effecting functions log what they did via the package logger.

### Determinism

- Given identical inputs, every function produces identical outputs.
- All randomness uses an injected seed (`Splitter(seed=...)`, not module
  global). Tests prove determinism with the same seed.
- No hidden global state. No implicit ordering dependencies.

### Test quality

- Every assertion proves behavior, not just type or shape. `assert x is not None`
  is forbidden unless that is the literal contract.
- Every public API has at least one behavior test, one edge-case test, one
  invalid-input test, and one error-path test.
- Tests must fail under mutation: if the implementation is intentionally
  broken, the test must fail. (Author reviews each test against this rule.)
- Property tests (`hypothesis`) for round-trip and invariant properties.
- Boundary tests for every numeric boundary (0, 1, max, off-by-one).
- Invalid-input tests reject malformed data with clear error messages.
- Integration tests use `typer.testing.CliRunner` for CLI subcommands.
- MLX tests are marked `@pytest.mark.mlx` and guarded by
  `pytest.importorskip("mlx.core")` so they skip on Linux without error.

### Documentation

- Every public function, method, class has a Google-style docstring.
- Every docstring states preconditions, postconditions, and invariants where
  applicable (Design by Contract).
- Every public module has a one-line module docstring at the top.
- `README.md` and `docs/*.md` are kept in sync with code (no stale
  `scripts/...` references after Phase 1).

### Repository hygiene

- No TODO, FIXME, XXX, or HACK comments in committed code.
- No dead code. Functions or constants not referenced are deleted.
- No unused imports.
- Imports alphabetized within each group; standard library, third-party,
  local. Blank line between groups.
- `ruff check`, `ruff format --check`, `mypy`, `pytest` all pass at the end
  of every phase. No phase ends with a broken quality gate.

### Commit hygiene

- Every atomic todo is exactly one commit.
- Commit message starts with the todo ID: `T1.1: delete scripts/ directory`.
- Commit body briefly explains what changed and why (one sentence).
- Commits are atomic: each commit leaves the tree in a coherent state.
  If a todo requires staging multiple files, that is fine, but the commit
  message reflects the todo ID and a single intent.

## Standing quality gates

These run after every phase:

```bash
uv run pytest -m "not mlx"
uv run ruff check hone tests
uv run ruff format --check hone tests
uv run mypy hone tests
```

If any of these fail at the end of a phase, the phase is not complete.
