# Phase 8 — Comprehensive tests

## Goal

Replace the existing weak tests with a comprehensive suite covering
every public API: behavior tests (every code path), edge cases,
invalid inputs, error paths, integration tests (CLI subcommands via
`typer.testing.CliRunner`), and property-based tests (`hypothesis`).
MLX tests are gated behind `pytest.importorskip("mlx.core")` and the
`mlx` marker.

## Standing acceptance criteria

See `todo/README.md`.

In addition, every test in this phase must satisfy:

- Mutation mindset: if the implementation is intentionally broken
  (e.g., off-by-one in a boundary), the test must fail.
- Behavior tests, not shape tests. No `assert x is not None` unless
  that is the literal contract.
- Property tests assert invariants that hold for all valid inputs,
  not just examples.
- Integration tests use `typer.testing.CliRunner` and check exit
  codes, stdout/stderr, and filesystem effects.

## Todos

### T8.1 — Delete `tests/test_data_contracts.py`

Replaced by `tests/unit/test_model.py` and
`tests/unit/test_normalize.py` (T8.5, T8.6).

Acceptance:
- File no longer exists.

### T8.2 — Delete `tests/test_dataset_engineering.py`

Replaced by `tests/unit/test_split.py` and
`tests/unit/test_jsonl.py` (T8.7, T8.8).

Acceptance:
- File no longer exists.

### T8.3 — Delete `tests/test_tuning.py`

Replaced by `tests/integration/test_tune.py` (T8.14).

Acceptance:
- File no longer exists.

### T8.4 — Move `tests/test_device_launcher.py` → `tests/mlx/test_run.py`

With renamed identifiers (Phase 4).

Acceptance:
- `tests/test_device_launcher.py` no longer exists.
- `tests/mlx/test_run.py` exists with all required mlx-test
  attributes.

### T8.5 — Create `tests/unit/test_model.py` (14 tests)

Required tests:

1. `test_role_accepts_system_user_assistant` — `Role("system") ==
   Role.system`, etc.
2. `test_role_rejects_unknown_value` — `Role("tool")` raises.
3. `test_role_serializes_to_string` — `str(Role.user) == "user"`.
4. `test_message_rejects_empty_content` — empty string raises.
5. `test_message_rejects_whitespace_only_content` — `"   "`
   raises.
6. `test_message_strips_surrounding_whitespace` — `" hi "` →
   stored as `"hi"`.
7. `test_message_preserves_internal_whitespace` — `"a  b"`
   preserved.
8. `test_message_is_immutable` — `message.content = "x"` raises
   `FrozenInstanceError` or `AttributeError`.
9. `test_example_requires_at_least_two_messages` — single-message
   raises.
10. `test_example_requires_assistant_last` — `(user, user)`
    raises.
11. `test_example_accepts_system_user_assistant` — three-message
    form works.
12. `test_example_character_count_sums_message_lengths` —
    arithmetic check.
13. `test_example_metadata_is_defensively_copied` — mutating the
    original dict does not change the example.
14. `test_example_to_record_roundtrip` — `to_record()` produces a
    dict that `Reader.record` (or equivalent) can re-parse.

Acceptance:
- 14 tests pass.
- All assertions check behavior (not shape).

### T8.6 — Create `tests/unit/test_normalize.py` (13 tests)

Required tests:

1. `test_normalize_prompt_completion_creates_user_assistant`
2. `test_normalize_messages_creates_message_list`
3. `test_normalize_messages_rejects_non_object`
4. `test_normalize_messages_rejects_missing_role`
5. `test_normalize_messages_rejects_missing_content`
6. `test_normalize_accepts_system_user_assistant`
7. `test_normalize_rejects_missing_prompt_and_messages`
8. `test_normalize_strips_surrounding_whitespace`
9. `test_swe_rejects_missing_problem_statement`
10. `test_swe_rejects_missing_patch`
11. `test_swe_includes_repo_and_version_in_prompt`
12. `test_swe_handles_missing_repo`
13. `test_swe_metadata_contains_instance_id`

Acceptance:
- 13 tests pass.
- Error messages are checked to mention the failing field.

### T8.7 — Create `tests/unit/test_split.py` (11 tests)

Required tests:

1. `test_splitter_rejects_ratio_zero`
2. `test_splitter_rejects_ratio_one`
3. `test_splitter_rejects_negative_ratio`
4. `test_splitter_rejects_ratio_above_one`
5. `test_splitter_rejects_single_example`
6. `test_splitter_rejects_empty_examples`
7. `test_splitter_same_seed_produces_same_split`
8. `test_splitter_different_seeds_produce_different_splits`
9. `test_splitter_validation_has_at_least_min_valid`
10. `test_splitter_returns_disjoint_sets`
11. `test_splitter_preserves_all_elements`

Acceptance:
- 11 tests pass.
- Determinism property proven by equality of two splits with the
  same seed.

### T8.8 — Create `tests/unit/test_jsonl.py` (12 tests)

Required tests:

1. `test_writer_roundtrips_messages`
2. `test_writer_roundtrips_metadata`
3. `test_writer_creates_parent_directories`
4. `test_writer_uses_utf8`
5. `test_writer_serializes_unicode`
6. `test_writer_returns_count`
7. `test_reader_reports_line_number_for_malformed_json`
8. `test_reader_skips_blank_lines`
9. `test_reader_rejects_non_dict_record`
10. `test_reader_rejects_missing_messages`
11. `test_reader_rejects_non_list_messages`
12. `test_reader_rejects_non_object_message_items`

Acceptance:
- 12 tests pass.
- Line-number assertions check that the error message includes
  the line number (mutation test: changing the line number in the
  error would still need the test to verify the line number is
  correct).

### T8.9 — Create `tests/unit/test_log.py` (4 tests)

1. `test_setup_returns_logger`
2. `test_setup_is_idempotent`
3. `test_setup_respects_verbose_flag`
4. `test_get_returns_named_logger`

Acceptance: 4 tests pass.

### T8.10 — Create `tests/unit/test_config.py` (7 tests)

1. `test_load_reads_yaml_file`
2. `test_load_raises_on_missing_file`
3. `test_validate_rejects_missing_model`
4. `test_validate_rejects_missing_train`
5. `test_validate_rejects_missing_data`
6. `test_save_writes_yaml_file`
7. `test_roundtrip_yaml`

Acceptance: 7 tests pass.

### T8.11 — Create `tests/integration/test_prepare.py` (10 tests)

Uses local JSONL fixtures (no HF download).

Required tests:

1. `test_prepare_file_creates_train_and_valid`
2. `test_prepare_file_respects_max_samples`
3. `test_prepare_file_respects_seed`
4. `test_prepare_file_deterministic_with_same_seed`
5. `test_prepare_swe_refuses_test_split`
6. `test_prepare_swe_filters_by_max_chars`
7. `test_prepare_code_reservoir_sampling_is_deterministic`
8. `test_prepare_code_filters_by_language`
9. `test_prepare_all_writes_valid_messages`
10. `test_prepare_evaluate_writes_lcb_prompts`

Acceptance:
- 10 tests pass.
- Tests use `typer.testing.CliRunner` and assert on exit codes,
  filesystem effects, and log output where appropriate.

### T8.12 — Create `tests/integration/test_train.py` (3 tests, mlx marker)

1. `test_train_code_runs_with_minimal_config`
2. `test_train_swe_runs_with_minimal_config`
3. `test_train_all_invokes_each_dataset_in_sequence`

Acceptance:
- 3 tests pass.
- All marked `@pytest.mark.mlx` and guarded by
  `pytest.importorskip("mlx.core")`.

### T8.13 — Create `tests/integration/test_generate.py` (3 tests)

1. `test_generate_prompt_runs_with_minimal_args`
2. `test_generate_file_reads_input_and_writes_output`
3. `test_generate_file_clean_strips_markdown_fences`

Acceptance: 3 tests pass.

### T8.14 — Create `tests/integration/test_tune.py` (3 tests)

1. `test_tune_runs_with_minimal_config`
2. `test_tune_writes_results_json`
3. `test_tune_writes_best_json`

Acceptance: 3 tests pass.

### T8.15 — Create `tests/integration/test_evaluate.py` (1 test)

1. `test_evaluate_runs_with_minimal_config` — LCB runner is mocked
   or stubbed; the test verifies the dispatch and file I/O.

Acceptance: 1 test passes.

### T8.16 — Create `tests/integration/test_pipeline.py` (3 tests, end-to-end)

1. `test_end_to_end_local_jsonl_to_trainable`
2. `test_splitter_after_normalize_roundtrips`
3. `test_writer_after_normalize_roundtrips`

Acceptance: 3 tests pass.

### T8.17 — Create `tests/property/test_roundtrip.py` (5 hypothesis tests)

1. `test_writer_reader_roundtrip_preserves_example`
   - For any valid `Example`, `read(write([example])) == [example]`.
2. `test_splitter_disjoint_for_random_examples`
   - For any sequence of ≥2 examples, `train ∩ valid == ∅`.
3. `test_splitter_preserves_element_count`
   - For any sequence of ≥2 examples, `len(train) + len(valid)
     == len(examples)`.
4. `test_normalize_produces_valid_example`
   - For any valid source record, `Normalizer().normalize(record)`
     passes `Example` invariants.
5. `test_message_normalize_messages_roundtrip`
   - For any valid `Message`, `Message.from_record(m.to_record())
     == m`.

Acceptance:
- 5 hypothesis tests pass.
- Each test uses `hypothesis.given(...)` with appropriate
  strategies; no `example()` short-circuits.
- Each test asserts an invariant, not an example-specific outcome.

### T8.18 — Create `tests/mlx/test_run.py` (10 tests)

10 tests covering device selection and the launcher module.
All `@pytest.mark.mlx` and guarded by
`pytest.importorskip("mlx.core")`.

Required tests:

1. `test_read_device_defaults_to_gpu`
2. `test_read_device_reads_env_case_insensitively`
3. `test_read_device_rejects_unsupported_value`
4. `test_setup_installs_cpu_when_requested`
5. `test_setup_installs_gpu_when_requested_and_metal_available`
6. `test_setup_refuses_gpu_when_metal_unavailable`
7. `test_setup_logs_device_info`
8. `test_run_runs_as_module` — `python -m hone.run --help`
9. `test_run_help_works`
10. `test_scripts_directory_is_gone` — regression test for T1.1

Acceptance:
- 10 tests pass on Apple Silicon.
- 10 tests skip cleanly on Linux (no failures).

## Success criteria for Phase 8

All 18 todos complete with their acceptance criteria met.

Total test count: ~100–120 tests across all files.

`pytest -m "not mlx"` exits 0 on Linux.

`pytest` exits 0 on Apple Silicon.

Every public API in `hone/__init__.py` has at least one behavior
test, one edge-case test, one invalid-input test, and one
error-path test.

At least one test in each module is a property test
(`hypothesis`) where round-trip or invariant properties apply.

CI workflows from Phase 6 pass on their respective platforms.
