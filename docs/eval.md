# Evaluation

## LiveCodeBench

```bash
hone evaluate run --version release_v2 --samples 10
```

`hone evaluate` prepares LiveCodeBench prompts, generates
completions via `hone generate file`, and invokes the official
LiveCodeBench `custom_evaluator` subprocess.

Use `VERSION=release_v2`, `VERSION=release_v6`, or another
explicitly recorded release. Set `SAMPLES=10` to match the official
sampling setup.

## SWE-bench

SWE-bench is not a generation-only benchmark. The official
evaluator requires an agent that can inspect a repository, edit
files, run tests, recover from failures, and return a patch.
Plain-text generation is **not** a valid substitute for the
official evaluation.

Train the SWE adapter with `hone train swe`, then evaluate it
inside an agent harness against SWE-bench Lite (development)
and SWE-bench Verified (final report).

## Benchmark hooks for `hone tune`

`hone tune run --benchmark-command "<cmd>"` invokes `<cmd>` after
each successful trial with `ADAPTER_PATH`, `TRIAL_DIR`, and
`METRICS_PATH` in the environment. The command should write a
JSON object of numeric metrics to `$METRICS_PATH`, for example:

```json
{"livecodebench_pass_at_1": 0.42}
```

Select the trial by metric:

```bash
hone tune run --objective livecodebench_pass_at_1 ...
```
