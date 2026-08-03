# Model-agnostic supervised fine-tuning pipeline

MiniCPM5 is one example configuration in this repository. The pipeline is
model- and dataset-agnostic at its data-contract layer and is configured here
for an Apple M3 Pro, 18 GB
unified memory, macOS. The native backend is MLX-LM with LoRA. The regular
Unsloth Python/CUDA path is included as a fallback for an NVIDIA machine;
Unsloth Studio now has an MLX training route, but that is different from the
CUDA-oriented `FastLanguageModel` script in this repository.

## Quick start

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[dev,mlx]'

# Put solution-bearing coding JSONL at data/raw.jsonl, then create splits.
finetune-prepare-data --input data/raw.jsonl --output-dir data/processed/code --seed 42

# Train the coding adapter.
./scripts/train_stage.sh code
./scripts/generate.sh "Write a Python solution for two sum. Return code only."
```

Python 3.12 is intentional: it is broadly supported by the MLX and benchmark
tooling. The model is the Apple-Silicon conversion of the requested
`openbmb/MiniCPM5-1B`, published as `mlx-community/MiniCPM5-1B-4bit`.

## Apple-Silicon device selection

All MLX training entry points (`scripts/train_stage.sh`,
`scripts/train_full_sequence.sh`, `scripts/tune_all.sh`, and
`scripts/tune_mlx.py`) route through `scripts/run_mlx_lora.py`. The launcher
selects the MLX device explicitly, verifies Metal is available when GPU is
requested, and logs the active accelerator at the top of the training run:

```
INFO finetune: MLX device: Metal GPU (FINETUNE_DEVICE=gpu)
INFO finetune: Metal device: Apple M3 Pro, memory=19327352832 bytes, architecture=applegpu_g15s
```

Set `FINETUNE_DEVICE=cpu` to fall back to the CPU backend; the launcher will
refuse `FINETUNE_DEVICE=gpu` with a clear error if Metal is unavailable, so
the pipeline never silently degrades to CPU.

## Data contract

The input must be JSONL. Preferred format:

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

`prompt`/`completion` rows are also accepted. Keep LiveCodeBench and
SWE-bench evaluation instances out of `data/raw.jsonl`. Use dated or official
training splits, then reserve a local validation split for model selection.

For LiveCodeBench-style training, make the assistant response the complete
solution in the requested language. For SWE-style training, include the issue
and only the repository context the agent is allowed to see, with a unified
diff as the assistant response. Do not include the gold `patch` or `test_patch`
from evaluation instances.

## Staged training

For the complete sequential run requested by the project, with no row cap,
sampling, or validation split:

```bash
chmod +x scripts/train_full_sequence.sh scripts/prepare_full_hf.py
./scripts/train_full_sequence.sh
```

This materializes and trains the complete training splits of KIMI (all four
configs), CodeX, Ling-Coder, and the default Codeforces config. It uses every
Codeforces row as continued pretraining text because that dataset contains
problem/editorial data rather than solution completions. It will take a very
long time and require substantial disk space.

The M3 Pro configurations are conservative for 18 GB unified memory:

- 4-bit base model with LoRA
- 3,072 tokens for coding and 4,096 for SWE
- batch size 1
- 12 LoRA layers for coding, 8 for SWE
- gradient accumulation instead of increasing memory-heavy batch size
- rank 8
- gradient checkpointing
- prompt masking so loss focuses on the solution

Train coding first, then prepare and train SWE separately:

```bash
python scripts/prepare_swe_data.py \
  --dataset SWE-bench/SWE-bench \
  --split train \
  --output-dir data/processed/swe
./scripts/train_stage.sh swe
```

Keep the adapters separate until evaluation proves that merging them helps.
The SWE preparer intentionally refuses non-training splits because an SWE
evaluation patch is the answer.

Start with `--max-samples 32` and 20 iterations to validate the stack. Then
use several thousand high-quality examples and tune `iters` based on the
validation loss. Do not mix short competitive-programming examples and long
repository patches in one run until each task works independently.

## LiveCodeBench evaluation

Use separate virtual environments for training and evaluation if installing
vLLM. Run LiveCodeBench against a fixed release version and record the model
commit, release version, temperature, and sampling count. Use SWE-bench Lite
while developing and SWE-bench Verified only for the final report. SWE-bench
requires an agent/harness that can inspect repositories and produce patches;
plain text generation is not a valid substitute for the official evaluation.

LiveCodeBench must remain evaluation-only. Prepare a fixed release and run the
native MLX generator through the official custom evaluator:

```bash
git clone https://github.com/LiveCodeBench/LiveCodeBench.git ../LiveCodeBench
cd LiveCodeBench
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
cd ../finetune
./scripts/eval_lcb.sh
```

Use `VERSION=release_v2`, `VERSION=release_v6`, or another explicitly recorded
release. Set `SAMPLES=10` to match the official sampling setup. The runner
reports pass@1/pass@5 after executing generated code.

The pipeline does not claim SWE-bench SOTA from SFT alone. SWE-bench requires
an agent that can inspect a repository, edit files, run tests, recover from
failures, and return a patch. Use the trained SWE adapter inside such an agent
and evaluate first on Lite, then Verified.

## NVIDIA/Unsloth fallback

## Project architecture

This repository is an installable Python package. Files under `scripts/` are
thin orchestration entry points; data contracts and transformations live under
`src/finetune/`.

The data path is explicit: `TrainingExampleNormalizer` converts source records
into immutable `ChatMessage` and `TrainingExample` objects, `DatasetSplitter`
performs a seeded disjoint split, and `JsonlDatasetWriter`/`JsonlDatasetReader`
own the canonical JSONL persistence contract.

Run the same quality gates used in CI with:

```bash
pytest
ruff check src scripts tests
ruff format --check src scripts tests
mypy src scripts tests
```

Record the dataset identifiers, seed, validation ratio, Python version, and
dependency extras for every published experiment. Generated datasets, model
adapters, and virtual environments are intentionally excluded from source
control.

## Hyperparameter tuning

Run the reproducible code and SWE searches sequentially with one command:

```bash
MAX_TRIALS=24 ./scripts/tune_all.sh
```

Each trial varies learning rate, LoRA rank, trainable layers, sequence length,
gradient accumulation, and training iterations. Every trial gets its own YAML
configuration, adapter directory, training log, and result record under
`artifacts/tuning/`. The runner selects the lowest validation loss and writes
`best.json`.

For benchmark-driven selection, a benchmark command must write numeric metrics
to `$METRICS_PATH`, for example `{"livecodebench_pass_at_1": 0.42}`. Set
`CODE_OBJECTIVE=livecodebench_pass_at_1` or `SWE_OBJECTIVE=swe_resolved_rate`
and provide the corresponding command through `CODE_BENCHMARK_COMMAND` or
`SWE_BENCHMARK_COMMAND`. The command receives `$ADAPTER_PATH`, `$TRIAL_DIR`,
and `$METRICS_PATH`. This keeps official benchmark harnesses configurable and
does not pretend that plain text generation is official SWE-bench evaluation.

On a CUDA machine, install `.[cuda]` and run `scripts/train_unsloth.py` with
the same processed JSONL. The script follows the OpenBMB MiniCPM5 Unsloth
recipe. It is not used by default on this Mac.
