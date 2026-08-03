#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

MAX_TRIALS="${MAX_TRIALS:-24}"
DEVICE="${FINETUNE_DEVICE:-gpu}"
CODE_ARGS=(--objective "${CODE_OBJECTIVE:-validation_loss}")
SWE_ARGS=(--objective "${SWE_OBJECTIVE:-validation_loss}")
if [[ -n "${CODE_BENCHMARK_COMMAND:-}" ]]; then
  CODE_ARGS+=(--benchmark-command "$CODE_BENCHMARK_COMMAND")
fi
if [[ -n "${SWE_BENCHMARK_COMMAND:-}" ]]; then
  SWE_ARGS+=(--benchmark-command "$SWE_BENCHMARK_COMMAND")
fi

"${PYTHON_BIN:-$PWD/.venv/bin/python}" scripts/tune_mlx.py \
  --config configs/m3pro-code.yaml \
  --search-space configs/tuning-code.yaml \
  --output-dir artifacts/tuning/code \
  --device "$DEVICE" \
  --max-trials "$MAX_TRIALS" \
  "${CODE_ARGS[@]}"

"${PYTHON_BIN:-$PWD/.venv/bin/python}" scripts/tune_mlx.py \
  --config configs/m3pro-swe.yaml \
  --search-space configs/tuning-swe.yaml \
  --output-dir artifacts/tuning/swe \
  --device "$DEVICE" \
  --max-trials "$MAX_TRIALS" \
  "${SWE_ARGS[@]}"
