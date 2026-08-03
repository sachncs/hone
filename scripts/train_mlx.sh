#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f data/processed/code/train.jsonl ]]; then
  echo "Missing data/processed/code/train.jsonl. Run scripts/prepare_data.py first." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python}"
RUN_SCRIPT="$PWD/scripts/run_mlx_lora.py"
exec "$PYTHON_BIN" "$RUN_SCRIPT" --config configs/m3pro.yaml "$@"
