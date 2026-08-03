#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

stage=${1:?usage: scripts/train_stage.sh code|swe [mlx_lm options]}
shift
case "$stage" in
  code) config=configs/m3pro-code.yaml; data=data/processed/code/train.jsonl ;;
  swe)  config=configs/m3pro-swe.yaml;  data=data/processed/swe/train.jsonl ;;
  *) echo "stage must be code or swe" >&2; exit 2 ;;
esac
[[ -f "$data" ]] || { echo "Missing $data; prepare the dataset first." >&2; exit 1; }
PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python}"
RUN_SCRIPT="$PWD/scripts/run_mlx_lora.py"
[[ -f "$RUN_SCRIPT" ]] || { echo "Missing $RUN_SCRIPT" >&2; exit 1; }
exec "$PYTHON_BIN" "$RUN_SCRIPT" --config "$config" "$@"
