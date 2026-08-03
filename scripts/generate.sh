#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

prompt=${*:?usage: scripts/generate.sh 'your prompt'}
exec "${PYTHON_BIN:-$PWD/.venv/bin/python}" scripts/generate_mlx.py \
  "$prompt" \
  --model "${MODEL:-mlx-community/MiniCPM5-1B-4bit}" \
  --adapter-path "${ADAPTER_PATH:-artifacts/code-lora}" \
  --max-tokens "${MAX_TOKENS:-1024}" \
  --temperature "${TEMPERATURE:-0.2}"
