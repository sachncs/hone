#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"

LCB_DIR=${LCB_DIR:-../LiveCodeBench}
VERSION=${VERSION:-release_v2}
SAMPLES=${SAMPLES:-1}
[[ -d "$LCB_DIR" ]] || { echo "Clone LiveCodeBench into $LCB_DIR first." >&2; exit 1; }
"$PYTHON_BIN" scripts/prepare_lcb_eval.py --version "$VERSION"
if [[ -n "${ADAPTER_PATH:-}" ]]; then
  "$PYTHON_BIN" scripts/generate_lcb.py --samples "$SAMPLES" --adapter-path "$ADAPTER_PATH"
else
  "$PYTHON_BIN" scripts/generate_lcb.py --samples "$SAMPLES"
fi
(cd "$LCB_DIR" && python -m lcb_runner.runner.custom_evaluator \
  --custom_output_file "$ROOT/artifacts/lcb_outputs.json")
