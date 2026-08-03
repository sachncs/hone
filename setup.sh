#!/usr/bin/env bash
# Bootstrap hone and verify the pipeline end-to-end:
#   1. create .venv
#   2. install hone with dev + mlx extras
#   3. run the full pytest suite
#   4. run a streamlined training job (configs/smoke.yaml)
#
# Env:
#   PYTHON_VERSION (default 3.12)
#   HONE_DEVICE    (default cpu)
#   HONE_SKIP_SMOKE=1   skip the smoke training job
#   HONE_SKIP_TESTS=1   skip pytest

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
HONE_DEVICE="${HONE_DEVICE:-gpu}"

if [[ ! -d .venv ]]; then
    echo "==> creating .venv (Python $PYTHON_VERSION)"
    uv venv --python "$PYTHON_VERSION"
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> installing hone [dev,mlx]"
uv pip install -e '.[dev,mlx]'

if [[ "${HONE_SKIP_TESTS:-0}" != "1" ]]; then
    echo "==> pytest (non-mlx)"
    uv run pytest -m "not mlx"
    echo "==> pytest (mlx; skips cleanly without Metal)"
    uv run pytest -m mlx
fi

if [[ "${HONE_SKIP_SMOKE:-0}" != "1" ]]; then
    echo "==> smoke training (configs/smoke.yaml, HONE_DEVICE=$HONE_DEVICE)"
    HONE_DEVICE="$HONE_DEVICE" uv run hone train code \
        --config configs/smoke.yaml \
        --device "$HONE_DEVICE"
fi

echo "==> done"
