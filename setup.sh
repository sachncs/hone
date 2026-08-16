#!/usr/bin/env bash
# Bootstrap hone and verify the pipeline end-to-end:
#   1. create .venv
#   2. install hone with dev + mlx extras
#   3. install Soup [mlx] + the AutoTokenizer Llama-fallback shim
#   4. run the full pytest suite
#   5. run a streamlined training job (configs/smoke.yaml)
#   6. run the Soup smoke (configs/soup-sft-codex-smoke.yaml)
#
# Env:
#   PYTHON_VERSION (default 3.12)
#   HONE_DEVICE    (default cpu)
#   HONE_SKIP_SMOKE=1        skip the hone smoke training job
#   HONE_SKIP_SMOKE_KIMI=1   skip the kimi smoke training job
#   HONE_SKIP_SOUP_SMOKE=1   skip the Soup MLX smoke job
#   HONE_SKIP_TESTS=1        skip pytest
#   HONE_SKIP_SOUP=1         skip installing soup-cli

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

if [[ "${HONE_SKIP_SOUP:-0}" != "1" ]]; then
    echo "==> installing soup-cli[mlx] for the Soup driver"
    # Pin transformers <5 because Soup needs <5 and mlx-lm 0.29/0.31 wants
    # 4.57; both are happy with 4.57.6. The .pth shim below patches
    # AutoTokenizer so MLX inference loads the Llama tokenizer when
    # transformers 4.57 cannot resolve TokenizersBackend without PyTorch.
    uv pip install "soup-cli[mlx]" "transformers>=4.57,<5"
    SITE_DIR="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    cp soup_mlx_compat.py "$SITE_DIR/soup_mlx_compat.py"
    printf 'import soup_mlx_compat\n' > "$SITE_DIR/soup_mlx_compat.pth"
fi

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

if [[ "${HONE_SKIP_SMOKE_KIMI:-0}" != "1" && -f data/full/kimi/train.jsonl ]]; then
    echo "==> smoke-kimi (configs/smoke-kimi.yaml, HONE_DEVICE=$HONE_DEVICE)"
    HONE_DEVICE="$HONE_DEVICE" uv run hone train code \
        --config configs/smoke-kimi.yaml \
        --device "$HONE_DEVICE"
fi

if [[ "${HONE_SKIP_SOUP_SMOKE:-0}" != "1" && "${HONE_SKIP_SOUP:-0}" != "1" ]]; then
    if [[ -f data/full/codex/train.jsonl ]]; then
        echo "==> Soup MLX smoke (configs/soup-sft-codex-smoke.yaml)"
        bash train-soup.sh smoke
    else
        echo "==> Soup MLX smoke skipped (data/full/codex/train.jsonl not found)"
    fi
fi

echo "==> done"
