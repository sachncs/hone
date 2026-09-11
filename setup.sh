#!/usr/bin/env bash
# Bootstrap hone (data-prep library) + Soup driver for Soup-first training.
#
# Steps:
#   1. create .venv
#   2. install hone + the prepare-layer dev extras
#   3. install soup-cli[mlx] + the AutoTokenizer Llama-fallback shim
#   4. run the pytest suite
#   5. (optional) run the Soup MLX smoke against an existing
#      data/full/codex/train.jsonl — validates the prepare pipeline
#      + the Soup driver end to end.
#
# Env:
#   PYTHON_VERSION (default 3.12)
#   HONE_SKIP_SOUP=1         skip installing soup-cli
#   HONE_SKIP_TESTS=1        skip pytest
#   HONE_SKIP_SOUP_SMOKE=1   skip the Soup smoke after bootstrap

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

if [[ ! -d .venv ]]; then
    echo "==> creating .venv (Python $PYTHON_VERSION)"
    uv venv --python "$PYTHON_VERSION"
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> installing hone [dev]"
uv pip install -e '.[dev]'

if [[ "${HONE_SKIP_SOUP:-0}" != "1" ]]; then
    echo "==> installing soup-cli[mlx] for the Soup driver"
    # Three pinned constraints (verified to coexist on M3 Pro 18 GB):
    #   * soup-cli==0.73.2 — current MLX-smoke-tested release
    #   * transformers>=4.57,<5 — mlx-lm 0.29 transitively requires 4.57
    #     and Soup refuses >=5; the upper bound is the Soup-imposed cap
    #   * huggingface-hub<1.0,>=0.34 — Soup's MLX trainer import-check
    # The same pins are documented in README.md Quick Start so the
    # README and setup.sh converge on the same dependency state.
    uv pip install \
        "soup-cli[mlx]==0.73.2" \
        "transformers>=4.57,<5" \
        "huggingface-hub<1.0,>=0.34"
    SITE_DIR="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    cp soup_mlx_compat.py "$SITE_DIR/soup_mlx_compat.py"
    printf 'import soup_mlx_compat\n' > "$SITE_DIR/soup_mlx_compat.pth"
fi

if [[ "${HONE_SKIP_TESTS:-0}" != "1" ]]; then
    echo "==> pytest"
    uv run pytest
fi

if [[ "${HONE_SKIP_SOUP_SMOKE:-0}" != "1" && "${HONE_SKIP_SOUP:-0}" != "1" ]]; then
    if [[ -f data/full/codex/train.jsonl ]]; then
        echo "==> Soup MLX smoke (configs/soup-sft-codex-smoke.yaml)"
        bash train-soup.sh smoke
    else
        echo "==> Soup MLX smoke skipped (data/full/codex/train.jsonl not found)"
        echo "    See docs/SOTA-EXPECTATIONS.md for the data prep workflow."
    fi
fi

echo "==> done"
