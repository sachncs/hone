#!/usr/bin/env bash
# Soup-driven MiniCPM5-1B SFT for coding (Apple Silicon, MLX backend).
#
# Usage:
#   ./train-soup.sh                       # default: smoke on CodeX
#   ./train-soup.sh smoke                 # 32 iters, ~30s, validates CodeX path
#   ./train-soup.sh smoke-ling            # 50 iters on Ling-Coder (different dist)
#   ./train-soup.sh full                  # full CodeX run, ~70 min on M3 Pro 18 GB
#   ./train-soup.sh full-combined         # full CodeX+Ling-Coder, ~100 min
#   ./train-soup.sh gen "prompt"          # single-prompt generation via mlx_lm
#   ./train-soup.sh export                # fuse LoRA into base for deployment
#   ./train-soup.sh ship                  # `soup ship` regression gate (needs PyTorch)
#
# Env:
#   SOUP_CONFIG_FULL          — override the full-run config path
#   SOUP_CONFIG_SMOKE         — override the smoke config path
#   SOUP_OUTPUT               — override the output dir (default artifacts/soup-codex-full)
#   SOUP_BASE                 — override the base model (default openbmb/MiniCPM5-1B-MLX)
#   SOUP_SKIP_SMOKE=1         — skip the smoke validation step before full
#   SOUP_DATA_DIR             — override the data dir used by `data` subcommand

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source .venv/bin/activate

ACTION="${1:-smoke}"
SOUP_CONFIG_SMOKE="${SOUP_CONFIG_SMOKE:-configs/soup-sft-codex-smoke.yaml}"
SOUP_CONFIG_SMOKE_LING="${SOUP_CONFIG_SMOKE_LING:-configs/soup-sft-lingcoder-smoke.yaml}"
SOUP_CONFIG_FULL="${SOUP_CONFIG_FULL:-configs/soup-sft-codex-full.yaml}"
SOUP_CONFIG_FULL_COMBINED="${SOUP_CONFIG_FULL_COMBINED:-configs/soup-sft-codex-lingcoder-full.yaml}"
SOUP_OUTPUT="${SOUP_OUTPUT:-artifacts/soup-codex-full}"
SOUP_BASE="${SOUP_BASE:-openbmb/MiniCPM5-1B-MLX}"

LOG_DIR="artifacts/logs"
mkdir -p "$LOG_DIR"
ts="$(date +%Y%m%d-%H%M%S)"

# Soup needs a HF token for higher rate limits (the smoke run warned about
# this). Export if available; otherwise Soup will still work, just slower.
if [[ -z "${HF_TOKEN:-}" && -f ~/.cache/huggingface/token ]]; then
    export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
fi

case "$ACTION" in
    smoke)
        echo "==> Soup MLX smoke (MiniCPM5-1B-MLX, ~32 iters on data/soup/codex/smoke-train-small.jsonl)"
        uv run soup train --config "$SOUP_CONFIG_SMOKE" --yes \
            2>&1 | tee "$LOG_DIR/soup-smoke-$ts.log"
        echo
        echo "==> smoke OK; adapter at artifacts/soup-codex-smoke/adapters.safetensors"
        echo "    validate with: ./train-soup.sh gen 'Write a Python hello world'"
        ;;
    smoke-ling)
        echo "==> Soup MLX Ling-Coder smoke (50 iters on data/soup/lingcoder/train.jsonl)"
        uv run soup train --config "$SOUP_CONFIG_SMOKE_LING" --yes \
            2>&1 | tee "$LOG_DIR/soup-smoke-ling-$ts.log"
        echo
        echo "==> Ling-Coder smoke OK; adapter at artifacts/soup-lingcoder-smoke/adapters.safetensors"
        ;;
    full)
        if [[ "${SOUP_SKIP_SMOKE:-0}" != "1" ]]; then
            echo "==> running smoke validation first (set SOUP_SKIP_SMOKE=1 to skip)"
            "$0" smoke
        fi
        echo "==> Soup MLX full SFT (MiniCPM5-1B-MLX on full CodeX, ~6000 iters)"
        echo "    log: $LOG_DIR/soup-full-$ts.log"
        echo "    expected: ~70 min on M3 Pro 18 GB, ~8 MB adapter"
        uv run soup train --config "$SOUP_CONFIG_FULL" --yes \
            2>&1 | tee "$LOG_DIR/soup-full-$ts.log"
        echo
        echo "==> full SFT done; adapter at $SOUP_OUTPUT/adapters.safetensors"
        echo "    next: ./train-soup.sh gen '...'    # prompt"
        echo "          ./train-soup.sh export       # fuse + merge"
        echo "          ./train-soup.sh ship         # regression gate (needs PyTorch)"
        ;;
    full-combined)
        if [[ "${SOUP_SKIP_SMOKE:-0}" != "1" ]]; then
            echo "==> running smoke validation first (set SOUP_SKIP_SMOKE=1 to skip)"
            "$0" smoke-ling
        fi
        echo "==> Soup MLX full SFT (MiniCPM5-1B-MLX on CodeX + Ling-Coder, 5k-row stratified subset)"
        echo "    log: $LOG_DIR/soup-full-combined-$ts.log"
        echo "    expected: ~80 min on M3 Pro 18 GB, ~8 MB adapter"
        SOUP_OUTPUT="artifacts/soup-combined-5k-full" \
            uv run soup train --config configs/soup-sft-combined-5k-full.yaml --yes \
            2>&1 | tee "$LOG_DIR/soup-full-combined-$ts.log"
        echo
        echo "==> combined SFT done; adapter at artifacts/soup-combined-5k-full/adapters.safetensors"
        echo "    next: ./train-soup.sh gen '...'    # prompt"
        echo "          ./train-soup.sh export       # fuse + merge"
        echo "          ./train-soup.sh ship         # regression gate (needs PyTorch)"
        ;;
    gen)
        # NOTE: `soup infer` and `soup chat` go through the transformers
        # backend, which needs PyTorch — that contradicts the MLX
        # backend we trained with. On Apple Silicon the right inference
        # path is direct `mlx_lm.generate`.
        ADAPTER="${SOUP_OUTPUT:-artifacts/soup-codex-smoke}"
        if [[ ! -f "$ADAPTER/adapters.safetensors" ]]; then
            ADAPTER="artifacts/soup-codex-smoke"
        fi
        if [[ "${2:-}" == "-h" || "${2:-}" == "--help" || -z "${2:-}" ]]; then
            echo "Usage: $0 gen \"PROMPT\" [MAX_TOKENS=200]" >&2
            exit 2
        fi
        PROMPT="$2"
        MAX_TOKENS="${3:-${SOUP_MAX_TOKENS:-200}}"
        echo "==> mlx_lm.generate on $SOUP_BASE + $ADAPTER  (max_tokens=$MAX_TOKENS)"
        exec uv run --extra mlx python -m mlx_lm.generate \
            --model "$SOUP_BASE" \
            --adapter-path "$ADAPTER" \
            --max-tokens "$MAX_TOKENS" \
            --prompt "$PROMPT"
        ;;
    export)
        # Soup's `soup export` requires PyTorch (transformers-based
        # export path). On MLX adapters, the right deployment path is
        # to fuse the LoRA into the base via mlx_lm.fuse, then export
        # GGUF with llama.cpp's convert_hf_to_gguf.py.
        ADAPTER="${SOUP_OUTPUT:-artifacts/soup-codex-smoke}"
        if [[ ! -f "$ADAPTER/adapters.safetensors" ]]; then
            ADAPTER="artifacts/soup-codex-smoke"
        fi
        MERGED="${SOUP_MERGED:-$ADAPTER/merged}"
        echo "==> fusing LoRA into base via mlx_lm fuse"
        uv run --extra mlx python -m mlx_lm fuse \
            --model "$SOUP_BASE" \
            --adapter-path "$ADAPTER" \
            --save-path "$MERGED"
        echo
        echo "==> merged model at $MERGED"
        echo "    next: clone llama.cpp and run convert_hf_to_gguf.py for GGUF."
        echo "    see https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py"
        ;;
    ship)
        # `soup ship` runs an extraction-based scorer over 7 bundled
        # offline suites (MCQ · arithmetic · tool-calling · JSON
        # validity · safety/refusal · benign-prompt over-refusal mirror ·
        # noise floor) — it needs PyTorch because the eval goes through
        # the transformers backend. We try a dry-run first to confirm
        # the adapter + base are detectable, and tell the user how to
        # actually run the gate.
        echo "==> soup ship dry-run (gate itself requires PyTorch)"
        uv run soup ship \
            --base "$SOUP_BASE" \
            --adapter "${SOUP_OUTPUT:-artifacts/soup-codex-smoke}" \
            --dry-run 2>&1 | tail -30 || true
        echo
        echo "==> for the real gate, install torch and rerun:"
        echo "       pip install 'soup-cli[train]'"
        echo "       ./train-soup.sh ship"
        echo "    see docs/SOTA-EXPECTATIONS.md for what 'SHIP' means on a 1B"
        ;;
    *)
        echo "Unknown action: $ACTION" >&2
        echo "Usage: $0 [smoke|smoke-ling|full|full-combined|gen|export|ship]" >&2
        exit 2
        ;;
esac
