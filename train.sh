#!/usr/bin/env bash
# Run the full end-to-end training pipeline across every dataset:
#
#   1. ianncity/KIMI-K2.5-1000000x   (general distillation + STEM)
#   2. Modotte/CodeX-7M-Non-Thinking (code SFT)
#   3. inclusionAI/Ling-Coder-SFT    (code SFT)
#   4. open-r1/codeforces            (codeforces problem text)
#   5. microsoft/rStar-Coder         (seed_sft; long-CoT code SFT)
#
# Each stage prepares its dataset (skipped if train.jsonl already
# exists) then trains a LoRA adapter, resuming from the previous
# stage's adapter.
#
# This is a multi-day run on Apple Silicon. Datasets total well
# over 100 GB on disk and the combined training is on the order
# of millions of iterations. Ensure ample free disk before
# starting; intermediate artifacts under artifacts/full/ are
# reused on resume.
#
# Recommended flags:
#   --max-tokens 4096     pass to 'hone prepare all' (via the data
#                         stage invoked inside train.sh) to drop
#                         records longer than the training window.
#                         Without this, sequences longer than
#                         --seq-len are truncated silently and can
#                         produce empty loss targets (NaN).
#   --seq-len 4096        caps attention cost. 8192 is the upstream
#                         default but quadruples attention time vs
#                         4096 and can blow past the M3 Pro's 18 GB
#                         unified memory with mixed-length batches.
#
# Usage:
#   ./train.sh                            # run all 5 stages with defaults
#   ./train.sh --layers 4 --seq-len 4096  # pass flags through
#
# Output:
#   artifacts/full/{01-kimi,02-codex,03-ling,04-codeforces,05-rstar}/
#       adapters.safetensors

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="artifacts/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train-$(date +%Y%m%d-%H%M%S).log"

echo "==> full training sequence"
echo "    log: $LOG_FILE"
echo "    recommendation: prepare with --max-tokens 4096 to avoid NaN losses"
echo

uv run --extra mlx hone train all "$@" 2>&1 | tee "$LOG_FILE"
