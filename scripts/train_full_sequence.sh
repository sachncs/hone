#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python}"
RUN_SCRIPT="$PWD/scripts/run_mlx_lora.py"
MODEL="${MODEL:-mlx-community/MiniCPM5-1B-4bit}"
LAYERS="${LAYERS:-8}"
ACCUM="${ACCUM:-16}"
SEQ_LEN="${SEQ_LEN:-4096}"
SAVE_EVERY="${SAVE_EVERY:-100000}"

prepare() {
  local repo=$1 configs=$2 output=$3 mode=${4:-sft}
  [[ -f "$output" ]] || "$PYTHON_BIN" scripts/prepare_full_hf.py \
    --repo "$repo" --configs "$configs" --split train --mode "$mode" --output "$output"
}

train_stage() {
  local data=$1 adapter=$2 previous=${3:-}
  local iters
  iters=$(wc -l < "$data" | tr -d ' ')
  [[ "$iters" -gt 0 ]] || { echo "empty dataset: $data" >&2; exit 1; }
  local args=(--model "$MODEL" --train --data "$(dirname "$data")"
    --adapter-path "$adapter" --iters "$iters" --batch-size 1
    --grad-accumulation-steps "$ACCUM" --num-layers "$LAYERS"
    --max-seq-length "$SEQ_LEN" --save-every "$SAVE_EVERY" --seed 42)
  [[ "$data" == */codeforces/train.jsonl ]] || args+=(--mask-prompt)
  [[ -z "$previous" ]] || args+=(--resume-adapter-file "$previous")
  "$PYTHON_BIN" "$RUN_SCRIPT" "${args[@]}"
}

mkdir -p data/full artifacts/full
prepare ianncity/KIMI-K2.5-1000000x \
  General-Distillation,PHD-Science,General-Math,MultilingualSTEM \
  data/full/kimi/train.jsonl
train_stage data/full/kimi/train.jsonl artifacts/full/01-kimi

prepare Modotte/CodeX-7M-Non-Thinking default data/full/codex/train.jsonl
train_stage data/full/codex/train.jsonl artifacts/full/02-codex \
  artifacts/full/01-kimi/adapters.safetensors

prepare inclusionAI/Ling-Coder-SFT default data/full/ling/train.jsonl
train_stage data/full/ling/train.jsonl artifacts/full/03-ling \
  artifacts/full/02-codex/adapters.safetensors

# open-r1/codeforces has no solution-completion field. This is full-row
# continued pretraining on problem/editorial text, not supervised code SFT.
prepare open-r1/codeforces default data/full/codeforces/train.jsonl codeforces-text
train_stage data/full/codeforces/train.jsonl artifacts/full/04-codeforces \
  artifacts/full/03-ling/adapters.safetensors
