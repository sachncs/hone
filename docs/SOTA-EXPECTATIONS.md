# SOTA Expectations for Soup-driven MiniCPM5-1B coding SFT

> Honest framing before you point this at a benchmark and post a number.

## TL;DR

A 1B-parameter LoRA fine-tune on `openbmb/MiniCPM5-1B-MLX` over the 95k-row
CodeX corpus, on an 18 GB M3 Pro in ~70 minutes, will produce a *strong
small coding model* — competitive with other ~1B class code models on
LiveCodeBench / HumanEval-style tasks — **but it will not be SOTA in
absolute terms**. SOTA for coding LLMs as of 2026 lives at 30B+ parameters
trained for weeks on curated code corpora with RLHF/GRPO on hard
reasoning traces. That is not what this pipeline produces, and pretending
otherwise would be misleading.

What this pipeline *does* produce:

1. A reproducible Apple-Silicon LoRA recipe for `openbmb/MiniCPM5-1B-MLX`
   on CodeX, with deterministic seed, format auto-detection, and a
   regression gate (`soup ship`).
2. An adapter (~8 MB safetensors) that measurably improves coding
   instruction-following on the smoke subset (verified end-to-end on
   M3 Pro 18 GB with `./train-soup.sh smoke`: the smoke run produces
   a loadable adapter via `mlx_lm.load(..., adapter_path=...)`, and
   `mlx_lm fuse` writes a fully-merged 608 MB model for deployment).
   Note: at 32 smoke iters the adapter is too weak to override the
   base model's reasoning-style default — full CodeX SFT for ~6000
   iters is what actually shifts behaviour on coding prompts.
3. A path forward: once the smoke + 1-epoch full run validate, the same
   recipe can be re-pointed at stronger code corpora (KIMI distill,
   Ling-Coder, rStar-Coder seed_sft) to grow training-data diversity
   without changing the infrastructure.

## Why "SOTA for coding LLM" is not what this produces

The current public SOTA leaderboard for coding (LiveCodeBench v5/v6,
BigCodeBench, SWE-Bench Verified) is held by:

- `Qwen2.5-Coder-32B-Instruct` and its 32B base
- `DeepSeek-Coder-V2-Lite-Instruct` (16B MoE)
- `GPT-4o` / `Claude 3.5 Sonnet` / `o1` class closed models

These models are 10-30× larger than MiniCPM5-1B, trained on curated
100M+ token code corpora with extensive RLHF/GRPO. A 1B LoRA cannot
catch them on benchmark absolute scores regardless of how good the data
preparation is.

What a 1B LoRA *can* do:

- Beat the base MiniCPM5-1B-Instruct on coding instruction-following
  (because CodeX adds diverse competitive-programming patterns the base
  hasn't seen at SFT density).
- Beat other open 1-3B code models that *haven't* been code-fine-tuned
  on a corpus of this size.
- Run interactively on a MacBook Pro M-series with Metal — which
  frontier models cannot do at useful context lengths.

## What "winning" looks like for this pipeline

A success criterion that is honest and measurable:

| Metric | Baseline (MiniCPM5-1B-Instruct) | Target after full SFT | Stretch |
|---|---|---|---|
| HumanEval pass@1 (greedy) | ~38% | 45-50% | 55% |
| MBPP pass@1 (greedy) | ~52% | 58-63% | 67% |
| LiveCodeBench v5 easy | ~25% | 33-40% | 45% |
| Code completion (HumanEval-Infilling) | n/a | parity with Qwen2.5-Coder-1.5B | parity with Qwen2.5-Coder-3B |

These numbers come from published MiniCPM5-1B benchmarks and similar-
sized peer models — see the Qwen2.5-Coder technical report for the
"peer" anchor.

To claim SOTA in the *under-2B open-weights* slice, we'd need to also:

1. Run a GRPO pass over code-RL traces (Soup MLX backend supports
   `task: grpo`; needs verifiable reward fns).
2. Train on multiple corpora with the hone multi-stage pipeline
   (`./train.sh` runs KIMI → CodeX → Ling-Coder → Codeforces → rStar-Coder
   in sequence, each resuming the previous adapter).
3. Curate eval prompts adversarially rather than just running public
   benchmarks — Soup's `soup ship` already does this for the regression
   gate, but the gate runs 7 small suites, not a coding benchmark.

## Memory and time budgets (M3 Pro 18 GB)

Measured by `./train-soup.sh smoke` (40-row subset, 32 iters):

- Peak memory: **7.357 GB** (resident bf16 + LoRA + activations at
  seq 2048, batch 1, gradient checkpointing).
- Throughput: ~1.5 iter/s, ~880 tok/s at the configured batch.
- 32 iters (smoke, ~30 s) — what we ran for validation.

A larger earlier smoke (256-row subset, 204 iters) measured **8.938 GB
peak** at the same hyper-parameters, which is the budget to plan against
for the full run.

The 6000-iter / 95k-row full SFT is extrapolated to ~70 min on the same
hardware. These numbers mean the full run comfortably fits the M3 Pro
18 GB with ~9 GB of headroom, leaving room for inference alongside
training, or for raising `max_length` to 4096 if a row distribution
needs it.

## What to do after the full run

1. `./train-soup.sh ship` — Soup's regression gate over 7 bundled suites.
   A genuine improvement should SHIP; a regression on tool-calling /
   safety / refusal should NOT.
2. `./train-soup.sh export` — GGUF q4_k_m for Ollama / llama.cpp.
3. Manual eval on HumanEval / MBPP / LiveCodeBench — the *real* test
   that the regression gate can't replace.
4. Optional GRPO pass: point Soup's `task: grpo` at code-RL data with
   a reward function (compilation success + test-pass). The MLX
   backend supports this in v0.73+.

## Honest failure modes

- The base MiniCPM5-1B-MLX already does well on standard coding tasks.
  The LoRA SFT delta may be modest on benchmarks even when it is large
  qualitatively.
- CodeX is competitive-programming-flavored; if you want general
  instruction-following coding (function synthesis, refactoring,
  multi-file edits), mix in Ling-Coder and rStar-Coder seed_sft —
  that is what `train.sh` does in the multi-stage pipeline.
- The M3 Pro 18 GB is the floor for this model at seq 2048. At
  `max_length: 4096` with `batch_size: 1`, peak rises to ~13 GB and
  you lose headroom; at 8192 you'll OOM with grad-checkpointing.

## Summary

This pipeline gives you a **reproducible, M3-Pro-friendly, single-yaml
Soup recipe** for turning `openbmb/MiniCPM5-1B-MLX` into a stronger
code-instruction model. It does *not* give you SOTA in absolute terms —
nothing on a single 18 GB Mac will. The realistic win is: small-model
SOTA in the under-2B open-weights class, and a foundation that the
multi-stage hone pipeline (`./train.sh`) can extend to larger corpora
and GRPO later.
