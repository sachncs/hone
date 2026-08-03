# Quickstart

Train a coding adapter on a small JSONL.

## 1. Prepare a JSONL

Each line is a JSON object with a chat-style `messages` array:

```jsonl
{"messages":[{"role":"user","content":"Write a Python two-sum solution."},{"role":"assistant","content":"def two_sum(nums, target):\n    seen = {}\n    for i, n in enumerate(nums):\n        if target - n in seen:\n            return [seen[target - n], i]\n        seen[n] = i\n"}]}
```

`prompt`/`completion` rows are also accepted:

```jsonl
{"prompt":"Write a Python two-sum solution.","completion":"def two_sum(nums, target): ..."}
```

Save as `data/raw.jsonl`.

## 2. Split the data

```bash
hone prepare file --input data/raw.jsonl --output data/processed/code
```

This writes `train.jsonl` and `valid.jsonl` under the output
directory using a 5% validation split with seed 42 by default.

## 3. Train

```bash
hone train code
```

Reads `configs/code.yaml`, invokes `python -m hone.run` with
`HONE_DEVICE=gpu`, and writes the adapter to
`artifacts/code-lora/`.

## 4. Generate

```bash
hone generate prompt --adapter artifacts/code-lora \
    "Write a Python solution for two sum."
```
