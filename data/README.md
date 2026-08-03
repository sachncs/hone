# Training data

Install the package from the repository root, then place a JSONL file at
`data/raw.jsonl` and run:

```bash
uv pip install -e '.[dev]'
finetune-prepare-data --input data/raw.jsonl --output-dir data/processed --seed 42
```

The canonical input contract is either `messages` or `prompt`/`completion`.
Every example must contain at least two non-empty messages and end with an
assistant message. The preparer creates disjoint `train.jsonl` and
`valid.jsonl` files deterministically from the recorded seed and validation
ratio.

Do not place LiveCodeBench or SWE-bench evaluation/test examples here.
