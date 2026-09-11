# Training data

`hone` ships a Python prepare library (`hone.prepare`) that
materializes the `train.jsonl` + `valid.jsonl` files Soup consumes.
Pick the function that matches your source data and call it
directly from a script or a notebook.

## Functions

| Source | Function |
|---|---|
| Local JSONL | `hone.prepare.prepare_local_file` |
| HF competitive-programming corpus | `hone.prepare.prepare_reservoir_sample` |
| HF SWE-bench | `hone.prepare.prepare_swe` |
| Any HF dataset (chat or codeforces-text) | `hone.prepare.prepare_stream` |
| Ling-Coder SFT | `hone.prepare.prepare_ling_coder` |
| Nemotron Competitive Programming + SWE | `hone.prepare.prepare_nemotron` |
| LiveCodeBench prompts (for eval) | `hone.prepare.prepare_eval_prompts` |

## Worked example — local JSONL to `train.jsonl` + `valid.jsonl`

```bash
uv pip install -e '.[dev]'
```

```python
import json
import logging
from pathlib import Path

from hone.prepare import prepare_local_file, PrepareRequest

logging.basicConfig(level=logging.INFO)

Path("data/raw.jsonl").write_text(
    "\n".join(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": f"Question {i}"},
                    {"role": "assistant", "content": f"Answer {i}"},
                ]
            }
        )
        for i in range(200)
    ),
    encoding="utf-8",
)

prepare_local_file(
    input_path=Path("data/raw.jsonl"),
    request=PrepareRequest(output=Path("data/processed"), seed=42),
    ratio=0.1,
)
```

The output lands under `data/processed/{train,valid}.jsonl`. The
prepare layer accepts either `{"messages": [...]}` or
`{"prompt": ..., "completion": ...}` rows; messages must have a
recognized `role` and non-empty `content`. See
`hone.prepare.service._parse_chat` for the validator.

Do not place LiveCodeBench or SWE-bench evaluation/test examples here.