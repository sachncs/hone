"""Data preparation application layer.

Pure-Python orchestration for turning a HuggingFace dataset stream
(or a local JSONL file) into a deterministic chat-format JSONL
ready for the MLX trainer. The CLI in :mod:`hone.cli.prepare`
imports only from here; downstream layers never touch HF or the
tokenizer directly.

Public surface:

* :func:`prepare_local_file` — normalize and split a local JSONL.
* :func:`prepare_reservoir_sample` — reservoir-sample an HF stream.
* :func:`prepare_swe` — build SWE-bench SFT rows.
* :func:`prepare_stream` — materialize every row of an HF config
  with optional token-length filtering.
* :func:`prepare_eval_prompts` — download LiveCodeBench prompts.
"""

from hone.prepare.service import (
    PrepareRequest,
    PrepareResult,
    prepare_eval_prompts,
    prepare_local_file,
    prepare_reservoir_sample,
    prepare_stream,
    prepare_swe,
)

__all__ = [
    "PrepareRequest",
    "PrepareResult",
    "prepare_eval_prompts",
    "prepare_local_file",
    "prepare_reservoir_sample",
    "prepare_stream",
    "prepare_swe",
]
