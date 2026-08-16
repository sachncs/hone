"""Soup-ready JSONL data preparation.

Turns raw HuggingFace streams or local JSONL files into chat-format
JSONL files that :mod:`soup-cli` consumes directly.

Public surface:

* :func:`prepare_local_file` — normalize and split a local JSONL.
* :func:`prepare_reservoir_sample` — reservoir-sample an HF stream.
* :func:`prepare_swe` — build SWE-bench SFT rows.
* :func:`prepare_stream` — materialize every row of an HF config
  with optional token-length filtering.
* :func:`prepare_eval_prompts` — download LiveCodeBench prompts.

This package has no CLI of its own; call its functions directly or
through ``train-soup.sh`` / your own driver script.

Errors raised by these functions are subclasses of
:class:`PrepareError` (also exported here); the base class lets
callers catch every prepare-layer failure with a single
``except``.
"""

from hone.prepare.service import (
    DataError,
    PrepareError,
    PrepareRequest,
    PrepareResult,
    Role,
    ValidationError,
    prepare_eval_prompts,
    prepare_lingsard,
    prepare_local_file,
    prepare_reservoir_sample,
    prepare_stream,
    prepare_swe,
)

__all__ = [
    "DataError",
    "PrepareError",
    "PrepareRequest",
    "PrepareResult",
    "Role",
    "ValidationError",
    "prepare_eval_prompts",
    "prepare_lingsard",
    "prepare_local_file",
    "prepare_reservoir_sample",
    "prepare_stream",
    "prepare_swe",
]
