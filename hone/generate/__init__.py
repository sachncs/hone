"""Generation (inference) layer.

Pure-Python helpers around :mod:`mlx_lm` so the CLI module stays
a thin transport. Public surface:

* :func:`unfence` — strip optional markdown code fences from a
  generated source-code snippet.
* :func:`format_chat_prompt` — apply a tokenizer's chat template
  with the standard ``add_generation_prompt=True`` flag.
* :func:`lcb_user_prompt` — render a LiveCodeBench question into
  the standard user prompt used by ``hone generate file``.
"""

from hone.generate.prompt import format_chat_prompt, lcb_user_prompt, unfence

__all__ = ["format_chat_prompt", "lcb_user_prompt", "unfence"]
