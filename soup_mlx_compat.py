"""Compatibility shim for mlx-lm 0.29 / 0.31 + transformers 4.57+.

When the venv has neither PyTorch nor a backend-capable transformers (>=5.x
or >=4.58), ``AutoTokenizer.from_pretrained`` raises
``ValueError: Tokenizer class TokenizersBackend does not exist or is not
currently imported.`` even though the underlying tokenizer loads fine.

This module patches ``AutoTokenizer.from_pretrained`` to fall back to
``LlamaTokenizerFast`` when the registry lookup fails. It is harmless when
a working backend IS installed (the try/except passes through).

Import once at process start:
    import soup_mlx_compat  # noqa: F401
"""
from __future__ import annotations

# The patching is a no-op when transformers can resolve TokenizersBackend
# itself; we only kick in on the specific failure mode.
try:
    from transformers import AutoTokenizer, LlamaTokenizerFast

    _orig_from_pretrained = AutoTokenizer.from_pretrained.__func__

    def _patched_from_pretrained(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
        try:
            return _orig_from_pretrained(cls, *args, **kwargs)
        except (ValueError, ImportError, AttributeError) as exc:
            msg = str(exc)
            if "TokenizersBackend" not in msg:
                raise
            # Fall back: the MiniCPM / Llama tokenizer config advertises
            # TokenizersBackend but transformers 4.57 does not register
            # it without a PyTorch backend. LlamaTokenizerFast covers
            # every Llama-architecture tokenizer we care about here.
            return LlamaTokenizerFast.from_pretrained(*args, **kwargs)

    AutoTokenizer.from_pretrained = classmethod(_patched_from_pretrained)
except Exception:  # pragma: no cover
    # transformers not installed yet — leave the import alone; downstream
    # callers will surface the real error.
    pass
