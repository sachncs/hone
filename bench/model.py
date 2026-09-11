"""MLX model wrapper used by every benchmark.

Loads the model + tokenizer + optional LoRA adapter once and
provides a uniform :meth:`generate` that returns a single
completion per prompt. The wrapper handles prompt templating via
the tokenizer's chat template; benchmarks only need to pass raw
user content.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationParams:
    """Hyperparameters that every benchmark pass through unchanged."""

    max_tokens: int = 512
    temperature: float = 0.0
    n_samples: int = 1
    """Number of completions per prompt; pass@10 needs ``n_samples=10``."""


class MlxCoder:
    """One loaded MLX model + tokenizer used for generation."""

    def __init__(
        self,
        *,
        model_id: str,
        adapter_path: Path | None = None,
    ) -> None:
        from mlx_lm import load

        log.info("loading model=%s adapter=%s", model_id, adapter_path or "(none)")
        loaded = load(
            model_id, adapter_path=str(adapter_path) if adapter_path else None
        )
        # mlx_lm.load returns (model, tokenizer) in v0.29/0.31
        self._model, self._tokenizer = loaded[0], loaded[1]
        self._model_id = model_id
        self._adapter_path = adapter_path

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def adapter_path(self) -> Path | None:
        return self._adapter_path

    def generate(
        self,
        prompts: Sequence[str],
        params: GenerationParams,
    ) -> list[list[str]]:
        """Return ``n_samples`` completions per prompt.

        Each completion is the raw generated text (no chat
        template wrapping); benchmarks compose the prompt
        upstream.
        """
        from mlx_lm import generate
        from mlx_lm import sample_utils as sample

        completions: list[list[str]] = []
        sampler = sample.make_sampler(temp=params.temperature)
        for prompt in prompts:
            samples: list[str] = []
            for _ in range(params.n_samples):
                text = generate(
                    self._model,
                    self._tokenizer,
                    prompt=prompt,
                    max_tokens=params.max_tokens,
                    sampler=sampler,
                    verbose=False,
                )
                samples.append(text)
            completions.append(samples)
        return completions


__all__ = ["GenerationParams", "MlxCoder"]
