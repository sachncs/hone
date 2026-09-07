"""CLI subcommand group: generation.

Thin transport layer over :mod:`hone.generate` and :mod:`mlx_lm`.
The ``prompt`` and ``file`` commands load the model, build the
chat-formatted prompt via :func:`hone.generate.format_chat_prompt`,
and stream the response (single prompt) or batch it (JSONL file).
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from hone.generate import format_chat_prompt, lcb_user_prompt, unfence
from hone.log import setup

app: typer.Typer = typer.Typer(
    help="Generate text from a trained adapter.", no_args_is_help=True
)


def _load_model(model: str, adapter: str | None) -> tuple:
    """Load the MLX model + tokenizer; central seam for test patching."""
    from mlx_lm import load

    loaded = load(model, adapter_path=adapter, return_config=False)
    return loaded[0], loaded[1]


@app.command("prompt")
def prompt(
    prompt_text: str = typer.Argument(..., help="Prompt text."),
    model: str = typer.Option("openbmb/MiniCPM5-1B", "--model"),
    adapter: str | None = typer.Option(None, "--adapter", help="Adapter path."),
    max_tokens: int = typer.Option(1024, "--max-tokens"),
    temperature: float = typer.Option(0.2, "--temperature"),
) -> None:
    """Single-prompt generation."""
    from mlx_lm import generate
    from mlx_lm import sample_utils as sample

    model_obj, tokenizer = _load_model(model, adapter)
    formatted = format_chat_prompt(tokenizer, user_content=prompt_text)
    response = generate(
        model_obj,
        tokenizer,
        prompt=formatted,
        max_tokens=max_tokens,
        sampler=sample.make_sampler(temp=temperature),
        verbose=False,
    )
    typer.echo(response + "\n")


@app.command("file")
def file(
    input: str = typer.Option(..., "--input", help="JSONL of prompts."),
    output: str = typer.Option("artifacts/lcb_outputs.json", "--output"),
    model: str = typer.Option("openbmb/MiniCPM5-1B", "--model"),
    adapter: str | None = typer.Option(None, "--adapter"),
    samples: int = typer.Option(1, "--samples"),
    max_tokens: int = typer.Option(1536, "--max-tokens"),
    temperature: float = typer.Option(0.2, "--temperature"),
) -> None:
    """Bulk generation from a JSONL prompts file."""
    from mlx_lm import generate
    from mlx_lm import sample_utils as sample

    logger = setup(verbose=False)
    model_obj, tokenizer = _load_model(model, adapter)
    outputs: list[dict[str, object]] = []
    with Path(input).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            formatted = format_chat_prompt(
                tokenizer,
                user_content=lcb_user_prompt(row["question_content"]),
            )
            codes = []
            for _ in range(samples):
                text = generate(
                    model_obj,
                    tokenizer,
                    prompt=formatted,
                    max_tokens=max_tokens,
                    sampler=sample.make_sampler(temp=temperature),
                    verbose=False,
                )
                codes.append(unfence(text))
            outputs.append({"question_id": row["question_id"], "code_list": codes})
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("wrote %d questions to %s", len(outputs), output_path)


__all__ = ["app", "unfence"]
