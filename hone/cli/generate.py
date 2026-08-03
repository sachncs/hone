"""CLI subcommand group: generation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from hone.log import setup

app: typer.Typer = typer.Typer(
    help="Generate text from a trained adapter.", no_args_is_help=True
)


@app.command("prompt")
def prompt(
    prompt: str = typer.Argument(..., help="Prompt text."),
    model: str = typer.Option("openbmb/MiniCPM5-1B", "--model"),
    adapter: str | None = typer.Option(None, "--adapter", help="Adapter path."),
    max_tokens: int = typer.Option(1024, "--max-tokens"),
    temperature: float = typer.Option(0.2, "--temperature"),
) -> None:
    """Single-prompt generation."""
    from mlx_lm import generate, load
    from mlx_lm import sample_utils as utils

    loaded = load(model, adapter_path=adapter, return_config=False)
    model_obj, tokenizer = loaded[0], loaded[1]
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    sys.stdout.write(
        generate(
            model_obj,
            tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            sampler=utils.make_sampler(temp=temperature),
            verbose=False,
        )
        + "\n"
    )


def strip_fences(text: str) -> str:
    """Remove optional markdown fences from generated source code."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


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
    from mlx_lm import generate, load
    from mlx_lm import sample_utils as utils

    logger = setup(verbose=False)
    loaded = load(model, adapter_path=adapter, return_config=False)
    model_obj, tokenizer = loaded[0], loaded[1]
    outputs: list[dict[str, object]] = []
    with Path(input).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            formatted_prompt = tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": (
                            "Solve this competitive-programming problem. "
                            "Return only the complete solution code, with no "
                            "markdown fences.\n\n" + str(row["question_content"])
                        ),
                    }
                ],
                add_generation_prompt=True,
                enable_thinking=False,
            )
            codes = []
            for _ in range(samples):
                text = generate(
                    model_obj,
                    tokenizer,
                    prompt=formatted_prompt,
                    max_tokens=max_tokens,
                    sampler=utils.make_sampler(temp=temperature),
                    verbose=False,
                )
                codes.append(strip_fences(text))
            outputs.append({"question_id": row["question_id"], "code_list": codes})
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("wrote %d questions to %s", len(outputs), output_path)


__all__ = ["app", "strip_fences"]
