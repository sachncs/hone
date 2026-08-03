"""CUDA fallback matching the OpenBMB/Unsloth MiniCPM5 recipe."""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openbmb/MiniCPM5-1B")
    parser.add_argument("--data", default="data/processed/code/train.jsonl")
    parser.add_argument("--output", default="artifacts/minicpm5-unsloth")
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_length,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        full_finetuning=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    rows = [
        json.loads(x) for x in Path(args.data).read_text().splitlines() if x.strip()
    ]
    dataset = Dataset.from_list(
        [
            {"text": tokenizer.apply_chat_template(x["messages"], tokenize=False)}
            for x in rows
        ]
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=args.output,
            dataset_text_field="text",
            max_length=args.max_length,
            num_train_epochs=2,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            learning_rate=2e-4,
            warmup_ratio=0.03,
            bf16=True,
            logging_steps=10,
            save_steps=200,
            report_to="none",
        ),
    )
    trainer.train()
    trainer.model.save_pretrained(f"{args.output}/adapter")
    tokenizer.save_pretrained(f"{args.output}/adapter")


if __name__ == "__main__":
    main()
