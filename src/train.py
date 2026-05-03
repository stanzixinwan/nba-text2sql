"""
train.py — Fine-tune T5/CodeT5/Flan-T5 on Spider for text-to-SQL.

Supports three methods:
  - full:  full parameter fine-tuning
  - lora:  LoRA (parameter-efficient)
  - qlora: QLoRA (4-bit quantized + LoRA)

Usage:
    python -m src.train --method full --model t5-base --epochs 3
    python -m src.train --method full --model google/flan-t5-base --epochs 10
    python -m src.train --method lora --model google/flan-t5-base --rank 16 --epochs 5
    python -m src.train --method lora --model google/flan-t5-large --rank 16 --epochs 5 --batch-size 4

Outputs go to models/<run_name>/.
"""

import argparse
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset

from src.data_utils import load_spider_splits


def tokenize_dataset(examples, tokenizer, max_input=1024, max_target=256):
    inputs = [ex["input_text"] for ex in examples]
    targets = [ex["target_text"] for ex in examples]
    model_inputs = tokenizer(inputs, max_length=max_input, truncation=True, padding=False)
    labels = tokenizer(text_target=targets, max_length=max_target,
                       truncation=True, padding=False)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def build_dataset(examples, tokenizer):
    ds = Dataset.from_list(examples)
    ds = ds.map(
        lambda batch: tokenize_dataset(
            [{"input_text": i, "target_text": t}
             for i, t in zip(batch["input_text"], batch["target_text"])],
            tokenizer,
        ),
        batched=True,
        remove_columns=ds.column_names,
    )
    return ds


def setup_model(model_name: str, method: str, lora_rank: int = 16):
    """Load base model and optionally wrap with PEFT.
    Uses AutoModelForSeq2SeqLM so it works for T5, Flan-T5, CodeT5 alike."""
    if method == "qlora":
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, quantization_config=bnb, device_map="auto"
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if method in ("lora", "qlora"):
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
        if method == "qlora":
            model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.1,
            # T5/Flan-T5 attention + FFN projections
            target_modules=["q", "k", "v", "o", "wi", "wo"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="t5-base",
                        help="HF model id, e.g. t5-base, google/flan-t5-base, "
                             "google/flan-t5-large, Salesforce/codet5-base")
    parser.add_argument("--method", choices=["full", "lora", "qlora"], required=True)
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    run_name = args.run_name or f"{args.method}_{args.model.split('/')[-1]}"
    if args.method == "lora":
        run_name += f"_r{args.rank}"
    output_dir = f"models/{run_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Run: {run_name}")
    print(f"Output: {output_dir}\n")

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print("Loading Spider data...")
    train, dev = load_spider_splits(max_examples=args.max_train)
    print(f"  Train: {len(train)} | Dev: {len(dev)}")

    print("Tokenizing...")
    train_ds = build_dataset(train, tokenizer)
    dev_ds = build_dataset(dev[:500], tokenizer)

    print(f"Setting up {args.method} model...")
    model = setup_model(args.model, args.method, args.rank)

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=100,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=False,
        bf16=torch.cuda.is_available() and args.method != "qlora",
        fp16=False,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        data_collator=collator,
    )

    print("Training...")
    trainer.train()

    print(f"\nSaving final model to {output_dir}/final")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    print("Done.")


if __name__ == "__main__":
    main()