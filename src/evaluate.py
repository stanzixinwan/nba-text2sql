"""
evaluate.py — Evaluate a fine-tuned checkpoint on NBA and/or Spider dev.

Usage:
    python -m src.evaluate --checkpoint models/lora_t5-base_r16/final --eval nba
    python -m src.evaluate --checkpoint models/lora_t5-base_r16/final --eval spider --max-examples 500
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm

from src.data_utils import load_nba_dataset, load_spider_splits
from src.prompt_baseline import execution_accuracy, exact_match, generate_sql


def load_checkpoint(checkpoint_path: str, base_model: str = "t5-base"):
    """
    Load a checkpoint. Auto-detects whether it's a full fine-tune or PEFT/LoRA.
    """
    ckpt = Path(checkpoint_path)
    is_peft = (ckpt / "adapter_config.json").exists()

    tokenizer = T5Tokenizer.from_pretrained(checkpoint_path if not is_peft else base_model)

    if is_peft:
        from peft import PeftModel
        print(f"Loading PEFT adapter from {checkpoint_path} on top of {base_model}")
        model = T5ForConditionalGeneration.from_pretrained(base_model)
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()  # merge LoRA weights for faster inference
    else:
        print(f"Loading full model from {checkpoint_path}")
        model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)

    return model, tokenizer


def evaluate(model, tokenizer, examples, db_path, device, eval_name: str):
    results = []
    by_diff = defaultdict(lambda: {"total": 0, "exec": 0, "exact": 0})

    for ex in tqdm(examples, desc=f"Eval {eval_name}"):
        pred_sql = generate_sql(model, tokenizer, ex["input_text"], device)
        gold_sql = ex.get("gold_sql", ex["target_text"])

        ex_exec = execution_accuracy(pred_sql, gold_sql, db_path) if db_path else False
        ex_exact = exact_match(pred_sql, gold_sql)

        diff = ex.get("difficulty", "unknown")
        by_diff[diff]["total"] += 1
        by_diff[diff]["exec"] += int(ex_exec)
        by_diff[diff]["exact"] += int(ex_exact)

        results.append({
            "question": ex.get("question", ex["input_text"][:100]),
            "gold": gold_sql,
            "pred": pred_sql,
            "exec_match": ex_exec,
            "exact_match": ex_exact,
            "difficulty": diff,
        })

    n = len(results)
    total_exec = sum(r["exec_match"] for r in results)
    total_exact = sum(r["exact_match"] for r in results)

    print(f"\n=== {eval_name} ({n} examples) ===")
    print(f"  Execution accuracy: {total_exec}/{n} = {total_exec/n:.1%}")
    print(f"  Exact match:        {total_exact}/{n} = {total_exact/n:.1%}")
    print(f"\n  By difficulty:")
    for diff in ["easy", "medium", "hard", "extra_hard", "unknown"]:
        if diff in by_diff and by_diff[diff]["total"] > 0:
            d = by_diff[diff]
            print(f"    {diff:11s}: exec {d['exec']:3d}/{d['total']:3d} ({d['exec']/d['total']:5.1%}) | "
                  f"exact {d['exact']:3d}/{d['total']:3d} ({d['exact']/d['total']:5.1%})")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", default="t5-base",
                        help="For PEFT, the base model the adapter was trained on")
    parser.add_argument("--eval", choices=["nba", "spider", "both"], required=True)
    parser.add_argument("--nba-questions", default="data/nba/nba_questions.json")
    parser.add_argument("--nba-db", default="data/raw/nba.sqlite")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--output-dir", default="eval")
    parser.add_argument("--oracle-tables", action="store_true",
                    help="Restrict NBA schema to gold-relevant tables (mimics perfect RAG)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, tokenizer = load_checkpoint(args.checkpoint, args.base_model)
    model.to(device).eval()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    run_name = Path(args.checkpoint).parent.name

    if args.eval in ("nba", "both"):
        nba = load_nba_dataset(args.nba_questions, args.nba_db, use_oracle_tables=args.oracle_tables)
        if args.max_examples:
            nba = nba[:args.max_examples]
        results = evaluate(model, tokenizer, nba, args.nba_db, device, f"NBA / {run_name}")
        out = f"{args.output_dir}/{run_name}_nba.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Saved → {out}")

    if args.eval in ("spider", "both"):
        _, dev = load_spider_splits(max_examples=args.max_examples)
        # Spider exec acc requires per-db sqlite paths; skip for now, exact match only
        results = evaluate(model, tokenizer, dev, None, device, f"Spider / {run_name}")
        out = f"{args.output_dir}/{run_name}_spider.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Saved → {out}")


if __name__ == "__main__":
    main()