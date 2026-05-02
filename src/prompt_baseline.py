"""
prompt_baseline.py — Zero-shot prompting baseline with pretrained T5-base.

No fine-tuning. Just feed (question + schema) to the pretrained model and
measure execution accuracy. Establishes the lower bound for all later experiments.

Usage:
    python -m src.prompt_baseline --eval nba --max-examples 100
    python -m src.prompt_baseline --eval spider --max-examples 200
"""

import argparse
import json
import sqlite3
from pathlib import Path
from collections import defaultdict

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm

from src.data_utils import load_nba_dataset, load_spider_splits


def execute_sql(sql: str, db_path: str, timeout: float = 5.0):
    """Run SQL against a SQLite DB. Returns (success, result_set or error_str)."""
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        # Normalize: sort rows so order doesn't matter for set comparison
        return True, sorted([tuple(str(v) for v in r) for r in rows])
    except Exception as e:
        return False, str(e)


def execution_accuracy(pred_sql: str, gold_sql: str, db_path: str) -> bool:
    """Both SQL queries execute and return the same row set."""
    p_ok, p_res = execute_sql(pred_sql, db_path)
    g_ok, g_res = execute_sql(gold_sql, db_path)
    if not (p_ok and g_ok):
        return False
    return p_res == g_res


def exact_match(pred_sql: str, gold_sql: str) -> bool:
    """String match after lowercasing and whitespace normalization."""
    norm = lambda s: " ".join(s.lower().split()).rstrip(";")
    return norm(pred_sql) == norm(gold_sql)


def generate_sql(model, tokenizer, input_text: str, device, max_length: int = 512) -> str:
    """Run a single forward pass to generate SQL from input."""
    inputs = tokenizer(input_text, return_tensors="pt",
                       max_length=1024, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=8,
            length_penalty=1.0,              # 鼓励生成完整SQL
            no_repeat_ngram_size=0,          # SQL允许重复
            early_stopping=True,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def evaluate(model, tokenizer, examples, db_path, device, eval_name: str):
    """Run model over examples, compute exec accuracy + exact match, stratified by difficulty."""
    results = []
    by_diff = defaultdict(lambda: {"total": 0, "exec": 0, "exact": 0})

    for ex in tqdm(examples, desc=f"Evaluating {eval_name}"):
        pred_sql = generate_sql(model, tokenizer, ex["input_text"], device)
        gold_sql = ex.get("gold_sql", ex["target_text"])

        # For Spider, db_path differs per example — skip exec acc, use exact match only
        if db_path is None:
            ex_exec = False
        else:
            ex_exec = execution_accuracy(pred_sql, gold_sql, db_path)
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

    # Print summary
    n = len(results)
    total_exec = sum(r["exec_match"] for r in results)
    total_exact = sum(r["exact_match"] for r in results)

    print(f"\n=== {eval_name} Results ({n} examples) ===")
    print(f"  Execution accuracy: {total_exec}/{n} = {total_exec/n:.1%}")
    print(f"  Exact match:        {total_exact}/{n} = {total_exact/n:.1%}")
    print(f"\n  By difficulty:")
    for diff in ["easy", "medium", "hard", "extra_hard", "unknown"]:
        if diff in by_diff:
            d = by_diff[diff]
            print(f"    {diff:11s}: exec {d['exec']}/{d['total']} ({d['exec']/d['total']:.1%}) | "
                  f"exact {d['exact']}/{d['total']} ({d['exact']/d['total']:.1%})")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="t5-base")
    parser.add_argument("--eval", choices=["nba", "spider"], required=True)
    parser.add_argument("--nba-questions", default="data/nba/nba_questions.json")
    parser.add_argument("--nba-db", default="data/raw/nba.sqlite")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--output", default=None,
                        help="Where to save per-example predictions (default: eval/baseline_<eval>.json)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading {args.model}...")
    tokenizer = T5Tokenizer.from_pretrained(args.model)
    model = T5ForConditionalGeneration.from_pretrained(args.model).to(device)
    model.eval()

    if args.eval == "nba":
        examples = load_nba_dataset(args.nba_questions, args.nba_db)
        if args.max_examples:
            examples = examples[:args.max_examples]
        results = evaluate(model, tokenizer, examples, args.nba_db, device, "NBA zero-shot")
    else:
        _, dev = load_spider_splits(max_examples=args.max_examples)
        # Spider has per-db SQLite files; skip exec acc here, exact match only
        results = evaluate(model, tokenizer, dev, None, device, "Spider zero-shot")

    output_path = args.output or f"eval/baseline_{args.eval}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()