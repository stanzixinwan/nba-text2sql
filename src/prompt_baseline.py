"""
prompt_baseline.py — Zero-shot prompting baseline.

Uses AutoTokenizer/AutoModelForSeq2SeqLM so it works for T5, Flan-T5, CodeT5 alike.

Usage:
    python -m src.prompt_baseline --model t5-base --eval nba
    python -m src.prompt_baseline --model google/flan-t5-base --eval nba
"""

import argparse
import json
import sqlite3
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

from src.data_utils import load_nba_dataset, load_spider_splits


def execute_sql(sql: str, db_path: str, timeout: float = 5.0):
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return True, sorted([tuple(str(v) for v in r) for r in rows])
    except Exception as e:
        return False, str(e)


def execution_accuracy(pred_sql: str, gold_sql: str, db_path: str) -> bool:
    p_ok, p_res = execute_sql(pred_sql, db_path)
    g_ok, g_res = execute_sql(gold_sql, db_path)
    if not (p_ok and g_ok):
        return False
    return p_res == g_res


def exact_match(pred_sql: str, gold_sql: str) -> bool:
    """Lenient match: lowercase, normalize whitespace, strip semicolon."""
    import re
    def norm(s):
        s = s.lower().strip().rstrip(";")
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"\s*([(),])\s*", r"\1", s)
        return s
    return norm(pred_sql) == norm(gold_sql)


def generate_sql(model, tokenizer, input_text: str, device, max_length: int = 256) -> str:
    """Run generation. Tuned for SQL: more beams, no n-gram restriction."""
    inputs = tokenizer(input_text, return_tensors="pt",
                       max_length=1024, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=8,                  # was 4 — wider search helps SQL
            length_penalty=1.0,
            no_repeat_ngram_size=0,       # SQL legitimately repeats tokens
            early_stopping=True,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def evaluate(model, tokenizer, examples, db_path, device, eval_name: str):
    results = []
    by_diff = defaultdict(lambda: {"total": 0, "exec": 0, "exact": 0})

    for ex in tqdm(examples, desc=f"Evaluating {eval_name}"):
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

    print(f"\n=== {eval_name} Results ({n} examples) ===")
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
    parser.add_argument("--model", default="t5-base")
    parser.add_argument("--eval", choices=["nba", "spider"], required=True)
    parser.add_argument("--nba-questions", default="data/nba/nba_questions.json")
    parser.add_argument("--nba-db", default="data/raw/nba.sqlite")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--oracle-tables", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    model.eval()

    if args.eval == "nba":
        examples = load_nba_dataset(args.nba_questions, args.nba_db,
                                    use_oracle_tables=args.oracle_tables)
        if args.max_examples:
            examples = examples[:args.max_examples]
        results = evaluate(model, tokenizer, examples, args.nba_db, device,
                           f"NBA zero-shot ({args.model})")
    else:
        _, dev = load_spider_splits(max_examples=args.max_examples)
        results = evaluate(model, tokenizer, dev, None, device,
                           f"Spider zero-shot ({args.model})")

    out = args.output or f"eval/baseline_{args.eval}_{args.model.split('/')[-1]}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()