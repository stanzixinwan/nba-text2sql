"""
build_results_report.py — Build comparable result tables from eval JSON files.

Outputs:
  - eval/results_summary.csv
  - eval/results_summary.md
  - eval/fewshot_curve.csv
  - eval/rag_ablation.csv
  - eval/spider_summary.csv
  - eval/spider_summary.md
"""

import argparse
import csv
import json
import re
from pathlib import Path


def _load_scores(path: Path) -> tuple[float, float, int]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    n = len(rows) or 1
    exec_acc = sum(bool(r.get("exec_match")) for r in rows) / n
    exact_acc = sum(bool(r.get("exact_match")) for r in rows) / n
    return exec_acc, exact_acc, len(rows)


def _label_from_filename(name: str) -> dict:
    return {
        "run": name.replace(".json", ""),
        "model": "codet5p-220m" if "codet5p-220m" in name else "t5-base" if "t5-base" in name else "unknown",
        "method": (
            "full" if name.startswith("full_")
            else "lora" if name.startswith("lora_")
            else "qlora" if name.startswith("qlora_")
            else "baseline"
        ),
        "mode": "rag" if "_rag_" in name else "oracle" if "_oracle_" in name else "full_schema" if "_full_" in name else "other",
        "split": "test" if name.endswith("_test.json") else "all",
    }


def _parse_n_train(name: str) -> str:
    m = re.search(r"_nba_n(10|20|70|all)_", name)
    if m:
        return m.group(1)
    if "_nba_oracle_test" in name:
        return "0"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default="eval")
    parser.add_argument("--glob", default="*_test.json")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    files = sorted(eval_dir.glob(args.glob))
    rows = []
    for path in files:
        exec_acc, exact_acc, n = _load_scores(path)
        meta = _label_from_filename(path.name)
        n_train = _parse_n_train(path.name)
        rows.append({
            **meta,
            "file": path.name,
            "n_examples": n,
            "n_train": n_train,
            "exec_acc": round(exec_acc, 4),
            "exact_acc": round(exact_acc, 4),
        })

    out_csv = eval_dir / "results_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "run", "model", "method", "mode", "split", "n_examples", "n_train", "exec_acc", "exact_acc"],
        )
        writer.writeheader()
        writer.writerows(rows)

    out_md = eval_dir / "results_summary.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("| file | model | method | mode | n_train | exec_acc | exact_acc |\n")
        f.write("|---|---|---|---|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['file']} | {r['model']} | {r['method']} | {r['mode']} | {r['n_train'] or '-'} | "
                f"{r['exec_acc']:.4f} | {r['exact_acc']:.4f} |\n"
            )

    fewshot = [r for r in rows if r["n_train"] in {"0", "10", "20", "70", "all"} and r["mode"] == "oracle"]
    fewshot = sorted(fewshot, key=lambda x: {"0": 0, "10": 1, "20": 2, "70": 3, "all": 4}[x["n_train"]])
    with open(eval_dir / "fewshot_curve.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "model", "n_train", "exec_acc", "exact_acc"])
        writer.writeheader()
        for r in fewshot:
            writer.writerow({k: r[k] for k in ["file", "model", "n_train", "exec_acc", "exact_acc"]})

    rag = [r for r in rows if r["mode"] == "rag"]
    with open(eval_dir / "rag_ablation.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "model", "exec_acc", "exact_acc"])
        writer.writeheader()
        for r in rag:
            writer.writerow({k: r[k] for k in ["file", "model", "exec_acc", "exact_acc"]})

    spider_files = sorted(eval_dir.glob("*_spider.json"))
    spider_rows = []
    for path in spider_files:
        exec_acc, exact_acc, n = _load_scores(path)
        lower = path.name.lower()
        if "codet5p-220m" in lower:
            model = "codet5p-220m"
        elif "flan-t5-base" in lower:
            model = "flan-t5-base"
        elif "t5-base" in lower:
            model = "t5-base"
        else:
            model = "unknown"
        spider_rows.append({
            "file": path.name,
            "run": path.stem,
            "model": model,
            "n_examples": n,
            "exec_acc": round(exec_acc, 4),
            "exact_acc": round(exact_acc, 4),
        })

    spider_rows = sorted(spider_rows, key=lambda r: r["exact_acc"], reverse=True)
    spider_csv = eval_dir / "spider_summary.csv"
    with open(spider_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "run", "model", "n_examples", "exec_acc", "exact_acc"],
        )
        writer.writeheader()
        writer.writerows(spider_rows)

    spider_md = eval_dir / "spider_summary.md"
    with open(spider_md, "w", encoding="utf-8") as f:
        f.write("| file | model | n_examples | exec_acc | exact_acc |\n")
        f.write("|---|---|---:|---:|---:|\n")
        for r in spider_rows:
            f.write(
                f"| {r['file']} | {r['model']} | {r['n_examples']} | "
                f"{r['exec_acc']:.4f} | {r['exact_acc']:.4f} |\n"
            )

    print(f"Saved -> {out_csv}")
    print(f"Saved -> {out_md}")
    print(f"Saved -> {eval_dir / 'fewshot_curve.csv'}")
    print(f"Saved -> {eval_dir / 'rag_ablation.csv'}")
    print(f"Saved -> {spider_csv}")
    print(f"Saved -> {spider_md}")


if __name__ == "__main__":
    main()
