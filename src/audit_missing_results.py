"""
audit_missing_results.py — Check missing expected eval outputs.
"""

import argparse
import csv
from pathlib import Path


EXPECTED = [
    "baseline_nba_zeroshot_t5-base_test.json",
    "baseline_nba_fewshot_t5-base_test.json",
    "full_t5-base_nba_oracle_test.json",
    "lora_t5-base_r4_nba_oracle_test.json",
    "lora_t5-base_r8_nba_oracle_test.json",
    "lora_t5-base_r16_nba_oracle_test.json",
    "lora_t5-base_r32_nba_oracle_test.json",
    "qlora_t5-base_nba_oracle_test.json",
    "lora_t5-base_r16_nba_n10_nba_oracle_test.json",
    "lora_t5-base_r16_nba_n20_nba_oracle_test.json",
    "lora_t5-base_r16_nba_n70_nba_oracle_test.json",
    "lora_t5-base_r16_nba_nall_nba_oracle_test.json",
    "lora_t5-base_r16_nba_nall_nba_rag_k1_test.json",
    "lora_t5-base_r16_nba_nall_nba_rag_k3_test.json",
    "lora_t5-base_r16_nba_nall_nba_rag_k5_test.json",
    "lora_codet5p-220m_r16_nba_nall_nba_rag_k1_test.json",
    "lora_codet5p-220m_r16_nba_nall_nba_rag_k3_test.json",
    "lora_codet5p-220m_r16_nba_nall_nba_rag_k5_test.json",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default="eval")
    parser.add_argument("--out", default="eval/missing_results.csv")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    existing = {p.name for p in eval_dir.glob("*.json")}

    rows = []
    for name in EXPECTED:
        rows.append({"file": name, "status": "present" if name in existing else "missing"})

    missing = [r for r in rows if r["status"] == "missing"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "status"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Expected: {len(rows)} | Missing: {len(missing)}")
    print(f"Saved -> {out_path}")
    if missing:
        print("Missing files:")
        for row in missing:
            print(f"  - {row['file']}")


if __name__ == "__main__":
    main()
