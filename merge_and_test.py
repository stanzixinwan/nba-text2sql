"""
merge_and_test.py — Merge new 100 NBA questions with existing 100, then test all 200 against SQLite.

Usage (run from project root):
    python merge_and_test.py

Expects:
    data/nba/nba_questions.json         (your existing 100, already in repo)
    data/nba/nba_questions_new100.json  (the new 100, downloaded from this conversation)
    data/raw/nba.sqlite                 (your NBA database)

Output:
    data/nba/nba_questions.json         (overwritten with all 200, ids 1-200, pretty-printed)
    data/nba/nba_questions_backup_100.json  (backup of your original 100)
"""
import json
import sqlite3
import shutil
from pathlib import Path
from collections import Counter

EXISTING = Path("data/nba/nba_questions.json")
NEW = Path("data/nba/nba_questions_new100.json")
BACKUP = Path("data/nba/nba_questions_backup_100.json")
DB = Path("data/raw/nba.sqlite")


def load(p):
    with open(p) as f:
        return json.load(f)


def main():
    if not EXISTING.exists():
        print(f"ERROR: {EXISTING} not found")
        return
    if not NEW.exists():
        print(f"ERROR: {NEW} not found — download nba_questions_new100.json from chat")
        return

    existing = load(EXISTING)
    new = load(NEW)

    print(f"Existing: {len(existing)} | New: {len(new)}")
    if len(existing) != 100 or len(new) != 100:
        print("WARNING: expected 100+100, got different counts")

    # Backup existing before overwriting
    if not BACKUP.exists():
        shutil.copy(EXISTING, BACKUP)
        print(f"Backup saved to {BACKUP}")

    # Merge — new questions already have ids 101-200
    merged = existing + new

    # Sanity check: ids unique and contiguous 1-200
    ids = [q["id"] for q in merged]
    assert sorted(ids) == list(range(1, 201)), f"id gap detected: {sorted(set(range(1, 201)) - set(ids))}"

    # Distribution
    print("\nDifficulty distribution:")
    for diff, n in sorted(Counter(q["difficulty"] for q in merged).items()):
        print(f"  {diff}: {n}")

    # Validate every SQL executes
    print("\nValidating all 200 SQL queries against the SQLite database...")
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    passed, failed = 0, 0
    failures = []
    for q in merged:
        try:
            cur.execute(q["query"])
            rows = cur.fetchall()
            passed += 1
        except Exception as e:
            failed += 1
            failures.append({"id": q["id"], "question": q["question"],
                             "sql": q["query"], "error": str(e)})

    conn.close()
    print(f"\n{passed}/{len(merged)} passed, {failed} failed")

    if failures:
        print("\nFailed queries:")
        for f in failures:
            print(f"  Q{f['id']}: {f['error']}")
            print(f"    SQL: {f['sql'][:120]}...")
        with open("data/nba/failures.json", "w") as fout:
            json.dump(failures, fout, indent=2)
        print("\nFailure details saved to data/nba/failures.json")
        print("Review and fix before saving merged file.")
        return

    # All good — save merged
    with open(EXISTING, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Merged 200 questions saved to {EXISTING}")
    print(f"✓ Original backed up at {BACKUP}")


if __name__ == "__main__":
    main()