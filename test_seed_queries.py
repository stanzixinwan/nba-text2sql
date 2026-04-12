import sqlite3
import json

with open("data/nba/nba_questions.json", "r", encoding="utf-8") as f:
    SEED_QUERIES = json.load(f)

conn = sqlite3.connect("data/raw/nba.sqlite")
cur = conn.cursor()

passed, failed = 0, 0
failures = []

for q in SEED_QUERIES:
    try:
        cur.execute(q["query"])
        result = cur.fetchall()
        print(f"[{q['difficulty']:11s}] Q{q['id']:2d}: OK ({len(result)} rows)")
        passed += 1
    except Exception as e:
        print(f"[{q['difficulty']:11s}] Q{q['id']:2d}: FAIL — {e}")
        failures.append({"id": q["id"], "error": str(e), "sql": q["query"]})
        failed += 1

print(f"\n{passed} passed, {failed} failed")

# 把失败的存下来方便修
if failures:
    with open("data/nba/failures.json", "w") as f:
        json.dump(failures, f, indent=2)
    print(f"Failures saved to data/nba/failures.json")

conn.close()