"""
Test all NBA seed question/SQL pairs against the local SQLite database.
Usage: python test_seed_queries.py
"""
import sqlite3
import json
 
DB_PATH = "data/raw/nba.sqlite"
QUESTIONS_PATH = "data/nba/nba_questions.json"
 
with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
    queries = json.load(f)
 
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
 
passed, failed = 0, 0
failures = []
 
for q in queries:
    try:
        cur.execute(q["query"])
        result = cur.fetchall()
        # Show first result for quick sanity check
        preview = str(result[0]) if result else "(empty)"
        print(f"  [PASS] Q{q['id']:2d} ({q['difficulty']:11s}) | {len(result):4d} rows | {preview[:60]}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Q{q['id']:2d} ({q['difficulty']:11s}) | {e}")
        failures.append({"id": q["id"], "question": q["question"], "sql": q["query"], "error": str(e)})
        failed += 1
 
print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed out of {len(queries)} total")
 
if failures:
    with open("data/nba/failures.json", "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2, ensure_ascii=False)
    print(f"Failure details saved to data/nba/failures.json")
    print("\nFailed queries:")
    for fail in failures:
        print(f"  Q{fail['id']}: {fail['error']}")
        print(f"    SQL: {fail['sql'][:100]}...")
 
conn.close()
 