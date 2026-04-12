import sqlite3
import json
import random

conn = sqlite3.connect("data/raw/nba.sqlite")
cur = conn.cursor()

with open("data/nba/nba_questions.json", "r", encoding="utf-8") as f:
    queries = json.load(f)

from collections import Counter
diff_counts = Counter(q["difficulty"] for q in queries)
print(f"Difficulty distribution: {dict(diff_counts)}")

empty_results = []
for q in queries:
    cur.execute(q["query"])
    rows = cur.fetchall()
    if not rows or (len(rows) == 1 and all(v is None for v in rows[0])):
        empty_results.append(q["id"])

print(f"\nQueries returning empty/null: {len(empty_results)} — IDs: {empty_results}")

print("\n=== Random audit (10 queries) ===")
sample = random.sample(queries, min(10, len(queries)))
for q in sample:
    cur.execute(q["query"])
    rows = cur.fetchall()
    print(f"\nQ{q['id']} [{q['difficulty']}]: {q['question']}")
    print(f"  SQL: {q['query'][:100]}...")
    print(f"  Result: {rows[:3]}")

conn.close()