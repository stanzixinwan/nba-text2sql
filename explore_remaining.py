"""
Explore NBA database schema for tables we haven't fully checked yet.
Usage: python explore_remaining.py
"""
import sqlite3

DB_PATH = "data/raw/nba.sqlite"
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

tables_to_check = ["draft_history", "team_details", "line_score", "game_summary", "officials"]

for table in tables_to_check:
    print(f"\n{'='*50}")
    print(f"=== {table} columns ===")
    cur.execute(f"PRAGMA table_info({table})")
    cols = cur.fetchall()
    for row in cols:
        print(f"  {row[1]:35s} {row[2]}")

    cur.execute(f"SELECT COUNT(*) FROM {table}")
    count = cur.fetchone()[0]
    print(f"\n  Total rows: {count}")

    if count > 0:
        print(f"\n=== {table} sample row ===")
        cur.execute(f"SELECT * FROM {table} LIMIT 1")
        col_names = [d[0] for d in cur.description]
        row = cur.fetchone()
        for c, v in zip(col_names, row):
            print(f"  {c:35s} = {v}")

conn.close()