import sqlite3

conn = sqlite3.connect("data/raw/nba.sqlite")
cur = conn.cursor()

# 列出所有表
cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = [r[0] for r in cur.fetchall()]
print(f"Found {len(tables)} tables:")
for t in tables:
    print(f"  - {t}")

# 看一下每张表的列数和行数
print("\nTable details:")
for t in tables:
    cur.execute(f"SELECT COUNT(*) FROM {t}")
    n_rows = cur.fetchone()[0]
    cur.execute(f"PRAGMA table_info({t})")
    n_cols = len(cur.fetchall())
    print(f"  {t}: {n_cols} cols, {n_rows} rows")

conn.close()