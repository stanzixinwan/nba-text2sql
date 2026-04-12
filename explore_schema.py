import sqlite3

conn = sqlite3.connect("data/raw/nba.sqlite")
cur = conn.cursor()

# 重点看 game 表的所有列
print("=== game table columns ===")
cur.execute("PRAGMA table_info(game)")
for row in cur.fetchall():
    print(f"  {row[1]:30s} {row[2]}")

# 看一行样本数据(转成 dict 好读)
print("\n=== sample game row ===")
cur.execute("SELECT * FROM game LIMIT 1")
cols = [d[0] for d in cur.description]
row = cur.fetchone()
for c, v in zip(cols, row):
    print(f"  {c:30s} = {v}")

# player 表的列
print("\n=== player table columns ===")
cur.execute("PRAGMA table_info(player)")
for row in cur.fetchall():
    print(f"  {row[1]:30s} {row[2]}")

# common_player_info 表的列(更详细)
print("\n=== common_player_info columns ===")
cur.execute("PRAGMA table_info(common_player_info)")
for row in cur.fetchall():
    print(f"  {row[1]:30s} {row[2]}")

# team 表的列
print("\n=== team table columns ===")
cur.execute("PRAGMA table_info(team)")
for row in cur.fetchall():
    print(f"  {row[1]:30s} {row[2]}")

# 看几个真实球员名, 确认数据格式
print("\n=== sample players ===")
cur.execute("SELECT * FROM player LIMIT 5")
for r in cur.fetchall():
    print(f"  {r}")

# 加在 conn.close() 之前
print("\n=== draft_history columns ===")
cur.execute("PRAGMA table_info(draft_history)")
for row in cur.fetchall():
    print(f"  {row[1]:30s} {row[2]}")

print("\n=== draft_history sample row ===")
cur.execute("SELECT * FROM draft_history LIMIT 1")
cols = [d[0] for d in cur.description]
for c, v in zip(cols, cur.fetchone()):
    print(f"  {c:30s} = {v}")

print("\n=== team_details columns ===")
cur.execute("PRAGMA table_info(team_details)")
for row in cur.fetchall():
    print(f"  {row[1]:30s} {row[2]}")
    
conn.close()

