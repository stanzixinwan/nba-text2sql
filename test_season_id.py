import sqlite3
conn = sqlite3.connect("data/raw/nba.sqlite")
cur = conn.cursor()

# 看 season_id 的真实分布
print("=== Recent season_ids in game table ===")
cur.execute("SELECT DISTINCT season_id, season_type FROM game ORDER BY season_id DESC LIMIT 20")
for row in cur.fetchall():
    print(f"  {row}")

# 看最新的几场比赛的日期和 season_id
print("\n=== Most recent games ===")
cur.execute("SELECT game_date, season_id, season_type, team_name_home FROM game ORDER BY game_date DESC LIMIT 5")
for row in cur.fetchall():
    print(f"  {row}")

conn.close()