"""
Replace `common_player_info` in nba.sqlite from data/raw/csv/common_player_info.csv.

The bundled SQLite snapshot can be older/smaller than the CSV export from nba-sql;
running this syncs roster rows (e.g. LeBron) with your CSV.
"""
import csv
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data/raw/nba.sqlite"
CSV_PATH = ROOT / "data/raw/csv/common_player_info.csv"


def main() -> None:
    if not CSV_PATH.is_file():
        raise SystemExit(f"Missing CSV: {CSV_PATH}")
    if not DB_PATH.is_file():
        raise SystemExit(f"Missing DB: {DB_PATH}")

    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise SystemExit("CSV has no header row")
        rows = list(reader)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys=OFF")
    cur.execute("DELETE FROM common_player_info")
    placeholders = ",".join("?" * len(fieldnames))
    cols = ",".join(fieldnames)
    sql = f"INSERT INTO common_player_info ({cols}) VALUES ({placeholders})"
    cur.executemany(sql, [tuple(r.get(c, "") for c in fieldnames) for r in rows])
    conn.commit()
    cur.execute("SELECT COUNT(*) FROM common_player_info")
    (n,) = cur.fetchone()
    cur.execute(
        "SELECT COUNT(*) FROM common_player_info WHERE display_first_last = ?",
        ("LeBron James",),
    )
    (lb,) = cur.fetchone()
    conn.close()
    print(f"Inserted {n} rows; LeBron James rows: {lb}")


if __name__ == "__main__":
    main()
