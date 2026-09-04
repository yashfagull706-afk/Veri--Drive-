"""Fresh-start reset for Veri-Drive (wipes all tables only).

Safety:
  - Requires explicit confirmation (interactive y/N, or the --yes flag).
  - Backs up data/veri_drive.db into data/backups/ BEFORE wiping.

Usage:
    py reset_fresh.py           # asks for confirmation first
    py reset_fresh.py --yes     # non-interactive (scripts)
"""
import os
import sys
import sqlite3

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "modules"))
import config
import database as db


def main():
    if "--yes" not in sys.argv:
        answer = input("This will DELETE all drivers, vehicles, gate logs "
                       "and alarms. A backup is taken first.\n"
                       "Type 'yes' to continue: ").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted - nothing was deleted.")
            return

    # Backup before any destructive operation (P2 policy)
    path = db.backup_database("before-reset")
    print(f"backup saved: {path}")

    conn = sqlite3.connect(config.DATABASE_PATH)
    for t in ("drivers", "vehicles", "gate_log", "alarms"):
        conn.execute(f"DELETE FROM {t}")
    conn.commit()
    tables = [r[0] for r in
              conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in tables}
    conn.close()
    print("fresh reset done:", counts)


if __name__ == "__main__":
    main()
