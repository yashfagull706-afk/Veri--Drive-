"""
Database Module — Veri-Drive

Manages the SQLite database for the gate security system:
  - Registered drivers and registered vehicles (cars)
  - Gate log: automatic ENTRY / EXIT time logging per registered car
  - Alarms: every unregistered car / unregistered driver event
  - Driver logs: how many times each driver drove, and which car
  - Driver-car logs: driver <-> vehicle usage history
  - Weekly reports: everything above for a chosen 7-day window

Core rule (security policy):
  - An event is ONLY logged when a plate is read.  A driver passing
    the gate with no readable plate is NOT logged.
  - Only REGISTERED cars get entry/exit trip logs.
  - Unregistered car or unregistered driver -> alarm record.

Usage:
    python -m modules.database     # runs a quick self-test
"""

import os
import re
import sys
import sqlite3
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------
def get_connection():
    """Returns a connection to the SQLite database, creating folders if needed."""
    os.makedirs(os.path.dirname(config.DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row  # access columns by name
    return conn


def init_db():
    """Creates all Veri-Drive tables if they don't exist."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS drivers (
            name         TEXT PRIMARY KEY,
            phone        TEXT DEFAULT '',
            registered_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS vehicles (
            plate         TEXT PRIMARY KEY,
            label         TEXT DEFAULT '',
            registered_at TEXT NOT NULL
        )
    """)

    # One row per trip: entry time, exit time, duration
    cur.execute("""
        CREATE TABLE IF NOT EXISTS gate_log (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            plate            TEXT NOT NULL,
            driver           TEXT,
            entry_time       TEXT NOT NULL,
            exit_time        TEXT,
            duration_minutes REAL,
            snapshot         TEXT
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_gate_plate ON gate_log(plate)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_gate_driver ON gate_log(driver)")

    # Migration: 'inferred' marks that entry/exit direction was INFERRED from
    # trip state (one camera cannot see direction of travel).  Older databases
    # get the column added automatically.
    cols = [r[1] for r in cur.execute("PRAGMA table_info(gate_log)")]
    if "inferred" not in cols:
        cur.execute("ALTER TABLE gate_log ADD COLUMN inferred INTEGER DEFAULT 1")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS alarms (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            time       TEXT NOT NULL,
            kind       TEXT NOT NULL,      -- UNKNOWN_VEHICLE / UNKNOWN_DRIVER / UNKNOWN_BOTH
            plate      TEXT,
            driver     TEXT,
            details    TEXT,
            snapshot   TEXT
        )
    """)

    conn.commit()
    conn.close()
    print("[DB] Veri-Drive database initialized.")


def _now():
    return datetime.now()


# ---------------------------------------------------------------------------
# Backups (P2): copy the DB file before destructive ops + daily snapshots
# ---------------------------------------------------------------------------
def backup_database(reason="manual"):
    """Copies veri_drive.db into data/backups with a timestamp + reason.

    Keeps only the newest config.BACKUP_KEEP files. Returns the backup path
    (or None if there is nothing to back up / the copy failed).
    """
    import shutil
    if not os.path.exists(config.DATABASE_PATH):
        return None
    os.makedirs(config.BACKUP_DIR, exist_ok=True)
    safe_reason = re.sub(r'[^A-Za-z0-9]+', '-', reason).strip('-') or "backup"
    stamp = _now().strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(config.BACKUP_DIR, f"veri_drive_{stamp}_{safe_reason}.db")
    try:
        shutil.copy2(config.DATABASE_PATH, dest)
    except OSError as exc:
        print(f"[DB] backup failed: {exc}")
        return None
    # prune old backups
    try:
        backups = sorted(
            (f for f in os.listdir(config.BACKUP_DIR)
             if f.startswith("veri_drive_") and f.endswith(".db")),
            reverse=True)
        for old in backups[config.BACKUP_KEEP:]:
            os.remove(os.path.join(config.BACKUP_DIR, old))
    except OSError:
        pass
    print(f"[DB] backup saved: {dest}")
    return dest


# ---------------------------------------------------------------------------
# Registration: drivers
# ---------------------------------------------------------------------------
def add_driver(name, phone=""):
    """Adds a driver record. Returns (ok, message)."""
    name = (name or "").strip()
    if not name:
        return False, "Driver name cannot be empty."
    if len(name) > 40:
        return False, "Driver name is too long (max 40 characters)."
    if not re.match(r"^[A-Za-z0-9 _\-]+$", name):
        return False, "Driver name contains invalid characters."

    conn = get_connection()
    try:
        conn.execute("INSERT INTO drivers (name, phone, registered_at) VALUES (?, ?, ?)",
                     (name, (phone or "").strip(), _now().isoformat(timespec="seconds")))
        conn.commit()
        return True, f"Driver '{name}' registered."
    except sqlite3.IntegrityError:
        return False, f"Driver '{name}' is already registered."
    finally:
        conn.close()


def remove_driver(name):
    """Removes a driver record (history rows are kept). Returns (ok, message)."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM drivers WHERE name = ?", (name,))
    conn.commit()
    removed = cur.rowcount > 0
    conn.close()
    if removed:
        return True, f"Driver '{name}' removed."
    return False, f"Driver '{name}' not found."


def get_drivers():
    """All registered drivers with their trip counts."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT d.name, d.phone, d.registered_at,
               (SELECT COUNT(*) FROM gate_log g WHERE g.driver = d.name) AS trips
        FROM drivers d ORDER BY d.name
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def is_driver_registered(name):
    if not name:
        return False
    conn = get_connection()
    row = conn.execute("SELECT 1 FROM drivers WHERE name = ?", (name,)).fetchone()
    conn.close()
    return row is not None


# ---------------------------------------------------------------------------
# Registration: vehicles
# ---------------------------------------------------------------------------
def add_vehicle(plate, label=""):
    """Adds a vehicle record. Returns (ok, message)."""
    plate = (plate or "").strip().upper()
    if not plate:
        return False, "Plate number cannot be empty."
    if not (config.MIN_PLATE_LENGTH <= len(plate) <= config.MAX_PLATE_LENGTH):
        return False, (f"Plate must be {config.MIN_PLATE_LENGTH}-"
                       f"{config.MAX_PLATE_LENGTH} characters.")
    if not re.match(r"^[A-Z0-9\- ]+$", plate):
        return False, "Plate contains invalid characters (letters/digits only)."

    conn = get_connection()
    try:
        conn.execute("INSERT INTO vehicles (plate, label, registered_at) VALUES (?, ?, ?)",
                     (plate, (label or "").strip(), _now().isoformat(timespec="seconds")))
        conn.commit()
        return True, f"Vehicle '{plate}' registered."
    except sqlite3.IntegrityError:
        return False, f"Vehicle '{plate}' is already registered."
    finally:
        conn.close()


def remove_vehicle(plate):
    """Removes a vehicle record (history rows are kept). Returns (ok, message)."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM vehicles WHERE plate = ?", (plate,))
    conn.commit()
    removed = cur.rowcount > 0
    conn.close()
    if removed:
        return True, f"Vehicle '{plate}' removed."
    return False, f"Vehicle '{plate}' not found."


def clear_records(kind, before=None):
    """Delete stored log records (used by the Logs page "Delete" button).

    kind: 'gate' | 'alarms'.  The 'drivers' and 'dcar' views are computed
    from the gate_log table, so they map to 'gate'.
    before: optional ISO date/datetime string; when given, only records with
    a timestamp strictly older than it are removed.  Otherwise ALL are removed.
    Returns (deleted_count, message).
    """
    table_map = {
        "gate":    ("gate_log", "entry_time"),
        "drivers": ("gate_log", "entry_time"),
        "dcar":    ("gate_log", "entry_time"),
        "alarms":  ("alarms",   "time"),
    }
    if kind not in table_map:
        return 0, "Unknown record type."
    table, tcol = table_map[kind]
    conn = get_connection()
    cur = conn.cursor()
    if before:
        cur.execute(f"DELETE FROM {table} WHERE {tcol} < ?", (before,))
    else:
        cur.execute(f"DELETE FROM {table}")
    conn.commit()
    n = cur.rowcount
    conn.close()
    label = "alarm record(s)" if kind == "alarms" else "trip record(s)"
    scope = f"older than {before}" if before else "all"
    return n, f"Deleted {n} {label} ({scope})."


def get_vehicles():
    """All registered vehicles with their trip counts."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT v.plate, v.label, v.registered_at,
               (SELECT COUNT(*) FROM gate_log g WHERE g.plate = v.plate) AS trips
        FROM vehicles v ORDER BY v.plate
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_registered_plates():
    """Set of all registered plate strings."""
    conn = get_connection()
    rows = conn.execute("SELECT plate FROM vehicles").fetchall()
    conn.close()
    return {r["plate"] for r in rows}


def is_vehicle_registered(plate):
    if not plate:
        return False
    conn = get_connection()
    row = conn.execute("SELECT 1 FROM vehicles WHERE plate = ?", (plate,)).fetchone()
    conn.close()
    return row is not None


# ---------------------------------------------------------------------------
# Gate log: entry / exit
# ---------------------------------------------------------------------------
def log_gate_event(plate, driver=None, timestamp=None, snapshot=None):
    """
    Logs an ENTRY or EXIT for a REGISTERED vehicle.

    - If the plate has no open entry (exit_time NULL) -> ENTRY row
    - If it has an open entry -> close it (EXIT), compute duration

    Returns a dict describing the event.
    """
    if timestamp is None:
        timestamp = _now()

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, entry_time, driver FROM gate_log
        WHERE plate = ? AND exit_time IS NULL
        ORDER BY entry_time DESC LIMIT 1
    """, (plate,))
    open_entry = cur.fetchone()

    if open_entry is None:
        cur.execute("""
            INSERT INTO gate_log (plate, driver, entry_time, snapshot, inferred)
            VALUES (?, ?, ?, ?, 1)
        """, (plate, driver, timestamp.isoformat(timespec="seconds"), snapshot))
        conn.commit()
        conn.close()
        return {
            "event_type": "entry",
            "plate": plate,
            "driver": driver or "Unknown",
            "time": timestamp.strftime("%I:%M %p"),
            "duration_minutes": None,
            "inferred": True,   # direction inferred from trip state (1 camera)
            "message": f"ENTRY: {plate} | driver {driver or 'Unknown'} @ "
                       f"{timestamp.strftime('%I:%M %p')}",
        }

    entry_time = datetime.fromisoformat(open_entry["entry_time"])
    duration_minutes = round((timestamp - entry_time).total_seconds() / 60, 1)
    final_driver = driver or open_entry["driver"] or "Unknown"

    cur.execute("""
        UPDATE gate_log SET exit_time = ?, duration_minutes = ?, driver = ?
        WHERE id = ?
    """, (timestamp.isoformat(timespec="seconds"), duration_minutes,
          final_driver, open_entry["id"]))
    conn.commit()
    conn.close()
    return {
        "event_type": "exit",
        "plate": plate,
        "driver": final_driver,
        "time": timestamp.strftime("%I:%M %p"),
        "duration_minutes": duration_minutes,
        "inferred": True,   # direction inferred from trip state (1 camera)
        "message": f"EXIT: {plate} | driver {final_driver} @ "
                   f"{timestamp.strftime('%I:%M %p')} | {duration_minutes} min",
    }


# ---------------------------------------------------------------------------
# Alarms
# ---------------------------------------------------------------------------
def raise_alarm(kind, plate=None, driver=None, details="", snapshot=None,
                timestamp=None):
    """
    Records a security alarm.

    kind: UNKNOWN_VEHICLE | UNKNOWN_DRIVER | UNKNOWN_BOTH
    Returns the alarm dict (includes its id so the UI can detect new ones).
    """
    if timestamp is None:
        timestamp = _now()

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO alarms (time, kind, plate, driver, details, snapshot)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (timestamp.isoformat(timespec="seconds"), kind, plate, driver,
          details, snapshot))
    conn.commit()
    alarm_id = cur.lastrowid
    conn.close()

    print(f"[ALARM] {kind} | plate={plate or '-'} driver={driver or '-'} "
          f"@ {timestamp.strftime('%I:%M %p')}")
    return {
        "id": alarm_id,
        "time": timestamp.strftime("%I:%M %p"),
        "iso_time": timestamp.isoformat(timespec="seconds"),
        "kind": kind,
        "plate": plate,
        "driver": driver,
        "details": details,
    }


def get_last_alarm():
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM alarms ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Log queries (tables / live views) — with pagination + search (P3)
# ---------------------------------------------------------------------------
def get_gate_log(limit=200, offset=0, search=""):
    """Trips, newest first.  `search` filters plate/driver (substring)."""
    conn = get_connection()
    where, params = "", []
    if search:
        where = "WHERE plate LIKE ? OR driver LIKE ?"
        like = f"%{search}%"
        params = [like, like]
    rows = conn.execute(f"""
        SELECT id, plate, driver, entry_time, exit_time, duration_minutes,
               snapshot, inferred
        FROM gate_log {where}
        ORDER BY entry_time DESC LIMIT ? OFFSET ?
    """, params + [limit, offset]).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def count_gate_log(search=""):
    conn = get_connection()
    if search:
        like = f"%{search}%"
        n = conn.execute(
            "SELECT COUNT(*) FROM gate_log WHERE plate LIKE ? OR driver LIKE ?",
            (like, like)).fetchone()[0]
    else:
        n = conn.execute("SELECT COUNT(*) FROM gate_log").fetchone()[0]
    conn.close()
    return n


def get_alarms(limit=200, offset=0, search=""):
    """Alarms, newest first.  `search` filters kind/plate/driver/details."""
    conn = get_connection()
    where, params = "", []
    if search:
        where = ("WHERE kind LIKE ? OR plate LIKE ? OR driver LIKE ? "
                 "OR details LIKE ?")
        like = f"%{search}%"
        params = [like, like, like, like]
    rows = conn.execute(f"""
        SELECT * FROM alarms {where} ORDER BY id DESC LIMIT ? OFFSET ?
    """, params + [limit, offset]).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def count_alarms(search=""):
    conn = get_connection()
    if search:
        like = f"%{search}%"
        n = conn.execute(
            "SELECT COUNT(*) FROM alarms WHERE kind LIKE ? OR plate LIKE ? "
            "OR driver LIKE ? OR details LIKE ?",
            (like, like, like, like)).fetchone()[0]
    else:
        n = conn.execute("SELECT COUNT(*) FROM alarms").fetchone()[0]
    conn.close()
    return n


def rows_to_csv(rows, columns):
    """Serializes a list of dicts to a CSV string (for the export buttons)."""
    import csv
    import io
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(columns)
    for r in rows:
        writer.writerow([r.get(c, "") for c in columns])
    return buf.getvalue()


def get_driver_logs():
    """
    Per-driver summary: how many trips each driver made, with which cars,
    total time on the road, first and last trip.
    """
    conn = get_connection()
    rows = conn.execute("""
        SELECT driver,
               COUNT(*)                                   AS total_trips,
               COUNT(DISTINCT plate)                      AS cars_used,
               ROUND(SUM(COALESCE(duration_minutes, 0)), 1) AS total_minutes,
               MIN(entry_time)                            AS first_trip,
               MAX(COALESCE(exit_time, entry_time))       AS last_seen
        FROM gate_log
        WHERE driver IS NOT NULL AND driver != ''
        GROUP BY driver
        ORDER BY total_trips DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_driver_vehicle_logs():
    """
    Driver <-> vehicle usage history: how many times each driver drove
    each specific car.
    """
    conn = get_connection()
    rows = conn.execute("""
        SELECT driver, plate,
               COUNT(*)                                       AS times_driven,
               ROUND(SUM(COALESCE(duration_minutes, 0)), 1)  AS total_minutes,
               MAX(COALESCE(exit_time, entry_time))          AS last_used
        FROM gate_log
        WHERE driver IS NOT NULL AND driver != ''
        GROUP BY driver, plate
        ORDER BY times_driven DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Weekly reports
# ---------------------------------------------------------------------------
def _week_bounds(end_date=None):
    """Returns (start, end) datetimes for a 7-day window ending at end_date."""
    if end_date is None:
        end = _now()
    elif isinstance(end_date, datetime):
        end = end_date
    else:
        raw = str(end_date).strip()
        end = datetime.fromisoformat(raw)
        # A bare date (YYYY-MM-DD) parses to midnight, which would EXCLUDE
        # that whole day's records (entry_time <= midnight = nothing today).
        # Treat a date-only value as the END of that day so it is inclusive.
        if ":" not in raw:
            end = end.replace(hour=23, minute=59, second=59, microsecond=999999)
    start = end - timedelta(days=config.WEEK_DAYS)
    return start, end


def get_weekly_report(end_date=None):
    """
    Builds the full weekly report:
      - gate log times in the window
      - per-driver logs
      - driver-car logs
      - alarms raised
      - headline numbers
    """
    start, end = _week_bounds(end_date)
    conn = get_connection()

    trips = conn.execute("""
        SELECT id, plate, driver, entry_time, exit_time, duration_minutes
        FROM gate_log
        WHERE entry_time >= ? AND entry_time <= ?
        ORDER BY entry_time DESC
    """, (start.isoformat(timespec="seconds"),
          end.isoformat(timespec="seconds"))).fetchall()

    driver_logs = conn.execute("""
        SELECT driver,
               COUNT(*)                                          AS total_trips,
               COUNT(DISTINCT plate)                             AS cars_used,
               ROUND(SUM(COALESCE(duration_minutes, 0)), 1)      AS total_minutes
        FROM gate_log
        WHERE driver IS NOT NULL AND driver != ''
          AND entry_time >= ? AND entry_time <= ?
        GROUP BY driver ORDER BY total_trips DESC
    """, (start.isoformat(timespec="seconds"),
          end.isoformat(timespec="seconds"))).fetchall()

    driver_vehicle = conn.execute("""
        SELECT driver, plate,
               COUNT(*)                                          AS times_driven,
               ROUND(SUM(COALESCE(duration_minutes, 0)), 1)      AS total_minutes
        FROM gate_log
        WHERE driver IS NOT NULL AND driver != ''
          AND entry_time >= ? AND entry_time <= ?
        GROUP BY driver, plate ORDER BY times_driven DESC
    """, (start.isoformat(timespec="seconds"),
          end.isoformat(timespec="seconds"))).fetchall()

    vehicle_logs = conn.execute("""
        SELECT plate,
               COUNT(*)                                          AS total_trips,
               COUNT(DISTINCT driver)                            AS drivers_used,
               ROUND(SUM(COALESCE(duration_minutes, 0)), 1)      AS total_minutes
        FROM gate_log
        WHERE entry_time >= ? AND entry_time <= ?
        GROUP BY plate ORDER BY total_trips DESC
    """, (start.isoformat(timespec="seconds"),
          end.isoformat(timespec="seconds"))).fetchall()

    alarm_rows = conn.execute("""
        SELECT * FROM alarms
        WHERE time >= ? AND time <= ?
        ORDER BY id DESC
    """, (start.isoformat(timespec="seconds"),
          end.isoformat(timespec="seconds"))).fetchall()

    conn.close()

    return {
        "start": start.strftime("%d %b %Y, %I:%M %p"),
        "end": end.strftime("%d %b %Y, %I:%M %p"),
        "totals": {
            "trips": len(trips),
            "drivers_active": len(driver_logs),
            "vehicles_active": len(vehicle_logs),
            "alarms": len(alarm_rows),
        },
        "trips": [dict(r) for r in trips],
        "driver_logs": [dict(r) for r in driver_logs],
        "driver_vehicle": [dict(r) for r in driver_vehicle],
        "vehicle_logs": [dict(r) for r in vehicle_logs],
        "alarms": [dict(r) for r in alarm_rows],
    }


# ---------------------------------------------------------------------------
# Dashboard stats
# ---------------------------------------------------------------------------
def get_summary_stats():
    conn = get_connection()
    today = _now().date().isoformat()

    def count(sql, params=()):
        return conn.execute(sql, params).fetchone()[0]

    stats = {
        "registered_drivers": count("SELECT COUNT(*) FROM drivers"),
        "registered_vehicles": count("SELECT COUNT(*) FROM vehicles"),
        "trips_today": count(
            "SELECT COUNT(*) FROM gate_log WHERE DATE(entry_time) = ?", (today,)),
        "active_vehicles": count(
            "SELECT COUNT(*) FROM gate_log WHERE exit_time IS NULL"),
        "alarms_today": count(
            "SELECT COUNT(*) FROM alarms WHERE DATE(time) = ?", (today,)),
        "alarms_total": count("SELECT COUNT(*) FROM alarms"),
    }
    conn.close()
    return stats


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _run_self_test():
    print("=== Veri-Drive Database Self-Test ===\n")

    if os.path.exists(config.DATABASE_PATH):
        os.remove(config.DATABASE_PATH)
    init_db()

    base = datetime(2026, 8, 24, 8, 0, 0)

    print("-- register drivers & vehicles --")
    print(add_driver("Ali"))
    print(add_driver("Sara"))
    print(add_driver("Ali"))            # duplicate -> error
    print(add_vehicle("ABC-1234"))
    print(add_vehicle("XYZ-5678"))
    print(add_vehicle("AB"))            # too short -> error

    print("\n-- gate events --")
    print(log_gate_event("ABC-1234", "Ali", base)["message"])
    print(log_gate_event("ABC-1234", "Ali", base + timedelta(minutes=45))["message"])
    print(log_gate_event("XYZ-5678", "Sara", base + timedelta(hours=1))["message"])

    print("\n-- alarm --")
    print(raise_alarm("UNKNOWN_VEHICLE", plate="ZZZ-999",
                      details="self-test alarm"))

    print("\n-- driver logs --")
    for row in get_driver_logs():
        print(" ", row)

    print("\n-- driver-vehicle logs --")
    for row in get_driver_vehicle_logs():
        print(" ", row)

    print("\n-- weekly report totals --")
    rep = get_weekly_report()
    print(" ", rep["totals"])

    print("\n=== Self-test passed ===")


if __name__ == "__main__":
    _run_self_test()
