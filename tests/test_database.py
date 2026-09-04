"""Veri-Drive automated tests — database entry/exit state transitions.

Run all tests:   py -m unittest discover tests -v
These tests use a TEMPORARY database; the real data/veri_drive.db is
never touched.
"""
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "modules"))

import config

# Redirect storage BEFORE database.py is used
_TMP = tempfile.mkdtemp(prefix="veridrive_test_")
config.DATABASE_PATH = os.path.join(_TMP, "test.db")
config.BACKUP_DIR = os.path.join(_TMP, "backups")

import database as db


class TestRegistration(unittest.TestCase):
    def setUp(self):
        db.init_db()

    def test_driver_validation(self):
        self.assertEqual(db.add_driver("")[0], False)
        self.assertEqual(db.add_driver("Bad!Name")[0], False)
        self.assertEqual(db.add_driver("X" * 41)[0], False)
        self.assertEqual(db.add_driver("Ali")[0], True)
        # duplicate rejected
        ok, msg = db.add_driver("Ali")
        self.assertFalse(ok)
        self.assertIn("already", msg)

    def test_vehicle_validation(self):
        self.assertEqual(db.add_vehicle("")[0], False)
        self.assertEqual(db.add_vehicle("AB")[0], False)          # too short
        self.assertEqual(db.add_vehicle("ABCDEFGHIJKLM")[0], False)  # too long
        self.assertEqual(db.add_vehicle("ABC@12")[0], False)      # bad chars
        self.assertEqual(db.add_vehicle("abc-1234")[0], True)     # uppercased
        self.assertTrue(db.is_vehicle_registered("ABC-1234"))

    def test_remove_keeps_history(self):
        db.add_driver("Sara")
        db.add_vehicle("XYZ-5678")
        db.log_gate_event("XYZ-5678", "Sara")
        db.remove_driver("Sara")
        db.remove_vehicle("XYZ-5678")
        # history rows survive deletion
        self.assertEqual(len(db.get_gate_log(search="XYZ-5678")), 1)


class TestEntryExitTransitions(unittest.TestCase):
    def setUp(self):
        db.init_db()
        # unique plate per test method: the temp DB is shared, so a fixed
        # plate would mix trips across tests
        self.plate = ("T-" + self._testMethodName.replace("_", "-").upper())[:12]
        db.add_vehicle(self.plate)
        self.base = datetime(2026, 9, 1, 8, 0, 0)

    def test_entry_then_exit(self):
        e1 = db.log_gate_event(self.plate, "Ali", self.base)
        self.assertEqual(e1["event_type"], "entry")
        self.assertTrue(e1["inferred"])     # single camera -> always inferred

        e2 = db.log_gate_event(self.plate, "Ali",
                               self.base + timedelta(minutes=45))
        self.assertEqual(e2["event_type"], "exit")
        self.assertEqual(e2["duration_minutes"], 45.0)
        self.assertTrue(e2["inferred"])

        rows = db.get_gate_log(search=self.plate)
        self.assertEqual(len(rows), 1)      # one trip, closed
        self.assertIsNotNone(rows[0]["exit_time"])
        self.assertEqual(rows[0]["inferred"], 1)

    def test_second_entry_opens_new_trip(self):
        db.log_gate_event(self.plate, "Ali", self.base)
        db.log_gate_event(self.plate, "Ali", self.base + timedelta(minutes=10))
        e = db.log_gate_event(self.plate, "Ali", self.base + timedelta(minutes=20))
        self.assertEqual(e["event_type"], "entry")   # new open trip
        open_trips = [r for r in db.get_gate_log(search=self.plate)
                      if r["exit_time"] is None]
        self.assertEqual(len(open_trips), 1)

    def test_driver_fallback_on_exit(self):
        db.log_gate_event(self.plate, "Ali", self.base)
        e = db.log_gate_event(self.plate, None, self.base + timedelta(minutes=5))
        self.assertEqual(e["driver"], "Ali")   # kept from the entry record


class TestPagingSearchBackup(unittest.TestCase):
    def setUp(self):
        db.init_db()

    def test_search_and_paging(self):
        db.add_vehicle("AAA-111")
        db.add_vehicle("BBB-222")
        before = db.count_gate_log()
        db.log_gate_event("AAA-111", "Ali")
        db.log_gate_event("BBB-222", "Sara")
        self.assertEqual(db.count_gate_log(), before + 2)
        self.assertEqual(db.count_gate_log("AAA"), 1)
        page = db.get_gate_log(limit=1, offset=0)
        self.assertEqual(len(page), 1)
        self.assertEqual(len(db.get_gate_log(limit=1, offset=1)), 1)

    def test_backup_prunes(self):
        db.add_driver("BackupTest")
        path = db.backup_database("unit-test")
        self.assertTrue(path and os.path.exists(path))
        backups = os.listdir(config.BACKUP_DIR)
        self.assertLessEqual(len(backups), config.BACKUP_KEEP)

    def test_rows_to_csv(self):
        csv_text = db.rows_to_csv([{"a": 1, "b": "x"}], ["a", "b"])
        self.assertIn("a,b", csv_text)
        self.assertIn("1,x", csv_text)


class TestClearAndReport(unittest.TestCase):
    def setUp(self):
        db.init_db()

    def test_clear_all_gate_records(self):
        db.add_vehicle("CLR-100")
        db.log_gate_event("CLR-100", "Ali")
        self.assertGreaterEqual(db.count_gate_log("CLR-100"), 1)
        n, msg = db.clear_records("gate")
        self.assertGreaterEqual(n, 1)
        self.assertIn("Deleted", msg)
        self.assertEqual(db.count_gate_log("CLR-100"), 0)

    def test_clear_before_cutoff_keeps_recent(self):
        db.add_vehicle("CLR-200")
        db.add_vehicle("CLR-201")
        db.log_gate_event("CLR-200", "Ali", datetime(2020, 1, 1, 8, 0, 0))
        db.log_gate_event("CLR-201", "Ali", datetime.now())
        db.clear_records("gate", before="2021-01-01T00:00:00")
        self.assertEqual(db.count_gate_log("CLR-200"), 0)        # old removed
        self.assertGreaterEqual(db.count_gate_log("CLR-201"), 1)  # recent kept

    def test_clear_unknown_kind_is_noop(self):
        n, msg = db.clear_records("bogus")
        self.assertEqual(n, 0)

    def test_weekly_report_includes_selected_day(self):
        """Guards the bare-date 'end of day' fix: a report generated for
        TODAY must include a trip logged today (previously excluded because
        a date-only 'end' parsed to midnight)."""
        db.add_vehicle("RPT-300")
        db.log_gate_event("RPT-300", "Ali", datetime.now())
        today = datetime.now().date().isoformat()
        rep = db.get_weekly_report(today)
        plates = [t["plate"] for t in rep["trips"]]
        self.assertIn("RPT-300", plates)


if __name__ == "__main__":
    unittest.main(verbosity=2)
