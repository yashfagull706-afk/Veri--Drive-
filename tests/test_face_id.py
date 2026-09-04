"""Veri-Drive automated tests — face matching distance/margin logic.

Run all tests:   py -m unittest discover tests -v
Uses synthetic encodings injected into face_id's module state; the real
data/veri_drivers.npz is never touched.
"""
import os
import sys
import unittest

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "modules"))

import face_id


def _vec(seed, n=900):
    """Deterministic pseudo-random unit-ish vector."""
    rng = np.random.default_rng(seed)
    return rng.random(n).astype(np.float64)


class TestDistance(unittest.TestCase):
    def test_identical_is_zero(self):
        v = _vec(1)
        self.assertAlmostEqual(face_id._distance(v, v.copy()), 0.0, places=6)

    def test_flat_vector_returns_one(self):
        flat = np.ones(100)
        self.assertEqual(face_id._distance(flat, _vec(2)), 1.0)

    def test_different_is_large(self):
        d = face_id._distance(_vec(3), _vec(4))
        self.assertGreater(d, face_id.MATCH_THRESHOLD)


class TestMatchEncoding(unittest.TestCase):
    def setUp(self):
        self._saved = (face_id._known_encodings, face_id._known_names)

    def tearDown(self):
        face_id._known_encodings, face_id._known_names = self._saved

    def _set_db(self, pairs):
        face_id._known_encodings = [e for _, e in pairs]
        face_id._known_names = [n for n, _ in pairs]

    def test_empty_db_no_match(self):
        self._set_db([])
        name, dist = face_id._match_encoding(_vec(10))
        self.assertIsNone(name)
        self.assertEqual(dist, 1.0)

    def test_exact_self_match(self):
        v = _vec(11)
        self._set_db([("Ali", v.copy()), ("Sara", _vec(12))])
        name, dist = face_id._match_encoding(v)
        self.assertEqual(name, "Ali")
        self.assertLess(dist, face_id.MATCH_THRESHOLD)

    def test_unknown_rejected(self):
        self._set_db([("Ali", _vec(13)), ("Sara", _vec(14))])
        name, _ = face_id._match_encoding(_vec(99))
        self.assertIsNone(name)

    def test_per_driver_collapse_multiple_photos(self):
        """A driver with several photos matches on their BEST photo, and the
        margin is measured against the next DIFFERENT driver (not against
        the same driver's other photo)."""
        query = _vec(20)
        noisy = query + np.random.default_rng(5).random(query.shape) * 0.05
        self._set_db([
            ("Ali", _vec(77)),        # bad photo of Ali (far away)
            ("Ali", noisy),           # good photo of Ali (close)
        ])
        name, _ = face_id._match_encoding(query)
        # With only one driver enrolled, margin vs "2nd driver" defaults to
        # 1.0, so a close match must be accepted.
        self.assertEqual(name, "Ali")

    def test_margin_blocks_close_second(self):
        """When two DIFFERENT drivers' encodings are nearly identical, the
        margin rule rejects everyone - no coin-flip identification."""
        base = _vec(30)
        twin = base + 0.001  # nearly identical encoding for a 2nd driver
        self._set_db([("Ali", base), ("AliTwin", twin)])
        # even an exact reading of Ali is rejected: AliTwin is within margin
        name, _ = face_id._match_encoding(base.copy())
        self.assertIsNone(name)
        # a query exactly between them is too ambiguous as well
        mid = (base + twin) / 2
        name2, _ = face_id._match_encoding(mid)
        self.assertIsNone(name2)


class TestDatabaseVersioning(unittest.TestCase):
    def setUp(self):
        self._saved = (face_id._known_encodings, face_id._known_names)
        self.tmp = os.path.join(os.path.dirname(__file__), "_tmp_test.npz")

    def tearDown(self):
        face_id._known_encodings, face_id._known_names = self._saved
        if os.path.exists(self.tmp):
            os.remove(self.tmp)

    def test_save_load_roundtrip(self):
        face_id._known_encodings = [_vec(40), _vec(41)]
        face_id._known_names = ["Ali", "Sara"]
        face_id.save_database(self.tmp)
        face_id._known_encodings, face_id._known_names = [], []
        self.assertTrue(face_id.load_database(self.tmp))
        self.assertEqual(face_id._known_names, ["Ali", "Sara"])

    def test_old_version_rejected(self):
        # simulate a version-1 database (no version field)
        np.savez_compressed(self.tmp,
                            encodings=np.array([_vec(42)]),
                            names=np.array(["OldGuy"]))
        face_id._known_encodings, face_id._known_names = [], []
        self.assertFalse(face_id.load_database(self.tmp))
        self.assertEqual(face_id._known_names, [])

    def test_empty_save_removes_file(self):
        face_id._known_encodings = [_vec(43)]
        face_id._known_names = ["Temp"]
        face_id.save_database(self.tmp)
        self.assertTrue(os.path.exists(self.tmp))
        face_id._known_encodings, face_id._known_names = [], []
        face_id.save_database(self.tmp)
        self.assertFalse(os.path.exists(self.tmp))


if __name__ == "__main__":
    unittest.main(verbosity=2)
