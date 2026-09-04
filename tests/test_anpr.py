"""Veri-Drive automated tests — ANPR text helpers + fuzzy plate matching.

Run all tests:   py -m unittest discover tests -v
Note: importing anpr pulls in EasyOCR (slow first import) but no OCR is
executed in these tests — they only cover the pure text/matching logic.
"""
import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "modules"))

import anpr


class TestTextHelpers(unittest.TestCase):
    def test_normalize_text(self):
        self.assertEqual(anpr.normalize_text(" abc·12.34 "), "ABC-12-34")
        self.assertEqual(anpr.normalize_text("leB@@1234"), "LEB 1234")

    def test_clean_plate(self):
        self.assertEqual(anpr.clean_plate("ABC-12 34"), "ABC1234")

    def test_is_likely_plate(self):
        self.assertTrue(anpr.is_likely_plate("ABC-1234"))
        self.assertTrue(anpr.is_likely_plate("LEB1234"))
        self.assertFalse(anpr.is_likely_plate("ABC"))        # too short
        self.assertFalse(anpr.is_likely_plate("12345"))      # no letters
        self.assertFalse(anpr.is_likely_plate("ABCDE"))      # no digits
        self.assertFalse(anpr.is_likely_plate(""))           # empty

    def test_levenshtein(self):
        self.assertEqual(anpr.levenshtein("BDE759", "BDE759"), 0)
        self.assertEqual(anpr.levenshtein("BDE759", "BDE758"), 1)
        self.assertEqual(anpr.levenshtein("ABC1234", "XYZ9999"), 7)
        self.assertEqual(anpr.levenshtein("", "AB"), 2)


class TestMatchPlate(unittest.TestCase):
    PLATES = {"BDE-759", "AYY-911"}

    def test_exact_match(self):
        self.assertEqual(anpr.match_plate("BDE-759", self.PLATES), "BDE-759")
        self.assertEqual(anpr.match_plate("bde759", self.PLATES), "BDE-759")
        self.assertEqual(anpr.match_plate("BDE 759", self.PLATES), "BDE-759")

    def test_fuzzy_match_ocr_noise(self):
        # one noisy character still matches (>= 0.80 similarity)
        self.assertEqual(anpr.match_plate("BDE-750", self.PLATES), "BDE-759")

    def test_no_match(self):
        self.assertIsNone(anpr.match_plate("ZZZ-9999", self.PLATES))
        self.assertIsNone(anpr.match_plate("", self.PLATES))
        self.assertIsNone(anpr.match_plate(None, self.PLATES))

    def test_near_duplicate_ambiguity_rejected(self):
        """BDE-759 vs BDE-758: a reading that fuzzy-matches BOTH must be
        rejected unless it exactly equals one of them (P1 guard)."""
        plates = {"BDE-759", "BDE-758"}
        # exact readings are fine
        self.assertEqual(anpr.match_plate("BDE-759", plates), "BDE-759")
        self.assertEqual(anpr.match_plate("BDE-758", plates), "BDE-758")
        # ambiguous noise reading close to both -> rejected
        self.assertIsNone(anpr.match_plate("BDE-75B", plates))

    def test_empty_registry(self):
        self.assertIsNone(anpr.match_plate("ABC-1234", set()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
