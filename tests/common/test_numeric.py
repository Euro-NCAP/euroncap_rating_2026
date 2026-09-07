# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for the protocol rounding primitives.

IMPORTANT test-design constraint: decimal-midpoint cases must be
constructed with ``math.nextafter`` or computed expressions, NOT decimal
literals. A literal like ``0.475`` parses to the nearest double, whose
``repr`` is the nice decimal string again — a test written that way passes
even against a broken helper that only fixes the literal case. The
realistic failure shape is a boundary *computed* 1 ulp below the midpoint, which is
a different double.
"""

import math
import unittest

import numpy as np
import pandas as pd

from euroncap_rating_2026.numeric import floor_score, round_half_up


class TestRoundHalfUpMidpoints(unittest.TestCase):
    """Excel-ROUND midpoint behaviour where Python round() diverges."""

    def test_literal_midpoints_round_up(self):
        # All of these round DOWN (half-even on binary) with Python round()
        cases = [
            (0.415, 2, 0.42),
            (0.425, 2, 0.43),
            (2.675, 2, 2.68),
            (0.125, 2, 0.13),
            (0.075, 2, 0.08),
        ]
        for value, ndigits, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(round_half_up(value, ndigits), expected)
                # sanity: this is a real divergence from builtin round()
                self.assertNotEqual(round(value, ndigits), expected)

    def test_computed_midpoint_one_ulp_below(self):
        # The realistic shape: a boundary that is mathematically 0.475 but is
        # computed 1 ulp below. repr() shows 0.4749999999999999, so the
        # a naive repr-based helper returns 0.47 — the guard
        # must absorb the artifact and give 0.48.
        one_ulp_below = math.nextafter(0.475, 0)
        self.assertEqual(round_half_up(one_ulp_below, 2), 0.48)

    def test_computed_midpoint_one_ulp_above(self):
        one_ulp_above = math.nextafter(0.475, 1)
        self.assertEqual(round_half_up(one_ulp_above, 2), 0.48)

    def test_tibia_tolerance_expression(self):
        # Tibia index: HPL 0.4, LPL 1.3 -> interval_step 0.3, tolerance
        # 25% of band width = 0.075 -> half-up 0.08 (Python round(): 0.07).
        interval_step = round_half_up((1.3 - 0.4) / 3, 2)
        self.assertEqual(interval_step, 0.3)
        self.assertEqual(round_half_up(interval_step / 4.0, 2), 0.08)

    def test_genuinely_below_midpoint_stays_down(self):
        # 0.4749 is really below 0.475 — the guard must NOT promote it.
        self.assertEqual(round_half_up(0.4749, 2), 0.47)

    def test_double_rounding_trap(self):
        # Classic double-rounding bug shape: 0.4449 -> (0.445) -> 0.45.
        # The guard precision is far finer than 3 dp, so this must stay 0.44.
        self.assertEqual(round_half_up(0.4449, 2), 0.44)

    def test_negative_midpoint_ties_away_from_zero(self):
        # Excel ROUND(-0.475, 2) == -0.48. An additive-epsilon helper gets
        # this wrong (rounds toward zero).
        self.assertEqual(round_half_up(-0.475, 2), -0.48)
        self.assertEqual(round_half_up(math.nextafter(-0.475, 0), 2), -0.48)
        self.assertEqual(round_half_up(-0.4749, 2), -0.47)


class TestRoundHalfUpGeneral(unittest.TestCase):
    def test_one_ulp_below_integer_snaps(self):
        value = 8.359 + 3.864 + 0.777  # 12.999999999999998
        self.assertNotEqual(value, 13.0)
        self.assertEqual(round_half_up(value, 2), 13.0)
        self.assertEqual(round_half_up(79.99999999999999, 2), 80.0)

    def test_plain_values_unchanged(self):
        self.assertEqual(round_half_up(354.25503, 2), 354.26)
        self.assertEqual(round_half_up(67.66667, 2), 67.67)
        self.assertEqual(round_half_up(0.83333, 4), 0.8333)
        self.assertEqual(round_half_up(8.86667, 3), 8.867)
        self.assertEqual(round_half_up(1.0, 2), 1.0)
        self.assertEqual(round_half_up(0.0, 2), 0.0)

    def test_various_ndigits(self):
        self.assertEqual(round_half_up(0.831333, 5), 0.83133)
        self.assertEqual(round_half_up(0.83333, 5), 0.83333)
        self.assertEqual(round_half_up(2.5, 0), 3.0)
        self.assertEqual(round_half_up(-2.5, 0), -3.0)

    def test_none_and_nan_pass_through(self):
        self.assertIsNone(round_half_up(None, 2))
        self.assertTrue(math.isnan(round_half_up(float("nan"), 2)))
        self.assertTrue(pd.isna(round_half_up(pd.NA, 2)))

    def test_numpy_and_int_inputs(self):
        self.assertEqual(round_half_up(np.float64(0.415), 2), 0.42)
        self.assertEqual(round_half_up(np.float32(1.5), 0), 2.0)
        self.assertEqual(round_half_up(7, 2), 7.0)
        self.assertIsInstance(round_half_up(np.float64(0.415), 2), float)


class TestFloorScore(unittest.TestCase):
    def test_one_ulp_below_integer_sums(self):
        cases = [
            (8.359 + 3.864 + 0.777, 13),  # 12.999999999999998
            (0.732 + 0.834 + 0.434, 2),  # 1.9999999999999998
            (79.99999999999999, 80),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(math.floor(value), expected - 1)  # the bug
                self.assertEqual(floor_score(value), expected)  # the fix

    def test_genuine_fractions_still_floor(self):
        self.assertEqual(floor_score(79.879), 79)
        self.assertEqual(floor_score(79.999), 79)
        self.assertEqual(floor_score(13.0), 13)
        self.assertEqual(floor_score(0.999999), 0)  # genuine 6-dp value below 1
        self.assertEqual(floor_score(0.9999996), 1)  # inside the 6-dp snap

    def test_returns_int(self):
        self.assertIsInstance(floor_score(12.3), int)


if __name__ == "__main__":
    unittest.main()
