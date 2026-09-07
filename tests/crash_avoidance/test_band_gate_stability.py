# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Regression tests for the extended-range bands and >=50% robustness
gates in crash_avoidance.test_info (float rounding at the band edge,
failure mode 3).

Edge cases are constructed with math.nextafter so the tested float really
is 1 ulp off the band edge — a decimal literal parses to the nearest
double of the edge itself and would pass even against the old exact
comparisons.
"""

import math
import unittest

from euroncap_rating_2026 import common
from euroncap_rating_2026.crash_avoidance import (
    data_model,
    matrix_processing,
    test_info,
)
from euroncap_rating_2026.crash_avoidance.test_info import (
    apply_extended_range_bands,
    meets_half_total,
)


class TestApplyExtendedRangeBands(unittest.TestCase):
    # Real Extended totals from data_model.TOTAL_SCORES: 0.25 is binary
    # exact, 0.15 is not — both must band identically.
    TOTALS = (0.25, 0.15)

    def test_exact_band_edges(self):
        for total in self.TOTALS:
            with self.subTest(total=total):
                self.assertEqual(apply_extended_range_bands(0.0, total), 0)
                self.assertEqual(
                    apply_extended_range_bands(0.5 * total, total), 0.5 * total
                )
                self.assertEqual(
                    apply_extended_range_bands(0.75 * total, total), 0.75 * total
                )
                self.assertEqual(apply_extended_range_bands(total, total), total)

    def test_one_ulp_below_edges_snaps_into_band(self):
        # A score that is "really" on the edge but computed 1 ulp below
        # must land in the band above, not the band below.
        for total in self.TOTALS:
            with self.subTest(total=total):
                self.assertEqual(
                    apply_extended_range_bands(math.nextafter(0.5 * total, 0), total),
                    0.5 * total,
                )
                self.assertEqual(
                    apply_extended_range_bands(math.nextafter(0.75 * total, 0), total),
                    0.75 * total,
                )
                self.assertEqual(
                    apply_extended_range_bands(math.nextafter(total, 0), total),
                    total,
                )

    def test_one_ulp_above_total_is_clamped(self):
        # The old chain ended with an exact == total and no else: a score
        # 1 ulp above the total passed through unsnapped.
        for total in self.TOTALS:
            with self.subTest(total=total):
                self.assertEqual(
                    apply_extended_range_bands(math.nextafter(total, 1), total),
                    total,
                )

    def test_genuinely_inside_bands_unaffected(self):
        for total in self.TOTALS:
            with self.subTest(total=total):
                self.assertEqual(apply_extended_range_bands(0.49 * total, total), 0)
                self.assertEqual(
                    apply_extended_range_bands(0.6 * total, total), 0.5 * total
                )
                self.assertEqual(
                    apply_extended_range_bands(0.9 * total, total), 0.75 * total
                )

    def test_zero_total(self):
        # Loadcases without an Extended range use total 0.0.
        self.assertEqual(apply_extended_range_bands(0, 0.0), 0.0)


class TestMeetsHalfTotal(unittest.TestCase):
    # Real Standard totals: 2.0 (binary exact) and 1.2 (not exact).
    TOTALS = (2.0, 1.2)

    def test_exactly_half_passes(self):
        for total in self.TOTALS:
            with self.subTest(total=total):
                self.assertTrue(meets_half_total(0.5 * total, total))

    def test_one_ulp_below_half_passes(self):
        # 0.4999999...*total is "really" half the total — must not be zeroed.
        for total in self.TOTALS:
            with self.subTest(total=total):
                self.assertTrue(meets_half_total(math.nextafter(0.5 * total, 0), total))

    def test_genuinely_below_half_fails(self):
        for total in self.TOTALS:
            with self.subTest(total=total):
                self.assertFalse(meets_half_total(0.49 * total, total))
                self.assertFalse(meets_half_total(0.0, total))

    def test_above_half_passes(self):
        for total in self.TOTALS:
            with self.subTest(total=total):
                self.assertTrue(meets_half_total(0.51 * total, total))
                self.assertTrue(meets_half_total(total, total))


class TestComputePredictedScoreUsesBands(unittest.TestCase):
    """End-to-end sanity: compute_predicted_score still bands via the helper."""

    def _make_points(self, extended_colors):
        extended_cells = data_model.EXTENDED_RANGE_CELLS["CCFhos"]
        extended_set = set(extended_cells)
        points = []
        ext_iter = iter(extended_colors)
        for r in range(8):
            for c in range(4):
                if (r, c) in extended_set:
                    color = next(ext_iter, common.PredictionColor.GREEN)
                    test_range = matrix_processing.TestRange.EXTENDED
                else:
                    color = common.PredictionColor.GREEN
                    test_range = matrix_processing.TestRange.STANDARD
                points.append(
                    matrix_processing.TestPoint(
                        row=r, col=c, color=color, test_range=test_range
                    )
                )
        return points

    def test_all_green_extended_hits_full_total(self):
        # Exercises the final else (clamp) branch: score == total.
        points = self._make_points([common.PredictionColor.GREEN] * 14)
        _, extended_score = test_info.compute_predicted_score("CCFhos", points)
        self.assertEqual(extended_score, 0.25)

    def test_half_green_extended_snaps_to_half_total(self):
        points = self._make_points(
            [common.PredictionColor.GREEN] * 7 + [common.PredictionColor.RED] * 7
        )
        _, extended_score = test_info.compute_predicted_score("CCFhos", points)
        self.assertEqual(extended_score, 0.125)


if __name__ == "__main__":
    unittest.main()
