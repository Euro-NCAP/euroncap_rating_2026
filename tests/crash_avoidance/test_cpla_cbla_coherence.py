# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Regression tests: check_cpla_cbla_coherence was written for
a template revision where CPLA/CBLA "Function" was hardcoded per sub-block.
Now that the second sub-block's Function is a grey-cell AEB/FCW dropdown,
selecting AEB there legitimately produces a duplicate test point (same VUT
speed/target speed/Function/Day-Night/impact location as the fixed AEB
sub-block above it). The old check only ever inspected the first of a
duplicate pair and whitelisted GREEN/RED there alone, instead of comparing
the pair's colors to each other -- so a legitimate, self-consistent AEB
declaration could never pass. It also re-selected test points by bare
`tp.row`, which pulls in unrelated impact-location columns sharing that row
(CPLA/CBLA matrices are 4 columns wide) instead of just the actual duplicate.
"""

import unittest

from euroncap_rating_2026 import common
from euroncap_rating_2026.crash_avoidance import matrix_processing
from euroncap_rating_2026.crash_avoidance import test_info

GREEN = common.PredictionColor.GREEN
RED = common.PredictionColor.RED
BROWN = common.PredictionColor.BROWN
GREY = common.PredictionColor.GREY


def _tp(row, col, color, **attributes):
    return matrix_processing.TestPoint(
        row=row,
        col=col,
        color=color,
        attributes=attributes,
    )


class TestFindDuplicateRows(unittest.TestCase):
    def test_no_duplicates_returns_empty(self):
        points = [
            _tp(0, 0, GREEN, vut_speed="50 km/h", function="AEB"),
            _tp(1, 0, GREEN, vut_speed="60 km/h", function="AEB"),
        ]
        self.assertEqual(test_info.find_duplicate_rows(points), [])

    def test_same_row_different_impact_location_is_not_a_duplicate(self):
        """Two TestPoints can share a bare `row` index across different
        impact-location columns; unless their full attributes (including
        impact location) match, they must not be grouped together."""
        points = [
            _tp(
                4, 0, GREEN, vut_speed="50 km/h", function="AEB", impact_location="25%"
            ),
            _tp(
                4, 1, BROWN, vut_speed="50 km/h", function="AEB", impact_location="50%"
            ),
        ]
        self.assertEqual(test_info.find_duplicate_rows(points), [])

    def test_matching_attributes_across_rows_are_grouped(self):
        points = [
            _tp(
                4, 2, GREEN, vut_speed="50 km/h", function="AEB", impact_location="25%"
            ),
            _tp(
                6, 2, GREEN, vut_speed="50 km/h", function="AEB", impact_location="25%"
            ),
        ]
        groups = test_info.find_duplicate_rows(points)
        self.assertEqual(len(groups), 1)
        self.assertCountEqual(groups[0], points)

    def test_three_way_duplicate_group_keeps_every_member(self):
        points = [
            _tp(4, 0, GREEN, vut_speed="50 km/h", function="AEB"),
            _tp(6, 0, GREEN, vut_speed="50 km/h", function="AEB"),
            _tp(8, 0, GREEN, vut_speed="50 km/h", function="AEB"),
        ]
        groups = test_info.find_duplicate_rows(points)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 3)
        self.assertCountEqual(groups[0], points)


class TestCheckCplaCblaCoherence(unittest.TestCase):
    def test_no_duplicates_is_coherent(self):
        points = [
            _tp(0, 0, GREEN, vut_speed="50 km/h", function="AEB"),
            _tp(1, 0, BROWN, vut_speed="60 km/h", function="AEB"),
        ]
        coherent, mismatches = test_info.check_cpla_cbla_coherence(points)
        self.assertTrue(coherent)
        self.assertEqual(mismatches, [])

    def test_duplicate_pair_with_matching_colors_is_coherent(self):
        """Realistic case: a legitimate 'AEB up to 80 km/h' declaration
        makes the choice-band row duplicate the fixed AEB-band row; as long
        as both rows agree, this must pass -- even though both are BROWN,
        which the old GREEN/RED-only whitelist would have rejected."""
        points = [
            _tp(
                4, 2, BROWN, vut_speed="50 km/h", function="AEB", impact_location="25%"
            ),
            _tp(
                6, 2, BROWN, vut_speed="50 km/h", function="AEB", impact_location="25%"
            ),
        ]
        coherent, mismatches = test_info.check_cpla_cbla_coherence(points)
        self.assertTrue(coherent)
        self.assertEqual(mismatches, [])

    def test_duplicate_pair_with_disagreeing_colors_is_incoherent(self):
        points = [
            _tp(
                4, 2, BROWN, vut_speed="50 km/h", function="AEB", impact_location="25%"
            ),
            _tp(
                6, 2, GREEN, vut_speed="50 km/h", function="AEB", impact_location="25%"
            ),
        ]
        coherent, mismatches = test_info.check_cpla_cbla_coherence(points)
        self.assertFalse(coherent)
        self.assertEqual(len(mismatches), 1)
        message = mismatches[0]
        self.assertIn("row=4, col=2", message)
        self.assertIn("row=6, col=2", message)
        self.assertIn(BROWN.value, message)
        self.assertIn(GREEN.value, message)

    def test_grey_member_is_ignored_remaining_members_still_compared(self):
        points = [
            _tp(4, 2, GREY, vut_speed="50 km/h", function="AEB", impact_location="25%"),
            _tp(
                6, 2, GREEN, vut_speed="50 km/h", function="AEB", impact_location="25%"
            ),
        ]
        coherent, mismatches = test_info.check_cpla_cbla_coherence(points)
        self.assertTrue(coherent)
        self.assertEqual(mismatches, [])

    def test_grey_member_does_not_hide_a_real_disagreement(self):
        points = [
            _tp(4, 2, GREY, vut_speed="50 km/h", function="AEB", impact_location="25%"),
            _tp(
                6, 2, BROWN, vut_speed="50 km/h", function="AEB", impact_location="25%"
            ),
            _tp(
                8, 2, GREEN, vut_speed="50 km/h", function="AEB", impact_location="25%"
            ),
        ]
        coherent, mismatches = test_info.check_cpla_cbla_coherence(points)
        self.assertFalse(coherent)
        self.assertEqual(len(mismatches), 1)


if __name__ == "__main__":
    unittest.main()
