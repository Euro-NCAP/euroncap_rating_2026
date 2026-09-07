# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Regression tests for the "same or better than predicted" rule in
crash_avoidance.test_info.compute_loadcase_score.

The rule credits a verification point whose measured colour is at least as
good as the OEM predicted one. It leans on a local `color_rank` table that
used to give ORANGE and BROWN the same rank, so `rank[computed] <
rank[predicted]` read `2 < 2` for the one pair where the measured outcome is
orange against a predicted brown: a genuinely better result counted as
INCORRECT. With the Extended discount for a `Self claimed` prediction being
all-or-nothing, a single mis-marked point zeroed the whole Extended layer.

The pairs are produced by driving the real ComputedTestPoint /
check_thresholds path with mid-band `v_rel_impact` values, so the tests fail
if the ranking, the thresholds, or the tolerance handling drift apart.
"""

import unittest

from euroncap_rating_2026 import common
from euroncap_rating_2026.crash_avoidance import (
    matrix_processing,
    robustness_layer,
    test_info,
)

# Severity order, best first. Only used to name the pairs; the credited/not
# expectation is derived from TestPoint.predicted_score below rather than
# restated here, so this cannot drift out of sync with the scoring scale.
COLORS = [
    common.PredictionColor.GREEN,
    common.PredictionColor.YELLOW,
    common.PredictionColor.ORANGE,
    common.PredictionColor.BROWN,
    common.PredictionColor.RED,
]

# v_rel_impact values landing mid-band in the VUT >= 50 km/h row of
# mitigation_or_avoidance_thresholds (green <=0, yellow <=10, orange <=20,
# brown <=30, red >30). Mid-band keeps every value at least 5 km/h clear of a
# boundary, so the 2 km/h IN_TOLERANCE fallback never rescues a pair and each
# one exercises the color_rank comparison itself.
MID_BAND_VALUE = {
    common.PredictionColor.GREEN: 0.0,
    common.PredictionColor.YELLOW: 5.0,
    common.PredictionColor.ORANGE: 15.0,
    common.PredictionColor.BROWN: 25.0,
    common.PredictionColor.RED: 40.0,
}

LOADCASE = "CCRm"  # a mitigation-or-avoidance scenario, as in the reported case
VUT_SPEED = "50 km/h"
STANDARD_TOTAL = 2.4  # data_model.TOTAL_SCORES["CCRm"]["Standard"]
EXTENDED_TOTAL = 0.3  # data_model.TOTAL_SCORES["CCRm"]["Extended"]


def make_point(predicted_color, measured_color, test_range):
    """A ComputedTestPoint whose measured colour comes out of the real
    threshold logic, not a hand-set attribute."""
    test_point = matrix_processing.TestPoint(
        row=0,
        col=0,
        color=predicted_color,
        test_range=test_range,
        attributes={"VUT speed": VUT_SPEED},
    )
    return matrix_processing.ComputedTestPoint(
        test_point,
        test_name=LOADCASE,
        value=MID_BAND_VALUE[measured_color],
    )


def score(points, standard_method, extended_method, predicted_extended_score=0.0):
    # A fresh RobustnessLayerScore per call: compute_loadcase_score mutates it.
    return test_info.compute_loadcase_score(
        LOADCASE,
        LOADCASE,
        points,
        standard_method,
        extended_method,
        STANDARD_TOTAL,
        predicted_extended_score,
        robustness_layer.RobustnessLayerScore(tested_score=0.0, constant_score=0.0),
    )


def severity_value(color):
    """The scale the library already uses for a predicted colour: green 1.0 >
    yellow 0.75 > orange 0.5 > brown 0.25 > red 0.0, speed-independent."""
    return matrix_processing.TestPoint(
        row=0,
        col=0,
        color=color,
        test_range=matrix_processing.TestRange.STANDARD,
    ).predicted_score


class TestMidBandFixture(unittest.TestCase):
    """The pairs below are only meaningful if the fixture values really
    produce the colour they are named after, with no tolerance rescue."""

    def test_each_value_produces_its_colour(self):
        for predicted in COLORS:
            for measured in COLORS:
                with self.subTest(predicted=predicted, measured=measured):
                    ctp = make_point(
                        predicted, measured, matrix_processing.TestRange.STANDARD
                    )
                    self.assertEqual(ctp.computed_color, measured)
                    expected = (
                        matrix_processing.PredictionResult.CORRECT
                        if measured == predicted
                        else matrix_processing.PredictionResult.INCORRECT
                    )
                    self.assertEqual(ctp.prediction_result, expected)


class TestAllColorPairs(unittest.TestCase):
    """Walk all 25 (predicted, measured) pairs and assert every pair whose
    measured colour is at least as good as predicted is credited."""

    def credited(self, predicted, measured):
        # One Standard point, VTA: the lookup is {1: {1: 100, 0: 0}}, so the
        # Standard score is the full total when the point is credited and 0
        # otherwise.
        result = score(
            [make_point(predicted, measured, matrix_processing.TestRange.STANDARD)],
            matrix_processing.PredictionSource.VTA,
            matrix_processing.PredictionSource.VTA,
        )
        self.assertIn(result.standard_score, (0.0, STANDARD_TOTAL))
        return result.standard_score == STANDARD_TOTAL

    def test_same_or_better_is_credited_and_worse_is_not(self):
        for predicted in COLORS:
            for measured in COLORS:
                with self.subTest(predicted=predicted, measured=measured):
                    at_least_as_good = severity_value(measured) >= severity_value(
                        predicted
                    )
                    self.assertEqual(
                        self.credited(predicted, measured),
                        at_least_as_good,
                        f"measured {measured.value} against predicted "
                        f"{predicted.value} should "
                        f"{'be' if at_least_as_good else 'not be'} credited",
                    )

    def test_measured_orange_against_predicted_brown_is_credited(self):
        # The one pair the equal ORANGE/BROWN ranks used to miss.
        self.assertTrue(
            self.credited(common.PredictionColor.BROWN, common.PredictionColor.ORANGE)
        )


class TestExtendedLayerRegression(unittest.TestCase):
    """The reported CCRm case: two Extended verification points, prediction
    method `Self claimed`, whose discount lookup is all-or-nothing
    ({2: {2: 100, 1: 0, 0: 0}}). Miscounting the second point zeroed the
    banded 0.15 Extended score outright."""

    PREDICTED_EXTENDED_SCORE = 0.15  # 14/22 * 0.3 = 0.191, banded to 0.15

    def extended_score(self):
        points = [
            # 60/20 at impact location 125%: predicted green, avoided.
            make_point(
                common.PredictionColor.GREEN,
                common.PredictionColor.GREEN,
                matrix_processing.TestRange.EXTENDED,
            ),
            # 90/30 at impact location 125%: predicted brown, measured
            # orange -- better than predicted.
            make_point(
                common.PredictionColor.BROWN,
                common.PredictionColor.ORANGE,
                matrix_processing.TestRange.EXTENDED,
            ),
            # A Standard point so the >=50% Standard gate is met.
            make_point(
                common.PredictionColor.GREEN,
                common.PredictionColor.GREEN,
                matrix_processing.TestRange.STANDARD,
            ),
        ]
        return score(
            points,
            matrix_processing.PredictionSource.VTA,
            matrix_processing.PredictionSource.SELF_CLAIMED,
            predicted_extended_score=self.PREDICTED_EXTENDED_SCORE,
        )

    def test_extended_layer_keeps_its_score(self):
        result = self.extended_score()
        self.assertAlmostEqual(result.standard_score, STANDARD_TOTAL)
        self.assertAlmostEqual(result.extended_score, self.PREDICTED_EXTENDED_SCORE)
        self.assertLessEqual(result.extended_score, EXTENDED_TOTAL)
        self.assertFalse(result.robustness_layer_failed)


if __name__ == "__main__":
    unittest.main()
