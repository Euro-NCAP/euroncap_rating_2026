# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
The 2 km/h impact-speed tolerance of Crash Avoidance - Frontal Collisions v1.2
sec 4.2.4.

Two things are pinned here:

* the accepted ranges of the worked table in that section, which the tolerance
  window must reproduce exactly. Before the fix both branches widened the
  predicted band *inward*, so no value anywhere reached IN_TOLERANCE and every
  "accepted" case below returned INCORRECT;
* that the tolerance is scoped to the v_rel_impact KPI. The remaining criteria
  score a speed reduction, a TTC in seconds, a distance in metres or a 0/1
  impact flag, so a global 2.0 would read as 2 seconds / 2 metres / 2 impacts
  and rescue predictions it must not;
* that it decides prediction accuracy only. sec 4.2.4 applies the OEM's
  predicted colour to a within-tolerance point, but the point's *true* colour
  stays the measured band -- so an absolute performance criterion (NOT_RED)
  must not be swayed by the prediction.
"""

import unittest

from euroncap_rating_2026 import common
from euroncap_rating_2026.crash_avoidance import data_model
from euroncap_rating_2026.crash_avoidance import matrix_processing
from euroncap_rating_2026.crash_avoidance import robustness_layer
from euroncap_rating_2026.crash_avoidance import test_info

GREEN = common.PredictionColor.GREEN
YELLOW = common.PredictionColor.YELLOW
ORANGE = common.PredictionColor.ORANGE
BROWN = common.PredictionColor.BROWN
RED = common.PredictionColor.RED

CORRECT = matrix_processing.PredictionResult.CORRECT
INCORRECT = matrix_processing.PredictionResult.INCORRECT
IN_TOLERANCE = matrix_processing.PredictionResult.IN_TOLERANCE


def compute(test_name, predicted, value, attributes=None, assessment_criteria=None):
    """One computed test point, as get_test_matrix_from_region would build it."""
    point_attributes = {"VUT speed": "60 km/h"}
    point_attributes.update(attributes or {})
    test_point = matrix_processing.TestPoint(
        row=1,
        col=2,
        color=predicted,
        test_range=matrix_processing.TestRange.STANDARD,
        attributes=point_attributes,
    )
    return matrix_processing.ComputedTestPoint(
        test_point=test_point,
        test_name=test_name,
        value=value,
        assessment_criteria=assessment_criteria,
    )


class TestImpactSpeedToleranceWorkedTable(unittest.TestCase):
    """
    sec 4.2.4's worked example, the 60 km/h CMRs matrix (v_rel_impact bands
    green v = 0, yellow <= 10, orange <= 20, brown <= 30, red > 30):

        Colour   Impact speed      Accepted range
        Green    v = 0             v < 2
        Yellow   0 < v <= 10       0 < v <= 12
        Orange   10 < v <= 20      8 < v <= 22
        Brown    20 < v <= 30      18 < v <= 32
        Red      30 < v            -
    """

    TEST_NAME = "CMRs"

    # (predicted colour, v_rel_impact, expected result, applied colour, true colour)
    CASES = [
        # In-band: the plain comparison, no tolerance involved.
        (GREEN, 0.0, CORRECT, GREEN, GREEN),
        (YELLOW, 5.0, CORRECT, YELLOW, YELLOW),
        (RED, 40.0, CORRECT, RED, RED),
        # Green accepts v < 2 strictly, as the table prints it.
        (GREEN, 1.9, IN_TOLERANCE, GREEN, YELLOW),
        (GREEN, 2.0, INCORRECT, YELLOW, YELLOW),
        # Yellow: 0 < v <= 12. The lower edge stops at 0 rather than -2, so a
        # fully avoided point keeps its own (better) Green colour.
        (YELLOW, 0.0, INCORRECT, GREEN, GREEN),
        (YELLOW, 12.0, IN_TOLERANCE, YELLOW, ORANGE),
        (YELLOW, 12.1, INCORRECT, ORANGE, ORANGE),
        # Orange: 8 < v <= 22.
        (ORANGE, 8.0, INCORRECT, YELLOW, YELLOW),
        (ORANGE, 8.1, IN_TOLERANCE, ORANGE, YELLOW),
        (ORANGE, 22.0, IN_TOLERANCE, ORANGE, BROWN),
        (ORANGE, 22.1, INCORRECT, BROWN, BROWN),
        # Brown: 18 < v <= 32.
        (BROWN, 18.0, INCORRECT, ORANGE, ORANGE),
        (BROWN, 18.1, IN_TOLERANCE, BROWN, ORANGE),
        (BROWN, 32.0, IN_TOLERANCE, BROWN, RED),
        (BROWN, 32.1, INCORRECT, RED, RED),
        # Red is unbounded above; the tolerance still applies downward, since
        # sec 4.2.4 applies it "in both directions".
        (RED, 28.1, IN_TOLERANCE, RED, BROWN),
        (RED, 27.9, INCORRECT, BROWN, BROWN),
    ]

    def test_accepted_ranges(self):
        for predicted, value, expected_result, applied, true_color in self.CASES:
            with self.subTest(predicted=predicted, value=value):
                ctp = compute(self.TEST_NAME, predicted, value)
                self.assertEqual(ctp.prediction_result, expected_result)
                self.assertEqual(ctp.computed_color, applied)
                self.assertEqual(ctp.measured_color, true_color)

    def test_true_colour_never_uses_the_tolerance(self):
        """measured_color is the plain band lookup for every case, tolerance
        or not: sec 4.2.4 determines it "without applying a tolerance"."""
        for predicted, value, _result, _applied, true_color in self.CASES:
            with self.subTest(predicted=predicted, value=value):
                self.assertEqual(
                    true_color,
                    matrix_processing.get_band_color(
                        [
                            (GREEN, 0.0),
                            (YELLOW, 10.0),
                            (ORANGE, 20.0),
                            (BROWN, 30.0),
                            (RED, float("inf")),
                        ],
                        value,
                    ),
                )

    def test_tolerance_is_reachable(self):
        """
        Guard against the inverted window coming back: it made IN_TOLERANCE
        unreachable for every threshold table and every predicted colour.
        """
        results = {
            compute(self.TEST_NAME, predicted, value).prediction_result
            for predicted, value, _result, _applied, _true in self.CASES
        }
        self.assertIn(IN_TOLERANCE, results)


class TestToleranceScopedToVRelImpact(unittest.TestCase):
    """
    Every case here would be wrongly rescued to IN_TOLERANCE by a global
    2.0 tolerance, because its KPI is not an impact speed in km/h.
    """

    def test_elk_target_impact_is_not_within_tolerance_of_no_impact(self):
        # impact_occurred, a 0/1 flag: 1 is an actual impact.
        ctp = compute("CC ELK On", GREEN, 1.0)
        self.assertEqual(ctp.prediction_result, INCORRECT)
        self.assertEqual(ctp.computed_color, RED)

    def test_fcw_warning_ttc_has_no_tolerance_in_seconds(self):
        # fcw_ttc, seconds: red at <= 1.7 s, so 1.6 s is a late warning and
        # 2 "km/h" must not become 2 seconds here.
        ctp = compute("CPLA day", GREEN, 1.6, attributes={"Function": "FCW"})
        self.assertEqual(ctp.prediction_result, INCORRECT)
        self.assertEqual(ctp.computed_color, RED)

    def test_road_edge_distance_has_no_tolerance_in_metres(self):
        # dtle_t_end, metres: red at <= -0.1 m.
        ctp = compute("ELK RE", GREEN, -0.5)
        self.assertEqual(ctp.prediction_result, INCORRECT)
        self.assertEqual(ctp.computed_color, RED)

    def test_avoidance_impact_speed_has_no_tolerance(self):
        # v_impact, km/h, but the band is green = 0 / red > 0: any impact is
        # red, tolerance or not.
        ctp = compute("CCFtap", GREEN, 1.5)
        self.assertEqual(ctp.prediction_result, INCORRECT)
        self.assertEqual(ctp.computed_color, RED)

    def test_mitigation_speed_reduction_has_no_tolerance(self):
        # v_reduction, km/h, a different KPI (Figure 5-2): red below 10 km/h
        # of reduction, so 11 km/h is orange and stays orange.
        ctp = compute("CCFhos", RED, 11.0)
        self.assertEqual(ctp.prediction_result, INCORRECT)
        self.assertEqual(ctp.computed_color, ORANGE)


class TestNotRedReadsTheTrueColour(unittest.TestCase):
    """
    The NOT_RED robustness criterion is an absolute performance bar, so the
    tolerance must not reach it: sec 4.2.4 scopes the tolerance to verifying
    the prediction, and the point's true colour is the measured band.

    Both cases below are within 2 km/h of the brown/red boundary of CCRb at
    VUT 60 km/h, i.e. exactly where the applied colour and the true colour
    disagree.
    """

    LOADCASE = "CCRb"
    STANDARD_TOTAL = 1.0

    def score(self, point):
        # One Standard point, VTA: the lookup is {1: {1: 100, 0: 0}}, so the
        # Standard score is the full total when the point is credited and 0
        # otherwise, and a failed Standard point collapses the robustness
        # layer to its constant score.
        layer_score = robustness_layer.RobustnessLayerScore(
            tested_score=1.0, constant_score=0.0
        )
        return test_info.compute_loadcase_score(
            self.LOADCASE,
            self.LOADCASE,
            [point],
            matrix_processing.PredictionSource.VTA,
            matrix_processing.PredictionSource.VTA,
            self.STANDARD_TOTAL,
            0.0,
            layer_score,
        )

    def test_measured_red_fails_even_when_brown_was_predicted(self):
        point = compute(
            self.LOADCASE,
            BROWN,
            32.0,
            assessment_criteria=robustness_layer.AssessmentCriteria.NOT_RED,
        )
        # The prediction itself is accepted -- brown, within tolerance ...
        self.assertEqual(point.prediction_result, IN_TOLERANCE)
        self.assertEqual(point.computed_color, BROWN)
        # ... but the run really produced a red outcome.
        self.assertEqual(point.measured_color, RED)
        self.assertEqual(self.score(point).standard_score, 0.0)

    def test_measured_brown_passes_even_when_red_was_predicted(self):
        point = compute(
            self.LOADCASE,
            RED,
            29.0,
            assessment_criteria=robustness_layer.AssessmentCriteria.NOT_RED,
        )
        self.assertEqual(point.prediction_result, IN_TOLERANCE)
        self.assertEqual(point.computed_color, RED)
        self.assertEqual(point.measured_color, BROWN)
        self.assertEqual(self.score(point).standard_score, self.STANDARD_TOTAL)

    def test_criterion_is_unmoved_by_the_tolerance(self):
        """Neither verdict differs from the one the plain band lookup gives,
        which is what the criterion used before the tolerance existed."""
        for predicted, value in [(BROWN, 32.0), (RED, 29.0), (GREEN, 1.5)]:
            with self.subTest(predicted=predicted, value=value):
                point = compute(
                    self.LOADCASE,
                    predicted,
                    value,
                    assessment_criteria=robustness_layer.AssessmentCriteria.NOT_RED,
                )
                credited = self.score(point).standard_score == self.STANDARD_TOTAL
                self.assertEqual(credited, point.measured_color != RED)


class TestToleranceMap(unittest.TestCase):
    def test_every_scored_criteria_declares_a_tolerance(self):
        """Same key set as the KPI map, so a new criteria cannot be scored
        without stating whether its KPI has a tolerance."""
        self.assertEqual(
            set(data_model.NCAP_TEST_CRITERIA_TO_TOLERANCE),
            set(data_model.NCAP_TEST_CRITERIA_TO_SCORE_PARAMETER),
        )

    def test_only_v_rel_impact_has_a_non_zero_tolerance(self):
        non_zero = {
            criteria
            for criteria, tolerance in data_model.NCAP_TEST_CRITERIA_TO_TOLERANCE.items()
            if tolerance
        }
        self.assertEqual(
            non_zero, {data_model.NcapTestCriteria.MITIGATION_OR_AVOIDANCE}
        )
        self.assertEqual(
            data_model.NCAP_TEST_CRITERIA_TO_TOLERANCE[
                data_model.NcapTestCriteria.MITIGATION_OR_AVOIDANCE
            ],
            matrix_processing.COLOR_THRESHOLD_TOLERANCE,
        )

    def test_zero_tolerance_window_is_a_no_op(self):
        """
        With tolerance 0 the widened window collapses to the predicted band,
        which check_thresholds has already returned CORRECT for -- so a
        0-tolerance criteria can never reach IN_TOLERANCE.
        """
        thresholds = [
            (GREEN, 0.0),
            (YELLOW, 10.0),
            (ORANGE, 20.0),
            (BROWN, 30.0),
            (RED, float("inf")),
        ]
        for predicted in (GREEN, YELLOW, ORANGE, BROWN, RED):
            value = 0.0
            while value <= 45.0:
                result, _ = matrix_processing.check_thresholds(
                    thresholds, predicted, value, 0.0
                )
                with self.subTest(predicted=predicted, value=value):
                    self.assertNotEqual(result, IN_TOLERANCE)
                value = round(value + 0.1, 2)


if __name__ == "__main__":
    unittest.main()
