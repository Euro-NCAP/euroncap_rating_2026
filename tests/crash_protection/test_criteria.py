# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import math
import unittest
from euroncap_rating_2026.crash_protection.criteria import Criteria, CriteriaType
import pandas as pd
import logging


class TestCriteria(unittest.TestCase):
    def setUp(self):
        self.row = {
            "Criteria": "Test",
            "HPL": 50.0,
            "LPL": 100.0,
            "Capping": 90.0,
            "Value": 75.0,
        }
        self.criteria = Criteria.get_criteria_from_row(self.row)

    def test_initialization(self):
        self.assertEqual(self.criteria.name, "Test")
        self.assertEqual(self.criteria.hpl, 50.0)
        self.assertEqual(self.criteria.lpl, 100.0)
        self.assertEqual(self.criteria.capping_value, 90.0)
        self.assertEqual(self.criteria.value, 75.0)

    def test_calculate_color_score(self):
        self.criteria.calculate_color_score()
        self.assertEqual(self.criteria.color, "orange")
        self.assertEqual(self.criteria.score, 40.0)

    def test_set_value(self):
        self.criteria.set_value(95.0)
        self.assertEqual(self.criteria.value, 95.0)
        self.assertTrue(self.criteria.capping)
        self.assertEqual(self.criteria.color, "brown")
        self.assertEqual(self.criteria.score, 20.0)

    def test_set_prediction_before_and_after(self):
        self.assertIsNone(self.criteria.prediction)
        self.criteria.set_prediction("yellow")
        self.assertEqual(self.criteria.prediction, "yellow")

    def test_set_capping_value(self):
        self.criteria.set_capping_value(80.0)
        self.assertEqual(self.criteria.capping_value, 80.0)
        self.assertFalse(self.criteria.capping)

    def test_set_prediction(self):
        self.criteria.set_prediction("green")
        self.assertEqual(self.criteria.prediction, "green")

    def test_value_at_green_yellow_threshold(self):
        self.criteria.set_value(self.criteria.green_yellow_threshold)
        self.criteria.calculate_color_score()
        self.assertEqual(self.criteria.color, "yellow")
        self.assertEqual(self.criteria.score, 80.0)

    def test_value_at_yellow_orange_threshold(self):
        self.criteria.set_value(self.criteria.yellow_orange_threshold)
        self.criteria.calculate_color_score()
        self.assertEqual(self.criteria.color, "orange")
        self.assertEqual(self.criteria.score, 40.0)

    def test_value_at_orange_brown_threshold(self):
        self.criteria.set_value(self.criteria.orange_brown_threshold)
        self.criteria.calculate_color_score()
        self.assertEqual(self.criteria.color, "brown")
        self.assertEqual(self.criteria.score, 20.0)

    def test_value_at_brown_red_threshold(self):
        self.criteria.set_value(self.criteria.brown_red_threshold)
        self.criteria.calculate_color_score()
        self.assertEqual(self.criteria.color, "red")
        self.assertEqual(self.criteria.score, 0.0)

    def test_value_just_below_green_yellow_threshold(self):
        self.criteria.set_value(self.criteria.green_yellow_threshold - 0.01)
        self.criteria.calculate_color_score()
        self.assertEqual(self.criteria.color, "green")
        self.assertEqual(self.criteria.score, 100.0)

    def test_value_just_above_green_yellow_threshold(self):
        self.criteria.set_value(self.criteria.green_yellow_threshold + 0.01)
        self.criteria.calculate_color_score()
        self.assertEqual(self.criteria.color, "yellow")
        self.assertEqual(self.criteria.score, 80.0)

    def test_value_with_more_digits(self):
        self.criteria.set_value(75.12345)
        self.criteria.calculate_color_score()
        self.assertEqual(self.criteria.color, "orange")
        self.assertEqual(self.criteria.score, 40.0)

    def test_value_just_below_yellow_orange_threshold(self):
        self.criteria.set_value(self.criteria.yellow_orange_threshold - 0.01)
        self.criteria.calculate_color_score()
        self.assertEqual(self.criteria.color, "yellow")
        self.assertEqual(self.criteria.score, 80.0)

    def test_value_just_above_yellow_orange_threshold(self):
        self.criteria.set_value(self.criteria.yellow_orange_threshold + 0.01)
        self.criteria.calculate_color_score()
        self.assertEqual(self.criteria.color, "orange")
        self.assertEqual(self.criteria.score, 40.0)

    def test_value_just_below_orange_brown_threshold(self):
        self.criteria.set_value(self.criteria.orange_brown_threshold - 0.01)
        self.criteria.calculate_color_score()
        self.assertEqual(self.criteria.color, "orange")
        self.assertEqual(self.criteria.score, 40.0)

    def test_value_just_above_orange_brown_threshold(self):
        self.criteria.set_value(self.criteria.orange_brown_threshold + 0.01)
        self.criteria.calculate_color_score()
        self.assertEqual(self.criteria.color, "brown")
        self.assertEqual(self.criteria.score, 20.0)

    def test_value_just_below_brown_red_threshold(self):
        self.criteria.set_value(self.criteria.brown_red_threshold - 0.01)
        self.criteria.calculate_color_score()
        self.assertEqual(self.criteria.color, "brown")
        self.assertEqual(self.criteria.score, 20.0)

    def test_value_just_above_brown_red_threshold(self):
        self.criteria.set_value(self.criteria.brown_red_threshold + 0.01)
        self.criteria.calculate_color_score()
        self.assertEqual(self.criteria.color, "red")
        self.assertEqual(self.criteria.score, 0.0)

    def test_prediction_correct(self):
        self.criteria.set_value(75.0)
        self.criteria.set_prediction("orange")
        self.criteria.calculate_color_score()
        self.assertEqual(self.criteria.prediction_result, "Correct")
        self.assertEqual(self.criteria.color, "orange")
        self.assertEqual(self.criteria.score, 40.0)

    def test_prediction_in_tolerance(self):
        self.criteria.set_value(51.0)
        self.criteria.set_prediction("green")
        self.criteria.calculate_color_score()
        self.assertEqual(self.criteria.prediction_result, "In Tolerance")
        self.assertEqual(self.criteria.color, "green")
        self.assertEqual(self.criteria.score, 100.0)

    def test_prediction_incorrect(self):
        self.criteria.set_value(75.0)
        self.criteria.set_prediction("green")
        self.criteria.calculate_color_score()
        self.assertEqual(self.criteria.prediction_result, "Incorrect")
        self.assertEqual(self.criteria.color, "orange")
        self.assertEqual(self.criteria.score, 40.0)

    def test_zero_switch_criteria(self):
        row = {
            "Criteria": "Zero Switch Test",
            "HPL": 80.0,
            "LPL": 80.0,
            "Capping": 90.0,
            "Value": 0.0,
        }
        criteria = Criteria.get_criteria_from_row(row)
        criteria.criteria_type = CriteriaType.ZERO_SWITCH

        criteria.set_value(50.0)
        criteria.calculate_color_score()
        self.assertEqual(criteria.score, 100.0)

        criteria.set_value(150.0)
        criteria.calculate_color_score()
        self.assertEqual(criteria.score, 0.0)

    def test_zero_switch_criteria_boundary_is_inclusive(self):
        # Euro NCAP protocol documents the Backset pass condition as
        # (ΔCP X) <= (ΔCP X)LIMIT, so a value exactly equal to the limit
        # must PASS (green/100.0), not FAIL.
        row = {
            "Criteria": "Backset - Mid",
            "HPL": 331.2,
            "LPL": 331.2,
            "Value": 331.2,
        }
        criteria = Criteria.get_criteria_from_row(row)
        self.assertEqual(criteria.criteria_type, CriteriaType.ZERO_SWITCH)
        self.assertEqual(criteria.color, "green")
        self.assertEqual(criteria.score, 100.0)

    def test_none_switch_criteria(self):
        row = {
            "Criteria": "None Switch Test",
            "LPL": 80.0,
            "HPL": None,
            "Capping": 80.0,
            "Value": 0.0,
        }
        criteria = Criteria.get_criteria_from_row(row)
        criteria.criteria_type = CriteriaType.NONE_SWITCH

        criteria.set_value(50.0)
        criteria.calculate_color_score()
        self.assertEqual(criteria.score, 100.0)

        criteria.set_value(150.0)
        criteria.calculate_color_score()
        self.assertIsNone(criteria.score)

    def test_standard_criteria(self):
        row = {
            "Criteria": "Standard Test",
            "HPL": 50.0,
            "LPL": 100.0,
            "Capping": 90.0,
            "Value": 75.0,
        }
        criteria = Criteria.get_criteria_from_row(row)
        criteria.criteria_type = CriteriaType.CRITERIA

        criteria.set_value(40.0)
        criteria.calculate_color_score()
        self.assertEqual(criteria.score, 100.0)

        criteria.set_value(120.0)
        criteria.calculate_color_score()
        self.assertEqual(criteria.score, 0.0)

    def test_unknown_with_capping(self):
        row = {
            "Criteria": "Unknown with Capping Test",
            "HPL": None,
            "LPL": None,
            "Capping": 90.0,
            "Value": 75.0,
        }
        criteria = Criteria.get_criteria_from_row(row)
        criteria.criteria_type = CriteriaType.UNKNOWN

        criteria.calculate_color_score()
        self.assertEqual(criteria.capping_value, 90.0)
        self.assertFalse(criteria.capping)
        criteria.set_value(120.0)
        self.assertTrue(criteria.capping)


class TestNoneSwitchColour(unittest.TestCase):
    """
    Verify that NONE_SWITCH criteria emit 'green' / 'red' colours.

    Rule: value < lpl → green (score 100), value >= lpl → red (score None).
    """

    def _make(self, lpl: float) -> Criteria:
        return Criteria(
            name="Ares",
            lpl=lpl,
            criteria_type=CriteriaType.NONE_SWITCH,
        )

    def test_below_lpl_is_green(self):
        c = self._make(80.0)
        c.set_value(50.0)
        self.assertEqual(c.color, "green")
        self.assertEqual(c.score, 100.0)

    def test_at_lpl_is_red(self):
        c = self._make(80.0)
        c.set_value(80.0)
        self.assertEqual(c.color, "red")
        self.assertIsNone(c.score)

    def test_above_lpl_is_red(self):
        c = self._make(80.0)
        c.set_value(100.0)
        self.assertEqual(c.color, "red")
        self.assertIsNone(c.score)

    def test_nan_value_leaves_colour_none(self):
        c = self._make(80.0)
        c.value = float("nan")
        c.calculate_color_score()
        self.assertIsNone(c.color)


class TestSetValueNan(unittest.TestCase):
    """set_value(NaN) must be accepted and blank the criteria.

    The A-pillar resolution pass (_resolve_headform_apillar_values) calls
    set_value(float("nan")) to force an untested run to score 0 instead of
    green; round_half_up passes NaN through and calculate_color_score maps
    it to color=None / score=0.0. This pins that contract.
    """

    def test_set_value_nan_scores_zero(self):
        c = Criteria(
            name="HIC15",
            hpl=650.0,
            lpl=1700.0,
            value=500.0,
            criteria_type=CriteriaType.CRITERIA,
        )
        c.calculate_color_score()
        self.assertEqual(c.color, "green")

        c.set_value(float("nan"))
        self.assertTrue(math.isnan(c.value))
        self.assertIsNone(c.color)
        self.assertEqual(c.score, 0.0)


class TestTibiaHalfUpRegression(unittest.TestCase):
    """Regression tests for the FW tibia index half-up rounding case.

    Tibia index has HPL 0.4 / LPL 1.3 (Frontal Impact protocol v1.2
    section 3.5.4), so interval_step = 0.3 and the prediction tolerance is
    25% of the colour band width = 0.075, which must round half-up to
    0.08 (Python's binary round() gives 0.07). With a green prediction
    and a measured TI of 0.47, the tolerance window must end at
    0.4 + 0.08 = 0.48, so 0.47 is In Tolerance -> green (the half-even
    tolerance of 0.07 ended the window at 0.47 -> Incorrect -> yellow).
    """

    def _make_tibia(self, value, prediction=None):
        row = {
            "Criteria": "Tibia Index",
            "HPL": 0.4,
            "LPL": 1.3,
            "Value": value,
        }
        if prediction is not None:
            row["OEM Prediction"] = prediction
        return Criteria.get_criteria_from_row(pd.Series(row))

    def test_tolerance_is_half_up(self):
        criteria = self._make_tibia(0.47)
        self.assertEqual(criteria.interval_step, 0.3)
        lower, upper = criteria._get_tolerance_bounds(criteria.green_yellow_threshold)
        self.assertEqual(lower, 0.32)
        self.assertEqual(upper, 0.48)

    def test_green_prediction_measured_047_is_in_tolerance(self):
        criteria = self._make_tibia(0.47, prediction="Green")
        self.assertEqual(criteria.prediction_result, "In Tolerance")
        self.assertEqual(criteria.color, "green")

    def test_green_prediction_measured_048_is_incorrect(self):
        # The window is closed-open: 0.48 is the first value outside it.
        criteria = self._make_tibia(0.48, prediction="Green")
        self.assertEqual(criteria.prediction_result, "Incorrect")
        self.assertEqual(criteria.color, "yellow")

    def test_thresholds_are_exact_decimals(self):
        criteria = self._make_tibia(0.47)
        self.assertEqual(criteria.green_yellow_threshold, 0.4)
        self.assertEqual(criteria.yellow_orange_threshold, 0.7)
        self.assertEqual(criteria.orange_brown_threshold, 1.0)
        self.assertEqual(criteria.brown_red_threshold, 1.3)


if __name__ == "__main__":
    unittest.main()
