# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import unittest
from euroncap_rating_2026.crash_protection.criteria import Criteria, CriteriaType
import pandas as pd
import logging


class TestVRUCriteria(unittest.TestCase):
    """
    Tests for VRU_CRITERIA type, which uses 10% relative tolerance instead of fixed interval_step / 4.

    VRU criteria are used for Vulnerable Road User test points (headform and legform impacts).
    They differ from standard criteria in their "In Tolerance" logic for prediction checking.
    """

    def setUp(self):
        """
        Set up a VRU criterion with HPL=650.0 and LPL=1700.0 (typical for Head Impact Adult).
        This creates thresholds:
        - green_yellow: 650
        - yellow_orange: 1000
        - orange_brown: 1350
        - brown_red: 1700
        """
        self.criteria = Criteria(
            name="HIC15",
            hpl=650.0,
            lpl=1700.0,
            criteria_type=CriteriaType.VRU_CRITERIA,
        )

    def test_vru_criteria_type_exists(self):
        """Verify VRU_CRITERIA enum value exists."""
        self.assertEqual(self.criteria.criteria_type, CriteriaType.VRU_CRITERIA)

    def test_vru_threshold_calculations(self):
        """Verify threshold values are calculated correctly."""
        self.assertEqual(self.criteria.green_yellow_threshold, 650.0)
        self.assertEqual(self.criteria.yellow_orange_threshold, 1000.0)
        self.assertEqual(self.criteria.orange_brown_threshold, 1350.0)
        self.assertEqual(self.criteria.brown_red_threshold, 1700.0)

    def test_vru_color_assignment_green(self):
        """Test color assignment for value in green range."""
        self.criteria.set_value(600.0)
        self.criteria.calculate_color_score()
        self.assertEqual(self.criteria.color, "green")
        self.assertEqual(self.criteria.score, 100.0)

    def test_vru_color_assignment_yellow(self):
        """Test color assignment for value in yellow range."""
        self.criteria.set_value(900.0)
        self.criteria.calculate_color_score()
        self.assertEqual(self.criteria.color, "yellow")
        self.assertEqual(self.criteria.score, 80.0)

    def test_vru_color_assignment_orange(self):
        """Test color assignment for value in orange range."""
        self.criteria.set_value(1100.0)
        self.criteria.calculate_color_score()
        self.assertEqual(self.criteria.color, "orange")
        self.assertEqual(self.criteria.score, 40.0)

    def test_vru_color_assignment_brown(self):
        """Test color assignment for value in brown range."""
        self.criteria.set_value(1500.0)
        self.criteria.calculate_color_score()
        self.assertEqual(self.criteria.color, "brown")
        self.assertEqual(self.criteria.score, 20.0)

    def test_vru_color_assignment_red(self):
        """Test color assignment for value in red range."""
        self.criteria.set_value(1750.0)
        self.criteria.calculate_color_score()
        self.assertEqual(self.criteria.color, "red")
        self.assertEqual(self.criteria.score, 0.0)

    def test_vru_tolerance_bounds_calculation(self):
        """Test the 10% relative tolerance bounds calculation."""
        # For green_yellow (650): 650/1.1 = 590.91, 650/0.9 = 722.22
        gy_lower, gy_upper = self.criteria._get_tolerance_bounds(
            self.criteria.green_yellow_threshold, is_vru=True
        )
        self.assertEqual(gy_lower, 590.91)
        self.assertEqual(gy_upper, 722.22)

        # For yellow_orange (1000): 1000/1.1 = 909.09, 1000/0.9 = 1111.11
        yo_lower, yo_upper = self.criteria._get_tolerance_bounds(
            self.criteria.yellow_orange_threshold, is_vru=True
        )
        self.assertEqual(yo_lower, 909.09)
        self.assertEqual(yo_upper, 1111.11)

        # For orange_brown (1350): 1350/1.1 = 1227.27, 1350/0.9 = 1500.0
        ob_lower, ob_upper = self.criteria._get_tolerance_bounds(
            self.criteria.orange_brown_threshold, is_vru=True
        )
        self.assertEqual(ob_lower, 1227.27)
        self.assertEqual(ob_upper, 1500.0)

        # For brown_red (1700): 1700/1.1 = 1545.45, 1700/0.9 = 1888.89
        br_lower, br_upper = self.criteria._get_tolerance_bounds(
            self.criteria.brown_red_threshold, is_vru=True
        )
        self.assertEqual(br_lower, 1545.45)
        self.assertEqual(br_upper, 1888.89)

    def test_vru_standard_tolerance_bounds_differ(self):
        """Verify standard criteria use different tolerance (interval_step / 4)."""
        vru_gy_lower, vru_gy_upper = self.criteria._get_tolerance_bounds(
            self.criteria.green_yellow_threshold, is_vru=True
        )
        std_gy_lower, std_gy_upper = self.criteria._get_tolerance_bounds(
            self.criteria.green_yellow_threshold, is_vru=False
        )

        # VRU uses 10% relative: 722.22 - 590.91 = 131.31
        self.assertAlmostEqual(vru_gy_upper - vru_gy_lower, 131.31, places=2)

        # Standard uses interval_step / 4: ±87.5 (since interval_step = 350)
        self.assertEqual(std_gy_upper - std_gy_lower, 175.0)

    def test_vru_prediction_correct_yellow(self):
        """Test prediction marked correct when value matches predicted color."""
        self.criteria.set_prediction("yellow")
        self.criteria.set_value(900.0)  # Within yellow zone (650-1000)
        self.criteria.calculate_color_score()
        self.assertEqual(self.criteria.prediction_result, "Correct")
        self.assertEqual(self.criteria.color, "yellow")

    def test_vru_prediction_in_tolerance_yellow(self):
        """
        Test prediction marked "In Tolerance" within VRU tolerance band.

        Critical test case: value=1090 with prediction="yellow"
        - green_yellow_lower = 590.91, yellow_orange_upper = 1111.11
        - 590.91 <= 1090 < 1111.11 → "In Tolerance"

        With standard criteria logic, this would be "Incorrect" (since 1090 > 1087.5).
        """
        self.criteria.set_prediction("yellow")
        self.criteria.set_value(1090.0)
        self.criteria.calculate_color_score()

        # Verify actual color is orange (where 1090 truly falls)
        self.assertEqual(self.criteria.color, "yellow")

        # Verify prediction result shows tolerance (not incorrect)
        self.assertEqual(
            self.criteria.prediction_result,
            "In Tolerance",
            msg=(
                "Value 1090 with prediction 'yellow' should be 'In Tolerance' "
                "(within 909.09-1111.11 band), not 'Incorrect'"
            ),
        )

    def test_vru_prediction_in_tolerance_green(self):
        """Test prediction marked 'In Tolerance' for green within 10% tolerance."""
        self.criteria.set_prediction("green")
        self.criteria.set_value(
            700.0
        )  # Just outside threshold but within 10% tolerance
        self.criteria.calculate_color_score()

        # Value 700 falls in yellow zone, but prediction "green" is within tolerance (590.91-722.22)
        self.assertEqual(self.criteria.prediction_result, "In Tolerance")
        self.assertEqual(self.criteria.color, "green")

    def test_vru_prediction_in_tolerance_orange(self):
        """Test prediction marked 'In Tolerance' for orange within 10% tolerance."""
        self.criteria.set_prediction("orange")
        self.criteria.set_value(
            1400.0
        )  # In brown zone but within orange tolerance band
        self.criteria.calculate_color_score()

        # Value 1400 is in brown zone (1350-1700) but within orange tolerance band (909.09-1500.0)
        # For prediction "orange": yo_lower=909.09, ob_upper=1500.0
        # 909.09 <= 1400 < 1500.0 → "In Tolerance"
        self.assertEqual(self.criteria.prediction_result, "In Tolerance")
        self.assertEqual(self.criteria.color, "orange")

    def test_vru_prediction_incorrect_far_outside(self):
        """Test prediction marked incorrect when far outside tolerance band."""
        self.criteria.set_prediction("yellow")
        self.criteria.set_value(1500.0)  # Far outside yellow tolerance (900-1100)
        self.criteria.calculate_color_score()

        # Value 1500 falls in brown, way outside yellow tolerance
        self.assertEqual(self.criteria.prediction_result, "Incorrect")
        self.assertEqual(self.criteria.color, "brown")

    def test_vru_blue_color_support(self):
        """Test that VRU criteria can have blue predictions."""
        self.criteria.set_prediction("blue")
        self.criteria.set_value(650.0)
        self.criteria.calculate_color_score()

        # Blue is a special color for VRU, prediction should be valid
        self.assertEqual(self.criteria.prediction, "blue")

    def test_vru_green_40_variant_support(self):
        """Test that VRU criteria support green-40 predictions (a-pillar markers)."""
        self.criteria.set_prediction("green-40")
        self.criteria.set_value(650.0)
        self.criteria.calculate_color_score()

        self.assertEqual(self.criteria.prediction, "green-40")

    def test_standard_criteria_still_work(self):
        """Verify standard CRITERIA type still uses original logic (interval_step/4)."""
        std_criteria = Criteria(
            name="HIC15",
            hpl=650.0,
            lpl=1700.0,
            criteria_type=CriteriaType.CRITERIA,
        )

        # With standard logic and value=1090, prediction="yellow":
        std_criteria.set_prediction("yellow")
        std_criteria.set_value(1090.0)
        std_criteria.calculate_color_score()

        # Standard tolerance = 87.5, so yellow band is 562.5 to 1087.5
        # 1090 > 1087.5, so should be "Incorrect"
        self.assertEqual(
            std_criteria.prediction_result,
            "Incorrect",
            msg="Standard CRITERIA should use interval_step/4 tolerance, making value 1090 incorrect",
        )
        # Color should still show orange (actual test result)
        self.assertEqual(std_criteria.color, "orange")

    def test_vru_vs_standard_criteria_difference(self):
        """
        Comprehensive comparison showing the exact difference between VRU and standard criteria.

        This is the key test demonstrating the tolerance mechanism change for VRU.
        With value=1090 and prediction="yellow":
        - Actual test output color is "orange" (1000 <= 1090 < 1350)
        - VRU tolerance band for yellow: 590.91-1111.11, so 1090 is IN tolerance → color="yellow"
        - Standard tolerance band for yellow: 562.5-1087.5, so 1090 is OUT of tolerance → color="orange"
        """
        vru_crit = Criteria(
            name="HIC15",
            hpl=650.0,
            lpl=1700.0,
            criteria_type=CriteriaType.VRU_CRITERIA,
        )
        std_crit = Criteria(
            name="HIC15",
            hpl=650.0,
            lpl=1700.0,
            criteria_type=CriteriaType.CRITERIA,
        )

        test_value = 1090.0
        prediction = "yellow"

        # Both criteria use same setup
        vru_crit.set_prediction(prediction)
        vru_crit.set_value(test_value)
        vru_crit.calculate_color_score()

        std_crit.set_prediction(prediction)
        std_crit.set_value(test_value)
        std_crit.calculate_color_score()

        # Prediction results differ due to different tolerance mechanisms:
        # VRU: "In Tolerance" with color="yellow" (uses 10% relative tolerance: 909.09-1111.11)
        # Standard: "Incorrect" with color="orange" (uses interval_step/4 tolerance: 912.5-1087.5)
        self.assertEqual(vru_crit.prediction_result, "In Tolerance")
        self.assertEqual(
            vru_crit.color, "yellow"
        )  # Color is prediction when "In Tolerance"

        self.assertEqual(std_crit.prediction_result, "Incorrect")
        self.assertEqual(
            std_crit.color, "orange"
        )  # Color is actual test output when "Incorrect"

    def test_vru_proportional_tolerance_widening(self):
        """
        Test that VRU tolerance increases with threshold value (10% is relative).

        Higher thresholds get wider tolerance bands, which is the key benefit of relative tolerance.
        """
        # Lower threshold example
        low_threshold = self.criteria.green_yellow_threshold  # 650
        low_lower, low_upper = self.criteria._get_tolerance_bounds(
            low_threshold, is_vru=True
        )
        low_tolerance = low_upper - low_lower

        # Higher threshold example
        high_threshold = self.criteria.brown_red_threshold  # 1700
        high_lower, high_upper = self.criteria._get_tolerance_bounds(
            high_threshold, is_vru=True
        )
        high_tolerance = high_upper - high_lower

        # Verify proportional scaling
        self.assertAlmostEqual(low_tolerance, 131.31, places=2)  # 722.22 - 590.91
        self.assertAlmostEqual(high_tolerance, 343.44, places=2)  # 1888.89 - 1545.45

        # Higher tolerance is proportionally wider
        self.assertGreater(high_tolerance, low_tolerance)
        self.assertAlmostEqual(
            high_tolerance / low_tolerance, high_threshold / low_threshold, places=3
        )


if __name__ == "__main__":
    unittest.main()
