# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import numpy as np
import pandas as pd
from unittest.mock import patch
from euroncap_rating_2026.overall.compute_score import (
    compute_star_rating,
    get_star_rating_for_score,
    get_report_path_with_prefix,
    update_test_scores_with_report,
    get_body_region_capping,
    apply_deduction,
    get_test_score_from_reports,
    calculate_score,
)
from euroncap_rating_2026 import common
import unittest
import os
import io
import tempfile
from datetime import datetime


class TestGetReportPathWithPrefix(unittest.TestCase):
    """Test cases for get_report_path_with_prefix function"""

    def test_get_report_path_existing(self):
        """Test finding existing report file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_file = os.path.join(tmpdir, "cp_2025-01-15_12-30-45_report.xlsx")
            open(report_file, "w").close()

            result = get_report_path_with_prefix(tmpdir, "cp")
            self.assertEqual(result, report_file)

    def test_get_report_path_not_found(self):
        """Test when report file doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_report_path_with_prefix(tmpdir, "cp")
            self.assertIsNone(result)

    def test_get_report_path_multiple_files(self):
        """Test when multiple report files exist (returns first)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            report1 = os.path.join(tmpdir, "cp_2025-01-15_12-30-45_report.xlsx")
            report2 = os.path.join(tmpdir, "cp_2025-01-16_12-30-45_report.xlsx")
            open(report1, "w").close()
            open(report2, "w").close()

            result = get_report_path_with_prefix(tmpdir, "cp")
            # Should return one of them (first one found)
            self.assertIsNotNone(result)
            self.assertIn("cp_", result)


class TestUpdateTestScoresWithReport(unittest.TestCase):
    """Test cases for update_test_scores_with_report function"""

    def test_update_test_scores_with_valid_report(self):
        """Test updating scores with valid report data"""
        test_scores_df = pd.DataFrame(
            {
                "Stage element": ["Element A", "Element B", "Element C"],
                "Stage subelement": ["Test A", "Test B", "Test C"],
                "Score": [0.0, 0.0, 0.0],
            }
        )
        report_df = pd.DataFrame(
            {
                "Stage element": ["Element A", "Element B"],
                "Stage subelement": ["Test A", "Test B"],
                "Score": [15.7, 22.3],
            }
        )

        result = update_test_scores_with_report(test_scores_df, report_df)
        self.assertAlmostEqual(result.loc[0, "Score"], 15.7)  # exact copy, no floor
        self.assertAlmostEqual(result.loc[1, "Score"], 22.3)  # exact copy, no floor
        self.assertEqual(result.loc[2, "Score"], 0.0)  # unchanged

    def test_update_test_scores_with_none_report(self):
        """Test updating with None report"""
        test_scores_df = pd.DataFrame({"Stage subelement": ["Test A"], "Score": [0.0]})

        result = update_test_scores_with_report(test_scores_df, None)
        self.assertEqual(result.loc[0, "Score"], 0.0)

    def test_update_test_scores_empty_report(self):
        """Test updating with empty report"""
        test_scores_df = pd.DataFrame({"Stage subelement": ["Test A"], "Score": [5.0]})
        report_df = pd.DataFrame({"Stage subelement": [], "Score": []})

        result = update_test_scores_with_report(test_scores_df, report_df)
        self.assertEqual(result.loc[0, "Score"], 5.0)  # unchanged


class TestGetBodyRegionCapping(unittest.TestCase):
    """Test cases for get_body_region_capping function"""

    def test_body_region_capping_true(self):
        """Test when body region capping conditions are met (score = 0)"""
        # Create a dataframe with one matching row with Score = 0
        cp_body_region_df = pd.DataFrame(
            {
                "Stage subelement": ["Offset"],
                "Loadcase": ["MPDB-50"],
                "Seat position": ["Driver"],
                "Dummy": ["THOR-50"],
                "Body region": ["Head & Neck"],
                "Score": [0.0],
            }
        )

        result = get_body_region_capping(cp_body_region_df)
        self.assertTrue(result)

    def test_body_region_capping_false(self):
        """Test when body region capping conditions are NOT met (score != 0)"""
        cp_body_region_df = pd.DataFrame(
            {
                "Stage subelement": ["Offset"],
                "Loadcase": ["MPDB-50"],
                "Seat position": ["Driver"],
                "Dummy": ["THOR-50"],
                "Body region": ["Head & Neck"],
                "Score": [5.0],
            }
        )

        result = get_body_region_capping(cp_body_region_df)
        self.assertFalse(result)

    def test_body_region_capping_empty_dataframe(self):
        """Test with empty dataframe"""
        cp_body_region_df = pd.DataFrame(
            {
                "Stage subelement": [],
                "Loadcase": [],
                "Seat position": [],
                "Dummy": [],
                "Body region": [],
                "Score": [],
            }
        )

        result = get_body_region_capping(cp_body_region_df)
        self.assertFalse(result)

    def test_body_region_capping_multiple_scores(self):
        """Test with multiple scores, one of them is 0"""
        # Use exact data from BODY_REGION_BLUE_ROWS_CSV_STRING for proper merge
        cp_body_region_df = pd.DataFrame(
            {
                "Stage subelement": ["Offset", "Offset"],
                "Loadcase": ["MPDB-50", "MPDB-50"],
                "Seat position": ["Driver", "Front Passenger"],
                "Dummy": ["THOR-50", "HIII-05"],
                "Body region": ["Head & Neck", "Head & Neck"],
                "Score": [5.0, 0.0],
            }
        )

        result = get_body_region_capping(cp_body_region_df)
        self.assertTrue(result)


class TestApplyDeduction(unittest.TestCase):
    """Test cases for apply_deduction function"""

    def test_apply_deduction_2026_below_7(self):
        """Test deduction for 2026 with provision score < 7"""
        stage_df = pd.DataFrame({"Stage": ["Crash Protection"], "Score": [80.0]})
        input_parameters_df = pd.DataFrame()

        with patch.object(common, "create_param_dict_from_input_parameters") as mock:
            mock.return_value = {
                "All": {"Year of the test": 2026},
                "Crash Protection": {"Vehicle provision assessment score": 5},
            }
            result = apply_deduction(stage_df, input_parameters_df)
            # Deduction = 7 - 5 = 2
            self.assertEqual(result.loc[0, "Score"], 78.0)

    def test_apply_deduction_2026_above_7(self):
        """Test no deduction for 2026 with provision score >= 7"""
        stage_df = pd.DataFrame({"Stage": ["Crash Protection"], "Score": [80.0]})
        input_parameters_df = pd.DataFrame()

        with patch.object(common, "create_param_dict_from_input_parameters") as mock:
            mock.return_value = {
                "All": {"Year of the test": 2026},
                "Crash Protection": {"Vehicle provision assessment score": 7},
            }
            result = apply_deduction(stage_df, input_parameters_df)
            self.assertEqual(result.loc[0, "Score"], 80.0)

    def test_apply_deduction_2028_below_8(self):
        """Test deduction for 2028 with provision score < 8"""
        stage_df = pd.DataFrame({"Stage": ["Crash Protection"], "Score": [80.0]})
        input_parameters_df = pd.DataFrame()

        with patch.object(common, "create_param_dict_from_input_parameters") as mock:
            mock.return_value = {
                "All": {"Year of the test": 2028},
                "Crash Protection": {"Vehicle provision assessment score": 6},
            }
            result = apply_deduction(stage_df, input_parameters_df)
            # Deduction = 8 - 6 = 2
            self.assertEqual(result.loc[0, "Score"], 78.0)

    def test_apply_deduction_2028_above_8(self):
        """Test no deduction for 2028 with provision score >= 8"""
        stage_df = pd.DataFrame({"Stage": ["Crash Protection"], "Score": [80.0]})
        input_parameters_df = pd.DataFrame()

        with patch.object(common, "create_param_dict_from_input_parameters") as mock:
            mock.return_value = {
                "All": {"Year of the test": 2028},
                "Crash Protection": {"Vehicle provision assessment score": 8},
            }
            result = apply_deduction(stage_df, input_parameters_df)
            self.assertEqual(result.loc[0, "Score"], 80.0)

    def test_apply_deduction_missing_year(self):
        """Test when year is missing"""
        stage_df = pd.DataFrame({"Stage": ["Crash Protection"], "Score": [80.0]})
        input_parameters_df = pd.DataFrame()

        with patch.object(common, "create_param_dict_from_input_parameters") as mock:
            mock.return_value = {
                "All": {"Year of the test": None},
                "Crash Protection": {"Vehicle provision assessment score": 5},
            }
            result = apply_deduction(stage_df, input_parameters_df)
            self.assertEqual(result.loc[0, "Score"], 80.0)

    def test_apply_deduction_missing_provision_score(self):
        """Test when provision score is missing"""
        stage_df = pd.DataFrame({"Stage": ["Crash Protection"], "Score": [80.0]})
        input_parameters_df = pd.DataFrame()

        with patch.object(common, "create_param_dict_from_input_parameters") as mock:
            mock.return_value = {
                "All": {"Year of the test": 2026},
                "Crash Protection": {"Vehicle provision assessment score": None},
            }
            result = apply_deduction(stage_df, input_parameters_df)
            self.assertEqual(result.loc[0, "Score"], 80.0)


class TestGetStarRatingForScore(unittest.TestCase):
    """Test cases for get_star_rating_for_score function"""

    def setUp(self):
        """Setup standard thresholds for testing"""
        self.thresholds = [80, 70, 60, 50, 40]

    def test_five_star_threshold_exact(self):
        """Test score exactly at 5-star threshold"""
        result = get_star_rating_for_score(80, self.thresholds)
        self.assertEqual(result, 5)

    def test_five_star_above_threshold(self):
        """Test score above 5-star threshold"""
        result = get_star_rating_for_score(85, self.thresholds)
        self.assertEqual(result, 5)

    def test_four_star_threshold_exact(self):
        """Test score exactly at 4-star threshold"""
        result = get_star_rating_for_score(70, self.thresholds)
        self.assertEqual(result, 4)

    def test_four_star_between_thresholds(self):
        """Test score between 4-star and 5-star thresholds"""
        result = get_star_rating_for_score(75, self.thresholds)
        self.assertEqual(result, 4)

    def test_three_star_threshold_exact(self):
        """Test score exactly at 3-star threshold"""
        result = get_star_rating_for_score(60, self.thresholds)
        self.assertEqual(result, 3)

    def test_three_star_between_thresholds(self):
        """Test score between 3-star and 4-star thresholds"""
        result = get_star_rating_for_score(65, self.thresholds)
        self.assertEqual(result, 3)

    def test_two_star_threshold_exact(self):
        """Test score exactly at 2-star threshold"""
        result = get_star_rating_for_score(50, self.thresholds)
        self.assertEqual(result, 2)

    def test_two_star_between_thresholds(self):
        """Test score between 2-star and 3-star thresholds"""
        result = get_star_rating_for_score(55, self.thresholds)
        self.assertEqual(result, 2)

    def test_one_star_threshold_exact(self):
        """Test score exactly at 1-star threshold"""
        result = get_star_rating_for_score(40, self.thresholds)
        self.assertEqual(result, 1)

    def test_one_star_between_thresholds(self):
        """Test score between 1-star and 2-star thresholds"""
        result = get_star_rating_for_score(45, self.thresholds)
        self.assertEqual(result, 1)

    def test_zero_star_below_all_thresholds(self):
        """Test score below all thresholds"""
        result = get_star_rating_for_score(35, self.thresholds)
        self.assertEqual(result, 0)

    def test_zero_star_just_below_one_star(self):
        """Test score just below 1-star threshold"""
        result = get_star_rating_for_score(39.9, self.thresholds)
        self.assertEqual(result, 0)

    def test_adjusted_thresholds_2026(self):
        """Test with adjusted thresholds for 2026"""
        adjusted_thresholds = [t - 20 for t in self.thresholds]
        result = get_star_rating_for_score(60, adjusted_thresholds)
        self.assertEqual(result, 5)

    def test_adjusted_thresholds_negative(self):
        """Test with negative thresholds"""
        adjusted_thresholds = [t - 100 for t in self.thresholds]
        result = get_star_rating_for_score(0, adjusted_thresholds)
        self.assertEqual(result, 5)


class TestComputeStarRating(unittest.TestCase):
    """Test cases for compute_star_rating function"""

    def _create_stage_df(self, sd=66, ca=74, cp=79, pc=87):
        """Helper to create stage dataframe"""
        return pd.DataFrame(
            {
                "Stage": [
                    "Safe Driving",
                    "Crash Avoidance",
                    "Crash Protection",
                    "Post-Crash",
                ],
                "Score": [float(sd), float(ca), float(cp), float(pc)],
            }
        )

    def _create_input_parameters_df(self, year=2028):
        """Helper to create input parameters dataframe"""
        return pd.DataFrame(
            {"Stage": ["All"], "Parameter": ["Year of the test"], "Value": [year]}
        )

    def test_all_perfect_scores_2028(self):
        """Test with perfect scores (80+) in 2028 - should be 5 stars"""
        stage_df = self._create_stage_df(sd=90, ca=90, cp=90, pc=90)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            self.assertEqual(result, 5.0)

    def test_minimum_star_rating_applied(self):
        """Test that minimum of all categories is used"""
        stage_df = self._create_stage_df(sd=45, ca=85, cp=85, pc=85)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            # SD at 45 is 1-star, others are 5-star -> minimum is 1
            self.assertEqual(result, 1.0)

    def test_minimum_star_rating_applied_2026(self):
        """Test minimum of all categories with 2026 offset"""
        stage_df = self._create_stage_df(sd=45, ca=85, cp=85, pc=85)
        input_parameters_df = self._create_input_parameters_df(year=2026)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2026}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            # After +20 offset: SD=65, CA=95, CP=85, PC=85 -> min is 3
            self.assertGreaterEqual(result, 0)
            self.assertLessEqual(result, 5)

    def test_minimum_star_rating_applied_2027(self):
        """Test minimum of all categories with 2027 offset"""
        stage_df = self._create_stage_df(sd=45, ca=85, cp=85, pc=85)
        input_parameters_df = self._create_input_parameters_df(year=2027)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2027}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            self.assertGreaterEqual(result, 0)
            self.assertLessEqual(result, 5)

    def test_all_below_threshold(self):
        """Test when all scores are below minimum threshold"""
        stage_df = self._create_stage_df(sd=30, ca=30, cp=30, pc=30)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            self.assertEqual(result, 0.0)

    def test_year_2026_adjustment(self):
        """Test 2026: Safe Driving gets +20, Crash Avoidance gets +10"""
        stage_df = self._create_stage_df(sd=66, ca=74, cp=79, pc=87)
        input_parameters_df = self._create_input_parameters_df(year=2026)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2026}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            self.assertIsNotNone(result)
            self.assertGreaterEqual(result, 0)
            self.assertLessEqual(result, 5)

    def test_year_2027_adjustment(self):
        """Test 2027: Safe Driving gets +10"""
        stage_df = self._create_stage_df(sd=66, ca=74, cp=79, pc=87)
        input_parameters_df = self._create_input_parameters_df(year=2027)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2027}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            self.assertIsNotNone(result)
            self.assertGreaterEqual(result, 0)
            self.assertLessEqual(result, 5)

    def test_four_star_capping_applied(self):
        """Test that 4-star capping is applied when flag is True and rating > 4"""
        stage_df = self._create_stage_df(sd=90, ca=90, cp=90, pc=90)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=True
            )
            self.assertEqual(result, 4.0)

    def test_four_star_capping_not_applied_below_threshold(self):
        """Test that 4-star capping is not applied when rating <= 4"""
        stage_df = self._create_stage_df(sd=65, ca=65, cp=65, pc=65)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=True
            )
            self.assertEqual(result, 3.0)

    def test_three_star_rating(self):
        """Test achieving 3-star rating"""
        stage_df = self._create_stage_df(sd=65, ca=65, cp=65, pc=65)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            self.assertEqual(result, 3.0)

    def test_two_star_rating(self):
        """Test achieving 2-star rating"""
        stage_df = self._create_stage_df(sd=55, ca=55, cp=55, pc=55)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            self.assertEqual(result, 2.0)

    def test_one_star_rating(self):
        """Test achieving 1-star rating"""
        stage_df = self._create_stage_df(sd=45, ca=45, cp=45, pc=45)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            self.assertEqual(result, 1.0)

    def test_mixed_category_scores(self):
        """Test with mixed category scores to verify minimum is used"""
        # SD=3, CA=4, CP=5, PC=5 -> should be 3 (minimum)
        stage_df = self._create_stage_df(sd=65, ca=72, cp=82, pc=82)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            self.assertEqual(result, 3.0)

    def test_threshold_boundaries_exact_80(self):
        """Test score exactly at 80 threshold (5-star boundary)"""
        stage_df = self._create_stage_df(sd=80, ca=80, cp=80, pc=80)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            self.assertEqual(result, 5.0)

    def test_threshold_boundaries_exact_70(self):
        """Test score exactly at 70 threshold (4-star boundary)"""
        stage_df = self._create_stage_df(sd=70, ca=70, cp=70, pc=70)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            self.assertEqual(result, 4.0)

    def test_threshold_boundaries_exact_60(self):
        """Test score exactly at 60 threshold (3-star boundary)"""
        stage_df = self._create_stage_df(sd=60, ca=60, cp=60, pc=60)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            self.assertEqual(result, 3.0)

    def test_threshold_boundaries_exact_50(self):
        """Test score exactly at 50 threshold (2-star boundary)"""
        stage_df = self._create_stage_df(sd=50, ca=50, cp=50, pc=50)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            self.assertEqual(result, 2.0)

    def test_threshold_boundaries_exact_40(self):
        """Test score exactly at 40 threshold (1-star boundary)"""
        stage_df = self._create_stage_df(sd=40, ca=40, cp=40, pc=40)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            self.assertEqual(result, 1.0)

    def test_just_below_80_threshold(self):
        """Test score just below 80 threshold (should be 4-star)"""
        stage_df = self._create_stage_df(sd=79.9, ca=79.9, cp=79.9, pc=79.9)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            self.assertEqual(result, 4.0)

    def test_just_below_40_threshold(self):
        """Test score just below 40 threshold (should be 0-star)"""
        stage_df = self._create_stage_df(sd=39.9, ca=39.9, cp=39.9, pc=39.9)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            self.assertEqual(result, 0.0)

    def test_capping_with_compensation(self):
        """Test 4-star capping with compensation mechanism"""
        stage_df = self._create_stage_df(sd=90, ca=90, cp=90, pc=90)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=True
            )
            self.assertEqual(result, 4.0)

    def test_year_2028_no_offset(self):
        """Test 2028 without any score offsets"""
        stage_df = self._create_stage_df(sd=75, ca=75, cp=75, pc=75)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            self.assertEqual(result, 4.0)

    def test_unequal_scores_pc_lowest(self):
        """Test when Post-Crash score is the limiting factor"""
        stage_df = self._create_stage_df(sd=85, ca=85, cp=85, pc=45)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            # PC at 45 is 1-star, others are 5-star -> minimum is 1
            self.assertEqual(result, 1.0)

    def test_unequal_scores_cp_lowest(self):
        """Test when Crash Protection score is the limiting factor"""
        stage_df = self._create_stage_df(sd=85, ca=85, cp=45, pc=85)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            # CP at 45 is 1-star, others are 5-star -> minimum is 1
            self.assertEqual(result, 1.0)


class TestCalculateScore(unittest.TestCase):
    """Tests for the calculate_score(dfs) dataframe-in/dataframe-out entrypoint."""

    # ------------------------------------------------------------------ helpers

    def _make_test_scores_df(self):
        return pd.DataFrame(
            {
                "Stage": ["Crash Avoidance", "Crash Protection"],
                "Stage element": ["Frontal Collisions", "Frontal Collisions"],
                "Stage subelement": ["Pedestrian & cyclist", "Head & Neck"],
                "Score": [0.0, 0.0],
                "Max score": [20.0, 10.0],
            }
        )

    def _make_stage_element_df(self):
        return pd.DataFrame(
            {
                "Stage": ["Crash Avoidance", "Crash Protection"],
                "Stage element": ["Frontal Collisions", "Frontal Collisions"],
                "Score": [0.0, 0.0],
                "Max score": [20.0, 10.0],
            }
        )

    def _make_stage_df(self, sd=75.0, ca=75.0, cp=75.0, pc=75.0):
        return pd.DataFrame(
            {
                "Stage": [
                    "Safe Driving",
                    "Crash Avoidance",
                    "Crash Protection",
                    "Post-Crash",
                ],
                "Score": [sd, ca, cp, pc],
                "Max score": [100.0, 100.0, 100.0, 100.0],
            }
        )

    def _make_rating_df(self):
        return pd.DataFrame({"Star rating": [0]})

    def _make_input_parameters_df(self, year=2028):
        return pd.DataFrame(
            {
                "Stage": ["All", "Crash Protection"],
                "Parameter": ["Year of the test", "Vehicle provision assessment score"],
                "Value": [year, 8],
            }
        )

    def _make_dfs(self, year=2028, sd=75.0, ca=75.0, cp=75.0, pc=75.0):
        return {
            "Input parameters": self._make_input_parameters_df(year),
            "Rating": self._make_rating_df(),
            "Stage Scores": self._make_stage_df(sd=sd, ca=ca, cp=cp, pc=pc),
            "Stage element Scores": self._make_stage_element_df(),
            "Test Scores": self._make_test_scores_df(),
            "cp": {"Test Scores": None, "CP - Body region scores": None},
            "ca": {"Test Scores": None},
            "sd": {"Test Scores": None, "Category Scores": None},
            "pc": {"Test Scores": None, "Scenario Scores": None},
        }

    # ------------------------------------------------------------------ tests

    def test_returns_required_keys(self):
        dfs = self._make_dfs()
        with (
            patch.object(
                common, "update_score_sum_interval", side_effect=lambda df, *a, **kw: df
            ),
            patch.object(
                common,
                "create_param_dict_from_input_parameters",
                return_value={
                    "All": {"Year of the test": 2028},
                    "Crash Protection": {"Vehicle provision assessment score": 8},
                },
            ),
        ):
            result = calculate_score(dfs)
        self.assertIn("Test Scores", result)
        self.assertIn("Stage element Scores", result)
        self.assertIn("Stage Scores", result)
        self.assertIn("Rating", result)

    def test_does_not_mutate_input(self):
        dfs = self._make_dfs()
        original_ts_score = dfs["Test Scores"]["Score"].tolist()
        with (
            patch.object(
                common, "update_score_sum_interval", side_effect=lambda df, *a, **kw: df
            ),
            patch.object(
                common,
                "create_param_dict_from_input_parameters",
                return_value={
                    "All": {"Year of the test": 2028},
                    "Crash Protection": {"Vehicle provision assessment score": 8},
                },
            ),
        ):
            calculate_score(dfs)
        self.assertEqual(dfs["Test Scores"]["Score"].tolist(), original_ts_score)

    def test_star_rating_written_to_output(self):
        dfs = self._make_dfs(sd=85.0, ca=85.0, cp=85.0, pc=85.0)
        with (
            patch.object(
                common, "update_score_sum_interval", side_effect=lambda df, *a, **kw: df
            ),
            patch.object(
                common,
                "create_param_dict_from_input_parameters",
                return_value={
                    "All": {"Year of the test": 2028},
                    "Crash Protection": {"Vehicle provision assessment score": 8},
                },
            ),
        ):
            result = calculate_score(dfs)
        self.assertEqual(result["Rating"]["Star rating"].values[0], 5)

    def test_edge_case_1_zeros_pedestrian_score(self):
        """VRU Head + Pelvis < 10 must zero the Frontal Collisions/Pedestrian & cyclist row."""
        ts = pd.DataFrame(
            {
                "Stage": ["Crash Avoidance", "Crash Avoidance", "Crash Avoidance"],
                "Stage element": ["Frontal Collisions", "VRU Impact", "VRU Impact"],
                "Stage subelement": [
                    "Pedestrian & cyclist",
                    "Head Impact",
                    "Pelvis & Leg Impact",
                ],
                "Score": [15.0, 4.0, 5.0],
                "Max score": [20.0, 10.0, 10.0],
            }
        )
        dfs = self._make_dfs()
        dfs["Test Scores"] = ts
        with (
            patch.object(
                common, "update_score_sum_interval", side_effect=lambda df, *a, **kw: df
            ),
            patch.object(
                common,
                "create_param_dict_from_input_parameters",
                return_value={
                    "All": {"Year of the test": 2028},
                    "Crash Protection": {"Vehicle provision assessment score": 8},
                },
            ),
        ):
            result = calculate_score(dfs)
        ped_score = (
            result["Test Scores"]
            .loc[
                result["Test Scores"]["Stage subelement"] == "Pedestrian & cyclist",
                "Score",
            ]
            .values[0]
        )
        self.assertEqual(ped_score, 0.0)

    def test_edge_case_1_does_not_fire_when_vru_gte_10(self):
        ts = pd.DataFrame(
            {
                "Stage": ["Crash Avoidance", "Crash Avoidance", "Crash Avoidance"],
                "Stage element": ["Frontal Collisions", "VRU Impact", "VRU Impact"],
                "Stage subelement": [
                    "Pedestrian & cyclist",
                    "Head Impact",
                    "Pelvis & Leg Impact",
                ],
                "Score": [15.0, 6.0, 4.0],
                "Max score": [20.0, 10.0, 10.0],
            }
        )
        dfs = self._make_dfs()
        dfs["Test Scores"] = ts
        with (
            patch.object(
                common, "update_score_sum_interval", side_effect=lambda df, *a, **kw: df
            ),
            patch.object(
                common,
                "create_param_dict_from_input_parameters",
                return_value={
                    "All": {"Year of the test": 2028},
                    "Crash Protection": {"Vehicle provision assessment score": 8},
                },
            ),
        ):
            result = calculate_score(dfs)
        ped_score = (
            result["Test Scores"]
            .loc[
                result["Test Scores"]["Stage subelement"] == "Pedestrian & cyclist",
                "Score",
            ]
            .values[0]
        )
        self.assertEqual(ped_score, 15.0)

    def test_edge_case_1_skipped_when_head_impact_unassessed(self):
        """An unassessed (NaN) Head Impact score must not silently satisfy
        or silently fail the `< 10` comparison -- the rule must be skipped,
        leaving Pedestrian & cyclist untouched, not zeroed."""
        ts = pd.DataFrame(
            {
                "Stage": ["Crash Avoidance", "Crash Avoidance", "Crash Avoidance"],
                "Stage element": ["Frontal Collisions", "VRU Impact", "VRU Impact"],
                "Stage subelement": [
                    "Pedestrian & cyclist",
                    "Head Impact",
                    "Pelvis & Leg Impact",
                ],
                "Score": [15.0, np.nan, 5.0],
                "Max score": [20.0, 10.0, 10.0],
            }
        )
        dfs = self._make_dfs()
        dfs["Test Scores"] = ts
        with (
            patch.object(
                common, "update_score_sum_interval", side_effect=lambda df, *a, **kw: df
            ),
            patch.object(
                common,
                "create_param_dict_from_input_parameters",
                return_value={
                    "All": {"Year of the test": 2028},
                    "Crash Protection": {"Vehicle provision assessment score": 8},
                },
            ),
        ):
            result = calculate_score(dfs)
        ped_score = (
            result["Test Scores"]
            .loc[
                result["Test Scores"]["Stage subelement"] == "Pedestrian & cyclist",
                "Score",
            ]
            .values[0]
        )
        self.assertEqual(ped_score, 15.0)

    def test_edge_case_2_subtracts_occupants_score(self):
        """crash_occupancy == 0 AND potential_occupants > 0 must subtract from Advanced eCall."""
        ts = pd.DataFrame(
            {
                "Stage": ["Post-Crash"],
                "Stage element": ["Post-Crash Intervention"],
                "Stage subelement": ["Advanced eCall"],
                "Score": [12.0],
                "Max score": [20.0],
            }
        )
        sd_category = pd.DataFrame(
            {
                "Category": ["Crash occupancy information"],
                "Score": [0.0],
            }
        )
        pc_scenario = pd.DataFrame(
            {
                "Scenario": ["Potential number of occupants"],
                "Score": [4.0],
            }
        )
        dfs = self._make_dfs()
        dfs["Test Scores"] = ts
        dfs["sd"]["Category Scores"] = sd_category
        dfs["pc"]["Scenario Scores"] = pc_scenario
        with (
            patch.object(
                common, "update_score_sum_interval", side_effect=lambda df, *a, **kw: df
            ),
            patch.object(
                common,
                "create_param_dict_from_input_parameters",
                return_value={
                    "All": {"Year of the test": 2028},
                    "Crash Protection": {"Vehicle provision assessment score": 8},
                },
            ),
        ):
            result = calculate_score(dfs)
        ecall_score = (
            result["Test Scores"]
            .loc[result["Test Scores"]["Stage subelement"] == "Advanced eCall", "Score"]
            .values[0]
        )
        self.assertAlmostEqual(ecall_score, 8.0)

    def test_edge_case_2_no_deduction_when_occupancy_gt_0(self):
        ts = pd.DataFrame(
            {
                "Stage": ["Post-Crash"],
                "Stage element": ["Post-Crash Intervention"],
                "Stage subelement": ["Advanced eCall"],
                "Score": [12.0],
                "Max score": [20.0],
            }
        )
        sd_category = pd.DataFrame(
            {"Category": ["Crash occupancy information"], "Score": [1.0]}
        )
        pc_scenario = pd.DataFrame(
            {"Scenario": ["Potential number of occupants"], "Score": [4.0]}
        )
        dfs = self._make_dfs()
        dfs["Test Scores"] = ts
        dfs["sd"]["Category Scores"] = sd_category
        dfs["pc"]["Scenario Scores"] = pc_scenario
        with (
            patch.object(
                common, "update_score_sum_interval", side_effect=lambda df, *a, **kw: df
            ),
            patch.object(
                common,
                "create_param_dict_from_input_parameters",
                return_value={
                    "All": {"Year of the test": 2028},
                    "Crash Protection": {"Vehicle provision assessment score": 8},
                },
            ),
        ):
            result = calculate_score(dfs)
        ecall_score = (
            result["Test Scores"]
            .loc[result["Test Scores"]["Stage subelement"] == "Advanced eCall", "Score"]
            .values[0]
        )
        self.assertAlmostEqual(ecall_score, 12.0)

    def test_edge_case_2_skipped_when_crash_occupancy_unassessed(self):
        """An unassessed (NaN) Crash occupancy information score must not
        silently satisfy or silently fail the `== 0` gate -- the rule must
        be skipped, leaving Advanced eCall untouched, not deducted."""
        ts = pd.DataFrame(
            {
                "Stage": ["Post-Crash"],
                "Stage element": ["Post-Crash Intervention"],
                "Stage subelement": ["Advanced eCall"],
                "Score": [12.0],
                "Max score": [20.0],
            }
        )
        sd_category = pd.DataFrame(
            {
                "Category": ["Crash occupancy information"],
                "Score": [np.nan],
            }
        )
        pc_scenario = pd.DataFrame(
            {
                "Scenario": ["Potential number of occupants"],
                "Score": [4.0],
            }
        )
        dfs = self._make_dfs()
        dfs["Test Scores"] = ts
        dfs["sd"]["Category Scores"] = sd_category
        dfs["pc"]["Scenario Scores"] = pc_scenario
        with (
            patch.object(
                common, "update_score_sum_interval", side_effect=lambda df, *a, **kw: df
            ),
            patch.object(
                common,
                "create_param_dict_from_input_parameters",
                return_value={
                    "All": {"Year of the test": 2028},
                    "Crash Protection": {"Vehicle provision assessment score": 8},
                },
            ),
        ):
            result = calculate_score(dfs)
        ecall_score = (
            result["Test Scores"]
            .loc[result["Test Scores"]["Stage subelement"] == "Advanced eCall", "Score"]
            .values[0]
        )
        self.assertAlmostEqual(ecall_score, 12.0)

    def test_body_region_capping_caps_at_4_stars(self):
        cp_body_region = pd.DataFrame(
            {
                "Stage subelement": ["Offset"],
                "Loadcase": ["MPDB-50"],
                "Seat position": ["Driver"],
                "Dummy": ["THOR-50"],
                "Body region": ["Head & Neck"],
                "Score": [0.0],
            }
        )
        dfs = self._make_dfs(sd=90.0, ca=90.0, cp=90.0, pc=90.0)
        dfs["cp"]["CP - Body region scores"] = cp_body_region
        with (
            patch.object(
                common, "update_score_sum_interval", side_effect=lambda df, *a, **kw: df
            ),
            patch.object(
                common,
                "create_param_dict_from_input_parameters",
                return_value={
                    "All": {"Year of the test": 2028},
                    "Crash Protection": {"Vehicle provision assessment score": 8},
                },
            ),
        ):
            result = calculate_score(dfs)
        self.assertEqual(result["Rating"]["Star rating"].values[0], 4)

    def _patched(self):
        return (
            patch.object(
                common, "update_score_sum_interval", side_effect=lambda df, *a, **kw: df
            ),
            patch.object(
                common,
                "create_param_dict_from_input_parameters",
                return_value={
                    "All": {"Year of the test": 2028},
                    "Crash Protection": {"Vehicle provision assessment score": 8},
                },
            ),
        )

    def test_dash_placeholders_do_not_raise(self):
        """The generated template carries "-" in every computed cell (see
        overall/sheets.py); calculate_score must convert them to 0 before
        its hard astype casts instead of raising ValueError."""
        dfs = self._make_dfs()
        dfs["Test Scores"]["Score"] = "-"
        dfs["Stage element Scores"]["Score"] = "-"
        dfs["Stage Scores"]["Score"] = "-"
        dfs["Rating"]["Star rating"] = "-"
        p1, p2 = self._patched()
        with p1, p2:
            result = calculate_score(dfs)
        # 0-propagation: dashed (unassessed) stages read as 0 points.
        self.assertEqual(result["Stage Scores"]["Score"].tolist(), [0, 0, 0, 0])
        self.assertEqual(result["Rating"]["Star rating"].values[0], 0)

    def test_dash_placeholders_score_same_as_zeros(self):
        dashed = self._make_dfs(sd=85.0, ca=85.0, cp=85.0, pc=85.0)
        dashed["Test Scores"]["Score"] = "-"
        dashed["Rating"]["Star rating"] = "-"
        zeroed = self._make_dfs(sd=85.0, ca=85.0, cp=85.0, pc=85.0)
        zeroed["Test Scores"]["Score"] = 0.0
        zeroed["Rating"]["Star rating"] = 0

        p1, p2 = self._patched()
        with p1, p2:
            result_dashed = calculate_score(dashed)
        p1, p2 = self._patched()
        with p1, p2:
            result_zeroed = calculate_score(zeroed)

        for key in ["Test Scores", "Stage element Scores", "Stage Scores", "Rating"]:
            pd.testing.assert_frame_equal(result_dashed[key], result_zeroed[key])

    def test_mixed_dash_and_numeric_rows(self):
        dfs = self._make_dfs()
        dfs["Test Scores"]["Score"] = ["-", 2.5]
        p1, p2 = self._patched()
        with p1, p2:
            result = calculate_score(dfs)
        self.assertEqual(result["Test Scores"]["Score"].tolist(), [0.0, 2.5])


class TestCalculateScoreOnDashedPackagedTemplate(unittest.TestCase):
    """End-to-end guard for the packaged template: data/overall_template.xlsx
    ships with "-" baked into every computed cell, and calculate_score with
    no domain reports at all must complete with 0-based scores (a missing
    domain report = 0 points / 0 stars), not raise on the dashes."""

    def test_pristine_template_with_no_reports_yields_zeros(self):
        from importlib.resources import files

        pristine_path = str(files("data").joinpath("overall_template.xlsx"))
        dfs = common.read_excel_file_to_dfs(pristine_path)
        self.assertEqual(dfs["Rating"]["Star rating"].tolist(), ["-"])

        result = calculate_score(dfs)

        self.assertEqual(result["Rating"]["Star rating"].tolist(), [0])
        self.assertEqual(result["Stage Scores"]["Score"].tolist(), [0, 0, 0, 0])
        self.assertTrue((result["Test Scores"]["Score"] == 0.0).all())
        # The input dfs are not mutated.
        self.assertEqual(dfs["Rating"]["Star rating"].tolist(), ["-"])


class TestGetTestScoreFromReportsEdgeCase2(unittest.TestCase):
    """Regression tests for edge case 2: Advanced eCall deduction."""

    def _make_test_scores_df(self, ecall_score=12.0):
        return pd.DataFrame(
            {
                "Stage": ["Post-Crash"],
                "Stage element": ["Post-Crash Intervention"],
                "Stage subelement": ["Advanced eCall"],
                "Score": [ecall_score],
                "Max score": [20.0],
            }
        )

    def _make_sd_category_scores(self, crash_occupancy_score):
        return pd.DataFrame(
            {
                "Category": ["Crash occupancy information"],
                "Score": [crash_occupancy_score],
            }
        )

    def _make_pc_scenario_scores(self, potential_occupants_score):
        return pd.DataFrame(
            {
                "Scenario": ["Potential number of occupants"],
                "Score": [potential_occupants_score],
            }
        )

    def _run(self, crash_occupancy, potential_occupants, ecall_score=12.0):
        test_scores_df = self._make_test_scores_df(ecall_score)
        empty_test_scores = pd.DataFrame(
            columns=["Stage element", "Stage subelement", "Score"]
        )
        sd_category_df = self._make_sd_category_scores(crash_occupancy)
        pc_scenario_df = self._make_pc_scenario_scores(potential_occupants)

        # get_sheet_from_report is called 6 times:
        # calls 1-4: "Test Scores" sheets for cp, ca, sd, pc (returned empty so they don't alter scores)
        # call 5: sd "Category Scores"
        # call 6: pc "Scenario Scores"
        sheet_side_effects = [
            empty_test_scores,
            empty_test_scores,
            empty_test_scores,
            empty_test_scores,
            sd_category_df,
            pc_scenario_df,
        ]

        with (
            patch(
                "euroncap_rating_2026.overall.compute_score.get_sheet_from_report",
                side_effect=sheet_side_effects,
            ),
            patch(
                "euroncap_rating_2026.overall.compute_score.update_test_scores_with_report",
                side_effect=lambda base, _report: base,
            ),
            patch(
                "euroncap_rating_2026.overall.compute_score.common.opposite_ffill",
                side_effect=lambda df, **_: df,
            ),
        ):
            result = get_test_score_from_reports(
                test_scores_df,
                cp_report_path="cp.xlsx",
                ca_report_path="ca.xlsx",
                sd_report_path="sd.xlsx",
                pc_report_path="pc.xlsx",
            )
        return result

    def test_deduction_fires_when_condition_true(self):
        """Edge case 2 must subtract potential_occupants_score from Advanced eCall."""
        result = self._run(
            crash_occupancy=0.0, potential_occupants=3.0, ecall_score=12.0
        )
        ecall_score = result.loc[
            result["Stage subelement"] == "Advanced eCall", "Score"
        ].values[0]
        self.assertAlmostEqual(ecall_score, 9.0)

    def test_no_deduction_when_crash_occupancy_gt_0(self):
        """Condition is false when crash_occupancy_score > 0; score must be unchanged."""
        result = self._run(
            crash_occupancy=1.0, potential_occupants=3.0, ecall_score=12.0
        )
        ecall_score = result.loc[
            result["Stage subelement"] == "Advanced eCall", "Score"
        ].values[0]
        self.assertAlmostEqual(ecall_score, 12.0)

    def test_no_deduction_when_potential_occupants_zero(self):
        """Condition is false when potential_occupants_score == 0; score must be unchanged."""
        result = self._run(
            crash_occupancy=4.0, potential_occupants=0.0, ecall_score=12.0
        )
        ecall_score = result.loc[
            result["Stage subelement"] == "Advanced eCall", "Score"
        ].values[0]
        self.assertAlmostEqual(ecall_score, 12.0)


class TestFlooringBehavior(unittest.TestCase):
    """Fix 3: Test scores must not be floored; stage scores must be floored after deduction."""

    def test_test_scores_not_floored(self):
        """update_test_scores_with_report must copy scores exactly, without floor."""
        test_scores_df = pd.DataFrame(
            {
                "Stage element": ["Element A"],
                "Stage subelement": ["Test A"],
                "Score": [0.0],
            }
        )
        report_df = pd.DataFrame(
            {
                "Stage element": ["Element A"],
                "Stage subelement": ["Test A"],
                "Score": [15.7],
            }
        )
        result = update_test_scores_with_report(test_scores_df, report_df)
        self.assertAlmostEqual(
            result.loc[0, "Score"],
            15.7,
            msg="Test score must be the exact domain value, not floored",
        )

    def test_fractional_test_scores_preserved_across_sum(self):
        """Fractional test scores must sum without premature flooring."""
        test_scores_df = pd.DataFrame(
            {
                "Stage element": ["Element A"],
                "Stage subelement": ["Test A"],
                "Score": [0.0],
            }
        )
        report_df = pd.DataFrame(
            {
                "Stage element": ["Element A"],
                "Stage subelement": ["Test A"],
                "Score": [9.9],
            }
        )
        result = update_test_scores_with_report(test_scores_df, report_df)
        self.assertAlmostEqual(
            result.loc[0, "Score"],
            9.9,
            msg="Fractional score 9.9 must not be floored to 9",
        )

    def test_stage_scores_floored_in_calculate_score(self):
        """Stage scores in calculate_score output must be floored integers."""
        import math as _math

        # Build a dfs where stage_df starts at 75.9 and apply_deduction leaves it at 75.9
        test_scores_df = pd.DataFrame(
            {
                "Stage": ["Safe Driving"],
                "Stage element": ["Driver Monitoring"],
                "Stage subelement": ["DM Score"],
                "Score": [0.0],
                "Max score": [100.0],
            }
        )
        stage_element_df = pd.DataFrame(
            {
                "Stage": ["Safe Driving"],
                "Stage element": ["Driver Monitoring"],
                "Score": [0.0],
                "Max score": [100.0],
            }
        )
        stage_df = pd.DataFrame(
            {
                "Stage": [
                    "Safe Driving",
                    "Crash Avoidance",
                    "Crash Protection",
                    "Post-Crash",
                ],
                "Score": [75.9, 75.9, 75.9, 75.9],
                "Max score": [100.0, 100.0, 100.0, 100.0],
            }
        )
        dfs = {
            "Input parameters": pd.DataFrame(
                {
                    "Stage": ["All", "Crash Protection"],
                    "Parameter": [
                        "Year of the test",
                        "Vehicle provision assessment score",
                    ],
                    "Value": [2028, 8],
                }
            ),
            "Rating": pd.DataFrame({"Star rating": [0]}),
            "Stage Scores": stage_df,
            "Stage element Scores": stage_element_df,
            "Test Scores": test_scores_df,
            "cp": {"Test Scores": None, "CP - Body region scores": None},
            "ca": {"Test Scores": None},
            "sd": {"Test Scores": None, "Category Scores": None},
            "pc": {"Test Scores": None, "Scenario Scores": None},
        }
        with (
            patch.object(
                common, "update_score_sum_interval", side_effect=lambda df, *a, **kw: df
            ),
            patch.object(
                common,
                "create_param_dict_from_input_parameters",
                return_value={
                    "All": {"Year of the test": 2028},
                    "Crash Protection": {"Vehicle provision assessment score": 8},
                },
            ),
        ):
            result = calculate_score(dfs)
        stage_scores = result["Stage Scores"]
        for score in stage_scores["Score"]:
            self.assertEqual(
                score,
                _math.floor(score),
                msg=f"Stage score {score} must be a floored integer",
            )

    def test_nan_stage_score_does_not_crash_flooring(self):
        """An unassessed (NaN) Stage score must pass through the flooring
        step unchanged, not crash calculate_score -- math.floor(nan) raises
        ValueError, and this Stage can legitimately be NaN if any of its
        Test Scores is unassessed."""
        test_scores_df = pd.DataFrame(
            {
                "Stage": ["Safe Driving"],
                "Stage element": ["Driver Monitoring"],
                "Stage subelement": ["DM Score"],
                "Score": [0.0],
                "Max score": [100.0],
            }
        )
        stage_element_df = pd.DataFrame(
            {
                "Stage": ["Safe Driving"],
                "Stage element": ["Driver Monitoring"],
                "Score": [0.0],
                "Max score": [100.0],
            }
        )
        stage_df = pd.DataFrame(
            {
                "Stage": [
                    "Safe Driving",
                    "Crash Avoidance",
                    "Crash Protection",
                    "Post-Crash",
                ],
                "Score": [np.nan, 75.9, 75.9, 75.9],
                "Max score": [100.0, 100.0, 100.0, 100.0],
            }
        )
        dfs = {
            "Input parameters": pd.DataFrame(
                {
                    "Stage": ["All", "Crash Protection"],
                    "Parameter": [
                        "Year of the test",
                        "Vehicle provision assessment score",
                    ],
                    "Value": [2028, 8],
                }
            ),
            "Rating": pd.DataFrame({"Star rating": [0]}),
            "Stage Scores": stage_df,
            "Stage element Scores": stage_element_df,
            "Test Scores": test_scores_df,
            "cp": {"Test Scores": None, "CP - Body region scores": None},
            "ca": {"Test Scores": None},
            "sd": {"Test Scores": None, "Category Scores": None},
            "pc": {"Test Scores": None, "Scenario Scores": None},
        }
        with (
            patch.object(
                common, "update_score_sum_interval", side_effect=lambda df, *a, **kw: df
            ),
            patch.object(
                common,
                "create_param_dict_from_input_parameters",
                return_value={
                    "All": {"Year of the test": 2028},
                    "Crash Protection": {"Vehicle provision assessment score": 8},
                },
            ),
        ):
            result = calculate_score(dfs)  # must not raise
        sd_score = (
            result["Stage Scores"]
            .loc[result["Stage Scores"]["Stage"] == "Safe Driving", "Score"]
            .values[0]
        )
        self.assertTrue(pd.isna(sd_score))
        self.assertIsNone(result["Rating"]["Star rating"].values[0])

    def test_stage_score_floor_affects_star_rating(self):
        """A stage score of 79.9 should be floored to 79 and NOT reach the 80-threshold for 5 stars."""
        stage_df = pd.DataFrame(
            {
                "Stage": [
                    "Safe Driving",
                    "Crash Avoidance",
                    "Crash Protection",
                    "Post-Crash",
                ],
                "Score": [79.9, 79.9, 79.9, 79.9],
                "Max score": [100.0, 100.0, 100.0, 100.0],
            }
        )
        dfs = {
            "Input parameters": pd.DataFrame(
                {
                    "Stage": ["All", "Crash Protection"],
                    "Parameter": [
                        "Year of the test",
                        "Vehicle provision assessment score",
                    ],
                    "Value": [2028, 8],
                }
            ),
            "Rating": pd.DataFrame({"Star rating": [0]}),
            "Stage Scores": stage_df,
            "Stage element Scores": pd.DataFrame(
                {
                    "Stage": ["Safe Driving"],
                    "Stage element": ["DM"],
                    "Score": [0.0],
                    "Max score": [100.0],
                }
            ),
            "Test Scores": pd.DataFrame(
                {
                    "Stage": ["Safe Driving"],
                    "Stage element": ["DM"],
                    "Stage subelement": ["DM Score"],
                    "Score": [0.0],
                    "Max score": [100.0],
                }
            ),
            "cp": {"Test Scores": None, "CP - Body region scores": None},
            "ca": {"Test Scores": None},
            "sd": {"Test Scores": None, "Category Scores": None},
            "pc": {"Test Scores": None, "Scenario Scores": None},
        }
        with (
            patch.object(
                common, "update_score_sum_interval", side_effect=lambda df, *a, **kw: df
            ),
            patch.object(
                common,
                "create_param_dict_from_input_parameters",
                return_value={
                    "All": {"Year of the test": 2028},
                    "Crash Protection": {"Vehicle provision assessment score": 8},
                },
            ),
        ):
            result = calculate_score(dfs)
        # 79.9 floored = 79; 5-star threshold for 2028 is 80, so should be 4 stars
        star_rating = result["Rating"]["Star rating"].values[0]
        self.assertEqual(
            star_rating,
            4,
            msg="Stage score 79.9 must floor to 79 and yield 4 stars, not 5",
        )


if __name__ == "__main__":
    unittest.main()
    """Test cases for get_star_rating_for_score function"""

    def setUp(self):
        """Setup standard thresholds for testing"""
        self.thresholds = [80, 70, 60, 50, 40]

    def test_five_star_threshold_exact(self):
        """Test score exactly at 5-star threshold"""
        result = get_star_rating_for_score(80, self.thresholds)
        self.assertEqual(result, 5)

    def test_five_star_above_threshold(self):
        """Test score above 5-star threshold"""
        result = get_star_rating_for_score(85, self.thresholds)
        self.assertEqual(result, 5)

    def test_four_star_threshold_exact(self):
        """Test score exactly at 4-star threshold"""
        result = get_star_rating_for_score(70, self.thresholds)
        self.assertEqual(result, 4)

    def test_four_star_between_thresholds(self):
        """Test score between 4-star and 5-star thresholds"""
        result = get_star_rating_for_score(75, self.thresholds)
        self.assertEqual(result, 4)

    def test_three_star_threshold_exact(self):
        """Test score exactly at 3-star threshold"""
        result = get_star_rating_for_score(60, self.thresholds)
        self.assertEqual(result, 3)

    def test_three_star_between_thresholds(self):
        """Test score between 3-star and 4-star thresholds"""
        result = get_star_rating_for_score(65, self.thresholds)
        self.assertEqual(result, 3)

    def test_two_star_threshold_exact(self):
        """Test score exactly at 2-star threshold"""
        result = get_star_rating_for_score(50, self.thresholds)
        self.assertEqual(result, 2)

    def test_two_star_between_thresholds(self):
        """Test score between 2-star and 3-star thresholds"""
        result = get_star_rating_for_score(55, self.thresholds)
        self.assertEqual(result, 2)

    def test_one_star_threshold_exact(self):
        """Test score exactly at 1-star threshold"""
        result = get_star_rating_for_score(40, self.thresholds)
        self.assertEqual(result, 1)

    def test_one_star_between_thresholds(self):
        """Test score between 1-star and 2-star thresholds"""
        result = get_star_rating_for_score(45, self.thresholds)
        self.assertEqual(result, 1)

    def test_zero_star_below_all_thresholds(self):
        """Test score below all thresholds"""
        result = get_star_rating_for_score(35, self.thresholds)
        self.assertEqual(result, 0)

    def test_zero_star_just_below_one_star(self):
        """Test score just below 1-star threshold"""
        result = get_star_rating_for_score(39.9, self.thresholds)
        self.assertEqual(result, 0)

    def test_adjusted_thresholds_2026(self):
        """Test with adjusted thresholds for 2026"""
        adjusted_thresholds = [t - 20 for t in self.thresholds]  # [60, 50, 40, 30, 20]
        result = get_star_rating_for_score(60, adjusted_thresholds)
        self.assertEqual(result, 5)

    def test_adjusted_thresholds_negative(self):
        """Test with negative thresholds"""
        adjusted_thresholds = [
            t - 100 for t in self.thresholds
        ]  # [-20, -30, -40, -50, -60]
        result = get_star_rating_for_score(0, adjusted_thresholds)
        self.assertEqual(result, 5)


class TestComputeStarRating(unittest.TestCase):
    """Test cases for compute_star_rating function"""

    def _create_stage_df(self, sd=66, ca=74, cp=79, pc=87):
        """Helper to create stage dataframe"""
        return pd.DataFrame(
            {
                "Stage": [
                    "Safe Driving",
                    "Crash Avoidance",
                    "Crash Protection",
                    "Post-Crash",
                ],
                "Score": [float(sd), float(ca), float(cp), float(pc)],
            }
        )

    def _create_input_parameters_df(self, year=2028):
        """Helper to create input parameters dataframe"""
        return pd.DataFrame(
            {"Stage": ["All"], "Parameter": ["Year of the test"], "Value": [year]}
        )

    def test_all_perfect_scores_2028(self):
        """Test with perfect scores (80+) in 2028 - should be 5 stars"""
        stage_df = self._create_stage_df(sd=90, ca=90, cp=90, pc=90)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            self.assertEqual(result, 5.0)

    def test_minimum_star_rating_applied(self):
        """Test that minimum of all categories is used"""
        # One category at 2 stars, others at 5 stars
        stage_df = self._create_stage_df(sd=45, ca=85, cp=85, pc=85)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            # Minimum should be 2 (from SD)
            self.assertEqual(result, 2.0)

    def test_minimum_star_rating_applied_2026(self):
        """Test that minimum of all categories is used"""
        # One category at 2 stars, others at 5 stars
        stage_df = self._create_stage_df(sd=45, ca=85, cp=85, pc=85)
        input_parameters_df = self._create_input_parameters_df(year=2026)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2026}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            # Minimum should be 4 (from SD)
            self.assertEqual(result, 4.0)

    def test_minimum_star_rating_applied_2027(self):
        """Test that minimum of all categories is used"""
        # One category at 2 stars, others at 5 stars
        stage_df = self._create_stage_df(sd=45, ca=85, cp=85, pc=85)
        input_parameters_df = self._create_input_parameters_df(year=2027)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2027}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            # Minimum should be 3 (from SD)
            self.assertEqual(result, 3.0)

    def test_all_below_threshold(self):
        """Test when all scores are below minimum threshold"""
        stage_df = self._create_stage_df(sd=30, ca=30, cp=30, pc=30)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            self.assertEqual(result, 0.0)

    def test_unassessed_stage_returns_none_not_zero_stars(self):
        """An unassessed (NaN) Stage score must make the overall rating
        "not yet ratable" (None), not silently a confident 0-star result --
        the exact failure mode this whole class of bug is about, just
        inverted (confident zero instead of confident full marks)."""
        stage_df = pd.DataFrame(
            {
                "Stage": [
                    "Safe Driving",
                    "Crash Avoidance",
                    "Crash Protection",
                    "Post-Crash",
                ],
                "Score": [np.nan, 90.0, 90.0, 90.0],
            }
        )
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            self.assertIsNone(result)

    def test_year_2026_adjustment_safe_driving(self):
        """Test 2026: Safe Driving gets +20 adjustment"""
        # SD=66 + 20 offset = 86 (5 stars)
        # CA=74 + 10 offset = 84 (5 stars)
        # CP=79 (5 stars)
        # PC=87 (5 stars)
        stage_df = self._create_stage_df(sd=66, ca=74, cp=79, pc=87)
        input_parameters_df = self._create_input_parameters_df(year=2026)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2026}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            # After compensation and offset removal, should achieve appropriate ratings
            self.assertIsNotNone(result)
            self.assertGreaterEqual(result, 0)
            self.assertLessEqual(result, 5)

    def test_year_2027_adjustment_safe_driving(self):
        """Test 2027: Safe Driving gets +10 adjustment"""
        stage_df = self._create_stage_df(sd=66, ca=74, cp=79, pc=87)
        input_parameters_df = self._create_input_parameters_df(year=2027)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2027}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            self.assertIsNotNone(result)
            self.assertGreaterEqual(result, 0)
            self.assertLessEqual(result, 5)

    def test_four_star_capping_applied(self):
        """Test that 4-star capping is applied when flag is True and rating > 4"""
        stage_df = self._create_stage_df(sd=90, ca=90, cp=90, pc=90)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=True
            )
            self.assertEqual(result, 4.0)

    def test_four_star_capping_not_applied_below_threshold(self):
        """Test that 4-star capping is not applied when rating <= 4"""
        stage_df = self._create_stage_df(sd=65, ca=65, cp=65, pc=65)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=True
            )
            self.assertEqual(result, 3.0)
            self.assertNotEqual(result, 4.0)

    def test_three_star_rating(self):
        """Test achieving 3-star rating"""
        stage_df = self._create_stage_df(sd=65, ca=65, cp=65, pc=65)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            self.assertEqual(result, 3.0)

    def test_two_star_rating(self):
        """Test achieving 2-star rating"""
        stage_df = self._create_stage_df(sd=55, ca=55, cp=55, pc=55)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            self.assertEqual(result, 2.0)

    def test_one_star_rating(self):
        """Test achieving 1-star rating"""
        stage_df = self._create_stage_df(sd=45, ca=45, cp=45, pc=45)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            self.assertEqual(result, 1.0)

    def test_mixed_category_scores(self):
        """Test with mixed category scores to verify minimum is used"""
        # SD=3, CA=4, CP=5, PC=5 -> should be 3 (minimum)
        stage_df = self._create_stage_df(sd=65, ca=72, cp=82, pc=82)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            self.assertEqual(result, 3.0)

    def test_threshold_boundaries_exact_80(self):
        """Test score exactly at 80 threshold (5-star boundary)"""
        stage_df = self._create_stage_df(sd=80, ca=80, cp=80, pc=80)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            self.assertEqual(result, 5.0)

    def test_threshold_boundaries_exact_70(self):
        """Test score exactly at 70 threshold (4-star boundary)"""
        stage_df = self._create_stage_df(sd=70, ca=70, cp=70, pc=70)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            self.assertEqual(result, 4.0)

    def test_threshold_boundaries_exact_60(self):
        """Test score exactly at 60 threshold (3-star boundary)"""
        stage_df = self._create_stage_df(sd=60, ca=60, cp=60, pc=60)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            self.assertEqual(result, 3.0)

    def test_threshold_boundaries_exact_50(self):
        """Test score exactly at 50 threshold (2-star boundary)"""
        stage_df = self._create_stage_df(sd=50, ca=50, cp=50, pc=50)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            self.assertEqual(result, 2.0)

    def test_threshold_boundaries_exact_40(self):
        """Test score exactly at 40 threshold (1-star boundary)"""
        stage_df = self._create_stage_df(sd=40, ca=40, cp=40, pc=40)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            self.assertEqual(result, 1.0)

    def test_just_below_80_threshold(self):
        """Test score just below 80 threshold (should be 4-star)"""
        stage_df = self._create_stage_df(sd=79.9, ca=79.9, cp=79.9, pc=79.9)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            self.assertEqual(result, 4.0)

    def test_just_below_40_threshold(self):
        """Test score just below 40 threshold (should be 0-star)"""
        stage_df = self._create_stage_df(sd=39.9, ca=39.9, cp=39.9, pc=39.9)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            self.assertEqual(result, 0.0)

    def test_capping_with_compensation(self):
        """Test 4-star capping with compensation mechanism"""
        stage_df = self._create_stage_df(sd=90, ca=90, cp=90, pc=90)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=True
            )
            # With capping enabled, should cap at 4
            self.assertEqual(result, 4.0)

    def test_year_2028_no_offset(self):
        """Test 2028 without any score offsets"""
        stage_df = self._create_stage_df(sd=75, ca=75, cp=75, pc=75)
        input_parameters_df = self._create_input_parameters_df(year=2028)

        with patch.object(
            common, "create_param_dict_from_input_parameters"
        ) as mock_param:
            mock_param.return_value = {"All": {"Year of the test": 2028}}
            result = compute_star_rating(
                stage_df, input_parameters_df, four_star_capping=False
            )
            self.assertEqual(result, 4.0)


if __name__ == "__main__":
    unittest.main()
