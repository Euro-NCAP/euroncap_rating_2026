# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import random
import unittest
import importlib
import pandas as pd
from unittest.mock import patch
from euroncap_rating_2026.crash_avoidance import robustness_layer
from euroncap_rating_2026.crash_avoidance import matrix_processing
from euroncap_rating_2026.crash_avoidance import test_info
from euroncap_rating_2026.crash_avoidance import data_model
from euroncap_rating_2026 import common
from euroncap_rating_2026.crash_avoidance.compute_score import (
    check_general_requirements_rules,
    get_element_for_row,
)


class TestCrashAvoidanceDataClasses(unittest.TestCase):
    def setUp(self):
        # Set up test data for TestPoint
        self.test_point = matrix_processing.TestPoint(
            row=1,
            col=2,
            color=common.PredictionColor.GREEN,
            test_range=matrix_processing.TestRange.STANDARD,
        )

        # Set up test data for LoadcaseScore
        self.loadcase_score = test_info.LoadcaseScore(
            loadcase_name="Test Loadcase",
            standard_score=85.0,
            extended_score=15.0,
            robustness_layer_score=5.0,
            total_score=105.0,
        )

        # Set up test data for ComputedTestPoint
        self.computed_test_point = matrix_processing.ComputedTestPoint(
            test_name="Test Scenario",
            test_point=self.test_point,
            value=0.75,
            assessment_criteria=robustness_layer.AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
            prediction_result=matrix_processing.PredictionResult.CORRECT,
        )

    def test_test_point_creation(self):
        """Test TestPoint dataclass creation and attributes"""
        self.assertEqual(self.test_point.row, 1)
        self.assertEqual(self.test_point.col, 2)
        self.assertEqual(self.test_point.color, common.PredictionColor.GREEN)
        self.assertEqual(
            self.test_point.test_range, matrix_processing.TestRange.STANDARD
        )

    def test_test_point_equality(self):
        """Test TestPoint equality comparison"""
        test_point_2 = matrix_processing.TestPoint(
            row=1,
            col=2,
            color=common.PredictionColor.GREEN,
            test_range=matrix_processing.TestRange.STANDARD,
        )
        self.assertEqual(self.test_point, test_point_2)

    def test_loadcase_score_creation(self):
        """Test LoadcaseScore dataclass creation and attributes"""
        self.assertEqual(self.loadcase_score.loadcase_name, "Test Loadcase")
        self.assertEqual(self.loadcase_score.standard_score, 85.0)
        self.assertEqual(self.loadcase_score.extended_score, 15.0)
        self.assertEqual(self.loadcase_score.robustness_layer_score, 5.0)
        self.assertEqual(self.loadcase_score.total_score, 105.0)

    def test_loadcase_score_calculation(self):
        """Test LoadcaseScore total score calculation logic"""
        expected_total = (
            self.loadcase_score.standard_score
            + self.loadcase_score.extended_score
            + self.loadcase_score.robustness_layer_score
        )
        self.assertEqual(self.loadcase_score.total_score, expected_total)

    def test_computed_test_point_creation(self):
        """Test ComputedTestPoint dataclass creation and attributes"""
        self.assertEqual(self.computed_test_point.test_name, "Test Scenario")
        self.assertEqual(self.computed_test_point.test_point, self.test_point)
        self.assertEqual(self.computed_test_point.value, 0.75)
        self.assertEqual(
            self.computed_test_point.assessment_criteria,
            robustness_layer.AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
        )
        self.assertEqual(
            self.computed_test_point.prediction_result,
            matrix_processing.PredictionResult.UNKNOWN,
        )

    def test_computed_test_point_with_none_assessment_criteria(self):
        """Test ComputedTestPoint with None assessment_criteria"""
        ctp = matrix_processing.ComputedTestPoint(
            test_name="Test Scenario 2",
            test_point=self.test_point,
            value=1.25,
            assessment_criteria=None,
            prediction_result=matrix_processing.PredictionResult.INCORRECT,
        )
        self.assertIsNone(ctp.assessment_criteria)
        self.assertEqual(
            ctp.prediction_result, matrix_processing.PredictionResult.UNKNOWN
        )

    def test_get_lsc_computed_color_avoidance_success(self):
        """Test LSC computed color for avoidance scenario with successful result"""
        # Create a test point for avoidance scenario
        avoidance_ctp = matrix_processing.ComputedTestPoint(
            test_name="CCCscp SfS",
            test_point=self.test_point,
            value=0.0,  # Success value
            assessment_criteria=None,
            prediction_result=None,
        )

        color = avoidance_ctp.get_lsc_computed_color(data_model.VehicleResponse.WARNING)
        self.assertEqual(color, common.PredictionColor.GREEN)
        self.assertEqual(
            avoidance_ctp.prediction_result, matrix_processing.PredictionResult.CORRECT
        )

    def test_get_lsc_computed_color_avoidance_failure(self):
        """Test LSC computed color for avoidance scenario with failure result"""
        avoidance_ctp = matrix_processing.ComputedTestPoint(
            test_name="CCCscp SfS",
            test_point=self.test_point,
            value=1.5,  # Failure value
            assessment_criteria=None,
            prediction_result=None,
        )

        color = avoidance_ctp.get_lsc_computed_color(data_model.VehicleResponse.WARNING)
        self.assertEqual(color, common.PredictionColor.RED)
        self.assertEqual(
            avoidance_ctp.prediction_result,
            matrix_processing.PredictionResult.INCORRECT,
        )

    def test_get_lsc_computed_color_mitigation_success(self):
        """Test LSC computed color for mitigation scenario with successful result"""
        mitigation_ctp = matrix_processing.ComputedTestPoint(
            test_name="CPMFC",
            test_point=self.test_point,
            value=0.0,  # Success value
            assessment_criteria=None,
            prediction_result=None,
        )

        color = mitigation_ctp.get_lsc_computed_color(
            data_model.VehicleResponse.WARNING
        )
        self.assertEqual(color, common.PredictionColor.GREEN)

    def test_get_lsc_computed_color_cbda_information_returns_brown(self):
        """Test CBDA information response computes brown when TTC is sufficient."""
        dooring_ctp = matrix_processing.ComputedTestPoint(
            test_name="CBDA",
            test_point=matrix_processing.TestPoint(
                row=1,
                col=2,
                color=common.PredictionColor.BROWN,
                test_range=matrix_processing.TestRange.STANDARD,
            ),
            value=2.5,
            assessment_criteria=None,
            prediction_result=None,
        )

        color = dooring_ctp.get_lsc_computed_color(
            data_model.VehicleResponse.INFORMATION
        )
        self.assertEqual(color, common.PredictionColor.BROWN)

    def test_get_lsc_computed_color_cbda_warning_orange_returns_orange(self):
        """Test CBDA warning response preserves orange matrix cells."""
        dooring_ctp = matrix_processing.ComputedTestPoint(
            test_name="CBDA",
            test_point=matrix_processing.TestPoint(
                row=1,
                col=2,
                color=common.PredictionColor.ORANGE,
                test_range=matrix_processing.TestRange.STANDARD,
            ),
            value=2.0,
            assessment_criteria=None,
            prediction_result=None,
        )

        color = dooring_ctp.get_lsc_computed_color(data_model.VehicleResponse.WARNING)
        self.assertEqual(color, common.PredictionColor.ORANGE)

    def test_get_lsc_computed_color_cbda_warning_yellow_returns_yellow(self):
        """Test CBDA warning response preserves yellow matrix cells."""
        dooring_ctp = matrix_processing.ComputedTestPoint(
            test_name="CBDA",
            test_point=matrix_processing.TestPoint(
                row=1,
                col=2,
                color=common.PredictionColor.YELLOW,
                test_range=matrix_processing.TestRange.STANDARD,
            ),
            value=2.0,
            assessment_criteria=None,
            prediction_result=None,
        )

        color = dooring_ctp.get_lsc_computed_color(data_model.VehicleResponse.WARNING)
        self.assertEqual(color, common.PredictionColor.YELLOW)

    def test_get_lsc_computed_color_cbda_retention_green_returns_green(self):
        """Test CBDA retention returns green when value passes and prediction is green."""
        dooring_ctp = matrix_processing.ComputedTestPoint(
            test_name="CBDA",
            test_point=matrix_processing.TestPoint(
                row=1,
                col=2,
                color=common.PredictionColor.GREEN,
                test_range=matrix_processing.TestRange.STANDARD,
            ),
            value=2.0,
            assessment_criteria=None,
            prediction_result=None,
        )

        color = dooring_ctp.get_lsc_computed_color(data_model.VehicleResponse.RETENTION)
        self.assertEqual(color, common.PredictionColor.GREEN)

    def test_get_lsc_computed_color_cbda_retention_yellow_returns_yellow(self):
        """Test CBDA retention returns yellow when value passes and prediction is yellow."""
        dooring_ctp = matrix_processing.ComputedTestPoint(
            test_name="CBDA",
            test_point=matrix_processing.TestPoint(
                row=1,
                col=2,
                color=common.PredictionColor.YELLOW,
                test_range=matrix_processing.TestRange.STANDARD,
            ),
            value=2.0,
            assessment_criteria=None,
            prediction_result=None,
        )

        color = dooring_ctp.get_lsc_computed_color(data_model.VehicleResponse.RETENTION)
        self.assertEqual(color, common.PredictionColor.YELLOW)

    def test_assessment_criteria_enum(self):
        """Test AssessmentCriteria enum values"""
        self.assertEqual(robustness_layer.AssessmentCriteria.NOT_RED.value, "Not red")
        self.assertEqual(
            robustness_layer.AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED.value,
            "Same or better than predicted",
        )

    def test_prediction_color_enum(self):
        """Test PredictionColor enum values"""
        self.assertEqual(common.PredictionColor.GREEN.value, "green")
        self.assertEqual(common.PredictionColor.YELLOW.value, "yellow")
        self.assertEqual(common.PredictionColor.ORANGE.value, "orange")
        self.assertEqual(common.PredictionColor.BROWN.value, "brown")
        self.assertEqual(common.PredictionColor.RED.value, "red")
        self.assertEqual(common.PredictionColor.GREY.value, "grey")

    def test_prediction_result_enum(self):
        """Test PredictionResult enum values"""
        self.assertEqual(matrix_processing.PredictionResult.CORRECT.value, "Correct")
        self.assertEqual(
            matrix_processing.PredictionResult.INCORRECT.value, "Incorrect"
        )
        self.assertEqual(
            matrix_processing.PredictionResult.IN_TOLERANCE.value, "In Tolerance"
        )

    def test_test_range_enum(self):
        """Test TestRange enum values"""
        self.assertEqual(matrix_processing.TestRange.STANDARD.value, "Standard")
        self.assertEqual(matrix_processing.TestRange.EXTENDED.value, "Extended")

    def test_vehicle_response_enum(self):
        """Test VehicleResponse enum values"""
        self.assertEqual(data_model.VehicleResponse.INFORMATION.value, "Information")
        self.assertEqual(data_model.VehicleResponse.WARNING.value, "Warning")
        self.assertEqual(data_model.VehicleResponse.RETENTION.value, "Retention")


class TestCbdaHandling(unittest.TestCase):
    def _make_cbda_test_point(self, color):
        return matrix_processing.TestPoint(
            row=0,
            col=0,
            color=color,
            test_range=matrix_processing.TestRange.STANDARD,
        )

    def _make_input_parameters_df(self, vehicle_response):
        return pd.DataFrame(
            {
                "Input parameter": ["Vehicle response"],
                "Value": [vehicle_response.value],
            }
        )

    def _make_cbda_points_df(self, predicted_color, value):
        return pd.DataFrame(
            {
                "Scenario": ["CBDA"],
                "Test point": ["(0, 0)"],
                "Range": [matrix_processing.TestRange.STANDARD.value],
                "OEM Prediction": [predicted_color.value],
                "Value": [value],
            }
        )

    def _make_standard_test_points(self, n):
        return [
            matrix_processing.TestPoint(
                row=i,
                col=0,
                color=common.PredictionColor.GREEN,
                test_range=matrix_processing.TestRange.STANDARD,
            )
            for i in range(n)
        ]

    def _make_cbda_retention_points_df(self, predicted_color, value):
        return pd.DataFrame(
            {
                "Scenario": ["CBDA"],
                "Test point": ["(0, 0)"],
                "Range": [matrix_processing.TestRange.STANDARD.value],
                "OEM Prediction": [predicted_color.value],
                "Expected value": ["TTC @ t_door_opening"],
                "Value": [value],
            }
        )

    def test_cbda_information_consistency_requires_brown(self):
        passing_points = [self._make_cbda_test_point(common.PredictionColor.BROWN)]
        failing_points = [self._make_cbda_test_point(common.PredictionColor.YELLOW)]

        self.assertTrue(
            test_info.check_lsc_consistency(
                passing_points, data_model.VehicleResponse.INFORMATION
            )
        )
        self.assertFalse(
            test_info.check_lsc_consistency(
                failing_points, data_model.VehicleResponse.INFORMATION
            )
        )

    def test_cbda_warning_consistency_requires_orange_or_yellow(self):
        passing_points = [
            self._make_cbda_test_point(common.PredictionColor.ORANGE),
            self._make_cbda_test_point(common.PredictionColor.YELLOW),
        ]
        failing_points = [self._make_cbda_test_point(common.PredictionColor.BROWN)]

        self.assertTrue(
            test_info.check_lsc_consistency(
                passing_points, data_model.VehicleResponse.WARNING
            )
        )
        self.assertFalse(
            test_info.check_lsc_consistency(
                failing_points, data_model.VehicleResponse.WARNING
            )
        )

    def test_cbda_retention_consistency_requires_green_or_yellow(self):
        passing_points = [
            self._make_cbda_test_point(common.PredictionColor.GREEN),
            self._make_cbda_test_point(common.PredictionColor.YELLOW),
        ]
        failing_points = [self._make_cbda_test_point(common.PredictionColor.ORANGE)]

        self.assertTrue(
            test_info.check_lsc_consistency(
                passing_points, data_model.VehicleResponse.RETENTION
            )
        )
        self.assertFalse(
            test_info.check_lsc_consistency(
                failing_points, data_model.VehicleResponse.RETENTION
            )
        )

    def test_cbda_empty_points_always_passes(self):
        # Empty grid (all grey) must pass regardless of vehicle response value,
        # including None (template default). This is the regression case from the
        # bug where an LDC-only upload was blocked by the template CBDA sheet.
        self.assertTrue(test_info.check_lsc_consistency([], None))

    def test_cbda_empty_points_with_valid_response_passes(self):
        self.assertTrue(
            test_info.check_lsc_consistency([], data_model.VehicleResponse.INFORMATION)
        )

    def test_cbda_information_scoring_uses_brown_match(self):
        loadcase_score, computed_color_dict = test_info.compute_lsc_loadcase_score(
            "CBDA",
            {"CBDA": self._make_cbda_points_df(common.PredictionColor.BROWN, 2.5)},
            self._make_input_parameters_df(data_model.VehicleResponse.INFORMATION),
            self._make_standard_test_points(1),
            matrix_total_cells=1,
        )

        self.assertEqual(loadcase_score.standard_score, 0.5)
        self.assertEqual(loadcase_score.total_score, 0.5)
        self.assertEqual(
            computed_color_dict["CBDA_(0, 0)"], common.PredictionColor.BROWN
        )

    def test_cbda_warning_scoring_distinguishes_orange_and_yellow(self):
        warning_cases = [
            (common.PredictionColor.ORANGE, 1.0),
            (common.PredictionColor.YELLOW, 1.5),
        ]

        for predicted_color, expected_score in warning_cases:
            with self.subTest(predicted_color=predicted_color):
                loadcase_score, computed_color_dict = (
                    test_info.compute_lsc_loadcase_score(
                        "CBDA",
                        {"CBDA": self._make_cbda_points_df(predicted_color, 2.0)},
                        self._make_input_parameters_df(
                            data_model.VehicleResponse.WARNING
                        ),
                        self._make_standard_test_points(1),
                        matrix_total_cells=1,
                    )
                )

                self.assertEqual(loadcase_score.standard_score, expected_score)
                self.assertEqual(loadcase_score.total_score, expected_score)
                self.assertEqual(computed_color_dict["CBDA_(0, 0)"], predicted_color)

    def test_cbda_warning_scoring_uses_computed_color_when_value_specified(self):
        # OEM=BROWN is inconsistent for WARNING (expects ORANGE/YELLOW), but when Value
        # is specified the computed color (YELLOW, TTC=2.0≥1.70) determines the score.
        loadcase_score, computed_color_dict = test_info.compute_lsc_loadcase_score(
            "CBDA",
            {"CBDA": self._make_cbda_points_df(common.PredictionColor.BROWN, 2.0)},
            self._make_input_parameters_df(data_model.VehicleResponse.WARNING),
            self._make_standard_test_points(1),
            matrix_total_cells=1,
        )

        self.assertAlmostEqual(loadcase_score.standard_score, 0.75 * 2.0, places=3)
        self.assertAlmostEqual(loadcase_score.total_score, 0.75 * 2.0, places=3)
        self.assertEqual(
            computed_color_dict["CBDA_(0, 0)"], common.PredictionColor.YELLOW
        )

    def test_cbda_retention_scoring_distinguishes_green_and_yellow(self):
        retention_cases = [
            (common.PredictionColor.GREEN, 2.0),
            (common.PredictionColor.YELLOW, 1.5),
        ]

        for predicted_color, expected_score in retention_cases:
            with self.subTest(predicted_color=predicted_color):
                loadcase_score, computed_color_dict = (
                    test_info.compute_lsc_loadcase_score(
                        "CBDA",
                        {
                            "CBDA": self._make_cbda_retention_points_df(
                                predicted_color, 2.0
                            )
                        },
                        self._make_input_parameters_df(
                            data_model.VehicleResponse.RETENTION
                        ),
                        self._make_standard_test_points(1),
                        matrix_total_cells=1,
                    )
                )

                self.assertEqual(loadcase_score.standard_score, expected_score)
                self.assertEqual(loadcase_score.total_score, expected_score)
                self.assertEqual(computed_color_dict["CBDA_(0, 0)"], predicted_color)

    def test_all_red_predictions_returns_zero_score(self):
        """All OEM predictions RED yield zero score (color_score_map[RED] == 0.0)."""
        points_df = pd.DataFrame(
            {
                "Scenario": ["CCCscp SfS", "CCCscp SfS"],
                "Test point": ["(0, 0)", "(1, 0)"],
                "Range": [
                    matrix_processing.TestRange.STANDARD.value,
                    matrix_processing.TestRange.STANDARD.value,
                ],
                "OEM Prediction": [
                    common.PredictionColor.RED.value,
                    common.PredictionColor.RED.value,
                ],
                "Value": [1.5, 1.5],
            }
        )
        input_params_df = pd.DataFrame(
            {
                "Input parameter": ["Vehicle response"],
                "Value": [data_model.VehicleResponse.WARNING.value],
            }
        )

        loadcase_score, computed_color_dict = test_info.compute_lsc_loadcase_score(
            "CCCscp SfS",
            {"CCCscp SfS": points_df},
            input_params_df,
            self._make_standard_test_points(2),
            matrix_total_cells=2,
        )

        self.assertEqual(loadcase_score.standard_score, 0.0)
        self.assertEqual(loadcase_score.extended_score, 0.0)
        self.assertEqual(loadcase_score.robustness_layer_score, 0.0)
        self.assertEqual(loadcase_score.total_score, 0.0)
        for color in computed_color_dict.values():
            self.assertEqual(color, common.PredictionColor.RED)

    def test_cmftap_sfs_red_green_green_red_score(self):
        """CMFtap SfS with 4 points (Red, Green, Green, Red) all matching yields 1.5/3.0."""
        points_df = pd.DataFrame(
            {
                "Scenario": ["CMFtap SfS"] * 4,
                "Test point": ["(0, 0)", "(1, 0)", "(2, 0)", "(3, 0)"],
                "Range": [matrix_processing.TestRange.STANDARD.value] * 4,
                "OEM Prediction": [
                    common.PredictionColor.RED.value,
                    common.PredictionColor.GREEN.value,
                    common.PredictionColor.GREEN.value,
                    common.PredictionColor.RED.value,
                ],
                "Value": [1.0, 0.0, 0.0, 1.0],
            }
        )
        input_params_df = pd.DataFrame(
            {
                "Input parameter": ["Vehicle response"],
                "Value": [data_model.VehicleResponse.WARNING.value],
            }
        )
        all_test_points = self._make_standard_test_points(4)

        loadcase_score, computed_color_dict = test_info.compute_lsc_loadcase_score(
            "CMFtap SfS",
            {"CMFtap SfS": points_df},
            input_params_df,
            all_test_points,
            matrix_total_cells=4,
        )

        self.assertAlmostEqual(loadcase_score.standard_score, 1.5)
        self.assertAlmostEqual(loadcase_score.total_score, 1.5)
        self.assertEqual(loadcase_score.extended_score, 0.0)
        self.assertEqual(loadcase_score.robustness_layer_score, 0.0)
        self.assertEqual(
            computed_color_dict["CMFtap SfS_(0, 0)"], common.PredictionColor.RED
        )
        self.assertEqual(
            computed_color_dict["CMFtap SfS_(1, 0)"], common.PredictionColor.GREEN
        )
        self.assertEqual(
            computed_color_dict["CMFtap SfS_(2, 0)"], common.PredictionColor.GREEN
        )
        self.assertEqual(
            computed_color_dict["CMFtap SfS_(3, 0)"], common.PredictionColor.RED
        )


class TestLoadcaseSelectionBoundaries(unittest.TestCase):
    def test_ccrb_target_speed_threshold_keeps_80_and_excludes_81(self):
        test_points = [
            matrix_processing.TestPoint(
                row=0,
                col=0,
                color=common.PredictionColor.GREEN,
                test_range=matrix_processing.TestRange.STANDARD,
                attributes={"Target speed": "79 km/h"},
            ),
            matrix_processing.TestPoint(
                row=0,
                col=1,
                color=common.PredictionColor.GREEN,
                test_range=matrix_processing.TestRange.STANDARD,
                attributes={"Target speed": "80 km/h"},
            ),
            matrix_processing.TestPoint(
                row=0,
                col=2,
                color=common.PredictionColor.GREEN,
                test_range=matrix_processing.TestRange.STANDARD,
                attributes={"Target speed": "81 km/h"},
            ),
        ]
        loadcase_info = test_info.LoadcaseInfo(
            test_name="CCRb",
            test_points=test_points,
            robustness_layers={},
            num_extended_tests=0,
            num_robustness_tests=0,
        )

        selection = loadcase_info.select_test_points()

        selected_target_speeds = {
            test_point.attributes["Target speed"]
            for test_point in selection.standard_points
        }

        self.assertEqual(selected_target_speeds, {"79 km/h", "80 km/h"})
        self.assertNotIn("81 km/h", selected_target_speeds)
        self.assertEqual(selection.extended_points, [])
        self.assertIsNone(selection.selected_robustness_layer)

    def test_ccrb_all_above_threshold_selects_no_points(self):
        test_points = [
            matrix_processing.TestPoint(
                row=0,
                col=0,
                color=common.PredictionColor.GREEN,
                test_range=matrix_processing.TestRange.STANDARD,
                attributes={"Target speed": "81 km/h"},
            ),
            matrix_processing.TestPoint(
                row=0,
                col=1,
                color=common.PredictionColor.GREEN,
                test_range=matrix_processing.TestRange.STANDARD,
                attributes={"Target speed": "90 km/h"},
            ),
        ]
        loadcase_info = test_info.LoadcaseInfo(
            test_name="CCRb",
            test_points=test_points,
            robustness_layers={},
            num_extended_tests=0,
            num_robustness_tests=0,
        )

        selection = loadcase_info.select_test_points()

        self.assertEqual(selection.standard_points, [])
        self.assertEqual(selection.extended_points, [])
        self.assertIsNone(selection.selected_robustness_layer)

    def _make_loadcase_info(self, test_name, speed_attr, speeds):
        test_points = [
            matrix_processing.TestPoint(
                row=0,
                col=i,
                color=common.PredictionColor.GREEN,
                test_range=matrix_processing.TestRange.STANDARD,
                attributes={speed_attr: f"{speed} km/h"},
            )
            for i, speed in enumerate(speeds)
        ]
        return test_info.LoadcaseInfo(
            test_name=test_name,
            test_points=test_points,
            robustness_layers={},
            num_extended_tests=0,
            num_robustness_tests=0,
        )

    def _selected_speeds(self, selection, speed_attr):
        return {tp.attributes[speed_attr] for tp in selection.standard_points}

    # --- CCFhos / CCFhol (Corner Cases in full; CA Frontal 4.1.3 + the 4.2.1
    # "Verification Test not applicable" footnote). CCFhol never gets a
    # verification test; CCFhos gets exactly the pinned VUT 30 km/h x 50%
    # impact location cell, data_model.CCFHOS_PINNED_TEST_POINT. ---

    # The real CCFhos/CCFhol matrices are 8 rows x 4 cols: rows are VUT
    # 30..100 km/h, cols are the 100%/75%/50%/25% impact locations.
    _CCFH_VUT_SPEEDS = [30, 40, 50, 60, 70, 80, 90, 100]
    _CCFH_TARGET_SPEEDS = [50, 50, 50, 70, 70, 70, 90, 100]
    _CCFH_IMPACT_LOCATIONS = ["1", "0.75", "0.5", "0.25"]

    def _make_full_ccfh_loadcase_info(self, test_name, colors=None):
        """Build a LoadcaseInfo over the full 8x4 CCFhos/CCFhol matrix.

        Every cell is GREEN unless overridden through *colors*, a
        {(row, col): PredictionColor} dict. test_range is taken from
        data_model.EXTENDED_RANGE_CELLS so the Standard/Extended split
        matches the real sheet.
        """
        colors = colors or {}
        extended_cells = set(data_model.EXTENDED_RANGE_CELLS[test_name])
        test_points = [
            matrix_processing.TestPoint(
                row=row,
                col=col,
                color=colors.get((row, col), common.PredictionColor.GREEN),
                test_range=(
                    matrix_processing.TestRange.EXTENDED
                    if (row, col) in extended_cells
                    else matrix_processing.TestRange.STANDARD
                ),
                attributes={
                    "VUT speed": f"{self._CCFH_VUT_SPEEDS[row]} km/h",
                    "Target speed": f"{self._CCFH_TARGET_SPEEDS[row]} km/h",
                    "Impact location": self._CCFH_IMPACT_LOCATIONS[col],
                },
            )
            for row in range(8)
            for col in range(4)
        ]
        return test_info.LoadcaseInfo(
            test_name=test_name,
            test_points=test_points,
            robustness_layers={},
            stage_subelement_key=data_model.StageSubelementKey.FC,
        )

    @staticmethod
    def _positions(points):
        return [(tp.row, tp.col) for tp in points]

    def test_ccfhos_pinned_point_is_standard_range(self):
        """The pinning relies on the pinned cell classifying as Standard --
        guard that against a future EXTENDED_RANGE_CELLS edit."""
        self.assertNotIn(
            data_model.CCFHOS_PINNED_TEST_POINT,
            set(data_model.EXTENDED_RANGE_CELLS["CCFhos"]),
        )

    def test_ccfhos_selects_only_the_pinned_point(self):
        info = self._make_full_ccfh_loadcase_info("CCFhos")

        selection = info.select_test_points()

        self.assertEqual(
            self._positions(selection.standard_points),
            [data_model.CCFHOS_PINNED_TEST_POINT],
        )
        self.assertEqual(selection.extended_points, [])

    def test_ccfhos_pinned_point_selected_on_every_seed(self):
        """ "Always selected", not "usually": the pinned cell is the only
        candidate, so every random seed must yield it."""
        self.addCleanup(random.seed)
        for seed in range(25):
            with self.subTest(seed=seed):
                random.seed(seed)
                selection = self._make_full_ccfh_loadcase_info(
                    "CCFhos"
                ).select_test_points()
                self.assertEqual(
                    self._positions(selection.standard_points),
                    [data_model.CCFHOS_PINNED_TEST_POINT],
                )

    def test_ccfhos_pinned_point_red_selects_no_points(self):
        info = self._make_full_ccfh_loadcase_info(
            "CCFhos",
            {data_model.CCFHOS_PINNED_TEST_POINT: common.PredictionColor.RED},
        )

        selection = info.select_test_points()

        self.assertEqual(selection.standard_points, [])
        self.assertEqual(selection.extended_points, [])

    def test_ccfhos_pinned_point_grey_selects_no_points(self):
        info = self._make_full_ccfh_loadcase_info(
            "CCFhos",
            {data_model.CCFHOS_PINNED_TEST_POINT: common.PredictionColor.GREY},
        )

        selection = info.select_test_points()

        self.assertEqual(selection.standard_points, [])
        self.assertEqual(selection.extended_points, [])

    def test_ccfhos_other_cells_never_selected_whatever_their_colour(self):
        """Only the pinned cell is eligible: reds elsewhere change nothing,
        and no other scored cell can substitute for it."""
        colors = {
            (row, col): common.PredictionColor.RED
            for row in range(8)
            for col in range(4)
            if (row, col) != data_model.CCFHOS_PINNED_TEST_POINT
        }
        info = self._make_full_ccfh_loadcase_info("CCFhos", colors)

        selection = info.select_test_points()

        self.assertEqual(
            self._positions(selection.standard_points),
            [data_model.CCFHOS_PINNED_TEST_POINT],
        )
        self.assertEqual(selection.extended_points, [])

    def test_ccfhol_selects_no_points(self):
        info = self._make_full_ccfh_loadcase_info("CCFhol")

        selection = info.select_test_points()

        self.assertEqual(selection.standard_points, [])
        self.assertEqual(selection.extended_points, [])

    def test_ccfhol_selects_no_points_on_every_seed(self):
        self.addCleanup(random.seed)
        for seed in range(25):
            with self.subTest(seed=seed):
                random.seed(seed)
                selection = self._make_full_ccfh_loadcase_info(
                    "CCFhol"
                ).select_test_points()
                self.assertEqual(selection.standard_points, [])
                self.assertEqual(selection.extended_points, [])

    def test_ccfhol_pinned_ccfhos_position_is_not_special(self):
        """CCFhos's exception must not leak into CCFhol."""
        info = self._make_full_ccfh_loadcase_info("CCFhol")

        selection = info.select_test_points()

        self.assertNotIn(
            data_model.CCFHOS_PINNED_TEST_POINT,
            self._positions(selection.standard_points),
        )

    # --- CMRb (Target speed <= 80) ---

    def test_cmrb_keeps_80_and_excludes_81(self):
        info = self._make_loadcase_info("CMRb", "Target speed", [79, 80, 81])
        selection = info.select_test_points()
        speeds = self._selected_speeds(selection, "Target speed")
        self.assertEqual(speeds, {"79 km/h", "80 km/h"})
        self.assertNotIn("81 km/h", speeds)

    def test_cmrb_all_above_threshold_selects_no_points(self):
        info = self._make_loadcase_info("CMRb", "Target speed", [81, 90])
        selection = info.select_test_points()
        self.assertEqual(selection.standard_points, [])

    # --- CCFtap (Target speed <= 60) ---

    def test_ccftap_keeps_60_and_excludes_61(self):
        info = self._make_loadcase_info("CCFtap", "Target speed", [59, 60, 61])
        selection = info.select_test_points()
        speeds = self._selected_speeds(selection, "Target speed")
        self.assertEqual(speeds, {"59 km/h", "60 km/h"})
        self.assertNotIn("61 km/h", speeds)

    def test_ccftap_all_above_threshold_selects_no_points(self):
        info = self._make_loadcase_info("CCFtap", "Target speed", [61, 70])
        selection = info.select_test_points()
        self.assertEqual(selection.standard_points, [])

    # --- CMFtap (Target speed <= 60) ---

    def test_cmftap_keeps_60_and_excludes_61(self):
        info = self._make_loadcase_info("CMFtap", "Target speed", [59, 60, 61])
        selection = info.select_test_points()
        speeds = self._selected_speeds(selection, "Target speed")
        self.assertEqual(speeds, {"59 km/h", "60 km/h"})
        self.assertNotIn("61 km/h", speeds)

    def test_cmftap_all_above_threshold_selects_no_points(self):
        info = self._make_loadcase_info("CMFtap", "Target speed", [61, 70])
        selection = info.select_test_points()
        self.assertEqual(selection.standard_points, [])

    # --- CCCscp (Target speed <= 60) ---

    def test_cccscp_keeps_60_and_excludes_61(self):
        info = self._make_loadcase_info("CCCscp", "Target speed", [59, 60, 61])
        selection = info.select_test_points()
        speeds = self._selected_speeds(selection, "Target speed")
        self.assertEqual(speeds, {"59 km/h", "60 km/h"})
        self.assertNotIn("61 km/h", speeds)

    def test_cccscp_all_above_threshold_selects_no_points(self):
        info = self._make_loadcase_info("CCCscp", "Target speed", [61, 70])
        selection = info.select_test_points()
        self.assertEqual(selection.standard_points, [])

    # --- CMCscp (Target speed <= 60) ---

    def test_cmcscp_keeps_60_and_excludes_61(self):
        info = self._make_loadcase_info("CMCscp", "Target speed", [59, 60, 61])
        selection = info.select_test_points()
        speeds = self._selected_speeds(selection, "Target speed")
        self.assertEqual(speeds, {"59 km/h", "60 km/h"})
        self.assertNotIn("61 km/h", speeds)

    def test_cmcscp_all_above_threshold_selects_no_points(self):
        info = self._make_loadcase_info("CMCscp", "Target speed", [61, 70])
        selection = info.select_test_points()
        self.assertEqual(selection.standard_points, [])

    # --- CM ELK (Target speed <= 80) ---

    def test_cm_elk_keeps_80_and_excludes_81(self):
        info = self._make_loadcase_info("CM ELK On", "Target speed", [79, 80, 81])
        selection = info.select_test_points()
        speeds = self._selected_speeds(selection, "Target speed")
        self.assertIn("79 km/h", speeds)
        self.assertIn("80 km/h", speeds)
        self.assertNotIn("81 km/h", speeds)

    def test_cm_elk_all_above_threshold_selects_no_points(self):
        info = self._make_loadcase_info("CM ELK On", "Target speed", [81, 90])
        selection = info.select_test_points()
        self.assertEqual(selection.standard_points, [])

    def test_cc_elk_ovi_initial_position_offset_has_verification_condition(self):
        info = self._make_loadcase_info("CC ELK OvI", "Target speed", [79, 80])
        info.robustness_layers = {"Initial position offset": "YES"}
        info.num_robustness_tests = 1
        info.stage_subelement_key = data_model.StageSubelementKey.LDC

        selection = info.select_test_points()

        self.assertEqual(
            selection.selected_robustness_layer,
            "Initial position offset",
        )
        self.assertIn(
            selection.verification_condition,
            {"0.25m away from VUT", "0.5m closer to VUT"},
        )
        self.assertEqual(
            selection.assessment_criteria,
            robustness_layer.AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
        )

    def test_cc_elk_ovi_assessment_criteria_uses_canonical_robustness_mapping(self):
        assessment_criteria = robustness_layer.get_assessment_criteria(
            robustness_layer.RobustnessLayer.INITIAL_POSITION_OFFSET,
            "CC ELK OvI",
            "0.25m away from VUT",
        )

        self.assertEqual(
            assessment_criteria,
            robustness_layer.AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
        )

    # --- ELK RE (VUT speed <= 80) ---

    def test_elk_re_keeps_80_and_excludes_81(self):
        info = self._make_loadcase_info("ELK RE", "VUT speed", [79, 80, 81])
        selection = info.select_test_points()
        speeds = self._selected_speeds(selection, "VUT speed")
        self.assertEqual(speeds, {"79 km/h", "80 km/h"})
        self.assertNotIn("81 km/h", speeds)

    def test_elk_re_all_above_threshold_selects_no_points(self):
        info = self._make_loadcase_info("ELK RE", "VUT speed", [81, 90])
        selection = info.select_test_points()
        self.assertEqual(selection.standard_points, [])

    # --- ELK On + "Impact location" → exclude 0.3 m/s lateral velocity ---

    def _make_elk_on_loadcase_info(self, test_name, lateral_velocities):
        """Build a LoadcaseInfo for an ELK On test with the given lateral velocities."""
        test_points = [
            matrix_processing.TestPoint(
                row=0,
                col=i,
                color=common.PredictionColor.GREEN,
                test_range=matrix_processing.TestRange.STANDARD,
                attributes={"Lateral velocity": lat_vel, "Target speed": "70 km/h"},
            )
            for i, lat_vel in enumerate(lateral_velocities)
        ]
        return test_info.LoadcaseInfo(
            test_name=test_name,
            test_points=test_points,
            robustness_layers={"Impact location": "YES"},
            num_extended_tests=0,
            num_robustness_tests=1,
            stage_subelement_key=data_model.StageSubelementKey.LDC,
        )

    def test_cc_elk_on_impact_location_excludes_0_3_ms_lateral_velocity(self):
        info = self._make_elk_on_loadcase_info(
            "CC ELK On", ["0.1 m/s", "0.3 m/s", "0.5 m/s"]
        )
        selection = info.select_test_points()
        lat_vels = {
            tp.attributes["Lateral velocity"] for tp in selection.standard_points
        }
        self.assertNotIn("0.3 m/s", lat_vels)
        self.assertEqual(selection.selected_robustness_layer, "Impact location")

    def test_cm_elk_on_impact_location_excludes_0_3_ms_lateral_velocity(self):
        info = self._make_elk_on_loadcase_info(
            "CM ELK On", ["0.1 m/s", "0.3 m/s", "0.5 m/s"]
        )
        selection = info.select_test_points()
        lat_vels = {
            tp.attributes["Lateral velocity"] for tp in selection.standard_points
        }
        self.assertNotIn("0.3 m/s", lat_vels)
        self.assertEqual(selection.selected_robustness_layer, "Impact location")

    def test_cc_elk_on_other_robustness_layer_keeps_0_3_ms_lateral_velocity(self):
        """When 'Initial position offset' is chosen instead, 0.3 m/s must remain selectable."""
        test_points = [
            matrix_processing.TestPoint(
                row=0,
                col=i,
                color=common.PredictionColor.GREEN,
                test_range=matrix_processing.TestRange.STANDARD,
                attributes={"Lateral velocity": lat_vel, "Target speed": "70 km/h"},
            )
            for i, lat_vel in enumerate(["0.1 m/s", "0.3 m/s", "0.5 m/s"])
        ]
        info = test_info.LoadcaseInfo(
            test_name="CC ELK On",
            test_points=test_points,
            robustness_layers={
                "Initial position offset": "YES",
                "Impact location": "NO",
            },
            num_extended_tests=0,
            num_robustness_tests=1,
            stage_subelement_key=data_model.StageSubelementKey.LDC,
        )
        # With num_standard_tests=3 and all points present, 0.3 m/s must be a candidate
        # (we can't assert it's selected since it's random, but the filtered pool must include it)
        from unittest.mock import patch as _patch

        with _patch(
            "euroncap_rating_2026.crash_avoidance.test_info.random.sample",
            side_effect=lambda pop, k: list(pop)[:k],
        ) as _mock:
            selection = info.select_test_points()
        self.assertEqual(selection.selected_robustness_layer, "Initial position offset")
        lat_vels = {
            tp.attributes["Lateral velocity"] for tp in selection.standard_points
        }
        self.assertIn("0.3 m/s", lat_vels)

    # --- ELK Ov + "Initial position offset" → exclude purple cells ---

    def _make_elk_ov_loadcase_info(self, test_name, lat_vels):
        """Build a LoadcaseInfo for an ELK Ov test with the given lateral velocities."""
        test_points = [
            matrix_processing.TestPoint(
                row=0,
                col=i,
                color=common.PredictionColor.GREEN,
                test_range=matrix_processing.TestRange.STANDARD,
                attributes={
                    "VUT speed": "70 km/h",
                    "Target speed": "70 km/h",
                    "Lateral velocity": lat_vel,
                },
            )
            for i, lat_vel in enumerate(lat_vels)
        ]
        return test_info.LoadcaseInfo(
            test_name=test_name,
            test_points=test_points,
            robustness_layers={"Initial position offset": "YES"},
            num_extended_tests=0,
            num_robustness_tests=1,
            stage_subelement_key=data_model.StageSubelementKey.LDC,
        )

    def test_cc_elk_ovi_initial_position_offset_excludes_purple_cells(self):
        purple = [
            {
                "Scenario": "CC ELK OvI",
                "VUT speed": "70 km/h",
                "Target speed": "70 km/h",
                "Lateral velocity": "0.3 m/s",
            }
        ]
        info = self._make_elk_ov_loadcase_info(
            "CC ELK OvI", ["0.1 m/s", "0.3 m/s", "0.5 m/s"]
        )
        original = data_model.ELK_OV_INITIAL_POSITION_OFFSET_EXCLUDED_CELLS
        try:
            data_model.ELK_OV_INITIAL_POSITION_OFFSET_EXCLUDED_CELLS = purple
            selection = info.select_test_points()
        finally:
            data_model.ELK_OV_INITIAL_POSITION_OFFSET_EXCLUDED_CELLS = original

        lat_vels = {
            tp.attributes["Lateral velocity"] for tp in selection.standard_points
        }
        self.assertNotIn("0.3 m/s", lat_vels)
        self.assertEqual(selection.selected_robustness_layer, "Initial position offset")

    def test_cc_elk_ovi_impact_location_keeps_purple_cells_selectable(self):
        """When 'Impact location' is chosen instead, purple cells must remain in the pool."""
        purple = [
            {
                "Scenario": "CC ELK OvI",
                "VUT speed": "70 km/h",
                "Target speed": "70 km/h",
                "Lateral velocity": "0.3 m/s",
            }
        ]
        test_points = [
            matrix_processing.TestPoint(
                row=0,
                col=i,
                color=common.PredictionColor.GREEN,
                test_range=matrix_processing.TestRange.STANDARD,
                attributes={
                    "VUT speed": "70 km/h",
                    "Target speed": "70 km/h",
                    "Lateral velocity": lat_vel,
                },
            )
            for i, lat_vel in enumerate(["0.1 m/s", "0.3 m/s", "0.5 m/s"])
        ]
        info = test_info.LoadcaseInfo(
            test_name="CC ELK OvI",
            test_points=test_points,
            robustness_layers={
                "Impact location": "YES",
                "Initial position offset": "NO",
            },
            num_extended_tests=0,
            num_robustness_tests=1,
            stage_subelement_key=data_model.StageSubelementKey.LDC,
        )
        original = data_model.ELK_OV_INITIAL_POSITION_OFFSET_EXCLUDED_CELLS
        try:
            data_model.ELK_OV_INITIAL_POSITION_OFFSET_EXCLUDED_CELLS = purple
            from unittest.mock import patch as _patch

            with _patch(
                "euroncap_rating_2026.crash_avoidance.test_info.random.sample",
                side_effect=lambda pop, k: list(pop)[:k],
            ):
                selection = info.select_test_points()
        finally:
            data_model.ELK_OV_INITIAL_POSITION_OFFSET_EXCLUDED_CELLS = original

        self.assertEqual(selection.selected_robustness_layer, "Impact location")
        lat_vels = {
            tp.attributes["Lateral velocity"] for tp in selection.standard_points
        }
        self.assertIn("0.3 m/s", lat_vels)

    def test_purple_cell_for_different_scenario_does_not_affect_this_test(self):
        """A purple cell registered under 'CM ELK OvI' must not affect 'CC ELK OvI'."""
        purple = [
            {
                "Scenario": "CM ELK OvI",  # different scenario
                "VUT speed": "70 km/h",
                "Target speed": "70 km/h",
                "Lateral velocity": "0.3 m/s",
            }
        ]
        info = self._make_elk_ov_loadcase_info(
            "CC ELK OvI", ["0.1 m/s", "0.3 m/s", "0.5 m/s"]
        )
        original = data_model.ELK_OV_INITIAL_POSITION_OFFSET_EXCLUDED_CELLS
        try:
            data_model.ELK_OV_INITIAL_POSITION_OFFSET_EXCLUDED_CELLS = purple
            from unittest.mock import patch as _patch

            with _patch(
                "euroncap_rating_2026.crash_avoidance.test_info.random.sample",
                side_effect=lambda pop, k: list(pop)[:k],
            ):
                selection = info.select_test_points()
        finally:
            data_model.ELK_OV_INITIAL_POSITION_OFFSET_EXCLUDED_CELLS = original

        lat_vels = {
            tp.attributes["Lateral velocity"] for tp in selection.standard_points
        }
        self.assertIn("0.3 m/s", lat_vels)

    # --- CC/CM ELK On/Ov: only GREEN predictions are selectable ---

    def _make_elk_color_loadcase_info(self, test_name, colors):
        """Build a LoadcaseInfo for an ELK test with the given prediction colors."""
        test_points = [
            matrix_processing.TestPoint(
                row=0,
                col=i,
                color=color,
                test_range=matrix_processing.TestRange.STANDARD,
                attributes={"Target speed": "70 km/h", "VUT speed": "70 km/h"},
            )
            for i, color in enumerate(colors)
        ]
        return test_info.LoadcaseInfo(
            test_name=test_name,
            test_points=test_points,
            robustness_layers={},
            num_extended_tests=0,
            num_robustness_tests=0,
        )

    def test_cc_elk_on_only_selects_green_points(self):
        info = self._make_elk_color_loadcase_info(
            "CC ELK On",
            [
                common.PredictionColor.GREEN,
                common.PredictionColor.YELLOW,
                common.PredictionColor.ORANGE,
                common.PredictionColor.BROWN,
                common.PredictionColor.RED,
                common.PredictionColor.GREY,
            ],
        )
        selection = info.select_test_points()
        colors = {tp.color for tp in selection.standard_points}
        self.assertEqual(colors, {common.PredictionColor.GREEN})

    def test_cm_elk_on_only_selects_green_points(self):
        info = self._make_elk_color_loadcase_info(
            "CM ELK On",
            [
                common.PredictionColor.GREEN,
                common.PredictionColor.YELLOW,
                common.PredictionColor.ORANGE,
                common.PredictionColor.BROWN,
                common.PredictionColor.RED,
            ],
        )
        selection = info.select_test_points()
        colors = {tp.color for tp in selection.standard_points}
        self.assertEqual(colors, {common.PredictionColor.GREEN})

    def test_cc_elk_ovu_only_selects_green_points(self):
        info = self._make_elk_color_loadcase_info(
            "CC ELK OvU",
            [
                common.PredictionColor.GREEN,
                common.PredictionColor.YELLOW,
                common.PredictionColor.ORANGE,
                common.PredictionColor.BROWN,
            ],
        )
        selection = info.select_test_points()
        colors = {tp.color for tp in selection.standard_points}
        self.assertEqual(colors, {common.PredictionColor.GREEN})

    def test_cc_elk_ovi_only_selects_green_points(self):
        info = self._make_elk_color_loadcase_info(
            "CC ELK OvI",
            [
                common.PredictionColor.GREEN,
                common.PredictionColor.YELLOW,
                common.PredictionColor.ORANGE,
                common.PredictionColor.BROWN,
            ],
        )
        selection = info.select_test_points()
        colors = {tp.color for tp in selection.standard_points}
        self.assertEqual(colors, {common.PredictionColor.GREEN})

    def test_cm_elk_ovu_only_selects_green_points(self):
        info = self._make_elk_color_loadcase_info(
            "CM ELK OvU",
            [
                common.PredictionColor.GREEN,
                common.PredictionColor.YELLOW,
                common.PredictionColor.ORANGE,
                common.PredictionColor.BROWN,
            ],
        )
        selection = info.select_test_points()
        colors = {tp.color for tp in selection.standard_points}
        self.assertEqual(colors, {common.PredictionColor.GREEN})

    def test_cm_elk_ovi_only_selects_green_points(self):
        info = self._make_elk_color_loadcase_info(
            "CM ELK OvI",
            [
                common.PredictionColor.GREEN,
                common.PredictionColor.YELLOW,
                common.PredictionColor.ORANGE,
                common.PredictionColor.BROWN,
            ],
        )
        selection = info.select_test_points()
        colors = {tp.color for tp in selection.standard_points}
        self.assertEqual(colors, {common.PredictionColor.GREEN})

    def test_cc_elk_on_all_non_green_selects_no_points(self):
        info = self._make_elk_color_loadcase_info(
            "CC ELK On",
            [
                common.PredictionColor.YELLOW,
                common.PredictionColor.ORANGE,
                common.PredictionColor.BROWN,
                common.PredictionColor.RED,
            ],
        )
        selection = info.select_test_points()
        self.assertEqual(selection.standard_points, [])

    def test_cm_elk_ovi_all_non_green_selects_no_points(self):
        info = self._make_elk_color_loadcase_info(
            "CM ELK OvI",
            [
                common.PredictionColor.YELLOW,
                common.PredictionColor.ORANGE,
                common.PredictionColor.BROWN,
            ],
        )
        selection = info.select_test_points()
        self.assertEqual(selection.standard_points, [])

    def test_elk_re_keeps_non_green_points_selectable(self):
        """The GREEN-only restriction is scoped to ELK On/Ov and must not affect ELK RE."""
        info = self._make_elk_color_loadcase_info(
            "ELK RE",
            [
                common.PredictionColor.GREEN,
                common.PredictionColor.YELLOW,
                common.PredictionColor.ORANGE,
            ],
        )
        selection = info.select_test_points()
        colors = {tp.color for tp in selection.standard_points}
        self.assertIn(common.PredictionColor.YELLOW, colors)
        self.assertIn(common.PredictionColor.ORANGE, colors)


class TestCheckGeneralRequirementsRules(unittest.TestCase):
    """Unit tests for check_general_requirements_rules in compute_score."""

    # ------------------------------------------------------------------ helpers

    def _make_verification_df(self, gen_req_value="PASS"):
        return pd.DataFrame(
            {"Scenario": ["General requirements"], "Value": [gen_req_value]}
        )

    def _make_dfs(self, overrides=None):
        """Return a dfs dict with all sheets passing, optionally overriding some."""
        base = {
            "FC - Car & PTW verif.": self._make_verification_df("PASS"),
            "FC - Ped & Cyc verif.": self._make_verification_df("PASS"),
            "LDC - Single Veh verif.": self._make_verification_df("PASS"),
            "LDC - Car & PTW verif.": self._make_verification_df("PASS"),
            "LSC - Car & PTW verif.": self._make_verification_df("PASS"),
            "LSC - Ped & Cyc verif.": self._make_verification_df("PASS"),
        }
        if overrides:
            base.update(overrides)
        return base

    def _make_all_green_prediction_df(self):
        """160x20 DataFrame filled with GREEN — large enough for all matrix indices."""
        nrows, ncols = 160, 20
        return pd.DataFrame(
            [[common.PredictionColor.GREEN] * ncols for _ in range(nrows)]
        )

    def _make_labeled_prediction_df(self, test_name, label_row, matrix_n_rows):
        """Prediction DataFrame with a discoverable test name in column 0."""
        df = self._make_all_green_prediction_df()
        df.iloc[:, 0] = pd.NA
        df.iloc[label_row, 0] = test_name
        for row_idx in range(label_row + 1, label_row + 1 + matrix_n_rows):
            df.iloc[row_idx, 0] = "matrix row"
        return df

    def _make_prediction_df_with_red_for_test(
        self, test_name, label_row, matrix_n_rows, matrix_row, matrix_col
    ):
        """Prediction DataFrame with RED at a matrix-relative cell for a named test."""
        df = self._make_labeled_prediction_df(test_name, label_row, matrix_n_rows)
        matrix_indices = matrix_processing.get_matrix_indices(df, test_name)
        self.assertIsNotNone(matrix_indices)
        start_row = matrix_indices["start_row"]
        if start_row > 1:
            start_row -= 1
        df.iloc[start_row + matrix_row, matrix_indices["start_col"] + matrix_col] = (
            common.PredictionColor.RED
        )
        return df

    def _make_loadcase_score_dict(self):
        """Loadcase dict with non-zero scores for FC, LDC, and LSC."""
        return {
            "Frontal Collisions": {
                "Car & PTW": {
                    "CCRs": test_info.LoadcaseScore("CCRs", 1.0, 0.1, 0.1, 1.2)
                },
            },
            "Lane Departure Collisions": {
                "Single Veh": {
                    "ELK RE": test_info.LoadcaseScore("ELK RE", 2.0, 0.2, 0.2, 2.4)
                },
            },
            "Low Speed Collisions": {
                "Car & PTW": {
                    "CCCscp SfS": test_info.LoadcaseScore(
                        "CCCscp SfS", 1.0, 0.0, 0.0, 1.0
                    )
                },
            },
        }

    def _make_scenario_scores_df(self):
        """Minimal scenario_scores_df covering all three stage elements."""
        return pd.DataFrame(
            {
                "Stage element": [
                    "Frontal Collisions",
                    None,
                    None,
                    "Lane Departure Collisions",
                    None,
                    None,
                    "Low Speed Collisions",
                    None,
                ],
                "Stage subelement": [
                    "Car & PTW",
                    None,
                    None,
                    "Single Veh",
                    None,
                    None,
                    "Car & PTW",
                    None,
                ],
                "Scenario": [
                    "CCRs",
                    None,
                    None,
                    "ELK RE",
                    None,
                    None,
                    "CCCscp SfS",
                    None,
                ],
                "Layer": [
                    "Standard",
                    "Extended",
                    "Robustness",
                    "Standard",
                    "Extended",
                    "Robustness",
                    "Standard",
                    "Extended",
                ],
                "Score": [1.0, 0.1, 0.1, 2.0, 0.2, 0.2, 1.0, 0.0],
            }
        )

    def _run(
        self,
        dfs,
        loadcase_score_dict,
        scenario_scores_df,
        pred_ped_cyc=None,
        pred_car_ptw=None,
    ):
        """Call check_general_requirements_rules with (bg) dfs entries."""

        if pred_ped_cyc is None:
            pred_ped_cyc = self._make_all_green_prediction_df()
        if pred_car_ptw is None:
            pred_car_ptw = self._make_all_green_prediction_df()

        dfs["FC - Ped & Cyc pred. (bg)"] = pred_ped_cyc
        dfs["FC - Car & PTW pred. (bg)"] = pred_car_ptw

        check_general_requirements_rules(dfs, loadcase_score_dict, scenario_scores_df)

    @staticmethod
    def _all_zeroed(stage_dict):
        """Return True if every LoadcaseScore in stage_dict has all scores == 0."""
        return all(
            lcs.total_score == 0.0 and lcs.standard_score == 0.0
            for sub in stage_dict.values()
            for lcs in sub.values()
        )

    # ------------------------------------------------------------------ tests

    def test_all_pass_no_scores_zeroed(self):
        lsd = self._make_loadcase_score_dict()
        self._run(self._make_dfs(), lsd, self._make_scenario_scores_df())
        self.assertFalse(self._all_zeroed(lsd["Frontal Collisions"]))
        self.assertFalse(self._all_zeroed(lsd["Lane Departure Collisions"]))
        self.assertFalse(self._all_zeroed(lsd["Low Speed Collisions"]))

    # --- FC rules ---

    def test_fc_fails_when_car_ptw_gr_fails(self):
        dfs = self._make_dfs(
            {"FC - Car & PTW verif.": self._make_verification_df("FAIL")}
        )
        lsd = self._make_loadcase_score_dict()
        self._run(dfs, lsd, self._make_scenario_scores_df())
        self.assertTrue(self._all_zeroed(lsd["Frontal Collisions"]))
        self.assertFalse(self._all_zeroed(lsd["Lane Departure Collisions"]))
        self.assertFalse(self._all_zeroed(lsd["Low Speed Collisions"]))

    def test_fc_fails_when_ped_cyc_gr_fails(self):
        dfs = self._make_dfs(
            {"FC - Ped & Cyc verif.": self._make_verification_df("FAIL")}
        )
        lsd = self._make_loadcase_score_dict()
        self._run(dfs, lsd, self._make_scenario_scores_df())
        self.assertTrue(self._all_zeroed(lsd["Frontal Collisions"]))

    def test_fc_fails_when_cpna_day_not_green(self):
        pred_df = self._make_prediction_df_with_red_for_test(
            "CPNA day",
            label_row=70,
            matrix_n_rows=6,
            matrix_row=0,
            matrix_col=3,
        )
        lsd = self._make_loadcase_score_dict()
        self._run(
            self._make_dfs(),
            lsd,
            self._make_scenario_scores_df(),
            pred_ped_cyc=pred_df,
        )
        self.assertTrue(self._all_zeroed(lsd["Frontal Collisions"]))

    def test_fc_fails_when_cpna_night_not_green(self):
        pred_df = self._make_prediction_df_with_red_for_test(
            "CPNA night",
            label_row=79,
            matrix_n_rows=6,
            matrix_row=0,
            matrix_col=3,
        )
        lsd = self._make_loadcase_score_dict()
        self._run(
            self._make_dfs(),
            lsd,
            self._make_scenario_scores_df(),
            pred_ped_cyc=pred_df,
        )
        self.assertTrue(self._all_zeroed(lsd["Frontal Collisions"]))

    def test_fc_fails_when_ccrs_leq20_not_green(self):
        pred_df = self._make_prediction_df_with_red_for_test(
            "CCRs",
            label_row=0,
            matrix_n_rows=8,
            matrix_row=0,
            matrix_col=1,
        )
        lsd = self._make_loadcase_score_dict()
        self._run(
            self._make_dfs(),
            lsd,
            self._make_scenario_scores_df(),
            pred_car_ptw=pred_df,
        )
        self.assertTrue(self._all_zeroed(lsd["Frontal Collisions"]))

    def test_fc_fail_does_not_affect_ldc_or_lsc(self):
        dfs = self._make_dfs(
            {"FC - Car & PTW verif.": self._make_verification_df("FAIL")}
        )
        lsd = self._make_loadcase_score_dict()
        self._run(dfs, lsd, self._make_scenario_scores_df())
        self.assertFalse(self._all_zeroed(lsd["Lane Departure Collisions"]))
        self.assertFalse(self._all_zeroed(lsd["Low Speed Collisions"]))

    # --- LDC rules ---

    def test_ldc_fails_when_single_veh_gr_fails(self):
        dfs = self._make_dfs(
            {"LDC - Single Veh verif.": self._make_verification_df("FAIL")}
        )
        lsd = self._make_loadcase_score_dict()
        self._run(dfs, lsd, self._make_scenario_scores_df())
        self.assertTrue(self._all_zeroed(lsd["Lane Departure Collisions"]))
        self.assertFalse(self._all_zeroed(lsd["Frontal Collisions"]))
        self.assertFalse(self._all_zeroed(lsd["Low Speed Collisions"]))

    def test_ldc_fails_when_car_ptw_gr_fails(self):
        dfs = self._make_dfs(
            {"LDC - Car & PTW verif.": self._make_verification_df("FAIL")}
        )
        lsd = self._make_loadcase_score_dict()
        self._run(dfs, lsd, self._make_scenario_scores_df())
        self.assertTrue(self._all_zeroed(lsd["Lane Departure Collisions"]))

    def test_ldc_fail_does_not_affect_fc_or_lsc(self):
        dfs = self._make_dfs(
            {"LDC - Single Veh verif.": self._make_verification_df("FAIL")}
        )
        lsd = self._make_loadcase_score_dict()
        self._run(dfs, lsd, self._make_scenario_scores_df())
        self.assertFalse(self._all_zeroed(lsd["Frontal Collisions"]))
        self.assertFalse(self._all_zeroed(lsd["Low Speed Collisions"]))

    # --- LSC rules ---

    def test_lsc_fails_when_car_ptw_gr_fails(self):
        dfs = self._make_dfs(
            {"LSC - Car & PTW verif.": self._make_verification_df("FAIL")}
        )
        lsd = self._make_loadcase_score_dict()
        self._run(dfs, lsd, self._make_scenario_scores_df())
        self.assertTrue(self._all_zeroed(lsd["Low Speed Collisions"]))
        self.assertFalse(self._all_zeroed(lsd["Frontal Collisions"]))
        self.assertFalse(self._all_zeroed(lsd["Lane Departure Collisions"]))

    def test_lsc_fails_when_ped_cyc_gr_fails(self):
        dfs = self._make_dfs(
            {"LSC - Ped & Cyc verif.": self._make_verification_df("FAIL")}
        )
        lsd = self._make_loadcase_score_dict()
        self._run(dfs, lsd, self._make_scenario_scores_df())
        self.assertTrue(self._all_zeroed(lsd["Low Speed Collisions"]))

    def test_lsc_fail_does_not_affect_fc_or_ldc(self):
        dfs = self._make_dfs(
            {"LSC - Car & PTW verif.": self._make_verification_df("FAIL")}
        )
        lsd = self._make_loadcase_score_dict()
        self._run(dfs, lsd, self._make_scenario_scores_df())
        self.assertFalse(self._all_zeroed(lsd["Frontal Collisions"]))
        self.assertFalse(self._all_zeroed(lsd["Lane Departure Collisions"]))

    # --- scenario_scores_df zeroing ---

    def test_scenario_scores_df_fc_rows_zeroed_on_fc_fail(self):
        dfs = self._make_dfs(
            {"FC - Car & PTW verif.": self._make_verification_df("FAIL")}
        )
        ssd = self._make_scenario_scores_df()
        self._run(dfs, self._make_loadcase_score_dict(), ssd)
        fc_indices = [
            idx
            for idx in ssd.index
            if get_element_for_row(ssd, "Stage element", idx) == "Frontal Collisions"
        ]
        self.assertTrue(all(ssd.loc[idx, "Score"] == 0.0 for idx in fc_indices))

    def test_scenario_scores_df_ldc_rows_zeroed_on_ldc_fail(self):
        dfs = self._make_dfs(
            {"LDC - Single Veh verif.": self._make_verification_df("FAIL")}
        )
        ssd = self._make_scenario_scores_df()
        self._run(dfs, self._make_loadcase_score_dict(), ssd)
        ldc_indices = [
            idx
            for idx in ssd.index
            if get_element_for_row(ssd, "Stage element", idx)
            == "Lane Departure Collisions"
        ]
        self.assertTrue(all(ssd.loc[idx, "Score"] == 0.0 for idx in ldc_indices))

    def test_scenario_scores_df_lsc_rows_zeroed_on_lsc_fail(self):
        dfs = self._make_dfs(
            {"LSC - Car & PTW verif.": self._make_verification_df("FAIL")}
        )
        ssd = self._make_scenario_scores_df()
        self._run(dfs, self._make_loadcase_score_dict(), ssd)
        lsc_indices = [
            idx
            for idx in ssd.index
            if get_element_for_row(ssd, "Stage element", idx) == "Low Speed Collisions"
        ]
        self.assertTrue(all(ssd.loc[idx, "Score"] == 0.0 for idx in lsc_indices))


class TestComputeStageScoreMissingVerificationSheet(unittest.TestCase):
    def test_missing_verification_sheet_zeroes_affected_scores(self):
        stage_info = {
            "Stage element": "Frontal Collisions",
            "Stage subelement": "Pedestrian & cyclist",
        }
        loadcase_score_dict = {"Frontal Collisions": {"Ped & Cyc": {}}}
        dfs = {
            "FC - Ped & Cyc pred.": pd.DataFrame(),
            "FC - Ped & Cyc robust. pred.": pd.DataFrame(),
        }

        test_info.compute_stage_score(dfs, stage_info, loadcase_score_dict)

        expected_loadcases = data_model.STAGE_SUBELEMENT_TO_LOADCASES["FC"]["Ped & Cyc"]
        actual_scores = loadcase_score_dict["Frontal Collisions"]["Ped & Cyc"]

        self.assertEqual(set(actual_scores.keys()), set(expected_loadcases))
        for loadcase_name in expected_loadcases:
            loadcase_score = actual_scores[loadcase_name]
            self.assertEqual(loadcase_score.loadcase_name, loadcase_name)
            self.assertEqual(loadcase_score.standard_score, 0.0)
            self.assertEqual(loadcase_score.extended_score, 0.0)
            self.assertEqual(loadcase_score.robustness_layer_score, 0.0)
            self.assertEqual(loadcase_score.total_score, 0.0)


class TestGetTestMatrixTargetApproach(unittest.TestCase):
    """
    Verify that get_test_matrix injects a 'Target approach' attribute (via
    random.choice) for CCCscp and CMCscp scenarios, and does not do so for
    other scenarios.
    """

    # Minimal matrix_indices_dict: 1 cell at [1, 1] inside a small DataFrame.
    _MINIMAL_INDICES = {"start_row": 1, "n_rows": 1, "start_col": 1, "n_cols": 1}

    def _make_minimal_df(self):
        """Two-row DataFrame that places a 'green' color cell at iloc[1, 1]."""
        return pd.DataFrame(
            {
                "speed_col": [None, "30 km/h"],
                "VUT speed": [None, "green"],
            }
        )

    def _get_test_points(self, test_name: str):
        indices = self._MINIMAL_INDICES
        return matrix_processing.get_test_matrix_from_region(
            self._make_minimal_df(),
            indices["start_row"],
            indices["n_rows"],
            indices["start_col"],
            indices["n_cols"],
            test_name=test_name,
            stage_subelement_key=data_model.StageSubelementKey.FC,
        )

    @patch("euroncap_rating_2026.crash_avoidance.matrix_processing.plot_matrix")
    def test_cccscp_adds_target_approach_attribute(self, _mock_plot):
        """CCCscp test points must carry a 'Target approach' attribute."""
        test_points = self._get_test_points("CCCscp")

        self.assertIsNotNone(test_points)
        self.assertEqual(len(test_points), 1)
        tp = test_points[0]
        self.assertIn("Target approach", tp.attributes)
        self.assertIn(tp.attributes["Target approach"], ["Farside", "Nearside"])

    @patch("euroncap_rating_2026.crash_avoidance.matrix_processing.plot_matrix")
    def test_cmcscp_adds_target_approach_attribute(self, _mock_plot):
        """CMCscp test points must carry a 'Target approach' attribute."""
        test_points = self._get_test_points("CMCscp")

        self.assertIsNotNone(test_points)
        self.assertEqual(len(test_points), 1)
        tp = test_points[0]
        self.assertIn("Target approach", tp.attributes)
        self.assertIn(tp.attributes["Target approach"], ["Farside", "Nearside"])

    @patch("euroncap_rating_2026.crash_avoidance.matrix_processing.plot_matrix")
    def test_other_scenario_does_not_add_target_approach(self, _mock_plot):
        """Scenarios other than CCCscp/CMCscp must NOT get a 'Target approach' attribute."""
        test_points = self._get_test_points("CCRs")

        self.assertIsNotNone(test_points)
        self.assertEqual(len(test_points), 1)
        self.assertNotIn("Target approach", test_points[0].attributes)

    @patch("euroncap_rating_2026.crash_avoidance.matrix_processing.plot_matrix")
    @patch("euroncap_rating_2026.crash_avoidance.matrix_processing.random")
    def test_target_approach_value_comes_from_random_choice(
        self, mock_random, _mock_plot
    ):
        """'Target approach' must be the direct return value of random.choice."""
        mock_random.choice.return_value = "Farside"

        test_points = self._get_test_points("CCCscp")

        mock_random.choice.assert_called_once_with(["Farside", "Nearside"])
        self.assertEqual(test_points[0].attributes["Target approach"], "Farside")


class TestPreprocessStageSubelementInputSelection(unittest.TestCase):

    @staticmethod
    def _make_lsc_dfs():
        import numpy as np

        # Minimal "LSC - Ped & Cyc pred." DataFrame with a 2-row CBDA matrix.
        # CBDA: start_col=1, n_cols=3. Marker at idx=0 → start_row=2.
        data = [
            ["CBDA", "Target speed", np.nan, np.nan],
            [np.nan, "10 km/h", "15 km/h", "20 km/h"],
            ["0.50 m", "Green", "Green", "Green"],
            ["1.00 m", "Green", "Green", "Green"],
        ]
        return {"LSC - Ped & Cyc pred.": pd.DataFrame(data)}

    @staticmethod
    def _make_fc_car_ptw_dfs():
        import numpy as np

        # Minimal "FC - Car & PTW pred." with CCCscp only (start_col=3, n_cols=7).
        # Two data rows with VUT speed 30 and 50 km/h, all RED (robustness disabled).
        FC_LOADCASES = [
            "CCRs",
            "CCRm",
            "CCRb",
            "CCFhos",
            "CCFhol",
            "CCFtap",
            "CCCscp",
            "CMRs",
            "CMRb",
            "CMFtap",
            "CMCscp",
        ]
        pred_data = [
            [
                "CCCscp",
                np.nan,
                np.nan,
                "Target speed",
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                "30 km/h",
                "40 km/h",
                "50 km/h",
                "60 km/h",
                "70 km/h",
                "80 km/h",
                "100 km/h",
            ],
            [
                "30 km/h",
                np.nan,
                np.nan,
                "Red",
                "Red",
                "Red",
                "Red",
                "Red",
                "Red",
                "Red",
            ],
            [
                "50 km/h",
                np.nan,
                np.nan,
                "Red",
                "Red",
                "Red",
                "Red",
                "Red",
                "Red",
                "Red",
            ],
        ]
        return {
            "FC - Car & PTW pred.": pd.DataFrame(pred_data),
            "FC - Car & PTW robust. pred.": pd.DataFrame(
                columns=["Robustness layer"] + FC_LOADCASES
            ),
            "Input parameters": pd.DataFrame(
                {
                    "Input parameter": ["Vehicle response"],
                    "Value": ["Retention"],
                }
            ),
        }

    def test_input_selection_round_trip_preserves_count_and_disables_robustness(self):
        input_selected_points = {
            "CCCscp": [
                {
                    "Function": "AEB",
                    "VUT speed": "50 km/h",
                    "Target speed": "40 km/h",
                    "Target approach": "Farside",
                },
                {
                    "Function": "AEB",
                    "VUT speed": "30 km/h",
                    "Target speed": "30 km/h",
                    "Target approach": "Nearside",
                },
            ]
        }

        stage_subelement = test_info.preprocess_stage_subelement(
            self._make_fc_car_ptw_dfs(),
            "Frontal Collisions",
            "Car & PTW",
            input_selected_points=input_selected_points,
        )

        selected_points = stage_subelement.selected_points_dict["CCCscp"]
        self.assertEqual(len(selected_points), len(input_selected_points["CCCscp"]))
        self.assertEqual(stage_subelement.selected_robustness_dict["CCCscp"], "N/A")

        cccscp_rows = stage_subelement.test_points_df[
            stage_subelement.test_points_df["Scenario"] == "CCCscp"
        ].reset_index(drop=True)
        self.assertEqual(len(cccscp_rows), len(input_selected_points["CCCscp"]))
        self.assertIn("Robustness", cccscp_rows.columns)
        self.assertNotIn("Robustness layer", cccscp_rows.columns)
        self.assertNotIn("Verification condition", cccscp_rows.columns)
        self.assertTrue(
            cccscp_rows["Robustness"].fillna("").astype(str).str.strip().eq("").all()
        )

        selected_target_approaches = [
            tp.attributes["Target approach"] for tp in selected_points
        ]
        self.assertEqual(selected_target_approaches, ["Farside", "Nearside"])

        round_trip_points, computed_info_dict = test_info.read_test_points(cccscp_rows)
        self.assertEqual(len(round_trip_points), len(input_selected_points["CCCscp"]))
        self.assertEqual(
            set(computed_info_dict.keys()),
            {f"CCCscp_({tp.row}, {tp.col})" for tp in round_trip_points},
        )
        self.assertTrue(
            all(
                info
                == {
                    "value": 0.0,
                    "value_baseline": None,
                    "verification_condition": None,
                }
                for info in computed_info_dict.values()
            )
        )
        self.assertEqual(
            [tp.attributes["Target approach"] for tp in round_trip_points],
            ["Farside", "Nearside"],
        )
        self.assertTrue(all(tp.robustness_layer is None for tp in round_trip_points))

    @patch(
        "euroncap_rating_2026.crash_avoidance.test_info.get_vehicle_response",
        return_value=data_model.VehicleResponse.RETENTION,
    )
    def test_input_selection_cbda_retention_round_trip_single_row_per_point(
        self, _mock_vehicle_response
    ):
        input_selected_points = {
            "CBDA": [
                {
                    "Gap": "0.50 m",
                    "Target speed": "10 km/h",
                    "Door selected": "Driver",
                    "Function": "Retention",
                },
                {
                    "Gap": "1.00 m",
                    "Target speed": "15 km/h",
                    "Door selected": "Behind passenger",
                    "Function": "Retention",
                },
            ]
        }

        stage_subelement = test_info.preprocess_stage_subelement(
            self._make_lsc_dfs(),
            "Low Speed Collisions",
            "Pedestrian & cyclist",
            input_selected_points=input_selected_points,
        )

        selected_points = stage_subelement.selected_points_dict["CBDA"]
        self.assertEqual(len(selected_points), len(input_selected_points["CBDA"]))

        cbda_rows = stage_subelement.test_points_df[
            stage_subelement.test_points_df["Scenario"] == "CBDA"
        ].reset_index(drop=True)
        # Now one row per selected point (no more duplicate rows)
        self.assertEqual(len(cbda_rows), len(input_selected_points["CBDA"]))
        self.assertTrue((cbda_rows["Expected value"] == "TTC @ t_door_opening").all())
        self.assertCountEqual(
            cbda_rows["Door selected"].tolist(),
            ["Driver", "Behind passenger"],
        )

        # Standard reader handles all CBDA response types
        round_trip_points, computed_info_dict = test_info.read_test_points(cbda_rows)
        self.assertEqual(len(round_trip_points), len(input_selected_points["CBDA"]))
        for test_point in round_trip_points:
            row_key = f"CBDA_({test_point.row}, {test_point.col})"
            self.assertIn("value", computed_info_dict[row_key])
            self.assertEqual(test_point.attributes["Function"], "Retention")
        self.assertCountEqual(
            [tp.attributes["Door selected"] for tp in round_trip_points],
            ["Driver", "Behind passenger"],
        )


class TestRobustnessColumnOverride(unittest.TestCase):
    def test_has_robustness_column_failures_considers_standard_rows(self):
        scenario_df = pd.DataFrame(
            {
                "Range": [
                    matrix_processing.TestRange.STANDARD.value,
                    matrix_processing.TestRange.EXTENDED.value,
                ],
                "Robustness layer": ["Impact location", None],
                "Robustness": ["FAIL", "PASS"],
            }
        )

        self.assertTrue(test_info._has_robustness_column_failures(scenario_df))

    def test_has_robustness_column_failures_ignores_extended_only_failures(self):
        scenario_df = pd.DataFrame(
            {
                "Range": [
                    matrix_processing.TestRange.STANDARD.value,
                    matrix_processing.TestRange.EXTENDED.value,
                ],
                "Robustness layer": ["Impact location", None],
                "Robustness": ["PASS", "FAIL"],
            }
        )

        self.assertFalse(test_info._has_robustness_column_failures(scenario_df))

    def test_has_robustness_column_failures_with_mixed_standard_pass_and_fail(self):
        scenario_df = pd.DataFrame(
            {
                "Range": [
                    matrix_processing.TestRange.STANDARD.value,
                    matrix_processing.TestRange.STANDARD.value,
                ],
                "Robustness layer": ["Impact location", "Impact location"],
                "Robustness": ["PASS", "FAIL"],
            }
        )

        self.assertTrue(test_info._has_robustness_column_failures(scenario_df))

    def test_has_robustness_column_failures_with_all_standard_pass_returns_false(self):
        scenario_df = pd.DataFrame(
            {
                "Range": [
                    matrix_processing.TestRange.STANDARD.value,
                    matrix_processing.TestRange.STANDARD.value,
                ],
                "Robustness layer": ["Impact location", "Impact location"],
                "Robustness": ["PASS", "PASS"],
            }
        )

        self.assertFalse(test_info._has_robustness_column_failures(scenario_df))


class TestComputeLoadcaseScoreRobustnessLayerFix(unittest.TestCase):
    """Tests for Bug 1 fix: compute_loadcase_score uses constant_score (not 0) on failure."""

    def _make_passing_ccftap_test_point(self, robustness_layer_val=None):
        """STANDARD CCFtap avoidance test point with value=0.0 → GREEN, assessment NOT_RED."""
        tp = matrix_processing.TestPoint(
            row=1,
            col=1,
            color=common.PredictionColor.GREEN,
            test_range=matrix_processing.TestRange.STANDARD,
            robustness_layer=robustness_layer_val,
        )
        return matrix_processing.ComputedTestPoint(
            test_name="CCFtap",
            test_point=tp,
            value=0.0,
            assessment_criteria=robustness_layer.AssessmentCriteria.NOT_RED,
        )

    def _make_ccftap_robustness_score(self):
        """CCFtap RobustnessLayerScore: 7 claimed, 8 applicable → tested=0.0625, constant=0.375."""
        return robustness_layer.RobustnessLayerScore(
            tested_score=0.0625,
            constant_score=0.375,
            n_applicable=8.0,
            claimed_layers={
                robustness_layer.RobustnessLayer.SPEED,
                robustness_layer.RobustnessLayer.DRIVER_INPUT_PRE_CRASH,
            },
        )

    def test_has_robustness_failure_sets_score_to_constant_score_not_zero(self):
        """Bug 1: has_robustness_failure=True must yield constant_score (0.375), not 0."""
        test_points = [self._make_passing_ccftap_test_point() for _ in range(3)]
        rob_score = self._make_ccftap_robustness_score()

        result = test_info.compute_loadcase_score(
            score_test_name="CCFtap",
            subtest_name="CCFtap",
            computed_test_points=test_points,
            standard_oem_prediction_method=matrix_processing.PredictionSource.VTA,
            extended_oem_prediction_method=matrix_processing.PredictionSource.VTA,
            predicted_standard_score=4.0,
            predicted_extended_score=0.5,
            robustness_layer_score=rob_score,
            has_robustness_failure=True,
        )

        # constant_score = 0.5 * 6/8 = 0.375, NOT zero
        self.assertAlmostEqual(result.robustness_layer_score, 0.375, places=6)
        self.assertTrue(result.robustness_layer_failed)

    def test_no_failure_preserves_full_robustness_score(self):
        """When no layer fails, robustness score equals tested_score + constant_score."""
        test_points = [self._make_passing_ccftap_test_point() for _ in range(3)]
        rob_score = self._make_ccftap_robustness_score()

        result = test_info.compute_loadcase_score(
            score_test_name="CCFtap",
            subtest_name="CCFtap",
            computed_test_points=test_points,
            standard_oem_prediction_method=matrix_processing.PredictionSource.VTA,
            extended_oem_prediction_method=matrix_processing.PredictionSource.VTA,
            predicted_standard_score=4.0,
            predicted_extended_score=0.5,
            robustness_layer_score=rob_score,
            has_robustness_failure=False,
        )

        # tested_score + constant_score = 0.0625 + 0.375 = 0.4375
        self.assertAlmostEqual(result.robustness_layer_score, 0.4375, places=6)
        self.assertFalse(result.robustness_layer_failed)

    def test_robustness_layer_failed_metadata_populated(self):
        """LoadcaseScore carries n_applicable and claimed_layers from RobustnessLayerScore."""
        test_points = [self._make_passing_ccftap_test_point() for _ in range(3)]
        rob_score = self._make_ccftap_robustness_score()

        result = test_info.compute_loadcase_score(
            score_test_name="CCFtap",
            subtest_name="CCFtap",
            computed_test_points=test_points,
            standard_oem_prediction_method=matrix_processing.PredictionSource.VTA,
            extended_oem_prediction_method=matrix_processing.PredictionSource.VTA,
            predicted_standard_score=4.0,
            predicted_extended_score=0.5,
            robustness_layer_score=rob_score,
            has_robustness_failure=True,
        )

        self.assertAlmostEqual(result.n_applicable, 8.0)
        self.assertIn(robustness_layer.RobustnessLayer.SPEED, result.claimed_layers)
        self.assertAlmostEqual(result.robustness_score_per_layer, 0.0625, places=6)

    def test_tested_robustness_layer_extracted_from_standard_test_points(self):
        """tested_robustness_layer is read from the STANDARD test point's robustness_layer field."""
        speed = robustness_layer.RobustnessLayer.SPEED
        test_points = [
            self._make_passing_ccftap_test_point(robustness_layer_val=speed)
            for _ in range(3)
        ]
        rob_score = self._make_ccftap_robustness_score()

        result = test_info.compute_loadcase_score(
            score_test_name="CCFtap",
            subtest_name="CCFtap",
            computed_test_points=test_points,
            standard_oem_prediction_method=matrix_processing.PredictionSource.VTA,
            extended_oem_prediction_method=matrix_processing.PredictionSource.VTA,
            predicted_standard_score=4.0,
            predicted_extended_score=0.5,
            robustness_layer_score=rob_score,
            has_robustness_failure=True,
        )

        self.assertEqual(result.tested_robustness_layer, speed)

    def test_no_test_points_sets_tested_robustness_layer_to_none(self):
        """When no STANDARD test points exist, tested_robustness_layer is None."""
        rob_score = robustness_layer.RobustnessLayerScore(
            tested_score=0.0, constant_score=0.0, n_applicable=0.0, claimed_layers=set()
        )

        result = test_info.compute_loadcase_score(
            score_test_name="CCFtap",
            subtest_name="CCFtap",
            computed_test_points=[],
            standard_oem_prediction_method=matrix_processing.PredictionSource.VTA,
            extended_oem_prediction_method=matrix_processing.PredictionSource.VTA,
            predicted_standard_score=0.0,
            predicted_extended_score=0.0,
            robustness_layer_score=rob_score,
            has_robustness_failure=False,
        )

        self.assertIsNone(result.tested_robustness_layer)


class TestApplyCrossScenarioRobustnessRule(unittest.TestCase):
    """Tests for Bug 2 fix: cross-scenario collision-partner propagation (section 4.2.2)."""

    STAGE_ELEMENT = "Car Frontal"
    STAGE_SUBELEMENT = "Car & PTW"

    def _make_score(
        self,
        loadcase_name,
        robustness_layer_score=0.5,
        tested_robustness_layer=None,
        robustness_layer_failed=False,
        n_applicable=8.0,
        claimed_layers=None,
        robustness_score_per_layer=0.0625,
    ):
        return test_info.LoadcaseScore(
            loadcase_name=loadcase_name,
            standard_score=4.0,
            extended_score=0.5,
            robustness_layer_score=robustness_layer_score,
            total_score=4.0 + 0.5 + robustness_layer_score,
            tested_robustness_layer=tested_robustness_layer,
            robustness_layer_failed=robustness_layer_failed,
            n_applicable=n_applicable,
            claimed_layers=claimed_layers if claimed_layers is not None else set(),
            robustness_score_per_layer=robustness_score_per_layer,
        )

    def _make_dict(self, scores_by_name):
        return {self.STAGE_ELEMENT: {self.STAGE_SUBELEMENT: dict(scores_by_name)}}

    def test_single_failure_no_propagation(self):
        """Speed fails in 1 car scenario: no cross-scenario reduction applied."""
        speed = robustness_layer.RobustnessLayer.SPEED
        ccrs = self._make_score(
            "CCRs",
            robustness_layer_score=0.125,
            tested_robustness_layer=speed,
            robustness_layer_failed=True,
            claimed_layers={speed},
        )
        ccftap = self._make_score(
            "CCFtap",
            robustness_layer_score=0.4375,
            claimed_layers={speed},
        )
        loadcase_dict = self._make_dict({"CCRs": ccrs, "CCFtap": ccftap})
        test_info.apply_cross_scenario_robustness_rule(loadcase_dict)

        self.assertAlmostEqual(ccftap.robustness_layer_score, 0.4375)
        self.assertAlmostEqual(ccftap.total_score, 4.0 + 0.5 + 0.4375)

    def test_two_failures_same_partner_propagates_to_third_car_scenario(self):
        """Speed fails in CCRs and CCRm: CCFtap robustness score reduced by robustness_score_per_layer."""
        speed = robustness_layer.RobustnessLayer.SPEED
        ccrs = self._make_score(
            "CCRs",
            robustness_layer_score=0.125,
            tested_robustness_layer=speed,
            robustness_layer_failed=True,
            claimed_layers={speed},
        )
        ccrm = self._make_score(
            "CCRm",
            robustness_layer_score=0.25,
            tested_robustness_layer=speed,
            robustness_layer_failed=True,
            claimed_layers={speed},
        )
        ccftap = self._make_score(
            "CCFtap",
            robustness_layer_score=0.4375,
            robustness_layer_failed=False,
            n_applicable=8.0,
            claimed_layers={speed},
            robustness_score_per_layer=0.0625,
        )
        loadcase_dict = self._make_dict({"CCRs": ccrs, "CCRm": ccrm, "CCFtap": ccftap})
        test_info.apply_cross_scenario_robustness_rule(loadcase_dict)

        self.assertAlmostEqual(ccftap.robustness_layer_score, 0.4375 - 0.0625, places=6)
        self.assertAlmostEqual(
            ccftap.total_score, 4.0 + 0.5 + 0.4375 - 0.0625, places=6
        )

    def test_layer_not_claimed_in_target_no_reduction(self):
        """Speed fails in 2 car scenarios: CCFtap that doesn't claim Speed is unaffected."""
        speed = robustness_layer.RobustnessLayer.SPEED
        driver_input = robustness_layer.RobustnessLayer.DRIVER_INPUT_PRE_CRASH
        ccrs = self._make_score(
            "CCRs",
            robustness_layer_score=0.125,
            tested_robustness_layer=speed,
            robustness_layer_failed=True,
            claimed_layers={speed},
        )
        ccrm = self._make_score(
            "CCRm",
            robustness_layer_score=0.25,
            tested_robustness_layer=speed,
            robustness_layer_failed=True,
            claimed_layers={speed},
        )
        ccftap = self._make_score(
            "CCFtap",
            robustness_layer_score=0.4375,
            claimed_layers={driver_input},  # Speed not claimed
        )
        loadcase_dict = self._make_dict({"CCRs": ccrs, "CCRm": ccrm, "CCFtap": ccftap})
        test_info.apply_cross_scenario_robustness_rule(loadcase_dict)

        self.assertAlmostEqual(ccftap.robustness_layer_score, 0.4375)

    def test_two_failures_different_stage_subelements_no_cross_propagation(self):
        """Speed fails once per stage subelement: scoping prevents cross-subelement propagation."""
        speed = robustness_layer.RobustnessLayer.SPEED
        ccrs = self._make_score(
            "CCRs",
            robustness_layer_score=0.125,
            tested_robustness_layer=speed,
            robustness_layer_failed=True,
            claimed_layers={speed},
        )
        ccftap = self._make_score(
            "CCFtap",
            robustness_layer_score=0.4375,
            claimed_layers={speed},
        )
        ccrm = self._make_score(
            "CCRm",
            robustness_layer_score=0.25,
            tested_robustness_layer=speed,
            robustness_layer_failed=True,
            claimed_layers={speed},
        )
        loadcase_dict = {
            self.STAGE_ELEMENT: {
                "Subelement A": {"CCRs": ccrs, "CCFtap": ccftap},
                "Subelement B": {"CCRm": ccrm},
            }
        }
        test_info.apply_cross_scenario_robustness_rule(loadcase_dict)

        # Each subelement has only 1 failure → no propagation in either
        self.assertAlmostEqual(ccftap.robustness_layer_score, 0.4375)

    def test_already_failed_target_not_double_reduced(self):
        """Speed already failed in CCFtap's own test: cross-scenario rule skips it."""
        speed = robustness_layer.RobustnessLayer.SPEED
        ccrs = self._make_score(
            "CCRs",
            robustness_layer_score=0.125,
            tested_robustness_layer=speed,
            robustness_layer_failed=True,
            claimed_layers={speed},
        )
        ccrm = self._make_score(
            "CCRm",
            robustness_layer_score=0.25,
            tested_robustness_layer=speed,
            robustness_layer_failed=True,
            claimed_layers={speed},
        )
        # CCFtap tested Speed and it already failed → score at constant_score (0.375)
        ccftap = self._make_score(
            "CCFtap",
            robustness_layer_score=0.375,
            tested_robustness_layer=speed,
            robustness_layer_failed=True,
            n_applicable=8.0,
            claimed_layers={speed},
            robustness_score_per_layer=0.0625,
        )
        loadcase_dict = self._make_dict({"CCRs": ccrs, "CCRm": ccrm, "CCFtap": ccftap})
        test_info.apply_cross_scenario_robustness_rule(loadcase_dict)

        # CCFtap's own Speed layer already failed → no further reduction
        self.assertAlmostEqual(ccftap.robustness_layer_score, 0.375)

    def test_different_collision_partner_not_affected(self):
        """Speed fails in 2 car scenarios: PTW scenario claiming Speed is unaffected."""
        speed = robustness_layer.RobustnessLayer.SPEED
        ccrs = self._make_score(
            "CCRs",
            robustness_layer_score=0.125,
            tested_robustness_layer=speed,
            robustness_layer_failed=True,
            claimed_layers={speed},
        )
        ccrm = self._make_score(
            "CCRm",
            robustness_layer_score=0.25,
            tested_robustness_layer=speed,
            robustness_layer_failed=True,
            claimed_layers={speed},
        )
        # CMRs is PTW collision partner
        cmrs = self._make_score(
            "CMRs",
            robustness_layer_score=0.3,
            claimed_layers={speed},
        )
        loadcase_dict = self._make_dict({"CCRs": ccrs, "CCRm": ccrm, "CMRs": cmrs})
        test_info.apply_cross_scenario_robustness_rule(loadcase_dict)

        # CMRs is PTW → different collision partner from car failures → no reduction
        self.assertAlmostEqual(cmrs.robustness_layer_score, 0.3)

    def test_multiple_layer_failures_cumulative_reduction(self):
        """Two layers each fail ≥2 times: target scenario gets two independent reductions."""
        speed = robustness_layer.RobustnessLayer.SPEED
        driver_input = robustness_layer.RobustnessLayer.DRIVER_INPUT_PRE_CRASH
        ccrs = self._make_score(
            "CCRs",
            robustness_layer_score=0.10,
            tested_robustness_layer=speed,
            robustness_layer_failed=True,
            claimed_layers={speed},
        )
        ccrm = self._make_score(
            "CCRm",
            robustness_layer_score=0.20,
            tested_robustness_layer=speed,
            robustness_layer_failed=True,
            claimed_layers={speed},
        )
        ccrb = self._make_score(
            "CCRb",
            robustness_layer_score=0.10,
            tested_robustness_layer=driver_input,
            robustness_layer_failed=True,
            claimed_layers={driver_input},
        )
        ccfhos = self._make_score(
            "CCFhos",
            robustness_layer_score=0.15,
            tested_robustness_layer=driver_input,
            robustness_layer_failed=True,
            claimed_layers={driver_input},
        )
        # CCFtap claims both layers with neither failed in its own tests
        ccftap = self._make_score(
            "CCFtap",
            robustness_layer_score=0.4375,
            robustness_layer_failed=False,
            n_applicable=8.0,
            claimed_layers={speed, driver_input},
            robustness_score_per_layer=0.0625,
        )
        loadcase_dict = self._make_dict(
            {
                "CCRs": ccrs,
                "CCRm": ccrm,
                "CCRb": ccrb,
                "CCFhos": ccfhos,
                "CCFtap": ccftap,
            }
        )
        test_info.apply_cross_scenario_robustness_rule(loadcase_dict)

        # Speed reduces by 0.0625, Driver Input reduces by 0.0625 → total 0.125 reduction
        expected = 0.4375 - 0.0625 - 0.0625
        self.assertAlmostEqual(ccftap.robustness_layer_score, expected, places=6)
        self.assertAlmostEqual(ccftap.total_score, 4.0 + 0.5 + expected, places=6)

    def test_zero_applicable_layers_skipped(self):
        """Scenarios with n_applicable=0 are skipped even when collision partner matches."""
        speed = robustness_layer.RobustnessLayer.SPEED
        ccrs = self._make_score(
            "CCRs",
            robustness_layer_score=0.0,
            tested_robustness_layer=speed,
            robustness_layer_failed=True,
            n_applicable=0.0,
            claimed_layers={speed},
        )
        ccrm = self._make_score(
            "CCRm",
            robustness_layer_score=0.0,
            tested_robustness_layer=speed,
            robustness_layer_failed=True,
            n_applicable=0.0,
            claimed_layers={speed},
        )
        ccftap = self._make_score(
            "CCFtap",
            robustness_layer_score=0.4375,
            n_applicable=0.0,
            claimed_layers={speed},
        )
        loadcase_dict = self._make_dict({"CCRs": ccrs, "CCRm": ccrm, "CCFtap": ccftap})
        test_info.apply_cross_scenario_robustness_rule(loadcase_dict)

        # n_applicable=0 → guard skips reduction
        self.assertAlmostEqual(ccftap.robustness_layer_score, 0.4375)

    def test_same_subelement_different_stage_elements_no_cross_propagation(self):
        """One failure in Frontal 'Car & PTW' and one in Lane Departure 'Car & PTW':
        they must NOT be aggregated — different stage elements, so count stays at 1 each.
        """
        speed = robustness_layer.RobustnessLayer.SPEED
        frontal_element = "Frontal Collisions"
        ldc_element = "Lane Departure Collisions"
        subelement = "Car & PTW"

        # One failing scenario under Frontal
        ccrs_frontal = test_info.LoadcaseScore(
            loadcase_name="CCRs",
            standard_score=4.0,
            extended_score=0.5,
            robustness_layer_score=0.125,
            total_score=4.625,
            tested_robustness_layer=speed,
            robustness_layer_failed=True,
            n_applicable=8.0,
            claimed_layers={speed},
            robustness_score_per_layer=0.0625,
        )
        ccftap_frontal = test_info.LoadcaseScore(
            loadcase_name="CCFtap",
            standard_score=4.0,
            extended_score=0.5,
            robustness_layer_score=0.4375,
            total_score=4.9375,
            tested_robustness_layer=None,
            robustness_layer_failed=False,
            n_applicable=8.0,
            claimed_layers={speed},
            robustness_score_per_layer=0.0625,
        )
        # One failing scenario under Lane Departure (CMRs maps to PTW partner,
        # so use CCRs which maps to "car" — reuse loadcase name under a different stage element)
        ccrs_ldc = test_info.LoadcaseScore(
            loadcase_name="CCRs",
            standard_score=4.0,
            extended_score=0.5,
            robustness_layer_score=0.125,
            total_score=4.625,
            tested_robustness_layer=speed,
            robustness_layer_failed=True,
            n_applicable=8.0,
            claimed_layers={speed},
            robustness_score_per_layer=0.0625,
        )
        ccftap_ldc = test_info.LoadcaseScore(
            loadcase_name="CCFtap",
            standard_score=4.0,
            extended_score=0.5,
            robustness_layer_score=0.4375,
            total_score=4.9375,
            tested_robustness_layer=None,
            robustness_layer_failed=False,
            n_applicable=8.0,
            claimed_layers={speed},
            robustness_score_per_layer=0.0625,
        )
        loadcase_dict = {
            frontal_element: {
                subelement: {"CCRs": ccrs_frontal, "CCFtap": ccftap_frontal}
            },
            ldc_element: {subelement: {"CCRs": ccrs_ldc, "CCFtap": ccftap_ldc}},
        }
        test_info.apply_cross_scenario_robustness_rule(loadcase_dict)

        # No propagation: each stage element has only 1 failure, count < 2
        self.assertAlmostEqual(ccftap_frontal.robustness_layer_score, 0.4375)
        self.assertAlmostEqual(ccftap_ldc.robustness_layer_score, 0.4375)

    def test_two_failures_same_stage_element_propagates_only_within_that_element(self):
        """Two failures in Frontal 'Car & PTW' trigger reduction there,
        but a passing scenario in Lane Departure 'Car & PTW' is untouched."""
        speed = robustness_layer.RobustnessLayer.SPEED
        frontal_element = "Frontal Collisions"
        ldc_element = "Lane Departure Collisions"
        subelement = "Car & PTW"

        ccrs = test_info.LoadcaseScore(
            loadcase_name="CCRs",
            standard_score=4.0,
            extended_score=0.5,
            robustness_layer_score=0.125,
            total_score=4.625,
            tested_robustness_layer=speed,
            robustness_layer_failed=True,
            n_applicable=8.0,
            claimed_layers={speed},
            robustness_score_per_layer=0.0625,
        )
        ccrm = test_info.LoadcaseScore(
            loadcase_name="CCRm",
            standard_score=4.0,
            extended_score=0.5,
            robustness_layer_score=0.25,
            total_score=4.75,
            tested_robustness_layer=speed,
            robustness_layer_failed=True,
            n_applicable=8.0,
            claimed_layers={speed},
            robustness_score_per_layer=0.0625,
        )
        ccftap_frontal = test_info.LoadcaseScore(
            loadcase_name="CCFtap",
            standard_score=4.0,
            extended_score=0.5,
            robustness_layer_score=0.4375,
            total_score=4.9375,
            tested_robustness_layer=None,
            robustness_layer_failed=False,
            n_applicable=8.0,
            claimed_layers={speed},
            robustness_score_per_layer=0.0625,
        )
        # Passing scenario in a different stage element with the same subelement name
        ccftap_ldc = test_info.LoadcaseScore(
            loadcase_name="CCFtap",
            standard_score=4.0,
            extended_score=0.5,
            robustness_layer_score=0.4375,
            total_score=4.9375,
            tested_robustness_layer=None,
            robustness_layer_failed=False,
            n_applicable=8.0,
            claimed_layers={speed},
            robustness_score_per_layer=0.0625,
        )
        loadcase_dict = {
            frontal_element: {
                subelement: {"CCRs": ccrs, "CCRm": ccrm, "CCFtap": ccftap_frontal}
            },
            ldc_element: {subelement: {"CCFtap": ccftap_ldc}},
        }
        test_info.apply_cross_scenario_robustness_rule(loadcase_dict)

        # Frontal CCFtap is reduced (2 failures in same stage_element + subelement)
        self.assertAlmostEqual(
            ccftap_frontal.robustness_layer_score, 0.4375 - 0.0625, places=6
        )
        self.assertAlmostEqual(ccftap_frontal.total_score, 4.9375 - 0.0625, places=6)
        # LDC CCFtap is NOT reduced (different stage element)
        self.assertAlmostEqual(ccftap_ldc.robustness_layer_score, 0.4375)
        self.assertAlmostEqual(ccftap_ldc.total_score, 4.9375)


if __name__ == "__main__":
    unittest.main()
