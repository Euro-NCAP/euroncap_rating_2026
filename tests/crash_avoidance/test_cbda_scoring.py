# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch

from euroncap_rating_2026 import common
from euroncap_rating_2026.crash_avoidance import (
    data_model,
    matrix_processing,
    test_info,
)


def _make_test_point(color):
    return matrix_processing.TestPoint(
        row=0,
        col=0,
        color=color,
        test_range=matrix_processing.TestRange.STANDARD,
    )


def _make_computed_point(color, value):
    return matrix_processing.ComputedTestPoint(
        test_name="CBDA",
        test_point=_make_test_point(color),
        value=value,
        assessment_criteria=None,
        prediction_result=None,
    )


def _make_input_params_df(vehicle_response):
    return pd.DataFrame(
        {
            "Input parameter": ["Vehicle response"],
            "Value": [vehicle_response.value],
        }
    )


def _make_cbda_df(predicted_color, value, expected_value_col="TTC @ t_door_opening"):
    return pd.DataFrame(
        {
            "Scenario": ["CBDA"],
            "Test point": ["(0, 0)"],
            "Range": [matrix_processing.TestRange.STANDARD.value],
            "OEM Prediction": [predicted_color.value],
            "Expected value": [expected_value_col],
            "Value": [value],
        }
    )


def _make_standard_test_points(n):
    return [
        matrix_processing.TestPoint(
            row=i,
            col=0,
            color=common.PredictionColor.GREEN,
            test_range=matrix_processing.TestRange.STANDARD,
        )
        for i in range(n)
    ]


class TestCbdaRetentionColor(unittest.TestCase):
    """Tests for Retention PASS/FAIL: FAIL if -0.4 < value < 1.7, PASS otherwise."""

    def _color(self, predicted_color, value):
        return _make_computed_point(predicted_color, value).get_lsc_computed_color(
            data_model.VehicleResponse.RETENTION
        )

    def test_pass_negative_value_green_prediction(self):
        self.assertEqual(
            self._color(common.PredictionColor.GREEN, -2.0),
            common.PredictionColor.GREEN,
        )

    def test_pass_negative_value_yellow_prediction(self):
        self.assertEqual(
            self._color(common.PredictionColor.YELLOW, -2.0),
            common.PredictionColor.YELLOW,
        )

    def test_pass_high_positive_value_green_prediction(self):
        self.assertEqual(
            self._color(common.PredictionColor.GREEN, 2.0), common.PredictionColor.GREEN
        )

    def test_pass_high_positive_value_yellow_prediction(self):
        self.assertEqual(
            self._color(common.PredictionColor.YELLOW, 2.0),
            common.PredictionColor.YELLOW,
        )

    def test_fail_zero_value(self):
        self.assertEqual(
            self._color(common.PredictionColor.GREEN, 0.0), common.PredictionColor.RED
        )

    def test_fail_mid_range_value(self):
        self.assertEqual(
            self._color(common.PredictionColor.YELLOW, 1.0), common.PredictionColor.RED
        )

    def test_boundary_pass_at_minus_0_4(self):
        # -0.4 is NOT > -0.4, so NOT in fail range → PASS
        self.assertEqual(
            self._color(common.PredictionColor.GREEN, -0.4),
            common.PredictionColor.GREEN,
        )

    def test_boundary_pass_at_1_7(self):
        # 1.7 is NOT < 1.7, so NOT in fail range → PASS
        self.assertEqual(
            self._color(common.PredictionColor.GREEN, 1.7), common.PredictionColor.GREEN
        )

    def test_boundary_fail_just_above_minus_0_4(self):
        # -0.39 > -0.4 and -0.39 < 1.7 → FAIL
        self.assertEqual(
            self._color(common.PredictionColor.GREEN, -0.39), common.PredictionColor.RED
        )

    def test_boundary_fail_just_below_1_7(self):
        # 1.69 > -0.4 and 1.69 < 1.7 → FAIL
        self.assertEqual(
            self._color(common.PredictionColor.GREEN, 1.69), common.PredictionColor.RED
        )

    def test_pass_with_invalid_prediction_color_returns_red(self):
        # ORANGE is not a valid retention prediction; even on PASS it must return RED
        self.assertEqual(
            self._color(common.PredictionColor.ORANGE, 2.0), common.PredictionColor.RED
        )

    def test_nan_value_returns_red(self):
        # NaN TTC must be treated as FAIL → RED (chained comparison with NaN would silently PASS)
        self.assertEqual(
            self._color(common.PredictionColor.GREEN, float("nan")),
            common.PredictionColor.RED,
        )


class TestCbdaWarningColor(unittest.TestCase):
    """Tests for Warning PASS/FAIL: FAIL if value < 1.7."""

    def _color(self, predicted_color, value):
        return _make_computed_point(predicted_color, value).get_lsc_computed_color(
            data_model.VehicleResponse.WARNING
        )

    def test_pass_orange_prediction(self):
        self.assertEqual(
            self._color(common.PredictionColor.ORANGE, 2.0),
            common.PredictionColor.ORANGE,
        )

    def test_pass_yellow_prediction(self):
        self.assertEqual(
            self._color(common.PredictionColor.YELLOW, 2.0),
            common.PredictionColor.YELLOW,
        )

    def test_fail_below_threshold(self):
        self.assertEqual(
            self._color(common.PredictionColor.ORANGE, 1.5), common.PredictionColor.RED
        )

    def test_fail_exactly_below_threshold(self):
        self.assertEqual(
            self._color(common.PredictionColor.YELLOW, 1.69), common.PredictionColor.RED
        )

    def test_pass_at_exact_threshold(self):
        self.assertEqual(
            self._color(common.PredictionColor.ORANGE, 1.7),
            common.PredictionColor.ORANGE,
        )


class TestCbdaInformationColor(unittest.TestCase):
    """Tests for Information PASS/FAIL: FAIL if value < 2.3."""

    def _color(self, predicted_color, value):
        return _make_computed_point(predicted_color, value).get_lsc_computed_color(
            data_model.VehicleResponse.INFORMATION
        )

    def test_pass_brown_prediction(self):
        self.assertEqual(
            self._color(common.PredictionColor.BROWN, 3.0), common.PredictionColor.BROWN
        )

    def test_fail_below_threshold(self):
        self.assertEqual(
            self._color(common.PredictionColor.BROWN, 2.0), common.PredictionColor.RED
        )

    def test_pass_at_exact_threshold(self):
        self.assertEqual(
            self._color(common.PredictionColor.BROWN, 2.3), common.PredictionColor.BROWN
        )

    def test_fail_just_below_threshold(self):
        self.assertEqual(
            self._color(common.PredictionColor.BROWN, 2.29), common.PredictionColor.RED
        )


class TestCbdaScoring(unittest.TestCase):
    """Integration tests for compute_lsc_loadcase_score with CBDA scenarios."""

    def _score(self, vehicle_response, predicted_color, value, expected_value_col=None):
        if expected_value_col is None:
            col_map = {
                data_model.VehicleResponse.RETENTION: "TTC @ t_door_opening",
                data_model.VehicleResponse.WARNING: "TTC @ t_warning",
                data_model.VehicleResponse.INFORMATION: "TTC @ t_information",
            }
            expected_value_col = col_map[vehicle_response]
        return test_info.compute_lsc_loadcase_score(
            "CBDA",
            {"CBDA": _make_cbda_df(predicted_color, value, expected_value_col)},
            _make_input_params_df(vehicle_response),
            _make_standard_test_points(1),
            matrix_total_cells=1,
        )

    # Information
    def test_information_pass_brown_scores_correctly(self):
        score, colors = self._score(
            data_model.VehicleResponse.INFORMATION, common.PredictionColor.BROWN, 2.5
        )
        self.assertEqual(score.standard_score, 0.5)
        self.assertEqual(colors["CBDA_(0, 0)"], common.PredictionColor.BROWN)

    def test_information_fail_returns_zero(self):
        score, colors = self._score(
            data_model.VehicleResponse.INFORMATION, common.PredictionColor.BROWN, 1.0
        )
        self.assertEqual(score.standard_score, 0.0)
        self.assertEqual(colors["CBDA_(0, 0)"], common.PredictionColor.RED)

    # Warning
    def test_warning_pass_orange_scores_correctly(self):
        score, colors = self._score(
            data_model.VehicleResponse.WARNING, common.PredictionColor.ORANGE, 2.0
        )
        self.assertEqual(score.standard_score, 1.0)
        self.assertEqual(colors["CBDA_(0, 0)"], common.PredictionColor.ORANGE)

    def test_warning_pass_yellow_scores_correctly(self):
        score, colors = self._score(
            data_model.VehicleResponse.WARNING, common.PredictionColor.YELLOW, 2.0
        )
        self.assertEqual(score.standard_score, 1.5)
        self.assertEqual(colors["CBDA_(0, 0)"], common.PredictionColor.YELLOW)

    def test_warning_fail_returns_zero(self):
        score, colors = self._score(
            data_model.VehicleResponse.WARNING, common.PredictionColor.ORANGE, 1.5
        )
        self.assertEqual(score.standard_score, 0.0)
        self.assertEqual(colors["CBDA_(0, 0)"], common.PredictionColor.RED)

    # Retention
    def test_retention_pass_green_scores_correctly(self):
        score, colors = self._score(
            data_model.VehicleResponse.RETENTION, common.PredictionColor.GREEN, 2.0
        )
        self.assertEqual(score.standard_score, 2.0)
        self.assertEqual(colors["CBDA_(0, 0)"], common.PredictionColor.GREEN)

    def test_retention_pass_yellow_scores_correctly(self):
        score, colors = self._score(
            data_model.VehicleResponse.RETENTION, common.PredictionColor.YELLOW, 2.0
        )
        self.assertEqual(score.standard_score, 1.5)
        self.assertEqual(colors["CBDA_(0, 0)"], common.PredictionColor.YELLOW)

    def test_retention_fail_returns_zero(self):
        score, colors = self._score(
            data_model.VehicleResponse.RETENTION, common.PredictionColor.GREEN, 0.0
        )
        self.assertEqual(score.standard_score, 0.0)
        self.assertEqual(colors["CBDA_(0, 0)"], common.PredictionColor.RED)

    def test_retention_nan_value_uses_oem_prediction_color(self):
        # NaN in the df → "not specified" → use OEM prediction color weight directly
        score, colors = self._score(
            data_model.VehicleResponse.RETENTION,
            common.PredictionColor.GREEN,
            float("nan"),
        )
        self.assertEqual(score.standard_score, 2.0)
        self.assertEqual(colors["CBDA_(0, 0)"], common.PredictionColor.GREEN)

    def test_na_vehicle_response_scores_zero(self):
        # An explicit "N/A" Vehicle response means no dooring system is
        # fitted: CBDA scores a real 0.0 with no computed colors.
        input_params_df = pd.DataFrame(
            {"Input parameter": ["Vehicle response"], "Value": ["N/A"]}
        )
        score, colors = test_info.compute_lsc_loadcase_score(
            "CBDA",
            {"CBDA": _make_cbda_df(common.PredictionColor.GREEN, 2.0)},
            input_params_df,
            _make_standard_test_points(1),
            matrix_total_cells=1,
        )
        self.assertEqual(score.standard_score, 0.0)
        self.assertEqual(score.total_score, 0.0)
        self.assertEqual(colors, {})

    def test_na_vehicle_response_does_not_zero_other_lsc_loadcases(self):
        # The N/A guard is CBDA-only: other LSC loadcases ignore the
        # Vehicle response parameter entirely.
        input_params_df = pd.DataFrame(
            {"Input parameter": ["Vehicle response"], "Value": ["N/A"]}
        )
        df = _make_cbda_df(common.PredictionColor.GREEN, float("nan"))
        df["Scenario"] = "CPMRCs"
        score, _ = test_info.compute_lsc_loadcase_score(
            "CPMRCs",
            {"CPMRCs": df},
            input_params_df,
            _make_standard_test_points(1),
            matrix_total_cells=1,
        )
        self.assertGreater(score.standard_score, 0.0)

    def test_blank_vehicle_response_keeps_current_behavior(self):
        # Blank (NaN) Vehicle response is NOT the N/A case: CBDA still
        # scores from the OEM prediction colors.
        input_params_df = pd.DataFrame(
            {"Input parameter": ["Vehicle response"], "Value": [np.nan]}
        )
        score, _ = test_info.compute_lsc_loadcase_score(
            "CBDA",
            {"CBDA": _make_cbda_df(common.PredictionColor.GREEN, float("nan"))},
            input_params_df,
            _make_standard_test_points(1),
            matrix_total_cells=1,
        )
        self.assertEqual(score.standard_score, 2.0)

    def test_partial_matrix_uses_full_denominator(self):
        # 9 selected GREEN cells in a 4×3=12 cell matrix.
        # Expected: 9 × (1/12) × 2.0 = 1.5, not 2.0.
        nine_green_points = [
            matrix_processing.TestPoint(
                row=i,
                col=j,
                color=common.PredictionColor.GREEN,
                test_range=matrix_processing.TestRange.STANDARD,
            )
            for i in range(3)
            for j in range(3)
        ]
        test_points_df = pd.DataFrame(
            {
                "Scenario": ["CBDA"] * 9,
                "Test point": [f"({i}, {j})" for i in range(3) for j in range(3)],
                "Range": [matrix_processing.TestRange.STANDARD.value] * 9,
                "OEM Prediction": [common.PredictionColor.GREEN.value] * 9,
                "Expected value": [float("nan")] * 9,
                "Value": [float("nan")] * 9,
            }
        )
        score, _ = test_info.compute_lsc_loadcase_score(
            "CBDA",
            {"CBDA": test_points_df},
            _make_input_params_df(data_model.VehicleResponse.RETENTION),
            nine_green_points,
            matrix_total_cells=12,
        )
        self.assertAlmostEqual(score.standard_score, 1.5)


class TestLscFullMatrixDenominator(unittest.TestCase):
    """
    Verify that compute_lsc_loadcase_score uses n_rows×n_cols as the denominator,
    not the count of selected test points.

    For every LSC scenario we build:
      - a full GREEN matrix (selected == total) → expect full marks
      - a partial GREEN matrix (selected < total) → expect proportionally reduced score
    The OEM prediction color is GREEN with no value specified (NaN), so the score
    contribution per selected cell is exactly 1.0 (GREEN = full weight).
    """

    def _make_all_green_points(self, scenario, n_rows, n_cols):
        return [
            matrix_processing.TestPoint(
                row=i,
                col=j,
                color=common.PredictionColor.GREEN,
                test_range=matrix_processing.TestRange.STANDARD,
            )
            for i in range(n_rows)
            for j in range(n_cols)
        ]

    def _make_points_df(self, scenario, test_points):
        return pd.DataFrame(
            {
                "Scenario": [scenario] * len(test_points),
                "Test point": [f"({tp.row}, {tp.col})" for tp in test_points],
                "Range": [matrix_processing.TestRange.STANDARD.value]
                * len(test_points),
                "OEM Prediction": [common.PredictionColor.GREEN.value]
                * len(test_points),
                "Expected value": [float("nan")] * len(test_points),
                "Value": [float("nan")] * len(test_points),
            }
        )

    def _make_input_params_df(self, vehicle_response):
        return pd.DataFrame(
            {
                "Input parameter": ["Vehicle response"],
                "Value": [vehicle_response.value],
            }
        )

    def _score(self, scenario, selected_points, matrix_total_cells, vehicle_response):
        points_df = self._make_points_df(scenario, selected_points)
        score, _ = test_info.compute_lsc_loadcase_score(
            scenario,
            {scenario: points_df},
            self._make_input_params_df(vehicle_response),
            selected_points,
            matrix_total_cells=matrix_total_cells,
        )
        return score.standard_score

    # --- LSC avoidance scenarios (vehicle_response=WARNING, value≤0 → GREEN) ---
    # For avoidance tests value=NaN → OEM color used directly (GREEN → 1.0 weight).

    def _avoidance_cases(self):
        # (scenario, n_rows, n_cols, total_score)
        return [
            ("CCCscp SfS", 4, 5, 3.0),
            ("CMCscp SfS", 4, 7, 3.0),
            ("CCFtap SfS", 4, 3, 1.0),
            ("CMFtap SfS", 4, 4, 3.0),
            ("CBNAO SfS", 3, 3, 3.0),
            ("CPMRCm", 3, 2, 1.5),
            ("CPMRCs", 3, 3, 1.5),
        ]

    def test_full_selection_gives_full_score(self):
        """When every matrix cell is selected GREEN, score == total_score."""
        for scenario, n_rows, n_cols, total_score in self._avoidance_cases():
            with self.subTest(scenario=scenario):
                all_points = self._make_all_green_points(scenario, n_rows, n_cols)
                result = self._score(
                    scenario,
                    all_points,
                    matrix_total_cells=n_rows * n_cols,
                    vehicle_response=data_model.VehicleResponse.WARNING,
                )
                self.assertAlmostEqual(
                    result,
                    total_score,
                    places=6,
                    msg=f"{scenario}: expected {total_score}, got {result}",
                )

    def test_partial_selection_reduces_score_proportionally(self):
        """
        Selecting k out of N cells (all GREEN, no value) gives score = (k/N) × total_score.
        We drop the last column of each scenario to create a predictable partial selection.
        """
        for scenario, n_rows, n_cols, total_score in self._avoidance_cases():
            with self.subTest(scenario=scenario):
                # Select all cells except the last column → k = n_rows × (n_cols - 1)
                selected = [
                    matrix_processing.TestPoint(
                        row=i,
                        col=j,
                        color=common.PredictionColor.GREEN,
                        test_range=matrix_processing.TestRange.STANDARD,
                    )
                    for i in range(n_rows)
                    for j in range(n_cols - 1)
                ]
                k = len(selected)
                n = n_rows * n_cols
                expected = (k / n) * total_score
                result = self._score(
                    scenario,
                    selected,
                    matrix_total_cells=n,
                    vehicle_response=data_model.VehicleResponse.WARNING,
                )
                self.assertAlmostEqual(
                    result,
                    expected,
                    places=6,
                    msg=f"{scenario}: expected {expected:.4f}, got {result:.4f}",
                )

    # --- CPMFC (LSC mitigation, total_score=2.0) ---

    def test_cpmfc_full_selection_gives_full_score(self):
        n_rows, n_cols, total_score = 3, 3, 2.0
        all_points = self._make_all_green_points("CPMFC", n_rows, n_cols)
        # CPMFC uses _compute_cpmfc_color when value is set; NaN → OEM color used directly.
        result = self._score(
            "CPMFC",
            all_points,
            matrix_total_cells=n_rows * n_cols,
            vehicle_response=data_model.VehicleResponse.WARNING,
        )
        self.assertAlmostEqual(result, total_score)

    def test_cpmfc_partial_selection_reduces_score(self):
        n_rows, n_cols, total_score = 3, 3, 2.0
        # Select 6 of 9 cells
        selected = self._make_all_green_points("CPMFC", n_rows, n_cols - 1)
        expected = (6 / 9) * total_score
        result = self._score(
            "CPMFC",
            selected,
            matrix_total_cells=n_rows * n_cols,
            vehicle_response=data_model.VehicleResponse.WARNING,
        )
        self.assertAlmostEqual(result, expected, places=6)

    # --- CBDA (LSC dooring, total_score=2.0) ---

    def test_cbda_full_selection_gives_full_score(self):
        n_rows, n_cols, total_score = 4, 3, 2.0
        all_points = self._make_all_green_points("CBDA", n_rows, n_cols)
        result = self._score(
            "CBDA",
            all_points,
            matrix_total_cells=n_rows * n_cols,
            vehicle_response=data_model.VehicleResponse.RETENTION,
        )
        self.assertAlmostEqual(result, total_score)

    def test_cbda_9_of_12_cells_selected_gives_1_5(self):
        """Concrete CBDA example from the bug report: 9/12 GREEN → 1.5/2.0."""
        n_rows, n_cols, total_score = 4, 3, 2.0
        # 9 cells: rows 0-2, all 3 cols (row 3 = missing 20 km/h column gap cells)
        selected = [
            matrix_processing.TestPoint(
                row=i,
                col=j,
                color=common.PredictionColor.GREEN,
                test_range=matrix_processing.TestRange.STANDARD,
            )
            for i in range(3)
            for j in range(3)
        ]
        result = self._score(
            "CBDA",
            selected,
            matrix_total_cells=n_rows * n_cols,
            vehicle_response=data_model.VehicleResponse.RETENTION,
        )
        self.assertAlmostEqual(result, 1.5, places=6)

    def test_cell_weight_denominator_is_total_not_selected(self):
        """
        Regression guard: with N selected cells and matrix_total_cells=M > N,
        the score must be strictly less than total_score even if all selected cells are GREEN.
        """
        n_rows, n_cols, total_score = 4, 3, 2.0
        # Pick only 1 cell from a 12-cell CBDA matrix
        selected = [
            matrix_processing.TestPoint(
                row=0,
                col=0,
                color=common.PredictionColor.GREEN,
                test_range=matrix_processing.TestRange.STANDARD,
            )
        ]
        result = self._score(
            "CBDA",
            selected,
            matrix_total_cells=n_rows * n_cols,
            vehicle_response=data_model.VehicleResponse.RETENTION,
        )
        # 1/12 × 2.0 = 0.1667 — must be well below 2.0
        self.assertAlmostEqual(result, total_score / (n_rows * n_cols), places=6)
        self.assertLess(result, total_score)


class TestCbdaPreprocessExpectedValueNames(unittest.TestCase):
    """Tests that preprocess_stage_subelement emits correct expected-value column names."""

    @staticmethod
    def _make_lsc_dfs():
        # Minimal "LSC - Ped & Cyc pred." DataFrame with a 2-row CBDA matrix.
        # Layout: CBDA marker at idx=0 → start_row=2; two data rows (Gap 0.50 m and 1.00 m).
        # Columns: [0=Gap/marker, 1=Target speed header/values, 2=values, 3=values]
        data = [
            ["CBDA", "Target speed", np.nan, np.nan],  # marker row
            [np.nan, "10 km/h", "15 km/h", "20 km/h"],  # speed header
            ["0.50 m", "Green", "Green", "Green"],  # data row 1
            ["1.00 m", "Green", "Green", "Green"],  # data row 2
        ]
        return {"LSC - Ped & Cyc pred.": pd.DataFrame(data)}

    def _preprocess_cbda(self, vehicle_response, input_selected_points):
        with patch(
            "euroncap_rating_2026.crash_avoidance.test_info.get_vehicle_response",
            return_value=vehicle_response,
        ):
            stage_subelement = test_info.preprocess_stage_subelement(
                self._make_lsc_dfs(),
                "Low Speed Collisions",
                "Pedestrian & cyclist",
                input_selected_points=input_selected_points,
            )
        cbda_rows = stage_subelement.test_points_df[
            stage_subelement.test_points_df["Scenario"] == "CBDA"
        ].reset_index(drop=True)
        return stage_subelement, cbda_rows

    def _make_input_points(self, function_name):
        return {
            "CBDA": [
                {
                    "Gap": "0.50 m",
                    "Target speed": "10 km/h",
                    "Door selected": "Driver",
                    "Function": function_name,
                }
            ]
        }

    def test_retention_expected_value_name(self):
        _, cbda_rows = self._preprocess_cbda(
            data_model.VehicleResponse.RETENTION,
            self._make_input_points("Retention"),
        )
        self.assertEqual(len(cbda_rows), 1)
        self.assertEqual(cbda_rows.iloc[0]["Expected value"], "TTC @ t_door_opening")

    def test_warning_expected_value_name(self):
        _, cbda_rows = self._preprocess_cbda(
            data_model.VehicleResponse.WARNING,
            self._make_input_points("Warning"),
        )
        self.assertEqual(len(cbda_rows), 1)
        self.assertEqual(cbda_rows.iloc[0]["Expected value"], "TTC @ t_warning")

    def test_information_expected_value_name(self):
        _, cbda_rows = self._preprocess_cbda(
            data_model.VehicleResponse.INFORMATION,
            self._make_input_points("Information"),
        )
        self.assertEqual(len(cbda_rows), 1)
        self.assertEqual(cbda_rows.iloc[0]["Expected value"], "TTC @ t_information")

    def test_retention_default_value_is_nan(self):
        _, cbda_rows = self._preprocess_cbda(
            data_model.VehicleResponse.RETENTION,
            self._make_input_points("Retention"),
        )
        self.assertTrue(pd.isna(cbda_rows.iloc[0]["Value"]))

    def test_retention_single_row_per_point(self):
        input_points = {
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
        _, cbda_rows = self._preprocess_cbda(
            data_model.VehicleResponse.RETENTION, input_points
        )
        self.assertEqual(len(cbda_rows), 2)


if __name__ == "__main__":
    unittest.main()
