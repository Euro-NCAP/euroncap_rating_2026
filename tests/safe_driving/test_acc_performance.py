# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import unittest
import pandas as pd
import numpy as np
import re
from unittest.mock import patch, MagicMock
from euroncap_rating_2026.safe_driving import acc_performance
from euroncap_rating_2026.safe_driving import data_model
from euroncap_rating_2026.crash_avoidance import (
    matrix_processing as ca_matrix_processing,
)
from euroncap_rating_2026 import common


class TestPreprocess(unittest.TestCase):
    """Test cases for the preprocess function."""

    def _create_param_df(self, acc_type_of_system="Automatic resume"):
        """Helper to create an Input parameters DataFrame."""
        return pd.DataFrame(
            {
                "param_code": ["SD - Vehicle ACC", "SD - Vehicle ACC"],
                "Stage Subelement": ["ACC", "ACC"],
                "Input parameter": ["Type of system", "Other param"],
                "Value": [acc_type_of_system, "Other value"],
            }
        )

    def test_preprocess_empty_prediction_df_returns_empty(self):
        """Test that preprocess returns an empty-but-valid 3-tuple when the
        prediction df is missing (the caller unpacks 3 values unconditionally)."""
        dfs = {}
        test_points, verification_df, scenario_scores_df = acc_performance.preprocess(
            dfs
        )

        self.assertEqual(test_points, [])
        self.assertTrue(verification_df.empty)
        self.assertTrue(scenario_scores_df.empty)

    def test_preprocess_empty_acc_prediction_df_returns_empty(self):
        """Test that preprocess returns an empty-but-valid 3-tuple when the
        prediction df is empty, and passes through any existing Scenario
        Scores sheet unchanged rather than clobbering it."""
        existing_scenario_scores_df = pd.DataFrame(
            {"Category": ["Foo"], "Scenario": ["Bar"], "Score": [1.0]}
        )
        dfs = {
            "VA - ACC pred.": pd.DataFrame(),
            "Scenario Scores": existing_scenario_scores_df,
        }
        test_points, verification_df, scenario_scores_df = acc_performance.preprocess(
            dfs
        )

        self.assertEqual(test_points, [])
        self.assertTrue(verification_df.empty)
        pd.testing.assert_frame_equal(scenario_scores_df, existing_scenario_scores_df)

    def test_preprocess_expected_value_column_is_v_reduction(self):
        """All generated ACC verification rows must carry 'v_reduction' as Expected value."""
        test_point = ca_matrix_processing.TestPoint(
            row=0,
            col=0,
            color=common.PredictionColor.GREEN,
            test_range=ca_matrix_processing.TestRange.STANDARD,
            attributes={
                "Scenario": "CCRs straight",
                "VUT speed": "50 km/h",
                "Target speed": "0 km/h",
            },
        )

        def fake_get_test_matrix(_df, matrix_name, *_args, **_kwargs):
            if matrix_name == "CCRs straight":
                return [test_point]
            return []

        dfs = {
            "VA - ACC pred.": pd.DataFrame({"stub": [1]}),
            "Input parameters": pd.DataFrame(
                {
                    "Category": ["Auto-resume"],
                    "Input parameter": ["Type of system"],
                    "Value": ["Automatic resume"],
                }
            ),
            "Scenario Scores": pd.DataFrame(
                {
                    "Category": ["Auto-resume"],
                    "Scenario": [""],
                    "Value": [""],
                }
            ),
        }

        with patch.object(
            acc_performance.matrix_processing,
            "get_test_matrix",
            side_effect=fake_get_test_matrix,
        ):
            _, verification_df, _ = acc_performance.preprocess(dfs)

        self.assertFalse(verification_df.empty)
        block_categories = {"Road features", "Auto-resume"}
        test_point_rows = verification_df[
            ~verification_df["Category"].fillna("").isin(block_categories)
        ]
        self.assertFalse(test_point_rows.empty)
        self.assertTrue(
            (test_point_rows["Expected value"] == "v_reduction").all(),
            "All test-point rows must have 'v_reduction' in the Expected value column",
        )


class TestMatrixTestKeys(unittest.TestCase):
    """Test matrix test key definitions."""

    def test_car_to_car_longitudinal_keys(self):
        """Test Car-to-Car Longitudinal matrix test keys."""
        matrix_test_keys = ["CCRs-straight", "CCRs-curved", "CCRm", "CCRb"]

        for key in matrix_test_keys:
            self.assertIn("CCR", key, f"Key {key} should contain 'CCR'")

    def test_car_to_car_cut_in_out_keys(self):
        """Test Car-to-Car Cut-in/Cut-out matrix test keys."""
        matrix_test_keys = ["CCRcut-in", "CCRcut-out"]

        self.assertEqual(len(matrix_test_keys), 2)
        self.assertIn("CCRcut-in", matrix_test_keys)
        self.assertIn("CCRcut-out", matrix_test_keys)

    def test_car_to_ptw_longitudinal_keys(self):
        """Test Car-to-PTW Longitudinal matrix test keys."""
        matrix_test_keys = ["CMRs-straight", "CMRs-curved", "CMRm", "CMRb"]

        for key in matrix_test_keys:
            self.assertIn("CMR", key, f"Key {key} should contain 'CMR'")

    def test_car_to_ptw_cut_in_out_keys(self):
        """Test Car-to-PTW Cut-in/Cut-out matrix test keys."""
        matrix_test_keys = ["CMRcut-in", "CMRcut-out"]

        self.assertEqual(len(matrix_test_keys), 2)
        self.assertIn("CMRcut-in", matrix_test_keys)
        self.assertIn("CMRcut-out", matrix_test_keys)

    def test_car_to_vru_longitudinal_keys(self):
        """Test Car-to-VRU Longitudinal matrix test keys."""
        matrix_test_keys = ["CPLA", "CBLA"]

        self.assertEqual(len(matrix_test_keys), 2)
        self.assertIn("CPLA", matrix_test_keys)
        self.assertIn("CBLA", matrix_test_keys)


class TestMatrixScoring(unittest.TestCase):
    """Matrix scoring tests that exercise acc_performance.compute_score()."""

    def _build_acc_verification_df(self, scenario_row: dict) -> pd.DataFrame:
        columns = [
            "Category",
            "Scenario",
            "VUT speed",
            "Target speed",
            "OEM Prediction",
            "Value",
        ]
        rows = [["", "", "", "", "", ""] for _ in range(9)]
        rows[7] = columns.copy()
        rows[8] = [
            "",
            scenario_row.get("Scenario", ""),
            scenario_row.get("VUT speed", ""),
            scenario_row.get("Target speed", "0 km/h"),
            scenario_row.get("OEM Prediction", ""),
            scenario_row.get("Value", ""),
        ]
        return pd.DataFrame(rows, columns=columns)

    def _expected_possible_scores(self, scenario: str) -> tuple[float, float]:
        n_rows = data_model.ACC_MATRIX_INDICES.get(scenario, {}).get("n_rows", 0)
        possible_score = 1.0 / n_rows if n_rows > 0 else 0
        if scenario in ["CMR cut-in", "CMR cut-out"]:
            possible_score = 0.5 * possible_score
        return possible_score, possible_score / 2

    def test_green_above_v_scenario_minus_2_full_score(self):
        scenario = "CCRs straight"
        possible_score, _ = self._expected_possible_scores(scenario)
        acc_verif_df = self._build_acc_verification_df(
            {
                "Scenario": scenario,
                "VUT speed": "50 km/h",
                "OEM Prediction": "Green",
                "Value": 49,
            }
        )

        result_df, score_dict = acc_performance.compute_score(
            {"VA - ACC verif.": acc_verif_df}
        )

        self.assertEqual(score_dict[scenario], possible_score)
        self.assertEqual(result_df.loc[8, "Score"], possible_score)

    def test_green_equal_v_scenario_minus_2_earns_half_score(self):
        """An exact tie with the green threshold lands in the orange band (half score)."""
        scenario = "CCRs straight"
        _, half_score = self._expected_possible_scores(scenario)
        acc_verif_df = self._build_acc_verification_df(
            {
                "Scenario": scenario,
                "VUT speed": "50 km/h",
                "OEM Prediction": "Green",
                "Value": 48,
            }
        )

        result_df, score_dict = acc_performance.compute_score(
            {"VA - ACC verif.": acc_verif_df}
        )

        self.assertEqual(score_dict[scenario], half_score)
        self.assertEqual(result_df.loc[8, "Score"], half_score)
        self.assertEqual(result_df.loc[8, self._color_col(result_df)], "Orange")

    def test_green_midband_half_score(self):
        scenario = "CCRs straight"
        _, half_score = self._expected_possible_scores(scenario)
        acc_verif_df = self._build_acc_verification_df(
            {
                "Scenario": scenario,
                "VUT speed": "50 km/h",
                "OEM Prediction": "Green",
                "Value": 20,
            }
        )

        result_df, score_dict = acc_performance.compute_score(
            {"VA - ACC verif.": acc_verif_df}
        )

        self.assertEqual(score_dict[scenario], half_score)
        self.assertEqual(result_df.loc[8, "Score"], half_score)

    def test_green_exactly_15_no_score(self):
        scenario = "CCRs straight"
        acc_verif_df = self._build_acc_verification_df(
            {
                "Scenario": scenario,
                "VUT speed": "50 km/h",
                "OEM Prediction": "Green",
                "Value": 15,
            }
        )

        result_df, score_dict = acc_performance.compute_score(
            {"VA - ACC verif.": acc_verif_df}
        )

        self.assertEqual(score_dict[scenario], 0)
        self.assertEqual(result_df.loc[8, "Score"], 0)

    def test_green_below_15_no_score(self):
        scenario = "CCRs straight"
        acc_verif_df = self._build_acc_verification_df(
            {
                "Scenario": scenario,
                "VUT speed": "50 km/h",
                "OEM Prediction": "Green",
                "Value": 10,
            }
        )

        result_df, score_dict = acc_performance.compute_score(
            {"VA - ACC verif.": acc_verif_df}
        )

        self.assertEqual(score_dict[scenario], 0)
        self.assertEqual(result_df.loc[8, "Score"], 0)

    def test_orange_with_green_result_full_score(self):
        """Orange prediction + green test result (v_reduction > threshold) => full score."""
        scenario = "CCRs straight"
        possible_score, _ = self._expected_possible_scores(scenario)
        # threshold = 50 - 0 - 2 = 48; 49 > 48 → full score despite orange prediction
        acc_verif_df = self._build_acc_verification_df(
            {
                "Scenario": scenario,
                "VUT speed": "50 km/h",
                "OEM Prediction": "Orange",
                "Value": 49,
            }
        )

        result_df, score_dict = acc_performance.compute_score(
            {"VA - ACC verif.": acc_verif_df}
        )

        self.assertEqual(score_dict[scenario], possible_score)
        self.assertEqual(result_df.loc[8, "Score"], possible_score)

    def test_orange_above_15_half_score(self):
        scenario = "CCRs straight"
        _, half_score = self._expected_possible_scores(scenario)
        # v_reduction=16 is above 15 but below threshold (48) → half score
        acc_verif_df = self._build_acc_verification_df(
            {
                "Scenario": scenario,
                "VUT speed": "50 km/h",
                "OEM Prediction": "Orange",
                "Value": 16,
            }
        )

        result_df, score_dict = acc_performance.compute_score(
            {"VA - ACC verif.": acc_verif_df}
        )

        self.assertEqual(score_dict[scenario], half_score)
        self.assertEqual(result_df.loc[8, "Score"], half_score)

    def test_orange_exactly_15_no_score(self):
        scenario = "CCRs straight"
        acc_verif_df = self._build_acc_verification_df(
            {
                "Scenario": scenario,
                "VUT speed": "50 km/h",
                "OEM Prediction": "Orange",
                "Value": 15,
            }
        )

        result_df, score_dict = acc_performance.compute_score(
            {"VA - ACC verif.": acc_verif_df}
        )

        self.assertEqual(score_dict[scenario], 0)
        self.assertEqual(result_df.loc[8, "Score"], 0)

    def test_orange_below_15_no_score(self):
        scenario = "CCRs straight"
        acc_verif_df = self._build_acc_verification_df(
            {
                "Scenario": scenario,
                "VUT speed": "50 km/h",
                "OEM Prediction": "Orange",
                "Value": 10,
            }
        )

        result_df, score_dict = acc_performance.compute_score(
            {"VA - ACC verif.": acc_verif_df}
        )

        self.assertEqual(score_dict[scenario], 0)
        self.assertEqual(result_df.loc[8, "Score"], 0)

    def test_blank_value_green_prediction_scores_full(self):
        """No measured v_reduction yet (Value blank) -- inherit the GREEN
        OEM prediction directly rather than defaulting v_reduction to 0."""
        scenario = "CCRs straight"
        possible_score, _ = self._expected_possible_scores(scenario)
        acc_verif_df = self._build_acc_verification_df(
            {
                "Scenario": scenario,
                "VUT speed": "50 km/h",
                "OEM Prediction": "Green",
                "Value": float("nan"),
            }
        )

        result_df, score_dict = acc_performance.compute_score(
            {"VA - ACC verif.": acc_verif_df}
        )

        self.assertEqual(score_dict[scenario], possible_score)
        self.assertEqual(result_df.loc[8, "Score"], possible_score)

    def test_blank_value_orange_prediction_scores_half(self):
        scenario = "CCRs straight"
        _, half_score = self._expected_possible_scores(scenario)
        acc_verif_df = self._build_acc_verification_df(
            {
                "Scenario": scenario,
                "VUT speed": "50 km/h",
                "OEM Prediction": "Orange",
                "Value": float("nan"),
            }
        )

        result_df, score_dict = acc_performance.compute_score(
            {"VA - ACC verif.": acc_verif_df}
        )

        self.assertEqual(score_dict[scenario], half_score)
        self.assertEqual(result_df.loc[8, "Score"], half_score)

    def test_blank_value_red_prediction_scores_zero(self):
        scenario = "CCRs straight"
        acc_verif_df = self._build_acc_verification_df(
            {
                "Scenario": scenario,
                "VUT speed": "50 km/h",
                "OEM Prediction": "Red",
                "Value": float("nan"),
            }
        )

        result_df, score_dict = acc_performance.compute_score(
            {"VA - ACC verif.": acc_verif_df}
        )

        self.assertEqual(score_dict[scenario], 0)
        self.assertEqual(result_df.loc[8, "Score"], 0)

    def test_cmr_cut_in_out_uses_half_possible_score(self):
        scenario = "CMR cut-in"
        possible_score, _ = self._expected_possible_scores(scenario)
        acc_verif_df = self._build_acc_verification_df(
            {
                "Scenario": scenario,
                "VUT speed": "50 km/h",
                "OEM Prediction": "Green",
                "Value": 49,
            }
        )

        result_df, score_dict = acc_performance.compute_score(
            {"VA - ACC verif.": acc_verif_df}
        )

        self.assertEqual(score_dict[scenario], possible_score)
        self.assertEqual(result_df.loc[8, "Score"], possible_score)

    def test_cmr_cut_out_stationary_target_half_score(self):
        """Regression case for the CMR cut-out 90 km/h test point: the VUT
        enters the window at roughly 70 km/h -- following the SOV, not at the
        90 km/h ACC set speed -- against a stationary EMT, and
        v_reduction = 31.5 km/h before AEB takes over. That should score half
        (Orange), not full, since 15 < 31.5 < 70 - 2 = 68.

        The row is written as the template prints it: VUT speed 90 km/h (the
        ACC set speed) and Target speed 70 km/h (the SOV the VUT is trailing,
        which is the speed it has to shed).
        """
        scenario = "CMR cut-out"
        _, half_score = self._expected_possible_scores(scenario)
        acc_verif_df = self._build_acc_verification_df(
            {
                "Scenario": scenario,
                "VUT speed": "90 km/h",
                "Target speed": "70 km/h",
                "OEM Prediction": "Green",
                "Value": 31.5,
            }
        )

        result_df, score_dict = acc_performance.compute_score(
            {"VA - ACC verif.": acc_verif_df}
        )

        self.assertEqual(score_dict[scenario], half_score)
        self.assertEqual(result_df.loc[8, "Score"], half_score)
        self.assertEqual(result_df.loc[8, self._color_col(result_df)], "Orange")

    def test_ccr_cut_out_90_km_h_full_avoidance_threshold(self):
        """The 90 km/h cut-out row of the template: the VUT follows the lead
        vehicle (SOV) at 70 km/h and sheds speed against the standstill target
        the SOV reveals, so the full-avoidance threshold is 70 - 2 = 68 km/h.
        67 is still half, 69 is full.

        Not 90 - 2 = 88: the VUT never reaches its 90 km/h ACC set speed while
        following the SOV, so an 88 km/h reduction is unreachable and the test
        point would be unwinnable."""
        scenario = "CCR cut-out"
        possible_score, half_score = self._expected_possible_scores(scenario)
        common_row = {
            "Scenario": scenario,
            "VUT speed": "90 km/h",
            "Target speed": "70 km/h",
            "OEM Prediction": "Green",
        }

        _, score_dict = acc_performance.compute_score(
            {
                "VA - ACC verif.": self._build_acc_verification_df(
                    {**common_row, "Value": 67}
                )
            }
        )
        self.assertEqual(score_dict[scenario], half_score)

        _, score_dict = acc_performance.compute_score(
            {
                "VA - ACC verif.": self._build_acc_verification_df(
                    {**common_row, "Value": 69}
                )
            }
        )
        self.assertEqual(score_dict[scenario], possible_score)

    def test_green_above_threshold_with_nonzero_target_full_score(self):
        """v_reduction > v_scenario - v_target - 2 earns full score."""
        scenario = "CCRs straight"
        possible_score, _ = self._expected_possible_scores(scenario)
        # threshold = 50 - 20 - 2 = 28; 29 > 28 → full score
        acc_verif_df = self._build_acc_verification_df(
            {
                "Scenario": scenario,
                "VUT speed": "50 km/h",
                "Target speed": "20 km/h",
                "OEM Prediction": "Green",
                "Value": 29,
            }
        )

        result_df, score_dict = acc_performance.compute_score(
            {"VA - ACC verif.": acc_verif_df}
        )

        self.assertEqual(score_dict[scenario], possible_score)
        self.assertEqual(result_df.loc[8, "Score"], possible_score)

    def test_green_at_threshold_with_nonzero_target_earns_half_score(self):
        """v_reduction == v_scenario - v_target - 2 lands in the orange band (half score)."""
        scenario = "CCRs straight"
        _, half_score = self._expected_possible_scores(scenario)
        # threshold = 50 - 20 - 2 = 28; 28 not > 28, but 28 > 15 → half score
        acc_verif_df = self._build_acc_verification_df(
            {
                "Scenario": scenario,
                "VUT speed": "50 km/h",
                "Target speed": "20 km/h",
                "OEM Prediction": "Green",
                "Value": 28,
            }
        )

        result_df, score_dict = acc_performance.compute_score(
            {"VA - ACC verif.": acc_verif_df}
        )

        self.assertEqual(score_dict[scenario], half_score)
        self.assertEqual(result_df.loc[8, "Score"], half_score)
        self.assertEqual(result_df.loc[8, self._color_col(result_df)], "Orange")

    def test_green_midband_with_nonzero_target_half_score(self):
        """15 < v_reduction < v_scenario - v_target - 2 earns half score."""
        scenario = "CCRs straight"
        _, half_score = self._expected_possible_scores(scenario)
        # threshold = 50 - 20 - 2 = 28; 20 > 15 and 20 < 28 → half score
        acc_verif_df = self._build_acc_verification_df(
            {
                "Scenario": scenario,
                "VUT speed": "50 km/h",
                "Target speed": "20 km/h",
                "OEM Prediction": "Green",
                "Value": 20,
            }
        )

        result_df, score_dict = acc_performance.compute_score(
            {"VA - ACC verif.": acc_verif_df}
        )

        self.assertEqual(score_dict[scenario], half_score)
        self.assertEqual(result_df.loc[8, "Score"], half_score)

    def test_missing_target_speed_defaults_to_zero(self):
        """When Target speed column is absent, v_target defaults to 0 (backward-compatible)."""
        scenario = "CCRs straight"
        possible_score, _ = self._expected_possible_scores(scenario)
        # No "Target speed" key → v_target=0, threshold = 50-0-2=48; 49 > 48 → full score
        columns = ["Category", "Scenario", "VUT speed", "OEM Prediction", "Value"]
        rows = [["", "", "", "", ""] for _ in range(9)]
        rows[7] = columns.copy()
        rows[8] = ["", scenario, "50 km/h", "Green", 49]
        acc_verif_df = pd.DataFrame(rows, columns=columns)

        result_df, score_dict = acc_performance.compute_score(
            {"VA - ACC verif.": acc_verif_df}
        )

        self.assertEqual(score_dict[scenario], possible_score)
        self.assertEqual(result_df.loc[8, "Score"], possible_score)

    def _color_col(self, result_df: pd.DataFrame) -> str:
        # "Color" is renamed to "" for output formatting; resolve it by
        # position (the column right after "Score").
        return result_df.columns[result_df.columns.get_loc("Score") + 1]

    def _assert_row_score(self, scenario_row, expected_score, expected_color):
        scenario = scenario_row["Scenario"]
        acc_verif_df = self._build_acc_verification_df(scenario_row)

        result_df, score_dict = acc_performance.compute_score(
            {"VA - ACC verif.": acc_verif_df}
        )

        self.assertEqual(score_dict[scenario], expected_score)
        self.assertEqual(result_df.loc[8, "Score"], expected_score)
        self.assertEqual(result_df.loc[8, self._color_col(result_df)], expected_color)

    def test_cpla_green_above_30_full_score_regardless_of_vut_speed(self):
        """CPLA green threshold is a fixed 30 km/h: 31 at VUT 90 earns full
        score, whereas the speed-dependent rule (90 - 5 - 2 = 83) would not."""
        possible_score, _ = self._expected_possible_scores("CPLA")
        self._assert_row_score(
            {
                "Scenario": "CPLA",
                "VUT speed": "90 km/h",
                "Target speed": "5 km/h",
                "OEM Prediction": "Green",
                "Value": 31,
            },
            possible_score,
            "Green",
        )

    def test_cpla_green_exactly_30_earns_half_score(self):
        """An exact tie with the fixed 30 km/h threshold lands in the orange band."""
        _, half_score = self._expected_possible_scores("CPLA")
        self._assert_row_score(
            {
                "Scenario": "CPLA",
                "VUT speed": "90 km/h",
                "Target speed": "5 km/h",
                "OEM Prediction": "Green",
                "Value": 30,
            },
            half_score,
            "Orange",
        )

    def test_cpla_green_midband_half_score(self):
        _, half_score = self._expected_possible_scores("CPLA")
        self._assert_row_score(
            {
                "Scenario": "CPLA",
                "VUT speed": "60 km/h",
                "Target speed": "5 km/h",
                "OEM Prediction": "Green",
                "Value": 20,
            },
            half_score,
            "Orange",
        )

    def test_cpla_green_exactly_15_no_score(self):
        self._assert_row_score(
            {
                "Scenario": "CPLA",
                "VUT speed": "60 km/h",
                "Target speed": "5 km/h",
                "OEM Prediction": "Green",
                "Value": 15,
            },
            0,
            "Red",
        )

    def test_cbla_orange_with_green_result_full_score(self):
        """Orange prediction + v_reduction above the fixed 30 km/h threshold
        earns full score (35 would only be half under 80 - 20 - 2 = 58)."""
        possible_score, _ = self._expected_possible_scores("CBLA")
        self._assert_row_score(
            {
                "Scenario": "CBLA",
                "VUT speed": "80 km/h",
                "Target speed": "20 km/h",
                "OEM Prediction": "Orange",
                "Value": 35,
            },
            possible_score,
            "Green",
        )

    def test_cbla_orange_above_15_half_score(self):
        _, half_score = self._expected_possible_scores("CBLA")
        self._assert_row_score(
            {
                "Scenario": "CBLA",
                "VUT speed": "80 km/h",
                "Target speed": "20 km/h",
                "OEM Prediction": "Orange",
                "Value": 16,
            },
            half_score,
            "Orange",
        )

    def test_cbla_red_prediction_no_score_despite_high_reduction(self):
        self._assert_row_score(
            {
                "Scenario": "CBLA",
                "VUT speed": "70 km/h",
                "Target speed": "20 km/h",
                "OEM Prediction": "Red",
                "Value": 60,
            },
            0,
            "Red",
        )

    def test_car_scenario_keeps_speed_dependent_threshold(self):
        """Regression guard: the fixed VRU 30 km/h threshold must not leak
        into Car scenarios — 31 at VUT 50 (threshold 50 - 0 - 2 = 48) stays
        half score, not full."""
        _, half_score = self._expected_possible_scores("CCRs straight")
        self._assert_row_score(
            {
                "Scenario": "CCRs straight",
                "VUT speed": "50 km/h",
                "Target speed": "0 km/h",
                "OEM Prediction": "Green",
                "Value": 31,
            },
            half_score,
            "Orange",
        )

    def test_ptw_scenario_keeps_speed_dependent_threshold(self):
        """Same leak guard for PTW (Car-to-Motorcyclist) scenarios."""
        _, half_score = self._expected_possible_scores("CMRs straight")
        self._assert_row_score(
            {
                "Scenario": "CMRs straight",
                "VUT speed": "50 km/h",
                "Target speed": "0 km/h",
                "OEM Prediction": "Green",
                "Value": 31,
            },
            half_score,
            "Orange",
        )

    def test_cpla_four_full_credit_rows_roll_up_to_one_point(self):
        """4 full-credit CPLA rows (0.25 each) sum to the scenario max of 1.0."""
        columns = [
            "Category",
            "Scenario",
            "VUT speed",
            "Target speed",
            "OEM Prediction",
            "Value",
        ]
        rows = [["", "", "", "", "", ""] for _ in range(12)]
        rows[7] = columns.copy()
        for i, vut_speed in enumerate(["60", "70", "80", "90"]):
            rows[8 + i] = [
                "",
                "CPLA",
                f"{vut_speed} km/h",
                "5 km/h",
                "Green",
                31 + 10 * i,
            ]
        acc_verif_df = pd.DataFrame(rows, columns=columns)

        result_df, score_dict = acc_performance.compute_score(
            {"VA - ACC verif.": acc_verif_df}
        )

        self.assertEqual(score_dict["CPLA"], 1.0)
        for i in range(4):
            self.assertEqual(result_df.loc[8 + i, "Score"], 0.25)


class TestGreenVReductionThreshold(unittest.TestCase):
    """Unit tests for get_green_v_reduction_threshold()."""

    def test_vru_scenarios_use_fixed_30(self):
        self.assertEqual(
            acc_performance.get_green_v_reduction_threshold("CPLA", 90, 5), 30.0
        )
        self.assertEqual(
            acc_performance.get_green_v_reduction_threshold("CBLA", 60, 20), 30.0
        )

    def test_other_scenarios_use_speed_dependent_criterion(self):
        self.assertEqual(
            acc_performance.get_green_v_reduction_threshold("CCRs straight", 50, 0), 48
        )
        self.assertEqual(
            acc_performance.get_green_v_reduction_threshold("CCRs straight", 70, 20), 48
        )

    def test_cut_out_measures_against_the_sov_speed(self):
        """Cut-out sheds the lead vehicle's (SOV) speed, which the protocol
        table keeps in the "Target speed" column -- not the ACC set speed in
        the "VUT speed" column, which the VUT never reaches while trailing the
        SOV. The two template rows are therefore 48 and 68, not 18
        (v_VUT - v_SOV - 2) and not 68 / 88 (v_VUT - 2)."""
        self.assertEqual(
            acc_performance.get_green_v_reduction_threshold("CCR cut-out", 70, 50), 48
        )
        self.assertEqual(
            acc_performance.get_green_v_reduction_threshold("CCR cut-out", 90, 70), 68
        )
        self.assertEqual(
            acc_performance.get_green_v_reduction_threshold("CMR cut-out", 70, 50), 48
        )
        self.assertEqual(
            acc_performance.get_green_v_reduction_threshold("CMR cut-out", 90, 70), 68
        )

    def test_cut_out_without_target_speed_falls_back_to_vut_column(self):
        """A sheet generated by a superseded library revision keeps
        the SOV speed in the "VUT speed" column and 0 in "Target speed"; those
        rows keep scoring against the VUT column, exactly as they do today. The
        one thing that must never happen is falling through to 0 - 2, which
        would score every measured reduction green."""
        self.assertEqual(
            acc_performance.get_green_v_reduction_threshold("CCR cut-out", 50, 0), 48
        )
        self.assertEqual(
            acc_performance.get_green_v_reduction_threshold("CCR cut-out", 70, 0), 68
        )

    def test_cut_in_keeps_the_standard_criterion(self):
        """Cut-in is not cut-out: its "Target speed" really is the collision
        target, and the VUT does travel at its own set speed."""
        self.assertEqual(
            acc_performance.get_green_v_reduction_threshold("CCR cut-in", 50, 10), 38
        )
        self.assertEqual(
            acc_performance.get_green_v_reduction_threshold("CMR cut-in", 120, 70), 48
        )


class TestCutOutTemplateLayoutInvariance(TestMatrixScoring):
    """Restoring the protocol layout of the cut-out rows must not move any
    score: the sheet is a
    visual, and the same measured reduction has to score the same whichever
    layout the OEM's template was generated with.

    The two layouts that were released, for the second cut-out test point:

    - superseded (every released build before 5.4.6): VUT speed 70, Target speed 0
    - protocol (this template):         VUT speed 90, Target speed 70

    Both mean "the VUT is trailing a 70 km/h lead vehicle", so both score
    against 68.
    """

    LAYOUTS = {
        "legacy layout": {"VUT speed": "70 km/h", "Target speed": "0 km/h"},
        "protocol layout": {"VUT speed": "90 km/h", "Target speed": "70 km/h"},
    }

    def _score(self, scenario, layout, value):
        _, score_dict = acc_performance.compute_score(
            {
                "VA - ACC verif.": self._build_acc_verification_df(
                    {
                        "Scenario": scenario,
                        "OEM Prediction": "Green",
                        "Value": value,
                        **layout,
                    }
                )
            }
        )
        return score_dict[scenario]

    def test_both_layouts_score_identically(self):
        for scenario in ("CCR cut-out", "CMR cut-out"):
            possible_score, half_score = self._expected_possible_scores(scenario)
            for value, expected in ((10, 0), (31.5, half_score), (69, possible_score)):
                scores = {
                    name: self._score(scenario, layout, value)
                    for name, layout in self.LAYOUTS.items()
                }
                with self.subTest(scenario=scenario, value=value):
                    self.assertEqual(
                        list(set(scores.values())),
                        [expected],
                        f"{scenario} at v_reduction={value}: {scores}",
                    )


class TestVutSpeedParsing(unittest.TestCase):
    """Test VUT speed parsing from string."""

    def test_vut_speed_parsing_simple(self):
        """Test VUT speed parsing from simple string."""
        vut_speed_raw = "50"

        current_vut_speed = 0
        match = re.search(r"(\d+)", vut_speed_raw)
        if match:
            current_vut_speed = int(match.group(1))

        self.assertEqual(current_vut_speed, 50)

    def test_vut_speed_parsing_with_units(self):
        """Test VUT speed parsing from string with units."""
        vut_speed_raw = "50 km/h"

        current_vut_speed = 0
        match = re.search(r"(\d+)", vut_speed_raw)
        if match:
            current_vut_speed = int(match.group(1))

        self.assertEqual(current_vut_speed, 50)

    def test_vut_speed_parsing_no_match(self):
        """Test VUT speed parsing returns 0 when no match."""
        vut_speed_raw = "N/A"

        current_vut_speed = 0
        match = re.search(r"(\d+)", vut_speed_raw)
        if match:
            current_vut_speed = int(match.group(1))

        self.assertEqual(current_vut_speed, 0)


class TestRoadFeaturesScoring(unittest.TestCase):
    """Test Road Features scoring logic."""

    def test_road_features_pass_scores_02(self):
        """Test that Road Features PASS scores 0.2."""
        value = "pass"

        if str(value).lower() == "pass":
            score = 0.2
        else:
            score = 0.0

        self.assertEqual(score, 0.2)

    def test_road_features_fail_scores_0(self):
        """Test that Road Features FAIL scores 0."""
        value = "fail"

        if str(value).lower() == "pass":
            score = 0.2
        else:
            score = 0.0

        self.assertEqual(score, 0.0)

    def test_road_features_elements(self):
        """Test expected Road Features elements."""
        expected_elements = [
            "Curves",
            "Roundabouts",
            "Intersection (no right-of-way)",
            "Traffic lights",
            "Stop signs",
        ]

        self.assertEqual(len(expected_elements), 5)
        self.assertIn("Curves", expected_elements)
        self.assertIn("Traffic lights", expected_elements)


class TestAutoResumeComputeScore(unittest.TestCase):
    """Auto-resume scoring driven through the real
    acc_performance.compute_score(): an "N/A" Type of system (stamped into
    the block row's Scenario at preprocess) scores a real 0.0 with no user
    input required, while a blank Value on a real system stays unassessed."""

    def _build_df(self, scenario, value):
        columns = [
            "Category",
            "Scenario",
            "VUT speed",
            "Target speed",
            "OEM Prediction",
            "Value",
        ]
        rows = [["Auto-resume", scenario, "", "", "", value]]
        rows.extend([["", "", "", "", "", ""] for _ in range(2)])
        return pd.DataFrame(rows, columns=columns)

    def test_na_scenario_blank_value_scores_real_zero(self):
        result_df, score_dict = acc_performance.compute_score(
            {"VA - ACC verif.": self._build_df("N/A", "")}
        )

        self.assertEqual(score_dict["Auto-resume"], 0.0)
        self.assertFalse(pd.isna(score_dict["Auto-resume"]))
        self.assertEqual(result_df.loc[0, "Score"], 0.0)

    def test_na_scenario_stray_pass_still_scores_zero(self):
        _, score_dict = acc_performance.compute_score(
            {"VA - ACC verif.": self._build_df("N/A", "PASS")}
        )

        self.assertEqual(score_dict["Auto-resume"], 0.0)

    def test_automatic_resume_pass_scores_1(self):
        _, score_dict = acc_performance.compute_score(
            {"VA - ACC verif.": self._build_df("Automatic resume", "PASS")}
        )

        self.assertEqual(score_dict["Auto-resume"], 1.0)

    def test_driver_input_pass_scores_05(self):
        _, score_dict = acc_performance.compute_score(
            {"VA - ACC verif.": self._build_df("Driver input", "PASS")}
        )

        self.assertEqual(score_dict["Auto-resume"], 0.5)

    def test_real_system_blank_value_stays_unassessed(self):
        _, score_dict = acc_performance.compute_score(
            {"VA - ACC verif.": self._build_df("Automatic resume", "")}
        )

        self.assertTrue(pd.isna(score_dict["Auto-resume"]))


class TestAutoResumeScoring(unittest.TestCase):
    """Test Auto-Resume scoring logic."""

    def test_auto_resume_automatic_resume_scores_1(self):
        """Test that 'Automatic resume' scores 1.0."""
        element = "automatic resume"
        value = "pass"

        auto_resume_score = 0.0
        if str(value).lower() == "pass":
            element = str(element).strip().lower()
            if element == "automatic resume":
                auto_resume_score = 1.0
            elif element == "driver input":
                auto_resume_score = 0.5
            else:
                auto_resume_score = 0.0

        self.assertEqual(auto_resume_score, 1.0)

    def test_auto_resume_driver_input_scores_05(self):
        """Test that 'Driver input' scores 0.5."""
        element = "driver input"
        value = "pass"

        auto_resume_score = 0.0
        if str(value).lower() == "pass":
            element = str(element).strip().lower()
            if element == "automatic resume":
                auto_resume_score = 1.0
            elif element == "driver input":
                auto_resume_score = 0.5
            else:
                auto_resume_score = 0.0

        self.assertEqual(auto_resume_score, 0.5)

    def test_auto_resume_other_element_scores_0(self):
        """Test that other elements score 0."""
        element = "other"
        value = "pass"

        auto_resume_score = 0.0
        if str(value).lower() == "pass":
            element = str(element).strip().lower()
            if element == "automatic resume":
                auto_resume_score = 1.0
            elif element == "driver input":
                auto_resume_score = 0.5
            else:
                auto_resume_score = 0.0

        self.assertEqual(auto_resume_score, 0.0)

    def test_auto_resume_fail_scores_0(self):
        """Test that FAIL auto-resume scores 0."""
        element = "automatic resume"
        value = "fail"

        auto_resume_score = 0.0
        if str(value).lower() == "pass":
            element = str(element).strip().lower()
            if element == "automatic resume":
                auto_resume_score = 1.0

        self.assertEqual(auto_resume_score, 0.0)


class TestComputeScoreRoadFeaturesAutoResumeUnassessed(unittest.TestCase):
    """compute_score must distinguish an unassessed (blank) Road features /
    Auto-resume Value from an explicit FAIL: blank scores NaN, FAIL scores 0.0."""

    def _make_dfs(self, values):
        rows = [
            {
                "Category": "Road features",
                "Scenario": "Curves",
                "Value": values.get("Curves"),
            },
            {
                "Category": "Road features",
                "Scenario": "Roundabouts",
                "Value": values.get("Roundabouts"),
            },
            {
                "Category": "Road features",
                "Scenario": "Intersection (no right-of-way)",
                "Value": values.get("Intersection (no right-of-way)"),
            },
            {
                "Category": "Road features",
                "Scenario": "Traffic lights",
                "Value": values.get("Traffic lights"),
            },
            {
                "Category": "Road features",
                "Scenario": "Stop signs",
                "Value": values.get("Stop signs"),
            },
            {
                "Category": "Auto-resume",
                "Scenario": "Automatic resume",
                "Value": values.get("Auto-resume"),
            },
        ]
        return {"VA - ACC verif.": pd.DataFrame(rows)}

    def test_all_blank_default_is_unassessed_not_zero(self):
        _, score_dict = acc_performance.compute_score(self._make_dfs({}))

        for key in [
            "Curves",
            "Roundabouts",
            "Intersection (no right-of-way)",
            "Traffic lights",
            "Stop signs",
            "Auto-resume",
        ]:
            self.assertTrue(pd.isna(score_dict[key]), key)

    def test_explicit_fail_scores_zero_not_nan(self):
        _, score_dict = acc_performance.compute_score(
            self._make_dfs({"Curves": "FAIL"})
        )

        self.assertEqual(score_dict["Curves"], 0.0)
        # Untouched scenarios stay unassessed independently.
        self.assertTrue(pd.isna(score_dict["Roundabouts"]))

    def test_pass_scores_point_two(self):
        _, score_dict = acc_performance.compute_score(
            self._make_dfs({"Curves": "PASS"})
        )

        self.assertEqual(score_dict["Curves"], 0.2)

    def test_auto_resume_pass_automatic_resume_scores_1(self):
        _, score_dict = acc_performance.compute_score(
            self._make_dfs({"Auto-resume": "PASS"})
        )

        self.assertEqual(score_dict["Auto-resume"], 1.0)

    def test_auto_resume_blank_stays_unassessed(self):
        _, score_dict = acc_performance.compute_score(self._make_dfs({}))

        self.assertTrue(pd.isna(score_dict["Auto-resume"]))

    def test_auto_resume_fail_scores_zero_not_nan(self):
        _, score_dict = acc_performance.compute_score(
            self._make_dfs({"Auto-resume": "FAIL"})
        )

        self.assertEqual(score_dict["Auto-resume"], 0.0)


class TestRoadFeaturesCategoryIsForwardFilled(unittest.TestCase):
    """Regression: in the real 'VA - ACC verif.'
    sheet, "Category" is blank on every row but the first of a block (the
    same "blank means same as the row above" convention as every other
    sheet in this codebase) -- it must be forward-filled before filtering by
    Category, or every scenario but the first of a block category is
    silently dropped from scoring instead of counted."""

    def _blank_continuation_dfs(self, values):
        """Same shape as the real sheet: only the first row of each block
        carries an explicit Category label; the rest leave it blank."""
        rows = [
            {
                "Category": "Road features",
                "Scenario": "Curves",
                "Value": values.get("Curves"),
            },
            {
                "Category": None,
                "Scenario": "Roundabouts",
                "Value": values.get("Roundabouts"),
            },
            {
                "Category": None,
                "Scenario": "Intersection (no right-of-way)",
                "Value": values.get("Intersection (no right-of-way)"),
            },
            {
                "Category": None,
                "Scenario": "Traffic lights",
                "Value": values.get("Traffic lights"),
            },
            {
                "Category": None,
                "Scenario": "Stop signs",
                "Value": values.get("Stop signs"),
            },
            {
                "Category": "Auto-resume",
                "Scenario": "Driver input",
                "Value": values.get("Auto-resume"),
            },
        ]
        return {"VA - ACC verif.": pd.DataFrame(rows)}

    def test_every_scenario_in_the_block_is_scored_not_just_the_first(self):
        _, score_dict = acc_performance.compute_score(
            self._blank_continuation_dfs(
                {
                    "Curves": "PASS",
                    "Roundabouts": "PASS",
                    "Intersection (no right-of-way)": "PASS",
                    "Traffic lights": "PASS",
                    "Stop signs": "PASS",
                }
            )
        )

        for key in [
            "Curves",
            "Roundabouts",
            "Intersection (no right-of-way)",
            "Traffic lights",
            "Stop signs",
        ]:
            self.assertEqual(score_dict[key], 0.2, key)

    def test_continuation_row_still_recognised_as_auto_resume(self):
        """Auto-resume itself has only one real-world row (no blank
        continuation to forward-fill), but must keep working once Category
        is read via ffill instead of a plain per-row lookup."""
        _, score_dict = acc_performance.compute_score(
            self._blank_continuation_dfs({"Auto-resume": "PASS"})
        )

        self.assertEqual(score_dict["Auto-resume"], 0.5)


class TestAccMatrixIndices(unittest.TestCase):
    """Test ACC_MATRIX_INDICES constant from data_model."""

    def test_acc_matrix_indices_defined(self):
        """Test that ACC_MATRIX_INDICES is defined in data_model."""
        self.assertTrue(hasattr(data_model, "ACC_MATRIX_INDICES"))

    def test_acc_matrix_indices_has_expected_keys(self):
        """Test that ACC_MATRIX_INDICES has expected matrix keys."""
        expected_keys = [
            "CCRs-straight",
            "CCRs-curved",
            "CCRm",
            "CCRb",
            "CMRs-straight",
            "CMRs-curved",
            "CMRm",
            "CMRb",
        ]

        for key in expected_keys:
            if key in data_model.ACC_MATRIX_INDICES:
                self.assertIn(key, data_model.ACC_MATRIX_INDICES)


class TestRoadFeaturesCsvString(unittest.TestCase):
    """Test that preprocess emits road-features/auto-resume block rows in the verification df."""

    def _make_preprocess_dfs(self, acc_type_of_system="Automatic resume"):
        return {
            "VA - ACC pred.": pd.DataFrame({"stub": [1]}),
            "Input parameters": pd.DataFrame(
                {
                    "Category": ["Auto-resume"],
                    "Input parameter": ["Type of system"],
                    "Value": [acc_type_of_system],
                }
            ),
            "Scenario Scores": pd.DataFrame(
                {"Category": ["Auto-resume"], "Scenario": [""], "Value": [""]}
            ),
        }

    def test_csv_string_defined(self):
        """Verification df must contain Road features rows."""
        with patch.object(
            acc_performance.matrix_processing, "get_test_matrix", return_value=[]
        ):
            _, verification_df, _ = acc_performance.preprocess(
                self._make_preprocess_dfs()
            )
        self.assertIn(
            "Road features",
            verification_df["Category"].fillna("").values,
        )

    def test_csv_string_contains_road_features(self):
        """Verification df must contain the standard road-feature scenarios."""
        with patch.object(
            acc_performance.matrix_processing, "get_test_matrix", return_value=[]
        ):
            _, verification_df, _ = acc_performance.preprocess(
                self._make_preprocess_dfs()
            )
        scenarios = verification_df["Scenario"].fillna("").values
        self.assertIn("Curves", scenarios)
        self.assertIn("Roundabouts", scenarios)

    def test_csv_string_contains_auto_resume(self):
        """Verification df must contain an Auto-resume row."""
        with patch.object(
            acc_performance.matrix_processing, "get_test_matrix", return_value=[]
        ):
            _, verification_df, _ = acc_performance.preprocess(
                self._make_preprocess_dfs()
            )
        self.assertIn(
            "Auto-resume",
            verification_df["Category"].fillna("").values,
        )


class TestPredictionColorFiltering(unittest.TestCase):
    """Test prediction color filtering in preprocess."""

    def test_grey_color_excluded(self):
        """Test that GREY color predictions are excluded."""
        # Create a mock test point with GREY color
        grey_color = common.PredictionColor.GREY

        test_points = [
            MagicMock(color=common.PredictionColor.GREEN),
            MagicMock(color=common.PredictionColor.GREY),
            MagicMock(color=common.PredictionColor.ORANGE),
        ]

        filtered_points = [
            tp for tp in test_points if tp.color != common.PredictionColor.GREY
        ]

        self.assertEqual(len(filtered_points), 2)
        for tp in filtered_points:
            self.assertNotEqual(tp.color, common.PredictionColor.GREY)


class TestInputDrivenSelection(unittest.TestCase):
    def _create_dfs(self):
        return {
            "VA - ACC pred.": pd.DataFrame({"stub": [1]}),
            "Input parameters": pd.DataFrame(
                {
                    "Category": ["Auto-resume"],
                    "Input parameter": ["Type of system"],
                    "Value": ["Automatic resume"],
                }
            ),
            "Scenario Scores": pd.DataFrame(
                {
                    "Category": ["Auto-resume"],
                    "Scenario": [""],
                    "Value": [""],
                }
            ),
        }

    def _make_test_point(
        self, row, col, color=common.PredictionColor.GREEN, **attributes
    ):
        return ca_matrix_processing.TestPoint(
            row=row,
            col=col,
            color=color,
            test_range=ca_matrix_processing.TestRange.STANDARD,
            attributes=attributes,
        )

    def test_input_selection_matches_attributes_and_skips_omitted_scenarios(self):
        ccrs_points = [
            self._make_test_point(
                0, 0, **{"VUT speed": "50 km/h", "Target speed": "10 km/h"}
            ),
            self._make_test_point(
                1, 0, **{"VUT speed": "60 km/h", "Target speed": "20 km/h"}
            ),
        ]
        cmrs_points = [
            self._make_test_point(
                2, 0, **{"VUT speed": "30 km/h", "Target speed": "0 km/h"}
            )
        ]

        def fake_get_test_matrix(_df, matrix_name, *_args, **_kwargs):
            if matrix_name == "CCRs straight":
                return ccrs_points
            if matrix_name == "CMRs straight":
                return cmrs_points
            return []

        input_selected_points = {
            "CCRs straight": [
                {
                    "VUT speed": "50 km/h",
                    "Target speed": "10 km/h",
                    "Extra field": "kept",
                }
            ]
        }

        with patch.object(
            acc_performance.matrix_processing,
            "get_test_matrix",
            side_effect=fake_get_test_matrix,
        ):
            selected_points, verification_df, _ = acc_performance.preprocess(
                self._create_dfs(), input_selected_points=input_selected_points
            )

        self.assertEqual(len(selected_points), 1)
        self.assertEqual(selected_points[0].attributes["Scenario"], "CCRs straight")
        self.assertEqual(selected_points[0].attributes["Extra field"], "kept")

        scenario_rows = verification_df[verification_df["Scenario"] == "CCRs straight"]
        self.assertEqual(len(scenario_rows), 1)
        self.assertEqual(scenario_rows.iloc[0]["Extra field"], "kept")
        self.assertFalse((verification_df["Scenario"] == "CMRs straight").any())

    def test_input_selection_keeps_explicit_additional_target_for_cmrs_straight(self):
        cmrs_points = [
            self._make_test_point(
                0, 0, **{"VUT speed": "50 km/h", "Target speed": "10 km/h"}
            ),
            self._make_test_point(
                1, 0, **{"VUT speed": "60 km/h", "Target speed": "20 km/h"}
            ),
        ]

        def fake_get_test_matrix(_df, matrix_name, *_args, **_kwargs):
            if matrix_name == "CMRs straight":
                return cmrs_points
            return []

        input_selected_points = {
            "CMRs straight": [
                {
                    "VUT speed": "50 km/h",
                    "Target speed": "10 km/h",
                    "Additional target": "side",
                },
                {
                    "VUT speed": "60 km/h",
                    "Target speed": "20 km/h",
                    "Additional target": "front",
                },
            ]
        }

        with (
            patch.object(
                acc_performance.matrix_processing,
                "get_test_matrix",
                side_effect=fake_get_test_matrix,
            ),
            patch.object(acc_performance.random, "shuffle") as mock_shuffle,
        ):
            _, verification_df, _ = acc_performance.preprocess(
                self._create_dfs(), input_selected_points=input_selected_points
            )

        mock_shuffle.assert_not_called()
        cmrs_rows = verification_df[verification_df["Scenario"] == "CMRs straight"]
        self.assertEqual(cmrs_rows["Additional target"].tolist(), ["side", "front"])

    def _cmrm_points_at_two_impact_locations(self):
        """One CMRm speed row with a colored 25% cell and a grey 50% cell."""
        return [
            self._make_test_point(
                0,
                0,
                color=common.PredictionColor.GREY,
                **{
                    "VUT speed": "60 km/h",
                    "Target speed": "20 km/h",
                    "Impact location": "50%",
                },
            ),
            self._make_test_point(
                0,
                1,
                color=common.PredictionColor.ORANGE,
                **{
                    "VUT speed": "60 km/h",
                    "Target speed": "20 km/h",
                    "Impact location": "25%",
                },
            ),
        ]

    def _preprocess_cmrm(self, input_selected_points):
        def fake_get_test_matrix(_df, matrix_name, *_args, **_kwargs):
            if matrix_name == "CMRm":
                return self._cmrm_points_at_two_impact_locations()
            return []

        with patch.object(
            acc_performance.matrix_processing,
            "get_test_matrix",
            side_effect=fake_get_test_matrix,
        ):
            return acc_performance.preprocess(
                self._create_dfs(), input_selected_points=input_selected_points
            )

    def test_input_selection_accepts_impact_location_key_alias(self):
        """An input point spelled "Impact Location" (capital L) must match the
        point whose attribute key is "Impact location" — not be silently
        ignored and fall through to the first candidate — and must not be
        merged back as a duplicate attribute/column under the alias spelling."""
        selected_points, verification_df, _ = self._preprocess_cmrm(
            {
                "CMRm": [
                    {
                        "VUT speed": "60 km/h",
                        "Target speed": "20 km/h",
                        "Impact Location": "25%",
                    }
                ]
            }
        )

        self.assertEqual(len(selected_points), 1)
        self.assertEqual(selected_points[0].attributes["Impact location"], "25%")
        self.assertEqual(selected_points[0].color, common.PredictionColor.ORANGE)
        self.assertNotIn("Impact Location", selected_points[0].attributes)
        self.assertNotIn("Impact Location", verification_df.columns)

    def test_input_selection_merges_alias_only_input_key_as_canonical(self):
        """An input key the matched point doesn't carry at all, provided under
        an alias spelling (e.g. "GVT speed"), must be merged under the
        canonical spelling ("Target speed") — the canonical form survives."""
        points = [
            self._make_test_point(
                0,
                0,
                color=common.PredictionColor.ORANGE,
                **{"VUT speed": "60 km/h", "Impact location": "25%"},
            )
        ]

        def fake_get_test_matrix(_df, matrix_name, *_args, **_kwargs):
            return points if matrix_name == "CMRm" else []

        with patch.object(
            acc_performance.matrix_processing,
            "get_test_matrix",
            side_effect=fake_get_test_matrix,
        ):
            selected_points, verification_df, _ = acc_performance.preprocess(
                self._create_dfs(),
                input_selected_points={
                    "CMRm": [{"VUT speed": "60 km/h", "GVT speed": "20 km/h"}]
                },
            )

        self.assertEqual(len(selected_points), 1)
        self.assertEqual(selected_points[0].attributes["Target speed"], "20 km/h")
        self.assertNotIn("GVT speed", selected_points[0].attributes)
        self.assertNotIn("GVT speed", verification_df.columns)

    def test_input_selection_normalizes_impact_location_value(self):
        """A fractional impact-location value ("0.25") must match the
        percentage form stored on the test point ("25%")."""
        selected_points, _, _ = self._preprocess_cmrm(
            {
                "CMRm": [
                    {
                        "VUT speed": "60 km/h",
                        "Target speed": "20 km/h",
                        "Impact location": "0.25",
                    }
                ]
            }
        )

        self.assertEqual(len(selected_points), 1)
        self.assertEqual(selected_points[0].attributes["Impact location"], "25%")

    def test_input_selection_warns_when_grey_point_selected(self):
        """A requested point landing on a grey (unpredicted) cell is still
        selected for backwards compatibility, but must warn loudly and point
        at the impact location(s) that do carry a prediction."""
        with self.assertLogs(
            "euroncap_rating_2026.safe_driving.acc_performance", level="WARNING"
        ) as captured:
            selected_points, verification_df, _ = self._preprocess_cmrm(
                {
                    "CMRm": [
                        {
                            "VUT speed": "60 km/h",
                            "Target speed": "20 km/h",
                            "Impact location": "50%",
                        }
                    ]
                }
            )

        self.assertEqual(len(selected_points), 1)
        self.assertEqual(selected_points[0].color, common.PredictionColor.GREY)
        cmrm_rows = verification_df[verification_df["Scenario"] == "CMRm"]
        self.assertEqual(cmrm_rows.iloc[0]["OEM Prediction"], "Grey")

        warning_text = "\n".join(captured.output)
        self.assertIn("no OEM prediction (grey)", warning_text)
        self.assertIn("25%", warning_text)

    def test_input_selection_colored_match_does_not_warn(self):
        """Matching a colored cell must stay silent — the stale-selection
        warning is reserved for grey cells."""
        with self.assertNoLogs(
            "euroncap_rating_2026.safe_driving.acc_performance", level="WARNING"
        ):
            selected_points, _, _ = self._preprocess_cmrm(
                {
                    "CMRm": [
                        {
                            "VUT speed": "60 km/h",
                            "Target speed": "20 km/h",
                            "Impact location": "25%",
                        }
                    ]
                }
            )

        self.assertEqual(len(selected_points), 1)
        self.assertEqual(selected_points[0].color, common.PredictionColor.ORANGE)


class TestComputeScoreRowProcessing(unittest.TestCase):
    """Test row processing in compute_score."""

    def _create_verification_df(self):
        """Helper to create a verification DataFrame."""
        return pd.DataFrame(
            {
                "Scenario": [
                    "CCRs-straight",
                    "CCRs-straight",
                    "Road Features",
                    np.nan,
                    "Auto-Resume",
                ],
                "VUT speed": ["50 km/h", "60 km/h", "", "", ""],
                "Target speed": ["0", "0", "", "", ""],
                "Impact Location": ["Center", "Center", "", "", ""],
                "TTC": ["2.0", "2.0", "", "", ""],
                "Additional target": ["", "", "", "", ""],
                "OEM.Prediction": ["Green", "Orange", "", "", ""],
                "Value": [1, 40, "PASS", "", "PASS"],
                "Score": [0.0, 0.0, 0.0, 0.0, 0.0],
                "Element": ["", "", "Curves", "", "Automatic resume"],
            }
        )

    def test_skip_nan_scenario_rows(self):
        """Test that rows with NaN scenario are skipped."""
        df = self._create_verification_df()

        nan_count = 0
        for index, row in df.iterrows():
            scenario = row.get("Scenario", "")
            if pd.isna(scenario) or scenario == "":
                nan_count += 1

        self.assertEqual(nan_count, 1)


class TestComputeScoreRobustParsing(unittest.TestCase):
    """Regression tests for robust Value/VUT speed parsing in compute_score."""

    def _build_acc_verification_df(self, scenario_row: dict) -> pd.DataFrame:
        columns = [
            "Category",
            "Scenario",
            "VUT speed",
            "Target speed",
            "OEM Prediction",
            "Value",
        ]
        rows = [["", "", "", "", "", ""] for _ in range(9)]
        rows[7] = columns.copy()
        rows[8] = [
            "",
            scenario_row.get("Scenario", ""),
            scenario_row.get("VUT speed", ""),
            scenario_row.get("Target speed", "0 km/h"),
            scenario_row.get("OEM Prediction", ""),
            scenario_row.get("Value", ""),
        ]
        return pd.DataFrame(rows, columns=columns)

    def test_invalid_vut_speed_nan_forces_zero_score(self):
        """NaN VUT speed should not crash and must force score 0."""
        acc_verif_df = self._build_acc_verification_df(
            {
                "Scenario": "CCRs straight",
                "VUT speed": np.nan,
                "OEM Prediction": "Green",
                "Value": 40,
            }
        )
        result_df, score_dict = acc_performance.compute_score(
            {"VA - ACC verif.": acc_verif_df}
        )

        self.assertEqual(result_df.loc[8, "Score"], 0)
        self.assertEqual(score_dict["CCRs straight"], 0.0)

    def test_non_numeric_value_is_coerced_and_scores_zero(self):
        """Non-numeric Value should be coerced and lead to zero score."""
        acc_verif_df = self._build_acc_verification_df(
            {
                "Scenario": "CCRs straight",
                "VUT speed": "50 km/h",
                "OEM Prediction": "Green",
                "Value": "not-a-number",
            }
        )
        result_df, score_dict = acc_performance.compute_score(
            {"VA - ACC verif.": acc_verif_df}
        )

        self.assertEqual(result_df.loc[8, "Score"], 0)
        self.assertEqual(score_dict["CCRs straight"], 0.0)


class TestAllRedScenarioScoresZero(unittest.TestCase):
    """A scenario predicted Red on every test point produces no verification
    rows (preprocess drops Grey and Red cells) but is an assessed FAIL: it
    must score an explicit 0.0, not stay NaN. Only an all-Grey
    matrix is genuinely "not assessed" and stays NaN."""

    def _build_acc_verification_df(self, scenario_row: dict) -> pd.DataFrame:
        columns = [
            "Category",
            "Scenario",
            "VUT speed",
            "Target speed",
            "OEM Prediction",
            "Value",
        ]
        rows = [["", "", "", "", "", ""] for _ in range(9)]
        rows[7] = columns.copy()
        rows[8] = [
            "",
            scenario_row.get("Scenario", ""),
            scenario_row.get("VUT speed", ""),
            scenario_row.get("Target speed", "0 km/h"),
            scenario_row.get("OEM Prediction", ""),
            scenario_row.get("Value", ""),
        ]
        return pd.DataFrame(rows, columns=columns)

    def _build_acc_prediction_df(self, scenario: str, colors: list) -> pd.DataFrame:
        """Minimal "VA - ACC pred." layout for one scenario, mirroring the
        template: the scenario name in column 0, a header row below it, then
        the matrix data rows (row labels in column 0, color cells at the
        scenario's start_col from ACC_MATRIX_INDICES)."""
        start_col = data_model.ACC_MATRIX_INDICES[scenario]["start_col"]
        n_rows = len(colors)
        n_cols = len(colors[0])
        start_row = 2  # get_start_row returns the name's row + 2
        df_rows = [
            [np.nan] * (start_col + n_cols) for _ in range(start_row + n_rows + 2)
        ]
        df_rows[0][0] = scenario
        for i in range(n_rows):
            df_rows[start_row + i][0] = f"{10 * (i + 1)} km/h"
            for j in range(n_cols):
                df_rows[start_row + i][start_col + j] = colors[i][j]
        return pd.DataFrame(df_rows)

    def test_all_red_scenario_scores_zero_not_nan(self):
        n = data_model.ACC_MATRIX_INDICES["CPLA"]["n_rows"]
        m = data_model.ACC_MATRIX_INDICES["CPLA"]["n_cols"]
        acc_pred_df = self._build_acc_prediction_df("CPLA", [["Red"] * m] * n)
        # Another scenario has a normal verification row, as in a real file.
        acc_verif_df = self._build_acc_verification_df(
            {
                "Scenario": "CCRs straight",
                "VUT speed": "50 km/h",
                "OEM Prediction": "Green",
                "Value": 49,
            }
        )

        _, score_dict = acc_performance.compute_score(
            {"VA - ACC verif.": acc_verif_df, "VA - ACC pred.": acc_pred_df}
        )

        self.assertEqual(score_dict["CPLA"], 0.0)

    def test_partially_red_matrix_stays_grey_gated(self):
        """Red cells mixed with Grey ones (no Green/Orange anywhere) still
        mean the OEM assessed the scenario: explicit 0.0."""
        n = data_model.ACC_MATRIX_INDICES["CPLA"]["n_rows"]
        m = data_model.ACC_MATRIX_INDICES["CPLA"]["n_cols"]
        colors = [[np.nan] * m for _ in range(n)]
        colors[0][0] = "Red"
        acc_pred_df = self._build_acc_prediction_df("CPLA", colors)
        acc_verif_df = self._build_acc_verification_df(
            {
                "Scenario": "CCRs straight",
                "VUT speed": "50 km/h",
                "OEM Prediction": "Green",
                "Value": 49,
            }
        )

        _, score_dict = acc_performance.compute_score(
            {"VA - ACC verif.": acc_verif_df, "VA - ACC pred.": acc_pred_df}
        )

        self.assertEqual(score_dict["CPLA"], 0.0)

    def test_all_grey_scenario_stays_nan(self):
        n = data_model.ACC_MATRIX_INDICES["CPLA"]["n_rows"]
        m = data_model.ACC_MATRIX_INDICES["CPLA"]["n_cols"]
        acc_pred_df = self._build_acc_prediction_df("CPLA", [[np.nan] * m] * n)
        acc_verif_df = self._build_acc_verification_df(
            {
                "Scenario": "CCRs straight",
                "VUT speed": "50 km/h",
                "OEM Prediction": "Green",
                "Value": 49,
            }
        )

        _, score_dict = acc_performance.compute_score(
            {"VA - ACC verif.": acc_verif_df, "VA - ACC pred.": acc_pred_df}
        )

        self.assertTrue(np.isnan(score_dict["CPLA"]))

    def test_matrix_with_selectable_cells_but_no_rows_stays_nan(self):
        """Green/Orange cells mean preprocess would have produced rows; if
        the rows are missing the scenario is left unassessed (NaN), not
        force-zeroed by the matrix scan."""
        n = data_model.ACC_MATRIX_INDICES["CPLA"]["n_rows"]
        m = data_model.ACC_MATRIX_INDICES["CPLA"]["n_cols"]
        colors = [["Red"] * m for _ in range(n)]
        colors[0][0] = "Green"
        acc_pred_df = self._build_acc_prediction_df("CPLA", colors)
        acc_verif_df = self._build_acc_verification_df(
            {
                "Scenario": "CCRs straight",
                "VUT speed": "50 km/h",
                "OEM Prediction": "Green",
                "Value": 49,
            }
        )

        _, score_dict = acc_performance.compute_score(
            {"VA - ACC verif.": acc_verif_df, "VA - ACC pred.": acc_pred_df}
        )

        self.assertTrue(np.isnan(score_dict["CPLA"]))

    def test_missing_prediction_sheet_keeps_nan_behavior(self):
        acc_verif_df = self._build_acc_verification_df(
            {
                "Scenario": "CCRs straight",
                "VUT speed": "50 km/h",
                "OEM Prediction": "Green",
                "Value": 49,
            }
        )

        _, score_dict = acc_performance.compute_score({"VA - ACC verif.": acc_verif_df})

        self.assertTrue(np.isnan(score_dict["CPLA"]))

    # --- Preprocessed shape: colours live in the "(bg)" fill frame -------
    #
    # compute-score runs on the preprocessed workbook, where preprocess has
    # re-encoded every typed colour word as a cell fill and blanked the text
    # (only selected test points keep an "X"). The colours then only exist in
    # dfs["VA - ACC pred. (bg)"], a PredictionColor/None frame positionally
    # aligned with the text frame. The text frame still carries the scenario
    # name and row labels, which is how the matrix is located.

    def _build_acc_prediction_bg_df(
        self, scenario: str, colors: list, extra_rows: int = 0, extra_cols: int = 0
    ) -> pd.DataFrame:
        """ "(bg)" counterpart of _build_acc_prediction_df: PredictionColor
        members (or None) at the same positions, nothing else."""
        text_df = self._build_acc_prediction_df(
            scenario, [[np.nan] * len(colors[0])] * len(colors)
        )
        n_rows, n_cols = text_df.shape
        values = np.full((n_rows + extra_rows, n_cols + extra_cols), None, dtype=object)
        start_col = data_model.ACC_MATRIX_INDICES[scenario]["start_col"]
        for i, row in enumerate(colors):
            for j, color in enumerate(row):
                values[2 + i][start_col + j] = color
        return pd.DataFrame(values)

    def _control_verification_df(self):
        return self._build_acc_verification_df(
            {
                "Scenario": "CCRs straight",
                "VUT speed": "50 km/h",
                "OEM Prediction": "Green",
                "Value": 49,
            }
        )

    def test_all_red_fills_with_blank_text_score_zero(self):
        n = data_model.ACC_MATRIX_INDICES["CPLA"]["n_rows"]
        m = data_model.ACC_MATRIX_INDICES["CPLA"]["n_cols"]
        RED = common.PredictionColor.RED
        acc_pred_df = self._build_acc_prediction_df("CPLA", [[np.nan] * m] * n)
        acc_pred_bg_df = self._build_acc_prediction_bg_df("CPLA", [[RED] * m] * n)

        _, score_dict = acc_performance.compute_score(
            {
                "VA - ACC verif.": self._control_verification_df(),
                "VA - ACC pred.": acc_pred_df,
                "VA - ACC pred. (bg)": acc_pred_bg_df,
            }
        )

        self.assertEqual(score_dict["CPLA"], 0.0)

    def test_only_the_input_cells_red_in_fills_scores_zero(self):
        # The real template has a single dropdown column per CPLA row; the
        # rest of the grid is never filled, so it reads as unfilled/None.
        n = data_model.ACC_MATRIX_INDICES["CPLA"]["n_rows"]
        m = data_model.ACC_MATRIX_INDICES["CPLA"]["n_cols"]
        RED = common.PredictionColor.RED
        colors = [[None] * m for _ in range(n)]
        for row in colors:
            row[5] = RED
        acc_pred_df = self._build_acc_prediction_df("CPLA", [[np.nan] * m] * n)
        acc_pred_bg_df = self._build_acc_prediction_bg_df("CPLA", colors)

        _, score_dict = acc_performance.compute_score(
            {
                "VA - ACC verif.": self._control_verification_df(),
                "VA - ACC pred.": acc_pred_df,
                "VA - ACC pred. (bg)": acc_pred_bg_df,
            }
        )

        self.assertEqual(score_dict["CPLA"], 0.0)

    def test_selected_green_fill_under_x_mark_keeps_nan(self):
        # Mixed matrix in preprocessed shape: one Green cell (marked "X" in
        # the text frame) among Red fills. Not all-Red, so the missing rows
        # mean "unassessed", not FAIL.
        n = data_model.ACC_MATRIX_INDICES["CPLA"]["n_rows"]
        m = data_model.ACC_MATRIX_INDICES["CPLA"]["n_cols"]
        RED, GREEN = common.PredictionColor.RED, common.PredictionColor.GREEN
        colors = [[RED] * m for _ in range(n)]
        colors[0][0] = GREEN
        text_colors = [[np.nan] * m for _ in range(n)]
        text_colors[0][0] = "X"
        acc_pred_df = self._build_acc_prediction_df("CPLA", text_colors)
        acc_pred_bg_df = self._build_acc_prediction_bg_df("CPLA", colors)

        _, score_dict = acc_performance.compute_score(
            {
                "VA - ACC verif.": self._control_verification_df(),
                "VA - ACC pred.": acc_pred_df,
                "VA - ACC pred. (bg)": acc_pred_bg_df,
            }
        )

        self.assertTrue(np.isnan(score_dict["CPLA"]))

    def test_text_colour_word_wins_over_fill(self):
        n = data_model.ACC_MATRIX_INDICES["CPLA"]["n_rows"]
        m = data_model.ACC_MATRIX_INDICES["CPLA"]["n_cols"]
        RED = common.PredictionColor.RED
        text_colors = [[np.nan] * m for _ in range(n)]
        text_colors[1][2] = "Green"
        acc_pred_df = self._build_acc_prediction_df("CPLA", text_colors)
        acc_pred_bg_df = self._build_acc_prediction_bg_df("CPLA", [[RED] * m] * n)

        _, score_dict = acc_performance.compute_score(
            {
                "VA - ACC verif.": self._control_verification_df(),
                "VA - ACC pred.": acc_pred_df,
                "VA - ACC pred. (bg)": acc_pred_bg_df,
            }
        )

        self.assertTrue(np.isnan(score_dict["CPLA"]))

    def test_bg_frame_with_extra_rows_and_columns_still_scores_zero(self):
        # openpyxl counts styled-but-empty cells that pd.read_excel trims, so
        # the "(bg)" frame can be larger than the text frame.
        n = data_model.ACC_MATRIX_INDICES["CPLA"]["n_rows"]
        m = data_model.ACC_MATRIX_INDICES["CPLA"]["n_cols"]
        RED = common.PredictionColor.RED
        acc_pred_df = self._build_acc_prediction_df("CPLA", [[np.nan] * m] * n)
        acc_pred_bg_df = self._build_acc_prediction_bg_df(
            "CPLA", [[RED] * m] * n, extra_rows=3, extra_cols=2
        )

        _, score_dict = acc_performance.compute_score(
            {
                "VA - ACC verif.": self._control_verification_df(),
                "VA - ACC pred.": acc_pred_df,
                "VA - ACC pred. (bg)": acc_pred_bg_df,
            }
        )

        self.assertEqual(score_dict["CPLA"], 0.0)

    def test_bg_frame_without_text_frame_is_ignored(self):
        n = data_model.ACC_MATRIX_INDICES["CPLA"]["n_rows"]
        m = data_model.ACC_MATRIX_INDICES["CPLA"]["n_cols"]
        RED = common.PredictionColor.RED
        acc_pred_bg_df = self._build_acc_prediction_bg_df("CPLA", [[RED] * m] * n)

        _, score_dict = acc_performance.compute_score(
            {
                "VA - ACC verif.": self._control_verification_df(),
                "VA - ACC pred. (bg)": acc_pred_bg_df,
            }
        )

        self.assertTrue(np.isnan(score_dict["CPLA"]))


if __name__ == "__main__":
    unittest.main()
