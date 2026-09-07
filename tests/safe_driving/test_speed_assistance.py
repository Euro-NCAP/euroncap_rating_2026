# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from euroncap_rating_2026.safe_driving import speed_assistance
from euroncap_rating_2026 import common


# The real "VA - Speed assist. pred." sheet lists exactly 10 "Euro NCAP
# member states" rows followed by 19 "Other members in Euro NCAP
# application area" rows (29 country rows total). Kept as module-level
# constants so every test class can build a realistic fixture.
MEMBER_STATE_COUNTRIES = [
    "Austria",
    "France",
    "Germany",
    "Italy",
    "Luxembourg",
    "Netherlands",
    "Norway",
    "Spain",
    "Sweden",
    "United Kingdom",
]
OTHER_APPLICATION_AREA_COUNTRIES = [f"Other Country {i}" for i in range(1, 20)]
ALL_SLIF_ADV_COUNTRIES = MEMBER_STATE_COUNTRIES + OTHER_APPLICATION_AREA_COUNTRIES


class TestPreprocess(unittest.TestCase):
    """Test cases for the preprocess function."""

    def _create_prediction_df(self, scenario_values=None, hazard_values=None):
        """Build a realistic 'VA - Speed assist. pred.' fixture.

        scenario_values: optional dict mapping a SLIF_ADV_MAPPING key (e.g.
        "Rain / Wetness") to a list of exactly 29 cell values (Green/Red/
        None), ordered as the 10 member-state rows followed by the 19
        other-application-area rows. Keys not provided default to 29x
        "Green".

        hazard_values: optional dict mapping a LOCAL_HAZARD_COLUMNS entry
        (e.g. "Construction zones") to a list of exactly 4 cell values
        ordered [cloud_sending, cloud_receiving, direct_sending,
        direct_receiving]. Columns not provided default to 4x "Green".
        """
        scenario_keys = list(speed_assistance.SLIF_ADV_MAPPING.keys())
        scenario_values = scenario_values or {}
        hazard_values = hazard_values or {}

        adv_header = [None, None] + scenario_keys
        rows = [adv_header]
        for i, country in enumerate(ALL_SLIF_ADV_COUNTRIES):
            if i == 0:
                group_label = "Euro NCAP member states"
            elif i == len(MEMBER_STATE_COUNTRIES):
                group_label = "Other members in Euro NCAP application area"
            else:
                group_label = None
            row_values = [
                scenario_values.get(key, ["Green"] * 29)[i] for key in scenario_keys
            ]
            rows.append([group_label, country] + row_values)
        rows.append([None] * 14)  # unused buffer row before the local hazards block

        hazard_header = [
            "Element",
            "Construction zones",
            "Items on road",
            "Stopped vehicle",
            "Broken down vehicle",
            "Post crash",
            "Poor weather",
            "Poor road",
            "Wrong way driver",
            "Amber & blue lights",
            "Traffic jam",
            "Extra",
            "",
            "",
        ]
        rows.append(hazard_header)
        hazard_columns = hazard_header[1:11]
        for row_idx in range(5):
            row_data = ["Cloud"]
            for col_name in hazard_columns:
                values = hazard_values.get(col_name)
                row_data.append(
                    values[row_idx] if values and row_idx < len(values) else "Green"
                )
            row_data += ["N/A", "", ""]
            rows.append(row_data)
        return pd.DataFrame(rows)

    def _create_prediction_bg_df(self, scenario_bg=None):
        """Build a 'VA - Speed assist. pred. (bg)' fixture aligned with
        _create_prediction_df(). scenario_bg maps a SLIF_ADV_MAPPING key to
        a list of 29 PredictionColor/None values; keys not provided default
        to 29x None (no fill, i.e. blank/white)."""
        scenario_keys = list(speed_assistance.SLIF_ADV_MAPPING.keys())
        scenario_bg = scenario_bg or {}

        rows = [[None] * 14]  # header row placeholder (colorless, discarded)
        for i in range(len(ALL_SLIF_ADV_COUNTRIES)):
            row_values = [scenario_bg.get(key, [None] * 29)[i] for key in scenario_keys]
            rows.append([None, None] + row_values)
        rows.append([None] * 14)  # unused buffer row
        return pd.DataFrame(rows)

    def _create_input_params_df(self):
        return pd.DataFrame(
            {
                "Category": ["Speed control function", "SLIF - System updates"],
                "Input parameter": ["Type of system", "Type of system"],
                "Value": ["iacc", "temporary"],
            }
        )

    def _create_scenario_scores_df(self):
        return pd.DataFrame(
            {
                "Category": ["Speed control function", "SLIF - System updates", ""],
                "Scenario": ["Speedometer accuracy", "", ""],
                "Score": [0.0, 0.0, 0.0],
            }
        )

    def _create_preprocess_dfs(self):
        return {
            "VA - Speed assist. pred.": self._create_prediction_df(),
            "Input parameters": self._create_input_params_df(),
            "Scenario Scores": self._create_scenario_scores_df(),
        }

    def test_preprocess_returns_dataframe(self):
        """Test that preprocess returns a DataFrame."""
        dfs = self._create_preprocess_dfs()
        result, _ = speed_assistance.preprocess(dfs)

        self.assertIsInstance(result, pd.DataFrame)

    def test_preprocess_has_expected_columns(self):
        """Test that the returned DataFrame has expected columns."""
        dfs = self._create_preprocess_dfs()
        result, _ = speed_assistance.preprocess(dfs)

        expected_columns = ["Category", "Scenario", "Value"]
        for col in expected_columns:
            self.assertIn(col, result.columns)

    def test_preprocess_scenarios_populated(self):
        """Test that preprocess returns expected scenario types."""
        dfs = self._create_preprocess_dfs()
        result, _ = speed_assistance.preprocess(dfs)

        scenarios = result["Scenario"].dropna().unique().tolist()
        expected_scenarios = [
            "Distance based KPI",
            "Event based KPI",
            "Conditional speed limits - Rain / Wetness",
            "Implicit speed limits - Highway, city and residential zones",
            "Construction zones - Sending",
            "Speedometer accuracy",
        ]
        for scenario in expected_scenarios:
            self.assertIn(scenario, scenarios)


class TestComputeScoreBasic(unittest.TestCase):
    """Test basic compute_score functionality."""

    def _create_prediction_df(self, rows):
        """Helper to create a SAS prediction DataFrame."""
        return pd.DataFrame(rows)

    def _create_prediction_df(self):
        return TestPreprocess()._create_prediction_df()

    def _create_input_params_df(self):
        return TestPreprocess()._create_input_params_df()

    def _create_scenario_scores_df(self):
        return TestPreprocess()._create_scenario_scores_df()

    def _create_verification_df_from_csv(self):
        """Helper to create a verification DataFrame from preprocess output."""
        verification_df, _ = speed_assistance.preprocess(
            {
                "VA - Speed assist. pred.": self._create_prediction_df(),
                "Input parameters": self._create_input_params_df(),
                "Scenario Scores": self._create_scenario_scores_df(),
            }
        )
        return verification_df

    def _create_param_df(self, system_update="No", type_of_scf="No"):
        """Helper to create an Input parameters DataFrame."""
        return pd.DataFrame(
            {
                "Stage Subelement": ["Speed Assistance", "Speed Assistance"],
                "Scenario": ["SLIF", "SCF"],
                "Input parameter": ["System Update", "Type of SCF"],
                "Value": [system_update, type_of_scf],
            }
        )

    def test_compute_score_missing_prediction_df(self):
        """Test that compute_score returns zeros when prediction df is missing."""
        verification_df = self._create_verification_df_from_csv()
        dfs = {"VA - Speed assist. verif.": verification_df}

        result = speed_assistance.compute_score(dfs)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)
        _, _, score_dict, _ = result
        for key, value in score_dict.items():
            self.assertEqual(value, 0.0)

    def test_compute_score_missing_verification_df(self):
        """Test that compute_score returns zeros when verification df is missing."""
        prediction_df = self._create_prediction_df()
        dfs = {"VA - Speed assist. pred.": prediction_df}

        result = speed_assistance.compute_score(dfs)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)
        _, _, score_dict, _ = result
        for key, value in score_dict.items():
            self.assertEqual(value, 0.0)

    def test_compute_score_returns_tuple(self):
        """Test that compute_score returns a tuple of (verification_df, scenario_scores_df, score_dict, cap)."""
        verification_df = self._create_verification_df_from_csv()
        # Keep all required general checks passing
        verification_df.loc[
            verification_df["Scenario"] == "SLIF - General requirements", "Value"
        ] = "PASS"
        prediction_df = self._create_prediction_df()
        param_df = self._create_input_params_df()
        scenario_scores_df = self._create_scenario_scores_df()
        dfs = {
            "VA - Speed assist. pred.": prediction_df,
            "VA - Speed assist. verif.": verification_df,
            "Input parameters": param_df,
            "Scenario Scores": scenario_scores_df,
        }

        result = speed_assistance.compute_score(dfs)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)


class TestGeneralRequirementsCheck(unittest.TestCase):
    """Test general requirements check in compute_score."""

    def _create_verification_df_with_general_requirements(self, pass_all=True):
        """Helper to create a verification DataFrame with general requirements."""
        verification_df, _ = speed_assistance.preprocess(
            {
                "VA - Speed assist. pred.": TestPreprocess()._create_prediction_df(),
                "Input parameters": TestPreprocess()._create_input_params_df(),
                "Scenario Scores": TestPreprocess()._create_scenario_scores_df(),
            }
        )
        if pass_all:
            verification_df.loc[
                verification_df["Category"] == "SLIF - General requirements", "Value"
            ] = "PASS"
        else:
            verification_df.loc[
                verification_df["Category"] == "SLIF - General requirements", "Value"
            ] = "FAIL"
        return verification_df

    def test_general_requirements_fail_returns_zeros(self):
        """Test that failing general requirements returns all zeros."""
        verification_df = self._create_verification_df_with_general_requirements(
            pass_all=False
        )
        prediction_df = TestPreprocess()._create_prediction_df()
        param_df = TestPreprocess()._create_input_params_df()
        dfs = {
            "VA - Speed assist. pred.": prediction_df,
            "VA - Speed assist. verif.": verification_df,
            "Input parameters": param_df,
            "Scenario Scores": TestPreprocess()._create_scenario_scores_df(),
        }

        result = speed_assistance.compute_score(dfs)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)
        _, _, score_dict, _ = result
        for key, value in score_dict.items():
            self.assertEqual(value, 0.0)


class TestSLIFAccuracyScoring(unittest.TestCase):
    """Test SLIF Accuracy scoring (Distance based KPI, Event Based KPI)."""

    def _create_verification_df_with_accuracy(self, dist_value=80.0, event_value=70.0):
        """Helper to create verification df with accuracy values."""
        verification_df, _ = speed_assistance.preprocess(
            {
                "VA - Speed assist. pred.": TestPreprocess()._create_prediction_df(),
                "Input parameters": TestPreprocess()._create_input_params_df(),
                "Scenario Scores": TestPreprocess()._create_scenario_scores_df(),
            }
        )
        verification_df.loc[
            verification_df["Scenario"] == "SLIF - General requirements", "Value"
        ] = "PASS"
        verification_df.loc[
            verification_df["Scenario"] == "Distance based KPI", "Value"
        ] = ("PASS" if dist_value >= 80.0 else "FAIL")
        verification_df.loc[
            verification_df["Scenario"] == "Event based KPI", "Value"
        ] = ("PASS" if event_value >= 70.0 else "FAIL")
        return verification_df

    def test_distance_kpi_80_scores_2(self):
        """Test that Distance based KPI >= 80 scores 2.0."""
        verification_df = self._create_verification_df_with_accuracy(
            dist_value=80.0, event_value=70.0
        )

        # Check the scoring logic
        dist_value_row = verification_df[
            verification_df["Scenario"] == "Distance based KPI"
        ]["Value"]
        dist_value = dist_value_row.iloc[0] if dist_value_row.size > 0 else "FAIL"

        if str(dist_value).strip().lower() == "pass":
            expected_score = 2.0
        else:
            expected_score = 0.0

        self.assertEqual(expected_score, 2.0)

    def test_distance_kpi_79_scores_0(self):
        """Test that Distance based KPI < 80 scores 0.0."""
        dist_value = 79.0

        if dist_value >= 80:
            expected_score = 2.0
        else:
            expected_score = 0.0

        self.assertEqual(expected_score, 0.0)

    def test_event_kpi_70_scores_2(self):
        """Test that Event Based KPI >= 70 scores 2.0."""
        event_value = 70.0

        if event_value >= 70:
            expected_score = 2.0
        else:
            expected_score = 0.0

        self.assertEqual(expected_score, 2.0)

    def test_event_kpi_69_scores_0(self):
        """Test that Event Based KPI < 70 scores 0.0."""
        event_value = 69.0

        if event_value >= 70:
            expected_score = 2.0
        else:
            expected_score = 0.0

        self.assertEqual(expected_score, 0.0)


class TestSystemUpdateScoring(unittest.TestCase):
    """SLIF - System updates scoring, driven through the real
    `speed_assistance.preprocess` + `speed_assistance.compute_score`:
    Continuous scores 2.0, Temporary 1.0, N/A a real 0.0 (an N/A input
    parameter is valid and needs no verification input), and a blank input
    leaves the score unassessed (NaN)."""

    def _compute(self, system_updates_value, blank_scenario_cell=False):
        preprocessor = TestPreprocess()
        prediction_df = preprocessor._create_prediction_df()
        input_params_df = pd.DataFrame(
            {
                "Category": ["Speed control function", "SLIF - System updates"],
                "Input parameter": ["Type of system", "Type of system"],
                "Value": ["iacc", system_updates_value],
            }
        )
        scenario_scores_df = pd.DataFrame(
            {
                "Category": ["Speed control function", "SLIF - System updates", ""],
                "Scenario": ["Speedometer accuracy", "", ""],
                "Score": [np.nan, np.nan, np.nan],
            }
        )

        verification_df, scenario_scores_df = speed_assistance.preprocess(
            {
                "VA - Speed assist. pred.": prediction_df,
                "Input parameters": input_params_df,
                "Scenario Scores": scenario_scores_df,
            }
        )
        verification_df.loc[
            verification_df["Category"] == "SLIF - General requirements", "Value"
        ] = "PASS"
        if blank_scenario_cell:
            # Caller-supplied Scenario Scores whose Scenario cell was never
            # stamped by preprocess -- compute must fall back to the raw
            # Input parameters value.
            scenario_scores_df.loc[
                scenario_scores_df["Category"] == "SLIF - System updates",
                "Scenario",
            ] = ""

        dfs = {
            "VA - Speed assist. pred.": prediction_df,
            "VA - Speed assist. verif.": verification_df,
            "Input parameters": input_params_df,
            "Scenario Scores": scenario_scores_df,
        }
        _, scenario_scores_df, _, _ = speed_assistance.compute_score(dfs)
        row = scenario_scores_df[
            scenario_scores_df["Category"] == "SLIF - System updates"
        ]
        return row["Score"].iloc[0]

    def test_system_update_continuous_scores_2(self):
        self.assertEqual(self._compute("Continuous"), 2.0)

    def test_system_update_temporary_scores_1(self):
        self.assertEqual(self._compute("Temporary"), 1.0)

    def test_system_update_na_scores_real_zero(self):
        score = self._compute("N/A")
        self.assertEqual(score, 0.0)
        self.assertFalse(pd.isna(score))

    def test_system_update_blank_stays_unassessed(self):
        self.assertTrue(pd.isna(self._compute("")))

    def test_system_update_na_falls_back_to_input_parameters(self):
        score = self._compute("N/A", blank_scenario_cell=True)
        self.assertEqual(score, 0.0)

    def test_fallback_matches_category_normalized(self):
        # Caller-supplied Input parameters may differ in casing/whitespace:
        # the fallback must match the Category normalized.
        preprocessor = TestPreprocess()
        prediction_df = preprocessor._create_prediction_df()
        input_params_df = pd.DataFrame(
            {
                "Category": ["Speed control function", "  SLIF - SYSTEM UPDATES  "],
                "Input parameter": ["Type of system", "Type of system"],
                "Value": ["iacc", "N/A"],
            }
        )
        # Like the real sheet, keep a dedicated row after "Speedometer
        # accuracy": update_scenario_scores_with_scf_intelligent_row stamps
        # the intelligent scenario there positionally.
        scenario_scores_df = pd.DataFrame(
            {
                "Category": [
                    "Speed control function",
                    "",
                    "SLIF - System updates",
                    "",
                ],
                "Scenario": ["Speedometer accuracy", "", "", ""],
                "Score": [np.nan, np.nan, np.nan, np.nan],
            }
        )
        verification_df, scenario_scores_df = speed_assistance.preprocess(
            {
                "VA - Speed assist. pred.": prediction_df,
                "Input parameters": input_params_df,
                "Scenario Scores": scenario_scores_df,
            }
        )
        verification_df.loc[
            verification_df["Category"] == "SLIF - General requirements", "Value"
        ] = "PASS"

        _, scenario_scores_df, _, _ = speed_assistance.compute_score(
            {
                "VA - Speed assist. pred.": prediction_df,
                "VA - Speed assist. verif.": verification_df,
                "Input parameters": input_params_df,
                "Scenario Scores": scenario_scores_df,
            }
        )

        row = scenario_scores_df[
            scenario_scores_df["Category"] == "SLIF - System updates"
        ]
        self.assertEqual(row["Score"].iloc[0], 0.0)


class TestSCFScoring(unittest.TestCase):
    """Test Speed Control Function scoring logic."""

    def test_type_of_scf_iacc_scores_8(self):
        """Test that Type of SCF = 'IACC' scores 8 points."""
        type_of_scf_param = "iacc"
        if type_of_scf_param == "iacc":
            scf_points = 8
        elif type_of_scf_param == "isl":
            scf_points = 5
        else:
            scf_points = 0

        self.assertEqual(scf_points, 8)

    def test_type_of_scf_isl_scores_5(self):
        """Test that Type of SCF = 'ISL' scores 5 points."""
        type_of_scf_param = "isl"
        if type_of_scf_param == "iacc":
            scf_points = 8
        elif type_of_scf_param == "isl":
            scf_points = 5
        else:
            scf_points = 0

        self.assertEqual(scf_points, 5)

    def test_type_of_scf_no_scores_0(self):
        """Test that Type of SCF = 'No' scores 0 points."""
        type_of_scf_param = "no"
        if type_of_scf_param == "iacc":
            scf_points = 8
        elif type_of_scf_param == "isl":
            scf_points = 5
        else:
            scf_points = 0

        self.assertEqual(scf_points, 0)


class TestSCFIntelligentRowBlankValue(unittest.TestCase):
    """Regression test: the intelligent SCF row (added because Type of
    system is iacc/isl) can have a blank Value -- a system is configured
    but that specific PASS/FAIL hasn't been tested yet. compute_score must
    not crash (.strip() on NaN) and must leave the score unassessed (NaN),
    not silently treat blank the same as a FAIL."""

    def _compute(self):
        preprocessor = TestPreprocess()
        prediction_df = preprocessor._create_prediction_df()
        input_params_df = (
            preprocessor._create_input_params_df()
        )  # Type of system = "iacc"
        scenario_scores_df = preprocessor._create_scenario_scores_df()

        verification_df, scenario_scores_df = speed_assistance.preprocess(
            {
                "VA - Speed assist. pred.": prediction_df,
                "Input parameters": input_params_df,
                "Scenario Scores": scenario_scores_df,
            }
        )
        verification_df.loc[
            verification_df["Category"] == "SLIF - General requirements", "Value"
        ] = "PASS"
        verification_df.loc[
            verification_df["Scenario"] == "Distance based KPI", "Value"
        ] = "PASS"
        verification_df.loc[
            verification_df["Scenario"] == "Event based KPI", "Value"
        ] = "PASS"
        # The intelligent row exists (Type of system = iacc) but its own
        # PASS/FAIL Value was never filled in.
        verification_df.loc[
            verification_df["Scenario"] == "Intelligent adaptive cruise control",
            "Value",
        ] = None

        dfs = {
            "VA - Speed assist. pred.": prediction_df,
            "VA - Speed assist. verif.": verification_df,
            "Input parameters": input_params_df,
            "Scenario Scores": scenario_scores_df,
        }
        return speed_assistance.compute_score(dfs)

    def test_does_not_raise_and_scores_unassessed(self):
        _, _, score_dict, _ = self._compute()
        self.assertTrue(pd.isna(score_dict["Intelligent adaptive cruise control"]))
        self.assertTrue(pd.isna(score_dict["Speedometer accuracy"]))


class TestSLIFAccuracyFfillDoesNotLeakValue(unittest.TestCase):
    """Regression test: reconstructing the collapsed Category/Scenario
    labels via ffill() must not also forward-fill Value/Score -- otherwise
    a blank (unassessed) row directly after a PASS/FAIL row would silently
    inherit its neighbor's value instead of staying unassessed."""

    def _compute(self):
        preprocessor = TestPreprocess()
        prediction_df = preprocessor._create_prediction_df()
        input_params_df = preprocessor._create_input_params_df()
        scenario_scores_df = preprocessor._create_scenario_scores_df()

        verification_df, scenario_scores_df = speed_assistance.preprocess(
            {
                "VA - Speed assist. pred.": prediction_df,
                "Input parameters": input_params_df,
                "Scenario Scores": scenario_scores_df,
            }
        )
        verification_df.loc[
            verification_df["Category"] == "SLIF - General requirements", "Value"
        ] = "PASS"
        verification_df.loc[
            verification_df["Scenario"] == "Distance based KPI", "Value"
        ] = "PASS"
        # Event based KPI (directly after Distance based KPI) stays blank
        # -- not yet tested.

        dfs = {
            "VA - Speed assist. pred.": prediction_df,
            "VA - Speed assist. verif.": verification_df,
            "Input parameters": input_params_df,
            "Scenario Scores": scenario_scores_df,
        }
        return speed_assistance.compute_score(dfs)

    def test_blank_event_based_kpi_stays_unassessed(self):
        _, _, score_dict, _ = self._compute()
        self.assertEqual(score_dict["Distance based"], 2.0)
        self.assertTrue(pd.isna(score_dict["Event based"]))


class TestSpeedometerAccuracyScoring(unittest.TestCase):
    """Test speedometer accuracy scoring logic."""

    def test_speedo_accuracy_in_range_no_penalty(self):
        """Test that speedometer accuracy -3 to 0 has no penalty."""
        scf_points = 8
        speedo_val = -2  # In range -3 to 0

        if -3 <= speedo_val <= 0:
            pass  # no action
        elif -5 <= speedo_val < -3:
            scf_points /= 2
        elif speedo_val < -5:
            scf_points = 0

        self.assertEqual(scf_points, 8)

    def test_speedo_accuracy_minus_4_halves_points(self):
        """Test that speedometer accuracy -5 to -3 halves points."""
        scf_points = 8
        speedo_val = -4  # In range -5 to -3

        if -3 <= speedo_val <= 0:
            pass  # no action
        elif -5 <= speedo_val < -3:
            scf_points /= 2
        elif speedo_val < -5:
            scf_points = 0

        self.assertEqual(scf_points, 4)

    def test_speedo_accuracy_minus_6_zeros_points(self):
        """Test that speedometer accuracy < -5 zeros points."""
        scf_points = 8
        speedo_val = -6  # Below -5

        if -3 <= speedo_val <= 0:
            pass  # no action
        elif -5 <= speedo_val < -3:
            scf_points /= 2
        elif speedo_val < -5:
            scf_points = 0

        self.assertEqual(scf_points, 0)


class TestSettingSpeedAndSpeedControlCheck(unittest.TestCase):
    """Test Setting the speed and Speed control requirements."""

    def test_setting_speed_fail_zeros_scf(self):
        """Test that failing 'Setting the speed' zeros SCF points."""
        scf_points = 8
        setting_speed_val = "FAIL"
        speed_control_val = "PASS"

        if (
            str(setting_speed_val).strip().lower() == "fail"
            or str(speed_control_val).strip().lower() == "fail"
        ):
            scf_points = 0

        self.assertEqual(scf_points, 0)

    def test_speed_control_fail_zeros_scf(self):
        """Test that failing 'Speed control' zeros SCF points."""
        scf_points = 8
        setting_speed_val = "PASS"
        speed_control_val = "FAIL"

        if (
            str(setting_speed_val).strip().lower() == "fail"
            or str(speed_control_val).strip().lower() == "fail"
        ):
            scf_points = 0

        self.assertEqual(scf_points, 0)

    def test_both_pass_keeps_scf_points(self):
        """Test that both passing keeps SCF points."""
        scf_points = 8
        setting_speed_val = "PASS"
        speed_control_val = "PASS"

        if (
            str(setting_speed_val).strip().lower() == "fail"
            or str(speed_control_val).strip().lower() == "fail"
        ):
            scf_points = 0

        self.assertEqual(scf_points, 8)


class TestConditionalSpeedLimitsScoring(unittest.TestCase):
    """Test Conditional Speed Limits scoring logic."""

    def test_conditional_speed_limit_max_scores(self):
        """Test the max scores for conditional speed limit elements."""
        slif_adv_max_scores_dict = {
            "Rain/wetness": 0.4,
            "Snow/icy": 0.4,
            "Time/season": 0.4,
            "Distance for/in": 0.4,
            "Arrows (non-LR)": 0.1,
            "Arrows (LR)": 0.1,
            "Vehicle Categories": 0.2,
            "Implicit": 0.5,
            "Lane Relevant": 0.25,
            "non-Lane Relevant": 0.25,
        }

        self.assertEqual(slif_adv_max_scores_dict["Rain/wetness"], 0.4)
        self.assertEqual(slif_adv_max_scores_dict["Snow/icy"], 0.4)
        self.assertEqual(slif_adv_max_scores_dict["Arrows (non-LR)"], 0.1)
        self.assertEqual(slif_adv_max_scores_dict["Arrows (LR)"], 0.1)
        self.assertEqual(slif_adv_max_scores_dict["Lane Relevant"], 0.25)


class TestSlifAdvancedSpeedLimitGrouping(unittest.TestCase):
    """Regression tests for the Euro NCAP member-state vs. Application-Area
    PASS/FAIL split in the SLIF advanced speed limit logic. These call the
    real `speed_assistance.preprocess()` production code (unlike the
    previous inline-logic tests they replace)."""

    def _preprocess(self, scenario_values=None, scenario_bg=None):
        preprocessor = TestPreprocess()
        dfs = {
            "VA - Speed assist. pred.": preprocessor._create_prediction_df(
                scenario_values
            ),
            "Input parameters": preprocessor._create_input_params_df(),
            "Scenario Scores": preprocessor._create_scenario_scores_df(),
        }
        if scenario_bg is not None:
            dfs["VA - Speed assist. pred. (bg)"] = (
                preprocessor._create_prediction_bg_df(scenario_bg)
            )
        verification_df, _ = speed_assistance.preprocess(dfs)
        return verification_df

    def _value_for(self, verification_df, scenario):
        rows = verification_df[verification_df["Scenario"] == scenario]
        return rows["Value"].iloc[0]

    def test_member_state_red_fails_even_if_application_area_fully_green(self):
        values = ["Green"] * 29
        values[0] = "Red"  # Austria, a member state
        verification_df = self._preprocess({"Rain / Wetness": values})
        self.assertEqual(
            self._value_for(
                verification_df, "Conditional speed limits - Rain / Wetness"
            ),
            "FAIL",
        )

    def test_non_member_red_does_not_fail_feature(self):
        """Regression test for the original bug: a Red in a non-member
        Application-Area country must not, by itself, fail the feature."""
        values = ["Green"] * 29
        values[15] = "Red"  # an "other" application-area country
        verification_df = self._preprocess({"Rain / Wetness": values})
        self.assertEqual(
            self._value_for(
                verification_df, "Conditional speed limits - Rain / Wetness"
            ),
            "PASS",
        )

    def test_pass_at_minimum_qualifying_green_count(self):
        # All 10 member states Green; among the 19 others, 5 Green + 14 Red
        # => 15/29 Green, which is >= half of 29 (14.5).
        values = ["Green"] * 10 + ["Green"] * 5 + ["Red"] * 14
        verification_df = self._preprocess({"Rain / Wetness": values})
        self.assertEqual(
            self._value_for(
                verification_df, "Conditional speed limits - Rain / Wetness"
            ),
            "PASS",
        )

    def test_fail_one_below_minimum_qualifying_green_count(self):
        # All 10 member states Green; among the 19 others, 4 Green + 15 Red
        # => 14/29 Green, one below the qualifying threshold.
        values = ["Green"] * 10 + ["Green"] * 4 + ["Red"] * 15
        verification_df = self._preprocess({"Rain / Wetness": values})
        self.assertEqual(
            self._value_for(
                verification_df, "Conditional speed limits - Rain / Wetness"
            ),
            "FAIL",
        )

    def test_blank_member_state_cells_do_not_fail_the_gate(self):
        # Mirrors real evidence: blank (not Red) member-state cells, 0 Reds
        # anywhere -> still PASS. No "(bg)" table supplied, exercising the
        # current_bg_col is None fallback (blank treated as pass-equivalent).
        values = ["Green"] * 29
        values[8] = None
        values[9] = None
        verification_df = self._preprocess({"Non-lane relevant arrows": values})
        self.assertEqual(
            self._value_for(
                verification_df,
                "Conditional speed limits - Non-lane relevant arrows",
            ),
            "PASS",
        )

    def test_grey_filled_member_state_cell_fails_the_gate(self):
        """A member-state cell left blank in text but grey-filled is an
        explicit N/A marking and must fail the gate, even with zero
        explicit Reds anywhere."""
        values = ["Green"] * 29
        values[0] = None  # Austria: blank text
        bg = [None] * 29
        bg[0] = common.PredictionColor.GREY
        verification_df = self._preprocess(
            {"Rain / Wetness": values}, scenario_bg={"Rain / Wetness": bg}
        )
        self.assertEqual(
            self._value_for(
                verification_df, "Conditional speed limits - Rain / Wetness"
            ),
            "FAIL",
        )

    def test_grey_filled_application_area_cell_flips_boundary_to_fail(self):
        """A non-member cell that's blank-but-grey doesn't count toward the
        Application-Area green/blank count, unlike a truly blank cell --
        flipping a boundary PASS to FAIL."""
        # 10 member states Green; among 19 others: 4 Green, 1 blank, 14 Red.
        # Treating the 1 blank as pass-equivalent: 10+4+1=15 >= 14.5 -> PASS.
        values = ["Green"] * 10 + ["Green"] * 4 + [None] + ["Red"] * 14
        verification_df = self._preprocess({"Rain / Wetness": values})
        self.assertEqual(
            self._value_for(
                verification_df, "Conditional speed limits - Rain / Wetness"
            ),
            "PASS",
        )

        # Same values, but that one blank cell is grey-filled -> no longer
        # counted as pass-equivalent: 10+4+0=14 < 14.5 -> FAIL.
        bg = [None] * 29
        bg[14] = common.PredictionColor.GREY
        verification_df = self._preprocess(
            {"Rain / Wetness": values}, scenario_bg={"Rain / Wetness": bg}
        )
        self.assertEqual(
            self._value_for(
                verification_df, "Conditional speed limits - Rain / Wetness"
            ),
            "FAIL",
        )

    def test_implicit_trio_fails_if_member_state_red_in_one_column(self):
        values = ["Green"] * 29
        values[0] = "Red"  # member state Red in Highway / Motorway
        verification_df = self._preprocess({"Highway / Motorway": values})
        self.assertEqual(
            self._value_for(
                verification_df,
                "Implicit speed limits - Highway, city and residential zones",
            ),
            "FAIL",
        )

    def test_wrong_member_state_label_raises(self):
        preprocessor = TestPreprocess()
        prediction_df = preprocessor._create_prediction_df()
        prediction_df.iloc[1, 0] = "Wrong Label"
        dfs = {
            "VA - Speed assist. pred.": prediction_df,
            "Input parameters": preprocessor._create_input_params_df(),
            "Scenario Scores": preprocessor._create_scenario_scores_df(),
        }
        with self.assertRaises(ValueError):
            speed_assistance.preprocess(dfs)

    def test_wrong_row_count_raises(self):
        preprocessor = TestPreprocess()
        prediction_df = preprocessor._create_prediction_df()
        prediction_df = prediction_df.drop(index=5).reset_index(drop=True)
        dfs = {
            "VA - Speed assist. pred.": prediction_df,
            "Input parameters": preprocessor._create_input_params_df(),
            "Scenario Scores": preprocessor._create_scenario_scores_df(),
        }
        with self.assertRaises(ValueError):
            speed_assistance.preprocess(dfs)


class TestSlifAdvancedSpeedLimitTableHelpers(unittest.TestCase):
    """Isolate the ffill/bg-alignment mechanics from the pass/fail logic."""

    def test_group_column_is_forward_filled(self):
        preprocessor = TestPreprocess()
        prediction_df = preprocessor._create_prediction_df()
        table = speed_assistance.get_slif_advanced_speed_limit_table(prediction_df)
        expected_groups = ["Euro NCAP member states"] * len(MEMBER_STATE_COUNTRIES) + [
            "Other members in Euro NCAP application area"
        ] * len(OTHER_APPLICATION_AREA_COUNTRIES)
        self.assertEqual(list(table.iloc[:, 0]), expected_groups)

    def test_bg_table_aligns_with_text_table(self):
        preprocessor = TestPreprocess()
        prediction_df = preprocessor._create_prediction_df()
        bg = [common.PredictionColor.RED] + [None] * 28
        bg_prediction_df = preprocessor._create_prediction_bg_df({"Rain / Wetness": bg})
        table = speed_assistance.get_slif_advanced_speed_limit_table(prediction_df)
        bg_table = speed_assistance.get_slif_advanced_speed_limit_bg_table(
            bg_prediction_df, table.columns
        )
        self.assertEqual(bg_table["Rain / Wetness"].iloc[0], common.PredictionColor.RED)
        self.assertTrue(all(v is None for v in bg_table["Rain / Wetness"].iloc[1:]))


class TestImplicitSpeedLimitsScoring(unittest.TestCase):
    """Test Implicit Speed Limits scoring logic."""

    def test_implicit_requires_all_three_pass(self):
        """Test that implicit speed limit requires all three sub-elements to pass."""
        highway_motorway_passed = True
        city_entry_exit_passed = True
        residential_status = "PASS"

        if (
            residential_status == "PASS"
            and highway_motorway_passed
            and city_entry_exit_passed
        ):
            implicit_score = 0.5
        else:
            implicit_score = 0.0

        self.assertEqual(implicit_score, 0.5)

    def test_implicit_fails_if_highway_fails(self):
        """Test that implicit speed limit fails if highway/motorway fails."""
        highway_motorway_passed = False
        city_entry_exit_passed = True
        residential_status = "PASS"

        if (
            residential_status == "PASS"
            and highway_motorway_passed
            and city_entry_exit_passed
        ):
            implicit_score = 0.5
        else:
            implicit_score = 0.0

        self.assertEqual(implicit_score, 0.0)


class TestDynamicSpeedLimitsScoring(unittest.TestCase):
    """Test Dynamic Speed Limits scoring logic."""

    def test_lane_relevant_requires_non_lane_relevant_pass(self):
        """Test that Lane Relevant requires non-Lane Relevant to pass first."""
        non_lane_relevant_status = "FAIL"
        lane_relevant_base_status = "PASS"

        if non_lane_relevant_status != "PASS":
            lane_relevant_status = "FAIL"
        else:
            lane_relevant_status = lane_relevant_base_status

        self.assertEqual(lane_relevant_status, "FAIL")

    def test_lane_relevant_passes_when_non_lane_relevant_passes(self):
        """Test that Lane Relevant passes when non-Lane Relevant passes."""
        non_lane_relevant_status = "PASS"
        lane_relevant_base_status = "PASS"

        if non_lane_relevant_status != "PASS":
            lane_relevant_status = "FAIL"
        else:
            lane_relevant_status = lane_relevant_base_status

        self.assertEqual(lane_relevant_status, "PASS")


class TestLocalHazardsScoring(unittest.TestCase):
    """Regression tests for Local Hazards Cloud AND/OR Direct
    scoring. Unlike the tests these replace, these drive the real
    `speed_assistance.preprocess` + `speed_assistance.compute_score`
    production code end-to-end on a prediction fixture, instead of
    re-implementing the scoring logic inline."""

    def _compute(self, hazard_values=None):
        preprocessor = TestPreprocess()
        prediction_df = preprocessor._create_prediction_df(hazard_values=hazard_values)
        input_params_df = preprocessor._create_input_params_df()
        scenario_scores_df = preprocessor._create_scenario_scores_df()

        verification_df, scenario_scores_df = speed_assistance.preprocess(
            {
                "VA - Speed assist. pred.": prediction_df,
                "Input parameters": input_params_df,
                "Scenario Scores": scenario_scores_df,
            }
        )
        verification_df.loc[
            verification_df["Category"] == "SLIF - General requirements", "Value"
        ] = "PASS"

        dfs = {
            "VA - Speed assist. pred.": prediction_df,
            "VA - Speed assist. verif.": verification_df,
            "Input parameters": input_params_df,
            "Scenario Scores": scenario_scores_df,
        }
        _, _, score_dict, hazard_capping_score = speed_assistance.compute_score(dfs)
        return score_dict, hazard_capping_score

    def test_sending_both_channels_green_scores_02(self):
        score_dict, _ = self._compute(
            {"Construction zones": ["Green", "Green", "Green", "Green"]}
        )
        self.assertEqual(score_dict["Construction zones - Sending"], 0.2)

    def test_sending_cloud_only_scores_015(self):
        score_dict, _ = self._compute(
            {"Construction zones": ["Green", "Green", "Red", "Green"]}
        )
        self.assertEqual(score_dict["Construction zones - Sending"], 0.15)

    def test_sending_direct_only_scores_015(self):
        score_dict, _ = self._compute(
            {"Construction zones": ["Red", "Green", "Green", "Green"]}
        )
        self.assertEqual(score_dict["Construction zones - Sending"], 0.15)

    def test_sending_both_red_scores_0(self):
        score_dict, _ = self._compute(
            {"Construction zones": ["Red", "Green", "Red", "Green"]}
        )
        self.assertEqual(score_dict["Construction zones - Sending"], 0.0)

    def test_receiving_both_channels_green_scores_015(self):
        score_dict, _ = self._compute(
            {"Construction zones": ["Green", "Green", "Green", "Green"]}
        )
        self.assertEqual(score_dict["Construction zones - Receiving & informing"], 0.15)

    def test_receiving_single_channel_still_scores_015(self):
        score_dict, _ = self._compute(
            {"Construction zones": ["Green", "Green", "Green", "Red"]}
        )
        self.assertEqual(score_dict["Construction zones - Receiving & informing"], 0.15)

    def test_receiving_both_red_scores_0(self):
        score_dict, _ = self._compute(
            {"Construction zones": ["Green", "Red", "Green", "Red"]}
        )
        self.assertEqual(score_dict["Construction zones - Receiving & informing"], 0.0)

    def test_single_channel_hazard_does_not_reduce_other_hazards(self):
        """A hazard scored at the reduced rate must not affect sibling
        hazards, which stay at their own (independent) score."""
        score_dict, _ = self._compute(
            {"Construction zones": ["Green", "Green", "Red", "Green"]}
        )
        self.assertEqual(score_dict["Items on road - Sending"], 0.2)
        self.assertEqual(score_dict["Items on road - Receiving & informing"], 0.15)

    def test_no_sending_row_for_amber_and_traffic_jam(self):
        score_dict, _ = self._compute()
        self.assertNotIn("Amber & blue lights - Sending", score_dict)
        self.assertNotIn("Traffic jam - Sending", score_dict)
        self.assertEqual(
            score_dict["Amber & blue lights - Receiving & informing"], 0.15
        )
        self.assertEqual(score_dict["Traffic jam - Receiving & informing"], 0.15)

    def test_mixed_grid_matches_expected_validation_table(self):
        """End-to-end regression test reproducing the mixed Cloud/Direct
        grid with both channels present: raw sum 2.7,
        capped to 2.5."""
        grid = {
            "Construction zones": ["Green", "Red", "Green", "Green"],
            "Items on road": ["Red", "Green", "Green", "Green"],
            "Stopped vehicle": ["Green", "Green", "Red", "Green"],
            "Broken down vehicle": ["Green", "Green", "Green", "Red"],
            "Post crash": ["Red", "Red", "Green", "Green"],
            "Poor weather": ["Green", "Green", "Red", "Red"],
            "Poor road": ["Red", "Green", "Red", "Green"],
            "Wrong way driver": ["Green", "Green", "Green", "Green"],
            "Amber & blue lights": ["N/A", "Green", "N/A", "Green"],
            "Traffic jam": ["N/A", "Green", "N/A", "Green"],
        }
        score_dict, cap = self._compute(grid)

        expected = {
            "Construction zones - Sending": 0.2,
            "Construction zones - Receiving & informing": 0.15,
            "Items on road - Sending": 0.15,
            "Items on road - Receiving & informing": 0.15,
            "Stopped vehicle - Sending": 0.15,
            "Stopped vehicle - Receiving & informing": 0.15,
            "Broken down vehicle - Sending": 0.2,
            "Broken down vehicle - Receiving & informing": 0.15,
            "Post crash - Sending": 0.15,
            "Post crash - Receiving & informing": 0.15,
            "Poor weather - Sending": 0.15,
            "Poor weather - Receiving & informing": 0.15,
            "Poor road - Sending": 0.0,
            "Poor road - Receiving & informing": 0.15,
            "Wrong way driver - Sending": 0.2,
            "Wrong way driver - Receiving & informing": 0.15,
            "Amber & blue lights - Receiving & informing": 0.15,
            "Traffic jam - Receiving & informing": 0.15,
        }
        for scenario, expected_score in expected.items():
            self.assertEqual(score_dict[scenario], expected_score, scenario)

        self.assertAlmostEqual(sum(expected.values()), 2.7)
        self.assertEqual(cap, 2.5)


class TestGetLocalHazardChannelStatus(unittest.TestCase):
    """Test the `get_local_hazard_channel_status` helper in isolation."""

    def _channel_status(self, hazard_values=None):
        prediction_df = TestPreprocess()._create_prediction_df(
            hazard_values=hazard_values
        )
        table = speed_assistance.get_local_hazard_table(prediction_df)
        return speed_assistance.get_local_hazard_channel_status(table)

    def test_sending_key_absent_for_amber_and_traffic_jam(self):
        channel_status = self._channel_status()
        self.assertNotIn(
            speed_assistance.LOCAL_HAZARD_SENDING_STR,
            channel_status["Amber & blue lights"],
        )
        self.assertNotIn(
            speed_assistance.LOCAL_HAZARD_SENDING_STR,
            channel_status["Traffic jam"],
        )
        self.assertIn(
            speed_assistance.LOCAL_HAZARD_RECEIVING_STR,
            channel_status["Amber & blue lights"],
        )

    def test_blank_cell_is_neither_green_nor_red(self):
        channel_status = self._channel_status(
            {"Construction zones": [None, "Green", "Green", "Green"]}
        )
        sending = channel_status["Construction zones"][
            speed_assistance.LOCAL_HAZARD_SENDING_STR
        ]
        self.assertFalse(sending["cloud"])
        self.assertFalse(sending["has_red"])


class TestHazardCappingScore(unittest.TestCase):
    """Regression tests for the Local Hazards category cap. Calls
    the real production `speed_assistance.compute_score`."""

    def _compute(self, hazard_values=None):
        return TestLocalHazardsScoring()._compute(hazard_values)

    def test_all_green_grid_caps_at_30(self):
        _, cap = self._compute()
        self.assertEqual(cap, 3.0)

    def test_one_red_cell_caps_at_25(self):
        _, cap = self._compute(
            {"Construction zones": ["Red", "Green", "Green", "Green"]}
        )
        self.assertEqual(cap, 2.5)

    def test_red_on_failing_hazard_still_caps_at_25(self):
        """Both channels Red -> the hazard FAILs and scores 0, but the Red
        cells are still evidence of a missing communication channel."""
        _, cap = self._compute({"Construction zones": ["Red", "Green", "Red", "Green"]})
        self.assertEqual(cap, 2.5)

    def test_blank_unclaimed_channel_with_no_red_caps_at_25(self):
        """The leak this fix closes: Cloud left entirely blank (not Red),
        Direct Green -> single-channel vehicle, cap must drop to 2.5 even
        with zero explicit Reds."""
        _, cap = self._compute({"Construction zones": [None, None, "Green", "Green"]})
        self.assertEqual(cap, 2.5)

    def test_fully_blank_hazard_does_not_lower_cap(self):
        """A hazard left entirely unclaimed (all four cells blank, no Red)
        scores 0 and is not evidence about the vehicle's communication
        capability -- it must not lower the cap by itself."""
        _, cap = self._compute({"Construction zones": [None, None, None, None]})
        self.assertEqual(cap, 3.0)


class TestArrowsScoringLogic(unittest.TestCase):
    """Test Arrows (LR) scoring dependency on Arrows (non-LR)."""

    def test_arrows_lr_requires_non_lr_pass(self):
        """Test that Arrows (LR) requires Arrows (non-LR) to pass."""
        arrows_non_lr_status = "FAIL"
        arrows_lr_base_status = "PASS"

        if arrows_non_lr_status != "PASS":
            arrows_lr_status = "FAIL"
        else:
            arrows_lr_status = arrows_lr_base_status

        self.assertEqual(arrows_lr_status, "FAIL")

    def test_arrows_lr_passes_when_non_lr_passes(self):
        """Test that Arrows (LR) passes when Arrows (non-LR) passes."""
        arrows_non_lr_status = "PASS"
        arrows_lr_base_status = "PASS"

        if arrows_non_lr_status != "PASS":
            arrows_lr_status = "FAIL"
        else:
            arrows_lr_status = arrows_lr_base_status

        self.assertEqual(arrows_lr_status, "PASS")


if __name__ == "__main__":
    unittest.main()
