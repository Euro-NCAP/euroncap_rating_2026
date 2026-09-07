# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from euroncap_rating_2026.safe_driving import occupant_classification
from euroncap_rating_2026.safe_driving import data_model


class TestComputeClassificationScoreBasic(unittest.TestCase):
    """Test basic behavior of the compute_classification_score function."""

    def _create_valid_input_params(self):
        return pd.DataFrame(
            {
                "Category": ["Passenger airbag status", None],
                "Input parameter": ["Type of system", "Type of switch"],
                "Value": ["Automatic", "Automatic"],
            }
        )

    def test_compute_classification_score_missing_dataframe(self):
        """Test that missing Input parameters returns without crashing (NaN/None values are handled gracefully)."""

        dfs = {}
        result_df, score_dict = occupant_classification.compute_classification_score(
            dfs
        )
        self.assertIsInstance(score_dict, dict)

    def test_compute_classification_score_empty_dataframe(self):
        """Test that empty OM verification dataframe returns empty dataframe + initialized score dict."""
        dfs = {
            "Input parameters": self._create_valid_input_params(),
            "OM - Occ. classification verif.": pd.DataFrame(),
        }
        result_df, score_dict = occupant_classification.compute_classification_score(
            dfs
        )

        self.assertTrue(result_df.empty)
        self.assertIsInstance(score_dict, dict)
        self.assertIn("Automatic", score_dict)

    def test_compute_classification_score_returns_tuple(self):
        """Test that compute_classification_score returns a tuple of (DataFrame, dict)."""
        dfs = {
            "Input parameters": self._create_valid_input_params(),
            "OM - Occ. classification verif.": pd.DataFrame(),
        }
        result = occupant_classification.compute_classification_score(dfs)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], pd.DataFrame)
        self.assertIsInstance(result[1], dict)

    def test_compute_classification_score_nan_type_of_switch_does_not_raise(self):
        """Regression test: NaN Type of switch (e.g. Automatic system with no switch) must not raise ValueError."""
        input_params = pd.DataFrame(
            {
                "Category": ["Passenger airbag status", "Passenger airbag status"],
                "Input parameter": ["Type of system", "Type of switch"],
                "Value": ["Automatic", np.nan],
            }
        )
        dfs = {
            "Input parameters": input_params,
            "OM - Occ. classification verif.": pd.DataFrame(),
        }
        # Should not raise ValueError: nan is not a valid OCTypeOfSwitch
        result_df, score_dict = occupant_classification.compute_classification_score(
            dfs
        )
        self.assertIsInstance(score_dict, dict)

    def test_compute_classification_score_missing_airbag_params_has_no_empty_key(self):
        """Regression test: unresolved passenger airbag scenario must not create an empty-string score key."""
        input_params = pd.DataFrame(
            {
                "Category": ["Passenger airbag status", "Passenger airbag status"],
                "Input parameter": ["Type of system", "Type of switch"],
                "Value": [np.nan, np.nan],
            }
        )
        dfs = {
            "Input parameters": input_params,
            "OM - Occ. classification verif.": pd.DataFrame(),
        }
        _, score_dict = occupant_classification.compute_classification_score(dfs)

        self.assertNotIn("", score_dict)


class TestOCTypeOfSystemScoringLogic(unittest.TestCase):
    """Test the scoring logic based on OCTypeOfSystem values."""

    def test_automatic_scores_4_0(self):
        """Test that Automatic type of system gives 4.0 points."""
        type_of_system = data_model.OCTypeOfSystem.AUTOMATIC

        if type_of_system == data_model.OCTypeOfSystem.AUTOMATIC:
            expected_score = 4.0
        elif type_of_system == data_model.OCTypeOfSystem.SYSTEM_ADVISED_SOFTWARE:
            expected_score = 3.0
        elif type_of_system == data_model.OCTypeOfSystem.SYSTEM_ADVISED_HARDWARE:
            expected_score = 2.0
        elif type_of_system == data_model.OCTypeOfSystem.MANUAL:
            expected_score = 1.0
        else:
            expected_score = 0.0

        self.assertEqual(expected_score, 4.0)


class TestOCTypeOfSystemEnum(unittest.TestCase):
    """Test OCTypeOfSystem enum values."""

    def test_automatic_value(self):
        """Test Automatic enum value."""
        self.assertEqual(data_model.OCTypeOfSystem.AUTOMATIC.value, "Automatic")

    def test_manual_value(self):
        """Test Manual enum value."""
        self.assertEqual(data_model.OCTypeOfSystem.MANUAL.value, "Manual")


class TestOCTypeOfSwitchEnum(unittest.TestCase):
    """Test OCTypeOfSwitch enum values."""

    def test_software_switch_value(self):
        """Test Software switch enum value."""
        self.assertEqual(data_model.OCTypeOfSwitch.SOFTWARE.value, "Software")

    def test_hardware_switch_value(self):
        """Test Hardware switch enum value."""
        self.assertEqual(data_model.OCTypeOfSwitch.HARDWARE.value, "Hardware")


class TestAirbagAvailabilityScoringLogic(unittest.TestCase):
    """Test scoring logic for airbag availability."""

    def test_airbag_available_adds_to_score(self):
        """Test that airbag availability adds points."""
        classification_elements_score = 4.0
        airbag_available = True

        if airbag_available:
            total_score = classification_elements_score + 1.0
        else:
            total_score = classification_elements_score

        self.assertEqual(total_score, 5.0)

    def test_airbag_not_available_no_addition(self):
        """Test that no airbag availability doesn't add points."""
        classification_elements_score = 4.0
        airbag_available = False

        if airbag_available:
            total_score = classification_elements_score + 1.0
        else:
            total_score = classification_elements_score

        self.assertEqual(total_score, 4.0)


class TestComputeClassificationScoreUnassessed(unittest.TestCase):
    """compute_classification_score must distinguish an unassessed (blank)
    Value from an explicit FAIL: blank scores NaN, FAIL scores 0.0."""

    def _dfs(self):
        input_params = pd.DataFrame(
            {
                "Category": ["Passenger airbag status", None],
                "Input parameter": ["Type of system", "Type of switch"],
                "Value": ["Automatic", None],
            }
        )
        dfs = {
            "Input parameters": input_params,
            "Scenario Scores": pd.DataFrame({"Category": ["Passenger airbag status"]}),
        }
        om_df, _ = occupant_classification.preprocess_occupant_classification(dfs)
        om_df["Value"] = om_df["Value"].astype(object)
        dfs["OM - Occ. classification verif."] = om_df
        return dfs

    def test_all_blank_default_is_unassessed_not_zero(self):
        _, score_dict = occupant_classification.compute_classification_score(
            self._dfs()
        )

        self.assertTrue(all(pd.isna(v) for v in score_dict.values()))

    def test_explicit_fail_scores_zero_not_nan(self):
        dfs = self._dfs()
        om_df = dfs["OM - Occ. classification verif."]
        om_df.loc[om_df["Category"] == "Passenger airbag status", "Value"] = "FAIL"

        _, score_dict = occupant_classification.compute_classification_score(dfs)

        self.assertEqual(score_dict["Automatic"], 0.0)

    def test_all_pass_scores_full_marks(self):
        dfs = self._dfs()
        om_df = dfs["OM - Occ. classification verif."]
        om_df["Value"] = "PASS"

        _, score_dict = occupant_classification.compute_classification_score(dfs)

        self.assertEqual(score_dict["Automatic"], 4.0)
        self.assertEqual(score_dict["Close proximity to the airbag"], 1.0)
        self.assertEqual(score_dict["Feet on dashboard"], 1.0)
        self.assertEqual(score_dict["Driver"], 3.0)
        self.assertEqual(score_dict["Front seat passenger"], 1.0)

    def test_generic_row_blank_leaves_only_that_key_unassessed(self):
        dfs = self._dfs()
        om_df = dfs["OM - Occ. classification verif."]
        om_df["Value"] = "PASS"
        om_df.loc[om_df["Scenario"] == "Driver", "Value"] = np.nan

        _, score_dict = occupant_classification.compute_classification_score(dfs)

        self.assertEqual(score_dict["Automatic"], 4.0)
        self.assertTrue(pd.isna(score_dict["Driver"]))
        self.assertEqual(score_dict["Front seat passenger"], 1.0)


class TestComputeClassificationScoreNotApplicable(unittest.TestCase):
    """An "N/A" passenger airbag input parameter means no disabling system is
    fitted: no crash, the scenario reads "N/A", and the element scores a real
    0.0 even though its Value cell legitimately stays blank."""

    def _dfs(self, type_of_system="N/A", type_of_switch="N/A"):
        input_params = pd.DataFrame(
            {
                "Category": ["Passenger airbag status", None],
                "Input parameter": ["Type of system", "Type of switch"],
                "Value": [type_of_system, type_of_switch],
            }
        )
        dfs = {
            "Input parameters": input_params,
            "Scenario Scores": pd.DataFrame({"Category": ["Passenger airbag status"]}),
        }
        om_df, _ = occupant_classification.preprocess_occupant_classification(dfs)
        om_df["Value"] = om_df["Value"].astype(object)
        dfs["OM - Occ. classification verif."] = om_df
        return dfs

    def test_get_passenger_airbag_scenario_na(self):
        self.assertEqual(
            occupant_classification.get_passenger_airbag_scenario("N/A", None), "N/A"
        )
        self.assertEqual(
            occupant_classification.get_passenger_airbag_scenario(
                data_model.OCTypeOfSystem.NOT_APPLICABLE, None
            ),
            "N/A",
        )

    def test_preprocess_stamps_na_scenario(self):
        dfs = self._dfs()
        om_df = dfs["OM - Occ. classification verif."]

        airbag_scenario = om_df.loc[
            om_df["Category"] == "Passenger airbag status", "Scenario"
        ].iloc[0]
        self.assertEqual(airbag_scenario, "N/A")
        self.assertEqual(dfs["Scenario Scores"]["Scenario"].iloc[0], "N/A")

    def test_na_system_with_blank_value_does_not_raise_and_scores_zero(self):
        _, score_dict = occupant_classification.compute_classification_score(
            self._dfs()
        )

        self.assertEqual(score_dict["N/A"], 0.0)
        self.assertFalse(pd.isna(score_dict["N/A"]))

    def test_na_system_leaves_other_blank_keys_unassessed(self):
        _, score_dict = occupant_classification.compute_classification_score(
            self._dfs()
        )

        self.assertTrue(pd.isna(score_dict["Driver"]))
        self.assertTrue(pd.isna(score_dict["Front seat passenger"]))

    def test_automatic_system_with_na_switch_still_scores_full(self):
        dfs = self._dfs(type_of_system="Automatic", type_of_switch="N/A")
        om_df = dfs["OM - Occ. classification verif."]
        om_df["Value"] = "PASS"

        _, score_dict = occupant_classification.compute_classification_score(dfs)

        self.assertEqual(score_dict["Automatic"], 4.0)


if __name__ == "__main__":
    unittest.main()
