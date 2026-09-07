# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from euroncap_rating_2026.safe_driving import general_vehicle_controls


class TestComputeScore(unittest.TestCase):
    """Test cases for the compute_score function."""

    def _create_gvc_verification_df(self, rows):
        """Helper to create a GVC verification DataFrame from row specifications."""
        return pd.DataFrame(rows)

    def test_compute_score_returns_dict(self):
        """Test that compute_score returns a dictionary."""
        dfs = {"DE - GVC verif.": pd.DataFrame()}
        result = general_vehicle_controls.compute_score(dfs)
        self.assertIsInstance(result, dict)

    def test_compute_score_empty_dataframe(self):
        """An empty verif. sheet means nothing has been assessed yet, so
        every scenario must be unassessed (NaN), not a false 0."""
        dfs = {"DE - GVC verif.": pd.DataFrame()}
        result = general_vehicle_controls.compute_score(dfs)

        expected_scenarios = [
            "Driving controls",
            "Vision",
            "Lights",
            "ADAS",
            "Audio entertainment",
            "Calling & dialling",
            "Navigation system",
            "Climate controls",
            "Windows",
            "Other",
        ]
        for scenario in expected_scenarios:
            self.assertTrue(pd.isna(result[scenario]))

    def test_compute_score_missing_dataframe(self):
        """Test scoring when dataframe is missing from dfs."""
        dfs = {}
        result = general_vehicle_controls.compute_score(dfs)

        # Missing sheet means nothing assessed -- unassessed (NaN), not 0.
        for value in result.values():
            self.assertTrue(pd.isna(value))

    def test_compute_score_all_pass(self):
        """Test scoring when all scenarios pass."""
        df = self._create_gvc_verification_df(
            [
                {"Scenario": "Driving controls", "Element": "E1", "Value": "PASS"},
                {"Scenario": "Driving controls", "Element": "E2", "Value": "PASS"},
                {"Scenario": "Vision", "Element": "E1", "Value": "PASS"},
                {"Scenario": "Lights", "Element": "E1", "Value": "PASS"},
                {"Scenario": "ADAS", "Element": "E1", "Value": "PASS"},
                {"Scenario": "Audio entertainment", "Element": "E1", "Value": "PASS"},
                {"Scenario": "Calling & dialling", "Element": "E1", "Value": "PASS"},
                {"Scenario": "Navigation system", "Element": "E1", "Value": "PASS"},
                {"Scenario": "Climate controls", "Element": "E1", "Value": "PASS"},
                {"Scenario": "Windows", "Element": "E1", "Value": "PASS"},
                {"Scenario": "Other", "Element": "E1", "Value": "PASS"},
            ]
        )
        dfs = {"DE - GVC verif.": df}

        with patch.object(
            general_vehicle_controls.common, "extract_section"
        ) as mock_extract:
            # Setup mock to return appropriate dataframes
            def extract_side_effect(df, scenario, col_name=None):
                scenario_df = df[df["Scenario"] == scenario]
                return scenario_df

            mock_extract.side_effect = extract_side_effect

            result = general_vehicle_controls.compute_score(dfs)

        # Check expected max scores
        self.assertEqual(result["Driving controls"], 1.00)
        self.assertEqual(result["Vision"], 0.50)
        self.assertEqual(result["Lights"], 0.50)
        self.assertEqual(result["ADAS"], 0.50)
        self.assertEqual(result["Audio entertainment"], 0.50)
        self.assertEqual(result["Calling & dialling"], 0.50)
        self.assertEqual(result["Navigation system"], 0.50)
        self.assertEqual(result["Climate controls"], 0.50)
        self.assertEqual(result["Windows"], 0.25)
        self.assertEqual(result["Other"], 0.25)

    def test_compute_score_driving_controls_fail(self):
        """Test scoring when Driving controls scenario fails."""
        df = self._create_gvc_verification_df(
            [
                {"Scenario": "Driving controls", "Element": "E1", "Value": "FAIL"},
            ]
        )
        dfs = {"DE - GVC verif.": df}

        with patch.object(
            general_vehicle_controls.common, "extract_section"
        ) as mock_extract:

            def extract_side_effect(df, scenario, col_name=None):
                if scenario == "Driving controls":
                    return pd.DataFrame(
                        [
                            {
                                "Scenario": "Driving controls",
                                "Element": "E1",
                                "Value": "FAIL",
                            }
                        ]
                    )
                return pd.DataFrame()

            mock_extract.side_effect = extract_side_effect

            result = general_vehicle_controls.compute_score(dfs)

        self.assertEqual(result["Driving controls"], 0)

    def test_compute_score_mixed_pass_fail(self):
        """Test scoring with mixed pass/fail results."""
        df = self._create_gvc_verification_df(
            [
                {"Scenario": "Driving controls", "Element": "E1", "Value": "PASS"},
                {"Scenario": "Vision", "Element": "E1", "Value": "FAIL"},
            ]
        )
        dfs = {"DE - GVC verif.": df}

        with patch.object(
            general_vehicle_controls.common, "extract_section"
        ) as mock_extract:

            def extract_side_effect(df, scenario, col_name=None):
                if scenario == "Driving controls":
                    return pd.DataFrame(
                        [
                            {
                                "Scenario": "Driving controls",
                                "Element": "E1",
                                "Value": "PASS",
                            }
                        ]
                    )
                elif scenario == "Vision":
                    return pd.DataFrame(
                        [{"Scenario": "Vision", "Element": "E1", "Value": "FAIL"}]
                    )
                return pd.DataFrame()

            mock_extract.side_effect = extract_side_effect

            result = general_vehicle_controls.compute_score(dfs)

        self.assertEqual(result["Driving controls"], 1.00)
        self.assertEqual(result["Vision"], 0)

    def test_compute_score_case_insensitive_pass(self):
        """Test that PASS matching is case-insensitive."""
        test_values = ["pass", "PASS", "Pass", "pAsS"]

        for value in test_values:
            with self.subTest(value=value):
                df = self._create_gvc_verification_df(
                    [
                        {
                            "Scenario": "Driving controls",
                            "Element": "E1",
                            "Value": value,
                        },
                    ]
                )
                dfs = {"DE - GVC verif.": df}

                with patch.object(
                    general_vehicle_controls.common, "extract_section"
                ) as mock_extract:
                    mock_extract.return_value = pd.DataFrame(
                        [
                            {
                                "Scenario": "Driving controls",
                                "Element": "E1",
                                "Value": value,
                            }
                        ]
                    )
                    result = general_vehicle_controls.compute_score(dfs)

                self.assertEqual(result["Driving controls"], 1.00)

    def test_compute_score_one_fail_in_scenario(self):
        """Test that one FAIL in a scenario causes the whole scenario to fail."""
        df = self._create_gvc_verification_df(
            [
                {"Scenario": "Driving controls", "Element": "E1", "Value": "PASS"},
                {"Scenario": "Driving controls", "Element": "E2", "Value": "PASS"},
                {"Scenario": "Driving controls", "Element": "E3", "Value": "FAIL"},
            ]
        )
        dfs = {"DE - GVC verif.": df}

        with patch.object(
            general_vehicle_controls.common, "extract_section"
        ) as mock_extract:
            mock_extract.return_value = df[df["Scenario"] == "Driving controls"]
            result = general_vehicle_controls.compute_score(dfs)

        self.assertEqual(result["Driving controls"], 0)

    def test_compute_score_blank_and_fail_mixed_stays_unassessed(self):
        """A scenario with one blank row and one FAIL row must stay
        unassessed (NaN), not collapse to a FAIL (0) -- the OEM still hasn't
        finished assessing it."""
        df = self._create_gvc_verification_df(
            [
                {"Scenario": "Driving controls", "Element": "E1", "Value": np.nan},
                {"Scenario": "Driving controls", "Element": "E2", "Value": "FAIL"},
            ]
        )
        dfs = {"DE - GVC verif.": df}

        with patch.object(
            general_vehicle_controls.common, "extract_section"
        ) as mock_extract:
            mock_extract.return_value = df[df["Scenario"] == "Driving controls"]
            result = general_vehicle_controls.compute_score(dfs)

        self.assertTrue(pd.isna(result["Driving controls"]))

    def test_compute_score_all_scenarios_have_expected_keys(self):
        """Test that all expected scenario keys are present in result."""
        dfs = {"DE - GVC verif.": pd.DataFrame()}
        result = general_vehicle_controls.compute_score(dfs)

        expected_scenarios = [
            "Driving controls",
            "Vision",
            "Lights",
            "ADAS",
            "Audio entertainment",
            "Calling & dialling",
            "Navigation system",
            "Climate controls",
            "Windows",
            "Other",
        ]
        for scenario in expected_scenarios:
            self.assertIn(scenario, result)


class TestScenarioMaxScores(unittest.TestCase):
    """Test cases for verifying max scores per scenario."""

    def test_max_scores_values(self):
        """Test that max scores match expected values."""
        # Create a non-empty DataFrame to avoid early return
        dfs = {"DE - GVC verif.": pd.DataFrame([{"Scenario": "Test", "Value": "PASS"}])}

        with patch.object(
            general_vehicle_controls.common, "extract_section"
        ) as mock_extract:

            def extract_side_effect(df, scenario, col_name=None):
                return pd.DataFrame(
                    [{"Scenario": scenario, "Element": "E1", "Value": "PASS"}]
                )

            mock_extract.side_effect = extract_side_effect

            result = general_vehicle_controls.compute_score(dfs)

        expected_max_scores = {
            "Driving controls": 1.00,
            "Vision": 0.50,
            "Lights": 0.50,
            "ADAS": 0.50,
            "Audio entertainment": 0.50,
            "Calling & dialling": 0.50,
            "Navigation system": 0.50,
            "Climate controls": 0.50,
            "Windows": 0.25,
            "Other": 0.25,
        }

        for scenario, expected_score in expected_max_scores.items():
            self.assertEqual(
                result[scenario],
                expected_score,
                f"Max score for {scenario} should be {expected_score}",
            )


class TestEdgeCases(unittest.TestCase):
    """Test edge cases for compute_score function."""

    def test_scenario_with_nan_values(self):
        """A blank/NA Value means the OEM hasn't assessed this scenario yet
        -- it must stay unassessed (NaN), not be treated as a FAIL (0)."""
        df = pd.DataFrame(
            [
                {"Scenario": "Driving controls", "Element": "E1", "Value": pd.NA},
            ]
        )
        dfs = {"DE - GVC verif.": df}

        with patch.object(
            general_vehicle_controls.common, "extract_section"
        ) as mock_extract:
            mock_extract.return_value = df
            result = general_vehicle_controls.compute_score(dfs)

        self.assertTrue(pd.isna(result["Driving controls"]))

    def test_scenario_not_present_returns_unassessed(self):
        """Test that missing scenarios stay unassessed (NaN), not a false 0."""
        df = pd.DataFrame(
            [
                {"Scenario": "Driving controls", "Element": "E1", "Value": "PASS"},
            ]
        )
        dfs = {"DE - GVC verif.": df}

        with patch.object(
            general_vehicle_controls.common, "extract_section"
        ) as mock_extract:

            def extract_side_effect(df, scenario, col_name=None):
                if scenario == "Driving controls":
                    return pd.DataFrame(
                        [
                            {
                                "Scenario": "Driving controls",
                                "Element": "E1",
                                "Value": "PASS",
                            }
                        ]
                    )
                return pd.DataFrame()  # Empty for other scenarios

            mock_extract.side_effect = extract_side_effect

            result = general_vehicle_controls.compute_score(dfs)

        # Present scenario should score, missing should stay unassessed.
        self.assertEqual(result["Driving controls"], 1.00)
        self.assertTrue(pd.isna(result["Vision"]))


if __name__ == "__main__":
    unittest.main()
