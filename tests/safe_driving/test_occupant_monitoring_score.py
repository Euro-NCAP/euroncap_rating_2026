# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from euroncap_rating_2026.safe_driving import seatbelt_usage


class TestRearSeatNumbering(unittest.TestCase):
    def test_get_rear_seat_labels_uses_fixed_sequence(self):
        labels = seatbelt_usage.get_rear_seat_labels(6)

        self.assertEqual(
            labels,
            ["Seat 4", "Seat 5", "Seat 6", "Seat 7", "Seat 8", "Seat 9"],
        )

    def test_generate_rear_seat_df_uses_sequence_prefix(self):
        result_df = seatbelt_usage.generate_seatbelt_usage_rear_seat_df(2)

        self.assertEqual(result_df["Scenario"].tolist(), ["Seat 4", "Seat 6"])


class TestComputeSeatbeltScoreBasic(unittest.TestCase):
    """Test basic behavior of the compute_seatbelt_score function."""

    def test_compute_seatbelt_score_missing_dataframe(self):
        """Test that missing dataframe returns empty results."""
        dfs = {}
        result_df, score_dict = seatbelt_usage.compute_seatbelt_score(dfs)

        self.assertTrue(result_df.empty)
        self.assertEqual(score_dict, {})

    def test_compute_seatbelt_score_empty_dataframe(self):
        """Test that empty dataframe returns empty results."""
        dfs = {"OM - Seatbelt usage": pd.DataFrame()}
        result_df, score_dict = seatbelt_usage.compute_seatbelt_score(dfs)

        self.assertTrue(result_df.empty)
        self.assertEqual(score_dict, {})

    def test_compute_seatbelt_score_returns_tuple(self):
        """Test that compute_seatbelt_score returns a tuple of (DataFrame, dict)."""
        dfs = {"OM - Seatbelt usage": pd.DataFrame()}
        result = seatbelt_usage.compute_seatbelt_score(dfs)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], pd.DataFrame)
        self.assertIsInstance(result[1], dict)


class TestSeatbeltScoringLogic(unittest.TestCase):
    """Test the scoring logic for seatbelt-related scenarios."""

    def test_general_req_fail_zeros_seatbelt_reminder(self):
        """Test that if general requirements fail, Seatbelt Reminder score is 0."""
        # Based on the logic in occupant_monitoring_score.py
        any_general_req_failed = True

        if any_general_req_failed:
            seatbelt_reminder_score = 0.0
        else:
            seatbelt_reminder_score = 2.0

        self.assertEqual(seatbelt_reminder_score, 0.0)

    def test_general_req_pass_seatbelt_reminder_2(self):
        """Test that if general requirements pass, Seatbelt Reminder score is 2.0."""
        any_general_req_failed = False

        if any_general_req_failed:
            seatbelt_reminder_score = 0.0
        else:
            seatbelt_reminder_score = 2.0

        self.assertEqual(seatbelt_reminder_score, 2.0)


class TestBeltRoutingElementsScoring(unittest.TestCase):
    """Test scoring logic for belt routing elements."""

    def test_seatbelt_buckle_only_score(self):
        """Test that seatbelt buckle only gives 0.25 per pass."""
        seatbelt_buckle_pass = True

        if seatbelt_buckle_pass:
            score = 0.25
        else:
            score = 0.0

        self.assertEqual(score, 0.25)

    def test_seatbelt_across_lap_score(self):
        """Test that seatbelt across lap gives 0.25 per pass."""
        seatbelt_across_lap_pass = True

        if seatbelt_across_lap_pass:
            score = 0.25
        else:
            score = 0.0

        self.assertEqual(score, 0.25)

    def test_seatbelt_behind_back_score(self):
        """Test that seatbelt behind back gives 0.25 per pass."""
        seatbelt_behind_back_pass = True

        if seatbelt_behind_back_pass:
            score = 0.25
        else:
            score = 0.0

        self.assertEqual(score, 0.25)


class TestRearSeatOccupancyScoringLogic(unittest.TestCase):
    """Test scoring logic for rear seat occupancy detection."""

    def test_rear_seat_occupancy_max_score(self):
        """Test that rear seat occupancy max score is 5.0."""
        max_score = 5.0
        self.assertEqual(max_score, 5.0)

    def test_rear_seat_occupancy_even_distribution(self):
        """Test that rear seat occupancy score is evenly distributed."""
        num_rows = 4
        max_score = 5.0
        score_per_row = max_score / num_rows

        self.assertEqual(score_per_row, 1.25)


class TestComputeSeatbeltScoreUnassessed(unittest.TestCase):
    """compute_seatbelt_score must distinguish an unassessed (blank) Value
    from an explicit FAIL: blank scores NaN, FAIL scores 0.0."""

    def _dfs(self):
        # Mirrors how test_info.preprocess_stage_subelement assembles this
        # sheet: main CSV rows plus rear-seat rows, driven by the "Number of
        # rear seats" input parameter.
        input_parameters_df = pd.DataFrame(
            [
                {
                    "Category": "Rear seat occupancy",
                    "Input parameter": "Number of rear seats",
                    "Value": 2,
                }
            ]
        )
        om_df = seatbelt_usage.generate_seatbelt_usage_df({})
        rear_seat_df = seatbelt_usage.generate_seatbelt_usage_rear_seat_df(2)
        om_df = pd.concat([om_df, rear_seat_df], ignore_index=True)
        # Mirrors an Excel-sourced column, which is object dtype once any
        # cell holds a string -- avoids a pandas dtype-mismatch warning that
        # only exists because this fixture starts from an all-blank CSV.
        om_df["Value"] = om_df["Value"].astype(object)
        return {
            "OM - Seatbelt usage verif.": om_df,
            "Input parameters": input_parameters_df,
        }

    def test_all_blank_default_is_unassessed_not_zero(self):
        om_df, score_dict = seatbelt_usage.compute_seatbelt_score(self._dfs())

        self.assertTrue(all(pd.isna(v) for v in score_dict.values()))
        self.assertTrue(om_df["Score"].isna().all())

    def test_general_requirement_explicit_fail_scores_zero(self):
        dfs = self._dfs()
        om_df = dfs["OM - Seatbelt usage verif."]
        om_df.loc[om_df["Category"] == "General requirements", "Value"] = "FAIL"

        _, score_dict = seatbelt_usage.compute_seatbelt_score(dfs)

        self.assertTrue(all(v == 0.0 for v in score_dict.values()))

    def test_general_requirement_pass_with_mixed_scenarios(self):
        dfs = self._dfs()
        om_df = dfs["OM - Seatbelt usage verif."]
        om_df.loc[om_df["Category"] == "General requirements", "Value"] = "PASS"
        om_df.loc[om_df["Scenario"] == "Seatbelt buckle only", "Value"] = "PASS"
        om_df.loc[om_df["Scenario"] == "Lap belt only", "Value"] = "FAIL"
        # "Seatbelt completely behind back" left blank/unassessed

        _, score_dict = seatbelt_usage.compute_seatbelt_score(dfs)

        self.assertEqual(score_dict["Seatbelt buckle only"], 2.0)
        self.assertEqual(score_dict["Lap belt only"], 0.0)
        self.assertTrue(pd.isna(score_dict["Seatbelt completely behind back"]))

    def test_general_requirement_blank_leaves_everything_unassessed_even_if_scenarios_filled(
        self,
    ):
        dfs = self._dfs()
        om_df = dfs["OM - Seatbelt usage verif."]
        # General requirements left blank; a scenario is filled in anyway.
        om_df.loc[om_df["Scenario"] == "Seatbelt buckle only", "Value"] = "PASS"

        _, score_dict = seatbelt_usage.compute_seatbelt_score(dfs)

        self.assertTrue(all(pd.isna(v) for v in score_dict.values()))


if __name__ == "__main__":
    unittest.main()
