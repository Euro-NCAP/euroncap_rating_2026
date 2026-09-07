# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import unittest
import pandas as pd
from euroncap_rating_2026.safe_driving import steering_assistance


class TestPreprocess(unittest.TestCase):

    def test_preprocess_returns_dataframe_with_expected_columns(self):
        result = steering_assistance.preprocess({})
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("Category", result.columns)
        self.assertIn("Scenario", result.columns)
        self.assertIn("Value", result.columns)

    def test_preprocess_contains_s_bend_scenarios(self):
        result = steering_assistance.preprocess({})
        scenarios = set(result["Scenario"].dropna().tolist())
        self.assertIn("S-bend @ 60 km/h - First turn", scenarios)
        self.assertIn("S-bend @ 130 km/h - Second turn", scenarios)

    def test_preprocess_values_are_none(self):
        result = steering_assistance.preprocess({})
        self.assertTrue(result["Value"].isna().all())


class TestComputeScore(unittest.TestCase):
    def _input_params(self, fitment="standard"):
        return pd.DataFrame(
            {
                "Category": ["Lane change assist"],
                "Input parameter": ["Fitment"],
                "Value": [fitment],
            }
        )

    def _df(self, scenarios, values):
        return pd.DataFrame(
            {
                "Category": ["Steering assistance"] * len(scenarios),
                "Scenario": scenarios,
                "Value": values,
            }
        )

    def test_missing_dataframe_returns_empty_and_initialized_scores(self):
        result_df, score_dict = steering_assistance.compute_score({})
        self.assertTrue(result_df.empty)
        self.assertIn("Lane change assist", score_dict)
        self.assertTrue(pd.isna(score_dict["Lane change assist"]))

    # --- First turn ---

    def test_first_turn_stays_in_lane_with_redirects_second_turn_scores_point_five(
        self,
    ):
        df = self._df(
            ["S-bend @ 60 km/h - First turn", "S-bend @ 60 km/h - Second turn"],
            ["Stays in lane", "Redirects"],
        )
        result_df, score_dict = steering_assistance.compute_score(
            {
                "VA - Steering assistance verif.": df,
                "Input parameters": self._input_params("no"),
            }
        )
        self.assertEqual(result_df.iloc[0]["Score"], 0.5)
        self.assertEqual(score_dict["S-bend @ 60 km/h - First turn"], 0.5)

    def test_first_turn_stays_in_lane_with_stays_in_lane_second_turn_scores_point_five(
        self,
    ):
        df = self._df(
            ["S-bend @ 60 km/h - First turn", "S-bend @ 60 km/h - Second turn"],
            ["Stays in lane", "Stays in lane"],
        )
        result_df, score_dict = steering_assistance.compute_score(
            {
                "VA - Steering assistance verif.": df,
                "Input parameters": self._input_params("no"),
            }
        )
        self.assertEqual(result_df.iloc[0]["Score"], 0.5)

    def test_first_turn_stays_in_lane_with_does_not_redirect_scores_zero(self):
        df = self._df(
            ["S-bend @ 60 km/h - First turn", "S-bend @ 60 km/h - Second turn"],
            ["Stays in lane", "Does not redirect"],
        )
        result_df, score_dict = steering_assistance.compute_score(
            {
                "VA - Steering assistance verif.": df,
                "Input parameters": self._input_params("no"),
            }
        )
        self.assertEqual(result_df.iloc[0]["Score"], 0.0)
        self.assertEqual(score_dict["S-bend @ 60 km/h - First turn"], 0.0)

    def test_first_turn_does_not_stay_in_lane_scores_zero(self):
        df = self._df(
            ["S-bend @ 60 km/h - First turn", "S-bend @ 60 km/h - Second turn"],
            ["Does not stay in lane", "Redirects"],
        )
        result_df, score_dict = steering_assistance.compute_score(
            {
                "VA - Steering assistance verif.": df,
                "Input parameters": self._input_params("no"),
            }
        )
        self.assertEqual(result_df.iloc[0]["Score"], 0.0)

    # --- Second turn ---

    def test_second_turn_stays_in_lane_with_first_turn_stays_scores_point_five(self):
        df = self._df(
            ["S-bend @ 80 km/h - First turn", "S-bend @ 80 km/h - Second turn"],
            ["Stays in lane", "Stays in lane"],
        )
        result_df, score_dict = steering_assistance.compute_score(
            {
                "VA - Steering assistance verif.": df,
                "Input parameters": self._input_params("no"),
            }
        )
        self.assertEqual(result_df.iloc[1]["Score"], 0.5)
        self.assertEqual(score_dict["S-bend @ 80 km/h - Second turn"], 0.5)

    def test_second_turn_redirects_scores_zero(self):
        df = self._df(
            ["S-bend @ 80 km/h - First turn", "S-bend @ 80 km/h - Second turn"],
            ["Stays in lane", "Redirects"],
        )
        result_df, score_dict = steering_assistance.compute_score(
            {
                "VA - Steering assistance verif.": df,
                "Input parameters": self._input_params("no"),
            }
        )
        self.assertEqual(result_df.iloc[1]["Score"], 0.0)

    def test_second_turn_does_not_redirect_scores_zero(self):
        df = self._df(
            ["S-bend @ 80 km/h - First turn", "S-bend @ 80 km/h - Second turn"],
            ["Stays in lane", "Does not redirect"],
        )
        result_df, score_dict = steering_assistance.compute_score(
            {
                "VA - Steering assistance verif.": df,
                "Input parameters": self._input_params("no"),
            }
        )
        self.assertEqual(result_df.iloc[1]["Score"], 0.0)

    def test_second_turn_requires_first_turn_stays_in_lane(self):
        df = self._df(
            ["S-bend @ 60 km/h - First turn", "S-bend @ 60 km/h - Second turn"],
            ["Does not stay in lane", "Stays in lane"],
        )
        result_df, score_dict = steering_assistance.compute_score(
            {
                "VA - Steering assistance verif.": df,
                "Input parameters": self._input_params("no"),
            }
        )
        self.assertEqual(result_df.iloc[1]["Score"], 0.0)
        self.assertEqual(score_dict["S-bend @ 60 km/h - Second turn"], 0.0)

    # --- Lane change assist ---

    def test_lane_change_assist_standard_fitment_scores_one(self):
        df = self._df(
            ["S-bend @ 100 km/h - First turn", "S-bend @ 100 km/h - Second turn"],
            ["Stays in lane", "Stays in lane"],
        )
        _, score_dict = steering_assistance.compute_score(
            {
                "VA - Steering assistance verif.": df,
                "Input parameters": self._input_params("standard"),
            }
        )
        self.assertEqual(score_dict["Lane change assist"], 1.0)

    def test_lane_change_assist_non_standard_fitment_scores_zero(self):
        df = self._df(
            ["S-bend @ 100 km/h - First turn", "S-bend @ 100 km/h - Second turn"],
            ["Stays in lane", "Stays in lane"],
        )
        _, score_dict = steering_assistance.compute_score(
            {
                "VA - Steering assistance verif.": df,
                "Input parameters": self._input_params("no"),
            }
        )
        self.assertEqual(score_dict["Lane change assist"], 0.0)

    def test_lane_change_assist_blank_fitment_is_unassessed(self):
        df = self._df(
            ["S-bend @ 100 km/h - First turn", "S-bend @ 100 km/h - Second turn"],
            ["Stays in lane", "Stays in lane"],
        )
        _, score_dict = steering_assistance.compute_score(
            {
                "VA - Steering assistance verif.": df,
                "Input parameters": self._input_params(None),
            }
        )
        self.assertTrue(pd.isna(score_dict["Lane change assist"]))

    # --- Blank Value cells ---

    def test_blank_value_is_unassessed_not_zero(self):
        df = self._df(
            ["S-bend @ 60 km/h - First turn", "S-bend @ 60 km/h - Second turn"],
            [None, None],
        )
        result_df, score_dict = steering_assistance.compute_score(
            {
                "VA - Steering assistance verif.": df,
                "Input parameters": self._input_params("no"),
            }
        )
        self.assertTrue(pd.isna(result_df.iloc[0]["Score"]))
        self.assertTrue(pd.isna(score_dict["S-bend @ 60 km/h - First turn"]))


if __name__ == "__main__":
    unittest.main()
