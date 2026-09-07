# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import unittest
from importlib import import_module
from unittest.mock import patch

import pandas as pd


safe_driving_compute_score = import_module(
    "euroncap_rating_2026.safe_driving.compute_score"
)


class TestSafeDrivingCalculateScore(unittest.TestCase):
    def test_calculate_score_merges_updates_and_strips_internal_color_keys(self):
        input_parameters_df = pd.DataFrame({"Value": [0]})
        original_scenario_scores_df = pd.DataFrame(
            {"Scenario": ["Original"], "Score": [1.0]}
        )
        updated_scenario_scores_df = pd.DataFrame(
            {"Scenario": ["Updated"], "Score": [2.0]}
        )
        updated_acc_df = pd.DataFrame({"Scenario": ["ACC"], "Value": ["PASS"]})

        dfs = {
            "Input parameters": input_parameters_df,
            "Scenario Scores": original_scenario_scores_df,
            "DE - DM pred. (cell_colors)": {"D5": "GREEN"},
            "VA - ACC pred. (bg)": pd.DataFrame({"Color": ["GREEN"]}),
        }

        with patch.object(
            safe_driving_compute_score,
            "get_updated_dfs",
            return_value={
                "Scenario Scores": updated_scenario_scores_df,
                "VA - ACC verif.": updated_acc_df,
            },
        ):
            result = safe_driving_compute_score.calculate_score(dfs)

        self.assertEqual(
            set(result.keys()),
            {"Input parameters", "Scenario Scores", "VA - ACC verif."},
        )
        self.assertTrue(result["Input parameters"].equals(input_parameters_df))
        self.assertTrue(result["Scenario Scores"].equals(updated_scenario_scores_df))
        self.assertTrue(result["VA - ACC verif."].equals(updated_acc_df))


class TestSafeDrivingGetUpdatedDfs(unittest.TestCase):
    def test_get_updated_dfs_calls_driver_monitoring_with_dfs_only(self):
        dfs = {
            "Scenario Scores": pd.DataFrame(columns=["Scenario", "Category", "Score"]),
            "Category Scores": pd.DataFrame(columns=["Category", "Score"]),
            "Test Scores": pd.DataFrame(columns=["Stage subelement", "Score"]),
        }
        empty_df = pd.DataFrame()
        dm_score_df = pd.DataFrame(columns=["Category", "Scenario", "Score"])

        with (
            # This test exercises the per-sub-module dispatch only, with a
            # minimal synthetic skeleton that doesn't match the pristine
            # template's row structure -- bypass the grey-cell rebuild
            # (covered separately in tests/safe_driving/test_integrity.py).
            patch.object(
                safe_driving_compute_score.integrity,
                "rebuild_trusted_input_dfs",
                side_effect=lambda d: d,
            ),
            patch.object(
                safe_driving_compute_score.seatbelt_usage,
                "compute_seatbelt_score",
                return_value=(empty_df, {}),
            ),
            patch.object(
                safe_driving_compute_score.occupant_classification,
                "compute_classification_score",
                return_value=(empty_df, {}),
            ),
            patch.object(
                safe_driving_compute_score.occupant_presence,
                "compute_classification_score",
                return_value=(empty_df, {}),
            ),
            patch.object(
                safe_driving_compute_score.driver_monitoring,
                "compute_score",
                return_value=dm_score_df,
            ) as mock_dm_compute_score,
            patch.object(
                safe_driving_compute_score.general_vehicle_controls,
                "compute_score",
                return_value={},
            ),
            patch.object(
                safe_driving_compute_score.speed_assistance,
                "compute_score",
                return_value=(empty_df, dfs["Scenario Scores"], {}, None),
            ),
            patch.object(
                safe_driving_compute_score.acc_performance,
                "compute_score",
                return_value=(empty_df, {}),
            ),
            patch.object(
                safe_driving_compute_score.steering_assistance,
                "compute_score",
                return_value=(empty_df, {}),
            ),
            patch.object(
                safe_driving_compute_score.common,
                "update_scoring_sheet",
                side_effect=lambda df, *_args, **_kwargs: df,
            ),
            patch.object(
                safe_driving_compute_score.common,
                "update_scoring_sheet_from_df",
                side_effect=lambda df, *_args, **_kwargs: df,
            ),
            patch.object(
                safe_driving_compute_score.common,
                "update_score_sum_interval",
                side_effect=lambda df, *_args, **_kwargs: df,
            ),
        ):
            safe_driving_compute_score.get_updated_dfs(dfs)

        mock_dm_compute_score.assert_called_once_with(dfs)

    def test_max_score_tamper_reaching_get_updated_dfs_directly_is_neutralized(self):
        """Regression test mirroring crash_protection's fix: get_updated_dfs(dfs)
        -- the real dataframe-in/dataframe-out entry point (sd_get_updated_dfs)
        -- must neutralize a tampered Max score with no file and no
        preprocess() call in between, since callers may invoke it directly
        with dataframes that never went through a file, e.g. after being
        reconstructed from parquet storage."""
        from importlib.resources import files

        from euroncap_rating_2026 import common

        pristine_path = str(files("data").joinpath("sd_template.xlsx"))
        dfs = common.read_excel_file_to_dfs(pristine_path)
        dfs = {
            k: v
            for k, v in dfs.items()
            if k
            in ("Input parameters", "Test Scores", "Category Scores", "Scenario Scores")
        }

        original_max_score = dfs["Test Scores"].loc[0, "Max score"]
        dfs["Test Scores"] = dfs["Test Scores"].astype(object)
        dfs["Test Scores"].loc[0, "Max score"] = 99999

        empty_df = pd.DataFrame()
        dm_score_df = pd.DataFrame(columns=["Category", "Scenario", "Score"])
        with (
            patch.object(
                safe_driving_compute_score.seatbelt_usage,
                "compute_seatbelt_score",
                return_value=(empty_df, {}),
            ),
            patch.object(
                safe_driving_compute_score.occupant_classification,
                "compute_classification_score",
                return_value=(empty_df, {}),
            ),
            patch.object(
                safe_driving_compute_score.occupant_presence,
                "compute_classification_score",
                return_value=(empty_df, {}),
            ),
            patch.object(
                safe_driving_compute_score.driver_monitoring,
                "compute_score",
                return_value=dm_score_df,
            ),
            patch.object(
                safe_driving_compute_score.general_vehicle_controls,
                "compute_score",
                return_value={},
            ),
            patch.object(
                safe_driving_compute_score.speed_assistance,
                "compute_score",
                return_value=(empty_df, dfs["Scenario Scores"], {}, None),
            ),
            patch.object(
                safe_driving_compute_score.acc_performance,
                "compute_score",
                return_value=(empty_df, {}),
            ),
            patch.object(
                safe_driving_compute_score.steering_assistance,
                "compute_score",
                return_value=(empty_df, {}),
            ),
        ):
            result = safe_driving_compute_score.get_updated_dfs(dfs)

        self.assertEqual(result["Test Scores"].loc[0, "Max score"], original_max_score)
        self.assertNotEqual(result["Test Scores"].loc[0, "Max score"], 99999)


if __name__ == "__main__":
    unittest.main()
