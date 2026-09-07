# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import unittest
from importlib import import_module
from importlib.resources import files
from unittest.mock import patch

import pandas as pd

from euroncap_rating_2026 import common
from euroncap_rating_2026.post_crash import data_model, test_info


post_crash_compute_score = import_module(
    "euroncap_rating_2026.post_crash.compute_score"
)


def _update_score_column(df, score_dict, key_column):
    updated_df = df.copy()
    for key, score in score_dict.items():
        updated_df.loc[updated_df[key_column] == key, "Score"] = score
    return updated_df


class TestPostCrashCalculateScore(unittest.TestCase):
    def test_calculate_score_returns_updated_dfs_and_applies_caps(self):
        scenario_scores_df = pd.DataFrame(
            {
                "Scenario": [
                    "Advanced eCall scenario",
                    "Energy scenario",
                    "Extrication scenario",
                ],
                "Score": [0.0, 0.0, 0.0],
            }
        )
        category_scores_df = pd.DataFrame(
            {
                "Category": [
                    "Advanced eCall - TPS",
                    "Thermal propagation",
                    "Submergence",
                ],
                "Score": [0.0, 0.0, 0.0],
            }
        )
        test_scores_df = pd.DataFrame(
            {
                "Stage subelement": ["Advanced eCall", "Rescue Information"],
                "Score": [25.0, 0.0],
            }
        )
        dfs = {
            "Scenario Scores": scenario_scores_df,
            "Category Scores": category_scores_df,
            "Test Scores": test_scores_df,
        }

        with (
            # This test exercises score aggregation/capping only, with
            # minimal synthetic fixtures that don't match the pristine
            # template's row structure -- bypass the grey-cell rebuild
            # (covered separately in tests/post_crash/test_integrity.py).
            patch.object(
                post_crash_compute_score.integrity,
                "rebuild_trusted_input_dfs",
                side_effect=lambda d: d,
            ),
            patch.object(
                post_crash_compute_score.rescue_info,
                "compute_score",
                return_value={"Rescue Information": 4.0},
            ),
            patch.object(
                post_crash_compute_score.advanced_ecall,
                "compute_score",
                return_value={"Advanced eCall scenario": 7.0},
            ),
            patch.object(
                post_crash_compute_score.mcb_hazard_lights,
                "compute_score",
                return_value={
                    "Advanced eCall - TPS": 30.0,
                    "Thermal propagation": 12.0,
                    "Submergence": 5.0,
                },
            ),
            patch.object(
                post_crash_compute_score.energy_management,
                "compute_score",
                return_value={"Energy scenario": 2.0},
            ),
            patch.object(
                post_crash_compute_score.occupant_extrication,
                "compute_score",
                return_value={"Extrication scenario": 3.0},
            ),
            patch.object(
                post_crash_compute_score.common,
                "update_scoring_sheet",
                side_effect=_update_score_column,
            ),
            patch.object(
                post_crash_compute_score.common,
                "update_score_sum_interval",
                side_effect=lambda df, *_args, **_kwargs: df,
            ),
        ):
            result = post_crash_compute_score.calculate_score(dfs)

        self.assertEqual(
            set(result.keys()), {"Scenario Scores", "Category Scores", "Test Scores"}
        )
        self.assertEqual(
            result["Category Scores"]
            .set_index("Category")
            .loc["Advanced eCall - TPS", "Score"],
            15.0,
        )
        self.assertEqual(
            result["Category Scores"]
            .set_index("Category")
            .loc["Thermal propagation", "Score"],
            9.0,
        )
        self.assertEqual(
            result["Category Scores"].set_index("Category").loc["Submergence", "Score"],
            3.0,
        )
        self.assertEqual(
            result["Test Scores"]
            .set_index("Stage subelement")
            .loc["Advanced eCall", "Score"],
            20.0,
        )
        self.assertEqual(
            result["Test Scores"]
            .set_index("Stage subelement")
            .loc["Rescue Information", "Score"],
            4.0,
        )


class TestResolveCappedUnassessedCategories(unittest.TestCase):
    """The capped categories are scored "cumulative up to a maximum of N";
    an unassessed scenario must only block the roll-up while it can still
    change the capped result (Post-Crash protocol 3.2.6: Submergence's
    rescue-tool row is an alternative route that legitimately stays blank
    when window opening already passed)."""

    @staticmethod
    def _submergence_dfs(window_score, rescue_score, category_score=float("nan")):
        scenario_scores_df = pd.DataFrame(
            {
                "Category": ["Submergence", None, "Next category"],
                "Scenario": [
                    "Window opening",
                    "Rescue tool / Emergency device",
                    "Other",
                ],
                "Score": [window_score, rescue_score, 0.0],
                "Max score": [3.0, 1.0, 2.0],
            }
        )
        category_scores_df = pd.DataFrame(
            {
                "Category": ["Submergence", "Next category"],
                "Score": [category_score, 0.0],
            }
        )
        return category_scores_df, scenario_scores_df

    def _resolve(self, category_scores_df, scenario_scores_df):
        return post_crash_compute_score.resolve_capped_unassessed_categories(
            category_scores_df, scenario_scores_df, {"Submergence": 3.0}
        )

    def test_window_pass_with_blank_rescue_tool_is_determined(self):
        category_scores_df, scenario_scores_df = self._submergence_dfs(
            3.0, float("nan")
        )
        result = self._resolve(category_scores_df, scenario_scores_df)
        self.assertEqual(result.loc[0, "Score"], 3.0)

    def test_blank_window_with_rescue_tool_pass_stays_unassessed(self):
        # Could still end up 1.0 (window fails) or 3.0 (window passes, capped).
        category_scores_df, scenario_scores_df = self._submergence_dfs(
            float("nan"), 1.0
        )
        result = self._resolve(category_scores_df, scenario_scores_df)
        self.assertTrue(pd.isna(result.loc[0, "Score"]))

    def test_window_fail_with_blank_rescue_tool_stays_unassessed(self):
        # Could still end up 0.0 or 1.0 depending on the rescue tool.
        category_scores_df, scenario_scores_df = self._submergence_dfs(
            0.0, float("nan")
        )
        result = self._resolve(category_scores_df, scenario_scores_df)
        self.assertTrue(pd.isna(result.loc[0, "Score"]))

    def test_fully_assessed_category_is_left_untouched(self):
        category_scores_df, scenario_scores_df = self._submergence_dfs(
            3.0, 1.0, category_score=3.0
        )
        result = self._resolve(category_scores_df, scenario_scores_df)
        self.assertEqual(result.loc[0, "Score"], 3.0)

    def test_blank_scenario_without_max_score_stays_unassessed(self):
        category_scores_df, scenario_scores_df = self._submergence_dfs(
            3.0, float("nan")
        )
        scenario_scores_df.loc[1, "Max score"] = float("nan")
        result = self._resolve(category_scores_df, scenario_scores_df)
        self.assertTrue(pd.isna(result.loc[0, "Score"]))

    def test_partial_tps_assessment_reaching_the_cap_is_determined(self):
        scenario_scores_df = pd.DataFrame(
            {
                "Category": ["Advanced eCall - TPS"] + [None] * 8,
                "Scenario": [f"TPS scenario {i}" for i in range(9)],
                "Score": [3.0] * 5 + [float("nan")] * 4,
                "Max score": [3.0] * 9,
            }
        )
        category_scores_df = pd.DataFrame(
            {"Category": ["Advanced eCall - TPS"], "Score": [float("nan")]}
        )
        result = post_crash_compute_score.resolve_capped_unassessed_categories(
            category_scores_df, scenario_scores_df, {"Advanced eCall - TPS": 15.0}
        )
        self.assertEqual(result.loc[0, "Score"], 15.0)


class TestSubmergenceDeterminedDespiteBlankRescueTool(unittest.TestCase):
    """Full-pipeline regression: an assessment sheet with
    Window opening = PASS and the rescue-tool alternative left blank must
    yield Submergence 3.0 (not unassessed), and must not blank the whole
    Occupant extrication test score."""

    def test_full_pipeline_scores_submergence_and_test(self):
        pristine_path = str(files("data").joinpath("pc_template.xlsx"))
        dfs = common.read_excel_file_to_dfs(pristine_path)
        for stage_info in data_model.STAGE_SUBELEMENTS:
            test_info.preprocess_stage_subelement(
                dfs, stage_info["Stage element"], stage_info["Stage subelement"]
            )

        dfs["VE - Occupant Extrication Verif"]["Value"] = [
            "PASS",  # Seat belt buckle unlatching (1)
            "PASS",  # Door opening - interior, post low voltage drop (3)
            "FAIL",  # Door opening - exterior, post crash (0 of 4)
            "PASS",  # Door opening - exterior, post crash, post low voltage drop (2)
            "FAIL",  # Tailgate opening (0 of 2)
            "PASS",  # Window opening (3)
            None,  # Rescue tool / Emergency device -- alternative route, unassessed
        ]

        result = post_crash_compute_score.calculate_score(dfs)

        category_scores = result["Category Scores"].set_index("Category")
        self.assertEqual(category_scores.loc["Doors and belts", "Score"], 6.0)
        self.assertEqual(category_scores.loc["Submergence", "Score"], 3.0)
        test_scores = result["Test Scores"].set_index("Stage subelement")
        self.assertEqual(test_scores.loc["Occupant extrication", "Score"], 9.0)


class TestMaxScoreTamperReachingCalculateScoreDirectlyIsNeutralized(unittest.TestCase):
    """Regression test mirroring crash_protection's fix: calculate_score(dfs)
    must neutralize a tampered Max score with no file and no preprocess()
    call in between, since callers may invoke it directly with dataframes
    that never went through openpyxl/an on-disk file."""

    def test_tampered_max_score_is_restored_from_pristine_template(self):
        pristine_path = str(files("data").joinpath("pc_template.xlsx"))
        dfs = common.read_excel_file_to_dfs(pristine_path)
        for stage_info in data_model.STAGE_SUBELEMENTS:
            test_info.preprocess_stage_subelement(
                dfs, stage_info["Stage element"], stage_info["Stage subelement"]
            )

        original_max_score = dfs["Test Scores"].loc[0, "Max score"]
        dfs["Test Scores"] = dfs["Test Scores"].astype(object)
        dfs["Test Scores"].loc[0, "Max score"] = 99999

        result = post_crash_compute_score.calculate_score(dfs)

        self.assertEqual(result["Test Scores"].loc[0, "Max score"], original_max_score)
        self.assertNotEqual(result["Test Scores"].loc[0, "Max score"], 99999)


if __name__ == "__main__":
    unittest.main()
