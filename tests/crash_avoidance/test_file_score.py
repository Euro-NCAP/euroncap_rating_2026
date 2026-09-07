# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import unittest
from importlib import import_module
from unittest.mock import patch
import numpy as np
import pandas as pd
import logging

from euroncap_rating_2026 import common
from euroncap_rating_2026.crash_avoidance import (
    data_model,
    matrix_processing,
    test_info,
)
from euroncap_rating_2026.crash_avoidance.matrix_processing import (
    get_matrix_indices,
    get_test_matrix_from_region,
)
from euroncap_rating_2026.crash_avoidance.data_model import StageSubelementKey
from euroncap_rating_2026.crash_avoidance.compute_score import (
    calculate_score,
    check_general_requirements_rules,
)
from euroncap_rating_2026.crash_avoidance.test_info import LoadcaseScore

# crash_avoidance/__init__.py rebinds the package attribute "compute_score"
# to the click Command (for the CLI), shadowing the module of the same name.
# import_module() goes through sys.modules and isn't fooled by that.
crash_avoidance_compute_score = import_module(
    "euroncap_rating_2026.crash_avoidance.compute_score"
)

logger = logging.getLogger(__name__)


class TestCrashAvoidanceFileScore(unittest.TestCase):

    @staticmethod
    def _make_short_tables_df():
        G = "Green"
        N = np.nan
        # "FC - Car & PTW pred." DataFrame mimicking short (truncated) matrices.
        # CCRs:  start_col=3, n_cols=7 → 1 data row  → 7 test points
        # CCRm:  start_col=3, n_cols=7 → 5 data rows → 35 test points
        # CCFhos: start_col=4, n_cols=4 → 1 data row → 4 test points
        data = [
            # idx 0  — CCRs marker (start_row = 0+2 = 2)
            ["CCRs", N, N, "Target speed", N, N, N, N, N, N],
            # idx 1  — CCRs column header
            [
                N,
                N,
                N,
                "40 km/h",
                "50 km/h",
                "60 km/h",
                "70 km/h",
                "80 km/h",
                "100 km/h",
                "120 km/h",
            ],
            # idx 2  — CCRs data row 1  (col-0 non-NaN → n_rows=1)
            ["20 km/h", N, N, G, G, G, G, G, G, G],
            # idx 3  — separator (col-0 NaN stops CCRs n_rows count)
            [N, N, N, N, N, N, N, N, N, N],
            # idx 4  — CCRm marker (start_row = 4+2 = 6)
            ["CCRm", N, N, "Target speed", N, N, N, N, N, N],
            # idx 5  — CCRm column header
            [
                N,
                N,
                N,
                "40 km/h",
                "50 km/h",
                "60 km/h",
                "70 km/h",
                "80 km/h",
                "100 km/h",
                "120 km/h",
            ],
            # idx 6-10 — CCRm data rows (n_rows=5 → 35 test points)
            ["30 km/h", N, N, G, G, G, G, G, G, G],
            ["40 km/h", N, N, G, G, G, G, G, G, G],
            ["50 km/h", N, N, G, G, G, G, G, G, G],
            ["60 km/h", N, N, G, G, G, G, G, G, G],
            ["70 km/h", N, N, G, G, G, G, G, G, G],
            # idx 11 — separator
            [N, N, N, N, N, N, N, N, N, N],
            # idx 12 — CCFhos marker (start_col=4, start_row = 12+2 = 14)
            ["CCFhos", N, N, N, "Target speed", N, N, N, N, N],
            # idx 13 — CCFhos column header
            [N, N, N, N, "40 km/h", "50 km/h", "60 km/h", "70 km/h", N, N],
            # idx 14 — CCFhos data row 1 (n_rows=1 → 4 test points)
            ["20 km/h", N, N, N, G, G, G, G, N, N],
        ]
        return pd.DataFrame(data)

    def test_short_tables(self):
        logger.info("Testing short tables scenario...")
        car_ptw_df = self._make_short_tables_df()

        ccrs_indices = get_matrix_indices(car_ptw_df, "CCRs")
        self.assertIsNotNone(ccrs_indices)
        ccrs = get_test_matrix_from_region(
            car_ptw_df,
            ccrs_indices["start_row"],
            ccrs_indices["n_rows"],
            ccrs_indices["start_col"],
            ccrs_indices["n_cols"],
            test_name="CCRs",
            stage_subelement_key=StageSubelementKey.FC,
        )
        self.assertIsNotNone(ccrs)
        self.assertEqual(len(ccrs), 7)
        ccrs_row_zero = [point for point in ccrs if point.row == 0]
        self.assertEqual(len(ccrs_row_zero), 7)
        self.assertTrue(all(point.row == 0 for point in ccrs_row_zero))

        ccrm_indices = get_matrix_indices(car_ptw_df, "CCRm")
        self.assertIsNotNone(ccrm_indices)
        ccrm = get_test_matrix_from_region(
            car_ptw_df,
            ccrm_indices["start_row"],
            ccrm_indices["n_rows"],
            ccrm_indices["start_col"],
            ccrm_indices["n_cols"],
            test_name="CCRm",
            stage_subelement_key=StageSubelementKey.FC,
        )
        self.assertIsNotNone(ccrm)
        self.assertEqual(len(ccrm), 35)

        ccfhos_indices = get_matrix_indices(car_ptw_df, "CCFhos")
        self.assertIsNotNone(ccfhos_indices)
        ccfhos = get_test_matrix_from_region(
            car_ptw_df,
            ccfhos_indices["start_row"],
            ccfhos_indices["n_rows"],
            ccfhos_indices["start_col"],
            ccfhos_indices["n_cols"],
            test_name="CCFhos",
            stage_subelement_key=StageSubelementKey.FC,
        )
        self.assertIsNotNone(ccfhos)
        self.assertEqual(len(ccfhos), 4)

    def test_min_score_all_zero(self):
        """Missing verification sheets → compute_stage_score sets all stage scores to zero."""
        NaN = float("nan")
        # Minimal score skeleton required by calculate_score(). All scores start at 0;
        # with no verification sheets, compute_stage_score returns early and nothing changes.
        dfs = {
            "Scenario Scores": pd.DataFrame(
                {
                    "Stage": ["Crash Avoidance", NaN, NaN],
                    "Stage element": ["Frontal Collisions", NaN, NaN],
                    "Stage subelement": ["Car & PTW", NaN, NaN],
                    "Category": ["Longitudinal", NaN, NaN],
                    "Scenario": ["CCRs", "CCRm", NaN],
                    "Layer": ["Standard", "Standard", "Extended"],
                    "Score": [0.0, 0.0, 0.0],
                    "Max score": [1.2, 2.4, 0.3],
                }
            ),
            "Category Scores": pd.DataFrame(
                {
                    "Stage": ["Crash Avoidance"],
                    "Stage element": ["Frontal Collisions"],
                    "Stage subelement": ["Car & PTW"],
                    "Category": ["Longitudinal"],
                    "Score": [0.0],
                    "Max score": [15.0],
                }
            ),
            "Test Scores": pd.DataFrame(
                {
                    "Stage": ["Crash Avoidance"] + [NaN] * 5,
                    "Stage element": [
                        "Frontal Collisions",
                        NaN,
                        "Lane Departure Collisions",
                        NaN,
                        "Low Speed Collisions",
                        NaN,
                    ],
                    "Stage subelement": [
                        "Car & PTW",
                        "Pedestrian & cyclist",
                        "Single vehicle",
                        "Car & PTW",
                        "Car & PTW",
                        "Pedestrian & cyclist",
                    ],
                    "Score": [0.0] * 6,
                    "Max score": [40.0, 20.0, 10.0, 10.0, 10.0, 10.0],
                }
            ),
            # Empty bg sheets satisfy the direct dict access in check_general_requirements_rules
            # while producing no fail conditions (get_matrix_indices returns None → colors=[]).
            "FC - Ped & Cyc pred. (bg)": pd.DataFrame(),
            "FC - Car & PTW pred. (bg)": pd.DataFrame(),
        }
        # This test exercises the "missing verification sheets" zero-score
        # path with a minimal synthetic skeleton that doesn't match the
        # pristine template's row structure -- bypass the grey-cell rebuild
        # (covered separately in tests/crash_avoidance/test_integrity.py).
        with patch.object(
            crash_avoidance_compute_score.integrity,
            "rebuild_trusted_input_dfs",
            side_effect=lambda d: d,
        ):
            result = calculate_score(dfs)
        scores = result["Test Scores"]["Score"].tolist()
        self.assertTrue(
            all(s == 0.0 for s in scores),
            f"Expected all Test Scores to be 0, got {scores}",
        )

    def test_max_score_tamper_reaching_calculate_score_directly_is_neutralized(self):
        """Regression test mirroring crash_protection's fix: calculate_score(dfs)
        must neutralize a tampered Max score with no file and no preprocess()
        call in between, since callers may invoke it directly with dataframes
        that never went through openpyxl/an on-disk file."""
        from importlib.resources import files

        pristine_path = str(files("data").joinpath("ca_template.xlsx"))
        dfs = common.read_excel_file_to_dfs(pristine_path)
        dfs = {
            k: v
            for k, v in dfs.items()
            if k
            in ("Input parameters", "Test Scores", "Category Scores", "Scenario Scores")
        }
        # Required by check_general_requirements_rules even when no real
        # verification sheets are present (see test_min_score_all_zero above).
        dfs["FC - Ped & Cyc pred. (bg)"] = pd.DataFrame()
        dfs["FC - Car & PTW pred. (bg)"] = pd.DataFrame()

        original_max_score = dfs["Test Scores"].loc[0, "Max score"]
        dfs["Test Scores"] = dfs["Test Scores"].astype(object)
        dfs["Test Scores"].loc[0, "Max score"] = 99999

        result = calculate_score(dfs)

        self.assertEqual(result["Test Scores"].loc[0, "Max score"], original_max_score)
        self.assertNotEqual(result["Test Scores"].loc[0, "Max score"], 99999)

    def test_ccrs_red_zeroes_fc_scores(self):
        """CCRs ≤20 km/h Red cells trigger the FC general-requirements FAIL rule,
        which zeroes all Frontal Collision scores in loadcase_score_dict and Scenario Scores.
        """
        R = common.PredictionColor.RED

        # FC - Car & PTW pred. (bg): first column named "CCRs" so build_start_row_lookup
        # returns start_row=1 from the column header.  check_general_requirements_rules
        # then reads cells (1+row, 3+col) for ccrs_leq20_cells → all Red → FAIL.
        bg_data = [
            [None] * 10,  # row 0: marker
            [None, None, None, R, R, R, R, R, R, R],  # row 1: first VUT speed, all Red
            [None, None, None, R, R, R, R, R, R, R],  # row 2: second VUT speed, all Red
        ]
        fc_car_ptw_bg_df = pd.DataFrame(
            bg_data,
            columns=["CCRs", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"],
        )

        # Pre-populate a non-zero CCRm score to verify it gets zeroed by the rule.
        ccrm_score = LoadcaseScore(
            loadcase_name="CCRm",
            standard_score=1.0,
            extended_score=0.5,
            robustness_layer_score=0.25,
            total_score=1.75,
        )
        loadcase_score_dict = {
            "Frontal Collisions": {"Car & PTW": {"CCRm": ccrm_score}}
        }

        scenario_scores_df = pd.DataFrame(
            {
                "Stage element": ["Frontal Collisions"],
                "Stage subelement": ["Car & PTW"],
                "Category": ["Longitudinal"],
                "Scenario": ["CCRm"],
                "Layer": ["Standard"],
                "Score": [1.0],
                "Max score": [2.4],
            }
        )

        check_general_requirements_rules(
            {
                "FC - Ped & Cyc pred. (bg)": pd.DataFrame(),
                "FC - Car & PTW pred. (bg)": fc_car_ptw_bg_df,
            },
            loadcase_score_dict,
            scenario_scores_df,
        )

        self.assertEqual(ccrm_score.standard_score, 0.0)
        self.assertEqual(ccrm_score.extended_score, 0.0)
        self.assertEqual(ccrm_score.robustness_layer_score, 0.0)
        self.assertEqual(scenario_scores_df.iloc[0]["Score"], 0.0)

    def test_lsc_avoidance_scoring_formula(self):
        """LSC avoidance score = max_score * n_pass / n_total_standard_points when Value is specified.
        value <= 0 → PASS (no impact, GREEN computed, scaling=1.0); value > 0 → FAIL (scaling=0.0).
        With 3 PASS and 2 FAIL out of 5, CCCscp SfS score = 3/5 * 3.0 = 1.8."""
        G = common.PredictionColor.GREEN

        # value <= 0 → PASS; value > 0 → FAIL regardless of OEM prediction color.
        test_points_df = pd.DataFrame(
            {
                "Scenario": ["CCCscp SfS"] * 5,
                "Test point": ["(0, 0)", "(0, 1)", "(0, 2)", "(0, 3)", "(0, 4)"],
                "Range": [matrix_processing.TestRange.STANDARD.value] * 5,
                "OEM Prediction": [G.value] * 5,
                "Value": [0.0, 0.0, 0.0, 1.0, 1.0],
            }
        )
        all_test_points = [
            matrix_processing.TestPoint(
                row=0,
                col=i,
                color=G,
                test_range=matrix_processing.TestRange.STANDARD,
            )
            for i in range(5)
        ]
        input_params_df = pd.DataFrame(
            {
                "Input parameter": ["Vehicle response"],
                "Value": [data_model.VehicleResponse.RETENTION.value],
            }
        )

        score, _ = test_info.compute_lsc_loadcase_score(
            "CCCscp SfS",
            {"CCCscp SfS": test_points_df},
            input_params_df,
            all_test_points,
            matrix_total_cells=5,
        )
        self.assertAlmostEqual(score.standard_score, 3.0 / 5.0 * 3.0, places=3)

    def test_lsc_all_green_gives_max_score(self):
        """All-Green OEM predictions for an LSC avoidance loadcase yield the max standard score."""
        G = common.PredictionColor.GREEN

        test_points_df = pd.DataFrame(
            {
                "Scenario": ["CCCscp SfS"] * 5,
                "Test point": ["(0, 0)", "(0, 1)", "(0, 2)", "(0, 3)", "(0, 4)"],
                "Range": [matrix_processing.TestRange.STANDARD.value] * 5,
                "OEM Prediction": [G.value] * 5,
                "Value": [0.0] * 5,
            }
        )
        all_test_points = [
            matrix_processing.TestPoint(
                row=0,
                col=i,
                color=G,
                test_range=matrix_processing.TestRange.STANDARD,
            )
            for i in range(5)
        ]
        input_params_df = pd.DataFrame(
            {
                "Input parameter": ["Vehicle response"],
                "Value": [data_model.VehicleResponse.RETENTION.value],
            }
        )

        score, _ = test_info.compute_lsc_loadcase_score(
            "CCCscp SfS",
            {"CCCscp SfS": test_points_df},
            input_params_df,
            all_test_points,
            matrix_total_cells=5,
        )
        expected_max = data_model.TOTAL_SCORES["CCCscp SfS"]["Standard"]
        self.assertAlmostEqual(score.standard_score, expected_max, places=3)


if __name__ == "__main__":
    unittest.main()
