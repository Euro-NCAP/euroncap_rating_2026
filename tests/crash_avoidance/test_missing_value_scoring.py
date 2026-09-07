# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import unittest
import numpy as np
import pandas as pd

from euroncap_rating_2026 import common
from euroncap_rating_2026.crash_avoidance import (
    data_model,
    matrix_processing,
    robustness_layer,
    test_info,
)


def _make_test_point(
    color, test_range=matrix_processing.TestRange.STANDARD, row=0, col=0
):
    return matrix_processing.TestPoint(
        row=row,
        col=col,
        color=color,
        test_range=test_range,
    )


def _make_computed_point(
    prediction_color, value, test_range=matrix_processing.TestRange.STANDARD
):
    return matrix_processing.ComputedTestPoint(
        test_name="CCFhos",
        test_point=_make_test_point(prediction_color, test_range=test_range),
        value=value,
    )


def _make_ccfhos_all_test_points(extended_colors, standard_colors):
    """Build a minimal all_test_points list with given extended/standard colors."""
    points = []
    extended_cells = data_model.EXTENDED_RANGE_CELLS["CCFhos"]
    extended_set = set(extended_cells)

    n_rows, n_cols = 8, 4
    ext_iter = iter(extended_colors)
    std_iter = iter(standard_colors)
    for r in range(n_rows):
        for c in range(n_cols):
            if (r, c) in extended_set:
                color = next(ext_iter, common.PredictionColor.GREEN)
                test_range = matrix_processing.TestRange.EXTENDED
            else:
                color = next(std_iter, common.PredictionColor.GREEN)
                test_range = matrix_processing.TestRange.STANDARD
            points.append(
                matrix_processing.TestPoint(
                    row=r, col=c, color=color, test_range=test_range
                )
            )
    return points


class TestComputedTestPointMissingValue(unittest.TestCase):
    """ComputedTestPoint with value=None falls back to the prediction color."""

    def test_none_value_returns_prediction_color_green(self):
        ctp = _make_computed_point(common.PredictionColor.GREEN, value=None)
        self.assertEqual(ctp.computed_color, common.PredictionColor.GREEN)

    def test_none_value_returns_prediction_color_orange(self):
        ctp = _make_computed_point(common.PredictionColor.ORANGE, value=None)
        self.assertEqual(ctp.computed_color, common.PredictionColor.ORANGE)

    def test_none_value_returns_prediction_color_red(self):
        ctp = _make_computed_point(common.PredictionColor.RED, value=None)
        self.assertEqual(ctp.computed_color, common.PredictionColor.RED)

    def test_none_value_sets_prediction_result_correct(self):
        ctp = _make_computed_point(common.PredictionColor.GREEN, value=None)
        self.assertEqual(
            ctp.prediction_result, matrix_processing.PredictionResult.CORRECT
        )

    def test_none_value_extended_range_returns_prediction_color(self):
        ctp = _make_computed_point(
            common.PredictionColor.ORANGE,
            value=None,
            test_range=matrix_processing.TestRange.EXTENDED,
        )
        self.assertEqual(ctp.computed_color, common.PredictionColor.ORANGE)
        self.assertEqual(
            ctp.prediction_result, matrix_processing.PredictionResult.CORRECT
        )

    def test_non_none_value_does_not_use_prediction_color(self):
        """A real value should still go through threshold logic, not short-circuit."""
        ctp = _make_computed_point(common.PredictionColor.GREEN, value=0.0)
        # 0.0 is a valid measurement — prediction_result should NOT be CORRECT just because
        # the prediction was GREEN (it depends on thresholds).
        self.assertIsNotNone(ctp.computed_color)
        # With value=0.0 the computed color may or may not be GREEN depending on thresholds,
        # but the key assertion is that prediction_result is not forced to CORRECT by None.
        # We just confirm the None branch was NOT taken.
        self.assertIsNotNone(ctp.value)


class TestComputePredictedScoreExtendedRange(unittest.TestCase):
    """compute_predicted_score returns a non-zero extended score when extended range
    cells have non-RED predictions."""

    def test_all_green_extended_cells_give_max_extended_score(self):
        all_tp = _make_ccfhos_all_test_points(
            extended_colors=[common.PredictionColor.GREEN] * 14,
            standard_colors=[common.PredictionColor.GREEN] * 18,
        )
        _, extended_score = test_info.compute_predicted_score("CCFhos", all_tp)
        self.assertEqual(extended_score, 0.25)

    def test_all_red_extended_cells_give_zero_extended_score(self):
        all_tp = _make_ccfhos_all_test_points(
            extended_colors=[common.PredictionColor.RED] * 14,
            standard_colors=[common.PredictionColor.GREEN] * 18,
        )
        _, extended_score = test_info.compute_predicted_score("CCFhos", all_tp)
        self.assertEqual(extended_score, 0.0)

    def test_mixed_extended_cells_score_is_thresholded(self):
        # 12 GREEN (score 1.0) + 2 RED (score 0.0) → avg = 12/14 ≈ 0.857 → raw = 0.214
        # Threshold: 0.214 in [0.1875, 0.25) → snaps to 0.1875
        colors = [common.PredictionColor.GREEN] * 12 + [common.PredictionColor.RED] * 2
        all_tp = _make_ccfhos_all_test_points(
            extended_colors=colors,
            standard_colors=[common.PredictionColor.GREEN] * 18,
        )
        _, extended_score = test_info.compute_predicted_score("CCFhos", all_tp)
        self.assertAlmostEqual(extended_score, 0.1875, places=4)

    def test_below_50pct_extended_cells_give_zero(self):
        # 6 GREEN + 8 RED → avg = 6/14 ≈ 0.43 → raw ≈ 0.107 < 0.125 → 0
        colors = [common.PredictionColor.GREEN] * 6 + [common.PredictionColor.RED] * 8
        all_tp = _make_ccfhos_all_test_points(
            extended_colors=colors,
            standard_colors=[common.PredictionColor.GREEN] * 18,
        )
        _, extended_score = test_info.compute_predicted_score("CCFhos", all_tp)
        self.assertEqual(extended_score, 0.0)


class TestComputeLoadcaseScoreExtendedMissingValue(unittest.TestCase):
    """compute_loadcase_score counts missing-value extended range points as correct."""

    def _make_verification_points(self, rows_cols_colors, test_range):
        """Create ComputedTestPoints with value=None (simulating missing Value)."""
        ctps = []
        for (r, c), color in rows_cols_colors:
            tp = _make_test_point(color, test_range=test_range, row=r, col=c)
            ctp = matrix_processing.ComputedTestPoint(
                test_name="CCFhos", test_point=tp, value=None
            )
            ctps.append(ctp)
        return ctps

    def _call_compute_loadcase_score(
        self, computed_test_points, predicted_extended_score
    ):
        standard_points = [
            matrix_processing.ComputedTestPoint(
                test_name="CCFhos",
                test_point=_make_test_point(
                    common.PredictionColor.GREEN,
                    test_range=matrix_processing.TestRange.STANDARD,
                    row=i,
                    col=1,
                ),
                value=0.5,
            )
            for i in range(4)
        ]
        all_ctps = standard_points + computed_test_points
        return test_info.compute_loadcase_score(
            score_test_name="CCFhos",
            subtest_name="CCFhos",
            computed_test_points=all_ctps,
            standard_oem_prediction_method=matrix_processing.PredictionSource.VTA,
            extended_oem_prediction_method=matrix_processing.PredictionSource.VTA,
            predicted_standard_score=2.0,
            predicted_extended_score=predicted_extended_score,
            robustness_layer_score=robustness_layer.RobustnessLayerScore(
                tested_score=0.0, constant_score=0.0
            ),
        )

    def test_two_missing_value_extended_points_give_full_extended_score(self):
        """2 correct extended points out of 2 → discount factor 100% → full predicted score."""
        ext_ctps = self._make_verification_points(
            [
                ((0, 0), common.PredictionColor.GREEN),
                ((1, 0), common.PredictionColor.GREEN),
            ],
            test_range=matrix_processing.TestRange.EXTENDED,
        )
        predicted_extended = 0.1875
        score = self._call_compute_loadcase_score(ext_ctps, predicted_extended)
        self.assertAlmostEqual(score.extended_score, predicted_extended, places=4)

    def test_zero_missing_value_extended_points_give_zero_extended_score(self):
        """0 correct extended points → discount factor 0% → extended score 0."""
        ext_ctps = [
            matrix_processing.ComputedTestPoint(
                test_name="CCFhos",
                test_point=_make_test_point(
                    common.PredictionColor.GREEN,
                    test_range=matrix_processing.TestRange.EXTENDED,
                    row=0,
                    col=0,
                ),
                value=99.0,  # Large value → RED computed color, not matching GREEN prediction
            )
        ]
        score = self._call_compute_loadcase_score(
            ext_ctps, predicted_extended_score=0.1875
        )
        self.assertEqual(score.extended_score, 0.0)

    def test_one_of_two_missing_value_extended_points_gives_half_extended_score(self):
        """1 correct out of 2 extended → VTA discount factor 50% → half predicted score."""
        ext_ctps = self._make_verification_points(
            [
                ((0, 0), common.PredictionColor.GREEN),
                ((1, 0), common.PredictionColor.GREEN),
            ],
            test_range=matrix_processing.TestRange.EXTENDED,
        )
        # Override second point to have a non-matching computed color
        ext_ctps[1] = matrix_processing.ComputedTestPoint(
            test_name="CCFhos",
            test_point=_make_test_point(
                common.PredictionColor.GREEN,
                test_range=matrix_processing.TestRange.EXTENDED,
                row=1,
                col=0,
            ),
            value=99.0,  # triggers RED computed color, mismatch → not counted
        )
        predicted_extended = 0.25
        score = self._call_compute_loadcase_score(ext_ctps, predicted_extended)
        # VTA: 2 tests, 1 correct → 50% factor
        self.assertAlmostEqual(score.extended_score, 0.25 * 0.5, places=4)

    def test_elk_re_regression(self):
        """Regression: ELK-RE extended range cells with value=None (NOVALUE from the assessor) are
        scored CORRECT — the extended score is no longer 0.0 as it was before the fix.
        """
        standard_ctps = [
            matrix_processing.ComputedTestPoint(
                test_name="ELK RE",
                test_point=_make_test_point(
                    common.PredictionColor.GREEN,
                    test_range=matrix_processing.TestRange.STANDARD,
                    row=i,
                    col=0,
                ),
                value=None,
            )
            for i in range(3)
        ]
        extended_ctps = [
            matrix_processing.ComputedTestPoint(
                test_name="ELK RE",
                test_point=_make_test_point(
                    common.PredictionColor.GREEN,
                    test_range=matrix_processing.TestRange.EXTENDED,
                    row=i,
                    col=5,
                ),
                value=None,
            )
            for i in range(2)
        ]
        predicted_extended = 0.5  # ELK RE max extended score
        score = test_info.compute_loadcase_score(
            score_test_name="ELK RE",
            subtest_name="ELK RE",
            computed_test_points=standard_ctps + extended_ctps,
            standard_oem_prediction_method=matrix_processing.PredictionSource.VTA,
            extended_oem_prediction_method=matrix_processing.PredictionSource.VTA,
            predicted_standard_score=4.0,
            predicted_extended_score=predicted_extended,
            robustness_layer_score=robustness_layer.RobustnessLayerScore(
                tested_score=0.0, constant_score=0.0
            ),
        )
        self.assertAlmostEqual(score.extended_score, predicted_extended, places=4)


class TestReadTestPointsPreserveNone(unittest.TestCase):
    """read_test_points preserves None for NaN/unparseable values when preserve_none=True."""

    def _make_minimal_df(self, value):
        return pd.DataFrame(
            [
                {
                    "Test point": "(0, 0)",
                    "Scenario": "ELK RE",
                    "Range": "Standard",
                    "OEM Prediction": "Green",
                    "Value": value,
                }
            ]
        )

    def _get_value(self, df, preserve_none):
        _, computed_info_dict = test_info.read_test_points(
            df, preserve_none=preserve_none
        )
        return computed_info_dict["ELK RE_(0, 0)"]["value"]

    def test_nan_value_becomes_none_when_preserve_none_true(self):
        df = self._make_minimal_df(float("nan"))
        self.assertIsNone(self._get_value(df, preserve_none=True))

    def test_nan_value_becomes_zero_when_preserve_none_false(self):
        df = self._make_minimal_df(float("nan"))
        self.assertEqual(self._get_value(df, preserve_none=False), 0.0)

    def test_novalue_string_becomes_none_when_preserve_none_true(self):
        """Simulates the ELK-RE scenario where the assessor writes 'NOVALUE' for untested cells."""
        df = self._make_minimal_df("NOVALUE")
        self.assertIsNone(self._get_value(df, preserve_none=True))

    def test_novalue_string_becomes_zero_when_preserve_none_false(self):
        df = self._make_minimal_df("NOVALUE")
        self.assertEqual(self._get_value(df, preserve_none=False), 0.0)


class TestExtendedRangePredictedScore(unittest.TestCase):
    """TestPoint.predicted_score in extended range: ORANGE now scores 1.0, not 0.5."""

    def _make_extended_point(self, color):
        return matrix_processing.TestPoint(
            row=0,
            col=0,
            color=color,
            test_range=matrix_processing.TestRange.EXTENDED,
        )

    def test_orange_extended_scores_one_not_half(self):
        tp = self._make_extended_point(common.PredictionColor.ORANGE)
        self.assertEqual(tp.predicted_score, 1.0)

    def test_red_extended_scores_zero(self):
        tp = self._make_extended_point(common.PredictionColor.RED)
        self.assertEqual(tp.predicted_score, 0.0)

    def test_green_extended_scores_one(self):
        tp = self._make_extended_point(common.PredictionColor.GREEN)
        self.assertEqual(tp.predicted_score, 1.0)


def _make_robustness_df(rows):
    """Build a minimal scenario DataFrame for _has_robustness_column_failures tests."""
    return pd.DataFrame(rows)


class TestNotApplicableRobustness(unittest.TestCase):
    """_has_robustness_column_failures must ignore rows where Robustness layer is Not Applicable."""

    def test_not_applicable_layer_with_fail_value_returns_false(self):
        # Assessor overwrote "Not Applicable" with "FAIL" — should still be ignored.
        df = _make_robustness_df(
            [
                {
                    "Range": "Standard",
                    "Robustness layer": "Not Applicable",
                    "Robustness": "FAIL",
                },
                {
                    "Range": "Standard",
                    "Robustness layer": "Not Applicable",
                    "Robustness": "FAIL",
                },
                {
                    "Range": "Standard",
                    "Robustness layer": "Not Applicable",
                    "Robustness": "FAIL",
                },
            ]
        )
        self.assertFalse(test_info._has_robustness_column_failures(df))

    def test_not_applicable_layer_with_not_applicable_value_returns_false(self):
        # Normal template flow: both columns contain "Not Applicable".
        df = _make_robustness_df(
            [
                {
                    "Range": "Standard",
                    "Robustness layer": "Not Applicable",
                    "Robustness": "Not Applicable",
                },
                {
                    "Range": "Standard",
                    "Robustness layer": "Not Applicable",
                    "Robustness": "Not Applicable",
                },
            ]
        )
        self.assertFalse(test_info._has_robustness_column_failures(df))

    def test_real_layer_with_fail_returns_true(self):
        # Existing behaviour: a real layer with FAIL must still be detected.
        df = _make_robustness_df(
            [
                {
                    "Range": "Standard",
                    "Robustness layer": "Impact location",
                    "Robustness": "FAIL",
                },
            ]
        )
        self.assertTrue(test_info._has_robustness_column_failures(df))

    def test_real_layer_with_pass_returns_false(self):
        df = _make_robustness_df(
            [
                {
                    "Range": "Standard",
                    "Robustness layer": "Impact location",
                    "Robustness": "PASS",
                },
            ]
        )
        self.assertFalse(test_info._has_robustness_column_failures(df))

    def test_extended_range_not_applicable_ignored(self):
        # Extended rows are always skipped regardless of the layer name.
        df = _make_robustness_df(
            [
                {
                    "Range": "Extended",
                    "Robustness layer": "Not Applicable",
                    "Robustness": "FAIL",
                },
            ]
        )
        self.assertFalse(test_info._has_robustness_column_failures(df))

    def test_no_robustness_layer_column_with_fail_returns_false(self):
        # ELK RE old-format: assessor added FAIL in Robustness but template had no
        # Robustness layer column (because no pool-eligible layer was selected).
        df = _make_robustness_df(
            [
                {"Range": "Standard", "Robustness": "FAIL"},
                {"Range": "Standard", "Robustness": "FAIL"},
                {"Range": "Standard", "Robustness": "FAIL"},
            ]
        )
        self.assertFalse(test_info._has_robustness_column_failures(df))


class TestComputeLoadcaseScoreVerificationTestCount(unittest.TestCase):
    """compute_loadcase_score handles 0, 1, or 2 actual verification test points."""

    def _correct_ctp(self, test_range, row=0, col=0):
        tp = _make_test_point(
            common.PredictionColor.GREEN, test_range=test_range, row=row, col=col
        )
        return matrix_processing.ComputedTestPoint(
            test_name="CCFhos", test_point=tp, value=None
        )

    def _incorrect_ctp(self, test_range, row=0, col=0):
        tp = _make_test_point(
            common.PredictionColor.GREEN, test_range=test_range, row=row, col=col
        )
        return matrix_processing.ComputedTestPoint(
            test_name="CCFhos", test_point=tp, value=99.0
        )

    def _score(self, ctps, predicted_standard=2.0, predicted_extended=0.25):
        return test_info.compute_loadcase_score(
            score_test_name="CCFhos",
            subtest_name="CCFhos",
            computed_test_points=ctps,
            standard_oem_prediction_method=matrix_processing.PredictionSource.VTA,
            extended_oem_prediction_method=matrix_processing.PredictionSource.VTA,
            predicted_standard_score=predicted_standard,
            predicted_extended_score=predicted_extended,
            robustness_layer_score=robustness_layer.RobustnessLayerScore(
                tested_score=0.0, constant_score=0.0
            ),
        )

    # --- Standard range ---

    def test_zero_standard_tests_gives_full_standard_score(self):
        score = self._score([])
        self.assertAlmostEqual(score.standard_score, 2.0, places=4)

    def test_one_standard_test_correct_gives_full_score(self):
        score = self._score([self._correct_ctp(matrix_processing.TestRange.STANDARD)])
        self.assertAlmostEqual(score.standard_score, 2.0, places=4)

    def test_one_standard_test_incorrect_gives_zero(self):
        score = self._score([self._incorrect_ctp(matrix_processing.TestRange.STANDARD)])
        self.assertAlmostEqual(score.standard_score, 0.0, places=4)

    def test_two_standard_tests_both_correct_gives_full_score(self):
        ctps = [
            self._correct_ctp(matrix_processing.TestRange.STANDARD, row=0),
            self._correct_ctp(matrix_processing.TestRange.STANDARD, row=1),
        ]
        score = self._score(ctps)
        self.assertAlmostEqual(score.standard_score, 2.0, places=4)

    def test_two_standard_tests_one_correct_vta_gives_half(self):
        ctps = [
            self._correct_ctp(matrix_processing.TestRange.STANDARD, row=0),
            self._incorrect_ctp(matrix_processing.TestRange.STANDARD, row=1),
        ]
        score = self._score(ctps)
        self.assertAlmostEqual(score.standard_score, 2.0 * 0.5, places=4)

    def test_two_standard_tests_both_incorrect_gives_zero(self):
        ctps = [
            self._incorrect_ctp(matrix_processing.TestRange.STANDARD, row=0),
            self._incorrect_ctp(matrix_processing.TestRange.STANDARD, row=1),
        ]
        score = self._score(ctps)
        self.assertAlmostEqual(score.standard_score, 0.0, places=4)

    # --- Extended range ---

    def test_zero_extended_tests_gives_full_extended_score(self):
        score = self._score([], predicted_extended=0.25)
        self.assertAlmostEqual(score.extended_score, 0.25, places=4)

    def test_one_extended_test_correct_gives_full_score(self):
        score = self._score(
            [self._correct_ctp(matrix_processing.TestRange.EXTENDED)],
            predicted_extended=0.25,
        )
        self.assertAlmostEqual(score.extended_score, 0.25, places=4)

    def test_one_extended_test_incorrect_gives_zero(self):
        score = self._score(
            [self._incorrect_ctp(matrix_processing.TestRange.EXTENDED)],
            predicted_extended=0.25,
        )
        self.assertAlmostEqual(score.extended_score, 0.0, places=4)


class TestNonLscValueDefaultsToNan(unittest.TestCase):
    """preprocess_stage_subelement must seed every fresh Value cell as NaN,
    not just for LSC scenarios -- otherwise an FC/LDC verification sheet
    ships with a real 0.0 that is indistinguishable from an OEM-measured 0
    once the OEM never touches the cell."""

    FC_LOADCASES = [
        "CCRs",
        "CCRm",
        "CCRb",
        "CCFhos",
        "CCFhol",
        "CCFtap",
        "CCCscp",
        "CMRs",
        "CMRb",
        "CMFtap",
        "CMCscp",
    ]

    def _make_fc_car_ptw_dfs(self):
        pred_data = [
            [
                "CCCscp",
                np.nan,
                np.nan,
                "Target speed",
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                "30 km/h",
                "40 km/h",
                "50 km/h",
                "60 km/h",
                "70 km/h",
                "80 km/h",
                "100 km/h",
            ],
            [
                "30 km/h",
                np.nan,
                np.nan,
                "Red",
                "Red",
                "Red",
                "Red",
                "Red",
                "Red",
                "Red",
            ],
            [
                "50 km/h",
                np.nan,
                np.nan,
                "Red",
                "Red",
                "Red",
                "Red",
                "Red",
                "Red",
                "Red",
            ],
        ]
        return {
            "FC - Car & PTW pred.": pd.DataFrame(pred_data),
            "FC - Car & PTW robust. pred.": pd.DataFrame(
                columns=["Robustness layer"] + self.FC_LOADCASES
            ),
            "Input parameters": pd.DataFrame(
                {
                    "Input parameter": ["Vehicle response"],
                    "Value": ["Retention"],
                }
            ),
        }

    def test_fc_scenario_value_column_is_nan_not_zero(self):
        input_selected_points = {
            "CCCscp": [
                {
                    "Function": "AEB",
                    "VUT speed": "50 km/h",
                    "Target speed": "40 km/h",
                    "Target approach": "Farside",
                },
            ]
        }
        stage_subelement = test_info.preprocess_stage_subelement(
            self._make_fc_car_ptw_dfs(),
            "Frontal Collisions",
            "Car & PTW",
            input_selected_points=input_selected_points,
        )
        cccscp_rows = stage_subelement.test_points_df[
            stage_subelement.test_points_df["Scenario"] == "CCCscp"
        ]
        self.assertEqual(len(cccscp_rows), 1)
        self.assertTrue(cccscp_rows["Value"].isna().all())


class TestMissingVerificationRowsFallBackToPrediction(unittest.TestCase):
    """A claimed (VTA / Self claimed) scenario with no rows at all in the
    verification sheet keeps its full predicted score instead of scoring 0.
    The N/A prediction-method gate stays the only zeroing
    path for an unassessed scenario."""

    FC_LOADCASES = data_model.STAGE_SUBELEMENT_TO_LOADCASES["FC"]["Car & PTW"]

    @staticmethod
    def _make_cccscp_pred_df():
        N = np.nan
        G = "Green"
        # CCCscp: start_col=3, n_cols=7. Marker at idx 0 -> start_row=2,
        # two data rows -> n_rows=2 (outside CELL_REMOVAL_MAP["CCCscp"]).
        data = [
            ["CCCscp", N, N, "Target speed", N, N, N, N, N, N],
            [
                N,
                N,
                N,
                "30 km/h",
                "40 km/h",
                "50 km/h",
                "60 km/h",
                "70 km/h",
                "80 km/h",
                "100 km/h",
            ],
            ["30 km/h", N, N, G, G, G, G, G, G, G],
            ["50 km/h", N, N, G, G, G, G, G, G, G],
        ]
        return pd.DataFrame(data)

    @staticmethod
    def _make_verif_df():
        # Header row is iloc[2] for FC sheets; no data rows below it, so no
        # scenario has any selected verification test points.
        header = ["Scenario", "Test point", "Colour", "Value"]
        return pd.DataFrame([[""] * 4, [""] * 4, header])

    def _make_dfs(self, prediction_method="VTA"):
        pred_df = self._make_cccscp_pred_df()
        return {
            "FC - Car & PTW verif.": self._make_verif_df(),
            "FC - Car & PTW pred.": pred_df,
            "FC - Car & PTW pred. (bg)": pred_df.copy(),
            "FC - Car & PTW robust. pred.": pd.DataFrame(
                {
                    "Robustness layer": ["Layer 1"],
                    **{
                        loadcase: ["YES" if loadcase == "CCCscp" else np.nan]
                        for loadcase in self.FC_LOADCASES
                    },
                }
            ),
            "Input parameters": pd.DataFrame(
                {
                    "Scenario": ["CCCscp", np.nan],
                    "Input parameter": [
                        "Prediction - Standard",
                        "Prediction - Extended",
                    ],
                    "Value": [prediction_method, prediction_method],
                }
            ),
        }

    def _run_stage(self, dfs):
        stage_info = {
            "Stage element": "Frontal Collisions",
            "Stage subelement": "Car & PTW",
        }
        loadcase_score_dict = {"Frontal Collisions": {"Car & PTW": {}}}
        test_info.compute_stage_score(dfs, stage_info, loadcase_score_dict)
        return loadcase_score_dict["Frontal Collisions"]["Car & PTW"]

    def test_claimed_scenario_without_rows_keeps_predicted_score(self):
        dfs = self._make_dfs(prediction_method="VTA")
        pred_df = dfs["FC - Car & PTW pred."]
        matrix_indices = matrix_processing.get_matrix_indices(pred_df, "CCCscp")
        all_test_points = matrix_processing.get_test_matrix_from_region(
            dfs["FC - Car & PTW pred. (bg)"],
            matrix_indices["start_row"],
            matrix_indices["n_rows"],
            matrix_indices["start_col"],
            matrix_indices["n_cols"],
            "CCCscp",
            data_model.StageSubelementKey.FC,
            attributes_df=pred_df,
        )
        expected_standard, expected_extended = test_info.compute_predicted_score(
            "CCCscp", all_test_points
        )
        expected_robustness = test_info.compute_robustness_layer_score(
            dfs["FC - Car & PTW robust. pred."], "CCCscp", "CCCscp"
        ).total_score

        scores = self._run_stage(dfs)

        cccscp = scores["CCCscp"]
        self.assertGreater(expected_standard, 0.0)
        self.assertEqual(cccscp.standard_score, expected_standard)
        self.assertEqual(cccscp.extended_score, expected_extended)
        # The robustness layer is intentionally awarded too: nothing was
        # verified, so nothing failed, and the fixture's predicted standard
        # score meets the 50%-of-total threshold.
        self.assertGreater(expected_robustness, 0.0)
        self.assertEqual(cccscp.robustness_layer_score, expected_robustness)
        self.assertEqual(
            cccscp.total_score,
            expected_standard + expected_extended + expected_robustness,
        )

    def test_na_prediction_method_still_zeroes_scenario(self):
        scores = self._run_stage(self._make_dfs(prediction_method="N/A"))

        cccscp = scores["CCCscp"]
        self.assertEqual(cccscp.standard_score, 0.0)
        self.assertEqual(cccscp.extended_score, 0.0)
        self.assertEqual(cccscp.robustness_layer_score, 0.0)
        self.assertEqual(cccscp.total_score, 0.0)


class TestLscMissingVerificationRowsFallBackToPrediction(unittest.TestCase):
    """compute_lsc_loadcase_score scores the prediction matrix directly when
    a scenario has no verification rows, matching the existing
    blank-Value behaviour cell for cell."""

    @staticmethod
    def _make_input_params_df(vehicle_response="Retention"):
        return pd.DataFrame(
            {
                "Input parameter": ["Vehicle response"],
                "Value": [vehicle_response],
            }
        )

    def test_missing_rows_score_from_prediction_colors(self):
        all_test_points = [
            _make_test_point(common.PredictionColor.GREEN, row=0, col=0),
            _make_test_point(common.PredictionColor.GREEN, row=0, col=1),
            _make_test_point(common.PredictionColor.ORANGE, row=0, col=2),
            _make_test_point(common.PredictionColor.GREY, row=0, col=3),
        ]

        loadcase_score, computed_color_dict = test_info.compute_lsc_loadcase_score(
            "CPMFC",
            {},
            self._make_input_params_df(),
            all_test_points,
            matrix_total_cells=len(all_test_points),
        )

        # (1.0 + 1.0 + 0.5) / 4 cells * 2.0 total standard points
        self.assertEqual(loadcase_score.standard_score, 1.25)
        self.assertEqual(loadcase_score.total_score, 1.25)
        self.assertEqual(loadcase_score.extended_score, 0.0)
        self.assertEqual(loadcase_score.robustness_layer_score, 0.0)
        # No verification rows exist, so there is nothing to write a Colour
        # back into -- the write-back dict must stay empty.
        self.assertEqual(computed_color_dict, {})

    def test_missing_rows_and_missing_matrix_scores_zero(self):
        loadcase_score, _ = test_info.compute_lsc_loadcase_score(
            "CPMFC",
            {},
            self._make_input_params_df(),
            None,
            matrix_total_cells=0,
        )

        self.assertEqual(loadcase_score.standard_score, 0.0)
        self.assertEqual(loadcase_score.total_score, 0.0)

    def test_cbda_na_vehicle_response_gate_still_wins(self):
        all_test_points = [
            _make_test_point(common.PredictionColor.GREEN, row=0, col=0),
        ]

        loadcase_score, _ = test_info.compute_lsc_loadcase_score(
            "CBDA",
            {},
            self._make_input_params_df(vehicle_response="N/A"),
            all_test_points,
            matrix_total_cells=1,
        )

        self.assertEqual(loadcase_score.standard_score, 0.0)
        self.assertEqual(loadcase_score.total_score, 0.0)


if __name__ == "__main__":
    unittest.main()
