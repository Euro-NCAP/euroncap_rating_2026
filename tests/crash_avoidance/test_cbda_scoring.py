# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from euroncap_rating_2026 import common
from euroncap_rating_2026.crash_avoidance import (
    data_model,
    matrix_processing,
    test_info,
)

GREEN = common.PredictionColor.GREEN
YELLOW = common.PredictionColor.YELLOW
ORANGE = common.PredictionColor.ORANGE
BROWN = common.PredictionColor.BROWN
RED = common.PredictionColor.RED
GREY = common.PredictionColor.GREY

INFORMATION = data_model.VehicleResponse.INFORMATION
WARNING = data_model.VehicleResponse.WARNING
RETENTION = data_model.VehicleResponse.RETENTION

DRIVER = test_info.DRIVER_DOOR
NON_DRIVER = "Passenger"


def _make_test_point(color, row=0, col=0):
    return matrix_processing.TestPoint(
        row=row,
        col=col,
        color=color,
        test_range=matrix_processing.TestRange.STANDARD,
    )


def _make_computed_point(color, value):
    return matrix_processing.ComputedTestPoint(
        test_name="CBDA",
        test_point=_make_test_point(color),
        value=value,
        assessment_criteria=None,
        prediction_result=None,
    )


def _make_input_params_df(vehicle_response):
    return pd.DataFrame(
        {
            "Input parameter": ["Vehicle response"],
            "Value": [vehicle_response.value if vehicle_response else np.nan],
        }
    )


def _make_cbda_df(
    predicted_color,
    value,
    door_selected=DRIVER,
    type_of_system=None,
    expected_value_col="TTC",
):
    df = pd.DataFrame(
        {
            "Scenario": ["CBDA"],
            "Test point": ["(0, 0)"],
            "Range": [matrix_processing.TestRange.STANDARD.value],
            "OEM Prediction": [predicted_color.value],
            "Door selected": [door_selected],
            "Expected value": [expected_value_col],
            "Value": [value],
        }
    )
    if type_of_system is not None:
        df["Type of system"] = [type_of_system]
    return df


def _make_standard_test_points(n):
    return [_make_test_point(GREEN, row=i) for i in range(n)]


#: Marks "caller did not pass a door" in the helpers below, so that an explicit
#: door=None (a blank cell) is not mistaken for "pick the implied door".
_UNSET = object()


def _implied_door(vehicle_response, color):
    """The door change 3 assigns to a consistent (claim, prediction) pair."""
    if (vehicle_response, color) in test_info.DOORING_ALL_DOOR_CLAIMS:
        return NON_DRIVER
    return DRIVER


class TestCbdaRetentionColor(unittest.TestCase):
    """Retention pass band: FAIL if -0.4 < ttc < 1.7, PASS otherwise. Each case
    is scored on the door its predicted colour implies (Green -> a non-driver
    door, Yellow -> Driver)."""

    def _color(self, predicted_color, value, door=None):
        door = door or _implied_door(RETENTION, predicted_color)
        return _make_computed_point(predicted_color, value).get_lsc_computed_color(
            RETENTION, door
        )

    def test_pass_negative_value_green_prediction(self):
        self.assertEqual(self._color(GREEN, -2.0), GREEN)

    def test_pass_negative_value_yellow_prediction(self):
        self.assertEqual(self._color(YELLOW, -2.0), YELLOW)

    def test_pass_high_positive_value_green_prediction(self):
        self.assertEqual(self._color(GREEN, 2.0), GREEN)

    def test_pass_high_positive_value_yellow_prediction(self):
        self.assertEqual(self._color(YELLOW, 2.0), YELLOW)

    def test_fail_zero_value(self):
        self.assertEqual(self._color(GREEN, 0.0), RED)

    def test_fail_mid_range_value(self):
        self.assertEqual(self._color(YELLOW, 1.0), RED)

    def test_boundary_pass_at_minus_0_4(self):
        # -0.4 is NOT > -0.4, so NOT in fail range → PASS
        self.assertEqual(self._color(GREEN, -0.4), GREEN)

    def test_boundary_pass_at_1_7(self):
        # 1.7 is NOT < 1.7, so NOT in fail range → PASS
        self.assertEqual(self._color(GREEN, 1.7), GREEN)

    def test_boundary_fail_just_above_minus_0_4(self):
        self.assertEqual(self._color(GREEN, -0.39), RED)

    def test_boundary_fail_just_below_1_7(self):
        self.assertEqual(self._color(GREEN, 1.69), RED)

    def test_pass_orange_prediction_stays_orange(self):
        # A Retention vehicle may predict Orange (equal or worse than its
        # claim); the prediction caps the row, so a pass scores Orange, not
        # the driver's-door Retention ceiling (Yellow).
        self.assertEqual(self._color(ORANGE, 2.0, door=DRIVER), ORANGE)

    def test_nan_value_returns_red(self):
        # NaN TTC must be treated as FAIL → RED (chained comparison with NaN would silently PASS)
        self.assertEqual(self._color(GREEN, float("nan")), RED)

    def test_none_value_returns_red(self):
        self.assertEqual(self._color(GREEN, None), RED)


class TestCbdaWarningColor(unittest.TestCase):
    """Warning pass band: FAIL if ttc < 1.7. Yellow (all doors) is scored on a
    non-driver door, Orange (driver's door only) on the driver's door."""

    def _color(self, predicted_color, value, door=None):
        door = door or _implied_door(WARNING, predicted_color)
        return _make_computed_point(predicted_color, value).get_lsc_computed_color(
            WARNING, door
        )

    def test_pass_orange_prediction(self):
        self.assertEqual(self._color(ORANGE, 2.0), ORANGE)

    def test_pass_yellow_prediction(self):
        self.assertEqual(self._color(YELLOW, 2.0), YELLOW)

    def test_fail_below_threshold(self):
        self.assertEqual(self._color(ORANGE, 1.5), RED)

    def test_fail_exactly_below_threshold(self):
        self.assertEqual(self._color(YELLOW, 1.69), RED)

    def test_pass_at_exact_threshold(self):
        self.assertEqual(self._color(ORANGE, 1.7), ORANGE)

    def test_nan_value_returns_red(self):
        self.assertEqual(self._color(YELLOW, float("nan")), RED)


class TestCbdaInformationColor(unittest.TestCase):
    """Information pass band: FAIL if ttc < 2.3. The door never matters."""

    def _color(self, predicted_color, value, door=DRIVER):
        return _make_computed_point(predicted_color, value).get_lsc_computed_color(
            INFORMATION, door
        )

    def test_pass_brown_prediction(self):
        self.assertEqual(self._color(BROWN, 3.0), BROWN)

    def test_pass_brown_prediction_any_door(self):
        self.assertEqual(self._color(BROWN, 3.0, door="Behind driver"), BROWN)

    def test_fail_below_threshold(self):
        self.assertEqual(self._color(BROWN, 2.0), RED)

    def test_pass_at_exact_threshold(self):
        self.assertEqual(self._color(BROWN, 2.3), BROWN)

    def test_fail_just_below_threshold(self):
        self.assertEqual(self._color(BROWN, 2.29), RED)

    def test_nan_value_returns_red(self):
        self.assertEqual(self._color(BROWN, float("nan")), RED)


class TestCbdaDoorCeiling(unittest.TestCase):
    """The per-row rule: computed = worst_of(ceiling(type, door), prediction)
    when the pass band is met, RED otherwise."""

    def _color(self, type_of_system, door, predicted_color, value):
        return _make_computed_point(predicted_color, value).get_lsc_computed_color(
            type_of_system, door
        )

    def test_ceiling_by_type_and_door_on_pass(self):
        cases = [
            (RETENTION, "Passenger", GREEN, 2.0, GREEN),
            (RETENTION, "Behind driver", GREEN, -1.0, GREEN),
            (RETENTION, DRIVER, GREEN, 2.0, YELLOW),
            (WARNING, "Behind passenger", YELLOW, 2.0, YELLOW),
            (WARNING, DRIVER, YELLOW, 2.0, ORANGE),
            (INFORMATION, DRIVER, BROWN, 3.0, BROWN),
            (INFORMATION, "Passenger", BROWN, 3.0, BROWN),
        ]
        for type_of_system, door, predicted, value, expected in cases:
            with self.subTest(type=type_of_system, door=door, predicted=predicted):
                self.assertEqual(
                    self._color(type_of_system, door, predicted, value), expected
                )

    def test_fail_is_red_for_every_type_and_door(self):
        cases = [
            (RETENTION, "Passenger", GREEN, 1.0),
            (RETENTION, DRIVER, YELLOW, 0.0),
            (WARNING, "Passenger", YELLOW, 1.5),
            (WARNING, DRIVER, ORANGE, 1.69),
            (INFORMATION, DRIVER, BROWN, 2.0),
            (INFORMATION, "Passenger", BROWN, float("nan")),
        ]
        for type_of_system, door, predicted, value in cases:
            with self.subTest(type=type_of_system, door=door, value=value):
                self.assertEqual(
                    self._color(type_of_system, door, predicted, value), RED
                )

    def test_prediction_worse_than_ceiling_caps_the_row(self):
        # A Retention row with a Brown cell must stay Brown, not rise to the
        # ceiling; likewise Warning + Brown and Retention + Orange.
        self.assertEqual(self._color(RETENTION, "Passenger", BROWN, 2.0), BROWN)
        self.assertEqual(self._color(RETENTION, DRIVER, ORANGE, 2.0), ORANGE)
        self.assertEqual(self._color(WARNING, "Passenger", BROWN, 2.0), BROWN)
        self.assertEqual(self._color(WARNING, DRIVER, ORANGE, 2.0), ORANGE)

    def test_prediction_better_than_ceiling_is_capped_by_ceiling(self):
        # The ceiling reads the tested door: a Green cell verified on the
        # driver's door can only reach Yellow.
        self.assertEqual(self._color(RETENTION, DRIVER, GREEN, 2.0), YELLOW)
        self.assertEqual(self._color(WARNING, DRIVER, YELLOW, 2.0), ORANGE)
        self.assertEqual(self._color(INFORMATION, "Passenger", GREEN, 3.0), BROWN)

    def test_missing_door_is_scored_as_driver(self):
        self.assertEqual(self._color(RETENTION, None, GREEN, 2.0), YELLOW)
        self.assertEqual(self._color(WARNING, "", YELLOW, 2.0), ORANGE)

    def test_door_comparison_is_case_and_space_insensitive(self):
        self.assertEqual(self._color(RETENTION, " driver ", GREEN, 2.0), YELLOW)
        self.assertEqual(self._color(RETENTION, "passenger", GREEN, 2.0), GREEN)

    def test_unrecognised_door_falls_back_to_the_driver_ceiling(self):
        """A door that is not one of the four the tool assigns must not be read
        as "some other door": that would lift the ceiling to the all-door band
        on the strength of a typo. Retention tops out at Yellow here, not
        Green."""
        for typo in ("Drivr", "Passanger", "co-driver", "rear left", "N/A"):
            with self.subTest(door=typo):
                self.assertEqual(self._color(RETENTION, typo, GREEN, 2.0), YELLOW)
                self.assertEqual(self._color(WARNING, typo, YELLOW, 2.0), ORANGE)

    def test_every_assigned_non_driver_door_lifts_the_ceiling(self):
        """The three real non-driver doors keep the all-door ceiling, so the
        conservative fallback above cannot swallow a legitimate value."""
        for door in test_info.NON_DRIVER_DOORS:
            with self.subTest(door=door):
                self.assertEqual(self._color(RETENTION, door, GREEN, 2.0), GREEN)
                self.assertEqual(self._color(WARNING, door, YELLOW, 2.0), YELLOW)

    def test_unknown_type_with_a_value_is_red(self):
        self.assertEqual(self._color(None, "Passenger", GREEN, 2.0), RED)

    def test_non_dooring_scenarios_ignore_type_and_door(self):
        for scenario in ("CCCscp SfS", "CPMRCs", "CPMFC"):
            with self.subTest(scenario=scenario):
                point = matrix_processing.ComputedTestPoint(
                    test_name=scenario,
                    test_point=_make_test_point(GREEN),
                    value=0.0,
                    assessment_criteria=None,
                    prediction_result=None,
                )
                self.assertEqual(point.get_lsc_computed_color(), GREEN)
                self.assertEqual(point.get_lsc_computed_color(WARNING, DRIVER), GREEN)


class TestCbdaScoring(unittest.TestCase):
    """Integration tests for compute_lsc_loadcase_score with CBDA scenarios."""

    def _score(
        self,
        vehicle_response,
        predicted_color,
        value,
        door=_UNSET,
        type_of_system=None,
        expected_value_col="TTC",
    ):
        if door is _UNSET:
            door = _implied_door(vehicle_response, predicted_color)
        return test_info.compute_lsc_loadcase_score(
            "CBDA",
            {
                "CBDA": _make_cbda_df(
                    predicted_color, value, door, type_of_system, expected_value_col
                )
            },
            _make_input_params_df(vehicle_response),
            _make_standard_test_points(1),
            matrix_total_cells=1,
        )

    # Information
    def test_information_pass_brown_scores_correctly(self):
        score, colors = self._score(INFORMATION, BROWN, 2.5)
        self.assertEqual(score.standard_score, 0.5)
        self.assertEqual(colors["CBDA_(0, 0)"], BROWN)

    def test_information_fail_returns_zero(self):
        score, colors = self._score(INFORMATION, BROWN, 1.0)
        self.assertEqual(score.standard_score, 0.0)
        self.assertEqual(colors["CBDA_(0, 0)"], RED)

    # Warning
    def test_warning_pass_orange_scores_correctly(self):
        score, colors = self._score(WARNING, ORANGE, 2.0)
        self.assertEqual(score.standard_score, 1.0)
        self.assertEqual(colors["CBDA_(0, 0)"], ORANGE)

    def test_warning_pass_yellow_scores_correctly(self):
        score, colors = self._score(WARNING, YELLOW, 2.0)
        self.assertEqual(score.standard_score, 1.5)
        self.assertEqual(colors["CBDA_(0, 0)"], YELLOW)

    def test_warning_fail_returns_zero(self):
        score, colors = self._score(WARNING, ORANGE, 1.5)
        self.assertEqual(score.standard_score, 0.0)
        self.assertEqual(colors["CBDA_(0, 0)"], RED)

    # Retention
    def test_retention_pass_green_scores_correctly(self):
        score, colors = self._score(RETENTION, GREEN, 2.0)
        self.assertEqual(score.standard_score, 2.0)
        self.assertEqual(colors["CBDA_(0, 0)"], GREEN)

    def test_retention_pass_yellow_scores_correctly(self):
        score, colors = self._score(RETENTION, YELLOW, 2.0)
        self.assertEqual(score.standard_score, 1.5)
        self.assertEqual(colors["CBDA_(0, 0)"], YELLOW)

    def test_retention_fail_returns_zero(self):
        score, colors = self._score(RETENTION, GREEN, 0.0)
        self.assertEqual(score.standard_score, 0.0)
        self.assertEqual(colors["CBDA_(0, 0)"], RED)

    def test_retention_nan_value_uses_oem_prediction_color(self):
        # NaN in the df → "not specified" → use OEM prediction color weight directly
        score, colors = self._score(RETENTION, GREEN, float("nan"))
        self.assertEqual(score.standard_score, 2.0)
        self.assertEqual(colors["CBDA_(0, 0)"], GREEN)

    # Per-row type of system and door
    def test_row_type_of_system_overrides_workbook_claim(self):
        # ttc = -1.0 passes the Retention band (end of retention at TTC <= -0.4)
        # but fails the Warning band. The lab marked the row Retention, so the
        # row is trusted for the band; the Yellow prediction still caps it.
        score, colors = self._score(
            WARNING, YELLOW, -1.0, door=NON_DRIVER, type_of_system="Retention"
        )
        self.assertEqual(colors["CBDA_(0, 0)"], YELLOW)
        self.assertEqual(score.standard_score, 1.5)

    def test_blank_type_of_system_falls_back_to_workbook_claim(self):
        for blank in (None, "", np.nan):
            with self.subTest(blank=blank):
                score, colors = self._score(
                    WARNING, YELLOW, -1.0, door=NON_DRIVER, type_of_system=blank
                )
                self.assertEqual(colors["CBDA_(0, 0)"], RED)
                self.assertEqual(score.standard_score, 0.0)

    def test_type_of_system_ceiling_reads_the_door(self):
        # Lab says Retention on a Green cell verified on the driver's door:
        # the ceiling is Yellow whatever the prediction claimed.
        score, colors = self._score(
            RETENTION, GREEN, 2.0, door=DRIVER, type_of_system="Retention"
        )
        self.assertEqual(colors["CBDA_(0, 0)"], YELLOW)
        self.assertEqual(score.standard_score, 1.5)

    def test_retention_row_with_brown_prediction_stays_brown(self):
        score, colors = self._score(
            RETENTION, BROWN, 2.0, door=NON_DRIVER, type_of_system="Retention"
        )
        self.assertEqual(colors["CBDA_(0, 0)"], BROWN)
        self.assertEqual(score.standard_score, 0.5)

    def test_missing_door_is_scored_as_driver(self):
        for blank in (None, "", np.nan):
            with self.subTest(blank=blank):
                score, colors = self._score(RETENTION, GREEN, 2.0, door=blank)
                self.assertEqual(colors["CBDA_(0, 0)"], YELLOW)
                self.assertEqual(score.standard_score, 1.5)

    def test_blank_type_and_blank_door_with_value(self):
        # Falls back to the workbook claim and the driver's door.
        score, colors = self._score(
            WARNING, YELLOW, 2.0, door=np.nan, type_of_system=np.nan
        )
        self.assertEqual(colors["CBDA_(0, 0)"], ORANGE)
        self.assertEqual(score.standard_score, 1.0)

    def test_blank_type_and_blank_door_without_value_uses_prediction(self):
        score, colors = self._score(
            WARNING, YELLOW, float("nan"), door=np.nan, type_of_system=np.nan
        )
        self.assertEqual(colors["CBDA_(0, 0)"], YELLOW)
        self.assertEqual(score.standard_score, 1.5)

    def test_type_of_system_is_read_case_insensitively(self):
        score, colors = self._score(
            WARNING, YELLOW, -1.0, door=NON_DRIVER, type_of_system="retention"
        )
        self.assertEqual(colors["CBDA_(0, 0)"], YELLOW)

    def test_na_vehicle_response_scores_zero(self):
        # An explicit "N/A" Vehicle response means no dooring system is
        # fitted: CBDA scores a real 0.0 with no computed colors.
        input_params_df = pd.DataFrame(
            {"Input parameter": ["Vehicle response"], "Value": ["N/A"]}
        )
        score, colors = test_info.compute_lsc_loadcase_score(
            "CBDA",
            {"CBDA": _make_cbda_df(GREEN, 2.0)},
            input_params_df,
            _make_standard_test_points(1),
            matrix_total_cells=1,
        )
        self.assertEqual(score.standard_score, 0.0)
        self.assertEqual(score.total_score, 0.0)
        self.assertEqual(colors, {})

    def test_na_vehicle_response_does_not_zero_other_lsc_loadcases(self):
        # The N/A guard is CBDA-only: other LSC loadcases ignore the
        # Vehicle response parameter entirely.
        input_params_df = pd.DataFrame(
            {"Input parameter": ["Vehicle response"], "Value": ["N/A"]}
        )
        df = _make_cbda_df(GREEN, float("nan"))
        df["Scenario"] = "CPMRCs"
        score, _ = test_info.compute_lsc_loadcase_score(
            "CPMRCs",
            {"CPMRCs": df},
            input_params_df,
            _make_standard_test_points(1),
            matrix_total_cells=1,
        )
        self.assertGreater(score.standard_score, 0.0)

    def test_blank_vehicle_response_keeps_current_behavior(self):
        # Blank (NaN) Vehicle response is NOT the N/A case: CBDA still
        # scores from the OEM prediction colors.
        input_params_df = pd.DataFrame(
            {"Input parameter": ["Vehicle response"], "Value": [np.nan]}
        )
        score, _ = test_info.compute_lsc_loadcase_score(
            "CBDA",
            {"CBDA": _make_cbda_df(GREEN, float("nan"))},
            input_params_df,
            _make_standard_test_points(1),
            matrix_total_cells=1,
        )
        self.assertEqual(score.standard_score, 2.0)

    def test_partial_matrix_uses_full_denominator(self):
        # 9 selected GREEN cells in a 4×3=12 cell matrix.
        # Expected: 9 × (1/12) × 2.0 = 1.5, not 2.0.
        nine_green_points = [
            _make_test_point(GREEN, row=i, col=j) for i in range(3) for j in range(3)
        ]
        test_points_df = pd.DataFrame(
            {
                "Scenario": ["CBDA"] * 9,
                "Test point": [f"({i}, {j})" for i in range(3) for j in range(3)],
                "Range": [matrix_processing.TestRange.STANDARD.value] * 9,
                "OEM Prediction": [GREEN.value] * 9,
                "Expected value": [float("nan")] * 9,
                "Value": [float("nan")] * 9,
            }
        )
        score, _ = test_info.compute_lsc_loadcase_score(
            "CBDA",
            {"CBDA": test_points_df},
            _make_input_params_df(RETENTION),
            nine_green_points,
            matrix_total_cells=12,
        )
        self.assertAlmostEqual(score.standard_score, 1.5)


class TestLscFullMatrixDenominator(unittest.TestCase):
    """
    Verify that compute_lsc_loadcase_score uses n_rows×n_cols as the denominator,
    not the count of selected test points.

    For every LSC scenario we build:
      - a full GREEN matrix (selected == total) → expect full marks
      - a partial GREEN matrix (selected < total) → expect proportionally reduced score
    The OEM prediction color is GREEN with no value specified (NaN), so the score
    contribution per selected cell is exactly 1.0 (GREEN = full weight).
    """

    def _make_all_green_points(self, scenario, n_rows, n_cols):
        return [
            _make_test_point(GREEN, row=i, col=j)
            for i in range(n_rows)
            for j in range(n_cols)
        ]

    def _make_points_df(self, scenario, test_points):
        return pd.DataFrame(
            {
                "Scenario": [scenario] * len(test_points),
                "Test point": [f"({tp.row}, {tp.col})" for tp in test_points],
                "Range": [matrix_processing.TestRange.STANDARD.value]
                * len(test_points),
                "OEM Prediction": [GREEN.value] * len(test_points),
                "Expected value": [float("nan")] * len(test_points),
                "Value": [float("nan")] * len(test_points),
            }
        )

    def _score(self, scenario, selected_points, matrix_total_cells, vehicle_response):
        points_df = self._make_points_df(scenario, selected_points)
        score, _ = test_info.compute_lsc_loadcase_score(
            scenario,
            {scenario: points_df},
            _make_input_params_df(vehicle_response),
            selected_points,
            matrix_total_cells=matrix_total_cells,
        )
        return score.standard_score

    # --- LSC avoidance scenarios (vehicle_response=WARNING, value≤0 → GREEN) ---
    # For avoidance tests value=NaN → OEM color used directly (GREEN → 1.0 weight).

    def _avoidance_cases(self):
        # (scenario, n_rows, n_cols, total_score)
        return [
            ("CCCscp SfS", 4, 5, 3.0),
            ("CMCscp SfS", 4, 7, 3.0),
            ("CCFtap SfS", 4, 3, 1.0),
            ("CMFtap SfS", 4, 4, 3.0),
            ("CBNAO SfS", 3, 3, 3.0),
            ("CPMRCm", 3, 2, 1.5),
            ("CPMRCs", 3, 3, 1.5),
        ]

    def test_full_selection_gives_full_score(self):
        """When every matrix cell is selected GREEN, score == total_score."""
        for scenario, n_rows, n_cols, total_score in self._avoidance_cases():
            with self.subTest(scenario=scenario):
                all_points = self._make_all_green_points(scenario, n_rows, n_cols)
                result = self._score(
                    scenario,
                    all_points,
                    matrix_total_cells=n_rows * n_cols,
                    vehicle_response=WARNING,
                )
                self.assertAlmostEqual(
                    result,
                    total_score,
                    places=6,
                    msg=f"{scenario}: expected {total_score}, got {result}",
                )

    def test_partial_selection_reduces_score_proportionally(self):
        """
        Selecting k out of N cells (all GREEN, no value) gives score = (k/N) × total_score.
        We drop the last column of each scenario to create a predictable partial selection.
        """
        for scenario, n_rows, n_cols, total_score in self._avoidance_cases():
            with self.subTest(scenario=scenario):
                selected = [
                    _make_test_point(GREEN, row=i, col=j)
                    for i in range(n_rows)
                    for j in range(n_cols - 1)
                ]
                k = len(selected)
                n = n_rows * n_cols
                expected = (k / n) * total_score
                result = self._score(
                    scenario,
                    selected,
                    matrix_total_cells=n,
                    vehicle_response=WARNING,
                )
                self.assertAlmostEqual(
                    result,
                    expected,
                    places=6,
                    msg=f"{scenario}: expected {expected:.4f}, got {result:.4f}",
                )

    # --- CPMFC (LSC mitigation, total_score=2.0) ---

    def test_cpmfc_full_selection_gives_full_score(self):
        n_rows, n_cols, total_score = 3, 3, 2.0
        all_points = self._make_all_green_points("CPMFC", n_rows, n_cols)
        result = self._score(
            "CPMFC",
            all_points,
            matrix_total_cells=n_rows * n_cols,
            vehicle_response=WARNING,
        )
        self.assertAlmostEqual(result, total_score)

    def test_cpmfc_partial_selection_reduces_score(self):
        n_rows, n_cols, total_score = 3, 3, 2.0
        selected = self._make_all_green_points("CPMFC", n_rows, n_cols - 1)
        expected = (6 / 9) * total_score
        result = self._score(
            "CPMFC",
            selected,
            matrix_total_cells=n_rows * n_cols,
            vehicle_response=WARNING,
        )
        self.assertAlmostEqual(result, expected, places=6)

    # --- CBDA (LSC dooring, total_score=2.0) ---

    def test_cbda_full_selection_gives_full_score(self):
        n_rows, n_cols, total_score = 4, 3, 2.0
        all_points = self._make_all_green_points("CBDA", n_rows, n_cols)
        result = self._score(
            "CBDA",
            all_points,
            matrix_total_cells=n_rows * n_cols,
            vehicle_response=RETENTION,
        )
        self.assertAlmostEqual(result, total_score)

    def test_cbda_9_of_12_cells_selected_gives_1_5(self):
        """Concrete CBDA example from the bug report: 9/12 GREEN → 1.5/2.0."""
        n_rows, n_cols, total_score = 4, 3, 2.0
        selected = [
            _make_test_point(GREEN, row=i, col=j) for i in range(3) for j in range(3)
        ]
        result = self._score(
            "CBDA",
            selected,
            matrix_total_cells=n_rows * n_cols,
            vehicle_response=RETENTION,
        )
        self.assertAlmostEqual(result, 1.5, places=6)

    def test_cell_weight_denominator_is_total_not_selected(self):
        """
        Regression guard: with N selected cells and matrix_total_cells=M > N,
        the score must be strictly less than total_score even if all selected cells are GREEN.
        """
        n_rows, n_cols, total_score = 4, 3, 2.0
        selected = [_make_test_point(GREEN)]
        result = self._score(
            "CBDA",
            selected,
            matrix_total_cells=n_rows * n_cols,
            vehicle_response=RETENTION,
        )
        self.assertAlmostEqual(result, total_score / (n_rows * n_cols), places=6)
        self.assertLess(result, total_score)


class TestCbdaDoorAssignment(unittest.TestCase):
    """Change 3: Door selected follows the (Vehicle response, prediction)
    pair. Only Retention+Green and Warning+Yellow claim all doors and draw
    from the three non-driver doors; every other allowed pair gets Driver."""

    ALL_DOOR_PAIRS = [(RETENTION, GREEN), (WARNING, YELLOW)]
    DRIVER_ONLY_PAIRS = [
        (RETENTION, YELLOW),
        (RETENTION, ORANGE),
        (RETENTION, BROWN),
        (WARNING, ORANGE),
        (WARNING, BROWN),
        (INFORMATION, BROWN),
    ]

    def _doors(self, vehicle_response, color, draws=60):
        random.seed(1234)
        doors = set()
        for _ in range(draws):
            (tp,) = test_info.add_dooring_columns(
                [_make_test_point(color)], vehicle_response
            )
            doors.add(tp.attributes["Door selected"])
        return doors

    def test_all_door_claims_never_draw_the_driver_door(self):
        for vehicle_response, color in self.ALL_DOOR_PAIRS:
            with self.subTest(response=vehicle_response, color=color):
                doors = self._doors(vehicle_response, color)
                self.assertNotIn(DRIVER, doors)
                self.assertTrue(doors <= set(test_info.NON_DRIVER_DOORS))
                self.assertEqual(doors, set(test_info.NON_DRIVER_DOORS))

    def test_every_other_pair_is_exactly_driver(self):
        for vehicle_response, color in self.DRIVER_ONLY_PAIRS:
            with self.subTest(response=vehicle_response, color=color):
                self.assertEqual(self._doors(vehicle_response, color), {DRIVER})

    def test_allow_list_is_exactly_the_two_pairs(self):
        self.assertEqual(
            set(test_info.DOORING_ALL_DOOR_CLAIMS),
            set(self.ALL_DOOR_PAIRS),
        )
        self.assertNotIn(DRIVER, test_info.NON_DRIVER_DOORS)
        self.assertEqual(len(test_info.NON_DRIVER_DOORS), 3)

    def test_stored_door_is_preserved_not_rerolled(self):
        tp = _make_test_point(GREEN)
        tp.attributes["Door selected"] = "Behind passenger"
        random.seed(0)
        test_info.add_dooring_columns([tp], RETENTION)
        self.assertEqual(tp.attributes["Door selected"], "Behind passenger")

    def test_blank_stored_door_is_assigned(self):
        for blank in (None, "", np.nan):
            with self.subTest(blank=blank):
                tp = _make_test_point(YELLOW)
                tp.attributes["Door selected"] = blank
                test_info.add_dooring_columns([tp], RETENTION)
                self.assertEqual(tp.attributes["Door selected"], DRIVER)

    def test_type_of_system_seeded_blank_and_function_removed(self):
        tp = _make_test_point(BROWN)
        tp.attributes["Function"] = "Information"
        test_info.add_dooring_columns([tp], INFORMATION)
        self.assertEqual(tp.attributes["Type of system"], "")
        self.assertNotIn("Function", tp.attributes)

    def test_stored_type_of_system_is_kept(self):
        tp = _make_test_point(BROWN)
        tp.attributes["Type of system"] = "Warning"
        test_info.add_dooring_columns([tp], INFORMATION)
        self.assertEqual(tp.attributes["Type of system"], "Warning")


class TestCbdaConsistencyRankRule(unittest.TestCase):
    """Change 1: a prediction may be equal to or worse than the claimed
    Vehicle response, never better. Grey is always allowed."""

    def _check(self, colors, vehicle_response):
        return test_info.check_lsc_consistency(
            [_make_test_point(c) for c in colors], vehicle_response
        )

    def test_allowed_colors_table(self):
        self.assertEqual(test_info.dooring_allowed_colors(INFORMATION), (BROWN, RED))
        self.assertEqual(
            test_info.dooring_allowed_colors(WARNING), (YELLOW, ORANGE, BROWN, RED)
        )
        self.assertEqual(
            test_info.dooring_allowed_colors(RETENTION),
            (GREEN, YELLOW, ORANGE, BROWN, RED),
        )
        self.assertIsNone(test_info.dooring_allowed_colors(None))

    def test_information_allows_brown_and_red_only(self):
        self.assertTrue(self._check([BROWN, RED], INFORMATION))
        for better in (ORANGE, YELLOW, GREEN):
            with self.subTest(color=better):
                self.assertFalse(self._check([better], INFORMATION))

    def test_warning_allows_yellow_and_worse(self):
        self.assertTrue(self._check([YELLOW, ORANGE, BROWN, RED], WARNING))
        self.assertFalse(self._check([GREEN], WARNING))

    def test_retention_allows_everything(self):
        self.assertTrue(self._check([GREEN, YELLOW, ORANGE, BROWN, RED], RETENTION))

    def test_grey_is_always_allowed(self):
        for vehicle_response in (INFORMATION, WARNING, RETENTION):
            with self.subTest(response=vehicle_response):
                self.assertTrue(self._check([GREY], vehicle_response))
                self.assertTrue(self._check([GREY, BROWN], vehicle_response))
        # An all-grey matrix passes even without a Vehicle response.
        self.assertTrue(self._check([GREY, GREY], None))

    def test_predicted_points_without_vehicle_response_fail(self):
        self.assertFalse(self._check([BROWN], None))

    def test_violations_are_collected_per_point(self):
        points = [_make_test_point(c) for c in (GREEN, BROWN, YELLOW, GREY)]
        violations = test_info.collect_lsc_consistency_violations(points, WARNING)
        self.assertEqual([tp.color for tp in violations], [GREEN])


class TestCbdaPreprocess(unittest.TestCase):
    """preprocess_stage_subelement on the CBDA matrix: Expected value, door
    assignment on both selection branches, the blank Type of system column,
    and the widened consistency check."""

    @staticmethod
    def _make_lsc_dfs(vehicle_response, colors=("Green",) * 6):
        # Minimal "LSC - Ped & Cyc pred." DataFrame with a 2-row CBDA matrix.
        # Layout: CBDA marker at idx=0 → start_row=2; two data rows (Gap 0.50 m and 1.00 m).
        # Columns: [0=Gap/marker, 1=Target speed header/values, 2=values, 3=values]
        colors = list(colors)
        data = [
            ["CBDA", "Target speed", np.nan, np.nan],  # marker row
            [np.nan, "10 km/h", "15 km/h", "20 km/h"],  # speed header
            ["0.50 m", *colors[0:3]],  # data row 1
            ["1.00 m", *colors[3:6]],  # data row 2
        ]
        return {
            "LSC - Ped & Cyc pred.": pd.DataFrame(data),
            "Input parameters": _make_input_params_df(vehicle_response),
        }

    def _preprocess_cbda(
        self, vehicle_response, input_selected_points=None, colors=None
    ):
        dfs = (
            self._make_lsc_dfs(vehicle_response, colors)
            if colors
            else self._make_lsc_dfs(vehicle_response)
        )
        stage_subelement = test_info.preprocess_stage_subelement(
            dfs,
            "Low Speed Collisions",
            "Pedestrian & cyclist",
            input_selected_points=input_selected_points,
        )
        cbda_rows = stage_subelement.test_points_df[
            stage_subelement.test_points_df["Scenario"] == "CBDA"
        ].reset_index(drop=True)
        return stage_subelement, cbda_rows

    def _make_input_points(self, door="Driver"):
        return {
            "CBDA": [
                {
                    "Gap": "0.50 m",
                    "Target speed": "10 km/h",
                    "Door selected": door,
                }
            ]
        }

    def test_expected_value_is_generic_ttc_for_every_response(self):
        for vehicle_response in (RETENTION, WARNING, INFORMATION):
            with self.subTest(response=vehicle_response):
                colors = ("Brown",) * 6 if vehicle_response != RETENTION else None
                _, cbda_rows = self._preprocess_cbda(
                    vehicle_response, self._make_input_points(), colors=colors
                )
                self.assertEqual(len(cbda_rows), 1)
                self.assertEqual(cbda_rows.iloc[0]["Expected value"], "TTC")

    def test_expected_value_is_generic_ttc_on_fresh_selection(self):
        _, cbda_rows = self._preprocess_cbda(RETENTION)
        self.assertEqual(len(cbda_rows), 6)
        self.assertTrue((cbda_rows["Expected value"] == "TTC").all())

    def test_default_value_is_nan(self):
        _, cbda_rows = self._preprocess_cbda(RETENTION, self._make_input_points())
        self.assertTrue(pd.isna(cbda_rows.iloc[0]["Value"]))

    def test_single_row_per_input_point(self):
        input_points = {
            "CBDA": [
                {"Gap": "0.50 m", "Target speed": "10 km/h", "Door selected": "Driver"},
                {
                    "Gap": "1.00 m",
                    "Target speed": "15 km/h",
                    "Door selected": "Behind passenger",
                },
            ]
        }
        _, cbda_rows = self._preprocess_cbda(RETENTION, input_points)
        self.assertEqual(len(cbda_rows), 2)

    def test_input_door_is_preserved_not_rerolled(self):
        random.seed(7)
        _, cbda_rows = self._preprocess_cbda(
            RETENTION, self._make_input_points(door="Driver")
        )
        # Retention + Green would draw a non-driver door on a fresh selection;
        # the stored door wins on the input-points branch.
        self.assertEqual(cbda_rows.iloc[0]["Door selected"], "Driver")

    def test_input_point_without_door_gets_one(self):
        input_points = {"CBDA": [{"Gap": "0.50 m", "Target speed": "10 km/h"}]}
        _, cbda_rows = self._preprocess_cbda(RETENTION, input_points)
        self.assertIn(cbda_rows.iloc[0]["Door selected"], test_info.NON_DRIVER_DOORS)

    def test_fresh_selection_assigns_doors_from_the_pair(self):
        random.seed(3)
        _, green_rows = self._preprocess_cbda(RETENTION)
        self.assertEqual(len(green_rows), 6)
        self.assertTrue(
            set(green_rows["Door selected"]) <= set(test_info.NON_DRIVER_DOORS)
        )
        _, yellow_rows = self._preprocess_cbda(RETENTION, colors=("Yellow",) * 6)
        self.assertEqual(set(yellow_rows["Door selected"]), {"Driver"})
        _, mixed_rows = self._preprocess_cbda(
            WARNING, colors=("Yellow", "Orange", "Brown", "Yellow", "Orange", "Brown")
        )
        by_color = dict(zip(mixed_rows["OEM Prediction"], mixed_rows["Door selected"]))
        self.assertIn(by_color["Yellow"], test_info.NON_DRIVER_DOORS)
        self.assertEqual(by_color["Orange"], "Driver")
        self.assertEqual(by_color["Brown"], "Driver")

    def test_type_of_system_is_blank_and_function_absent_on_both_branches(self):
        for input_points in (None, self._make_input_points()):
            with self.subTest(using_input_points=input_points is not None):
                _, cbda_rows = self._preprocess_cbda(RETENTION, input_points)
                self.assertIn("Type of system", cbda_rows.columns)
                self.assertTrue((cbda_rows["Type of system"] == "").all())
                if "Function" in cbda_rows.columns:
                    self.assertTrue(cbda_rows["Function"].isna().all())

    def test_stale_function_from_input_is_dropped(self):
        input_points = {
            "CBDA": [
                {
                    "Gap": "0.50 m",
                    "Target speed": "10 km/h",
                    "Door selected": "Driver",
                    "Function": "Retention",
                }
            ]
        }
        _, cbda_rows = self._preprocess_cbda(RETENTION, input_points)
        if "Function" in cbda_rows.columns:
            self.assertTrue(cbda_rows["Function"].isna().all())

    def test_stored_type_of_system_from_input_is_kept(self):
        input_points = {
            "CBDA": [
                {
                    "Gap": "0.50 m",
                    "Target speed": "10 km/h",
                    "Door selected": "Passenger",
                    "Type of system": "Retention",
                }
            ]
        }
        _, cbda_rows = self._preprocess_cbda(RETENTION, input_points)
        self.assertEqual(cbda_rows.iloc[0]["Type of system"], "Retention")

    def test_lower_than_claimed_predictions_are_accepted(self):
        _, cbda_rows = self._preprocess_cbda(
            RETENTION, colors=("Green", "Yellow", "Orange", "Brown", "Red", "Grey")
        )
        # Red and Grey get no verification row.
        self.assertEqual(
            sorted(cbda_rows["OEM Prediction"]), ["Brown", "Green", "Orange", "Yellow"]
        )

    def test_prediction_better_than_claim_is_refused_on_fresh_selection(self):
        with self.assertRaises(ValueError) as ctx:
            self._preprocess_cbda(WARNING, colors=("Green",) * 6)
        self.assertIn("LSC consistency check failed", str(ctx.exception))

    def test_prediction_better_than_claim_is_refused_on_input_points_branch(self):
        # The check runs over the full matrix on the evaluator's second pass
        # too, not only when points are freshly selected.
        with self.assertRaises(ValueError) as ctx:
            self._preprocess_cbda(
                INFORMATION,
                self._make_input_points(),
                colors=("Brown", "Brown", "Brown", "Brown", "Brown", "Yellow"),
            )
        self.assertIn("LSC consistency check failed", str(ctx.exception))

    def test_na_vehicle_response_skips_the_check(self):
        dfs = self._make_lsc_dfs(WARNING)
        dfs["Input parameters"]["Value"] = ["N/A"]
        stage_subelement = test_info.preprocess_stage_subelement(
            dfs, "Low Speed Collisions", "Pedestrian & cyclist"
        )
        self.assertIsNotNone(stage_subelement)

    def test_type_of_system_round_trips_to_scoring(self):
        # preprocess -> xlsx -> read back -> compute-score, mirroring the
        # Door selected round trip: a blank cell becomes an absent attribute
        # (workbook fallback), a filled one is read per row.
        _, cbda_rows = self._preprocess_cbda(
            RETENTION,
            {
                "CBDA": [
                    {
                        "Gap": "0.50 m",
                        "Target speed": "10 km/h",
                        "Door selected": "Passenger",
                    },
                    {
                        "Gap": "1.00 m",
                        "Target speed": "15 km/h",
                        "Door selected": "Passenger",
                    },
                ]
            },
        )
        cbda_rows = common.reorder_columns_by_header_order(
            cbda_rows, common.VERIFICATION_COLUMN_ORDER
        )
        self.assertIn("Type of system", cbda_rows.columns)
        # One row is filled; the other stays blank. Both rows measured -1.0:
        # passes Retention, fails Warning.
        cbda_rows.loc[0, "Type of system"] = "Warning"
        cbda_rows["Value"] = [-1.0, -1.0]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "verif.xlsx"
            cbda_rows.to_excel(path, index=False)
            read_back = pd.read_excel(path)

        points, _ = test_info.read_test_points(read_back, preserve_none=True)
        by_key = {f"({tp.row}, {tp.col})": tp for tp in points}
        filled = read_back.loc[0, "Test point"]
        blank = read_back.loc[1, "Test point"]
        self.assertEqual(by_key[filled].attributes["Type of system"], "Warning")
        self.assertNotIn("Type of system", by_key[blank].attributes)
        self.assertEqual(by_key[blank].attributes["Door selected"], "Passenger")

        score, colors = test_info.compute_lsc_loadcase_score(
            "CBDA",
            {"CBDA": read_back},
            _make_input_params_df(RETENTION),
            points,
            matrix_total_cells=2,
        )
        # Row marked Warning fails its band (-1.0 < 1.7); the blank row falls
        # back to the workbook's Retention and passes on a non-driver door.
        self.assertEqual(colors[f"CBDA_{filled}"], RED)
        self.assertEqual(colors[f"CBDA_{blank}"], GREEN)
        self.assertAlmostEqual(score.standard_score, 0.5 * 2.0)


if __name__ == "__main__":
    unittest.main()
