# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Regression tests for the CBLA/CPLA extended-range and AEB/FCW row-classification
bug: EXTENDED_RANGE_CELLS and AEB_ONLY_ROW/FCW_AEB_ROW identified cells by
absolute (row, col) index, which broke as soon as a sheet was produced with a
different row count than the current template (e.g. a legacy CBLA layout with
an extra low-speed row). check_extended_range and
matrix_processing.is_fcw_declared_as_aeb now resolve CBLA/CPLA from test-point
attributes (and, for CPLA, position relative to the bottom of the matrix)
instead.

Also covers a follow-up bug found while validating on filled workbooks: the
DataFrame path always feeds get_test_matrix_from_region the bg-color-derived
prediction sheet, which nulls out the label cells this attribute-based logic
depends on.
get_test_matrix_from_region now takes a separate attributes_df for label
lookups (see TestGetTestMatrixFromRegionAttributesDf below).
"""

import unittest
from unittest.mock import patch

import pandas as pd

from euroncap_rating_2026 import common
from euroncap_rating_2026.crash_avoidance import data_model
from euroncap_rating_2026.crash_avoidance import matrix_processing
from euroncap_rating_2026.crash_avoidance import test_info


def _build_matrix_df(rows_info, il_values, start_row=2, start_col=6):
    """
    Build a minimal DataFrame shaped like a CBLA/CPLA prediction sheet region.

    rows_info: list of (vut_speed, target_speed, function) tuples, one per
        matrix row.
    il_values: list of raw "Impact location" values (fractions, e.g. 0.5 for
        50%), one per matrix column.
    """
    n_rows = len(rows_info)
    n_cols = len(il_values)
    total_rows = start_row + n_rows
    total_cols = start_col + n_cols
    data = [[None] * total_cols for _ in range(total_rows)]

    header_row = start_row - 2
    il_header_row = start_row - 1
    data[header_row][1] = "Target speed"
    data[header_row][2] = "Function"
    data[header_row][start_col] = "Impact location"
    for j, il in enumerate(il_values):
        data[il_header_row][start_col + j] = il

    for i, (vut_speed, target_speed, function) in enumerate(rows_info):
        data[start_row + i][0] = vut_speed
        data[start_row + i][1] = target_speed
        data[start_row + i][2] = function
        for j in range(n_cols):
            data[start_row + i][start_col + j] = "green"

    return pd.DataFrame(data)


def _build_color_only_df(rows_info, il_values, start_row=2, start_col=6, color="green"):
    """
    Build a DataFrame shaped like the real bg-color-derived prediction sheet
    (``common.load_sheet_with_bg_colors``): every cell is None except the
    matrix's own data cells, which hold a color string. Row/column label
    cells (Target speed, Function, Impact location, ...) are always None,
    since that loader discards all non-color cell content.
    """
    n_rows = len(rows_info)
    n_cols = len(il_values)
    total_rows = start_row + n_rows
    total_cols = start_col + n_cols
    data = [[None] * total_cols for _ in range(total_rows)]
    for i in range(n_rows):
        for j in range(n_cols):
            data[start_row + i][start_col + j] = color
    return pd.DataFrame(data)


IL_VALUES = [
    0.0,
    0.5,
    0.25,
    0.75,
]  # col1 = 50% (fixed-band standard), col2 = 25% (choice-band standard)


@patch("euroncap_rating_2026.crash_avoidance.matrix_processing.plot_matrix")
class TestCblaExtendedRangeIsAttributeBased(unittest.TestCase):
    def _get_points(self, rows_info):
        df = _build_matrix_df(rows_info, IL_VALUES)
        return matrix_processing.get_test_matrix_from_region(
            df,
            start_row=2,
            n_rows=len(rows_info),
            start_col=6,
            n_cols=len(IL_VALUES),
            test_name="CBLA",
            stage_subelement_key=data_model.StageSubelementKey.FC,
        )

    def _range_grid(self, points, n_rows, n_cols):
        grid = {}
        for tp in points:
            grid[(tp.row, tp.col)] = tp.test_range
        return grid

    def test_current_9_row_template_matches_legacy_index_based_ranges(self, _mock_plot):
        """Sanity check: current 9-row layout gets the same ranges as before."""
        rows_info = (
            [("20 km/h", "15 km/h", "AEB")]
            + [(f"{v} km/h", "15 km/h", "AEB") for v in (30, 40, 50, 60)]
            + [(f"{v} km/h", "20 km/h", "AEB") for v in (50, 60, 70, 80)]
        )
        points = self._get_points(rows_info)
        grid = self._range_grid(points, len(rows_info), len(IL_VALUES))

        # Fixed band (rows 0-4): standard only at col 1 (50% IL).
        for row in range(5):
            for col in range(4):
                expected = (
                    matrix_processing.TestRange.STANDARD
                    if col == 1
                    else matrix_processing.TestRange.EXTENDED
                )
                self.assertEqual(grid[(row, col)], expected, f"row={row}, col={col}")

        # Choice band (rows 5-8): standard only at col 2 (25% IL).
        for row in range(5, 9):
            for col in range(4):
                expected = (
                    matrix_processing.TestRange.STANDARD
                    if col == 2
                    else matrix_processing.TestRange.EXTENDED
                )
                self.assertEqual(grid[(row, col)], expected, f"row={row}, col={col}")

    def test_legacy_10_row_layout_resolves_correctly(self, _mock_plot):
        """
        Reproduces the reported bug: a sheet produced with the legacy 10-row
        layout (extra '10 km/h' row on top of the fixed band) must still get
        correct ranges, even though every row shifts by one against the
        current template's absolute indices.
        """
        rows_info = [
            (f"{v} km/h", "15 km/h", "AEB") for v in (10, 20, 30, 40, 50, 60)
        ] + [(f"{v} km/h", "20 km/h", "FCW") for v in (50, 60, 70, 80)]
        points = self._get_points(rows_info)
        grid = self._range_grid(points, len(rows_info), len(IL_VALUES))

        # "60 km/h @ 15" is matrix row 5, which the old absolute-index dict
        # (calibrated for the 9-row template) mistook for the start of the
        # choice band. It must resolve as fixed-band standard at 50% IL.
        self.assertEqual(grid[(5, 1)], matrix_processing.TestRange.STANDARD)
        self.assertEqual(grid[(5, 2)], matrix_processing.TestRange.EXTENDED)

        # "80 km/h @ 20" is matrix row 9, entirely outside the old dict's
        # range (which only went up to row 8) - it must not silently fall
        # back to STANDARD on every column.
        self.assertEqual(grid[(9, 2)], matrix_processing.TestRange.STANDARD)
        self.assertEqual(grid[(9, 0)], matrix_processing.TestRange.EXTENDED)
        self.assertEqual(grid[(9, 1)], matrix_processing.TestRange.EXTENDED)
        self.assertEqual(grid[(9, 3)], matrix_processing.TestRange.EXTENDED)

    def test_collapsed_choice_duplicate_promotes_fixed_row_to_both_ils(
        self, _mock_plot
    ):
        """
        After test_info.collapse_cpla_cbla_aeb_duplicate_rows removes the
        choice-band (Target speed=20) duplicate at 50 km/h because it
        declared AEB, the surviving fixed-band (Target speed=15) row at 50
        km/h is standard at *both* 50% and 25% IL. 60 km/h's duplicate was
        kept (declared FCW), so it keeps the unpromoted two-row shape.
        """
        rows_info = [(f"{v} km/h", "15 km/h", "AEB") for v in (20, 30, 40, 50, 60)] + [
            (f"{v} km/h", "20 km/h", "FCW") for v in (60, 70, 80)
        ]
        points = self._get_points(rows_info)
        grid = self._range_grid(points, len(rows_info), len(IL_VALUES))

        # 50 km/h @ 15 (row 3): promoted, standard at both 50% and 25% IL.
        self.assertEqual(grid[(3, 1)], matrix_processing.TestRange.STANDARD)
        self.assertEqual(grid[(3, 2)], matrix_processing.TestRange.STANDARD)
        self.assertEqual(grid[(3, 0)], matrix_processing.TestRange.EXTENDED)
        self.assertEqual(grid[(3, 3)], matrix_processing.TestRange.EXTENDED)

        # 60 km/h @ 15 (row 4): duplicate still present (kept as FCW), not
        # promoted -- standard only at 50% IL as usual.
        self.assertEqual(grid[(4, 1)], matrix_processing.TestRange.STANDARD)
        self.assertEqual(grid[(4, 2)], matrix_processing.TestRange.EXTENDED)

        # 60 km/h @ 20 (row 5): the surviving choice-band duplicate, standard
        # only at 25% IL as usual.
        self.assertEqual(grid[(5, 2)], matrix_processing.TestRange.STANDARD)
        self.assertEqual(grid[(5, 1)], matrix_processing.TestRange.EXTENDED)


@patch("euroncap_rating_2026.crash_avoidance.matrix_processing.plot_matrix")
class TestCplaExtendedRangeIsContentBased(unittest.TestCase):
    """
    CPLA's fixed/choice band split (and the resulting standard-range
    columns) is derived from each row's actual VUT speed (see
    matrix_processing.classify_cpla_rows), not row position -- it must keep
    resolving correctly once
    test_info.collapse_cpla_cbla_aeb_duplicate_rows has removed a variable
    number of the matrix's duplicate 50/60 km/h rows.
    """

    def _get_points(self, rows_info):
        df = _build_matrix_df(rows_info, IL_VALUES)
        return matrix_processing.get_test_matrix_from_region(
            df,
            start_row=2,
            n_rows=len(rows_info),
            start_col=6,
            n_cols=len(IL_VALUES),
            test_name="CPLA day",
            stage_subelement_key=data_model.StageSubelementKey.FC,
        )

    def _grid(self, points):
        return {(tp.row, tp.col): tp.test_range for tp in points}

    def _assert_row_standard_at(self, grid, row, standard_cols):
        for col in range(4):
            expected = (
                matrix_processing.TestRange.STANDARD
                if col in standard_cols
                else matrix_processing.TestRange.EXTENDED
            )
            self.assertEqual(grid[(row, col)], expected, f"row={row}, col={col}")

    def test_10_row_layout_choice_band_is_last_4_rows(self, _mock_plot):
        """Uncollapsed 10-row matrix: same shape as the real template --
        fixed band standard at col 1 (50% IL), choice band at col 2 (25%
        IL), matching the old bottom-anchored result bit-for-bit."""
        rows_info = [
            (f"{v} km/h", "5 km/h", "AEB") for v in (10, 20, 30, 40, 50, 60)
        ] + [(f"{v} km/h", "5 km/h", "FCW") for v in (50, 60, 70, 80)]
        grid = self._grid(self._get_points(rows_info))
        for row in range(6):
            self._assert_row_standard_at(grid, row, {1})
        for row in range(6, 10):
            self._assert_row_standard_at(grid, row, {2})

    def test_11_row_layout_choice_band_is_still_last_4_rows(self, _mock_plot):
        """An extra fixed-band row (own distinct VUT speed) must not shift
        the choice band detection."""
        rows_info = [
            (f"{v} km/h", "5 km/h", "AEB") for v in (5, 10, 20, 30, 40, 50, 60)
        ] + [(f"{v} km/h", "5 km/h", "FCW") for v in (50, 60, 70, 80)]
        grid = self._grid(self._get_points(rows_info))
        for row in range(7):
            self._assert_row_standard_at(grid, row, {1})
        for row in range(7, 11):
            self._assert_row_standard_at(grid, row, {2})

    def test_both_speeds_collapsed_promotes_both_to_standard_at_both_ils(
        self, _mock_plot
    ):
        """8-row shape: both 50 and 60 km/h duplicates were collapsed
        (Function=AEB on both) -- the surviving single rows are standard at
        *both* 50% and 25% IL."""
        rows_info = [
            (f"{v} km/h", "5 km/h", "AEB") for v in (10, 20, 30, 40, 50, 60)
        ] + [(f"{v} km/h", "5 km/h", "AEB") for v in (70, 80)]
        grid = self._grid(self._get_points(rows_info))
        for row in range(4):
            self._assert_row_standard_at(grid, row, {1})
        self._assert_row_standard_at(grid, 4, {1, 2})  # promoted 50 km/h
        self._assert_row_standard_at(grid, 5, {1, 2})  # promoted 60 km/h
        for row in (6, 7):
            self._assert_row_standard_at(grid, row, {2})

    def test_only_50_collapsed_60_kept_as_fcw_duplicate(self, _mock_plot):
        """9-row shape, per-speed: 50 km/h's duplicate was collapsed (AEB),
        60 km/h's was kept because it declared FCW instead."""
        rows_info = (
            [(f"{v} km/h", "5 km/h", "AEB") for v in (10, 20, 30, 40, 50, 60)]
            + [("60 km/h", "5 km/h", "FCW")]
            + [(f"{v} km/h", "5 km/h", "FCW") for v in (70, 80)]
        )
        grid = self._grid(self._get_points(rows_info))
        for row in range(4):
            self._assert_row_standard_at(grid, row, {1})
        self._assert_row_standard_at(grid, 4, {1, 2})  # promoted 50 km/h
        self._assert_row_standard_at(grid, 5, {1})  # 60 km/h fixed, kept dup
        self._assert_row_standard_at(grid, 6, {2})  # 60 km/h choice, kept dup
        for row in (7, 8):
            self._assert_row_standard_at(grid, row, {2})


@patch("euroncap_rating_2026.crash_avoidance.matrix_processing.plot_matrix")
class TestInternalRowInfoDoesNotLeakIntoAttributes(unittest.TestCase):
    """_cpla_row_info/_cbla_standard_ils are precomputed bookkeeping needed
    by is_fcw_declared_as_aeb and _check_extended_range_cpla/cbla, but must
    never appear in TestPoint.attributes -- external callers (e.g. a
    prediction-cell-values API used by other services) return that dict
    verbatim, and it leaked there before."""

    def _no_underscore_keys(self, points):
        for tp in points:
            for key in tp.attributes:
                self.assertFalse(
                    key.startswith("_"),
                    f"private key {key!r} leaked into attributes at "
                    f"row={tp.row}, col={tp.col}",
                )

    def test_cpla_attributes_are_clean_but_internal_carries_row_info(self, _mock_plot):
        rows_info = [
            (f"{v} km/h", "5 km/h", "AEB") for v in (10, 20, 30, 40, 50, 60)
        ] + [(f"{v} km/h", "5 km/h", "FCW") for v in (50, 60, 70, 80)]
        df = _build_matrix_df(rows_info, IL_VALUES)
        points = matrix_processing.get_test_matrix_from_region(
            df,
            start_row=2,
            n_rows=len(rows_info),
            start_col=6,
            n_cols=len(IL_VALUES),
            test_name="CPLA day",
            stage_subelement_key=data_model.StageSubelementKey.FC,
        )

        self._no_underscore_keys(points)

        choice_band_point = next(p for p in points if p.row == 6)
        fixed_band_point = next(p for p in points if p.row == 0)
        self.assertIn("_cpla_row_info", choice_band_point.internal)
        self.assertTrue(
            matrix_processing.is_fcw_declared_as_aeb(choice_band_point, "CPLA day")
        )
        self.assertIn("_cpla_row_info", fixed_band_point.internal)
        self.assertFalse(
            matrix_processing.is_fcw_declared_as_aeb(fixed_band_point, "CPLA day")
        )

    def test_cbla_attributes_are_clean_but_internal_carries_standard_ils(
        self, _mock_plot
    ):
        rows_info = [
            ("20 km/h", "15 km/h", "AEB"),
            ("20 km/h", "20 km/h", "AEB"),
        ]
        df = _build_matrix_df(rows_info, IL_VALUES)
        points = matrix_processing.get_test_matrix_from_region(
            df,
            start_row=2,
            n_rows=len(rows_info),
            start_col=6,
            n_cols=len(IL_VALUES),
            test_name="CBLA",
            stage_subelement_key=data_model.StageSubelementKey.FC,
        )

        self._no_underscore_keys(points)
        self.assertTrue(any("_cbla_standard_ils" in p.internal for p in points))


@patch("euroncap_rating_2026.crash_avoidance.matrix_processing.plot_matrix")
class TestGetTestMatrixFromRegionAttributesDf(unittest.TestCase):
    """
    Regression tests for the follow-up bug found while validating the fix
    against real production data: production always calls
    get_test_matrix_from_region with the bg-color-derived prediction sheet
    (common.load_sheet_with_bg_colors), which nulls out every non-color cell
    -- including the "Target speed"/"Impact location"/"Function" labels the
    CBLA/CPLA attribute-based logic reads. Attributes were always {} on that
    path, so check_extended_range's new branch never fired on filled workbooks and
    silently fell back to the (still wrong, for a shifted row count) static
    EXTENDED_RANGE_CELLS table. get_test_matrix_from_region now accepts a
    separate `attributes_df` for label lookups, decoupled from the
    color-bearing `prediction_df`.
    """

    # Fixed band is 6 rows here (not the static table's 5), same shape as the
    # sheet shape that exposed the bug.
    ROWS_INFO = [(f"{v} km/h", "15 km/h", "AEB") for v in (10, 20, 30, 40, 50, 60)] + [
        (f"{v} km/h", "20 km/h", "FCW") for v in (50, 60, 70, 80)
    ]

    def test_attributes_come_from_attributes_df_not_color_only_prediction_df(
        self, _mock_plot
    ):
        color_df = _build_color_only_df(self.ROWS_INFO, IL_VALUES)
        text_df = _build_matrix_df(self.ROWS_INFO, IL_VALUES)

        points = matrix_processing.get_test_matrix_from_region(
            color_df,
            start_row=2,
            n_rows=len(self.ROWS_INFO),
            start_col=6,
            n_cols=len(IL_VALUES),
            test_name="CBLA",
            stage_subelement_key=data_model.StageSubelementKey.FC,
            attributes_df=text_df,
        )
        grid = {(tp.row, tp.col): tp for tp in points}

        # Real labels from attributes_df, and the row-count-aware
        # classification kicks in (row 5 is fixed-band standard at 50% IL,
        # matching test_legacy_10_row_layout_resolves_correctly).
        self.assertEqual(grid[(5, 1)].attributes.get("Target speed"), "15 km/h")
        self.assertEqual(grid[(5, 1)].attributes.get("Impact location"), "50%")
        self.assertEqual(grid[(5, 1)].test_range, matrix_processing.TestRange.STANDARD)
        self.assertEqual(grid[(5, 2)].test_range, matrix_processing.TestRange.EXTENDED)

        # Colors still come from the color-only prediction_df, not attributes_df.
        self.assertEqual(grid[(5, 1)].color, common.PredictionColor.GREEN)

    def test_attributes_empty_and_static_fallback_used_when_attributes_df_omitted(
        self, _mock_plot
    ):
        color_df = _build_color_only_df(self.ROWS_INFO, IL_VALUES)

        points = matrix_processing.get_test_matrix_from_region(
            color_df,
            start_row=2,
            n_rows=len(self.ROWS_INFO),
            start_col=6,
            n_cols=len(IL_VALUES),
            test_name="CBLA",
            stage_subelement_key=data_model.StageSubelementKey.FC,
        )
        grid = {(tp.row, tp.col): tp for tp in points}

        # No attributes_df provided and prediction_df carries no label text
        # (mirrors the real bg-color-only sheet before the fix): attributes
        # are empty and the static EXTENDED_RANGE_CELLS table is used, which
        # is wrong for this 6-fixed-row layout (row 5 is actually fixed-band
        # standard, but the 5-row-calibrated static table marks it EXTENDED).
        self.assertEqual(grid[(5, 1)].attributes, {})
        self.assertEqual(grid[(5, 1)].test_range, matrix_processing.TestRange.EXTENDED)

    def test_colors_come_from_prediction_df_when_attributes_df_differs(
        self, _mock_plot
    ):
        color_df = _build_color_only_df(self.ROWS_INFO, IL_VALUES, color="red")
        text_df = _build_matrix_df(self.ROWS_INFO, IL_VALUES)

        points = matrix_processing.get_test_matrix_from_region(
            color_df,
            start_row=2,
            n_rows=len(self.ROWS_INFO),
            start_col=6,
            n_cols=len(IL_VALUES),
            test_name="CBLA",
            stage_subelement_key=data_model.StageSubelementKey.FC,
            attributes_df=text_df,
        )
        grid = {(tp.row, tp.col): tp for tp in points}

        # attributes_df's own matrix cells are "green", but color must still
        # follow prediction_df (the color-bearing df), i.e. RED.
        self.assertEqual(grid[(5, 1)].color, common.PredictionColor.RED)
        self.assertEqual(grid[(5, 1)].attributes.get("Target speed"), "15 km/h")


class TestAebDeclaredAsFcwClassification(unittest.TestCase):
    """
    Covers matrix_processing.is_fcw_declared_as_aeb, used by ComputedTestPoint
    to decide whether an AEB-labeled CPLA/CBLA point should be scored as
    avoidance (FCW declared as AEB) or mitigation-or-avoidance (original AEB).
    """

    def _computed_color(
        self, test_name, attributes, total_rows=None, row=0, value=15.0
    ):
        tp = matrix_processing.TestPoint(
            row=row,
            col=0,
            color=common.PredictionColor.GREEN,
            test_range=matrix_processing.TestRange.STANDARD,
            attributes=attributes,
            total_rows=total_rows,
        )
        ctp = matrix_processing.ComputedTestPoint(
            test_name=test_name, test_point=tp, value=value
        )
        return ctp.computed_color

    def test_cbla_choice_band_target_speed_is_scored_as_avoidance(self):
        color = self._computed_color(
            "CBLA",
            {"Function": "AEB", "Target speed": "20 km/h", "VUT speed": "60 km/h"},
        )
        # Avoidance thresholds: any value > 0 is RED.
        self.assertEqual(color, common.PredictionColor.RED)

    def test_cbla_fixed_band_target_speed_is_scored_as_mitigation_or_avoidance(self):
        color = self._computed_color(
            "CBLA",
            {"Function": "AEB", "Target speed": "15 km/h", "VUT speed": "60 km/h"},
        )
        # Mitigation-or-avoidance thresholds at (capped) 50 km/h: value=15 -> ORANGE.
        self.assertEqual(color, common.PredictionColor.ORANGE)

    def test_cpla_choice_band_row_is_scored_as_avoidance(self):
        color = self._computed_color(
            "CPLA day",
            {"Function": "AEB", "Target speed": "5 km/h", "VUT speed": "60 km/h"},
            total_rows=10,
            row=6,  # last 4 rows of a 10-row matrix = choice band
        )
        self.assertEqual(color, common.PredictionColor.RED)

    def test_cpla_fixed_band_row_is_scored_as_mitigation_or_avoidance(self):
        color = self._computed_color(
            "CPLA day",
            {"Function": "AEB", "Target speed": "5 km/h", "VUT speed": "60 km/h"},
            total_rows=10,
            row=5,  # outside the last 4 rows = fixed band
        )
        self.assertEqual(color, common.PredictionColor.ORANGE)

    def test_cpla_falls_back_to_legacy_row_index_when_total_rows_unknown(self):
        # FCW_AEB_ROW must cover the whole 4-row choice band (rows 6-9 of the
        # current 10-row template), not just its first two rows -- a partial
        # list would silently misclassify rows 8-9 as original AEB instead of
        # FCW-declared-as-AEB whenever total_rows is unavailable.
        for row in (6, 7, 8, 9):
            avoidance_color = self._computed_color(
                "CPLA day",
                {"Function": "AEB", "Target speed": "5 km/h", "VUT speed": "60 km/h"},
                total_rows=None,
                row=row,
            )
            self.assertEqual(avoidance_color, common.PredictionColor.RED, f"row={row}")
        mitigation_color = self._computed_color(
            "CPLA day",
            {"Function": "AEB", "Target speed": "5 km/h", "VUT speed": "60 km/h"},
            total_rows=None,
            row=4,  # not in legacy FCW_AEB_ROW fallback
        )
        self.assertEqual(mitigation_color, common.PredictionColor.ORANGE)


class TestReadTestPointsThreadsTotalRows(unittest.TestCase):
    def test_total_rows_argument_is_set_on_reconstructed_test_points(self):
        df = pd.DataFrame(
            [
                {
                    "Scenario": "CPLA day",
                    "Test point": "(6, 1)",
                    "Range": "Standard",
                    "OEM Prediction": "Green",
                    "Function": "AEB",
                    "Target speed": "5 km/h",
                    "VUT speed": "60 km/h",
                }
            ]
        )
        test_points, _ = test_info.read_test_points(df, total_rows=10)
        self.assertEqual(test_points[0].total_rows, 10)

    def test_total_rows_defaults_to_none(self):
        df = pd.DataFrame(
            [
                {
                    "Scenario": "CPLA day",
                    "Test point": "(6, 1)",
                    "Range": "Standard",
                    "OEM Prediction": "Green",
                }
            ]
        )
        test_points, _ = test_info.read_test_points(df)
        self.assertIsNone(test_points[0].total_rows)


if __name__ == "__main__":
    unittest.main()
