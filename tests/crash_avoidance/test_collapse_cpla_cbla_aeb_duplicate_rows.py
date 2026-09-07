# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Tests for test_info.collapse_cpla_cbla_aeb_duplicate_rows: the
collapse-and-promote rule replacing the old reject-based
check_cpla_cbla_aeb_impact_location. Per speed (50, 60 km/h independently),
if the choice band's duplicate row declares AEB it is deleted entirely; a
duplicate left on FCW is a genuine warning test and is kept.
"""

import unittest
from unittest.mock import patch

import pandas as pd

from euroncap_rating_2026 import common
from euroncap_rating_2026.crash_avoidance import data_model
from euroncap_rating_2026.crash_avoidance import matrix_processing
from euroncap_rating_2026.crash_avoidance import test_info

IL_VALUES = [0.75, 0.5, 0.25, 0.10]
IL_LABELS = ["75%", "50%", "25%", "10%"]


def _build_sheet_df(rows, test_name, start_row=2, start_col=6):
    """
    Build a DataFrame shaped like a real CPLA/CBLA prediction sheet region,
    including the matrix title cell (needed for
    matrix_processing.get_matrix_indices/common.get_start_row to locate it
    by searching, exactly like collapse_cpla_cbla_aeb_duplicate_rows does).

    rows: list of dicts with keys "vut_speed", "target_speed", "function",
        and "colors" (dict of IL label -> color string; a label absent from
        "colors" is left blank/uncolored, not GREY -- see
        test_grey_cell_is_not_collapsed for an explicit GREY cell).
    """
    n_rows = len(rows)
    n_cols = len(IL_VALUES)
    total_rows = start_row + n_rows
    total_cols = start_col + n_cols
    data = [[None] * total_cols for _ in range(total_rows)]

    title_row = start_row - 2
    il_header_row = start_row - 1
    data[title_row][0] = test_name
    data[title_row][1] = "Target speed"
    data[title_row][2] = "Function"
    data[title_row][start_col] = "Impact location"
    for j, il in enumerate(IL_VALUES):
        data[il_header_row][start_col + j] = il

    for i, row in enumerate(rows):
        data[start_row + i][0] = row["vut_speed"]
        data[start_row + i][1] = row["target_speed"]
        data[start_row + i][2] = row["function"]
        for j, label in enumerate(IL_LABELS):
            color = row.get("colors", {}).get(label)
            if color is not None:
                data[start_row + i][start_col + j] = color

    return pd.DataFrame(data)


def _get_test_points(df, test_name, n_rows, start_row=2, start_col=6):
    return matrix_processing.get_test_matrix_from_region(
        df,
        start_row=start_row,
        n_rows=n_rows,
        start_col=start_col,
        n_cols=len(IL_VALUES),
        test_name=test_name,
        stage_subelement_key=data_model.StageSubelementKey.FC,
    )


# Fixed band rows 0-5 (10, 20, 30, 40, 50, 60 km/h @ Target speed 5 km/h,
# always Function=AEB, colored at 50% IL like the real template); choice
# band rows 6-9 (50, 60, 70, 80 km/h) whose Function is supplied per test.
def _cpla_fixed_rows():
    return [
        {
            "vut_speed": f"{v} km/h",
            "target_speed": "5 km/h",
            "function": "AEB",
            "colors": {"50%": "green"},
        }
        for v in (10, 20, 30, 40, 50, 60)
    ]


@patch("euroncap_rating_2026.crash_avoidance.matrix_processing.plot_matrix")
class TestCollapseCpla(unittest.TestCase):
    def _collapse(self, choice_rows, test_name="CPLA day"):
        rows = _cpla_fixed_rows() + choice_rows
        df = _build_sheet_df(rows, test_name)
        test_points = _get_test_points(df, test_name, n_rows=len(rows))
        return (
            test_info.collapse_cpla_cbla_aeb_duplicate_rows(df, test_points, test_name),
            df,
            rows,
        )

    def test_both_speeds_aeb_removes_both_choice_rows(self, _mock_plot):
        choice_rows = [
            {
                "vut_speed": "50 km/h",
                "target_speed": "5 km/h",
                "function": "AEB",
                "colors": {"25%": "green"},
            },
            {
                "vut_speed": "60 km/h",
                "target_speed": "5 km/h",
                "function": "AEB",
                "colors": {"25%": "green"},
            },
            {
                "vut_speed": "70 km/h",
                "target_speed": "5 km/h",
                "function": "AEB",
                "colors": {"25%": "green"},
            },
            {
                "vut_speed": "80 km/h",
                "target_speed": "5 km/h",
                "function": "AEB",
                "colors": {"25%": "green"},
            },
        ]
        (updated_df, removed_rows, matrix_start_row), df, rows = self._collapse(
            choice_rows
        )
        self.assertEqual(removed_rows, [6, 7])
        self.assertEqual(matrix_start_row, 2)
        self.assertEqual(len(updated_df), len(df) - 2)

        # The surviving matrix (re-extracted from updated_df) has 8 rows:
        # fixed 10-60, then 70, 80 -- and the 50/60 fixed rows are now
        # promoted to standard at both 50% and 25% IL.
        points = _get_test_points(updated_df, "CPLA day", n_rows=8)
        grid = {(tp.row, tp.col): tp for tp in points}
        self.assertEqual(
            grid[(4, 1)].test_range, matrix_processing.TestRange.STANDARD
        )  # 50 km/h @ 50%
        self.assertEqual(
            grid[(4, 2)].test_range, matrix_processing.TestRange.STANDARD
        )  # 50 km/h @ 25%, promoted
        self.assertEqual(grid[(6, 2)].color, common.PredictionColor.GREEN)  # 70 km/h

    def test_one_aeb_one_fcw_removes_only_the_aeb_speed(self, _mock_plot):
        choice_rows = [
            {
                "vut_speed": "50 km/h",
                "target_speed": "5 km/h",
                "function": "AEB",
                "colors": {"25%": "green"},
            },
            {
                "vut_speed": "60 km/h",
                "target_speed": "5 km/h",
                "function": "FCW",
                "colors": {"25%": "green"},
            },
            {
                "vut_speed": "70 km/h",
                "target_speed": "5 km/h",
                "function": "AEB",
                "colors": {"25%": "green"},
            },
            {
                "vut_speed": "80 km/h",
                "target_speed": "5 km/h",
                "function": "AEB",
                "colors": {"25%": "green"},
            },
        ]
        (updated_df, removed_rows, matrix_start_row), df, rows = self._collapse(
            choice_rows
        )
        self.assertEqual(removed_rows, [6])
        self.assertEqual(len(updated_df), len(df) - 1)

    def test_all_fcw_is_a_no_op(self, _mock_plot):
        choice_rows = [
            {
                "vut_speed": v,
                "target_speed": "5 km/h",
                "function": "FCW",
                "colors": {"25%": "green"},
            }
            for v in ("50 km/h", "60 km/h", "70 km/h", "80 km/h")
        ]
        (updated_df, removed_rows, matrix_start_row), df, rows = self._collapse(
            choice_rows
        )
        self.assertEqual(removed_rows, [])
        self.assertIs(updated_df, df)

    def test_already_collapsed_matrix_is_a_no_op(self, _mock_plot):
        """No duplicate present at all (8-row shape) -- nothing to collapse."""
        rows = _cpla_fixed_rows() + [
            {
                "vut_speed": "70 km/h",
                "target_speed": "5 km/h",
                "function": "AEB",
                "colors": {"25%": "green"},
            },
            {
                "vut_speed": "80 km/h",
                "target_speed": "5 km/h",
                "function": "AEB",
                "colors": {"25%": "green"},
            },
        ]
        df = _build_sheet_df(rows, "CPLA day")
        test_points = _get_test_points(df, "CPLA day", n_rows=len(rows))
        updated_df, removed_rows, matrix_start_row = (
            test_info.collapse_cpla_cbla_aeb_duplicate_rows(df, test_points, "CPLA day")
        )
        self.assertEqual(removed_rows, [])
        self.assertIs(updated_df, df)

    def test_grey_cell_is_not_collapsed(self, _mock_plot):
        """A choice-band row with no color at all produces no TestPoint (see
        get_test_matrix_from_region), so there's nothing to match/collapse --
        the row survives untouched."""
        choice_rows = [
            {
                "vut_speed": "50 km/h",
                "target_speed": "5 km/h",
                "function": "AEB",
                "colors": {},
            },  # not selected by the OEM
            {
                "vut_speed": "60 km/h",
                "target_speed": "5 km/h",
                "function": "AEB",
                "colors": {"25%": "green"},
            },
            {
                "vut_speed": "70 km/h",
                "target_speed": "5 km/h",
                "function": "AEB",
                "colors": {"25%": "green"},
            },
            {
                "vut_speed": "80 km/h",
                "target_speed": "5 km/h",
                "function": "AEB",
                "colors": {"25%": "green"},
            },
        ]
        (updated_df, removed_rows, matrix_start_row), df, rows = self._collapse(
            choice_rows
        )
        self.assertEqual(removed_rows, [7])  # only 60 km/h's row is collapsed


@patch("euroncap_rating_2026.crash_avoidance.matrix_processing.plot_matrix")
class TestCollapseCbla(unittest.TestCase):
    def _fixed_rows(self):
        return [
            {
                "vut_speed": f"{v} km/h",
                "target_speed": "15 km/h",
                "function": "AEB",
                "colors": {"50%": "green"},
            }
            for v in (20, 30, 40, 50, 60)
        ]

    def _collapse(self, choice_rows):
        rows = self._fixed_rows() + choice_rows
        df = _build_sheet_df(rows, "CBLA")
        test_points = _get_test_points(df, "CBLA", n_rows=len(rows))
        return (
            test_info.collapse_cpla_cbla_aeb_duplicate_rows(df, test_points, "CBLA"),
            df,
        )

    def test_both_speeds_aeb_removes_both_choice_rows(self, _mock_plot):
        choice_rows = [
            {
                "vut_speed": f"{v} km/h",
                "target_speed": "20 km/h",
                "function": "AEB",
                "colors": {"25%": "green"},
            }
            for v in (50, 60, 70, 80)
        ]
        (updated_df, removed_rows, matrix_start_row), df = self._collapse(choice_rows)
        self.assertEqual(removed_rows, [5, 6])
        self.assertEqual(len(updated_df), len(df) - 2)

        points = _get_test_points(updated_df, "CBLA", n_rows=7)
        grid = {(tp.row, tp.col): tp for tp in points}
        self.assertEqual(
            grid[(3, 1)].test_range, matrix_processing.TestRange.STANDARD
        )  # 50 km/h @ 50%, still standard
        self.assertEqual(
            grid[(3, 2)].test_range, matrix_processing.TestRange.STANDARD
        )  # 50 km/h @ 25%, promoted

    def test_one_aeb_one_fcw_removes_only_the_aeb_speed(self, _mock_plot):
        choice_rows = [
            {
                "vut_speed": "50 km/h",
                "target_speed": "20 km/h",
                "function": "AEB",
                "colors": {"25%": "green"},
            },
            {
                "vut_speed": "60 km/h",
                "target_speed": "20 km/h",
                "function": "FCW",
                "colors": {"25%": "green"},
            },
            {
                "vut_speed": "70 km/h",
                "target_speed": "20 km/h",
                "function": "AEB",
                "colors": {"25%": "green"},
            },
            {
                "vut_speed": "80 km/h",
                "target_speed": "20 km/h",
                "function": "AEB",
                "colors": {"25%": "green"},
            },
        ]
        (updated_df, removed_rows, matrix_start_row), df = self._collapse(choice_rows)
        self.assertEqual(removed_rows, [5])


if __name__ == "__main__":
    unittest.main()
