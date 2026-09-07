# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Regression test for the row-shift bookkeeping in
test_info.preprocess_stage_subelement: CPLA day, CPLA night, and CBLA all
share one sheet ("FC - Ped & Cyc pred."), top to bottom. When CPLA day's
collapse removes rows, CBLA's matrix is relocated (by
matrix_processing.get_matrix_indices' label search) correctly in the
already-shrunk in-memory DataFrame -- but the worksheet rows reported in
StageSubelement.removed_prediction_rows must be translated back to the
*original* (pre-collapse) worksheet's coordinates, since report_writer later
replays the deletion on an untouched copy of that sheet. Getting this wrong
silently deletes the wrong physical rows.
"""

import unittest
from importlib.resources import files
from unittest.mock import patch

from euroncap_rating_2026 import common
from euroncap_rating_2026.crash_avoidance import data_model
from euroncap_rating_2026.crash_avoidance import test_info

PREDICTION_SHEET = "FC - Ped & Cyc pred."

# Column indices within the color matrix (start_col=6): 75%, 50%, 25%, 10%.
IL_25_PERCENT_COL = 8
FUNCTION_COL = 2


def _declare_choice_band_aeb(df, df_row, color="green"):
    # The pristine template's color columns are all-NaN (float64 dtype) --
    # cast to object first so assigning a color string doesn't trip
    # pandas' FutureWarning about incompatible-dtype item assignment.
    if df.dtypes.iloc[IL_25_PERCENT_COL] != object:
        df.isetitem(IL_25_PERCENT_COL, df.iloc[:, IL_25_PERCENT_COL].astype(object))
    df.iloc[df_row, FUNCTION_COL] = "AEB"
    df.iloc[df_row, IL_25_PERCENT_COL] = color


class TestCplaCblaCollapseRowShift(unittest.TestCase):
    def setUp(self):
        pristine_path = str(files("data").joinpath("ca_template.xlsx"))
        self.dfs = common.read_excel_file_to_dfs(pristine_path)

    def test_cpla_day_collapse_does_not_corrupt_cbla_row_numbers(self):
        df = self.dfs[PREDICTION_SHEET]

        # CPLA day's choice-band rows for 50/60 km/h are DataFrame rows 7, 8
        # (matrix start_row=1, matrix-relative rows 6, 7) -- see
        # matrix_processing.get_start_row's special case for a sheet's first
        # matrix, whose title is absorbed into the column header.
        _declare_choice_band_aeb(df, 7)  # 50 km/h
        _declare_choice_band_aeb(df, 8)  # 60 km/h

        # CBLA's choice-band rows for 50/60 km/h are DataFrame rows 32, 33
        # (matrix start_row=27, matrix-relative rows 5, 6).
        _declare_choice_band_aeb(df, 32)  # 50 km/h
        _declare_choice_band_aeb(df, 33)  # 60 km/h

        with patch(
            "euroncap_rating_2026.crash_avoidance.test_info.get_vehicle_response",
            return_value=data_model.VehicleResponse.WARNING,
        ):
            result = test_info.preprocess_stage_subelement(
                self.dfs, "Frontal Collisions", "Pedestrian & cyclist"
            )

        removed_rows = result.removed_prediction_rows[PREDICTION_SHEET]

        # Worksheet (1-based, pre-collapse) rows: CPLA day's 50/60 choice
        # rows are 9, 10; CBLA's are 34, 35 -- both in *original* sheet
        # coordinates, unaffected by CPLA day's collapse happening first.
        self.assertEqual(sorted(removed_rows), [9, 10, 34, 35])

    def test_only_cbla_collapses_reports_its_own_original_rows(self):
        """Sanity check isolating CBLA: with nothing collapsed above it,
        its rows must still be reported correctly (no accidental off-by-N
        from the offset bookkeeping when the offset is zero)."""
        df = self.dfs[PREDICTION_SHEET]
        _declare_choice_band_aeb(df, 32)  # CBLA 50 km/h
        _declare_choice_band_aeb(df, 33)  # CBLA 60 km/h

        with patch(
            "euroncap_rating_2026.crash_avoidance.test_info.get_vehicle_response",
            return_value=data_model.VehicleResponse.WARNING,
        ):
            result = test_info.preprocess_stage_subelement(
                self.dfs, "Frontal Collisions", "Pedestrian & cyclist"
            )

        removed_rows = result.removed_prediction_rows[PREDICTION_SHEET]
        self.assertEqual(sorted(removed_rows), [34, 35])


if __name__ == "__main__":
    unittest.main()
