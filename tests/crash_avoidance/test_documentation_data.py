# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for documentation_data.py's Excel-formula -> pandas translation.

The setup path for every trigger/boundary test below writes cells via
openpyxl using the exact same A1 coordinates that appear in the source
Excel formulas -- independent of documentation_data's own name/identity
based lookups (_input_parameters_has_vta, _robust_pred_has_yes,
_matrix_edge_slice) -- so a test passing is a real cross-check of those
lookups, not just a restatement of them.
"""

import os
import tempfile
import unittest
from importlib.resources import files

import openpyxl
import pandas as pd

from euroncap_rating_2026 import common
from euroncap_rating_2026.crash_avoidance import documentation_data


def _dfs_with_cells_set(pristine_path, sheet_cell_values):
    """Load the pristine template, set (sheet, a1_coordinate) -> value pairs
    via openpyxl, save to a temp file, and return it read back as dfs."""
    wb = openpyxl.load_workbook(pristine_path)
    for sheet_name, coord, value in sheet_cell_values:
        wb[sheet_name][coord] = value
    fd, path = tempfile.mkstemp(suffix=".xlsx")
    os.close(fd)
    try:
        wb.save(path)
        return common.read_excel_file_to_dfs(path)
    finally:
        os.remove(path)


class TestMatrixEdgeSlice(unittest.TestCase):
    """_matrix_edge_slice locates a named test's matrix via
    matrix_processing.get_matrix_indices (no hardcoded position) and
    returns the requested edge (last_rows/last_cols/first_rows). These
    tests confirm the returned edge lines up with the exact same absolute
    cells the old hardcoded ranges pointed to, for one test of each edge
    shape used by the real checks."""

    def setUp(self):
        self.pristine_path = str(files("data").joinpath("ca_template.xlsx"))

    def test_last_rows_all_cols(self):
        # CCRb's matrix is D28:J38 (11 rows); "last_rows": 3 -> D36:J38.
        sheet_name = "FC - Car & PTW pred."
        dfs = _dfs_with_cells_set(
            self.pristine_path,
            [(sheet_name, "D36", "sentinel-1"), (sheet_name, "J38", "sentinel-2")],
        )
        sliced = documentation_data._matrix_edge_slice(
            dfs[sheet_name], "CCRb", last_rows=3
        )
        self.assertEqual(sliced.shape, (3, 7))
        self.assertEqual(sliced.iloc[0, 0], "sentinel-1")
        self.assertEqual(sliced.iloc[2, 6], "sentinel-2")

    def test_last_cols_single_row(self):
        # CMFtap SfS's matrix is C13:F13 (1 row); "last_cols": 1 -> F15 (=F13 data row).
        sheet_name = "LSC - Car & PTW pred."
        dfs = _dfs_with_cells_set(self.pristine_path, [(sheet_name, "F15", "Green")])
        sliced = documentation_data._matrix_edge_slice(
            dfs[sheet_name], "CMFtap SfS", last_cols=1
        )
        self.assertEqual(sliced.shape, (1, 1))
        self.assertEqual(sliced.iloc[0, 0], "Green")

    def test_first_rows_last_cols(self):
        # CCCscp's matrix is D71:J77 (7 rows); "first_rows": 2, "last_cols": 2 -> I71:J72.
        sheet_name = "FC - Car & PTW pred."
        dfs = _dfs_with_cells_set(
            self.pristine_path,
            [(sheet_name, "I71", "sentinel-1"), (sheet_name, "J72", "sentinel-2")],
        )
        sliced = documentation_data._matrix_edge_slice(
            dfs[sheet_name], "CCCscp", first_rows=2, last_cols=2
        )
        self.assertEqual(sliced.shape, (2, 2))
        self.assertEqual(sliced.iloc[0, 0], "sentinel-1")
        self.assertEqual(sliced.iloc[1, 1], "sentinel-2")

    def test_unknown_test_name_returns_empty(self):
        dfs = common.read_excel_file_to_dfs(self.pristine_path)
        sliced = documentation_data._matrix_edge_slice(
            dfs["FC - Car & PTW pred."], "NoSuchTest", last_rows=1
        )
        self.assertEqual(sliced.shape, (0, 0))


class TestInputParametersSemanticMatching(unittest.TestCase):
    """_input_parameters_has_vta matches by Stage element + Input parameter
    identity (forward-filled), not by a hardcoded Excel range -- these tests
    exercise that directly against a synthetic 'Input parameters' frame,
    including a row-insertion scenario a hardcoded range (e.g. "G2:G43")
    could not have survived."""

    @staticmethod
    def _input_parameters_df(rows):
        return pd.DataFrame(
            rows,
            columns=[
                "Stage",
                "Stage element",
                "Stage subelement",
                "Category",
                "Scenario",
                "Input parameter",
                "Value",
            ],
        )

    def test_forward_fill_and_stage_element_match(self):
        dfs = {
            "Input parameters": self._input_parameters_df(
                [
                    [
                        "Crash Avoidance",
                        "Frontal Collisions",
                        "Car & PTW",
                        "Longitudinal",
                        "CCRs",
                        "Prediction - Standard",
                        None,
                    ],
                    [None, None, None, None, None, "Prediction - Extended", "VTA"],
                    [
                        None,
                        "Lane Departure Collisions",
                        "Single vehicle",
                        "Driver acceptance",
                        "Driveability",
                        "Heading correction",
                        None,
                    ],
                ]
            )
        }
        self.assertTrue(
            documentation_data._input_parameters_has_vta(dfs, "Frontal Collisions")
        )
        self.assertFalse(
            documentation_data._input_parameters_has_vta(
                dfs, "Lane Departure Collisions"
            )
        )

    def test_extra_inserted_row_does_not_shift_the_match(self):
        """Inserting a brand new row ahead of the Frontal Collisions block
        (e.g. a future template revision adding a parameter) must not
        change the outcome -- a hardcoded row range would have silently
        started reading the wrong rows instead."""
        dfs = {
            "Input parameters": self._input_parameters_df(
                [
                    [
                        "Crash Avoidance",
                        "Some Future Stage",
                        "X",
                        "Y",
                        "Z",
                        "Prediction - Standard",
                        None,
                    ],
                    [None, None, None, None, None, "Prediction - Extended", None],
                    [
                        None,
                        "Frontal Collisions",
                        "Car & PTW",
                        "Longitudinal",
                        "CCRs",
                        "Prediction - Standard",
                        "VTA",
                    ],
                ]
            )
        }
        self.assertTrue(
            documentation_data._input_parameters_has_vta(dfs, "Frontal Collisions")
        )
        self.assertFalse(
            documentation_data._input_parameters_has_vta(dfs, "Some Future Stage")
        )

    def test_non_prediction_input_parameter_under_target_stage_is_ignored(self):
        dfs = {
            "Input parameters": self._input_parameters_df(
                [
                    [
                        "Crash Avoidance",
                        "Lane Departure Collisions",
                        "Single vehicle",
                        "Driver acceptance",
                        "Driveability",
                        "Heading correction",
                        "VTA",
                    ],
                ]
            )
        }
        self.assertFalse(
            documentation_data._input_parameters_has_vta(
                dfs, "Lane Departure Collisions"
            )
        )


class TestComputeDocumentationBaseline(unittest.TestCase):
    def test_pristine_input_is_all_false(self):
        pristine_path = str(files("data").joinpath("ca_template.xlsx"))
        dfs = common.read_excel_file_to_dfs(pristine_path)
        doc_df = documentation_data.compute_documentation(dfs)
        self.assertEqual(list(doc_df["Status"]), ["FALSE"] * 4)
        data_df = documentation_data.compute_data(dfs)
        self.assertEqual(list(data_df["Status"]), ["FALSE"] * 5)


class TestComputeDocumentationTriggers(unittest.TestCase):
    """One positive-trigger test per Documentation-sheet row: setting the
    triggering value flips exactly that row to TRUE, all others stay FALSE."""

    def setUp(self):
        self.pristine_path = str(files("data").joinpath("ca_template.xlsx"))

    def _assert_only_row_true(self, doc_df, row_id):
        statuses = dict(zip(doc_df["Documentation ID"], doc_df["Status"]))
        for rid, status in statuses.items():
            expected = "TRUE" if rid == row_id else "FALSE"
            self.assertEqual(status, expected, f"row {rid}")

    def test_fc_vta_dossier(self):
        dfs = _dfs_with_cells_set(
            self.pristine_path, [("Input parameters", "G2", "VTA")]
        )
        self._assert_only_row_true(
            documentation_data.compute_documentation(dfs), "PR_CA-FC-VTADossier"
        )

    def test_fc_vta_dossier_is_case_insensitive(self):
        dfs = _dfs_with_cells_set(
            self.pristine_path, [("Input parameters", "G43", "vta")]
        )
        self._assert_only_row_true(
            documentation_data.compute_documentation(dfs), "PR_CA-FC-VTADossier"
        )

    def test_fc_perception_from_car_ptw_robust(self):
        dfs = _dfs_with_cells_set(
            self.pristine_path, [("FC - Car & PTW robust. pred.", "C7", "YES")]
        )
        self._assert_only_row_true(
            documentation_data.compute_documentation(dfs), "PR_CA-FC-Perception"
        )

    def test_fc_perception_from_ped_cyc_robust(self):
        dfs = _dfs_with_cells_set(
            self.pristine_path, [("FC - Ped & Cyc robust. pred.", "L14", "YES")]
        )
        self._assert_only_row_true(
            documentation_data.compute_documentation(dfs), "PR_CA-FC-Perception"
        )

    def test_ldc_vta_dossier(self):
        dfs = _dfs_with_cells_set(
            self.pristine_path, [("Input parameters", "G55", "VTA")]
        )
        self._assert_only_row_true(
            documentation_data.compute_documentation(dfs), "PR_CA-LDC-VTADossier"
        )

    def test_ldc_vta_dossier_gap_rows_are_excluded(self):
        """Excel rows 44 ("Heading correction") and 47 ("Extended range
        performance") sit between the two VTA ranges (G45:G46, G48:G55) and
        must NOT trigger the flag even if set to "VTA"."""
        dfs = _dfs_with_cells_set(
            self.pristine_path,
            [("Input parameters", "G44", "VTA"), ("Input parameters", "G47", "VTA")],
        )
        self._assert_only_row_true(documentation_data.compute_documentation(dfs), None)

    def test_ldc_perception(self):
        dfs = _dfs_with_cells_set(
            self.pristine_path, [("LDC - robust. pred.", "G9", "Yes")]
        )
        self._assert_only_row_true(
            documentation_data.compute_documentation(dfs), "PR_CA-LDC-Perception"
        )

    def test_fc_perception_ignores_vut_target_kinematic_rows(self):
        """Row 3 ("Speed") and row 6 ("Trajectory/Heading") are VUT/Target
        kinematic rows, not Type/Appearance/Environment -- "YES" there must
        not trigger PR_CA-FC-Perception."""
        dfs = _dfs_with_cells_set(
            self.pristine_path,
            [
                ("FC - Car & PTW robust. pred.", "C3", "YES"),
                ("FC - Car & PTW robust. pred.", "M6", "YES"),
            ],
        )
        self._assert_only_row_true(documentation_data.compute_documentation(dfs), None)


class TestComputeDataTriggers(unittest.TestCase):
    """One positive-trigger test per Data-sheet row."""

    def setUp(self):
        self.pristine_path = str(files("data").joinpath("ca_template.xlsx"))

    def _assert_only_row_true(self, data_df, row_id):
        statuses = dict(zip(data_df["Data ID"], data_df["Status"]))
        for rid, status in statuses.items():
            expected = "TRUE" if rid == row_id else "FALSE"
            self.assertEqual(status, expected, f"row {rid}")

    def test_fc_vta_data(self):
        dfs = _dfs_with_cells_set(
            self.pristine_path, [("Input parameters", "G10", "VTA")]
        )
        self._assert_only_row_true(
            documentation_data.compute_data(dfs), "PR_CA-FC-VTAData"
        )

    def test_fc_corner_first_range(self):
        dfs = _dfs_with_cells_set(
            self.pristine_path, [("FC - Car & PTW pred.", "D36", "Green")]
        )
        self._assert_only_row_true(
            documentation_data.compute_data(dfs), "PR_CA-FC-Corner"
        )

    def test_fc_corner_last_range_yellow(self):
        dfs = _dfs_with_cells_set(
            self.pristine_path, [("FC - Car & PTW pred.", "I113", "Yellow")]
        )
        self._assert_only_row_true(
            documentation_data.compute_data(dfs), "PR_CA-FC-Corner"
        )

    def test_fc_corner_ignores_red_and_grey(self):
        """PR_CA-FC-Corner only counts Green/Yellow/Orange/Brown -- Red and
        Grey (not a "scored" color) must not trigger it."""
        dfs = _dfs_with_cells_set(
            self.pristine_path,
            [
                ("FC - Car & PTW pred.", "D36", "Red"),
                ("FC - Car & PTW pred.", "D37", "Grey"),
            ],
        )
        self._assert_only_row_true(documentation_data.compute_data(dfs), None)

    def test_fc_corner_cell_outside_range_does_not_trigger(self):
        dfs = _dfs_with_cells_set(
            self.pristine_path,
            [
                ("FC - Car & PTW pred.", "D35", "Green"),  # one row above D36:J38
                ("FC - Car & PTW pred.", "K36", "Green"),  # one col right of D36:J38
            ],
        )
        self._assert_only_row_true(documentation_data.compute_data(dfs), None)

    def test_ldc_vta_data(self):
        dfs = _dfs_with_cells_set(
            self.pristine_path, [("Input parameters", "G46", "VTA")]
        )
        self._assert_only_row_true(
            documentation_data.compute_data(dfs), "PR_CA-LDC-VTAData"
        )

    def test_ldc_corner_single_veh(self):
        dfs = _dfs_with_cells_set(
            self.pristine_path, [("LDC - Single Veh pred.", "B7", "Green")]
        )
        self._assert_only_row_true(
            documentation_data.compute_data(dfs), "PR_CA-LDC-Corner"
        )

    def test_ldc_corner_car_ptw_last_range(self):
        dfs = _dfs_with_cells_set(
            self.pristine_path, [("LDC - Car & PTW pred.", "I57", "Green")]
        )
        self._assert_only_row_true(
            documentation_data.compute_data(dfs), "PR_CA-LDC-Corner"
        )

    def test_ldc_corner_ignores_non_green(self):
        dfs = _dfs_with_cells_set(
            self.pristine_path, [("LDC - Single Veh pred.", "B7", "Yellow")]
        )
        self._assert_only_row_true(documentation_data.compute_data(dfs), None)

    def test_lsc_corner_row_range(self):
        dfs = _dfs_with_cells_set(
            self.pristine_path, [("LSC - Car & PTW pred.", "H7", "Green")]
        )
        self._assert_only_row_true(
            documentation_data.compute_data(dfs), "PR_CA-LSC-Corner"
        )

    def test_lsc_corner_single_cell(self):
        dfs = _dfs_with_cells_set(
            self.pristine_path, [("LSC - Car & PTW pred.", "F15", "Green")]
        )
        self._assert_only_row_true(
            documentation_data.compute_data(dfs), "PR_CA-LSC-Corner"
        )


if __name__ == "__main__":
    unittest.main()
