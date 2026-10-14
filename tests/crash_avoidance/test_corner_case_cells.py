# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""data_model.CORNER_CASE_CELLS must enumerate exactly the cells of the packaged
ca_template.xlsx matrices that the protocols' Corner Case clauses select
(CA Frontal Collisions 4.1.3, Lane Departure Collisions 4.1.3, Low Speed
Collisions 4.1.2), expressed with the run attributes preprocess writes."""

import os
import unittest

import pandas as pd

from euroncap_rating_2026.crash_avoidance import data_model, matrix_processing

TEMPLATE = os.path.join(
    os.path.dirname(data_model.__file__), "..", "..", "data", "ca_template.xlsx"
)

SHEET_OF_TEST = {
    **{
        t: "FC - Car & PTW pred."
        for t in (
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
        )
    },
    "ELK RE": "LDC - Single Veh pred.",
    **{
        t: "LDC - Car & PTW pred."
        for t in (
            "CC ELK On",
            "CC ELK OvU",
            "CC ELK OvI",
            "CM ELK On",
            "CM ELK OvU",
            "CM ELK OvI",
        )
    },
    **{
        t: "LSC - Car & PTW pred."
        for t in ("CCCscp SfS", "CCFtap SfS", "CMCscp SfS", "CMFtap SfS")
    },
}


def _kmh(value):
    return float(str(value).replace("km/h", "").strip())


def _is_corner_case(test_name: str, attrs: dict) -> bool:
    """The protocol clauses, applied to one cell's run attributes."""
    if test_name in ("CCFhos", "CCFhol"):
        return True
    if test_name == "CCRb":
        return _kmh(attrs["Target speed"]) > 100
    if test_name == "CMRb" or test_name.startswith("CM ELK"):
        return _kmh(attrs["Target speed"]) > 80
    if test_name.startswith("CC ELK"):
        # Not in the printed LDC 4.1.3 list; Euro NCAP confirmed the omission is
        # a typo and the CCRb threshold applies.
        return _kmh(attrs["Target speed"]) > 100
    if test_name == "ELK RE":
        return _kmh(attrs["VUT speed"]) > 80
    if test_name.split(" ")[0] in ("CCFtap", "CMFtap", "CCCscp", "CMCscp"):
        return _kmh(attrs["Target speed"]) > 60
    return False


class TestCornerCaseCells(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        xls = pd.ExcelFile(TEMPLATE, engine_kwargs={"read_only": False})
        sheets = {
            name: pd.read_excel(xls, sheet_name=name, header=0)
            for name in set(SHEET_OF_TEST.values())
        }
        xls.close()
        cls.template_cells = {}
        for test_name, sheet_name in SHEET_OF_TEST.items():
            df = sheets[sheet_name]
            idx = matrix_processing.get_matrix_indices(df, test_name)
            removed = matrix_processing.CELL_REMOVAL_MAP.get(test_name, [])
            cells = []
            for i in range(idx["n_rows"]):
                for j in range(idx["n_cols"]):
                    if (i, j) in removed:
                        # Blanked out of the parsed matrix (e.g. the CCFtap/
                        # CMFtap 50 and 70 km/h columns): never a run, so
                        # never a Corner Case cell either.
                        continue
                    attrs = matrix_processing._extract_test_point_attributes(
                        df, idx["start_row"], idx["start_col"], i, j, test_name
                    )
                    attrs.pop("Target approach", None)  # random, never matched on
                    cells.append(attrs)
            cls.template_cells[test_name] = cells

    def test_enumeration_matches_protocol_selection_on_template(self):
        for test_name, cells in self.template_cells.items():
            expected = [c for c in cells if _is_corner_case(test_name, c)]
            with self.subTest(test=test_name):
                actual = data_model.CORNER_CASE_CELLS.get(test_name, [])
                self.assertCountEqual(actual, expected)

    def test_scenarios_without_corner_cases_are_absent(self):
        for test_name in (
            "CCRs",
            "CCRm",
            "CMRs",
            "CC ELK On",
            "CC ELK OvI",
            "CCCscp SfS",
            "CCFtap SfS",
        ):
            self.assertNotIn(test_name, data_model.CORNER_CASE_CELLS)

    def test_no_duplicate_cells(self):
        for test_name, cells in data_model.CORNER_CASE_CELLS.items():
            with self.subTest(test=test_name):
                keys = [tuple(sorted(c.items())) for c in cells]
                self.assertEqual(len(keys), len(set(keys)))


if __name__ == "__main__":
    unittest.main()
