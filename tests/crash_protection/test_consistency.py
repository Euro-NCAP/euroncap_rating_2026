# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Report-only legform symmetry collector for crash_protection
(consistency.find_prediction_inconsistencies) and its integration into
integrity.find_consistency_findings."""

import unittest

import openpyxl

from euroncap_rating_2026 import common
from euroncap_rating_2026.crash_protection import consistency, integrity, vru_processing


# Worksheet geometry of the legform matrix (see consistency._legform_cell):
# matrix rows 0..2 are worksheet rows 27..29, matrix column j is worksheet
# column 5+j (E..Y, 21 positions).
LEGFORM_WS_ROWS = (27, 28, 29)


class TestLegformSymmetryFindings(unittest.TestCase):
    def setUp(self):
        from importlib.resources import files

        self.pristine_path = str(files("data").joinpath("cp_template.xlsx"))
        self.wb = openpyxl.load_workbook(self.pristine_path, data_only=True)
        self.ws = self.wb["CP - VRU Prediction"]

    def test_pristine_template_has_no_findings(self):
        # An untouched template has no legform_symmetry violations (nothing
        # filled in to contradict), but it does trip
        # no_prediction_provided for every section the missing-input check
        # no longer covers: Headforms,
        # Pelvis Impact (Upper legform), and Leg Impact (both aPLI rows).
        findings = consistency.find_prediction_inconsistencies(self.wb)

        self.assertEqual(
            {f.check for f in findings}, {common.CHECK_NO_PREDICTION_PROVIDED}
        )
        self.assertEqual(
            {f.row_identity.get("Section") for f in findings},
            {
                "Headforms",
                "Upper legform (Pelvis Impact)",
                "aPLI - Femur / aPLI - Knee & Tibia (Leg Impact)",
            },
        )
        self.assertEqual({f.cell for f in findings}, {"A1", "A27", "A28"})

    def test_symmetric_pattern_is_clean(self):
        self.ws["E27"] = "T"
        self.ws["Y27"] = "ST"  # mirror of E: T/ST both satisfy the T/ST rule

        findings = consistency.find_prediction_inconsistencies(self.wb)

        # No symmetry violations from this pattern -- Pelvis Impact now has
        # a prediction too, so only Leg Impact and Headforms (still
        # untouched) trip no_prediction_provided.
        self.assertEqual(
            {f.check for f in findings}, {common.CHECK_NO_PREDICTION_PROVIDED}
        )
        self.assertEqual(
            {f.row_identity.get("Section") for f in findings},
            {"Headforms", "aPLI - Femur / aPLI - Knee & Tibia (Leg Impact)"},
        )

    def test_asymmetric_t_st_reported_with_cells(self):
        self.ws["G28"] = "SA"  # mirror W28 left grey -> SA rule violated
        self.ws["W28"] = "ST"  # its mirror is SA, not T/ST -> T/ST rule violated

        findings = consistency.find_prediction_inconsistencies(self.wb)

        symmetry = [f for f in findings if f.check == common.CHECK_LEGFORM_SYMMETRY]
        self.assertEqual({f.cell for f in symmetry}, {"G28", "W28"})
        for f in symmetry:
            self.assertEqual(f.sheet, "CP - VRU Prediction")
            self.assertEqual(f.row_identity.get("Section"), "aPLI - Femur")

        # Leg Impact now has a prediction (G28/W28), so only Pelvis Impact
        # (still untouched) and Headforms trip no_prediction_provided.
        no_prediction = [
            f for f in findings if f.check == common.CHECK_NO_PREDICTION_PROVIDED
        ]
        self.assertEqual(
            {f.row_identity.get("Section") for f in no_prediction},
            {"Headforms", "Upper legform (Pelvis Impact)"},
        )

    def test_scope_attribution_per_legform_row(self):
        # Upper legform (worksheet row 27) belongs to Pelvis Impact; the two
        # aPLI rows (28/29) to Leg Impact. Consuming applications query the
        # two scopes separately and merge them -- attribution must not overlap.
        self.ws["E27"] = "T"  # mirror Y27 grey -> Pelvis violation
        self.ws["F29"] = "SA"  # mirror X29 grey -> Leg violation

        pelvis = consistency.find_prediction_inconsistencies(
            self.wb, "VRU Impact", "Pelvis Impact"
        )
        leg = consistency.find_prediction_inconsistencies(
            self.wb, "VRU Impact", "Leg Impact"
        )
        head = consistency.find_prediction_inconsistencies(
            self.wb, "VRU Impact", "Head Impact"
        )
        combined = consistency.find_prediction_inconsistencies(
            self.wb, "VRU Impact", "Pelvis & Leg Impact"
        )

        self.assertEqual({f.cell for f in pelvis}, {"E27"})
        self.assertEqual({f.cell for f in leg}, {"F29"})
        # Headforms is untouched and is its own scope -- it trips
        # no_prediction_provided regardless of the legform edits above, and
        # none of the legform findings leak into this scope.
        self.assertEqual({f.check for f in head}, {common.CHECK_NO_PREDICTION_PROVIDED})
        self.assertEqual({f.row_identity.get("Section") for f in head}, {"Headforms"})
        self.assertEqual({f.cell for f in combined}, {"E27", "F29"})

    def test_blocking_check_still_raises_first_violation(self):
        self.ws["G28"] = "SA"
        df = common.sheet_to_dataframe(self.ws)
        matrix = vru_processing.get_legform_matrix(df)
        with self.assertRaises(ValueError) as ctx:
            vru_processing.check_legform_blue_symmetry(matrix)
        self.assertIn("SA prediction at legform matrix row=1", str(ctx.exception))

    def test_find_consistency_findings_reports_symmetry_not_missing(self):
        # An empty legform grid is a
        # legitimate submission (no missing_input findings), but what the
        # OEM does fill is still validated by the symmetry rule.
        self.ws["G28"] = "SA"

        findings = integrity.find_consistency_findings(
            self.wb, "VRU Impact", "Leg Impact"
        )

        by_check = {f.check for f in findings}
        self.assertIn(common.CHECK_LEGFORM_SYMMETRY, by_check)
        grid_missing = [
            f
            for f in findings
            if f.check == common.CHECK_MISSING_INPUT
            and f.sheet == "CP - VRU Prediction"
        ]
        self.assertEqual(grid_missing, [])


class TestNoPredictionProvidedFindings(unittest.TestCase):
    """CP - VRU Prediction's Headforms/Pelvis Impact/Leg Impact sections
    have no per-cell required-input check -- this is what still guarantees
    the OEM entered at least one prediction in each."""

    def setUp(self):
        from importlib.resources import files

        self.pristine_path = str(files("data").joinpath("cp_template.xlsx"))
        self.wb = openpyxl.load_workbook(self.pristine_path, data_only=True)
        self.ws = self.wb["CP - VRU Prediction"]

    def test_headform_prediction_clears_the_finding(self):
        self.ws["E5"] = "Green"

        findings = consistency.find_prediction_inconsistencies(
            self.wb, "VRU Impact", "Head Impact"
        )

        self.assertEqual(findings, [])

    def test_either_apli_row_clears_leg_impact(self):
        # aPLI - Knee & Tibia (row 29) alone is enough to satisfy the
        # combined Leg Impact element -- aPLI - Femur (row 28) can stay
        # untouched.
        self.ws["E29"] = "T"
        self.ws["Y29"] = "ST"  # keep it symmetry-clean, unrelated to this check

        findings = consistency.find_prediction_inconsistencies(
            self.wb, "VRU Impact", "Leg Impact"
        )

        self.assertEqual(
            [f for f in findings if f.check == common.CHECK_NO_PREDICTION_PROVIDED],
            [],
        )

    def test_message_and_anchor_cell(self):
        findings = consistency.find_prediction_inconsistencies(
            self.wb, "VRU Impact", "Head Impact"
        )

        self.assertEqual(len(findings), 1)
        finding = findings[0]
        self.assertEqual(finding.check, common.CHECK_NO_PREDICTION_PROVIDED)
        self.assertEqual(finding.sheet, "CP - VRU Prediction")
        self.assertEqual(finding.cell, "A1")
        self.assertIn("Headforms", finding.description)
        self.assertIn("no prediction provided", finding.description)


if __name__ == "__main__":
    unittest.main()
