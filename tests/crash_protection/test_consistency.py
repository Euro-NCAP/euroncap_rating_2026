# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Report-only prediction collector for crash_protection
(consistency.find_prediction_inconsistencies) and its integration into
integrity.find_consistency_findings. Asymmetric legform patterns are
accepted: only no_prediction_provided is reported on the VRU sheet."""

import unittest

import openpyxl

from euroncap_rating_2026 import common
from euroncap_rating_2026.crash_protection import consistency, integrity


# Worksheet geometry of the legform matrix (see consistency._legform_row):
# matrix rows 0..2 are worksheet rows 27..29, matrix column j is worksheet
# column 5+j (E..Y, 21 positions).
LEGFORM_WS_ROWS = (27, 28, 29)


class TestLegformPredictionFindings(unittest.TestCase):
    def setUp(self):
        from importlib.resources import files

        self.pristine_path = str(files("data").joinpath("cp_template.xlsx"))
        self.wb = openpyxl.load_workbook(self.pristine_path, data_only=True)
        self.ws = self.wb["CP - VRU Prediction"]

    def test_pristine_template_has_no_findings(self):
        # An untouched template trips no_prediction_provided for every
        # section the missing-input check does not cover: Headforms,
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
        self.ws["Y27"] = "ST"

        findings = consistency.find_prediction_inconsistencies(self.wb)

        # Pelvis Impact now has a prediction, so only Leg Impact and
        # Headforms (still untouched) trip no_prediction_provided.
        self.assertEqual(
            {f.check for f in findings}, {common.CHECK_NO_PREDICTION_PROVIDED}
        )
        self.assertEqual(
            {f.row_identity.get("Section") for f in findings},
            {"Headforms", "aPLI - Femur / aPLI - Knee & Tibia (Leg Impact)"},
        )

    def test_asymmetric_sa_st_is_accepted(self):
        # SA at G28 mirrored by ST at W28: the retired symmetry rule flagged
        # both cells; Euro NCAP accepts the pattern, so nothing may
        # be reported for them.
        self.ws["G28"] = "SA"
        self.ws["W28"] = "ST"

        findings = consistency.find_prediction_inconsistencies(self.wb)

        self.assertEqual(
            {f.check for f in findings}, {common.CHECK_NO_PREDICTION_PROVIDED}
        )
        # Leg Impact now has a prediction (G28/W28), so only Pelvis Impact
        # (still untouched) and Headforms trip no_prediction_provided.
        self.assertEqual(
            {f.row_identity.get("Section") for f in findings},
            {"Headforms", "Upper legform (Pelvis Impact)"},
        )

    def test_scope_attribution_per_legform_row(self):
        # Upper legform (worksheet row 27) belongs to Pelvis Impact; the two
        # aPLI rows (28/29) to Leg Impact. Consuming applications query the
        # two scopes separately and merge them -- attribution must not overlap.
        # Fill only the Pelvis row (asymmetric, mirror left grey, accepted) and
        # leave both aPLI rows empty so the two scopes differ.
        self.ws["E27"] = "T"

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

        leg_section = "aPLI - Femur / aPLI - Knee & Tibia (Leg Impact)"
        # Pelvis has a prediction, so its scope is clean; the empty aPLI rows
        # are reported under Leg Impact only, and the combined alias sees the
        # same single finding. Headforms is its own scope and stays untouched.
        self.assertEqual(pelvis, [])
        self.assertEqual({f.check for f in leg}, {common.CHECK_NO_PREDICTION_PROVIDED})
        self.assertEqual({f.row_identity.get("Section") for f in leg}, {leg_section})
        self.assertEqual(
            {f.row_identity.get("Section") for f in combined}, {leg_section}
        )
        self.assertEqual({f.check for f in head}, {common.CHECK_NO_PREDICTION_PROVIDED})
        self.assertEqual({f.row_identity.get("Section") for f in head}, {"Headforms"})

    def test_find_consistency_findings_accepts_asymmetric_grid(self):
        # A partially filled legform grid is a legitimate submission: no
        # missing_input finding for the empty cells, and no finding for the
        # asymmetric SA either.
        self.ws["G28"] = "SA"

        findings = integrity.find_consistency_findings(
            self.wb, "VRU Impact", "Leg Impact"
        )

        grid = [f for f in findings if f.sheet == "CP - VRU Prediction"]
        self.assertEqual(grid, [])


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
