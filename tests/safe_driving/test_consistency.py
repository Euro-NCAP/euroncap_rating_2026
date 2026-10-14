# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Report-only prediction-rule collector for safe_driving
(consistency.find_prediction_inconsistencies) and its integration into
integrity.find_consistency_findings."""

import unittest
from importlib.resources import files

import openpyxl

from euroncap_rating_2026 import common
from euroncap_rating_2026.safe_driving import consistency, integrity


class TestMixedDmPredictionGroupIsAccepted(unittest.TestCase):
    """A (Task, Movement type, Vehicle response) group may hold both Red and
    Green -- or Red and N/A -- cells. The upload-feedback path must stay silent
    about it: the two rules that used to report it are gone, and nothing else
    may take their place.

    The "Long distraction" table starts at worksheet row 15; its first group
    (Task=Non-driving task, Movement type=Owl) spans rows 17-21, with the
    Warning strategy in column D and Forward support in column E."""

    def setUp(self):
        self.pristine_path = str(files("data").joinpath("sd_template.xlsx"))
        self.wb = openpyxl.load_workbook(self.pristine_path, data_only=True)
        self.ws = self.wb["DE - DM pred."]

    def tearDown(self):
        self.wb.close()

    def test_green_and_red_in_one_group_reports_nothing(self):
        self.ws["D17"] = "Green"
        self.ws["D18"] = "Red"
        self.ws["D19"] = "Green"

        self.assertEqual(consistency.find_prediction_inconsistencies(self.wb), [])

    def test_na_and_red_in_one_group_reports_nothing(self):
        self.ws["E17"] = "Red"
        self.ws["E18"] = "N/A"

        self.assertEqual(consistency.find_prediction_inconsistencies(self.wb), [])

    def test_find_consistency_findings_reports_no_prediction_rule(self):
        """Through the whole upload-feedback entry point: a mixed group leaves
        only the ordinary missing-input findings for the blank cells."""
        self.ws["D17"] = "Green"
        self.ws["D18"] = "Red"
        self.ws["D19"] = "N/A"

        findings = integrity.find_consistency_findings(
            self.wb, "Driver Engagement", "Driver monitoring"
        )

        self.assertEqual({f.check for f in findings}, {common.CHECK_MISSING_INPUT})


class TestSlifEmptyScopeFindingCount(unittest.TestCase):
    """Regression: a "Failed (1)" badge for an entirely-blank Speed
    assistance submission is driven by len(find_consistency_findings(...))
    -- a silent count drift (e.g. a template layout change altering how
    many cells are required) would desync the badge from what the UI
    actually lists, so the exact count is worth pinning rather than just
    asserting "non-empty"."""

    def test_pristine_speed_assistance_scope_reports_330_findings(self):
        pristine_path = str(files("data").joinpath("sd_template.xlsx"))

        findings = integrity.find_consistency_findings_in_file(
            pristine_path, "Vehicle Assistance", "Speed assistance"
        )

        self.assertEqual(len(findings), 330)
        self.assertEqual({f.check for f in findings}, {common.CHECK_MISSING_INPUT})


class TestStaleAccCutOutFindings(unittest.TestCase):
    """The cut-out matrices must carry the protocol's (VUT speed, Target speed)
    pairs. A sheet that carries a superseded layout matches no test run in a
    consuming application, so its Value cells stay blank -- and a blank Value inherits the
    OEM's prediction at full credit. This rule is the only thing that says so:
    every layout stamped the same library version.

    "CCR cut-out" labels worksheet row 50, so its two test points are rows
    52-53; "CMR cut-out" labels row 96, giving rows 98-99.
    """

    def setUp(self):
        self.pristine_path = str(files("data").joinpath("sd_template.xlsx"))
        self.wb = openpyxl.load_workbook(self.pristine_path, data_only=True)
        self.ws = self.wb["VA - ACC pred."]

    def tearDown(self):
        self.wb.close()

    def _cut_out_findings(self, stage_element=None, stage_subelement=None):
        return [
            f
            for f in consistency.find_prediction_inconsistencies(
                self.wb, stage_element, stage_subelement
            )
            if f.check == common.CHECK_STALE_ACC_CUT_OUT
        ]

    def test_pristine_template_has_no_findings(self):
        self.assertEqual(self._cut_out_findings(), [])

    def test_superseded_target_speed_layout_is_reported_for_both_matrices(self):
        """VUT speed 50/70 with Target speed 0 -- the layout of every released
        build before 5.4.6."""
        for base in (52, 98):
            for offset, vut in ((0, "50 km/h"), (1, "70 km/h")):
                self.ws.cell(row=base + offset, column=1).value = vut
                self.ws.cell(row=base + offset, column=2).value = "0 km/h"

        findings = self._cut_out_findings()

        self.assertEqual([f.column for f in findings], ["CCR cut-out", "CMR cut-out"])
        self.assertEqual(findings[0].sheet, "VA - ACC pred.")
        self.assertEqual(findings[0].cell, "A52")
        self.assertEqual(findings[1].cell, "A98")
        self.assertIn("50 km/h / 0 km/h", findings[0].description)
        self.assertIn("70 km/h / 50 km/h", findings[0].description)
        self.assertIn("superseded template", findings[0].description)

    def test_shifted_column_layout_is_reported(self):
        """VUT speed restored to 70/90 but Target speed still 0 -- shipped only
        in an unreleased pre-release build, but it matches no
        test run either."""
        for row in (52, 53):
            self.ws.cell(row=row, column=2).value = "0 km/h"

        findings = self._cut_out_findings()

        self.assertEqual([f.column for f in findings], ["CCR cut-out"])
        self.assertIn("70 km/h / 0 km/h", findings[0].description)

    def test_blank_target_speed_is_reported(self):
        self.ws.cell(row=52, column=2).value = None

        findings = self._cut_out_findings()

        self.assertEqual(len(findings), 1)
        self.assertIn("(empty)", findings[0].description)

    def test_rule_is_scoped_to_acc_performance(self):
        self.ws.cell(row=52, column=2).value = "0 km/h"

        self.assertEqual(
            self._cut_out_findings("Vehicle Assistance", "Speed assistance"), []
        )
        self.assertEqual(
            len(self._cut_out_findings("Vehicle Assistance", "ACC performance")), 1
        )

    def test_find_consistency_findings_merges_missing_and_rules(self):
        """find_consistency_findings returns one list covering both the
        missing-input check and the prediction rules."""
        self.ws.cell(row=52, column=2).value = "0 km/h"

        findings = integrity.find_consistency_findings(
            self.wb, "Vehicle Assistance", "ACC performance"
        )

        checks = {f.check for f in findings}
        self.assertIn(common.CHECK_STALE_ACC_CUT_OUT, checks)
        self.assertIn(common.CHECK_MISSING_INPUT, checks)


if __name__ == "__main__":
    unittest.main()
