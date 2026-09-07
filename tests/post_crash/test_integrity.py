# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import os
import tempfile
import unittest
from unittest.mock import patch

import openpyxl

from euroncap_rating_2026 import common, version
from euroncap_rating_2026.post_crash import integrity


class TestRebuildTrustedInput(unittest.TestCase):
    def setUp(self):
        from importlib.resources import files

        self.pristine_path = str(files("data").joinpath("pc_template.xlsx"))

    def _save_wb(self, wb):
        fd, path = tempfile.mkstemp(suffix=".xlsx")
        os.close(fd)
        wb.save(path)
        self.addCleanup(os.remove, path)
        return path

    def test_max_score_tamper_is_neutralized(self):
        """Editing Max score in the user-supplied template must have zero
        effect on the sanitized output: it must reflect the official
        template's value instead."""
        wb = openpyxl.load_workbook(self.pristine_path)
        ws = wb["Test Scores"]
        header = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
        max_score_col = header.index("Max score") + 1
        original_max_score = ws.cell(row=2, column=max_score_col).value
        ws.cell(row=2, column=max_score_col).value = 99999
        user_path = self._save_wb(wb)

        sanitized_path = integrity.rebuild_trusted_input(user_path)
        self.addCleanup(os.remove, sanitized_path)

        sanitized_wb = openpyxl.load_workbook(sanitized_path, data_only=True)
        sanitized_value = (
            sanitized_wb["Test Scores"].cell(row=2, column=max_score_col).value
        )
        self.assertEqual(sanitized_value, original_max_score)
        self.assertNotEqual(sanitized_value, 99999)

    def test_input_parameters_value_is_preserved(self):
        """Legitimate grey-cell input must survive the rebuild unchanged."""
        wb = openpyxl.load_workbook(self.pristine_path)
        ws = wb["Input parameters"]
        header = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
        value_col = header.index("Value") + 1
        ws.cell(row=2, column=value_col).value = "Manual"
        user_path = self._save_wb(wb)

        sanitized_path = integrity.rebuild_trusted_input(user_path)
        self.addCleanup(os.remove, sanitized_path)

        sanitized_wb = openpyxl.load_workbook(sanitized_path, data_only=True)
        self.assertEqual(
            sanitized_wb["Input parameters"].cell(row=2, column=value_col).value,
            "Manual",
        )

    def test_reordered_rows_are_rejected(self):
        wb = openpyxl.load_workbook(self.pristine_path)
        ws = wb["Category Scores"]
        row2 = [ws.cell(row=2, column=c).value for c in range(1, 7)]
        row3 = [ws.cell(row=3, column=c).value for c in range(1, 7)]
        for c in range(1, 7):
            ws.cell(row=2, column=c).value = row3[c - 1]
            ws.cell(row=3, column=c).value = row2[c - 1]
        user_path = self._save_wb(wb)

        with self.assertRaises(common.TemplateIntegrityError):
            integrity.rebuild_trusted_input(user_path)

    def test_deleted_row_is_rejected(self):
        wb = openpyxl.load_workbook(self.pristine_path)
        ws = wb["Scenario Scores"]
        ws.delete_rows(3)
        user_path = self._save_wb(wb)

        with self.assertRaises(common.TemplateIntegrityError):
            integrity.rebuild_trusted_input(user_path)


class TestRebuildTrustedInputDfs(unittest.TestCase):
    """DataFrame-native sibling of TestRebuildTrustedInput: same tamper
    scenarios, but through rebuild_trusted_input_dfs (used by
    calculate_score(dfs)) instead of rebuild_trusted_input (used by the
    file-based preprocess() CLI command)."""

    def setUp(self):
        from importlib.resources import files

        pristine_path = str(files("data").joinpath("pc_template.xlsx"))
        self.pristine_dfs = common.read_excel_file_to_dfs(pristine_path)

    def _dfs(self):
        return {name: df.copy() for name, df in self.pristine_dfs.items()}

    def test_max_score_tamper_is_neutralized(self):
        dfs = self._dfs()
        original_max_score = dfs["Test Scores"].loc[0, "Max score"]
        dfs["Test Scores"].loc[0, "Max score"] = 99999

        sanitized = integrity.rebuild_trusted_input_dfs(dfs)

        self.assertEqual(
            sanitized["Test Scores"].loc[0, "Max score"], original_max_score
        )
        self.assertNotEqual(sanitized["Test Scores"].loc[0, "Max score"], 99999)

    def test_input_parameters_value_is_preserved(self):
        dfs = self._dfs()
        dfs["Input parameters"] = dfs["Input parameters"].astype(object)
        dfs["Input parameters"].loc[0, "Value"] = "Manual"

        sanitized = integrity.rebuild_trusted_input_dfs(dfs)

        self.assertEqual(sanitized["Input parameters"].loc[0, "Value"], "Manual")

    def test_reordered_rows_are_rejected(self):
        dfs = self._dfs()
        df = dfs["Category Scores"]
        df.iloc[[0, 1]] = df.iloc[[1, 0]].values

        with self.assertRaises(common.TemplateIntegrityError):
            integrity.rebuild_trusted_input_dfs(dfs)

    def test_deleted_row_is_rejected(self):
        dfs = self._dfs()
        dfs["Scenario Scores"] = (
            dfs["Scenario Scores"].drop(index=1).reset_index(drop=True)
        )

        with self.assertRaises(common.TemplateIntegrityError):
            integrity.rebuild_trusted_input_dfs(dfs)

    def test_sheet_missing_from_dfs_is_skipped_not_rejected(self):
        """calculate_score(dfs) callers may legitimately pass a partial dfs
        dict; a schema'd sheet that isn't present must not raise (deviates
        from rebuild_trusted_input, which requires the complete monolithic
        file)."""
        dfs = {"Input parameters": self.pristine_dfs["Input parameters"].copy()}

        sanitized = integrity.rebuild_trusted_input_dfs(dfs)

        self.assertIn("Input parameters", sanitized)
        self.assertNotIn("Test Scores", sanitized)


class TestDocumentationDataSheetCoverage(unittest.TestCase):
    """Regression test for the "sharp edge" documented in integrity.py's
    module docstring: any pristine-template sheet not covered by
    SHEET_SCHEMAS or PASSTHROUGH_SHEETS is silently reset to blank by
    rebuild_trusted_workbook/rebuild_trusted_dfs."""

    def setUp(self):
        from importlib.resources import files

        self.pristine_path = str(files("data").joinpath("pc_template.xlsx"))

    def test_every_pristine_sheet_is_covered(self):
        wb = openpyxl.load_workbook(self.pristine_path)
        covered = set(integrity.SHEET_SCHEMAS) | set(integrity.PASSTHROUGH_SHEETS)
        uncovered = set(wb.sheetnames) - covered
        self.assertEqual(uncovered, set())

    def test_documentation_and_data_sheets_are_hidden(self):
        wb = openpyxl.load_workbook(self.pristine_path)
        for name in ("Documentation", "Data"):
            self.assertIn(name, wb.sheetnames)
            self.assertEqual(wb[name].sheet_state, "hidden")

    def test_missing_input_check_covers_only_template_sheets(self):
        """find_missing_required_inputs must inspect prediction-template
        sheets only: every sheet it iterates has to exist in the pristine
        template, so assessment-only sheets (the preprocess-generated
        '...Verif' sheets) can never produce findings."""
        wb = openpyxl.load_workbook(self.pristine_path)
        self.assertEqual(set(integrity.SHEET_SCHEMAS) - set(wb.sheetnames), set())

    """The "...Verif" sheets (RI - RS Verification, PCI - Advanced eCall
    Verif, ...) exist only in preprocessed/assessment workbooks (preprocess
    generates them; pc_template.xlsx has none), and their Values are
    assessment-stage inputs, not OEM predictions --
    find_missing_required_inputs covers prediction-template sheets only, so
    blank Verif Values must NOT be reported. Prediction-stage coverage is
    exercised through the "Input parameters" sheet, built in-memory."""

    _INPUT_PARAMETERS_HEADER = [
        "Stage",
        "Stage element",
        "Stage subelement",
        "Category",
        "Input parameter",
        "Value",
    ]

    def _wb_with_sheet(self, sheet_name, header, rows):
        import openpyxl

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = sheet_name
        for col_idx, column in enumerate(header, start=1):
            ws.cell(row=1, column=col_idx, value=column)
        for row_idx, row in enumerate(rows, start=2):
            for col_idx, value in enumerate(row, start=1):
                ws.cell(row=row_idx, column=col_idx, value=value)
        return wb

    def test_blank_verif_values_are_not_reported(self):
        wb = self._wb_with_sheet(
            "PCI - Advanced eCall Verif",
            ["Category", "Scenario", "Value"],
            [
                ("Advanced eCall - 112", "General requirements", None),
                (None, "Potential number of occupants", None),
            ],
        )
        ws = wb.create_sheet("RI - RS Verification")
        ws.cell(row=1, column=1, value="Stage subelement")
        ws.cell(row=1, column=2, value="Value")
        ws.cell(row=2, column=1, value="Rescue sheet")

        self.assertEqual(integrity.find_missing_required_inputs(wb), [])

    def test_reports_blank_input_parameter_value(self):
        wb = self._wb_with_sheet(
            "Input parameters",
            self._INPUT_PARAMETERS_HEADER,
            [
                (
                    "Post-Crash",
                    "Rescue Information",
                    "Rescue sheets",
                    "Rescue sheet",
                    "Rescue sheet available",
                    None,
                ),
            ],
        )

        missing = integrity.find_missing_required_inputs(wb)

        self.assertEqual(len(missing), 1)
        entry = missing[0]
        self.assertEqual(entry.sheet, "Input parameters")
        self.assertEqual(entry.column, "Value")
        self.assertEqual(
            entry.row_identity["Input parameter"], "Rescue sheet available"
        )
        self.assertIn("Rescue sheet available", entry.description)

    def test_scoping_excludes_other_protocol_elements(self):
        wb = self._wb_with_sheet(
            "Input parameters",
            self._INPUT_PARAMETERS_HEADER,
            [
                (
                    "Post-Crash",
                    "Post-Crash Intervention",
                    "Advanced eCall",
                    "Advanced eCall - 112",
                    "eCall system fitted",
                    None,
                ),
                (
                    "Post-Crash",
                    "Rescue Information",
                    "Rescue sheets",
                    "Rescue sheet",
                    "Rescue sheet available",
                    None,
                ),
            ],
        )

        scoped = integrity.find_missing_required_inputs(
            wb,
            stage_element="Post-Crash Intervention",
            stage_subelement="Advanced eCall",
        )

        self.assertEqual(len(scoped), 1)
        self.assertEqual(
            scoped[0].row_identity["Input parameter"], "eCall system fitted"
        )

    def test_find_missing_required_inputs_in_file_round_trips(self):
        wb = self._wb_with_sheet(
            "Input parameters",
            self._INPUT_PARAMETERS_HEADER,
            [
                (
                    "Post-Crash",
                    "Rescue Information",
                    "Rescue sheets",
                    "Rescue sheet",
                    "Rescue sheet available",
                    None,
                ),
            ],
        )
        ws = wb.create_sheet("RI - RS Verification")
        ws.cell(row=1, column=1, value="Stage subelement")
        ws.cell(row=1, column=2, value="Value")
        ws.cell(row=2, column=1, value="Rescue sheet")
        fd, path = tempfile.mkstemp(suffix=".xlsx")
        os.close(fd)
        wb.save(path)
        self.addCleanup(os.remove, path)

        missing = integrity.find_missing_required_inputs_in_file(path)

        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0].sheet, "Input parameters")


class TestScopeValidation(unittest.TestCase):
    """An unknown (stage_element, stage_subelement) must raise instead of
    silently returning zero findings -- a typo'd caller-side mapping (e.g.
    snake_case identifiers) would otherwise silently disable the check.
    VALID_SCOPE_PAIRS must in turn cover every pair the packaged
    template's own sheets use, so no legitimate scope ever raises."""

    def setUp(self):
        from importlib.resources import files

        self.pristine_path = str(files("data").joinpath("pc_template.xlsx"))

    def test_unknown_scope_raises(self):
        wb = openpyxl.load_workbook(self.pristine_path, data_only=True)

        with self.assertRaises(ValueError):
            integrity.find_missing_required_inputs(
                wb, stage_element="rescue_information", stage_subelement="rescue_sheets"
            )
        with self.assertRaises(ValueError):
            integrity.find_missing_required_inputs(wb, stage_element="Bogus")

    def test_declared_pairs_cover_the_templates_own_literals(self):
        wb = openpyxl.load_workbook(self.pristine_path, data_only=True)
        template_pairs = set()
        for sheet in ("Test Scores", "Input parameters"):
            ws = wb[sheet]
            header = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
            element_col = header.index("Stage element") + 1
            subelement_col = header.index("Stage subelement") + 1
            last_element = last_subelement = None
            for row in range(2, ws.max_row + 1):
                element = ws.cell(row=row, column=element_col).value
                subelement = ws.cell(row=row, column=subelement_col).value
                if not common.is_empty_cell(element):
                    last_element = element
                    last_subelement = None
                if not common.is_empty_cell(subelement):
                    last_subelement = subelement
                if last_element is not None and last_subelement is not None:
                    template_pairs.add((last_element, last_subelement))

        self.assertTrue(template_pairs)
        undeclared = template_pairs - set(integrity.VALID_SCOPE_PAIRS)
        self.assertEqual(undeclared, set())


class TestPristineTemplateRegression(unittest.TestCase):
    """The audit assertion: on the pure packaged
    template every required cell is empty, so the unscoped check must
    report exactly the template's grey-painted input set -- no more (a
    false requirement fails valid submissions) and no less (a false pass
    ships an unchecked element). The literal per-sheet counts pin the
    required set; evolving the template legitimately means updating them
    in the same change. The fill cross-check then verifies each reported
    cell against the independent ground truth -- the grey paint the OEM
    actually sees."""

    EXPECTED_PER_SHEET = {
        "Input parameters": 7,
    }

    def setUp(self):
        from importlib.resources import files

        self.pristine_path = str(files("data").joinpath("pc_template.xlsx"))

    def test_unscoped_check_reports_exactly_the_expected_set(self):
        from collections import Counter

        wb = openpyxl.load_workbook(self.pristine_path, data_only=True)
        missing = integrity.find_missing_required_inputs(wb)

        self.assertEqual(
            dict(Counter(m.sheet for m in missing)), self.EXPECTED_PER_SHEET
        )

    def test_every_reported_cell_is_grey_painted_in_the_template(self):
        """Each reported coordinate must carry the grey input paint in the
        packaged template (a theme-shaded or literal-grey placeholder
        fill) -- a reported cell without it means the check requires
        something the template never asked the OEM to fill. Deliberately
        one-directional: some grey cells (Inspection [%], a criteria
        sheet's Value, non-sliding-scale OEM Predictions) are legitimately
        not required at prediction-upload time."""
        wb = openpyxl.load_workbook(self.pristine_path, data_only=True)
        missing = integrity.find_missing_required_inputs(wb)
        styled_wb = openpyxl.load_workbook(self.pristine_path)

        for entry in missing:
            fill = styled_wb[entry.sheet][entry.cell].fill
            self.assertEqual(
                fill.fill_type,
                "solid",
                f"{entry.sheet}!{entry.cell} reported but not painted",
            )
            if fill.fgColor.type == "theme":
                self.assertLess(
                    fill.fgColor.tint,
                    0,
                    f"{entry.sheet}!{entry.cell} has an unshaded theme fill",
                )
            else:
                grey_hex = str(fill.fgColor.rgb or "")[-6:].upper()
                self.assertIn(
                    grey_hex,
                    ("808080", "D9D9D9"),
                    f"{entry.sheet}!{entry.cell} reported but painted "
                    f"{fill.fgColor.rgb}",
                )


class TestFindMissingRequiredInputsVersionMismatch(unittest.TestCase):
    """A workbook generated by an incompatible library version must surface
    that drift as a finding, on top of -- not instead
    of -- the ordinary missing-input findings."""

    def setUp(self):
        from importlib.resources import files

        self.pristine_path = str(files("data").joinpath("pc_template.xlsx"))

    def _wb_with_version(self, version_string):
        wb = openpyxl.load_workbook(self.pristine_path, data_only=True)
        ws = wb["Version"] if "Version" in wb.sheetnames else wb.create_sheet("Version")
        ws["A1"] = "Generated with version"
        ws["B1"] = version_string
        return wb

    def test_incompatible_version_adds_finding_alongside_missing_inputs(self):
        with patch.object(common, "VERSION", "5.4.6"):
            wb = self._wb_with_version("5.4.0")
            findings = integrity.find_missing_required_inputs(wb)

        version_findings = [
            f for f in findings if f.check == common.CHECK_VERSION_MISMATCH
        ]
        self.assertEqual(len(version_findings), 1)
        self.assertTrue(any(f.check == common.CHECK_MISSING_INPUT for f in findings))

    def test_incompatible_version_surfaces_through_find_consistency_findings_too(self):
        with patch.object(common, "VERSION", "5.4.6"):
            wb = self._wb_with_version("5.4.0")
            findings = integrity.find_consistency_findings(wb)

        self.assertTrue(any(f.check == common.CHECK_VERSION_MISMATCH for f in findings))

    def test_version_finding_is_reported_even_when_scoped(self):
        """A version mismatch is a whole-workbook problem, not a per-element
        one -- it must not be dropped just because the caller scoped the
        check to one protocol element."""
        with patch.object(common, "VERSION", "5.4.6"):
            wb = self._wb_with_version("5.4.0")
            findings = integrity.find_missing_required_inputs(
                wb, stage_element="Rescue Information", stage_subelement="Rescue sheets"
            )

        self.assertTrue(any(f.check == common.CHECK_VERSION_MISMATCH for f in findings))

    def test_matching_version_adds_no_finding(self):
        wb = self._wb_with_version(version.VERSION)
        findings = integrity.find_missing_required_inputs(wb)

        self.assertFalse(
            any(f.check == common.CHECK_VERSION_MISMATCH for f in findings)
        )

    def test_no_version_sheet_adds_no_finding(self):
        """Regression guard: the many fixtures across this suite that never
        add a Version sheet at all must not start reporting a spurious
        mismatch."""
        wb = openpyxl.load_workbook(self.pristine_path, data_only=True)
        wb.remove(wb["Version"])
        findings = integrity.find_missing_required_inputs(wb)

        self.assertFalse(
            any(f.check == common.CHECK_VERSION_MISMATCH for f in findings)
        )

    def _rename_stage_subelement_header(self, wb):
        """Reproduces a real-world shape: a renamed identity
        column ("Stage Subelement" vs "Stage subelement") that makes the
        schema walk raise TemplateIntegrityError."""
        ws = wb["Input parameters"]
        header = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
        col = header.index("Stage subelement") + 1
        ws.cell(row=1, column=col).value = "Stage Subelement"

    def test_structurally_broken_and_version_mismatched_returns_only_version_finding(
        self,
    ):
        """The real-world case: an old-version file's renamed header would
        otherwise raise TemplateIntegrityError before any finding --
        including the version mismatch itself -- reaches the caller."""
        with patch.object(common, "VERSION", "5.4.6"):
            wb = self._wb_with_version("5.4.0")
            self._rename_stage_subelement_header(wb)
            findings = integrity.find_missing_required_inputs(wb)

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].check, common.CHECK_VERSION_MISMATCH)

    def test_structurally_broken_without_version_mismatch_still_raises(self):
        """Regression guard: with no version mismatch to blame, a
        genuinely broken template must still raise as before -- the
        schema-integrity contract predates and is independent of the version finding."""
        wb = openpyxl.load_workbook(self.pristine_path, data_only=True)
        self._rename_stage_subelement_header(wb)

        with self.assertRaises(common.TemplateIntegrityError):
            integrity.find_missing_required_inputs(wb)


if __name__ == "__main__":
    unittest.main()
