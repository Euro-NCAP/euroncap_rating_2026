# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""report_writer.write_report must collapse CPLA/CBLA duplicate-AEB rows in
the *output* worksheet on its own, without any caller having to compute and
pass `rows_to_delete` (see test_info.collapse_cpla_cbla_aeb_duplicate_rows:
a caller that never threads that value through -- as a report-download
call site didn't -- must still get a correctly collapsed download).
"""

import shutil
import tempfile
import unittest
from importlib.resources import files
from pathlib import Path

import openpyxl

from euroncap_rating_2026 import common
from euroncap_rating_2026.crash_avoidance import report_writer

PREDICTION_SHEET = "FC - Ped & Cyc pred."

# CPLA day's choice-band rows for 50/60 km/h, worksheet (1-based) coordinates.
CPLA_DAY_50_ROW = 9
CPLA_DAY_60_ROW = 10
# CPLA day's fixed-band rows for 50/60 km/h -- these are the ones promoted
# to standard at both impact locations once their choice-band duplicate
# above is collapsed away.
CPLA_DAY_50_FIXED_ROW = 7
CPLA_DAY_60_FIXED_ROW = 8
FUNCTION_COL = 3  # "Function"
IL_75_PERCENT_COL = 7
IL_50_PERCENT_COL = 8  # "Impact location" == 0.5
IL_25_PERCENT_COL = 9  # "Impact location" == 0.25
IL_10_PERCENT_COL = 10

# CBLA's choice-band rows for 50/60 km/h and their fixed-band counterparts.
CBLA_50_ROW = 34
CBLA_60_ROW = 35
CBLA_50_FIXED_ROW = 32
CBLA_60_FIXED_ROW = 33


def _has_full_box(cell) -> bool:
    b = cell.border
    return all(
        side is not None and side.style is not None
        for side in (b.left, b.right, b.top, b.bottom)
    )


def _make_input_file(input_dir, declare_choice_band=False, function="AEB"):
    """A temp copy of the pristine template, optionally with CPLA day's
    50/60 km/h choice-band rows declared with *function* and a real
    (non-blank) 25% impact-location colour -- the shape of a genuine OEM
    prediction sheet, whose cells hold literal colour-name text rather than
    "X" marks (see the investigation into the resulting cosmetic bug)."""
    pristine_path = str(files("data").joinpath("ca_template.xlsx"))
    input_file = str(Path(input_dir) / "ca_template.xlsx")
    shutil.copy(pristine_path, input_file)
    common.add_version_sheet(
        input_file, cli_command=common.CliCommand.GENERATE_TEMPLATE
    )

    if declare_choice_band:
        wb = openpyxl.load_workbook(input_file)
        ws = wb[PREDICTION_SHEET]
        for row in (CPLA_DAY_50_ROW, CPLA_DAY_60_ROW):
            ws.cell(row=row, column=FUNCTION_COL, value=function)
            ws.cell(row=row, column=IL_25_PERCENT_COL, value="Green")
        wb.save(input_file)

    return input_file


def _make_cbla_input_file(input_dir, function="AEB"):
    """A temp copy of the pristine template with CBLA's 50/60 km/h
    choice-band rows (Target speed 20 km/h) declared with *function*."""
    pristine_path = str(files("data").joinpath("ca_template.xlsx"))
    input_file = str(Path(input_dir) / "ca_template.xlsx")
    shutil.copy(pristine_path, input_file)
    common.add_version_sheet(
        input_file, cli_command=common.CliCommand.GENERATE_TEMPLATE
    )

    wb = openpyxl.load_workbook(input_file)
    ws = wb[PREDICTION_SHEET]
    for row in (CBLA_50_ROW, CBLA_60_ROW):
        ws.cell(row=row, column=FUNCTION_COL, value=function)
        ws.cell(row=row, column=IL_25_PERCENT_COL, value="Green")
    wb.save(input_file)

    return input_file


class TestDeriveCplaCblaRowsToDelete(unittest.TestCase):
    def test_collapses_choice_band_declared_aeb(self):
        with tempfile.TemporaryDirectory() as input_dir:
            input_file = _make_input_file(input_dir, declare_choice_band=True)
            rows_to_delete = report_writer._derive_cpla_cbla_rows_to_delete(input_file)
            self.assertEqual(
                rows_to_delete,
                {PREDICTION_SHEET: [CPLA_DAY_50_ROW, CPLA_DAY_60_ROW]},
            )

    def test_no_collapse_when_choice_band_declared_fcw(self):
        with tempfile.TemporaryDirectory() as input_dir:
            input_file = _make_input_file(
                input_dir, declare_choice_band=True, function="FCW"
            )
            rows_to_delete = report_writer._derive_cpla_cbla_rows_to_delete(input_file)
            self.assertEqual(rows_to_delete, {})

    def test_no_collapse_on_pristine_template(self):
        with tempfile.TemporaryDirectory() as input_dir:
            input_file = _make_input_file(input_dir, declare_choice_band=False)
            rows_to_delete = report_writer._derive_cpla_cbla_rows_to_delete(input_file)
            self.assertEqual(rows_to_delete, {})

    def test_idempotent_once_rows_are_physically_removed(self):
        """Running the derivation again against an already-collapsed sheet
        must find nothing left to remove (write_report may be called more
        than once against the same downstream file, e.g. COMPUTE_SCORE
        after PREPROCESS)."""
        with tempfile.TemporaryDirectory() as input_dir:
            input_file = _make_input_file(input_dir, declare_choice_band=True)
            rows = report_writer._derive_cpla_cbla_rows_to_delete(input_file)[
                PREDICTION_SHEET
            ]

            wb = openpyxl.load_workbook(input_file)
            common.delete_rows_keeping_merges(wb[PREDICTION_SHEET], rows)
            wb.save(input_file)

            self.assertEqual(
                report_writer._derive_cpla_cbla_rows_to_delete(input_file), {}
            )


class TestWriteReportCollapsesWithoutCallerInput(unittest.TestCase):
    def test_collapses_prediction_sheet_with_rows_to_delete_omitted(self):
        with (
            tempfile.TemporaryDirectory() as input_dir,
            tempfile.TemporaryDirectory() as output_dir,
        ):
            input_file = _make_input_file(input_dir, declare_choice_band=True)

            # No rows_to_delete passed -- mirrors a caller's actual
            # (unmodified) call site.
            report_writer.write_report(
                common.CliCommand.PREPROCESS,
                input_file,
                updated_dfs={},
                output_path=output_dir,
                format_prediction_cells=False,
            )

            output_file = Path(output_dir) / "ca_preprocessed_template.xlsx"
            ws = openpyxl.load_workbook(str(output_file))[PREDICTION_SHEET]

            # 10 rows collapsed to 8: 70/80 km/h shift up into rows 9/10,
            # and the row after them (the blank separator before "CPLA
            # night") is now one row higher too.
            self.assertEqual(ws.cell(row=9, column=1).value, "70 km/h")
            self.assertEqual(ws.cell(row=10, column=1).value, "80 km/h")
            self.assertIsNone(ws.cell(row=11, column=1).value)
            self.assertEqual(ws.cell(row=12, column=1).value, "CPLA night")

    def test_preserves_caller_supplied_rows_to_delete_for_other_sheets(self):
        with (
            tempfile.TemporaryDirectory() as input_dir,
            tempfile.TemporaryDirectory() as output_dir,
        ):
            input_file = _make_input_file(input_dir, declare_choice_band=True)
            pristine_row_count = openpyxl.load_workbook(input_file)[
                "Input parameters"
            ].max_row

            report_writer.write_report(
                common.CliCommand.PREPROCESS,
                input_file,
                updated_dfs={},
                output_path=output_dir,
                format_prediction_cells=False,
                # Last row, blank in the pristine template -- safe to drop
                # without disturbing meaningful content, this test only
                # cares whether the instruction survives the merge.
                rows_to_delete={"Input parameters": [pristine_row_count]},
            )

            output_file = Path(output_dir) / "ca_preprocessed_template.xlsx"
            wb = openpyxl.load_workbook(str(output_file))

            # The explicit instruction for an unrelated sheet was honoured...
            self.assertEqual(wb["Input parameters"].max_row, pristine_row_count - 1)
            # ...and the CPLA/CBLA collapse still ran on top of it.
            ws = wb[PREDICTION_SHEET]
            self.assertEqual(ws.cell(row=9, column=1).value, "70 km/h")

    def test_unreadable_input_file_falls_back_to_uncollapsed_sheet(self):
        """A caller-supplied file that can't be parsed for the CPLA/CBLA
        rule (corrupt, unexpected layout, etc.) must not break the report --
        collapsing is an enhancement, not a hard dependency."""
        with (
            tempfile.TemporaryDirectory() as input_dir,
            tempfile.TemporaryDirectory() as output_dir,
        ):
            input_file = _make_input_file(input_dir, declare_choice_band=True)

            original_read = common.read_excel_file_to_dfs

            def _boom(*args, **kwargs):
                raise RuntimeError("simulated parse failure")

            common.read_excel_file_to_dfs = _boom
            try:
                report_writer.write_report(
                    common.CliCommand.PREPROCESS,
                    input_file,
                    updated_dfs={},
                    output_path=output_dir,
                    format_prediction_cells=False,
                )
            finally:
                common.read_excel_file_to_dfs = original_read

            output_file = Path(output_dir) / "ca_preprocessed_template.xlsx"
            ws = openpyxl.load_workbook(str(output_file))[PREDICTION_SHEET]
            # Uncollapsed: the duplicate 50/60 km/h rows are still there.
            self.assertEqual(ws.cell(row=9, column=1).value, "50 km/h")
            self.assertEqual(ws.cell(row=10, column=1).value, "60 km/h")


class TestBoxesPromotedStandardRangeCells(unittest.TestCase):
    """A row promoted to standard range at both 50% and 25% IL (its
    choice-band duplicate at that speed was collapsed) must get a matching
    box on the newly-promoted cell, not just the one it already had (see
    the follow-up cosmetic issue described above)."""

    def test_cpla_promoted_rows_boxed_at_both_50_and_25_percent(self):
        with (
            tempfile.TemporaryDirectory() as input_dir,
            tempfile.TemporaryDirectory() as output_dir,
        ):
            input_file = _make_input_file(input_dir, declare_choice_band=True)

            report_writer.write_report(
                common.CliCommand.PREPROCESS,
                input_file,
                updated_dfs={},
                output_path=output_dir,
                format_prediction_cells=True,
            )

            output_file = Path(output_dir) / "ca_preprocessed_template.xlsx"
            ws = openpyxl.load_workbook(str(output_file))[PREDICTION_SHEET]

            for row in (CPLA_DAY_50_FIXED_ROW, CPLA_DAY_60_FIXED_ROW):
                self.assertTrue(
                    _has_full_box(ws.cell(row=row, column=IL_50_PERCENT_COL)),
                    f"row {row} missing its original 50% box",
                )
                self.assertTrue(
                    _has_full_box(ws.cell(row=row, column=IL_25_PERCENT_COL)),
                    f"row {row} missing the newly-promoted 25% box",
                )

    def test_cpla_non_promoted_row_is_not_over_boxed(self):
        with (
            tempfile.TemporaryDirectory() as input_dir,
            tempfile.TemporaryDirectory() as output_dir,
        ):
            input_file = _make_input_file(input_dir, declare_choice_band=True)

            report_writer.write_report(
                common.CliCommand.PREPROCESS,
                input_file,
                updated_dfs={},
                output_path=output_dir,
                format_prediction_cells=True,
            )

            output_file = Path(output_dir) / "ca_preprocessed_template.xlsx"
            ws = openpyxl.load_workbook(str(output_file))[PREDICTION_SHEET]

            # Row 3 (10 km/h, fixed band only) keeps exactly its original
            # box: 50% boxed, 75%/25%/10% not.
            self.assertTrue(_has_full_box(ws.cell(row=3, column=IL_50_PERCENT_COL)))
            for col in (IL_75_PERCENT_COL, IL_25_PERCENT_COL, IL_10_PERCENT_COL):
                self.assertFalse(_has_full_box(ws.cell(row=3, column=col)))

    def test_pristine_template_border_pattern_is_reproduced_bit_for_bit(self):
        """With nothing collapsed, recomputing the box from scratch must
        match the original pattern exactly."""
        with (
            tempfile.TemporaryDirectory() as input_dir,
            tempfile.TemporaryDirectory() as output_dir,
        ):
            input_file = _make_input_file(input_dir, declare_choice_band=False)
            pristine_ws = openpyxl.load_workbook(input_file)[PREDICTION_SHEET]

            report_writer.write_report(
                common.CliCommand.PREPROCESS,
                input_file,
                updated_dfs={},
                output_path=output_dir,
                format_prediction_cells=True,
            )

            output_file = Path(output_dir) / "ca_preprocessed_template.xlsx"
            output_ws = openpyxl.load_workbook(str(output_file))[PREDICTION_SHEET]

            for row in range(3, 26):
                for col in (
                    IL_75_PERCENT_COL,
                    IL_50_PERCENT_COL,
                    IL_25_PERCENT_COL,
                    IL_10_PERCENT_COL,
                ):
                    self.assertEqual(
                        _has_full_box(output_ws.cell(row=row, column=col)),
                        _has_full_box(pristine_ws.cell(row=row, column=col)),
                        f"row={row}, col={col}",
                    )

    def test_cbla_promoted_rows_boxed_at_both_50_and_25_percent(self):
        with (
            tempfile.TemporaryDirectory() as input_dir,
            tempfile.TemporaryDirectory() as output_dir,
        ):
            input_file = _make_cbla_input_file(input_dir)

            report_writer.write_report(
                common.CliCommand.PREPROCESS,
                input_file,
                updated_dfs={},
                output_path=output_dir,
                format_prediction_cells=True,
            )

            output_file = Path(output_dir) / "ca_preprocessed_template.xlsx"
            ws = openpyxl.load_workbook(str(output_file))[PREDICTION_SHEET]

            for row in (CBLA_50_FIXED_ROW, CBLA_60_FIXED_ROW):
                self.assertTrue(
                    _has_full_box(ws.cell(row=row, column=IL_50_PERCENT_COL)),
                    f"row {row} missing its original 50% box",
                )
                self.assertTrue(
                    _has_full_box(ws.cell(row=row, column=IL_25_PERCENT_COL)),
                    f"row {row} missing the newly-promoted 25% box",
                )


if __name__ == "__main__":
    unittest.main()
