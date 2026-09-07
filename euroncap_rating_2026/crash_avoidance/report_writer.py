# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import os
from datetime import datetime

import pandas as pd

from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side

from euroncap_rating_2026.common import (
    PREDICTION_COLOR_MAP,
    load_or_create_workbook,
    hard_copy_sheet,
)
from euroncap_rating_2026 import common
from euroncap_rating_2026.crash_avoidance import data_model
from euroncap_rating_2026.crash_avoidance import matrix_processing
from euroncap_rating_2026.crash_avoidance.test_info import (
    get_sheet_prefix,
    get_stage_subelement_key,
    read_loadcase_info,
    check_cpla_cbla_coherence,
    collapse_cpla_cbla_aeb_duplicate_rows,
)

import logging
from openpyxl.cell.cell import MergedCell
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

PREPROCESS_SHEETS_TO_COPY = [
    "Version",
    "Input parameters",
    "Test Scores",
    "Scenario Scores",
    "Category Scores",
    "FC - Car & PTW pred.",
    "FC - Car & PTW robust. pred.",
    "FC - Ped & Cyc pred.",
    "FC - Ped & Cyc robust. pred.",
    "FC - Driver State Link",
    "LDC - Single Veh pred.",
    "LDC - Car & PTW pred.",
    "LDC - robust. pred.",
    "LSC - Car & PTW pred.",
    "LSC - Ped & Cyc pred.",
    "Documentation",
    "Data",
]

COMPUTE_SHEETS_TO_COPY = [
    "Version",
    "Input parameters",
    "FC - Car & PTW pred.",
    "FC - Car & PTW robust. pred.",
    "FC - Ped & Cyc pred.",
    "FC - Ped & Cyc robust. pred.",
    "FC - Driver State Link",
    "LDC - Single Veh pred.",
    "LDC - Car & PTW pred.",
    "LDC - robust. pred.",
    "LSC - Car & PTW pred.",
    "LSC - Ped & Cyc pred.",
    "Documentation",
    "Data",
]

# Sheets that carry OEM input via cell formatting (background color, merged
# multi-level headers) rather than plain text -- hard_copy_sheet above is
# their only writer. Re-running them through pandas.to_excel below would
# discard that formatting and replace blank header cells with literal
# "Unnamed: N" text.
_FORMATTING_ONLY_SHEETS = {"FC - Driver State Link"}


def get_output_file_path(output_path: str) -> str:
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(output_path, f"ca_{current_datetime}_report.xlsx")


header_align_right = [
    "Score",
    "Max Score",
    "OEM Prediction",
    "Value",
    "Prediction.Check",
    "Colour",
]
header_align_left = [
    "Test",
    "Seat position",
    "Body region",
    "Criteria",
    "Stage",
    "Stage element",
    "Stage subelement",
    "Dummy",
]


CM_TO_EXCEL_WIDTH = 3.78


logger = logging.getLogger(__name__)


def adjust_column_widths(columns):
    """
    Adjusts the width of columns in an Excel worksheet based on the maximum length of cell values.

    Args:
        columns: Iterable of worksheet columns (e.g., ws.columns).
    """
    for column in columns:
        max_length = 0
        if not hasattr(column[0], "column_letter"):
            continue
        column_letter = column[0].column_letter  # Get the column letter
        for cell in column:
            try:
                if cell.value:
                    max_length = max(max_length, len(str(cell.value)))
            except (TypeError, AttributeError) as e:
                logger.warning(f"Error processing cell value '{cell.value}': {e}")
        adjusted_width = max_length + 2  # Add some padding
        column[0].parent.column_dimensions[column_letter].width = adjusted_width


_CPLA_CBLA_PREDICTION_SHEET = "FC - Ped & Cyc pred."


def _derive_cpla_cbla_rows_to_delete(input_file) -> dict:
    """Recompute which "FC - Ped & Cyc pred." worksheet rows the CPLA/CBLA
    AEB-duplicate collapse (test_info.collapse_cpla_cbla_aeb_duplicate_rows)
    would remove, straight from *input_file* -- so callers of write_report
    never need to know this rule exists or thread its result through.

    Pure and idempotent: running it against an already-collapsed sheet finds
    no duplicate occurrence left to match, so it safely no-ops. Every matrix
    is read fresh from the untouched input on every call, so -- unlike
    test_info.preprocess_stage_subelement's internal collapse -- there's no
    cross-matrix row-offset bookkeeping needed: CPLA night's and CBLA's rows
    are already correct in *original* worksheet coordinates even though CPLA
    day is processed first.
    """
    try:
        dfs = common.read_excel_file_to_dfs(input_file)
    except Exception:
        logger.exception(
            "Could not read %s to derive CPLA/CBLA collapsed rows.", input_file
        )
        return {}
    prediction_df = dfs.get(_CPLA_CBLA_PREDICTION_SHEET)
    if prediction_df is None:
        return {}

    rows_to_delete = []
    for test_name in ("CPLA day", "CPLA night", "CBLA"):
        try:
            loadcase_info = read_loadcase_info(
                prediction_df, None, test_name, data_model.StageSubelementKey.FC
            )
            coherent, mismatches = check_cpla_cbla_coherence(loadcase_info.test_points)
            if not coherent:
                logger.warning(
                    "Skipping CPLA/CBLA collapse for %s: coherence check "
                    "failed (%s).",
                    test_name,
                    "; ".join(mismatches),
                )
                continue
            _, removed_matrix_rows, matrix_start_row = (
                collapse_cpla_cbla_aeb_duplicate_rows(
                    prediction_df, loadcase_info.test_points, test_name
                )
            )
        except Exception:
            logger.exception("Could not derive collapsed rows for %s.", test_name)
            continue
        # +2: pandas header=0 offset from 0-based df row index to 1-based
        # worksheet row (matches test_info.preprocess_stage_subelement).
        rows_to_delete.extend(matrix_start_row + row + 2 for row in removed_matrix_rows)

    return {_CPLA_CBLA_PREDICTION_SHEET: rows_to_delete} if rows_to_delete else {}


_STANDARD_RANGE_BOX = Border(
    left=Side(style="thin"),
    right=Side(style="thin"),
    top=Side(style="thin"),
    bottom=Side(style="thin"),
)


def _box_cpla_cbla_standard_range_cells(ws) -> None:
    """Draw (or re-draw) the thin box around every CPLA/CBLA standard-range
    cell in *ws*, derived fresh from the sheet's own current VUT-speed
    layout via matrix_processing.classify_cpla_rows/classify_cbla_rows --
    the same logic already used for scoring. A row promoted to standard
    range at an extra impact location (its choice-band duplicate at that
    speed was collapsed, see test_info.collapse_cpla_cbla_aeb_duplicate_rows)
    gets a matching box on the newly-promoted cell too, not just the one it
    already had.

    Applied additively (cell.border + _STANDARD_RANGE_BOX) so an existing
    side -- e.g. the thick edge closing a matrix block -- survives; a cell
    that already has the box is re-applied harmlessly.

    Cosmetic only: any failure is logged and swallowed rather than breaking
    report generation.
    """
    try:
        rows = list(ws.values)
        if not rows:
            return
        pred_df = pd.DataFrame(rows[1:], columns=rows[0])

        for test_name in ("CPLA day", "CPLA night", "CBLA"):
            matrix_indices = matrix_processing.get_matrix_indices(pred_df, test_name)
            if not matrix_indices:
                continue
            start_row = matrix_indices["start_row"]
            start_col = matrix_indices["start_col"]
            n_cols = matrix_indices["n_cols"]
            n_rows = common.get_n_rows(pred_df, start_row, col_idx=0)

            # Column label ("50%", "25%", ...) for each matrix column, from
            # the header row directly above the data -- same lookup as
            # matrix_processing._extract_test_point_attributes.
            col_labels = {}
            for col_j in range(n_cols):
                raw = pred_df.iloc[start_row - 1, start_col + col_j]
                if raw is None:
                    continue
                col_labels[col_j] = str(int(float(raw) * 100)) + "%"

            row_vut_speeds = [pred_df.iloc[start_row + i, 0] for i in range(n_rows)]
            if test_name == "CBLA":
                row_target_speeds = [
                    pred_df.iloc[start_row + i, 1] for i in range(n_rows)
                ]
                standard_ils_by_row = matrix_processing.classify_cbla_rows(
                    row_vut_speeds, row_target_speeds
                )
            else:
                standard_ils_by_row = {
                    row: info.standard_ils
                    for row, info in matrix_processing.classify_cpla_rows(
                        row_vut_speeds
                    ).items()
                }

            for i in range(n_rows):
                standard_ils = standard_ils_by_row.get(i)
                if not standard_ils:
                    continue
                for col_j, label in col_labels.items():
                    if label not in standard_ils:
                        continue
                    cell = ws.cell(row=start_row + i + 2, column=start_col + col_j + 1)
                    cell.border = cell.border + _STANDARD_RANGE_BOX
    except Exception:
        logger.exception("Could not box CPLA/CBLA standard range cells in %s.", ws)


def write_report(
    command,
    input_file,
    updated_dfs,
    output_path,
    format_prediction_cells=True,
    rows_to_delete=None,
):
    """
    Writes the report Excel file based on the provided command, input file, updated DataFrames, and output path.

    Args:
        command: The CLI command indicating the context of the report (e.g., PREPROCESS, COMPUTE_SCORE).
        input_file: Path to the input Excel file containing the template sheets to copy.
        updated_dfs: Dictionary of updated DataFrames to be written to the report, keyed by sheet name.
        output_path: Directory where the output Excel file will be saved.
        format_prediction_cells: Boolean flag to determine whether to apply color formatting to prediction cells.
        rows_to_delete: Optional dict of sheet name -> list of 1-based worksheet
            rows to delete from that sheet's hard copy, merged with the rows
            this function derives itself for CPLA/CBLA (see
            _derive_cpla_cbla_rows_to_delete) -- callers do not need to
            compute or pass those explicitly. Applied right after the sheets
            are copied from the input template, before any formatting below
            runs.
    """
    if command == common.CliCommand.PREPROCESS:
        output_file = os.path.join(output_path, "ca_preprocessed_template.xlsx")
        sheets_to_copy = PREPROCESS_SHEETS_TO_COPY
    elif command == common.CliCommand.COMPUTE_SCORE:
        output_file = get_output_file_path(output_path)
        sheets_to_copy = COMPUTE_SHEETS_TO_COPY
    else:
        raise ValueError(f"Unsupported command: {command}")

    prediction_mark_coordinates = (
        common.build_crash_avoidance_prediction_mark_coordinates(updated_dfs)
    )

    # Copy base sheets from input to output via workbook objects
    input_wb = load_or_create_workbook(input_file)
    output_wb = load_or_create_workbook(output_file)
    for sheet in sheets_to_copy:
        hard_copy_sheet(input_wb, sheet, output_wb)

    rows_to_delete = dict(rows_to_delete or {})
    if _CPLA_CBLA_PREDICTION_SHEET in sheets_to_copy:
        for sheet_name, rows in _derive_cpla_cbla_rows_to_delete(input_file).items():
            rows_to_delete.setdefault(sheet_name, []).extend(rows)

    for sheet_name, rows in rows_to_delete.items():
        if rows and sheet_name in output_wb.sheetnames:
            common.delete_rows_keeping_merges(output_wb[sheet_name], rows)

    if (
        format_prediction_cells
        and _CPLA_CBLA_PREDICTION_SHEET in sheets_to_copy
        and _CPLA_CBLA_PREDICTION_SHEET in output_wb.sheetnames
    ):
        _box_cpla_cbla_standard_range_cells(output_wb[_CPLA_CBLA_PREDICTION_SHEET])

    output_wb.save(output_file)

    # Write updated DataFrames via ExcelWriter to preserve all rows
    try:
        with pd.ExcelWriter(
            output_file, engine="openpyxl", mode="a", if_sheet_exists="replace"
        ) as writer:
            for sheet_name, updated_df in updated_dfs.items():
                if "pred." in sheet_name or sheet_name in _FORMATTING_ONLY_SHEETS:
                    continue
                updated_df.to_excel(writer, sheet_name=sheet_name, index=False)
    except Exception as e:
        logger.error(f"Failed to write updated sheets: {e}")

    wb = load_workbook(output_file)

    calculation_color = PatternFill(
        start_color="FFDD04", end_color="FFDD04", fill_type="solid"
    )
    input_color = PatternFill(
        start_color=common.INPUT_CELL_RGB,
        end_color=common.INPUT_CELL_RGB,
        fill_type="solid",
    )

    header_fill = PatternFill(
        start_color="000000", end_color="000000", fill_type="solid"
    )
    header_font = Font(name="Calibri", color="FFFFFF", bold=True)
    sheetnames = wb.sheetnames

    for sheet_name in sheetnames:
        ws = wb[sheet_name]

        if sheet_name == "Version":
            white_font = Font(color="FFFFFF", name="Aptos Narrow")
            black_fill = PatternFill(
                start_color="000000", end_color="000000", fill_type="solid"
            )
            left_align = Alignment(horizontal="left")
            first_col_width = 10 / CM_TO_EXCEL_WIDTH  # Convert cm to Excel width units
            ws.column_dimensions["A"].width = first_col_width
            for row in ws.iter_rows():
                for cell in row:
                    cell.font = white_font
                    cell.fill = black_fill
                    cell.alignment = left_align
            continue

        if "robust. pred." in sheet_name:
            for col in ws.iter_cols():
                for cell in col[1:]:
                    if isinstance(cell.value, str):
                        if cell.value.strip().upper() == "YES":
                            cell.fill = PatternFill(
                                start_color="E8F2A1",
                                end_color="E8F2A1",
                                fill_type="solid",
                            )  # Light green
                        elif cell.value.strip().upper() == "NO":
                            cell.fill = PatternFill(
                                start_color="FFA6A6",
                                end_color="FFA6A6",
                                fill_type="solid",
                            )  # Light red

        ###############################################################
        # Align text
        ###############################################################
        for col in ws.iter_cols():
            header = col[0].value
            align = Alignment(horizontal="center")
            if header in header_align_right:
                align = Alignment(horizontal="right")
            elif header in header_align_left:
                align = Alignment(horizontal="left")

            for cell in col:
                cell.alignment = align

            if col[0].value in ["Colour", "OEM Prediction"]:
                for cell in col[1:]:
                    if (
                        isinstance(cell.value, str)
                        and cell.value
                        and str(cell.value).lower() not in ["nan", "none"]
                    ):
                        cell.value = cell.value.capitalize()

        common.set_number_precision(ws)
        common.set_title_capitalization(ws)

        ###############################################################
        # Color columns based on their headers
        ###############################################################
        for col in ws.iter_cols():
            if col[0].value in [
                "Score",
                "Points",
                "Prediction.Check",
                "Body regionscore",
                "Modifiers",
                "Modifier",
                "Colour",
                "Capping?",
            ]:
                for cell in col[1:]:  # Skip the first row (header)
                    cell.fill = calculation_color
                    if common.is_empty_cell(cell.value):
                        cell.value = ""
            elif col[0].value in [
                "Inspection [%]",
                "Value",
                "Robustness",
                "OEM Prediction",
            ]:
                for cell in col[1:]:  # Skip the first row (header)
                    if (
                        sheet_name
                        in [
                            "FC - Car & PTW verif.",
                            "FC - Ped & Cyc verif.",
                            "LDC - Single Veh verif.",
                            "LDC - Car & PTW verif.",
                            "LSC - Car & PTW verif.",
                            "LSC - Ped & Cyc verif.",
                        ]
                        and col[0].value == "OEM Prediction"
                    ):
                        logger.debug(
                            f"Skipping OEM Prediction color fill for sheet {sheet_name} and column {col[0].value}"
                        )
                    else:
                        cell.fill = input_color
                    if common.is_empty_cell(cell.value):
                        cell.value = ""

        ###############################################################
        # Grey background on "Value baseline" cells for CPMFC rows
        ###############################################################
        if "verif." in sheet_name:
            header_row = 6 if "LDC - Single Veh" in sheet_name else 4
            headers = [cell.value for cell in ws[header_row]]
            if "Scenario" in headers and "Value baseline" in headers:
                scenario_col_idx = headers.index("Scenario") + 1
                vb_col_idx = headers.index("Value baseline") + 1
                for row in ws.iter_rows(min_row=header_row + 1):
                    if (
                        ws.cell(row=row[0].row, column=scenario_col_idx).value
                        == "CPMFC"
                    ):
                        ws.cell(row=row[0].row, column=vb_col_idx).fill = input_color

    ###############################################################
    # Adjust column widths based on content
    ###############################################################
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]

        if "pred." in sheet_name:
            for i, column in enumerate(ws.columns):
                if i < 4:
                    adjust_column_widths([column])
                else:
                    col_letter = get_column_letter(i)
                    ws.column_dimensions[col_letter].width = 2.25 * CM_TO_EXCEL_WIDTH
        else:
            adjust_column_widths(ws.columns)

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        logger.debug(f"Adding borders to sheet: {sheet_name}")
        max_row = ws.max_row
        max_col = ws.max_column
        # Add thick border to the last column for all rows including header
        for row in range(1, max_row + 1):  # Start from 1 to include header
            cell = ws.cell(row=row, column=max_col)
            cell.border = cell.border + Border(
                right=Side(border_style="thick", color="000000")
            )
        # Add thick border to the last row for all columns
        for col in range(1, max_col + 1):
            cell = ws.cell(row=max_row, column=col)
            cell.border = cell.border + Border(
                bottom=Side(border_style="thick", color="000000")
            )

    # Set font size to 11 and center alignment for all cells in the VRU Prediction sheet
    font_large = Font(name="Calibri", size=11)
    center_align = Alignment(horizontal="center", vertical="center")
    prediction_mark_coordinates = prediction_mark_coordinates or {}

    if format_prediction_cells:
        for index, stage_info in enumerate(data_model.STAGE_SUBELEMENTS):
            stage_element = stage_info["Stage element"]
            stage_subelement = stage_info["Stage subelement"]
            sheet_prefix = get_sheet_prefix(stage_element, stage_subelement)
            stage_subelement_key = get_stage_subelement_key(stage_element)
            prediction_sheet_name = f"{sheet_prefix} pred."

            if prediction_sheet_name not in [
                "FC - Car & PTW pred.",
                "FC - Ped & Cyc pred.",
                "LDC - Single Veh pred.",
                "LDC - Car & PTW pred.",
                "LSC - Car & PTW pred.",
                "LSC - Ped & Cyc pred.",
            ]:
                continue

            # A pred sheet missing from the input workbook was skipped by
            # hard_copy_sheet above, so there is nothing to format either.
            if prediction_sheet_name not in wb.sheetnames:
                continue

            pred_ws = wb[prediction_sheet_name]
            rows = list(pred_ws.values)
            pred_df = pd.DataFrame(rows[1:], columns=rows[0])
            # Remove the value of stage_subelement_key + " - " from sheet_prefix
            prefix_to_remove = f"{stage_subelement_key.value} - "
            if sheet_prefix.startswith(prefix_to_remove):
                subdict_key = sheet_prefix[len(prefix_to_remove) :]

            loadcase_list = data_model.STAGE_SUBELEMENT_TO_LOADCASES[
                stage_subelement_key
            ][subdict_key]
            for scenario in loadcase_list:
                matrix_indices = matrix_processing.get_matrix_indices(pred_df, scenario)
                if matrix_indices:
                    n_rows = common.get_n_rows(
                        pred_df, matrix_indices["start_row"], col_idx=0
                    )
                else:
                    n_rows = 0

                for row_index, row in enumerate(pred_ws.iter_rows()):
                    for col_index, cell in enumerate(row):
                        key = str(cell.value).lower()
                        if (
                            matrix_indices
                            and matrix_indices["start_row"] + 1
                            <= row_index
                            <= matrix_indices["start_row"] + n_rows
                            and matrix_indices["start_col"]
                            <= col_index
                            < matrix_indices["start_col"] + matrix_indices["n_cols"]
                        ):
                            logger.debug(
                                f"Processing cell at row {row_index + 1}, col {col_index + 1} key: {key}"
                            )
                            if key in PREDICTION_COLOR_MAP:
                                color = PREDICTION_COLOR_MAP[key]
                            else:
                                color = None
                            logger.debug(
                                f"Setting cell at row {row_index + 1}, col {col_index + 1} to color: {color}"
                            )
                            if color:
                                color_int = tuple(int(round(c * 255)) for c in color)
                                color_hex = "{:02X}{:02X}{:02X}".format(*color_int)
                                fill = PatternFill(
                                    start_color=color_hex,
                                    end_color=color_hex,
                                    fill_type="solid",
                                )
                                cell.fill = fill
                            if not isinstance(cell, MergedCell):
                                cell.value = ""

            sheet_coordinates = prediction_mark_coordinates.get(
                prediction_sheet_name, []
            )
            if not sheet_coordinates:
                logger.info(
                    f"No selected points found in sheet {prediction_sheet_name}."
                )
                continue

            for point_row, point_col in sheet_coordinates:
                cell_to_mark = pred_ws.cell(row=point_row, column=point_col)
                logger.debug(f"Marking point at row={point_row}, col={point_col}")
                logger.debug(
                    f"Updated coordinates = row={cell_to_mark.row}, col={cell_to_mark.column}"
                )
                cell_to_mark.value = "X"
                cell_to_mark.alignment = Alignment(
                    horizontal="center", vertical="center"
                )
                cell_to_mark.font = font_large
                cell_to_mark.alignment = center_align

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        ###############################################################
        # Set first row (header) to black background and white text
        ###############################################################

        # Original header formatting for first row
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font

        # Remove everything until the first " - " occurrence, included
        stage_subelement = sheet_name.replace(" pred.", "")
        if " - " in stage_subelement:
            parts = stage_subelement.split(" - ", 1)
            stage_subelement = parts[1]
            prefix = parts[0]
        else:
            stage_subelement = None
            prefix = None
        scenario_list = data_model.STAGE_SUBELEMENT_TO_LOADCASES.get(prefix, {}).get(
            stage_subelement, []
        )

        if len(scenario_list) == 0:
            continue

        # Apply header formatting to row-1 and row-2 of every start_row in MATRIX_INDICES (if sheet matches)

        rows = list(ws.values)
        pred_df = (
            pd.DataFrame(rows[1:], columns=rows[0]) if len(rows) > 1 else pd.DataFrame()
        )
        for scenario in data_model.MATRIX_INDICES.keys():
            logger.debug(f"Processing scenario: {scenario}")
            # Only apply if this sheet is relevant (sheet_name contains scenario or is a prediction sheet)
            if any(scenario in s for s in scenario_list):
                indices = matrix_processing.get_matrix_indices(pred_df, scenario)
                if not indices:
                    continue
                start_row = indices["start_row"]
                for offset in [0, 1]:  # row-1 and row-2
                    row_num = start_row - offset + 1  # openpyxl is 1-based
                    if row_num > 0 and row_num <= ws.max_row:
                        for cell in ws[row_num]:
                            cell.fill = header_fill
                            cell.font = header_font
                last_num = start_row - 1  # openpyxl is 1-based
                if last_num > 0 and last_num <= ws.max_row:
                    for cell in ws[last_num]:
                        cell.value = ""
                        cell.border = cell.border + Border(
                            top=Side(border_style="thick", color="000000")
                        )
                        cell.border = Border(
                            left=cell.border.left,
                            top=cell.border.top,
                            bottom=cell.border.bottom,
                            right=None,
                        )
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        if "verif." in sheet_name:
            if "LDC - Single Veh" in sheet_name:
                header_row = 6
            else:
                header_row = 4
            common.set_header_by_row(
                ws,
                header_row,
                header_fill=header_fill,
                header_font=header_font,
            )

    ###############################################################
    # Loop through all sheets, find pass/fail columns, and apply data validation.
    for ws in wb.worksheets:
        headers = [cell.value for cell in ws[1]]
        pass_fail_col_indices = []

        if "Value" in headers:
            value_col_idx = headers.index("Value") + 1  # openpyxl is 1-based
            pass_fail_col_indices.append(value_col_idx)

            if "verif." in ws.title and value_col_idx > 1:
                pass_fail_col_indices.append(value_col_idx - 1)
            elif "Robustness" in headers:
                pass_fail_col_indices.append(headers.index("Robustness") + 1)

        for pass_fail_col_idx in dict.fromkeys(pass_fail_col_indices):
            dv = DataValidation(type="list", formula1='"PASS,FAIL"', allow_blank=False)
            ws.add_data_validation(dv)
            for row in range(2, ws.max_row + 1):
                cell = ws.cell(row=row, column=pass_fail_col_idx)
                if isinstance(cell.value, str) and cell.value.strip().upper() in [
                    "PASS",
                    "FAIL",
                ]:
                    cell.fill = input_color
                    dv.add(cell)

    ###############################################################
    # "General requirements" / "Driveability" / "Driver state link" rows
    # are manual OEM inputs that start blank, so the loop above (which only
    # adds validation to cells already containing "PASS"/"FAIL") never
    # reaches their Value cell. Add the same PASS/FAIL dropdown there,
    # scoped to just those rows.
    ###############################################################
    general_requirement_row_labels = {
        "General requirements",
        "Driveability",
        "Driver state link",
    }
    for ws in wb.worksheets:
        if "verif." not in ws.title:
            continue
        headers = [cell.value for cell in ws[1]]
        if "Scenario" not in headers or "Value" not in headers:
            continue
        scenario_col_idx = headers.index("Scenario") + 1
        value_col_idx = headers.index("Value") + 1
        dv = DataValidation(type="list", formula1='"PASS,FAIL"', allow_blank=True)
        ws.add_data_validation(dv)
        for row in range(2, ws.max_row + 1):
            scenario_cell = ws.cell(row=row, column=scenario_col_idx)
            if scenario_cell.value in general_requirement_row_labels:
                value_cell = ws.cell(row=row, column=value_col_idx)
                value_cell.fill = input_color
                dv.add(value_cell)

    for ws in wb.worksheets:
        common.reset_empty_row_background(ws)

    # Reorder sheets according to the specified order
    def reorder_sheets(workbook):
        # Define the main order
        main_order = [
            "Input parameters",
            "Test Scores",
            "Category Scores",
            "Scenario Scores",
            # Matches its position in the pristine ca_template.xlsx.
            "FC - Driver State Link",
        ]
        # Define the prefixes for the remaining groups
        prefixes = [
            "FC - Car & PTW",
            "FC - Ped & Cyc",
            "LDC - Single Veh",
            "LDC - Car & PTW",
            "LSC - Car & PTW",
            "LSC - Ped & Cyc",
        ]
        suffixes = ["pred.", "robust. pred.", "verif."]

        # Build the desired order
        desired_order = []
        # Add main sheets if present
        for sheet in main_order:
            if sheet in workbook.sheetnames:
                desired_order.append(sheet)
        # Add grouped sheets in the specified order
        for prefix in prefixes:
            for suffix in suffixes:
                sheet_name = f"{prefix} {suffix}"
                if sheet_name in workbook.sheetnames:
                    desired_order.append(sheet_name)
        # Insert "LDC - robust. pred." after all LDC sheets if present
        ldc_robustness = "LDC - robust. pred."
        if ldc_robustness in workbook.sheetnames:
            # Find the last LDC sheet in desired_order
            ldc_indices = [
                i for i, name in enumerate(desired_order) if name.startswith("LDC - ")
            ]
            if ldc_indices:
                insert_pos = max(ldc_indices) + 1
            else:
                insert_pos = len(desired_order)
            # Remove if already present
            if ldc_robustness in desired_order:
                desired_order.remove(ldc_robustness)
            desired_order.insert(insert_pos, ldc_robustness)
        # Add any remaining sheets that were not specified
        for sheet in workbook.sheetnames:
            if sheet not in desired_order:
                desired_order.append(sheet)

        # Reorder sheets in workbook
        for idx, sheet_name in enumerate(desired_order):
            if workbook.sheetnames[idx] != sheet_name:
                sheet = workbook[sheet_name]
                workbook._sheets.insert(
                    idx, workbook._sheets.pop(workbook._sheets.index(sheet))
                )

    reorder_sheets(wb)
    common.hide_documentation_data_sheets(wb)
    common.hide_integrity_sheet(wb)
    common.add_version_sheet_to_wb(wb, command)
    if command == common.CliCommand.PREPROCESS:
        # Sign the protected (non-grey) reference cells so compute-score can
        # detect if this preprocessed file gets edited outside the tool
        # before scoring.
        from euroncap_rating_2026.crash_avoidance import integrity as ca_integrity

        common.add_integrity_sheet_to_wb(wb, ca_integrity.SHEET_SCHEMAS)
    wb.save(output_file)

    if command == common.CliCommand.COMPUTE_SCORE:
        print(
            f"Log available at {os.path.join(output_path, 'euroncap_rating_2026.log')}"
        )
        print(" " * 40)
        print(f"Final report available at {output_file}")
        logger.info(f"Final report available at {output_file}")
    elif command == common.CliCommand.PREPROCESS:
        print(
            f"Log available at {os.path.join(output_path, 'euroncap_rating_2026.log')}"
        )
        print(" " * 40)
        print(f"Final report available at {output_file}")
        logger.info(f"Final report available at {output_file}")
