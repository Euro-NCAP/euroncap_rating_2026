# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from datetime import datetime
import os
from openpyxl.utils import get_column_letter
import pandas as pd

from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from euroncap_rating_2026.crash_protection.vru_processing import (
    VRU_PREDICTION_COLOR_MAP,
    VruPredictionColor,
)
from euroncap_rating_2026 import common


header_align_right = [
    "Body regionscore",
    "Modifiers",
    "Inspection [%]",
    "Score",
    "Max score",
    "HPL",
    "LPL",
    "Capping",
    "OEM Prediction",
    "Value",
    "Prediction.Check",
    "Colour",
    "Points",
    "Modifier",
]
header_align_left = [
    "Loadcase",
    "Seat position",
    "Body region",
    "Criteria",
    "Stage",
    "Stage element",
    "Stage subelement",
    "Dummy",
    "Version",
]


CM_TO_EXCEL_WIDTH = 3.78

VRU_PREDICTION_MAX_ROW = 29
VRU_PREDICTION_MAX_COL = 25

import logging

logger = logging.getLogger(__name__)

PREPROCESS_SHEETS_TO_COPY = [
    "Version",
    "Test Scores",
    "Input parameters",
    "CP - Dummy Scores",
    "CP - Body region scores",
    "CP - Frontal Offset",
    "CP - Frontal FW",
    "CP - Frontal Sled & VT",
    "CP - Side MDB",
    "CP - Side Pole",
    "CP - Side Farside",
    "CP - Rear Whiplash",
    "CP - VRU Prediction",
    "Documentation",
    "Data",
]

COMPUTE_SHEETS_TO_COPY = [
    "Version",
    "Test Scores",
    "Input parameters",
    "CP - Dummy Scores",
    "CP - Body region scores",
    "CP - VRU Prediction",
    "Documentation",
    "Data",
]


def get_output_file_path(output_path: str) -> str:
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(output_path, f"cp_{current_datetime}_report.xlsx")
    return output_file


def write_report(
    command: common.CliCommand,
    input_file: str,
    updated_dfs: dict,
    output_path: str,
    format_prediction_cells: bool = True,
):
    """Writes the report Excel file based on the provided command, input file, updated DataFrames, and output path.

    Args:
        command: The CLI command indicating the context of the report (e.g., PREPROCESS, COMPUTE_SCORE).
        input_file: Path to the input Excel file containing the template sheets to copy.
        updated_dfs: Dictionary of updated DataFrames to be written to the report, keyed by sheet name.
        output_path: Directory where the output Excel file will be saved.
        format_prediction_cells: Boolean flag to determine whether to apply color formatting to prediction cells.
    """
    if command == common.CliCommand.PREPROCESS:
        output_file = os.path.join(output_path, "cp_preprocessed_template.xlsx")
        sheets_to_copy = PREPROCESS_SHEETS_TO_COPY
    elif command == common.CliCommand.COMPUTE_SCORE:
        output_file = get_output_file_path(output_path)
        sheets_to_copy = COMPUTE_SHEETS_TO_COPY
    else:
        logger.error(f"Unsupported command: {command}")
        return

    print(f"Saving to {output_file}...")

    prediction_mark_coordinates = (
        common.build_crash_protection_prediction_mark_coordinates(updated_dfs)
    )

    input_wb = common.load_or_create_workbook(input_file)
    wb = common.load_or_create_workbook(output_file)
    for sheet in sheets_to_copy:
        common.hard_copy_sheet(input_wb, sheet, wb)

    if (
        command == common.CliCommand.PREPROCESS
        and "CP - Body region scores" in wb.sheetnames
    ):
        # The packaged template carries unused blank columns (L:S) on this
        # sheet past "Max score" (K), including stray merged cells further
        # down (e.g. Q90:Q92) that keep the sheet dimension stretched to S.
        # compute-score overwrites the sheet from a DataFrame that never has
        # them, but preprocess only hard-copies the sheet as-is, so both need
        # to be dropped explicitly here.
        ws = wb["CP - Body region scores"]
        max_col = ws.max_column
        if max_col > 11:
            for merged_range in list(ws.merged_cells.ranges):
                if merged_range.max_col > 11:
                    ws.unmerge_cells(str(merged_range))
            ws.delete_cols(12, max_col - 11)

    wb.save(output_file)

    try:
        with pd.ExcelWriter(
            output_file, engine="openpyxl", mode="a", if_sheet_exists="replace"
        ) as writer:
            for sheet_name, updated_df in updated_dfs.items():
                if sheet_name in {"CP - VRU Prediction Points", "CP - VRU Prediction"}:
                    # "CP - VRU Prediction Points" is an intermediate sheet,
                    # never written to Excel. "CP - VRU Prediction" carries
                    # OEM input via cell background color and a merged
                    # multi-level header (not plain text) -- hard_copy_sheet
                    # above is its only writer; re-running it through
                    # pandas.to_excel would discard that formatting and
                    # replace blank header cells with literal "Unnamed: N"
                    # text.
                    continue
                updated_df.to_excel(writer, sheet_name=sheet_name, index=False)
    except Exception as e:
        logger.error(f"Failed to write updated sheets: {e}")

    wb = common.load_or_create_workbook(output_file)
    prediction_mark_coordinates = prediction_mark_coordinates or {}
    # Dict mapping (excel_row, excel_col) → display label ("X" for headform, "T"/"ST"/"SA" for blue legform)
    vru_mark_coordinates: dict = prediction_mark_coordinates.get(
        "CP - VRU Prediction", {}
    )
    # The master template hides the colour text with a 1 pt font; mark cells
    # need an explicit readable size or the X/T/ST/SA labels are invisible.
    vru_mark_font = Font(name="Aptos Narrow", size=10)

    calculation_color = PatternFill(
        start_color="FFDD04", end_color="FFDD04", fill_type="solid"
    )
    input_color = PatternFill(
        start_color=common.INPUT_CELL_RGB,
        end_color=common.INPUT_CELL_RGB,
        fill_type="solid",
    )
    # If not formatting VRU prediction, keep a list of sheetnames without "CP - VRU Prediction"
    if not format_prediction_cells and "CP - VRU Prediction" in wb.sheetnames:
        sheetnames = [name for name in wb.sheetnames if name != "CP - VRU Prediction"]
        # Move "CP - VRU Prediction" to the second last position in wb._sheets if present
        sheets = wb._sheets
        for i, ws in enumerate(sheets):
            if ws.title == "CP - VRU Prediction":
                vru_ws = sheets.pop(i)
                sheets.insert(len(sheets) - 2, vru_ws)
                break
        sheetnames = [name for name in wb.sheetnames if name != "CP - VRU Prediction"]

    else:
        sheetnames = wb.sheetnames

    for sheet_name in sheetnames:
        if sheet_name == "CP - VRU Prediction":
            continue
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

        ###############################################################
        # Set first row (header) to black background and white text
        ###############################################################
        header_fill = PatternFill(
            start_color="000000", end_color="000000", fill_type="solid"
        )
        header_font = Font(name="Calibri", color="FFFFFF", bold=True)
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font

        ###############################################################
        # Align text
        ###############################################################
        for col in ws.iter_cols():
            header = col[0].value
            if header in header_align_right:
                align = Alignment(horizontal="right")
            else:
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
            elif col[0].value in ["Inspection [%]", "Value", "OEM Prediction"]:
                # For "CP - VRU Pelvis & Leg Impact", build a row→OEM Prediction lookup
                # so SA rows get white (no fill) while T/ST rows get the standard grey fill.
                if (
                    sheet_name == "CP - VRU Pelvis & Leg Impact"
                    and col[0].value == "Value"
                ):
                    oem_col = next(
                        (
                            c
                            for c in ws.iter_cols(1, ws.max_column)
                            if c[0].value == "OEM Prediction"
                        ),
                        None,
                    )
                    oem_by_row = {}
                    if oem_col is not None:
                        for oem_cell in oem_col[1:]:
                            val = oem_cell.value
                            if val and str(val).strip():
                                oem_by_row[oem_cell.row] = str(val).strip().lower()
                    for cell in col[1:]:
                        prediction = oem_by_row.get(cell.row, "")
                        if prediction != "sa":
                            cell.fill = input_color
                        if common.is_empty_cell(cell.value):
                            cell.value = ""
                    continue

                for cell in col[1:]:  # Skip the first row (header)
                    if (
                        sheet_name
                        in ["CP - VRU Head Impact", "CP - VRU Pelvis & Leg Impact"]
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
    # Adjust column widths based on content
    ###############################################################
    for sheet_name in wb.sheetnames:
        if sheet_name == "CP - VRU Prediction":
            continue
        ws = wb[sheet_name]
        for column in ws.columns:
            max_length = 0
            column_letter = column[0].column_letter  # Get the column letter
            for cell in column:
                try:
                    if cell.value:
                        max_length = max(max_length, len(str(cell.value)))
                except (TypeError, AttributeError) as e:
                    logger.warning(f"Error processing cell value '{cell.value}': {e}")
            adjusted_width = max_length + 2  # Add some padding
            ws.column_dimensions[column_letter].width = adjusted_width

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        max_row = ws.max_row
        max_col = ws.max_column
        if sheet_name == "CP - VRU Prediction":
            max_row = min(max_row, VRU_PREDICTION_MAX_ROW)
            max_col = min(max_col, VRU_PREDICTION_MAX_COL)
        prev_value = None
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
        for row in range(1, max_row + 1):  # Skip header row
            cell = ws.cell(row=row, column=1)
            # Set top border when previous value is empty (nan/None) and current is not empty
            if (
                (prev_value is None or str(prev_value).lower() in ["nan", "none", ""])
                and (
                    cell.value is not None
                    and str(cell.value).lower() not in ["nan", "none", ""]
                )
                and row != 2
            ):
                for col in range(1, max_col + 1):
                    ws.cell(row=row, column=col).border = ws.cell(
                        row=row, column=col
                    ).border + Border(top=Side(border_style="thick", color="000000"))
            prev_value = cell.value

    if format_prediction_cells:
        vru_ws = wb["CP - VRU Prediction"]
        vru_max_row = min(vru_ws.max_row, VRU_PREDICTION_MAX_ROW)
        vru_max_col = min(vru_ws.max_column, VRU_PREDICTION_MAX_COL)
        for row in vru_ws.iter_rows(
            min_row=1, max_row=vru_max_row, min_col=1, max_col=vru_max_col
        ):
            for cell in row:
                row_index = cell.row - 1
                col_index = cell.column - 1
                key = str(cell.value).lower()
                if (
                    # A typed "N/A" (an authored dropdown option in the
                    # headform matrix) means "not filled" -- render it
                    # exactly like a blank cell.
                    key in ["nan", "none", "", "n/a"]
                    and col_index > 3
                    and row_index in list(range(3, 22)) + list(range(26, 29))
                ):
                    key = VruPredictionColor.GREY
                if key in VRU_PREDICTION_COLOR_MAP:
                    color = VRU_PREDICTION_COLOR_MAP[key]

                    # Convert float RGB (0-1) to int (0-255) and then to hex
                    color_int = tuple(int(round(c * 255)) for c in color)
                    color_hex = "{:02X}{:02X}{:02X}".format(*color_int)
                    fill = PatternFill(
                        start_color=color_hex,
                        end_color=color_hex,
                        fill_type="solid",
                    )
                    cell.fill = fill
                    if (cell.row, cell.column) in vru_mark_coordinates:
                        cell.value = vru_mark_coordinates[(cell.row, cell.column)]
                        cell.font = vru_mark_font
                        cell.alignment = Alignment(
                            horizontal="center", vertical="center"
                        )
                    else:
                        cell.value = ""

        for (row_index, col_index), label in vru_mark_coordinates.items():
            if row_index > vru_max_row or col_index > vru_max_col:
                continue
            cell = vru_ws.cell(row=row_index, column=col_index)
            cell.value = label
            cell.font = vru_mark_font
            cell.alignment = Alignment(horizontal="center", vertical="center")

        # Set thick black border for all cells
        for row in list(range(4, 23)) + list(
            range(27, 30)
        ):  # Excel rows 4-21 and 27-29 inclusive
            for col in range(5, 26):  # Excel columns 3 to 25 inclusive
                cell = vru_ws.cell(row=row, column=col)
                cell.border = Border(
                    left=Side(border_style="thin", color="000000"),
                    right=Side(border_style="thin", color="000000"),
                    top=Side(border_style="thin", color="000000"),
                    bottom=Side(border_style="thin", color="000000"),
                )

        for idx in range(1, VRU_PREDICTION_MAX_COL + 1):
            col_letter = get_column_letter(idx)
            if idx <= 3:
                vru_ws.column_dimensions[col_letter].width = 2 * CM_TO_EXCEL_WIDTH
            else:
                vru_ws.column_dimensions[col_letter].width = CM_TO_EXCEL_WIDTH

        center_align = Alignment(horizontal="center", vertical="center")

        # Merge cells in the first row for the "Headform" header and keep the original text
        headform_text = vru_ws.cell(
            row=1, column=1
        ).value  # Preserve existing text if any
        vru_ws.merge_cells(start_row=1, start_column=1, end_row=2, end_column=25)
        vru_ws.cell(row=1, column=1).value = headform_text
        vru_ws.cell(row=1, column=1).alignment = center_align
        vru_ws.cell(row=1, column=1).fill = header_fill
        vru_ws.cell(row=1, column=1).font = header_font

        # Merge cells in the first row for the "Headform" header and keep the original text
        centerline_text = vru_ws.cell(
            row=4, column=1
        ).value  # Preserve existing text if any
        vru_ws.merge_cells(start_row=4, start_column=1, end_row=22, end_column=1)
        vru_ws.cell(row=4, column=1).value = centerline_text
        vru_ws.cell(row=4, column=1).alignment = center_align
        vru_ws.cell(row=4, column=1).fill = header_fill
        vru_ws.cell(row=4, column=1).font = header_font
        # Set orientation to 90 degrees (vertical text) for the merged center cell (centerline)
        vru_ws.cell(row=4, column=1).alignment = Alignment(
            horizontal="center", vertical="center", text_rotation=90
        )

        # Merge cells in the 24th row for the "Legform" header and keep the original text
        legform_text = vru_ws.cell(
            row=24, column=1
        ).value  # Preserve existing text if any
        vru_ws.merge_cells(start_row=24, start_column=1, end_row=25, end_column=25)
        vru_ws.cell(row=24, column=1).value = legform_text
        vru_ws.cell(row=24, column=1).alignment = center_align
        vru_ws.cell(row=24, column=1).fill = header_fill
        vru_ws.cell(row=24, column=1).font = header_font

        # Merge cells in rows 24, 27, 28, and 29 for the headers and keep the original text
        for row_num in [27, 28, 29]:
            header_text = vru_ws.cell(
                row=row_num, column=1
            ).value  # Preserve existing text if any
            vru_ws.merge_cells(
                start_row=row_num, start_column=1, end_row=row_num, end_column=4
            )
            vru_ws.cell(row=row_num, column=1).value = header_text
            vru_ws.row_dimensions[row_num].height = 0.55 * 28.35

            vru_ws.cell(row=row_num, column=1).fill = header_fill
            vru_ws.cell(row=row_num, column=1).font = header_font

    # Reorder sheets according to the specified order
    def reorder_sheets(workbook):
        # Define the main order
        main_order = [
            "Input parameters",
            "Test Scores",
            "CP - Dummy Scores",
            "CP - Body region scores",
            "CP - Frontal Offset",
            "CP - Frontal FW",
            "CP - Frontal Sled & VT",
            "CP - Side MDB",
            "CP - Side Pole",
            "CP - Side Farside",
            "CP - Rear Whiplash",
            "CP - VRU Prediction",
            "CP - VRU Head Impact",
            "CP - VRU Pelvis & Leg Impact",
            "Version",
        ]
        # Build the desired order
        desired_order = []
        # Add main sheets if present
        for sheet in main_order:
            if sheet in workbook.sheetnames:
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
        from euroncap_rating_2026.crash_protection import integrity as cp_integrity

        common.add_integrity_sheet_to_wb(wb, cp_integrity.SHEET_SCHEMAS)
    wb.save(output_file)

    if command == common.CliCommand.COMPUTE_SCORE:
        print(
            f"Log available at {os.path.join(output_path, 'euroncap_rating_2026.log')}"
        )
        print(" " * 40)
        print(f"Final report available at {output_file}")
        logger.info(f"Final report available at {output_file}")
    elif command == common.CliCommand.PREPROCESS:
        print(" " * 40)
        print(f"Preprocessed template available at {output_file}")
        logger.info(f"Preprocessed template available at {output_file}")
