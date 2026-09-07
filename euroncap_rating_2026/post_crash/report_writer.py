# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from datetime import datetime
import os
import pandas as pd

from openpyxl.styles import PatternFill, Font, Alignment, Border, Side


from euroncap_rating_2026.post_crash import data_model
from euroncap_rating_2026 import common

import logging
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation


CM_TO_EXCEL = 3.78

header_align_right = [
    "Score",
    "Max Score",
    "OEM.Prediction",
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


logger = logging.getLogger(__name__)

PREPROCESS_SHEETS_TO_COPY = [
    "Version",
    "Input parameters",
    "Test Scores",
    "Category Scores",
    "Scenario Scores",
    "Documentation",
    "Data",
]

COMPUTE_SHEETS_TO_COPY = [
    "Version",
    "Input parameters",
    "RI - RS Verification",
    "RI - ERG Verification",
    "PCI - Advanced eCall Verif",
    "PCI - MCB & Hazard lights Verif",
    "VE - Energy Management Verif",
    "VE - Occupant Extrication Verif",
    "Documentation",
    "Data",
]


def get_output_file_path(output_path: str) -> str:
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(output_path, f"pc_{current_datetime}_report.xlsx")
    return output_file


def set_dynamic_header(ws, header_fill, header_font):
    # Find the second row where the first column has value 'Scenario'
    scenario_row_indices = []
    for idx, row in enumerate(
        ws.iter_rows(min_row=1, max_row=ws.max_row, min_col=1, max_col=1),
        start=1,
    ):
        cell = row[0]
        if cell.value == "Scenario":
            scenario_row_indices.append(idx)
    if len(scenario_row_indices) >= 2:
        header_row = scenario_row_indices[1]
        for row in ws.iter_rows(
            min_row=header_row,
            max_row=header_row,
            min_col=1,
            max_col=ws.max_column,
        ):
            for cell in row:
                cell.fill = header_fill
                cell.font = header_font


def set_header(ws, col_idx, value, header_fill, header_font):
    header_row = [
        i
        for i, cell in enumerate(
            ws.iter_rows(
                min_row=1, max_row=ws.max_row, min_col=col_idx, max_col=col_idx
            ),
            start=1,
        )
        for c in cell
        if c.value == value
    ][0]
    for row in ws.iter_rows(
        min_row=header_row,
        max_row=header_row,
        min_col=1,
        max_col=ws.max_column,
    ):
        for cell in row:
            cell.fill = header_fill
            cell.font = header_font


def adjust_column_widths(columns):
    """
    Adjusts the width of columns in an Excel worksheet based on the maximum length of cell values.
    Sets a maximum width and enables text wrapping if content exceeds it.

    Args:
        columns: Iterable of worksheet columns (e.g., ws.columns).
    """
    max_width = 150  # Set maximum column width
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
        adjusted_width = min(max_length + 2, max_width)  # Cap at max_width
        column[0].parent.column_dimensions[column_letter].width = adjusted_width


def write_report(
    command: common.CliCommand,
    input_file: str,
    updated_dfs: dict,
    selected_points_dict,
    output_path: str,
    format_prediction_cells=True,
    sheet_order=None,
):
    if command == common.CliCommand.PREPROCESS:
        output_file = os.path.join(output_path, "pc_preprocessed_template.xlsx")
        sheets_to_copy = PREPROCESS_SHEETS_TO_COPY
    elif command == common.CliCommand.COMPUTE_SCORE:
        output_file = get_output_file_path(output_path)
        sheets_to_copy = COMPUTE_SHEETS_TO_COPY
    else:
        logger.error(f"Unsupported command: {command}")
        return

    print(f"Saving to {output_file}...")

    input_wb = common.load_or_create_workbook(input_file)
    wb = common.load_or_create_workbook(output_file)
    for sheet in sheets_to_copy:
        common.hard_copy_sheet(input_wb, sheet, wb)
    wb.save(output_file)

    try:
        with pd.ExcelWriter(
            output_file, engine="openpyxl", mode="a", if_sheet_exists="replace"
        ) as writer:
            for sheet_name, updated_df in updated_dfs.items():
                updated_df.to_excel(writer, sheet_name=sheet_name, index=False)
    except Exception as e:
        logger.error(f"Failed to write updated sheets: {e}")

    wb = common.load_or_create_workbook(output_file)

    calculation_color = PatternFill(
        start_color="FFDD04", end_color="FFDD04", fill_type="solid"
    )
    input_color = PatternFill(
        start_color=common.INPUT_CELL_RGB,
        end_color=common.INPUT_CELL_RGB,
        fill_type="solid",
    )

    sheetnames = wb.sheetnames

    for sheet_name in sheetnames:
        ws = wb[sheet_name]
        logger.debug(f"Formatting sheet: {sheet_name}")
        if sheet_name == "Version":
            white_font = Font(color="FFFFFF", name="Aptos Narrow")
            black_fill = PatternFill(
                start_color="000000", end_color="000000", fill_type="solid"
            )
            left_align = Alignment(horizontal="left")
            first_col_width = 10 / CM_TO_EXCEL  # Convert cm to Excel width units
            ws.column_dimensions["A"].width = first_col_width
            for row in ws.iter_rows():
                for cell in row:
                    cell.font = white_font
                    cell.fill = black_fill
                    cell.alignment = left_align
            continue

        ###############################################################
        # Align text
        ###############################################################
        for col in ws.iter_cols():
            header = col[0].value
            align = Alignment(horizontal="left")
            if header in header_align_right:
                align = Alignment(horizontal="right")
            elif header in header_align_left:
                align = Alignment(horizontal="left")

            for cell in col:
                cell.alignment = align

            if col[0].value in ["Colour", "OEM.Prediction"]:
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
            elif col[0].value in ["Inspection [%]", "Value", "OEM.Prediction"]:
                for cell in col[1:]:  # Skip the first row (header)
                    if (
                        sheet_name
                        in [
                            "FC - Car & PTW Verification",
                            "FC - Ped & Cyc Verification",
                            "LDC - Single Veh Verification",
                            "LDC - Car & PTW Verification",
                            "LSC - Car & PTW Verification",
                            "LSC - Ped & Cyc Verification",
                        ]
                        and col[0].value == "OEM.Prediction"
                    ):
                        logger.debug(
                            f"Skipping OEM.Prediction color fill for sheet {sheet_name} and column {col[0].value}"
                        )
                    else:
                        cell.fill = input_color
                    if common.is_empty_cell(cell.value):
                        cell.value = ""

    ###############################################################
    # Adjust column widths based on content
    ###############################################################
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        adjust_column_widths(ws.columns)

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        logger.debug(f"Adding borders to sheet: {sheet_name}")
        # Add thick border to the last column for all rows including header
        for row in range(1, ws.max_row + 1):  # Start from 1 to include header
            cell = ws.cell(row=row, column=ws.max_column)
            cell.border = cell.border + Border(
                right=Side(border_style="thick", color="000000")
            )
        # Add thick border to the last row for all columns
        for col in range(1, ws.max_column + 1):
            cell = ws.cell(row=ws.max_row, column=col)
            cell.border = cell.border + Border(
                bottom=Side(border_style="thick", color="000000")
            )

    # Set font size to 11 and center alignment for all cells in the VRU Prediction sheet
    font_large = Font(name="Calibri", size=11)
    center_align = Alignment(horizontal="center", vertical="center")

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        # Skip formatting if worksheet is empty
        if ws.max_row < 2 and ws.max_column < 2:
            continue
        ###############################################################
        # Set first row (header) to black background and white text
        ###############################################################
        header_fill = PatternFill(
            start_color="000000", end_color="000000", fill_type="solid"
        )
        header_font = Font(name="Calibri", color="FFFFFF", bold=True)
        # Original header formatting for first row
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font

    ###############################################################
    # Loop through all sheets, find 'Value' column, and apply data validation to PASS/FAIL cells
    #
    # A blank Value cell means "not yet assessed by the OEM" and must still
    # get a dropdown (not just cells that already say PASS/FAIL/a leadtime
    # band) -- otherwise a freshly-generated sheet's default blank cells
    # would have no dropdown at all. "Input parameters" is excluded above
    # since its "Value" column isn't PASS/FAIL at all; every other sheet
    # reaching this loop is one of the 6 "...Verif" sheets, so a blank cell
    # here is unambiguous -- except VE - Energy Management Verif, which
    # mixes ordinary PASS/FAIL rows with one leadtime-band row (identified
    # by Scenario) in the same Value column.
    for ws in wb.worksheets:
        headers = [cell.value for cell in ws[1]]
        if ws.title == "Input parameters":
            continue
        if "Value" in headers:
            value_col_idx = headers.index("Value") + 1  # openpyxl is 1-based
            scenario_col_idx = (
                headers.index("Scenario") + 1 if "Scenario" in headers else None
            )
            dv = DataValidation(type="list", formula1='"PASS,FAIL"', allow_blank=True)
            ws.add_data_validation(dv)
            duration_90_min_dv = DataValidation(
                type="list",
                formula1='"≥90 min,>40 min,>20 min,≤20 min"',
                allow_blank=True,
            )
            ws.add_data_validation(duration_90_min_dv)
            for row in range(2, ws.max_row + 1):
                cell = ws.cell(row=row, column=value_col_idx)
                is_leadtime_row = (
                    scenario_col_idx is not None
                    and str(
                        ws.cell(row=row, column=scenario_col_idx).value or ""
                    ).strip()
                    == "Fulfilment of UN-R100.03 with predefined leadtime"
                )
                if isinstance(cell.value, str) and cell.value.strip().upper() in [
                    "PASS",
                    "FAIL",
                ]:
                    dv.add(cell)
                elif isinstance(cell.value, str) and cell.value.strip().upper() in [
                    "≥90 MIN",
                    ">40 MIN",
                    ">20 MIN",
                    "≤20 MIN",
                ]:
                    duration_90_min_dv.add(cell)
                elif common.is_empty_cell(cell.value):
                    (duration_90_min_dv if is_leadtime_row else dv).add(cell)

    # Reorder sheets according to the specified order
    def reorder_sheets(workbook, main_order=None):
        # Define the main order
        if main_order is None:
            main_order = [
                "Input parameters",
                "Test Scores",
                "Category Scores",
                "Scenario Scores",
                "RI - RS Verification",
                "RI - ERG Verification",
                "PCI - Advanced eCall Verif",
                "PCI - MCB & Hazard lights Verif",
                "VE - Energy Management Verif",
                "VE - Occupant Extrication Verif",
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

    if command == common.CliCommand.PREPROCESS:
        # Dash the not-yet-computed columns on the openpyxl workbook: the
        # score sheets are hard-copied with their full template formatting
        # (borders, row heights, number formats), which a pandas sheet
        # replacement would discard. The DataFrame path gets the same
        # dash via common.reset_computed_columns_to_dash instead.
        from euroncap_rating_2026.post_crash import integrity as pc_integrity

        common.blank_computed_columns(wb, pc_integrity.COMPUTED_SHEET_COLUMNS)
    reorder_sheets(wb, sheet_order)
    common.hide_documentation_data_sheets(wb)
    common.hide_integrity_sheet(wb)
    common.add_version_sheet_to_wb(wb, command)
    if command == common.CliCommand.PREPROCESS:
        # Sign the protected (non-grey) reference cells so compute-score can
        # detect if this preprocessed file gets edited outside the tool
        # before scoring.
        from euroncap_rating_2026.post_crash import integrity as pc_integrity

        common.add_integrity_sheet_to_wb(wb, pc_integrity.SHEET_SCHEMAS)
    wb.save(output_file)

    if command == common.CliCommand.COMPUTE_SCORE:
        print(
            f"Log available at {os.path.join(output_path, 'euroncap_rating_2026.log')}"
        )
        print(" " * 40)
        print(f"Final report available at {output_file}")
        logger.info(f"Final report available at {output_file}")
