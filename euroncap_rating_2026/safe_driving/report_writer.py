# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from datetime import datetime
import os
import pandas as pd

from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side


from euroncap_rating_2026.safe_driving import data_model
from euroncap_rating_2026.safe_driving import acc_performance
from euroncap_rating_2026 import common

import logging
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation


CM_TO_EXCEL = 3.78

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

PREPROCESS_SHEETS_TO_COPY = [
    "Version",
    "Input parameters",
    "Test Scores",
    "Category Scores",
    "Scenario Scores",
    "DE - DM pred.",
    "VA - Speed assist. pred.",
    "VA - ACC pred.",
    "Documentation",
    "Data",
]

COMPUTE_SHEETS_TO_COPY = [
    "Version",
    "Input parameters",
    "DE - DM verif.",
    "DE - GVC verif.",
    "DE - DM pred.",
    "VA - Speed assist. pred.",
    "VA - ACC pred.",
    "Documentation",
    "Data",
]

logger = logging.getLogger(__name__)

# Verif. sheets whose "Value" column is entirely PASS/FAIL (blank cells here
# always mean "not yet assessed" and get the PASS/FAIL dropdown).
PASS_FAIL_BLANK_ELIGIBLE_SHEETS = {
    "OM - Seatbelt usage verif.",
    "OM - Occ. classification verif.",
    "OM - Occ. presence verif.",
    "DE - GVC verif.",
    "DE - DM verif.",
    "VA - Speed assist. verif.",
}
# VA - ACC verif. mixes PASS/FAIL rows (Road features/Auto-resume) with
# numeric ACC-matrix rows in the same "Value" column -- only the former's
# blank cells should get the PASS/FAIL dropdown.
PASS_FAIL_BLANK_ELIGIBLE_ACC_CATEGORIES = {"Road features", "Auto-resume"}

# Rows whose Scenario is "N/A" (Input parameters dropdown) score 0 with no
# user input: their Value cell stays white (no grey fill) and gets no
# PASS/FAIL dropdown. Keyed by sheet, valued by the (forward-filled)
# Category the rule applies to.
NA_VALUE_ROW_CATEGORIES = {
    "OM - Occ. classification verif.": {"Passenger airbag status"},
    "VA - ACC verif.": {"Auto-resume"},
}


def _get_na_value_rows(ws) -> set:
    """1-based worksheet row numbers whose Value cell needs no user input
    because the row's Scenario is "N/A" (see NA_VALUE_ROW_CATEGORIES).

    Detection is content-based on the written worksheet rather than mapped
    from the source DataFrame: "VA - ACC verif."'s layout (block rows, blank
    separator, repeated header above the test-point matrix) makes df-index to
    worksheet-row mapping fragile.
    """
    categories = NA_VALUE_ROW_CATEGORIES.get(ws.title)
    if not categories:
        return set()
    headers = [cell.value for cell in ws[1]]
    if "Category" not in headers or "Scenario" not in headers:
        return set()
    category_col_idx = headers.index("Category") + 1
    scenario_col_idx = headers.index("Scenario") + 1
    max_row, _ = get_populated_sheet_bounds(ws)
    na_rows = set()
    # Category is only written on the first row of a group -- forward-fill it
    # so continuation rows are still recognised (same convention as the
    # PASS/FAIL dropdown loop).
    current_category = None
    for row in range(2, max_row + 1):
        raw_category = str(
            ws.cell(row=row, column=category_col_idx).value or ""
        ).strip()
        if raw_category:
            current_category = raw_category
        if current_category not in categories:
            continue
        scenario = str(ws.cell(row=row, column=scenario_col_idx).value or "").strip()
        if scenario.lower() == "n/a":
            na_rows.add(row)
    return na_rows


def _get_acc_type_of_system(updated_dfs: dict) -> str:
    input_parameters_df = updated_dfs.get("Input parameters", pd.DataFrame())
    if isinstance(input_parameters_df, pd.DataFrame) and not input_parameters_df.empty:
        param_dict = common.create_param_dict_from_input_parameters(
            input_parameters_df, col="Category"
        )
        auto_resume = param_dict.get("Auto-resume", {})
        type_of_system = auto_resume.get("Type of system")
        if pd.notna(type_of_system):
            return str(type_of_system).strip()

    scenario_scores_df = updated_dfs.get("Scenario Scores", pd.DataFrame())
    if isinstance(scenario_scores_df, pd.DataFrame) and not scenario_scores_df.empty:
        category_col = scenario_scores_df.get("Category")
        scenario_col = scenario_scores_df.get("Scenario")
        if category_col is not None and scenario_col is not None:
            mask = (
                category_col.fillna("").astype(str).str.strip().str.lower()
                == "auto-resume"
            )
            matching_rows = scenario_scores_df.loc[mask, "Scenario"].dropna()
            if not matching_rows.empty:
                return str(matching_rows.iloc[0]).strip()

    logger.warning(
        "Could not resolve ACC Auto-resume type of system from updated data; leaving it blank in the exported ACC sheet."
    )
    return ""


def get_output_file_path(output_path: str) -> str:
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(output_path, f"sd_{current_datetime}_report.xlsx")
    return output_file


def set_dynamic_header(ws, header_fill, header_font, col_name="Scenario"):
    # Find the second row where the first column has value col_name
    scenario_row_indices = []
    for idx, row in enumerate(
        ws.iter_rows(min_row=1, max_row=ws.max_row, min_col=1, max_col=1),
        start=1,
    ):
        cell = row[0]
        if cell.value == col_name:
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
    max_width = 50  # Set maximum column width
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

        # Enable text wrapping if content exceeds max width
        if max_length + 2 > max_width:
            for cell in column:
                cell.alignment = Alignment(
                    wrap_text=True,
                    horizontal=(
                        cell.alignment.horizontal if cell.alignment else "center"
                    ),
                    vertical=cell.alignment.vertical if cell.alignment else "center",
                )


def get_populated_sheet_bounds(ws):
    max_row = 1
    max_col = 1

    for cell in ws._cells.values():
        if common.is_empty_cell(cell.value):
            continue
        max_row = max(max_row, cell.row)
        max_col = max(max_col, cell.column)

    for merged_range in ws.merged_cells.ranges:
        top_left_cell = ws.cell(merged_range.min_row, merged_range.min_col)
        if common.is_empty_cell(top_left_cell.value):
            continue
        max_row = max(max_row, merged_range.max_row)
        max_col = max(max_col, merged_range.max_col)

    return max_row, max_col


def write_report(
    command: common.CliCommand,
    input_file: str,
    updated_dfs: dict,
    output_path: str,
    format_prediction_cells: bool = True,
    dm_prediction_forced_red: pd.DataFrame = None,
):
    """Writes the report to an Excel file, applying formatting and copying necessary sheets.
    Args:
        command: The CLI command indicating the context of the report (e.g., PREPROCESS, COMPUTE_SCORE).
        input_file: Path to the input Excel file containing the template sheets to copy.
        updated_dfs: Dictionary of updated DataFrames to be written to the report, keyed by sheet name.
        output_path: Directory where the output Excel file will be saved.
        format_prediction_cells: Boolean flag to determine whether to apply color formatting to prediction cells.
        dm_prediction_forced_red: DataFrame with "row"/"col" (1-based, Excel-coordinate)
            columns marking "DE - DM pred." cells that must render as RED -- either
            explicitly, left blank in a required slot, or forced RED by
            ``driver_monitoring.apply_prediction_consistency``.
    """

    if command == common.CliCommand.PREPROCESS:
        output_file = os.path.join(output_path, "sd_preprocessed_template.xlsx")
        sheets_to_copy = PREPROCESS_SHEETS_TO_COPY
    elif command == common.CliCommand.COMPUTE_SCORE:
        output_file = get_output_file_path(output_path)
        sheets_to_copy = COMPUTE_SHEETS_TO_COPY
    else:
        logger.error(f"Unsupported command: {command}")
        return

    print(f"Saving to {output_file}...")

    prepared_dfs = dict(updated_dfs)
    acc_verification_df = prepared_dfs.get("VA - ACC verif.", pd.DataFrame())
    if (
        isinstance(acc_verification_df, pd.DataFrame)
        and not acc_verification_df.empty
        and "Test point" in acc_verification_df.columns
        and "Scenario" in acc_verification_df.columns
    ):
        acc_type_of_system = _get_acc_type_of_system(prepared_dfs)
        prepared_dfs["VA - ACC verif."] = acc_performance.build_verification_report_df(
            acc_verification_df,
            acc_type_of_system,
        )

    prediction_mark_coordinates = common.build_safe_driving_prediction_mark_coordinates(
        prepared_dfs
    )

    pred_sheet_names = {
        "DE - DM pred.",
        "VA - Speed assist. pred.",
        "VA - ACC pred.",
    }
    input_wb = common.load_or_create_workbook(input_file)
    output_wb = common.load_or_create_workbook(output_file)
    hard_copied_sheets = set()
    for i, sheet in enumerate(sheets_to_copy):
        if sheet in input_wb.sheetnames:
            common.hard_copy_sheet(input_wb, sheet, output_wb)
            hard_copied_sheets.add(sheet)
        else:
            logger.debug(
                f"Sheet '{sheet}' not in input workbook, will be written from updated_dfs if present."
            )
    output_wb.save(output_file)
    # Write the updated copied sheets back to the output file
    try:
        with pd.ExcelWriter(
            output_file, engine="openpyxl", mode="a", if_sheet_exists="replace"
        ) as writer:
            for sheet_name, updated_df in prepared_dfs.items():
                # Skip pred sheets that were already hard-copied with real data;
                # write pred sheets that were NOT hard-copied so format_prediction_cells can apply colors.
                if sheet_name in pred_sheet_names and sheet_name in hard_copied_sheets:
                    continue
                updated_df.to_excel(writer, sheet_name=sheet_name, index=False)
    except Exception as e:
        logger.error(f"Failed to write updated sheets: {e}")

    calculation_color = PatternFill(
        start_color="FFDD04", end_color="FFDD04", fill_type="solid"
    )
    input_color = PatternFill(
        start_color=common.INPUT_CELL_RGB,
        end_color=common.INPUT_CELL_RGB,
        fill_type="solid",
    )
    output_wb = load_workbook(output_file)
    sheetnames = output_wb.sheetnames

    for sheet_name in sheetnames:
        ws = output_wb[sheet_name]
        max_row, max_col = get_populated_sheet_bounds(ws)
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

        if "pred." not in sheet_name:
            for col in ws.iter_cols(
                min_row=1, max_row=max_row, min_col=1, max_col=max_col
            ):
                header = col[0].value
                align = Alignment(horizontal="left")
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

        # Set number format for "Number of rear seats" in "Input parameters" sheet
        if sheet_name == "Input parameters":
            headers = [cell.value for cell in ws[1]]
            if "Input parameter" in headers and "Value" in headers:
                input_param_col_idx = headers.index("Input parameter") + 1
                value_col_idx = headers.index("Value") + 1
                for row in range(2, ws.max_row + 1):
                    input_param_cell = ws.cell(row=row, column=input_param_col_idx)
                    if input_param_cell.value == "Number of rear seats":
                        value_cell = ws.cell(row=row, column=value_col_idx)
                        value_cell.number_format = "0"

        ###############################################################
        # Color columns based on their headers
        ###############################################################
        na_value_rows = _get_na_value_rows(ws)
        for col in ws.iter_cols(min_row=1, max_row=max_row, min_col=1, max_col=max_col):
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
                        and col[0].value == "OEM Prediction"
                    ):
                        logger.debug(
                            f"Skipping OEM Prediction color fill for sheet {sheet_name} and column {col[0].value}"
                        )
                    elif col[0].value == "Value" and cell.row in na_value_rows:
                        # N/A row: no input required, keep the Value cell white.
                        pass
                    else:
                        cell.fill = input_color
                    if common.is_empty_cell(cell.value):
                        cell.value = ""

    ###############################################################
    # Adjust column widths based on content
    ###############################################################
    for sheet_name in output_wb.sheetnames:
        ws = output_wb[sheet_name]
        max_row, max_col = get_populated_sheet_bounds(ws)

        if sheet_name == "VA - Speed assist. pred.":
            for i, column in enumerate(
                ws.iter_cols(min_row=1, max_row=max_row, min_col=1, max_col=max_col)
            ):
                col_letter = get_column_letter(i + 1)
                ws.column_dimensions[col_letter].width = 5.5 * CM_TO_EXCEL

        if sheet_name == "DE - DM pred.":
            for i, column in enumerate(
                ws.iter_cols(min_row=1, max_row=max_row, min_col=1, max_col=max_col)
            ):
                if i < 4:
                    adjust_column_widths([column])
                else:
                    col_letter = get_column_letter(i + 1)
                    ws.column_dimensions[col_letter].width = 4.0 * CM_TO_EXCEL
        else:
            adjust_column_widths(
                ws.iter_cols(min_row=1, max_row=max_row, min_col=1, max_col=max_col)
            )

    for sheet_name in output_wb.sheetnames:
        ws = output_wb[sheet_name]
        logger.debug(f"Adding borders to sheet: {sheet_name}")
        max_row, max_col = get_populated_sheet_bounds(ws)
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
            stage_element_initials = "".join(
                [word[0].upper() for word in stage_element.split()]
            )
            stage_subelement_initials = "".join(
                [word[0].upper() for word in stage_subelement.split()]
            )
            if stage_subelement == "Speed Assistance":
                stage_subelement_initials = "Speed assist."
            if stage_subelement == "ACC Performance":
                stage_subelement_initials = "ACC"
            if stage_subelement in [
                "Steering Assistance",
                "General vehicle controls",
                "Seatbelt usage",
                "Occupant classification",
                "Occupant presence",
            ]:
                continue  # Skip formatting for these subelements as they do not have a prediction sheet
            prediction_sheet_name = (
                f"{stage_element_initials} - {stage_subelement_initials} pred."
            )

            if prediction_sheet_name not in output_wb.sheetnames:
                logger.warning(
                    f"Prediction sheet '{prediction_sheet_name}' is missing; "
                    "skipping prediction-cell formatting for it."
                )
                continue

            pred_ws = output_wb[prediction_sheet_name]

            if (
                prediction_sheet_name == "DE - DM pred."
                and isinstance(dm_prediction_forced_red, pd.DataFrame)
                and not dm_prediction_forced_red.empty
            ):
                for row, col in dm_prediction_forced_red[["row", "col"]].itertuples(
                    index=False
                ):
                    pred_ws.cell(row=row, column=col, value="Red")

            pred_max_row, pred_max_col = get_populated_sheet_bounds(pred_ws)
            scenario = stage_info["Stage subelement"]
            for row_index, row in enumerate(
                pred_ws.iter_rows(
                    min_row=1,
                    max_row=pred_max_row,
                    min_col=1,
                    max_col=pred_max_col,
                )
            ):
                for col_index, cell in enumerate(row):
                    key = str(cell.value).lower()
                    logger.debug(
                        f"Processing cell at row {row_index + 1}, col {col_index + 1} key: {key}"
                    )
                    if key in common.PREDICTION_COLOR_MAP:
                        color = common.PREDICTION_COLOR_MAP[key]
                    else:
                        continue
                    logger.debug(
                        f"Setting cell at row {row_index + 1}, col {col_index + 1} to color: {color}"
                    )
                    # Convert float RGB (0-1) to int (0-255) and then to hex
                    color_int = tuple(int(round(c * 255)) for c in color)
                    color_hex = "{:02X}{:02X}{:02X}".format(*color_int)
                    fill = PatternFill(
                        start_color=color_hex,
                        end_color=color_hex,
                        fill_type="solid",
                    )
                    cell.fill = fill

                    if stage_subelement == "Speed Assistance":
                        cell.font = Font(color=color_hex)
                    else:
                        cell.value = ""
                    # Iterate over selected_points and mark them using data_model with offsets

            sheet_coordinates = prediction_mark_coordinates.get(
                prediction_sheet_name, []
            )
            if len(sheet_coordinates) == 0:
                logger.debug(
                    f"No selected points found in sheet '{prediction_sheet_name}'"
                )
                continue
            for point_row, point_col in sheet_coordinates:
                logger.debug(
                    f"Marking selected point at row={point_row}, col={point_col} in sheet {prediction_sheet_name}"
                )
                cell_to_mark = pred_ws.cell(
                    row=point_row,
                    column=point_col,
                )
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

    for sheet_name in output_wb.sheetnames:
        ws = output_wb[sheet_name]
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

        # Row 4 is the driver-state sub-header in the normal multi-column DM
        # verif layout. When mandatory noise variables fail, preprocess
        # instead writes a single-column "Category" message table (see
        # driver_monitoring.preprocess) -- skip the row-4 header treatment
        # there, it would just paint one of the message rows as a header.
        if sheet_name == "DE - DM verif." and ws.max_row >= 4 and ws.max_column > 1:
            common.set_header_by_row(ws, 4, header_fill, header_font)

        if sheet_name == "VA - ACC verif.":
            common.set_header_by_row(ws, 9, header_fill, header_font)

    ###############################################################
    # Loop through all sheets, find 'Value' column, and apply data validation to PASS/FAIL cells
    #
    # A blank Value cell means "not yet assessed by the OEM" and must still
    # get the PASS/FAIL dropdown (not just cells that already say PASS/FAIL) --
    # otherwise a freshly-generated sheet's default blank cells would have no
    # dropdown at all. This is scoped to a known allow-list rather than every
    # blank "Value" cell in the workbook: "Input parameters" and other
    # domains' sheets also have a "Value" column with unrelated (non
    # PASS/FAIL) blank cells that must not get this dropdown.
    for ws in output_wb.worksheets:
        headers = [cell.value for cell in ws[1]]
        if "Value" in headers:
            max_row, _ = get_populated_sheet_bounds(ws)
            value_col_idx = headers.index("Value") + 1  # openpyxl is 1-based
            category_col_idx = (
                headers.index("Category") + 1 if "Category" in headers else None
            )
            scenario_col_idx = (
                headers.index("Scenario") + 1 if "Scenario" in headers else None
            )
            validation_cells = []
            # Category is only written on the first row of a group (e.g. "Road
            # features" on Curves, blank on Roundabouts/Intersection/... right
            # below it) -- forward-fill so continuation rows are still
            # recognised as belonging to that category.
            current_acc_category = None
            na_value_rows = _get_na_value_rows(ws)
            for row in range(2, max_row + 1):
                cell = ws.cell(row=row, column=value_col_idx)
                # N/A rows need no user input at all -- no dropdown, even if
                # an old file carries a stray PASS/FAIL in the cell.
                if row in na_value_rows:
                    continue
                if isinstance(cell.value, str) and cell.value.strip().upper() in [
                    "PASS",
                    "FAIL",
                ]:
                    validation_cells.append(cell)
                    continue

                # Blank separator rows between sections (e.g. between "DE - DM
                # verif."'s General requirements row and its test-point matrix
                # header) must not get a dropdown -- they aren't a real row.
                row_is_blank = all(
                    common.is_empty_cell(ws.cell(row=row, column=col).value)
                    for col in range(1, ws.max_column + 1)
                )
                if row_is_blank:
                    continue

                # "Speedometer accuracy" gets its own dedicated dropdown
                # (below) with a different allowed-values list -- it must
                # not also pick up the generic PASS/FAIL one here.
                if (
                    ws.title == "VA - Speed assist. verif."
                    and scenario_col_idx is not None
                    and str(
                        ws.cell(row=row, column=scenario_col_idx).value or ""
                    ).strip()
                    == "Speedometer accuracy"
                ):
                    continue

                if ws.title == "VA - ACC verif." and category_col_idx is not None:
                    raw_category = str(
                        ws.cell(row=row, column=category_col_idx).value or ""
                    ).strip()
                    if raw_category:
                        current_acc_category = raw_category
                    is_eligible = (
                        current_acc_category in PASS_FAIL_BLANK_ELIGIBLE_ACC_CATEGORIES
                    )
                else:
                    is_eligible = ws.title in PASS_FAIL_BLANK_ELIGIBLE_SHEETS

                if is_eligible and common.is_empty_cell(cell.value):
                    validation_cells.append(cell)

            if validation_cells:
                dv = DataValidation(
                    type="list", formula1='"PASS,FAIL"', allow_blank=True
                )
                ws.add_data_validation(dv)
                for cell in validation_cells:
                    dv.add(cell)

    for ws in output_wb.worksheets:
        headers = [cell.value for cell in ws[1]]
        # Special validation for VA - Speed assist. verif. sheet, Speedometer accuracy row
        if (
            ws.title == "VA - Speed assist. verif."
            and "Scenario" in headers
            and "Value" in headers
        ):
            max_row, _ = get_populated_sheet_bounds(ws)
            scenario_col_idx = headers.index("Scenario") + 1
            value_col_idx = headers.index("Value") + 1
            for row in range(2, max_row + 1):
                scenario_cell = ws.cell(row=row, column=scenario_col_idx)
                if scenario_cell.value == "Speedometer accuracy":
                    value_cell = ws.cell(row=row, column=value_col_idx)
                    sas_dv = DataValidation(
                        type="list",
                        formula1='"≥-3 km/h,≥-5 km/h,<-5 km/h"',
                        allow_blank=False,
                    )
                    ws.add_data_validation(sas_dv)
                    sas_dv.add(value_cell)

    for ws in output_wb.worksheets:
        if ws.title == "VA - Steering assistance verif.":
            headers = [cell.value for cell in ws[1]]
            if "Scenario" in headers and "Value" in headers:
                max_row, _ = get_populated_sheet_bounds(ws)
                scenario_col_idx = headers.index("Scenario") + 1
                value_col_idx = headers.index("Value") + 1
                first_turn_dv = DataValidation(
                    type="list",
                    formula1='"Stays in lane,Does not stay in lane"',
                    allow_blank=True,
                )
                second_turn_dv = DataValidation(
                    type="list",
                    formula1='"Stays in lane,Redirects,Does not redirect"',
                    allow_blank=True,
                )
                ws.add_data_validation(first_turn_dv)
                ws.add_data_validation(second_turn_dv)
                for row in range(2, max_row + 1):
                    scenario_val = str(
                        ws.cell(row=row, column=scenario_col_idx).value or ""
                    )
                    value_cell = ws.cell(row=row, column=value_col_idx)
                    if "First turn" in scenario_val:
                        first_turn_dv.add(value_cell)
                    elif "Second turn" in scenario_val:
                        second_turn_dv.add(value_cell)

    for ws in output_wb.worksheets:
        common.reset_empty_row_background(ws)

    # Reorder sheets according to the specified order
    def reorder_sheets(workbook):
        # Define the main order
        main_order = [
            "Input parameters",
            "Test Scores",
            "Category Scores",
            "Scenario Scores",
            "OM - Seatbelt usage verif.",
            "OM - Occ. classification verif.",
            "OM - Occ. presence verif.",
            "DE - DM pred.",
            "DE - DM verif.",
            "DE - GVC verif.",
            "VA - Speed assist. pred.",
            "VA - Speed assist. verif.",
            "VA - ACC pred.",
            "VA - ACC verif.",
            "VA - Steering assistance verif.",
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
        from euroncap_rating_2026.safe_driving import integrity as sd_integrity

        common.blank_computed_columns(output_wb, sd_integrity.COMPUTED_SHEET_COLUMNS)
    reorder_sheets(output_wb)
    common.hide_documentation_data_sheets(output_wb)
    common.hide_integrity_sheet(output_wb)
    # Internal DM prediction required-cell mask carried through the dfs
    # merge -- not meant to be seen, like Documentation/Data above.
    if "DE - DM pred. (required)" in output_wb.sheetnames:
        output_wb["DE - DM pred. (required)"].sheet_state = "hidden"
    common.add_version_sheet_to_wb(output_wb, command)
    if command == common.CliCommand.PREPROCESS:
        # Sign the protected (non-grey) reference cells so compute-score can
        # detect if this preprocessed file gets edited outside the tool
        # before scoring.
        from euroncap_rating_2026.safe_driving import integrity as sd_integrity

        common.add_integrity_sheet_to_wb(output_wb, sd_integrity.SHEET_SCHEMAS)
    output_wb.save(output_file)

    if command == common.CliCommand.COMPUTE_SCORE:
        print(
            f"Log available at {os.path.join(output_path, 'euroncap_rating_2026.log')}"
        )
        print(" " * 40)
        print(f"Final report available at {output_file}")
        logger.info(f"Final report available at {output_file}")
