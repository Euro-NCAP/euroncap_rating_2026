# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from functools import wraps
from typing import Tuple
from euroncap_rating_2026.version import VERSION
from importlib.resources import files
import logging
import pandas as pd
import random
import numpy as np
from collections import Counter
import openpyxl
from openpyxl.utils import get_column_letter
from openpyxl.utils.cell import coordinate_to_tuple
from openpyxl.cell.cell import MergedCell
import copy
import sys
import hashlib
import hmac
from zipfile import BadZipFile
from openpyxl.styles import PatternFill, Font
from dataclasses import dataclass, field
from enum import Enum
import re
from openpyxl.styles import Border, Side

logger = logging.getLogger(__name__)

CM_TO_EXCEL_WIDTH = 0.2116

# "N/A" is a protocol-valid input value on several sheets (e.g. the
# safe_driving and crash_avoidance "Input parameters" dropdowns) meaning the
# system/feature is not fitted / not applicable: the affected element scores
# 0 and no further input is required from the user.
NOT_APPLICABLE = "N/A"


def is_not_applicable(value) -> bool:
    """True when *value* is the "N/A" input selection."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    if isinstance(value, Enum):
        value = value.value
    return str(value).strip().lower() == "n/a"


class PredictionColor(str, Enum):
    BLUE = "blue"
    BROWN = "brown"
    GREEN = "green"
    GREY = "grey"
    ORANGE = "orange"
    RED = "red"
    YELLOW = "yellow"


PREDICTION_COLOR_MAP = {
    PredictionColor.BLUE: (0 / 255, 102 / 255, 204 / 255),
    PredictionColor.BROWN: (150 / 255, 75 / 255, 0 / 255),
    PredictionColor.GREEN: (0 / 255, 153 / 255, 51 / 255),
    PredictionColor.GREY: (128 / 255, 128 / 255, 128 / 255),
    PredictionColor.ORANGE: (255 / 255, 153 / 255, 51 / 255),
    PredictionColor.RED: (255 / 255, 51 / 255, 51 / 255),
    PredictionColor.YELLOW: (255 / 255, 221 / 255, 51 / 255),
}


class CliCommand(Enum):
    """Enum representing the different CLI commands available in the package."""

    UNKNOWN = "unknown"
    GENERATE_TEMPLATE = "generate-template"
    PREPROCESS = "preprocess"
    COMPUTE_SCORE = "compute-score"


def with_footer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        # Print footer after all command output
        footer_lines = [
            " " * 40,
            f"Generated with version {VERSION}",
            " " * 40,
            "Copyright 2025-2026, Euro NCAP IVZW\nCreated by IVEX NV (https://ivex.ai)",
        ]
        for line in footer_lines:
            logger.info(line)
            print(line)
        return result

    return wrapper


def _rgb_tuple_from_hex(hex_str):
    if hex_str is None or len(hex_str) != 8:
        return None
    r = int(hex_str[2:4], 16) / 255
    g = int(hex_str[4:6], 16) / 255
    b = int(hex_str[6:8], 16) / 255
    return (r, g, b)


def _get_prediction_color_from_bgcolor(bg_color_hex):
    rgb = _rgb_tuple_from_hex(bg_color_hex)
    if rgb is None:
        return None
    for color, rgb_val in PREDICTION_COLOR_MAP.items():
        if all(abs(a - b) < 0.02 for a, b in zip(rgb, rgb_val)):
            return color
    return None


def normalize_prediction_bg_value(value):
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if stripped.startswith("PredictionColor."):
        color_name = stripped.split(".", 1)[1].lower()
        return PredictionColor(color_name)

    lowered = stripped.lower()
    if lowered in PREDICTION_COLOR_MAP:
        return PredictionColor(lowered)

    return None


def prediction_df_to_bg_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda column: column.map(normalize_prediction_bg_value))


def merge_prediction_color_sources(
    text_df: pd.DataFrame, bg_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Return the per-cell PredictionColor of a prediction sheet, read from
    whichever of its two representations carries it.

    A prediction grid cell holds its colour in one of two shapes (see
    `_cell_has_grid_prediction`): as literal text -- the OEM's typed dropdown
    word ("Green", "Red", ...) in a freshly filled template, or in a
    DataFrame built from such an upload -- or as a background fill only, the
    shape it has after `preprocess`, whose `format_prediction_cells` step
    re-encodes the word as a cell fill and blanks the text (leaving just an
    "X" on the selected test points). `compute-score` runs on the latter, so
    reading the plain-text frame alone silently turns every coloured cell
    into GREY there.

    `text_df` is the plain-text sheet (`dfs["<sheet>"]`), `bg_df` the matching
    fill-derived frame (`dfs["<sheet> (bg)"]`, PredictionColor or None per
    cell, produced by `load_sheet_with_bg_colors` or a caller's equivalent
    builder). The merge is positional -- both frames start at Excel row 2 /
    column A -- and per cell: a recognised colour word in the text wins,
    otherwise the fill colour, otherwise None. The bg frame is cropped or
    padded to `text_df`'s shape: `pd.read_excel` drops trailing empty
    rows/columns that openpyxl's `max_row`/`max_column` still count when the
    cells are styled, so the two may legitimately differ in size. A missing
    or empty `bg_df` (the normal shape on the DataFrame path today) simply
    yields the text colours. The result keeps `text_df`'s columns and is
    object-dtyped, so it plugs straight into
    `matrix_processing.get_test_matrix_from_region` (pass `text_df` as its
    `attributes_df`, since the merged frame carries no row/column labels).

    Note that, like every fill-based reader in this package, a colour fill
    outside the designated grey input cells counts as a prediction too.
    """
    text_values = (
        text_df.astype(object)
        .apply(lambda column: column.map(normalize_prediction_bg_value))
        .to_numpy(dtype=object)
    )
    n_rows, n_cols = text_values.shape
    bg_values = np.full((n_rows, n_cols), None, dtype=object)
    if bg_df is None or bg_df.empty:
        logger.debug(
            "No background-colour prediction frame given; reading prediction "
            "colours from the text sheet only."
        )
    else:
        raw_bg = bg_df.to_numpy(dtype=object)
        overlap_rows = min(n_rows, raw_bg.shape[0])
        overlap_cols = min(n_cols, raw_bg.shape[1])
        bg_values[:overlap_rows, :overlap_cols] = raw_bg[:overlap_rows, :overlap_cols]
        if raw_bg.shape != (n_rows, n_cols):
            logger.debug(
                "Background-colour prediction frame shape %s aligned to the "
                "text sheet shape %s.",
                raw_bg.shape,
                (n_rows, n_cols),
            )

    is_color = np.vectorize(lambda v: isinstance(v, PredictionColor), otypes=[bool])
    merged = np.where(
        is_color(text_values),
        text_values,
        np.where(is_color(bg_values), bg_values, None),
    )
    return pd.DataFrame(merged, columns=text_df.columns, index=text_df.index)


def load_sheet_with_bg_colors(wb, sheet_name: str) -> pd.DataFrame:
    """Load an Excel sheet from an already-opened openpyxl workbook and return a
    DataFrame where each cell value is the PredictionColor derived from its
    background fill (or None if the color is unrecognised / unfilled)."""
    ws = wb[sheet_name]
    columns = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    num_cols = len(columns)
    data_pred_color = []
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        pred_color_row = []
        for cell in row:
            fill = cell.fill
            bg_color = None
            if fill and fill.fgColor and fill.fgColor.type == "rgb":
                bg_color = fill.fgColor.rgb
            pred_color_row.append(_get_prediction_color_from_bgcolor(bg_color))
        data_pred_color.append(pred_color_row)
    return pd.DataFrame(data_pred_color, columns=columns)


def _get_dm_prediction_required_coordinates(ws) -> set:
    """Coordinates of cells governed by the sheet's "N/A,Green,Red" dropdown
    validation -- i.e. cells where the OEM is required to enter a prediction."""
    required_coords = set()
    for dv in ws.data_validations.dataValidation:
        if dv.type != "list" or not dv.formula1:
            continue
        options = {
            o.strip().strip('"').upper() for o in dv.formula1.strip('"').split(",")
        }
        if options != {"N/A", "GREEN", "RED"}:
            continue
        for cell_range in dv.sqref.ranges:
            for row_cells in ws.iter_rows(
                min_row=cell_range.min_row,
                max_row=cell_range.max_row,
                min_col=cell_range.min_col,
                max_col=cell_range.max_col,
            ):
                for cell in row_cells:
                    required_coords.add(cell.coordinate)
    return required_coords


def get_dm_prediction_required_mask(ws) -> pd.DataFrame:
    """DataFrame shaped like pd.read_excel(ws, header=0): for the prediction
    response columns (D/E/F, idx 3-5) each cell is replaced with a bool marking
    whether it falls inside the sheet's "N/A,Green,Red" dropdown range. All other
    columns (and fully-blank separator rows) are passed through unchanged so
    extract_table's marker/NaN-boundary lookups keep working on this DataFrame
    exactly as they do on the plain-text sheet read.
    """
    required_coords = _get_dm_prediction_required_coordinates(ws)

    columns = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    prediction_col_idxs = {3, 4, 5}
    data = []
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        raw_values = [cell.value for cell in row]
        if all(v is None for v in raw_values):
            data.append(raw_values)
            continue
        row_values = list(raw_values)
        for col_idx in prediction_col_idxs:
            if col_idx < len(row):
                row_values[col_idx] = row[col_idx].coordinate in required_coords
        data.append(row_values)
    return pd.DataFrame(data, columns=columns)


def get_default_dm_prediction_required_mask() -> pd.DataFrame:
    """Required-mask for "DE - DM pred.", derived from the packaged
    sd_template.xlsx instead of an uploaded workbook. The "N/A,Green,Red"
    dropdown ranges are authored by Euro NCAP in the template and don't vary
    per OEM submission, so dataframe-only callers without an openpyxl
    worksheet can still get required-but-blank enforcement.
    """
    template_path = str(files("data").joinpath("sd_template.xlsx"))
    wb = openpyxl.load_workbook(template_path, data_only=True)
    try:
        return get_dm_prediction_required_mask(wb["DE - DM pred."])
    finally:
        wb.close()


def sheet_to_dataframe(ws) -> pd.DataFrame:
    """An openpyxl worksheet as the DataFrame pd.read_excel(header=0) would
    produce: first row as column headers, remaining rows as data. Lets code
    written against read_excel_file_to_dfs' shape run on an already-opened
    workbook (e.g. the consistency checks, whose callers hold a Workbook,
    not a path).

    Bounded by get_populated_sheet_bounds rather than ws.values' own
    ws.max_row/max_column: a handful of sheets in the packaged templates
    (e.g. "CP - VRU Prediction") carry a stale, far-inflated dimension from
    some past formatting action -- a single real cell out past column
    16384 -- which ws.values honours literally, building a row of that
    width for every row in the sheet. get_populated_sheet_bounds ignores
    cells with an empty value, so it isn't fooled by that, and costs only
    as many iterations as the sheet's real (sparse) cell count."""
    max_row, max_col = get_populated_sheet_bounds(ws)
    if not ws._cells:
        return pd.DataFrame()
    rows = [
        tuple(ws.cell(row=r, column=c).value for c in range(1, max_col + 1))
        for r in range(1, max_row + 1)
    ]
    return pd.DataFrame(rows[1:], columns=list(rows[0]))


def _restore_na_text_values(df, ws, column_names):
    """Restore literal "N/A" strings that pd.read_excel silently turned into
    NaN (pandas' default na_values list includes "N/A"/"n/a"), only in the
    columns named in *column_names*.

    Worksheet row r (r >= 2, row 1 is the header) maps to df row r-2 -- the
    same header=0 alignment pd.read_excel produced the DataFrame with.
    Iteration is bounded to the DataFrame's rows and the named columns, so
    the cost never scales with the worksheet's (possibly formatting-inflated)
    used range.
    """
    if df.empty:
        return
    header = {cell.value: cell.column for cell in ws[1]}
    for column_name in column_names:
        column_idx = header.get(column_name)
        if column_idx is None:
            continue
        for (cell,) in ws.iter_rows(
            min_row=2, max_row=len(df) + 1, min_col=column_idx, max_col=column_idx
        ):
            if (
                isinstance(cell.value, str)
                and cell.value.strip().lower() == "n/a"
                and cell.column - 1 < len(df.columns)
            ):
                df.iat[cell.row - 2, cell.column - 1] = cell.value.strip()


def read_excel_file_to_dfs(
    input_file, load_bg_colors: bool = False, preserve_na_sheets=None
):
    # engine_kwargs={"read_only": False} overrides pandas' default openpyxl
    # engine_kwargs (read_only=True): its streaming reader trusts a sheet's
    # raw XML <dimension> tag literally, and several packaged templates have
    # that tag inflated far past their real used range (a template-authoring
    # artifact, e.g. ca_template.xlsx's prediction sheets) -- read_only mode
    # then iterates millions of phantom empty cells per sheet where the
    # normal loader iterates thousands. Output is identical either way
    # (same pandas parsing code, just fed a differently-loaded openpyxl
    # Workbook); only the load time changes, by 1-2 orders of magnitude.
    xls = pd.ExcelFile(input_file, engine_kwargs={"read_only": False})

    sheet_names = xls.sheet_names
    if not sheet_names:
        raise ValueError("The Excel file does not contain any sheets.")
    dfs = {}

    for sheet_name in sheet_names:
        dfs[sheet_name] = pd.read_excel(xls, sheet_name=sheet_name, header=0)

    xls.close()

    if load_bg_colors or preserve_na_sheets:
        wb = openpyxl.load_workbook(input_file, data_only=True)
        # "N/A" is a protocol-valid input on some sheets (e.g. safe_driving's
        # "Input parameters" dropdowns), but pd.read_excel's default na_values
        # collapse it into NaN, erasing the distinction from a blank cell --
        # restore the literal string for the sheets/columns that opt in.
        # preserve_na_sheets maps sheet name -> tuple of column names, so a
        # stray "N/A" typed into any other column keeps the historical
        # NaN/blank behavior.
        for sheet_name, column_names in (preserve_na_sheets or {}).items():
            if sheet_name in sheet_names:
                _restore_na_text_values(dfs[sheet_name], wb[sheet_name], column_names)
        if load_bg_colors:
            for sheet_name in sheet_names:
                if sheet_name.endswith("pred."):
                    dfs[f"{sheet_name} (bg)"] = load_sheet_with_bg_colors(
                        wb, sheet_name
                    )
            if "DE - DM pred." in sheet_names:
                ws = wb["DE - DM pred."]
                dfs["DE - DM pred. (cell_colors)"] = get_cell_background_colors(ws)
                dfs["DE - DM pred. (required)"] = get_dm_prediction_required_mask(ws)
        wb.close()

    return dfs


def frequency_proportional_sample(data, N, seed=None):
    """
    Sample N (row, col, color) tuples from data, using color frequency for proportional sampling.
    Keeps (row, col) in the result.
    """
    if seed is not None:
        random.seed(seed)

    if N >= len(data):
        return data.copy()
    # Count color frequencies
    color_counts = Counter([color for _, _, color in data])
    total = sum(color_counts.values())

    # Compute ideal proportions
    proportions = {color: count / total for color, count in color_counts.items()}
    logger.info(f"proportions: {proportions}")

    # Initial rounded down count
    raw_counts = {}
    for color, p in proportions.items():
        count = int(np.floor(p * N))
        raw_counts[color] = count
    logger.debug(f"raw_counts pre remaining distribution: {raw_counts}")
    # Distribute remaining samples
    remaining = N - sum(raw_counts.values())
    residuals = {
        color: (proportions[color] * N - raw_counts[color]) for color in color_counts
    }
    for color in sorted(residuals, key=residuals.get, reverse=True)[:remaining]:
        raw_counts[color] += 1
    logger.debug(f"raw_counts post remaining distribution: {raw_counts}")

    # Sample actual items, keeping (row, col, color)
    sampled = []
    for color, count in raw_counts.items():
        choices = [x for x in data if x[2] == color]
        sampled += random.sample(choices, min(len(choices), count))

    return sampled


def get_initials(text):
    if pd.isna(text):
        return ""
    return "".join([word[0].upper() for word in str(text).split()])


def get_first_word(text):
    if pd.isna(text):
        return ""
    return str(text).split()[0]


def get_param_df_from_input_parameters(input_parameters_df):
    """Extract parameters from input_parameters_df."""
    # Extract input parameters with their corresponding stage, stage element, and stage subelement
    input_parameters = []
    current_stage = None
    current_stage_element = None
    current_stage_subelement = None

    for _, row in input_parameters_df.iterrows():
        if not pd.isna(row.get("Stage")):
            current_stage = row["Stage"]
        if not pd.isna(row.get("Stage element")):
            current_stage_element = row["Stage element"]
        if not pd.isna(row.get("Stage subelement")):
            current_stage_subelement = row["Stage subelement"]
        input_param = row.get("Input parameter")
        input_value = row.get("Value")
        if not pd.isna(input_param) and not pd.isna(input_value):
            input_parameters.append(
                {
                    "Input parameter": input_param,
                    "Stage": current_stage,
                    "Stage element": current_stage_element,
                    "Stage subelement": current_stage_subelement,
                    "Value": input_value,
                }
            )
    # Create a DataFrame of input parameters
    param_df = pd.DataFrame(
        input_parameters,
        columns=[
            "Input parameter",
            "Stage",
            "Stage element",
            "Stage subelement",
            "Value",
        ],
    )

    param_df["param_code"] = (
        param_df["Stage"].apply(get_initials)
        + " - "
        + param_df["Stage element"].apply(get_first_word)
        + " "
        + param_df["Stage subelement"].astype(str)
    )
    param_df = param_df[["param_code", "Input parameter", "Value"]]
    logger.debug(f"Extracted param_df: {param_df}")
    return param_df


def get_param_df(dfs):
    """Extract parameters from dfs dictionary."""
    input_parameters_df = dfs.get("Input parameters")
    if input_parameters_df is None:
        raise ValueError(
            "The Excel file does not contain the required 'Input parameters' sheet."
        )
    return get_param_df_from_input_parameters(input_parameters_df)


def load_or_create_workbook(file_path: str):
    """Load an existing workbook or create a new one when the file is empty."""
    try:
        wb = openpyxl.load_workbook(file_path)
    except (FileNotFoundError, BadZipFile, OSError):
        wb = openpyxl.Workbook()
        if wb.active.max_row == 1 and wb.active.max_column == 1:
            active_cell = wb.active["A1"]
            if active_cell.value is None:
                wb.remove(wb.active)
    return wb


def hard_copy_sheet(input_wb, sheet_name: str, output_wb) -> bool:
    """Copy *sheet_name* from *input_wb* into *output_wb* verbatim.

    Returns True when the sheet was copied. A sheet missing from *input_wb*
    is skipped with a warning and returns False instead of raising: hard-
    copied sheets are carry-through only (nothing is computed from them at
    report-writing time), and workbooks uploaded against an older template
    revision may legitimately lack sheets a newer template has (e.g.
    "Documentation"/"Data" in older template revisions).
    """
    if sheet_name not in input_wb.sheetnames:
        logger.warning(
            f"Sheet '{sheet_name}' not found in input workbook, skipping copy"
        )
        return False

    in_ws = input_wb[sheet_name]
    # Remove sheet if it already exists in output
    if sheet_name in output_wb.sheetnames:
        std = output_wb[sheet_name]
        output_wb.remove(std)
    out_ws = output_wb.create_sheet(sheet_name)
    max_row, max_col = get_populated_sheet_bounds(in_ws)

    # Copy row heights
    for row in range(1, max_row + 1):
        out_ws.row_dimensions[row].height = in_ws.row_dimensions[row].height

    # Copy cell values, styles, and fills
    for row in in_ws.iter_rows(min_row=1, max_row=max_row, min_col=1, max_col=max_col):
        for cell in row:
            new_cell = out_ws.cell(row=cell.row, column=cell.column, value=cell.value)
            if cell.has_style:
                new_cell.font = copy.copy(cell.font)
                new_cell.border = copy.copy(cell.border)
                new_cell.fill = copy.copy(cell.fill)
                new_cell.number_format = copy.copy(cell.number_format)
                new_cell.protection = copy.copy(cell.protection)
                new_cell.alignment = copy.copy(cell.alignment)

    # Copy merged cells
    for merged_range in in_ws.merged_cells.ranges:
        out_ws.merge_cells(str(merged_range))

    # Copy column widths
    # Set column widths for columns 0 to 25 (i.e., columns A to Z)
    for col_idx in range(0, min(max_col, 26)):
        col_letter = get_column_letter(col_idx + 1)  # openpyxl is 1-based
        if (
            col_letter in in_ws.column_dimensions
            and in_ws.column_dimensions[col_letter].width is not None
        ):
            out_ws.column_dimensions[col_letter].width = in_ws.column_dimensions[
                col_letter
            ].width

    return True


def delete_rows_keeping_merges(ws, rows):
    """
    Delete the given 1-based worksheet rows from ws, then recompute and
    reapply merged-cell ranges shifted accordingly.

    openpyxl's Worksheet.delete_rows moves cell values and styles up but
    leaves merged ranges pinned to their original coordinates -- detaching
    every merge below the deletion point (e.g. a multi-column header). This
    releases all merges first, deletes the rows, then rebuilds each
    surviving merge at its shifted coordinates.

    Args:
        ws: an openpyxl Worksheet.
        rows: iterable of 1-based row numbers to delete.
    """
    rows_to_delete = sorted(set(rows))
    if not rows_to_delete:
        return

    merged_ranges = [
        (r.min_row, r.min_col, r.max_row, r.max_col)
        for r in list(ws.merged_cells.ranges)
    ]
    for min_row, min_col, max_row, max_col in merged_ranges:
        ws.unmerge_cells(
            start_row=min_row,
            start_column=min_col,
            end_row=max_row,
            end_column=max_col,
        )

    # Delete highest row number first: deleting a row only shifts rows
    # *below* it, so a pending lower-numbered deletion further up stays
    # valid without needing to be renumbered as we go.
    for row in reversed(rows_to_delete):
        ws.delete_rows(row, 1)

    def shifted(row_num):
        return row_num - sum(1 for deleted in rows_to_delete if deleted < row_num)

    deleted_set = set(rows_to_delete)
    for min_row, min_col, max_row, max_col in merged_ranges:
        if all(row in deleted_set for row in range(min_row, max_row + 1)):
            continue  # entirely inside the deleted rows, nothing to rebuild
        new_min_row = shifted(min_row)
        new_max_row = shifted(max_row)
        if new_max_row < new_min_row:
            continue
        ws.merge_cells(
            start_row=new_min_row,
            start_column=min_col,
            end_row=new_max_row,
            end_column=max_col,
        )


# The "Documentation"/"Data" sheets carry Euro-NCAP-facing "is supporting
# evidence required" flags. They're hidden in the pristine template, but
# hard_copy_sheet (used to carry them through preprocess/compute-score,
# see each domain's report_writer.py) copies cell content only -- not
# sheet_state -- so a hard-copied sheet always comes back visible. Call this
# once, right before saving, in every domain's write_report.
DOCUMENTATION_DATA_SHEETS = ["Documentation", "Data"]


def hide_documentation_data_sheets(wb) -> None:
    """Re-hide the "Documentation"/"Data" sheets after hard_copy_sheet has
    (re)created them visible in *wb*."""
    for name in DOCUMENTATION_DATA_SHEETS:
        if name in wb.sheetnames:
            wb[name].sheet_state = "hidden"


def blank_computed_columns(wb, sheet_columns: dict) -> None:
    """Set every column named in *sheet_columns* to "-" for its sheet.

    *sheet_columns* maps a sheet name to the list of its columns that are
    only ever produced by compute-score, never by the OEM or preprocess
    (e.g. a domain's GreyCellSheetSchema.computed_columns). Sheets or
    columns absent from *wb* are skipped rather than erroring, so the same
    mapping works across the pristine template, preprocessed files, and
    reports.
    """
    for sheet_name, columns in sheet_columns.items():
        if sheet_name not in wb.sheetnames:
            continue
        ws = wb[sheet_name]
        header = _sheet_header(ws)
        max_row, _ = get_populated_sheet_bounds(ws)
        for column in columns:
            if column not in header:
                continue
            col_idx = header.index(column) + 1
            for row in range(2, max_row + 1):
                ws.cell(row=row, column=col_idx).value = "-"


def reset_computed_columns_to_dash(dfs: dict, sheet_columns: dict) -> dict:
    """DataFrame-native sibling of blank_computed_columns: set every column
    named in *sheet_columns* to "-" in its sheet.

    *sheet_columns* maps a sheet name to the list of its columns that are
    only ever produced by compute-score, never by the OEM or preprocess
    (each domain's integrity.COMPUTED_SHEET_COLUMNS). Left untouched, a
    freshly preprocessed file would carry the template's placeholder (0 or
    blank) through, reading as "already scored" instead of "not yet
    assessed". Sheets or columns absent from *dfs* are skipped rather than
    erroring. Returns a new dict; the input *dfs* is not mutated.
    """
    result = dict(dfs)
    for sheet_name, columns in sheet_columns.items():
        df = result.get(sheet_name)
        if df is None:
            continue
        df = df.copy()
        for column in columns:
            if column in df.columns:
                df[column] = "-"
        result[sheet_name] = df
    return result


def numericize_computed_columns(dfs: dict, sheet_columns: dict) -> dict:
    """Replace the "-" placeholder in every column named in *sheet_columns*
    with 0, restoring numeric dtype.

    *sheet_columns* maps a sheet name to the list of its columns that are
    genuinely numeric (e.g. Score, crash_protection's "Body regionscore"/
    "Modifiers") -- NOT a domain's full GreyCellSheetSchema.computed_columns
    verbatim, since that can include non-numeric flags like crash_protection's
    "Capping?" (a "YES"/"Capped"/"" string), which this must not touch.

    The packaged template's computed columns hold "-" as a display-only
    placeholder (see blank_computed_columns) -- but callers that treat the
    pristine template as ground truth (e.g. compute-score's
    df["Score"].astype("float64") calls on a pristine-sourced column, via
    rebuild_trusted_df) need it numeric. Call this right after loading the
    pristine template into DataFrames, before it's cached/merged, so the
    dash never reaches those callers.
    """
    result = dict(dfs)
    for sheet_name, columns in sheet_columns.items():
        df = result.get(sheet_name)
        if df is None:
            continue
        df = df.copy()
        for column in columns:
            if column in df.columns:
                # .mask, not .replace: replace triggers a pandas
                # FutureWarning about silent downcasting on object columns.
                df[column] = pd.to_numeric(
                    df[column].mask(df[column] == "-", 0), errors="coerce"
                ).astype("float64")
        result[sheet_name] = df
    return result


def find_version_label_row(ws, label_substring: str):
    """
    Find the row number in column A of *ws* whose value contains
    *label_substring* as a case-insensitive substring.

    Args:
        ws: An openpyxl worksheet.
        label_substring (str): The substring to search for in column A.

    Returns:
        int or None: The 1-indexed row number of the first match, or None if
        no cell in column A contains the substring.
    """
    label_substring_lower = label_substring.lower()
    for row in range(1, ws.max_row + 1):
        cell_value = ws.cell(row=row, column=1).value
        if cell_value and label_substring_lower in str(cell_value).lower():
            return row
    return None


def get_version_search_substring_for_command(cli_command: CliCommand) -> str:
    """Get the label substring to search for in column A for a given CLI command.

    Deprecated in favor of get_version_search_substrings_for_command: the
    template row's label changed from "Generated with version" to "Template
    version", so a single substring can no longer match both generations of
    files. Kept (returning the legacy substring) because external consumers
    import it.
    """
    command_substrings = {
        CliCommand.GENERATE_TEMPLATE: "generated with",
        CliCommand.PREPROCESS: "preprocessed with",
        CliCommand.COMPUTE_SCORE: "score computed with",
        CliCommand.UNKNOWN: "generated with",
    }
    return command_substrings.get(cli_command, "generated with")


def get_version_search_substrings_for_command(cli_command: CliCommand) -> tuple:
    """Label substrings accepted in column A for a given CLI command, in
    match order. The template row accepts both the current "Template version"
    label and the legacy "Generated with version" one written by versions
    <= 5.4.5, so files stamped by either keep being read."""
    command_substrings = {
        CliCommand.GENERATE_TEMPLATE: ("template version", "generated with"),
        CliCommand.PREPROCESS: ("preprocessed with",),
        CliCommand.COMPUTE_SCORE: ("score computed with",),
        CliCommand.UNKNOWN: ("template version", "generated with"),
    }
    return command_substrings.get(cli_command, ("template version", "generated with"))


def get_version_from_workbook(
    wb, cli_command: CliCommand = CliCommand.UNKNOWN
) -> Tuple[str, str]:
    """
    Read the version string from the 'Version' sheet of an already-open
    openpyxl Workbook. Workbook-native sibling of get_version_from_sheet, for
    callers (e.g. find_version_mismatch) that already hold a loaded Workbook
    rather than a file path -- reopening the file from disk would be
    wasteful, and some callers (e.g. the grey-cell domains' find_missing_
    required_inputs) never have a path to begin with.

    The row is located dynamically by searching column A for a
    case-insensitive substring match (e.g. "generated with"), and the
    version string is read from column B of that same row.
    Args:
        wb: An opened openpyxl Workbook.
        cli_command (CliCommand): The CLI command context for version checking.
    Returns:
        Tuple[str, str]: A tuple containing:
            - version (str or None): The version string from column B of the
              matched row, or None if the sheet, label, or value is missing.
            - message (str): A status message indicating success or describing
              the error encountered.
    """
    if "Version" not in wb.sheetnames:
        return None, "Sheet 'Version' not found in workbook"

    ws = wb["Version"]
    label_substrings = get_version_search_substrings_for_command(cli_command)
    row = None
    for label_substring in label_substrings:
        row = find_version_label_row(ws, label_substring)
        if row is not None:
            break

    if row is None:
        quoted = " or ".join(f"'{s}'" for s in label_substrings)
        return (
            None,
            f"Label containing {quoted} not found in column A of " f"'Version' sheet",
        )

    version = ws.cell(row=row, column=2).value

    if version is None:
        return None, f"Version cell for command {cli_command.value} is empty"

    return str(version), "Version sheet read successfully"


def get_version_from_sheet(
    input_file: str, cli_command: CliCommand = CliCommand.UNKNOWN
) -> Tuple[str, str]:
    """
    Read the version string from the 'Version' sheet of an Excel workbook.
    This function opens an Excel file and extracts the version information from a
    dedicated 'Version' sheet. The row is located dynamically by searching column A
    for a case-insensitive substring match (e.g. "generated with"), and the version
    string is read from column B of that same row.
    Args:
        input_file (str): Path to the Excel workbook file to read from.
        cli_command (CliCommand): The CLI command context for version checking.
    Returns:
        Tuple[str, str]: A tuple containing:
            - version (str or None): The version string from column B of the
              matched row, or None if the sheet, label, or value is missing.
            - message (str): A status message indicating success or describing
              the error encountered.
    Examples:
        >>> version, msg = get_version_from_sheet("ratings.xlsx")
        >>> if version:
        ...     print(f"Version: {version}")
        ... else:
        ...     print(f"Error: {msg}")
    """
    wb = openpyxl.load_workbook(input_file)
    try:
        return get_version_from_workbook(wb, cli_command)
    finally:
        wb.close()


@dataclass
class VersionInfo:
    """
    Represents a semantic version with major, minor, and patch components.

    Attributes:
        major (int): Major version number
        minor (int): Minor version number
        patch (int): Patch version number

    Examples:
        >>> v = Version(1, 2, 3)
        >>> str(v)
        '1.2.3'
        >>> v.major
        1
    """

    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    @classmethod
    def from_string(cls, version_string: str) -> "VersionInfo":
        """
        Parse a version string in the format X.Y.Z into a Version object.

        Args:
            version_string (str): Version string in the format "X.Y.Z" (e.g., "1.2.3")

        Returns:
            Version: A Version object with parsed major, minor, and patch values

        Raises:
            ValueError: If the version string is not in the correct format

        Examples:
            >>> Version.from_string("1.2.3")
            Version(major=1, minor=2, patch=3)
            >>> Version.from_string("10.0.5")
            Version(major=10, minor=0, patch=5)
        """
        try:
            parts = version_string.split(".")
            if len(parts) != 3:
                raise ValueError(
                    f"Version string must have exactly 3 parts separated by '.': {version_string}"
                )

            major = int(parts[0])
            minor = int(parts[1])
            patch = int(parts[2])

            return cls(major, minor, patch)
        except (ValueError, AttributeError) as e:
            raise ValueError(
                f"Invalid version string format: {version_string}. Expected X.Y.Z format."
            ) from e


def parse_version(version_string: str) -> VersionInfo:
    """
    Parse a version string in the format X.Y.Z into a Version object.

    Args:
        version_string (str): Version string in the format "X.Y.Z" (e.g., "1.2.3")

    Returns:
        Version: A Version object with major, minor, and patch components

    Raises:
        ValueError: If the version string is not in the correct format

    Examples:
        >>> parse_version("1.2.3")
        Version(major=1, minor=2, patch=3)
        >>> parse_version("10.0.5")
        Version(major=10, minor=0, patch=5)
    """
    return VersionInfo.from_string(version_string)


def check_version(input_file: str, cli_command: CliCommand = CliCommand.UNKNOWN):
    """
    Verify that the version in the input file matches the current VERSION and exit if mismatch.
    This function reads the version from the specified Excel file and compares it with
    the current VERSION constant. If the versions don't match or if the version cannot
    be read, the program will exit with status code 1.
    Args:
        input_file (str): Path to the Excel file containing version information.
        cli_command (CliCommand): The CLI command context for version checking.
    Returns:
            None: This function does not return a value. It either completes successfully
              or exits the program.
    Raises:
        SystemExit: If the version cannot be read from the file or if there is a
                    version mismatch between the file and the current VERSION.
    Note:
        This function has side effects:
        - Prints error messages to stdout on failure
        - Logs success message using logger
        - Calls sys.exit(1) on any version-related issues
    """
    file_version, message = get_version_from_sheet(input_file, cli_command)

    if file_version is None:
        print(f"Error reading version: {message}")
        sys.exit(1)

    file_version_info = parse_version(file_version.strip())
    current_version_info = parse_version(VERSION.strip())

    if (
        file_version_info.major != current_version_info.major
        or file_version_info.minor != current_version_info.minor
    ):
        error_message = (
            f"Version mismatch: file version '{file_version}' != current version '{VERSION}'\n"
            f"Major or minor version difference detected.\n"
            f"Please upgrade the package by running:\n"
            f"    pip install --upgrade euroncap_rating_2026"
        )
        print(error_message)
        sys.exit(1)
    elif file_version_info.patch != current_version_info.patch:
        warning_message = (
            f"Warning: Patch version mismatch detected.\n"
            f"File version: '{file_version}' != Current version: '{VERSION}'\n"
            f"Consider upgrading the package by running:\n"
            f"    pip install --upgrade euroncap_rating_2026\n"
            f"Proceeding with current execution..."
        )
        logger.warning(warning_message)
        print(warning_message)

    logger.info(f"Version verification passed: {VERSION}")


def get_version_label_for_command(cli_command: CliCommand) -> str:
    """Get the appropriate version label text for a given CLI command.

    The template row is labeled "Template version" (not "Generated with
    version") since the packaged data/*.xlsx masters carry the stamp
    themselves and generate-template is a pure copy; files written by
    versions <= 5.4.5 with the old label are still read fine
    (get_version_search_substrings_for_command).
    """
    command_labels = {
        CliCommand.GENERATE_TEMPLATE: "Template version",
        CliCommand.PREPROCESS: "Preprocessed with version",
        CliCommand.COMPUTE_SCORE: "Score computed with version",
        CliCommand.UNKNOWN: "Template version",
    }
    return command_labels.get(cli_command, "Template version")


def get_version_row_for_command(cli_command: CliCommand) -> int:
    """Get the appropriate row number for a given CLI command."""
    command_rows = {
        CliCommand.GENERATE_TEMPLATE: 1,
        CliCommand.PREPROCESS: 2,
        CliCommand.COMPUTE_SCORE: 3,
        CliCommand.UNKNOWN: 1,
    }
    return command_rows.get(cli_command, 1)


def add_version_sheet(file_path: str, cli_command: CliCommand = CliCommand.UNKNOWN):
    """Add a version sheet to the workbook loaded from file_path."""
    wb = openpyxl.load_workbook(file_path)
    add_version_sheet_to_wb(wb, cli_command)
    wb.save(file_path)


def add_version_sheet_to_wb(wb, cli_command: CliCommand = CliCommand.UNKNOWN):
    """Helper to add a version sheet to the given workbook object."""
    if "Version" not in wb.sheetnames:
        ws = wb.create_sheet("Version")
    else:
        ws = wb["Version"]

    white_font = Font(color="FFFFFF", name="Aptos Narrow")
    black_fill = PatternFill(
        start_color="000000", end_color="000000", fill_type="solid"
    )
    first_col_width = 5 / CM_TO_EXCEL_WIDTH  # Convert cm to Excel width units
    row = get_version_row_for_command(cli_command)
    ws[f"A{row}"] = get_version_label_for_command(cli_command)
    ws[f"B{row}"] = VERSION
    ws.column_dimensions["A"].width = first_col_width
    ws[f"A{row}"].fill = black_fill
    ws[f"B{row}"].fill = black_fill
    ws[f"A{row}"].font = white_font
    ws[f"B{row}"].font = white_font


def extract_section(df, section_name, col_name="Scenario"):
    """
    Extract a section from a DataFrame based on a scenario name.

    Args:
        df (pd.DataFrame): The DataFrame containing a 'Scenario' column.
        section_name (str): The name of the scenario section to extract.

    Returns:
        pd.DataFrame: The extracted section as a DataFrame. If the section is not found, returns an empty DataFrame.
    """
    start_idx = df[df[col_name] == section_name].index
    if start_idx.empty:
        logging.warning(f"No '{section_name}' scenario found in provided DataFrame.")
        return pd.DataFrame()
    start_idx = start_idx[0]
    after_start = df.loc[start_idx + 1 :]
    next_non_nan = after_start[after_start[col_name].notna()].index
    if not next_non_nan.empty:
        end_idx = next_non_nan[0]
    else:
        end_idx = df.index[-1] + 1
    return df.loc[start_idx : end_idx - 1]


def extract_table(df, value, col_idx):
    """
    Extracts a table from a DataFrame starting from the row where the column at col_idx equals value,
    and includes all subsequent rows until the next NaN value in that column. All columns are included.

    Args:
        df (pd.DataFrame): The DataFrame to search.
        value: The value to search for in the specified column index.
        col_idx (int): The column index to search in.

    Returns:
        pd.DataFrame: The extracted table as a DataFrame. If the value is not found, returns an empty DataFrame.
    """
    col = df.columns[col_idx]
    start_idx = df[df[col] == value].index

    # If not found, try searching in column names
    if start_idx.empty:
        if value in df.columns:
            col_idx = df.columns.get_loc(value)
            col = df.columns[col_idx]
            logger.debug(
                f"Value '{value}' found as column name at index {col_idx} ('{col}')"
            )
            # Set start_idx to first index (start of DataFrame)
            start_idx = pd.Index([df.index[0]])

    if start_idx.empty:
        logger.warning(
            f"Value '{value}' not found in column index {col_idx} ('{col}') or as column name of provided DataFrame."
        )
        return pd.DataFrame()

    start_idx_val = start_idx[0]
    after_start = df.loc[start_idx_val:]
    # Find the next row where all columns are NaN
    next_full_nan = after_start[after_start.isnull().all(axis=1)].index
    if not next_full_nan.empty:
        end_idx = next_full_nan[0]
    else:
        end_idx = df.index[-1] + 1

    table_df = df.loc[start_idx_val : end_idx - 1]
    return table_df


def opposite_ffill(df, exclude_columns=[]):
    # For each column, set consecutive duplicate values to NaN except the first
    for col in df.columns:
        if col in exclude_columns:
            continue
        mask = df[col] == df[col].shift()
        df.loc[mask, col] = pd.NA
    return df


def extract_sub_table(df, start_row, n_rows, start_col, n_cols):
    """
    Extracts a sub-table from a DataFrame given starting row/column and number of rows/columns.

    Args:
        df (pd.DataFrame): The DataFrame to extract from.
        start_row (int): The starting row index (0-based).
        n_rows (int): The number of rows to include.
        start_col (int): The starting column index (0-based).
        n_cols (int): The number of columns to include.

    Returns:
        pd.DataFrame: The extracted sub-table as a DataFrame.
    """
    end_row = start_row + n_rows
    end_col = start_col + n_cols
    sub_table_df = df.iloc[start_row:end_row, start_col:end_col]
    return sub_table_df


RGB_TO_COLOR = {
    "FF009933": PredictionColor.GREEN,
    "FFFF3333": PredictionColor.RED,
    "00009933": PredictionColor.GREEN,
    "00000000": PredictionColor.GREY,
}

# Background fill used by report writers to mark cells the OEM must fill in.
INPUT_CELL_RGB = "D9D9D9"


def get_cell_background_colors(ws, cells=None):
    colors = {}
    if cells is None:
        max_row, max_col = get_populated_sheet_bounds(ws)
        cells = [
            cell.coordinate
            for row in ws.iter_rows(max_row=max_row, max_col=max_col)
            for cell in row
        ]
    for cell in cells:
        fill = ws[cell].fill
        color = None
        if fill and fill.fgColor:
            if fill.fgColor.type == "rgb" and fill.fgColor.rgb:
                rgb_code = fill.fgColor.rgb
                color = RGB_TO_COLOR.get(rgb_code, rgb_code)
            elif fill.fgColor.type == "theme" and hasattr(fill.fgColor, "theme"):
                color = f"theme:{fill.fgColor.theme}"
            elif fill.fgColor.type == "indexed" and hasattr(fill.fgColor, "indexed"):
                color = f"indexed:{fill.fgColor.indexed}"
        colors[cell] = color
    return colors


# =============================================================================
# Grey-cell trust boundary
#
# The xlsx template is the sole interchange format between generate-template,
# preprocess and compute-score, and is fully editable by the user at every
# stage. A GreyCellSheetSchema declares, per sheet, which columns are
# legitimate OEM input ("grey" cells) and which are reference/structural data
# (e.g. HPL/LPL/Capping/Max score, row labels) that must never be trusted from
# a user-supplied file. rebuild_trusted_workbook() sources every non-grey
# cell from the pristine packaged template instead, keyed by row identity
# rather than row position so inserted/deleted/reordered rows are also caught.
# =============================================================================


class TemplateIntegrityError(Exception):
    """Raised when a user-supplied workbook's protected structure or
    reference data doesn't match the trusted template."""


@dataclass
class LegacyUnlabeledCell:
    """One narrow, explicit exemption for a specific known template
    revision that gave a name to a previously-blank identity cell (e.g.
    a later sd_template.xlsx revision naming "Input parameters"'s SLIF -
    System updates row). A prediction file legitimately produced against
    the prior revision has that one cell blank; without this, forward-fill
    makes it inherit the wrong label from the row above and the identity
    comparison rejects a row that never actually moved.

    This is deliberately scoped to exactly one (row, column) pair, not a
    general "blank identity cell is OK if some other column is populated"
    rule -- that would let any current-template upload blank an arbitrary
    identity cell and have it silently substituted from the pristine
    template, which would mask real tampering or a reordered row on the
    same line. All three fields must match before the exemption applies:

    column: the identity column that's allowed to be blank.
    anchor: {other identity column: pristine value} that must hold at the
        row for this exemption to apply -- identifies the row using the
        PRISTINE template's own (forward-filled) values, never the user's,
        so the exemption can't be redirected to a different row by
        tampering with the user's file.
    value: the pristine template's current (non-blank) value for *column*
        at that row -- checked defensively so this entry can't silently
        start matching a different row if the template changes again.
    """

    column: str
    anchor: dict
    value: str


def _matches_legacy_unlabeled_cell(
    column_name: str, pristine_row_identity: dict, legacy_unlabeled_cells: list
) -> bool:
    return any(
        entry.column == column_name
        and pristine_row_identity.get(column_name) == entry.value
        and all(pristine_row_identity.get(k) == v for k, v in entry.anchor.items())
        for entry in legacy_unlabeled_cells
    )


@dataclass
class GreyCellSheetSchema:
    """Declares the trust boundary for one worksheet.

    identity_columns: columns used to match a row to its counterpart in the
        pristine template. Values are forward-filled before matching (blank
        cell means "same as the row above"), matching how these sheets lay
        out hierarchical labels like Loadcase/Seat position/Dummy.
    grey_columns: the only columns whose values are taken from the
        user-supplied file. Every other column is sourced from the pristine
        template, regardless of what the user's file contains there.
    """

    identity_columns: list
    grey_columns: list = field(default_factory=list)
    # Explicit, narrow exemptions for known template revisions that named a
    # previously-blank identity cell -- see LegacyUnlabeledCell. Empty by
    # default: a blank identity cell is ordinary forward-filled-and-compared
    # unless a schema explicitly whitelists it here.
    legacy_unlabeled_cells: list = field(default_factory=list)
    # Columns that are neither user input nor protected reference data: they
    # are overwritten by this tool's own downstream computation (e.g. the
    # "Score"/"Capping?" columns filled in by compute-score) and may hold a
    # placeholder formula or blank cell in the pristine template whose
    # representation isn't stable across a data_only save/reload round trip.
    # Excluded from both the grey overlay and the protected-cell signature.
    computed_columns: list = field(default_factory=list)
    # Subset of grey_columns the OEM is expected to fill in themselves --
    # narrower than grey_columns, which only means "not sourced from the
    # pristine template". E.g. "Inspection [%]" is grey (OEM-editable,
    # never overwritten by rebuild_trusted_*) but is filled in by the
    # inspector later in the tool, not by the OEM; a criteria sheet's
    # "Value" is grey but is discarded and re-merged from calculated
    # test-data KPIs during scoring, so it can't be filled by anyone before
    # the test has run. Used by find_missing_required_inputs. None (the
    # default) means "same as grey_columns", for sheets where the two sets
    # coincide. Set to an explicit [] to declare that grey_columns exist
    # but none of them are OEM-required -- don't confuse the two.
    prediction_required_columns: list | None = None
    # For sheets that are entirely dedicated to one protocol element (e.g.
    # "CP - Frontal Offset" has no "Stage element"/"Stage subelement" column
    # of its own -- the whole sheet *is* Frontal Impact/Offset), declare the
    # (stage_element, stage_subelement) pair(s) this sheet belongs to here.
    # find_missing_required_inputs uses this instead of row-level identity
    # matching to scope the sheet to a requested protocol element. Leave
    # None for shared, multi-element sheets (Input parameters, Test Scores,
    # ...) whose own "Stage element"/"Stage subelement" identity columns
    # already provide row-level scoping.
    applies_to_stage_elements: list | None = None
    # Row-level gate on prediction_required_columns: when set, a row's
    # required columns are only required if every listed column has a
    # value in the row's PRISTINE-template counterpart (never the user's
    # file -- an OEM must not be able to make a cell optional by blanking
    # a reference column). E.g. crash_protection's criteria sheets only
    # expect an OEM Prediction on sliding-scale criteria rows -- the ones
    # where both HPL and LPL are present -- while modifier and
    # single-limit rows have no predictable color and scoring treats
    # their OEM Prediction as optional. Requires the caller to pass
    # pristine_wb to find_missing_required_inputs; without it the gate
    # fails closed (every row required, never fewer).
    required_when_columns_present: list | None = None


def _sheet_header(ws) -> list:
    return [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]


def _column_indexes_by_name(header: list, names: list, sheet_name: str) -> dict:
    indexes = {}
    for name in names:
        if name not in header:
            raise TemplateIntegrityError(
                f"Sheet '{sheet_name}' is missing expected column '{name}'."
            )
        indexes[name] = header.index(name) + 1
    return indexes


def _normalize_identity_value(value):
    """Strip a trailing footnote marker ("*") from an identity-column value
    before it's used for template-integrity comparison. Footnote markers are
    a display detail (the footnote text itself lives in its own column) and
    must not be part of the trust boundary -- otherwise every future tweak
    to which labels carry a footnote invalidates every in-flight prediction
    file uploaded against the previous template revision."""
    if isinstance(value, str):
        return re.sub(r"\*+\s*$", "", value).rstrip()
    return value


def _forward_filled_identity_key_pairs(
    pristine_ws,
    pristine_col_idxs: list,
    user_ws,
    user_col_idxs: list,
    max_row: int,
    identity_columns: list,
    legacy_unlabeled_cells: list,
) -> tuple:
    """Forward-fill *pristine_ws* and *user_ws*'s identity columns together,
    row by row, and return (pristine_keys, user_keys). A user cell that's
    blank where the pristine cell at the same row/column is not is treated
    as carrying no constraint -- instead of forward-filling the wrong value
    into it -- ONLY when that exact (row, column) is explicitly whitelisted
    by *legacy_unlabeled_cells* (see LegacyUnlabeledCell); every other blank
    identity cell is compared exactly as before. The exemption doesn't seed
    the user's own forward-fill state with the pristine value, so a later,
    genuinely blank continuation cell in the same column still inherits the
    user's own prior label rather than the template's."""
    n = len(pristine_col_idxs)
    pristine_last = [None] * n
    user_last = [None] * n
    pristine_keys, user_keys = [], []
    for row in range(2, max_row + 1):
        user_row_key = [None] * n
        for i in range(n):
            pristine_raw = pristine_ws.cell(row=row, column=pristine_col_idxs[i]).value
            user_raw = user_ws.cell(row=row, column=user_col_idxs[i]).value
            if not is_empty_cell(pristine_raw):
                pristine_last[i] = _normalize_identity_value(pristine_raw)
            pristine_row_identity = dict(zip(identity_columns, pristine_last))
            if (
                is_empty_cell(user_raw)
                and not is_empty_cell(pristine_raw)
                and _matches_legacy_unlabeled_cell(
                    identity_columns[i], pristine_row_identity, legacy_unlabeled_cells
                )
            ):
                user_row_key[i] = pristine_last[i]
            else:
                if not is_empty_cell(user_raw):
                    user_last[i] = _normalize_identity_value(user_raw)
                user_row_key[i] = user_last[i]
        pristine_keys.append(tuple(pristine_last))
        user_keys.append(tuple(user_row_key))
    return pristine_keys, user_keys


def _describe_identity_mismatch(
    pristine_keys: list, user_keys: list, identity_columns: list
) -> str:
    """Return a human-readable pinpoint of the first row/column where
    *pristine_keys* and *user_keys* diverge, e.g. "Row 6: column 'Input
    parameter' is 'Prediction - Standard*' in your file vs 'Prediction -
    Standard' in the official template." Returns "" if the two lists have no
    row where a corresponding identity value differs (e.g. divergence is
    purely due to a row-count/shift difference)."""
    for row_idx, (pristine_key, user_key) in enumerate(zip(pristine_keys, user_keys)):
        if pristine_key == user_key:
            continue
        diffs = [
            f"column '{col_name}' is '{user_val}' in your file vs "
            f"'{pristine_val}' in the official template"
            for col_name, pristine_val, user_val in zip(
                identity_columns, pristine_key, user_key
            )
            if pristine_val != user_val
        ]
        if diffs:
            return f"Row {row_idx + 2}: " + "; ".join(diffs)
    return ""


def rebuild_trusted_sheet(
    pristine_ws, user_ws, schema: GreyCellSheetSchema, sheet_name: str
) -> None:
    """Overlay *schema.grey_columns* from *user_ws* onto *pristine_ws*, in
    place, after verifying both sheets have the same identity-column
    structure (same rows, in the same order). Column positions are looked up
    independently by header name in each sheet, so reordering columns can't
    be used to smuggle a value into a differently-named protected column.

    Raises TemplateIntegrityError if a required column is missing from
    either sheet, or if the identity-column rows don't line up (rows were
    inserted, deleted, or reordered relative to the pristine template).
    """
    pristine_header = _sheet_header(pristine_ws)
    user_header = _sheet_header(user_ws)

    pristine_identity_idxs = _column_indexes_by_name(
        pristine_header, schema.identity_columns, sheet_name
    )
    user_identity_idxs = _column_indexes_by_name(
        user_header, schema.identity_columns, sheet_name
    )
    pristine_grey_idxs = _column_indexes_by_name(
        pristine_header, schema.grey_columns, sheet_name
    )
    user_grey_idxs = _column_indexes_by_name(
        user_header, schema.grey_columns, sheet_name
    )

    # The pristine template is the sole authority on how many real rows this
    # table has. Rows beyond it are compared against identity columns only
    # (not via get_populated_sheet_bounds), because sheets commonly have
    # data-validation ranges that reach further down than the actual table,
    # and a stray value dropped in one of those otherwise-blank cells must
    # not be mistaken for tampering.
    pristine_max_row, _ = get_populated_sheet_bounds(pristine_ws)

    pristine_keys, user_keys = _forward_filled_identity_key_pairs(
        pristine_ws,
        list(pristine_identity_idxs.values()),
        user_ws,
        list(user_identity_idxs.values()),
        pristine_max_row,
        schema.identity_columns,
        schema.legacy_unlabeled_cells,
    )

    if pristine_keys != user_keys:
        detail = _describe_identity_mismatch(
            pristine_keys, user_keys, schema.identity_columns
        )
        raise TemplateIntegrityError(
            f"Sheet '{sheet_name}' rows do not match the official template "
            "(rows appear to have been reordered, inserted, or deleted). "
            "Please start from a freshly generated template and re-enter "
            "your values." + (f" {detail}." if detail else "")
        )

    user_identity_col_idxs = list(user_identity_idxs.values())
    for row in range(pristine_max_row + 1, user_ws.max_row + 1):
        if any(
            not is_empty_cell(user_ws.cell(row=row, column=col_idx).value)
            for col_idx in user_identity_col_idxs
        ):
            raise TemplateIntegrityError(
                f"Sheet '{sheet_name}' has extra rows beyond the official "
                "template (rows appear to have been inserted). Please start "
                "from a freshly generated template and re-enter your values."
            )

    for row in range(2, pristine_max_row + 1):
        for col_name in schema.grey_columns:
            pristine_ws.cell(row=row, column=pristine_grey_idxs[col_name]).value = (
                user_ws.cell(row=row, column=user_grey_idxs[col_name]).value
            )


def rebuild_trusted_workbook(
    pristine_path: str,
    user_wb,
    schemas: dict,
    passthrough_sheets: list = None,
):
    """Return a freshly-loaded workbook from *pristine_path*, with each
    schema'd sheet's grey columns overlaid with the corresponding values from
    *user_wb*, and *passthrough_sheets* copied verbatim from *user_wb*.

    Every cell not covered by a grey column or a passthrough sheet is
    guaranteed to come from the pristine template, never from *user_wb*.
    """
    out_wb = openpyxl.load_workbook(pristine_path)
    for sheet_name, schema in schemas.items():
        if sheet_name not in out_wb.sheetnames:
            continue
        if sheet_name not in user_wb.sheetnames:
            raise TemplateIntegrityError(
                f"Required sheet '{sheet_name}' not found in input file."
            )
        rebuild_trusted_sheet(
            out_wb[sheet_name], user_wb[sheet_name], schema, sheet_name
        )

    for sheet_name in passthrough_sheets or []:
        if sheet_name in user_wb.sheetnames:
            hard_copy_sheet(user_wb, sheet_name, out_wb)

    return out_wb


# =============================================================================
# Missing required-input check
#
# Which cells the OEM must fill in themselves is declared in code
# (GreyCellSheetSchema.prediction_required_columns, per domain
# SHEET_SCHEMAS), never read back from the submitted file's own cell fill
# colors -- the same trust-boundary principle as rebuild_trusted_workbook
# above. This intentionally makes the check backwards-compatible with older
# templates that used different (or no) grey fills, and immune to an OEM
# simply removing a fill to make a cell optional.
# =============================================================================


# Identifiers for the rule that produced a finding, carried in
# MissingInput.check so one findings list can cover every consistency rule
# ("one list of findings covering all rules") and a
# consumer can still group or style by rule without parsing the message.
CHECK_MISSING_INPUT = "missing_input"
CHECK_LSC_DOORING = "lsc_dooring"
CHECK_CPLA_CBLA_COHERENCE = "cpla_cbla_coherence"
CHECK_LEGFORM_SYMMETRY = "legform_symmetry"
CHECK_DM_GROUP_CONSISTENCY = "dm_group_consistency"
CHECK_FCW_COLOR = "fcw_color"
CHECK_DM_NA_FORCED_RED = "dm_na_forced_red"
CHECK_NO_PREDICTION_PROVIDED = "no_prediction_provided"
CHECK_SCENARIO_NA_CONSISTENCY = "scenario_na_consistency"
CHECK_LDW_EXTENDED_COLOR = "ldw_extended_color"
CHECK_VERSION_MISMATCH = "version_mismatch"
CHECK_STALE_ACC_CUT_OUT = "stale_acc_cut_out"


@dataclass
class MissingInput:
    """One consistency-check finding, with enough context to act on without
    opening the workbook: the sheet, the column header, and the
    forward-filled row identity (e.g. Loadcase/Seat position/Dummy/
    Body region/Criteria, or Stage element/Stage subelement/Input
    parameter) -- not just a bare cell coordinate.

    Historically this reported only empty required cells (hence the name,
    kept for API compatibility -- ConsistencyFinding below is the alias new
    code should read); ``check`` says which rule produced the finding
    (CHECK_* above), and rule violations that are not about an empty cell
    carry their full sentence in ``message``."""

    sheet: str
    column: str
    cell: str
    row_identity: dict = field(default_factory=dict)
    check: str = CHECK_MISSING_INPUT
    message: str = None

    @property
    def description(self) -> str:
        if self.message:
            return self.message
        identity = " / ".join(
            str(value) for value in self.row_identity.values() if value is not None
        )
        location = f"{identity} / {self.column}" if identity else self.column
        return f"{self.sheet} — {location} is missing"

    def __str__(self) -> str:
        return self.description


# What MissingInput has grown into: the record type for every consistency
# rule's findings, not just missing required inputs.
ConsistencyFinding = MissingInput


# Sheet/column renames landed in 5.4.4 made 5.4.0-generated
# templates structurally incompatible even though they only differ by patch
# version -- check_version's normal policy would just warn-and-proceed on a
# patch-only difference. Special-cased here rather than generalizing "any
# patch difference is incompatible" (untrue for most patch bumps).
KNOWN_INCOMPATIBLE_FILE_VERSIONS = {"5.4.0"}


def find_version_mismatch(wb, cli_command: CliCommand = CliCommand.UNKNOWN) -> list:
    """Non-exiting sibling of check_version: compares *wb*'s Version sheet
    against the running library's VERSION and returns the mismatch as a
    ConsistencyFinding instead of exiting the process, so upload-time checks
    (find_missing_required_inputs/find_consistency_findings) can surface a
    version drift through the same findings list a consuming application already
    reads, rather than that drift silently corrupting the workbook and only
    surfacing later as an unrelated KeyError.

    Returns [] when the Version sheet/label can't be read at all -- nothing
    to compare, not the same claim as "compared and it matches" -- or when
    the version matches under the same policy check_version uses: an exact
    match, or a patch-only difference that isn't a known-incompatible file
    version (KNOWN_INCOMPATIBLE_FILE_VERSIONS). Returns a single-element list
    otherwise.
    """
    file_version, _ = get_version_from_workbook(wb, cli_command)
    if file_version is None:
        return []

    try:
        file_version_info = parse_version(file_version.strip())
        current_version_info = parse_version(VERSION.strip())
    except ValueError:
        return []

    major_minor_mismatch = (
        file_version_info.major != current_version_info.major
        or file_version_info.minor != current_version_info.minor
    )
    known_incompatible = file_version.strip() in KNOWN_INCOMPATIBLE_FILE_VERSIONS and (
        current_version_info.major,
        current_version_info.minor,
        current_version_info.patch,
    ) >= (5, 4, 4)

    if not major_minor_mismatch and not known_incompatible:
        return []

    ws = wb["Version"]
    row = None
    for label_substring in get_version_search_substrings_for_command(cli_command):
        row = find_version_label_row(ws, label_substring)
        if row is not None:
            break
    column_header = str(ws.cell(row=row, column=1).value)

    return [
        ConsistencyFinding(
            sheet="Version",
            column=column_header,
            cell=f"B{row}",
            check=CHECK_VERSION_MISMATCH,
            message=(
                f"This template was generated with version {file_version}; "
                f"this system requires version {VERSION}. Please download a "
                f"fresh template."
            ),
        )
    ]


def validate_scope(
    stage_element, stage_subelement, valid_scope_pairs, domain: str
) -> None:
    """Raise ValueError when a requested (stage_element, stage_subelement)
    scope matches nothing the domain declares. An unrecognised scope would
    otherwise silently return zero findings -- indistinguishable from a
    fully filled workbook -- so a typo'd caller-side mapping (e.g.
    snake_case identifiers instead of the sheets' own display strings)
    would silently disable the check: the same silent-pass failure mode
    the declared-in-code design exists to prevent.

    valid_scope_pairs: the domain's declared (stage_element,
    stage_subelement) pairs. Passing neither argument (whole-workbook
    check) is always valid; passing only one validates that one.
    """
    if stage_element is None and stage_subelement is None:
        return

    def declared() -> str:
        return ", ".join(f"({e!r}, {s!r})" for e, s in valid_scope_pairs)

    if stage_element is not None and stage_element not in {
        element for element, _ in valid_scope_pairs
    }:
        raise ValueError(
            f"Unknown stage_element {stage_element!r} for the {domain} domain. "
            f"Declared (stage_element, stage_subelement) pairs: {declared()}"
        )
    if stage_subelement is not None:
        subelements = {
            subelement
            for element, subelement in valid_scope_pairs
            if stage_element is None or element == stage_element
        }
        if stage_subelement not in subelements:
            raise ValueError(
                f"Unknown stage_subelement {stage_subelement!r}"
                + (f" under {stage_element!r}" if stage_element is not None else "")
                + f" for the {domain} domain. Declared (stage_element, "
                f"stage_subelement) pairs: {declared()}"
            )


def stage_elements_in_scope(
    applies_to_stage_elements, stage_element, stage_subelement
) -> bool:
    """Whether a sheet declared to belong to *applies_to_stage_elements*
    (a list of (stage_element, stage_subelement) pairs) is relevant to a
    requested scope. Shared by GreyCellSheetSchema-driven sheets and grid
    sheets (find_missing_grid_inputs' callers, which have no schema of
    their own to hang this off of)."""
    if applies_to_stage_elements is None:
        return True
    return any(
        (stage_element is None or sheet_stage_element == stage_element)
        and (stage_subelement is None or sheet_stage_subelement == stage_subelement)
        for sheet_stage_element, sheet_stage_subelement in applies_to_stage_elements
    )


def _sheet_applies_to_scope(schema, stage_element, stage_subelement) -> bool:
    return stage_elements_in_scope(
        schema.applies_to_stage_elements, stage_element, stage_subelement
    )


def _row_matches_scope(schema, row_identity, stage_element, stage_subelement) -> bool:
    if schema.applies_to_stage_elements is not None:
        # Scoping already decided at the sheet level in _sheet_applies_to_scope.
        return True
    if stage_element is not None and row_identity.get("Stage element") != stage_element:
        return False
    if (
        stage_subelement is not None
        and row_identity.get("Stage subelement") != stage_subelement
    ):
        return False
    return True


def _iter_forward_filled_rows(
    ws, identity_columns: list, identity_idxs: dict, max_row: int
):
    """Yield (row, row_identity) for every data row of *ws*, with identity
    values forward-filled.

    Forward-fill cascades: once a column gets an explicit value that
    differs from a blank, every column to its right resets to None (not
    "inherited from above") unless it also has its own explicit value in
    this row. Without this, a column whose own value is legitimately blank
    for a whole group (e.g. "Scenario" under a category that has none)
    would otherwise leak the last *unrelated* group's stale value forward
    once an ancestor column (e.g. "Category") changes underneath it.
    """
    last = [None] * len(identity_columns)
    for row in range(2, max_row + 1):
        reset = False
        row_identity = {}
        for i, column in enumerate(identity_columns):
            value = ws.cell(row=row, column=identity_idxs[column]).value
            if not is_empty_cell(value):
                last[i] = value
                reset = True
            elif reset:
                last[i] = None
            row_identity[column] = last[i]
        yield row, row_identity


def _normalized_identity_tuple(row_identity: dict, identity_columns: list) -> tuple:
    return tuple(
        _normalize_identity_value(row_identity[column]) for column in identity_columns
    )


def _required_row_identities(pristine_ws, schema, sheet_name: str) -> set:
    """Identity tuples of the pristine rows where every
    schema.required_when_columns_present column has a value -- the only
    rows whose prediction_required_columns are actually required. Gate
    values are read exclusively from the pristine sheet, so blanking a
    reference column in a submitted file can't make its row optional."""
    header = _sheet_header(pristine_ws)
    identity_idxs = _column_indexes_by_name(header, schema.identity_columns, sheet_name)
    gate_idxs = _column_indexes_by_name(
        header, schema.required_when_columns_present, sheet_name
    )
    max_row, _ = get_populated_sheet_bounds(pristine_ws)
    identities = set()
    for row, row_identity in _iter_forward_filled_rows(
        pristine_ws, schema.identity_columns, identity_idxs, max_row
    ):
        if row_identity[schema.identity_columns[-1]] is None:
            continue
        if all(
            not is_empty_cell(pristine_ws.cell(row=row, column=col_idx).value)
            for col_idx in gate_idxs.values()
        ):
            identities.add(
                _normalized_identity_tuple(row_identity, schema.identity_columns)
            )
    return identities


def find_missing_required_inputs(
    wb,
    schemas: dict,
    stage_element: str = None,
    stage_subelement: str = None,
    pristine_wb=None,
) -> list:
    """Find empty cells the OEM is required to fill in, per *schemas* (a
    domain's SHEET_SCHEMAS -- see GreyCellSheetSchema.prediction_required_columns).

    Args:
        wb: An opened openpyxl Workbook (e.g. openpyxl.load_workbook(path, data_only=True)).
        schemas: The domain's SHEET_SCHEMAS dict.
        stage_element: If given, only report cells belonging to this
            protocol ("Stage element" in the workbook, e.g. "Frontal Impact").
        stage_subelement: If given, only report cells belonging to this
            protocol element ("Stage subelement", e.g. "Offset"). Sheets
            shared across elements (e.g. "Input parameters") are filtered
            row by row; sheets dedicated to a single element (declared via
            GreyCellSheetSchema.applies_to_stage_elements) are included or
            skipped as a whole.
        pristine_wb: The domain's packaged pristine template, needed by
            schemas that gate required-ness on reference columns
            (GreyCellSheetSchema.required_when_columns_present). Without
            it such gates fail closed (every row stays required).

    Returns:
        list[MissingInput]: one entry per empty required cell.
    """
    missing = []
    for sheet_name, schema in schemas.items():
        required_columns = (
            schema.grey_columns
            if schema.prediction_required_columns is None
            else schema.prediction_required_columns
        )
        if not required_columns or sheet_name not in wb.sheetnames:
            continue
        if not _sheet_applies_to_scope(schema, stage_element, stage_subelement):
            continue

        ws = wb[sheet_name]
        header = _sheet_header(ws)
        identity_idxs = _column_indexes_by_name(
            header, schema.identity_columns, sheet_name
        )
        required_idxs = {
            column: header.index(column) + 1
            for column in required_columns
            if column in header
        }
        if not required_idxs:
            continue

        required_identities = None
        if (
            schema.required_when_columns_present
            and pristine_wb is not None
            and sheet_name in pristine_wb.sheetnames
        ):
            required_identities = _required_row_identities(
                pristine_wb[sheet_name], schema, sheet_name
            )

        max_row, _ = get_populated_sheet_bounds(ws)
        for row, row_identity in _iter_forward_filled_rows(
            ws, schema.identity_columns, identity_idxs, max_row
        ):
            # A row whose own finest-grained identity (the last identity
            # column, e.g. "Input parameter"/"Criteria") is blank isn't a
            # real data row (e.g. a CP "Input parameters" row with a grey
            # Value cell but no "Input parameter" name) -- nothing to
            # require here regardless of fill color. Other identity
            # columns (e.g. "Scenario") may legitimately stay blank for an
            # entire group.
            if row_identity[schema.identity_columns[-1]] is None:
                continue

            if not _row_matches_scope(
                schema, row_identity, stage_element, stage_subelement
            ):
                continue

            if (
                required_identities is not None
                and _normalized_identity_tuple(row_identity, schema.identity_columns)
                not in required_identities
            ):
                continue

            for column, col_idx in required_idxs.items():
                cell = ws.cell(row=row, column=col_idx)
                if isinstance(cell, MergedCell):
                    continue
                if is_empty_cell(cell.value):
                    missing.append(
                        MissingInput(
                            sheet=sheet_name,
                            column=column,
                            cell=cell.coordinate,
                            row_identity=dict(row_identity),
                        )
                    )
    return missing


def find_missing_required_inputs_in_file(
    input_file: str,
    schemas: dict,
    stage_element: str = None,
    stage_subelement: str = None,
    pristine_wb=None,
) -> list:
    """Load an Excel file and return find_missing_required_inputs() for it."""
    wb = openpyxl.load_workbook(input_file, data_only=True)
    try:
        return find_missing_required_inputs(
            wb, schemas, stage_element, stage_subelement, pristine_wb=pristine_wb
        )
    finally:
        wb.close()


# =============================================================================
# Missing required-input check -- dynamic/matrix grid sheets
#
# Some sheets (crash_avoidance's "...pred."/"...robust. pred.", safe_driving's
# "VA - ACC pred.") carry entirely OEM-input matrices with no forward-
# fillable identity columns of their own -- GreyCellSheetSchema doesn't fit
# them. They already declare which of their cells are OEM dropdown input via
# Excel Data Validation (list) ranges authored directly in the packaged
# template -- the same mechanism _get_dm_prediction_required_coordinates
# (further below) already reads for "DE - DM pred.". These helpers
# generalize that: read the *pristine* template's DV ranges (never the
# user's file) as the required-cell declaration, on any sheet shape, and
# infer a semantic row/column label generically from the sheet's own layout
# rather than hardcoding each sheet's matrix math.
# =============================================================================


def get_required_dropdown_coordinates(ws) -> set:
    """Coordinates of cells governed by any list-type Data Validation in
    *ws*. Whatever applicability logic produced those ranges when the
    template was authored/generated (matrix_processing.CELL_REMOVAL_MAP,
    robustness_layer.ROBUSTNESS_LAYER_TO_VERIFICATION_CONDITION, ...) is
    already reflected in them -- this reads that result back instead of
    re-deriving it, so it works unchanged across sheet shapes."""
    required = set()
    for dv in ws.data_validations.dataValidation:
        if dv.type != "list":
            continue
        for cell_range in dv.sqref.ranges:
            for row_cells in ws.iter_rows(
                min_row=cell_range.min_row,
                max_row=cell_range.max_row,
                min_col=cell_range.min_col,
                max_col=cell_range.max_col,
            ):
                for cell in row_cells:
                    if isinstance(cell, MergedCell):
                        continue
                    required.add(cell.coordinate)
    return required


def _resolve_display_value(ws, row, col):
    """*cell.value*, following a non-anchor merged cell to its anchor."""
    cell = ws.cell(row=row, column=col)
    if not isinstance(cell, MergedCell):
        return cell.value
    for merged_range in ws.merged_cells.ranges:
        if (
            merged_range.min_row <= row <= merged_range.max_row
            and merged_range.min_col <= col <= merged_range.max_col
        ):
            return ws.cell(row=merged_range.min_row, column=merged_range.min_col).value
    return None


def _forward_filled_value(ws, row, col):
    """Nearest non-blank value in *col*, scanning from *row* upward
    (inclusive) -- "blank means same as the row above", same convention as
    _forward_filled_identity_key_pairs."""
    for r in range(row, 0, -1):
        value = _resolve_display_value(ws, r, col)
        if not is_empty_cell(value):
            return value
    return None


def rows_in_sections(ws, section_headers: list) -> dict:
    """Partition *ws*'s populated rows into the labelled sections that
    start at each column-A cell whose value equals one of
    *section_headers* (e.g. "CP - VRU Prediction"'s "Headforms"/"Legform"
    blocks). A section runs from its header row down to the row before
    the next declared header, or to the end of the populated area. Rows
    above the first declared header belong to no section. Anchored on the
    headers' own text so template layout changes (rows added, moved)
    never need a code update -- no row numbers are declared anywhere.

    Returns {section_header: set_of_row_numbers}. Raises
    TemplateIntegrityError if a declared header is not found -- a silent
    empty section would make every cell in it silently optional.
    """
    max_row, _ = get_populated_sheet_bounds(ws)
    header_rows = {}
    for row in range(1, max_row + 1):
        value = _resolve_display_value(ws, row, 1)
        if isinstance(value, str) and value.strip() in section_headers:
            header_rows.setdefault(value.strip(), row)
    missing_headers = set(section_headers) - set(header_rows)
    if missing_headers:
        raise TemplateIntegrityError(
            f"Sheet '{ws.title}' is missing expected section header(s) "
            f"{sorted(missing_headers)} in its first column."
        )
    boundaries = sorted(header_rows.values())
    sections = {}
    for header, start_row in header_rows.items():
        next_starts = [b for b in boundaries if b > start_row]
        end_row = (next_starts[0] - 1) if next_starts else max_row
        sections[header] = set(range(start_row, end_row + 1))
    return sections


def _cell_has_grid_prediction(cell) -> bool:
    """Whether *cell* carries an OEM answer for a dropdown-grid prediction
    cell -- either as literal text (the shape the cell has before running
    preprocess) or as a background fill color (the shape it has after:
    report_writer.write_report's format_prediction_cells step reads the
    OEM's typed dropdown value and re-encodes it as a cell fill, clearing
    the text). Checking both representations makes the caller correct
    regardless of whether preprocess has run yet.

    Only a fill matching a real PREDICTION_COLOR_MAP color other than GREY
    counts as an answer. No dropdown in any template offers "Grey", so a
    grey fill can never be an OEM answer: it is either the
    unanswered-placeholder paint some sheets use (several sheets paint
    their empty dropdown cells literal grey rather than a theme fill) or
    the tool's own encoding of test points the OEM did not select. The
    same goes for any other fill that isn't a prediction color (the
    D9D9D9 input-cell paint, decorative tints, ...). The robustness
    YES/NO tint fills never appear without their cell text, so answered
    robustness cells are covered by the text branch.
    """
    if not is_empty_cell(cell.value):
        return True
    fill = cell.fill
    if not fill or not fill.fgColor or fill.fgColor.type != "rgb":
        return False
    color = _get_prediction_color_from_bgcolor((fill.fgColor.rgb or "").upper())
    return color is not None and color is not PredictionColor.GREY


def find_missing_grid_inputs(user_ws, pristine_ws, sheet_name: str, rows=None) -> list:
    """Find empty required cells in a dropdown-grid sheet with no
    forward-fillable identity columns of its own. Required cells come from
    *pristine_ws*'s own Data Validation ranges -- never from *user_ws* --
    so clearing a dropdown/fill in the user's file can't make a cell
    optional, and an older/differently-formatted file still gets checked
    against the current template's requirements.

    rows: if given (a set of row numbers, e.g. one section from
    rows_in_sections), only required cells on those rows are checked --
    for sheets whose labelled sections belong to different protocol
    elements.

    Row/column identity is inferred generically from the sheet's own
    layout, not hardcoded per sheet: columns that are never part of a
    required range anywhere in the sheet are treated as label columns
    (forward-filled top-to-bottom, same "blank means same as above"
    convention as the rest of this module); the nearest non-blank cell
    strictly above a required cell, in the same column, is treated as that
    column's header/sub-header. Labels are resolved from *pristine_ws*,
    not the user's file: pristine required cells are empty by definition,
    so the upward scan always lands on the real header -- on a text-shape
    user file it would land on a neighbouring *answer* (e.g. "Green")
    instead -- and a label the user edited can't misdescribe the finding.
    """
    required_coords = get_required_dropdown_coordinates(pristine_ws)
    if not required_coords:
        return []

    # A Data Validation range can carry coordinates outside the sheet's
    # actual content (a sqref copy/paste artifact pointing into empty
    # space). Bound the required set to the pristine sheet's populated
    # area so a stray DV coordinate can never become an unsatisfiable
    # requirement the OEM has no cell to fill.
    pristine_max_row, pristine_max_col = get_populated_sheet_bounds(pristine_ws)
    required_cells = {
        (row, col)
        for row, col in (coordinate_to_tuple(c) for c in required_coords)
        if row <= pristine_max_row and col <= pristine_max_col
    }
    if rows is not None:
        required_cells = {(row, col) for row, col in required_cells if row in rows}
    if not required_cells:
        return []
    required_cols = {col for _row, col in required_cells}
    label_cols = [c for c in range(1, pristine_max_col + 1) if c not in required_cols]

    missing = []
    for row, col in sorted(required_cells):
        cell = user_ws.cell(row=row, column=col)
        if isinstance(cell, MergedCell):
            continue
        if _cell_has_grid_prediction(cell):
            continue

        row_identity = {}
        previous_label = None
        for label_col in label_cols:
            if label_col >= col:
                continue
            value = _forward_filled_value(pristine_ws, row, label_col)
            # A row label merged across several label columns (e.g. an
            # A:D-wide section row) resolves to the same value in each of
            # them -- keep it once, not once per covered column.
            if value is not None and value != previous_label:
                row_identity[get_column_letter(label_col)] = value
            previous_label = value

        column_label = _forward_filled_value(pristine_ws, row - 1, col)
        missing.append(
            MissingInput(
                sheet=sheet_name,
                column=(
                    str(column_label)
                    if column_label is not None
                    else get_column_letter(col)
                ),
                cell=cell.coordinate,
                row_identity=row_identity,
            )
        )
    return missing


# =============================================================================
# Grey-cell trust boundary -- DataFrame-native siblings
#
# rebuild_trusted_sheet/rebuild_trusted_workbook above need an on-disk
# pristine file and an opened openpyxl Workbook: they exist for the CLI's
# file-in/file-out preprocess() command. Callers that only ever have
# DataFrames (no .xlsx file at hand -- e.g. an application that parses an upload
# once and works in dataframe/parquet land from there) can't use them. These
# functions apply the exact same trust boundary -- same GreyCellSheetSchema,
# same "identity columns forward-filled and matched against the pristine
# template" logic -- directly to DataFrames.
# =============================================================================


def _is_blank_identity_value(value) -> bool:
    """Like is_empty_cell, but also treats a blank/whitespace-only string as
    empty. A blank openpyxl cell always reads back as None, but a DataFrame
    that round-tripped through parquet can hold "" in a continuation row's
    identity columns instead (a parquet round-trip can turn a blank cell into
    an empty string). Without this, a plain
    is_empty_cell check would treat every such row as starting a brand new
    identity instead of carrying the previous row's value forward, and
    falsely reject a perfectly legitimate dataframe."""
    if is_empty_cell(value):
        return True
    return isinstance(value, str) and value.strip() == ""


def _forward_filled_identity_key_pairs_df(
    pristine_df: pd.DataFrame,
    user_df: pd.DataFrame,
    identity_columns: list,
    n_rows: int,
    legacy_unlabeled_cells: list,
) -> tuple:
    """DataFrame-native sibling of _forward_filled_identity_key_pairs -- see
    its docstring for the scoped blank-user/non-blank-pristine exemption
    rule."""
    n = len(identity_columns)
    pristine_columns = [pristine_df[col].tolist() for col in identity_columns]
    user_columns = [user_df[col].tolist() for col in identity_columns]
    pristine_last = [None] * n
    user_last = [None] * n
    pristine_keys, user_keys = [], []
    for row in range(n_rows):
        user_row_key = [None] * n
        for i in range(n):
            pristine_raw = pristine_columns[i][row]
            user_raw = user_columns[i][row]
            if not _is_blank_identity_value(pristine_raw):
                pristine_last[i] = _normalize_identity_value(pristine_raw)
            pristine_row_identity = dict(zip(identity_columns, pristine_last))
            if (
                _is_blank_identity_value(user_raw)
                and not _is_blank_identity_value(pristine_raw)
                and _matches_legacy_unlabeled_cell(
                    identity_columns[i], pristine_row_identity, legacy_unlabeled_cells
                )
            ):
                user_row_key[i] = pristine_last[i]
            else:
                if not _is_blank_identity_value(user_raw):
                    user_last[i] = _normalize_identity_value(user_raw)
                user_row_key[i] = user_last[i]
        pristine_keys.append(tuple(pristine_last))
        user_keys.append(tuple(user_row_key))
    return pristine_keys, user_keys


def rebuild_trusted_df(
    pristine_df: pd.DataFrame,
    user_df: pd.DataFrame,
    schema: GreyCellSheetSchema,
    sheet_name: str,
) -> pd.DataFrame:
    """DataFrame-native sibling of rebuild_trusted_sheet: returns a copy of
    *pristine_df* with *schema.grey_columns* overlaid from *user_df*, after
    verifying both frames have the same identity-column row structure.

    Raises TemplateIntegrityError if a required column is missing from
    either frame, or if the identity-column rows don't line up (rows were
    inserted, deleted, or reordered relative to the pristine template).
    """
    for name in schema.identity_columns + schema.grey_columns:
        if name not in pristine_df.columns:
            raise TemplateIntegrityError(
                f"Sheet '{sheet_name}' is missing expected column '{name}'."
            )
        if name not in user_df.columns:
            raise TemplateIntegrityError(
                f"Sheet '{sheet_name}' is missing expected column '{name}'."
            )

    pristine_row_count = len(pristine_df)

    if len(user_df) < pristine_row_count:
        raise TemplateIntegrityError(
            f"Sheet '{sheet_name}' rows do not match the official template "
            "(rows appear to have been reordered, inserted, or deleted). "
            "Please start from a freshly generated template and re-enter "
            "your values."
        )

    pristine_keys, user_keys = _forward_filled_identity_key_pairs_df(
        pristine_df,
        user_df,
        schema.identity_columns,
        pristine_row_count,
        schema.legacy_unlabeled_cells,
    )

    if pristine_keys != user_keys:
        detail = _describe_identity_mismatch(
            pristine_keys, user_keys, schema.identity_columns
        )
        raise TemplateIntegrityError(
            f"Sheet '{sheet_name}' rows do not match the official template "
            "(rows appear to have been reordered, inserted, or deleted). "
            "Please start from a freshly generated template and re-enter "
            "your values." + (f" {detail}." if detail else "")
        )

    if len(user_df) > pristine_row_count:
        extra = user_df.iloc[pristine_row_count:]
        for col in schema.identity_columns:
            if extra[col].apply(lambda v: not _is_blank_identity_value(v)).any():
                raise TemplateIntegrityError(
                    f"Sheet '{sheet_name}' has extra rows beyond the "
                    "official template (rows appear to have been "
                    "inserted). Please start from a freshly generated "
                    "template and re-enter your values."
                )

    result = pristine_df.reset_index(drop=True).copy()
    for col in schema.grey_columns:
        result[col] = (
            user_df[col].reset_index(drop=True).iloc[:pristine_row_count].to_numpy()
        )

    return result


def rebuild_trusted_dfs(
    pristine_dfs: dict,
    user_dfs: dict,
    schemas: dict,
    passthrough_sheets: list = None,
) -> dict:
    """DataFrame-native sibling of rebuild_trusted_workbook: returns a copy
    of *user_dfs* with each schema'd sheet present in both *pristine_dfs* and
    *user_dfs* replaced by rebuild_trusted_df(...).

    Unlike rebuild_trusted_workbook, a schema'd sheet missing from *user_dfs*
    is skipped rather than treated as an error: the file-based CLI path
    always has the complete monolithic template, but dataframe-path callers
    legitimately pass a partial dict containing only the sheets they need.
    Skipping means nothing is protected for a sheet that isn't there, but a
    legitimate partial call doesn't spuriously fail either.

    *passthrough_sheets* need no special handling here: since the result
    starts from a copy of *user_dfs*, any sheet not covered by a schema
    (including passthrough sheets) already carries over unchanged.
    """
    result = dict(user_dfs)
    for sheet_name, schema in schemas.items():
        if sheet_name not in pristine_dfs or sheet_name not in user_dfs:
            continue
        result[sheet_name] = rebuild_trusted_df(
            pristine_dfs[sheet_name], user_dfs[sheet_name], schema, sheet_name
        )
    return result


INTEGRITY_SHEET_NAME = "Integrity"

# Local tamper-evidence only; not a cryptographic guarantee.
_INTEGRITY_HMAC_KEY = bytes.fromhex(
    "dc33ae4f544993a001098c18aacf446f380e50a5da3c8a7d638fe4d44456ae18"
)


def _iter_protected_cells(wb, schemas: dict):
    """Yield (sheet_name, row, col_name, value) for every cell in every
    schema'd sheet present in *wb* that is NOT a designated grey column, in a
    stable order."""
    for sheet_name in sorted(schemas):
        if sheet_name not in wb.sheetnames:
            continue
        schema = schemas[sheet_name]
        ws = wb[sheet_name]
        header = _sheet_header(ws)
        excluded_columns = set(schema.grey_columns) | set(schema.computed_columns)
        protected_columns = [
            (idx + 1, name)
            for idx, name in enumerate(header)
            if name and name not in excluded_columns
        ]
        max_row, _ = get_populated_sheet_bounds(ws)
        for row in range(2, max_row + 1):
            for col_idx, col_name in protected_columns:
                yield sheet_name, row, col_name, ws.cell(row=row, column=col_idx).value


def compute_protected_signature(wb, schemas: dict) -> str:
    """HMAC over every protected (non-grey) cell in every schema'd sheet."""
    hasher = hmac.new(_INTEGRITY_HMAC_KEY, digestmod=hashlib.sha256)
    for sheet_name, row, col_name, value in _iter_protected_cells(wb, schemas):
        hasher.update(f"{sheet_name}|{row}|{col_name}|{value!r}\n".encode("utf-8"))
    return hasher.hexdigest()


def add_integrity_sheet_to_wb(wb, schemas: dict) -> None:
    """Stamp *wb* with a hidden sheet recording the HMAC of its own protected
    cells, so a later `verify_protected_signature` call can detect edits made
    to those cells outside this tool."""
    signature = compute_protected_signature(wb, schemas)
    if INTEGRITY_SHEET_NAME in wb.sheetnames:
        ws = wb[INTEGRITY_SHEET_NAME]
    else:
        ws = wb.create_sheet(INTEGRITY_SHEET_NAME)
    ws["A1"] = "Signature"
    ws["B1"] = signature
    ws.sheet_state = "veryHidden"


def hide_integrity_sheet(wb) -> None:
    """Re-hide the "Integrity" sheet after it's been carried through a
    dfs merge and rewritten as a plain DataFrame (e.g. by compute-score,
    which doesn't call add_integrity_sheet_to_wb) -- that round-trip always
    comes back visible, same as Documentation/Data (see
    hide_documentation_data_sheets)."""
    if INTEGRITY_SHEET_NAME in wb.sheetnames:
        wb[INTEGRITY_SHEET_NAME].sheet_state = "veryHidden"


def verify_protected_signature(wb, schemas: dict) -> None:
    """Raise TemplateIntegrityError if *wb*'s protected reference cells
    (HPL/LPL/Capping/Max score/row labels/...) don't match the signature that
    was embedded when the file was produced by `preprocess`."""
    if INTEGRITY_SHEET_NAME not in wb.sheetnames:
        raise TemplateIntegrityError(
            "This file is missing its integrity signature (no hidden "
            "'Integrity' sheet). It may not have been produced by "
            "'preprocess', or was edited by a tool that stripped hidden "
            "sheets. Please regenerate it by running 'preprocess' again."
        )
    stored_signature = wb[INTEGRITY_SHEET_NAME]["B1"].value
    actual_signature = compute_protected_signature(wb, schemas)
    if stored_signature != actual_signature:
        raise TemplateIntegrityError(
            "This file's reference data (e.g. HPL/LPL/Capping/Max score "
            "columns) does not match what 'preprocess' produced -- it "
            "appears to have been modified afterwards. Please re-run "
            "'preprocess' on the original template and only edit the grey "
            "input cells."
        )


def create_param_dict_from_input_parameters(input_parameters_df, col="Scenario"):
    """
    Creates a nested dictionary from input parameters dataframe.

    Structure:
    {
        <col>: {
            "InputParameter": "Value",
            ...
        }
    }
    """
    result = {}
    current_scenario = None

    for _, row in input_parameters_df.iterrows():
        # Update scenario if not NA
        if pd.notna(row.get(col)):
            current_scenario = str(row[col]).strip()
            if current_scenario not in result:
                result[current_scenario] = {}

        # Add input parameter and value
        if current_scenario:
            input_param = row.get("Input parameter")
            value = row.get("Value")
            if pd.notna(input_param):
                result[current_scenario][str(input_param)] = value

    return result


def _score_column_as_float(score: pd.Series) -> pd.Series:
    """The Score column as float64, reading the "-" placeholder ("not yet
    computed", see reset_computed_columns_to_dash) as unassessed NaN.

    The CLI path never sees the dash here because get_updated_dfs resets
    the whole Score column to NaN first, but DataFrame callers pass their
    Scenario Scores sheet straight into the update_scoring_sheet* helpers
    without that reset -- mapping "-" to NaN keeps both paths identical.
    Any other non-numeric string still raises, as before.
    """
    return score.mask(score == "-").astype("float64")


def update_scoring_sheet(
    df: pd.DataFrame, score_dict: dict, key_column: str
) -> pd.DataFrame:
    """
    Update the 'Score' column in the DataFrame based on the score_dict.
    For each key in score_dict, search for it in key_column and update 'Score'.
    """
    df = df.copy()
    # Ensure case-insensitive comparison
    df_key_lower = df[key_column].astype(str).str.lower()
    # Ensure Score column is float64 to avoid dtype incompatibility warnings
    df["Score"] = _score_column_as_float(df["Score"])
    for key, value in score_dict.items():
        key_lower = str(key).lower()
        mask = df_key_lower == key_lower
        df.loc[mask, "Score"] = value
    return df


def update_scoring_sheet_from_df(
    df: pd.DataFrame, score_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Update the 'Score' column in df based on matching 'Category' and 'Scenario' columns in score_df.
    For each row in score_df, find matching rows in df and update 'Score'.
    """
    df = df.copy()
    df[["Category"]] = df[["Category"]].ffill()
    df["Score"] = _score_column_as_float(df["Score"])
    # Ensure case-insensitive comparison for both columns
    df_cat_lower = df["Category"].astype(str).str.lower()
    df_scen_lower = df["Scenario"].astype(str).str.lower()
    score_df_cat_lower = score_df["Category"].astype(str).str.lower()
    score_df_scen_lower = score_df["Scenario"].astype(str).str.lower()
    for idx, row in score_df.iterrows():
        cat = score_df_cat_lower.iloc[idx]
        scen = score_df_scen_lower.iloc[idx]
        score = row.get("Score", None)
        mask = (df_cat_lower == cat) & (df_scen_lower == scen)
        if score is not None:
            df.loc[mask, "Score"] = score
    df = opposite_ffill(df, exclude_columns=df.columns.difference(["Category"]))
    return df


def update_score_sum_interval(
    df: pd.DataFrame, interval_df: pd.DataFrame, key_column: str, interval_column: str
) -> pd.DataFrame:
    """
    For each value in interval_df[interval_column], sum all 'Score' values in df[key_column] starting from that value
    until a NaN is encountered in df[key_column]. Update the corresponding row in df with the sum in 'Score'.
    """

    df["Score"] = _score_column_as_float(df["Score"])
    interval_scores = _score_column_as_float(interval_df["Score"])
    # For each value in interval_df[interval_column], sum 'Score' values in df from the first match until the next non-NaN value
    for interval_value in df[key_column].dropna().unique():
        # Find the first index in df where key_column == interval_value
        mask = interval_df[interval_column] == interval_value
        if not mask.any():
            continue
        start_idx = mask.idxmax()
        # Find the next index after start_idx where key_column is not NaN (i.e., next section start)
        next_idxs = interval_df.index[
            (interval_df.index > start_idx) & (interval_df[interval_column].notna())
        ]
        if len(next_idxs) > 0:
            end_idx = next_idxs[0]
            section = interval_scores.loc[start_idx : end_idx - 1]
        else:
            section = interval_scores.loc[start_idx:]
        # Any unassessed (NaN) Score in the section makes the whole rolled-up
        # Score unassessed too, rather than pandas' default skipna=True
        # silently treating it as a 0 contribution to the sum.
        score_sum = np.nan if section.isna().any() else section.sum()
        # Update df at the index of the unique value from the for loop
        unique_idx = df.index[df[key_column] == interval_value]
        if len(unique_idx) > 0:
            df.loc[unique_idx[0], "Score"] = score_sum
    return df


def is_empty_cell(cell_value):
    """
    Checks if a cell value is considered empty.

    Args:
        cell_value: The value of the cell to check.

    Returns:
        bool: True if the cell is empty, False otherwise.
    """
    if isinstance(cell_value, str) and cell_value.strip() == "":
        return True
    return (
        pd.isna(cell_value)
        or cell_value is None
        or cell_value == "None"
        or cell_value == "nan"
    )


def classify_pass_fail_value(cell_value) -> str:
    """Classify a "PASS"/"FAIL" verification cell into "pass", "fail", or
    "blank" (the OEM has not assessed this scenario yet). Blank must not be
    treated as "fail" -- an unassessed scenario should score as unassessed
    (NaN), not 0, so callers should branch on this before scoring.
    """
    if is_empty_cell(cell_value):
        return "blank"
    return "pass" if str(cell_value).strip().lower() == "pass" else "fail"


def classify_pass_fail_section(df, column: str = "Value") -> str:
    """Classify a whole section/sheet of PASS/FAIL verification cells into
    "blank" (any cell not yet assessed), "fail" (any cell is an explicit
    FAIL and none are blank), or "pass" (every cell is an explicit PASS).

    A missing/empty section (df is None, empty, or lacks *column*) is
    "blank" -- nothing has been assessed yet, so it must not silently read
    as a passing "all()" (vacuous truth) or as a failing gate.
    """
    if df is None or df.empty or column not in df.columns:
        return "blank"
    states = df[column].apply(classify_pass_fail_value)
    if (states == "blank").any():
        return "blank"
    if (states == "fail").any():
        return "fail"
    return "pass"


def set_number_precision(ws):
    """
    Set number precision for specific columns based on worksheet title.
    """
    # Define precision rules based on worksheet title (case-insensitive)
    title = ws.title.lower()
    # Map worksheet titles to column precision
    precision_map = {
        "scenario scores": {
            "score": "0.000",
            "max score": "0.000",
        },
        "category scores": {
            "score": "0.000",
            "max score": "0.000",
        },
        "test scores": {
            "score": "0.000",
            "max score": "0.000",
        },
        "stage subelement scores": {
            "score": "0.000",
            "max score": "0.000",
        },
        "stage element scores": {
            "score": "0.000",
            "max score": "0.000",
        },
        "stage scores": {
            "score": "0",
            "max score": "0",
        },
        "cp - dummy scores": {
            "score": "0.0000",
            "max score": "0.0000",
        },
        "cp - body region scores": {
            "score": "0.0000",
            "max score": "0.0000",
        },
    }
    # Default precision for columns if not matched above
    default_precision = {
        "inspection [%]": "0.00",
        "value": "0.00",
        "capping": "0.00",
        "body regionscore": "0.00",
        "modifiers": "0.00",
        "score": "0.0000",
        "max score": "0.0000",
    }

    # Get precision for this worksheet title, or use default
    sheet_precision = precision_map.get(title.lower(), default_precision)
    logger.debug(f"Setting number precision for worksheet '{ws.title}'")
    max_row, max_col = get_populated_sheet_bounds(ws)
    for col in ws.iter_cols(min_row=1, max_row=max_row, min_col=1, max_col=max_col):
        header = (
            col[0].value.lower()
            if col[0].value and isinstance(col[0].value, str)
            else ""
        )
        if header in sheet_precision:
            logger.debug(
                f"Processing column {col[0].column_letter} with header '{col[0].value}'"
            )
            fmt = sheet_precision[header]
            for cell in col[1:]:
                if isinstance(cell.value, (int, float)) and not is_empty_cell(
                    cell.value
                ):
                    cell.number_format = fmt
        # Criteria score precision 2
        elif "cp" in title and title not in [
            "cp - dummy scores",
            "cp - body region scores",
        ]:
            sheet_precision = {
                "score": "0.00",
                "max score": "0.00",
            }
            if header in sheet_precision:
                fmt = sheet_precision[header]
                for cell in col[1:]:
                    if isinstance(cell.value, (int, float)) and not is_empty_cell(
                        cell.value
                    ):
                        cell.number_format = fmt
        elif header in default_precision and header not in sheet_precision:
            fmt = default_precision[header]
            for cell in col[1:]:
                if isinstance(cell.value, (int, float)) and not is_empty_cell(
                    cell.value
                ):
                    cell.number_format = fmt


# Capitalize after " - " or " / "


def capitalize_special(title):
    # List of substrings to always capitalize fully
    always_upper = {
        "CP",  # crash protection
        "CA",  # crash avoidance
        "SD",  # safe driving
        "PC",  # post crash
        "FC",  # forward collision
        "LDC",  # lane departure collision
        "LSC",  # low speed collisions
        "ACC",  # adaptive cruise control
        "DE",  # driver engagement
        "DM",  # driver monitoring
        "OM",  # occupant monitoring
        "GVC",  # general vehicle control
        "VA",  # vehicle assessment
        "VRU",  # vulnerable road user
        "CCRs",  # Car to car rear stationary
        "CCR",  # Car to car rear
        "CMR",  # Car to motorcycle rear
        "CMRs",  # Car to motorcycle rear stationary
        "VUT",  # Vehicle Under Test
        "TTC",  # Time To Collision
        "LPL",  # Low Performance Limit
        "HPL",  # High Performance Limit
        "OEM",  # Original Equipment Manufacturer
        "ID",  # Documentation ID / Data ID (Documentation & Data sheets)
        "CCRm",
        "CCRb",
        "CCFhos",
        "CCFhol",
        "CCFtap",
        "CCCscp",
        "CMCscp",
        "CMRb",
        "CMFtap",
        "CBNA",
        "CBFA",
        "CBNAO",
        "CBTA",
        "CPTA",
        "CPLA",
        "CPLA day",
        "CPNA",
        "CPFA",
        "CPNCO",
        "CBLA",
        "ELK RE",
        "CC ELK On",
        "CC ELK OvU",
        "CC ELK OvI",
        "CC ELK Ov",
        "CM ELK On",
        "CM ELK OvU",
        "CM ELK OvI",
        "CM ELK Ov",
    }

    def cap(match):
        word = match.group(2)
        if word.upper() in always_upper:
            return match.group(1) + word.upper()
        return match.group(1) + word.capitalize()

    # Capitalize after " - " or " / "
    title = re.sub(r"( - | / )(\w+)", cap, title)
    # Replace all occurrences of the special words with their uppercase version
    for word in always_upper:
        title = re.sub(rf"\b{word}\b", word, title, flags=re.IGNORECASE)
    return title


def set_title_capitalization(ws):
    """
    Set title capitalization for header cells in the worksheet.
    Capitalize only the first word for titles in capitalized_titles,
    capitalize all words for titles in all_words_capital_titles.
    """
    capitalized_titles = [
        "Stage subelement",
        "Category",
        "Input parameter",
        "Stage element",
    ]
    all_words_capital_titles = ["Stage"]

    for cell in ws[1]:  # Assuming the first row contains headers
        if isinstance(cell.value, str):
            title = cell.value.strip()
            if title in capitalized_titles:
                cell.value = title.capitalize()
            elif title in all_words_capital_titles:
                cell.value = " ".join([w.capitalize() for w in title.split()])
            elif title == "OEM Prediction":
                continue
            else:
                cell.value = title.capitalize()
            # Ensure special capitalization rules are applied after general title case
            cell.value = capitalize_special(cell.value)


def reset_empty_row_background(ws):
    """
    If any row in the worksheet is completely empty, reset all cell backgrounds in that row to white,
    but preserve existing borders.
    """
    white_fill = PatternFill(
        start_color="FFFFFF", end_color="FFFFFF", fill_type="solid"
    )
    max_row, max_col = get_populated_sheet_bounds(ws)
    for row in ws.iter_rows(min_row=1, max_row=max_row, min_col=1, max_col=max_col):
        # Check if all cells in the row are empty
        if all(cell.value in [None, ""] for cell in row):
            for cell in row:
                # Preserve border, only update fill
                cell.fill = copy.copy(white_fill)
                # Preserve border by reassigning the original border
                cell.border = copy.copy(cell.border)


def get_populated_sheet_bounds(ws):
    max_row = 1
    max_col = 1

    for cell in ws._cells.values():
        if is_empty_cell(cell.value):
            continue
        max_row = max(max_row, cell.row)
        max_col = max(max_col, cell.column)

    for merged_range in ws.merged_cells.ranges:
        top_left_cell = ws.cell(merged_range.min_row, merged_range.min_col)
        if is_empty_cell(top_left_cell.value):
            continue
        max_row = max(max_row, merged_range.max_row)
        max_col = max(max_col, merged_range.max_col)

    return max_row, max_col


def set_header_by_row(ws, row_id, header_fill, header_font):
    """
    Sets the specified row as header by applying fill and font styles.

    Args:
        ws: The worksheet object.
        row_id: The row number (1-based index) to set as header.
        header_fill: The fill style to apply.
        header_font: The font style to apply.
    """
    for row in ws.iter_rows(
        min_row=row_id,
        max_row=row_id,
        min_col=1,
        max_col=ws.max_column,
    ):
        for cell in row:
            cell.fill = header_fill
            cell.font = header_font


VERIFICATION_COLUMN_ORDER = [
    "Scenario",
    "Function",
    "VUT speed",
    "Lateral velocity",
    "Target speed",
    "Impact location",
    "Turn",
    "Target direction",
    "Target approach",
    "Illumination",
    "Distance",
    "Gap",
    "Range",
    "Door selected",
    "Robustness layer",
    "Verification condition",
    "Test point",
    "OEM Prediction",
    "Expected value baseline",
    "Value baseline",
    "Expected value",
    "Robustness",
    "Value",
    "Colour",
]


def reorder_columns_by_header_order(
    df: pd.DataFrame, desired_order: list
) -> pd.DataFrame:
    """
    Reorder columns in the DataFrame so that only columns in desired_order appear,
    in the order specified. Columns not in desired_order are excluded.
    Returns a new DataFrame with reordered columns.
    """
    cols = [col for col in desired_order if col in df.columns]
    return df[cols]


def get_start_row(df: pd.DataFrame, value, col_idx: int = 0) -> int | None:
    """
    Return the first row index where the specified column matches value.

    If no row matches, also check whether the requested value matches a column
    header, and in that case return the first row index if the DataFrame is not
    empty.

    Args:
        df (pd.DataFrame): The DataFrame to inspect.
        value: The value to search for.
        col_idx (int): The column index (0-based) to use as reference.

    Returns:
        int | None: The first matching row index, or None if no match is found.
    """
    if col_idx < 0 or col_idx >= len(df.columns):
        return None

    column_name = df.columns[col_idx]
    if isinstance(column_name, str) and isinstance(value, str):
        if column_name.strip().lower() == value.strip().lower():
            return 1 if len(df) > 0 else None
    elif column_name == value:
        return 1 if len(df) > 0 else None

    for row_idx in range(len(df)):
        cell_value = df.iloc[row_idx, col_idx]
        if pd.isna(cell_value):
            continue
        if isinstance(cell_value, str) and isinstance(value, str):
            if cell_value.strip().lower() == value.strip().lower():
                return row_idx + 2
        elif cell_value == value:
            return row_idx + 2

    return None


def get_n_rows(df: pd.DataFrame, start_row: int, col_idx: int = 0) -> int:
    """
    Return the number of consecutive non-NaN rows starting at start_row,
    using the specified column as the reference.

    Args:
        df (pd.DataFrame): The DataFrame to inspect.
        start_row (int): The starting row index (0-based).
        col_idx (int): The column index (0-based) to use as reference.

    Returns:
        int: Number of consecutive rows where the specified column value is not NaN.
    """
    if start_row is None:
        return 0

    if start_row < 0 or start_row >= len(df):
        return 0

    n_rows = 0
    for row_idx in range(start_row, len(df)):
        if pd.isna(df.iloc[row_idx, col_idx]):
            break
        n_rows += 1

    return n_rows


def get_n_cols(df: pd.DataFrame, row_idx: int, start_col: int) -> int:
    """
    Return the number of consecutive non-NaN columns starting at start_col,
    using the specified row as the reference.

    Args:
        df (pd.DataFrame): The DataFrame to inspect.
        row_idx (int): The row index (0-based) to use as reference.
        start_col (int): The starting column index (0-based).

    Returns:
        int: Number of consecutive columns where the specified row value is not NaN.
    """
    if (
        row_idx < 0
        or row_idx >= len(df)
        or start_col < 0
        or start_col >= len(df.columns)
    ):
        return 0

    n_cols = 0
    for col_idx in range(start_col, len(df.columns)):
        if pd.isna(df.iloc[row_idx, col_idx]):
            break
        n_cols += 1

    return n_cols


def format_test_point(row: int, col: int) -> str:
    """Return the canonical serialized representation for a test point."""
    return f"({int(row)}, {int(col)})"


def validate_input_selected_points(
    payload,
    *,
    context: str = "",
    allowed_top_keys: set | None = None,
    nested_group_values: bool = False,
) -> None:
    """Validate the structure of an ``input_selected_points`` payload.

    Expected shape by default: ``dict[str, list[dict]]`` — a mapping from group
    name to a list of attribute dicts. If ``nested_group_values`` is true, the
    expected shape becomes ``dict[str, dict[str, list[dict]]]``. ``None`` is
    always accepted (means: use protocol-driven random selection).

    Args:
        payload: The value to validate.
        context: A short label included in error messages (e.g. ``"ACC"``,
            ``"DM"``).
        allowed_top_keys: If given, the top-level keys must be a subset of this
            set. Used for the VRU payload that only allows
            ``{"Headform", "Legform"}``.
        nested_group_values: If true, each top-level value must be a dict whose
            values are ``list[dict]``.

    Raises:
        TypeError: If the payload has the wrong type at any level.
        ValueError: If a top-level key is not in *allowed_top_keys*.
    """
    if payload is None:
        return

    prefix = f"[{context}] " if context else ""

    if not isinstance(payload, dict):
        raise TypeError(
            f"{prefix}input_selected_points must be a dict, got {type(payload).__name__}."
        )

    def _validate_points_list(points, group_path: str) -> None:
        if not isinstance(points, list):
            raise TypeError(
                f"{prefix}input_selected_points{group_path} must be a list, "
                f"got {type(points).__name__}."
            )
        for idx, item in enumerate(points):
            if not isinstance(item, dict):
                raise TypeError(
                    f"{prefix}input_selected_points{group_path}[{idx}] must be a "
                    f"dict of attributes, got {type(item).__name__}."
                )

    for group_name, group_value in payload.items():
        if allowed_top_keys is not None and group_name not in allowed_top_keys:
            raise ValueError(
                f"{prefix}Unexpected key {group_name!r} in input_selected_points. "
                f"Allowed keys: {sorted(allowed_top_keys)}."
            )
        if nested_group_values:
            if not isinstance(group_value, dict):
                raise TypeError(
                    f"{prefix}input_selected_points[{group_name!r}] must be a dict, "
                    f"got {type(group_value).__name__}."
                )
            for nested_name, nested_value in group_value.items():
                _validate_points_list(
                    nested_value,
                    f"[{group_name!r}][{nested_name!r}]",
                )
            continue

        _validate_points_list(group_value, f"[{group_name!r}]")


def parse_test_point(value) -> Tuple[int, int]:
    """Parse a serialized test point in the format ``(row, col)`` or ``row,col``."""
    if pd.isna(value):
        raise ValueError("Test point value cannot be empty.")

    value_str = str(value).strip()
    match = re.fullmatch(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", value_str)
    if not match:
        match = re.fullmatch(r"(-?\d+)\s*,\s*(-?\d+)", value_str)
    if not match:
        raise ValueError(
            f"Invalid test point format: {value!r}. Expected '(row, col)' or 'row,col'."
        )

    return int(match.group(1)), int(match.group(2))


def extract_embedded_table(
    df: pd.DataFrame, required_headers: list[str]
) -> pd.DataFrame:
    """Extract a table whose header is stored as a row inside a worksheet dataframe."""
    if df is None or df.empty:
        return pd.DataFrame()

    normalized_headers = {header.strip().lower() for header in required_headers}
    header_row_idx = None
    header_values = None

    for row_idx, row in df.iterrows():
        row_values = [
            "" if pd.isna(value) else str(value).strip() for value in row.tolist()
        ]
        row_header_set = {value.lower() for value in row_values if value}
        if normalized_headers.issubset(row_header_set):
            header_row_idx = row_idx
            header_values = row_values
            break

    if header_row_idx is None or header_values is None:
        return pd.DataFrame()

    extracted_df = pd.DataFrame(
        df.iloc[header_row_idx + 1 :].values,
        columns=header_values,
    )
    if extracted_df.empty:
        return extracted_df

    return extracted_df.dropna(how="all").reset_index(drop=True)


def build_safe_driving_prediction_mark_coordinates(
    dfs: dict[str, pd.DataFrame],
) -> dict[str, list[Tuple[int, int]]]:
    from euroncap_rating_2026.safe_driving import data_model as safe_driving_data_model

    coordinates_by_sheet: dict[str, list[Tuple[int, int]]] = {}

    dm_verification_df = extract_embedded_table(
        dfs.get("DE - DM verif.", pd.DataFrame()),
        ["Scenario", "Test point"],
    )
    if not dm_verification_df.empty and "Test point" in dm_verification_df.columns:
        dm_coordinates = []
        for _, row in dm_verification_df.iterrows():
            test_point = row.get("Test point")
            if pd.isna(test_point) or str(test_point).strip() == "":
                continue
            point_row, point_col = parse_test_point(test_point)
            dm_coordinates.append((point_row, point_col))

        if dm_coordinates:
            coordinates_by_sheet["DE - DM pred."] = dm_coordinates

    acc_verification_df = extract_embedded_table(
        dfs.get("VA - ACC verif.", pd.DataFrame()),
        ["Scenario", "Test point"],
    )
    if not acc_verification_df.empty and "Test point" in acc_verification_df.columns:
        acc_coordinates = []
        for _, row in acc_verification_df.iterrows():
            scenario = row.get("Scenario")
            test_point = row.get("Test point")
            if pd.isna(test_point) or str(test_point).strip() == "":
                continue
            scenario_name = "" if pd.isna(scenario) else str(scenario).strip()
            if scenario_name not in safe_driving_data_model.ACC_MATRIX_INDICES:
                continue

            point_row, point_col = parse_test_point(test_point)
            matrix_indices = safe_driving_data_model.ACC_MATRIX_INDICES[scenario_name]
            acc_coordinates.append(
                (
                    point_row + matrix_indices["start_row"],
                    point_col + matrix_indices["start_col"],
                )
            )

        if acc_coordinates:
            coordinates_by_sheet["VA - ACC pred."] = acc_coordinates

    return coordinates_by_sheet


def build_crash_protection_prediction_mark_coordinates(
    dfs: dict[str, pd.DataFrame],
) -> dict[str, dict[Tuple[int, int], str]]:
    from euroncap_rating_2026.crash_protection import vru_processing

    prediction_df = dfs.get("CP - VRU Prediction", pd.DataFrame())
    if prediction_df.empty or len(prediction_df) < 2 or prediction_df.shape[1] < 4:
        return {}

    prediction_col_lookup = prediction_df.iloc[1]
    prediction_row_lookup = prediction_df.iloc[:, 3]
    prediction_col_lookup_numeric = pd.to_numeric(
        prediction_col_lookup, errors="coerce"
    )
    prediction_row_lookup_numeric = pd.to_numeric(
        prediction_row_lookup, errors="coerce"
    )
    # Maps (excel_row, excel_col) → display label ("X" for headform, "T"/"ST"/"SA" for blue legform)
    prediction_coordinates: dict[Tuple[int, int], str] = {}

    head_impact_df = dfs.get("CP - VRU Head Impact", pd.DataFrame())
    if "Test point" in head_impact_df.columns:
        for test_point in head_impact_df["Test point"].dropna().unique():
            point_row, point_col = parse_test_point(test_point)
            prediction_col = next(
                (
                    idx
                    for idx, value in enumerate(prediction_col_lookup_numeric)
                    if pd.notna(value) and int(value) == int(point_col)
                ),
                None,
            )
            prediction_row = next(
                (
                    idx
                    for idx, value in enumerate(prediction_row_lookup_numeric)
                    if pd.notna(value) and int(value) == int(point_row)
                ),
                None,
            )
            if prediction_row is None or prediction_col is None:
                continue
            prediction_coordinates[(prediction_row + 2, prediction_col + 1)] = "X"

    pelvis_leg_impact_df = dfs.get("CP - VRU Pelvis & Leg Impact", pd.DataFrame())
    # Build test-point → display label: blue points show "T"/"ST"/"SA", others show "X"
    tp_to_label: dict[str, str] = {}
    if (
        "Test point" in pelvis_leg_impact_df.columns
        and "OEM Prediction" in pelvis_leg_impact_df.columns
    ):
        for _, df_row in pelvis_leg_impact_df.dropna(subset=["Test point"]).iterrows():
            tp_str = str(df_row["Test point"])
            if tp_str in tp_to_label:
                continue
            pred = df_row.get("OEM Prediction", "")
            pred_str = str(pred).strip().lower() if pd.notna(pred) else ""
            tp_to_label[tp_str] = (
                pred_str.upper() if pred_str in ("t", "st", "sa") else "X"
            )

    if "Test point" in pelvis_leg_impact_df.columns:
        for test_point in pelvis_leg_impact_df["Test point"].dropna().unique():
            point_row, point_col = parse_test_point(test_point)
            prediction_col = next(
                (
                    idx
                    for idx, value in enumerate(prediction_col_lookup_numeric)
                    if pd.notna(value) and int(value) == int(point_col)
                ),
                None,
            )
            if prediction_col is None:
                continue

            # Preserve the original legform remap used by preprocess/report writing:
            # matrix-local legform rows are offset into the VRU prediction sheet block.
            prediction_row = point_row + vru_processing.LEGFORMS_START_ROW_INDEX + 2
            label = tp_to_label.get(str(test_point), "X")
            prediction_coordinates[(prediction_row + 2, prediction_col + 1)] = label

    if not prediction_coordinates:
        return {}
    return {"CP - VRU Prediction": prediction_coordinates}


def build_crash_avoidance_prediction_mark_coordinates(
    dfs: dict[str, pd.DataFrame],
) -> dict[str, list[Tuple[int, int]]]:
    from euroncap_rating_2026.crash_avoidance import matrix_processing

    coordinates_by_sheet: dict[str, list[Tuple[int, int]]] = {}

    for sheet_name, verification_sheet_df in dfs.items():
        if not sheet_name.endswith(" verif."):
            continue

        verification_df = extract_embedded_table(
            verification_sheet_df,
            ["Scenario", "Test point"],
        )
        if verification_df.empty or "Test point" not in verification_df.columns:
            continue

        prediction_sheet_name = sheet_name.replace(" verif.", " pred.")
        prediction_df = dfs.get(prediction_sheet_name, pd.DataFrame())
        if prediction_df.empty:
            continue

        prediction_coordinates: set[Tuple[int, int]] = set()
        for _, row in verification_df.iterrows():
            scenario = row.get("Scenario")
            test_point = row.get("Test point")
            if pd.isna(scenario) or pd.isna(test_point):
                continue

            scenario_name = str(scenario).strip()
            if scenario_name == "" or str(test_point).strip() == "":
                continue

            matrix_indices = matrix_processing.get_matrix_indices(
                prediction_df, scenario_name
            )
            if not matrix_indices:
                continue

            point_row, point_col = parse_test_point(test_point)
            prediction_coordinates.add(
                (
                    point_row + matrix_indices["start_row"] + 2,
                    point_col + matrix_indices["start_col"] + 1,
                )
            )

        if prediction_coordinates:
            coordinates_by_sheet[prediction_sheet_name] = sorted(prediction_coordinates)

    return coordinates_by_sheet
