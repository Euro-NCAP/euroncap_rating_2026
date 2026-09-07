# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import sys
import pandas as pd
import numpy as np
from collections import Counter
import logging
import openpyxl
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple

from euroncap_rating_2026.crash_protection.load_case import LoadCase
from euroncap_rating_2026.crash_protection.seat import Seat
from euroncap_rating_2026.crash_protection.dummy import Dummy
from euroncap_rating_2026.crash_protection.body_region import BodyRegion
from euroncap_rating_2026.crash_protection.criteria import Criteria, CriteriaType
from euroncap_rating_2026 import common
from euroncap_rating_2026.numeric import round_half_up

logger = logging.getLogger(__name__)

HEADFORMS_START_ROW_INDEX = 0
HEADFORMS_END_ROW_INDEX = 21
LEGFORMS_START_ROW_INDEX = 23
LEGFORMS_END_ROW_INDEX = 28
# The VRU matrices only span columns E:Y (21 columns); columns beyond that hold
# spacer/legend cells (e.g. legend text) that must not be treated as matrix data.
VRU_MATRIX_COL_END_INDEX = 25


def headform_body_region(row):
    """Map a headform point row (WAD band) to its body region name.

    Point rows count from the bottom of the grid: 0 = WAD 1000 up to
    18 = WAD 2800. Child spans WAD 1000-1500, Adult 1600-2100 and
    Cyclist 2200-2800.
    """
    if 18 >= row >= 12:
        return "Cyclist"
    elif 11 >= row >= 6:
        return "Adult"
    elif 5 >= row >= 0:
        return "Child"
    return None


class VruPredictionColor(str, Enum):
    BLUE = "blue"
    BROWN = "brown"
    D_RED = "d red"
    GREEN = "green"
    GREEN_20 = "green-20"
    GREEN_30 = "green-30"
    GREEN_40 = "green-40"
    GREY = "grey"
    ORANGE = "orange"
    RED = "red"
    YELLOW = "yellow"
    BLUE_T = "t"
    BLUE_ST = "st"
    BLUE_SA = "sa"
    D_GREEN = "d green"


VRU_PREDICTION_COLOR_MAP = {
    VruPredictionColor.BLUE: (0 / 255, 102 / 255, 204 / 255),
    VruPredictionColor.BLUE_T: (0 / 255, 102 / 255, 204 / 255),
    VruPredictionColor.BLUE_ST: (0 / 255, 102 / 255, 204 / 255),
    VruPredictionColor.BLUE_SA: (0 / 255, 102 / 255, 204 / 255),
    VruPredictionColor.BROWN: (150 / 255, 75 / 255, 0 / 255),
    VruPredictionColor.D_RED: (139 / 255, 0 / 255, 0 / 255),  # dark red
    VruPredictionColor.GREEN: (0 / 255, 153 / 255, 51 / 255),
    VruPredictionColor.GREEN_20: (
        204 / 255,
        255 / 255,
        221 / 255,
    ),  # light green (20% green)
    VruPredictionColor.GREEN_30: (
        153 / 255,
        255 / 255,
        187 / 255,
    ),  # lighter green (30% green)
    VruPredictionColor.GREEN_40: (
        102 / 255,
        255 / 255,
        153 / 255,
    ),  # even lighter green (40% green)
    VruPredictionColor.GREY: (128 / 255, 128 / 255, 128 / 255),
    VruPredictionColor.ORANGE: (255 / 255, 153 / 255, 51 / 255),
    VruPredictionColor.RED: (255 / 255, 51 / 255, 51 / 255),
    VruPredictionColor.YELLOW: (255 / 255, 221 / 255, 51 / 255),
    VruPredictionColor.D_GREEN: (0 / 255, 102 / 255, 0 / 255),  # dark green
}


@dataclass
class VruTestPoint:
    """
    Represents a VRU test point with its coordinates and color.
    """

    row: int
    col: int
    color: VruPredictionColor = None
    loadcase_name: str = None
    attributes: dict = field(default_factory=dict)

    @property
    def body_region(self):
        return headform_body_region(self.row)


@dataclass
class LegformTestPoint:
    """
    Represents a legform test point with its coordinates and color.
    """

    row: int
    col: int
    color: VruPredictionColor = None
    loadcase_name: str = None
    attributes: dict = field(default_factory=dict)


@dataclass
class VruScore:
    """
    Represents a VRU score with its name and value.
    """

    name: str
    predicted_score: float = 0.0
    blue_points: float = 0.0
    a_pillar: float = 0.0
    max_points: float = 0.0

    COLOR_WEIGHTS = {
        VruPredictionColor.GREEN: 1.0,
        VruPredictionColor.D_GREEN: 1.0,
        VruPredictionColor.YELLOW: 0.75,
        VruPredictionColor.ORANGE: 0.5,
        VruPredictionColor.BROWN: 0.25,
    }

    def compute_blue_points(self, loadcases):
        blue_scores = 0
        for load_case in loadcases:
            for seat in load_case.seats:
                dummy = seat.dummy
                for body_region in dummy.body_region_list:
                    if body_region.name == self.name:
                        for criteria in body_region._criteria:
                            if (
                                criteria.prediction is not None
                                and criteria.prediction.lower()
                                == VruPredictionColor.BLUE.value
                            ):
                                logger.debug(
                                    f"[CP - VRU Head Impact] Found blue prediction in {body_region}: {criteria}"
                                )
                                blue_scores += criteria.get_score()

        self.blue_points = blue_scores / 100.0

    def compute_a_pillar(self, loadcases):
        # Implement weighted sum for Green-40, Green-30, Green-20
        green_40_sum = 0
        green_30_sum = 0
        green_20_sum = 0
        for load_case in loadcases:
            for seat in load_case.seats:
                dummy = seat.dummy
                for body_region in dummy.body_region_list:
                    if body_region.name == self.name:
                        for criteria in body_region._criteria:
                            if criteria.prediction is None:
                                continue
                            pred = criteria.prediction.lower()
                            score = criteria.get_score()
                            if pred == "green-40":
                                green_40_sum += score
                            elif pred == "green-30":
                                green_30_sum += score
                            elif pred == "green-20":
                                green_20_sum += score
        weighted_sum = green_40_sum * 3 + green_30_sum * 2 + green_20_sum * 1
        self.a_pillar = weighted_sum / 100.0


def loadcases_to_rows(load_cases):
    """
    Converts a list of load cases to a list of rows for DataFrame creation.

    Args:
    load_cases (list): A list of LoadCase objects.

    Returns:
    list: A list of lists, each representing a row.
    """
    rows = []
    for load_case in load_cases:
        for seat in load_case.seats:
            dummy = seat.dummy
            first_row = True
            for body_region in dummy.body_region_list:
                for criteria in body_region._criteria:
                    if first_row:
                        row = [
                            load_case.name,
                            seat.name,
                            dummy.name,
                        ]
                        first_row = False
                    else:
                        row = [None, None, None]
                    row += [
                        body_region.name,
                        criteria.test_point,
                        criteria.name,
                        criteria.hpl,
                        criteria.lpl,
                        criteria.capping_value,
                        criteria.prediction,
                        criteria.value,
                    ]
                    rows.append(row)
    # Remove duplicated content in col0 (Loadcase), skip None/empty cells
    last_non_nan = None
    for idx in range(len(rows)):
        if rows[idx][0] and pd.notna(rows[idx][0]):
            if rows[idx][0] == last_non_nan:
                rows[idx][0] = None
            else:
                last_non_nan = rows[idx][0]
        elif last_non_nan is not None:
            rows[idx][0] = None

    # Remove duplicated content in col3 (Body region), use None so pd.isna() works in dummy.py
    last_non_nan_col3 = None
    for idx in range(len(rows)):
        if rows[idx][3] and pd.notna(rows[idx][3]):
            if rows[idx][3] == last_non_nan_col3:
                rows[idx][3] = None
            else:
                last_non_nan_col3 = rows[idx][3]
        elif last_non_nan_col3 is not None:
            rows[idx][3] = None
    return rows


def pretty_print_loadcases(loadcases):
    logger.info("pretty_print_loadcases:")
    for lc in loadcases:
        logger.info(f"LoadCase: {lc.name}")
        for seat in lc.seats:
            logger.info(f"  Seat: {seat.name}")
            dummy = seat.dummy
            logger.info(f"    Dummy: {dummy.name}")
            for body_region in getattr(dummy, "body_region_list", []):
                logger.info(f"      BodyRegion: {body_region.name}")
                for criteria in getattr(body_region, "_criteria", []):
                    logger.info(
                        f"        Criteria: {criteria.name}, HPL: {getattr(criteria, 'hpl', None)}, "
                        f"LPL: {getattr(criteria, 'lpl', None)}, "
                        f"Capping: {getattr(criteria, 'capping_value', None)}, "
                        f"Prediction: {getattr(criteria, 'prediction', None)}, "
                        f"Value: {getattr(criteria, 'value', None)}, "
                        f"Score: {getattr(criteria, 'score', None)}, "
                        f"Color: {getattr(criteria, 'color', None)}, "
                        f"Test point: {getattr(criteria, 'test_point', None)}"
                    )


def get_vru_points(vru_prediction_df, vru_prediction_ws):
    """
    Extract VRU (Vulnerable Road User) prediction test points from Excel cell colors.
    Parameters:
        vru_prediction_df (pd.DataFrame): DataFrame containing VRU prediction data.
        vru_prediction_ws (openpyxl.worksheet.worksheet.Worksheet): Worksheet object representing the Excel sheet.
    Returns:
        list[VruTestPoint]: List of VruTestPoint objects, each representing a test point with row, column, and color information
        extracted from the Excel sheet.
    The function processes the VRU prediction DataFrame and corresponding worksheet to extract test points,
    using the cell fill color to determine the color attribute for each point.
    """

    if not vru_prediction_df.empty:
        # Use the same logic as in process_headform_test to extract all cells
        headforms_matrix = vru_prediction_df.iloc[
            HEADFORMS_START_ROW_INDEX:HEADFORMS_END_ROW_INDEX, :
        ].reset_index(drop=True)
        headforms_matrix = headforms_matrix.iloc[2:, 4:].astype(str).values.tolist()
        for i in range(len(headforms_matrix)):
            for j in range(len(headforms_matrix[0])):

                cell = vru_prediction_ws.cell(
                    row=HEADFORMS_START_ROW_INDEX + 3 + i + 1, column=5 + j
                )
                fill = cell.fill
                rgb_tuple = None
                if fill and fill.fgColor is not None:
                    fg = fill.fgColor
                    if hasattr(fg, "rgb") and fg.rgb is not None:
                        rgb = str(fg.rgb).strip()  # e.g., 'FF00FF00'
                        if len(rgb) == 8:  # ARGB
                            rgb_tuple = tuple(
                                int(rgb[k : k + 2], 16) for k in (2, 4, 6)
                            )
                        elif len(rgb) == 6:  # RGB
                            rgb_tuple = tuple(
                                int(rgb[k : k + 2], 16) for k in (0, 2, 4)
                            )
                    elif hasattr(fg, "theme") or hasattr(fg, "indexed"):
                        # Handle theme/indexed colors if needed
                        pass
                logger.debug(
                    f"Cell ({HEADFORMS_START_ROW_INDEX + 3 + i + 1}, {5 + j}): RGB: {rgb_tuple}"
                )
                # If rgb_tuple is available, try to map it back to a VruPredictionColor using VRU_PREDICTION_COLOR_MAP
                if rgb_tuple is not None:
                    rgb_normalized = tuple(x / 255 for x in rgb_tuple)
                    # Find the closest color in VRU_PREDICTION_COLOR_MAP
                    for color_enum, rgb_val in VRU_PREDICTION_COLOR_MAP.items():
                        if all(
                            abs(a - b) < 1e-3 for a, b in zip(rgb_normalized, rgb_val)
                        ):
                            enum_val = color_enum
                            break
                    else:
                        enum_val = VruPredictionColor.GREY
                else:
                    enum_val = VruPredictionColor.GREY

                headforms_matrix[i][j] = enum_val
        return [
            VruTestPoint(
                row=18 - i,
                col=10 - j,
                color=headforms_matrix[i][j],
            )
            for i in range(len(headforms_matrix))
            for j in range(len(headforms_matrix[0]))
        ]
    else:
        return []


def load_vru_prediction_points(vru_prediction_points_df: pd.DataFrame) -> list:
    """
    Load VRU prediction points from a DataFrame.

    Args:
        vru_prediction_points_df (pd.DataFrame): DataFrame containing VRU prediction points.

    Returns:
        list: A list of VruTestPoint objects.
    """
    vru_points = []
    for _, row in vru_prediction_points_df.iterrows():
        vru_point = VruTestPoint(
            row=row.get("row"),
            col=row.get("col"),
            color=row.get("color"),
            loadcase_name=row.get("loadcase_name"),
        )
        vru_points.append(vru_point)
    return vru_points


def vru_points_to_df(vru_points_list):
    """
    Convert a list of VruTestPoint objects to a DataFrame.

    Args:
        vru_points_list (list): List of VruTestPoint objects.

    Returns:
        pd.DataFrame: DataFrame with columns ['row', 'col', 'loadcase_name', 'color', 'body_region'].
    """
    vru_points_df = pd.DataFrame(
        [
            {
                "row": point.row,
                "col": point.col,
                "loadcase_name": point.loadcase_name,
                "color": point.color,
                **(
                    {"body_region": point.body_region}
                    if hasattr(point, "body_region") and point.body_region is not None
                    else {}
                ),
            }
            for point in vru_points_list
        ]
    )
    return vru_points_df


def df_to_vru_points(vru_points_df, legform=False):
    """
    Convert a DataFrame to a list of VruTestPoint objects.

    Args:
        vru_points_df (pd.DataFrame): DataFrame with columns ['row', 'col', 'loadcase_name', 'color', 'body_region'].

    Returns:
        list: List of VruTestPoint objects.
    """
    vru_points_list = []
    for _, row in vru_points_df.iterrows():
        if legform:
            vru_point = LegformTestPoint(
                row=row.get("row"),
                col=row.get("col"),
                color=row.get("color"),
                loadcase_name=row.get("loadcase_name"),
            )
        else:
            vru_point = VruTestPoint(
                row=row.get("row"),
                col=row.get("col"),
                color=row.get("color"),
                loadcase_name=row.get("loadcase_name"),
            )
        vru_points_list.append(vru_point)
    return vru_points_list


def _attributes_match(candidate_attrs: dict, input_attrs: dict) -> bool:
    for key, input_val in input_attrs.items():
        if key not in candidate_attrs:
            continue
        if str(candidate_attrs[key]).strip() != str(input_val).strip():
            return False
    return True


def select_vru_points_from_input(candidate_points, input_selected_points):
    selected_points = []

    if input_selected_points is None:
        return selected_points

    grouped_candidates = {}
    for point in candidate_points:
        grouped_candidates.setdefault(point.loadcase_name, []).append(point)

    for group_name, requested_points in input_selected_points.items():
        for input_attrs in requested_points:
            matched_point = None
            for candidate in grouped_candidates.get(group_name, []):
                if _attributes_match(candidate.attributes, input_attrs):
                    matched_point = candidate
                    break

            if matched_point is None:
                logger.warning(
                    "[VRU %s] No candidate point found matching attributes %s. Skipping.",
                    group_name,
                    input_attrs,
                )
                continue

            selected_points.append(matched_point)

    return selected_points


def get_headform_matrix(vru_prediction_df):
    """Extract and convert the headform matrix from the VRU prediction
    DataFrame, mirroring get_legform_matrix. Factored out of
    extract_headform_candidate_points/process_headform_test (which
    duplicate this slicing inline) so a caller that only needs the raw
    color grid -- e.g. consistency.find_prediction_inconsistencies'
    no_prediction_provided check -- doesn't have to re-slice it a third
    way. Existing call sites are left as they are.

    Unlike those two, this bounds the column slice at
    VRU_MATRIX_COL_END_INDEX (the headform matrix has the same E:Y width as
    the legform one) instead of leaving it open-ended: an open ``4:`` slice
    picks up thousands of blank trailing columns on a real workbook (Excel
    inflates a sheet's reported max column from formatting alone), which
    costs nothing when this runs once per upload but adds up fast for a
    check called on every find_prediction_inconsistencies invocation.

    Returns:
        list: Headform matrix with VruPredictionColor enum values.
    """
    if vru_prediction_df.empty:
        return []

    headforms_matrix = vru_prediction_df.iloc[
        HEADFORMS_START_ROW_INDEX:HEADFORMS_END_ROW_INDEX, :
    ].reset_index(drop=True)
    headforms_matrix = (
        headforms_matrix.iloc[2:, 4:VRU_MATRIX_COL_END_INDEX]
        .astype(str)
        .values.tolist()
    )

    for i in range(len(headforms_matrix)):
        for j in range(len(headforms_matrix[0])):
            color_str = headforms_matrix[i][j].lower()
            headforms_matrix[i][j] = next(
                (k for k in VruPredictionColor if k.value.lower() == color_str),
                VruPredictionColor.GREY,
            )
    return headforms_matrix


def extract_headform_candidate_points(vru_prediction_df):
    if vru_prediction_df.empty:
        logger.error("The input DataFrame is empty.")
        return []

    headforms_matrix = vru_prediction_df.iloc[
        HEADFORMS_START_ROW_INDEX:HEADFORMS_END_ROW_INDEX, :
    ].reset_index(drop=True)
    headforms_matrix = headforms_matrix.iloc[2:, 4:].astype(str).values.tolist()

    candidate_points = []
    for i in range(len(headforms_matrix)):
        for j in range(len(headforms_matrix[0])):
            color_str = headforms_matrix[i][j].lower()
            color = next(
                (k for k in VruPredictionColor if k.value.lower() == color_str),
                VruPredictionColor.GREY,
            )
            if color in (
                VruPredictionColor.GREY,
                VruPredictionColor.D_RED,
                VruPredictionColor.D_GREEN,
            ):
                continue

            point = VruTestPoint(row=18 - i, col=10 - j, color=color)
            point.loadcase_name = (
                "Headform Apillar"
                if color
                in (
                    VruPredictionColor.GREEN_40,
                    VruPredictionColor.GREEN_30,
                    VruPredictionColor.GREEN_20,
                )
                else "Headform"
            )
            point.attributes = {
                "Loadcase": point.loadcase_name,
                "Body region": point.body_region,
                "Color": point.color.value,
                "row": point.row,
                "col": point.col,
            }
            candidate_points.append(point)

    return candidate_points


def extract_legform_candidate_points(vru_prediction_df):
    if vru_prediction_df.empty:
        logger.error("The input DataFrame is empty.")
        return []

    legform_matrix = get_legform_matrix(vru_prediction_df)
    candidate_points = []

    for row in range(len(legform_matrix)):
        for col in range(len(legform_matrix[row])):
            color = legform_matrix[row][col]
            if color == VruPredictionColor.GREY:
                continue

            point = LegformTestPoint(row=row, col=10 - col, color=color)
            if row == 0:
                point.loadcase_name = "Upper Leg"
                body_region = "Pelvis"
            elif row == 1:
                point.loadcase_name = "Lower Leg"
                body_region = "Femur"
            else:
                point.loadcase_name = "Lower Leg"
                body_region = "Knee/Tibia"

            point.attributes = {
                "Loadcase": point.loadcase_name,
                "Body region": body_region,
                "Color": point.color.value,
                "row": point.row,
                "col": point.col,
            }
            candidate_points.append(point)

    return candidate_points


def get_vru_point_df(input_file: str, dfs_dict=None) -> Tuple[str, pd.DataFrame]:
    """
    Load VRU (Vulnerable Road User) points from an Excel file.
    This function reads the specified Excel file, extracts the "CP - VRU Prediction" worksheet,
    and processes its contents to generate a DataFrame of VRU test points.
    Args:
        input_file (str): Path to the Excel file containing VRU prediction data.
        dfs_dict (dict, optional): Already-read dataframes of input_file (as returned
            by common.read_excel_file_to_dfs), to avoid parsing the workbook a second
            time. When None, the file is read here.
    Returns:
        Tuple[str, pd.DataFrame]: A tuple containing the sheet name ("CP - VRU Prediction Points")
        and a DataFrame with columns ['row', 'col', 'loadcase_name', 'color', 'body_region']
        representing the VRU test points.
    """

    if dfs_dict is None:
        dfs_dict = common.read_excel_file_to_dfs(input_file)
    wb = openpyxl.load_workbook(input_file, data_only=False)

    vru_prediction_ws = wb["CP - VRU Prediction"]
    vru_prediction_df = dfs_dict["CP - VRU Prediction"]

    vru_points_list = get_vru_points(vru_prediction_df, vru_prediction_ws)

    vru_points_df = vru_points_to_df(vru_points_list)
    return "CP - VRU Prediction Points", vru_points_df


def generate_vru_head_impact_df(headform_loadcases):
    """
    Generate DataFrame for VRU Head Impact sheet from headform loadcases.

    Args:
        headform_loadcases (list): List of LoadCase objects for headform tests.

    Returns:
        pd.DataFrame: DataFrame for CP - VRU Head Impact sheet.
    """
    df_rows = loadcases_to_rows(headform_loadcases)
    df = pd.DataFrame(
        df_rows,
        columns=[
            "Loadcase",
            "Seat position",
            "Dummy",
            "Body region",
            "Test point",
            "Criteria",
            "HPL",
            "LPL",
            "Capping",
            "OEM Prediction",
            "Value",
        ],
    )
    return df


def generate_vru_pelvis_leg_impact_df(legform_loadcases):
    """
    Generate DataFrame for VRU Pelvis & Leg Impact sheet from legform loadcases.

    Args:
        legform_loadcases (list): List of LoadCase objects for legform tests.

    Returns:
        pd.DataFrame: DataFrame for CP - VRU Pelvis & Leg Impact sheet.
    """
    df_rows = loadcases_to_rows(legform_loadcases)
    df = pd.DataFrame(
        df_rows,
        columns=[
            "Loadcase",
            "Seat position",
            "Dummy",
            "Body region",
            "Test point",
            "Criteria",
            "HPL",
            "LPL",
            "Capping",
            "OEM Prediction",
            "Value",
        ],
    )
    return df


def generate_vru_df(loadcase_dict):
    """
    Generate DataFrames for VRU sheets from loadcase dictionary.

    Args:
        loadcase_dict (dict): Dictionary mapping sheet names to lists of LoadCase objects.

    Returns:
        dict: Dictionary mapping sheet names to their corresponding DataFrames.
    """
    df_dict = {}
    df_dict["CP - VRU Head Impact"] = generate_vru_head_impact_df(
        loadcase_dict["CP - VRU Head Impact"]
    )
    df_dict["CP - VRU Pelvis & Leg Impact"] = generate_vru_pelvis_leg_impact_df(
        loadcase_dict["CP - VRU Pelvis & Leg Impact"]
    )
    return df_dict


def _parse_test_point(tp_str):
    """Parse '(row, col)' string into (row, col) integers, or (None, None) on failure."""
    if not tp_str:
        return None, None
    stripped = str(tp_str).strip().strip("()")
    parts = stripped.split(",")
    if len(parts) != 2:
        return None, None
    try:
        return int(parts[0].strip()), int(parts[1].strip())
    except ValueError:
        return None, None


def _resolve_legform_blue_values(legform_loadcases):
    """Infer missing Values for blue (T/ST/SA) legform criteria before scoring.

    Returns the number of warnings emitted so the caller can decide whether to
    emit a 'Score successfully computed' message.
    """
    _BLUE_VARIANTS = {
        VruPredictionColor.BLUE_T.value,
        VruPredictionColor.BLUE_ST.value,
        VruPredictionColor.BLUE_SA.value,
    }

    # First pass: collect all blue criteria and build a value lookup for provided values.
    # key: (row, col, criteria_name) → value (only non-NaN entries)
    value_map = {}
    all_blue = []

    for load_case in legform_loadcases:
        for seat in load_case.seats:
            for body_region in seat.dummy.body_region_list:
                last_tp = None  # carry-forward: Test point is deduplicated in the sheet
                for criteria in body_region._criteria:
                    pred = getattr(criteria, "prediction", None)
                    if pred is None or pred.lower() not in _BLUE_VARIANTS:
                        continue
                    tp = getattr(criteria, "test_point", None)
                    if tp and not pd.isna(tp):
                        last_tp = tp
                    elif last_tp:
                        tp = last_tp
                    row, col = _parse_test_point(tp)
                    if row is None:
                        continue
                    all_blue.append((row, col, criteria))
                    if not pd.isna(criteria.value):
                        value_map[(row, col, criteria.name)] = criteria.value

    # Second pass: resolve missing values. The outcome must not depend on
    # sheet order:
    #
    # - An ST point copies its symmetric partner. That partner is guaranteed
    #   to be a T or an ST — collect_legform_symmetry_violations (enforced via
    #   check_legform_blue_symmetry) rejects matrices where the mirror of a
    #   T/ST is anything else — so an ST never inherits from an inferred SA.
    #   Nor can it inherit from another *inferred* ST: such a chain would have
    #   to be circular (the partner's own mirror is this blank point), so the
    #   partner's value can only be measured, and every resolvable ST settles
    #   in one sweep from the measured values alone.
    # - An SA point takes the max of its adjacent columns, and a neighbour may
    #   settle later with a *larger* value than whatever was available when
    #   the SA was first visited. Finalizing an SA on its first available
    #   neighbour therefore still made the result depend on which side of the
    #   matrix the measured value was entered on. Since max is monotone,
    #   sweeping the SA points with upward revision until nothing changes
    #   converges to the unique order-independent fixed point. Measured
    #   values are never revised; only inferred ones can grow.
    pending_st = []
    pending_sa = []
    for row, col, criteria in all_blue:
        if not pd.isna(criteria.value):
            continue
        pred = criteria.prediction.lower()
        if pred == VruPredictionColor.BLUE_ST.value:
            pending_st.append((row, col, criteria))
        elif pred == VruPredictionColor.BLUE_SA.value:
            pending_sa.append((row, col, criteria))
        # BLUE_T never resolves from neighbours; it is warned about below.

    for row, col, criteria in pending_st:
        resolved = value_map.get((row, -col, criteria.name))
        if resolved is not None:
            criteria.set_value(resolved)
            value_map[(row, col, criteria.name)] = criteria.value
            logger.debug(
                "ST point at %s (%s): copied value %.4g from symmetric col=%d",
                criteria.test_point,
                criteria.name,
                resolved,
                -col,
            )

    changed = True
    while changed:
        changed = False
        for row, col, criteria in pending_sa:
            adj_values = [
                v
                for adj_col in (col - 1, col + 1)
                for v in [value_map.get((row, adj_col, criteria.name))]
                if v is not None
            ]
            if not adj_values:
                continue
            # Compare on the same 2-dp lattice that set_value() stores, so the
            # sweep is strictly increasing and guaranteed to terminate even if
            # a neighbour value ever carried more than 2 decimals.
            resolved = round_half_up(max(adj_values), 2)
            current = value_map.get((row, col, criteria.name))
            if current is not None and resolved <= current:
                continue
            criteria.set_value(resolved)
            value_map[(row, col, criteria.name)] = criteria.value
            logger.debug(
                "SA point at %s (%s): value inferred as %.4g from adjacent columns",
                criteria.test_point,
                criteria.name,
                resolved,
            )
            changed = True

    # Whatever is still missing cannot be resolved from any other point.
    warning_count = 0
    for row, col, criteria in all_blue:
        if not pd.isna(criteria.value):
            continue
        pred = criteria.prediction.lower()

        if pred == VruPredictionColor.BLUE_T.value:
            logger.warning(
                "T point at %s (%s) has no Value; score=0",
                criteria.test_point,
                criteria.name,
            )
        elif pred == VruPredictionColor.BLUE_ST.value:
            logger.warning(
                "ST point at %s (%s) and its symmetric (col=%d) both have no Value; score=0",
                criteria.test_point,
                criteria.name,
                -col,
            )
        else:
            logger.warning(
                "SA point at %s (%s): no adjacent Values found; score=0",
                criteria.test_point,
                criteria.name,
            )
        warning_count += 1

    return warning_count


_APILLAR_PREDICTIONS = {
    VruPredictionColor.GREEN_20.value,
    VruPredictionColor.GREEN_30.value,
    VruPredictionColor.GREEN_40.value,
}


def _apillar_value_missing(value):
    """An A-pillar run counts as untested when its Value is blank or the
    preprocess-seeded 0.0 - a HIC15 of exactly 0.0 is not a physical result."""
    return pd.isna(value) or value == 0.0


def _resolve_headform_apillar_values(headform_loadcases):
    """Resolve missing HIC15 Values for A-pillar (green-20/30/40) runs before scoring.

    Only one physical test is run for a symmetric A-pillar pair (the same
    green-N0 shade at mirrored lateral positions), so the measured HIC15 of
    the tested run is copied to its untested mirror instead of expecting the
    lab to enter the same value twice. A run without a same-shade tested
    mirror must be tested itself: when its Value is missing, its Value is
    blanked so the run scores 0 (the seeded 0.0 would otherwise silently pass
    as green) and a warning is emitted.

    Returns the number of warnings emitted so the caller can decide whether
    to emit a 'Score successfully computed' message.
    """
    # First pass: collect all A-pillar criteria and the provided values.
    # key: (row, col, criteria_name, prediction) -> value
    value_map = {}
    all_apillar = []

    for load_case in headform_loadcases:
        if load_case.name != "Headform Apillar":
            continue
        for seat in load_case.seats:
            for body_region in seat.dummy.body_region_list:
                last_tp = None  # carry-forward: Test point is deduplicated in the sheet
                for criteria in body_region._criteria:
                    pred = getattr(criteria, "prediction", None)
                    if pred is None or pred.lower() not in _APILLAR_PREDICTIONS:
                        continue
                    tp = getattr(criteria, "test_point", None)
                    if tp and not pd.isna(tp):
                        last_tp = tp
                    elif last_tp:
                        tp = last_tp
                    row, col = _parse_test_point(tp)
                    if row is None:
                        continue
                    all_apillar.append((row, col, criteria))
                    if not _apillar_value_missing(criteria.value):
                        value_map[(row, col, criteria.name, pred.lower())] = (
                            criteria.value
                        )

    # Second pass: resolve missing values.
    warning_count = 0
    for row, col, criteria in all_apillar:
        if not _apillar_value_missing(criteria.value):
            continue

        pred = criteria.prediction.lower()
        # The centre column (col == 0) is its own mirror; a same-shade mirror
        # only exists on the opposite signed column.
        sym_value = (
            value_map.get((row, -col, criteria.name, pred)) if col != 0 else None
        )
        if sym_value is not None:
            criteria.set_value(sym_value)
            value_map[(row, col, criteria.name, pred)] = criteria.value
            logger.debug(
                "A-pillar point at %s (%s, prediction %s): copied value %.4g "
                "from symmetric col=%d",
                criteria.test_point,
                criteria.name,
                pred,
                sym_value,
                -col,
            )
        else:
            logger.warning(
                "A-pillar point at %s (%s, prediction %s) has no Value and no "
                "tested symmetric run with the same prediction; it must be "
                "tested - score=0",
                criteria.test_point,
                criteria.name,
                pred,
            )
            # NaN yields color=None/score=0.0 in calculate_color_score, and the
            # score refresh in from_sheet_dict leaves color-less criteria alone,
            # so the untested run scores 0 instead of green.
            criteria.set_value(float("nan"))
            warning_count += 1

    return warning_count


@dataclass
class VruTestData:
    """
    Represents a collection of VRU test data.
    """

    prediction_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    headform_test_points: list[VruTestPoint] = field(default_factory=list)
    legform_test_points: list[LegformTestPoint] = field(default_factory=list)

    headform_loadcases: list[LoadCase] = field(default_factory=list)
    legform_loadcases: list[LoadCase] = field(default_factory=list)

    headform_scores: dict[str, VruScore] = field(default_factory=dict)
    legform_scores: dict[str, VruScore] = field(default_factory=dict)

    loadcase_dict: dict[str, list[LoadCase]] = field(default_factory=dict)
    df_dict: dict[str, pd.DataFrame] = field(default_factory=dict)

    legform_blue_warnings: int = 0
    headform_apillar_warnings: int = 0

    vru_test_points_all: list[VruTestPoint] = field(default_factory=list)

    def get_vru_test_points(self):
        """
        Returns a combined list of all VRU test points.
        """
        return self.headform_test_points + self.legform_test_points

    @staticmethod
    def from_sheet_dict(sheet_dict):
        """
        Creates and returns a VruTestData instance populated from a sheet_dict containing DataFrames for
        "CP - VRU Head Impact" and "CP - VRU Pelvis & Leg Impact".
        """
        vru_test_data = VruTestData()
        # Headform test points and loadcases
        vru_test_data.headform_loadcases = sheet_dict.get("CP - VRU Head Impact", [])
        vru_test_data.headform_test_points = []

        # Build headform test points from headform_loadcases
        for load_case in vru_test_data.headform_loadcases:
            for seat in load_case.seats:
                dummy = seat.dummy
                for body_region in dummy.body_region_list:
                    for criteria in body_region._criteria:
                        test_point_str = getattr(criteria, "test_point", None)
                        if not test_point_str or pd.isna(test_point_str):
                            continue
                        try:
                            row_idx, col_idx = map(int, str(test_point_str).split(", "))
                        except Exception:
                            continue
                        color_str = getattr(criteria, "prediction", None)
                        color = None
                        if isinstance(color_str, str):
                            color = next(
                                (
                                    k
                                    for k in VruPredictionColor
                                    if k.value.lower() == color_str.lower()
                                ),
                                VruPredictionColor.GREY,
                            )
                        vtp = VruTestPoint(
                            row=row_idx,
                            col=col_idx,
                            color=color,
                            loadcase_name=load_case.name,
                        )
                        vru_test_data.headform_test_points.append(vtp)

        # Legform test points and loadcases
        vru_test_data.legform_loadcases = sheet_dict.get(
            "CP - VRU Pelvis & Leg Impact", []
        )
        vru_test_data.legform_test_points = []
        for load_case in vru_test_data.legform_loadcases:
            for seat in load_case.seats:
                dummy = seat.dummy
                for body_region in dummy.body_region_list:
                    for criteria in body_region._criteria:
                        test_point_str = getattr(criteria, "test_point", None)
                        if not test_point_str or pd.isna(test_point_str):
                            continue
                        try:
                            row_idx, col_idx = map(int, str(test_point_str).split(", "))
                        except Exception:
                            continue
                        color_str = getattr(criteria, "prediction", None)
                        color = None
                        if isinstance(color_str, str):
                            color = next(
                                (
                                    k
                                    for k in VruPredictionColor
                                    if k.value.lower() == color_str.lower()
                                ),
                                VruPredictionColor.GREY,
                            )
                        ltp = LegformTestPoint(
                            row=row_idx,
                            col=col_idx,
                            color=color,
                            loadcase_name=load_case.name,
                        )
                        vru_test_data.legform_test_points.append(ltp)
        # Resolve missing Values for blue (T/ST/SA) legform criteria before scoring
        vru_test_data.legform_blue_warnings = _resolve_legform_blue_values(
            vru_test_data.legform_loadcases
        )
        # Resolve missing HIC15 Values for A-pillar (green-20/30/40) runs: one
        # physical test covers a symmetric pair, so the tested value is copied
        # to the untested mirror before the criteria scores are refreshed.
        vru_test_data.headform_apillar_warnings = _resolve_headform_apillar_values(
            vru_test_data.headform_loadcases
        )

        # Update criteria.score for all criteria in headform and legform loadcases
        for load_case in (
            vru_test_data.headform_loadcases + vru_test_data.legform_loadcases
        ):
            for seat in load_case.seats:
                dummy = seat.dummy
                for body_region in dummy.body_region_list:
                    for criteria in body_region._criteria:
                        color = criteria.color
                        if load_case.name == "Upper Leg":
                            logger.debug(
                                f"Setting score for criteria at test_point={getattr(criteria, 'test_point', None)}, color={color}"
                            )
                        if color is not None:
                            score = (
                                VruScore.COLOR_WEIGHTS.get(str(color).lower(), 0.0)
                                * 100
                            )
                            criteria.score = score

        return vru_test_data

    _HEADFORM_REGIONS_BY_DUMMY = {
        "Adult Headform": ["Adult", "Cyclist"],
        "Child Headform": ["Child"],
    }

    def _ensure_headform_bodyregions(self, headform_region_scores):
        """Guarantee that every WAD band with predicted grid points exists as a
        BodyRegion under the 'Headform' loadcase.

        A band whose grid cells are coloured but that received no verification
        row (no sampled point, no blue point, no A-pillar point) has no
        BodyRegion in the loadcase tree, so its predicted score could never be
        scored nor written back to the report. When that happens the missing
        structure is synthesized here: both headform dummies and all three body
        regions are created (regions without coloured cells stay score-less
        placeholders so the report block is walkable end to end).

        A workbook with no head-impact rows at all keeps its current all-zero
        behaviour: nothing is synthesized.
        """
        if not self.headform_loadcases:
            return
        needed = [
            region
            for region in ("Cyclist", "Adult", "Child")
            if headform_region_scores[region].max_points > 0
        ]
        if not needed:
            return

        headform_loadcase = next(
            (lc for lc in self.headform_loadcases if lc.name == "Headform"), None
        )
        existing_regions = set()
        if headform_loadcase is not None:
            existing_regions = {
                body_region.name
                for seat in headform_loadcase.seats
                for body_region in seat.dummy.body_region_list
            }
        missing = [region for region in needed if region not in existing_regions]
        if not missing:
            return
        logger.info(
            "WAD bands %s have predicted grid points but no verification row; "
            "synthesizing their body regions so the predicted score stands.",
            missing,
        )

        if headform_loadcase is None:
            headform_loadcase = LoadCase(name="Headform", seats=[])
            self.headform_loadcases.append(headform_loadcase)
        elif headform_loadcase.seats and not headform_loadcase.raw_seats:
            # Keep the original seat structure as the alternative loadcase id so
            # the verification-sheet rows still match after seats are added
            # (same mechanism as the farside Pole-32 merge).
            headform_loadcase.raw_seats = headform_loadcase.seats.copy()

        for dummy_name, regions in self._HEADFORM_REGIONS_BY_DUMMY.items():
            seat = next(
                (
                    s
                    for s in headform_loadcase.seats
                    if s.name == "Driver" and s.dummy.name == dummy_name
                ),
                None,
            )
            if seat is None:
                seat = Seat(name="Driver", dummy=Dummy(name=dummy_name))
                # Adult Headform must precede Child Headform to match the
                # static report block id.
                if dummy_name == "Adult Headform":
                    headform_loadcase.seats.insert(0, seat)
                else:
                    headform_loadcase.seats.append(seat)
            for region in regions:
                if not any(
                    body_region.name == region
                    for body_region in seat.dummy.body_region_list
                ):
                    seat.dummy.body_region_list.append(BodyRegion(name=region))

    def compute_vru_bodyregion_score(self):
        """
        Compute and set headform and legform scores for the VRU test data.
        Updates self.headform_scores and self.legform_scores.
        Returns the legform_final_scores dict.
        """
        # Use instance data
        headform_test_points = self.headform_test_points
        vru_head_impact_loadcases = self.headform_loadcases
        vru_legform_loadcases = self.legform_loadcases

        logger.debug("headform_test_points:")
        for point in headform_test_points:
            logger.debug(f"  {point}")

        logger.debug("vru_head_impact_loadcases:")
        for lc in vru_head_impact_loadcases:
            logger.debug(f"  {lc}")

        logger.debug("vru_legform_loadcases:")
        for lc in vru_legform_loadcases:
            logger.debug(f"  {lc}")

        # Compute headform region scores
        headform_region_scores = {}
        for region in ["Cyclist", "Adult", "Child"]:
            score = compute_vru_score(self.vru_test_points_all, region)
            headform_region_scores[region] = score

        # Per protocol the predicted score of all grid points stands regardless
        # of where the verification tests landed, so every WAD band with
        # coloured grid points must exist in the loadcase tree even when no
        # verification row fell into it.
        self._ensure_headform_bodyregions(headform_region_scores)

        pretty_print_loadcases(vru_head_impact_loadcases)
        pretty_print_loadcases(vru_legform_loadcases)

        # Compute blue points and a_pillar for each region
        for region in ["Cyclist", "Adult", "Child"]:
            region_score = headform_region_scores[region]
            region_score.compute_blue_points(vru_head_impact_loadcases)
            region_score.compute_a_pillar(vru_head_impact_loadcases)

        # Log scores
        logger.debug(f"VRU Cyclist Score: {headform_region_scores['Cyclist']}")
        logger.debug(f"VRU Adult Score: {headform_region_scores['Adult']}")
        logger.debug(f"VRU Child Score: {headform_region_scores['Child']}")

        # Store in self.headform_scores
        self.headform_scores = headform_region_scores

        vru_factors = get_vru_factors(vru_head_impact_loadcases)
        logger.debug(f"VRU Factors: {vru_factors}")

        # Per protocol every WAD band with coloured grid points is assessed:
        # the denominator sums the max_points of all three regions. A band with
        # no coloured cells at all has max_points == 0 and naturally drops out;
        # a band that merely received no verification test keeps its predicted
        # score (its BodyRegion is synthesized in _ensure_headform_bodyregions).
        vru_max_points_sum = sum(
            headform_region_scores[region].max_points
            for region in ["Cyclist", "Adult", "Child"]
        )
        logger.debug(f"vru_max_points_sum: {vru_max_points_sum}")

        # Set bodyregion scores in loadcases
        if vru_max_points_sum == 0:
            logger.debug(
                "vru_max_points_sum is 0 (no VRU prediction points available); "
                "skipping headform bodyregion score computation."
            )
        else:
            region_dummy_names = {
                region: dummy_name
                for dummy_name, regions in self._HEADFORM_REGIONS_BY_DUMMY.items()
                for region in regions
            }
            for load_case in vru_head_impact_loadcases:
                if load_case.name != "Headform":
                    continue
                for seat in load_case.seats:
                    if seat.name != "Driver":
                        continue
                    dummy = seat.dummy
                    logger.debug(
                        f"Processing LoadCase: {load_case.name}, Seat: {seat.name}, Dummy: {dummy.name}"
                    )
                    for body_region in dummy.body_region_list:
                        if region_dummy_names.get(body_region.name) != dummy.name:
                            continue
                        vru_score = headform_region_scores[body_region.name]
                        if vru_score.max_points <= 0:
                            continue
                        logger.debug(f"vru score : {vru_score}")
                        bodyregion_score = (
                            vru_score.predicted_score * vru_factors["correction_factor"]
                            + vru_score.blue_points
                            + vru_score.a_pillar
                        ) / vru_score.max_points
                        bodyregion_score = min(bodyregion_score * 100.0, 100.0)
                        body_region.set_bodyregion_score(bodyregion_score)
                        body_region.set_max_score(
                            vru_score.max_points / vru_max_points_sum * 10.0
                        )

        # Compute the sum of minimum criteria scores and count for each VRU legform body region
        legform_body_regions = ["Pelvis", "Femur", "Knee & Tibia"]
        legform_min_score_sum = {br: 0.0 for br in legform_body_regions}
        legform_counts = {br: 0 for br in legform_body_regions}
        pelvis_score = 0.0
        for load_case in vru_legform_loadcases:
            for seat in load_case.seats:
                dummy = seat.dummy
                # For Knee & Tibia, collect all criteria from both regions
                knee_tibia_scores = []
                for body_region in dummy.body_region_list:
                    br_name = body_region.name
                    if br_name == "Knee" or br_name == "Tibia":
                        criteria_scores = [
                            c.get_score()
                            for c in body_region._criteria
                            if c.get_score() is not None
                        ]
                        knee_tibia_scores.extend(criteria_scores)
                        for c in body_region._criteria:
                            logger.debug(
                                f"criteria score for {br_name}: {c.get_score()}, test point: {getattr(c, 'test_point', None)}"
                            )
                    elif br_name in legform_body_regions:
                        criteria_scores = [
                            c.get_score()
                            for c in body_region._criteria
                            if c.get_score() is not None
                        ]
                        for c in body_region._criteria:
                            logger.debug(
                                f"criteria score for {br_name}: {c.get_score()}, test point: {getattr(c, 'test_point', None)}"
                            )
                    if br_name == "Pelvis":
                        pelvis_score += sum(criteria_scores)
                        logger.debug(
                            f"Pelvis score: {pelvis_score}, after adding test point: {getattr(c, 'test_point', None)}"
                        )
                        legform_counts[br_name] += len(criteria_scores)
                    elif br_name == "Femur":
                        min_score = min(criteria_scores)
                        logger.debug(
                            f"Minimum score for {br_name}: {min_score}, test point: {getattr(c, 'test_point', None)}"
                        )
                        legform_min_score_sum[br_name] += min_score
                        legform_counts[br_name] += 1
                # After collecting, process Knee & Tibia as one region
                if knee_tibia_scores:
                    min_score = min(knee_tibia_scores)
                    logger.debug(f"Minimum score for Knee & Tibia: {min_score}")
                    legform_min_score_sum["Knee & Tibia"] += min_score
                    legform_counts["Knee & Tibia"] += 1
        logger.debug("legform_min_score_sum: %s", legform_min_score_sum)
        logger.debug("legform_counts: %s", legform_counts)
        # Calculate the final values: sum divided by 100 and by the count for each body region
        legform_final_scores = {}
        for br in legform_body_regions:
            count = legform_counts[br]

            if count > 0:
                if br == "Pelvis":
                    legform_final_scores[br] = pelvis_score / 100.0 / count * 100.0
                else:
                    legform_final_scores[br] = (
                        legform_min_score_sum[br] / 100.0 / count * 100.0
                    )
            else:
                legform_final_scores[br] = 0.0
        logger.debug("legform_final_scores: %s", legform_final_scores)

        # Store in self.legform_scores
        self.legform_scores = legform_final_scores

        if self.legform_blue_warnings == 0:
            logger.info("Score successfully computed for all blue legform criteria.")
        if self.headform_apillar_warnings == 0:
            logger.info(
                "Score successfully computed for all A-pillar headform criteria."
            )


def compute_vru_score(vru_test_points, body_region):
    region_cells = [
        point for point in vru_test_points if point.body_region == body_region
    ]
    logger.debug(f"region_cells for {body_region}: {region_cells}")
    region_color_counts = Counter(point.color for point in region_cells)

    region_score = VruScore(name=body_region)
    region_score.predicted_score = sum(
        region_color_counts.get(color, 0) * weight
        for color, weight in VruScore.COLOR_WEIGHTS.items()
    )
    # Calculate max_points using vru_test_points_all for the region

    count_valid = sum(
        p.color
        not in (
            VruPredictionColor.GREY,
            VruPredictionColor.GREEN_40,
            VruPredictionColor.GREEN_30,
        )
        for p in region_cells
    )
    count_green_40 = sum(p.color == VruPredictionColor.GREEN_40 for p in region_cells)
    count_green_30 = sum(p.color == VruPredictionColor.GREEN_30 for p in region_cells)
    logger.info(
        f"Max points for {body_region}: valid={count_valid}, green_40={count_green_40}, green_30={count_green_30}"
    )
    region_score.max_points = count_valid + count_green_40 * 3 + count_green_30 * 2
    logger.info(f"{body_region} score: {region_score}")
    return region_score


def get_vru_factors(loadcases):
    # Compute weighted VRU prediction score.
    # Only genuine grid colors (green/yellow/orange/brown/red) are eligible for the
    # correction factor - blue variants (blue/t/st/sa) and other loadcases (e.g.
    # "Headform Apillar") must never contribute to either the numerator or denominator.
    color_weights = {
        "green": 1.0,
        "yellow": 0.75,
        "orange": 0.5,
        "brown": 0.25,
        "red": 0.0,
    }
    color_counts = {"green": 0, "yellow": 0, "orange": 0, "brown": 0, "red": 0}

    # Sum of criteria.score for eligible (non-blue-variant) predictions
    score_sum = 0.0

    for load_case in loadcases:
        if load_case.name != "Headform":
            continue
        for seat in load_case.seats:
            dummy = seat.dummy
            for body_region in dummy.body_region_list:
                for criteria in body_region._criteria:
                    logger.debug(
                        f"[CP - VRU Prediction] Processing criteria: {criteria} in {body_region}"
                    )
                    prediction = (
                        criteria.prediction.lower() if criteria.prediction else ""
                    )
                    if prediction in color_counts:
                        color_counts[prediction] += 1
                        score_sum += criteria.score

    predicted_score_verification = sum(
        color_counts[color] * color_weights[color] for color in color_counts
    )
    # Divide the total score sum by 100
    tested_score_verification = score_sum / 100.0
    if predicted_score_verification == 0:
        vru_correction_factor = 0
    else:
        vru_correction_factor = tested_score_verification / predicted_score_verification
    if not (0.85 <= vru_correction_factor <= 1.15):
        logger.warning(
            "[CP - VRU Head Impact] VRU correction factor %.4f out of expected band "
            "[0.85, 1.15] before clamping (tested=%.4f, predicted=%.4f); "
            "this may indicate a data or scoring issue and should be investigated per protocol §4.1.",
            vru_correction_factor,
            tested_score_verification,
            predicted_score_verification,
        )
    vru_correction_factor = np.clip(vru_correction_factor, 0.85, 1.15)

    return {
        "predicted_score_verification": predicted_score_verification,
        "tested_score_verification": tested_score_verification,
        "correction_factor": vru_correction_factor,
    }


def get_num_verification_tests(vru_params, param_code):
    """
    Read the 'Number of verification tests' input for a given param code.

    Empty template cells are dropped when building param_df, so a missing or
    non-numeric value is expected for unfilled templates: warn and fall back
    to 0 verification tests instead of failing.

    Args:
        vru_params (pd.DataFrame): Parameter DataFrame filtered to VRU rows.
        param_code (str): The param code to look up (e.g. "CP - VRU Head Impact").

    Returns:
        int: The number of verification tests, or 0 if missing/invalid.
    """
    mask = (vru_params["param_code"] == param_code) & (
        vru_params["Input parameter"].str.contains(
            "Number of verification tests", na=False
        )
    )
    values = vru_params.loc[mask, "Value"].values
    if len(values) == 0 or pd.isna(values[0]):
        logger.warning(
            f"'Number of verification tests' for '{param_code}' is missing or empty; "
            "defaulting to 0 verification test points."
        )
        return 0
    try:
        return int(values[0])
    except (TypeError, ValueError):
        logger.warning(
            f"'Number of verification tests' for '{param_code}' has non-numeric value "
            f"{values[0]!r}; defaulting to 0 verification test points."
        )
        return 0


def _allocate_band_quotas(band_sizes, n_total):
    """Split n_total across WAD bands proportionally to their eligible-cell
    counts (largest-remainder rounding), guaranteeing at least one point per
    non-empty band whenever n_total allows. Quotas never exceed a band's size.

    Args:
        band_sizes (dict): band name -> number of eligible cells (all > 0).
        n_total (int): requested number of verification points.

    Returns:
        dict: band name -> allocated number of points.
    """
    bands = list(band_sizes)
    total = sum(band_sizes.values())
    if n_total >= total:
        return dict(band_sizes)
    if n_total < len(bands):
        # Not enough points for every band: give one each to the largest bands.
        alloc = {band: 0 for band in bands}
        for band in sorted(bands, key=lambda b: band_sizes[b], reverse=True)[:n_total]:
            alloc[band] = 1
        return alloc

    raw = {band: n_total * band_sizes[band] / total for band in bands}
    alloc = {band: int(np.floor(raw[band])) for band in bands}
    remainder = n_total - sum(alloc.values())
    for band in sorted(bands, key=lambda b: raw[b] - alloc[b], reverse=True)[
        :remainder
    ]:
        alloc[band] += 1

    # Every non-empty band must receive at least one verification point.
    for band in bands:
        if alloc[band] == 0:
            donor = max(bands, key=lambda b: alloc[b])
            if alloc[donor] > 1:
                alloc[donor] -= 1
                alloc[band] = 1

    # Cap each quota at the band size and hand the excess to bands with capacity.
    excess = 0
    for band in bands:
        if alloc[band] > band_sizes[band]:
            excess += alloc[band] - band_sizes[band]
            alloc[band] = band_sizes[band]
    while excess > 0:
        candidates = [band for band in bands if alloc[band] < band_sizes[band]]
        if not candidates:
            break
        band = max(candidates, key=lambda b: band_sizes[b] - alloc[b])
        alloc[band] += 1
        excess -= 1
    return alloc


def _stratified_headform_sample(filtered_colors, n_total, seed=None):
    """Sample n_total verification points from the headform grid, stratified
    by WAD band (Cyclist/Adult/Child) so that every band containing eligible
    cells receives verification points whenever n_total allows. A single
    colour-proportional draw over the whole grid can leave a band with no
    verification test purely by chance, which would previously drop that
    band's predicted score entirely.

    Args:
        filtered_colors (list): (i, j, color) matrix cells eligible for sampling.
        n_total (int): requested number of verification points.
        seed: forwarded to common.frequency_proportional_sample.

    Returns:
        list: sampled (i, j, color) cells.
    """
    if n_total <= 0:
        return []
    if n_total >= len(filtered_colors):
        return list(filtered_colors)

    band_cells = {}
    for cell in filtered_colors:
        # Matrix row i counts from the top of the grid; point row = 18 - i.
        band = headform_body_region(18 - cell[0])
        band_cells.setdefault(band, []).append(cell)
    band_cells.pop(None, None)

    quotas = _allocate_band_quotas(
        {band: len(cells) for band, cells in band_cells.items()}, n_total
    )
    logger.info(f"Stratified headform sampling quotas per WAD band: {quotas}")

    sampled = []
    for band, cells in band_cells.items():
        quota = quotas.get(band, 0)
        if quota > 0:
            sampled.extend(
                common.frequency_proportional_sample(cells, quota, seed=seed)
            )
    return sampled


def process_headform_test(vru_prediction_df, vru_params, input_selected_points=None):
    """
    Process the VRU predictions DataFrame to extract headforms  matrices,
    compute color percentages, and randomly select samples based on color distribution.

    Args:
        vru_prediction_df (pd.DataFrame): DataFrame containing VRU predictions.

    Returns:
        None
    """

    # Ensure the DataFrame is not empty
    if vru_prediction_df.empty:
        logger.error("The input DataFrame is empty.")
        return
    logger.info(f"vru_params: {vru_params}")

    if input_selected_points is not None:
        candidate_points = extract_headform_candidate_points(vru_prediction_df)
        matched_points = select_vru_points_from_input(
            candidate_points,
            input_selected_points,
        )
        # A-pillar (green-20/30/40) points are mandatory on every path: the
        # protocol scores all of them, so they are appended even when the
        # explicit selection omits them - mirroring the sampling path, where
        # they are excluded from the sampled verification count and appended
        # unconditionally afterwards. Blue cells are deliberately not
        # force-appended here: the caller stays responsible for selecting them.
        selected_apillar_coords = {
            (point.row, point.col)
            for point in matched_points
            if point.loadcase_name == "Headform Apillar"
        }
        for point in candidate_points:
            if (
                point.loadcase_name == "Headform Apillar"
                and (point.row, point.col) not in selected_apillar_coords
            ):
                logger.info(
                    "A-pillar point (%d, %d) with prediction '%s' was not in the "
                    "explicit selection; adding it (A-pillar points are mandatory).",
                    point.row,
                    point.col,
                    point.color.value,
                )
                matched_points.append(point)
        return matched_points

    num_verification_tests_head = get_num_verification_tests(
        vru_params, "CP - VRU Head Impact"
    )
    logger.info(
        f"Number of verification tests for headform: {num_verification_tests_head}"
    )

    headforms_matrix = None

    # Headforms matrix: from HEADFORMS_START_ROW_INDEX to HEADFORMS_END_ROW_INDEX - 1
    headforms_matrix = vru_prediction_df.iloc[
        HEADFORMS_START_ROW_INDEX:HEADFORMS_END_ROW_INDEX, :
    ].reset_index(drop=True)

    headforms_matrix = headforms_matrix.iloc[2:, 4:].astype(str).values.tolist()
    # Convert all values in headforms_matrix to VruPredictionColor enum
    for i in range(len(headforms_matrix)):
        for j in range(len(headforms_matrix[0])):
            color_str = headforms_matrix[i][j].lower()
            enum_val = next(
                (k for k in VruPredictionColor if k.value.lower() == color_str),
                VruPredictionColor.GREY,
            )
            headforms_matrix[i][j] = enum_val

    # Compute the percentage of each color occurrence in headforms_matrix and print it
    # Use all color values, including 'd red', 'green-40', 'green-30', 'green-20'
    filtered_colors = [
        (i, j, color)
        for i, row in enumerate(headforms_matrix)
        for j, color in enumerate(row)
        if color
        not in (
            VruPredictionColor.GREY,
            VruPredictionColor.BLUE,
            VruPredictionColor.GREEN_40,
            VruPredictionColor.GREEN_30,
            VruPredictionColor.GREEN_20,
            VruPredictionColor.D_RED,
            VruPredictionColor.D_GREEN,
            VruPredictionColor.BLUE_T,
            VruPredictionColor.BLUE_ST,
            VruPredictionColor.BLUE_SA,
        )
    ]

    selected_row_col_pairs = _stratified_headform_sample(
        filtered_colors, num_verification_tests_head, seed=None
    )

    logger.info(
        f"Randomly selected {num_verification_tests_head} (row, col) pairs (excluding grey/blue): {selected_row_col_pairs}"
    )

    all_cells = [
        (row, col, headforms_matrix[row][col])
        for row in range(len(headforms_matrix))
        for col in range(len(headforms_matrix[0]))
    ]
    # Add blue points to the selected_row_col_pairs
    for row, col, color in all_cells:
        if color in {
            VruPredictionColor.BLUE,
            VruPredictionColor.BLUE_T,
            VruPredictionColor.BLUE_ST,
            VruPredictionColor.BLUE_SA,
        }:
            selected_row_col_pairs.append((row, col, color))

    # Use selected_row_col_pairs to select test points
    headform_test_points = []
    for row, col, color in selected_row_col_pairs:
        # Directly create a new VruTestPoint without searching
        vtp = VruTestPoint(
            row=18 - row,
            col=10 - col,
            color=color,
        )
        vtp.loadcase_name = "Headform"
        headform_test_points.append(vtp)

    a_pillar_test_points = []
    for row, col, color in all_cells:
        if color in (
            VruPredictionColor.GREEN_40,
            VruPredictionColor.GREEN_30,
            VruPredictionColor.GREEN_20,
        ):
            vtp = VruTestPoint(
                row=18 - row,
                col=10 - col,
                color=color,
            )
            vtp.loadcase_name = "Headform Apillar"
            a_pillar_test_points.append(vtp)
    logger.info("A-Pillar test points created: %s", a_pillar_test_points)

    # Combine headform and a-pillar test points
    test_points = headform_test_points + a_pillar_test_points

    return test_points


def compute_all_vru_test_points(vru_prediction_df):
    """
    Compute all VRU test points from the prediction matrix without sampling.

    Args:
        vru_prediction_df (pd.DataFrame): DataFrame containing VRU predictions.

    Returns:
        list[VruTestPoint]: List of all VRU test points with their colors.
    """
    if vru_prediction_df.empty:
        logger.error("The input DataFrame is empty.")
        return []

    # Extract headforms matrix
    headforms_matrix = vru_prediction_df.iloc[
        HEADFORMS_START_ROW_INDEX:HEADFORMS_END_ROW_INDEX, :
    ].reset_index(drop=True)

    headforms_matrix = headforms_matrix.iloc[2:, 4:].astype(str).values.tolist()

    # Convert all values to VruPredictionColor enum
    for i in range(len(headforms_matrix)):
        for j in range(len(headforms_matrix[0])):
            color_str = headforms_matrix[i][j].lower()
            enum_val = next(
                (k for k in VruPredictionColor if k.value.lower() == color_str),
                VruPredictionColor.GREY,
            )
            headforms_matrix[i][j] = enum_val

    # Create all test points
    vru_test_points_all = []
    for row in range(len(headforms_matrix)):
        for col in range(len(headforms_matrix[0])):
            color = headforms_matrix[row][col]
            vru_test_point = VruTestPoint(
                row=18 - row,
                col=10 - col,
                color=color,
            )
            vru_test_points_all.append(vru_test_point)

    return vru_test_points_all


@dataclass
class LegformSymmetryViolation:
    """One legform T/ST/SA symmetry violation, in enough detail for either
    caller to build its own message: check_legform_blue_symmetry's blocking
    ValueError (matrix-index wording, unchanged) and
    consistency.find_prediction_inconsistencies' report-only finding
    (worksheet-cell wording). ``sym_color`` is None for "sa_out_of_range",
    the one rule with no symmetric cell to name."""

    row_idx: int
    j: int
    col: int
    j_sym: int
    sym_color: "VruPredictionColor | None"
    rule: str  # "t_st" | "sa_mismatch" | "sa_out_of_range"


def _legform_violation_message(v: "LegformSymmetryViolation") -> str:
    """check_legform_blue_symmetry's exact historical wording, rebuilt from
    a LegformSymmetryViolation -- kept separate from the violation
    collection itself so the report-only collector can word its own
    message differently without touching this one."""
    if v.rule == "t_st":
        return (
            f"Asymmetric T/ST prediction at legform matrix row={v.row_idx}, "
            f"col={v.col} (matrix index j={v.j}): symmetric cell (col={-v.col}, "
            f"j={v.j_sym}) has color '{v.sym_color.value}', expected T or ST."
        )
    if v.rule == "sa_out_of_range":
        return (
            f"SA prediction at legform matrix row={v.row_idx}, col={v.col} "
            f"(matrix index j={v.j}): symmetric column index {v.j_sym} is out of range."
        )
    return (
        f"SA prediction at legform matrix row={v.row_idx}, col={v.col} "
        f"(matrix index j={v.j}): symmetric cell (col={-v.col}, j={v.j_sym}) "
        f"has color '{v.sym_color.value}', expected SA."
    )


def collect_legform_symmetry_violations(matrix) -> list:
    """The structured form of check_legform_blue_symmetry's rule: every
    T/ST or SA cell whose required symmetric partner is wrong -- all of
    them, not just the first. Returns a list of LegformSymmetryViolation in
    matrix order. Shared by the blocking preprocess check below and the
    report-only collector (consistency.find_prediction_inconsistencies),
    each of which words its own message from the structured fields."""
    num_cols = len(matrix[0])
    col_center = num_cols // 2

    blue_t_st = {VruPredictionColor.BLUE_T, VruPredictionColor.BLUE_ST}

    violations = []
    for row_idx, row in enumerate(matrix):
        for j, color in enumerate(row):
            if color == VruPredictionColor.GREY:
                continue
            col = col_center - j
            j_sym = num_cols - 1 - j
            if color in blue_t_st:
                sym_color = row[j_sym]
                if sym_color not in blue_t_st:
                    violations.append(
                        LegformSymmetryViolation(
                            row_idx, j, col, j_sym, sym_color, "t_st"
                        )
                    )

            elif color == VruPredictionColor.BLUE_SA:
                if j_sym < 0 or j_sym >= num_cols:
                    violations.append(
                        LegformSymmetryViolation(
                            row_idx, j, col, j_sym, None, "sa_out_of_range"
                        )
                    )
                    continue
                sym_color = row[j_sym]
                if sym_color != VruPredictionColor.BLUE_SA:
                    violations.append(
                        LegformSymmetryViolation(
                            row_idx, j, col, j_sym, sym_color, "sa_mismatch"
                        )
                    )
    return violations


def check_legform_blue_symmetry(matrix):
    """Validate that every T/ST and SA cell has the required symmetric partner.

    For a legform matrix with N columns the centre column index is N//2, and the
    signed col coordinate is ``col = N//2 - j``.  The symmetric of column ``j`` is
    ``j_sym = N - 1 - j``.

    Rules:
    - Every T/ST cell at (row, col) must have its symmetric cell at (row, -col) also in {T, ST}.
    - Every SA cell at (row, col) with col != 0 must have its symmetric cell be SA.
    - Every SA cell at (row, col) must have its symmetric cell be SA (for col == 0 the symmetric is itself).
    Raises:
        ValueError: with a descriptive message identifying the offending cell.
    """
    violations = collect_legform_symmetry_violations(matrix)
    if violations:
        raise ValueError(_legform_violation_message(violations[0]))


def _resolve_row_coin(valid_cols, row):
    """Decide whether an all-blue legform row needs a coin flip, and which parity to use.

    An SA cell is "orphaned" when neither of its neighbors within valid_cols is a
    T/ST -- _resolve_legform_blue_values needs at least one adjacent tested column
    to infer an SA's value, so only orphaned SA cells actually require a promotion.

    Returns (coin, needs_flip). needs_flip=False means every SA in valid_cols already
    has an adjacent tested (T/ST) neighbor -- the row must be emitted exactly as
    declared, with no coin flip and no promotion. needs_flip=True means at least one
    SA is orphaned; coin is inferred (unanimous/majority) from any T/ST-SA-T/ST
    triples in the row, falling back to a genuine random.randint(0, 1) only when
    there are no such triples or their votes are tied.
    """
    t_st = {VruPredictionColor.BLUE_T, VruPredictionColor.BLUE_ST}
    colors = [row[c] for c in valid_cols]

    def has_tested_neighbor(i):
        left = i > 0 and colors[i - 1] in t_st
        right = i < len(colors) - 1 and colors[i + 1] in t_st
        return left or right

    orphaned = any(
        colors[i] == VruPredictionColor.BLUE_SA and not has_tested_neighbor(i)
        for i in range(len(colors))
    )
    if not orphaned:
        return None, False

    votes = [
        (i - 1) % 2
        for i in range(1, len(colors) - 1)
        if colors[i] == VruPredictionColor.BLUE_SA
        and colors[i - 1] in t_st
        and colors[i + 1] in t_st
    ]
    zeros, ones = votes.count(0), votes.count(1)
    if zeros == ones:
        coin = random.randint(0, 1)
        logger.info(f"Coin flip result: {coin}")
    else:
        coin = 0 if zeros > ones else 1
        logger.info(
            f"Coin inferred from symmetry votes (0s={zeros}, 1s={ones}): {coin}"
        )
    return coin, True


def select_blue_pattern(row_idx, matrix, num_verification_tests_leg=None):
    row = matrix[row_idx]
    row_no_grey = [color for color in row if color != VruPredictionColor.GREY]
    logger.info(f"Row without grey: {row_no_grey}")
    all_blue = all(
        color
        in {
            VruPredictionColor.BLUE,
            VruPredictionColor.BLUE_T,
            VruPredictionColor.BLUE_ST,
            VruPredictionColor.BLUE_SA,
        }
        for color in row_no_grey
    )
    logger.info(f"All row is blue: {all_blue}")
    if all_blue and len(row_no_grey) > 1:
        blue_indices = [
            j
            for j, color in enumerate(row)
            if color
            in {
                VruPredictionColor.BLUE,
                VruPredictionColor.BLUE_T,
                VruPredictionColor.BLUE_ST,
                VruPredictionColor.BLUE_SA,
            }
        ]
        start_blue = blue_indices[0]
        end_blue = blue_indices[-1]
        valid_cols = [
            col
            for col in range(start_blue, end_blue + 1)
            if row[col] != VruPredictionColor.GREY
        ]
        coin, needs_flip = _resolve_row_coin(valid_cols, row)
        if needs_flip:
            selected_cols = [
                col for idx, col in enumerate(valid_cols) if (idx % 2 == coin)
            ]
            # SA→ST promotion: for each selected column, if the cell (and row=2 if applicable) is SA, promote to ST
            for col in selected_cols:
                if matrix[row_idx][col] == VruPredictionColor.BLUE_SA:
                    matrix[row_idx][col] = VruPredictionColor.BLUE_ST
                    logger.debug(
                        f"SA→ST promotion at legform matrix row={row_idx}, col_j={col} (coin={coin})"
                    )
                # Row 2 (Knee/Tibia) always mirrors row 1 (Femur) selections; promote there too
                if (
                    row_idx == 1
                    and len(matrix) > 2
                    and matrix[2][col] == VruPredictionColor.BLUE_SA
                ):
                    matrix[2][col] = VruPredictionColor.BLUE_ST
                    logger.debug(
                        f"SA→ST promotion at legform matrix row=2 (Tibia/Knee), col_j={col} (coin={coin})"
                    )
        # Every blue cell in the row's span is always returned -- the coin only ever
        # decides *which* orphaned SA cells get promoted, never which cells survive.
        return [(row_idx, col, matrix[row_idx][col]) for col in valid_cols]
    else:
        # Prepare filtered colors for the row (excluding grey/blue/green-40/green-30/green-20/d red)
        filtered_colors_leg = [
            (row_idx, j, color)
            for j, color in enumerate(row)
            if color
            not in (
                VruPredictionColor.GREY,
                VruPredictionColor.BLUE,
                VruPredictionColor.GREEN_40,
                VruPredictionColor.GREEN_30,
                VruPredictionColor.GREEN_20,
                VruPredictionColor.D_RED,
                VruPredictionColor.D_GREEN,
                VruPredictionColor.BLUE_T,
                VruPredictionColor.BLUE_ST,
                VruPredictionColor.BLUE_SA,
            )
        ]
        if num_verification_tests_leg is not None:
            selected = common.frequency_proportional_sample(
                filtered_colors_leg, num_verification_tests_leg, seed=None
            )
        else:
            selected = filtered_colors_leg
        # Also add blue points in this row
        for j, color in enumerate(row):
            if color in {
                VruPredictionColor.BLUE,
                VruPredictionColor.BLUE_T,
                VruPredictionColor.BLUE_ST,
                VruPredictionColor.BLUE_SA,
            }:
                selected.append((row_idx, j, color))
        return selected


def get_legform_matrix(vru_prediction_df):
    """
    Extract and convert legform matrix from VRU prediction DataFrame.

    Args:
        vru_prediction_df (pd.DataFrame): DataFrame containing VRU predictions.

    Returns:
        list: Legform matrix with VruPredictionColor enum values.
    """
    legform_matrix = None

    # Legforms matrix: from LEGFORMS_START_ROW_INDEX to LEGFORMS_END_ROW_INDEX - 1
    legform_matrix = vru_prediction_df.iloc[
        LEGFORMS_START_ROW_INDEX:LEGFORMS_END_ROW_INDEX, :
    ].reset_index(drop=True)

    legform_matrix = (
        legform_matrix.iloc[2:, 4:VRU_MATRIX_COL_END_INDEX].astype(str).values.tolist()
    )
    # Convert all values in legforms_string_matrix to VruPredictionColor enum
    for i in range(len(legform_matrix)):
        for j in range(len(legform_matrix[0])):
            color_str = legform_matrix[i][j].lower()
            enum_val = next(
                (k for k in VruPredictionColor if k.value.lower() == color_str),
                VruPredictionColor.GREY,
            )
            legform_matrix[i][j] = enum_val

    for idx, row in enumerate(legform_matrix):
        logger.info("%d: %s", idx, row)

    return legform_matrix


def compute_all_legform_points(vru_prediction_df):
    """
    Convert legform matrix to a list of LegformTestPoint objects.

    Args:
        legform_matrix (list): Legform matrix with VruPredictionColor enum values.

    Returns:
        list: List of LegformTestPoint objects.
    """
    legform_matrix = get_legform_matrix(vru_prediction_df)
    legform_test_points = []

    for row in range(len(legform_matrix)):
        for col in range(len(legform_matrix[row])):
            color = legform_matrix[row][col]

            vtp = LegformTestPoint(
                row=row,
                col=10 - col,
                color=color,
            )

            if row == 0:
                vtp.loadcase_name = "Upper Leg"
            elif row >= 1:
                vtp.loadcase_name = "Lower Leg"

            legform_test_points.append(vtp)

    return legform_test_points


def process_legform_test(vru_prediction_df, vru_params, input_selected_points=None):
    """
    Process the VRU predictions DataFrame to extract legform matrices,
    compute color percentages, and randomly select samples based on color distribution.

    Args:
        vru_prediction_df (pd.DataFrame): DataFrame containing VRU predictions.

    Returns:
        None
    """

    # Ensure the DataFrame is not empty
    if vru_prediction_df.empty:
        logger.error("The input DataFrame is empty.")
        return
    logger.info(f"vru_params: {vru_params}")

    if input_selected_points is not None:
        legform_matrix = get_legform_matrix(vru_prediction_df)
        check_legform_blue_symmetry(legform_matrix)

        matched_points = select_vru_points_from_input(
            extract_legform_candidate_points(vru_prediction_df),
            input_selected_points,
        )

        # All manually supplied points are treated as BLUE_ST — no coin flip, no promotion.
        selected_row_col_pairs = [
            (point.row, 10 - point.col, VruPredictionColor.BLUE_ST)
            for point in matched_points
        ]

        # Row-2 mirror: for each row-1 selection add the corresponding row-2 cell,
        # but only when that cell is not GREY (i.e., it represents a real test point).
        for row, col, _ in list(selected_row_col_pairs):
            if row == 1 and len(legform_matrix) > 2:
                color_at_row2 = legform_matrix[2][col]
                if color_at_row2 != VruPredictionColor.GREY:
                    selected_row_col_pairs.append((2, col, color_at_row2))

        # Unselected SA cells always get a row in the verification sheet (no fillable KPI).
        # This branch never calls select_blue_pattern (manually supplied points are forced
        # to BLUE_ST above), so it's the only place that adds SA cells the OEM didn't select.
        selected_j_per_row = {(r, c) for r, c, _ in selected_row_col_pairs}
        for row_idx, matrix_row in enumerate(legform_matrix):
            for j, color in enumerate(matrix_row):
                if (
                    color == VruPredictionColor.BLUE_SA
                    and (row_idx, j) not in selected_j_per_row
                ):
                    selected_row_col_pairs.append(
                        (row_idx, j, VruPredictionColor.BLUE_SA)
                    )

        legform_test_points = []
        for row, col, color in selected_row_col_pairs:
            vtp = LegformTestPoint(row=row, col=10 - col, color=color)
            if row == 0:
                vtp.loadcase_name = "Upper Leg"
            elif row >= 1:
                vtp.loadcase_name = "Lower Leg"
            legform_test_points.append(vtp)

        return legform_test_points

    # Get number of verification tests for Pelvis and Leg separately
    num_verification_tests_pelvis = get_num_verification_tests(
        vru_params, "CP - VRU Pelvis Impact"
    )
    num_verification_tests_leg = get_num_verification_tests(
        vru_params, "CP - VRU Leg Impact"
    )
    logger.info(
        f"Number of verification tests for pelvis: {num_verification_tests_pelvis}, leg: {num_verification_tests_leg}"
    )

    legform_matrix = get_legform_matrix(vru_prediction_df)
    check_legform_blue_symmetry(legform_matrix)

    selected_row_pelvis = select_blue_pattern(
        0,
        legform_matrix,
        num_verification_tests_leg=num_verification_tests_pelvis,
    )

    # Check if all second row is blue
    # Remove GREY from legform_matrix[1] before checking if all are BLUE
    # Use select_blue_pattern for both all-blue and not-all-blue cases
    selected_row_leg = select_blue_pattern(
        1,
        legform_matrix,
        num_verification_tests_leg=num_verification_tests_leg,
    )
    # Combine pelvis and leg selections, and for each selected col in row 1 (leg), also select the same col in row 2
    selected_row_col_pairs = selected_row_pelvis + selected_row_leg

    # For each (row=1, col, color) in selected_row_leg, add (row=2, col, color_at_row2)
    for row, col, _ in selected_row_leg:
        color_at_row2 = legform_matrix[2][col]
        selected_row_col_pairs.append((2, col, color_at_row2))

    # Include remaining SA cells (not selected/promoted): they always get a row in the report
    # with a white Value cell (auto-computed from adjacent columns during score computation).
    # select_blue_pattern already returns every blue cell in an all-blue row's span, so for
    # rows 0/1/2 this is normally a no-op; kept as a defensive fallback for any row shape it
    # doesn't cover (e.g. a row that never goes through the all-blue branch).
    selected_j_per_row = {(r, c) for r, c, _ in selected_row_col_pairs}
    for row_idx, matrix_row in enumerate(legform_matrix):
        for j, color in enumerate(matrix_row):
            if (
                color == VruPredictionColor.BLUE_SA
                and (row_idx, j) not in selected_j_per_row
            ):
                selected_row_col_pairs.append((row_idx, j, VruPredictionColor.BLUE_SA))

    logger.info(
        f"Selected %d legform (row, col) points: %s",
        len(selected_row_col_pairs),
        selected_row_col_pairs,
    )

    legform_test_points = []
    for row, col, color in selected_row_col_pairs:
        # Directly create a new LegformTestPoint without searching
        vtp = LegformTestPoint(
            row=row,
            col=10 - col,
            color=color,
        )
        if row == 0:
            vtp.loadcase_name = "Upper Leg"
        elif row >= 1:
            vtp.loadcase_name = "Lower Leg"

        legform_test_points.append(vtp)

    return legform_test_points


def process_headform_vru_loadcases(vru_test_points):
    vru_head_impact_loadcases = []

    vru_points_data = []
    for point in vru_test_points:
        point_loadcase_name = point.loadcase_name
        seat_name = "Driver"
        point_body_region = point.body_region
        if point_body_region == "Cyclist":
            dummy_name = "Adult " + point_loadcase_name.split()[0]
        else:
            dummy_name = point_body_region + " " + point_loadcase_name.split()[0]
        logger.debug(
            f"I need to create: loadcase_name={point_loadcase_name}, seat_name={seat_name}, "
            f"dummy.name={dummy_name}, point.body_region={point_body_region}, "
        )
        vru_points_data.append(
            {
                "loadcase": point_loadcase_name,
                "seat": seat_name,
                "dummy": dummy_name,
                "body_region": point_body_region,
                "OEM Prediction": (
                    getattr(point, "color", None).title()
                    if getattr(point, "color", None)
                    else None
                ),
                "Test point": common.format_test_point(
                    getattr(point, "row", ""), getattr(point, "col", "")
                ),
            }
        )
    vru_points_df = pd.DataFrame(vru_points_data)
    if vru_points_df.empty:
        return vru_head_impact_loadcases

    vru_points_df = vru_points_df.sort_values(
        by=["loadcase", "dummy", "body_region"]
    ).reset_index(drop=True)

    # Split vru_points_df based on 'loadcase' column
    vru_points_dfs_by_loadcase = {
        loadcase: group.reset_index(drop=True)
        for loadcase, group in vru_points_df.groupby("loadcase")
    }

    for loadcase_name, vru_points_df in vru_points_dfs_by_loadcase.items():
        logger.debug(f"Processing loadcase: {loadcase_name}")
        adult_body_regions = []
        child_body_regions = []
        headform_seats = []
        body_region_dict = {}
        for _, row in vru_points_df.iterrows():

            body_region_name = row["body_region"]
            oem_prediction = row["OEM Prediction"]
            test_point = row["Test point"]

            if body_region_name not in body_region_dict:
                body_region = BodyRegion(name=body_region_name)
                body_region_dict[body_region_name] = body_region
            else:
                body_region = body_region_dict[body_region_name]

            if oem_prediction.lower() in ["green-20"]:
                hpl = 1000.0
                lpl = 1000.0
            elif oem_prediction.lower() in ["green-40", "green-30"]:
                hpl = 1700.0
                lpl = 1700.0
            else:
                hpl = 650.0
                lpl = 1700.0
            criteria = Criteria(name="HIC15", hpl=hpl, lpl=lpl)
            criteria.criteria_type = CriteriaType.VRU_CRITERIA
            criteria.set_prediction(oem_prediction.lower() if oem_prediction else None)
            # Write blank Value (NaN, not 0.0) so the generated sheet shows an
            # empty cell the user must fill, matching the legform sheet.  The
            # real color/score are computed at compute-score time from the
            # measured value; a seeded 0.0 would render as a literal 0 in the
            # sheet and silently score green when left unedited.
            criteria.set_value(float("nan"))
            criteria.test_point = test_point
            # Add criteria to the body region
            body_region._criteria.append(criteria)

        logger.debug(f"body_region_dict: {body_region_dict}")

        adult_headform_seat = None
        if "Adult" in body_region_dict:
            adult_body_regions.append(body_region_dict["Adult"])
        if "Cyclist" in body_region_dict:
            adult_body_regions.append(body_region_dict["Cyclist"])

        if adult_body_regions:
            adult_headform_dummy = Dummy(
                name="Adult Headform", body_region_list=adult_body_regions
            )
            adult_headform_seat = Seat(name="Driver", dummy=adult_headform_dummy)

        child_headform_seat = None
        if "Child" in body_region_dict:
            child_body_regions.append(body_region_dict["Child"])

        if child_body_regions:
            child_headform_dummy = Dummy(
                name="Child Headform", body_region_list=child_body_regions
            )
            child_headform_seat = Seat(name="Driver", dummy=child_headform_dummy)

        # Always create one seat "Driver" with dummy "Adult Headform" (Adult + Cyclist) and one with "Child Headform" (Child)
        headform_seats = []
        if adult_headform_seat is not None:
            headform_seats.append(adult_headform_seat)
        if child_headform_seat is not None:
            headform_seats.append(child_headform_seat)

        headform_loadcase = LoadCase(name=loadcase_name, seats=headform_seats)

        vru_head_impact_loadcases.append(headform_loadcase)

    return vru_head_impact_loadcases


def process_legform_criteria(criteria, test_point, oem_prediction):
    criteria.criteria_type = CriteriaType.VRU_CRITERIA
    # Write blank Value so the user fills T/ST cells and SA cells are auto-computed
    # from adjacent T/ST values at compute-score time.  Using NaN (not 0.0) ensures
    # that _resolve_legform_blue_values detects SA as "not yet resolved" and looks up
    # the adjacent columns instead of skipping the SA criteria.
    criteria.set_value(float("nan"))
    criteria.test_point = test_point
    criteria.set_prediction(oem_prediction.lower() if oem_prediction else None)


def _get_lower_leg_body_region_names(point):
    if point.row == 1:
        return ["Femur"]
    if point.row == 2:
        return ["Knee", "Tibia"]

    logger.warning(
        "Unexpected lower-leg test point row %s for column %s; keeping legacy body-region mapping.",
        point.row,
        point.col,
    )
    return ["Femur", "Knee", "Tibia"]


def process_legform_vru_loadcases(vru_test_points):
    vru_legform_loadcases = []
    # Group test points by loadcase
    loadcase_points = {}
    for point in vru_test_points:
        loadcase_points.setdefault(point.loadcase_name, []).append(point)

    upper_legform_criteria = []

    for loadcase_name, points in loadcase_points.items():
        seat_name = "Driver"
        if loadcase_name == "Upper Leg":
            dummy_name = "Upper legform"
            body_region_names = ["Pelvis"]
        elif loadcase_name == "Lower Leg":
            dummy_name = "aPLI"
            body_region_names = ["Femur", "Knee", "Tibia"]
        else:
            continue  # skip unknown loadcase

        seats = []
        if loadcase_name == "Lower Leg":
            body_region_dict = {
                name: BodyRegion(name=name) for name in body_region_names
            }
        else:
            body_region_dict = None

        for point in points:
            current_body_region_names = body_region_names
            if loadcase_name == "Lower Leg":
                current_body_region_names = _get_lower_leg_body_region_names(point)
            else:
                body_region_dict = {
                    name: BodyRegion(name=name) for name in current_body_region_names
                }

            oem_prediction = (
                getattr(point, "color", None).title()
                if getattr(point, "color", None)
                else None
            )
            test_point = common.format_test_point(
                getattr(point, "row", ""), getattr(point, "col", "")
            )

            # Assign criteria to each body region as per Euro NCAP protocol
            if "Pelvis" in current_body_region_names:
                criteria = Criteria(name="Sum of forces", hpl=5.00, lpl=6.00)
                process_legform_criteria(criteria, test_point, oem_prediction)
                upper_legform_criteria.append(criteria)
            if "Femur" in current_body_region_names:
                for f in ["F1", "F2", "F3"]:
                    criteria = Criteria(
                        name=f"Bending moment, {f}", hpl=390.00, lpl=440.00
                    )
                    process_legform_criteria(criteria, test_point, oem_prediction)
                    body_region_dict["Femur"]._criteria.append(criteria)
            if "Knee" in current_body_region_names:
                criteria = Criteria(name="MCL elongation", hpl=27.00, lpl=32.00)
                process_legform_criteria(criteria, test_point, oem_prediction)
                body_region_dict["Knee"]._criteria.append(criteria)
            if "Tibia" in current_body_region_names:
                for t in ["T1", "T2", "T3", "T4"]:
                    criteria = Criteria(
                        name=f"Bending moment, {t}", hpl=275.00, lpl=320.00
                    )
                    process_legform_criteria(criteria, test_point, oem_prediction)
                    body_region_dict["Tibia"]._criteria.append(criteria)
            if loadcase_name != "Lower Leg":
                body_regions = [
                    body_region_dict[name]
                    for name in current_body_region_names
                    if name in body_region_dict
                ]
                if body_regions:
                    dummy = Dummy(name=dummy_name, body_region_list=body_regions)
                    seat = Seat(name=seat_name, dummy=dummy)
                    seats.append(seat)

        # Log body_region_dict in order
        if loadcase_name == "Lower Leg":
            body_regions = [
                body_region_dict[name]
                for name in ["Femur", "Knee", "Tibia"]
                if name in body_region_dict and body_region_dict[name]._criteria
            ]
            if body_regions:
                dummy = Dummy(name=dummy_name, body_region_list=body_regions)
                seat = Seat(name=seat_name, dummy=dummy)
                seats.append(seat)

        if seats:
            for seat in seats:
                for body_region in seat.dummy.body_region_list:
                    logger.debug(f"body_region_dict[{body_region.name}]: {body_region}")
                    for criteria in body_region._criteria:
                        logger.debug(f"    Criteria: {criteria}")

        if loadcase_name == "Upper Leg":
            # For Upper Leg: 1 seat, 1 dummy, 1 body region (Pelvis) with all criteria
            # Create the Pelvis body region and assign all upper_legform_criteria to it
            pelvis_body_region = BodyRegion(name="Pelvis")
            pelvis_body_region._criteria = upper_legform_criteria
            body_regions = [pelvis_body_region]
            dummy = Dummy(name=dummy_name, body_region_list=body_regions)
            seat = Seat(name=seat_name, dummy=dummy)
            seats.append(seat)

        legform_loadcase = LoadCase(name=loadcase_name, seats=seats)
        vru_legform_loadcases.append(legform_loadcase)

    return vru_legform_loadcases
