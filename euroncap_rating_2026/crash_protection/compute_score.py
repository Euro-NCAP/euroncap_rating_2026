# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from dataclasses import dataclass
from typing import List, Optional, Dict
import pandas as pd
import logging
import openpyxl

from euroncap_rating_2026.crash_protection import vru_processing
from euroncap_rating_2026.crash_protection import data_loader
from euroncap_rating_2026.crash_protection import integrity
from euroncap_rating_2026 import common
from euroncap_rating_2026 import config
import sys
from pandas.api.types import is_string_dtype
import os
from datetime import datetime
from importlib.resources import files
from euroncap_rating_2026.crash_protection.report_writer import write_report

import click


logger = logging.getLogger(__name__)
settings = config.Settings()


def get_current_loadcase_id(df, index):
    current_loadcase_id = df.loc[index, "Loadcase"]
    if not pd.isna(df.iloc[index]["Seat position"]):
        current_loadcase_id += f"_{df.iloc[index]['Seat position']}"
    if not pd.isna(df.iloc[index]["Dummy"]):
        current_loadcase_id += f"_{df.iloc[index]['Dummy']}"

    # Then process the remaining rows
    for i in range(index + 1, len(df)):
        next_row = df.iloc[i]
        if not pd.isna(next_row["Loadcase"]):
            break
        if not pd.isna(next_row["Seat position"]):
            current_loadcase_id += f"_{next_row['Seat position']}"
        if not pd.isna(next_row["Dummy"]):
            current_loadcase_id += f"_{next_row['Dummy']}"
    return current_loadcase_id


def get_loadcase_id_from_seats(loadcase_name, seats):
    id_parts = [loadcase_name]
    for seat in seats:
        id_parts.append(seat.name)
        id_parts.append(seat.dummy.name)
    return "_".join(id_parts)


def update_loadcase(df, loadcase):
    last_load_case = None
    last_seat_name = None
    last_dummy_name = None
    last_body_region = None
    last_test_point = None

    if len(loadcase.raw_seats) > 0:
        logger.debug(
            f"Processing loadcase: {loadcase.name} with {len(loadcase.raw_seats)} raw seats"
        )
        logger.debug(f"raw_seats: {[s.name for s in loadcase.raw_seats]}")
    # Ensure the DataFrame has the "Score" and "Capping?" columns
    if "Score" not in df.columns:
        df["Score"] = ""
    if "Capping?" not in df.columns:
        df["Capping?"] = ""

    for column in ["Colour", "Capping?", "Prediction.Check"]:
        if column in df.columns and not is_string_dtype(df[column]):
            df[column] = df[column].astype(str)

    for index, row in df.iterrows():
        if not pd.isna(row["Loadcase"]):
            last_load_case = row["Loadcase"]
            current_loadcase_id = get_current_loadcase_id(df, index)
        if not pd.isna(row["Seat position"]):
            last_seat_name = row["Seat position"]
        if not pd.isna(row["Dummy"]):
            last_dummy_name = row["Dummy"]
        if not pd.isna(row["Body region"]):
            last_body_region = row["Body region"]
        if "Test point" in df.columns and not pd.isna(row["Test point"]):
            last_test_point = row["Test point"]
        valid_loadcase_ids = {loadcase.id}
        if hasattr(loadcase, "raw_seats") and len(loadcase.raw_seats) > 0:
            valid_loadcase_ids.add(
                get_loadcase_id_from_seats(loadcase.name, loadcase.raw_seats)
            )

        if current_loadcase_id not in valid_loadcase_ids:
            continue

        criteria = row["Criteria"]
        # Use loadcase.raw_seats if available, otherwise use loadcase.seats
        seat_list = (
            loadcase.raw_seats
            if hasattr(loadcase, "raw_seats") and len(loadcase.raw_seats) > 0
            else loadcase.seats
        )
        logger.debug(f"seat_list: {[f'{s.name} ({s.dummy.name})' for s in seat_list]}")
        seat = next(
            (
                s
                for s in seat_list
                if s.name == last_seat_name and s.dummy.name == last_dummy_name
            ),
            None,
        )
        if seat is None:
            logger.debug(
                f"Seat {last_seat_name} with dummy {last_dummy_name} not found in loadcase {loadcase.name}"
            )
            return df

        body_region = next(
            (br for br in seat.dummy.body_region_list if br.name == last_body_region),
            None,
        )
        if body_region is None:
            logger.debug(
                f"Body region {last_body_region} not found in dummy {seat.dummy.name}"
            )
            return df

        if "Test point" in df.columns:
            # Try to find criteria_obj in the current body_region first
            criteria_obj = next(
                (
                    c
                    for c in body_region._criteria
                    if c.name == criteria and c.test_point == last_test_point
                ),
                None,
            )
            # If not found, search all seats, dummies, body regions, and criteria
            if criteria_obj is None:
                for seat_search in seat_list:
                    for body_region_search in seat_search.dummy.body_region_list:
                        for c in body_region_search._criteria:
                            if c.name == criteria and c.test_point == last_test_point:
                                criteria_obj = c
                                break
                        if criteria_obj is not None:
                            break
                    if criteria_obj is not None:
                        break
        else:
            criteria_obj = next(
                (c for c in body_region._criteria if c.name == criteria),
                None,
            )
        logger.debug(f"Loadcase: {loadcase.id}, Criteria object: {criteria_obj}")
        if criteria_obj:
            df.loc[index, "HPL"] = criteria_obj.hpl
            df.loc[index, "LPL"] = criteria_obj.lpl
            pred = getattr(criteria_obj, "prediction", None)
            # SA legform values are inferred from adjacent columns and A-pillar
            # (green-20/30/40) values may be copied from the symmetric run, so
            # the resolved value is written back for a self-explanatory report.
            if (
                pred
                and str(pred).lower() in ("sa", "green-20", "green-30", "green-40")
                and not pd.isna(criteria_obj.value)
            ):
                df.loc[index, "Value"] = criteria_obj.value
            df.loc[index, "Colour"] = criteria_obj.color
            df.loc[index, "Score"] = criteria_obj.score
            df.loc[index, "Capping?"] = "YES" if criteria_obj.capping else ""
            if "Prediction.Check" in df.columns:
                # Written unconditionally: a criteria whose assessment was
                # cleared (prediction_result None, e.g. HIC15/Ares-3ms below
                # the 80 g Ares gate) must blank any stale value already in
                # the cell when a previously scored dataframe is re-scored.
                prediction_result = criteria_obj.prediction_result
                df.loc[index, "Prediction.Check"] = (
                    "".join(word.capitalize() for word in prediction_result.split())
                    if prediction_result
                    else ""
                )
            logger.debug(
                f"Updated row - Loadcase: {current_loadcase_id}, Seat: {last_seat_name}, Dummy: {last_dummy_name}, Body region: {last_body_region}, Criteria: {criteria}, Colour: {criteria_obj.color}, Score: {criteria_obj.score}, Capping: {criteria_obj.capping}, Prediction: {criteria_obj.prediction_result}"
            )
        else:
            measurement_obj = next(
                (m for m in body_region._measurement if m.name == criteria),
                None,
            )
            if measurement_obj and "Colour" in df.columns:
                df.loc[index, "Colour"] = measurement_obj.color
                logger.debug(
                    f"Updated measurement row - Criteria: {criteria}, Colour: {measurement_obj.color}"
                )
    return df


def update_dummy_scores(df, loadcase):

    last_load_case = None
    last_seat_name = None
    last_dummy_name = None

    for column in ["Capping?"]:
        if column in df.columns and not is_string_dtype(df[column]):
            df[column] = df[column].astype(str)

    for index, row in df.iterrows():
        if not pd.isna(row["Loadcase"]):
            last_load_case = row["Loadcase"]
            current_loadcase_id = get_current_loadcase_id(df, index)
        if not pd.isna(row["Seat position"]):
            last_seat_name = row["Seat position"]
        if not pd.isna(row["Dummy"]):
            last_dummy_name = row["Dummy"]

        if current_loadcase_id != loadcase.id:
            continue

        seat = next(
            (
                s
                for s in loadcase.seats
                if s.name == last_seat_name and s.dummy.name == last_dummy_name
            ),
            None,
        )
        if seat is None:
            logger.debug(
                f"Seat {last_seat_name} with dummy {last_dummy_name} not found in loadcase {loadcase.name}"
            )
            return

        if seat.dummy.score is not None:
            df.loc[index, "Score"] = seat.dummy.score
        if seat.dummy.max_score is not None:
            df.loc[index, "Max score"] = seat.dummy.max_score
        df.loc[index, "Capping?"] = "Capped" if seat.dummy.capping else ""

        logger.debug(
            f"Updated row - Loadcase: {last_load_case}, Seat: {last_seat_name}, Dummy: {last_dummy_name}, Score: {seat.dummy.score}"
        )

    return df


def update_bodyregion(df, loadcase):

    last_load_case = None
    last_seat_name = None
    last_dummy_name = None
    last_body_region = None

    for index, row in df.iterrows():
        if not pd.isna(row["Loadcase"]):
            last_load_case = row["Loadcase"]
            current_loadcase_id = get_current_loadcase_id(df, index)
        if not pd.isna(row["Seat position"]):
            last_seat_name = row["Seat position"]
        if not pd.isna(row["Dummy"]):
            last_dummy_name = row["Dummy"]
        if not pd.isna(row["Body region"]):
            last_body_region = row["Body region"]

        if current_loadcase_id != loadcase.id:
            continue

        seat = next(
            (
                s
                for s in loadcase.seats
                if s.name == last_seat_name and s.dummy.name == last_dummy_name
            ),
            None,
        )
        if seat is None:
            logger.debug(
                f"Seat {last_seat_name} with dummy {last_dummy_name} not found in loadcase {loadcase.name}"
            )
            return df

        body_region = next(
            (br for br in seat.dummy.body_region_list if br.name == last_body_region),
            None,
        )
        if body_region is None:
            logger.debug(
                f"Body region {last_body_region} not found in dummy {seat.dummy.name}"
            )
            return df

        # Synthesized placeholder regions (no criteria, no scores - e.g. a WAD
        # band with no coloured grid points) must leave the template's cells
        # untouched instead of overwriting them with empty values.
        if (
            body_region.bodyregion_score is None
            and body_region.score is None
            and not body_region._criteria
            and not body_region._measurement
        ):
            continue

        df.loc[index, "Body regionscore"] = body_region.bodyregion_score
        df.loc[index, "Score"] = body_region.score
        df.loc[index, "Modifiers"] = sum(
            measurement.modifier
            for measurement in body_region._measurement
            if measurement.modifier is not None
        )
        if body_region.max_score is not None:
            df.loc[index, "Max score"] = body_region.max_score
        logger.debug(
            f"Updated row - Loadcase: {last_load_case}, Seat: {last_seat_name}, Dummy: {last_dummy_name}, Body region: {last_body_region}, Score: {body_region.score}"
        )
    return df


def update_stage_scores(df, final_scores):

    last_stage_subelement = None

    for index, row in df.iterrows():

        if not pd.isna(row["Stage subelement"]):
            last_stage_subelement = row["Stage subelement"]

        for key in final_scores:
            if key == last_stage_subelement:
                df.loc[index, "Score"] = final_scores[key]

    return df


def print_score(final_scores, final_max_scores, overall_score, overall_max_score):
    print("-" * 40)
    print("Score:")
    print("-" * 40)
    print(
        f"{'Stage Element':<20}{'Stage subelement':<20}{'Score':<10}{'Max score':<10}"
    )
    print("-" * 40)
    score_order = [
        ("Frontal Impact", ["Offset", "FW", "Sled & VT"]),
        ("Side Impact", ["MDB", "Pole", "Farside"]),
        ("Rear Impact", ["Whiplash"]),
        ("VRU Impact", ["Head Impact", "Pelvis & Leg Impact"]),
    ]

    printed_categories = set()
    for category, subcategories in score_order:
        for subcategory in subcategories:
            if subcategory in final_scores:
                logger.info(
                    f"Final score for {subcategory}: {final_scores[subcategory]}/{final_max_scores[subcategory]}"
                )
                category_to_print = (
                    category if category not in printed_categories else ""
                )
                print(
                    f"{category_to_print:<20}{subcategory:<20}{final_scores[subcategory]:<10}{final_max_scores[subcategory]:<10}"
                )
                printed_categories.add(category)
            else:
                logger.warning(f"Score for {subcategory} not found.")
                category_to_print = (
                    category if category not in printed_categories else ""
                )
                print(f"{category_to_print:<20}{subcategory:<20}{'N/A':<10}{'N/A':<10}")
                printed_categories.add(category)

    print("-" * 40)
    print(" " * 40)

    overall_str = "Final score"
    print(f"{overall_str:<20}{overall_score:<10}{overall_max_score:<10}")
    print(" " * 40)


def get_updated_dfs(dfs, sheet_dict, final_scores):
    """Prepare and update score dataframes with computed values.

    Args:
        dfs: Dictionary of dataframes from the input Excel file
        sheet_dict: Dictionary of sheet data with loadcases
        final_scores: Dictionary of final computed scores

    Returns:
        dict: updated_dfs containing updated dataframes
    """
    updated_dfs = {}
    for sheet in sheet_dict.keys():
        if sheet != "CP - VRU Prediction" and sheet in dfs:
            updated_dfs[sheet] = dfs[sheet].copy()
    updated_dfs["CP - Body region scores"] = dfs["CP - Body region scores"].copy()
    updated_dfs["CP - Dummy Scores"] = dfs["CP - Dummy Scores"].copy()
    updated_dfs["Test Scores"] = dfs["Test Scores"].copy()
    # Update score dataframes with computed values
    for sheet_name in sheet_dict:
        if sheet_name in dfs:
            df = dfs[sheet_name].copy()

            logger.debug(f"updated_dfs keys: {list(updated_dfs.keys())}")
            logger.debug(f"sheet_dict keys: {list(sheet_dict.keys())}")

            for i, loadcase_df in enumerate(sheet_dict[sheet_name]):
                df = update_loadcase(df, loadcase_df)
                logger.debug(f"Processing loadcase_df: {loadcase_df}")
                updated_dfs["CP - Body region scores"] = update_bodyregion(
                    updated_dfs["CP - Body region scores"], loadcase_df
                )
                updated_dfs["CP - Dummy Scores"] = update_dummy_scores(
                    updated_dfs["CP - Dummy Scores"], loadcase_df
                )

            updated_dfs[sheet_name] = df

    updated_dfs["Test Scores"] = update_stage_scores(
        updated_dfs["Test Scores"], final_scores
    )

    return updated_dfs


@dataclass
class ModifierInfo:
    dummy: str
    seat_position: str
    body_region: str
    score: float
    modifier_name: str


def get_modifiers(
    dfs: Dict[str, pd.DataFrame], loadcase_name: str
) -> List[ModifierInfo]:
    sheet_dict, _ = data_loader.load_data(dfs)
    loadcase_measurements = []
    for _, load_cases in sheet_dict.items():
        for load_case in load_cases:
            if load_case.name == loadcase_name:
                for seat in load_case.seats:
                    for body_region in seat.dummy.body_region_list:
                        body_region_measurements = body_region.get_measurement_list()
                        for measurement in body_region_measurements:
                            modifier_info = ModifierInfo(
                                dummy=seat.dummy.name,
                                seat_position=seat.name,
                                body_region=body_region.name,
                                score=measurement.modifier,
                                modifier_name=measurement.name,
                            )
                            loadcase_measurements.append(modifier_info)
                break

    return loadcase_measurements


@dataclass
class VruHeadformRegionKpi:
    body_region: str  # "Child" | "Adult" | "Cyclist"
    color_counts: Dict[str, int]
    predicted_score: float
    max_points: float
    # Tested bonus credit, through the same code path as scoring
    # (VruScore.compute_blue_points / compute_a_pillar). All three are 0.0
    # when no verification Value is filled in (see get_vru_headform_kpis).
    blue_points: float = 0.0  # blue-cell credit, sum(tested score) / 100
    a_pillar_bonus: float = 0.0  # green-20/30/40 credit, weighted x1/x2/x3
    bonus: float = 0.0  # a_pillar_bonus + blue_points


@dataclass
class VruHeadformKpis:
    regions: List[VruHeadformRegionKpi]  # fixed order: Child, Adult, Cyclist
    correction_factor: float  # 1.0 when no verification value is filled in
    predicted_score_verification: float
    tested_score_verification: float


# The colour breakdown shown per body region. The whole blue family (blue and
# the t/st/sa variants, which share one grid fill colour) is counted under the
# single "blue" key. Note the count is the grid colour family, while the
# blue_points credit follows scoring, which pays out on literal blue cells
# only.
_VRU_KPI_COLOR_KEYS = [
    vru_processing.VruPredictionColor.D_GREEN.value,
    vru_processing.VruPredictionColor.GREEN.value,
    vru_processing.VruPredictionColor.YELLOW.value,
    vru_processing.VruPredictionColor.ORANGE.value,
    vru_processing.VruPredictionColor.BROWN.value,
    vru_processing.VruPredictionColor.RED.value,
    vru_processing.VruPredictionColor.D_RED.value,
    vru_processing.VruPredictionColor.GREEN_20.value,
    vru_processing.VruPredictionColor.GREEN_30.value,
    vru_processing.VruPredictionColor.GREEN_40.value,
    vru_processing.VruPredictionColor.BLUE.value,
]

_VRU_KPI_BLUE_VARIANTS = {
    vru_processing.VruPredictionColor.BLUE.value,
    vru_processing.VruPredictionColor.BLUE_T.value,
    vru_processing.VruPredictionColor.BLUE_ST.value,
    vru_processing.VruPredictionColor.BLUE_SA.value,
}


def get_vru_headform_kpis(dfs: Dict[str, pd.DataFrame]) -> VruHeadformKpis:
    """Read the VRU headform KPIs from a workbook's dataframes.

    Programmatic accessor (no CLI wiring): returns, per WAD body region, the colour
    counts of the headform prediction matrix and the predicted score, plus the
    whole-test correction factor derived from the verification loadcases -
    computed through the same code path as scoring (VruTestData), so the
    factor always equals the one used in the score.

    Grid points are taken from the "CP - VRU Prediction Points" sheet when
    present (the add_vru_sheets_to_dfs output carries it); otherwise they are
    parsed from the "CP - VRU Prediction" cell texts, which is only reliable
    for workbooks whose prediction cells were not yet overwritten with the
    X/T/ST/SA selection marks at preprocess time. For an on-disk preprocessed
    file use get_vru_headform_kpis_in_file, which reads the cell fill colours.

    When no verification Value is filled in (all blank or the seeded 0.0), the
    correction factor is 1.0 - no correction applied - and the tested
    verification sum is 0.0.

    The per-region bonus fields (blue_points, a_pillar_bonus, bonus) carry the
    tested blue and A-pillar (green-20/30/40) credit, computed through the
    same VruScore path as scoring - so they equal the credit added to the
    body-region score numerator (a region already at the 100-point cap
    realises less of it). One deliberate difference from scoring: when no
    verification Value is filled in, all three fields are 0.0 - the same
    neutral fallback as the correction factor - whereas scoring would treat
    the preprocess-seeded 0.0 Values as tested. Without a head-impact sheet
    they are also 0.0.
    """
    if "CP - VRU Prediction Points" in dfs:
        points = vru_processing.load_vru_prediction_points(
            dfs["CP - VRU Prediction Points"]
        )
    else:
        points = vru_processing.compute_all_vru_test_points(dfs["CP - VRU Prediction"])

    vru_test_data = None
    has_verification_values = False
    head_impact_df = dfs.get("CP - VRU Head Impact")
    if head_impact_df is not None and "Loadcase" in head_impact_df.columns:
        loadcases = data_loader.get_loadcases_from_df(head_impact_df)
        vru_test_data = vru_processing.VruTestData.from_sheet_dict(
            {"CP - VRU Head Impact": loadcases}
        )
        # A Value of 0.0 is the preprocess-seeded placeholder, not a filled
        # measurement; a workbook where nothing else is filled carries no
        # verification data at all.
        has_verification_values = any(
            not (pd.isna(criteria.value) or criteria.value == 0.0)
            for load_case in vru_test_data.headform_loadcases
            if load_case.name == "Headform"
            for seat in load_case.seats
            for body_region in seat.dummy.body_region_list
            for criteria in body_region._criteria
        )

    regions = []
    for region in ("Child", "Adult", "Cyclist"):
        vru_score = vru_processing.compute_vru_score(points, region)
        if vru_test_data is not None and has_verification_values:
            vru_score.compute_blue_points(vru_test_data.headform_loadcases)
            vru_score.compute_a_pillar(vru_test_data.headform_loadcases)
        color_counts = {key: 0 for key in _VRU_KPI_COLOR_KEYS}
        for point in points:
            if point.body_region != region:
                continue
            # Points arrive as VruPredictionColor on the CLI path but as plain
            # strings after a parquet round-trip.
            color_key = str(getattr(point.color, "value", point.color)).lower()
            if color_key in _VRU_KPI_BLUE_VARIANTS:
                color_key = vru_processing.VruPredictionColor.BLUE.value
            if color_key in color_counts:
                color_counts[color_key] += 1
        regions.append(
            VruHeadformRegionKpi(
                body_region=region,
                color_counts=color_counts,
                predicted_score=vru_score.predicted_score,
                max_points=vru_score.max_points,
                blue_points=vru_score.blue_points,
                a_pillar_bonus=vru_score.a_pillar,
                bonus=vru_score.a_pillar + vru_score.blue_points,
            )
        )

    correction_factor = 1.0
    predicted_score_verification = 0.0
    tested_score_verification = 0.0
    if vru_test_data is not None:
        factors = vru_processing.get_vru_factors(vru_test_data.headform_loadcases)
        predicted_score_verification = factors["predicted_score_verification"]
        if has_verification_values:
            correction_factor = float(factors["correction_factor"])
            tested_score_verification = factors["tested_score_verification"]

    return VruHeadformKpis(
        regions=regions,
        correction_factor=correction_factor,
        predicted_score_verification=predicted_score_verification,
        tested_score_verification=tested_score_verification,
    )


def get_vru_headform_kpis_in_file(input_file: str) -> VruHeadformKpis:
    """File-path twin of get_vru_headform_kpis.

    Reads the grid points from the "CP - VRU Prediction" cell fill colours
    (the source compute-score itself uses, valid even after the prediction
    texts were replaced by selection marks at preprocess time), falling back
    to the cell texts when the fill scan finds no coloured cell.
    """
    dfs = common.read_excel_file_to_dfs(input_file)
    vru_points_key, vru_points_df = vru_processing.get_vru_point_df(
        input_file, dfs_dict=dfs
    )
    has_colored_fill = not vru_points_df.empty and any(
        str(getattr(color, "value", color)).lower()
        != vru_processing.VruPredictionColor.GREY.value
        for color in vru_points_df["color"]
    )
    if has_colored_fill:
        dfs[vru_points_key] = vru_points_df
    return get_vru_headform_kpis(dfs)


def calculate_score(dfs: dict) -> dict:
    """Calculate NCAP scores from input dataframes.

    Args:
        dfs: Dictionary of dataframes from the input Excel file

    Returns:
        dict: Dictionary containing score dataframes and metadata
    """
    # Normalize column casing in CP - Body region scores so parquet-sourced and
    # Excel-sourced dataframes both work (e.g. 'Body Region' -> 'Body region').
    if "CP - Body region scores" in dfs:
        _canonical = {"Body Region": "Body region", "Max Score": "Max score"}
        dfs = dict(dfs)
        dfs["CP - Body region scores"] = dfs["CP - Body region scores"].rename(
            columns=_canonical
        )

    # Re-derive every protected reference cell (HPL, LPL, Capping, Max score,
    # row labels, ...) from the official packaged template instead of trusting
    # it from the caller's dfs -- callers that only ever pass DataFrames (no
    # file, no preprocess() call) never went through rebuild_trusted_input, so
    # this must happen unconditionally here rather than being assumed done
    # upstream. Only the designated grey input cells are carried over.
    dfs = integrity.rebuild_trusted_input_dfs(dfs)

    sheet_dict, test_score_inspection = data_loader.load_data(dfs)
    logger.info(f"Loaded sheet_dict: {sheet_dict.keys()}")

    # Compute scores
    print("Computing NCAP scores...")
    overall_score, overall_max_score, final_scores, final_max_scores = (
        data_loader.get_score(sheet_dict, test_score_inspection)
    )

    updated_dfs = get_updated_dfs(dfs, sheet_dict, final_scores)
    print_score(final_scores, final_max_scores, overall_score, overall_max_score)

    result_dict = {}
    result_dict.update(updated_dfs)
    # Add sheet_dict dataframes to result_dict
    for key in dfs.keys():
        if key in sheet_dict:
            if key not in result_dict:
                result_dict[key] = dfs[key]

    return result_dict


@click.command()
@common.with_footer
@click.option(
    "--input_file",
    "-i",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the input Excel file containing NCAP test measurements.",
)
@click.option(
    "--output_path",
    "-o",
    type=click.Path(file_okay=False, writable=True),
    default=os.getcwd(),
    show_default=True,
    help="Path to the output directory where the report will be saved.",
)
def compute_score(input_file, output_path):
    """Compute NCAP scores from an input Excel file."""
    common.check_version(input_file, common.CliCommand.PREPROCESS)
    print(f"[Crash Protection] Computing NCAP scores from input Excel file...")

    if not input_file:
        logger.error("Input file path is required.")
        sys.exit(1)
    if not input_file.endswith(".xlsx"):
        logger.error("Input file must be an Excel file with .xlsx extension.")
        sys.exit(1)

    # Verify the preprocessed file's protected reference cells (HPL, LPL,
    # Capping, Max score, row labels, ...) haven't been edited since
    # 'preprocess' produced it.
    wb_to_verify = openpyxl.load_workbook(input_file, data_only=False)
    try:
        common.verify_protected_signature(wb_to_verify, integrity.SHEET_SCHEMAS)
    except common.TemplateIntegrityError as e:
        # Files preprocessed by an older version legitimately fail this
        # check (no Integrity sheet yet, or a signature computed over a
        # different schema revision), so it cannot be a hard gate.
        logger.warning(f"{e} Proceeding anyway for backward compatibility.")
        print(f"Warning: {e}")
        print(
            "Proceeding anyway for backward compatibility with files "
            "produced by older versions..."
        )
    finally:
        wb_to_verify.close()

    print("Loading data from spreadsheet...")

    # Read data from file
    dfs_dict = common.read_excel_file_to_dfs(input_file)
    vru_points_key, vru_points_df = vru_processing.get_vru_point_df(
        input_file, dfs_dict=dfs_dict
    )
    dfs_dict[vru_points_key] = vru_points_df

    result_dict = calculate_score(dfs_dict)

    # Separate result_dict back into score_df_dict and sheet_dict
    updated_dfs = result_dict

    write_report(
        common.CliCommand.COMPUTE_SCORE,
        input_file,
        updated_dfs,
        output_path=output_path,
        format_prediction_cells=False,
    )
