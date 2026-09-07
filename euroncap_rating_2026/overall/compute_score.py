# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from dataclasses import dataclass
import sys
import os
import glob
import io
import pandas as pd
import numpy as np
import logging
import click

from euroncap_rating_2026.common import with_footer
from euroncap_rating_2026 import config
from euroncap_rating_2026.numeric import floor_score
from euroncap_rating_2026 import common
from euroncap_rating_2026.overall import report_writer
from euroncap_rating_2026.overall import sheets


logger = logging.getLogger(__name__)
settings = config.Settings()


# Copy specific sheets to the output file

BODY_REGION_BLUE_ROWS_CSV_STRING = """Stage subelement;Loadcase;Seat position;Dummy;Body region
Offset;MPDB-50;Driver;THOR-50;Head & Neck
Offset;MPDB-50;Driver;THOR-50;Chest & Abdomen
Offset;MPDB-50;Front Passenger;HIII-05;Head & Neck
Offset;MPDB-50;Front Passenger;HIII-05;Chest & Abdomen
Offset;MPDB-50;Behind Driver;Q6;Head
Offset;MPDB-50;Behind Driver;Q6;Neck
Offset;MPDB-50;Behind Driver;Q6;Chest
Offset;MPDB-50;Behind Passenger;Q10;Head
Offset;MPDB-50;Behind Passenger;Q10;Neck
Offset;MPDB-50;Behind Passenger;Q10;Chest
FW;FWDB-35;Driver;HIII-05;Head & Neck
FW;FWDB-35;Driver;HIII-05;Chest & Abdomen
FW;FWDB-35;Front Passenger;THOR-50;Head & Neck
FW;FWDB-35;Front Passenger;THOR-50;Chest & Abdomen
FW;FWDB-35;Behind Passenger;HIII-05;Head & Neck
FW;FWDB-35;Behind Passenger;HIII-05;Chest & Abdomen
Sled & VT;Sled-Mid;Driver;HIII-50;Head & Neck
Sled & VT;Sled-Mid;Driver;HIII-50;Chest & Abdomen
Sled & VT;Sled-Mid;Front Passenger;HIII-95;Head & Neck
Sled & VT;Sled-Mid;Front Passenger;HIII-95;Chest & Abdomen
Sled & VT;Sled-High;Driver;HIII-95;Head & Neck
Sled & VT;Sled-High;Driver;HIII-95;Chest & Abdomen
Sled & VT;Sled-High;Front Passenger;HIII-05;Head & Neck
Sled & VT;Sled-High;Front Passenger;HIII-05;Chest & Abdomen
MDB;AEMDB-60;Driver;WorldSID-50;Head
MDB;AEMDB-60;Driver;WorldSID-50;Chest
MDB;AEMDB-60;Driver;WorldSID-50;Abdomen
MDB;AEMDB-60;Driver;WorldSID-50;Pelvis
MDB;AEMDB-60;Behind Driver;Q10-Side;Head
MDB;AEMDB-60;Behind Driver;Q10-Side;Neck
MDB;AEMDB-60;Behind Passenger;Q6-Side;Head
MDB;AEMDB-60;Behind Passenger;Q6-Side;Neck
Pole;Pole-32;Driver;WorldSID-50;Head
Pole;Pole-32;Driver;WorldSID-50;Chest
Pole;Pole-32;Driver;WorldSID-50;Abdomen
Pole;Pole-32;Driver;WorldSID-50;Pelvis
Farside;Pole-32;Driver and Front Passenger;WorldSID-50-Farside;Head
Whiplash;Rear-Mid;Driver;BioRID-50;Neck
Whiplash;Rear-High;Driver;BioRID-50;Neck
Head Impact;Headform;Driver;Adult Headform;Cyclist
Head Impact;Headform;Driver;Adult Headform;Adult
Head Impact;Headform;Driver;Child Headform;Child
Pelvis & Leg Impact;Upper Leg;Driver;Upper legform;Pelvis
Pelvis & Leg Impact;Lower Leg;Driver;aPLI;Femur
Pelvis & Leg Impact;Lower Leg;Driver;aPLI;Knee & Tibia
"""


PREFIX_TO_DOMAIN = {
    "cp": "Crash Protection",
    "ca": "Crash Avoidance",
    "sd": "Safe Driving",
    "pc": "Post-Crash",
}


@dataclass
class CompensationInfo:
    sd_to_ca: int
    ca_to_cp: int


def get_sheet_from_report(report_path, sheet_name="Test Scores"):
    sheet_df = None
    if report_path:
        # engine_kwargs={"read_only": False}: see common.read_excel_file_to_dfs
        # -- avoids pandas' read_only-mode openpyxl engine, whose streaming
        # reader can iterate millions of phantom cells on a sheet with an
        # inflated XML <dimension> tag. Same output, much faster load.
        sheet_df = pd.read_excel(
            report_path,
            sheet_name=sheet_name,
            header=0,
            engine_kwargs={"read_only": False},
        )
    else:
        logger.warning(f"Report path is None, cannot read sheet {sheet_name}.")
    return sheet_df


def get_report_path_with_prefix(reports_path, prefix):
    reports = glob.glob(os.path.join(reports_path, f"{prefix}_*_report.xlsx"))
    if len(reports) == 0:
        logger.warning(
            f"No {prefix} report found in {reports_path}. {PREFIX_TO_DOMAIN.get(prefix, prefix)} scores will not be included in the overall score."
        )
    elif len(reports) > 1:
        logger.warning(
            f"Multiple {prefix} reports found in {reports_path}. Only the first one will be used for overall score computation."
        )
    report = reports[0] if len(reports) > 0 else None
    logger.debug(f"Report path for prefix '{prefix}': {report}")
    return report


def update_test_scores_with_report(test_scores_df, report_df):
    if report_df is not None:
        for _, row in report_df.iterrows():
            stage_element = row.get("Stage element")
            stage_subelement = row.get("Stage subelement")
            score = row.get("Score")
            logger.debug(
                f"Processing report row: Stage element={stage_element}, Stage subelement={stage_subelement}, Score={score}"
            )
            if stage_element and stage_subelement and score is not None:
                mask = (test_scores_df["Stage element"] == stage_element) & (
                    test_scores_df["Stage subelement"] == stage_subelement
                )
                if not mask.any():
                    logger.debug(
                        f"  - No matching row found in Test Scores for Stage element '{stage_element}' and Stage subelement '{stage_subelement}'. Skipping score update for this row."
                    )
                    continue

                test_scores_df.loc[mask, "Score"] = score
                logger.debug(
                    f"  - Updated Test Score for test {stage_element}/{stage_subelement} to {score}"
                )
    else:
        logger.warning(f"No report data available to update scores.")

    return test_scores_df


def get_body_region_capping(cp_body_region_df) -> bool:
    if cp_body_region_df is None:
        logger.warning(
            "CP Body region scores DataFrame is None, cannot apply body region score capping."
        )
        return False
    logger.debug(f"CP Body region scores DataFrame: \n{cp_body_region_df}")
    cp_body_region_df = cp_body_region_df.ffill()

    blue_regions_df = pd.read_csv(
        io.StringIO(BODY_REGION_BLUE_ROWS_CSV_STRING), sep=";"
    )

    merged_df = cp_body_region_df.merge(
        blue_regions_df,
        on=["Stage subelement", "Loadcase", "Seat position", "Dummy", "Body region"],
        how="inner",
    )
    if not merged_df.empty and (merged_df["Score"] == 0).any():
        return True

    return False


def _merge_test_scores_from_dfs(
    test_scores_df: pd.DataFrame,
    cp_test_scores_df,
    ca_test_scores_df,
    sd_test_scores_df,
    pc_test_scores_df,
    sd_category_scores_df,
    pc_scenario_scores_df,
) -> pd.DataFrame:
    if cp_test_scores_df is not None:
        cp_test_scores_df = cp_test_scores_df.ffill()
    if ca_test_scores_df is not None:
        ca_test_scores_df = ca_test_scores_df.ffill()
    if sd_test_scores_df is not None:
        sd_test_scores_df = sd_test_scores_df.ffill()
    if pc_test_scores_df is not None:
        pc_test_scores_df = pc_test_scores_df.ffill()
    test_scores_df = test_scores_df.ffill()
    test_scores_df = update_test_scores_with_report(test_scores_df, cp_test_scores_df)
    test_scores_df = update_test_scores_with_report(test_scores_df, ca_test_scores_df)
    test_scores_df = update_test_scores_with_report(test_scores_df, sd_test_scores_df)
    test_scores_df = update_test_scores_with_report(test_scores_df, pc_test_scores_df)

    test_scores_df = test_scores_df.ffill()

    ##############
    # Edge case 1
    # Frontal Collision - Pedestrian Cyclist score depends on the scores of VRU Impact/Head Impact and VRU Impact/Pelvis & Leg Impact.
    ##############
    # IF (VRU Iimpact/Head Impact+VRU Iimpact/Pelvis & Leg Impact)<10
    # THEN 0 for Frontal Collision/Ped. & cyclist
    # and highlight in red
    head_impact_score = test_scores_df[
        test_scores_df["Stage subelement"] == "Head Impact"
    ]["Score"].values
    pelvis_leg_score = test_scores_df[
        test_scores_df["Stage subelement"] == "Pelvis & Leg Impact"
    ]["Score"].values

    # NaN (not 0.0) when absent/unassessed -- an unassessed input must not
    # silently satisfy (or silently fail to satisfy) this rule via NaN
    # arithmetic evaluating any comparison to False.
    head_impact_score = head_impact_score[0] if len(head_impact_score) > 0 else np.nan
    pelvis_leg_score = pelvis_leg_score[0] if len(pelvis_leg_score) > 0 else np.nan
    logger.debug(f"VRU Impact/Head Impact score: {head_impact_score}")
    logger.debug(f"VRU Impact/Pelvis & Leg Impact score: {pelvis_leg_score}")
    if pd.isna(head_impact_score) or pd.isna(pelvis_leg_score):
        logger.info(
            "Skipping edge case rule (VRU Impact/Head Impact + Pelvis & Leg Impact < 10): "
            "at least one input is unassessed."
        )
    elif head_impact_score + pelvis_leg_score < 10:
        test_scores_df.loc[
            (test_scores_df["Stage element"] == "Frontal Collisions")
            & (test_scores_df["Stage subelement"] == "Pedestrian & cyclist"),
            "Score",
        ] = 0.0
        logger.info(
            "Applied rule: (VRU Impact/Head Impact + VRU Impact/Pelvis & Leg Impact) < 10, setting Frontal Collision/Pedestrian & cyclist score to 0."
        )

    ##############
    # Edge case 2
    # Safe Driving - Advanced eCall score depends on the scores of
    # Safe Driving/Occupant Monitoring/Occupant presence/Crash occupancy information
    # and
    # Post-Crash/Potential number of occupants.
    ##############
    # IF (Safe Driving/Occupant Monitoring/Occupant presence/Crash occupancy information == 0
    # AND
    # IF Post-Crash/Post-Crash Intervention/Advanced eCall/Advanced eCall – 112/Potential number of occupants>0)
    # THEN Remove the points allocated to Post-Crash/Post-Crash Intervention/Advanced eCall/Advanced eCall – 112/Potential number of occupants
    # and highlight in red
    crash_occupancy_score = (
        sd_category_scores_df[
            sd_category_scores_df["Category"] == "Crash occupancy information"
        ]["Score"].values
        if sd_category_scores_df is not None
        else []
    )
    # NaN (not 0.0) when absent/unassessed -- see edge case 1's comment.
    crash_occupancy_score = (
        crash_occupancy_score[0] if len(crash_occupancy_score) > 0 else np.nan
    )

    potential_occupants_score = (
        pc_scenario_scores_df[
            pc_scenario_scores_df["Scenario"] == "Potential number of occupants"
        ]["Score"].values
        if pc_scenario_scores_df is not None
        else []
    )
    potential_occupants_score = (
        potential_occupants_score[0] if len(potential_occupants_score) > 0 else np.nan
    )

    logger.debug(f"Crash occupancy information score: {crash_occupancy_score}")
    logger.debug(f"Potential number of occupants score: {potential_occupants_score}")
    if pd.isna(crash_occupancy_score) or pd.isna(potential_occupants_score):
        logger.info(
            "Skipping edge case rule (Crash occupancy information == 0 and Potential "
            "number of occupants > 0): at least one input is unassessed."
        )
    elif crash_occupancy_score == 0 and potential_occupants_score > 0:
        test_scores_df.loc[
            test_scores_df["Stage subelement"] == "Advanced eCall", "Score"
        ] -= potential_occupants_score
        logger.info(
            f"Applied rule: Crash occupancy information == 0 and Potential number of occupants > 0, subtracting {potential_occupants_score} points from Advanced eCall score."
        )
    test_scores_df = common.opposite_ffill(
        test_scores_df, exclude_columns=["Score", "Max score", "Stage subelement"]
    )

    return test_scores_df


def get_test_score_from_reports(
    test_scores_df: pd.DataFrame,
    cp_report_path: str,
    ca_report_path: str,
    sd_report_path: str,
    pc_report_path: str,
) -> pd.DataFrame:
    """
    Extract and aggregate test scores from multiple Euro NCAP category reports.
    This function retrieves test scores from Crash Protection (CP), Crash Avoidance (CA),
    Safe Driving (SD), and Post-Crash (PC) reports, merges them with the provided test scores
    dataframe, and applies scoring rules based on specific edge cases.
    Edge Case 1: If the combined score of VRU Impact/Head Impact and VRU Impact/Pelvis & Leg
    Impact is less than 10, the Frontal Collisions/Pedestrian & cyclist score is set to 0.
    Edge Case 2: If Crash occupancy information score is less than 5 AND Potential number of
    occupants score is greater than 0, the points allocated to Advanced eCall are subtracted.
    Args:
        test_scores_df (pd.DataFrame): Base dataframe containing test scores with columns
            including "Stage element", "Stage subelement", and "Score".
        cp_report_path (str): File path to the Crash Protection report.
        ca_report_path (str): File path to the Crash Avoidance report.
        sd_report_path (str): File path to the Safe Driving report.
        pc_report_path (str): File path to the Post-Crash report.
    Returns:
        pd.DataFrame: Updated test scores dataframe with aggregated scores from all
            reports and edge case rules applied.
    """
    cp_test_scores_df = get_sheet_from_report(cp_report_path, sheet_name="Test Scores")
    ca_test_scores_df = get_sheet_from_report(ca_report_path, sheet_name="Test Scores")
    sd_test_scores_df = get_sheet_from_report(sd_report_path, sheet_name="Test Scores")
    pc_test_scores_df = get_sheet_from_report(pc_report_path, sheet_name="Test Scores")
    sd_category_scores_df = get_sheet_from_report(
        sd_report_path, sheet_name="Category Scores"
    )
    pc_scenario_scores_df = get_sheet_from_report(
        pc_report_path, sheet_name="Scenario Scores"
    )
    return _merge_test_scores_from_dfs(
        test_scores_df,
        cp_test_scores_df,
        ca_test_scores_df,
        sd_test_scores_df,
        pc_test_scores_df,
        sd_category_scores_df,
        pc_scenario_scores_df,
    )


def apply_deduction(
    stage_df: pd.DataFrame, input_parameters_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply protocol year-based deduction to the Crash Protection score based on the Vehicle provision assessment score and Year of the test.
    Rules:
        - IF (Vehicle provision assessment score < 7 AND Year of the test < 2028) THEN (7 - Vehicle provision assessment score) to be subtracted from the Crash Protection score in the Stage scores tab.
        - IF (Vehicle provision assessment score < 8 AND Year of the test = 2028) THEN (8 - Vehicle provision assessment score) to be subtracted from the Crash Protection score in the Stage scores tab.

        args:
            stage_df: DataFrame containing the Stage Scores sheet data.
            input_parameters_df: DataFrame containing the Input parameters sheet data.
        returns:
            Updated stage_df with the provision score deduction applied to the Crash Protection score.
    """
    #############
    # Edge case 3
    # Crash Protection - Vehicle provision assessment score impacts the overall score based on the year of the test.
    #############
    # IF (value < 7 AND Year of the test <2028) THEN (7-value) to be substracted from the CP score in the Stage scores tab.
    # IF (value < 8 AND Year of the test = 2028) THEN (8-value) to be substracted from the CP score in the Stage scores tab
    param_dict = common.create_param_dict_from_input_parameters(
        input_parameters_df, col="Stage"
    )
    logger.debug(f"Extracted parameters from input file: {param_dict}")
    test_year = param_dict["All"].get("Year of the test", None)
    logger.debug(f"Extracted test year: {test_year}")
    provision_score = param_dict["Crash Protection"].get(
        "Vehicle provision assessment score", None
    )
    logger.debug(f"Extracted Vehicle provision assessment score: {provision_score}")

    if test_year is None or provision_score is None:
        deduction = 0
        logger.warning(
            "Year of the test or Vehicle provision assessment score is missing from input parameters. Skipping provision score deduction."
        )
    elif provision_score < 7 and test_year < 2028:
        deduction = 7 - provision_score
        logger.info(
            f"Applied rule: Vehicle provision assessment score < 7 and Year of the test < 2028, subtracting {deduction} points from Crash Protection score."
        )
    elif provision_score < 8 and test_year == 2028:
        deduction = 8 - provision_score
        logger.info(
            f"Applied rule: Vehicle provision assessment score < 8 and Year of the test = 2028, subtracting {deduction} points from Crash Protection score."
        )
    else:
        deduction = 0
        logger.debug(
            "No provision score deduction applied based on the input parameters."
        )

    stage_df.loc[stage_df["Stage"] == "Crash Protection", "Score"] -= deduction
    logger.info(
        f"Updated Crash Protection score after provision score deduction: {stage_df.loc[stage_df['Stage'] == 'Crash Protection', 'Score'].values[0]}"
    )

    return stage_df


# Compute star rating for each category based on which bin the score falls into
def get_star_rating_for_score(score, thresholds):
    """Determine star rating: 5 if >= thresholds[0], 4 if >= thresholds[1], etc."""
    for star in range(5, 0, -1):
        if score >= thresholds[5 - star]:
            return star
    return 0  # Below all thresholds


def compute_star_rating(stage_df, input_parameters_df, four_star_capping):
    star_rating = 0.0
    param_dict = common.create_param_dict_from_input_parameters(
        input_parameters_df, col="Stage"
    )
    sd_score = stage_df[stage_df["Stage"] == "Safe Driving"]["Score"].values
    sd_score = float(sd_score[0]) if len(sd_score) > 0 else np.nan
    logger.debug(f"Safe Driving score for star rating computation: {sd_score}")
    ca_score = stage_df[stage_df["Stage"] == "Crash Avoidance"]["Score"].values
    ca_score = float(ca_score[0]) if len(ca_score) > 0 else np.nan
    logger.debug(f"Crash Avoidance score for star rating computation: {ca_score}")
    cp_score = stage_df[stage_df["Stage"] == "Crash Protection"]["Score"].values
    cp_score = float(cp_score[0]) if len(cp_score) > 0 else np.nan
    logger.debug(f"Crash Protection score for star rating computation: {cp_score}")
    pc_score = stage_df[stage_df["Stage"] == "Post-Crash"]["Score"].values
    pc_score = float(pc_score[0]) if len(pc_score) > 0 else np.nan
    logger.debug(f"Post-Crash score for star rating computation: {pc_score}")

    # An unassessed Stage must not silently turn into a 0-star rating: the
    # compensation search below picks the first (arbitrary) combination when
    # every candidate's error is NaN, and get_star_rating_for_score's `>=`
    # comparisons are all False against a NaN score -- both would otherwise
    # produce a confidently-wrong 0-star result instead of "not yet ratable".
    if any(pd.isna(s) for s in (sd_score, ca_score, cp_score, pc_score)):
        logger.info(
            "At least one Stage score is unassessed; overall star rating is not yet ratable."
        )
        return None

    logger.debug(
        f"Starting scores: SD={sd_score}, CA={ca_score}, CP={cp_score}, PC={pc_score}"
    )
    test_year = param_dict["All"].get("Year of the test", None)

    if test_year == 2026:
        sd_score = sd_score + 20
        ca_score = ca_score + 10
    if test_year == 2027:
        sd_score = sd_score + 10

    logger.info(
        f"Test year {test_year} Adjusted scores: SD={sd_score}, CA={ca_score}, CP={cp_score}, PC={pc_score}"
    )
    avg_score = float(np.mean([sd_score, ca_score, cp_score]))
    logger.info(
        f"Average score across categories for star rating computation: {avg_score}"
    )

    compensation_errors = []
    compensation_updates = []

    sd_to_ca_list = range(-5, 6)
    logger.debug(f"SD to CA values for star rating computation: {list(sd_to_ca_list)}")
    for sd_to_ca in sd_to_ca_list:
        ca_to_cp_min = -5 - sd_to_ca if sd_to_ca < 0 else (0 if sd_to_ca > 0 else -5)
        ca_to_cp_max = 0 if sd_to_ca < 0 else (5 - sd_to_ca if sd_to_ca > 0 else 5)
        ca_to_cp_list = range(ca_to_cp_min, ca_to_cp_max + 1)
        for ca_to_cp in ca_to_cp_list:
            curr_sd_score = sd_score + sd_to_ca
            curr_ca_score = ca_score - sd_to_ca - ca_to_cp
            curr_cp_score = cp_score + ca_to_cp

            deltas = (
                np.array([curr_sd_score, curr_ca_score, curr_cp_score], dtype=float)
                - avg_score
            )
            current_error = float(np.sum(np.square(deltas)))
            compensation_info = CompensationInfo(sd_to_ca=sd_to_ca, ca_to_cp=ca_to_cp)
            logger.debug(
                f"current_error for SD to CA={sd_to_ca}, CA to CP={ca_to_cp}: {current_error}"
            )
            compensation_errors.append(current_error)
            compensation_updates.append(compensation_info)

    logger.debug(
        f"Min error for star rating compensation: {compensation_errors[np.argmin(compensation_errors)]}"
    )
    best_compensation = compensation_updates[np.argmin(compensation_errors)]
    logger.info(f"Best compensation for star rating computation: {best_compensation}")

    # Compensate the scores based on the best compensation found
    compensated_sd_score = sd_score + best_compensation.sd_to_ca
    compensated_ca_score = (
        ca_score - best_compensation.sd_to_ca - best_compensation.ca_to_cp
    )
    compensated_cp_score = cp_score + best_compensation.ca_to_cp
    logger.info(
        f"Compensated scores for star rating computation: SD={compensated_sd_score}, CA={compensated_ca_score}, CP={compensated_cp_score}, PC={pc_score}"
    )

    # Remove the offset from the compensated scores to get the final scores for star rating computation
    if test_year == 2026:
        compensated_sd_score = compensated_sd_score - 20
        compensated_ca_score = compensated_ca_score - 10
    if test_year == 2027:
        compensated_sd_score = compensated_sd_score - 10

    logger.info(
        f"Final compensated scores for star rating computation after removing offsets: SD={compensated_sd_score}, CA={compensated_ca_score}, CP={compensated_cp_score}, PC={pc_score}"
    )

    star_thresholds = [80, 70, 60, 50, 40]
    post_crash_thresholds = star_thresholds
    crash_protection_thresholds = star_thresholds

    if test_year == 2026:
        safe_driving_thresholds = [t - 20 for t in star_thresholds]
        crash_avoidance_thresholds = [t - 10 for t in star_thresholds]
    elif test_year == 2027:
        safe_driving_thresholds = [t - 10 for t in star_thresholds]
        crash_avoidance_thresholds = star_thresholds
    else:
        safe_driving_thresholds = star_thresholds
        crash_avoidance_thresholds = star_thresholds

    sd_star = get_star_rating_for_score(compensated_sd_score, safe_driving_thresholds)
    ca_star = get_star_rating_for_score(
        compensated_ca_score, crash_avoidance_thresholds
    )
    cp_star = get_star_rating_for_score(
        compensated_cp_score, crash_protection_thresholds
    )
    pc_star = get_star_rating_for_score(pc_score, post_crash_thresholds)

    logger.info(
        f"Star ratings by category: SD={sd_star}, CA={ca_star}, CP={cp_star}, PC={pc_star}"
    )

    # Overall star rating is the minimum of all category ratings
    star_rating = floor_score(float(min(sd_star, ca_star, cp_star, pc_star)))
    logger.info(f"Computed overall star rating before capping: {star_rating}")
    # 4 start capping if body region score 0-conditions are met
    if star_rating > 4 and four_star_capping:
        star_rating = 4.0
        logger.info(
            f"Applied 4-star capping due to body region score conditions. Final star rating: {star_rating}"
        )
    return star_rating


def calculate_score(dfs: dict) -> dict:
    """Calculate the overall NCAP score from pre-loaded dataframes.

    dfs keys:
      # overall template sheets
      "Input parameters", "Rating", "Stage Scores", "Stage element Scores", "Test Scores"
      # per-domain report sheets (dataframes, not paths)
      "cp": {"Test Scores": df, "CP - Body region scores": df}
      "ca": {"Test Scores": df}
      "sd": {"Test Scores": df, "Category Scores": df}
      "pc": {"Test Scores": df, "Scenario Scores": df}

    Returns:
      dict with keys "Test Scores", "Stage element Scores", "Stage Scores", "Rating"
    """
    # "-" marks a not-yet-computed cell in the template (see sheets.py).
    # Convert to 0 (0-propagation kept: a missing domain report = 0 points /
    # 0 stars) before the hard astype casts below. Only the four overall
    # template sheets are touched; the nested "cp"/"ca"/"sd"/"pc" report
    # entries pass through untouched.
    dfs = common.numericize_computed_columns(dfs, sheets.COMPUTED_SHEET_COLUMNS)

    test_scores_df = dfs["Test Scores"].copy()
    stage_element_df = dfs["Stage element Scores"].copy()
    stage_df = dfs["Stage Scores"].copy()
    star_rating_df = dfs["Rating"].copy()

    test_scores_df["Score"] = test_scores_df["Score"].astype(float)
    stage_element_df["Score"] = stage_element_df["Score"].astype(float)
    stage_df["Score"] = stage_df["Score"].astype(float)
    star_rating_df["Star rating"] = star_rating_df["Star rating"].astype(int)

    cp = dfs.get("cp", {})
    ca = dfs.get("ca", {})
    sd = dfs.get("sd", {})
    pc = dfs.get("pc", {})

    test_scores_df = _merge_test_scores_from_dfs(
        test_scores_df,
        cp.get("Test Scores"),
        ca.get("Test Scores"),
        sd.get("Test Scores"),
        pc.get("Test Scores"),
        sd.get("Category Scores"),
        pc.get("Scenario Scores"),
    )

    stage_element_df = common.update_score_sum_interval(
        stage_element_df,
        test_scores_df,
        key_column="Stage element",
        interval_column="Stage element",
    )
    stage_df = common.update_score_sum_interval(
        stage_df, stage_element_df, key_column="Stage", interval_column="Stage"
    )

    input_parameters_df = dfs["Input parameters"]
    stage_df = apply_deduction(stage_df, input_parameters_df)
    # floor_score's math.floor crashes on NaN (a Stage can be unassessed if
    # any of its Test Scores is) -- pass an unassessed Stage Score through
    # unchanged rather than flooring it.
    stage_df["Score"] = stage_df["Score"].apply(
        lambda v: v if pd.isna(v) else floor_score(v)
    )

    four_star_capping = get_body_region_capping(cp.get("CP - Body region scores"))
    star_rating = compute_star_rating(stage_df, input_parameters_df, four_star_capping)
    star_rating_df["Star rating"] = star_rating

    return {
        "Test Scores": test_scores_df,
        "Stage element Scores": stage_element_df,
        "Stage Scores": stage_df,
        "Rating": star_rating_df,
    }


@click.command()
@with_footer
@click.option(
    "--input_file",
    "-i",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the input Excel file containing NCAP test measurements.",
)
@click.option(
    "--reports_path",
    "-p",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=os.getcwd(),
    help="Path to the folder where the individual test reports are stored. This is used to link the overall report to the individual test reports.",
)
@click.option(
    "--output_path",
    "-o",
    type=click.Path(file_okay=False, writable=True),
    default=os.getcwd(),
    show_default=True,
    help="Path to the output directory where the report will be saved.",
)
def compute_score(input_file, reports_path, output_path):
    """Collect NCAP scores from the report directory and compute the overall score."""
    common.check_version(input_file, common.CliCommand.GENERATE_TEMPLATE)
    print(f"[Overall] Collecting NCAP scores from report directory {reports_path}...")

    if not input_file:
        logger.error("Input file path is required.")
        sys.exit(1)
    if not input_file.endswith(".xlsx"):
        logger.error("Input file must be an Excel file with .xlsx extension.")
        sys.exit(1)
    if not reports_path:
        logger.error("Reports path is required.")
        sys.exit(1)
    if not os.path.exists(reports_path):
        logger.error(f"Reports path {reports_path} does not exist.")
        sys.exit(1)

    print("Loading data from spreadsheet...")
    dfs = common.read_excel_file_to_dfs(input_file)
    logger.debug(f"Loaded sheets: {list(dfs.keys())}")

    print("Collecting NCAP scores...")
    domain_sheets = {
        "cp": ["Test Scores", "CP - Body region scores"],
        "ca": ["Test Scores"],
        "sd": ["Test Scores", "Category Scores"],
        "pc": ["Test Scores", "Scenario Scores"],
    }
    for prefix, sheets in domain_sheets.items():
        path = get_report_path_with_prefix(reports_path, prefix)
        dfs[prefix] = {s: get_sheet_from_report(path, s) for s in sheets}

    updated_dfs = calculate_score(dfs)

    logger.debug(
        f"Starting to write output file with copied sheets and computed scores..."
    )

    sheet_order = [
        "Input parameters",
        "Rating",
        "Stage Scores",
        "Stage element Scores",
        "Test Scores",
        "Version",
    ]
    report_writer.write_report(
        common.CliCommand.COMPUTE_SCORE,
        input_file,
        updated_dfs=updated_dfs,
        selected_points_dict={},
        output_path=output_path,
        format_prediction_cells=False,
        sheet_order=sheet_order,
    )
