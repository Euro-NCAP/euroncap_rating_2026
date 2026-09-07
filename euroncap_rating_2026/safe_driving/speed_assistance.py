# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import io
from typing import Dict
import numpy as np
import pandas as pd
from euroncap_rating_2026 import common
from euroncap_rating_2026.safe_driving import data_model
import logging

logger = logging.getLogger(__name__)

CONDITIONAL_SPEED_LIMIT_STR = "Conditional speed limits"
IMPLICIT_SPEED_LIMIT_STR = "Implicit speed limits"
DYNAMIC_SPEED_LIMIT_STR = "Dynamic speed limits"
SLIF_ADV_MAPPING = {
    "Rain / Wetness": CONDITIONAL_SPEED_LIMIT_STR,
    "Snow / Icy": CONDITIONAL_SPEED_LIMIT_STR,
    "Time / Season": CONDITIONAL_SPEED_LIMIT_STR,
    "Distance for / Distance in": CONDITIONAL_SPEED_LIMIT_STR,
    "Non-lane relevant arrows": CONDITIONAL_SPEED_LIMIT_STR,
    "Lane relevant arrows": CONDITIONAL_SPEED_LIMIT_STR,
    "Vehicle categories": CONDITIONAL_SPEED_LIMIT_STR,
    "Highway / Motorway": IMPLICIT_SPEED_LIMIT_STR,
    "City entry / City exit": IMPLICIT_SPEED_LIMIT_STR,
    "Residential zones": IMPLICIT_SPEED_LIMIT_STR,
    "Non-lane relevant": DYNAMIC_SPEED_LIMIT_STR,
    "Lane relevant": DYNAMIC_SPEED_LIMIT_STR,
}

EURO_NCAP_MEMBER_STATES_LABEL = "euro ncap member states"
EXPECTED_MEMBER_STATE_ROW_COUNT = 10
EXPECTED_APPLICATION_AREA_ROW_COUNT = 29

LOCAL_HAZARD_COLUMNS = [
    "Construction zones",
    "Items on road",
    "Stopped vehicle",
    "Broken down vehicle",
    "Post crash",
    "Poor weather",
    "Poor road",
    "Wrong way driver",
    "Amber & blue lights",
    "Traffic jam",
]

LOCAL_HAZARD_SENDING_STR = "Sending"
LOCAL_HAZARD_RECEIVING_STR = "Receiving & informing"
LOCAL_HAZARD_DIRECTIONS = [LOCAL_HAZARD_SENDING_STR, LOCAL_HAZARD_RECEIVING_STR]

# Hazards for which "Sending" is N/A in the protocol scoring table
LOCAL_HAZARD_NO_SENDING_COLUMNS = ["Amber & blue lights", "Traffic jam"]

# Positional row index of each (communication channel, direction) pair inside the
# SLIF Local Hazards prediction table. The table is 4 data rows tall:
# Cloud/Sending, Cloud/Receiving & informing, Direct/Sending, Direct/Receiving &
# informing (see sheet 'VA - Speed assist. pred.').
LOCAL_HAZARD_CHANNEL_ROW_IDX = {
    ("cloud", LOCAL_HAZARD_SENDING_STR): 0,
    ("cloud", LOCAL_HAZARD_RECEIVING_STR): 1,
    ("direct", LOCAL_HAZARD_SENDING_STR): 2,
    ("direct", LOCAL_HAZARD_RECEIVING_STR): 3,
}

# Points per direction as (Direct AND Cloud communication, Direct OR Cloud
# communication), per protocol section 1.2.3 Local Hazards.
LOCAL_HAZARD_POINTS = {
    LOCAL_HAZARD_SENDING_STR: (0.2, 0.15),
    LOCAL_HAZARD_RECEIVING_STR: (0.15, 0.15),
}

LOCAL_HAZARD_CAP_BOTH_CHANNELS = 3.0
LOCAL_HAZARD_CAP_SINGLE_CHANNEL = 2.5

SLIF_SCORING_CSV_STRING = """
Category;Scenario;Value
SLIF - General requirements;;
SLIF - Accuracy;Distance based KPI;
SLIF - Accuracy;Event based KPI;
SLIF - Advanced speed limit;Conditional speed limits - Rain / Wetness;FAIL
SLIF - Advanced speed limit;Conditional speed limits - Snow / Icy;FAIL
SLIF - Advanced speed limit;Conditional speed limits - Time / Season;FAIL
SLIF - Advanced speed limit;Conditional speed limits - Distance for / Distance in;FAIL
SLIF - Advanced speed limit;Conditional speed limits - Non-lane relevant arrows;FAIL
SLIF - Advanced speed limit;Conditional speed limits - Lane relevant arrows;FAIL
SLIF - Advanced speed limit;Conditional speed limits - Vehicle categories;FAIL
SLIF - Advanced speed limit;Implicit speed limits - Highway, city and residential zones;FAIL
SLIF - Advanced speed limit;Dynamic speed limits - Non-lane relevant;FAIL
SLIF - Advanced speed limit;Dynamic speed limits - Lane relevant;FAIL
SLIF - Local hazards;Construction zones - Sending;FAIL
SLIF - Local hazards;Construction zones - Receiving & informing;FAIL
SLIF - Local hazards;Items on road - Sending;FAIL
SLIF - Local hazards;Items on road - Receiving & informing;FAIL
SLIF - Local hazards;Stopped vehicle - Sending;FAIL
SLIF - Local hazards;Stopped vehicle - Receiving & informing;FAIL
SLIF - Local hazards;Broken down vehicle - Sending;FAIL
SLIF - Local hazards;Broken down vehicle - Receiving & informing;FAIL
SLIF - Local hazards;Post crash - Sending;FAIL
SLIF - Local hazards;Post crash - Receiving & informing;FAIL
SLIF - Local hazards;Poor weather - Sending;FAIL
SLIF - Local hazards;Poor weather - Receiving & informing;FAIL
SLIF - Local hazards;Poor road - Sending;FAIL
SLIF - Local hazards;Poor road - Receiving & informing;FAIL
SLIF - Local hazards;Wrong way driver - Sending;FAIL
SLIF - Local hazards;Wrong way driver - Receiving & informing;FAIL
SLIF - Local hazards;Amber & blue lights - Receiving & informing;FAIL
SLIF - Local hazards;Traffic jam - Receiving & informing;FAIL
Speed control function;Speedometer accuracy;≥-3 km/h
"""

ISL_ROW = """
Category;Scenario;Value
Speed control function;Intelligent speed limiter;PASS
"""
IACC_ROW = """
Category;Scenario;Value
Speed control function;Intelligent adaptive cruise control;PASS
"""


def get_slif_advanced_speed_limit_table(
    sas_prediction_df: pd.DataFrame,
) -> pd.DataFrame:
    slif_adv_table = common.extract_sub_table(
        sas_prediction_df, start_row=0, n_rows=30, start_col=0, n_cols=14
    )
    slif_adv_table.columns = slif_adv_table.iloc[0]
    slif_adv_table = slif_adv_table[1:].reset_index(drop=True)
    # Column 0 (group label, e.g. "Euro NCAP member states") is sparse and
    # shares a duplicate NaN header with column 1 (country name), so both
    # must be forward-filled and accessed positionally, never by column
    # label, since `slif_adv_table[nan]` returns a 2-column DataFrame.
    group_labels = slif_adv_table.iloc[:, 0].ffill()
    slif_adv_table.fillna("N/A", inplace=True)
    slif_adv_table.iloc[:, 0] = group_labels
    return slif_adv_table


def get_slif_advanced_speed_limit_bg_table(
    sas_prediction_bg_df: pd.DataFrame, columns
) -> pd.DataFrame:
    slif_adv_bg_table = common.extract_sub_table(
        sas_prediction_bg_df, start_row=0, n_rows=30, start_col=0, n_cols=14
    )
    # Unlike the text table, the bg table's own sub-header row (scenario
    # names) carries no fill color, so it can't supply column labels itself.
    # Reuse the already-promoted labels from the text table instead; the two
    # tables are row/column aligned since both are built from the same
    # sheet region starting at the same Excel row.
    slif_adv_bg_table = slif_adv_bg_table[1:].reset_index(drop=True)
    slif_adv_bg_table.columns = columns
    return slif_adv_bg_table


def get_local_hazard_table(sas_prediction_df: pd.DataFrame) -> pd.DataFrame:
    slif_local_hazards_table = common.extract_sub_table(
        sas_prediction_df, start_row=31, n_rows=6, start_col=0, n_cols=12
    )
    slif_local_hazards_table.columns = slif_local_hazards_table.iloc[0]
    slif_local_hazards_table = slif_local_hazards_table[1:].reset_index(drop=True)
    # Forward fill only the first column
    first_col = slif_local_hazards_table.columns[0]
    slif_local_hazards_table[first_col] = slif_local_hazards_table[first_col].ffill()

    slif_local_hazards_table.fillna("N/A", inplace=True)
    logger.debug(f"SLIF Local Hazards Table:\n{slif_local_hazards_table}")
    return slif_local_hazards_table


def get_local_hazard_channel_status(slif_local_hazards_table: pd.DataFrame) -> Dict:
    """Read the Cloud / Direct communication predictions per hazard and direction.

    Returns {hazard: {direction: {"cloud": bool, "direct": bool, "has_red": bool}}}
    """
    channel_status = {}
    for hazard_col in LOCAL_HAZARD_COLUMNS:
        if hazard_col not in slif_local_hazards_table.columns:
            logger.warning(
                f"Column '{hazard_col}' is missing in SLIF Local Hazards Table."
            )
            continue
        hazard_values = [
            str(value).strip().lower()
            for value in slif_local_hazards_table[hazard_col].iloc[:4].tolist()
        ]
        if len(hazard_values) < 4:
            logger.warning(
                f"SLIF Local Hazards Table has only {len(hazard_values)} prediction "
                f"rows for '{hazard_col}', 4 expected."
            )
            continue

        hazard_status = {}
        for direction in LOCAL_HAZARD_DIRECTIONS:
            if (
                direction == LOCAL_HAZARD_SENDING_STR
                and hazard_col in LOCAL_HAZARD_NO_SENDING_COLUMNS
            ):
                continue
            cloud_value = hazard_values[
                LOCAL_HAZARD_CHANNEL_ROW_IDX[("cloud", direction)]
            ]
            direct_value = hazard_values[
                LOCAL_HAZARD_CHANNEL_ROW_IDX[("direct", direction)]
            ]
            hazard_status[direction] = {
                "cloud": cloud_value == "green",
                "direct": direct_value == "green",
                "has_red": "red" in (cloud_value, direct_value),
            }
        channel_status[hazard_col] = hazard_status

    logger.debug(f"SLIF Local Hazards channel status: {channel_status}")
    return channel_status


def update_scenario_scores_with_scf_intelligent_row(
    scenario_scores_df: pd.DataFrame,
    type_of_scf_param,
    intelligent_scenario: str,
) -> pd.DataFrame:
    if scenario_scores_df.empty:
        logger.warning("Scenario Scores sheet is missing or empty.")
        return scenario_scores_df

    if type_of_scf_param is None:
        logger.warning("Type of SCF parameter not found.")
        return scenario_scores_df

    type_of_scf_param = str(type_of_scf_param).strip().lower()
    if type_of_scf_param not in ["iacc", "isl"]:
        logger.debug(
            f"Type of SCF read {type_of_scf_param} does not match 'iacc' or 'isl'. No Scenario Scores row update done."
        )
        return scenario_scores_df

    if not intelligent_scenario:
        logger.warning("Intelligent scenario string is empty.")
        return scenario_scores_df

    category_col = scenario_scores_df["Category"].fillna("").astype(str)
    scenario_col = scenario_scores_df["Scenario"].fillna("").astype(str)

    mask = (category_col.str.strip().str.lower() == "speed control function") & (
        scenario_col.str.strip().str.lower() == "speedometer accuracy"
    )
    indices = scenario_scores_df.index[mask].tolist()

    if not indices:
        logger.warning(
            "No row with Category 'Speed control function' and Scenario 'Speedometer accuracy' found in Scenario Scores sheet."
        )
        return scenario_scores_df

    insert_idx = indices[-1] + 1
    if insert_idx >= len(scenario_scores_df):
        logger.warning(
            "No blank row found after 'Speedometer accuracy' in Scenario Scores sheet."
        )
        return scenario_scores_df

    scenario_scores_df.loc[insert_idx, "Scenario"] = str(intelligent_scenario)

    return scenario_scores_df


def update_scenario_scores_with_system_updates_row(
    scenario_scores_df: pd.DataFrame,
    system_updates_str: str,
) -> pd.DataFrame:
    if scenario_scores_df.empty:
        logger.warning("Scenario Scores sheet is missing or empty.")
        return scenario_scores_df

    if not system_updates_str:
        logger.warning("System updates string is empty.")
        return scenario_scores_df

    category_col = scenario_scores_df["Category"].fillna("").astype(str)
    mask = category_col.str.strip().str.lower() == "slif - system updates"
    indices = scenario_scores_df.index[mask].tolist()

    if not indices:
        logger.warning(
            "No row with Category 'SLIF - System updates' found in Scenario Scores sheet."
        )
        return scenario_scores_df

    idx = indices[0]
    scenario_scores_df.loc[idx, "Scenario"] = str(system_updates_str)

    return scenario_scores_df


def preprocess(dfs: dict[str, pd.DataFrame]) -> Dict:
    logger.info("Preprocess SAS ...")
    slif_verification_df = pd.read_csv(io.StringIO(SLIF_SCORING_CSV_STRING), sep=";")
    sas_prediction_df = dfs.get("VA - Speed assist. pred.")
    sas_prediction_bg_df = dfs.get("VA - Speed assist. pred. (bg)")

    slif_adv_table = get_slif_advanced_speed_limit_table(sas_prediction_df)
    slif_adv_bg_table = (
        get_slif_advanced_speed_limit_bg_table(
            sas_prediction_bg_df, slif_adv_table.columns
        )
        if sas_prediction_bg_df is not None
        else None
    )

    total_country_count = len(slif_adv_table)
    if total_country_count != EXPECTED_APPLICATION_AREA_ROW_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_APPLICATION_AREA_ROW_COUNT} country rows in the SLIF "
            f"advanced speed limit table, found {total_country_count}. The "
            "'VA - Speed assist. pred.' sheet layout may have changed."
        )

    member_state_mask = (
        slif_adv_table.iloc[:, 0].map(lambda v: str(v).strip().lower())
        == EURO_NCAP_MEMBER_STATES_LABEL
    )
    if member_state_mask.sum() != EXPECTED_MEMBER_STATE_ROW_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_MEMBER_STATE_ROW_COUNT} 'Euro NCAP member states' rows "
            f"in the SLIF advanced speed limit table, found {member_state_mask.sum()}. "
            "The 'VA - Speed assist. pred.' sheet layout may have changed."
        )

    implicit_scenario_status_list = []

    for slif_adv_key, scenario_str in SLIF_ADV_MAPPING.items():
        current_text_col = slif_adv_table[slif_adv_key]
        current_bg_col = (
            slif_adv_bg_table[slif_adv_key] if slif_adv_bg_table is not None else None
        )
        scenario_full_name = f"{scenario_str} - {slif_adv_key}"

        def classify(idx):
            norm = str(current_text_col.iloc[idx]).strip().lower()
            if norm == "green":
                return "GREEN"
            if norm == "red":
                return "RED"
            # Blank text: distinguish truly-blank (white/no-fill) from an
            # explicit grey/other manual fill, which the sheet's Green/Red-only
            # dropdown can't express as text.
            if current_bg_col is not None:
                color = current_bg_col.iloc[idx]
                if color not in (
                    None,
                    common.PredictionColor.GREEN,
                    common.PredictionColor.RED,
                ):
                    return "GREY_NA"
            return "BLANK"

        classified = [classify(i) for i in range(total_country_count)]
        member_classified = [
            c for c, is_member in zip(classified, member_state_mask) if is_member
        ]
        logger.debug(
            f"Processing SLIF prediction column '{slif_adv_key}' for scenario '{scenario_str}' with values:\n{classified}"
        )

        member_states_pass = not any(c in ("RED", "GREY_NA") for c in member_classified)
        green_or_blank_count = sum(c in ("GREEN", "BLANK") for c in classified)
        application_area_pass = green_or_blank_count >= (total_country_count / 2)

        if member_states_pass and application_area_pass:
            scenario_status = "PASS"
            logger.debug(
                f"Setting scenario '{scenario_full_name}' to PASS: no member-state "
                f"Red/N-A, and {green_or_blank_count}/{total_country_count} application-area "
                "countries are Green or blank."
            )
        else:
            scenario_status = "FAIL"
            logger.debug(
                f"Setting scenario '{scenario_full_name}' to FAIL: member_states_pass="
                f"{member_states_pass}, application_area_pass={application_area_pass} "
                f"({green_or_blank_count}/{total_country_count})."
            )

        if scenario_str == IMPLICIT_SPEED_LIMIT_STR:
            implicit_scenario_status_list.append(scenario_status)
            continue

        slif_verification_df.loc[
            (slif_verification_df["Scenario"] == scenario_full_name),
            "Value",
        ] = scenario_status

    implicit_final_status = (
        "PASS"
        if all(status == "PASS" for status in implicit_scenario_status_list)
        else "FAIL"
    )

    slif_verification_df.loc[
        slif_verification_df["Scenario"]
        == "Implicit speed limits - Highway, city and residential zones",
        "Value",
    ] = implicit_final_status
    logger.debug(
        f"Final status for 'Implicit speed limits - Highway, city and residential zones': {implicit_final_status}"
    )

    slif_local_hazards_table = get_local_hazard_table(sas_prediction_df)
    logger.debug(
        f"Processing SLIF Local Hazards predictions for scenarios starting with 'SLIF - Local Hazards' with values:\n{slif_local_hazards_table}"
    )
    # Determine PASS/FAIL for SLIF Local Hazards scenarios based on prediction table
    # For each hazard, set PASS/FAIL in slif_verification_df
    for hazard_col in LOCAL_HAZARD_COLUMNS:
        hazard_col_df = slif_local_hazards_table[hazard_col]
        if hazard_col_df.empty:
            logger.warning(
                f"Column '{hazard_col}' is missing in SLIF Local Hazards Table."
            )
            continue
        hazard_values = hazard_col_df.iloc[:4].tolist()

        # Sending status
        is_cloud_sending = hazard_values[0].lower() == "green"
        is_direct_sending = hazard_values[2].lower() == "green"
        sending_status = "PASS" if (is_cloud_sending or is_direct_sending) else "FAIL"
        slif_verification_df.loc[
            (slif_verification_df["Category"] == "SLIF - Local hazards")
            & (slif_verification_df["Scenario"] == f"{hazard_col} - Sending"),
            "Value",
        ] = sending_status

        # Receiving status
        is_cloud_receiving = hazard_values[1].lower() == "green"
        is_direct_receiving = hazard_values[3].lower() == "green"
        receiving_status = (
            "PASS" if (is_cloud_receiving or is_direct_receiving) else "FAIL"
        )
        slif_verification_df.loc[
            (slif_verification_df["Category"] == "SLIF - Local hazards")
            & (
                slif_verification_df["Scenario"]
                == f"{hazard_col} - Receiving & informing"
            ),
            "Value",
        ] = receiving_status

        logger.debug(
            f"Hazard '{hazard_col}' - Sending status: {sending_status}, Receiving status: {receiving_status}"
        )

    param_dict = common.create_param_dict_from_input_parameters(
        dfs.get("Input parameters", pd.DataFrame()), col="Category"
    )
    logger.debug(f"Parameter dictionary created from input parameters:\n{param_dict}")

    type_of_scf_param = param_dict.get("Speed control function", {}).get(
        "Type of system", ""
    )
    logger.info(f"Type of SCF parameter value: {type_of_scf_param}")
    type_of_scf_param = str(type_of_scf_param).strip().lower()

    if type_of_scf_param == "iacc":
        intelligent_row = pd.read_csv(io.StringIO(IACC_ROW), sep=";")
    elif type_of_scf_param == "isl":
        intelligent_row = pd.read_csv(io.StringIO(ISL_ROW), sep=";")
    else:
        logger.debug(
            f"Type of SCF read {type_of_scf_param} does not match 'iacc' or 'isl'. No additional rows added to SLIF verification DataFrame."
        )
        intelligent_row = pd.DataFrame()

    slif_verification_df = pd.concat(
        [slif_verification_df, intelligent_row],
        ignore_index=True,
    )

    slif_verification_df = common.opposite_ffill(
        slif_verification_df, exclude_columns=["Value"]
    )

    intelligent_scenario_str = (
        intelligent_row["Scenario"].iloc[0] if not intelligent_row.empty else ""
    )
    # Update Scenario Scores sheet
    scenario_scores_df = dfs.get("Scenario Scores", pd.DataFrame())
    scenario_scores_df = update_scenario_scores_with_scf_intelligent_row(
        scenario_scores_df, type_of_scf_param, intelligent_scenario_str
    )
    input_parameters_df = dfs.get("Input parameters", pd.DataFrame())
    system_updates_row = input_parameters_df[
        input_parameters_df["Category"] == "SLIF - System updates"
    ]
    system_updates_str = (
        system_updates_row["Value"].iloc[0] if not system_updates_row.empty else ""
    )
    logger.debug(
        f"System updates string extracted from Input Parameters: '{system_updates_str}'"
    )

    scenario_scores_df = update_scenario_scores_with_system_updates_row(
        scenario_scores_df, system_updates_str
    )

    return slif_verification_df, scenario_scores_df


def compute_score(dfs: dict[str, pd.DataFrame]) -> Dict:
    logger.info("Computing SAS prediction score...")
    scenario_scores_df = dfs.get("Scenario Scores", pd.DataFrame())
    accuracy_score_dict = {
        "Distance based": 0.0,
        "Event based": 0.0,
    }
    advanced_score_dict = {
        "Conditional speed limits - Rain / Wetness": 0.0,
        "Conditional speed limits - Snow / Icy": 0.0,
        "Conditional speed limits - Time / Season": 0.0,
        "Conditional speed limits - Distance for / Distance in": 0.0,
        "Conditional speed limits - Non-lane relevant arrows": 0.0,
        "Conditional speed limits - Lane relevant arrows": 0.0,
        "Conditional speed limits - Vehicle categories": 0.0,
        "Implicit speed limits - Highway, city and residential zones": 0.0,
        "Dynamic speed limits - Non-lane relevant": 0.0,
        "Dynamic speed limits - Lane relevant": 0.0,
    }

    hazard_score_dict = {
        "Construction zones - Sending": 0.0,
        "Construction zones - Receiving & informing": 0.0,
        "Items on road - Sending": 0.0,
        "Items on road - Receiving & informing": 0.0,
        "Stopped vehicle - Sending": 0.0,
        "Stopped vehicle - Receiving & informing": 0.0,
        "Broken down vehicle - Sending": 0.0,
        "Broken down vehicle - Receiving & informing": 0.0,
        "Post crash - Sending": 0.0,
        "Post crash - Receiving & informing": 0.0,
        "Poor weather - Sending": 0.0,
        "Poor weather - Receiving & informing": 0.0,
        "Poor road - Sending": 0.0,
        "Poor road - Receiving & informing": 0.0,
        "Wrong way driver - Sending": 0.0,
        "Wrong way driver - Receiving & informing": 0.0,
        "Amber & blue lights - Receiving & informing": 0.0,
        "Traffic jam - Receiving & informing": 0.0,
    }

    scf_score_dict = {
        "Speedometer accuracy": 0.0,
        "Intelligent speed limiter": 0.0,
        "Intelligent adaptive cruise control": 0.0,
    }

    slif_score_dict = {
        **accuracy_score_dict,
        **advanced_score_dict,
        **hazard_score_dict,
        **scf_score_dict,
    }

    ####### 1. Get relevant DataFrames
    sas_prediction_df = dfs.get("VA - Speed assist. pred.")
    sas_verification_df = dfs.get("VA - Speed assist. verif.")
    if sas_prediction_df is None or sas_verification_df is None:
        logger.error(
            "Required DataFrames 'VA - Speed assist. pred.' or 'VA - Speed assist. verif.' are missing in dfs."
        )
        return pd.DataFrame(), scenario_scores_df, slif_score_dict, 0.0

    sas_verification_df["Score"] = 0.0
    sas_verification_df["Score"] = sas_verification_df["Score"].astype(float)

    logger.debug(
        f"Initial SAS Verification DataFrame with Score column added:\n{sas_verification_df}"
    )
    ####### 2. Read general requirements
    # Extract General requirements section
    general_req_rows = common.extract_section(
        sas_verification_df, "SLIF - General requirements", col_name="Category"
    )
    # Remove all rows where every value is NaN
    general_req_rows = general_req_rows.dropna(how="all")

    if general_req_rows.empty:
        logger.warning("General requirements section is missing.")
        return sas_verification_df, scenario_scores_df, slif_score_dict, 0.0

    general_req_state = common.classify_pass_fail_section(
        general_req_rows, column="Value"
    )

    # Not yet assessed by the OEM -- unassessed (NaN), not a silent 0/FAIL.
    if general_req_state == "blank":
        logger.info(
            "General requirements not yet assessed. SLIF score remains unassessed."
        )
        slif_score_dict = {key: np.nan for key in slif_score_dict}
        return sas_verification_df, scenario_scores_df, slif_score_dict, None

    # If any general requirement failed, return score dict with all zeros
    if general_req_state == "fail":
        logger.info("One or more General requirements failed. Keeping score to 0.")
        return sas_verification_df, scenario_scores_df, slif_score_dict, 0.0

    ####### 3. SLIF Accuracy scoring
    # Reconstruct the collapsed Category/Scenario labels only -- NOT
    # "Value"/"Score": those are per-row OEM input, and forward-filling
    # them would silently copy a previous row's PASS/FAIL onto a row the
    # OEM left genuinely blank, turning an unassessed scenario into an
    # assessed one. DataFrame-sourced data can represent a blank label cell as
    # an empty string rather than NaN/pd.NA -- .ffill() only propagates
    # over actual nulls, so normalize is_empty_cell()-blank values first.
    ffill_columns = [
        c for c in sas_verification_df.columns if c not in ("Value", "Score")
    ]
    sas_verification_df[ffill_columns] = (
        sas_verification_df[ffill_columns]
        .map(lambda v: pd.NA if common.is_empty_cell(v) else v)
        .ffill()
    )
    accuracy_df = sas_verification_df[
        sas_verification_df["Category"] == "SLIF - Accuracy"
    ]
    # Distance based KPI
    dist_value_row = accuracy_df[accuracy_df["Scenario"] == "Distance based KPI"][
        "Value"
    ]
    dist_value = dist_value_row.iloc[0] if dist_value_row.size > 0 else "FAIL"
    dist_state = common.classify_pass_fail_value(dist_value)
    if dist_state == "pass":
        accuracy_score_dict["Distance based"] = 2.0
    elif dist_state == "blank":
        accuracy_score_dict["Distance based"] = np.nan
    if dist_state in ("pass", "blank"):
        sas_verification_df.loc[
            (sas_verification_df["Category"] == "SLIF - Accuracy")
            & (sas_verification_df["Scenario"] == "Distance based KPI"),
            "Score",
        ] = accuracy_score_dict["Distance based"]

    # Event Based KPI
    event_value_row = accuracy_df[accuracy_df["Scenario"] == "Event based KPI"]["Value"]
    event_value = event_value_row.iloc[0] if event_value_row.size > 0 else "FAIL"
    event_state = common.classify_pass_fail_value(event_value)
    if event_state == "pass":
        accuracy_score_dict["Event based"] = 2.0
    elif event_state == "blank":
        accuracy_score_dict["Event based"] = np.nan
    if event_state in ("pass", "blank"):
        sas_verification_df.loc[
            (sas_verification_df["Category"] == "SLIF - Accuracy")
            & (sas_verification_df["Scenario"] == "Event based KPI"),
            "Score",
        ] = accuracy_score_dict["Event based"]

    logger.info(
        f"SAS Accuracy Scores: Distance based: {accuracy_score_dict['Distance based']}, Event based: {accuracy_score_dict['Event based']}"
    )

    ####### 4. Conditional Speed Limits scoring
    # Read specified columns from sas_prediction_df
    slif_adv_max_scores_dict = {
        "Conditional speed limits - Rain / Wetness": 0.4,
        "Conditional speed limits - Snow / Icy": 0.4,
        "Conditional speed limits - Time / Season": 0.4,
        "Conditional speed limits - Distance for / Distance in": 0.4,
        "Conditional speed limits - Non-lane relevant arrows": 0.1,
        "Conditional speed limits - Lane relevant arrows": 0.1,
        "Conditional speed limits - Vehicle categories": 0.2,
        "Implicit speed limits - Highway, city and residential zones": 0.5,
        "Dynamic speed limits - Non-lane relevant": 0.25,
        "Dynamic speed limits - Lane relevant": 0.25,
    }

    for full_scenario_name in slif_adv_max_scores_dict.keys():

        current_scenario_row = sas_verification_df[
            (sas_verification_df["Scenario"] == full_scenario_name)
        ]

        if current_scenario_row["Value"].empty:
            logger.debug(
                f"Scenario '{full_scenario_name}' is missing in SLIF verification DataFrame."
            )
            continue
        current_scenario_value = current_scenario_row["Value"].iloc[0]
        if str(current_scenario_value).strip().lower() == "pass":
            max_score = slif_adv_max_scores_dict[full_scenario_name]
            advanced_score_dict[full_scenario_name] = max_score
            sas_verification_df.loc[
                (sas_verification_df["Scenario"] == full_scenario_name),
                "Score",
            ] = max_score
            logger.info(
                f"Scenario '{full_scenario_name}' passed. Assigned score: {max_score}"
            )
        else:
            logger.info(
                f"Scenario '{full_scenario_name}' did not pass. Score remains 0."
            )

    # If Conditional speed limits - Non-lane relevant arrows is 0, set Lane relevant arrows to 0
    if (
        advanced_score_dict["Conditional speed limits - Non-lane relevant arrows"]
        == 0.0
    ):
        advanced_score_dict["Conditional speed limits - Lane relevant arrows"] = 0.0
        sas_verification_df.loc[
            (
                sas_verification_df["Scenario"]
                == "Conditional speed limits - Lane relevant arrows"
            ),
            "Score",
        ] = 0.0

    # If Dynamic speed limits - Non-lane relevant is 0, set Lane relevant to 0
    if advanced_score_dict["Dynamic speed limits - Non-lane relevant"] == 0.0:
        advanced_score_dict["Dynamic speed limits - Lane relevant"] = 0.0
        sas_verification_df.loc[
            (sas_verification_df["Scenario"] == "Dynamic speed limits - Lane relevant"),
            "Score",
        ] = 0.0

    ####### 5. Local Hazards scoring
    # The PASS/FAIL gate is unchanged: a hazard/direction passes as soon as either
    # the Cloud or the Direct communication prediction is 'Green'. What the
    # available points are, however, depends on HOW MANY channels are 'Green':
    # 'Direct AND Cloud' scores the full points, 'Direct OR Cloud' the reduced
    # ones, per protocol section 1.2.3. The 'Max score' columns of the
    # verification and scoring sheets are never modified - only 'Score' changes.
    slif_local_hazards_table = get_local_hazard_table(sas_prediction_df)
    hazard_channel_status = get_local_hazard_channel_status(slif_local_hazards_table)

    # Tracks whether any hazard/direction is missing one of the two communication
    # channels ('Red' or not claimed), which also lowers the category total cap
    # from 3.0 to 2.5.
    single_channel_communication = False

    for current_scenario in LOCAL_HAZARD_COLUMNS:
        for direction in LOCAL_HAZARD_DIRECTIONS:
            scenario_name = f"{current_scenario} - {direction}"

            scenario_row = sas_verification_df[
                (sas_verification_df["Scenario"] == scenario_name)
            ]
            if scenario_row["Value"].empty:
                logger.debug(
                    f"Scenario '{scenario_name}' is missing in SLIF verification DataFrame."
                )
                continue

            channel_status = hazard_channel_status.get(current_scenario, {}).get(
                direction
            )
            if channel_status is None:
                logger.warning(
                    f"No Cloud / Direct communication prediction found for "
                    f"'{scenario_name}'. Assuming both communication channels."
                )
                channel_status = {"cloud": True, "direct": True, "has_red": False}

            # A 'Red' prediction lowers the category cap even when the other
            # channel still earns the (reduced) points.
            if channel_status["has_red"]:
                single_channel_communication = True

            scenario_value = scenario_row["Value"].iloc[0]
            if str(scenario_value).strip().lower() != "pass":
                logger.info(
                    f"Hazard scenario '{scenario_name}' did not pass. Score remains 0."
                )
                continue

            both_channels = channel_status["cloud"] and channel_status["direct"]
            if not both_channels:
                single_channel_communication = True

            full_points, reduced_points = LOCAL_HAZARD_POINTS[direction]
            score = full_points if both_channels else reduced_points

            hazard_score_dict[scenario_name] = score
            sas_verification_df.loc[
                (sas_verification_df["Scenario"] == scenario_name), "Score"
            ] = score
            logger.info(
                f"Hazard scenario '{scenario_name}' passed with "
                f"{'Cloud AND Direct' if both_channels else 'a single'} "
                f"communication channel (cloud={channel_status['cloud']}, "
                f"direct={channel_status['direct']}). Assigned score: {score}"
            )

    # Cap the Local Hazards category total: 3.0 when every scored hazard has both
    # Cloud AND Direct communication, 2.5 as soon as one channel is missing.
    hazard_capping_score = LOCAL_HAZARD_CAP_BOTH_CHANNELS
    if single_channel_communication:
        hazard_capping_score = LOCAL_HAZARD_CAP_SINGLE_CHANNEL
        logger.info(
            "At least one SLIF Local Hazard is not covered by both Cloud and Direct "
            f"communication. Capping total Local Hazards score to {hazard_capping_score}"
        )

    ####### 6. SLIF - System Updates scoring
    # temporary 1, continous 2
    system_updates_row = scenario_scores_df[
        scenario_scores_df["Category"].str.strip().str.lower()
        == "slif - system updates"
    ]
    system_updates_value = (
        system_updates_row["Scenario"].iloc[0] if not system_updates_row.empty else ""
    )
    if common.is_empty_cell(system_updates_value):
        # Caller-supplied dfs may carry a Scenario Scores sheet whose Scenario cell
        # was never stamped by preprocess; fall back to the raw Input
        # parameters value in that case. DataFrame-sourced inputs may also
        # differ in casing/whitespace, so match Category normalized (same
        # convention as the Scenario Scores lookup above).
        input_parameters_df = dfs.get("Input parameters", pd.DataFrame())
        if not input_parameters_df.empty and "Category" in input_parameters_df.columns:
            input_row = input_parameters_df[
                input_parameters_df["Category"].astype(str).str.strip().str.lower()
                == "slif - system updates"
            ]
            if not input_row.empty:
                system_updates_value = input_row["Value"].iloc[0]
    logger.info(
        f"System updates scenario value from Scenario Scores: '{system_updates_value}'"
    )
    idx = system_updates_row.index[0] if not system_updates_row.empty else None
    if idx is not None:
        if str(system_updates_value).strip().lower() == "continuous":
            scenario_scores_df.loc[idx, "Score"] = 2.0
            logger.info(f"System updates scenario is 'Continuous'. Assigned score: 2.0")
        elif str(system_updates_value).strip().lower() == "temporary":
            scenario_scores_df.loc[idx, "Score"] = 1.0
            logger.info(f"System updates scenario is 'Temporary'. Assigned score: 1.0")
        elif data_model.is_not_applicable(system_updates_value):
            scenario_scores_df.loc[idx, "Score"] = 0.0
            logger.info("System updates scenario is 'N/A'. Assigned score: 0.0")
    else:
        logger.info(
            f"System updates scenario is '{system_updates_value}'. "
            "Score remains unassessed (NaN)."
        )

    ####### 7. SCF - General requirements scoring

    speedo_row = sas_verification_df[
        (
            sas_verification_df["Category"].str.strip().str.lower()
            == "speed control function"
        )
        & (
            sas_verification_df["Scenario"].str.strip().str.lower()
            == "speedometer accuracy"
        )
    ]
    speedo_idx = speedo_row.index[0] if not speedo_row.empty else None

    if speedo_idx is None:
        raise ValueError(
            "No row with Category 'Speed control function' and Scenario 'Speedometer accuracy' found in SLIF verification DataFrame."
        )

    speedo_row_value = (
        sas_verification_df.loc[speedo_idx, "Value"] if speedo_idx is not None else ""
    )
    # Set intelligent_row_value to the next row after speedo_row
    intelligent_row_idx = speedo_idx + 1
    intelligent_row = (
        sas_verification_df.loc[intelligent_row_idx]
        if intelligent_row_idx in sas_verification_df.index
        else None
    )
    scf_points = 0

    if intelligent_row is None:
        # No ISL/ACC system configured (blank "Type of system" input) -- the
        # intelligent scenario row was never added during preprocessing.
        # scf_score_dict's defaults for both scenarios already stay at 0.0.
        logger.info(
            "No intelligent scenario row found after 'Speedometer accuracy' in SLIF "
            "verification DataFrame. Intelligent SCF score remains 0."
        )
    else:
        intelligent_row_scenario = intelligent_row["Scenario"]
        # A configured ISL/ACC system can still have a blank Value (not yet
        # tested) -- use the shared classifier instead of .strip() directly,
        # which crashes on NaN, and leave the row unassessed (NaN) rather
        # than silently treating blank as a FAIL.
        intelligent_row_state = common.classify_pass_fail_value(
            intelligent_row["Value"]
        )

        if intelligent_row_state == "blank":
            scf_points = np.nan
        elif intelligent_row_state == "pass":
            if intelligent_row_scenario == "Intelligent adaptive cruise control":
                scf_points = 8
            elif intelligent_row_scenario == "Intelligent speed limiter":
                scf_points = 5

        scf_score_dict[intelligent_row_scenario] = scf_points
        sas_verification_df.loc[
            (sas_verification_df["Scenario"] == intelligent_row_scenario),
            "Score",
        ] = scf_points

    speed_multiplier = 0
    if speedo_row_value == "≥-3 km/h":
        speed_multiplier = 0
    elif speedo_row_value == "≥-5 km/h":
        speed_multiplier = 0.5
    elif speedo_row_value == "<-5 km/h":
        speed_multiplier = 1.0

    speedometer_accuracy_score = -scf_points * speed_multiplier
    scf_score_dict["Speedometer accuracy"] = speedometer_accuracy_score
    sas_verification_df.loc[
        (sas_verification_df["Scenario"] == "Speedometer accuracy"),
        "Score",
    ] = speedometer_accuracy_score
    logger.debug(
        f"Speedometer accuracy value: '{speedo_row_value}', Speed multiplier: {speed_multiplier}, Speedometer accuracy score: {speedometer_accuracy_score}"
    )

    sas_verification_df = common.opposite_ffill(
        sas_verification_df, exclude_columns=["Value", "Score"]
    )

    slif_score_dict = {
        **accuracy_score_dict,
        **advanced_score_dict,
        **hazard_score_dict,
        **scf_score_dict,
    }

    return (
        sas_verification_df,
        scenario_scores_df,
        slif_score_dict,
        hazard_capping_score,
    )
