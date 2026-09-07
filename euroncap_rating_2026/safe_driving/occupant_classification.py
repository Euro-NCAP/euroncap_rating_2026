# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import io

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple
from euroncap_rating_2026 import common
from euroncap_rating_2026.safe_driving import data_model


logger = logging.getLogger(__name__)

OCCUPANT_CLASSIFICATION_CSV_STRING = """
Category;Scenario;Value
Passenger airbag status;;
Out of position;Close proximity to the airbag;
;Feet on dashboard;
Occupant stature classification;Driver;
;Front seat passenger;
"""


def get_passenger_airbag_scenario(airbag_system, airbag_switch):
    if airbag_system == data_model.OCTypeOfSystem.NOT_APPLICABLE:
        return data_model.NOT_APPLICABLE
    elif airbag_system == data_model.OCTypeOfSystem.AUTOMATIC:
        return "Automatic"
    elif (
        airbag_system == data_model.OCTypeOfSystem.SYSTEM_ADVISED
        and airbag_switch == data_model.OCTypeOfSwitch.SOFTWARE
    ):
        return "System advised with manual software switch"
    elif (
        airbag_system == data_model.OCTypeOfSystem.SYSTEM_ADVISED
        and airbag_switch == data_model.OCTypeOfSwitch.HARDWARE
    ):
        return "System advised with manual hardware switch"
    elif airbag_system == data_model.OCTypeOfSystem.MANUAL:
        return "Manual (software or hardware switch)"
    else:
        return ""


def preprocess_occupant_classification(dfs: dict[str, pd.DataFrame]) -> Dict:
    logger.info("Preprocess Occupant Classification ...")
    occupant_classification_df = pd.read_csv(
        io.StringIO(OCCUPANT_CLASSIFICATION_CSV_STRING), sep=";"
    )
    occupant_classification_df = common.opposite_ffill(
        occupant_classification_df, exclude_columns=["Value", "Score"]
    )

    param_dict = common.create_param_dict_from_input_parameters(
        dfs.get("Input parameters", pd.DataFrame()), col="Category"
    )

    airbat_status_info = param_dict.get("Passenger airbag status", {})
    airbag_system = airbat_status_info.get("Type of system")
    airbag_switch = airbat_status_info.get("Type of switch")

    passenger_airbag_scenario = get_passenger_airbag_scenario(
        airbag_system, airbag_switch
    )

    occupant_classification_df.loc[
        occupant_classification_df["Category"] == "Passenger airbag status", "Scenario"
    ] = passenger_airbag_scenario

    scenario_scores_df = dfs.get("Scenario Scores", pd.DataFrame())
    # Update scenario_scores_df where Category is "Passenger airbag status"
    scenario_scores_df.loc[
        scenario_scores_df["Category"] == "Passenger airbag status", "Scenario"
    ] = passenger_airbag_scenario

    return occupant_classification_df, scenario_scores_df


def compute_classification_score(
    dfs: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Compute the occupant classification score."""

    param_dict = common.create_param_dict_from_input_parameters(
        dfs.get("Input parameters", pd.DataFrame()), col="Category"
    )

    airbag_status_info = param_dict.get("Passenger airbag status", {})
    type_of_system_raw = airbag_status_info.get("Type of system")
    type_of_switch_raw = airbag_status_info.get("Type of switch")
    type_of_system = (
        data_model.OCTypeOfSystem(type_of_system_raw)
        if pd.notna(type_of_system_raw)
        else None
    )
    type_of_switch = (
        data_model.OCTypeOfSwitch(type_of_switch_raw)
        if pd.notna(type_of_switch_raw)
        else None
    )

    passenger_airbag_scenario = get_passenger_airbag_scenario(
        type_of_system, type_of_switch
    )

    classification_score_dict = {}

    # Merge belt keys and their max scores into a single dictionary
    classification_scores = {
        "Close proximity to the airbag": 1.0,
        "Feet on dashboard": 1.0,
        "Driver": 3.0,
        "Front seat passenger": 1.0,
    }

    # Initialize dict. NaN means "not yet assessed by the OEM" -- distinct
    # from an explicit FAIL (0.0). Only overwritten below once a row is
    # actually PASS/FAIL.
    for key in classification_scores:
        classification_score_dict[key] = np.nan

    if passenger_airbag_scenario:
        classification_score_dict[passenger_airbag_scenario] = np.nan
    else:
        logger.warning(
            "Passenger airbag scenario could not be determined from input parameters; skipping scenario-specific score key initialization."
        )

    om_df = dfs.get("OM - Occ. classification verif.")
    if om_df is None or om_df.empty:
        logging.warning("OM - Occ. classification verif. sheet is missing or empty.")
        return pd.DataFrame(), classification_score_dict

    logger.info(
        f"Occupant Classification type of system: {type_of_system} - type of switch: {type_of_switch}"
    )

    # NaN means "not yet assessed by the OEM" -- distinct from an explicit
    # FAIL (0.0). Only overwritten below once a row is actually PASS/FAIL.
    om_df["Score"] = np.nan

    passenger_airbag_status_rows = common.extract_section(
        om_df, "Passenger airbag status", col_name="Category"
    )
    out_of_position_rows = common.extract_section(
        om_df, "Out of position", col_name="Category"
    )
    occupant_stature_classification_rows = common.extract_section(
        om_df, "Occupant stature classification", col_name="Category"
    )

    passenger_airbag_states = passenger_airbag_status_rows["Value"].apply(
        common.classify_pass_fail_value
    )
    any_passenger_airbag_blank = (passenger_airbag_states == "blank").any()
    all_passenger_airbag_status_pass = (passenger_airbag_states == "pass").all()

    if type_of_system == data_model.OCTypeOfSystem.NOT_APPLICABLE:
        # N/A means no passenger airbag disabling system is fitted: the
        # element scores a real 0.0 and its Value cell legitimately stays
        # blank (no input required), so the blank->NaN gate must not apply.
        passenger_airbag_score = 0.0
    elif any_passenger_airbag_blank:
        passenger_airbag_score = np.nan
    else:
        passenger_airbag_score = 0.0
        if (
            type_of_system == data_model.OCTypeOfSystem.AUTOMATIC
            and all_passenger_airbag_status_pass
        ):
            passenger_airbag_score = 4.0
        elif (
            type_of_system == data_model.OCTypeOfSystem.SYSTEM_ADVISED
            and type_of_switch == data_model.OCTypeOfSwitch.SOFTWARE
            and all_passenger_airbag_status_pass
        ):
            passenger_airbag_score = 3.0
        elif (
            type_of_system == data_model.OCTypeOfSystem.SYSTEM_ADVISED
            and type_of_switch == data_model.OCTypeOfSwitch.HARDWARE
            and all_passenger_airbag_status_pass
        ):
            passenger_airbag_score = 2.0
        elif (
            type_of_system == data_model.OCTypeOfSystem.MANUAL
            and all_passenger_airbag_status_pass
        ):
            passenger_airbag_score = 1.0

    if passenger_airbag_scenario:
        classification_score_dict[passenger_airbag_scenario] = passenger_airbag_score

    generic_rows_df = pd.concat(
        [out_of_position_rows, occupant_stature_classification_rows]
    )
    # Iterate over each classification key and assign scores if conditions are met
    # This loop checks if each classification scenario (key) is present in the concatenated DataFrame (generic_rows_df),
    # and assigns the max score on PASS, 0.0 on an explicit FAIL, or leaves it
    # unassessed (NaN) if the OEM hasn't filled the row in yet.
    for key, max_score in classification_scores.items():
        row = generic_rows_df[generic_rows_df["Scenario"].str.lower() == key.lower()]
        if row.empty:
            continue
        state = common.classify_pass_fail_value(row.iloc[0]["Value"])
        if state == "blank":
            continue
        idx = row.index[0]
        score = max_score if state == "pass" else 0.0
        om_df.at[idx, "Score"] = score
        classification_score_dict[key] = score

    return om_df, classification_score_dict
