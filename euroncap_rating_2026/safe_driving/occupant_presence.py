# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import io

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple
from euroncap_rating_2026 import common
from euroncap_rating_2026.safe_driving import data_model

logger = logging.getLogger(__name__)

OCCUPANT_PRESENCE_ALL_CSV_STRING = """
CPD_code;Scenario;Element
clb_front_warning; Child left behind;Front Seat - Warning
clb_front_intervention; Child left behind;Front Seat - Intervention
clb_rear_warning; Child left behind;Rear Seat - Warning
clb_rear_intervention; Child left behind;Rear Seat - Intervention
ceuv_front_warning; Child enters unlocked vehicle;Front compartment - Warning
ceuv_front_intervention; Child enters unlocked vehicle;Front compartment - Intervention
ceuv_rear_warning; Child enters unlocked vehicle;Rear Seat - Warning
ceuv_rear_intervention; Child enters unlocked vehicle;Rear Seat - Intervention
"""


def get_cpd_score_df() -> pd.DataFrame:
    """Get the DataFrame with the scenarios and elements for Child Presence Detection scoring."""
    cols = [
        "clb_front_warning",
        "clb_front_intervention",
        "clb_rear_warning",
        "clb_rear_intervention",
        "ceuv_front_warning",
        "ceuv_front_intervention",
        "ceuv_rear_warning",
        "ceuv_rear_intervention",
        "child_left_behind_score",
        "child_enters_unlocked_vehicle_score",
        "final_score",
    ]

    data = [
        ["PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", 4, 1, 5],
        ["PASS", "PASS", "PASS", "PASS", "FAIL", "FAIL", "PASS", "PASS", 4, 0.5, 4.5],
        ["PASS", "PASS", "PASS", "PASS", "FAIL", "FAIL", "FAIL", "FAIL", 4, 0, 4],
        ["PASS", "FAIL", "PASS", "FAIL", "PASS", "PASS", "PASS", "PASS", 3, 1, 4],
        ["PASS", "FAIL", "PASS", "FAIL", "PASS", "FAIL", "PASS", "FAIL", 3, 1, 4],
        ["PASS", "FAIL", "PASS", "FAIL", "FAIL", "FAIL", "PASS", "PASS", 3, 0.5, 3.5],
        ["PASS", "FAIL", "PASS", "FAIL", "FAIL", "FAIL", "PASS", "FAIL", 3, 0.5, 3.5],
        ["PASS", "FAIL", "PASS", "FAIL", "FAIL", "FAIL", "FAIL", "FAIL", 3, 0, 3],
        ["FAIL", "FAIL", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", 3, 1, 4],
        ["FAIL", "FAIL", "PASS", "PASS", "FAIL", "PASS", "PASS", "FAIL", 3, 0.5, 3.5],
        ["FAIL", "FAIL", "PASS", "PASS", "FAIL", "FAIL", "FAIL", "FAIL", 3, 0, 3],
        ["FAIL", "FAIL", "PASS", "FAIL", "PASS", "PASS", "PASS", "PASS", 1.5, 1, 2.5],
        ["FAIL", "FAIL", "PASS", "FAIL", "PASS", "FAIL", "PASS", "FAIL", 1.5, 1, 2.5],
        ["FAIL", "FAIL", "PASS", "FAIL", "FAIL", "FAIL", "PASS", "PASS", 1.5, 0.5, 2],
        ["FAIL", "FAIL", "PASS", "FAIL", "FAIL", "FAIL", "PASS", "FAIL", 1.5, 0.5, 2],
        ["FAIL", "FAIL", "PASS", "FAIL", "FAIL", "FAIL", "FAIL", "FAIL", 1.5, 0, 1.5],
    ]

    df = pd.DataFrame(data, columns=cols)
    return df


def _generate_occupant_presence_rows(category_name, scenario_data, scenario_name):
    """Helper function to generate rows for occupant presence scenarios."""
    rows = []

    system_type = scenario_data.get("Type of system")
    if system_type is None:
        logger.warning(f"{category_name} scenario not found in param_dict")
        return rows

    seat_coverage = scenario_data.get("Seat coverage")
    # Determine which seats to cover
    seats_to_cover = []
    if seat_coverage == data_model.OPSeatCoverage.ALL_PASSENGER_SEATS.value:
        seats_to_cover = ["Front Seat", "Rear Seat"]
    elif seat_coverage == data_model.OPSeatCoverage.REAR_SEATS.value:
        seats_to_cover = ["Rear Seat"]

    # Generate rows based on type of system and seat coverage
    for seat in seats_to_cover:
        if seat == "Front Seat" and category_name == "Child enters unlocked vehicle":
            seat = "Front compartment"

        if system_type == data_model.OPTypeOfSystem.WARNING.value:
            rows.append(
                {
                    "Category": "Child presence detection",
                    "Scenario": scenario_name,
                    "Element": f"{seat} - Warning",
                    "Value": np.nan,
                }
            )
        elif system_type == data_model.OPTypeOfSystem.WARNING_AND_INTERVENTION.value:
            rows.append(
                {
                    "Category": "Child presence detection",
                    "Scenario": scenario_name,
                    "Element": f"{seat} - Warning",
                    "Value": np.nan,
                }
            )
            rows.append(
                {
                    "Category": "Child presence detection",
                    "Scenario": scenario_name,
                    "Element": f"{seat} - Intervention",
                    "Value": np.nan,
                }
            )

    return rows


def _extract_scenario_input_params(
    input_parameters_df: pd.DataFrame, scenario_name: str
) -> dict:
    """Input parameter -> Value for *scenario_name*'s row and the row below it.

    The "Input parameters" sheet lists each child-presence scenario on one row
    (Type of system) with its second parameter (Seat coverage) on the next,
    Scenario cell blank -- so the row after each match is included too.
    """
    scenario_data = {}
    if input_parameters_df.empty or "Scenario" not in input_parameters_df.columns:
        return scenario_data
    # Work on positions, not index labels: caller-supplied dfs may carry a
    # non-default index, where label arithmetic (.index + 1) and positional
    # selection (.iloc) would disagree.
    matched_positions = np.flatnonzero(
        input_parameters_df["Scenario"].astype(str).str.lower() == scenario_name.lower()
    )
    # Add also the next row after each matched row
    selected_positions = sorted(
        set(matched_positions)
        | {p + 1 for p in matched_positions if p + 1 < len(input_parameters_df)}
    )
    for pos in selected_positions:
        row = input_parameters_df.iloc[pos]
        key = row.get("Input parameter")
        value = row.get("Value")
        if pd.notna(key):
            scenario_data[key] = value
    return scenario_data


def generate_occupant_presence_df(input_parameters_df: pd.DataFrame) -> pd.DataFrame:

    # Extract parameters for 'Child left behind'
    child_left_behind_data = _extract_scenario_input_params(
        input_parameters_df, "Child left behind"
    )
    logger.debug(
        f"Extracted child left behind data from input parameters:\n{child_left_behind_data}"
    )
    # Get Child Left Behind data
    child_left_behind_rows = _generate_occupant_presence_rows(
        "Child left behind", child_left_behind_data, "Child left behind"
    )

    # Get Child enters unlocked vehicle data
    child_enters_data = _extract_scenario_input_params(
        input_parameters_df, "Child enters unlocked vehicle"
    )
    logger.debug(
        f"Extracted child enters unlocked vehicle data from input parameters:\n{child_enters_data}"
    )
    child_enters_rows = _generate_occupant_presence_rows(
        "Child enters unlocked vehicle",
        child_enters_data,
        "Child enters unlocked vehicle",
    )
    # Add the required row on top
    header_row = {
        "Category": "Child presence detection",
        "Scenario": "General requirements",
        "Element": "",
        "Value": np.nan,
    }
    crash_rows = [
        {
            "Category": "Crash occupancy information",
            "Scenario": "General requirements",
            "Element": "",
            "Value": np.nan,
        },
        {"Category": "", "Scenario": "Adult occupants", "Element": "", "Value": np.nan},
        {
            "Category": "",
            "Scenario": "Children in all CRS",
            "Element": "",
            "Value": np.nan,
        },
    ]
    rows = [header_row] + child_left_behind_rows + child_enters_rows + crash_rows
    occupant_presence_df = pd.DataFrame(rows)
    occupant_presence_df = common.opposite_ffill(
        occupant_presence_df, exclude_columns=["Element", "Value"]
    )
    return occupant_presence_df


def preprocess(dfs: Dict[str, pd.DataFrame]) -> Dict:
    """Preprocess the dataframes for occupant presence"""

    input_parameters_df = dfs.get("Input parameters", pd.DataFrame())
    occupant_presence_df = generate_occupant_presence_df(input_parameters_df)
    return occupant_presence_df


def compute_classification_score(
    dfs: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Compute the occupant classification score."""

    om_df = dfs.get("OM - Occ. presence verif.")
    if om_df is None or om_df.empty:
        logging.warning("OM - Occ. presence verif. sheet is missing or empty.")
        return pd.DataFrame(), {}

    # NaN means "not yet assessed by the OEM" -- distinct from an explicit
    # FAIL (0.0). Only overwritten below once a row is actually PASS/FAIL.
    om_df["Score"] = np.nan
    presence_score_dict = {}

    # Child Presence Detection Scores
    child_presence_detection = common.extract_section(
        om_df, "Child presence detection", col_name="Category"
    )
    child_general_req = common.extract_section(
        child_presence_detection, "General requirements", col_name="Scenario"
    )

    child_general_req_states = child_general_req["Value"].apply(
        common.classify_pass_fail_value
    )
    any_child_general_req_blank = (child_general_req_states == "blank").any()
    any_child_general_req_failed = (child_general_req_states == "fail").any()

    # Reconstruct the Category/Scenario/Element columns that opposite_ffill
    # collapsed at generation time -- but NOT "Value"/"Score": those are
    # per-row OEM input, and forward-filling them would silently copy a
    # previous row's PASS/FAIL onto a row the OEM left genuinely blank.
    # DataFrame-sourced data (e.g. parquet) represents a collapsed cell as an
    # empty string, not NaN/pd.NA -- .ffill() only propagates over actual
    # nulls, so normalize is_empty_cell()-blank values to NaN first or a
    # caller-supplied empty string here silently blocks the whole reconstruction.
    ffill_columns = [c for c in om_df.columns if c not in ("Value", "Score")]
    om_df[ffill_columns] = (
        om_df[ffill_columns]
        .map(lambda v: pd.NA if common.is_empty_cell(v) else v)
        .ffill()
    )

    if any_child_general_req_blank:
        presence_score_dict["Child left behind"] = np.nan
        presence_score_dict["Child enters unlocked vehicle"] = np.nan
    elif any_child_general_req_failed:
        presence_score_dict["Child left behind"] = 0.0
        presence_score_dict["Child enters unlocked vehicle"] = 0.0
    else:
        # Create the DataFrame with the required scenarios and elements
        scenario_element_df = pd.read_csv(
            io.StringIO(OCCUPANT_PRESENCE_ALL_CSV_STRING), sep=";"
        )
        scenario_element_df["Scenario"] = (
            scenario_element_df["Scenario"].astype(str).str.strip()
        )
        scenario_element_df["Element"] = (
            scenario_element_df["Element"].astype(str).str.strip()
        )

        om_for_merge = om_df.copy()
        om_for_merge["Scenario"] = om_for_merge["Scenario"].astype(str).str.strip()
        om_for_merge["Element"] = om_for_merge["Element"].astype(str).str.strip()

        # Merge scenario+element with OM values. indicator=True distinguishes
        # "this CPD element doesn't exist for this vehicle" (no match --
        # correctly treated as FAIL below, a real scoring rule, not a bug)
        # from "this CPD element exists but the OEM hasn't assessed it yet"
        # (matched, but Value is blank).
        merged = scenario_element_df.merge(
            om_for_merge[["Scenario", "Element", "Value"]],
            on=["Scenario", "Element"],
            how="left",
            indicator=True,
        )

        any_applicable_element_blank = (
            (merged["_merge"] == "both") & merged["Value"].apply(common.is_empty_cell)
        ).any()

        if any_applicable_element_blank:
            presence_score_dict["Child left behind"] = np.nan
            presence_score_dict["Child enters unlocked vehicle"] = np.nan
        else:
            # Missing entries (element not applicable to this vehicle) are FAIL
            merged["Value"] = (
                merged["Value"].fillna("FAIL").astype(str).str.strip().str.upper()
            )
            logger.debug(
                f"Merged DataFrame for CPD scoring after filling missing values:\n{merged}"
            )

            # Long -> wide: CPD_code becomes columns, Value becomes row values
            cpd_input_wide = (
                merged[["CPD_code", "Value"]]
                .drop_duplicates(subset=["CPD_code"], keep="last")
                .set_index("CPD_code")
                .T.reset_index(drop=True)
            )
            logger.debug(f"CPD input wide DataFrame:\n{cpd_input_wide}")

            # Merge with CPD score lookup table
            cpd_score_df = get_cpd_score_df()
            cpd_cols = [
                "clb_front_warning",
                "clb_front_intervention",
                "clb_rear_warning",
                "clb_rear_intervention",
                "ceuv_front_warning",
                "ceuv_front_intervention",
                "ceuv_rear_warning",
                "ceuv_rear_intervention",
            ]

            # Ensure all required CPD columns exist
            for col in cpd_cols:
                if col not in cpd_input_wide.columns:
                    cpd_input_wide[col] = "FAIL"

            cpd_input_wide = cpd_input_wide[cpd_cols]

            scored = cpd_input_wide.merge(cpd_score_df, on=cpd_cols, how="left")

            if scored.empty or scored["final_score"].isna().all():
                presence_score_dict["Child left behind"] = 0.0
                presence_score_dict["Child enters unlocked vehicle"] = 0.0
            else:
                score_row = scored.iloc[0]
                presence_score_dict["Child left behind"] = float(
                    score_row["child_left_behind_score"]
                )
                presence_score_dict["Child enters unlocked vehicle"] = float(
                    score_row["child_enters_unlocked_vehicle_score"]
                )

    # N/A override: an explicit "N/A" for Type of system or Seat coverage in
    # the Input parameters sheet means the system is not fitted -- the
    # scenario scores a real 0.0 (never NaN), regardless of the General
    # requirements row, which may legitimately stay blank in that case.
    input_parameters_df = dfs.get("Input parameters", pd.DataFrame())
    for scenario in ("Child left behind", "Child enters unlocked vehicle"):
        scenario_params = _extract_scenario_input_params(input_parameters_df, scenario)
        if data_model.is_not_applicable(
            scenario_params.get("Type of system")
        ) or data_model.is_not_applicable(scenario_params.get("Seat coverage")):
            logger.info(
                f"'{scenario}' input parameters are N/A; scoring the scenario 0.0."
            )
            presence_score_dict[scenario] = 0.0

    # Crash Occupancy Information Scores
    crash_occupancy_information = om_df[
        om_df["Category"] == "Crash occupancy information"
    ]
    crash_general_req = crash_occupancy_information[
        crash_occupancy_information["Scenario"] == "General requirements"
    ]

    crash_general_req_states = crash_general_req["Value"].apply(
        common.classify_pass_fail_value
    )
    any_crash_general_req_blank = (crash_general_req_states == "blank").any()
    any_crash_general_req_failed = (crash_general_req_states == "fail").any()

    if any_crash_general_req_blank:
        presence_score_dict["Adult occupants"] = np.nan
        presence_score_dict["Children in all CRS"] = np.nan
    elif any_crash_general_req_failed:
        presence_score_dict["Adult occupants"] = 0.0
        presence_score_dict["Children in all CRS"] = 0.0
    else:
        presence_score_dict["Adult occupants"] = np.nan
        presence_score_dict["Children in all CRS"] = np.nan

        belt_in_use_row = crash_occupancy_information[
            crash_occupancy_information["Scenario"].str.lower() == "adult occupants"
        ]
        state = common.classify_pass_fail_value(belt_in_use_row.iloc[0]["Value"])
        if state != "blank":
            idx = belt_in_use_row.index[0]
            score = 4.0 if state == "pass" else 0.0
            om_df.at[idx, "Score"] = score
            presence_score_dict["Adult occupants"] = score

        children_crs_row = crash_occupancy_information[
            crash_occupancy_information["Scenario"].str.lower() == "children in all crs"
        ]
        if not children_crs_row.empty:
            state = common.classify_pass_fail_value(children_crs_row.iloc[0]["Value"])
            if state != "blank":
                idx = children_crs_row.index[0]
                score = 1.0 if state == "pass" else 0.0
                om_df.at[idx, "Score"] = score
                presence_score_dict["Children in all CRS"] = score

    logger.info(f"presence_score_dict:\n{presence_score_dict}")

    om_df = common.opposite_ffill(om_df, exclude_columns=["Value"])
    return om_df, presence_score_dict
