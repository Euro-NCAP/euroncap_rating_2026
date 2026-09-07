# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import pandas as pd
import logging
from typing import Dict, Tuple
import logging
from euroncap_rating_2026 import common
import numpy as np
import io

logger = logging.getLogger(__name__)

REAR_SEAT_NUMBER_SEQUENCE = [4, 6, 5, 7, 9, 8]

SEATBELT_USAGE_CSV_STRING = """
Category;Scenario;Value
General requirements;;
Correct belt routing;Seatbelt buckle only;
;Seatbelt completely behind back;
;Lap belt only;
"""


def generate_seatbelt_usage_df(dfs: dict[str, pd.DataFrame]) -> Dict:
    logger.info("Preprocess Seatbelt usage ...")
    seatbelt_usage_df = pd.read_csv(io.StringIO(SEATBELT_USAGE_CSV_STRING), sep=";")
    seatbelt_usage_df = common.opposite_ffill(
        seatbelt_usage_df, exclude_columns=["Value", "Score"]
    )
    return seatbelt_usage_df


def get_rear_seat_labels(num_rear_seats: int) -> list[str]:
    if common.is_empty_cell(num_rear_seats):
        logger.warning("Number of rear seats not found in param_dict")
        return []

    num_rear_seats = int(num_rear_seats)
    if num_rear_seats > len(REAR_SEAT_NUMBER_SEQUENCE):
        logger.error(
            f"Number of rear seats ({num_rear_seats}) exceeds maximum allowed ({len(REAR_SEAT_NUMBER_SEQUENCE)})"
        )
        raise ValueError(
            f"Number of rear seats ({num_rear_seats}) exceeds maximum allowed ({len(REAR_SEAT_NUMBER_SEQUENCE)})"
        )

    seat_numbers = sorted(REAR_SEAT_NUMBER_SEQUENCE[:num_rear_seats])
    return [f"Seat {seat_number}" for seat_number in seat_numbers]


def generate_seatbelt_usage_rear_seat_df(num_rear_seats):
    rear_seat_labels = get_rear_seat_labels(num_rear_seats)
    if not rear_seat_labels:
        return None

    # Generate rows for each rear seat
    rows = []
    for index, seat_name in enumerate(rear_seat_labels):
        row = {
            "Category": np.nan,
            "Scenario": seat_name,
            "Value": np.nan,
        }
        if index == 0:
            row["Category"] = "Rear seat occupancy"
        rows.append(row)

    return pd.DataFrame(rows)


def update_scenario_scores_with_rear_seats(scenario_scores_df, num_rear_seats):
    if scenario_scores_df.empty:
        logger.warning("Scenario Scores sheet is missing or empty.")
        return scenario_scores_df

    if common.is_empty_cell(num_rear_seats):
        logger.warning("Number of rear seats not found in param_dict")
        return scenario_scores_df

    rear_seat_labels = get_rear_seat_labels(num_rear_seats)
    num_rear_seats = len(rear_seat_labels)

    # Find indices where Category is 'Rear Seat Occupancy'
    mask = (
        scenario_scores_df["Category"].str.strip().str.lower() == "rear seat occupancy"
    )
    indices = scenario_scores_df.index[mask].tolist()

    # If there are rear seat occupancy rows, replace them in-place
    if indices:
        # Prepare new rows
        new_rear_seat_rows = []
        for index, seat_name in enumerate(rear_seat_labels):
            new_row = {
                "Stage element": "",
                "Stage subelement": "",
                "Scenario": seat_name,
                "Score": 0.0,
                "Max score": float(5 / num_rear_seats),
            }
            if index == 0:
                new_row["Category"] = "Rear seat occupancy"
            else:
                new_row["Category"] = ""
            new_rear_seat_rows.append(new_row)

        # Build new DataFrame
        df_before = scenario_scores_df.iloc[: indices[0]]
        df_after = scenario_scores_df.iloc[indices[-1] + 1 :]
        scenario_scores_df = pd.concat(
            [df_before, pd.DataFrame(new_rear_seat_rows), df_after], ignore_index=True
        )
    else:
        logger.warning(
            "No rows with Category 'Rear Seat Occupancy' found in Scenario Scores sheet."
        )

    return scenario_scores_df


def compute_seatbelt_score(
    dfs: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Compute the seatbelt score based on occupant monitoring data."""

    om_df = dfs.get("OM - Seatbelt usage verif.")
    if om_df is None or om_df.empty:
        logging.warning("OM - Seatbelt usage verif. sheet is missing or empty.")
        return pd.DataFrame(), {}

    param_dict = common.create_param_dict_from_input_parameters(
        dfs.get("Input parameters", pd.DataFrame()), col="Category"
    )
    rear_seats_info = param_dict.get("Rear seat occupancy", {})
    num_rear_seats = rear_seats_info.get("Number of rear seats")
    rear_seat_labels = get_rear_seat_labels(num_rear_seats)

    # NaN means "not yet assessed by the OEM" -- distinct from an explicit
    # FAIL (0.0). Only overwritten below once a row is actually PASS/FAIL.
    om_df["Score"] = np.nan
    seatbelt_score_dict = {}
    seatbelt_score_dict["Seatbelt buckle only"] = np.nan
    seatbelt_score_dict["Seatbelt completely behind back"] = np.nan
    seatbelt_score_dict["Lap belt only"] = np.nan
    for seat_name in rear_seat_labels:
        seatbelt_score_dict[seat_name] = np.nan

    general_req_rows = common.extract_section(
        om_df, "General requirements", col_name="Category"
    )
    correct_belt_routing_rows = common.extract_section(
        om_df, "Correct belt routing", col_name="Category"
    )
    rear_seat_occupancy_rows = common.extract_section(
        om_df, "Rear seat occupancy", col_name="Category"
    )

    if general_req_rows.empty:
        logging.warning("General requirements section is missing.")

    general_req_states = general_req_rows["Value"].apply(
        common.classify_pass_fail_value
    )

    if (general_req_states == "blank").any():
        logging.info(
            "General requirements not yet assessed. Leaving Seatbelt usage unassessed."
        )
        return om_df, seatbelt_score_dict

    if (general_req_states == "fail").any():
        logging.info("One or more General requirements failed. Keeping score to 0.")
        for key in seatbelt_score_dict:
            seatbelt_score_dict[key] = 0.0
        return om_df, seatbelt_score_dict

    # Check for 'Seatbelt buckle only'
    seatbelt_buckle_row = correct_belt_routing_rows[
        correct_belt_routing_rows["Scenario"].str.lower() == "seatbelt buckle only"
    ]
    idx = seatbelt_buckle_row.index[0]
    state = common.classify_pass_fail_value(seatbelt_buckle_row.iloc[0]["Value"])
    if state != "blank":
        score = 2.0 if state == "pass" else 0.0
        om_df.at[idx, "Score"] = score
        seatbelt_score_dict["Seatbelt buckle only"] = score

    # Check for 'Seatbelt completely behind back'
    seatbelt_behind_back_row = correct_belt_routing_rows[
        correct_belt_routing_rows["Scenario"].str.lower()
        == "seatbelt completely behind back"
    ]
    idx = seatbelt_behind_back_row.index[0]
    state = common.classify_pass_fail_value(seatbelt_behind_back_row.iloc[0]["Value"])
    if state != "blank":
        score = 1.0 if state == "pass" else 0.0
        om_df.at[idx, "Score"] = score
        seatbelt_score_dict["Seatbelt completely behind back"] = score

    # Check for 'Lap belt only'
    lap_belt_only_row = correct_belt_routing_rows[
        correct_belt_routing_rows["Scenario"].str.lower() == "lap belt only"
    ]
    idx = lap_belt_only_row.index[0]
    state = common.classify_pass_fail_value(lap_belt_only_row.iloc[0]["Value"])
    if state != "blank":
        score = 2.0 if state == "pass" else 0.0
        om_df.at[idx, "Score"] = score
        seatbelt_score_dict["Lap belt only"] = score

    # Assign score for rear seat occupancy rows
    score_per_row = (1 / len(rear_seat_occupancy_rows)) * 5
    for seat_name, (idx, row) in zip(
        rear_seat_labels, rear_seat_occupancy_rows.iterrows()
    ):
        state = common.classify_pass_fail_value(row["Value"])
        if state != "blank":
            score = score_per_row if state == "pass" else 0.0
            om_df.at[idx, "Score"] = score
            seatbelt_score_dict[seat_name] = score
    return om_df, seatbelt_score_dict
