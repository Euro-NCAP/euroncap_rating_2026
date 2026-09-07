# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import io
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from euroncap_rating_2026 import common
import logging


logger = logging.getLogger(__name__)

STEERING_ASSISTANCE_CSV_STRING = """
Category;Scenario;Value
Steering assistance;S-bend @ 60 km/h - First turn;
;S-bend @ 60 km/h - Second turn;
;S-bend @ 80 km/h - First turn;
;S-bend @ 80 km/h - Second turn;
;S-bend @ 100 km/h - First turn;
;S-bend @ 100 km/h - Second turn;
;S-bend @ 130 km/h - First turn;
;S-bend @ 130 km/h - Second turn;
"""


def preprocess(dfs: dict[str, pd.DataFrame]) -> Dict:
    logger.info("Preprocess Steering assistance ...")
    steering_assistance_df = pd.read_csv(
        io.StringIO(STEERING_ASSISTANCE_CSV_STRING), sep=";"
    )
    steering_assistance_df = common.opposite_ffill(
        steering_assistance_df, exclude_columns=["Value", "Score"]
    )
    return steering_assistance_df


def compute_score(
    dfs: dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    steering_assistance_score_dict = {}

    # NaN means "not yet assessed by the OEM" -- distinct from an explicit
    # FAIL (0.0). Only overwritten below once a row/parameter is actually filled in.
    steering_assistance_score_dict["S-bend @ 60 km/h - First turn"] = np.nan
    steering_assistance_score_dict["S-bend @ 60 km/h - Second turn"] = np.nan
    steering_assistance_score_dict["S-bend @ 80 km/h - First turn"] = np.nan
    steering_assistance_score_dict["S-bend @ 80 km/h - Second turn"] = np.nan
    steering_assistance_score_dict["S-bend @ 100 km/h - First turn"] = np.nan
    steering_assistance_score_dict["S-bend @ 100 km/h - Second turn"] = np.nan
    steering_assistance_score_dict["S-bend @ 130 km/h - First turn"] = np.nan
    steering_assistance_score_dict["S-bend @ 130 km/h - Second turn"] = np.nan

    steering_assistance_score_dict["Lane change assist"] = np.nan

    steering_assistance_df = dfs.get("VA - Steering assistance verif.")
    if steering_assistance_df is None or steering_assistance_df.empty:
        logger.warning("VA - Steering assistance verif. sheet is missing or empty.")
        return pd.DataFrame(), steering_assistance_score_dict

    # NaN means "not yet assessed by the OEM" -- distinct from an explicit
    # FAIL (0.0). Only overwritten below once a row is actually filled in.
    steering_assistance_df["Score"] = np.nan
    steering_assistance_df["Score"] = steering_assistance_df["Score"].astype("float64")

    rows_list = list(steering_assistance_df.iterrows())
    for pos, (index, row) in enumerate(rows_list):
        scenario = row["Scenario"]
        value = row["Value"]

        if common.is_empty_cell(value):
            score = np.nan
        elif "First turn" in str(scenario):
            next_row = rows_list[pos + 1][1] if pos + 1 < len(rows_list) else None
            second_turn_not_redirect = (
                next_row is not None
                and "Second turn" in str(next_row["Scenario"])
                and next_row["Value"] in ("Stays in lane", "Redirects")
            )
            score = (
                0.5 if (value == "Stays in lane" and second_turn_not_redirect) else 0.0
            )
        elif "Second turn" in str(scenario):
            prev_row = rows_list[pos - 1][1] if pos > 0 else None
            score = (
                0.5
                if (
                    value == "Stays in lane"
                    and prev_row is not None
                    and "First turn" in str(prev_row["Scenario"])
                    and prev_row["Value"] == "Stays in lane"
                )
                else 0.0
            )
        else:
            score = 0.0

        steering_assistance_df.at[index, "Score"] = score
        steering_assistance_score_dict[scenario] = score

    param_dict = common.create_param_dict_from_input_parameters(
        dfs.get("Input parameters", pd.DataFrame()), col="Category"
    )
    fitment_param = param_dict.get("Lane change assist", {}).get("Fitment")

    # "Fitment" is a vehicle-spec input (like other OEM-declared parameters
    # elsewhere), not a test result -- blank means "not yet specified"
    # (NaN), not "confirmed not fitted" (0.0).
    if common.is_empty_cell(fitment_param):
        steering_assistance_score_dict["Lane change assist"] = np.nan
    else:
        fitment_param = str(fitment_param).strip().lower()
        steering_assistance_score_dict["Lane change assist"] = (
            1.0 if fitment_param == "standard" else 0.0
        )

    logger.info(
        f"Computed Steering assistance scores: {steering_assistance_score_dict}"
    )
    return steering_assistance_df, steering_assistance_score_dict
