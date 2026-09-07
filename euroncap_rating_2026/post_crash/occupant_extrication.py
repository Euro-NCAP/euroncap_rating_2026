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
import logging

logger = logging.getLogger(__name__)
OCCUPANT_EXTRICATION_CSV_STRING = """
Category;Scenario;Value
Doors and belts;Seat belt buckle unlatching;
Doors and belts;Door opening - interior, post low voltage drop;
Doors and belts;Door opening - exterior, post crash;
Doors and belts;Door opening - exterior, post crash, post low voltage drop;
Doors and belts;Tailgate opening;
Submergence;Window opening;
Submergence;Rescue tool / Emergency device;
"""


def preprocess(dfs: dict[str, pd.DataFrame]) -> Dict:
    logger.info("Preprocess Occupant Extrication ...")

    # Start with seatbelt unlatching
    occupant_extrication_df = pd.read_csv(
        io.StringIO(OCCUPANT_EXTRICATION_CSV_STRING.strip()), sep=";"
    )

    logger.info(
        f"Occupant Extrication DataFrame created with {len(occupant_extrication_df)} rows"
    )

    return occupant_extrication_df


SCENARIO_MAX_SCORES = {
    "Seat belt buckle unlatching": 1.0,
    "Door opening - interior, post low voltage drop": 3.0,
    "Door opening - exterior, post crash": 4.0,
    "Door opening - exterior, post crash, post low voltage drop": 2.0,
    "Tailgate opening": 2.0,
    "Window opening": 3.0,
    "Rescue tool / Emergency device": 1.0,
}


def compute_score(dfs: dict[str, pd.DataFrame]) -> Dict:
    logger.info("Computing Occupant Extrication score...")

    # NaN means "not yet assessed by the OEM" -- distinct from an explicit
    # FAIL (0.0). Only overwritten below once a scenario is actually assessed.
    occupant_extrication_score_dict = {k: np.nan for k in SCENARIO_MAX_SCORES}

    occupant_extrication_df = dfs.get("VE - Occupant Extrication Verif", pd.DataFrame())
    if occupant_extrication_df.empty:
        return occupant_extrication_score_dict

    for scenario, max_score in SCENARIO_MAX_SCORES.items():
        scenario_df = common.extract_section(
            occupant_extrication_df, scenario, col_name="Scenario"
        )
        if scenario_df.empty:
            continue
        state = common.classify_pass_fail_value(scenario_df["Value"].iloc[0])
        if state == "pass":
            occupant_extrication_score_dict[scenario] = max_score
        elif state == "fail":
            occupant_extrication_score_dict[scenario] = 0.0
        # "blank": leave unassessed (NaN)

    return occupant_extrication_score_dict
