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

GVC_CSV_STRING = """
Category;Scenario;Value
Driving;Driving controls;
;Vision;
;Lights;
;ADAS;
Comfort & infotainment;Audio entertainment;
;Calling & dialling;
;Navigation system;
;Climate controls;
;Windows;
;Other;
"""


def preprocess(dfs: dict[str, pd.DataFrame]) -> Dict:
    logger.info("Preprocess General Vehicle Controls usage ...")
    gvc_df = pd.read_csv(io.StringIO(GVC_CSV_STRING), sep=";")
    gvc_df = common.opposite_ffill(gvc_df, exclude_columns=["Value", "Score"])
    return gvc_df


def compute_score(dfs: dict[str, pd.DataFrame]) -> Dict:
    scenario_list = [
        "Driving controls",
        "Vision",
        "Lights",
        "ADAS",
        "Audio entertainment",
        "Calling & dialling",
        "Navigation system",
        "Climate controls",
        "Windows",
        "Other",
    ]
    # NaN means "not yet assessed by the OEM" -- distinct from an explicit
    # FAIL (0). Only overwritten below once a scenario is actually PASS/FAIL.
    gvc_score_dict = {scenario: np.nan for scenario in scenario_list}
    scenario_scores = [1.00, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.25, 0.25]
    gvc_max_score_dict = {
        scenario: score for scenario, score in zip(scenario_list, scenario_scores)
    }
    gvc_verification_df = dfs.get("DE - GVC verif.", pd.DataFrame())
    if gvc_verification_df.empty:
        return gvc_score_dict

    for scenario in scenario_list:
        scenario_rows = common.extract_section(
            gvc_verification_df, scenario, col_name="Scenario"
        )
        # Remove all rows where every value is NaN
        scenario_rows = scenario_rows.dropna(how="all")
        if scenario_rows.empty:
            continue

        states = scenario_rows["Value"].apply(common.classify_pass_fail_value)
        if (states == "blank").any():
            continue  # leave unassessed (NaN)
        any_failed = (states == "fail").any()
        gvc_score_dict[scenario] = 0 if any_failed else gvc_max_score_dict[scenario]

    return gvc_score_dict
