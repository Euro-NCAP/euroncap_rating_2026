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

ENERGY_MANAGEMENT_CSV_STRING = """
Category;Scenario;Value
Energy isolation;Compliance with UN Regulation requirements;
;Deactivation of high voltage energy - Automatic;
;Deactivation of high voltage energy - First manual;
;Deactivation of high voltage energy - Second manual;
Thermal propagation;Fulfilment of UN-R100.03 with predefined leadtime;
;Thermal propagation detection communication inside the vehicle;
;Thermal propagation detection communication to the car owner during charging;
;Thermal propagation detection communication for the people around the car during charging;
"""


def preprocess(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    logger.info("Preprocess Energy Management ...")
    # Add ENERGY_MANAGEMENT_CSV_STRING
    energy_management_df = pd.read_csv(
        io.StringIO(ENERGY_MANAGEMENT_CSV_STRING.strip()), sep=";"
    )

    logger.info(
        f"Energy Management DataFrame created with {len(energy_management_df)} rows"
    )

    return energy_management_df


def compute_score(dfs: dict[str, pd.DataFrame]) -> Dict:

    logger.info("Computing Energy Management score...")

    # NaN means "not yet assessed by the OEM" -- distinct from an explicit
    # FAIL (0.0). Only overwritten below once a scenario is actually assessed.
    energy_management_dict = {}
    energy_management_dict["Compliance with UN Regulation requirements"] = np.nan
    energy_management_dict["Deactivation of high voltage energy - Automatic"] = np.nan
    energy_management_dict["Deactivation of high voltage energy - First manual"] = (
        np.nan
    )
    energy_management_dict["Deactivation of high voltage energy - Second manual"] = (
        np.nan
    )
    energy_management_dict["Fulfilment of UN-R100.03 with predefined leadtime"] = np.nan
    energy_management_dict[
        "Thermal propagation detection communication inside the vehicle"
    ] = np.nan
    energy_management_dict[
        "Thermal propagation detection communication to the car owner during charging"
    ] = np.nan
    energy_management_dict[
        "Thermal propagation detection communication for the people around the car during charging"
    ] = np.nan

    energy_management_max_scores = {
        "Compliance with UN Regulation requirements": 3.0,
        "Deactivation of high voltage energy - Automatic": 5.0,
        "Deactivation of high voltage energy - First manual": 2.0,
        "Deactivation of high voltage energy - Second manual": 1.0,
        "Fulfilment of UN-R100.03 with predefined leadtime": 9.0,
        "Thermal propagation detection communication inside the vehicle": 2.0,
        "Thermal propagation detection communication to the car owner during charging": 1.0,
        "Thermal propagation detection communication for the people around the car during charging": 1.0,
    }
    energy_management_verification_df = dfs.get(
        "VE - Energy Management Verif", pd.DataFrame()
    )
    if energy_management_verification_df.empty:
        return energy_management_dict

    for scenario in energy_management_dict.keys():
        scenario_df = common.extract_section(
            energy_management_verification_df, scenario
        )
        if scenario_df.empty:
            logger.warning(
                f"No data found for scenario '{scenario}' in Energy Management verification DataFrame."
            )
            continue
        if len(scenario_df) > 1:
            logger.warning(
                f"Multiple rows found for scenario '{scenario}' in Energy Management verification DataFrame. Using the first row."
            )

        scenario_value = scenario_df["Value"].iloc[0]
        if common.is_empty_cell(scenario_value):
            logger.info(
                f"Value for scenario '{scenario}' is not yet assessed. Leaving unassessed."
            )
            continue

        logger.debug(
            f"Processing scenario '{scenario}' with value '{scenario_value}' for Energy Management scoring."
        )

        if scenario == "Fulfilment of UN-R100.03 with predefined leadtime":
            if scenario_value == "≥90 min":
                energy_management_dict[scenario] = 9.0
            elif scenario_value == ">40 min":
                energy_management_dict[scenario] = 6.0
            elif scenario_value == ">20 min":
                energy_management_dict[scenario] = 3.0
            else:
                # Explicitly assessed but not one of the scoring bands
                # (e.g. "≤20 min") -- a real 0, not unassessed.
                energy_management_dict[scenario] = 0.0
        elif scenario_value.lower() == "pass":
            energy_management_dict[scenario] = energy_management_max_scores[scenario]
        else:
            energy_management_dict[scenario] = 0.0

    return energy_management_dict
