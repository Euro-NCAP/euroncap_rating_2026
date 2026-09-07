# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import logging
from euroncap_rating_2026.post_crash import data_model
from euroncap_rating_2026.post_crash import energy_management
from euroncap_rating_2026.post_crash import occupant_extrication
from euroncap_rating_2026 import common
import pandas as pd
import io

logger = logging.getLogger(__name__)


def preprocess_stage_subelement(dfs, stage_element_name, subelement_name):

    logger.debug(
        f"Preprocessing stage element: {stage_element_name}, subelement: {subelement_name}"
    )

    input_parameters_df = dfs.get("Input parameters")

    param_dict = common.create_param_dict_from_input_parameters(
        input_parameters_df, col="Stage subelement"
    )
    logger.debug(f"Created parameter dictionary from input parameters:\n{param_dict}")

    if subelement_name == "Rescue sheets":
        rescue_sheet_df = pd.read_csv(
            io.StringIO(data_model.RESCUE_SHEET_CSV_STRING), sep=";"
        )
        rescue_sheet_df = common.opposite_ffill(
            rescue_sheet_df, exclude_columns=["Value"]
        )
        dfs["RI - RS Verification"] = rescue_sheet_df
        return
    elif subelement_name == "Emergency response guide":
        emergency_response_guide_df = pd.read_csv(
            io.StringIO(data_model.EMERGENCY_RESPONSE_GUIDE_CSV_STRING), sep=";"
        )
        emergency_response_guide_df = common.opposite_ffill(
            emergency_response_guide_df, exclude_columns=["Value"]
        )
        dfs["RI - ERG Verification"] = emergency_response_guide_df
        return
    elif subelement_name == "Advanced eCall":
        advanced_ecall_df = pd.read_csv(
            io.StringIO(data_model.ADVANCED_ECALL_CSV_STRING), sep=";"
        )
        advanced_ecall_df = common.opposite_ffill(
            advanced_ecall_df, exclude_columns=["Value"]
        )
        dfs["PCI - Advanced eCall Verif"] = advanced_ecall_df
        return
    elif subelement_name == "Multi-collision brake & hazard lights":
        mcb_hazard_lights_df = pd.read_csv(
            io.StringIO(data_model.MCB_HAZARD_LIGHTS_CSV_STRING), sep=";"
        )
        mcb_hazard_lights_df = common.opposite_ffill(
            mcb_hazard_lights_df, exclude_columns=["Value"]
        )
        dfs["PCI - MCB & Hazard lights Verif"] = mcb_hazard_lights_df
        return
    elif subelement_name == "Energy management":
        energy_management_df = energy_management.preprocess(dfs)
        energy_management_df = common.opposite_ffill(
            energy_management_df, exclude_columns=["Value"]
        )
        dfs["VE - Energy Management Verif"] = energy_management_df
        return
    elif subelement_name == "Occupant extrication":
        occupant_extrication_df = occupant_extrication.preprocess(dfs)
        occupant_extrication_df = common.opposite_ffill(
            occupant_extrication_df, exclude_columns=["Value"]
        )
        dfs["VE - Occupant Extrication Verif"] = occupant_extrication_df
        return
