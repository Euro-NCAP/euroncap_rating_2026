# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import logging
from euroncap_rating_2026.safe_driving import data_model
from euroncap_rating_2026.safe_driving import general_vehicle_controls
from euroncap_rating_2026.safe_driving import steering_assistance
from euroncap_rating_2026.safe_driving import speed_assistance
from euroncap_rating_2026.safe_driving import driver_monitoring
from euroncap_rating_2026.safe_driving import acc_performance
from euroncap_rating_2026.safe_driving import seatbelt_usage
from euroncap_rating_2026.safe_driving import occupant_classification
from euroncap_rating_2026.safe_driving import occupant_presence


from euroncap_rating_2026 import common
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import numpy as np


@dataclass
class StageSubelement:
    name: str
    test_points_df: None


class GVCImplementation(str, Enum):
    PASS = "Pass"
    FAIL = "Fail"


logger = logging.getLogger(__name__)


def get_stage_subelement_key(stage_element_name):
    element_initials = "".join([word[0].upper() for word in stage_element_name.split()])
    if element_initials not in data_model.StageSubelementKey.__members__:
        raise ValueError(f"Invalid stage element name: {stage_element_name}")
    return data_model.StageSubelementKey[element_initials]


def preprocess_stage_subelement(
    dfs, stage_element_name, subelement_name, input_selected_points=None
):
    # Key is needed to access STAGE_SUBELEMENT_TO_LOADCASES with the right prefix (LDC, FC, ..)
    stage_subelement_key = get_stage_subelement_key(stage_element_name)

    logger.debug(
        f"Preprocessing stage element: {stage_element_name}, subelement: {subelement_name}"
    )
    logger.debug(f"Using stage subelement key: {stage_subelement_key}")

    input_parameters_df = dfs.get("Input parameters")

    param_dict = common.create_param_dict_from_input_parameters(
        input_parameters_df, col="Category"
    )
    logger.debug(f"Created parameter dictionary from input parameters:\n{param_dict}")

    if subelement_name == "Seatbelt usage":

        rear_seats_info = param_dict.get("Rear seat occupancy", {})
        num_rear_seats = rear_seats_info.get("Number of rear seats")
        rear_seat_df = seatbelt_usage.generate_seatbelt_usage_rear_seat_df(
            num_rear_seats
        )
        logger.debug(f"Generated rear seat seatbelt usage dataframe:\n{rear_seat_df}")
        seatbelt_usage_df = seatbelt_usage.generate_seatbelt_usage_df(param_dict)
        logger.debug(f"Generated seatbelt usage dataframe:\n{seatbelt_usage_df}")

        if rear_seat_df is not None:
            seatbelt_usage_df = pd.concat(
                [seatbelt_usage_df, rear_seat_df], ignore_index=True
            )

        logger.debug(f"Final seatbelt usage dataframe:\n{seatbelt_usage_df}")

        scenario_scores_df = dfs.get("Scenario Scores", pd.DataFrame())
        scenario_scores_df = seatbelt_usage.update_scenario_scores_with_rear_seats(
            scenario_scores_df, num_rear_seats
        )

        dfs["OM - Seatbelt usage verif."] = seatbelt_usage_df
        dfs["Scenario Scores"] = scenario_scores_df

    elif subelement_name == "Occupant classification":
        occupant_classification_df, scenario_scores_df = (
            occupant_classification.preprocess_occupant_classification(dfs)
        )
        logger.debug(
            f"Generated occupant classification dataframe:\n{occupant_classification_df}"
        )
        dfs["OM - Occ. classification verif."] = occupant_classification_df
        dfs["Scenario Scores"] = scenario_scores_df

    elif subelement_name == "Occupant presence":
        om_occupant_presence_df = occupant_presence.preprocess(dfs)
        logger.debug(
            f"Generated occupant presence dataframe:\n{om_occupant_presence_df}"
        )

        dfs["OM - Occ. presence verif."] = om_occupant_presence_df

    elif subelement_name == "Driver monitoring":
        dm_prediction_df = dfs.get("DE - DM pred.", pd.DataFrame())
        if dm_prediction_df.empty:
            logger.warning(
                "'DE - DM pred.' sheet is missing or empty; skipping Driver "
                "monitoring preprocessing."
            )
            dfs["DE - DM verif."] = pd.DataFrame()
            return []
        param_df = common.get_param_df(dfs)
        dm_required_mask_df = dfs.get("DE - DM pred. (required)")
        logger.debug(f"Loaded DM Prediction dataframe:\n{dm_prediction_df}")
        dm_verification_df, selected_test_points, forced_red_coordinates = (
            driver_monitoring.preprocess(
                dm_prediction_df,
                param_df,
                input_selected_points=input_selected_points,
                required_mask_df=dm_required_mask_df,
            )
        )
        dfs["DE - DM verif."] = dm_verification_df
        dfs["DE - DM pred. (forced_red)"] = pd.DataFrame(
            forced_red_coordinates, columns=["row", "col"]
        )
        return selected_test_points

    elif subelement_name == "General vehicle controls":
        gvc_verification_df = general_vehicle_controls.preprocess(dfs)
        logger.debug(
            f"Generated General Vehicle Controls verification dataframe:\n{gvc_verification_df}"
        )
        dfs["DE - GVC verif."] = gvc_verification_df
        # Since no test points are selected for General Vehicle Controls, we return an empty list
        return []

    elif subelement_name == "Speed Assistance":

        sas_verification_df, scenario_scores_df = speed_assistance.preprocess(dfs)
        logger.debug(
            f"Generated Speed Assistance verification dataframe:\n{sas_verification_df}"
        )
        dfs["VA - Speed assist. verif."] = sas_verification_df
        dfs["Scenario Scores"] = scenario_scores_df

        # Since no test points are selected for Speed Assistance, we return an empty list
        return []

    elif subelement_name == "ACC Performance":
        test_points, verification_df, scenario_scores_df = acc_performance.preprocess(
            dfs,
            input_selected_points=input_selected_points,
        )
        dfs["VA - ACC verif."] = verification_df
        dfs["Scenario Scores"] = scenario_scores_df
        return test_points

    elif subelement_name == "Steering Assistance":
        steering_assistance_df = steering_assistance.preprocess(dfs)
        dfs["VA - Steering assistance verif."] = steering_assistance_df
        # Since no test points are selected for Steering Assistance, we return an empty list
        return []
