# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Dict
import numpy as np
import pandas as pd
from euroncap_rating_2026 import common
import logging

logger = logging.getLogger(__name__)


def compute_score(dfs: dict[str, pd.DataFrame]) -> Dict:
    """
    Compute the score for the MCB & Hazard lights section.

    Assigns scores to:
    - "Advanced multi-collision brake"
    - "Automatic activation of hazard warning lights"

    :param dfs: Dictionary containing DataFrames for each section
    :type dfs: dict[str, pd.DataFrame]
    :return: Dictionary with scores for each category
    :rtype: Dict
    """
    # NaN means "not yet assessed by the OEM" -- distinct from an explicit
    # FAIL (0.0). Only overwritten below once a section is actually assessed.
    mcb_score_dict = {
        "Advanced multi-collision brake": np.nan,
        "Automatic activation of hazard warning lights": np.nan,
    }

    mcb_verification_df = dfs.get("PCI - MCB & Hazard lights Verif", pd.DataFrame())
    if mcb_verification_df.empty:
        return mcb_score_dict

    # MCB sheets scoring
    advanced_mcb_df = common.extract_section(
        mcb_verification_df, "Advanced multi-collision brake", "Category"
    )
    mcb_state = common.classify_pass_fail_section(advanced_mcb_df)
    if mcb_state == "fail":
        logger.info(
            "One or more Advanced multi-collision brake requirements failed. Keeping score to 0."
        )
        mcb_score_dict["Advanced multi-collision brake"] = 0.0
    elif mcb_state == "pass":
        mcb_score_dict["Advanced multi-collision brake"] = 4.0
    # "blank": leave unassessed (NaN)

    # Automatic activation of hazard warning lights scoring
    hazard_lights_df = common.extract_section(
        mcb_verification_df, "Automatic activation of hazard warning lights", "Category"
    )
    hazard_lights_state = common.classify_pass_fail_section(hazard_lights_df)
    if hazard_lights_state == "fail":
        logger.info(
            "One or more Automatic activation of hazard warning lights requirements failed. Keeping score to 0."
        )
        mcb_score_dict["Automatic activation of hazard warning lights"] = 0.0
    elif hazard_lights_state == "pass":
        mcb_score_dict["Automatic activation of hazard warning lights"] = 1.0
    # "blank": leave unassessed (NaN)

    return mcb_score_dict
