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
    # NaN means "not yet assessed by the OEM" -- distinct from an explicit
    # FAIL (0.0). Only overwritten below once a sheet is actually assessed.
    rescue_score_dict = {
        "Rescue sheets": np.nan,
        "Emergency response guide": np.nan,
    }

    rs_verification_df = dfs.get("RI - RS Verification", pd.DataFrame())
    erg_verification_df = dfs.get("RI - ERG Verification", pd.DataFrame())

    rs_state = common.classify_pass_fail_section(rs_verification_df)
    if rs_state == "fail":
        logger.info(
            "One or more Rescue sheets requirements failed. Keeping score to 0."
        )
        rescue_score_dict["Rescue sheets"] = 0.0
    elif rs_state == "pass":
        rescue_score_dict["Rescue sheets"] = 35.0
    # "blank": leave unassessed (NaN)

    erg_state = common.classify_pass_fail_section(erg_verification_df)
    if erg_state == "fail":
        logger.info(
            "One or more Emergency response guide requirements failed. Keeping score to 0."
        )
        rescue_score_dict["Emergency response guide"] = 0.0
    elif erg_state == "pass":
        rescue_score_dict["Emergency response guide"] = 5.0
    # "blank": leave unassessed (NaN)

    return rescue_score_dict
