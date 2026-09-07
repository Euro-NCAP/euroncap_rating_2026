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


def compute_112_score(ecall_112_df: pd.DataFrame) -> Dict:
    """
    Compute the score for the Advanced eCall - 112 section.

    Assigns scores to:
    - Potential number of occupants
    - Direction of impacts - Front impact
    - Direction of impacts - Side impact
    - Direction of impacts - Rear impact
    - Direction of impacts - Rollovers as 1st impact
    - Delta V - Front impact
    - Delta V - Side impact
    - Delta V - Rear impact

    :param ecall_112_df: DataFrame containing the Advanced eCall - 112 data
    :type ecall_112_df: pd.DataFrame
    :return: Dictionary with scores for each category
    :rtype: Dict
    """

    # NaN means "not yet assessed by the OEM" -- distinct from an explicit
    # FAIL (0.0). Only overwritten below once a row is actually assessed.
    ecall_112_score_dict = {
        "Potential number of occupants": np.nan,
        "Direction of impacts - Front impact": np.nan,
        "Direction of impacts - Side impact": np.nan,
        "Direction of impacts - Rear impact": np.nan,
        "Direction of impacts - Rollovers as 1st impact": np.nan,
        "Delta V - Front impact": np.nan,
        "Delta V - Side impact": np.nan,
        "Delta V - Rear impact": np.nan,
    }

    ecall_112_max_scores = {
        "Potential number of occupants": 3.0,
        "Direction of impacts - Front impact": 2.0,
        "Direction of impacts - Side impact": 2.0,
        "Direction of impacts - Rear impact": 2.0,
        "Direction of impacts - Rollovers as 1st impact": 1.5,
        "Delta V - Front impact": 1.5,
        "Delta V - Side impact": 1.5,
        "Delta V - Rear impact": 1.5,
    }

    general_req_rows_112 = common.extract_section(ecall_112_df, "General requirements")
    # Remove all rows where every value is NaN
    general_req_rows_112 = general_req_rows_112.dropna(how="all")

    general_req_112_state = common.classify_pass_fail_section(general_req_rows_112)

    if general_req_112_state == "blank":
        return ecall_112_score_dict

    if general_req_112_state == "fail":
        for category in ecall_112_score_dict:
            ecall_112_score_dict[category] = 0.0
        return ecall_112_score_dict

    for category in ecall_112_score_dict.keys():
        category_df = common.extract_section(ecall_112_df, category)
        state = common.classify_pass_fail_section(category_df)
        if state == "pass":
            ecall_112_score_dict[category] = ecall_112_max_scores[category]
        elif state == "fail":
            ecall_112_score_dict[category] = 0.0
        # "blank": leave unassessed (NaN)

    return ecall_112_score_dict


def compute_tps_score(ecall_tps_df: pd.DataFrame) -> Dict:
    """
    Compute the score for the Advanced eCall - TPS section.

    Assigns scores to:
    - "Country coverage"
    - "Multiple languages - EN, DE, FR, ES"
    - "Multiple languages - 4 additional languages"
    - "Hazard detection after crash"
    - "Telephone pairing"
    - "Vehicle information"
    - "Vehicle attitude"
    - "Any additional information"
    - "AACN and OEM severity index"

    :param ecall_tps_df: DataFrame containing the Advanced eCall - TPS data
    :type ecall_tps_df: pd.DataFrame
    :return: Dictionary with scores for each category
    :rtype: Dict
    """
    # NaN means "not yet assessed by the OEM" -- distinct from an explicit
    # FAIL (0.0). Only overwritten below once a row is actually assessed.
    ecall_tps_score_dict = {
        "Country coverage": np.nan,
        "Multiple languages - EN, DE, FR, ES": np.nan,
        "Multiple languages - 4 additional languages": np.nan,
        "Hazard detection after crash": np.nan,
        "Telephone pairing": np.nan,
        "Vehicle information": np.nan,
        "Vehicle attitude": np.nan,
        "Any additional information": np.nan,
        "AACN and OEM severity index": np.nan,
    }
    ecall_tps_max_scores = {
        "Country coverage": 3.0,
        "Multiple languages - EN, DE, FR, ES": 3.0,
        "Multiple languages - 4 additional languages": 3.0,
        "Hazard detection after crash": 3.0,
        "Telephone pairing": 3.0,
        "Vehicle information": 3.0,
        "Vehicle attitude": 3.0,
        "Any additional information": 3.0,
        "AACN and OEM severity index": 3.0,
    }
    general_req_rows_tps = common.extract_section(ecall_tps_df, "General requirements")
    # Remove all rows where every value is NaN
    general_req_rows_tps = general_req_rows_tps.dropna(how="all")

    general_req_tps_state = common.classify_pass_fail_section(general_req_rows_tps)

    if general_req_tps_state == "blank":
        return ecall_tps_score_dict

    if general_req_tps_state == "fail":
        for category in ecall_tps_score_dict:
            ecall_tps_score_dict[category] = 0.0
        return ecall_tps_score_dict

    # Compute country coverage score
    country_coverage_df = common.extract_section(ecall_tps_df, "Country coverage")
    country_coverage_state = common.classify_pass_fail_section(country_coverage_df)
    if country_coverage_state == "pass":
        ecall_tps_score_dict["Country coverage"] = ecall_tps_max_scores[
            "Country coverage"
        ]
    elif country_coverage_state == "fail":
        ecall_tps_score_dict["Country coverage"] = 0.0
    # "blank": leave unassessed (NaN)

    # Compute multiple language score
    language_df = common.extract_section(
        ecall_tps_df, "Multiple languages - EN, DE, FR, ES"
    )
    language_state = common.classify_pass_fail_section(language_df)

    # The remaining categories can only earn points if country coverage AND
    # language support are both explicitly PASS. If either is still blank,
    # the remaining categories stay unassessed too (NaN, the dict default) --
    # we can't yet know whether they'll be gated to 0 or allowed to score. If
    # either is an explicit FAIL, the remaining categories are deterministic
    # zeros, matching the pre-existing "gated to 0" rule.
    gate_states = {country_coverage_state, language_state}
    if "blank" in gate_states:
        gate_state = "blank"
    elif "fail" in gate_states:
        gate_state = "fail"
    else:
        gate_state = "pass"

    if gate_state == "pass":
        ecall_tps_score_dict["Multiple languages - EN, DE, FR, ES"] = (
            ecall_tps_max_scores["Multiple languages - EN, DE, FR, ES"]
        )
    elif gate_state == "fail":
        ecall_tps_score_dict["Multiple languages - EN, DE, FR, ES"] = 0.0
    # "blank": leave unassessed (NaN)

    # Compute remaining TPS scores
    # If country coverage and language support requirements are met, then check the remaining categories
    # If either country coverage or language support fails, the remaining categories cannot earn points, even if they are marked as "PASS"
    remaining_categories = [
        "Multiple languages - 4 additional languages",
        "Hazard detection after crash",
        "Telephone pairing",
        "Vehicle information",
        "Vehicle attitude",
        "Any additional information",
        "AACN and OEM severity index",
    ]
    if gate_state == "pass":
        for category in remaining_categories:
            category_df = common.extract_section(ecall_tps_df, category)
            state = common.classify_pass_fail_section(category_df)
            if state == "pass":
                ecall_tps_score_dict[category] = ecall_tps_max_scores[category]
            elif state == "fail":
                ecall_tps_score_dict[category] = 0.0
            # "blank": leave unassessed (NaN)
    elif gate_state == "fail":
        for category in remaining_categories:
            ecall_tps_score_dict[category] = 0.0
    # gate_state == "blank": leave all remaining categories unassessed (NaN)

    return ecall_tps_score_dict


def compute_score(dfs: dict[str, pd.DataFrame]) -> Dict:
    ecall_score_dict = {}

    ecall_verification_df = dfs.get("PCI - Advanced eCall Verif", pd.DataFrame())
    if ecall_verification_df.empty:
        return ecall_score_dict

    ## 112 scoring

    ecall_112_df = common.extract_section(
        ecall_verification_df, "Advanced eCall - 112", "Category"
    )

    ecall_112_score_dict = compute_112_score(ecall_112_df)
    ecall_score_dict.update(ecall_112_score_dict)

    ## TPS scoring
    ecall_tps_df = common.extract_section(
        ecall_verification_df, "Advanced eCall - TPS", "Category"
    )

    ecall_tps_score_dict = compute_tps_score(ecall_tps_df)
    ecall_score_dict.update(ecall_tps_score_dict)

    return ecall_score_dict
