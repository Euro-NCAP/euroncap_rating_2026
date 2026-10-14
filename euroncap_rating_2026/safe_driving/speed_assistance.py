# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import io
from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, Mapping, Optional, Tuple
import numpy as np
import pandas as pd
from euroncap_rating_2026 import common
from euroncap_rating_2026.numeric import round_half_up
from euroncap_rating_2026.safe_driving import data_model
import logging

logger = logging.getLogger(__name__)

CONDITIONAL_SPEED_LIMIT_STR = "Conditional speed limits"
IMPLICIT_SPEED_LIMIT_STR = "Implicit speed limits"
DYNAMIC_SPEED_LIMIT_STR = "Dynamic speed limits"

SLIF_ADVANCED_SPEED_LIMIT_CATEGORY = "SLIF - Advanced speed limit"


class SlifAdvancedPassRule(str, Enum):
    """How the per-market Green/Red predictions of one Advanced Speed Limits
    column roll up into the scenario's PASS/FAIL."""

    # Euro NCAP: no member-state Red / grey-N/A AND at least half of all
    # application-area rows Green or blank.
    TWO_TIER = "two_tier"
    # Every market row must be Green (or blank / N/A): a single Red or
    # grey-N/A row fails the scenario.
    ALL_ROWS = "all_rows"


@dataclass(frozen=True)
class SpeedAssistanceProfile:
    """Everything in Speed Assistance that differs between rating protocols.

    Only SLIF -> Advanced Speed Limits (taxonomy, points, coupling, caps) and
    the geometry of the 'VA - Speed assist. pred.' sheet vary; Accuracy,
    Local Hazards, System updates and the Speed Control Function are shared
    and stay in module-level code. The Euro NCAP profile is built in; another
    programme's package registers its own with register_profile (see
    common.register_protocol).
    """

    protocol: str
    # Prediction-sheet column header -> category string, in sheet order.
    slif_adv_mapping: Mapping[str, str]
    # Scored scenario (verification-sheet row name) -> points, in verification
    # row order. These are the canonical scenario strings consuming
    # applications match by exact equality.
    advanced_points: Mapping[str, float]
    # Euro NCAP scores the three implicit columns as one all-or-nothing row.
    collapse_implicit: bool
    implicit_collapsed_scenario: Optional[str]
    # (dependent scenario, prerequisite scenario): the dependent scores 0
    # whenever the prerequisite scored 0.
    coupling_rules: Tuple[Tuple[str, str], ...]
    # (scenarios, cap): the summed points of *scenarios* count at most *cap*
    # towards the category total (applied at Category Scores level, the raw
    # per-scenario points stay -- same convention as Local Hazards).
    sub_caps: Tuple[Tuple[FrozenSet[str], float], ...]
    # Geometry of the prediction sheet as read by read_excel_file_to_dfs
    # (0-based DataFrame rows; the sheet's first row is the header).
    adv_table_n_rows: int
    adv_table_n_cols: int
    hazard_table_start_row: int
    # Advanced Speed Limits grid rows and their first-column group label.
    expected_row_count: int
    group_label: str
    expected_group_row_count: int
    pass_rule: SlifAdvancedPassRule

    @property
    def advanced_scenarios(self) -> Tuple[str, ...]:
        return tuple(self.advanced_points.keys())

    @property
    def verification_csv(self) -> str:
        return build_slif_scoring_csv(self.advanced_scenarios)


ENCAP_SLIF_ADV_MAPPING = {
    "Rain / Wetness": CONDITIONAL_SPEED_LIMIT_STR,
    "Snow / Icy": CONDITIONAL_SPEED_LIMIT_STR,
    "Time / Season": CONDITIONAL_SPEED_LIMIT_STR,
    "Distance for / Distance in": CONDITIONAL_SPEED_LIMIT_STR,
    "Non-lane relevant arrows": CONDITIONAL_SPEED_LIMIT_STR,
    "Lane relevant arrows": CONDITIONAL_SPEED_LIMIT_STR,
    "Vehicle categories": CONDITIONAL_SPEED_LIMIT_STR,
    "Highway / Motorway": IMPLICIT_SPEED_LIMIT_STR,
    "City entry / City exit": IMPLICIT_SPEED_LIMIT_STR,
    "Residential zones": IMPLICIT_SPEED_LIMIT_STR,
    "Non-lane relevant": DYNAMIC_SPEED_LIMIT_STR,
    "Lane relevant": DYNAMIC_SPEED_LIMIT_STR,
}

ENCAP_IMPLICIT_COLLAPSED_SCENARIO = (
    "Implicit speed limits - Highway, city and residential zones"
)

# Points are protocol constants (Euro NCAP Safe Driving - Vehicle Assistance
# v1.2, section 1.2.2), not read from the template's 'Max score' column.
ENCAP_ADVANCED_SPEED_LIMIT_POINTS = {
    "Conditional speed limits - Rain / Wetness": 0.4,
    "Conditional speed limits - Snow / Icy": 0.4,
    "Conditional speed limits - Time / Season": 0.4,
    "Conditional speed limits - Distance for / Distance in": 0.4,
    "Conditional speed limits - Non-lane relevant arrows": 0.1,
    "Conditional speed limits - Lane relevant arrows": 0.1,
    "Conditional speed limits - Vehicle categories": 0.2,
    ENCAP_IMPLICIT_COLLAPSED_SCENARIO: 0.5,
    "Dynamic speed limits - Non-lane relevant": 0.25,
    "Dynamic speed limits - Lane relevant": 0.25,
}
# The 10 canonical Euro NCAP scenario strings, matched by exact string
# equality by consuming applications.
ENCAP_ADVANCED_SPEED_LIMIT_SCENARIOS = tuple(ENCAP_ADVANCED_SPEED_LIMIT_POINTS)

ENCAP_26_28_PROFILE = SpeedAssistanceProfile(
    protocol=common.Protocol.ENCAP_26_28.value,
    slif_adv_mapping=ENCAP_SLIF_ADV_MAPPING,
    advanced_points=ENCAP_ADVANCED_SPEED_LIMIT_POINTS,
    collapse_implicit=True,
    implicit_collapsed_scenario=ENCAP_IMPLICIT_COLLAPSED_SCENARIO,
    coupling_rules=(
        (
            "Conditional speed limits - Lane relevant arrows",
            "Conditional speed limits - Non-lane relevant arrows",
        ),
        (
            "Dynamic speed limits - Lane relevant",
            "Dynamic speed limits - Non-lane relevant",
        ),
    ),
    sub_caps=(),
    adv_table_n_rows=30,
    adv_table_n_cols=14,
    hazard_table_start_row=31,
    expected_row_count=29,
    group_label="euro ncap member states",
    expected_group_row_count=10,
    pass_rule=SlifAdvancedPassRule.TWO_TIER,
)

# Registry: protocol id -> profile. Euro NCAP is built in; another
# programme's package adds its own with register_profile at import time.
PROFILES = {
    ENCAP_26_28_PROFILE.protocol: ENCAP_26_28_PROFILE,
}


def register_profile(profile: SpeedAssistanceProfile) -> SpeedAssistanceProfile:
    """Register *profile* for its protocol. The protocol must already be
    registered (common.register_protocol); every scenario named by the
    coupling rules and sub-caps must be one of the profile's own scenarios.
    Registering the identical profile twice is a no-op; a different profile
    under an existing protocol raises ValueError."""
    protocol = common.normalize_protocol(profile.protocol)
    scenarios = set(profile.advanced_scenarios)
    referenced = {s for pair in profile.coupling_rules for s in pair}
    referenced |= {s for group, _cap in profile.sub_caps for s in group}
    unknown = sorted(referenced - scenarios)
    if unknown:
        raise ValueError(
            f"Profile for '{protocol}' references scenarios it does not score: "
            f"{unknown}"
        )
    if profile.collapse_implicit and not profile.implicit_collapsed_scenario:
        raise ValueError(
            f"Profile for '{protocol}' collapses implicit scenarios but names no "
            "collapsed scenario."
        )
    # preprocess fills the verification row "<category> - <column header>"
    # for every mapped column (the collapsed implicit row instead, when
    # collapsing); compute_score awards advanced_points by those same row
    # names. The two must agree exactly, or a scenario is silently left at 0.
    emitted = set()
    implicit_columns = 0
    for header, category in profile.slif_adv_mapping.items():
        if profile.collapse_implicit and category == IMPLICIT_SPEED_LIMIT_STR:
            implicit_columns += 1
            continue
        emitted.add(f"{category} - {header}")
    if profile.collapse_implicit:
        if implicit_columns == 0:
            raise ValueError(
                f"Profile for '{protocol}' collapses implicit scenarios but maps "
                "no implicit column."
            )
        emitted.add(profile.implicit_collapsed_scenario)
    if emitted != scenarios:
        raise ValueError(
            f"Profile for '{protocol}': advanced_points and slif_adv_mapping "
            f"disagree. Scored but never filled by preprocess: "
            f"{sorted(scenarios - emitted)}; filled but never scored: "
            f"{sorted(emitted - scenarios)}."
        )
    existing = PROFILES.get(protocol)
    if existing is not None and existing != profile:
        raise ValueError(
            f"A Speed Assistance profile is already registered for '{protocol}'; "
            "refusing to replace it."
        )
    PROFILES[protocol] = profile
    return profile


def get_profile(protocol=None) -> SpeedAssistanceProfile:
    """The SpeedAssistanceProfile for *protocol* (see common.normalize_protocol;
    None means Euro NCAP). Raises ValueError for a registered protocol whose
    package registered no Speed Assistance profile."""
    protocol = common.normalize_protocol(protocol)
    try:
        return PROFILES[protocol]
    except KeyError:
        raise ValueError(
            f"No Speed Assistance profile is registered for protocol '{protocol}'."
        ) from None


def resolve_profile(protocol=None, dfs: dict = None) -> SpeedAssistanceProfile:
    """Profile for an explicit *protocol*, else for the protocol recorded in
    *dfs*' Version sheet, else Euro NCAP (common.resolve_protocol)."""
    return get_profile(common.resolve_protocol(protocol, dfs=dfs))


# Euro NCAP values kept under their historical names for existing importers.
SLIF_ADV_MAPPING = ENCAP_SLIF_ADV_MAPPING
EURO_NCAP_MEMBER_STATES_LABEL = ENCAP_26_28_PROFILE.group_label
EXPECTED_MEMBER_STATE_ROW_COUNT = ENCAP_26_28_PROFILE.expected_group_row_count
EXPECTED_APPLICATION_AREA_ROW_COUNT = ENCAP_26_28_PROFILE.expected_row_count

LOCAL_HAZARD_COLUMNS = [
    "Construction zones",
    "Items on road",
    "Stopped vehicle",
    "Broken down vehicle",
    "Post crash",
    "Poor weather",
    "Poor road",
    "Wrong way driver",
    "Amber & blue lights",
    "Traffic jam",
]

LOCAL_HAZARD_SENDING_STR = "Sending"
LOCAL_HAZARD_RECEIVING_STR = "Receiving & informing"
LOCAL_HAZARD_DIRECTIONS = [LOCAL_HAZARD_SENDING_STR, LOCAL_HAZARD_RECEIVING_STR]

# Hazards for which "Sending" is N/A in the protocol scoring table
LOCAL_HAZARD_NO_SENDING_COLUMNS = ["Amber & blue lights", "Traffic jam"]

# Positional row index of each (communication channel, direction) pair inside the
# SLIF Local Hazards prediction table. The table is 4 data rows tall:
# Cloud/Sending, Cloud/Receiving & informing, Direct/Sending, Direct/Receiving &
# informing (see sheet 'VA - Speed assist. pred.').
LOCAL_HAZARD_CHANNEL_ROW_IDX = {
    ("cloud", LOCAL_HAZARD_SENDING_STR): 0,
    ("cloud", LOCAL_HAZARD_RECEIVING_STR): 1,
    ("direct", LOCAL_HAZARD_SENDING_STR): 2,
    ("direct", LOCAL_HAZARD_RECEIVING_STR): 3,
}

# Points per direction as (Direct AND Cloud communication, Direct OR Cloud
# communication), per protocol section 1.2.3 Local Hazards.
LOCAL_HAZARD_POINTS = {
    LOCAL_HAZARD_SENDING_STR: (0.2, 0.15),
    LOCAL_HAZARD_RECEIVING_STR: (0.15, 0.15),
}

LOCAL_HAZARD_CAP_BOTH_CHANNELS = 3.0
LOCAL_HAZARD_CAP_SINGLE_CHANNEL = 2.5

SLIF_LOCAL_HAZARD_VERIFICATION_ROWS = (
    "Construction zones - Sending",
    "Construction zones - Receiving & informing",
    "Items on road - Sending",
    "Items on road - Receiving & informing",
    "Stopped vehicle - Sending",
    "Stopped vehicle - Receiving & informing",
    "Broken down vehicle - Sending",
    "Broken down vehicle - Receiving & informing",
    "Post crash - Sending",
    "Post crash - Receiving & informing",
    "Poor weather - Sending",
    "Poor weather - Receiving & informing",
    "Poor road - Sending",
    "Poor road - Receiving & informing",
    "Wrong way driver - Sending",
    "Wrong way driver - Receiving & informing",
    "Amber & blue lights - Receiving & informing",
    "Traffic jam - Receiving & informing",
)


def build_slif_scoring_csv(advanced_scenarios) -> str:
    """The 'VA - Speed assist. verif.' skeleton as a ';'-separated CSV string:
    General requirements, Accuracy, one 'SLIF - Advanced speed limit' row per
    scenario in *advanced_scenarios*, Local hazards, Speedometer accuracy."""
    lines = [
        "",
        "Category;Scenario;Value",
        "SLIF - General requirements;;",
        "SLIF - Accuracy;Distance based KPI;",
        "SLIF - Accuracy;Event based KPI;",
    ]
    lines.extend(
        f"{SLIF_ADVANCED_SPEED_LIMIT_CATEGORY};{scenario};FAIL"
        for scenario in advanced_scenarios
    )
    lines.extend(
        f"SLIF - Local hazards;{scenario};FAIL"
        for scenario in SLIF_LOCAL_HAZARD_VERIFICATION_ROWS
    )
    lines.append("Speed control function;Speedometer accuracy;≥-3 km/h")
    lines.append("")
    return "\n".join(lines)


# Euro NCAP verification skeleton, kept under its historical name.
SLIF_SCORING_CSV_STRING = ENCAP_26_28_PROFILE.verification_csv

ISL_ROW = """
Category;Scenario;Value
Speed control function;Intelligent speed limiter;PASS
"""
IACC_ROW = """
Category;Scenario;Value
Speed control function;Intelligent adaptive cruise control;PASS
"""


def get_slif_advanced_speed_limit_table(
    sas_prediction_df: pd.DataFrame,
    profile: SpeedAssistanceProfile = ENCAP_26_28_PROFILE,
) -> pd.DataFrame:
    slif_adv_table = common.extract_sub_table(
        sas_prediction_df,
        start_row=0,
        n_rows=profile.adv_table_n_rows,
        start_col=0,
        n_cols=profile.adv_table_n_cols,
    )
    slif_adv_table.columns = slif_adv_table.iloc[0]
    slif_adv_table = slif_adv_table[1:].reset_index(drop=True)
    # Column 0 (group label, e.g. "Euro NCAP member states") is sparse and
    # shares a duplicate NaN header with column 1 (country name), so both
    # must be forward-filled and accessed positionally, never by column
    # label, since `slif_adv_table[nan]` returns a 2-column DataFrame.
    group_labels = slif_adv_table.iloc[:, 0].ffill()
    slif_adv_table.fillna("N/A", inplace=True)
    slif_adv_table.iloc[:, 0] = group_labels
    return slif_adv_table


def get_slif_advanced_speed_limit_bg_table(
    sas_prediction_bg_df: pd.DataFrame,
    columns,
    profile: SpeedAssistanceProfile = ENCAP_26_28_PROFILE,
) -> pd.DataFrame:
    slif_adv_bg_table = common.extract_sub_table(
        sas_prediction_bg_df,
        start_row=0,
        n_rows=profile.adv_table_n_rows,
        start_col=0,
        n_cols=profile.adv_table_n_cols,
    )
    # Unlike the text table, the bg table's own sub-header row (scenario
    # names) carries no fill color, so it can't supply column labels itself.
    # Reuse the already-promoted labels from the text table instead; the two
    # tables are row/column aligned since both are built from the same
    # sheet region starting at the same Excel row.
    slif_adv_bg_table = slif_adv_bg_table[1:].reset_index(drop=True)
    slif_adv_bg_table.columns = columns
    return slif_adv_bg_table


def get_local_hazard_table(
    sas_prediction_df: pd.DataFrame,
    profile: SpeedAssistanceProfile = ENCAP_26_28_PROFILE,
) -> pd.DataFrame:
    slif_local_hazards_table = common.extract_sub_table(
        sas_prediction_df,
        start_row=profile.hazard_table_start_row,
        n_rows=6,
        start_col=0,
        n_cols=12,
    )
    slif_local_hazards_table.columns = slif_local_hazards_table.iloc[0]
    slif_local_hazards_table = slif_local_hazards_table[1:].reset_index(drop=True)
    # Forward fill only the first column
    first_col = slif_local_hazards_table.columns[0]
    slif_local_hazards_table[first_col] = slif_local_hazards_table[first_col].ffill()

    slif_local_hazards_table.fillna("N/A", inplace=True)
    logger.debug(f"SLIF Local Hazards Table:\n{slif_local_hazards_table}")
    return slif_local_hazards_table


def get_local_hazard_channel_status(slif_local_hazards_table: pd.DataFrame) -> Dict:
    """Read the Cloud / Direct communication predictions per hazard and direction.

    Returns {hazard: {direction: {"cloud": bool, "direct": bool, "has_red": bool}}}
    """
    channel_status = {}
    for hazard_col in LOCAL_HAZARD_COLUMNS:
        if hazard_col not in slif_local_hazards_table.columns:
            logger.warning(
                f"Column '{hazard_col}' is missing in SLIF Local Hazards Table."
            )
            continue
        hazard_values = [
            str(value).strip().lower()
            for value in slif_local_hazards_table[hazard_col].iloc[:4].tolist()
        ]
        if len(hazard_values) < 4:
            logger.warning(
                f"SLIF Local Hazards Table has only {len(hazard_values)} prediction "
                f"rows for '{hazard_col}', 4 expected."
            )
            continue

        hazard_status = {}
        for direction in LOCAL_HAZARD_DIRECTIONS:
            if (
                direction == LOCAL_HAZARD_SENDING_STR
                and hazard_col in LOCAL_HAZARD_NO_SENDING_COLUMNS
            ):
                continue
            cloud_value = hazard_values[
                LOCAL_HAZARD_CHANNEL_ROW_IDX[("cloud", direction)]
            ]
            direct_value = hazard_values[
                LOCAL_HAZARD_CHANNEL_ROW_IDX[("direct", direction)]
            ]
            hazard_status[direction] = {
                "cloud": cloud_value == "green",
                "direct": direct_value == "green",
                "has_red": "red" in (cloud_value, direct_value),
            }
        channel_status[hazard_col] = hazard_status

    logger.debug(f"SLIF Local Hazards channel status: {channel_status}")
    return channel_status


def update_scenario_scores_with_scf_intelligent_row(
    scenario_scores_df: pd.DataFrame,
    type_of_scf_param,
    intelligent_scenario: str,
) -> pd.DataFrame:
    if scenario_scores_df.empty:
        logger.warning("Scenario Scores sheet is missing or empty.")
        return scenario_scores_df

    if type_of_scf_param is None:
        logger.warning("Type of SCF parameter not found.")
        return scenario_scores_df

    type_of_scf_param = str(type_of_scf_param).strip().lower()
    if type_of_scf_param not in ["iacc", "isl"]:
        logger.debug(
            f"Type of SCF read {type_of_scf_param} does not match 'iacc' or 'isl'. No Scenario Scores row update done."
        )
        return scenario_scores_df

    if not intelligent_scenario:
        logger.warning("Intelligent scenario string is empty.")
        return scenario_scores_df

    category_col = scenario_scores_df["Category"].fillna("").astype(str)
    scenario_col = scenario_scores_df["Scenario"].fillna("").astype(str)

    mask = (category_col.str.strip().str.lower() == "speed control function") & (
        scenario_col.str.strip().str.lower() == "speedometer accuracy"
    )
    indices = scenario_scores_df.index[mask].tolist()

    if not indices:
        logger.warning(
            "No row with Category 'Speed control function' and Scenario 'Speedometer accuracy' found in Scenario Scores sheet."
        )
        return scenario_scores_df

    insert_idx = indices[-1] + 1
    if insert_idx >= len(scenario_scores_df):
        logger.warning(
            "No blank row found after 'Speedometer accuracy' in Scenario Scores sheet."
        )
        return scenario_scores_df

    scenario_scores_df.loc[insert_idx, "Scenario"] = str(intelligent_scenario)

    return scenario_scores_df


def update_scenario_scores_with_system_updates_row(
    scenario_scores_df: pd.DataFrame,
    system_updates_str: str,
) -> pd.DataFrame:
    if scenario_scores_df.empty:
        logger.warning("Scenario Scores sheet is missing or empty.")
        return scenario_scores_df

    if not system_updates_str:
        logger.warning("System updates string is empty.")
        return scenario_scores_df

    category_col = scenario_scores_df["Category"].fillna("").astype(str)
    mask = category_col.str.strip().str.lower() == "slif - system updates"
    indices = scenario_scores_df.index[mask].tolist()

    if not indices:
        logger.warning(
            "No row with Category 'SLIF - System updates' found in Scenario Scores sheet."
        )
        return scenario_scores_df

    idx = indices[0]
    scenario_scores_df.loc[idx, "Scenario"] = str(system_updates_str)

    return scenario_scores_df


def preprocess(dfs: dict[str, pd.DataFrame], protocol=None) -> Dict:
    """Derive the 'VA - Speed assist. verif.' skeleton from the prediction
    sheet. *protocol* selects the SpeedAssistanceProfile; None reads the
    marker in dfs["Version"] and falls back to Euro NCAP
    (common.resolve_protocol)."""
    profile = resolve_profile(protocol, dfs)
    logger.info(f"Preprocess SAS ({profile.protocol}) ...")
    slif_verification_df = pd.read_csv(io.StringIO(profile.verification_csv), sep=";")
    sas_prediction_df = dfs.get("VA - Speed assist. pred.")
    sas_prediction_bg_df = dfs.get("VA - Speed assist. pred. (bg)")

    slif_adv_table = get_slif_advanced_speed_limit_table(sas_prediction_df, profile)
    slif_adv_bg_table = (
        get_slif_advanced_speed_limit_bg_table(
            sas_prediction_bg_df, slif_adv_table.columns, profile
        )
        if sas_prediction_bg_df is not None
        else None
    )

    total_country_count = len(slif_adv_table)
    if total_country_count != profile.expected_row_count:
        raise ValueError(
            f"Expected {profile.expected_row_count} country rows in the SLIF "
            f"advanced speed limit table for protocol '{profile.protocol}', found "
            f"{total_country_count}. The 'VA - Speed assist. pred.' sheet layout "
            "may have changed, or the file was generated for a different protocol."
        )

    member_state_mask = (
        slif_adv_table.iloc[:, 0].map(lambda v: str(v).strip().lower())
        == profile.group_label.strip().lower()
    )
    if member_state_mask.sum() != profile.expected_group_row_count:
        raise ValueError(
            f"Expected {profile.expected_group_row_count} '{profile.group_label}' "
            f"rows in the SLIF advanced speed limit table, found "
            f"{member_state_mask.sum()}. The 'VA - Speed assist. pred.' sheet "
            "layout may have changed."
        )

    missing_headers = [
        key for key in profile.slif_adv_mapping if key not in slif_adv_table.columns
    ]
    if missing_headers:
        raise ValueError(
            f"Advanced speed limit columns {missing_headers} are missing from the "
            f"'VA - Speed assist. pred.' sheet for protocol '{profile.protocol}'. "
            "The file may have been generated for a different protocol."
        )

    implicit_scenario_status_list = []

    for slif_adv_key, scenario_str in profile.slif_adv_mapping.items():
        current_text_col = slif_adv_table[slif_adv_key]
        current_bg_col = (
            slif_adv_bg_table[slif_adv_key] if slif_adv_bg_table is not None else None
        )
        scenario_full_name = f"{scenario_str} - {slif_adv_key}"

        def classify(idx):
            norm = str(current_text_col.iloc[idx]).strip().lower()
            if norm == "green":
                return "GREEN"
            if norm == "red":
                return "RED"
            # Any other text (pasted or written programmatically, bypassing
            # the Green/Red dropdown) is not a blank N/A: it fails the row.
            # Empty cells arrive here as "N/A" (fillna above).
            if not common.is_empty_cell(norm) and norm not in ("n/a", "-"):
                logger.warning(
                    f"Unexpected SLIF prediction '{current_text_col.iloc[idx]}' in "
                    f"column '{slif_adv_key}', row {idx}: treated as failing."
                )
                return "INVALID"
            # Blank text: distinguish truly-blank (white/no-fill) from an
            # explicit grey/other manual fill, which the sheet's Green/Red-only
            # dropdown can't express as text.
            if current_bg_col is not None:
                color = current_bg_col.iloc[idx]
                if color not in (
                    None,
                    common.PredictionColor.GREEN,
                    common.PredictionColor.RED,
                ):
                    return "GREY_NA"
            return "BLANK"

        classified = [classify(i) for i in range(total_country_count)]
        logger.debug(
            f"Processing SLIF prediction column '{slif_adv_key}' for scenario '{scenario_str}' with values:\n{classified}"
        )

        if profile.pass_rule == SlifAdvancedPassRule.ALL_ROWS:
            # Every market row must be Green or blank (N/A): a single Red,
            # grey-N/A or unknown-text market fails the scenario.
            failing_rows = [c for c in classified if c in ("RED", "GREY_NA", "INVALID")]
            if not failing_rows:
                scenario_status = "PASS"
                logger.debug(
                    f"Setting scenario '{scenario_full_name}' to PASS: no market "
                    "row is Red/N-A."
                )
            else:
                scenario_status = "FAIL"
                logger.debug(
                    f"Setting scenario '{scenario_full_name}' to FAIL: "
                    f"{len(failing_rows)}/{total_country_count} market rows are Red/N-A."
                )
        else:
            member_classified = [
                c for c, is_member in zip(classified, member_state_mask) if is_member
            ]
            member_states_pass = not any(
                c in ("RED", "GREY_NA", "INVALID") for c in member_classified
            )
            green_or_blank_count = sum(c in ("GREEN", "BLANK") for c in classified)
            application_area_pass = green_or_blank_count >= (total_country_count / 2)

            if member_states_pass and application_area_pass:
                scenario_status = "PASS"
                logger.debug(
                    f"Setting scenario '{scenario_full_name}' to PASS: no member-state "
                    f"Red/N-A, and {green_or_blank_count}/{total_country_count} application-area "
                    "countries are Green or blank."
                )
            else:
                scenario_status = "FAIL"
                logger.debug(
                    f"Setting scenario '{scenario_full_name}' to FAIL: member_states_pass="
                    f"{member_states_pass}, application_area_pass={application_area_pass} "
                    f"({green_or_blank_count}/{total_country_count})."
                )

        if profile.collapse_implicit and scenario_str == IMPLICIT_SPEED_LIMIT_STR:
            implicit_scenario_status_list.append(scenario_status)
            continue

        slif_verification_df.loc[
            (slif_verification_df["Scenario"] == scenario_full_name),
            "Value",
        ] = scenario_status

    if profile.collapse_implicit:
        implicit_final_status = (
            "PASS"
            if all(status == "PASS" for status in implicit_scenario_status_list)
            else "FAIL"
        )

        slif_verification_df.loc[
            slif_verification_df["Scenario"] == profile.implicit_collapsed_scenario,
            "Value",
        ] = implicit_final_status
        logger.debug(
            f"Final status for '{profile.implicit_collapsed_scenario}': {implicit_final_status}"
        )

    slif_local_hazards_table = get_local_hazard_table(sas_prediction_df, profile)
    logger.debug(
        f"Processing SLIF Local Hazards predictions for scenarios starting with 'SLIF - Local Hazards' with values:\n{slif_local_hazards_table}"
    )
    # Determine PASS/FAIL for SLIF Local Hazards scenarios based on prediction table
    # For each hazard, set PASS/FAIL in slif_verification_df
    for hazard_col in LOCAL_HAZARD_COLUMNS:
        hazard_col_df = slif_local_hazards_table[hazard_col]
        if hazard_col_df.empty:
            logger.warning(
                f"Column '{hazard_col}' is missing in SLIF Local Hazards Table."
            )
            continue
        hazard_values = hazard_col_df.iloc[:4].tolist()

        # Sending status
        is_cloud_sending = hazard_values[0].lower() == "green"
        is_direct_sending = hazard_values[2].lower() == "green"
        sending_status = "PASS" if (is_cloud_sending or is_direct_sending) else "FAIL"
        slif_verification_df.loc[
            (slif_verification_df["Category"] == "SLIF - Local hazards")
            & (slif_verification_df["Scenario"] == f"{hazard_col} - Sending"),
            "Value",
        ] = sending_status

        # Receiving status
        is_cloud_receiving = hazard_values[1].lower() == "green"
        is_direct_receiving = hazard_values[3].lower() == "green"
        receiving_status = (
            "PASS" if (is_cloud_receiving or is_direct_receiving) else "FAIL"
        )
        slif_verification_df.loc[
            (slif_verification_df["Category"] == "SLIF - Local hazards")
            & (
                slif_verification_df["Scenario"]
                == f"{hazard_col} - Receiving & informing"
            ),
            "Value",
        ] = receiving_status

        logger.debug(
            f"Hazard '{hazard_col}' - Sending status: {sending_status}, Receiving status: {receiving_status}"
        )

    param_dict = common.create_param_dict_from_input_parameters(
        dfs.get("Input parameters", pd.DataFrame()), col="Category"
    )
    logger.debug(f"Parameter dictionary created from input parameters:\n{param_dict}")

    type_of_scf_param = param_dict.get("Speed control function", {}).get(
        "Type of system", ""
    )
    logger.info(f"Type of SCF parameter value: {type_of_scf_param}")
    type_of_scf_param = str(type_of_scf_param).strip().lower()

    if type_of_scf_param == "iacc":
        intelligent_row = pd.read_csv(io.StringIO(IACC_ROW), sep=";")
    elif type_of_scf_param == "isl":
        intelligent_row = pd.read_csv(io.StringIO(ISL_ROW), sep=";")
    else:
        logger.debug(
            f"Type of SCF read {type_of_scf_param} does not match 'iacc' or 'isl'. No additional rows added to SLIF verification DataFrame."
        )
        intelligent_row = pd.DataFrame()

    slif_verification_df = pd.concat(
        [slif_verification_df, intelligent_row],
        ignore_index=True,
    )

    slif_verification_df = common.opposite_ffill(
        slif_verification_df, exclude_columns=["Value"]
    )

    intelligent_scenario_str = (
        intelligent_row["Scenario"].iloc[0] if not intelligent_row.empty else ""
    )
    # Update Scenario Scores sheet
    scenario_scores_df = dfs.get("Scenario Scores", pd.DataFrame())
    scenario_scores_df = update_scenario_scores_with_scf_intelligent_row(
        scenario_scores_df, type_of_scf_param, intelligent_scenario_str
    )
    input_parameters_df = dfs.get("Input parameters", pd.DataFrame())
    system_updates_row = input_parameters_df[
        input_parameters_df["Category"] == "SLIF - System updates"
    ]
    system_updates_str = (
        system_updates_row["Value"].iloc[0] if not system_updates_row.empty else ""
    )
    logger.debug(
        f"System updates string extracted from Input Parameters: '{system_updates_str}'"
    )

    scenario_scores_df = update_scenario_scores_with_system_updates_row(
        scenario_scores_df, system_updates_str
    )

    return slif_verification_df, scenario_scores_df


def apply_advanced_speed_limit_sub_caps(
    category_scores_df: pd.DataFrame,
    scenario_scores_df: pd.DataFrame,
    profile: SpeedAssistanceProfile = None,
    protocol=None,
    dfs: dict = None,
) -> pd.DataFrame:
    """Apply the profile's Advanced Speed Limits sub-caps (a programme may cap
    the summed points of a scenario group) to the 'SLIF - Advanced speed
    limit' row of *category_scores_df*, after the category has been summed
    from *scenario_scores_df*.

    Mirrors the Local Hazards cap: the raw per-scenario points are left as
    they are, only the category total is reduced, by the amount the capped
    group's summed scenario scores exceed its cap. An unassessed (NaN)
    scenario in the group leaves the (already NaN) category untouched. A
    profile without sub-caps (Euro NCAP) is a no-op.

    *profile* wins over *protocol*, which wins over the marker in *dfs*.
    """
    if profile is None:
        profile = resolve_profile(protocol, dfs)
    if not profile.sub_caps:
        return category_scores_df
    if (
        category_scores_df is None
        or category_scores_df.empty
        or "Category" not in category_scores_df.columns
        or scenario_scores_df is None
        or scenario_scores_df.empty
        or "Scenario" not in scenario_scores_df.columns
    ):
        return category_scores_df

    category_mask = (
        category_scores_df["Category"].astype(str).str.strip()
        == SLIF_ADVANCED_SPEED_LIMIT_CATEGORY
    )
    if not category_mask.any():
        logger.warning(
            f"No '{SLIF_ADVANCED_SPEED_LIMIT_CATEGORY}' row in Category Scores; "
            "sub-caps not applied."
        )
        return category_scores_df
    category_idx = category_scores_df.index[category_mask][0]
    category_score = pd.to_numeric(
        pd.Series([category_scores_df.at[category_idx, "Score"]]), errors="coerce"
    ).iloc[0]
    if pd.isna(category_score):
        return category_scores_df

    scenario_scores = pd.to_numeric(scenario_scores_df["Score"], errors="coerce")
    scenario_names = scenario_scores_df["Scenario"].astype(str).str.strip()
    for scenarios, cap in profile.sub_caps:
        group_scores = scenario_scores[scenario_names.isin(scenarios)]
        if group_scores.empty or group_scores.isna().any():
            continue
        excess = group_scores.sum() - cap
        if excess <= 0:
            continue
        # 0.075-point steps are not exactly representable; round the
        # adjusted total to the protocol's decimal precision, half-up.
        capped = round_half_up(category_score - excess, 6)
        logger.info(
            f"Advanced speed limit sub-cap: {len(group_scores)} capped scenarios "
            f"sum to {group_scores.sum():.3f} > {cap}; category score "
            f"{category_score:.3f} -> {capped:.3f}"
        )
        category_scores_df.loc[category_idx, "Score"] = capped
        category_score = capped
    return category_scores_df


def compute_score(dfs: dict[str, pd.DataFrame], protocol=None) -> Dict:
    """Score Speed Assistance from the preprocessed verification/prediction
    sheets. *protocol* selects the SpeedAssistanceProfile; None reads the
    marker in dfs["Version"] and falls back to Euro NCAP.

    Returns (verification_df, scenario_scores_df, slif_score_dict,
    hazard_capping_score); the Advanced Speed Limits sub-caps are applied
    separately, at Category Scores level, by
    apply_advanced_speed_limit_sub_caps.
    """
    profile = resolve_profile(protocol, dfs)
    logger.info(f"Computing SAS prediction score ({profile.protocol})...")
    scenario_scores_df = dfs.get("Scenario Scores", pd.DataFrame())
    accuracy_score_dict = {
        "Distance based": 0.0,
        "Event based": 0.0,
    }
    advanced_score_dict = {scenario: 0.0 for scenario in profile.advanced_scenarios}

    hazard_score_dict = {
        "Construction zones - Sending": 0.0,
        "Construction zones - Receiving & informing": 0.0,
        "Items on road - Sending": 0.0,
        "Items on road - Receiving & informing": 0.0,
        "Stopped vehicle - Sending": 0.0,
        "Stopped vehicle - Receiving & informing": 0.0,
        "Broken down vehicle - Sending": 0.0,
        "Broken down vehicle - Receiving & informing": 0.0,
        "Post crash - Sending": 0.0,
        "Post crash - Receiving & informing": 0.0,
        "Poor weather - Sending": 0.0,
        "Poor weather - Receiving & informing": 0.0,
        "Poor road - Sending": 0.0,
        "Poor road - Receiving & informing": 0.0,
        "Wrong way driver - Sending": 0.0,
        "Wrong way driver - Receiving & informing": 0.0,
        "Amber & blue lights - Receiving & informing": 0.0,
        "Traffic jam - Receiving & informing": 0.0,
    }

    scf_score_dict = {
        "Speedometer accuracy": 0.0,
        "Intelligent speed limiter": 0.0,
        "Intelligent adaptive cruise control": 0.0,
    }

    slif_score_dict = {
        **accuracy_score_dict,
        **advanced_score_dict,
        **hazard_score_dict,
        **scf_score_dict,
    }

    ####### 1. Get relevant DataFrames
    sas_prediction_df = dfs.get("VA - Speed assist. pred.")
    sas_verification_df = dfs.get("VA - Speed assist. verif.")
    if sas_prediction_df is None or sas_verification_df is None:
        logger.error(
            "Required DataFrames 'VA - Speed assist. pred.' or 'VA - Speed assist. verif.' are missing in dfs."
        )
        return pd.DataFrame(), scenario_scores_df, slif_score_dict, 0.0

    sas_verification_df["Score"] = 0.0
    sas_verification_df["Score"] = sas_verification_df["Score"].astype(float)

    logger.debug(
        f"Initial SAS Verification DataFrame with Score column added:\n{sas_verification_df}"
    )
    ####### 2. Read general requirements
    # Extract General requirements section
    general_req_rows = common.extract_section(
        sas_verification_df, "SLIF - General requirements", col_name="Category"
    )
    # Remove all rows where every value is NaN
    general_req_rows = general_req_rows.dropna(how="all")

    if general_req_rows.empty:
        logger.warning("General requirements section is missing.")
        return sas_verification_df, scenario_scores_df, slif_score_dict, 0.0

    general_req_state = common.classify_pass_fail_section(
        general_req_rows, column="Value"
    )

    # Not yet assessed by the OEM -- unassessed (NaN), not a silent 0/FAIL.
    if general_req_state == "blank":
        logger.info(
            "General requirements not yet assessed. SLIF score remains unassessed."
        )
        slif_score_dict = {key: np.nan for key in slif_score_dict}
        return sas_verification_df, scenario_scores_df, slif_score_dict, None

    # If any general requirement failed, return score dict with all zeros
    if general_req_state == "fail":
        logger.info("One or more General requirements failed. Keeping score to 0.")
        # System updates is scored on the Scenario Scores row (step 6), not
        # through slif_score_dict, so zero it here too -- otherwise it stays
        # unassessed, or keeps points stamped by an earlier calculation.
        if "Category" in scenario_scores_df.columns:
            system_updates_mask = (
                scenario_scores_df["Category"].astype(str).str.strip().str.lower()
                == "slif - system updates"
            )
            scenario_scores_df.loc[system_updates_mask, "Score"] = 0.0
        return sas_verification_df, scenario_scores_df, slif_score_dict, 0.0

    ####### 3. SLIF Accuracy scoring
    # Reconstruct the collapsed Category/Scenario labels only -- NOT
    # "Value"/"Score": those are per-row OEM input, and forward-filling
    # them would silently copy a previous row's PASS/FAIL onto a row the
    # OEM left genuinely blank, turning an unassessed scenario into an
    # assessed one. DataFrame-sourced data can represent a blank label cell as
    # an empty string rather than NaN/pd.NA -- .ffill() only propagates
    # over actual nulls, so normalize is_empty_cell()-blank values first.
    ffill_columns = [
        c for c in sas_verification_df.columns if c not in ("Value", "Score")
    ]
    sas_verification_df[ffill_columns] = (
        sas_verification_df[ffill_columns]
        .map(lambda v: pd.NA if common.is_empty_cell(v) else v)
        .ffill()
    )
    accuracy_df = sas_verification_df[
        sas_verification_df["Category"] == "SLIF - Accuracy"
    ]
    # Distance based KPI
    dist_value_row = accuracy_df[accuracy_df["Scenario"] == "Distance based KPI"][
        "Value"
    ]
    dist_value = dist_value_row.iloc[0] if dist_value_row.size > 0 else "FAIL"
    dist_state = common.classify_pass_fail_value(dist_value)
    if dist_state == "pass":
        accuracy_score_dict["Distance based"] = 2.0
    elif dist_state == "blank":
        accuracy_score_dict["Distance based"] = np.nan
    if dist_state in ("pass", "blank"):
        sas_verification_df.loc[
            (sas_verification_df["Category"] == "SLIF - Accuracy")
            & (sas_verification_df["Scenario"] == "Distance based KPI"),
            "Score",
        ] = accuracy_score_dict["Distance based"]

    # Event Based KPI
    event_value_row = accuracy_df[accuracy_df["Scenario"] == "Event based KPI"]["Value"]
    event_value = event_value_row.iloc[0] if event_value_row.size > 0 else "FAIL"
    event_state = common.classify_pass_fail_value(event_value)
    if event_state == "pass":
        accuracy_score_dict["Event based"] = 2.0
    elif event_state == "blank":
        accuracy_score_dict["Event based"] = np.nan
    if event_state in ("pass", "blank"):
        sas_verification_df.loc[
            (sas_verification_df["Category"] == "SLIF - Accuracy")
            & (sas_verification_df["Scenario"] == "Event based KPI"),
            "Score",
        ] = accuracy_score_dict["Event based"]

    logger.info(
        f"SAS Accuracy Scores: Distance based: {accuracy_score_dict['Distance based']}, Event based: {accuracy_score_dict['Event based']}"
    )

    ####### 4. Conditional Speed Limits scoring
    # Points are protocol constants carried by the profile, not read from
    # the template's 'Max score' column.
    slif_adv_max_scores_dict = dict(profile.advanced_points)

    for full_scenario_name in slif_adv_max_scores_dict.keys():

        current_scenario_row = sas_verification_df[
            (sas_verification_df["Scenario"] == full_scenario_name)
        ]

        if current_scenario_row["Value"].empty:
            logger.debug(
                f"Scenario '{full_scenario_name}' is missing in SLIF verification DataFrame."
            )
            continue
        current_scenario_value = current_scenario_row["Value"].iloc[0]
        if str(current_scenario_value).strip().lower() == "pass":
            max_score = slif_adv_max_scores_dict[full_scenario_name]
            advanced_score_dict[full_scenario_name] = max_score
            sas_verification_df.loc[
                (sas_verification_df["Scenario"] == full_scenario_name),
                "Score",
            ] = max_score
            logger.info(
                f"Scenario '{full_scenario_name}' passed. Assigned score: {max_score}"
            )
        else:
            logger.info(
                f"Scenario '{full_scenario_name}' did not pass. Score remains 0."
            )

    # Coupled pairs (Euro NCAP arrows / dynamic): the dependent half scores 0
    # whenever its prerequisite scored 0.
    for dependent_scenario, prerequisite_scenario in profile.coupling_rules:
        if advanced_score_dict[prerequisite_scenario] == 0.0:
            advanced_score_dict[dependent_scenario] = 0.0
            sas_verification_df.loc[
                (sas_verification_df["Scenario"] == dependent_scenario),
                "Score",
            ] = 0.0

    ####### 5. Local Hazards scoring
    # The PASS/FAIL gate is unchanged: a hazard/direction passes as soon as either
    # the Cloud or the Direct communication prediction is 'Green'. What the
    # available points are, however, depends on HOW MANY channels are 'Green':
    # 'Direct AND Cloud' scores the full points, 'Direct OR Cloud' the reduced
    # ones, per protocol section 1.2.3. The 'Max score' columns of the
    # verification and scoring sheets are never modified - only 'Score' changes.
    slif_local_hazards_table = get_local_hazard_table(sas_prediction_df, profile)
    hazard_channel_status = get_local_hazard_channel_status(slif_local_hazards_table)

    # Tracks whether any hazard/direction is missing one of the two communication
    # channels ('Red' or not claimed), which also lowers the category total cap
    # from 3.0 to 2.5.
    single_channel_communication = False

    for current_scenario in LOCAL_HAZARD_COLUMNS:
        for direction in LOCAL_HAZARD_DIRECTIONS:
            scenario_name = f"{current_scenario} - {direction}"

            scenario_row = sas_verification_df[
                (sas_verification_df["Scenario"] == scenario_name)
            ]
            if scenario_row["Value"].empty:
                logger.debug(
                    f"Scenario '{scenario_name}' is missing in SLIF verification DataFrame."
                )
                continue

            channel_status = hazard_channel_status.get(current_scenario, {}).get(
                direction
            )
            if channel_status is None:
                logger.warning(
                    f"No Cloud / Direct communication prediction found for "
                    f"'{scenario_name}'. Assuming both communication channels."
                )
                channel_status = {"cloud": True, "direct": True, "has_red": False}

            # A 'Red' prediction lowers the category cap even when the other
            # channel still earns the (reduced) points.
            if channel_status["has_red"]:
                single_channel_communication = True

            scenario_value = scenario_row["Value"].iloc[0]
            if str(scenario_value).strip().lower() != "pass":
                logger.info(
                    f"Hazard scenario '{scenario_name}' did not pass. Score remains 0."
                )
                continue

            both_channels = channel_status["cloud"] and channel_status["direct"]
            if not both_channels:
                single_channel_communication = True

            full_points, reduced_points = LOCAL_HAZARD_POINTS[direction]
            score = full_points if both_channels else reduced_points

            hazard_score_dict[scenario_name] = score
            sas_verification_df.loc[
                (sas_verification_df["Scenario"] == scenario_name), "Score"
            ] = score
            logger.info(
                f"Hazard scenario '{scenario_name}' passed with "
                f"{'Cloud AND Direct' if both_channels else 'a single'} "
                f"communication channel (cloud={channel_status['cloud']}, "
                f"direct={channel_status['direct']}). Assigned score: {score}"
            )

    # Cap the Local Hazards category total: 3.0 when every scored hazard has both
    # Cloud AND Direct communication, 2.5 as soon as one channel is missing.
    hazard_capping_score = LOCAL_HAZARD_CAP_BOTH_CHANNELS
    if single_channel_communication:
        hazard_capping_score = LOCAL_HAZARD_CAP_SINGLE_CHANNEL
        logger.info(
            "At least one SLIF Local Hazard is not covered by both Cloud and Direct "
            f"communication. Capping total Local Hazards score to {hazard_capping_score}"
        )

    ####### 6. SLIF - System Updates scoring
    # temporary 1, continous 2
    system_updates_row = scenario_scores_df[
        scenario_scores_df["Category"].str.strip().str.lower()
        == "slif - system updates"
    ]
    system_updates_value = (
        system_updates_row["Scenario"].iloc[0] if not system_updates_row.empty else ""
    )
    if common.is_empty_cell(system_updates_value):
        # Caller-supplied dfs may carry a Scenario Scores sheet whose Scenario cell
        # was never stamped by preprocess; fall back to the raw Input
        # parameters value in that case. DataFrame-sourced inputs may also
        # differ in casing/whitespace, so match Category normalized (same
        # convention as the Scenario Scores lookup above).
        input_parameters_df = dfs.get("Input parameters", pd.DataFrame())
        if not input_parameters_df.empty and "Category" in input_parameters_df.columns:
            input_row = input_parameters_df[
                input_parameters_df["Category"].astype(str).str.strip().str.lower()
                == "slif - system updates"
            ]
            if not input_row.empty:
                system_updates_value = input_row["Value"].iloc[0]
    logger.info(
        f"System updates scenario value from Scenario Scores: '{system_updates_value}'"
    )
    idx = system_updates_row.index[0] if not system_updates_row.empty else None
    if idx is not None:
        if str(system_updates_value).strip().lower() == "continuous":
            scenario_scores_df.loc[idx, "Score"] = 2.0
            logger.info(f"System updates scenario is 'Continuous'. Assigned score: 2.0")
        elif str(system_updates_value).strip().lower() == "temporary":
            scenario_scores_df.loc[idx, "Score"] = 1.0
            logger.info(f"System updates scenario is 'Temporary'. Assigned score: 1.0")
        elif data_model.is_not_applicable(system_updates_value):
            scenario_scores_df.loc[idx, "Score"] = 0.0
            logger.info("System updates scenario is 'N/A'. Assigned score: 0.0")
    else:
        logger.info(
            f"System updates scenario is '{system_updates_value}'. "
            "Score remains unassessed (NaN)."
        )

    ####### 7. SCF - General requirements scoring

    speedo_row = sas_verification_df[
        (
            sas_verification_df["Category"].str.strip().str.lower()
            == "speed control function"
        )
        & (
            sas_verification_df["Scenario"].str.strip().str.lower()
            == "speedometer accuracy"
        )
    ]
    speedo_idx = speedo_row.index[0] if not speedo_row.empty else None

    if speedo_idx is None:
        raise ValueError(
            "No row with Category 'Speed control function' and Scenario 'Speedometer accuracy' found in SLIF verification DataFrame."
        )

    speedo_row_value = (
        sas_verification_df.loc[speedo_idx, "Value"] if speedo_idx is not None else ""
    )
    # Set intelligent_row_value to the next row after speedo_row
    intelligent_row_idx = speedo_idx + 1
    intelligent_row = (
        sas_verification_df.loc[intelligent_row_idx]
        if intelligent_row_idx in sas_verification_df.index
        else None
    )
    scf_points = 0

    if intelligent_row is None:
        # No ISL/ACC system configured (blank "Type of system" input) -- the
        # intelligent scenario row was never added during preprocessing.
        # scf_score_dict's defaults for both scenarios already stay at 0.0.
        logger.info(
            "No intelligent scenario row found after 'Speedometer accuracy' in SLIF "
            "verification DataFrame. Intelligent SCF score remains 0."
        )
    else:
        intelligent_row_scenario = intelligent_row["Scenario"]
        # A configured ISL/ACC system can still have a blank Value (not yet
        # tested) -- use the shared classifier instead of .strip() directly,
        # which crashes on NaN, and leave the row unassessed (NaN) rather
        # than silently treating blank as a FAIL.
        intelligent_row_state = common.classify_pass_fail_value(
            intelligent_row["Value"]
        )

        if intelligent_row_state == "blank":
            scf_points = np.nan
        elif intelligent_row_state == "pass":
            if intelligent_row_scenario == "Intelligent adaptive cruise control":
                scf_points = 8
            elif intelligent_row_scenario == "Intelligent speed limiter":
                scf_points = 5

        scf_score_dict[intelligent_row_scenario] = scf_points
        sas_verification_df.loc[
            (sas_verification_df["Scenario"] == intelligent_row_scenario),
            "Score",
        ] = scf_points

    speed_multiplier = 0
    if speedo_row_value == "≥-3 km/h":
        speed_multiplier = 0
    elif speedo_row_value == "≥-5 km/h":
        speed_multiplier = 0.5
    elif speedo_row_value == "<-5 km/h":
        speed_multiplier = 1.0

    speedometer_accuracy_score = -scf_points * speed_multiplier
    scf_score_dict["Speedometer accuracy"] = speedometer_accuracy_score
    sas_verification_df.loc[
        (sas_verification_df["Scenario"] == "Speedometer accuracy"),
        "Score",
    ] = speedometer_accuracy_score
    logger.debug(
        f"Speedometer accuracy value: '{speedo_row_value}', Speed multiplier: {speed_multiplier}, Speedometer accuracy score: {speedometer_accuracy_score}"
    )

    sas_verification_df = common.opposite_ffill(
        sas_verification_df, exclude_columns=["Value", "Score"]
    )

    slif_score_dict = {
        **accuracy_score_dict,
        **advanced_score_dict,
        **hazard_score_dict,
        **scf_score_dict,
    }

    return (
        sas_verification_df,
        scenario_scores_df,
        slif_score_dict,
        hazard_capping_score,
    )
