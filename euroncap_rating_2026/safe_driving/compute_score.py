# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import numpy as np
import pandas as pd
import logging
import openpyxl

from euroncap_rating_2026.common import with_footer, hard_copy_sheet
from euroncap_rating_2026 import config
from euroncap_rating_2026 import common
from euroncap_rating_2026.numeric import round_half_up
from euroncap_rating_2026.safe_driving import (
    data_model,
    occupant_classification,
    occupant_presence,
    report_writer,
    driver_monitoring,
    general_vehicle_controls,
    seatbelt_usage,
    speed_assistance,
    acc_performance,
    steering_assistance,
    integrity,
)

import sys
import os

import click


logger = logging.getLogger(__name__)
settings = config.Settings()


def get_updated_dfs(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    # Re-derive every protected reference cell (Max score, row labels, ...)
    # from the official packaged template instead of trusting it from the
    # caller's dfs -- this is the real dataframe-in/dataframe-out entry point
    # external callers use (sd_get_updated_dfs), never chained in-process from
    # any preprocess-equivalent's output, so this must happen unconditionally
    # here. Only the designated grey input cells are carried over. Note:
    # "Scenario Scores" is deliberately not covered (see integrity.py).
    dfs = integrity.rebuild_trusted_input_dfs(dfs)

    scenario_scores_df = dfs.get("Scenario Scores", pd.DataFrame())
    category_scores_df = dfs.get("Category Scores", pd.DataFrame())
    test_scores_df = dfs.get("Test Scores", pd.DataFrame())
    # The packaged template hardcodes 0 (not blank) in every Score cell.
    # Reset to NaN before any domain module runs, so a row a module never
    # touches (e.g. a scenario skipped because a gating input parameter is
    # blank) surfaces as unassessed instead of silently keeping that
    # template placeholder 0. Category/Test Scores are fully re-derived
    # from this sheet below via update_score_sum_interval.
    if "Score" in scenario_scores_df.columns:
        scenario_scores_df["Score"] = np.nan
    # Compute Occupant Monitoring Score
    updated_omseatbelt_df, seatbelt_score_dict = seatbelt_usage.compute_seatbelt_score(
        dfs
    )
    logger.debug(f"Seatbelt score dict: {seatbelt_score_dict}")
    updated_occupant_classificationv_df, classificationv_score_dict = (
        occupant_classification.compute_classification_score(dfs)
    )
    logger.debug(f"Occupant classification score dict: {classificationv_score_dict}")
    updated_occupant_presence_df, presence_score_dict = (
        occupant_presence.compute_classification_score(dfs)
    )
    logger.debug(f"Occupant presence score dict: {presence_score_dict}")

    # # Compute Driver Monitoring Score
    driver_monitoring_score_df = driver_monitoring.compute_score(dfs)
    updated_dm_verification_df = dfs.get("DE - DM verif.")
    if updated_dm_verification_df is not None:
        updated_dm_verification_df = updated_dm_verification_df.copy()
        updated_dm_verification_df.columns = [
            col if "Unnamed" not in str(col) else ""
            for col in updated_dm_verification_df.columns.tolist()
        ]
    general_vehicle_controls_score_dict = general_vehicle_controls.compute_score(dfs)
    logger.debug(
        f"General vehicle controls score dict: {general_vehicle_controls_score_dict}"
    )

    # Compute Vehicle Assistance Score - Speed Assistance
    (
        updated_sas_verification_df,
        scenario_scores_df,
        ve_sas_score_dict,
        hazard_capping_score,
    ) = speed_assistance.compute_score(dfs)
    logger.debug(f"Speed assistance score dict: {ve_sas_score_dict}")

    # # Compute ACC Performance Score
    updated_acc_verification_df, acc_score_dict = acc_performance.compute_score(dfs)
    logger.debug(f"ACC performance score dict: {acc_score_dict}")

    # # Compute Steering Assistance Score
    updated_steering_assistance_df, steering_assistance_score_dict = (
        steering_assistance.compute_score(dfs)
    )
    logger.debug(f"Steering assistance score dict: {steering_assistance_score_dict}")
    # Update Scenario Scores sheet
    scenario_scores_df = common.update_scoring_sheet(
        scenario_scores_df, seatbelt_score_dict, key_column="Scenario"
    )
    # When the passenger airbag input parameter is N/A, this dict carries an
    # "N/A" key (value 0.0) that also matches the "SLIF - System updates" and
    # "Auto-resume" rows whenever their Scenario is "N/A" too -- benign, since
    # those rows only carry an "N/A" scenario when their own input parameter
    # is N/A, in which case 0.0 is exactly their correct score.
    scenario_scores_df = common.update_scoring_sheet(
        scenario_scores_df, classificationv_score_dict, key_column="Scenario"
    )
    scenario_scores_df = common.update_scoring_sheet(
        scenario_scores_df, presence_score_dict, key_column="Scenario"
    )
    scenario_scores_df = common.update_scoring_sheet_from_df(
        scenario_scores_df, driver_monitoring_score_df
    )
    scenario_scores_df = common.update_scoring_sheet(
        scenario_scores_df, general_vehicle_controls_score_dict, key_column="Scenario"
    )
    scenario_scores_df = common.update_scoring_sheet(
        scenario_scores_df, ve_sas_score_dict, key_column="Scenario"
    )
    # Some scenario score for ve sas are on Category col
    scenario_scores_df = common.update_scoring_sheet(
        scenario_scores_df, ve_sas_score_dict, key_column="Category"
    )
    scenario_scores_df = common.update_scoring_sheet(
        scenario_scores_df, acc_score_dict, key_column="Scenario"
    )
    scenario_scores_df = common.update_scoring_sheet(
        scenario_scores_df, acc_score_dict, key_column="Category"
    )
    scenario_scores_df = common.update_scoring_sheet(
        scenario_scores_df, steering_assistance_score_dict, key_column="Scenario"
    )
    scenario_scores_df = common.update_scoring_sheet(
        scenario_scores_df, steering_assistance_score_dict, key_column="Category"
    )

    category_scores_df = common.update_score_sum_interval(
        category_scores_df,
        scenario_scores_df,
        key_column="Category",
        interval_column="Category",
    )
    # Hazard edge case: if any SLIF Local Hazard has 'Red' status, cap the score to 2.5
    # Apply hazard capping score if present
    if (
        "SLIF - Local hazards" in category_scores_df["Category"].values
        and hazard_capping_score is not None
    ):
        category_scores_df.loc[
            category_scores_df["Category"] == "SLIF - Local hazards",
            "Score",
        ] = min(
            category_scores_df.loc[
                category_scores_df["Category"] == "SLIF - Local hazards",
                "Score",
            ].values[0],
            hazard_capping_score,
        )
    test_scores_df = common.update_score_sum_interval(
        test_scores_df,
        category_scores_df,
        key_column="Stage subelement",
        interval_column="Stage subelement",
    )

    # Remove 'Score' column from the relevant DataFrames if present
    if "Score" in updated_omseatbelt_df.columns:
        updated_omseatbelt_df = updated_omseatbelt_df.drop(columns=["Score"])
    if "Score" in updated_occupant_classificationv_df.columns:
        updated_occupant_classificationv_df = updated_occupant_classificationv_df.drop(
            columns=["Score"]
        )
    if "Score" in updated_occupant_presence_df.columns:
        updated_occupant_presence_df = updated_occupant_presence_df.drop(
            columns=["Score"]
        )
    if "Score" in updated_sas_verification_df.columns:
        updated_sas_verification_df = updated_sas_verification_df.drop(
            columns=["Score"]
        )
    if "Score" in updated_acc_verification_df.columns:
        updated_acc_verification_df = updated_acc_verification_df.drop(
            columns=["Score"]
        )
    if "Score" in updated_steering_assistance_df.columns:
        updated_steering_assistance_df = updated_steering_assistance_df.drop(
            columns=["Score"]
        )

    updated_dfs = {}
    updated_dfs["OM - Seatbelt usage verif."] = updated_omseatbelt_df
    updated_dfs["OM - Occ. classification verif."] = updated_occupant_classificationv_df
    updated_dfs["OM - Occ. presence verif."] = updated_occupant_presence_df
    updated_dfs["VA - Speed assist. verif."] = updated_sas_verification_df
    updated_dfs["VA - ACC verif."] = updated_acc_verification_df
    updated_dfs["VA - Steering assistance verif."] = updated_steering_assistance_df
    if updated_dm_verification_df is not None:
        updated_dfs["DE - DM verif."] = updated_dm_verification_df
    scenario_scores_df["Score"] = pd.to_numeric(
        scenario_scores_df["Score"], errors="coerce"
    ).map(lambda v: round_half_up(v, 3))
    category_scores_df["Score"] = pd.to_numeric(
        category_scores_df["Score"], errors="coerce"
    ).map(lambda v: round_half_up(v, 3))
    test_scores_df["Score"] = pd.to_numeric(
        test_scores_df["Score"], errors="coerce"
    ).map(lambda v: round_half_up(v, 3))

    updated_dfs["Scenario Scores"] = scenario_scores_df
    updated_dfs["Category Scores"] = category_scores_df
    updated_dfs["Test Scores"] = test_scores_df

    return updated_dfs


def calculate_score(dfs: dict) -> dict:
    """Calculate safe driving NCAP scores from input dataframes.

    Args:
        dfs: Dictionary of dataframes from the input Excel file, including
             ``"DE - DM pred. (cell_colors)"`` produced by
             ``common.read_excel_file_to_dfs(..., load_bg_colors=True)``.

    Returns:
        dict: Updated dataframes with computed scores (no internal color keys).
    """
    updated_dfs = get_updated_dfs(dfs)
    result = {**dfs, **updated_dfs}
    result = {
        k: v
        for k, v in result.items()
        if not k.endswith("(bg)") and not k.endswith("(cell_colors)")
    }
    # "Documentation"/"Data" are carry-through-only (see crash_avoidance's
    # compute_score.calculate_score for the full rationale): drop them here
    # so report_writer.write_report's hard-copy from the preprocessed input
    # file is the sole source, instead of a pandas round-trip that could
    # coerce "TRUE"/"FALSE" strings to native bool once this domain gets its
    # own documentation_data.py logic.
    result = {
        k: v for k, v in result.items() if k not in common.DOCUMENTATION_DATA_SHEETS
    }
    return result


@click.command()
@with_footer
@click.option(
    "--input_file",
    "-i",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the input Excel file containing NCAP test measurements.",
)
@click.option(
    "--output_path",
    "-o",
    type=click.Path(file_okay=False, writable=True),
    default=os.getcwd(),
    show_default=True,
    help="Path to the output directory where the report will be saved.",
)
def compute_score(input_file, output_path):
    """Compute NCAP scores from an input Excel file."""
    common.check_version(input_file, common.CliCommand.PREPROCESS)
    print(f"[Safe Driving] Computing NCAP scores from input Excel file...")

    if not input_file:
        logger.error("Input file path is required.")
        sys.exit(1)
    if not input_file.endswith(".xlsx"):
        logger.error("Input file must be an Excel file with .xlsx extension.")
        sys.exit(1)

    # Verify the preprocessed file's protected reference cells (Max score,
    # row labels, ...) haven't been edited since 'preprocess' produced it.
    wb_to_verify = openpyxl.load_workbook(input_file, data_only=False)
    try:
        common.verify_protected_signature(wb_to_verify, integrity.SHEET_SCHEMAS)
    except common.TemplateIntegrityError as e:
        # Files preprocessed by an older version legitimately fail this
        # check (no Integrity sheet yet, or a signature computed over a
        # different schema revision), so it cannot be a hard gate.
        logger.warning(f"{e} Proceeding anyway for backward compatibility.")
        print(f"Warning: {e}")
        print(
            "Proceeding anyway for backward compatibility with files "
            "produced by older versions..."
        )
    finally:
        wb_to_verify.close()

    print("Loading data from spreadsheet...")

    dfs = common.read_excel_file_to_dfs(
        input_file,
        load_bg_colors=True,
        preserve_na_sheets=data_model.PRESERVE_NA_SHEETS,
    )
    logger.debug(f"Loaded sheets: {list(dfs.keys())}")
    print("Computing NCAP scores...")

    updated_dfs = calculate_score(dfs)

    report_writer.write_report(
        common.CliCommand.COMPUTE_SCORE,
        input_file,
        updated_dfs=updated_dfs,
        output_path=output_path,
        format_prediction_cells=False,
    )
