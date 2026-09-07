# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import pandas as pd
import logging
import openpyxl

from euroncap_rating_2026.common import with_footer
from euroncap_rating_2026 import config
from euroncap_rating_2026 import common
from euroncap_rating_2026.numeric import round_half_up
from euroncap_rating_2026.post_crash import (
    report_writer,
)
from euroncap_rating_2026.post_crash import integrity
from euroncap_rating_2026.post_crash import rescue_info
from euroncap_rating_2026.post_crash import advanced_ecall
from euroncap_rating_2026.post_crash import mcb_hazard_lights
from euroncap_rating_2026.post_crash import energy_management
from euroncap_rating_2026.post_crash import occupant_extrication

import sys
import os
from datetime import datetime

import click


logger = logging.getLogger(__name__)
settings = config.Settings()


# Copy specific sheets to the output file

COMPUTE_SHEETS_TO_COPY = [
    "Version",
    "Input parameters",
    "RI - RS Verification",
    "RI - ERG Verification",
    "PCI - Advanced eCall Verif",
    "PCI - MCB & Hazard lights Verif",
    "VE - Energy Management Verif",
    "VE - Occupant Extrication Verif",
]


def get_output_file_path(output_path: str) -> str:
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(output_path, f"pc_{current_datetime}_report.xlsx")
    return output_file


def cap_category_scores(
    category_scores_df: pd.DataFrame, categories_to_cap: dict
) -> pd.DataFrame:
    # Define the categories to cap and their maximum scores

    for category, max_score in categories_to_cap.items():
        if category in category_scores_df["Category"].values:
            category_scores_df.loc[
                category_scores_df["Category"] == category, "Score"
            ] = category_scores_df.loc[
                category_scores_df["Category"] == category, "Score"
            ].clip(
                upper=max_score
            )

    return category_scores_df


def resolve_capped_unassessed_categories(
    category_scores_df: pd.DataFrame,
    scenario_scores_df: pd.DataFrame,
    categories_to_cap: dict,
) -> pd.DataFrame:
    """Emit a capped category's score even when some of its scenarios are
    unassessed, as long as the blanks cannot change the capped result.

    update_score_sum_interval rolls a category up as unassessed (NaN) when any
    scenario in it is NaN. For the capped categories that is over-strict: the
    protocol scores them as "cumulative up to a maximum of N points", where
    some scenarios are alternative routes that legitimately stay unassessed
    (Post-Crash protocol 3.2.6: Submergence is assessed with *either* electric
    window opening *or* the rescue tool -- a PASS on window opening alone is a
    complete assessment). Whenever min(cap, assessed) == min(cap, assessed +
    max of the blanks), every possible outcome of the blank rows yields the
    same capped score, so that score is emitted; otherwise the category stays
    unassessed.
    """
    required_columns = {"Category", "Score", "Max score"}
    if not required_columns.issubset(scenario_scores_df.columns):
        return category_scores_df
    for category, cap in categories_to_cap.items():
        category_mask = category_scores_df["Category"] == category
        if not category_mask.any():
            continue
        if pd.notna(category_scores_df.loc[category_mask, "Score"].iloc[0]):
            # Fully assessed: update_score_sum_interval + cap already produced
            # the exact same min(cap, sum), nothing to resolve.
            continue
        # Same Category-interval walk as update_score_sum_interval: the
        # section runs from the labeled row until the next non-NaN Category.
        scenario_mask = scenario_scores_df["Category"] == category
        if not scenario_mask.any():
            continue
        start_idx = scenario_mask.idxmax()
        next_idxs = scenario_scores_df.index[
            (scenario_scores_df.index > start_idx)
            & (scenario_scores_df["Category"].notna())
        ]
        end_idx = (
            next_idxs[0] if len(next_idxs) > 0 else scenario_scores_df.index[-1] + 1
        )
        section = scenario_scores_df.loc[start_idx : end_idx - 1]
        scores = pd.to_numeric(section["Score"], errors="coerce")
        max_scores = pd.to_numeric(section["Max score"], errors="coerce")
        if max_scores[scores.isna()].isna().any():
            # A blank scenario without a usable Max score has unbounded
            # potential -- the outcome can't be pinned down.
            continue
        assessed_sum = scores.fillna(0.0).sum()
        blank_potential = max_scores[scores.isna()].sum()
        if min(cap, assessed_sum) == min(cap, assessed_sum + blank_potential):
            category_scores_df.loc[
                category_scores_df.index[category_mask][0], "Score"
            ] = min(cap, assessed_sum)
    return category_scores_df


def calculate_score(dfs: dict) -> dict:
    """Calculate post-crash NCAP scores from input dataframes.

    Args:
        dfs: Dictionary of dataframes from the input Excel file.

    Returns:
        dict: Updated dataframes with computed scores.
    """
    # Re-derive every protected reference cell (Max score, row labels, ...)
    # from the official packaged template instead of trusting it from the
    # caller's dfs -- callers that only ever pass DataFrames (no file, no
    # preprocess() call) never went through rebuild_trusted_input, so this
    # must happen unconditionally here rather than being assumed done
    # upstream. Only the designated grey input cells are carried over.
    dfs = integrity.rebuild_trusted_input_dfs(dfs)

    scenario_scores_df = dfs.get("Scenario Scores", pd.DataFrame())
    category_scores_df = dfs.get("Category Scores", pd.DataFrame())
    test_scores_df = dfs.get("Test Scores", pd.DataFrame())

    rescue_score_dict = rescue_info.compute_score(dfs)
    ecall_score_dict = advanced_ecall.compute_score(dfs)
    mcb_score_dict = mcb_hazard_lights.compute_score(dfs)
    energy_management_score_dict = energy_management.compute_score(dfs)
    occupant_extrication_score_dict = occupant_extrication.compute_score(dfs)
    logger.info(f"Rescue Score: {rescue_score_dict}")
    logger.info(f"eCall Score: {ecall_score_dict}")
    logger.info(f"MCB Score: {mcb_score_dict}")
    logger.info(f"Energy Management Score: {energy_management_score_dict}")
    logger.info(f"Occupant Extrication Score: {occupant_extrication_score_dict}")

    scenario_scores_df = common.update_scoring_sheet(
        scenario_scores_df, ecall_score_dict, key_column="Scenario"
    )
    scenario_scores_df = common.update_scoring_sheet(
        scenario_scores_df, energy_management_score_dict, key_column="Scenario"
    )
    scenario_scores_df = common.update_scoring_sheet(
        scenario_scores_df, occupant_extrication_score_dict, key_column="Scenario"
    )

    category_scores_df = common.update_scoring_sheet(
        category_scores_df, mcb_score_dict, key_column="Category"
    )
    test_scores_df = common.update_scoring_sheet(
        test_scores_df, rescue_score_dict, key_column="Stage subelement"
    )

    category_scores_df = common.update_score_sum_interval(
        category_scores_df,
        scenario_scores_df,
        key_column="Category",
        interval_column="Category",
    )

    categories_to_cap = {
        "Advanced eCall - TPS": 15.0,
        "Thermal propagation": 9.0,
        "Submergence": 3.0,
    }
    category_scores_df = cap_category_scores(category_scores_df, categories_to_cap)
    category_scores_df = resolve_capped_unassessed_categories(
        category_scores_df, scenario_scores_df, categories_to_cap
    )

    test_scores_df = common.update_score_sum_interval(
        test_scores_df,
        category_scores_df,
        key_column="Stage subelement",
        interval_column="Stage subelement",
    )

    ecall_subelement = "Advanced eCall"
    max_score = 20.0
    test_scores_df.loc[
        test_scores_df["Stage subelement"] == ecall_subelement, "Score"
    ] = test_scores_df.loc[
        test_scores_df["Stage subelement"] == ecall_subelement, "Score"
    ].clip(
        upper=max_score
    )

    scenario_scores_df["Score"] = scenario_scores_df["Score"].map(
        lambda v: round_half_up(v, 3)
    )
    category_scores_df["Score"] = category_scores_df["Score"].map(
        lambda v: round_half_up(v, 3)
    )
    test_scores_df["Score"] = test_scores_df["Score"].map(lambda v: round_half_up(v, 3))

    return {
        "Scenario Scores": scenario_scores_df,
        "Category Scores": category_scores_df,
        "Test Scores": test_scores_df,
    }


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
    print(f"[Post-Crash] Computing NCAP scores from input Excel file...")

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

    dfs = common.read_excel_file_to_dfs(input_file)
    logger.debug(f"Loaded sheets: {list(dfs.keys())}")

    print("Computing NCAP scores...")

    updated_dfs = calculate_score(dfs)

    report_writer.write_report(
        common.CliCommand.COMPUTE_SCORE,
        input_file,
        updated_dfs,
        selected_points_dict={},
        output_path=output_path,
        format_prediction_cells=False,
    )
