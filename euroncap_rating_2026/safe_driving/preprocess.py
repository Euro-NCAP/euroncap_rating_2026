# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import pandas as pd
import logging
import os
import sys

from euroncap_rating_2026 import config
from euroncap_rating_2026 import common
from euroncap_rating_2026.safe_driving import data_model
from euroncap_rating_2026.safe_driving import integrity
from euroncap_rating_2026.safe_driving import test_info
from euroncap_rating_2026.safe_driving import report_writer
import click

logger = logging.getLogger(__name__)
settings = config.Settings()


def add_verification_sheets_to_dfs(
    dfs: dict,
    acc_input_selected_points=None,
    dm_input_selected_points=None,
) -> tuple:
    """Run SD preprocess for ACC Performance and Driver Monitoring and return updated dfs.

    Args:
        dfs: Dictionary of DataFrames loaded from parquets or Excel.
        acc_input_selected_points: Optional fixed test-point dict for ACC Performance.
        dm_input_selected_points: Optional fixed test-point dict for Driver Monitoring.

    Returns:
        Tuple of (acc_test_points, dm_test_points, updated_dfs) where updated_dfs is a
        copy of *dfs* with the generated verification sheets merged in.
    """
    # Re-derive every protected reference cell (Max score, row labels, ...)
    # present in dfs from the official packaged template instead of trusting
    # it from the caller -- this is the dataframe-in/dataframe-out sibling
    # of preprocess()'s rebuild_trusted_input call, for callers that
    # only ever have DataFrames.
    dfs = integrity.rebuild_trusted_input_dfs(dfs)

    acc_test_points = test_info.preprocess_stage_subelement(
        dfs,
        "Vehicle Assistance",
        "ACC Performance",
        input_selected_points=acc_input_selected_points,
    )

    dm_test_points = test_info.preprocess_stage_subelement(
        dfs,
        "Driver Engagement",
        "Driver monitoring",
        input_selected_points=dm_input_selected_points,
    )

    result = dict(dfs)
    result["DE - DM verif."] = dfs.get("DE - DM verif.", pd.DataFrame())
    result["VA - ACC verif."] = dfs.get("VA - ACC verif.", pd.DataFrame())
    result = common.reset_computed_columns_to_dash(
        result, integrity.COMPUTED_SHEET_COLUMNS
    )

    return acc_test_points, dm_test_points, result


@click.command()
@common.with_footer
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
def preprocess(input_file, output_path):
    """Select test points and generate loadcases from input Excel file."""
    common.check_version(input_file, common.CliCommand.GENERATE_TEMPLATE)
    print(f"[Safe Driving] Preprocessing test points and generating loadcases...")

    # Re-derive a trusted copy of the input: every reference/structural cell
    # (Max score, row labels, ...) is sourced from the official packaged
    # template instead of the user-supplied file; only the designated grey
    # input cells are carried over from the user's file.
    try:
        sanitized_input_file = integrity.rebuild_trusted_input(input_file)
    except common.TemplateIntegrityError as e:
        print(f"Error: {e}")
        sys.exit(1)

    try:
        dfs = common.read_excel_file_to_dfs(
            sanitized_input_file,
            load_bg_colors=True,
            preserve_na_sheets=data_model.PRESERVE_NA_SHEETS,
        )
        logger.debug(f"Loaded sheets: {list(dfs.keys())}")

        selected_test_points = {}
        for stage_info in data_model.STAGE_SUBELEMENTS:
            current_stage_info = stage_info
            current_stage_element = current_stage_info["Stage element"]
            current_stage_subelement = current_stage_info["Stage subelement"]

            current_selected_test_points = test_info.preprocess_stage_subelement(
                dfs, current_stage_element, current_stage_subelement
            )
            selected_test_points[current_stage_subelement] = (
                current_selected_test_points
            )

        updated_dfs = {}
        updated_dfs.update(
            {
                "OM - Seatbelt usage verif.": dfs.get(
                    "OM - Seatbelt usage verif.", pd.DataFrame()
                )
            }
        )
        updated_dfs.update(
            {
                "OM - Occ. classification verif.": dfs.get(
                    "OM - Occ. classification verif.", pd.DataFrame()
                )
            }
        )
        updated_dfs.update(
            {
                "OM - Occ. presence verif.": dfs.get(
                    "OM - Occ. presence verif.", pd.DataFrame()
                )
            }
        )
        updated_dfs.update(
            {"DE - DM verif.": dfs.get("DE - DM verif.", pd.DataFrame())}
        )
        updated_dfs.update(
            {"DE - GVC verif.": dfs.get("DE - GVC verif.", pd.DataFrame())}
        )
        updated_dfs.update(
            {
                "VA - Speed assist. verif.": dfs.get(
                    "VA - Speed assist. verif.", pd.DataFrame()
                )
            }
        )
        updated_dfs.update(
            {"VA - ACC verif.": dfs.get("VA - ACC verif.", pd.DataFrame())}
        )
        updated_dfs.update(
            {
                "VA - Steering assistance verif.": dfs.get(
                    "VA - Steering assistance verif.", pd.DataFrame()
                )
            }
        )
        updated_dfs.update(
            {"Scenario Scores": dfs.get("Scenario Scores", pd.DataFrame())}
        )

        report_writer.write_report(
            common.CliCommand.PREPROCESS,
            sanitized_input_file,
            updated_dfs=updated_dfs,
            output_path=output_path,
            format_prediction_cells=True,
            dm_prediction_forced_red=dfs.get(
                "DE - DM pred. (forced_red)", pd.DataFrame()
            ),
        )
    finally:
        os.remove(sanitized_input_file)
