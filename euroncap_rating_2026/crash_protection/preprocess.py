# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import pandas as pd
import logging
import os
import sys

from euroncap_rating_2026.crash_protection import vru_processing
from euroncap_rating_2026.crash_protection import data_loader
from euroncap_rating_2026.crash_protection import integrity
from euroncap_rating_2026 import common
from euroncap_rating_2026 import config
from euroncap_rating_2026.crash_protection.report_writer import write_report

import click

logger = logging.getLogger(__name__)
settings = config.Settings()


def generate_headform_verification_df(headform_test_points):
    headform_loadcases = data_loader.generate_vru_head_impact_loadcases(
        headform_test_points
    )
    headform_df = vru_processing.generate_vru_head_impact_df(headform_loadcases)
    return headform_df


def generate_legform_verification_df(legform_test_points):
    legform_loadcases = data_loader.generate_vru_legform_loadcases(legform_test_points)
    legform_df = vru_processing.generate_vru_pelvis_leg_impact_df(legform_loadcases)
    return legform_df


def add_vru_sheets_to_dfs(dfs: dict, input_selected_points=None) -> dict:
    """Return a copy of *dfs* with the three VRU sheets added.

    Adds (or overwrites):
      * ``"CP - VRU Prediction Points"``  – all prediction-matrix cells derived
        from ``dfs["CP - VRU Prediction"]`` cell values (color text), so this
        works even when the source Excel uses conditional formatting for colors.
      * ``"CP - VRU Head Impact"``         – headform verification sheet.
      * ``"CP - VRU Pelvis & Leg Impact"`` – legform verification sheet.

    The original *dfs* dict is not mutated.

    Args:
        dfs: Dictionary of DataFrames.  Must contain
             ``"CP - VRU Prediction"`` and ``"Input parameters"``.
        input_selected_points: Optional manual-selection dict forwarded to
            ``data_loader.select_vru_test_points``.  ``None`` triggers
            automatic selection.

    Returns:
        Tuple containing headform test points, legform test points, and the updated dfs dictionary.
    """

    # Re-derive every protected reference cell (HPL, LPL, Capping, Max score,
    # row labels, ...) present in dfs from the official packaged template
    # instead of trusting it from the caller -- this is the dataframe-in/
    # dataframe-out sibling of preprocess()'s rebuild_trusted_input call, for
    # callers that only ever have DataFrames.
    dfs = integrity.rebuild_trusted_input_dfs(dfs)

    vru_prediction_df = dfs.get("CP - VRU Prediction", pd.DataFrame())
    input_parameters_df = dfs.get("Input parameters", pd.DataFrame())

    vru_points_all = vru_processing.compute_all_vru_test_points(vru_prediction_df)
    vru_prediction_points_df = vru_processing.vru_points_to_df(vru_points_all)

    headform_test_point_df, legform_test_point_df = data_loader.select_vru_test_points(
        vru_prediction_df,
        input_parameters_df,
        input_selected_points=input_selected_points,
    )

    headform_test_points = vru_processing.df_to_vru_points(headform_test_point_df)
    legform_test_points = vru_processing.df_to_vru_points(
        legform_test_point_df, legform=True
    )

    result = dict(dfs)
    result["CP - VRU Prediction Points"] = vru_prediction_points_df
    result["CP - VRU Head Impact"] = generate_headform_verification_df(
        headform_test_points
    )
    result["CP - VRU Pelvis & Leg Impact"] = generate_legform_verification_df(
        legform_test_points
    )
    result = common.reset_computed_columns_to_dash(
        result, integrity.COMPUTED_SHEET_COLUMNS
    )
    return headform_test_points, legform_test_points, result


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
    """Preprocess VRU test points and generate loadcases from input Excel file."""
    common.check_version(input_file, common.CliCommand.GENERATE_TEMPLATE)
    print(
        f"[Crash Protection] Preprocessing VRU test points and generating loadcases..."
    )

    # Re-derive a trusted copy of the input: every reference/structural cell
    # (HPL, LPL, Capping, Max score, row labels, ...) is sourced from the
    # official packaged template instead of the user-supplied file; only the
    # designated grey input cells are carried over from the user's file.
    try:
        sanitized_input_file = integrity.rebuild_trusted_input(input_file)
    except common.TemplateIntegrityError as e:
        print(f"Error: {e}")
        sys.exit(1)

    try:
        # Generate test point i,j
        vru_test_data = data_loader.generate_vru_test_points(sanitized_input_file)
        ########################################################################

        headform_test_points = vru_test_data.headform_test_points
        legform_test_points = vru_test_data.legform_test_points

        # Generate loadcases from test points and export to Excel
        headform_loadcases = data_loader.generate_vru_head_impact_loadcases(
            headform_test_points
        )
        legform_loadcases = data_loader.generate_vru_legform_loadcases(
            legform_test_points
        )
        vru_test_data.headform_loadcases = headform_loadcases
        vru_test_data.legform_loadcases = legform_loadcases

        vru_test_data.loadcase_dict["CP - VRU Head Impact"] = headform_loadcases
        vru_test_data.loadcase_dict["CP - VRU Pelvis & Leg Impact"] = legform_loadcases

        vru_test_data.df_dict["CP - VRU Head Impact"] = (
            vru_processing.generate_vru_head_impact_df(headform_loadcases)
        )
        vru_test_data.df_dict["CP - VRU Pelvis & Leg Impact"] = (
            vru_processing.generate_vru_pelvis_leg_impact_df(legform_loadcases)
        )

        vru_test_data.loadcase_dict["CP - VRU Prediction"] = vru_test_data.prediction_df
        vru_test_data.df_dict["CP - VRU Prediction"] = vru_test_data.prediction_df

        vru_processing.pretty_print_loadcases(vru_test_data.headform_loadcases)
        vru_processing.pretty_print_loadcases(vru_test_data.legform_loadcases)

        score_sheets = common.reset_computed_columns_to_dash(
            common.read_excel_file_to_dfs(sanitized_input_file),
            integrity.COMPUTED_SHEET_COLUMNS,
        )
        vru_test_data.df_dict.update(
            {
                sheet_name: score_sheets[sheet_name]
                for sheet_name in integrity.COMPUTED_SHEET_COLUMNS
                if sheet_name in score_sheets
            }
        )

        write_report(
            common.CliCommand.PREPROCESS,
            sanitized_input_file,
            updated_dfs=vru_test_data.df_dict,
            output_path=output_path,
            format_prediction_cells=True,
        )
    finally:
        os.remove(sanitized_input_file)
