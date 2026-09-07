# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import logging
import os
import sys

from euroncap_rating_2026 import config
from euroncap_rating_2026 import common
from euroncap_rating_2026.post_crash import data_model
from euroncap_rating_2026.post_crash import integrity
from euroncap_rating_2026.post_crash import test_info
from euroncap_rating_2026.post_crash import report_writer
import click

logger = logging.getLogger(__name__)
settings = config.Settings()

VERIFICATION_SHEETS = [
    "RI - RS Verification",
    "RI - ERG Verification",
    "PCI - Advanced eCall Verif",
    "PCI - MCB & Hazard lights Verif",
    "VE - Energy Management Verif",
    "VE - Occupant Extrication Verif",
]


def add_verification_sheets_to_dfs(dfs: dict) -> dict:
    """DataFrame-native sibling of preprocess() for callers that
    only ever have DataFrames: rebuild trusted cells, generate the 6 Verif
    sheets, dash the not-yet-computed Score columns.

    Returns a new dict; the input *dfs* is not mutated.
    """
    dfs = dict(integrity.rebuild_trusted_input_dfs(dfs))
    for stage_info in data_model.STAGE_SUBELEMENTS:
        test_info.preprocess_stage_subelement(
            dfs, stage_info["Stage element"], stage_info["Stage subelement"]
        )
    return common.reset_computed_columns_to_dash(dfs, integrity.COMPUTED_SHEET_COLUMNS)


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
    print(f"[Post-Crash] Preprocessing and generating loadcases...")

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
        dfs = common.read_excel_file_to_dfs(sanitized_input_file)
        logger.debug(f"Loaded sheets: {list(dfs.keys())}")

        # rebuild_trusted_input_dfs on the already-sanitized dfs is
        # idempotent (and its pristine template load is lru-cached). Only
        # the generated Verif sheets are written from dfs; the score
        # sheets stay hard-copied with their template formatting and get
        # their dash from write_report's blank_computed_columns call.
        result = add_verification_sheets_to_dfs(dfs)
        updated_dfs = {
            name: result[name] for name in VERIFICATION_SHEETS if name in result
        }

        report_writer.write_report(
            common.CliCommand.PREPROCESS,
            sanitized_input_file,
            updated_dfs,
            selected_points_dict={},
            output_path=output_path,
            format_prediction_cells=False,
        )
    finally:
        os.remove(sanitized_input_file)
