# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import pandas as pd
import logging
import os
import io
import sys
import numpy as np
from euroncap_rating_2026.crash_avoidance import matrix_processing
from euroncap_rating_2026 import config
from euroncap_rating_2026 import common
import click

from euroncap_rating_2026.crash_avoidance.test_info import (
    preprocess_stage_subelement,
    get_sheet_prefix,
)
from euroncap_rating_2026.crash_avoidance import data_model
from euroncap_rating_2026.crash_avoidance import documentation_data
from euroncap_rating_2026.crash_avoidance import integrity
from euroncap_rating_2026.crash_avoidance import report_writer

logger = logging.getLogger(__name__)
settings = config.Settings()


def drop_empty_optional_columns(verification_test_dfs: dict) -> dict:
    """Drop columns that are entirely empty (NaN or '') after default-value normalization."""
    optional_cols = ["Robustness"]
    result = {}
    for sheet_name, df in verification_test_dfs.items():
        for col in optional_cols:
            if col in df.columns and df[col].replace("", float("nan")).isna().all():
                df = df.drop(columns=[col])
        result[sheet_name] = df
    return result


def set_relevant_columns_to_default_values(verification_test_dfs: dict) -> dict:
    normalized_dfs = {}

    for sheet_name, df in verification_test_dfs.items():
        df = df.copy()
        df.columns = pd.Index([str(col).strip() for col in df.columns])

        if "Function" in df.columns:
            df["Function"] = df["Function"].fillna("AEB")

        if "Day/Night" in df.columns:
            df["Day/Night"] = df["Day/Night"].fillna("Day")
            df.rename(columns={"Day/Night": "Illumination"}, inplace=True)

        if "Colour" not in df.columns:
            df["Colour"] = ""

        if "Robustness" not in df.columns:
            df["Robustness"] = ""

        ordered_columns = [
            col for col in common.VERIFICATION_COLUMN_ORDER if col in df.columns
        ]
        unordered_columns = [
            col for col in df.columns if col not in set(ordered_columns)
        ]
        df = df[ordered_columns + unordered_columns]

        normalized_dfs[sheet_name] = df

    return normalized_dfs


def add_general_requirements_rows(verification_test_dfs: dict) -> dict:
    updated_dfs = {}

    for sheet_name, df in verification_test_dfs.items():
        df = df.copy()
        columns = [str(col).strip() for col in df.columns]
        if not columns:
            columns = ["Scenario", "Value"]
        else:
            if "Scenario" not in columns:
                columns.insert(0, "Scenario")
            if "Value" not in columns:
                columns.append("Value")
        df = df.reindex(columns=columns, fill_value="")
        value_col_idx = df.columns.get_loc("Value")

        general_requirements_columns = [""] * df.shape[1]
        general_requirements_columns[0] = "Scenario"
        general_requirements_columns[value_col_idx] = "Value"
        # Value stays blank ("" / not yet assessed), not "PASS" -- a fresh
        # sheet must not silently score as if the OEM had already passed
        # this check.
        general_requirements_row = [""] * df.shape[1]
        general_requirements_row[0] = "General requirements"

        if "LDC - Single Veh" in sheet_name:
            driveability_row = [""] * df.shape[1]
            driveability_row[0] = "Driveability"

            driver_state_row = [""] * df.shape[1]
            driver_state_row[0] = "Driver state link"
        else:
            driveability_row = None
            driver_state_row = None

        row_list = [general_requirements_row]
        if driveability_row is not None:
            row_list.append(driveability_row)
        if driver_state_row is not None:
            row_list.append(driver_state_row)

        general_requirements_df = pd.DataFrame(
            row_list,
            columns=general_requirements_columns,
        )

        empty_row = pd.DataFrame(
            [[""] * df.shape[1]], columns=general_requirements_columns
        )

        updated_dfs[sheet_name] = pd.DataFrame(
            np.vstack(
                [
                    general_requirements_df.values,
                    empty_row.values,
                    np.array([df.columns.tolist()]),
                    df.values,
                ]
            ),
            columns=general_requirements_columns,
        )

    return updated_dfs


def add_verification_sheets_to_dfs(dfs: dict, input_selected_points=None) -> dict:
    """Return a copy of *dfs* with CA verification sheets added.

    Runs ``preprocess_stage_subelement`` for every stage/subelement pair in
    ``STAGE_SUBELEMENTS``, applies column reordering, default-value normalisation,
    and general-requirements rows, then merges the results into *dfs*.

    Args:
        dfs: Dictionary of DataFrames (typically loaded from parquets).  Must
             contain every prediction sheet required by each stage subelement.
        input_selected_points: Optional dict of manually chosen test points
            forwarded to ``preprocess_stage_subelement``.  ``None`` triggers
            automatic (random) selection.

    Returns:
        New dict equal to *dfs* plus the ``"{prefix} verif."`` sheets.
    """
    # Re-derive every protected reference cell (Max score, row labels, ...)
    # present in dfs from the official packaged template instead of trusting
    # it from the caller -- this is the dataframe-in/dataframe-out sibling
    # of preprocess()'s rebuild_trusted_input call, for callers that
    # only ever have DataFrames.
    dfs = integrity.rebuild_trusted_input_dfs(dfs)

    verification_test_dfs = {}

    # Normalise robustness-sheet column name from parquet casing to code-expected casing.
    dfs = dict(dfs)
    for key in list(dfs):
        if key.endswith("robust. pred."):
            dfs[key] = dfs[key].rename(columns={"Robustness Layer": "Robustness layer"})

    stage_subelements_result = {}

    for stage_info in data_model.STAGE_SUBELEMENTS:
        stage_element = stage_info["Stage element"]
        stage_subelement = stage_info["Stage subelement"]
        sheet_prefix = get_sheet_prefix(stage_element, stage_subelement)
        prediction_sheet_name = f"{sheet_prefix} pred."
        if prediction_sheet_name not in dfs:
            logger.warning(
                "Skipping preprocess for %s - %s because sheet '%s' is missing. "
                "The scorer will treat the missing verification sheet as zero.",
                stage_element,
                stage_subelement,
                prediction_sheet_name,
            )
            continue

        stage_subelement_result = preprocess_stage_subelement(
            dfs,
            stage_element,
            stage_subelement,
            input_selected_points=input_selected_points,
        )

        stage_subelements_result[(stage_element, stage_subelement)] = (
            stage_subelement_result
        )
        verification_test_dfs[f"{sheet_prefix} verif."] = (
            stage_subelement_result.test_points_df
        )

    for sheet_name, df in verification_test_dfs.items():
        verification_test_dfs[sheet_name] = common.reorder_columns_by_header_order(
            df, common.VERIFICATION_COLUMN_ORDER
        )

    verification_test_dfs = set_relevant_columns_to_default_values(
        verification_test_dfs
    )
    verification_test_dfs = drop_empty_optional_columns(verification_test_dfs)
    verification_test_dfs = add_general_requirements_rows(verification_test_dfs)

    result = dict(dfs)
    result.update(verification_test_dfs)
    result["Documentation"] = documentation_data.compute_documentation(dfs)
    result["Data"] = documentation_data.compute_data(dfs)
    result = common.reset_computed_columns_to_dash(
        result, integrity.COMPUTED_SHEET_COLUMNS
    )
    return stage_subelements_result, result


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
    print(f"[Crash Avoidance] Preprocessing test points and generating loadcases...")

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
            sanitized_input_file, preserve_na_sheets=data_model.PRESERVE_NA_SHEETS
        )
        logger.debug(f"Loaded sheets: {list(dfs.keys())}")

        verification_test_dfs = {}
        rows_to_delete = {}

        stage_subelements = []
        for stage_info in data_model.STAGE_SUBELEMENTS:
            current_stage_info = stage_info
            current_stage_element = current_stage_info["Stage element"]
            current_stage_subelement = current_stage_info["Stage subelement"]

            try:
                stage_subelement = preprocess_stage_subelement(
                    dfs, current_stage_element, current_stage_subelement
                )
            except ValueError as e:
                print(f"Error: {e}")
                sys.exit(1)
            sheet_prefix = get_sheet_prefix(
                current_stage_element, current_stage_subelement
            )
            verification_test_dfs[f"{sheet_prefix} verif."] = (
                stage_subelement.test_points_df
            )
            stage_subelements.append(stage_subelement)
            for sheet_name, sheet_rows in (
                stage_subelement.removed_prediction_rows or {}
            ).items():
                rows_to_delete.setdefault(sheet_name, []).extend(sheet_rows)

        for sheet_name, df in verification_test_dfs.items():
            verification_test_dfs[sheet_name] = common.reorder_columns_by_header_order(
                df, common.VERIFICATION_COLUMN_ORDER
            )

        verification_test_dfs = set_relevant_columns_to_default_values(
            verification_test_dfs
        )
        verification_test_dfs = drop_empty_optional_columns(verification_test_dfs)
        verification_test_dfs = add_general_requirements_rows(verification_test_dfs)
        dfs = {**dfs, **verification_test_dfs}
        dfs["Documentation"] = documentation_data.compute_documentation(dfs)
        dfs["Data"] = documentation_data.compute_data(dfs)
        dfs = common.reset_computed_columns_to_dash(
            dfs, integrity.COMPUTED_SHEET_COLUMNS
        )
        report_writer.write_report(
            common.CliCommand.PREPROCESS,
            sanitized_input_file,
            updated_dfs=dfs,
            output_path=output_path,
            format_prediction_cells=True,
            rows_to_delete=rows_to_delete,
        )
    finally:
        os.remove(sanitized_input_file)
