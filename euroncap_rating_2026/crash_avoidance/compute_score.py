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
from euroncap_rating_2026.crash_avoidance.test_info import (
    compute_stage_score,
    get_stage_subelement_key,
    apply_cross_scenario_robustness_rule,
)
from euroncap_rating_2026.crash_avoidance.data_model import (
    PRESERVE_NA_SHEETS,
    STAGE_SUBELEMENTS,
    STAGE_SUBELEMENT_TO_CATEGORIES,
)
import euroncap_rating_2026.crash_avoidance.matrix_processing as matrix_processing
from euroncap_rating_2026.crash_avoidance import integrity
from euroncap_rating_2026.crash_avoidance import report_writer

import sys
import os
import click


logger = logging.getLogger(__name__)
settings = config.Settings()


# Copy specific sheets to the output file


def _normalize_stage_subelement_name(value):
    if value == "Pedestrian & cyclist":
        return "Ped & Cyc"
    if value == "Single vehicle":
        return "Single Veh"
    return value


def _prepare_effective_columns(
    df: pd.DataFrame, columns: list[str]
) -> dict[str, pd.Series]:
    return {
        col: (
            df[col].ffill()
            if col in df.columns
            else pd.Series(index=df.index, dtype=object)
        )
        for col in columns
    }


def get_element_for_row(df, col_name, row_idx):
    """
    Given a DataFrame, column name, and row index, return the value of the column for the interval containing the row.
    Assumes the column is set at the start of an interval and NaN until the next.
    """
    col_elements = df[col_name]
    # Find all indices where col_name is notna and less than or equal to row_idx
    valid_indices = col_elements[col_elements.notna()].index
    prev_indices = valid_indices[valid_indices <= row_idx]
    if prev_indices.empty:
        return None
    last_idx = prev_indices[-1]
    return col_elements.loc[last_idx]


def check_general_requirements_rules(dfs, loadcase_score_dict, scenario_scores_df):
    """
    Applies the Frontal Collision general requirements rule.

    FAIL condition (any one is sufficient):
    - FC - Car & PTW General requirements == FAIL
    - FC - Ped & Cyc General requirements == FAIL
    - Prediction CPNA day  @ Impact location 75% / speed 10 km/h != Green
    - Prediction CPNA night @ Impact location 75% / speed 10 km/h != Green
    - Any CCRs standard-range test point at speed <= 20 km/h != Green

    When FAIL: all Frontal Collisions scores are zeroed out.
    """

    def _get_gen_req_value(verification_sheet_name):
        """Return lowercased 'Value' for the 'General requirements' row, or None."""
        df = dfs.get(verification_sheet_name)
        if df is None or "Scenario" not in df.columns or "Value" not in df.columns:
            logger.warning(
                f"Sheet '{verification_sheet_name}' not available for general requirements check."
            )
            return None
        matches = df[df["Scenario"] == "General requirements"]
        if matches.empty:
            logger.warning(
                f"'General requirements' row not found in '{verification_sheet_name}'."
            )
            return None
        return str(matches.iloc[0]["Value"]).strip().lower()

    def _get_colors_at_cells(prediction_df, test_name, cells):
        """
        Extract prediction colors at specific cells within a test matrix.

        Uses MATRIX_INDICES to locate the sub-table in the prediction sheet,
        then reads the color at each (row, col) position (0-based within the
        test matrix).

        Args:
            prediction_df: DataFrame returned by load_verification_sheet_with_bg_colors.
            test_name: Key in MATRIX_INDICES (e.g. "CPNA day", "CCRs").
            cells: List of (row, col) tuples, 0-based within the test matrix.

        Returns:
            List of PredictionColor values (None for out-of-bounds cells).
        """
        matrix_indices = matrix_processing.get_matrix_indices(prediction_df, test_name)
        if not matrix_indices:
            logger.warning(f"Test name '{test_name}' not found in MATRIX_INDICES.")
            return []
        # Prediction sheets are read with Excel row 1 consumed as DataFrame headers.
        # Top matrices already align at row index 1, while later blocks are shifted
        # up by one DataFrame row relative to the template coordinates.
        start_row = matrix_indices["start_row"]
        if start_row > 1:
            start_row -= 1
        start_col = matrix_indices["start_col"]
        logger.debug(
            f"Extracting colors for test '{test_name}' from prediction DataFrame "
            f"starting at (row={start_row}, col={start_col}) for cells {cells}"
        )
        logger.debug(
            [prediction_df.iloc[start_row + row, start_col + col] for row, col in cells]
        )
        colors = []
        for row, col in cells:
            try:
                color = prediction_df.iloc[start_row + row, start_col + col]
                colors.append(color)
            except IndexError:
                logger.warning(
                    f"Cell ({row}, {col}) out of bounds for test '{test_name}'."
                )
                colors.append(None)
        return colors

    # --- Evaluate each condition ---

    fc_car_ptw_gr = _get_gen_req_value("FC - Car & PTW verif.")
    fc_ped_cyc_gr = _get_gen_req_value("FC - Ped & Cyc verif.")

    fc_ped_cyc_pred_df = dfs["FC - Ped & Cyc pred. (bg)"]
    fc_car_ptw_pred_df = dfs["FC - Car & PTW pred. (bg)"]

    # CPNA day: matrix is 6 rows x 5 cols (MATRIX_INDICES start_row=72, start_col=5)
    # Set (row, col) to the cell for 75% impact location / 10 km/h within that sub-table.
    cpna_day_cells = [(0, 3)]
    cpna_day_colors = _get_colors_at_cells(
        fc_ped_cyc_pred_df, "CPNA day", cpna_day_cells
    )

    # CPNA night: same matrix shape (MATRIX_INDICES start_row=81, start_col=5)
    cpna_night_cells = [(0, 3)]
    cpna_night_colors = _get_colors_at_cells(
        fc_ped_cyc_pred_df, "CPNA night", cpna_night_cells
    )

    # CCRs: matrix is 8 rows x 7 cols (MATRIX_INDICES start_row=1, start_col=3)
    # List all standard-range cells whose column corresponds to speed <= 20 km/h.
    ccrs_leq20_cells = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
    ]
    ccrs_leq20_colors = _get_colors_at_cells(
        fc_car_ptw_pred_df, "CCRs", ccrs_leq20_cells
    )

    fail_fc_car_ptw = fc_car_ptw_gr == "fail"
    fail_fc_ped_cyc = fc_ped_cyc_gr == "fail"
    fail_cpna_day = any(
        c is not None and c != common.PredictionColor.GREEN for c in cpna_day_colors
    )
    fail_cpna_night = any(
        c is not None and c != common.PredictionColor.GREEN for c in cpna_night_colors
    )
    fail_ccrs = any(
        c is not None and c != common.PredictionColor.GREEN for c in ccrs_leq20_colors
    )

    logger.info(
        f"Frontal Collision general requirements check: "
        f"FC Car&PTW GR='{fc_car_ptw_gr}', FC Ped&Cyc GR='{fc_ped_cyc_gr}', "
        f"CPNA day colors={cpna_day_colors}, "
        f"CPNA night colors={cpna_night_colors}, "
        f"CCRs <=20km/h colors={ccrs_leq20_colors}"
    )

    frontal_collision_fail = (
        fail_fc_car_ptw
        or fail_fc_ped_cyc
        or fail_cpna_day
        or fail_cpna_night
        or fail_ccrs
    )

    effective_cols = _prepare_effective_columns(scenario_scores_df, ["Stage element"])
    scenario_stage_element = effective_cols["Stage element"]

    if frontal_collision_fail:
        reasons = []
        if fail_fc_car_ptw:
            reasons.append(f"FC Car&PTW General requirements='{fc_car_ptw_gr}'")
        if fail_fc_ped_cyc:
            reasons.append(f"FC Ped&Cyc General requirements='{fc_ped_cyc_gr}'")
        if fail_cpna_day:
            reasons.append(f"CPNA day cells={cpna_day_cells} colors={cpna_day_colors}")
        if fail_cpna_night:
            reasons.append(
                f"CPNA night cells={cpna_night_cells} colors={cpna_night_colors}"
            )
        if fail_ccrs:
            reasons.append(
                f"CCRs <=20km/h cells={ccrs_leq20_cells} colors={ccrs_leq20_colors}"
            )
        logger.warning(f"[!!!] Frontal Collision FAIL. Reasons: {', '.join(reasons)}")

        # Zero out all Frontal Collisions scores in loadcase_score_dict
        fc_stage = "Frontal Collisions"
        if fc_stage in loadcase_score_dict:
            for sub_dict in loadcase_score_dict[fc_stage].values():
                for lcs in sub_dict.values():
                    lcs.standard_score = 0.0
                    lcs.extended_score = 0.0
                    lcs.robustness_layer_score = 0.0
                    lcs.total_score = 0.0

        # Zero out corresponding rows in scenario_scores_df
        fc_mask = scenario_stage_element == fc_stage
        scenario_scores_df.loc[fc_mask, "Score"] = 0.0

    # --- LDC rule ---
    ldc_single_veh_gr = _get_gen_req_value("LDC - Single Veh verif.")
    ldc_car_ptw_gr = _get_gen_req_value("LDC - Car & PTW verif.")

    fail_ldc_single_veh = ldc_single_veh_gr == "fail"
    fail_ldc_car_ptw = ldc_car_ptw_gr == "fail"

    logger.info(
        f"LDC general requirements check: "
        f"LDC Single Veh GR='{ldc_single_veh_gr}', LDC Car&PTW GR='{ldc_car_ptw_gr}'"
    )

    ldc_fail = fail_ldc_single_veh or fail_ldc_car_ptw

    if ldc_fail:
        ldc_reasons = []
        if fail_ldc_single_veh:
            ldc_reasons.append(
                f"LDC Single Veh General requirements='{ldc_single_veh_gr}'"
            )
        if fail_ldc_car_ptw:
            ldc_reasons.append(f"LDC Car&PTW General requirements='{ldc_car_ptw_gr}'")
        logger.warning(
            f"[!!!] Lane Departure Collisions FAIL. Reasons: {', '.join(ldc_reasons)}"
        )

        ldc_stage = "Lane Departure Collisions"
        if ldc_stage in loadcase_score_dict:
            for sub_dict in loadcase_score_dict[ldc_stage].values():
                for lcs in sub_dict.values():
                    lcs.standard_score = 0.0
                    lcs.extended_score = 0.0
                    lcs.robustness_layer_score = 0.0
                    lcs.total_score = 0.0

        ldc_mask = scenario_stage_element == ldc_stage
        scenario_scores_df.loc[ldc_mask, "Score"] = 0.0

    # --- LSC rule ---
    lsc_car_ptw_gr = _get_gen_req_value("LSC - Car & PTW verif.")
    lsc_ped_cyc_gr = _get_gen_req_value("LSC - Ped & Cyc verif.")

    fail_lsc_car_ptw = lsc_car_ptw_gr == "fail"
    fail_lsc_ped_cyc = lsc_ped_cyc_gr == "fail"

    logger.info(
        f"LSC general requirements check: "
        f"LSC Car&PTW GR='{lsc_car_ptw_gr}', LSC Ped&Cyc GR='{lsc_ped_cyc_gr}'"
    )

    lsc_fail = fail_lsc_car_ptw or fail_lsc_ped_cyc

    if lsc_fail:
        lsc_reasons = []
        if fail_lsc_car_ptw:
            lsc_reasons.append(f"LSC Car&PTW General requirements='{lsc_car_ptw_gr}'")
        if fail_lsc_ped_cyc:
            lsc_reasons.append(f"LSC Ped&Cyc General requirements='{lsc_ped_cyc_gr}'")
        logger.warning(
            f"[!!!] Low Speed Collisions FAIL. Reasons: {', '.join(lsc_reasons)}"
        )

        lsc_stage = "Low Speed Collisions"
        if lsc_stage in loadcase_score_dict:
            for sub_dict in loadcase_score_dict[lsc_stage].values():
                for lcs in sub_dict.values():
                    lcs.standard_score = 0.0
                    lcs.extended_score = 0.0
                    lcs.robustness_layer_score = 0.0
                    lcs.total_score = 0.0

        lsc_mask = scenario_stage_element == lsc_stage
        scenario_scores_df.loc[lsc_mask, "Score"] = 0.0


def get_updated_dfs(dfs, loadcase_score_dict):
    scenario_scores_df = dfs["Scenario Scores"]
    scenario_scores_df["Score"] = scenario_scores_df["Score"].astype(float)
    for stage_element, subelements_dict in loadcase_score_dict.items():
        for stage_subelement, loadcases_dict in subelements_dict.items():
            for loadcase_key, loadcase_score in loadcases_dict.items():
                row_idx = scenario_scores_df.index[
                    scenario_scores_df["Scenario"] == loadcase_key
                ]
                if row_idx.empty:
                    logger.warning(f"Scenario {loadcase_key} not found in DataFrame.")
                    continue
                start_idx = row_idx[0]
                # Find the next index where "Scenario" is not NaN after start_idx
                next_idxs = scenario_scores_df.index[
                    (scenario_scores_df.index > start_idx)
                    & (scenario_scores_df["Scenario"].notna())
                ]
                end_idx = (
                    next_idxs[0] if not next_idxs.empty else len(scenario_scores_df)
                )
                loadcase_subset_df = scenario_scores_df.iloc[start_idx:end_idx]

                standard_layer_indices = loadcase_subset_df.index[
                    loadcase_subset_df["Layer"] == "Standard"
                ]
                scenario_scores_df.loc[standard_layer_indices, "Score"] = (
                    loadcase_score.standard_score
                )
                extended_layer_indices = loadcase_subset_df.index[
                    loadcase_subset_df["Layer"] == "Extended"
                ]
                scenario_scores_df.loc[extended_layer_indices, "Score"] = (
                    loadcase_score.extended_score
                )
                robustness_layer_indices = loadcase_subset_df.index[
                    loadcase_subset_df["Layer"] == "Robustness"
                ]
                scenario_scores_df.loc[robustness_layer_indices, "Score"] = (
                    loadcase_score.robustness_layer_score
                )

    check_general_requirements_rules(dfs, loadcase_score_dict, scenario_scores_df)

    category_scores_df = dfs["Category Scores"]
    category_scores_df["Score"] = category_scores_df["Score"].astype(float)
    test_scores_df = dfs["Test Scores"]
    test_scores_df["Score"] = test_scores_df["Score"].astype(float)

    scenario_effective = _prepare_effective_columns(
        scenario_scores_df,
        ["Stage element", "Stage subelement", "Category"],
    )
    category_effective = _prepare_effective_columns(
        category_scores_df,
        ["Stage element", "Stage subelement", "Category"],
    )
    test_effective = _prepare_effective_columns(
        dfs["Test Scores"],
        ["Stage element", "Stage subelement"],
    )

    category_row_lookup = {}
    for category_idx in category_scores_df.index:
        elem = category_effective["Stage element"].loc[category_idx]
        subelem = _normalize_stage_subelement_name(
            category_effective["Stage subelement"].loc[category_idx]
        )
        category_name = category_effective["Category"].loc[category_idx]
        key = (elem, subelem, category_name)
        category_row_lookup.setdefault(key, category_idx)

    idx = 0
    logger.debug("Printing loadcase_score_dict for debugging:")
    for stage_element, subelements_dict in loadcase_score_dict.items():
        for stage_subelement, loadcases_dict in subelements_dict.items():
            for loadcase_key, loadcase_score in loadcases_dict.items():
                logger.debug(
                    f"Stage element: {stage_element}, Stage subelement: {stage_subelement}, Scenario: {loadcase_key}, Score: {loadcase_score}"
                )

    while idx < len(scenario_scores_df):
        logger.debug(f"IDX: {idx}")
        current_idx = scenario_scores_df.index[idx]
        stage_element = scenario_effective["Stage element"].loc[current_idx]
        stage_subelement = _normalize_stage_subelement_name(
            scenario_effective["Stage subelement"].loc[current_idx]
        )
        category_name = scenario_effective["Category"].loc[current_idx]
        start_idx = idx
        # Find the next index where "Category" is notna after start_idx
        next_idxs = scenario_scores_df.index[
            (scenario_scores_df.index > start_idx)
            & (scenario_scores_df["Category"].notna())
        ]
        end_idx = next_idxs[0] if not next_idxs.empty else len(scenario_scores_df)
        subset_df = scenario_scores_df.iloc[start_idx:end_idx]
        idx += len(subset_df)
        logger.debug(f"IDX: {idx} after adding subset length {len(subset_df)}")
        score_sum = subset_df["Score"].sum()
        logger.debug(
            f"Computed score for {category_name} [{stage_element} - {stage_subelement}]: {score_sum} from rows {start_idx} to {end_idx}"
        )
        category_idx = category_row_lookup.get(
            (stage_element, stage_subelement, category_name)
        )
        if category_idx is not None:
            logger.debug(
                f"Updating score for {category_name} in {stage_element} - {stage_subelement} with value {score_sum}"
            )
            category_scores_df.loc[category_idx, "Score"] = score_sum

    for idx in test_scores_df.index:
        stage_element = test_effective["Stage element"].loc[idx]
        stage_subelement = _normalize_stage_subelement_name(
            test_effective["Stage subelement"].loc[idx]
        )

        scenario_scores = 0.0
        stage_subelement_key = get_stage_subelement_key(stage_element).value
        logger.debug(
            f"stage_subelement_key: {stage_subelement_key}, categories: {STAGE_SUBELEMENT_TO_CATEGORIES[stage_subelement_key]}"
        )
        processed_categories = set()
        for category_name in STAGE_SUBELEMENT_TO_CATEGORIES[stage_subelement_key]:
            category_idx = category_row_lookup.get(
                (stage_element, stage_subelement, category_name)
            )
            if category_idx is None:
                continue
            score = category_scores_df.loc[category_idx, "Score"]
            if pd.notna(score) and category_name not in processed_categories:
                scenario_scores += score
                processed_categories.add(category_name)

        logger.debug(
            f"Updating total score for {stage_element} - {stage_subelement} with category scores {scenario_scores}"
        )
        test_scores_df.loc[idx, "Score"] = scenario_scores

    scenario_scores_df["Score"] = scenario_scores_df["Score"].map(
        lambda v: round_half_up(v, 3)
    )
    category_scores_df["Score"] = category_scores_df["Score"].map(
        lambda v: round_half_up(v, 3)
    )
    test_scores_df["Score"] = test_scores_df["Score"].map(lambda v: round_half_up(v, 3))

    updated_dfs = {
        sheet_name: df for sheet_name, df in dfs.items() if "verif." in sheet_name
    }
    updated_dfs["Scenario Scores"] = scenario_scores_df
    updated_dfs["Category Scores"] = category_scores_df
    updated_dfs["Test Scores"] = test_scores_df

    return updated_dfs


def calculate_score(dfs: dict) -> dict:
    """Calculate crash avoidance NCAP scores from input dataframes.

    Args:
        dfs: Dictionary of dataframes from the input Excel file, including
             ``"{sheet} (bg)"`` entries for every ``"pred."`` sheet (produced
             by ``common.read_excel_file_to_dfs(..., load_bg_colors=True)``).

    Returns:
        dict: Updated dataframes with computed scores (no ``"(bg)"`` keys).
    """
    # Re-derive every protected reference cell (Max score, row labels, ...)
    # from the official packaged template instead of trusting it from the
    # caller's dfs -- callers that only ever pass DataFrames (no file, no
    # preprocess() call) never went through rebuild_trusted_input, so this
    # must happen unconditionally here rather than being assumed done
    # upstream. Only the designated grey input cells are carried over.
    dfs = integrity.rebuild_trusted_input_dfs(dfs)

    if "Scenario Scores" not in dfs:
        raise ValueError("The input file must contain a 'Scenario Scores' sheet.")
    if "Test Scores" not in dfs:
        raise ValueError("The input file must contain a 'Test Scores' sheet.")

    loadcase_score_dict = {}

    for stage_info in STAGE_SUBELEMENTS:
        stage_element = stage_info["Stage element"]
        stage_subelement = stage_info["Stage subelement"]
        if stage_subelement == "Pedestrian & cyclist":
            stage_subelement = "Ped & Cyc"
        if stage_subelement == "Single vehicle":
            stage_subelement = "Single Veh"
        if stage_element not in loadcase_score_dict:
            loadcase_score_dict[stage_element] = {}
        if stage_subelement not in loadcase_score_dict[stage_element]:
            loadcase_score_dict[stage_element][stage_subelement] = {}

        compute_stage_score(dfs, stage_info, loadcase_score_dict)

    apply_cross_scenario_robustness_rule(loadcase_score_dict)

    for stage_element, subelements_dict in loadcase_score_dict.items():
        for stage_subelement, loadcases_dict in subelements_dict.items():
            for loadcase_key, loadcase_score in loadcases_dict.items():
                logger.debug(
                    f"Computed scenario score: Stage element={stage_element}, "
                    f"Stage subelement={stage_subelement}, Scenario={loadcase_key}, "
                    f"Scores={loadcase_score}"
                )

    updated_dfs = get_updated_dfs(dfs, loadcase_score_dict)

    result = {**dfs, **updated_dfs}
    # Strip internal bg-color sheets — they are never written to the output file
    result = {k: v for k, v in result.items() if not k.endswith("(bg)")}
    # "Documentation"/"Data" are carry-through-only: computed once by
    # preprocess, never recomputed here. report_writer.write_report hard-
    # copies them straight from the preprocessed input file instead, so drop
    # them here to avoid a spurious pandas round-trip, which would coerce
    # the "TRUE"/"FALSE" strings preprocess wrote into native bool values
    # and diverge from what preprocess actually produced.
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
    print(f"[Crash Avoidance] Computing NCAP scores from input Excel file...")

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
        input_file, load_bg_colors=True, preserve_na_sheets=PRESERVE_NA_SHEETS
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
