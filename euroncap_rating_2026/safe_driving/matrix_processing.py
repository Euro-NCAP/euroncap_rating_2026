# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import sys
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from enum import Enum
import os
import warnings

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


from euroncap_rating_2026.safe_driving import data_model

logger = logging.getLogger(__name__)


class PredictionInput(str, Enum):
    UNKNOWN = "Unknown"
    YES = "Yes"
    NO = "No"


PREDICTION_COLOR_MAP = {
    PredictionInput.YES: (0 / 255, 153 / 255, 51 / 255),
    PredictionInput.UNKNOWN: (128 / 255, 128 / 255, 128 / 255),
    PredictionInput.NO: (255 / 255, 51 / 255, 51 / 255),
}


def plot_matrix(matrix, extended_range_cells, name="matrix"):
    unique_colors = sorted(PREDICTION_COLOR_MAP.keys())
    # Prepare the matrix for plotting
    # Log the length of legforms_string_matrix
    n_rows = len(matrix)
    n_cols = len(matrix[0]) if n_rows > 0 else 0

    fig, ax = plt.subplots(figsize=(n_cols, n_rows))
    # Draw colored cells
    for i in range(n_rows):
        for j in range(n_cols):
            color = matrix[i][j]
            rect = plt.Rectangle(
                (j, n_rows - 1 - i),
                1,
                1,
                facecolor=PREDICTION_COLOR_MAP.get(color, "white"),
                edgecolor="black",
            )
            ax.add_patch(rect)

    # Highlight selected cells with an X
    for row, col in extended_range_cells:
        rect = plt.Rectangle(
            (col, n_rows - 1 - row),
            1,
            1,
            fill=False,
            edgecolor="black",
            linewidth=3,
            linestyle="--",
        )
        ax.add_patch(rect)

    # Set axis limits and labels
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title(f"{name} with Selected Cells")

    # Create a legend for colors
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=PREDICTION_COLOR_MAP[color])
        for color in unique_colors
        if color not in ["grey", "blue"]
    ]
    labels = [
        color.capitalize() for color in unique_colors if color not in ["grey", "blue"]
    ]
    if "grey" in unique_colors:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=PREDICTION_COLOR_MAP["grey"]))
        labels.append("Grey")

    ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc="upper left")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            plt.tight_layout()
    except Exception as e:
        logger.warning(f"tight_layout() failed: {e}")

    # Save to /output directory
    output_dir = "matrices"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{name}_matrix.png"))
    plt.close()


@dataclass
class TestPoint:
    """
    Represents a VRU test point with its coordinates and color.
    """

    row: int
    col: int
    prediction_input: PredictionInput = None
    attributes: dict = field(default_factory=dict)


def get_dm_test_matrix(prediction_df, test_name, stage_subelement_key):
    """
    Process the DE DM predictions DataFrame to extract headforms  matrices,
    compute color percentages, and randomly select samples based on color distribution.

    Args:
        prediction_df (pd.DataFrame): DataFrame containing VRU predictions.

    Returns:
        None
    """

    # Ensure the DataFrame is not empty
    if prediction_df.empty:
        logger.error("The input DataFrame is empty.")
        return

    test_matrix = None
    matrix_indices = data_model.DM_MATRIX_INDICES.get(test_name, None)

    if not matrix_indices:
        logger.error(f"Test name '{test_name}' not found in DM_MATRIX_INDICES.")
        return

    start_row = matrix_indices["start_row"]
    n_rows = matrix_indices["n_rows"]
    start_col = matrix_indices["start_col"]
    n_cols = matrix_indices["n_cols"]

    test_matrix = prediction_df.iloc[
        start_row : start_row + n_rows,
        start_col : start_col + n_cols,
    ].reset_index(drop=True)

    test_matrix = test_matrix.iloc[:, :].astype(str).values.tolist()

    test_points = []

    test_start_col = matrix_indices["start_col"]
    test_start_row = matrix_indices["start_row"]
    # Convert all values in test_matrix to PredictionColor enum
    for i in range(len(test_matrix)):
        for j in range(len(test_matrix[0])):
            color_str = test_matrix[i][j].lower()
            enum_val = next(
                (k for k in PredictionInput if k.value.lower() == color_str),
                PredictionInput.UNKNOWN,
            )
            test_matrix[i][j] = enum_val

            attributes = {}
            # Add start_col -1, -2, -3 if available
            offset = 1
            col_idx = test_start_col - offset
            while col_idx >= 0:
                if col_idx == 0:
                    if test_name == "CBNAO - SfS":
                        col_name = "d"
                    elif test_name in ["CBDA", "CPMRCm"]:
                        col_name = "Gap"
                    elif test_name in ["CPMFC"]:
                        col_name = "Distance"
                    else:
                        col_name = "VUT speed"
                else:
                    # col_name = str(prediction_df.columns[col_idx])
                    target_row = test_start_row - 2
                    col_name = str(prediction_df.iloc[target_row, col_idx])
                    if test_start_row == 1:
                        col_name = prediction_df.columns[col_idx]
                col_value = prediction_df.iloc[test_start_row + i, col_idx]
                if col_value is not None and not pd.isna(col_value):
                    attributes[col_name] = col_value.strip()
                offset += 1
                col_idx = test_start_col - offset

            # Add row at index start_row -2 for the name and start_row -1 for the value
            if test_start_row == 1:
                row_name = str(prediction_df.columns[test_start_col])
            else:
                row_name = str(prediction_df.iloc[test_start_row - 2, test_start_col])

            if (
                pd.isna(row_name)
                or str(row_name).strip() == "nan"
                or str(row_name).strip() == "<NA>"
                or "Unnamed" in str(row_name).strip()
            ):
                if test_start_row == 1:
                    # Search backward from test_start_col for the first non-NaN value in columns array
                    for col_idx in range(test_start_col, -1, -1):
                        potential_row_name = prediction_df.columns[col_idx]
                        if (
                            not pd.isna(potential_row_name)
                            and str(potential_row_name).strip() != "nan"
                            and "Unnamed" not in str(potential_row_name).strip()
                        ):
                            row_name = str(potential_row_name)
                            break
                else:
                    # Search backward from test_start_col for the first non-NaN value in row test_start_row - 2
                    for col_idx in range(test_start_col, -1, -1):
                        potential_row_name = prediction_df.iloc[
                            test_start_row - 2, col_idx
                        ]
                        if (
                            not pd.isna(potential_row_name)
                            and str(potential_row_name).strip() != "nan"
                        ):
                            row_name = str(potential_row_name)
                            break

            row_value = prediction_df.iloc[test_start_row - 1, test_start_col + j]

            if row_value is not None and not pd.isna(row_value):
                attributes[row_name] = row_value

            test_points.append(
                TestPoint(
                    row=i,
                    col=j,
                    prediction_input=enum_val,
                    attributes=attributes,
                )
            )
    logger.debug("Test points for %s:", test_name)
    for tp in test_points:
        logger.debug(
            "  Row %d, Col %d: %s, Attributes: %s",
            tp.row,
            tp.col,
            tp.prediction_input,
            tp.attributes,
        )

    for idx, row in enumerate(test_matrix):
        logger.debug("%d: %s", idx, row)

    if HAS_MPL:
        logger.debug("Plotting matrix for test: %s", test_name)
        plot_matrix(test_matrix, [], name=test_name)
    return test_points
