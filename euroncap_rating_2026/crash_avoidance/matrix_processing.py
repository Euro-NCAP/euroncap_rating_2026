# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import random
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


from euroncap_rating_2026.crash_avoidance.robustness_layer import (
    RobustnessLayer,
    AssessmentCriteria,
)
from euroncap_rating_2026.crash_avoidance import data_model
from euroncap_rating_2026 import common
from euroncap_rating_2026 import config

logger = logging.getLogger(__name__)
# The FC v1.2 sec 4.2.4 impact-speed tolerance, kept as public module
# surface. It is the v_rel_impact entry of the per-criteria map, which is
# the single source of truth -- see _tolerance_for.
COLOR_THRESHOLD_TOLERANCE = data_model.NCAP_TEST_CRITERIA_TO_TOLERANCE[
    data_model.NcapTestCriteria.MITIGATION_OR_AVOIDANCE
]  # km/h tolerance
settings = config.Settings()


def _normalize_lookup_value(value):
    if isinstance(value, str):
        return value.strip().lower()
    return value


def build_start_row_lookup(df: pd.DataFrame, col_idx: int = 0) -> dict:
    """Build a lookup map for first occurrence start rows in the given column."""
    lookup = {}
    if col_idx < 0 or col_idx >= len(df.columns):
        return lookup

    column_name = df.columns[col_idx]
    normalized_header = _normalize_lookup_value(column_name)
    if len(df) > 0 and normalized_header is not None:
        lookup[normalized_header] = 1

    for row_idx in range(len(df)):
        cell_value = df.iloc[row_idx, col_idx]
        if pd.isna(cell_value):
            continue
        normalized_value = _normalize_lookup_value(cell_value)
        if normalized_value is None or normalized_value in lookup:
            continue
        lookup[normalized_value] = row_idx + 2

    return lookup


def plot_matrix(matrix, extended_range_cells, name="matrix"):
    unique_colors = sorted(common.PREDICTION_COLOR_MAP.keys())
    # Prepare the matrix for plotting
    # Log the length of legforms_string_matrix
    n_rows = len(matrix)
    n_cols = len(matrix[0]) if n_rows > 0 else 0

    fig, ax = plt.subplots(figsize=(n_cols, n_rows))

    # Draw colored cells
    for i in range(n_rows):
        for j in range(n_cols):
            color = matrix[i][j].lower() if matrix[i][j] is not None else None
            rect = plt.Rectangle(
                (j, n_rows - 1 - i),
                1,
                1,
                facecolor=common.PREDICTION_COLOR_MAP.get(color, "black"),
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
        plt.Rectangle((0, 0), 1, 1, color=common.PREDICTION_COLOR_MAP[color])
        for color in unique_colors
        if color not in ["grey", "blue"]
    ]
    labels = [
        color.capitalize() for color in unique_colors if color not in ["grey", "blue"]
    ]
    if "grey" in unique_colors:
        handles.append(
            plt.Rectangle((0, 0), 1, 1, color=common.PREDICTION_COLOR_MAP["grey"])
        )
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


class TestRange(Enum):
    STANDARD = "Standard"
    EXTENDED = "Extended"
    UNKNOWN = "Unknown"


class PredictionSource(Enum):
    VTA = "VTA"
    SELF_CLAIMED = "Self claimed"
    UNKNOWN = "Unknown"


@dataclass
class TestPoint:
    """
    Represents a VRU test point with its coordinates and color.
    """

    row: int
    col: int
    color: common.PredictionColor = None
    test_range: TestRange = TestRange.UNKNOWN
    attributes: dict = field(default_factory=dict)
    robustness_layer: RobustnessLayer = None
    total_rows: int = None
    # Library-private, precomputed bookkeeping (e.g. "_cpla_row_info",
    # "_cbla_standard_ils" -- see classify_cpla_rows/classify_cbla_rows) that
    # downstream logic in this module needs but that must never appear in
    # `attributes`, since external callers (e.g. an application's prediction cell
    # values API) read `attributes` verbatim.
    internal: dict = field(default_factory=dict)

    @property
    def predicted_score(self) -> float:
        if self.test_range == TestRange.STANDARD:
            if self.color == common.PredictionColor.GREEN:
                return 1.0
            elif self.color == common.PredictionColor.YELLOW:
                return 0.75
            elif self.color == common.PredictionColor.ORANGE:
                return 0.5
            elif self.color == common.PredictionColor.BROWN:
                return 0.25
            elif self.color == common.PredictionColor.RED:
                return 0.0
            else:
                return 0.0
        else:
            if self.color == common.PredictionColor.RED:
                return 0.0
            else:
                return 1.0


class PredictionResult(str, Enum):
    UNKNOWN = "Unknown"
    CORRECT = "Correct"
    INCORRECT = "Incorrect"
    IN_TOLERANCE = "In Tolerance"


def get_band_color(thresholds, value):
    """
    The colour band `value` falls in, with no tolerance applied -- the "true
    colour of the test point" of FC v1.2 sec 4.2.4, obtained "by comparing the
    actual measured impact speed with the colour bands in section 5.2".

    thresholds: list of (PredictionColor, upper_bound), sorted ascending.
    Intervals: (-inf, t0], (t0, t1], ..., (tn-1, tn], tn==inf.
    """
    thresholds = sorted(thresholds, key=lambda x: x[1])
    lower = float("-inf")
    for color, upper in thresholds:
        if lower < value <= upper:
            return color
        lower = upper
    # Should not happen if last threshold is inf, but fallback
    return thresholds[-1][0]


def check_thresholds(thresholds, oem_predicted_color, value, tolerance):
    """
    Verify one measured value against the OEM's predicted colour band.

    Returns the prediction result and the colour to *apply* to the point: the
    band colour when the prediction is right or wrong outright, and -- per
    sec 4.2.4, "when a tested point scores better than predicted, but within
    tolerance, the predicted result is applied" -- the predicted colour when
    the value falls in the band widened by `tolerance`. Consumers that need
    the measured outcome rather than the applied one (an absolute performance
    criterion, say) must read the band colour: get_band_color, or
    ComputedTestPoint.measured_color.
    """
    interval_color = get_band_color(thresholds, value)
    thresholds = sorted(thresholds, key=lambda x: x[1])

    # Compare with OEM predicted color
    if interval_color == oem_predicted_color:
        return PredictionResult.CORRECT, interval_color

    # Not the predicted colour: the point still counts as predicted when it
    # falls inside the predicted band widened by the tolerance.
    # Find the interval for OEM color
    oem_idx = next(
        (i for i, (c, _) in enumerate(thresholds) if c == oem_predicted_color), None
    )
    if oem_idx is not None:
        # Widen the predicted band outward by the tolerance, in both
        # directions: FC v1.2 sec 4.2.4 applies it "when a tested point scores
        # better than predicted" as well as when it scores worse, and in both
        # cases "the predicted result is applied".
        upper_bound = thresholds[oem_idx][1]
        if oem_idx == 0:
            # Lowest band (Green, v = 0): sec 4.2.4's worked table prints its
            # accepted range as the strict "v < 2", unlike the half-open rows
            # below it.
            if value < upper_bound + tolerance:
                return PredictionResult.IN_TOLERANCE, oem_predicted_color
        else:
            # An impact speed is never negative, so the widened lower edge
            # stops at 0 -- which is why the same table accepts
            # "0 < v <= 12" for Yellow rather than "-2 < v <= 12". Criteria
            # whose KPI can go negative (dtle_t_end, in metres) carry
            # tolerance 0.0, so the clamp never reaches them.
            lower_bound = max(thresholds[oem_idx - 1][1] - tolerance, 0.0)
            if lower_bound < value <= upper_bound + tolerance:
                return PredictionResult.IN_TOLERANCE, oem_predicted_color

    return PredictionResult.INCORRECT, interval_color


def _tolerance_for(criteria):
    """
    The colour-band tolerance to apply for one criteria's KPI, in that KPI's
    own unit (see data_model.NCAP_TEST_CRITERIA_TO_TOLERANCE).

    Callers pass the criteria of the branch they actually take, not
    data_model.get_criteria_for_test(test_name): a CPLA/CBLA test point is
    re-classified from MITIGATION_OR_AVOIDANCE into WARNING (declared FCW) or
    AVOIDANCE (FCW declared as AEB) after that lookup, and only the branch
    knows which KPI is being compared.
    """
    return data_model.NCAP_TEST_CRITERIA_TO_TOLERANCE.get(criteria, 0.0)


# Fallback row indices for is_fcw_declared_as_aeb, only used when a
# TestPoint's total_rows isn't known (e.g. constructed without it). Covers the
# current template's 4-row CPLA choice band (rows 6-9 of its 10-row layout);
# prefer the attribute/relative-position checks there wherever possible, since
# this list -- like the static EXTENDED_RANGE_CELLS index map -- goes stale
# the moment a sheet's row count differs from the layout it was calibrated
# against.
FCW_AEB_ROW = [6, 7, 8, 9]

CPLA_CHOICE_BAND_ROW_COUNT = 4


def is_fcw_declared_as_aeb(test_point, test_name):
    """
    True when an AEB-labeled CPLA/CBLA test point is really the OEM's FCW
    function declared as AEB (scored as avoidance), rather than a genuinely
    original AEB point (scored as mitigation-or-avoidance).

    CBLA tells the two apart by content: "Target speed" 15 km/h is the
    fixed/original-AEB band, 20 km/h is the choice band where FCW may be
    declared as AEB.

    CPLA has no such content difference (both bands run at 5 km/h target
    speed) - when the matrix was built via get_test_matrix_from_region, the
    test point carries a precomputed "_cpla_row_info" entry in `internal`
    (see classify_cpla_rows) derived from the matrix's actual VUT speeds,
    which correctly identifies the choice band even after
    test_info.collapse_cpla_cbla_aeb_duplicate_rows has removed a variable
    number of its duplicate rows. Otherwise it falls back to position (the
    choice band is always the last CPLA_CHOICE_BAND_ROW_COUNT rows), which
    only holds for an uncollapsed matrix.
    """
    if "CBLA" in test_name:
        return test_point.attributes.get("Target speed") == "20 km/h"
    row_info = test_point.internal.get("_cpla_row_info")
    if row_info is not None:
        return row_info.is_choice
    if test_point.total_rows:
        return test_point.row >= test_point.total_rows - CPLA_CHOICE_BAND_ROW_COUNT
    return test_point.row in FCW_AEB_ROW


@dataclass
class ComputedTestPoint:
    """
    Represents a computed test point with its attributes.
    """

    test_point: TestPoint
    test_name: str = None
    value: float = None
    prediction_result: PredictionResult = PredictionResult.UNKNOWN
    assessment_criteria: AssessmentCriteria = None
    # The point's true colour: the band the measured value falls in, with no
    # tolerance applied (sec 4.2.4). `computed_color` may differ -- it is the
    # colour applied to the point, which is the OEM's prediction when the
    # value is only within tolerance of it. None when there is nothing
    # measured to place in a band (no value, or an LSC point, whose colour
    # comes from get_lsc_computed_color).
    measured_color: common.PredictionColor = None

    def __init__(
        self,
        test_point,
        test_name=None,
        value=None,
        prediction_result=PredictionResult.UNKNOWN,
        assessment_criteria=None,
    ):
        self.test_point = test_point
        self.test_name = test_name
        self.value = value
        self.prediction_result = prediction_result
        self.assessment_criteria = assessment_criteria
        self.measured_color = None
        self.computed_color = self._compute_color_logic()

    def _verify(self, thresholds, criteria):
        """
        Verify this point against one criteria's colour bands: record its true
        colour and the prediction result, and return the colour to apply.

        The tolerance comes from the criteria of the branch actually taken,
        not from data_model.get_criteria_for_test(self.test_name): a
        CPLA/CBLA point is re-classified into WARNING (declared FCW) or
        AVOIDANCE (FCW declared as AEB) before reaching here.
        """
        self.measured_color = get_band_color(thresholds, self.value)
        prediction_result, applied_color = check_thresholds(
            thresholds,
            self.test_point.color,
            self.value,
            _tolerance_for(criteria),
        )
        self.prediction_result = prediction_result
        return applied_color

    def _compute_color_logic(self):
        if self.value is None:
            self.prediction_result = PredictionResult.CORRECT
            logger.debug("Value is None, using prediction color.")
            return self.test_point.color

        vut_speed_str = self.test_point.attributes.get("VUT speed")
        if vut_speed_str and vut_speed_str.strip().lower() == "sfs":
            vut_speed = 0.0
        else:
            vut_speed = (
                float(vut_speed_str.replace(" km/h", "")) if vut_speed_str else 0.0
            )

        # Lookup tables for thresholds and tolerances
        mitigation_or_avoidance_thresholds = {
            10.0: [
                (common.PredictionColor.GREEN, 0.0),
                (common.PredictionColor.RED, float("inf")),
            ],
            20.0: [
                (common.PredictionColor.GREEN, 0.0),
                (common.PredictionColor.RED, float("inf")),
            ],
            30.0: [
                (common.PredictionColor.GREEN, 0.0),
                (common.PredictionColor.BROWN, 10.0),
                (common.PredictionColor.RED, float("inf")),
            ],
            40.0: [
                (common.PredictionColor.GREEN, 0.0),
                (common.PredictionColor.ORANGE, 10.0),
                (common.PredictionColor.BROWN, 20.0),
                (common.PredictionColor.RED, float("inf")),
            ],
            50.0: [
                (common.PredictionColor.GREEN, 0.0),
                (common.PredictionColor.YELLOW, 10.0),
                (common.PredictionColor.ORANGE, 20.0),
                (common.PredictionColor.BROWN, 30.0),
                (common.PredictionColor.RED, float("inf")),
            ],
        }

        mitigation_scenarios_thresholds = [
            (common.PredictionColor.RED, 10.0),
            (common.PredictionColor.ORANGE, 20.0),
            (common.PredictionColor.GREEN, 30.0),
            (common.PredictionColor.RED, float("inf")),
        ]
        avoidance_thresholds = [
            (common.PredictionColor.GREEN, 0.0),
            (common.PredictionColor.RED, float("inf")),
        ]
        warning_thresholds = [
            (common.PredictionColor.RED, 1.7),
            (common.PredictionColor.GREEN, float("inf")),
        ]
        road_edge_thresholds = [
            (common.PredictionColor.RED, -0.1),
            (common.PredictionColor.GREEN, float("inf")),
        ]
        elk_target_thresholds = [
            (common.PredictionColor.GREEN, 0.0),
            (common.PredictionColor.RED, float("inf")),
        ]

        avoidance_scenarios = data_model.NCAP_TEST_SCENARIOS[
            data_model.NcapTestCriteria.AVOIDANCE
        ]
        mitigation_scenarios = data_model.NCAP_TEST_SCENARIOS[
            data_model.NcapTestCriteria.MITIGATION
        ]
        mitigation_or_avoidance_scenarios = data_model.NCAP_TEST_SCENARIOS[
            data_model.NcapTestCriteria.MITIGATION_OR_AVOIDANCE
        ]
        warning_scenarios = data_model.NCAP_TEST_SCENARIOS[
            data_model.NcapTestCriteria.WARNING
        ]
        road_edge_scenarios = data_model.NCAP_TEST_SCENARIOS[
            data_model.NcapTestCriteria.ROAD_EDGE
        ]
        elk_target_scenarios = data_model.NCAP_TEST_SCENARIOS[
            data_model.NcapTestCriteria.ELK_TARGET
        ]

        is_mitigation_or_avoidance = self.test_name in mitigation_or_avoidance_scenarios
        is_mitigation = self.test_name in mitigation_scenarios
        is_avoidance = self.test_name in avoidance_scenarios
        is_warning = self.test_name in warning_scenarios
        is_road_edge = self.test_name in road_edge_scenarios
        is_elk_target = self.test_name in elk_target_scenarios

        logger.debug(
            f"is_mitigation_or_avoidance: {is_mitigation_or_avoidance}, "
            f"is_mitigation: {is_mitigation}, "
            f"is_avoidance: {is_avoidance}, "
            f"is_warning: {is_warning}, "
            f"is_road_edge: {is_road_edge}, "
            f"is_elk_target: {is_elk_target}"
        )
        logger.debug(f"VUT speed: {vut_speed} km/h")

        if "CPLA" in self.test_name or "CBLA" in self.test_name:
            if self.test_point.attributes.get("Function") == "FCW":
                logger.debug(
                    f"Test point row {self.test_point.row} indicates FCW function. Setting is_warning to True."
                )
                is_warning = True
                is_mitigation_or_avoidance = False
            # Special case for CPLA scenarios
            elif self.test_point.attributes.get("Function") == "AEB":
                if is_fcw_declared_as_aeb(self.test_point, self.test_name):
                    logger.debug(
                        f"Test point row {self.test_point.row} indicates FCW declared as AEB. Setting is_avoidance to True."
                    )
                    is_avoidance = True
                    is_mitigation_or_avoidance = False
                else:
                    logger.debug(
                        f"Test point row {self.test_point.row} indicates original AEB. Setting is_mitigation_or_avoidance to True."
                    )
                    is_mitigation_or_avoidance = True

        # Mitigation or avoidance scenarios
        if is_mitigation_or_avoidance:
            logger.debug(
                f"Test name '{self.test_name}' is in mitigation or avoidance scenarios."
            )
            speed = 50.0 if vut_speed >= 50.0 else vut_speed
            thresholds_at_speed = mitigation_or_avoidance_thresholds.get(speed, [])
            # Find the index of the OEM predicted color in the thresholds list
            logger.debug(f"Speed: {speed}")
            logger.debug(f"Thresholds: {thresholds_at_speed}")
            logger.debug(f"OEM predicted color: {self.test_point.color}")

            return self._verify(
                thresholds_at_speed,
                data_model.NcapTestCriteria.MITIGATION_OR_AVOIDANCE,
            )

        # Mitigation scenarios
        elif is_mitigation:
            logger.debug(f"Test name '{self.test_name}' is in mitigation scenarios.")
            # Use similar logic as mitigation_or_avoidance_scenarios
            speed = 30.0 if vut_speed == 30.0 else 35.0

            predicted_color = self._verify(
                mitigation_scenarios_thresholds,
                data_model.NcapTestCriteria.MITIGATION,
            )
            logger.debug(
                f"Predicted color: {predicted_color}, Prediction result: {self.prediction_result}"
            )
            return predicted_color

        # Avoidance scenarios
        elif is_avoidance:
            logger.debug(f"Test name '{self.test_name}' is in avoidance scenarios.")
            return self._verify(
                avoidance_thresholds, data_model.NcapTestCriteria.AVOIDANCE
            )

        # Warning scenarios
        elif is_warning:
            logger.debug(f"Test name '{self.test_name}' is in warning scenarios.")
            return self._verify(warning_thresholds, data_model.NcapTestCriteria.WARNING)
        elif is_road_edge:
            logger.debug(f"Test name '{self.test_name}' is in road edge scenarios.")

            return self._verify(
                road_edge_thresholds, data_model.NcapTestCriteria.ROAD_EDGE
            )
        elif is_elk_target:
            logger.debug(f"Test name '{self.test_name}' is in ELK target scenarios.")
            return self._verify(
                elk_target_thresholds, data_model.NcapTestCriteria.ELK_TARGET
            )

        # Unknown category
        else:
            self.prediction_result = PredictionResult.UNKNOWN
            logger.debug(
                f"Test name '{self.test_name}' not recognized for computed color logic."
            )
            return common.PredictionColor.GREY

    def get_lsc_computed_color(self, vehicle_response):
        lsc_avoidance_scenarios = [
            "CCCscp SfS",
            "CMCscp SfS",
            "CCFtap SfS",
            "CMFtap SfS",
            "CBNAO SfS",
            "CPMRCm",
            "CPMRCs",
        ]
        lsc_mitigation_scenarios = ["CPMFC"]
        lsc_dooring_scenarios = ["CBDA"]

        is_avoidance = self.test_name in lsc_avoidance_scenarios
        is_mitigation = self.test_name in lsc_mitigation_scenarios
        is_dooring = self.test_name in lsc_dooring_scenarios
        logger.debug(
            f"is_avoidance: {is_avoidance}, is_mitigation: {is_mitigation}, is_dooring: {is_dooring}"
        )

        if is_avoidance or is_mitigation:
            if self.value <= 0.0:
                logger.debug(
                    f"Test name '{self.test_name}' is in avoidance or mitigation scenario with value <= 0.0."
                )
                logger.debug(f"Returning GREEN color.")

                self.prediction_result = PredictionResult.CORRECT
                return common.PredictionColor.GREEN
            else:
                logger.debug(
                    f"Test name '{self.test_name}' is in avoidance or mitigation scenario with value > 0.0."
                )
                logger.debug(f"Returning RED color.")
                self.prediction_result = PredictionResult.INCORRECT
                return common.PredictionColor.RED

        elif is_dooring:
            # LSC dooring thresholds logic
            # Vehicle response | Criteria | Doors | Colour Band
            # Information: TTC ≥ 2.30s, Driver's only -> Brown; TTC < 2.30s -> Red
            # Warning: TTC ≥ 1.70s, Driver's only -> Orange; All -> Yellow; TTC < 1.70s, Driver's only or All -> Red
            # Retention: Start @ TTC ≥ 1.70s AND End @ TTC ≤ -0.40s, Driver's only -> Yellow; All -> Green
            # Start @ TTC < 1.70s OR End @ TTC > -0.40s, Driver's only or All -> Red

            ttc = self.value  # TTC value
            logger.debug(
                f"LSC Door scenario: value={ttc}, vehicle_response={vehicle_response}, predicted_color={self.test_point.color}"
            )
            logger.debug(
                f"vehicle_response==VehicleResponse.INFORMATION: {vehicle_response==data_model.VehicleResponse.INFORMATION}"
            )
            if vehicle_response == data_model.VehicleResponse.INFORMATION:
                if ttc >= 2.30:
                    logger.debug(f"Returning BROWN color.")
                    return common.PredictionColor.BROWN
                else:
                    logger.debug(f"Returning RED color.")
                    return common.PredictionColor.RED
            elif vehicle_response == data_model.VehicleResponse.WARNING:
                if ttc >= 1.70:
                    if self.test_point.color == common.PredictionColor.ORANGE:
                        return common.PredictionColor.ORANGE
                    else:
                        return common.PredictionColor.YELLOW
                else:
                    return common.PredictionColor.RED
            elif vehicle_response == data_model.VehicleResponse.RETENTION:
                if pd.isna(ttc):
                    return common.PredictionColor.RED
                # FAIL if -0.4 < value < 1.7, PASS otherwise
                passes = not (-0.40 < ttc < 1.70)
                if passes and self.test_point.color in (
                    common.PredictionColor.GREEN,
                    common.PredictionColor.YELLOW,
                ):
                    return self.test_point.color
                else:
                    return common.PredictionColor.RED


CBLA_STANDARD_IMPACT_LOCATION_BY_TARGET_SPEED = {
    "15 km/h": "50%",
    "20 km/h": "25%",
}

CPLA_STANDARD_IMPACT_LOCATION_BY_BAND = {
    "fixed": "50%",
    "choice": "25%",
}


@dataclass(frozen=True)
class CplaRowInfo:
    """
    Frozen (hashable) so it can be embedded in TestPoint.attributes without
    breaking test_info.find_duplicate_rows, which hashes the full attributes
    tuple to detect duplicate declarations.
    """

    standard_ils: frozenset
    is_choice: bool


def _parse_vut_speed_km_h(vut_speed):
    """
    Parse a "50 km/h"-style VUT speed attribute to a float. Returns None if
    it's missing or not in that shape (e.g. "SfS").
    """
    if not vut_speed:
        return None
    try:
        return float(vut_speed.replace(" km/h", ""))
    except (ValueError, AttributeError):
        return None


def classify_cpla_rows(row_vut_speeds):
    """
    Determine, for each row of a CPLA matrix, the set of "Impact location"
    values that count as standard range and whether the row belongs to the
    choice band -- from the matrix's actual VUT speeds, not row position.

    CPLA's fixed band (rows always Function=AEB) and choice band (grey-cell
    Function dropdown) both have a row at 50 and 60 km/h, the only speeds
    where a row is duplicated. When
    test_info.collapse_cpla_cbla_aeb_duplicate_rows removes a choice-band
    duplicate (because it declared AEB), the surviving single row at that
    speed is promoted: its extended-range 25% IL cell becomes standard too,
    on top of the fixed band's usual 50% IL. Row *position* can no longer
    tell fixed and choice bands apart once a variable number of rows have
    been removed -- this resolves it from content instead, and produces the
    same result as the old bottom-anchored logic whenever nothing has been
    collapsed.

    Args:
        row_vut_speeds: ordered list of each row's "VUT speed" attribute
            value (e.g. "50 km/h"), one per physical row of the matrix.

    Returns:
        dict: row index -> CplaRowInfo
    """
    speed_counts = {}
    for speed in row_vut_speeds:
        speed_counts[speed] = speed_counts.get(speed, 0) + 1

    occurrence_seen = {}
    result = {}
    for row, speed in enumerate(row_vut_speeds):
        occurrence_seen[speed] = occurrence_seen.get(speed, 0) + 1
        occurrence = occurrence_seen[speed]
        speed_value = _parse_vut_speed_km_h(speed)
        is_overlap_speed = speed_value in (50.0, 60.0)

        if is_overlap_speed and speed_counts[speed] == 2:
            if occurrence == 1:
                result[row] = CplaRowInfo(frozenset({"50%"}), is_choice=False)
            else:
                result[row] = CplaRowInfo(frozenset({"25%"}), is_choice=True)
        elif is_overlap_speed:
            # Singular occurrence: the choice-band duplicate was collapsed
            # away (or never existed) -- promote this row to standard at
            # both impact locations.
            result[row] = CplaRowInfo(frozenset({"50%", "25%"}), is_choice=False)
        elif speed_value is not None and speed_value >= 70.0:
            result[row] = CplaRowInfo(frozenset({"25%"}), is_choice=True)
        else:
            result[row] = CplaRowInfo(frozenset({"50%"}), is_choice=False)

    return result


def classify_cbla_rows(row_vut_speeds, row_target_speeds):
    """
    Determine, for each row of a CBLA matrix, the set of "Impact location"
    values that count as standard range -- from the matrix's actual
    VUT-speed/Target-speed pairing, so a fixed-band row correctly picks up
    the promoted 25% IL once its choice-band sibling at the same VUT speed
    has been collapsed away (see classify_cpla_rows for the CPLA analog, and
    test_info.collapse_cpla_cbla_aeb_duplicate_rows for the collapse itself).

    CBLA's bands are already tellable apart by content ("Target speed" 15
    km/h fixed vs 20 km/h choice -- see is_fcw_declared_as_aeb), so unlike
    CPLA this only needs the promotion, not a fixed/choice classification.

    Args:
        row_vut_speeds: ordered list of each row's "VUT speed" attribute.
        row_target_speeds: ordered list of each row's "Target speed"
            attribute.

    Returns:
        dict: row index -> frozenset[str] of standard impact locations, or
        None for a row whose Target speed isn't recognized.
    """
    choice_band_vut_speeds = {
        vut
        for vut, target in zip(row_vut_speeds, row_target_speeds)
        if target == "20 km/h"
    }
    result = {}
    for row, (vut, target) in enumerate(zip(row_vut_speeds, row_target_speeds)):
        if target == "20 km/h":
            result[row] = frozenset({"25%"})
        elif target == "15 km/h":
            if vut in ("50 km/h", "60 km/h") and vut not in choice_band_vut_speeds:
                result[row] = frozenset({"50%", "25%"})
            else:
                result[row] = frozenset({"50%"})
        else:
            result[row] = None
    return result


def _check_extended_range_cbla(attributes):
    """
    Resolve CBLA's test range from test-point attributes instead of an
    absolute row index. CBLA's fixed band ("Target speed" 15 km/h) and choice
    band (20 km/h) are each identifiable by content, and within a band the
    standard column is the one whose "Impact location" matches the band's
    known standard value. This makes the check independent of how many rows
    the sheet has (e.g. a legacy layout with an extra low-speed row above the
    fixed band).

    When the test point carries a precomputed "_cbla_standard_ils" attribute
    (see classify_cbla_rows), that takes precedence -- it additionally
    accounts for a collapsed choice-band duplicate promoting the fixed row's
    25% IL cell, which the static per-Target-speed mapping alone can't know.

    Returns None (defer to the static EXTENDED_RANGE_CELLS fallback) if the
    attributes needed aren't present.
    """
    if not attributes:
        return None
    impact_location = attributes.get("Impact location")
    if impact_location is None:
        return None

    standard_ils = attributes.get("_cbla_standard_ils")
    if standard_ils is not None:
        return (
            TestRange.STANDARD
            if impact_location in standard_ils
            else TestRange.EXTENDED
        )

    standard_impact_location = CBLA_STANDARD_IMPACT_LOCATION_BY_TARGET_SPEED.get(
        attributes.get("Target speed")
    )
    if standard_impact_location is None:
        return None
    return (
        TestRange.STANDARD
        if impact_location == standard_impact_location
        else TestRange.EXTENDED
    )


def _check_extended_range_cpla(row, attributes, n_rows):
    """
    Resolve CPLA's test range from the row's band and its "Impact location"
    attribute.

    When the test point carries a precomputed "_cpla_row_info" attribute
    (see classify_cpla_rows), that takes precedence -- it resolves the band
    from the matrix's actual VUT speeds, which stays correct after a
    collapsed duplicate row and additionally accounts for the resulting
    promotion. Otherwise falls back to the legacy bottom-anchored position
    logic (both CPLA bands run at "Target speed" 5 km/h, so content alone
    can't tell them apart without the VUT-speed-based classification).

    Returns None (defer to the static EXTENDED_RANGE_CELLS fallback) if the
    attributes/row count needed aren't present.
    """
    if not attributes:
        return None
    impact_location = attributes.get("Impact location")
    if impact_location is None:
        return None

    row_info = attributes.get("_cpla_row_info")
    if row_info is not None:
        return (
            TestRange.STANDARD
            if impact_location in row_info.standard_ils
            else TestRange.EXTENDED
        )

    if not n_rows:
        return None
    band = "choice" if row >= n_rows - CPLA_CHOICE_BAND_ROW_COUNT else "fixed"
    standard_impact_location = CPLA_STANDARD_IMPACT_LOCATION_BY_BAND[band]
    return (
        TestRange.STANDARD
        if impact_location == standard_impact_location
        else TestRange.EXTENDED
    )


def check_extended_range(row, col, test_name, attributes=None, n_rows=None):
    """
    Check if the given row and column indices correspond to an extended range cell
    for the specified test name. Returns TestRange.

    CBLA and CPLA are resolved from test-point attributes (and, for CPLA,
    relative row position) rather than the static EXTENDED_RANGE_CELLS index
    map, since their sheets can be produced with a different row count across
    converter versions. All other tests keep using the static index map,
    since their layouts are fixed by the current template.
    """
    if test_name == "CBLA":
        range_from_attributes = _check_extended_range_cbla(attributes)
        if range_from_attributes is not None:
            return range_from_attributes
    elif test_name in ("CPLA day", "CPLA night"):
        range_from_attributes = _check_extended_range_cpla(row, attributes, n_rows)
        if range_from_attributes is not None:
            return range_from_attributes

    extended_range_cells = data_model.EXTENDED_RANGE_CELLS.get(test_name, [])
    if (row, col) in extended_range_cells:
        return TestRange.EXTENDED
    else:
        return TestRange.STANDARD


CELL_REMOVAL_MAP = {
    # Example: "TestName1": [(0, 1), (2, 3)]
    "CCFtap": [
        (0, 2),
        (1, 2),
        (2, 2),
        (3, 2),
        (0, 4),
        (1, 4),
        (2, 4),
        (3, 4),
    ],  # Remove column index 2 and 4 for all rows
    "CMFtap": [
        (0, 2),
        (1, 2),
        (2, 2),
        (3, 2),
        (0, 4),
        (1, 4),
        (2, 4),
        (3, 4),
    ],  # Remove column index 2 and 4 for all rows
    "CCCscp": [
        # Row 2
        (2, 5),
        (2, 6),
        # Row 3
        (3, 5),
        (3, 6),
        # Row 4
        (4, 5),
        (4, 6),
        # Row 5
        (5, 2),
        (5, 3),
        (5, 4),
        (5, 5),
        (5, 6),
        # Row 6
        (6, 2),
        (6, 3),
        (6, 4),
        (6, 5),
        (6, 6),
    ],
    "CMCscp": [
        # Row 2
        (2, 5),
        (2, 6),
        # Row 3
        (3, 5),
        (3, 6),
        # Row 4
        (4, 5),
        (4, 6),
        # Row 5
        (5, 2),
        (5, 3),
        (5, 4),
        (5, 5),
        (5, 6),
        # Row 6
        (6, 2),
        (6, 3),
        (6, 4),
        (6, 5),
        (6, 6),
    ],
    "CBTAfs & CBTAns": [
        # Col 0
        (4, 0),
        # Col 1
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        # Col 3
        (0, 3),
        (1, 3),
        (2, 3),
        (3, 3),
        (4, 3),
        # Col 4
        (0, 4),
        (1, 4),
        (2, 4),
        (3, 4),
        (4, 4),
        # Col 5
        (0, 5),
        (1, 5),
        (2, 5),
        (3, 5),
        (4, 5),
        # Col 6
        (0, 6),
        (1, 6),
        (2, 6),
        (3, 6),
    ],
    "CBTAfo & CBTAno": [
        (4, 0),
        (4, 4),
    ],
}


def filter_test_matrix_cells(test_matrix, test_name):
    """
    Remove specific cells from the test_matrix for a given test_name.

    Args:
        test_matrix (list of list): The matrix to filter (list of lists of PredictionColor or similar).
        test_name (str): The name of the test.

    Returns:
        list of list: A new matrix with specified cells set to None.
    """
    cells_to_remove = CELL_REMOVAL_MAP.get(test_name, [])
    if len(cells_to_remove) == 0:
        logger.debug(
            f"No cells to remove for test '{test_name}'. Returning original matrix."
        )
        return test_matrix

    filtered_matrix = [row.copy() for row in test_matrix if row is not None]
    for row_idx, col_idx in cells_to_remove:
        if 0 <= row_idx < len(filtered_matrix) and 0 <= col_idx < len(
            filtered_matrix[0]
        ):
            filtered_matrix[row_idx][col_idx] = None
    return filtered_matrix


def _extract_test_point_attributes(
    prediction_df: pd.DataFrame,
    test_start_row: int,
    test_start_col: int,
    row_i: int,
    col_j: int,
    test_name: str,
) -> dict:
    """
    Extract row/column label attributes for a single cell in a test matrix.

    Column attributes come from cells to the left of the matrix (VUT speed, etc.).
    The row attribute comes from the header row immediately above the data.

    Args:
        prediction_df: The predictions DataFrame.
        test_start_row: First data row of the matrix (0-based).
        test_start_col: First data column of the matrix (0-based).
        row_i: Row offset within the matrix (0-based).
        col_j: Column offset within the matrix (0-based).
        test_name: The test name (used for special-case column naming).

    Returns:
        A dict mapping attribute names to their values.
    """
    attributes = {}

    # --- Column attributes (cells to the left of the matrix) ---
    offset = 1
    col_idx = test_start_col - offset
    while col_idx >= 0:
        if col_idx == 0:
            if test_name == "CBNAO SfS":
                col_name = "d"
            elif test_name in ["CBDA", "CPMRCm"]:
                col_name = "Gap"
            elif test_name in ["CPMFC"]:
                col_name = "Distance"
            else:
                col_name = "VUT speed"
        else:
            target_row = test_start_row - 2
            col_name = str(prediction_df.iloc[target_row, col_idx])
            if test_start_row == 1:
                col_name = prediction_df.columns[col_idx]
        col_value = prediction_df.iloc[test_start_row + row_i, col_idx]
        if col_value is not None and not pd.isna(col_value):
            attributes[col_name] = col_value.strip()
        offset += 1
        col_idx = test_start_col - offset

    # --- Row attribute (header row above the matrix) ---
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
            for c in range(test_start_col, -1, -1):
                candidate = prediction_df.columns[c]
                if (
                    not pd.isna(candidate)
                    and str(candidate).strip() != "nan"
                    and "Unnamed" not in str(candidate).strip()
                ):
                    row_name = str(candidate)
                    break
        else:
            for c in range(test_start_col, -1, -1):
                candidate = prediction_df.iloc[test_start_row - 2, c]
                if not pd.isna(candidate) and str(candidate).strip() != "nan":
                    row_name = str(candidate)
                    break

    row_value = prediction_df.iloc[test_start_row - 1, test_start_col + col_j]
    if row_value is not None and not pd.isna(row_value):
        attributes[row_name] = row_value

    # --- Post-processing ---
    if "Impact location" in attributes:
        attributes["Impact location"] = (
            str(int(float(attributes["Impact location"]) * 100)) + "%"
        )
    if test_name in {"CCCscp", "CMCscp"}:
        target_approach = random.choice(["Farside", "Nearside"])
        logger.debug(
            f"Adding target approach for test '{test_name}' at row {row_i}, col {col_j}: {target_approach}"
        )
        attributes["Target approach"] = target_approach

    return attributes


def _extract_raw_matrix(
    prediction_df: pd.DataFrame,
    test_name: str,
    start_row: int,
    start_col: int,
    n_rows: int,
    n_cols: int,
) -> list:
    """
    Slice the prediction DataFrame into a raw string matrix and apply cell-removal filtering.

    Args:
        prediction_df: The predictions DataFrame.
        test_name: Used to look up cells to remove in CELL_REMOVAL_MAP.
        start_row: First data row (0-based).
        start_col: First data column (0-based).
        n_rows: Number of rows to include.
        n_cols: Number of columns to include.

    Returns:
        A list-of-lists of str values (or None for filtered-out cells).
    """
    raw = (
        prediction_df.iloc[
            start_row : start_row + n_rows, start_col : start_col + n_cols
        ]
        .reset_index(drop=True)
        .astype(str)
        .values.tolist()
    )
    return filter_test_matrix_cells(raw, test_name)


def get_test_matrix(
    prediction_df,
    test_name,
    stage_subelement_key,
    matrix_indices_dict=data_model.MATRIX_INDICES,
):
    """
    Process the predictions DataFrame to extract the test matrix for a given test name,
    convert values to PredictionColor enums, and create TestPoint objects with attributes.

    Args:
        prediction_df (pd.DataFrame): DataFrame containing predictions.
        test_name (str): The name of the test.
        stage_subelement_key (data_model.StageSubelementKey): The stage subelement key.
        matrix_indices_dict (dict): Dictionary with start_row, start_col, n_rows, n_cols per test.

    Returns:
        list[TestPoint] or None
    """
    if prediction_df.empty:
        logger.error("The input DataFrame is empty.")
        return None

    matrix_indices = get_matrix_indices(prediction_df, test_name, matrix_indices_dict)
    if not matrix_indices:
        logger.warning(f"Test name '{test_name}' not found in MATRIX_INDICES.")
        return None

    start_row = matrix_indices["start_row"]
    start_col = matrix_indices["start_col"]
    n_rows = common.get_n_rows(prediction_df, start_row, col_idx=0)
    n_cols = matrix_indices["n_cols"]

    return get_test_matrix_from_region(
        prediction_df,
        start_row,
        n_rows,
        start_col,
        n_cols,
        test_name=test_name,
        stage_subelement_key=stage_subelement_key,
    )


def get_test_matrix_from_region(
    prediction_df,
    start_row,
    n_rows,
    start_col,
    n_cols,
    test_name="",
    stage_subelement_key=None,
    attributes_df=None,
):
    """
    Process an explicitly defined prediction matrix region, convert values to
    PredictionColor enums, and create TestPoint objects with attributes.

    Args:
        prediction_df (pd.DataFrame): DataFrame containing predictions.
        start_row (int): First data row of the matrix.
        n_rows (int): Number of rows in the matrix.
        start_col (int): First data column of the matrix.
        n_cols (int): Number of columns in the matrix.
        test_name (str): Optional test name for filtering and special-case logic.
        stage_subelement_key (data_model.StageSubelementKey | None): Optional stage subelement key.
        attributes_df (pd.DataFrame | None): DataFrame to read row/column label
            attributes (Target speed, Impact location, ...) from. Callers that
            derive `prediction_df` from cell background color (which carries
            no text) should pass the plain-text prediction sheet here instead;
            defaults to `prediction_df` for callers where it already is one.

    Returns:
        list[TestPoint] or None
    """
    if prediction_df.empty:
        logger.error("The input DataFrame is empty.")
        return None

    if attributes_df is None:
        attributes_df = prediction_df

    logger.debug(
        f"Extracting test matrix for test '{test_name}' from region starting at row {start_row}, col {start_col} with size {n_rows}x{n_cols}."
    )
    test_matrix = _extract_raw_matrix(
        prediction_df, test_name, start_row, start_col, n_rows, n_cols
    )

    # Precompute CPLA/CBLA's content-based row classification once per
    # matrix (needs every row's VUT/Target speed at once), so
    # check_extended_range/is_fcw_declared_as_aeb can resolve it per test
    # point without needing the whole matrix in scope. See
    # classify_cpla_rows/classify_cbla_rows.
    cpla_row_info_by_row = None
    cbla_standard_ils_by_row = None
    if test_name in ("CPLA day", "CPLA night"):
        row_vut_speeds = [
            _extract_test_point_attributes(
                attributes_df, start_row, start_col, i, 0, test_name
            ).get("VUT speed")
            for i in range(len(test_matrix))
        ]
        cpla_row_info_by_row = classify_cpla_rows(row_vut_speeds)
    elif test_name == "CBLA":
        row_labels = [
            _extract_test_point_attributes(
                attributes_df, start_row, start_col, i, 0, test_name
            )
            for i in range(len(test_matrix))
        ]
        cbla_standard_ils_by_row = classify_cbla_rows(
            [label.get("VUT speed") for label in row_labels],
            [label.get("Target speed") for label in row_labels],
        )

    test_points = []

    for i in range(len(test_matrix)):
        for j in range(len(test_matrix[0])):
            if test_matrix[i][j] is None:
                continue
            color_str = test_matrix[i][j].lower()
            enum_val = next(
                (k for k in common.PredictionColor if k.value.lower() == color_str),
                common.PredictionColor.GREY,
            )
            test_matrix[i][j] = enum_val
            attributes = _extract_test_point_attributes(
                attributes_df, start_row, start_col, i, j, test_name
            )
            internal = {}
            if attributes:
                if cpla_row_info_by_row is not None:
                    row_info = cpla_row_info_by_row.get(i)
                    if row_info is not None:
                        internal["_cpla_row_info"] = row_info
                elif cbla_standard_ils_by_row is not None:
                    standard_ils = cbla_standard_ils_by_row.get(i)
                    if standard_ils is not None:
                        internal["_cbla_standard_ils"] = standard_ils
            if stage_subelement_key == data_model.StageSubelementKey.LSC:
                test_range = TestRange.STANDARD
            else:
                # check_extended_range's CPLA/CBLA resolution needs the
                # internal precomputed value, but it must not be persisted in
                # the public `attributes` dict below -- merge it into a
                # throwaway view for this lookup only.
                range_attributes = (
                    {**attributes, **internal} if internal else attributes
                )
                test_range = check_extended_range(
                    i,
                    j,
                    test_name,
                    attributes=range_attributes,
                    n_rows=len(test_matrix),
                )
            test_points.append(
                TestPoint(
                    row=i,
                    col=j,
                    color=enum_val,
                    test_range=test_range,
                    attributes=attributes,
                    internal=internal,
                    total_rows=len(test_matrix),
                )
            )

    for idx, row in enumerate(test_matrix):
        logger.debug("%d: %s", idx, row)

    if stage_subelement_key == data_model.StageSubelementKey.LSC:
        extended_range_cells = []
    elif test_name:
        extended_range_cells = data_model.EXTENDED_RANGE_CELLS.get(test_name, [])
    else:
        extended_range_cells = []

    if (
        HAS_MPL
        and settings.enable_ca_matrix_plots
        and logger.isEnabledFor(logging.DEBUG)
    ):
        logger.debug("Plotting matrix for test: %s", test_name)
        plot_matrix(test_matrix, extended_range_cells, name=test_name)
    return test_points


def get_matrix_indices(
    prediction_df,
    test_name,
    matrix_indices_dict=data_model.MATRIX_INDICES,
    start_row_lookup: dict | None = None,
    matrix_indices_cache: dict | None = None,
):
    """
    Get the matrix indices for a given test name from the matrix_indices_dict.

    Args:
        prediction_df (pd.DataFrame): The DataFrame containing the prediction data.
        test_name (str): The name of the test.
        matrix_indices_dict (dict): Dictionary containing the start row, start column, and number of rows and columns for each test in the predictions DataFrame.

    Returns:
        dict: A dictionary with keys 'start_row', 'start_col', 'n_rows', 'n_cols' if test_name is found, else None.
    """
    cached_indices = None
    if matrix_indices_cache is not None:
        cached_indices = matrix_indices_cache.get(test_name)
    if cached_indices is not None:
        return dict(cached_indices)

    base_matrix_indices = matrix_indices_dict.get(test_name, None)
    if not base_matrix_indices:
        return None

    matrix_indices = dict(base_matrix_indices)

    if start_row_lookup is None:
        start_row = common.get_start_row(prediction_df, test_name)
    else:
        start_row = start_row_lookup.get(_normalize_lookup_value(test_name))

    if start_row is None:
        logger.error(
            f"Start row for test '{test_name}' not found in prediction DataFrame."
        )
        return None
    matrix_indices["start_row"] = start_row
    matrix_indices["n_rows"] = common.get_n_rows(
        prediction_df,
        matrix_indices["start_row"],
        col_idx=0,
    )

    if matrix_indices_cache is not None:
        matrix_indices_cache[test_name] = dict(matrix_indices)

    return matrix_indices


def _is_na_or_blank(value) -> bool:
    """True for an explicit "N/A" selection or an unfilled cell in the Input
    parameters Value column. Delegates to the shared helpers so every null
    flavour (None, float NaN, pd.NA, empty string) classifies the same way
    across modules."""
    return common.is_empty_cell(value) or common.is_not_applicable(value)


def get_prediction_methods(input_parameters_df, test_name):
    # Find the row where "Scenario" equals test_name
    logger.info(f"Test name: {test_name}")
    start_idx = input_parameters_df.index[input_parameters_df["Scenario"] == test_name]
    if len(start_idx) == 0:
        logger.error(
            f"Test name '{test_name}' not found in input_parameters_df['Scenario']."
        )
        return None, None
    start_idx = start_idx[0]

    # Find the next row where "Scenario" is not NaN after start_idx
    next_indices = input_parameters_df.index[
        (input_parameters_df.index > start_idx)
        & (input_parameters_df["Scenario"].notna())
    ]
    end_idx = next_indices[0] if len(next_indices) > 0 else len(input_parameters_df)

    # Get the relevant rows for this test_name
    input_params_rows = input_parameters_df.iloc[start_idx:end_idx]
    input_params_dict = dict(
        zip(
            input_params_rows["Input parameter"].astype(str),
            input_params_rows["Value"],
        )
    )
    logger.info(f"Input parameters for test '{test_name}': {input_params_dict}")

    standard_pred_raw = input_params_dict.get("Prediction - Standard")
    extended_pred_raw = input_params_dict.get("Prediction - Extended")

    # An "N/A" in ANY of the scenario's input parameters (the prediction
    # methods, or e.g. ELK RE's "Extended range performance") means the
    # scenario is not claimed: the caller scores all three layers
    # (Standard/Extended/Robustness) 0. The scenario_na_consistency check
    # requires all of a scenario's parameters to be N/A together; compute is
    # defensive and zeroes on any of them. A blank prediction method is
    # treated the same way (it previously crashed here on .strip() of NaN).
    # Genuinely unknown non-blank values still raise below.
    if (
        any(common.is_not_applicable(v) for v in input_params_dict.values())
        or _is_na_or_blank(standard_pred_raw)
        or _is_na_or_blank(extended_pred_raw)
    ):
        logger.info(
            f"Input parameters for test '{test_name}' contain N/A (or a blank "
            f"prediction method): {input_params_dict}; the scenario is not "
            "claimed and scores 0."
        )
        return None, None

    standard_pred_value = str(standard_pred_raw).strip().lower()
    extended_pred_value = str(extended_pred_raw).strip().lower()

    standard_oem_prediction_method = PredictionSource.UNKNOWN
    if standard_pred_value == "vta":
        standard_oem_prediction_method = PredictionSource.VTA
    elif standard_pred_value == "self claimed":
        standard_oem_prediction_method = PredictionSource.SELF_CLAIMED

    extended_oem_prediction_method = PredictionSource.UNKNOWN
    if extended_pred_value == "vta":
        extended_oem_prediction_method = PredictionSource.VTA
    elif extended_pred_value == "self claimed":
        extended_oem_prediction_method = PredictionSource.SELF_CLAIMED
    logger.info(f"Standard OEM prediction method: {standard_oem_prediction_method}")
    logger.info(f"Extended OEM prediction method: {extended_oem_prediction_method}")

    if (
        standard_oem_prediction_method == PredictionSource.UNKNOWN
        or extended_oem_prediction_method == PredictionSource.UNKNOWN
    ):
        raise ValueError(
            f"OEM prediction method is 'Unknown' for test '{test_name}'. Please specify both standard and extended OEM prediction methods."
        )

    return (
        standard_oem_prediction_method,
        extended_oem_prediction_method,
    )
