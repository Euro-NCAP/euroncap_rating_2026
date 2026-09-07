# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Report-only prediction-consistency rules for the safe_driving domain.

Both rules read the same rule this module's grouping enforces
(driver_monitoring.apply_prediction_consistency, which preprocess runs by
silently forcing every cell of a mixed (Task, Movement type, Vehicle
response) group to RED). This module runs the same grouping
(driver_monitoring.iter_prediction_groups) on the raw uploaded workbook and
reports, per group, what the OEM should know before the sheet is locked:

- dm_group_consistency: an *explicit* Green-vs-Red contradiction -- the
  Green cells will be scored as Red.
- dm_na_forced_red: an explicit N/A (GREY) cell sharing a group with a Red
  cell -- that N/A cell will also be scored Red, silently, at preprocessing.

A required cell left blank (no explicit color, no N/A text) is the
missing-input check's finding, not either rule here.

- stale_acc_cut_out: the "VA - ACC pred." cut-out matrices carry speeds from a
  superseded template layout. Unlike the DM rules this one is not about what
  the OEM entered but about which template they entered it in: a consuming
  application matches a test run to a prediction row by those very speeds, so a superseded
  sheet matches nothing, its Value cells stay blank, and a blank Value inherits
  the OEM's prediction at full credit (acc_performance.compute_score). Nothing
  else in the pipeline notices -- the version stamp cannot, since every layout
  stamped 5.4.5.
"""

import logging

from openpyxl.utils import get_column_letter

from euroncap_rating_2026 import common
from euroncap_rating_2026.safe_driving import driver_monitoring

logger = logging.getLogger(__name__)

DM_PREDICTION_SHEET = "DE - DM pred."

_DM_TABLES = ("Long distraction", "Short distraction", "Phone use", "Non-transient")

ACC_PREDICTION_SHEET = "VA - ACC pred."
# (VUT speed, Target speed) of each cut-out test point, in matrix row order --
# Safe Driving - Vehicle Assistance v1.2, pp. 17 (Car-to-Car) and 21
# (Car-to-Motorcyclist). "Target speed" is the lead vehicle (SOV) that cuts out
# to reveal the standstill GVT/EMT; see
# acc_performance.get_green_v_reduction_threshold.
_EXPECTED_CUT_OUT_ROWS = (("70 km/h", "50 km/h"), ("90 km/h", "70 km/h"))
_CUT_OUT_MATRICES = ("CCR cut-out", "CMR cut-out")


def _dm_cell(coords) -> str:
    """Worksheet coordinate for a TableProcessor.cell_dict key: row_idx 0 is
    Excel row 2, col_idx 0 is Excel column A (the convention documented on
    driver_monitoring.get_prediction_forced_red_coordinates)."""
    row_idx, col_idx = coords
    return f"{get_column_letter(col_idx + 1)}{row_idx + 2}"


def _cell_text(value) -> str:
    """A label cell as a comparable string: None becomes "", and internal
    whitespace is collapsed so a stray double space in "70  km/h" doesn't read
    as a superseded template."""
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _find_stale_acc_cut_out_rows(wb, stage_element, stage_subelement) -> list:
    """One finding per cut-out matrix whose (VUT speed, Target speed) pairs are
    not the protocol's. Locates each matrix by its label in column A rather
    than by a fixed row, so inserting rows elsewhere in the sheet can't
    silently disable the check."""
    if not common.stage_elements_in_scope(
        [("Vehicle Assistance", "ACC performance")], stage_element, stage_subelement
    ):
        return []
    if ACC_PREDICTION_SHEET not in wb.sheetnames:
        return []

    ws = wb[ACC_PREDICTION_SHEET]
    findings = []
    for label in _CUT_OUT_MATRICES:
        header_row = next(
            (
                row
                for row in range(1, ws.max_row + 1)
                if _cell_text(ws.cell(row=row, column=1).value) == label
            ),
            None,
        )
        if header_row is None:
            # A missing matrix is a structural problem the integrity layer
            # reports; not this rule's finding.
            continue

        # The header row carries the column labels, the row after it the
        # impact locations, then one row per test point.
        first_point_row = header_row + 2
        found = tuple(
            (
                _cell_text(ws.cell(row=row, column=1).value),
                _cell_text(ws.cell(row=row, column=2).value),
            )
            for row in range(
                first_point_row, first_point_row + len(_EXPECTED_CUT_OUT_ROWS)
            )
        )
        if found == _EXPECTED_CUT_OUT_ROWS:
            continue

        expected_desc = ", ".join(
            f"{vut} / {target}" for vut, target in _EXPECTED_CUT_OUT_ROWS
        )
        found_desc = ", ".join(
            f"{vut or '(empty)'} / {target or '(empty)'}" for vut, target in found
        )
        findings.append(
            common.ConsistencyFinding(
                sheet=ACC_PREDICTION_SHEET,
                column=label,
                cell=f"A{first_point_row}",
                row_identity={"Scenario": label},
                check=common.CHECK_STALE_ACC_CUT_OUT,
                message=(
                    f"{ACC_PREDICTION_SHEET} — {label}: the VUT speed / Target "
                    f"speed pairs are {found_desc}, but this protocol version "
                    f"expects {expected_desc}. This prediction sheet comes from "
                    f"a superseded template, so its test results cannot be "
                    f"matched and the predictions would be scored as if no data "
                    f"had been uploaded. Please re-enter the predictions in a "
                    f"freshly generated template."
                ),
            )
        )
    return findings


def _find_dm_prediction_inconsistencies(
    wb, stage_element: str = None, stage_subelement: str = None
) -> list:
    """The Driver monitoring prediction-rule violations in *wb* (check =
    dm_group_consistency or dm_na_forced_red). Report-only: preprocess keeps
    its silent forced-red rewrite; this only tells the OEM about it up front.
    """
    if not common.stage_elements_in_scope(
        [("Driver Engagement", "Driver monitoring")], stage_element, stage_subelement
    ):
        return []
    if DM_PREDICTION_SHEET not in wb.sheetnames:
        return []

    dm_prediction_df = common.sheet_to_dataframe(wb[DM_PREDICTION_SHEET])
    if dm_prediction_df.empty:
        return []

    # No required-mask here on purpose: without it a blank cell never counts
    # as RED, so the rule below sees only what the OEM explicitly entered.
    table_processors = {
        name: driver_monitoring.TableProcessor(dm_prediction_df, name)
        for name in _DM_TABLES
    }

    findings = []
    for (
        name,
        tp,
        (task, movement_type, vehicle_response),
        coords_list,
    ) in driver_monitoring.iter_prediction_groups(table_processors):
        red = sorted(c for c in coords_list if tp.cell_dict[c].get("Color") == "RED")
        green = sorted(
            c for c in coords_list if tp.cell_dict[c].get("Color") == "GREEN"
        )
        grey = sorted(c for c in coords_list if tp.cell_dict[c].get("Color") == "GREY")
        group_desc = (
            f"(Task={task}, Movement type={movement_type}, "
            f"Vehicle response={vehicle_response})"
        )
        row_identity = {
            "Table": name,
            "Task": task,
            "Movement type": movement_type,
            "Vehicle response": vehicle_response,
        }

        if red and green:
            red_cells = ", ".join(_dm_cell(c) for c in red)
            green_cells = ", ".join(_dm_cell(c) for c in green)
            findings.append(
                common.ConsistencyFinding(
                    sheet=DM_PREDICTION_SHEET,
                    column=str(vehicle_response),
                    cell=_dm_cell(green[0]),
                    row_identity=row_identity,
                    check=common.CHECK_DM_GROUP_CONSISTENCY,
                    message=(
                        f"{DM_PREDICTION_SHEET} — {name}: the {group_desc} "
                        f"group mixes Green ({green_cells}) and Red "
                        f"({red_cells}) predictions; mixed predictions are "
                        f"not accepted within the same group."
                    ),
                )
            )

        if red and grey:
            red_cells = ", ".join(_dm_cell(c) for c in red)
            grey_cells = ", ".join(_dm_cell(c) for c in grey)
            findings.append(
                common.ConsistencyFinding(
                    sheet=DM_PREDICTION_SHEET,
                    column=str(vehicle_response),
                    cell=_dm_cell(grey[0]),
                    row_identity=row_identity,
                    check=common.CHECK_DM_NA_FORCED_RED,
                    message=(
                        f"{DM_PREDICTION_SHEET} — {name}: the {group_desc} "
                        f"group has a Red prediction ({red_cells}); if one "
                        f"cell in a group is Red, every cell in that group "
                        f"is scored Red, including N/A -- so N/A cell(s) "
                        f"({grey_cells}) will also be scored Red at "
                        f"preprocessing. Set them to Red to match."
                    ),
                )
            )
    return findings


def find_prediction_inconsistencies(
    wb, stage_element: str = None, stage_subelement: str = None
) -> list:
    """Every prediction-rule violation in *wb* for the safe_driving domain, as
    a list of common.ConsistencyFinding. Each rule applies its own (Stage
    element, Stage subelement) scope guard, so adding one never narrows
    another; scope validity is the caller's concern
    (integrity.find_consistency_findings validates it once).
    """
    findings = _find_dm_prediction_inconsistencies(wb, stage_element, stage_subelement)
    findings.extend(_find_stale_acc_cut_out_rows(wb, stage_element, stage_subelement))
    return findings
