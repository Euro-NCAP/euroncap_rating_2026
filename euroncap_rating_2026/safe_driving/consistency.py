# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Report-only prediction-consistency rules for the safe_driving domain.

- stale_acc_cut_out: the "VA - ACC pred." cut-out matrices carry speeds from a
  superseded template layout. A consuming application matches a test run to a
  prediction row by those very speeds, so a superseded sheet matches nothing,
  its Value cells stay blank, and a blank Value inherits the OEM's prediction
  at full credit (acc_performance.compute_score). Nothing else in the pipeline
  notices -- the version stamp cannot, since every layout stamped 5.4.5.
"""

import logging

from euroncap_rating_2026 import common

logger = logging.getLogger(__name__)

ACC_PREDICTION_SHEET = "VA - ACC pred."
# (VUT speed, Target speed) of each cut-out test point, in matrix row order --
# Safe Driving - Vehicle Assistance v1.2, pp. 17 (Car-to-Car) and 21
# (Car-to-Motorcyclist). "Target speed" is the lead vehicle (SOV) that cuts out
# to reveal the standstill GVT/EMT; see
# acc_performance.get_green_v_reduction_threshold.
_EXPECTED_CUT_OUT_ROWS = (("70 km/h", "50 km/h"), ("90 km/h", "70 km/h"))
_CUT_OUT_MATRICES = ("CCR cut-out", "CMR cut-out")


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


def find_prediction_inconsistencies(
    wb, stage_element: str = None, stage_subelement: str = None
) -> list:
    """Every prediction-rule violation in *wb* for the safe_driving domain, as
    a list of common.ConsistencyFinding. Each rule applies its own (Stage
    element, Stage subelement) scope guard, so adding one never narrows
    another; scope validity is the caller's concern
    (integrity.find_consistency_findings validates it once).
    """
    return _find_stale_acc_cut_out_rows(wb, stage_element, stage_subelement)
