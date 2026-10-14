# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Report-only consistency rules for the crash_protection domain.

One rule on "Input parameters":

- rear_torso_angle: the second-row torso angle must be a number and the
  third-row one a number or "N/A" (see find_rear_torso_angle_findings).

One rule on "CP - VRU Prediction":

- no_prediction_provided: the Headforms/Upper legform/aPLI sections have no
  per-cell required-input check (the OEM fills only the cells matching the
  car's layout and its own prediction),
  so nothing else guarantees the OEM entered *any* prediction there. This
  rule reports a section that is entirely untouched (every cell GREY).

There is deliberately no legform T/ST/SA symmetry rule: Euro NCAP accepts
asymmetric legform predictions.
"""

import logging

import numpy as np

from euroncap_rating_2026 import common
from euroncap_rating_2026.crash_protection import vru_processing

logger = logging.getLogger(__name__)

VRU_PREDICTION_SHEET = "CP - VRU Prediction"

# The legform matrix rows, top to bottom, with the template section each row
# renders under and the (Stage element, Stage subelement) scope it belongs
# to. Only the scope tuples are read; the section labels are documentation.
# Kept per-row -- not per-block -- because consuming applications check
# Pelvis Impact and Leg Impact as two separate scopes whose results they
# merge: row 0 belongs to Pelvis Impact, rows 1/2 to Leg Impact (checked
# together -- either aPLI row having a prediction satisfies the element).
# The combined "Pelvis & Leg Impact" alias (Test Scores' vocabulary, accepted
# by integrity.VALID_SCOPE_PAIRS) matches every row.
_LEGFORM_ROW_SCOPES = (
    (
        "Upper legform",
        (("VRU Impact", "Pelvis Impact"), ("VRU Impact", "Pelvis & Leg Impact")),
    ),
    (
        "aPLI - Femur",
        (("VRU Impact", "Leg Impact"), ("VRU Impact", "Pelvis & Leg Impact")),
    ),
    (
        "aPLI - Knee & Tibia",
        (("VRU Impact", "Leg Impact"), ("VRU Impact", "Pelvis & Leg Impact")),
    ),
)

_HEADFORM_SCOPE = (("VRU Impact", "Head Impact"),)

# Column-A section-header row for "Headforms", confirmed against the
# packaged template (cp_template.xlsx: "Headforms" literally sits in A1).
# Used only to anchor a no_prediction_provided finding, which has no single
# offending data cell to point at.
_HEADFORMS_HEADER_ROW = 1


def _legform_row(row_idx: int) -> int:
    """Worksheet row of legform matrix row row_idx. This is also the row
    column A's own section header (Upper legform/aPLI - Femur/aPLI - Knee &
    Tibia) sits on -- confirmed against the packaged template -- so it
    doubles as the anchor for a no_prediction_provided finding.

    get_legform_matrix slices vru_prediction_df.iloc[LEGFORMS_START_ROW_INDEX:
    ...][2:, ...]: matrix row 0 is DataFrame row LEGFORMS_START_ROW_INDEX+2;
    the trailing +2 converts the 0-based, header-stripped DataFrame index to
    a 1-based worksheet row.
    """
    return vru_processing.LEGFORMS_START_ROW_INDEX + 2 + row_idx + 2


def _section_has_prediction(matrix_rows) -> bool:
    """Whether any cell across the given legform/headform matrix row(s)
    carries a real prediction -- anything other than GREY. GREY is the
    literal "nothing entered" value throughout this sheet (see
    vru_processing.VruPredictionColor); other excluded-from-sampling colors
    like D_RED/D_GREEN are still real OEM predictions, just not
    verification-test candidates, so they must count here."""
    return any(
        color != vru_processing.VruPredictionColor.GREY
        for row in matrix_rows
        for color in row
    )


def _no_prediction_finding(
    section_label: str, ws_row: int, expected: str
) -> "common.ConsistencyFinding":
    cell = f"A{ws_row}"
    return common.ConsistencyFinding(
        sheet=VRU_PREDICTION_SHEET,
        column="A",
        cell=cell,
        row_identity={"Section": section_label},
        check=common.CHECK_NO_PREDICTION_PROVIDED,
        message=(
            f"{VRU_PREDICTION_SHEET} — {section_label}: no prediction provided; "
            f"at least one cell in this section must have a {expected} prediction."
        ),
    )


def find_prediction_inconsistencies(
    wb, stage_element: str = None, stage_subelement: str = None
) -> list:
    """Every prediction-rule violation in *wb* for the crash_protection
    domain, as a list of common.ConsistencyFinding (check =
    no_prediction_provided). Report-only. Scope filtering follows
    find_missing_required_inputs' (Stage element, Stage subelement)
    convention; scope validity is the caller's concern
    (integrity.find_consistency_findings validates it once).
    """
    if VRU_PREDICTION_SHEET not in wb.sheetnames:
        return []

    df = common.sheet_to_dataframe(wb[VRU_PREDICTION_SHEET])
    if df.empty:
        return []

    findings = []

    legform_matrix = vru_processing.get_legform_matrix(df)
    if legform_matrix and legform_matrix[0]:
        if common.stage_elements_in_scope(
            list(_LEGFORM_ROW_SCOPES[0][1]), stage_element, stage_subelement
        ) and not _section_has_prediction([legform_matrix[0]]):
            findings.append(
                _no_prediction_finding(
                    "Upper legform (Pelvis Impact)", _legform_row(0), "T, ST or SA"
                )
            )

        if (
            len(legform_matrix) >= 3
            and common.stage_elements_in_scope(
                list(_LEGFORM_ROW_SCOPES[1][1]), stage_element, stage_subelement
            )
            and not _section_has_prediction(legform_matrix[1:3])
        ):
            findings.append(
                _no_prediction_finding(
                    "aPLI - Femur / aPLI - Knee & Tibia (Leg Impact)",
                    _legform_row(1),
                    "T, ST or SA",
                )
            )

    if common.stage_elements_in_scope(
        list(_HEADFORM_SCOPE), stage_element, stage_subelement
    ):
        headform_matrix = vru_processing.get_headform_matrix(df)
        if (
            headform_matrix
            and headform_matrix[0]
            and not _section_has_prediction(headform_matrix)
        ):
            findings.append(
                _no_prediction_finding("Headforms", _HEADFORMS_HEADER_ROW, "color")
            )

    return findings


INPUT_PARAMETERS_SHEET = "Input parameters"
_REAR_WHIPLASH_SCOPE = ("Rear Impact", "Whiplash")
# Current spelling first, then the previous template revision's.
_SECOND_ROW_TORSO_ANGLE_NAMES = ("Torso angle (rear second row)", "Torso angle (rear)")
_THIRD_ROW_TORSO_ANGLE_NAME = "Torso angle (rear third row)"


def _is_number(value) -> bool:
    """A finite number, as a numeric cell or as text (what the torso-angle
    limit can actually be computed from)."""
    if isinstance(value, bool):
        return False
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return False
    if isinstance(value, (int, float)):
        return bool(np.isfinite(value))
    return False


def find_rear_torso_angle_findings(
    wb, stage_element: str = None, stage_subelement: str = None
) -> list:
    """rear_torso_angle: the two Rear Impact torso-angle input parameters
    must hold what their seat row needs. The second row is always assessed,
    so its angle must be a number; the third row's angle must be a number
    (the vehicle has a third row) or "N/A" (it has none) -- the two answers
    score differently, so anything else is reported. Blank cells are left to
    the missing-input check, which already reports them.

    Rows are looked up in the user's own workbook, so a file produced
    against the previous template revision (no third-row parameter) is
    simply never asked about it."""
    if INPUT_PARAMETERS_SHEET not in wb.sheetnames:
        return []
    if stage_element is not None and (stage_element, stage_subelement) not in (
        _REAR_WHIPLASH_SCOPE,
        (_REAR_WHIPLASH_SCOPE[0], None),
    ):
        return []
    ws = wb[INPUT_PARAMETERS_SHEET]
    header = {cell.value: cell.column for cell in ws[1]}
    needed = ("Stage element", "Stage subelement", "Input parameter", "Value")
    if any(name not in header for name in needed):
        return []

    findings = []
    element = subelement = None
    for row in range(2, ws.max_row + 1):
        raw_element = ws.cell(row=row, column=header["Stage element"]).value
        raw_subelement = ws.cell(row=row, column=header["Stage subelement"]).value
        if not common.is_empty_cell(raw_element):
            element = raw_element
        if not common.is_empty_cell(raw_subelement):
            subelement = raw_subelement
        if (element, subelement) != _REAR_WHIPLASH_SCOPE:
            continue
        parameter = ws.cell(row=row, column=header["Input parameter"]).value
        value_cell = ws.cell(row=row, column=header["Value"])
        value = value_cell.value
        if common.is_empty_cell(value):
            continue
        if parameter in _SECOND_ROW_TORSO_ANGLE_NAMES:
            if _is_number(value):
                continue
            expected = "a number (the second row is always assessed)"
        elif parameter == _THIRD_ROW_TORSO_ANGLE_NAME:
            if _is_number(value) or common.is_not_applicable(value):
                continue
            expected = "a number, or N/A when the vehicle has no third row"
        else:
            continue
        findings.append(
            common.ConsistencyFinding(
                sheet=INPUT_PARAMETERS_SHEET,
                column="Value",
                cell=value_cell.coordinate,
                row_identity={
                    "Stage element": element,
                    "Stage subelement": subelement,
                    "Input parameter": parameter,
                },
                check=common.CHECK_REAR_TORSO_ANGLE,
                message=(
                    f"{INPUT_PARAMETERS_SHEET} — {element} / {subelement} / "
                    f"{parameter}: '{value}' is not valid; expected {expected}."
                ),
            )
        )
    return findings
