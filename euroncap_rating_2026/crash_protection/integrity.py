# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Grey-cell trust boundary for the crash_protection domain.

Declares, per sheet, which columns are legitimate OEM input ("grey" cells --
OEM Prediction, Value, Inspection [%]) versus protected reference/structural
data (HPL, LPL, Capping, Max score, row labels) that must always come from
the official packaged template rather than a user-supplied file. See
`euroncap_rating_2026.common` for the underlying mechanism.

NOTE: "Documentation"/"Data" are listed in PASSTHROUGH_SHEETS -- there is no
computed logic for this domain yet (unlike crash_avoidance's
documentation_data.py), but they're still not schema'd: nothing to protect
until a domain-specific check exists.
"""

import functools
import os
import tempfile
from importlib.resources import files

import openpyxl

from euroncap_rating_2026 import common
from euroncap_rating_2026.crash_protection import consistency

_CRITERIA_IDENTITY_COLUMNS = [
    "Loadcase",
    "Seat position",
    "Dummy",
    "Body region",
    "Criteria",
]

SHEET_SCHEMAS = {
    "Input parameters": common.GreyCellSheetSchema(
        identity_columns=[
            "Stage",
            "Stage element",
            "Stage subelement",
            "Input parameter",
        ],
        grey_columns=["Value"],
    ),
    "Test Scores": common.GreyCellSheetSchema(
        identity_columns=["Stage", "Stage element", "Stage subelement"],
        grey_columns=["Inspection [%]"],
        # Inspection [%] is filled in by the inspector once physical/VT
        # testing has happened, not by the OEM.
        prediction_required_columns=[],
        computed_columns=["Score"],
    ),
    "CP - Body region scores": common.GreyCellSheetSchema(
        identity_columns=[
            "Stage element",
            "Stage subelement",
            "Loadcase",
            "Seat position",
            "Dummy",
            "Body region",
        ],
        grey_columns=["Inspection [%]"],
        prediction_required_columns=[],
        computed_columns=["Body regionscore", "Modifiers", "Score"],
    ),
    "CP - Dummy Scores": common.GreyCellSheetSchema(
        identity_columns=[
            "Stage",
            "Stage element",
            "Stage subelement",
            "Loadcase",
            "Seat position",
            "Dummy",
        ],
        grey_columns=[],
        computed_columns=["Capping?", "Score"],
    ),
    "CP - Frontal Offset": common.GreyCellSheetSchema(
        identity_columns=_CRITERIA_IDENTITY_COLUMNS,
        grey_columns=["OEM Prediction", "Value"],
        # Value is the measured result: dropped and re-merged from the
        # physical test's calculated KPIs at scoring time, so it can't be
        # filled by the OEM (the test hasn't run yet) -- only their own
        # predicted outcome is required here.
        prediction_required_columns=["OEM Prediction"],
        # ... and only on sliding-scale criteria rows: a color prediction
        # exists only where the criterion has both a higher and a lower
        # performance limit. Modifier and single-limit/capping rows
        # (Ares, DAMAGE, Shoulder belt load, Pedal displacement, ...)
        # have no predictable color, and scoring treats their OEM
        # Prediction as optional -- exactly the rows the template leaves
        # ungreyed. Gated on the pristine template's own HPL/LPL cells.
        required_when_columns_present=["HPL", "LPL"],
        applies_to_stage_elements=[("Frontal Impact", "Offset")],
    ),
    "CP - Frontal FW": common.GreyCellSheetSchema(
        identity_columns=_CRITERIA_IDENTITY_COLUMNS,
        grey_columns=["OEM Prediction", "Value"],
        prediction_required_columns=["OEM Prediction"],
        required_when_columns_present=["HPL", "LPL"],
        applies_to_stage_elements=[("Frontal Impact", "FW")],
    ),
    "CP - Frontal Sled & VT": common.GreyCellSheetSchema(
        identity_columns=_CRITERIA_IDENTITY_COLUMNS,
        grey_columns=["Value"],
        # No "OEM Prediction" column on this sheet at all -- Value is
        # measured-result-only here too, so nothing on this sheet is
        # OEM-fillable at prediction time.
        prediction_required_columns=[],
        applies_to_stage_elements=[("Frontal Impact", "Sled & VT")],
    ),
    "CP - Side MDB": common.GreyCellSheetSchema(
        identity_columns=_CRITERIA_IDENTITY_COLUMNS,
        grey_columns=["OEM Prediction", "Value"],
        prediction_required_columns=["OEM Prediction"],
        required_when_columns_present=["HPL", "LPL"],
        applies_to_stage_elements=[("Side Impact", "MDB")],
    ),
    "CP - Side Pole": common.GreyCellSheetSchema(
        identity_columns=_CRITERIA_IDENTITY_COLUMNS,
        grey_columns=["OEM Prediction", "Value"],
        prediction_required_columns=["OEM Prediction"],
        required_when_columns_present=["HPL", "LPL"],
        applies_to_stage_elements=[("Side Impact", "Pole")],
    ),
    "CP - Side Farside": common.GreyCellSheetSchema(
        identity_columns=_CRITERIA_IDENTITY_COLUMNS,
        grey_columns=["Value"],
        prediction_required_columns=[],
        applies_to_stage_elements=[("Side Impact", "Farside")],
    ),
    "CP - Rear Whiplash": common.GreyCellSheetSchema(
        identity_columns=_CRITERIA_IDENTITY_COLUMNS,
        grey_columns=["Value"],
        prediction_required_columns=[],
        applies_to_stage_elements=[("Rear Impact", "Whiplash")],
    ),
}


@functools.lru_cache(maxsize=1)
def _pristine_wb():
    """The packaged pristine template as an openpyxl Workbook, cached for
    the lifetime of the process -- backs find_missing_required_inputs'
    HPL/LPL required-ness gate and the VRU grid check, both of which read
    the pristine file rather than the user's own workbook.
    """
    pristine_path = str(files("data").joinpath(PRISTINE_TEMPLATE_NAME))
    return openpyxl.load_workbook(pristine_path)


# "CP - VRU Prediction" has no forward-fillable identity columns for
# GreyCellSheetSchema, but -- like crash_avoidance's grid sheets -- it
# declares its OEM-input cells via Data Validation dropdown ranges authored
# directly in the packaged template (the color grid and the legform T/ST/SA
# grid). Unlike those sheets it hosts several protocol elements stacked
# vertically, so each is anchored on its section's own header text in
# column A (see common.rows_in_sections) rather than the whole sheet.
#
# The legform block splits along the same line as
# vru_processing.extract_legform_candidate_points: the "Upper legform" row
# is the Pelvis Impact element, the two aPLI rows are Leg Impact -- these
# are also the names the "Input parameters" sheet uses, so each element's
# scope covers its input-parameter row plus its matrix cells. "Pelvis &
# Leg Impact" (the combined name "Test Scores" uses for the shared score)
# is kept as an alias for the whole legform matrix -- a valid scope must
# never silently match nothing -- but note it does NOT pull in the two
# split-named "Input parameters" rows.
#
# Every section must stay declared here even when it is not
# required-checked: rows_in_sections bounds each section at the next
# declared header, so dropping one would silently absorb its rows into the
# section above.
GRID_SHEET_SECTION_STAGE_ELEMENTS = {
    "CP - VRU Prediction": {
        "Headforms": [("VRU Impact", "Head Impact")],
        "Upper legform": [
            ("VRU Impact", "Pelvis Impact"),
            ("VRU Impact", "Pelvis & Leg Impact"),
        ],
        "aPLI - Femur": [
            ("VRU Impact", "Leg Impact"),
            ("VRU Impact", "Pelvis & Leg Impact"),
        ],
        "aPLI - Knee & Tibia": [
            ("VRU Impact", "Leg Impact"),
            ("VRU Impact", "Pelvis & Leg Impact"),
        ],
    },
}

# Sections exempt from the missing-required-input check: the OEM chooses which legform
# and headform prediction cells to fill based on the car's layout and its
# own prediction, so an empty cell there is a legitimate submission, never
# a missing input. What the OEM *does* fill is still validated -- the
# T/ST/SA symmetry rule reports through
# consistency.find_prediction_inconsistencies (legform_symmetry findings),
# and preprocess keeps rejecting asymmetric patterns -- and every exempt
# section must still get at least one prediction somewhere in it
# (consistency.find_prediction_inconsistencies' no_prediction_provided
# findings) now that no individual cell is required. The Pelvis/Leg
# "Input parameters" rows stay required via the schema'd sheets.
GRID_SHEET_EXEMPT_SECTIONS = {
    "CP - VRU Prediction": frozenset(
        {"Headforms", "Upper legform", "aPLI - Femur", "aPLI - Knee & Tibia"}
    ),
}


# Every (Stage element, Stage subelement) pair the packaged template's own
# sheets use -- the literal strings find_missing_required_inputs matches
# against, kept here so an unknown scope raises instead of silently
# reporting zero findings (common.validate_scope). This domain has no
# data_model.py to derive them from. Note the template itself carries two
# vocabularies for the legform elements: "Input parameters" splits them
# into "Pelvis Impact"/"Leg Impact" while "Test Scores" (and the VRU grid
# wiring above, which follows it) uses the combined "Pelvis & Leg Impact"
# -- all three are accepted.
VALID_SCOPE_PAIRS = [
    ("Frontal Impact", "Offset"),
    ("Frontal Impact", "FW"),
    ("Frontal Impact", "Sled & VT"),
    ("Side Impact", "MDB"),
    ("Side Impact", "Pole"),
    ("Side Impact", "Farside"),
    ("Rear Impact", "Whiplash"),
    ("VRU Impact", "Head Impact"),
    ("VRU Impact", "Pelvis Impact"),
    ("VRU Impact", "Leg Impact"),
    ("VRU Impact", "Pelvis & Leg Impact"),
]


def find_missing_required_inputs(
    wb, stage_element: str = None, stage_subelement: str = None
) -> list:
    """Find empty cells the OEM is required to fill in for the
    crash_protection domain, optionally scoped to one protocol element
    (e.g. stage_element="Frontal Impact", stage_subelement="Offset").
    Covers both the schema'd sheets and the "CP - VRU Prediction" grid.
    Raises ValueError for a scope the domain doesn't declare.
    See common.find_missing_required_inputs / GreyCellSheetSchema.

    A version-mismatched workbook (common.find_version_mismatch) is almost
    always also structurally different (renamed sheets/columns), which
    would otherwise raise common.TemplateIntegrityError before any finding
    -- including the version mismatch itself -- makes it back to the
    caller. When a version mismatch is detected, a resulting
    TemplateIntegrityError is swallowed in favour of just the version
    finding: a "wrong template version" message is strictly more useful
    than the low-level column-name error it would otherwise surface.
    """
    common.validate_scope(
        stage_element, stage_subelement, VALID_SCOPE_PAIRS, "crash_protection"
    )
    version_mismatch = common.find_version_mismatch(wb)
    try:
        pristine_wb = _pristine_wb()
        missing = version_mismatch + common.find_missing_required_inputs(
            wb, SHEET_SCHEMAS, stage_element, stage_subelement, pristine_wb=pristine_wb
        )

        for sheet_name, sections in GRID_SHEET_SECTION_STAGE_ELEMENTS.items():
            if sheet_name not in wb.sheetnames:
                continue
            pristine_ws = pristine_wb[sheet_name]
            section_rows = common.rows_in_sections(pristine_ws, list(sections))
            exempt_sections = GRID_SHEET_EXEMPT_SECTIONS.get(sheet_name, frozenset())
            for section_header, applies_to in sections.items():
                if section_header in exempt_sections:
                    continue
                if not common.stage_elements_in_scope(
                    applies_to, stage_element, stage_subelement
                ):
                    continue
                missing.extend(
                    common.find_missing_grid_inputs(
                        wb[sheet_name],
                        pristine_ws,
                        sheet_name,
                        rows=section_rows[section_header],
                    )
                )
        return missing
    except common.TemplateIntegrityError:
        if version_mismatch:
            return version_mismatch
        raise


def find_missing_required_inputs_in_file(
    input_file: str, stage_element: str = None, stage_subelement: str = None
) -> list:
    """Load an Excel file and return find_missing_required_inputs() for it."""
    wb = openpyxl.load_workbook(input_file, data_only=True)
    try:
        return find_missing_required_inputs(wb, stage_element, stage_subelement)
    finally:
        wb.close()


def find_consistency_findings(
    wb, stage_element: str = None, stage_subelement: str = None
) -> list:
    """One list of findings covering every consistency rule this domain
    declares: the missing required inputs
    (find_missing_required_inputs) plus every report-only prediction-rule
    violation (consistency.find_prediction_inconsistencies --
    legform_symmetry). Each finding says which rule produced it in
    ``check`` (common.CHECK_*). Same scope convention and error behaviour
    as find_missing_required_inputs: ValueError for an undeclared scope,
    common.TemplateIntegrityError for structural problems."""
    findings = find_missing_required_inputs(wb, stage_element, stage_subelement)
    findings.extend(
        consistency.find_prediction_inconsistencies(wb, stage_element, stage_subelement)
    )
    return findings


def find_consistency_findings_in_file(
    input_file: str, stage_element: str = None, stage_subelement: str = None
) -> list:
    """Load an Excel file and return find_consistency_findings() for it."""
    wb = openpyxl.load_workbook(input_file, data_only=True)
    try:
        return find_consistency_findings(wb, stage_element, stage_subelement)
    finally:
        wb.close()


# Sheets copied verbatim from the user's file: either genuine user content
# with no fixed reference counterpart to protect (the VRU prediction grid),
# or content whose integrity is already enforced separately (Version, via
# common.check_version).
#
# NOTE: "CP - VRU Prediction" is passthrough for rebuild_trusted_* only --
# its missing-input coverage comes from GRID_SHEET_SECTION_STAGE_ELEMENTS
# below. The VRU verification sheets generated during preprocess ("CP - VRU
# Head Impact", "CP - VRU Pelvis & Leg Impact") are not covered by a
# row-identity schema here -- their HPL/LPL thresholds are hardcoded in
# vru_processing.py rather than read from the sheet, so they don't share
# this module's core risk, but their test-point/row structure isn't
# verified against tampering. Left as a follow-up if this mechanism is
# generalized further.
PASSTHROUGH_SHEETS = ["Version", "CP - VRU Prediction", "Documentation", "Data"]

PRISTINE_TEMPLATE_NAME = "cp_template.xlsx"

# Sheet -> columns that are only ever produced by compute-score, never by
# the OEM or preprocess, derived from the same schemas that declare the
# trust boundary. Consumed by generate_template (blank_computed_columns)
# and preprocess (reset_computed_columns_to_dash).
COMPUTED_SHEET_COLUMNS = {
    name: list(schema.computed_columns)
    for name, schema in SHEET_SCHEMAS.items()
    if schema.computed_columns
}

# "Capping?" is a "YES"/"Capped"/"" string flag, not a numeric score, so it
# must not be coerced by numericize_computed_columns (see _pristine_dfs).
NUMERIC_COMPUTED_SHEET_COLUMNS = {
    name: [c for c in cols if c != "Capping?"]
    for name, cols in COMPUTED_SHEET_COLUMNS.items()
}


def rebuild_trusted_input(user_file_path: str) -> str:
    """Build a sanitized copy of *user_file_path* where every protected
    reference/structural cell (HPL, LPL, Capping, Max score, row labels, ...)
    is sourced from the official packaged template instead of the
    user-supplied file, and only the designated grey input cells (OEM
    Prediction, Value, Inspection [%], ...) are carried over from the user.

    Returns the path to a new temporary .xlsx file. The caller is
    responsible for deleting it.

    Raises common.TemplateIntegrityError if the user's file is missing a
    required sheet/column, or if a sheet's rows don't match the pristine
    template's row identity structure (inserted, deleted, or reordered
    rows).
    """
    pristine_path = str(files("data").joinpath(PRISTINE_TEMPLATE_NAME))
    user_wb = openpyxl.load_workbook(user_file_path, data_only=True)
    try:
        trusted_wb = common.rebuild_trusted_workbook(
            pristine_path, user_wb, SHEET_SCHEMAS, PASSTHROUGH_SHEETS
        )
    finally:
        user_wb.close()

    fd, sanitized_path = tempfile.mkstemp(suffix=".xlsx")
    os.close(fd)
    trusted_wb.save(sanitized_path)
    return sanitized_path


@functools.lru_cache(maxsize=1)
def _pristine_dfs() -> dict:
    """The packaged pristine template loaded as a dict of DataFrames, cached
    for the lifetime of the process. Backing rebuild_trusted_input_dfs,
    which -- unlike rebuild_trusted_input -- can be called on every single
    calculate_score(dfs)/add_vru_sheets_to_dfs invocation by a long-running caller,
    not once per CLI process; re-parsing the template file on every call
    would be a real, avoidable cost."""
    pristine_path = str(files("data").joinpath(PRISTINE_TEMPLATE_NAME))
    dfs = common.read_excel_file_to_dfs(pristine_path)
    return common.numericize_computed_columns(dfs, NUMERIC_COMPUTED_SHEET_COLUMNS)


def rebuild_trusted_input_dfs(user_dfs: dict) -> dict:
    """DataFrame-native sibling of rebuild_trusted_input: same SHEET_SCHEMAS
    and PASSTHROUGH_SHEETS, same pristine template, but operating directly
    on a dict of DataFrames instead of an on-disk .xlsx file -- for callers
    (e.g. an application that parses an upload once and works in dataframe/
    parquet land afterwards) that never have the file/openpyxl workbook
    rebuild_trusted_input needs.

    Returns a copy of *user_dfs* with every protected reference cell (HPL,
    LPL, Capping, Max score, row labels, ...) in a schema'd sheet sourced
    from the official packaged template instead of *user_dfs*; only the
    designated grey input cells are carried over from *user_dfs*. A schema'd
    sheet missing from *user_dfs* is left untouched (see
    common.rebuild_trusted_dfs).

    Raises common.TemplateIntegrityError if a present schema'd sheet is
    missing a required column, or if its rows don't match the pristine
    template's row identity structure (inserted, deleted, or reordered
    rows).
    """
    return common.rebuild_trusted_dfs(
        _pristine_dfs(), user_dfs, SHEET_SCHEMAS, PASSTHROUGH_SHEETS
    )
