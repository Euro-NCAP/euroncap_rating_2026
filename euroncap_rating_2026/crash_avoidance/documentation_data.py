# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Python implementation of the "Documentation"/"Data" sheet formulas

Those two sheets are hidden in the pristine template and hold TRUE/FALSE
"is supporting evidence required" flags (e.g. did the OEM claim Virtual
Testing Assessment for any input parameter? did they mark any
prediction-grid cell as a scored color, implying a corner-case test was
run?).

Every check below :
  - 'Input parameters' checks match by Stage element + Input parameter
    identity (_input_parameters_has_vta).
  - "...robust. pred." checks match by the Type/Robustness layer row
    labels (_robust_pred_has_yes).
  - "...pred." checks locate each named test's matrix via
    matrix_processing.get_matrix_indices (the same lookup the rest of
    crash_avoidance already uses to read these sheets) and then read a
    documented edge of that matrix (_matrix_edge_slice)
"""

import pandas as pd

from euroncap_rating_2026.crash_avoidance import matrix_processing


def _any_color_in(df: pd.DataFrame, targets: list) -> bool:
    """True if any cell in *df* case-insensitively equals one of *targets*
    -- the DataFrame equivalent of Excel's (case-insensitive)
    COUNTIF(range, target) > 0."""
    targets_upper = {t.upper() for t in targets}
    values = df.to_numpy().ravel()
    return any(
        isinstance(v, str) and v.strip().upper() in targets_upper for v in values
    )


def _status(flag: bool) -> str:
    return "TRUE" if flag else "FALSE"


# --- shared: 'Input parameters' VTA checks, by row semantics rather than a
# hardcoded cell range -------------------------------------------------------

# The only two "Input parameter" labels that represent an OEM-supplied test
# prediction (as opposed to e.g. LDC's "Heading correction"/"Extended range
# performance", which are never VTA-eligible even though they sit under the
# same Stage element).
_PREDICTION_INPUT_PARAMETERS = {"Prediction - Standard", "Prediction - Extended"}

_FC_STAGE_ELEMENT = "Frontal Collisions"
_LDC_STAGE_ELEMENT = "Lane Departure Collisions"


def _forward_fill_stage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """'Input parameters' lays out its hierarchical Stage/Stage element/
    Stage subelement/Category labels like merged cells: only the first row
    of each group has a value, the rest are blank ("same as the row
    above"). Forward-fill them so every row can be matched by its full
    identity, the same convention common.get_param_df_from_input_parameters
    already uses."""
    df = df.copy()
    for col in ("Stage", "Stage element", "Stage subelement", "Category"):
        if col in df.columns:
            df[col] = df[col].ffill()
    return df


def _input_parameters_has_vta(dfs: dict, stage_element: str) -> bool:
    """True if any "Prediction" input parameter under *stage_element* has
    Value == "VTA" (case-insensitive) -- e.g. PR_CA-FC-VTADossier/VTAData
    pass stage_element="Frontal Collisions", PR_CA-LDC-VTADossier/VTAData
    pass "Lane Departure Collisions"."""
    df = _forward_fill_stage_columns(dfs["Input parameters"])
    is_prediction_row = df["Input parameter"].isin(_PREDICTION_INPUT_PARAMETERS)
    is_target_stage_element = df["Stage element"] == stage_element
    values = df.loc[is_prediction_row & is_target_stage_element, "Value"].to_numpy()
    return any(isinstance(v, str) and v.strip().upper() == "VTA" for v in values)


# --- shared: "...robust. pred." Perception checks, by row semantics -------

# "...robust. pred." sheets are a fixed table: one column per test, one row
# per robustness-layer attribute, with a "Type" column (VUT/Target/
# Environment) forward-filled like merged cells and a "Robustness layer"
# column naming the attribute (e.g. "Speed", "Appearance", "Adverse weather
# conditions"). The Perception check only counts "YES" in the target's
# Type/Appearance row or any Environment-group row -- never the VUT/Target
# kinematic rows (Speed, Acceleration, Initial position offset,
# Trajectory/Heading), which aren't part of what "Perception" documentation
# covers. This holds across all 3 "...robust. pred." sheets even though
# their exact row counts differ (e.g. "FC - Ped & Cyc robust. pred." has an
# extra "Illumination (Headlamp glare)" row the "Car & PTW" sheet doesn't).
_ENVIRONMENT_TYPE_GROUP = "Environment"
_TARGET_ROBUSTNESS_LAYERS = {"Type", "Appearance"}  # "Appearance*" also matches


def _robust_pred_has_yes(df: pd.DataFrame) -> bool:
    df = df.copy()
    df["Type"] = df["Type"].ffill()
    robustness_layer = df["Robustness layer"].astype(str).str.strip().str.rstrip("*")
    mask = (df["Type"] == _ENVIRONMENT_TYPE_GROUP) | robustness_layer.isin(
        _TARGET_ROBUSTNESS_LAYERS
    )
    test_columns = [c for c in df.columns if c not in ("Type", "Robustness layer")]
    return _any_color_in(df.loc[mask, test_columns], ["YES"])


# --- shared: "...pred." Corner checks, by named-matrix edge --------------


def _matrix_edge_slice(
    df: pd.DataFrame,
    test_name: str,
    last_rows: int | None = None,
    last_cols: int | None = None,
    first_rows: int | None = None,
) -> pd.DataFrame:
    """Return the edge of *test_name*'s matrix within *df*. The matrix
    itself is located by name via matrix_processing.get_matrix_indices (the
    same lookup used everywhere else in crash_avoidance to read these
    sheets), not a hardcoded position. *last_rows*/*first_rows* selects
    which rows of the matrix to keep (default: all); *last_cols* selects
    which columns (default: all)."""
    matrix_indices = matrix_processing.get_matrix_indices(df, test_name)
    if not matrix_indices:
        return df.iloc[0:0, 0:0]

    start_row = matrix_indices["start_row"]
    n_rows = matrix_indices["n_rows"]
    start_col = matrix_indices["start_col"]
    n_cols = matrix_indices["n_cols"]

    if first_rows is not None:
        row_start, row_stop = start_row, start_row + first_rows
    elif last_rows is not None:
        row_start, row_stop = start_row + n_rows - last_rows, start_row + n_rows
    else:
        row_start, row_stop = start_row, start_row + n_rows

    if last_cols is not None:
        col_start, col_stop = start_col + n_cols - last_cols, start_col + n_cols
    else:
        col_start, col_stop = start_col, start_col + n_cols

    return df.iloc[row_start:row_stop, col_start:col_stop]


_ANY_COLOR = ["Green", "Yellow", "Orange", "Brown"]
_GREEN_ONLY = ["Green"]

# (test name, matrix edge) -- the "corner" cells of each test's own matrix,
# i.e. the most demanding end of whichever axis (speed, offset, ...) that
# matrix varies across.
_FC_CAR_PTW_CORNER_TESTS = [
    ("CCRb", {"last_rows": 3}),
    ("CCFhos", {"last_rows": 5}),
    ("CCFhol", {"last_rows": 5}),
    ("CCFtap", {"last_cols": 1}),
    ("CCCscp", {"first_rows": 2, "last_cols": 2}),
    ("CMRb", {"last_rows": 5}),
    ("CMFtap", {"last_cols": 1}),
    ("CMCscp", {"first_rows": 2, "last_cols": 2}),
]

_LDC_CORNER_TESTS = [
    ("LDC - Single Veh pred.", "ELK RE", {"last_rows": 2}),
    ("LDC - Car & PTW pred.", "CC ELK OvU", {"last_rows": 4}),
    ("LDC - Car & PTW pred.", "CM ELK On", {"last_rows": 2}),
    ("LDC - Car & PTW pred.", "CM ELK OvU", {"last_rows": 6}),
    ("LDC - Car & PTW pred.", "CM ELK OvI", {"last_rows": 2}),
]

_LSC_CAR_PTW_CORNER_TESTS = [
    ("CMCscp SfS", {"last_cols": 2}),
    ("CMFtap SfS", {"last_cols": 1}),
]


# --- 'Documentation' sheet -------------------------------------------------


def compute_documentation(dfs: dict) -> pd.DataFrame:
    """Recompute the 4 "Documentation" sheet flags from *dfs*.

    Source formulas :
      PR_CA-FC-VTADossier:  COUNTIF('Input parameters'!G2:G43,"VTA")>0
      PR_CA-FC-Perception:  OR(COUNTIF('FC - Car & PTW robust. pred.'!C7:M13,"YES")>0,
                               COUNTIF('FC - Ped & Cyc robust. pred.'!C7:L14,"YES")>0)
      PR_CA-LDC-VTADossier: COUNTIF(G45:G46,"VTA")+COUNTIF(G48:G55,"VTA")>0
      PR_CA-LDC-Perception: COUNTIF('LDC - robust. pred.'!C5:G9,"YES")>0

    The ranges above are kept in this docstring only to document what the
    official template's formula covers -- see this module's docstring for
    how each is reimplemented by name/identity instead.
    """
    rows = [
        ("PR_CA-FC-VTADossier", _input_parameters_has_vta(dfs, _FC_STAGE_ELEMENT)),
        (
            "PR_CA-FC-Perception",
            _robust_pred_has_yes(dfs["FC - Car & PTW robust. pred."])
            or _robust_pred_has_yes(dfs["FC - Ped & Cyc robust. pred."]),
        ),
        ("PR_CA-LDC-VTADossier", _input_parameters_has_vta(dfs, _LDC_STAGE_ELEMENT)),
        (
            "PR_CA-LDC-Perception",
            _robust_pred_has_yes(dfs["LDC - robust. pred."]),
        ),
    ]
    return pd.DataFrame(
        [(row_id, _status(flag)) for row_id, flag in rows],
        columns=["Documentation ID", "Status"],
    )


# --- 'Data' sheet -----------------------------------------------------------


def compute_data(dfs: dict) -> pd.DataFrame:
    """Recompute the 5 "Data" sheet flags from *dfs*.

    Source formulas :
      PR_CA-FC-VTAData:  same as PR_CA-FC-VTADossier
      PR_CA-FC-Corner:   SUM of COUNTIF({"Green","Yellow","Orange","Brown"}) > 0
                         over 8 ranges on 'FC - Car & PTW pred.'
      PR_CA-LDC-VTAData: same as PR_CA-LDC-VTADossier
      PR_CA-LDC-Corner:  SUM of COUNTIF({"Green"}) > 0 over 5 ranges across
                         'LDC - Single Veh pred.' and 'LDC - Car & PTW pred.'
      PR_CA-LSC-Corner:  SUM of COUNTIF({"Green"}) > 0 over 2 ranges on
                         'LSC - Car & PTW pred.'

    The ranges above are kept in this docstring only to document what the
    official template's formula covers -- see this module's docstring for
    how each is reimplemented by name/identity instead.
    """
    fc_car_ptw_pred = dfs["FC - Car & PTW pred."]
    lsc_car_ptw_pred = dfs["LSC - Car & PTW pred."]

    rows = [
        ("PR_CA-FC-VTAData", _input_parameters_has_vta(dfs, _FC_STAGE_ELEMENT)),
        (
            "PR_CA-FC-Corner",
            any(
                _any_color_in(
                    _matrix_edge_slice(fc_car_ptw_pred, test, **edge), _ANY_COLOR
                )
                for test, edge in _FC_CAR_PTW_CORNER_TESTS
            ),
        ),
        ("PR_CA-LDC-VTAData", _input_parameters_has_vta(dfs, _LDC_STAGE_ELEMENT)),
        (
            "PR_CA-LDC-Corner",
            any(
                _any_color_in(
                    _matrix_edge_slice(dfs[sheet_name], test, **edge), _GREEN_ONLY
                )
                for sheet_name, test, edge in _LDC_CORNER_TESTS
            ),
        ),
        (
            "PR_CA-LSC-Corner",
            any(
                _any_color_in(
                    _matrix_edge_slice(lsc_car_ptw_pred, test, **edge), _GREEN_ONLY
                )
                for test, edge in _LSC_CAR_PTW_CORNER_TESTS
            ),
        ),
    ]
    return pd.DataFrame(
        [(row_id, _status(flag)) for row_id, flag in rows],
        columns=["Data ID", "Status"],
    )
