# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from enum import Enum

# Re-exported so existing data_model.NOT_APPLICABLE / data_model.is_not_applicable
# callers keep working; the shared N/A vocabulary lives in common.py.
from euroncap_rating_2026.common import NOT_APPLICABLE, is_not_applicable

# Sheet -> columns where a literal "N/A" is protocol-meaningful and must
# survive the pd.read_excel default na_values (which silently collapse "N/A"
# into NaN): the Input parameters dropdowns, plus the Scenario labels that
# preprocess stamps the selected "N/A" into (Scenario Scores rows and the
# verif rows whose Value cell is rendered white / not required). Scoped to
# named columns so a stray "N/A" typed anywhere else (e.g. a PASS/FAIL Value
# cell) keeps the historical blank/unassessed behavior.
PRESERVE_NA_SHEETS = {
    "Input parameters": ("Value",),
    "Scenario Scores": ("Scenario",),
    "OM - Occ. classification verif.": ("Scenario",),
    "VA - ACC verif.": ("Scenario",),
}


STAGE_SUBELEMENTS = [
    {
        "Stage ": "Safe Driving",
        "Stage element": "Occupant Monitoring",
        "Stage subelement": "Seatbelt usage",
    },
    {
        "Stage ": "Safe Driving",
        "Stage element": "Occupant Monitoring",
        "Stage subelement": "Occupant classification",
    },
    {
        "Stage ": "Safe Driving",
        "Stage element": "Occupant Monitoring",
        "Stage subelement": "Occupant presence",
    },
    {
        "Stage ": "Safe Driving",
        "Stage element": "Driver Engagement",
        "Stage subelement": "Driver monitoring",
    },
    {
        "Stage ": "Safe Driving",
        "Stage element": "Driver Engagement",
        "Stage subelement": "General vehicle controls",
    },
    {
        "Stage ": "Safe Driving",
        "Stage element": "Vehicle Assistance",
        "Stage subelement": "Speed Assistance",
    },
    {
        "Stage ": "Safe Driving",
        "Stage element": "Vehicle Assistance",
        "Stage subelement": "ACC Performance",
    },
    {
        "Stage ": "Safe Driving",
        "Stage element": "Vehicle Assistance",
        "Stage subelement": "Steering Assistance",
    },
]

ACC_MATRIX_INDICES = {
    "CCRs straight": {"start_row": 1, "n_rows": 8, "start_col": 3, "n_cols": 7},
    "CCRs curved": {"start_row": 12, "n_rows": 8, "start_col": 3, "n_cols": 7},
    "CCRm": {"start_row": 23, "n_rows": 15, "start_col": 3, "n_cols": 7},
    "CCRb": {"start_row": 41, "n_rows": 1, "start_col": 3, "n_cols": 7},
    "CCR cut-in": {"start_row": 45, "n_rows": 2, "start_col": 3, "n_cols": 7},
    "CCR cut-out": {"start_row": 50, "n_rows": 2, "start_col": 3, "n_cols": 7},
    "CMRs straight": {"start_row": 55, "n_rows": 4, "start_col": 3, "n_cols": 7},
    "CMRs curved": {"start_row": 62, "n_rows": 4, "start_col": 3, "n_cols": 7},
    "CMRm": {"start_row": 69, "n_rows": 15, "start_col": 3, "n_cols": 7},
    "CMRb": {"start_row": 87, "n_rows": 1, "start_col": 3, "n_cols": 7},
    "CMR cut-in": {"start_row": 91, "n_rows": 2, "start_col": 3, "n_cols": 7},
    "CMR cut-out": {"start_row": 96, "n_rows": 2, "start_col": 3, "n_cols": 7},
    "CPLA": {"start_row": 101, "n_rows": 4, "start_col": 3, "n_cols": 7},
    "CBLA": {"start_row": 108, "n_rows": 4, "start_col": 3, "n_cols": 7},
}


class StageSubelementKey(str, Enum):
    OM = "OM"
    DE = "DE"
    VA = "VA"


class OCTypeOfSystem(str, Enum):
    AUTOMATIC = "Automatic"
    SYSTEM_ADVISED = "System advised"
    MANUAL = "Manual"
    NOT_APPLICABLE = "N/A"


class OCTypeOfSwitch(str, Enum):
    AUTOMATIC = "Automatic"
    HARDWARE = "Hardware"
    SOFTWARE = "Software"
    NOT_APPLICABLE = "N/A"


class OPTypeOfSystem(str, Enum):
    WARNING = "Warning"
    WARNING_AND_INTERVENTION = "Warning and intervention"
    NO = "No"


class OPSeatCoverage(str, Enum):
    REAR_SEATS = "Rear seats"
    ALL_PASSENGER_SEATS = "All seats"
