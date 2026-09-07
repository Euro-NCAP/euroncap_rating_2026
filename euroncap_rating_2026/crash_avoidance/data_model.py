# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from enum import Enum
import pandas as pd

# Sheet -> columns where a literal "N/A" is protocol-meaningful (the Input
# parameters "Prediction - Standard/Extended", "Extended range performance"
# and "Vehicle response" dropdowns offer it) and must survive pd.read_excel's
# default na_values, which silently collapse "N/A" into NaN.
PRESERVE_NA_SHEETS = {"Input parameters": ("Value",)}


STAGE_SUBELEMENTS = [
    {
        "Stage ": "Crash Avoidance",
        "Stage element": "Frontal Collisions",
        "Stage subelement": "Car & PTW",
    },
    {
        "Stage ": "Crash Avoidance",
        "Stage element": "Frontal Collisions",
        "Stage subelement": "Pedestrian & cyclist",
    },
    {
        "Stage ": "Crash Avoidance",
        "Stage element": "Lane Departure Collisions",
        "Stage subelement": "Single vehicle",
    },
    {
        "Stage ": "Crash Avoidance",
        "Stage element": "Lane Departure Collisions",
        "Stage subelement": "Car & PTW",
    },
    {
        "Stage ": "Crash Avoidance",
        "Stage element": "Low Speed Collisions",
        "Stage subelement": "Car & PTW",
    },
    {
        "Stage ": "Crash Avoidance",
        "Stage element": "Low Speed Collisions",
        "Stage subelement": "Pedestrian & cyclist",
    },
]

MATRIX_INDICES = {
    "CCRs": {"start_col": 3, "n_cols": 7},
    "CCRm": {"start_col": 3, "n_cols": 7},
    "CCRb": {"start_col": 3, "n_cols": 7},
    "CCFhos": {"start_col": 4, "n_cols": 4},
    "CCFhol": {"start_col": 4, "n_cols": 4},
    "CCFtap": {"start_col": 4, "n_cols": 6},
    "CCCscp": {"start_col": 3, "n_cols": 7},
    "CMRs": {"start_col": 4, "n_cols": 5},
    "CMRb": {"start_col": 4, "n_cols": 5},
    "CMFtap": {"start_col": 4, "n_cols": 6},
    "CMCscp": {"start_col": 3, "n_cols": 7},
    "CPLA day": {"start_col": 6, "n_cols": 4},
    "CPLA night": {"start_col": 6, "n_cols": 4},
    "CBLA": {"start_col": 6, "n_cols": 4},
    "CPTAfs & CPTAns": {"start_col": 5, "n_cols": 5},
    "CPTAfo & CPTAno": {"start_col": 5, "n_cols": 5},
    "CBTAfs & CBTAns": {"start_col": 4, "n_cols": 7},
    "CBTAfo & CBTAno": {"start_col": 5, "n_cols": 5},
    "CPNA day": {"start_col": 5, "n_cols": 5},
    "CPNA night": {"start_col": 5, "n_cols": 5},
    "CPFA day": {"start_col": 5, "n_cols": 5},
    "CPFA night": {"start_col": 5, "n_cols": 5},
    "CPNCO day": {"start_col": 6, "n_cols": 3},
    "CPNCO night": {"start_col": 6, "n_cols": 3},
    "CBNA": {"start_col": 5, "n_cols": 5},
    "CBNAO": {"start_col": 5, "n_cols": 5},
    "CBFA": {"start_col": 5, "n_cols": 5},
    "ELK RE": {"start_col": 1, "n_cols": 6},
    "CC ELK On": {"start_col": 3, "n_cols": 4},
    "CC ELK OvU": {"start_col": 2, "n_cols": 6},
    "CC ELK OvI": {"start_col": 4, "n_cols": 5},
    "CM ELK On": {"start_col": 3, "n_cols": 4},
    "CM ELK OvU": {"start_col": 2, "n_cols": 6},
    "CM ELK OvI": {"start_col": 4, "n_cols": 5},
    "CCCscp SfS": {"start_col": 1, "n_cols": 5},
    "CMCscp SfS": {"start_col": 1, "n_cols": 7},
    "CCFtap SfS": {"start_col": 2, "n_cols": 3},
    "CMFtap SfS": {"start_col": 2, "n_cols": 4},
    "CBNAO SfS": {"start_col": 1, "n_cols": 3},
    "CPMRCm": {"start_col": 1, "n_cols": 2},
    "CPMRCs": {"start_col": 1, "n_cols": 3},
    "CPMFC": {"start_col": 1, "n_cols": 3},
    "CBDA": {"start_col": 1, "n_cols": 3},
}

TOTAL_SCORES = {
    "CCRs": {"Standard": 1.2, "Extended": 0.15, "Robustness": 0.15},
    "CCRm": {"Standard": 2.4, "Extended": 0.3, "Robustness": 0.3},
    "CCRb": {"Standard": 1.6, "Extended": 0.2, "Robustness": 0.2},
    "CCFhos": {"Standard": 2.0, "Extended": 0.25, "Robustness": 0.25},
    "CCFhol": {"Standard": 2.0, "Extended": 0.25, "Robustness": 0.25},
    "CCFtap": {"Standard": 4.0, "Extended": 0.5, "Robustness": 0.5},
    "CCCscp": {"Standard": 6.0, "Extended": 0.75, "Robustness": 0.75},
    "CMRs": {"Standard": 1.2, "Extended": 0.15, "Robustness": 0.15},
    "CMRb": {"Standard": 1.6, "Extended": 0.2, "Robustness": 0.2},
    "CMFtap": {"Standard": 4.0, "Extended": 0.5, "Robustness": 0.5},
    "CMCscp": {"Standard": 6.0, "Extended": 0.75, "Robustness": 0.75},
    "CPLA day": {"Standard": 1.0, "Extended": 0.125, "Robustness": 0.25},
    "CPLA night": {"Standard": 1.0, "Extended": 0.125, "Robustness": None},
    "CBLA": {"Standard": 2.0, "Extended": 0.25, "Robustness": 0.25},
    "CPTAfs & CPTAns": {"Standard": 1.0, "Extended": 0.125, "Robustness": 0.125},
    "CPTAfo & CPTAno": {"Standard": 1.0, "Extended": 0.125, "Robustness": 0.125},
    "CBTAfs & CBTAns": {"Standard": 1.0, "Extended": 0.125, "Robustness": 0.125},
    "CBTAfo & CBTAno": {"Standard": 1.0, "Extended": 0.125, "Robustness": 0.125},
    "CPNA day": {"Standard": 0.5, "Extended": 0.0625, "Robustness": 0.125},
    "CPNA night": {"Standard": 0.5, "Extended": 0.0625, "Robustness": None},
    "CPFA day": {"Standard": 0.5, "Extended": 0.0625, "Robustness": 0.125},
    "CPFA night": {"Standard": 0.5, "Extended": 0.0625, "Robustness": None},
    "CPNCO day": {"Standard": 1.0, "Extended": 0.125, "Robustness": 0.25},
    "CPNCO night": {"Standard": 1.0, "Extended": 0.125, "Robustness": None},
    "CBNA": {"Standard": 1.0, "Extended": 0.125, "Robustness": 0.125},
    "CBFA": {"Standard": 1.0, "Extended": 0.125, "Robustness": 0.125},
    "CBNAO": {"Standard": 2.0, "Extended": 0.25, "Robustness": 0.25},
    "ELK RE": {"Standard": 4.0, "Extended": 0.5, "Robustness": 0.5},
    "CC ELK On": {"Standard": 2.0, "Extended": 0.25, "Robustness": 0.25},
    "CC ELK OvU": {"Standard": 1.0, "Extended": 0.125, "Robustness": 0.125},
    "CC ELK OvI": {"Standard": 1.0, "Extended": 0.125, "Robustness": 0.125},
    "CM ELK On": {"Standard": 2.0, "Extended": 0.25, "Robustness": 0.25},
    "CM ELK OvU": {"Standard": 1.0, "Extended": 0.125, "Robustness": 0.125},
    "CM ELK OvI": {"Standard": 1.0, "Extended": 0.125, "Robustness": 0.125},
    "CCCscp SfS": {"Standard": 3.0},
    "CMCscp SfS": {"Standard": 3.0},
    "CCFtap SfS": {"Standard": 1.0},
    "CMFtap SfS": {"Standard": 3.0},
    "CBNAO SfS": {"Standard": 3.0},
    "CPMRCm": {"Standard": 1.5},
    "CPMRCs": {"Standard": 1.5},
    "CPMFC": {"Standard": 2.0},
    "CBDA": {"Standard": 2.0},
}

STANDARD_RANGE_VERIFICATION_TEST_NUM = {
    "CCRs": 3,
    "CCRm": 3,
    "CCRb": 3,
    "CCFhos": 4,
    "CCFhol": 4,
    "CCFtap": 3,
    "CCCscp": 5,
    "CMRs": 3,
    "CMRb": 3,
    "CMFtap": 3,
    "CMCscp": 5,
    "CPLA day": 3,
    "CPLA night": 3,
    "CBLA": 3,
    "CPTAfs & CPTAns": 3,
    "CPTAfo & CPTAno": 3,
    "CBTAfs & CBTAns": 3,
    "CBTAfo & CBTAno": 3,
    "CPNA day": 3,
    "CPNA night": 3,
    "CPFA day": 3,
    "CPFA night": 3,
    "CPNCO day": 3,
    "CPNCO night": 3,
    "CBNA": 3,
    "CBFA": 3,
    "CBNAO": 3,
    "ELK RE": 3,
    "CC ELK On": 3,
    "CC ELK OvU": 3,
    "CC ELK OvI": 3,
    "CM ELK On": 3,
    "CM ELK OvU": 3,
    "CM ELK OvI": 3,
}

COLLISION_PARTNER = {
    "car": ["CCRs", "CCRm", "CCRb", "CCFhos", "CCFhol", "CCFtap", "CCCscp"],
    "PTW": ["CMRs", "CMRb", "CMFtap", "CMCscp"],
    "pedestrian": ["CPNA", "CPFA", "CPLA", "CPNCO", "CPTA"],
    "bicyclist": ["CBNA", "CBFA", "CBNAO", "CBTA"],
}

SUBTEST_TO_COLLISION_PARTNER = {
    subtest: partner
    for partner, subtests in COLLISION_PARTNER.items()
    for subtest in subtests
}

# CCFhos and CCFhol are Corner Cases in full: CA Frontal Collisions 4.1.3
# lists "CCFhos/hol: all cases", and the 4.2.1 verification-test table
# footnotes both scenarios "Corner Cases - Verification Test not applicable;
# evidence to be provided by Vehicle Manufacturer". CCFhol therefore gets no
# verification test at all, and CCFhos gets exactly one: the cell below is
# always tested by the assessor.
#
# Matrix-relative (row, col): row 0 is the VUT 30 km/h row and col 2 is the
# 50% impact location (CCFhos's 4 matrix columns are the 100%/75%/50%/25%
# impact locations -- see the CCFhos table in 3.1.1.2).
CCFHOS_PINNED_TEST_POINT = (0, 2)

EXTENDED_RANGE_CELLS = {
    "CCRs": [
        *((i, 0) for i in range(8)),  # first col
        *((i, 6) for i in range(8)),  # last col
    ],
    "CCRm": [
        *((i, 0) for i in range(11)),  # first col
        *((i, 6) for i in range(11)),  # last col
    ],
    "CCRb": [
        *((i, 0) for i in range(11)),  # first col
        *((i, 6) for i in range(11)),  # last col
        *(
            (row, col) for row in range(6, 11) for col in range(7)
        ),  # last 5 rows, all cols
    ],
    "CCFhos": [
        *((i, 0) for i in range(8)),  # first col
        *(
            (row, col) for row in range(6, 8) for col in range(4)
        ),  # last 2 rows, all cols
    ],
    "CCFhol": [
        *((i, 0) for i in range(8)),  # first col
        *(
            (row, col) for row in range(6, 8) for col in range(4)
        ),  # last 2 rows, all cols
    ],
    "CCFtap": [
        *((i, 5) for i in range(4)),  # last col
        *((3, j) for j in range(6)),  # last row
    ],
    "CCCscp": [
        (0, 5),
        (0, 6),
        (1, 5),
        (1, 6),
        (5, 0),
        (6, 0),
        (5, 1),
        (6, 1),
    ],
    "CMRs": [
        *((i, 0) for i in range(8)),  # first col
        *((i, 4) for i in range(8)),  # last col
    ],
    "CMRb": [
        *((i, 0) for i in range(11)),  # first col
        *((i, 4) for i in range(11)),  # last col
        *(
            (row, col) for row in range(6, 11) for col in range(4)
        ),  # last 5 rows, all cols
    ],
    "CMFtap": [
        *((i, 5) for i in range(4)),  # last col
        *((3, j) for j in range(6)),  # last row
    ],
    "CMCscp": [
        (0, 5),
        (0, 6),
        (1, 5),
        (1, 6),
        (5, 0),
        (6, 0),
        (5, 1),
        (6, 1),
    ],
    "CPLA day": [
        *(
            (row, col) for row in range(6) for col in [0, 2, 3]
        ),  # first 6 rows, cols 0,2,3
        *(
            (row, col) for row in range(6, 10) for col in [0, 1, 3]
        ),  # next 4 rows, cols 0,1,3
    ],
    "CPLA night": [
        *(
            (row, col) for row in range(6) for col in [0, 2, 3]
        ),  # first 6 rows, cols 0,2,3
        *(
            (row, col) for row in range(6, 10) for col in [0, 1, 3]
        ),  # next 4 rows, cols 0,1,3
    ],
    # CBLA and CPLA's actual test range is resolved from test-point attributes
    # (and, for CPLA, relative row position) in
    # matrix_processing.check_extended_range instead of these absolute indices,
    # because their sheets can be produced with a different row count across
    # converter versions (e.g. a legacy CBLA layout with an extra low-speed
    # row). These entries only remain as a fallback for when attributes aren't
    # available.
    "CBLA": [
        *(
            (row, col) for row in range(5) for col in [0, 2, 3]
        ),  # first 5 rows, cols 0,2,3
        *(
            (row, col) for row in range(5, 9) for col in [0, 1, 3]
        ),  # next 4 rows, cols 0,1,3
    ],
    "CPTAfs & CPTAns": [
        *(
            (row, col) for row in range(5) for col in [0, 1, 3, 4]
        ),  # next 4 rows, cols 0,1,3,4
        (3, 2),  # last row-1, col 2
    ],
    "CPTAfo & CPTAno": [
        *(
            (row, col) for row in range(5) for col in [0, 1, 3, 4]
        ),  # next 4 rows, cols 0,1,3,4
        (3, 2),  # last row-1, col 2
    ],
    "CBTAfs & CBTAns": [
        (0, 2),  # 2nd col
        (1, 2),
        (2, 2),
        (3, 2),
        (4, 2),  # last row, col 2
        (3, 0),  # last row - 1, first col
    ],
    "CBTAfo & CBTAno": [
        *(
            (row, col) for row in range(4) for col in [0, 1, 3, 4]
        ),  # next 4 rows, cols 0,1,3,4
        (3, 2),  # last row-1, col 2
        (4, 1),
        (4, 3),
    ],
    "CPNA day": [
        *((row, col) for row in range(6) for col in [0, 4]),
    ],
    "CPNA night": [
        *((row, col) for row in range(6) for col in [0, 4]),
    ],
    "CPFA day": [
        *((row, col) for row in range(6) for col in [0, 1, 3, 4]),
    ],
    "CPFA night": [
        *((row, col) for row in range(6) for col in [0, 1, 3, 4]),
    ],
    "CPNCO day": [
        *((row, col) for row in range(6) for col in [0, 2]),
    ],
    "CPNCO night": [
        *((row, col) for row in range(6) for col in [0, 2]),
    ],
    "CBNA": [
        *((row, col) for row in range(6) for col in [0, 3, 4]),
    ],
    "CBNAO": [
        *((row, col) for row in range(6) for col in [0, 3, 4]),
    ],
    "CBFA": [
        *((row, col) for row in range(6) for col in [0, 1, 4]),
    ],
    "ELK RE": [
        *((row, col) for row in range(6) for col in [5]),
        *((row, col) for row in [0, 1, 5] for col in range(6)),
    ],
    "CC ELK On": [
        *((row, col) for row in [0, 1, 3, 4, 5] for col in range(4)),
    ],
    "CC ELK OvU": [
        *((row, col) for row in range(9) for col in [0, 5]),
        *((row, col) for row in [0, 1, 3, 4, 5, 6, 7, 8] for col in range(6)),
    ],
    "CC ELK OvI": [
        *((row, col) for row in range(5) for col in [0, 4]),
        *((row, col) for row in [0, 1, 3, 4] for col in range(5)),
    ],
    "CM ELK On": [
        *((row, col) for row in [0, 1, 3, 4, 5] for col in range(4)),
    ],
    "CM ELK OvU": [
        *((row, col) for row in range(9) for col in [0, 5]),
        *((row, col) for row in [3, 4, 5, 6, 7, 8] for col in range(6)),
    ],
    "CM ELK OvI": [
        *((row, col) for row in range(5) for col in [0, 4]),
        *((row, col) for row in [3, 4] for col in range(5)),
    ],
}

STAGE_SUBELEMENT_TO_LOADCASES = {
    "FC": {
        "Car & PTW": [
            "CCRs",
            "CCRm",
            "CCRb",
            "CCFhos",
            "CCFhol",
            "CCFtap",
            "CCCscp",
            "CMRs",
            "CMRb",
            "CMFtap",
            "CMCscp",
        ],
        "Ped & Cyc": [
            "CPLA day",
            "CPLA night",
            "CBLA",
            "CPTAfs & CPTAns",
            "CPTAfo & CPTAno",
            "CBTAfs & CBTAns",
            "CBTAfo & CBTAno",
            "CPNA day",
            "CPNA night",
            "CPFA day",
            "CPFA night",
            "CPNCO day",
            "CPNCO night",
            "CBNA",
            "CBFA",
            "CBNAO",
        ],
    },
    "LDC": {
        "Single Veh": [
            "Driveability",
            "Driver state link",
            "ELK RE",
        ],
        "Car & PTW": [
            "CC ELK On",
            "CC ELK OvU",
            "CC ELK OvI",
            "CM ELK On",
            "CM ELK OvU",
            "CM ELK OvI",
        ],
    },
    "LSC": {
        "Ped & Cyc": ["CBNAO SfS", "CPMRCm", "CPMRCs", "CPMFC", "CBDA"],
        "Car & PTW": [
            "CCCscp SfS",
            "CMCscp SfS",
            "CCFtap SfS",
            "CMFtap SfS",
        ],
    },
}

SCORE_FROM_VERIFICATION_TEST_OUTCOME = {
    "Standard": {
        "VTA": {
            5: {5: 100, 4: 80, 3: 60, 2: 40, 1: 20, 0: 0},
            4: {4: 100, 3: 75, 2: 50, 1: 25, 0: 0},
            3: {3: 100, 2: 67, 1: 33, 0: 0},
            2: {2: 100, 1: 50, 0: 0},
            1: {1: 100, 0: 0},
            0: {0: 100},
        },
        "Self claimed": {
            5: {5: 100, 4: 80, 3: 0, 2: 0, 1: 0, 0: 0},
            4: {4: 100, 3: 75, 2: 0, 1: 0, 0: 0},
            3: {3: 100, 2: 67, 1: 0, 0: 0},
            2: {2: 100, 1: 50, 0: 0},
            1: {1: 100, 0: 0},
            0: {0: 100},
        },
    },
    "Extended": {
        "VTA": {
            2: {2: 100, 1: 50, 0: 0},
            1: {1: 100, 0: 0},
            0: {0: 100},
        },
        "Self claimed": {
            2: {2: 100, 1: 0, 0: 0},
            1: {1: 100, 0: 0},
            0: {0: 100},
        },
    },
}


SUBTEST_TO_TEST_DICT = {
    "CPLA day": "CPLA",
    "CPLA night": "CPLA",
    "CPTAfs & CPTAns": "CPTA",
    "CPTAfo & CPTAno": "CPTA",
    "CBTAfs & CBTAns": "CBTA",
    "CBTAfo & CBTAno": "CBTA",
    "CPNA day": "CPNA",
    "CPNA night": "CPNA",
    "CPFA day": "CPFA",
    "CPFA night": "CPFA",
    "CPNCO day": "CPNCO",
    "CPNCO night": "CPNCO",
    "CC ELK OvU": "CC ELK Ov",
    "CC ELK OvI": "CC ELK Ov",
    "CM ELK OvU": "CM ELK Ov",
    "CM ELK OvI": "CM ELK Ov",
}

# List of test-point attribute dicts that must NOT be selected for CC/CM ELK OvU/OvI
# when the "Initial position offset" robustness layer is chosen.
# Each entry must contain a "Scenario" key plus the attribute key/value pairs that
# uniquely identify the excluded cell (matching uses _attributes_match semantics).
ELK_OV_INITIAL_POSITION_OFFSET_EXCLUDED_CELLS: list = [
    {
        "Scenario": "CC ELK On",
        "VUT speed": "70 km/h",
        "Target speed": "70 km/h",
        "Lateral velocity": "0.3 m/s",
    },
    {
        "Scenario": "CM ELK On",
        "VUT speed": "70 km/h",
        "Target speed": "70 km/h",
        "Lateral velocity": "0.3 m/s",
    },
    {
        "Scenario": "CM ELK OvU",
        "VUT speed": "50 km/h",
        "Target speed": "60 km/h",
        "Lateral velocity": "0.3 m/s",
    },
    {
        "Scenario": "CM ELK OvU",
        "VUT speed": "60 km/h",
        "Target speed": "70 km/h",
        "Lateral velocity": "0.3 m/s",
    },
    {
        "Scenario": "CM ELK OvU",
        "VUT speed": "70 km/h",
        "Target speed": "80 km/h",
        "Lateral velocity": "0.3 m/s",
    },
]

STAGE_SUBELEMENT_TO_CATEGORIES = {
    "FC": ["Turning", "Crossing", "Longitudinal"],
    "LDC": [
        "Driver Acceptance",
        "Lane Departure",
        "ELK Car-to-car",
        "ELK Car-to-motorcyclist",
    ],
    "LSC": [
        "Turning",
        "Crossing",
        "Crossing",
        "Manoeuvring",
        "Dooring",
    ],
}


class StageSubelementKey(str, Enum):
    LSC = "LSC"
    LDC = "LDC"
    FC = "FC"


class VehicleResponse(str, Enum):
    INFORMATION = "Information"
    WARNING = "Warning"
    RETENTION = "Retention"


class NcapTestCriteria(str, Enum):
    UNKNOWN = "Unknown"
    MITIGATION_OR_AVOIDANCE = "mitigation or avoidance"
    MITIGATION = "mitigation"
    AVOIDANCE = "avoidance"
    WARNING = "warning"
    ROAD_EDGE = "road edge"
    ELK_TARGET = "elk target"
    LSC_DOORING = "lsc dooring"
    LSC_MITIGATION = "lsc mitigation"
    LSC_AVOIDANCE = "lsc avoidance"


NCAP_TEST_SCENARIOS = {
    NcapTestCriteria.AVOIDANCE: [
        "CBTAfs & CBTAns",
        "CBTAfo & CBTAno",
        "CPTAfs & CPTAns",
        "CPTAfo & CPTAno",
        "CCFtap",
        "CMFtap",
        "CCCscp",
        "CMCscp",
    ],
    NcapTestCriteria.MITIGATION: ["CCFhos", "CCFhol"],
    NcapTestCriteria.MITIGATION_OR_AVOIDANCE: [
        "CCRs",
        "CCRm",
        "CCRb",
        "CMRs",
        "CMRb",
        "CPNA day",
        "CPNA night",
        "CPFA day",
        "CPFA night",
        "CPNCO day",
        "CPNCO night",
        "CBNA",
        "CBFA",
        "CBNAO",
        "CPLA day",
        "CPLA night",
        "CBLA",
    ],
    NcapTestCriteria.WARNING: [
        "CPLA day",
        "CPLA night",
        "CBLA",
    ],
    NcapTestCriteria.ROAD_EDGE: ["ELK RE"],
    NcapTestCriteria.ELK_TARGET: [
        "CC ELK On",
        "CC ELK OvI",
        "CC ELK OvU",
        "CM ELK On",
        "CM ELK Ov",
        "CM ELK OvI",
        "CM ELK OvU",
    ],
    NcapTestCriteria.LSC_AVOIDANCE: [
        "CCCscp SfS",
        "CMCscp SfS",
        "CCFtap SfS",
        "CMFtap SfS",
        "CBNAO SfS",
        "CPMRCm",
        "CPMRCs",
    ],
    NcapTestCriteria.LSC_MITIGATION: ["CPMFC"],
    NcapTestCriteria.LSC_DOORING: ["CBDA"],
}

NCAP_TEST_CRITERIA_TO_SCORE_PARAMETER = {
    NcapTestCriteria.MITIGATION_OR_AVOIDANCE: "v_rel_impact",
    NcapTestCriteria.MITIGATION: "v_reduction",
    NcapTestCriteria.AVOIDANCE: "v_impact",
    NcapTestCriteria.WARNING: "fcw_ttc",
    NcapTestCriteria.ROAD_EDGE: "dtle_t_end",
    NcapTestCriteria.ELK_TARGET: "impact_occurred",
    NcapTestCriteria.LSC_AVOIDANCE: "v_impact",
    NcapTestCriteria.LSC_MITIGATION: "v_impact",
    NcapTestCriteria.LSC_DOORING: "TTC @ t_door_opening",
}

# Tolerance (in the unit of the criteria's KPI above) applied to the OEM's
# predicted colour band when verifying a tested point, per criteria.
#
# Crash Avoidance - Frontal Collisions v1.2 sec 4.2.4 defines a 2 km/h
# tolerance on the *impact speed* of a verification test, so it applies to the
# v_rel_impact KPI and to nothing else: every other criteria scores a
# different quantity -- v_reduction (a speed reduction, Figure 5-2), a warning
# TTC in seconds, a distance to the lane edge in metres, a 0/1 impact flag --
# for which no protocol defines a tolerance, and where 2 "km/h" would mean 2
# seconds, 2 metres or 2 impacts. Those therefore get 0.0, which makes the
# tolerance window in matrix_processing.check_thresholds collapse to the
# predicted band itself, i.e. a no-op.
NCAP_TEST_CRITERIA_TO_TOLERANCE = {
    NcapTestCriteria.MITIGATION_OR_AVOIDANCE: 2.0,
    NcapTestCriteria.MITIGATION: 0.0,
    NcapTestCriteria.AVOIDANCE: 0.0,
    NcapTestCriteria.WARNING: 0.0,
    NcapTestCriteria.ROAD_EDGE: 0.0,
    NcapTestCriteria.ELK_TARGET: 0.0,
    NcapTestCriteria.LSC_AVOIDANCE: 0.0,
    NcapTestCriteria.LSC_MITIGATION: 0.0,
    NcapTestCriteria.LSC_DOORING: 0.0,
}


def get_criteria_for_test(test_name: str) -> NcapTestCriteria:
    """
    Maps a test name to its corresponding NCAP test criteria.

    Args:
        test_name: The name of the test (e.g., "CBNAO", "CCRs", "ELK RE")

    Returns:
        The NcapTestCriteria enum value for the test

    Raises:
        ValueError: If the test name is not found in any criteria mapping
    """
    for criteria, tests in NCAP_TEST_SCENARIOS.items():
        if test_name in tests:
            return criteria

    raise ValueError(f"Test '{test_name}' not found in NCAP_TEST_SCENARIOS")


def get_score_parameter_for_test(test_name: str) -> str:
    """
    Maps a test name to its corresponding score parameter.

    Args:
        test_name: The name of the test (e.g., "CBNAO", "CCRs", "ELK RE")

    Returns:
        The score parameter name (e.g., "v_rel_impact", "v_reduction", "dtle_t_end")

    Raises:
        ValueError: If the test name is not found in any criteria mapping
    """
    criteria = get_criteria_for_test(test_name)
    return NCAP_TEST_CRITERIA_TO_SCORE_PARAMETER[criteria]


def get_test_point_df_from_verification_df(df_verification, sheet_name):
    """
    Returns a DataFrame with test points, filtering out the 'General requirements' header.
    The header row and content start row depend on the sheet_name.

    Args:
        df_verification: The input DataFrame (as read from Excel, with all rows).
        sheet_name: The name of the sheet.

    Returns:
        A DataFrame with the correct header and content rows.
    """

    if "single veh" in sheet_name.lower():
        header_row = 4
    else:
        header_row = 2

    # Set header and slice content
    header = df_verification.iloc[header_row]
    df_content = df_verification.iloc[header_row + 1 :].copy()
    df_content.columns = header

    # All rows before the header row as a DataFrame
    # df_header = df_verification.iloc[:header_row].copy()
    # df_header.columns = df_verification.columns

    return df_content
