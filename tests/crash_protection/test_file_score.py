# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import unittest
import numpy as np
import pandas as pd

from euroncap_rating_2026.crash_protection.compute_score import calculate_score
from euroncap_rating_2026.crash_protection.vru_processing import VruPredictionColor

N = np.nan
G = "Green"
R = "Red"

STAGE_SUBELEMENT_LIST = [
    "Offset",
    "FW",
    "Sled & VT",
    "MDB",
    "Pole",
    "Farside",
    "Whiplash",
    "Head Impact",
    "Pelvis & Leg Impact",
]

MAX_SCORES = {
    "Offset": 20,
    "FW": 10,
    "Sled & VT": 10,
    "MDB": 15,
    "Pole": 10,
    "Farside": 10,
    "Whiplash": 5,
    "Head Impact": 10,
    "Pelvis & Leg Impact": 10,
}


# ---------------------------------------------------------------------------
# Input parameters sheet (identical for max and min)
# ---------------------------------------------------------------------------
def _make_input_parameters_df():
    return pd.DataFrame(
        {
            "Stage": [
                "Crash Protection",
                N,
                N,
                N,
                N,
                N,
                N,
                N,
                N,
                N,
                N,
                N,
            ],
            "Stage element": [
                "Frontal Impact",
                N,
                N,
                "Side Impact",
                N,
                N,
                N,
                "Rear Impact",
                "VRU Impact",
                N,
                N,
                N,
            ],
            "Stage subelement": [
                "Offset",
                "FW",
                "Sled & VT",
                "MDB",
                "Pole",
                "Farside",
                N,
                "Whiplash",
                "Head Impact",
                "Pelvis Impact",
                "Leg Impact",
                N,
            ],
            "Input parameter": [
                N,
                N,
                N,
                N,
                N,
                "Countermeasure?",
                "Red line >125 mm outboard of the orange line",
                "Torso angle (rear)",
                "Number of verification tests (min 10)",
                "Number of verification tests (min 5)",
                "Number of verification tests (min 5)",
                N,
            ],
            "Value": [
                N,
                N,
                N,
                N,
                N,
                "Yes",
                "Yes",
                1,
                10,
                5,
                5,
                N,
            ],
        }
    )


# ---------------------------------------------------------------------------
# Test Scores skeleton (identical for max and min — Score=NaN, filled by pipeline)
# ---------------------------------------------------------------------------
def _make_test_scores_df():
    return pd.DataFrame(
        {
            "Stage": ["Crash Protection"] + [N] * 8,
            "Stage element": [
                "Frontal Impact",
                N,
                N,
                "Side Impact",
                N,
                N,
                "Rear Impact",
                "VRU Impact",
                N,
            ],
            "Stage subelement": [
                "Offset",
                "FW",
                "Sled & VT",
                "MDB",
                "Pole",
                "Farside",
                "Whiplash",
                "Head Impact",
                "Pelvis & Leg Impact",
            ],
            "Inspection [%]": [0] * 9,
            "Score": [N] * 9,
            "Max score": [20, 10, 10, 15, 10, 10, 5, 10, 10],
        }
    )


# ---------------------------------------------------------------------------
# CP - Dummy Scores skeleton
# ---------------------------------------------------------------------------
def _make_dummy_scores_df():
    cols = [
        "Stage",
        "Stage element",
        "Stage subelement",
        "Loadcase",
        "Seat position",
        "Dummy",
        "Capping?",
        "Score",
        "Max score",
    ]
    rows = [
        # Frontal / Offset
        [
            "Crash Protection",
            "Frontal Impact",
            "Offset",
            "MPDB-50",
            "Driver",
            "THOR-50",
            N,
            N,
            5.0,
        ],
        [N, N, N, N, "Front Passenger", "HIII-05", N, N, 5.0],
        [N, N, N, N, "Behind Driver", "Q6", N, N, 5.0],
        [N, N, N, N, "Behind Passenger", "Q10", N, N, 5.0],
        # Frontal / FW
        [N, N, "FW", "FWDB-35", "Driver", "HIII-05", N, N, 5.0],
        [N, N, N, N, "Front Passenger", "THOR-50", N, N, 2.5],
        [N, N, N, N, "Behind Passenger", "HIII-05", N, N, 2.5],
        # Frontal / Sled & VT
        [N, N, "Sled & VT", "Sled-Mid", "Driver", "HIII-50", N, N, 1.25],
        [N, N, N, N, "Front Passenger", "HIII-95", N, N, 1.25],
        [N, N, N, "Sled-High", "Driver", "HIII-95", N, N, 1.25],
        [N, N, N, N, "Front Passenger", "HIII-05", N, N, 1.25],
        [N, N, N, "Virtual-Low", "Driver", "HIII-50", N, N, 0.833333],
        [N, N, N, N, "Front Passenger", "HIII-05", N, N, 0.833333],
        [N, N, N, "Virtual-Low", "Driver", "HIII-05", N, N, 0.833333],
        [N, N, N, N, "Front Passenger", "HIII-50", N, N, 0.833333],
        [N, N, N, "Virtual-High", "Driver", "HIII-05", N, N, 0.833333],
        [N, N, N, N, "Front Passenger", "HIII-95", N, N, 0.833333],
        # Side / MDB
        [
            "Crash Protection",
            "Side Impact",
            "MDB",
            "AEMDB-60",
            "Driver",
            "WorldSID-50",
            N,
            N,
            10.0,
        ],
        [N, N, N, N, "Behind Driver", "Q10-Side", N, N, 2.5],
        [N, N, N, N, "Behind Passenger", "Q6-Side", N, N, 2.5],
        # Side / Pole
        [N, N, "Pole", "Pole-32", "Driver", "WorldSID-50", N, N, 10.0],
        # Side / Farside
        [
            N,
            N,
            "Farside",
            "Pole-32",
            "Driver and Front Passenger",
            "WorldSID-50-Farside",
            N,
            N,
            2.0,
        ],
        [N, N, N, "Main-AEMDB", "Driver", "WorldSID-50", N, N, 2.0],
        [N, N, N, "Main-Pole", "Driver", "WorldSID-50", N, N, 2.0],
        [N, N, N, "Robustness-AEMDB", "Driver", "WorldSID-50", N, N, 2.0],
        [N, N, N, "Robustness-Pole", "Driver", "WorldSID-50", N, N, 2.0],
        # Rear / Whiplash
        [
            "Crash Protection",
            "Rear Impact",
            "Whiplash",
            "Rear-Mid",
            "Driver",
            "BioRID-50",
            N,
            N,
            1.5,
        ],
        [N, N, N, "Rear-High", "Driver", "BioRID-50", N, N, 1.5],
        [N, N, N, "Static-Front", "Front", "HPM", N, N, 1.0],
        [N, N, N, "Static-Rear", "Rear", "HPM", N, N, 1.0],
        # VRU / Head Impact
        [
            "Crash Protection",
            "VRU Impact",
            "Head Impact",
            "Headform",
            "Driver",
            "Adult Headform",
            N,
            N,
            N,
        ],
        [N, N, N, N, "Driver", "Child Headform", N, N, N],
        # VRU / Pelvis & Leg Impact
        [
            N,
            N,
            "Pelvis & Leg Impact",
            "Upper Leg",
            "Driver",
            "Upper legform",
            N,
            N,
            2.5,
        ],
        [N, N, N, "Lower Leg", "Driver", "aPLI", N, N, 7.5],
    ]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# CP - Body region scores skeleton
# ---------------------------------------------------------------------------
def _make_body_region_scores_df():
    cols = [
        "Stage element",
        "Stage subelement",
        "Loadcase",
        "Seat position",
        "Dummy",
        "Body region",
        "Body regionscore",
        "Modifiers",
        "Inspection [%]",
        "Score",
        "Max score",
    ]
    # fmt: off
    rows = [
        # Frontal / Offset / MPDB-50 / Driver / THOR-50
        ["Frontal Impact","Offset",    "MPDB-50",     "Driver",                "THOR-50",            "Head & Neck",           N, N, 0, N, 1.25],
        [N,               N,           N,             N,                       N,                    "Chest & Abdomen",       N, N, 0, N, 1.25],
        [N,               N,           N,             N,                       N,                    "Knee, Femur & Pelvis",  N, N, 0, N, 1.25],
        [N,               N,           N,             N,                       N,                    "Lower Leg, Foot & Ankle",N,N, 0, N, 1.25],
        # Frontal / Offset / MPDB-50 / Front Passenger / HIII-05
        [N,               N,           N,             "Front Passenger",       "HIII-05",            "Head & Neck",           N, N, 0, N, 1.25],
        [N,               N,           N,             N,                       N,                    "Chest & Abdomen",       N, N, 0, N, 1.25],
        [N,               N,           N,             N,                       N,                    "Knee, Femur & Pelvis",  N, N, 0, N, 1.25],
        [N,               N,           N,             N,                       N,                    "Lower Leg, Foot & Ankle",N,N, 0, N, 1.25],
        # Frontal / Offset / MPDB-50 / Behind Driver / Q6
        [N,               N,           N,             "Behind Driver",         "Q6",                 "Head",                  N, N, 0, N, 2.5],
        [N,               N,           N,             N,                       N,                    "Neck",                  N, N, 0, N, 1.25],
        [N,               N,           N,             N,                       N,                    "Chest",                 N, N, 0, N, 1.25],
        # Frontal / Offset / MPDB-50 / Behind Passenger / Q10
        [N,               N,           N,             "Behind Passenger",      "Q10",                "Head",                  N, N, 0, N, 2.5],
        [N,               N,           N,             N,                       N,                    "Neck",                  N, N, 0, N, 1.25],
        [N,               N,           N,             N,                       N,                    "Chest",                 N, N, 0, N, 1.25],
        # Frontal / FW / FWDB-35 / Driver / HIII-05
        [N,               "FW",        "FWDB-35",     "Driver",                "HIII-05",            "Head & Neck",           N, N, 0, N, 1.25],
        [N,               N,           N,             N,                       N,                    "Chest & Abdomen",       N, N, 0, N, 1.25],
        [N,               N,           N,             N,                       N,                    "Knee, Femur & Pelvis",  N, N, 0, N, 1.25],
        [N,               N,           N,             N,                       N,                    "Lower Leg, Foot & Ankle",N,N, 0, N, 1.25],
        # Frontal / FW / FWDB-35 / Front Passenger / THOR-50
        [N,               N,           N,             "Front Passenger",       "THOR-50",            "Head & Neck",           N, N, 0, N, 0.625],
        [N,               N,           N,             N,                       N,                    "Chest & Abdomen",       N, N, 0, N, 0.625],
        [N,               N,           N,             N,                       N,                    "Knee, Femur & Pelvis",  N, N, 0, N, 0.625],
        [N,               N,           N,             N,                       N,                    "Lower Leg, Foot & Ankle",N,N, 0, N, 0.625],
        # Frontal / FW / FWDB-35 / Behind Passenger / HIII-05
        [N,               N,           N,             "Behind Passenger",      "HIII-05",            "Head & Neck",           N, N, 0, N, 0.625],
        [N,               N,           N,             N,                       N,                    "Chest & Abdomen",       N, N, 0, N, 1.25],
        [N,               N,           N,             N,                       N,                    "Knee, Femur & Pelvis",  N, N, 0, N, 0.625],
        # Frontal / Sled & VT / Sled-Mid / Driver / HIII-50
        [N,               "Sled & VT", "Sled-Mid",    "Driver",                "HIII-50",            "Head & Neck",           N, N, 0, N, 0.3125],
        [N,               N,           N,             N,                       N,                    "Chest & Abdomen",       N, N, 0, N, 0.3125],
        [N,               N,           N,             N,                       N,                    "Knee, Femur & Pelvis",  N, N, 0, N, 0.3125],
        [N,               N,           N,             N,                       N,                    "Lower Leg, Foot & Ankle",N,N, 0, N, 0.3125],
        # Frontal / Sled & VT / Sled-Mid / Front Passenger / HIII-95
        [N,               N,           N,             "Front Passenger",       "HIII-95",            "Head & Neck",           N, N, 0, N, 0.3125],
        [N,               N,           N,             N,                       N,                    "Chest & Abdomen",       N, N, 0, N, 0.3125],
        [N,               N,           N,             N,                       N,                    "Knee, Femur & Pelvis",  N, N, 0, N, 0.3125],
        [N,               N,           N,             N,                       N,                    "Lower Leg, Foot & Ankle",N,N, 0, N, 0.3125],
        # Frontal / Sled & VT / Sled-High / Driver / HIII-95
        [N,               N,           "Sled-High",   "Driver",                "HIII-95",            "Head & Neck",           N, N, 0, N, 0.3125],
        [N,               N,           N,             N,                       N,                    "Chest & Abdomen",       N, N, 0, N, 0.3125],
        [N,               N,           N,             N,                       N,                    "Knee, Femur & Pelvis",  N, N, 0, N, 0.3125],
        [N,               N,           N,             N,                       N,                    "Lower Leg, Foot & Ankle",N,N, 0, N, 0.3125],
        # Frontal / Sled & VT / Sled-High / Front Passenger / HIII-05
        [N,               N,           N,             "Front Passenger",       "HIII-05",            "Head & Neck",           N, N, 0, N, 0.3125],
        [N,               N,           N,             N,                       N,                    "Chest & Abdomen",       N, N, 0, N, 0.3125],
        [N,               N,           N,             N,                       N,                    "Knee, Femur & Pelvis",  N, N, 0, N, 0.3125],
        [N,               N,           N,             N,                       N,                    "Lower Leg, Foot & Ankle",N,N, 0, N, 0.3125],
        # Frontal / Sled & VT / Virtual-Low / Driver / HIII-50
        [N,               N,           "Virtual-Low", "Driver",                "HIII-50",            "Head & Neck",           N, N, 0, N, 0.208333],
        [N,               N,           N,             N,                       N,                    "Chest & Abdomen",       N, N, 0, N, 0.208333],
        [N,               N,           N,             N,                       N,                    "Knee, Femur & Pelvis",  N, N, 0, N, 0.208333],
        [N,               N,           N,             N,                       N,                    "Lower Leg, Foot & Ankle",N,N, 0, N, 0.208333],
        # Frontal / Sled & VT / Virtual-Low / Front Passenger / HIII-05
        [N,               N,           N,             "Front Passenger",       "HIII-05",            "Head & Neck",           N, N, 0, N, 0.208333],
        [N,               N,           N,             N,                       N,                    "Chest & Abdomen",       N, N, 0, N, 0.208333],
        [N,               N,           N,             N,                       N,                    "Knee, Femur & Pelvis",  N, N, 0, N, 0.208333],
        [N,               N,           N,             N,                       N,                    "Lower Leg, Foot & Ankle",N,N, 0, N, 0.208333],
        # Frontal / Sled & VT / Virtual-Low / Driver / HIII-05
        [N,               N,           "Virtual-Low", "Driver",                "HIII-05",            "Head & Neck",           N, N, 0, N, 0.208333],
        [N,               N,           N,             N,                       N,                    "Chest & Abdomen",       N, N, 0, N, 0.208333],
        [N,               N,           N,             N,                       N,                    "Knee, Femur & Pelvis",  N, N, 0, N, 0.208333],
        [N,               N,           N,             N,                       N,                    "Lower Leg, Foot & Ankle",N,N, 0, N, 0.208333],
        # Frontal / Sled & VT / Virtual-Low / Front Passenger / HIII-50
        [N,               N,           N,             "Front Passenger",       "HIII-50",            "Head & Neck",           N, N, 0, N, 0.208333],
        [N,               N,           N,             N,                       N,                    "Chest & Abdomen",       N, N, 0, N, 0.208333],
        [N,               N,           N,             N,                       N,                    "Knee, Femur & Pelvis",  N, N, 0, N, 0.208333],
        [N,               N,           N,             N,                       N,                    "Lower Leg, Foot & Ankle",N,N, 0, N, 0.208333],
        # Frontal / Sled & VT / Virtual-High / Driver / HIII-05
        [N,               N,           "Virtual-High","Driver",                "HIII-05",            "Head & Neck",           N, N, 0, N, 0.208333],
        [N,               N,           N,             N,                       N,                    "Chest & Abdomen",       N, N, 0, N, 0.208333],
        [N,               N,           N,             N,                       N,                    "Knee, Femur & Pelvis",  N, N, 0, N, 0.208333],
        [N,               N,           N,             N,                       N,                    "Lower Leg, Foot & Ankle",N,N, 0, N, 0.208333],
        # Frontal / Sled & VT / Virtual-High / Front Passenger / HIII-95
        [N,               N,           N,             "Front Passenger",       "HIII-95",            "Head & Neck",           N, N, 0, N, 0.208333],
        [N,               N,           N,             N,                       N,                    "Chest & Abdomen",       N, N, 0, N, 0.208333],
        [N,               N,           N,             N,                       N,                    "Knee, Femur & Pelvis",  N, N, 0, N, 0.208333],
        [N,               N,           N,             N,                       N,                    "Lower Leg, Foot & Ankle",N,N, 0, N, 0.208333],
        # Side / MDB / AEMDB-60 / Driver / WorldSID-50
        ["Side Impact",   "MDB",       "AEMDB-60",    "Driver",                "WorldSID-50",        "Head",                  N, N, 0, N, 2.5],
        [N,               N,           N,             N,                       N,                    "Chest",                 N, N, 0, N, 2.5],
        [N,               N,           N,             N,                       N,                    "Abdomen",               N, N, 0, N, 2.5],
        [N,               N,           N,             N,                       N,                    "Pelvis",                N, N, 0, N, 2.5],
        # Side / MDB / AEMDB-60 / Behind Driver / Q10-Side
        [N,               N,           N,             "Behind Driver",         "Q10-Side",           "Head",                  N, N, 0, N, 1.25],
        [N,               N,           N,             N,                       N,                    "Neck",                  N, N, 0, N, 0.625],
        [N,               N,           N,             N,                       N,                    "Chest",                 N, N, 0, N, 0.625],
        # Side / MDB / AEMDB-60 / Behind Passenger / Q6-Side
        [N,               N,           N,             "Behind Passenger",      "Q6-Side",            "Head",                  N, N, 0, N, 1.25],
        [N,               N,           N,             N,                       N,                    "Neck",                  N, N, 0, N, 0.625],
        [N,               N,           N,             N,                       N,                    "Chest",                 N, N, 0, N, 0.625],
        # Side / Pole / Pole-32 / Driver / WorldSID-50
        [N,               "Pole",      "Pole-32",     "Driver",                "WorldSID-50",        "Head",                  N, N, 0, N, 2.5],
        [N,               N,           N,             N,                       N,                    "Chest",                 N, N, 0, N, 2.5],
        [N,               N,           N,             N,                       N,                    "Abdomen",               N, N, 0, N, 2.5],
        [N,               N,           N,             N,                       N,                    "Pelvis",                N, N, 0, N, 2.5],
        # Side / Farside
        [N,               "Farside",   "Pole-32",     "Driver and Front Passenger","WorldSID-50-Farside","Head",              N, N, 0, N, 2.0],
        [N,               N,           "Main-AEMDB",  "Driver",                "WorldSID-50",        "Head",                  N, N, 0, N, 2.0],
        [N,               N,           "Main-Pole",   "Driver",                "WorldSID-50",        "Head",                  N, N, 0, N, 2.0],
        [N,               N,           "Robustness-AEMDB","Driver",            "WorldSID-50",        "Head",                  N, N, 0, N, 2.0],
        [N,               N,           "Robustness-Pole","Driver",             "WorldSID-50",        "Head",                  N, N, 0, N, 2.0],
        # Rear / Whiplash
        ["Rear Impact",   "Whiplash",  "Rear-Mid",    "Driver",                "BioRID-50",          "Neck",                  N, N, 0, N, 1.5],
        [N,               N,           "Rear-High",   "Driver",                "BioRID-50",          "Neck",                  N, N, 0, N, 1.5],
        [N,               N,           "Static-Front","Front",                 "HPM",                "Neck",                  N, N, 0, N, 1.0],
        [N,               N,           "Static-Rear", "Rear",                  "HPM",                "Neck",                  N, N, 0, N, 1.0],
        # VRU / Head Impact
        ["VRU Impact",    "Head Impact","Headform",    "Driver",                "Adult Headform",     "Cyclist",               N, N, 0, N, N],
        [N,               N,           N,             N,                       N,                    "Adult",                 N, N, 0, N, N],
        [N,               N,           N,             "Driver",                "Child Headform",     "Child",                 N, N, 0, N, N],
        # VRU / Pelvis & Leg Impact
        [N,               "Pelvis & Leg Impact","Upper Leg","Driver",          "Upper legform",      "Pelvis",                N, N, 0, N, 2.5],
        [N,               N,           "Lower Leg",   "Driver",                "aPLI",               "Femur",                 N, N, 0, N, 2.5],
        [N,               N,           N,             N,                       N,                    "Knee & Tibia",          N, N, 0, N, 5.0],
        [N,               N,           N,             N,                       N,                    N,                       N, N, 0, N, N],
        [N,               N,           N,             N,                       N,                    N,                       N, N, 0, N, N],
        [N,               N,           N,             N,                       N,                    N,                       N, N, 0, N, N],
    ]
    # fmt: on
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# CP - Frontal Offset
# ---------------------------------------------------------------------------
def _make_frontal_offset_df(prediction, value_good, value_bad):
    """prediction: "Green" or None; value_good: value for rows that score (0 or 5000);
    value_bad is unused (same as value_good for all numeric rows)."""
    cols = [
        "Loadcase",
        "Seat position",
        "Dummy",
        "Body region",
        "Criteria",
        "HPL",
        "LPL",
        "Capping",
        "OEM Prediction",
        "Value",
    ]
    v = value_good  # shorthand for numeric values
    p = prediction  # shorthand for OEM Prediction
    # fmt: off
    rows = [
        # Driver / THOR-50
        ["MPDB-50","Driver","THOR-50",        "Head & Neck",           "Ares",                             N,     80.0,  N,    N,   v],
        [N,        N,       N,                N,                       "HIC15",                          500.0,  700.0, 700.0, p,   v],
        [N,        N,       N,                N,                       "Ares-3ms",                        72.0,   80.0,  80.0, p,   v],
        [N,        N,       N,                N,                       "Fx,shear",                         1.9,    3.1,   3.1, p,   v],
        [N,        N,       N,                N,                       "Fz,tension",                       2.7,    3.3,   3.3, p,   v],
        [N,        N,       N,                N,                       "My,extension",                    42.0,   57.0,  57.0, p,   v],
        [N,        N,       N,                N,                       "DAMAGE",                            N,      N,    N,   N,   v],
        [N,        N,       N,                "Chest & Abdomen",       "Dchest compression, upper right", 35.0,   60.0,  60.0, p,   v],
        [N,        N,       N,                N,                       "Dchest compression, upper left",  35.0,   60.0,  60.0, p,   v],
        [N,        N,       N,                N,                       "Dchest compression, lower right", 35.0,   60.0,  60.0, p,   v],
        [N,        N,       N,                N,                       "Dchest compression, lower left",  35.0,   60.0,  60.0, p,   v],
        [N,        N,       N,                N,                       "Dabdomen compression, right",     88.0,   88.0,   N,   p,   v],
        [N,        N,       N,                N,                       "Dabdomen compression, left",      88.0,   88.0,   N,   p,   v],
        [N,        N,       N,                N,                       "Shoulder belt load",               N,      N,    N,   N,   v],
        [N,        N,       N,                "Knee, Femur & Pelvis",  "Facetabulum, right",               3.3,    4.1,   N,   p,   v],
        [N,        N,       N,                N,                       "Ffemur, right",                    3.8,    9.1,   N,   p,   v],
        [N,        N,       N,                N,                       "Dknee, right",                     6.0,   15.0,   N,   p,   v],
        [N,        N,       N,                N,                       "Facetabulum, left",                3.3,    4.1,   N,   p,   v],
        [N,        N,       N,                N,                       "Ffemur, left",                     3.8,    9.1,   N,   p,   v],
        [N,        N,       N,                N,                       "Dknee, left",                      6.0,   15.0,   N,   p,   v],
        [N,        N,       N,                "Lower Leg, Foot & Ankle","Itibia, right",                   0.4,    1.3,   N,   p,   v],
        [N,        N,       N,                N,                       "Ftibia, right",                    2.0,    8.0,   N,   p,   v],
        [N,        N,       N,                N,                       "Itibia, left",                     0.4,    1.3,   N,   p,   v],
        [N,        N,       N,                N,                       "Ftibia, left",                     2.0,    8.0,   N,   p,   v],
        [N,        N,       N,                N,                       "Pedal displacement - rearward",    N,      N,    N,   N,   v],
        [N,        N,       N,                N,                       "Pedal displacement - vertical",    N,      N,    N,   N,   v],
        [N,        N,       N,                N,                       "Pedal blocking",                   N,      N,    N,   N,   v],
        # Front Passenger / HIII-05
        [N,        "Front Passenger","HIII-05","Head & Neck",          "Ares",                             N,     80.0,  N,   N,   v],
        [N,        N,       N,                N,                       "HIC15",                          500.0,  700.0, 700.0, p,  v],
        [N,        N,       N,                N,                       "Ares-3ms",                        72.0,   80.0,  80.0, p,  v],
        [N,        N,       N,                N,                       "Fx,shear",                         1.2,    2.0,   N,   p,  v],
        [N,        N,       N,                N,                       "Fz,tension",                       1.7,    2.6,   N,   p,  v],
        [N,        N,       N,                N,                       "My,extension",                    36.0,   49.0,   N,   p,  v],
        [N,        N,       N,                "Chest & Abdomen",       "Dchest compression",              18.0,   34.0,  34.0, p,  v],
        [N,        N,       N,                N,                       "Viscous Criterion",                0.5,    1.0,   1.0, p,  v],
        [N,        N,       N,                N,                       "Shoulder belt load",               N,      N,    N,   N,  v],
        [N,        N,       N,                "Knee, Femur & Pelvis",  "Ffemur, right",                    2.6,    6.2,   N,   p,  v],
        [N,        N,       N,                N,                       "Dknee, right",                     6.0,   15.0,   N,   p,  v],
        [N,        N,       N,                N,                       "Ffemur, left",                     2.6,    6.2,   N,   p,  v],
        [N,        N,       N,                N,                       "Dknee, left",                      6.0,   15.0,   N,   p,  v],
        [N,        N,       N,                "Lower Leg, Foot & Ankle","Itibia, right",                   1.3,    1.3,   N,   p,  v],
        [N,        N,       N,                N,                       "Ftibia, right",                    8.0,    8.0,   N,   p,  v],
        [N,        N,       N,                N,                       "Itibia, left",                     1.3,    1.3,   N,   p,  v],
        [N,        N,       N,                N,                       "Ftibia, left",                     8.0,    8.0,   N,   p,  v],
        # Behind Driver / Q6
        [N,        "Behind Driver","Q6",      "Head",                  "Ares",                             N,     80.0,  N,   N,  v],
        [N,        N,       N,                N,                       "HIC15",                          500.0,  700.0, 700.0, p, v],
        [N,        N,       N,                N,                       "Ares-3ms",                        60.0,   80.0,  80.0, p, v],
        [N,        N,       N,                "Neck",                  "Fz,tension",                       1.7,    2.62,  N,   p, v],
        [N,        N,       N,                N,                       "My,extension",                    36.0,   36.0,   N,   p, v],
        [N,        N,       N,                "Chest",                 "Dchest compression",              30.0,   42.0,   N,   p, v],
        # Behind Passenger / Q10
        [N,        "Behind Passenger","Q10",  "Head",                  "Ares",                             N,     80.0,  N,   N,  v],
        [N,        N,       N,                N,                       "HIC15",                          500.0,  700.0, 700.0, p, v],
        [N,        N,       N,                N,                       "Ares-3ms",                        60.0,   80.0,  80.0, p, v],
        [N,        N,       N,                "Neck",                  "Fz,tension",                       1.7,    2.62,  N,   p, v],
        [N,        N,       N,                N,                       "My,extension",                    49.0,   49.0,   N,   p, v],
        [N,        N,       N,                "Chest",                 "Dchest compression",              56.0,   56.0,  56.0, p, v],
        [N,        N,       N,                N,                       "Ares-3ms",                        41.0,   55.0,  55.0, p, v],
    ]
    # fmt: on
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# CP - Frontal FW
# ---------------------------------------------------------------------------
def _make_frontal_fw_df(prediction, value):
    cols = [
        "Loadcase",
        "Seat position",
        "Dummy",
        "Body region",
        "Criteria",
        "HPL",
        "LPL",
        "Capping",
        "OEM Prediction",
        "Value",
    ]
    v = value
    p = prediction
    # fmt: off
    rows = [
        # Driver / HIII-05
        ["FWDB-35","Driver","HIII-05",        "Head & Neck",           "Ares",                             N,    80.0,  N,    N,  v],
        [N,        N,       N,                N,                       "HIC15",                          500.0, 700.0, 700.0, p,  v],
        [N,        N,       N,                N,                       "Ares-3ms",                        72.0,  80.0,  80.0, p,  v],
        [N,        N,       N,                N,                       "Fx,shear",                         1.2,   2.0,   2.7, p,  v],
        [N,        N,       N,                N,                       "Fz,tension",                       1.7,   2.6,   2.9, p,  v],
        [N,        N,       N,                N,                       "My,extension",                    36.0,  49.0,  57.0, p,  v],
        [N,        N,       N,                "Chest & Abdomen",       "Dchest compression",              18.0,  34.0,  34.0, p,  v],
        [N,        N,       N,                N,                       "Viscous Criterion",                0.5,   1.0,   1.0, p,  v],
        [N,        N,       N,                N,                       "Shoulder belt load",               N,     N,    N,    N,  v],
        [N,        N,       N,                "Knee, Femur & Pelvis",  "Ffemur, right",                    2.6,   6.2,   N,   p,  v],
        [N,        N,       N,                N,                       "Dknee, right",                     6.0,  15.0,   N,   p,  v],
        [N,        N,       N,                N,                       "Ffemur, left",                     2.6,   6.2,   N,   p,  v],
        [N,        N,       N,                N,                       "Dknee, left",                      6.0,  15.0,   N,   p,  v],
        [N,        N,       N,                "Lower Leg, Foot & Ankle","Itibia, right",                   1.3,   1.3,   N,   p,  v],
        [N,        N,       N,                N,                       "Ftibia, right",                    8.0,   8.0,   N,   p,  v],
        [N,        N,       N,                N,                       "Itibia, left",                     1.3,   1.3,   N,   p,  v],
        [N,        N,       N,                N,                       "Ftibia, left",                     8.0,   8.0,   N,   p,  v],
        [N,        N,       N,                N,                       "Pedal displacement - rearward",    N,     N,    N,    N,  v],
        [N,        N,       N,                N,                       "Pedal displacement - vertical",    N,     N,    N,    N,  v],
        [N,        N,       N,                N,                       "Pedal blocking",                   N,     N,    N,    N,  v],
        # Front Passenger / THOR-50
        [N,        "Front Passenger","THOR-50","Head & Neck",          "Ares",                             N,    80.0,  N,    N,  v],
        [N,        N,       N,                N,                       "HIC15",                          500.0, 700.0, 700.0, p,  v],
        [N,        N,       N,                N,                       "Ares-3ms",                        72.0,  80.0,  80.0, p,  v],
        [N,        N,       N,                N,                       "Fx,shear",                         1.9,   3.1,   3.1, p,  v],
        [N,        N,       N,                N,                       "Fz,tension",                       2.7,   3.3,   3.3, p,  v],
        [N,        N,       N,                N,                       "My,extension",                    42.0,  57.0,  57.0, p,  v],
        [N,        N,       N,                N,                       "DAMAGE",                           N,     N,    N,    N,  v],
        [N,        N,       N,                "Chest & Abdomen",       "Dchest compression, upper right", 29.0,  54.0,  54.0, p,  v],
        [N,        N,       N,                N,                       "Dchest compression, upper left",  29.0,  54.0,  54.0, p,  v],
        [N,        N,       N,                N,                       "Dchest compression, lower right", 29.0,  54.0,  54.0, p,  v],
        [N,        N,       N,                N,                       "Dchest compression, lower left",  29.0,  54.0,  54.0, p,  v],
        [N,        N,       N,                N,                       "Dabdomen compression, right",     88.0,  88.0,  N,    p,  v],
        [N,        N,       N,                N,                       "Dabdomen compression, left",      88.0,  88.0,  N,    p,  v],
        [N,        N,       N,                N,                       "Shoulder belt load",               N,     N,    N,    N,  v],
        [N,        N,       N,                "Knee, Femur & Pelvis",  "Facetabulum, right",               3.3,   4.1,  N,    p,  v],
        [N,        N,       N,                N,                       "Ffemur, right",                    3.8,   9.1,  N,    p,  v],
        [N,        N,       N,                N,                       "Dknee, right",                     6.0,  15.0,  N,    p,  v],
        [N,        N,       N,                N,                       "Facetabulum, left",                3.3,   4.1,  N,    p,  v],
        [N,        N,       N,                N,                       "Ffemur, left",                     3.8,   9.1,  N,    p,  v],
        [N,        N,       N,                N,                       "Dknee, left",                      6.0,  15.0,  N,    p,  v],
        [N,        N,       N,                "Lower Leg, Foot & Ankle","Itibia, right",                   0.4,   1.3,  N,    p,  v],
        [N,        N,       N,                N,                       "Ftibia, right",                    2.0,   8.0,  N,    p,  v],
        [N,        N,       N,                N,                       "Itibia, left",                     0.4,   1.3,  N,    p,  v],
        [N,        N,       N,                N,                       "Ftibia, left",                     2.0,   8.0,  N,    p,  v],
        # Behind Passenger / HIII-05
        [N,        "Behind Passenger","HIII-05","Head & Neck",         "Ares",                             N,    80.0,  N,    N,  v],
        [N,        N,       N,                N,                       "HIC15",                          500.0, 700.0, 700.0, p,  v],
        [N,        N,       N,                N,                       "Ares-3ms",                        72.0,  80.0,  80.0, p,  v],
        [N,        N,       N,                N,                       "Fx,shear",                         1.2,   2.0,  N,    p,  v],
        [N,        N,       N,                N,                       "Fz,tension",                       1.7,   2.6,  N,    p,  v],
        [N,        N,       N,                N,                       "My,extension",                    36.0,  49.0,  N,    p,  v],
        [N,        N,       N,                "Chest & Abdomen",       "Dchest compression",              18.0,  34.0,  34.0, p,  v],
        [N,        N,       N,                N,                       "Viscous Criterion",                0.5,   1.0,   1.0, p,  v],
        [N,        N,       N,                N,                       "Shoulder belt load",               N,     N,    N,    N,  v],
        [N,        N,       N,                "Knee, Femur & Pelvis",  "Ffemur, right",                    2.6,   6.2,  N,    p,  v],
        [N,        N,       N,                N,                       "Dknee, right",                     6.0,  15.0,  N,    p,  v],
        [N,        N,       N,                N,                       "Ffemur, left",                     2.6,   6.2,  N,    p,  v],
        [N,        N,       N,                N,                       "Dknee, left",                      6.0,  15.0,  N,    p,  v],
    ]
    # fmt: on
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# CP - Frontal Sled & VT   (no OEM Prediction, no Capping column)
# ---------------------------------------------------------------------------
def _make_frontal_sled_vt_df(value):
    cols = [
        "Loadcase",
        "Seat position",
        "Dummy",
        "Body region",
        "Criteria",
        "HPL",
        "LPL",
        "Value",
    ]
    v = value
    # fmt: off
    rows = [
        # Sled-Mid / Driver / HIII-50
        ["Sled-Mid","Driver","HIII-50",      "Head & Neck",           "Ares",                   N,     80.0, v],
        [N,         N,       N,              N,                       "HIC15",                500.0,  700.0, v],
        [N,         N,       N,              N,                       "Ares-3ms",              72.0,   80.0, v],
        [N,         N,       N,              N,                       "Fx,shear",               1.9,    3.1, v],
        [N,         N,       N,              N,                       "Fz,tension",             2.7,    3.3, v],
        [N,         N,       N,              N,                       "My,extension",           42.0,   57.0, v],
        [N,         N,       N,              "Chest & Abdomen",       "Dchest compression",     20.0,   42.0, v],
        [N,         N,       N,              N,                       "Viscous Criterion",       0.5,    1.0, v],
        [N,         N,       N,              N,                       "Shoulder belt load",      N,      N,  v],
        [N,         N,       N,              "Knee, Femur & Pelvis",  "Ffemur, right",           3.8,    9.1, v],
        [N,         N,       N,              N,                       "Dknee, right",            6.0,   15.0, v],
        [N,         N,       N,              N,                       "Ffemur, left",            3.8,    9.1, v],
        [N,         N,       N,              N,                       "Dknee, left",             6.0,   15.0, v],
        [N,         N,       N,              "Lower Leg, Foot & Ankle","Itibia, right",           0.4,    1.3, v],
        [N,         N,       N,              N,                       "Ftibia, right",           2.0,    8.0, v],
        [N,         N,       N,              N,                       "Itibia, left",            0.4,    1.3, v],
        [N,         N,       N,              N,                       "Ftibia, left",            2.0,    8.0, v],
        # Sled-Mid / Front Passenger / HIII-95
        [N,         "Front Passenger","HIII-95","Head & Neck",        "Ares",                   N,     80.0, v],
        [N,         N,       N,              N,                       "HIC15",                500.0,  700.0, v],
        [N,         N,       N,              N,                       "Ares-3ms",              72.0,   80.0, v],
        [N,         N,       N,              N,                       "Fx,shear",               2.3,    3.8, v],
        [N,         N,       N,              N,                       "Fz,tension",             3.3,    4.0, v],
        [N,         N,       N,              N,                       "My,extension",           56.0,   76.0, v],
        [N,         N,       N,              "Chest & Abdomen",       "Dchest compression",     28.0,   55.0, v],
        [N,         N,       N,              N,                       "Viscous Criterion",       0.5,    1.0, v],
        [N,         N,       N,              "Knee, Femur & Pelvis",  "Ffemur, right",           4.8,   11.5, v],
        [N,         N,       N,              N,                       "Dknee, right",            6.0,   17.0, v],
        [N,         N,       N,              N,                       "Ffemur, left",            4.8,   11.5, v],
        [N,         N,       N,              N,                       "Dknee, left",             6.0,   17.0, v],
        [N,         N,       N,              "Lower Leg, Foot & Ankle","Itibia, right",           1.3,    1.3, v],
        [N,         N,       N,              N,                       "Ftibia, right",          10.0,   10.0, v],
        [N,         N,       N,              N,                       "Itibia, left",            1.3,    1.3, v],
        [N,         N,       N,              N,                       "Ftibia, left",           10.0,   10.0, v],
        # Sled-High / Driver / HIII-95
        ["Sled-High","Driver","HIII-95",     "Head & Neck",           "Ares",                   N,     80.0, v],
        [N,          N,       N,             N,                       "HIC15",                500.0,  700.0, v],
        [N,          N,       N,             N,                       "Ares-3ms",              72.0,   80.0, v],
        [N,          N,       N,             N,                       "Fx,shear",               2.3,    3.8, v],
        [N,          N,       N,             N,                       "Fz,tension",             3.3,    4.0, v],
        [N,          N,       N,             N,                       "My,extension",           56.0,   76.0, v],
        [N,          N,       N,             "Chest & Abdomen",       "Dchest compression",     28.0,   55.0, v],
        [N,          N,       N,             N,                       "Viscous Criterion",       0.5,    1.0, v],
        [N,          N,       N,             "Knee, Femur & Pelvis",  "Ffemur, right",           4.8,   11.5, v],
        [N,          N,       N,             N,                       "Dknee, right",            6.0,   17.0, v],
        [N,          N,       N,             N,                       "Ffemur, left",            4.8,   11.5, v],
        [N,          N,       N,             N,                       "Dknee, left",             6.0,   17.0, v],
        [N,          N,       N,             "Lower Leg, Foot & Ankle","Itibia, right",           1.3,    1.3, v],
        [N,          N,       N,             N,                       "Ftibia, right",          10.0,   10.0, v],
        [N,          N,       N,             N,                       "Itibia, left",            1.3,    1.3, v],
        [N,          N,       N,             N,                       "Ftibia, left",           10.0,   10.0, v],
        # Sled-High / Front Passenger / HIII-05
        [N,          "Front Passenger","HIII-05","Head & Neck",       "Ares",                   N,     80.0, v],
        [N,          N,       N,             N,                       "HIC15",                500.0,  700.0, v],
        [N,          N,       N,             N,                       "Ares-3ms",              72.0,   80.0, v],
        [N,          N,       N,             N,                       "Fx,shear",               1.2,    2.0, v],
        [N,          N,       N,             N,                       "Fz,tension",             1.7,    2.6, v],
        [N,          N,       N,             N,                       "My,extension",           36.0,   49.0, v],
        [N,          N,       N,             "Chest & Abdomen",       "Dchest compression",     22.0,   42.0, v],
        [N,          N,       N,             N,                       "Viscous Criterion",       0.5,    1.0, v],
        [N,          N,       N,             N,                       "Shoulder belt load",      N,      N,  v],
        [N,          N,       N,             "Knee, Femur & Pelvis",  "Ffemur, right",           2.6,    6.2, v],
        [N,          N,       N,             N,                       "Dknee, right",            6.0,   15.0, v],
        [N,          N,       N,             N,                       "Ffemur, left",            2.6,    6.2, v],
        [N,          N,       N,             N,                       "Dknee, left",             6.0,   15.0, v],
        [N,          N,       N,             "Lower Leg, Foot & Ankle","Itibia, right",           1.3,    1.3, v],
        [N,          N,       N,             N,                       "Ftibia, right",           8.0,    8.0, v],
        [N,          N,       N,             N,                       "Itibia, left",            1.3,    1.3, v],
        [N,          N,       N,             N,                       "Ftibia, left",            8.0,    8.0, v],
        # Virtual-Low / Driver / HIII-50
        ["Virtual-Low","Driver","HIII-50",   "Head & Neck",           "Ares",                   N,     80.0, v],
        [N,            N,       N,           N,                       "HIC15",                500.0,  700.0, v],
        [N,            N,       N,           N,                       "Ares-3ms",              72.0,   80.0, v],
        [N,            N,       N,           N,                       "Fx,shear",               1.9,    3.1, v],
        [N,            N,       N,           N,                       "Fz,tension",             2.7,    3.3, v],
        [N,            N,       N,           N,                       "My,extension",           42.0,   57.0, v],
        [N,            N,       N,           "Chest & Abdomen",       "Dchest compression",     20.0,   42.0, v],
        [N,            N,       N,           N,                       "Viscous Criterion",       0.5,    1.0, v],
        [N,            N,       N,           N,                       "Shoulder belt load",      N,      N,  v],
        [N,            N,       N,           "Knee, Femur & Pelvis",  "Ffemur, right",           3.8,    9.1, v],
        [N,            N,       N,           N,                       "Ffemur, left",            3.8,    9.1, v],
        # Virtual-Low / Front Passenger / HIII-05
        [N,            "Front Passenger","HIII-05","Head & Neck",     "Ares",                   N,     80.0, v],
        [N,            N,       N,           N,                       "HIC15",                500.0,  700.0, v],
        [N,            N,       N,           N,                       "Ares-3ms",              72.0,   80.0, v],
        [N,            N,       N,           N,                       "Fx,shear",               1.2,    2.0, v],
        [N,            N,       N,           N,                       "Fz,tension",             1.7,    2.6, v],
        [N,            N,       N,           N,                       "My,extension",           36.0,   49.0, v],
        [N,            N,       N,           "Chest & Abdomen",       "Dchest compression",     18.0,   34.0, v],
        [N,            N,       N,           N,                       "Viscous Criterion",       0.5,    1.0, v],
        [N,            N,       N,           N,                       "Shoulder belt load",      N,      N,  v],
        [N,            N,       N,           "Knee, Femur & Pelvis",  "Ffemur, right",           2.6,    6.2, v],
        [N,            N,       N,           N,                       "Ffemur, left",            2.6,    6.2, v],
        # Virtual-Low / Driver / HIII-05
        ["Virtual-Low","Driver","HIII-05",   "Head & Neck",           "Ares",                   N,     80.0, v],
        [N,            N,       N,           N,                       "HIC15",                500.0,  700.0, v],
        [N,            N,       N,           N,                       "Ares-3ms",              72.0,   80.0, v],
        [N,            N,       N,           N,                       "Fx,shear",               1.2,    2.0, v],
        [N,            N,       N,           N,                       "Fz,tension",             1.7,    2.6, v],
        [N,            N,       N,           N,                       "My,extension",           36.0,   49.0, v],
        [N,            N,       N,           "Chest & Abdomen",       "Dchest compression",     18.0,   34.0, v],
        [N,            N,       N,           N,                       "Viscous Criterion",       0.5,    1.0, v],
        [N,            N,       N,           N,                       "Shoulder belt load",      N,      N,  v],
        [N,            N,       N,           "Knee, Femur & Pelvis",  "Ffemur, right",           2.6,    6.2, v],
        [N,            N,       N,           N,                       "Ffemur, left",            2.6,    6.2, v],
        # Virtual-Low / Front Passenger / HIII-50
        [N,            "Front Passenger","HIII-50","Head & Neck",     "Ares",                   N,     80.0, v],
        [N,            N,       N,           N,                       "HIC15",                500.0,  700.0, v],
        [N,            N,       N,           N,                       "Ares-3ms",              72.0,   80.0, v],
        [N,            N,       N,           N,                       "Fx,shear",               1.9,    3.1, v],
        [N,            N,       N,           N,                       "Fz,tension",             2.7,    3.3, v],
        [N,            N,       N,           N,                       "My,extension",           42.0,   57.0, v],
        [N,            N,       N,           "Chest & Abdomen",       "Dchest compression",     20.0,   42.0, v],
        [N,            N,       N,           N,                       "Viscous Criterion",       0.5,    1.0, v],
        [N,            N,       N,           N,                       "Shoulder belt load",      N,      N,  v],
        [N,            N,       N,           "Knee, Femur & Pelvis",  "Ffemur, right",           3.8,    9.1, v],
        [N,            N,       N,           N,                       "Ffemur, left",            3.8,    9.1, v],
        # Virtual-High / Driver / HIII-05
        ["Virtual-High","Driver","HIII-05",  "Head & Neck",           "Ares",                   N,     80.0, v],
        [N,             N,       N,          N,                       "HIC15",                500.0,  700.0, v],
        [N,             N,       N,          N,                       "Ares-3ms",              72.0,   80.0, v],
        [N,             N,       N,          N,                       "Fx,shear",               1.2,    2.0, v],
        [N,             N,       N,          N,                       "Fz,tension",             1.7,    2.6, v],
        [N,             N,       N,          N,                       "My,extension",           36.0,   49.0, v],
        [N,             N,       N,          "Chest & Abdomen",       "Dchest compression",     22.0,   42.0, v],
        [N,             N,       N,          N,                       "Viscous Criterion",       0.5,    1.0, v],
        [N,             N,       N,          N,                       "Shoulder belt load",      N,      N,  v],
        [N,             N,       N,          "Knee, Femur & Pelvis",  "Ffemur, right",           2.6,    6.2, v],
        [N,             N,       N,          N,                       "Ffemur, left",            2.6,    6.2, v],
        # Virtual-High / Front Passenger / HIII-95
        [N,             "Front Passenger","HIII-95","Head & Neck",    "Ares",                   N,     80.0, v],
        [N,             N,       N,          N,                       "HIC15",                500.0,  700.0, v],
        [N,             N,       N,          N,                       "Ares-3ms",              72.0,   80.0, v],
        [N,             N,       N,          N,                       "Fx,shear",               2.3,    3.8, v],
        [N,             N,       N,          N,                       "Fz,tension",             3.3,    4.0, v],
        [N,             N,       N,          N,                       "My,extension",           56.0,   76.0, v],
        [N,             N,       N,          "Chest & Abdomen",       "Dchest compression",     28.0,   55.0, v],
        [N,             N,       N,          N,                       "Viscous Criterion",       0.5,    1.0, v],
        [N,             N,       N,          "Knee, Femur & Pelvis",  "Ffemur, right",           4.8,   11.5, v],
        [N,             N,       N,          N,                       "Ffemur, left",            4.8,   11.5, v],
    ]
    # fmt: on
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# CP - Side MDB
# ---------------------------------------------------------------------------
def _make_side_mdb_df(prediction, value):
    cols = [
        "Loadcase",
        "Seat position",
        "Dummy",
        "Body region",
        "Criteria",
        "HPL",
        "LPL",
        "Capping",
        "OEM Prediction",
        "Value",
    ]
    v = value
    p = prediction
    # fmt: off
    rows = [
        # Driver / WorldSID-50
        ["AEMDB-60","Driver","WorldSID-50", "Head",    "Ares",                          N,     N,    80.0, N,  v],
        [N,          N,       N,            N,         "HIC15",                       500.0, 700.0, 700.0, p,  v],
        [N,          N,       N,            N,         "Ares-3ms",                     72.0,  80.0,  80.0, p,  v],
        [N,          N,       N,            "Chest",   "Dchest compression, top",       28.0,  50.0,  50.0, p,  v],
        [N,          N,       N,            N,         "Dchest compression, middle",    28.0,  50.0,  50.0, p,  v],
        [N,          N,       N,            N,         "Dchest compression, bottom",    28.0,  50.0,  50.0, p,  v],
        [N,          N,       N,            N,         "Shoulder load",                 N,     N,    N,    N,  v],
        [N,          N,       N,            N,         "Viscous Criterion",             N,     N,    N,    N,  v],
        [N,          N,       N,            "Abdomen", "Dabdomen compression, top",     47.0,  65.0,  65.0, p,  v],
        [N,          N,       N,            N,         "Dabdomen compression, bottom",  47.0,  65.0,  65.0, p,  v],
        [N,          N,       N,            N,         "Viscous Criterion",             N,     N,    N,    N,  v],
        [N,          N,       N,            "Pelvis",  "Fpubic symphysis",              1.7,   2.8,   2.8, p,  v],
        # Behind Driver / Q10-Side
        [N,          "Behind Driver","Q10-Side","Head","Ares",                          N,    80.0,  N,    N,  v],
        [N,          N,       N,            N,         "HIC15",                       500.0, 700.0, 700.0, p,  v],
        [N,          N,       N,            N,         "Ares-3ms",                     60.0,  80.0,  80.0, p,  v],
        [N,          N,       N,            "Neck",    "Fz,tension",                    2.2,   2.2,  N,    p,  v],
        [N,          N,       N,            "Chest",   "Ares-3ms",                     67.0,  67.0,  N,    p,  v],
        # Behind Passenger / Q6-Side
        [N,          "Behind Passenger","Q6-Side","Head","Ares",                        N,    80.0,  N,    N,  v],
        [N,          N,       N,            N,         "HIC15",                       500.0, 700.0, 700.0, p,  v],
        [N,          N,       N,            N,         "Ares-3ms",                     60.0,  80.0,  80.0, p,  v],
        [N,          N,       N,            "Neck",    "Fz,tension",                    2.4,   2.4,  N,    p,  v],
        [N,          N,       N,            "Chest",   "Ares-3ms",                     67.0,  67.0,  N,    p,  v],
    ]
    # fmt: on
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# CP - Side Pole
# ---------------------------------------------------------------------------
def _make_side_pole_df(prediction, value):
    cols = [
        "Loadcase",
        "Seat position",
        "Dummy",
        "Body region",
        "Criteria",
        "HPL",
        "LPL",
        "Capping",
        "OEM Prediction",
        "Value",
    ]
    v = value
    p = prediction
    # fmt: off
    rows = [
        ["Pole-32","Driver","WorldSID-50","Head",   "Ares",                          N,     N,    80.0, N,    v],
        [N,         N,       N,           N,        "HIC15",                       500.0, 700.0, 700.0, p,    v],
        [N,         N,       N,           N,        "Ares-3ms",                     72.0,  80.0,  80.0, p,    v],
        [N,         N,       N,           N,        "Direct contact with pole",      N,     N,    N,    N,  False],
        [N,         N,       N,           "Chest",  "Dchest compression, top",       28.0,  50.0,  55.0, p,    v],
        [N,         N,       N,           N,        "Dchest compression, middle",    28.0,  50.0,  55.0, p,    v],
        [N,         N,       N,           N,        "Dchest compression, bottom",    28.0,  50.0,  55.0, p,    v],
        [N,         N,       N,           N,        "Shoulder load",                 N,     N,    N,    N,    v],
        [N,         N,       N,           N,        "Viscous Criterion",             N,     N,    N,    N,    v],
        [N,         N,       N,           "Abdomen","Dabdomen compression, top",     47.0,  65.0,  65.0, p,    v],
        [N,         N,       N,           N,        "Dabdomen compression, bottom",  47.0,  65.0,  65.0, p,    v],
        [N,         N,       N,           N,        "Viscous Criterion",             N,     N,    N,    N,    v],
        [N,         N,       N,           "Pelvis", "Fpubic symphysis",              1.7,   2.8,   2.8, p,    v],
    ]
    # fmt: on
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# CP - Side Farside
# ---------------------------------------------------------------------------
def _make_side_farside_df(excursion_color, value):
    """excursion_color: "Green" for max, "Red" for min.
    value: 0 for max score (all criteria pass), 5000 for min score."""
    cols = [
        "Loadcase",
        "Seat position",
        "Dummy",
        "Body region",
        "Criteria",
        "HPL",
        "LPL",
        "Capping",
        "Value",
    ]
    ec = excursion_color
    v = value
    # fmt: off
    rows = [
        # Pole-32 — Driver
        ["Pole-32",              "Driver",          "WorldSID-50-Farside","Head", "HIC15",                     700.0, 700.0,   N,   v],
        [N,                      N,                 N,                   N,      "Ares-3ms",                   80.0,  80.0,    N,   v],
        # Pole-32 — Front Passenger
        [N,                      "Front Passenger", "WorldSID-50-Farside","Head", "HIC15",                     700.0, 700.0,   N,   v],
        [N,                      N,                 N,                   N,      "Ares-3ms",                   80.0,  80.0,    N,   v],
        # Main-AEMDB
        ["Main-AEMDB",           "Driver",          "WorldSID-50",       "Head", "Ares",                       N,     N,     80.0,  v],
        [N,                      N,                 N,                   N,      "HIC15",                      N,     N,    700.0,  v],
        [N,                      N,                 N,                   N,      "Ares-3ms",                   N,     N,     80.0,  v],
        [N,                      N,                 N,                   N,      "Farside excursion",          N,     N,      N,   ec],
        [N,                      N,                 N,                   "Neck", "Ftension, upper",            N,     N,     3.74,  v],
        [N,                      N,                 N,                   N,      "Mflexion, upper",            N,     N,   248.0,   v],
        [N,                      N,                 N,                   N,      "Mextension, upper",          N,     N,    50.0,   v],
        [N,                      N,                 N,                   N,      "Ftension, lower",            N,     N,     3.74,  v],
        [N,                      N,                 N,                   N,      "Mflexion, lower",            N,     N,   248.0,   v],
        [N,                      N,                 N,                   N,      "Mextension, lower",          N,     N,   100.0,   v],
        [N,                      N,                 N,                   "Chest","Dchest compression, top",    N,     N,    50.0,   v],
        [N,                      N,                 N,                   N,      "Dchest compression, middle", N,     N,    50.0,   v],
        [N,                      N,                 N,                   N,      "Dchest compression, bottom", N,     N,    50.0,   v],
        [N,                      N,                 N,                   "Abdomen","Dabdomen compression, top",N,     N,    65.0,   v],
        [N,                      N,                 N,                   N,      "Dabdomen compression, bottom",N,    N,    65.0,   v],
        [N,                      N,                 N,                   "Pelvis","Fpubic symphysis",          N,     N,      N,   v],
        [N,                      N,                 N,                   N,      "Fy lumbar",                  N,     N,      N,   v],
        [N,                      N,                 N,                   N,      "Fz lumbar",                  N,     N,      N,   v],
        [N,                      N,                 N,                   N,      "Mx lumbar",                  N,     N,      N,   v],
        # Main-Pole
        ["Main-Pole",            "Driver",          "WorldSID-50",       "Head", "Ares",                       N,     N,     80.0,  v],
        [N,                      N,                 N,                   N,      "HIC15",                      N,     N,    700.0,  v],
        [N,                      N,                 N,                   N,      "Ares-3ms",                   N,     N,     80.0,  v],
        [N,                      N,                 N,                   N,      "Farside excursion",          N,     N,      N,   ec],
        [N,                      N,                 N,                   "Neck", "Ftension, upper",            N,     N,     3.74,  v],
        [N,                      N,                 N,                   N,      "Mflexion, upper",            N,     N,   248.0,   v],
        [N,                      N,                 N,                   N,      "Mextension, upper",          N,     N,    50.0,   v],
        [N,                      N,                 N,                   N,      "Ftension, lower",            N,     N,     3.74,  v],
        [N,                      N,                 N,                   N,      "Mflexion, lower",            N,     N,   248.0,   v],
        [N,                      N,                 N,                   N,      "Mextension, lower",          N,     N,   100.0,   v],
        [N,                      N,                 N,                   "Chest","Dchest compression, top",    N,     N,    50.0,   v],
        [N,                      N,                 N,                   N,      "Dchest compression, middle", N,     N,    50.0,   v],
        [N,                      N,                 N,                   N,      "Dchest compression, bottom", N,     N,    50.0,   v],
        [N,                      N,                 N,                   "Abdomen","Dabdomen compression, top",N,     N,    65.0,   v],
        [N,                      N,                 N,                   N,      "Dabdomen compression, bottom",N,    N,    65.0,   v],
        [N,                      N,                 N,                   "Pelvis","Fpubic symphysis",          N,     N,      N,   v],
        [N,                      N,                 N,                   N,      "Fy lumbar",                  N,     N,      N,   v],
        [N,                      N,                 N,                   N,      "Fz lumbar",                  N,     N,      N,   v],
        [N,                      N,                 N,                   N,      "Mx lumbar",                  N,     N,      N,   v],
        # Robustness loadcases (excursion color only)
        ["Robustness-AEMDB60",   "Driver",          "WorldSID-50",       "Head", "Farside excursion",          N,     N,      N,  ec],
        ["Robustness-AEMDB75high","Driver",         "WorldSID-50",       "Head", "Farside excursion",          N,     N,      N,  ec],
        ["Robustness-AEMDB90",   "Driver",          "WorldSID-50",       "Head", "Farside excursion",          N,     N,      N,  ec],
        ["Robustness-AEMDB90high","Driver",         "WorldSID-50",       "Head", "Farside excursion",          N,     N,      N,  ec],
        ["Robustness-Pole75high","Driver",          "WorldSID-50",       "Head", "Farside excursion",          N,     N,      N,  ec],
        ["Robustness-Pole90",    "Driver",          "WorldSID-50",       "Head", "Farside excursion",          N,     N,      N,  ec],
    ]
    # fmt: on
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# CP - Rear Whiplash   (no OEM Prediction column)
# ---------------------------------------------------------------------------
def _make_rear_whiplash_df(value):
    cols = [
        "Loadcase",
        "Seat position",
        "Dummy",
        "Body region",
        "Criteria",
        "HPL",
        "LPL",
        "Capping",
        "Value",
    ]
    v = value
    # fmt: off
    rows = [
        # Rear-Mid
        ["Rear-Mid",    "Driver","BioRID-50","Neck","Head rebound velocity",               N,       N,     5.2,   v],
        [N,             N,       N,          N,     "T-HRC start",                         N,       N,    92.0,   v],
        [N,             N,       N,          N,     "NIC",                                11.0,   24.0,   27.0,   v],
        [N,             N,       N,          N,     "Nkm",                                 N,       N,     0.69,  v],
        [N,             N,       N,          N,     "Upper Neck Fx (+)",                  30.0,  190.0,  290.0,   v],
        [N,             N,       N,          N,     "Upper Neck Fx (-)",                   N,       N,   360.0,   v],
        [N,             N,       N,          N,     "Upper Neck Fz",                     360.0,  750.0,  900.0,   v],
        [N,             N,       N,          N,     "Upper Neck My extension",             N,       N,    30.0,   v],
        [N,             N,       N,          N,     "Upper Neck My flexion",               N,       N,    30.0,   v],
        [N,             N,       N,          N,     "Lower Neck Fx (abs)",                 N,       N,   360.0,   v],
        [N,             N,       N,          N,     "Lower Neck My extension",             N,       N,    30.0,   v],
        [N,             N,       N,          N,     "Lower Neck My flexion",               N,       N,    30.0,   v],
        [N,             N,       N,          N,     "T1 acceleration",                     N,       N,    15.55,  v],
        # Rear-High
        ["Rear-High",   "Driver","BioRID-50","Neck","Head rebound velocity",               N,       N,     6.0,   v],
        [N,             N,       N,          N,     "T-HRC start",                         N,       N,    92.0,   v],
        [N,             N,       N,          N,     "NIC",                                13.0,   23.0,   25.5,   v],
        [N,             N,       N,          N,     "Nkm",                                 N,       N,     0.78,  v],
        [N,             N,       N,          N,     "Upper Neck Fx (+)",                  30.0,  210.0,  364.0,   v],
        [N,             N,       N,          N,     "Upper Neck Fx (-)",                   N,       N,   360.0,   v],
        [N,             N,       N,          N,     "Upper Neck Fz",                     470.0,  770.0, 1024.0,   v],
        [N,             N,       N,          N,     "Upper Neck My extension",             N,       N,    30.0,   v],
        [N,             N,       N,          N,     "Upper Neck My flexion",               N,       N,    30.0,   v],
        [N,             N,       N,          N,     "Lower Neck Fx (abs)",                 N,       N,   360.0,   v],
        [N,             N,       N,          N,     "Lower Neck My extension",             N,       N,    30.0,   v],
        [N,             N,       N,          N,     "Lower Neck My flexion",               N,       N,    30.0,   v],
        [N,             N,       N,          N,     "T1 acceleration",                     N,       N,    17.8,   v],
        [N,             N,       N,          N,     "Seatback deflection",                 N,       N,    32.0,   v],
        # Static-Front  (HPL/LPL drive pass/fail; Value=0 for max score rows)
        ["Static-Front","Driver","HPM",      "Neck","Effective height - Test position",  825.0,  790.0,   N,    1000],
        [N,             N,       N,          N,     "Effective height - Lowest and rearmost",790.0,789.99,N,   1000],
        [N,             N,       N,          N,     "Effective height modifier",           N,       N,    N,    1000],
        [N,             N,       N,          N,     "Backset - Test position",            45.0,   45.0,   N,      0],
        [N,             N,       N,          N,     "Backset - Lowest and rearmost",      70.0,   70.0,   N,      0],
        [N,  "Front Passenger","HPM",        "Neck","Effective height - Test position",  825.0,  790.0,   N,    1000],
        [N,             N,       N,          N,     "Effective height - Lowest and rearmost",790.0,789.99,N,   1000],
        [N,             N,       N,          N,     "Effective height modifier",           N,       N,    N,    1000],
        [N,             N,       N,          N,     "Backset - Test position",            45.0,   45.0,   N,      0],
        [N,             N,       N,          N,     "Backset - Lowest and rearmost",      70.0,   70.0,   N,      0],
        # Static-Rear
        ["Static-Rear", "Rear",  "HPM",      "Neck","Effective height - Lowest",         720.0,  719.99,  N,    1000],
        [N,             N,       N,          N,     "Effective height - Highest",         770.0,  769.99,  N,    1000],
        [N,             N,       N,          N,     "Effective height modifier",           N,       N,    N,    1000],
        [N,             N,       N,          N,     "Backset - Lowest",                  160.128,160.128,  N,      0],
        [N,             N,       N,          N,     "Backset - Mid",                     160.128,160.128,  N,      0],
    ]
    # fmt: on
    return pd.DataFrame(rows, columns=cols)


def _make_rear_whiplash_df_min():
    """Min score version of Rear Whiplash: BioRID rows → 5000, Static rows use worst-case."""
    cols = [
        "Loadcase",
        "Seat position",
        "Dummy",
        "Body region",
        "Criteria",
        "HPL",
        "LPL",
        "Capping",
        "Value",
    ]
    # fmt: off
    rows = [
        # Rear-Mid (all criteria → 5000 to force worst colour)
        ["Rear-Mid",    "Driver","BioRID-50","Neck","Head rebound velocity",               N,       N,     5.2,   5000],
        [N,             N,       N,          N,     "T-HRC start",                         N,       N,    92.0,   5000],
        [N,             N,       N,          N,     "NIC",                                11.0,   24.0,   27.0,   5000],
        [N,             N,       N,          N,     "Nkm",                                 N,       N,     0.69,  5000],
        [N,             N,       N,          N,     "Upper Neck Fx (+)",                  30.0,  190.0,  290.0,   5000],
        [N,             N,       N,          N,     "Upper Neck Fx (-)",                   N,       N,   360.0,   5000],
        [N,             N,       N,          N,     "Upper Neck Fz",                     360.0,  750.0,  900.0,   5000],
        [N,             N,       N,          N,     "Upper Neck My extension",             N,       N,    30.0,   5000],
        [N,             N,       N,          N,     "Upper Neck My flexion",               N,       N,    30.0,   5000],
        [N,             N,       N,          N,     "Lower Neck Fx (abs)",                 N,       N,   360.0,   5000],
        [N,             N,       N,          N,     "Lower Neck My extension",             N,       N,    30.0,   5000],
        [N,             N,       N,          N,     "Lower Neck My flexion",               N,       N,    30.0,   5000],
        [N,             N,       N,          N,     "T1 acceleration",                     N,       N,    15.55,  5000],
        # Rear-High (all criteria → 5000)
        ["Rear-High",   "Driver","BioRID-50","Neck","Head rebound velocity",               N,       N,     6.0,   5000],
        [N,             N,       N,          N,     "T-HRC start",                         N,       N,    92.0,   5000],
        [N,             N,       N,          N,     "NIC",                                13.0,   23.0,   25.5,   5000],
        [N,             N,       N,          N,     "Nkm",                                 N,       N,     0.78,  5000],
        [N,             N,       N,          N,     "Upper Neck Fx (+)",                  30.0,  210.0,  364.0,   5000],
        [N,             N,       N,          N,     "Upper Neck Fx (-)",                   N,       N,   360.0,   5000],
        [N,             N,       N,          N,     "Upper Neck Fz",                     790.0,  825.0, 1024.0,   5000],
        [N,             N,       N,          N,     "Upper Neck My extension",             N,       N,    30.0,   5000],
        [N,             N,       N,          N,     "Upper Neck My flexion",               N,       N,    30.0,   5000],
        [N,             N,       N,          N,     "Lower Neck Fx (abs)",                 N,       N,   360.0,   5000],
        [N,             N,       N,          N,     "Lower Neck My extension",             N,       N,    30.0,   5000],
        [N,             N,       N,          N,     "Lower Neck My flexion",               N,       N,    30.0,   5000],
        [N,             N,       N,          N,     "T1 acceleration",                     N,       N,    17.8,   5000],
        [N,             N,       N,          N,     "Seatback deflection",                 N,       N,    32.0,   5000],
        # Static-Front: effective height rows fail, backset rows fail
        ["Static-Front","Driver","HPM",      "Neck","Effective height - Test position",  825.0,  790.0,   N,      0],
        [N,             N,       N,          N,     "Effective height - Lowest and rearmost",790.0,789.99,N,   5000],
        [N,             N,       N,          N,     "Effective height modifier",           N,       N,    N,      0],
        [N,             N,       N,          N,     "Backset - Test position",            45.0,   45.0,   N,   5000],
        [N,             N,       N,          N,     "Backset - Lowest and rearmost",      70.0,   70.0,   N,   5000],
        [N,  "Front Passenger","HPM",        "Neck","Effective height - Test position",  825.0,  790.0,   N,      0],
        [N,             N,       N,          N,     "Effective height - Lowest and rearmost",790.0,789.99,N,   5000],
        [N,             N,       N,          N,     "Effective height modifier",           N,       N,    N,      0],
        [N,             N,       N,          N,     "Backset - Test position",            45.0,   45.0,   N,   5000],
        [N,             N,       N,          N,     "Backset - Lowest and rearmost",      70.0,   70.0,   N,   5000],
        # Static-Rear: effective height rows pass, backset rows fail
        ["Static-Rear", "Rear",  "HPM",      "Neck","Effective height - Lowest",         720.0,  719.99,  N,      0],
        [N,             N,       N,          N,     "Effective height - Highest",         770.0,  769.99,  N,      0],
        [N,             N,       N,          N,     "Effective height modifier",           N,       N,    N,      0],
        [N,             N,       N,          N,     "Backset - Lowest",                    0.0,     0.0,  N,   5000],
        [N,             N,       N,          N,     "Backset - Mid",                       0.0,     0.0,  N,   5000],
    ]
    # fmt: on
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# CP - VRU Head Impact
# ---------------------------------------------------------------------------
def _make_vru_head_impact_df(prediction, value):
    cols = [
        "Loadcase",
        "Seat position",
        "Dummy",
        "Body region",
        "Test point",
        "Criteria",
        "HPL",
        "LPL",
        "Capping",
        "OEM Prediction",
        "Value",
    ]
    v = value
    p = prediction
    # fmt: off
    rows = [
        ["Headform","Driver","Adult Headform","Adult",   "(6, 9)",   "HIC15", 650, 1700, N, p, v],
        [N,         N,       N,              "Cyclist",  "(18, -6)", "HIC15", 650, 1700, N, p, v],
        [N,         N,       N,              N,          "(15, -7)", "HIC15", 650, 1700, N, p, v],
        [N,         N,       N,              N,          "(13, -2)", "HIC15", 650, 1700, N, p, v],
        [N,         N,       N,              N,          "(17, -10)","HIC15", 650, 1700, N, p, v],
        [N,         N,       N,              N,          "(12, -8)", "HIC15", 650, 1700, N, p, v],
        [N,         N,       "Child Headform","Child",   "(0, -3)",  "HIC15", 650, 1700, N, p, v],
        [N,         N,       N,              N,          "(1, -7)",  "HIC15", 650, 1700, N, p, v],
        [N,         N,       N,              N,          "(1, -8)",  "HIC15", 650, 1700, N, p, v],
        [N,         N,       N,              N,          "(0, 9)",   "HIC15", 650, 1700, N, p, v],
    ]
    # fmt: on
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# CP - VRU Pelvis & Leg Impact
# ---------------------------------------------------------------------------
def _make_vru_pelvis_leg_df(prediction, value):
    cols = [
        "Loadcase",
        "Seat position",
        "Dummy",
        "Body region",
        "Test point",
        "Criteria",
        "HPL",
        "LPL",
        "Capping",
        "OEM Prediction",
        "Value",
    ]
    v = value
    p = prediction
    # fmt: off
    rows = [
        # Upper Leg / Driver / Upper legform / Pelvis
        ["Upper Leg","Driver","Upper legform","Pelvis","(0, 10)", "Sum of forces", 5, 6, N, p, v],
        [N,          N,       N,             N,        "(0, -9)", "Sum of forces", 5, 6, N, p, v],
        [N,          N,       N,             N,        "(0, 7)",  "Sum of forces", 5, 6, N, p, v],
        [N,          N,       N,             N,        "(0, -2)", "Sum of forces", 5, 6, N, p, v],
        [N,          N,       N,             N,        "(0, -5)", "Sum of forces", 5, 6, N, p, v],
        # Lower Leg / Driver / aPLI / Femur
        ["Lower Leg","Driver","aPLI",        "Femur",  "(1, -1)", "Bending moment, F1", 390, 440, N, p, v],
        [N,          N,       N,             N,        N,         "Bending moment, F2", 390, 440, N, p, v],
        [N,          N,       N,             N,        N,         "Bending moment, F3", 390, 440, N, p, v],
        [N,          N,       N,             N,        "(1, -8)", "Bending moment, F1", 390, 440, N, p, v],
        [N,          N,       N,             N,        N,         "Bending moment, F2", 390, 440, N, p, v],
        [N,          N,       N,             N,        N,         "Bending moment, F3", 390, 440, N, p, v],
        [N,          N,       N,             N,        "(1, 2)",  "Bending moment, F1", 390, 440, N, p, v],
        [N,          N,       N,             N,        N,         "Bending moment, F2", 390, 440, N, p, v],
        [N,          N,       N,             N,        N,         "Bending moment, F3", 390, 440, N, p, v],
        [N,          N,       N,             N,        "(1, 8)",  "Bending moment, F1", 390, 440, N, p, v],
        [N,          N,       N,             N,        N,         "Bending moment, F2", 390, 440, N, p, v],
        [N,          N,       N,             N,        N,         "Bending moment, F3", 390, 440, N, p, v],
        [N,          N,       N,             N,        "(1, -5)", "Bending moment, F1", 390, 440, N, p, v],
        [N,          N,       N,             N,        N,         "Bending moment, F2", 390, 440, N, p, v],
        [N,          N,       N,             N,        N,         "Bending moment, F3", 390, 440, N, p, v],
        # Knee
        [N,          N,       N,             "Knee",   "(2, -1)", "MCL elongation", 27, 32, N, p, v],
        [N,          N,       N,             N,        "(2, -8)", "MCL elongation", 27, 32, N, p, v],
        [N,          N,       N,             N,        "(2, 2)",  "MCL elongation", 27, 32, N, p, v],
        [N,          N,       N,             N,        "(2, 8)",  "MCL elongation", 27, 32, N, p, v],
        [N,          N,       N,             N,        "(2, -5)", "MCL elongation", 27, 32, N, p, v],
        # Tibia
        [N,          N,       N,             "Tibia",  "(2, -1)", "Bending moment, T1", 275, 320, N, p, v],
        [N,          N,       N,             N,        N,         "Bending moment, T2", 275, 320, N, p, v],
        [N,          N,       N,             N,        N,         "Bending moment, T3", 275, 320, N, p, v],
        [N,          N,       N,             N,        N,         "Bending moment, T4", 275, 320, N, p, v],
        [N,          N,       N,             N,        "(2, -8)", "Bending moment, T1", 275, 320, N, p, v],
        [N,          N,       N,             N,        N,         "Bending moment, T2", 275, 320, N, p, v],
        [N,          N,       N,             N,        N,         "Bending moment, T3", 275, 320, N, p, v],
        [N,          N,       N,             N,        N,         "Bending moment, T4", 275, 320, N, p, v],
        [N,          N,       N,             N,        "(2, 2)",  "Bending moment, T1", 275, 320, N, p, v],
        [N,          N,       N,             N,        N,         "Bending moment, T2", 275, 320, N, p, v],
        [N,          N,       N,             N,        N,         "Bending moment, T3", 275, 320, N, p, v],
        [N,          N,       N,             N,        N,         "Bending moment, T4", 275, 320, N, p, v],
        [N,          N,       N,             N,        "(2, 8)",  "Bending moment, T1", 275, 320, N, p, v],
        [N,          N,       N,             N,        N,         "Bending moment, T2", 275, 320, N, p, v],
        [N,          N,       N,             N,        N,         "Bending moment, T3", 275, 320, N, p, v],
        [N,          N,       N,             N,        N,         "Bending moment, T4", 275, 320, N, p, v],
        [N,          N,       N,             N,        "(2, -5)", "Bending moment, T1", 275, 320, N, p, v],
        [N,          N,       N,             N,        N,         "Bending moment, T2", 275, 320, N, p, v],
        [N,          N,       N,             N,        N,         "Bending moment, T3", 275, 320, N, p, v],
        [N,          N,       N,             N,        N,         "Bending moment, T4", 275, 320, N, p, v],
    ]
    # fmt: on
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# CP - VRU Prediction Points
# Mirrors the test points from _make_vru_head_impact_df so that
# compute_vru_bodyregion_score() gets a non-empty vru_test_points_all and
# can compute max_points > 0 for each body region.
# ---------------------------------------------------------------------------
def _make_vru_prediction_points_df(color: VruPredictionColor):
    # (row, col) pairs parsed from the test point strings in _make_vru_head_impact_df
    # Adult Headform: Adult region
    # Child Headform: Child region
    test_points = [
        # Adult
        (6, 9, "Adult"),
        (18, -6, "Cyclist"),
        (15, -7, "Cyclist"),
        (13, -2, "Cyclist"),
        (17, -10, "Cyclist"),
        (12, -8, "Cyclist"),
        # Child
        (0, -3, "Child"),
        (1, -7, "Child"),
        (1, -8, "Child"),
        (0, 9, "Child"),
    ]
    return pd.DataFrame(
        [
            {"row": r, "col": c, "color": color, "loadcase_name": "Headform"}
            for r, c, _ in test_points
        ]
    )


# ---------------------------------------------------------------------------
# Full dfs builders
# ---------------------------------------------------------------------------
def _make_dfs_max():
    return {
        "Input parameters": _make_input_parameters_df(),
        "Test Scores": _make_test_scores_df(),
        "CP - Dummy Scores": _make_dummy_scores_df(),
        "CP - Body region scores": _make_body_region_scores_df(),
        "CP - Frontal Offset": _make_frontal_offset_df(
            prediction=G, value_good=0, value_bad=0
        ),
        "CP - Frontal FW": _make_frontal_fw_df(prediction=G, value=0),
        "CP - Frontal Sled & VT": _make_frontal_sled_vt_df(value=0),
        "CP - Side MDB": _make_side_mdb_df(prediction=G, value=0),
        "CP - Side Pole": _make_side_pole_df(prediction=G, value=0),
        "CP - Side Farside": _make_side_farside_df(excursion_color=G, value=0),
        "CP - Rear Whiplash": _make_rear_whiplash_df(value=0),
        "CP - VRU Head Impact": _make_vru_head_impact_df(prediction=G, value=0),
        "CP - VRU Pelvis & Leg Impact": _make_vru_pelvis_leg_df(prediction=G, value=0),
        "CP - VRU Prediction Points": _make_vru_prediction_points_df(
            VruPredictionColor.GREEN
        ),
    }


def _make_dfs_min():
    return {
        "Input parameters": _make_input_parameters_df(),
        "Test Scores": _make_test_scores_df(),
        "CP - Dummy Scores": _make_dummy_scores_df(),
        "CP - Body region scores": _make_body_region_scores_df(),
        "CP - Frontal Offset": _make_frontal_offset_df(
            prediction=None, value_good=5000, value_bad=5000
        ),
        "CP - Frontal FW": _make_frontal_fw_df(prediction=None, value=5000),
        "CP - Frontal Sled & VT": _make_frontal_sled_vt_df(value=5000),
        "CP - Side MDB": _make_side_mdb_df(prediction=None, value=5000),
        "CP - Side Pole": _make_side_pole_df(prediction=None, value=5000),
        "CP - Side Farside": _make_side_farside_df(excursion_color=R, value=5000),
        "CP - Rear Whiplash": _make_rear_whiplash_df_min(),
        "CP - VRU Head Impact": _make_vru_head_impact_df(prediction=R, value=5000),
        "CP - VRU Pelvis & Leg Impact": _make_vru_pelvis_leg_df(
            prediction=R, value=5000
        ),
        "CP - VRU Prediction Points": _make_vru_prediction_points_df(
            VruPredictionColor.RED
        ),
    }


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------
class TestCrashProtectionFileScore(unittest.TestCase):

    def test_max_score_equals_max(self):
        """All criteria at green / below LPL → score == max_score for every stage subelement."""
        dfs = _make_dfs_max()
        result = calculate_score(dfs)
        ts = result["Test Scores"]

        # Build a mapping stage_subelement → (score, max_score) from the result DataFrame
        last_subelement = None
        scores = {}
        for _, row in ts.iterrows():
            if not pd.isna(row.get("Stage subelement")):
                last_subelement = row["Stage subelement"]
            if last_subelement is not None:
                scores[last_subelement] = (row["Score"], row["Max score"])

        for subelement in STAGE_SUBELEMENT_LIST:
            score, max_score = scores[subelement]
            self.assertAlmostEqual(
                score,
                max_score,
                places=3,
                msg=f"Stage subelement '{subelement}': score {score} != max_score {max_score}",
            )

    def test_min_score_all_zero(self):
        """All criteria at worst values → score == 0 for every stage subelement."""
        dfs = _make_dfs_min()
        result = calculate_score(dfs)
        ts = result["Test Scores"]

        last_subelement = None
        scores = {}
        for _, row in ts.iterrows():
            if not pd.isna(row.get("Stage subelement")):
                last_subelement = row["Stage subelement"]
            if last_subelement is not None:
                scores[last_subelement] = row["Score"]

        for subelement in STAGE_SUBELEMENT_LIST:
            score = scores[subelement]
            self.assertEqual(
                score,
                0,
                msg=f"Stage subelement '{subelement}': expected score 0, got {score}",
            )

    def test_backset_limit_recomputed_from_torso_angle_ignores_stale_sheet_values(self):
        """Regression test: reported bug where a rear seat Backset check FAILed even
        though the measured value was below the torso-angle-derived limit.

        Torso angle = 25 deg -> (ΔCP X)LIMIT = 7.128*25 + 153 = 331.2 mm (per Euro NCAP
        Rear Impact Protocol v1.1 section 3.1.3.3/3.1.4.2). The 'CP - Rear Whiplash'
        sheet's HPL/LPL for Backset - Lowest/Mid are deliberately stale/wrong (100.0),
        lower than the measured value (313.8) but far below the true limit. Scoring
        must recompute the limit from torso angle rather than trust the stale cells,
        so the seating position should PASS (score == max_score for 'Whiplash').
        """
        dfs = _make_dfs_max()

        input_parameters_df = _make_input_parameters_df().copy()
        input_parameters_df.loc[
            input_parameters_df["Input parameter"] == "Torso angle (rear)", "Value"
        ] = 25
        dfs["Input parameters"] = input_parameters_df

        rear_whiplash_df = _make_rear_whiplash_df(value=0).copy()
        rear_whiplash_df["Value"] = rear_whiplash_df["Value"].astype(float)
        backset_mask = rear_whiplash_df["Criteria"].isin(
            ["Backset - Lowest", "Backset - Mid"]
        )
        rear_whiplash_df.loc[backset_mask, "HPL"] = 100.0
        rear_whiplash_df.loc[backset_mask, "LPL"] = 100.0
        rear_whiplash_df.loc[backset_mask, "Value"] = 313.8
        dfs["CP - Rear Whiplash"] = rear_whiplash_df

        result = calculate_score(dfs)
        ts = result["Test Scores"]

        last_subelement = None
        scores = {}
        for _, row in ts.iterrows():
            if not pd.isna(row.get("Stage subelement")):
                last_subelement = row["Stage subelement"]
            if last_subelement is not None:
                scores[last_subelement] = (row["Score"], row["Max score"])

        score, max_score = scores["Whiplash"]
        self.assertAlmostEqual(
            score,
            max_score,
            places=3,
            msg=(
                f"Whiplash score {score} != max score {max_score}: Backset limit was "
                "not recomputed from torso angle at scoring time"
            ),
        )

    def test_hpl_lpl_tamper_reaching_calculate_score_directly_is_neutralized(self):
        """Regression test: callers may never
        call the file-based preprocess()/compute_score() CLI (which
        already neutralized HPL/LPL tampering via rebuild_trusted_input) --
        instead calling calculate_score(dfs) directly with dataframes
        assembled from parquet storage, so a tampered HPL/LPL used to flow
        straight into scoring unchecked. Tamper HIC15's HPL/LPL (row index 1 of
        "CP - Frontal Offset", normally 500.0/700.0, matching the packaged
        template) to absurd values with no file and no preprocess() call in
        between, call calculate_score(dfs) directly, and confirm both the
        resulting sheet and the score reflect the official template's
        thresholds, not the tamper.
        """
        dfs = _make_dfs_max()
        frontal_offset = dfs["CP - Frontal Offset"].copy()
        self.assertEqual(frontal_offset.loc[1, "Criteria"], "HIC15")
        self.assertEqual(frontal_offset.loc[1, "HPL"], 500.0)
        self.assertEqual(frontal_offset.loc[1, "LPL"], 700.0)
        frontal_offset.loc[1, "HPL"] = -999
        frontal_offset.loc[1, "LPL"] = 99999
        dfs["CP - Frontal Offset"] = frontal_offset

        result = calculate_score(dfs)

        sanitized_row = result["CP - Frontal Offset"].loc[1]
        self.assertEqual(sanitized_row["Criteria"], "HIC15")
        self.assertEqual(sanitized_row["HPL"], 500.0)
        self.assertEqual(sanitized_row["LPL"], 700.0)

        ts = result["Test Scores"]
        last_subelement = None
        scores = {}
        for _, row in ts.iterrows():
            if not pd.isna(row.get("Stage subelement")):
                last_subelement = row["Stage subelement"]
            if last_subelement is not None:
                scores[last_subelement] = (row["Score"], row["Max score"])

        score, max_score = scores["Offset"]
        self.assertAlmostEqual(
            score,
            max_score,
            places=3,
            msg=(
                f"Offset score {score} != max score {max_score}: tampered "
                "HPL/LPL reached scoring instead of being neutralized"
            ),
        )


class TestAddVruSheetsToDfsRebuildsTrustedInput(unittest.TestCase):
    """Regression test for the other half of the same gap:
    add_vru_sheets_to_dfs(dfs) -- the dataframe-in/dataframe-out
    entry point used for VRU test-run selection -- must also reject a
    tampered "Input parameters" sheet, exactly like rebuild_trusted_input
    already does for the file-based preprocess() CLI command."""

    def test_reordered_input_parameters_rows_are_rejected(self):
        from importlib.resources import files

        from euroncap_rating_2026 import common
        from euroncap_rating_2026.crash_protection.preprocess import (
            add_vru_sheets_to_dfs,
        )

        pristine_path = str(files("data").joinpath("cp_template.xlsx"))
        dfs = common.read_excel_file_to_dfs(pristine_path)
        input_parameters = dfs["Input parameters"].copy()
        input_parameters.iloc[[0, 1]] = input_parameters.iloc[[1, 0]].values
        dfs["Input parameters"] = input_parameters

        with self.assertRaises(common.TemplateIntegrityError):
            add_vru_sheets_to_dfs(dfs)


class TestUpdateLoadcaseClearsStalePredictionCheck(unittest.TestCase):
    """Re-scoring a dataframe that already carries Colour/Prediction.Check
    values (callers may re-score previously scored sheets) must blank them
    for criteria whose assessment was cleared - the write has
    to be unconditional, not skipped when prediction_result is None."""

    def test_gated_criteria_blank_stale_colour_and_prediction_check(self):
        from euroncap_rating_2026 import common
        from euroncap_rating_2026.crash_protection.body_region import BodyRegion
        from euroncap_rating_2026.crash_protection.compute_score import update_loadcase
        from euroncap_rating_2026.crash_protection.criteria import (
            Criteria,
            CriteriaType,
        )
        from euroncap_rating_2026.crash_protection.dummy import Dummy
        from euroncap_rating_2026.crash_protection.load_case import LoadCase
        from euroncap_rating_2026.crash_protection.seat import Seat

        body_region = BodyRegion(name="Head & Neck")
        body_region.set_criteria_list(
            [
                Criteria(
                    name="Ares",
                    hpl=None,
                    lpl=80.0,
                    value=0.0,
                    criteria_type=CriteriaType.NONE_SWITCH,
                ),
                Criteria(
                    name="HIC15",
                    hpl=500.0,
                    lpl=700.0,
                    value=0.0,
                    criteria_type=CriteriaType.CRITERIA,
                ),
                Criteria(
                    name="Ares-3ms",
                    hpl=72.0,
                    lpl=80.0,
                    value=0.0,
                    criteria_type=CriteriaType.CRITERIA,
                ),
            ]
        )
        body_region.set_criteria_value("HIC15", 650.0)
        body_region.set_criteria_value("Ares-3ms", 75.0)
        body_region.set_criteria_value("Ares", 70.0)  # below the gate

        loadcase = LoadCase(
            name="MPDB-50",
            seats=[
                Seat(
                    name="Driver",
                    dummy=Dummy(name="THOR-50", body_region_list=[body_region]),
                )
            ],
        )

        # Sheet as a previous scoring run (pre-gate-fix) left it: the gated
        # criteria still carry a colour and a Prediction.Check.
        df = pd.DataFrame(
            {
                "Loadcase": ["MPDB-50", None, None],
                "Seat position": ["Driver", None, None],
                "Dummy": ["THOR-50", None, None],
                "Body region": ["Head & Neck", None, None],
                "Criteria": ["Ares", "HIC15", "Ares-3ms"],
                "HPL": [None, 500.0, 72.0],
                "LPL": [80.0, 700.0, 80.0],
                "Value": [70.0, 650.0, 75.0],
                "Score": [None, None, None],
                "Colour": ["green", "brown", "orange"],
                "Capping?": ["", "", ""],
                "Prediction.Check": ["Correct", "Correct", "InTolerance"],
            }
        )

        updated = update_loadcase(df, loadcase)

        hic15 = updated[updated["Criteria"] == "HIC15"].iloc[0]
        ares3ms = updated[updated["Criteria"] == "Ares-3ms"].iloc[0]
        ares = updated[updated["Criteria"] == "Ares"].iloc[0]
        for row in (hic15, ares3ms):
            self.assertTrue(common.is_empty_cell(row["Colour"]), row["Colour"])
            self.assertTrue(
                common.is_empty_cell(row["Prediction.Check"]),
                row["Prediction.Check"],
            )
        self.assertEqual(ares["Colour"], "green")
        self.assertEqual(ares["Score"], 100.0)


if __name__ == "__main__":
    unittest.main()
