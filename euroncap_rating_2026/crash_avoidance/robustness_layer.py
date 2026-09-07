# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from dataclasses import dataclass
from enum import Enum
import random


def _get_canonical_test_name(test_name: str) -> str:
    from euroncap_rating_2026.crash_avoidance import data_model

    return data_model.SUBTEST_TO_TEST_DICT.get(test_name, test_name)


class AssessmentCriteria(str, Enum):
    SAME_OR_BETTER_THAN_PREDICTED = "Same or better than predicted"
    NOT_RED = "Not red"


class RobustnessLayer(str, Enum):
    DRIVER_INPUT_PRE_CRASH = "Driver input pre-crash"
    SPEED = "Speed"
    ACCELERATION = "Acceleration"
    INITIAL_POSITION_OFFSET = "Initial position offset"
    TRAJECTORY_HEADING = "Trajectory/Heading"
    TYPE = "Type"
    APPEARANCE = "Appearance"
    ADVERSE_WEATHER_CONDITIONS = "Adverse weather conditions"
    ILLUMINATION_NIGHT = "Illumination (Night)"
    ILLUMINATION_GLARE = "Illumination (Sun glare)"
    ILLUMINATION_HLAMP_GLARE = "Illumination (Headlamp glare)"
    INFRASTRUCTURE_CLUTTER = "Infrastructure / Clutter"
    OBSCURATION_OBSTRUCTION = "Obscuration / Obstruction"
    IMPACT_LOCATION = "Impact location"


@dataclass
class RobustnessLayerScore:
    tested_score: float
    constant_score: float
    total_score: float
    n_applicable: float = 0.0
    claimed_layers: set = None

    def __init__(
        self, tested_score, constant_score, n_applicable=0.0, claimed_layers=None
    ):
        self.tested_score = tested_score
        self.constant_score = constant_score
        self.total_score = tested_score + constant_score
        self.n_applicable = n_applicable
        self.claimed_layers = claimed_layers if claimed_layers is not None else set()


ROBUSTNESS_LAYER_TO_VERIFICATION_CONDITION = {
    RobustnessLayer.DRIVER_INPUT_PRE_CRASH: {
        category: (
            "Steer and throttle condition. See protocol",
            AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
        )
        for category in [
            "CCRs",
            "CCRm",
            "CCRb",
            "CCFhos",
            "CCFhol",
            "CCFtap",
            "CCCscp",
            "CMCscp",
            "CMRs",
            "CMRb",
            "CMFtap",
            "CBNA",
            "CBFA",
            "CBNAO",
            "CBTA",
            "CPTA",
            "CPLA",
            "CPNA",
            "CPFA",
            "CPNCO",
        ]
    },
    RobustnessLayer.SPEED: {
        "CCFhos": [
            ("+5 km/h", AssessmentCriteria.NOT_RED),
            ("-5 km/h", AssessmentCriteria.NOT_RED),
        ],
        "CCFhol": [
            ("+5 km/h", AssessmentCriteria.NOT_RED),
            ("-5 km/h", AssessmentCriteria.NOT_RED),
        ],
        "CCFtap": [
            ("+5 km/h", AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED),
            ("-5 km/h", AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED),
        ],
        "CMFtap": [
            ("+5 km/h", AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED),
            ("-5 km/h", AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED),
        ],
        "CCCscp": [
            ("+5 km/h", AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED),
            ("-5 km/h", AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED),
        ],
        "CMCscp": [
            ("+5 km/h", AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED),
            ("-5 km/h", AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED),
        ],
        "CPTA": ("+ 3 km/h", AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED),
        "CBTA": ("+ 5 km/h", AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED),
        "CPNA": [
            ("+3 km/h", AssessmentCriteria.NOT_RED),
            ("-2 km/h", AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED),
        ],
        "CPFA": [
            ("+3 km/h", AssessmentCriteria.NOT_RED),
            ("-2 km/h", AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED),
        ],
        "CPNCO": [
            ("+3 km/h", AssessmentCriteria.NOT_RED),
            ("-2 km/h", AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED),
        ],
        "CBNA": [
            ("+5 km/h", AssessmentCriteria.NOT_RED),
            ("-3 km/h", AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED),
        ],
        "CBFA": [
            ("+5 km/h", AssessmentCriteria.NOT_RED),
            ("-3 km/h", AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED),
        ],
        "CBNAO": [
            ("+5 km/h", AssessmentCriteria.NOT_RED),
            ("-3 km/h", AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED),
        ],
    },
    RobustnessLayer.ACCELERATION: {
        "CCRb": [
            ("-2 m/s²", AssessmentCriteria.NOT_RED),
            ("+2 m/s²", AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED),
        ],
        "CMRb": [
            ("-2 m/s²", AssessmentCriteria.NOT_RED),
            ("+2 m/s²", AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED),
        ],
    },
    RobustnessLayer.INITIAL_POSITION_OFFSET: {
        "CPNA": ("-25% m of distance to impact point", AssessmentCriteria.NOT_RED),
        "CPFA": ("-25% m of distance to impact point", AssessmentCriteria.NOT_RED),
        "CBNA": ("-25% m of distance to impact point", AssessmentCriteria.NOT_RED),
        "CBFA": ("-25% m of distance to impact point", AssessmentCriteria.NOT_RED),
        "CPNCO": ("-25% m of distance to impact point", AssessmentCriteria.NOT_RED),
        "CBNAO": ("-25% m of distance to impact point", AssessmentCriteria.NOT_RED),
        "CPTA": (
            "-25% m of distance to impact point",
            AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
        ),
        "CCFtap": [
            ("+0.5m Path offset", AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED),
            ("-0.5m Path offset", AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED),
        ],
        "CMFtap": [
            ("+0.5m Path offset", AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED),
            ("-0.5m Path offset", AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED),
        ],
        "CBTA": [
            ("+0.5m Path offset", AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED),
            ("-0.5m Path offset", AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED),
        ],
        "CCRb": [
            ("+0.5s Time headway", AssessmentCriteria.NOT_RED),
            ("-0.5s Time headway", AssessmentCriteria.NOT_RED),
        ],
        "CMRb": [
            ("+0.5s Time headway", AssessmentCriteria.NOT_RED),
            ("-0.5s Time headway", AssessmentCriteria.NOT_RED),
        ],
        "CC ELK On": [
            ("0.25m away from VUT", AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED),
            ("0.5m closer to VUT", AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED),
        ],
        "CC ELK Ov": [
            ("0.25m away from VUT", AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED),
            ("0.5m closer to VUT", AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED),
        ],
        "CM ELK On": (
            "safety_margin_violation",
            AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
        ),
        "CM ELK Ov": (
            "safety_margin_violation",
            AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
        ),
    },
    RobustnessLayer.TRAJECTORY_HEADING: {
        "CCRs": [
            (
                "+20° (rotation around the impact point)",
                AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
            ),
            (
                "-20° (rotation around the impact point)",
                AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
            ),
        ],
        "CMRs": [
            (
                "+20° (rotation around the impact point)",
                AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
            ),
            (
                "-20° (rotation around the impact point)",
                AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
            ),
        ],
        "CPNA": [
            (
                "+20° (rotation around the impact point)",
                AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
            ),
            (
                "-20° (rotation around the impact point)",
                AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
            ),
        ],
        "CPFA": [
            (
                "+20° (rotation around the impact point)",
                AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
            ),
            (
                "-20° (rotation around the impact point)",
                AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
            ),
        ],
        "CBNA": [
            (
                "+20° (rotation around the impact point)",
                AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
            ),
            (
                "-20° (rotation around the impact point)",
                AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
            ),
        ],
        "CBFA": [
            (
                "+20° (rotation around the impact point)",
                AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
            ),
            (
                "-20° (rotation around the impact point)",
                AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
            ),
        ],
        "CPNCO": [
            (
                "+20° (rotation around the impact point)",
                AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
            ),
            (
                "-20° (rotation around the impact point)",
                AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
            ),
        ],
        "CBNAO": [
            (
                "+20° (rotation around the impact point)",
                AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
            ),
            (
                "-20° (rotation around the impact point)",
                AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
            ),
        ],
    },
    RobustnessLayer.ILLUMINATION_NIGHT: {
        "CCRs": ("Performance in darkness (1 lux)", AssessmentCriteria.NOT_RED),
        "CCRm": ("Performance in darkness (1 lux)", AssessmentCriteria.NOT_RED),
        "CCRb": ("Performance in darkness (1 lux)", AssessmentCriteria.NOT_RED),
        "CCFhos": ("Performance in darkness (1 lux)", AssessmentCriteria.NOT_RED),
        "CCFhol": ("Performance in darkness (1 lux)", AssessmentCriteria.NOT_RED),
        "CCFtap": ("Performance in darkness (1 lux)", AssessmentCriteria.NOT_RED),
        "CCCscp": ("Performance in darkness (1 lux)", AssessmentCriteria.NOT_RED),
        "CMCscp": ("Performance in darkness (1 lux)", AssessmentCriteria.NOT_RED),
        "CMRs": ("Performance in darkness (1 lux)", AssessmentCriteria.NOT_RED),
        "CMRb": ("Performance in darkness (1 lux)", AssessmentCriteria.NOT_RED),
        "CMFtap": ("Performance in darkness (1 lux)", AssessmentCriteria.NOT_RED),
        "CBNA": ("Performance in darkness (1 lux)", AssessmentCriteria.NOT_RED),
        "CBFA": ("Performance in darkness (1 lux)", AssessmentCriteria.NOT_RED),
        "CBNAO": ("Performance in darkness (1 lux)", AssessmentCriteria.NOT_RED),
        "CBTA": ("Performance in darkness (1 lux)", AssessmentCriteria.NOT_RED),
        "CPTA": ("Performance in darkness (1 lux)", AssessmentCriteria.NOT_RED),
        "ELK RE": (
            "Performance in darkness  (1 lux) ",
            AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
        ),
    },
    RobustnessLayer.ILLUMINATION_GLARE: {
        "CPLA": (
            "Headlight of stationary vehicle on adjacent lane",
            AssessmentCriteria.NOT_RED,
        ),
        "CPNA": (
            "Headlight of stationary vehicle on adjacent lane",
            AssessmentCriteria.NOT_RED,
        ),
        "CPFA": (
            "Headlight of stationary vehicle on adjacent lane",
            AssessmentCriteria.NOT_RED,
        ),
        "CPNCO": (
            "Headlight of stationary vehicle on adjacent lane",
            AssessmentCriteria.NOT_RED,
        ),
    },
    RobustnessLayer.INFRASTRUCTURE_CLUTTER: {
        "CCRs": (
            "Vehicle aside of main target",
            AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
        ),
        "CMRs": (
            "Vehicle aside of main target OR GVT in front of main target",
            AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
        ),
        "CCRm": (
            "Vehicle aside of main target (moving)",
            AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
        ),
        "CCRb": (
            "Vehicle aside of main target (moving)",
            AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
        ),
        "CMRb": (
            "Vehicle aside of main target (moving)",
            AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
        ),
        "CPTA": (
            "Typical crossing scenery e.g., traffic sign, refuge, trash bin",
            AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
        ),
        "CBTA": (
            "Typical crossing scenery e.g., traffic sign, refuge, trash bin",
            AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
        ),
        "CCCscp": (
            "Typical crossing scenery e.g., traffic sign, stationary pedestrians on sidewalk, stationary (secondary) GVT on crossing1 road",
            AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
        ),
        "CMCscp": (
            "Typical crossing scenery e.g., traffic sign, stationary pedestrians on sidewalk, stationary (secondary) GVT on crossing1 road",
            AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
        ),
        "CPNA": (
            "Randomly selected case from Technical Bulletin CA 002",
            AssessmentCriteria.NOT_RED,
        ),
        "CPFA": (
            "Randomly selected case from Technical Bulletin CA 002",
            AssessmentCriteria.NOT_RED,
        ),
        "CBNA": (
            "Randomly selected case from Technical Bulletin CA 002",
            AssessmentCriteria.NOT_RED,
        ),
        "CBFA": (
            "Randomly selected case from Technical Bulletin CA 002",
            AssessmentCriteria.NOT_RED,
        ),
    },
    RobustnessLayer.OBSCURATION_OBSTRUCTION: {
        "CPNCO": (
            "Randomly selected CPNCO case from Technical Bulletin CA 002",
            AssessmentCriteria.NOT_RED,
        ),
        "CBNAO": (
            "Randomly selected CBNAO case from Technical Bulletin CA 002",
            AssessmentCriteria.NOT_RED,
        ),
    },
    RobustnessLayer.IMPACT_LOCATION: {
        "CC ELK On": [
            (
                "+10% Range change",
                AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
            ),
            (
                "-10% Range change",
                AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
            ),
        ],
        "CC ELK Ov": (
            "+50% Range change",
            AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
        ),
        "CM ELK On": [
            (
                "+10% Range change",
                AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
            ),
            (
                "-10% Range change",
                AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
            ),
        ],
        "CM ELK Ov": (
            "+50% Range change",
            AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
        ),
    },
    RobustnessLayer.APPEARANCE: {
        "ELK RE": (
            "Lane marking type, colour, width",
            AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED,
        ),
    },
}


def get_robustness_layer_verification_condition(layer: RobustnessLayer, test_name: str):
    conditions = ROBUSTNESS_LAYER_TO_VERIFICATION_CONDITION.get(layer)
    if conditions is None:
        return None, None
    canonical_test_name = _get_canonical_test_name(test_name)
    value = conditions.get(canonical_test_name, None)
    if value is None:
        return None, None
    if isinstance(value, list):
        value = random.choice(value)
    # value is a tuple (string, AssessmentCriteria)
    return value


def get_assessment_criteria(
    robustness_layer: RobustnessLayer, test_name: str, verification_condition: str
):
    """
    Returns the AssessmentCriteria if an exact match is found
    for the given robustness_layer, test_name, and verification_condition string.
    Returns None if not found.
    """
    conditions = ROBUSTNESS_LAYER_TO_VERIFICATION_CONDITION.get(robustness_layer)
    if not conditions:
        return None
    canonical_test_name = _get_canonical_test_name(test_name)
    value = conditions.get(canonical_test_name)
    if value is None:
        return None
    # value can be a tuple or a list of tuples
    if isinstance(value, tuple):
        if value[0] == verification_condition:
            return value[1]
    elif isinstance(value, list):
        for item in value:
            if item[0] == verification_condition:
                return item[1]
    return None
