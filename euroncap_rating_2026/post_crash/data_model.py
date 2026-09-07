# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from enum import Enum


STAGE_SUBELEMENTS = [
    {
        "Stage": "Post-Crash",
        "Stage element": "Rescue Information",
        "Stage subelement": "Rescue sheets",
    },
    {
        "Stage": "Post-Crash",
        "Stage element": "Rescue Information",
        "Stage subelement": "Emergency response guide",
    },
    {
        "Stage": "Post-Crash",
        "Stage element": "Post-Crash Intervention",
        "Stage subelement": "Advanced eCall",
    },
    {
        "Stage": "Post-Crash",
        "Stage element": "Post-Crash Intervention",
        "Stage subelement": "Multi-collision brake & hazard lights",
    },
    {
        "Stage": "Post-Crash",
        "Stage element": "Vehicle Extrication",
        "Stage subelement": "Energy management",
    },
    {
        "Stage": "Post-Crash",
        "Stage element": "Vehicle Extrication",
        "Stage subelement": "Occupant extrication",
    },
]


class StageSubelementKey(str, Enum):
    RI = "RI"
    PCI = "PCI"
    VE = "VE"


class PowertrainType(str, Enum):
    HYBRID = "Hybrid"
    ELECTRIC = "Electric"
    ICE = "ICE"


class SlideDoorType(str, Enum):
    SLIDING = "Sliding"
    CONVENTIONAL = "Conventional"


RESCUE_SHEET_CSV_STRING = """
Stage subelement;Value
Rescue sheet;
"""

EMERGENCY_RESPONSE_GUIDE_CSV_STRING = """
Stage subelement;Value
Emergency response guide;
"""

ADVANCED_ECALL_CSV_STRING = """
Category;Scenario;Value
Advanced eCall - 112;General requirements;
;Potential number of occupants;
;Direction of impacts - Front impact;
;Direction of impacts - Side impact;
;Direction of impacts - Rear impact;
;Direction of impacts - Rollovers as 1st impact;
;Delta V - Front impact;
;Delta V - Side impact;
;Delta V - Rear impact;
Advanced eCall - TPS;General requirements;
;Country coverage;
;Multiple languages - EN, DE, FR, ES;
;Multiple languages - 4 additional languages;
;Hazard detection after crash;
;Telephone pairing;
;Vehicle information;
;Vehicle attitude;
;Any additional information;
;AACN and OEM severity index;
"""


MCB_HAZARD_LIGHTS_CSV_STRING = """
Category;Value
Advanced multi-collision brake;
Automatic activation of hazard warning lights;
"""
