# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import List, Optional
import logging
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Measurement(BaseModel):
    name: str
    value: Optional[float]

    @property
    def modifier(self) -> Optional[float]:
        return self.evaluate_measurement()

    @property
    def color(self) -> Optional[str]:
        if self.modifier is None:
            return None
        return "green" if self.modifier == 0.0 else "red"

    def evaluate_measurement(self) -> Optional[float]:
        if self.value is None:
            return None
        if self.name == "DAMAGE":
            return 0.0 if self.value < 0.42 else (20.0 if self.value < 0.47 else 40.0)
        elif self.name == "Shoulder belt load":
            return 0.0 if self.value < 6.0 else 40.0
        elif self.name == "Pedal blocking":
            return 0.0 if self.value < 50.0 else 20.0
        elif self.name == "Pedal displacement - rearward":
            return (
                0.0 if self.value < 100.0 else (50.0 if self.value < 200.0 else 100.0)
            )
        elif self.name == "Pedal displacement - vertical":
            return 0.0 if self.value < 72.0 else 20.0
        elif self.name == "Excursion":
            return (
                0.0 if self.value < 450.0 else (50.0 if self.value < 550.0 else 100.0)
            )
        elif self.name == "Shoulder load":
            return 0.0 if self.value < 3.0 else 100.0
        elif self.name == "Viscous Criterion":
            return 0.0 if self.value < 1.0 else 100.0
        elif self.name == "Effective height modifier":
            return 100.0 if self.value < 790.0 else 0.0
        elif self.name == "Fpubic symphysis":
            return 0.0 if self.value < 2.8 else 25.0
        elif self.name == "Fy lumbar":
            return 0.0 if self.value < 2.0 else 25.0
        elif self.name == "Fz lumbar":
            return 0.0 if self.value < 3.5 else 25.0
        elif self.name == "Mx lumbar":
            return 0.0 if self.value < 120.0 else 25.0
        else:
            raise ValueError(f"Unknown measurement: {self.name}")
