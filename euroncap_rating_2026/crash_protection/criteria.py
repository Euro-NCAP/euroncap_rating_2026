# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import Optional
from decimal import Decimal, ROUND_HALF_UP
from euroncap_rating_2026.numeric import round_half_up
import pandas as pd
import numpy as np
import logging
from enum import Enum

logger = logging.getLogger(__name__)

_TWO_DP = Decimal("0.01")


def _dec(value):
    """Exact Decimal for a protocol constant (HPL/LPL/threshold, 2-dp).

    repr() of the float gives back the constant's decimal string (e.g.
    "0.4"), so threshold arithmetic is exact decimal arithmetic: a
    boundary like 0.475 is really 0.475, never 0.4749999999999999, and
    half-up rounding at midpoints is unambiguous.
    """
    return Decimal(repr(float(value)))


def _quantize2(d):
    """Round a Decimal to 2 dp half-up and return it as float."""
    return float(d.quantize(_TWO_DP, rounding=ROUND_HALF_UP))


# Blue-variant prediction types: type indicators, not colour predictions
_BLUE_VARIANT_PREDICTIONS = frozenset({"t", "st", "sa"})

# Mapping of colors to their corresponding score values
color_to_value = {
    "green": 100.0,
    "yellow": 80.0,
    "orange": 40.0,
    "brown": 20.0,
    "red": 0.0,
}


class CriteriaType(Enum):
    UNKNOWN = 0
    CRITERIA = 1
    ZERO_SWITCH = 2
    NONE_SWITCH = 3
    DISABLED = 4
    INVERTED_CRITERIA = 5
    COLOR_CRITERIA = 6
    CAPPING_CRITERIA = 7
    VRU_CRITERIA = 8


class Criteria(BaseModel):
    """
    Represents a criteria used in vehicle crash tests.

    Attributes:
        name (str): The name of the criteria.
        hpl (Optional[float]): The high-performance limit.
        lpl (float): The low-performance limit.
        capping (bool): Indicates if capping is applied to the criteria.
        capping_value (Optional[float]): The capping value for the criteria.
        value (float): The value of the criteria.
        score (float): The score of the criteria.
        color (Optional[str]): The color representing the criteria's performance.
        prediction (Optional[str]): The predicted color for the criteria.
        prediction_result (Optional[str]): The result of the prediction.
        depends_on (str): The criteria that this criteria depends on.
    """

    name: str
    hpl: Optional[float] = None
    lpl: Optional[float] = None
    capping: bool = False
    capping_value: Optional[float] = None
    value: float = 0.0
    score: float = 0.0
    color: Optional[str] = Field(None, pattern="^(green|yellow|orange|brown|red)$")
    prediction: Optional[str] = Field(
        None,
        pattern="^(green|yellow|orange|brown|red|blue|green-40|green-30|green-20|t|st|sa)$",
    )
    prediction_result: Optional[str] = Field(
        None, pattern="^(Correct|Incorrect|In Tolerance)$"
    )
    depends_on: str = None
    criteria_type: CriteriaType = CriteriaType.UNKNOWN
    test_point: Optional[str] = None

    @field_validator("value", mode="before")
    @classmethod
    def ensure_two_decimal_places(cls, v):
        """
        Ensures the value has two decimal places.

        Args:
            v (float): The value.

        Returns:
            float: The rounded value.
        """
        if isinstance(v, (int, float, np.int64)):
            return round_half_up(float(v), 2)  # Ensure 2 decimal places
        raise ValueError("Value must be a float or int")

    @field_validator("capping_value", mode="before")
    @classmethod
    def ensure_two_decimal_places_cap(cls, v):
        """
        Ensures the capping value has two decimal places.

        Args:
            v (float): The capping value.

        Returns:
            float: The rounded capping value.
        """
        if v is None:
            return v  # Allow None values
        if isinstance(v, (int, float, np.int64)):
            return round_half_up(float(v), 2)  # Ensure 2 decimal places
        raise ValueError("Value must be a float or int")

    @property
    def interval_step(self):
        """
        Calculates the interval step for the criteria.

        Returns:
            float: The interval step.
        """
        return _quantize2((_dec(self.lpl) - _dec(self.hpl)) / 3)

    @property
    def green_yellow_threshold(self):
        """
        Calculates the green-yellow threshold for the criteria.

        Returns:
            float: The green-yellow threshold.
        """
        if self.criteria_type == CriteriaType.NONE_SWITCH:
            return None
        else:
            return _quantize2(_dec(self.hpl))

    @property
    def yellow_orange_threshold(self):
        """
        Calculates the yellow-orange threshold for the criteria.

        Returns:
            float: The yellow-orange threshold.
        """
        if self.criteria_type == CriteriaType.NONE_SWITCH:
            return None
        else:
            return _quantize2(_dec(self.hpl) + (_dec(self.lpl) - _dec(self.hpl)) / 3)

    @property
    def orange_brown_threshold(self):
        """
        Calculates the orange-brown threshold for the criteria.

        Returns:
            float: The orange-brown threshold.
        """
        if self.criteria_type == CriteriaType.NONE_SWITCH:
            return None
        else:
            return _quantize2(
                _dec(self.hpl) + 2 * (_dec(self.lpl) - _dec(self.hpl)) / 3
            )

    @property
    def brown_red_threshold(self):
        """
        Calculates the brown-red threshold for the criteria.

        Returns:
            float: The brown-red threshold.
        """
        if self.criteria_type == CriteriaType.NONE_SWITCH:
            return None
        else:
            return _quantize2(_dec(self.lpl))

    def set_prediction(self, prediction_color):
        """
        Sets the prediction color for the criteria.

        Args:
            prediction_color (str): The prediction color.
        """
        self.prediction = prediction_color

    def get_value(self):
        """
        Returns the value for the criteria.

        Returns:
            float: The value of the criteria.
        """
        return self.value

    def set_value(self, value):
        """
        Sets the value for the criteria and calculates the color and score.

        Args:
            value (float): The value to set.
        """
        self.value = round_half_up(value, 2)
        if self.capping_value:
            if self.value >= self.capping_value:
                self.capping = True
            else:
                self.capping = False
        self.calculate_color_score()

    def get_score(self):
        """
        Returns the score for the criteria.

        Returns:
            float: The score of the criteria.
        """
        return self.score

    def set_capping_value(self, value):
        """
        Sets the capping value for the criteria.

        Args:
            value (float): The capping value to set.
        """
        self.capping_value = round_half_up(value, 2)
        if self.capping_value:
            if self.value >= self.capping_value:
                self.capping = True
            else:
                self.capping = False

    def _get_tolerance_bounds(self, threshold_value, is_vru=False):
        """
        Calculates lower and upper tolerance bounds for a threshold.

        For normal criteria: uses interval_step / 4 as fixed tolerance
        For VRU criteria: uses 10% of threshold value as relative tolerance

        Args:
            threshold_value (float): The threshold value (green-yellow, yellow-orange, etc.)
            is_vru (bool): Whether to use VRU relative tolerance (True) or standard absolute tolerance (False)

        Returns:
            tuple: (lower_bound, upper_bound) as rounded values
        """
        threshold = _dec(threshold_value)
        if is_vru:
            # VRU uses 10% relative tolerance
            lower_bound = _quantize2(threshold / Decimal("1.1"))
            upper_bound = _quantize2(threshold / Decimal("0.9"))
        else:
            # Standard criteria use interval_step / 4 (25% of the colour
            # band width, Frontal Impact protocol v1.2 section 3.4.1)
            increase_factor = (_dec(self.interval_step) / 4).quantize(
                _TWO_DP, rounding=ROUND_HALF_UP
            )
            lower_bound = float(threshold - increase_factor)
            upper_bound = float(threshold + increase_factor)
        return lower_bound, upper_bound

    def calculate_color_score(self):
        """
        Calculates the color and score for the criteria based on its value and thresholds.
        """
        logger.debug(f"Calculating color and score for value: {self.value}")
        if pd.isna(self.value):
            logger.debug("Value is None, setting color to None and score to 0.0.")
            self.color = None
            self.score = 0.0

            return

        if self.criteria_type == CriteriaType.CAPPING_CRITERIA:
            if self.value is not None and self.capping_value is not None:
                if self.value >= self.capping_value:
                    self.capping = True
                    self.color = "red"
                else:
                    self.capping = False
                    self.color = "green"
            logger.debug(
                f"[RESULT] Criteria type: {self.criteria_type}, Capping: {self.capping}"
            )
            self.score = None
            return
        if self.criteria_type == CriteriaType.UNKNOWN:
            if self.value is not None and self.capping_value is not None:
                if self.value >= self.capping_value:
                    self.capping = True
                else:
                    self.capping = False
            logger.debug(
                f"[RESULT] Criteria type: {self.criteria_type}, Capping: {self.capping}"
            )
            return

        if self.criteria_type == CriteriaType.NONE_SWITCH:
            if self.value < self.lpl:
                self.score = 100.0
                self.color = "green"
            else:
                self.score = None
                self.color = "red"
            logger.debug(
                f"[RESULT] Criteria type: {self.criteria_type}, Color: {self.color}, Score: {self.score}"
            )
            return

        if (
            self.criteria_type == CriteriaType.CRITERIA
            or self.criteria_type == CriteriaType.ZERO_SWITCH
            or self.criteria_type == CriteriaType.VRU_CRITERIA
        ):
            logger.debug(
                f"Thresholds for {self.name}: "
                f"green_yellow={self.green_yellow_threshold}, "
                f"yellow_orange={self.yellow_orange_threshold}, "
                f"orange_brown={self.orange_brown_threshold}, "
                f"brown_red={self.brown_red_threshold}"
            )
            # Current assumption is that interval are ALWAYS
            # Closed on the left (<=) - Open on the right (<)
            # v1 <= x < v2
            # ZERO_SWITCH criteria (HPL == LPL, a single pass/fail limit, e.g.
            # rear seat Backset) instead document the pass condition as
            # value <= limit (inclusive), per the Euro NCAP protocol wording.
            if self.criteria_type == CriteriaType.ZERO_SWITCH:
                if self.value <= self.green_yellow_threshold:
                    test_output_color = "green"
                else:
                    test_output_color = "red"
            elif self.value < self.green_yellow_threshold:
                test_output_color = "green"
            elif self.value < self.yellow_orange_threshold:
                test_output_color = "yellow"
            elif self.value < self.orange_brown_threshold:
                test_output_color = "orange"
            elif self.value < self.brown_red_threshold:
                test_output_color = "brown"
            else:
                test_output_color = "red"

            if (
                self.prediction is not None
                and self.prediction.lower() not in _BLUE_VARIANT_PREDICTIONS
            ):
                is_vru = self.criteria_type == CriteriaType.VRU_CRITERIA

                if test_output_color == self.prediction:
                    self.prediction_result = "Correct"
                    self.color = self.prediction
                else:
                    # Calculate tolerance bounds based on criteria type
                    gy_lower, gy_upper = self._get_tolerance_bounds(
                        self.green_yellow_threshold, is_vru=is_vru
                    )
                    yo_lower, yo_upper = self._get_tolerance_bounds(
                        self.yellow_orange_threshold, is_vru=is_vru
                    )
                    ob_lower, ob_upper = self._get_tolerance_bounds(
                        self.orange_brown_threshold, is_vru=is_vru
                    )
                    br_lower, br_upper = self._get_tolerance_bounds(
                        self.brown_red_threshold, is_vru=is_vru
                    )

                    if (
                        (self.prediction == "green" and self.value < gy_upper)
                        or (
                            self.prediction == "yellow"
                            and gy_lower <= self.value < yo_upper
                        )
                        or (
                            self.prediction == "orange"
                            and yo_lower <= self.value < ob_upper
                        )
                        or (
                            self.prediction == "brown"
                            and ob_lower <= self.value < br_upper
                        )
                        or (self.prediction == "red" and self.value >= br_lower)
                    ):
                        self.prediction_result = "In Tolerance"
                        self.color = self.prediction
                    else:
                        self.prediction_result = "Incorrect"
                        self.color = test_output_color
                logger.debug(
                    f"Prediction: {self.prediction}, "
                    f"Prediction Result: {self.prediction_result}, "
                    f"Test Output Color: {test_output_color}, "
                    f"Final Color: {self.color}"
                )
            else:
                self.color = test_output_color
        elif self.criteria_type == CriteriaType.INVERTED_CRITERIA:
            # For inverted criteria intervals are
            # Open on the left (<) - Closed on the right (<=)
            # v1 < x <= v2
            if self.value > self.green_yellow_threshold:
                test_output_color = "green"
            elif self.value > self.yellow_orange_threshold:
                test_output_color = "yellow"
            elif self.value > self.orange_brown_threshold:
                test_output_color = "orange"
            elif self.value > self.brown_red_threshold:
                test_output_color = "brown"
            else:
                test_output_color = "red"

            if self.prediction is not None:
                if test_output_color == self.prediction:
                    self.prediction_result = "Correct"
                    self.color = self.prediction
                else:
                    gy_lower, gy_upper = self._get_tolerance_bounds(
                        self.green_yellow_threshold
                    )
                    yo_lower, yo_upper = self._get_tolerance_bounds(
                        self.yellow_orange_threshold
                    )
                    ob_lower, ob_upper = self._get_tolerance_bounds(
                        self.orange_brown_threshold
                    )
                    br_lower, br_upper = self._get_tolerance_bounds(
                        self.brown_red_threshold
                    )

                    if (
                        (self.prediction == "green" and gy_lower <= self.value)
                        or (
                            self.prediction == "yellow"
                            and gy_upper > self.value >= yo_lower
                        )
                        or (
                            self.prediction == "orange"
                            and yo_upper > self.value >= ob_lower
                        )
                        or (
                            self.prediction == "brown"
                            and ob_upper > self.value >= br_lower
                        )
                        or (self.prediction == "red" and self.value < br_upper)
                    ):
                        self.prediction_result = "In Tolerance"
                        self.color = self.prediction
                    else:
                        self.prediction_result = "Incorrect"
                        self.color = test_output_color
                logger.debug(
                    f"Prediction: {self.prediction}, "
                    f"Prediction Result: {self.prediction_result}, "
                    f"Test Output Color: {test_output_color}, "
                    f"Final Color: {self.color}"
                )
            else:
                self.color = test_output_color

        if self.color is not None:
            self.score = color_to_value[self.color]

        if self.capping_value:
            if self.value >= self.capping_value:
                self.capping = True
            else:
                self.capping = False
        logger.debug(
            f"[RESULT] Criteria type: {self.criteria_type}, Color: {self.color}, Score: {self.score}"
        )

    @staticmethod
    def get_criteria_from_row(row, dummy_name=None, criteria_type=CriteriaType.UNKNOWN):
        """
        Creates a Criteria object from a row of data.

        Args:
            row (pd.Series): The row of data.
            dummy_name (str, optional): The name of the dummy this criteria belongs to.

        Returns:
            Criteria: The created Criteria object.
        """
        name = row["Criteria"]
        hpl = row["HPL"]
        lpl = row["LPL"]
        value = row["Value"]
        logger.debug(f"Name: {name}, HPL: {hpl}, LPL: {lpl}, Value: {value}")
        if "Test point" in row:
            test_point = row["Test point"]
            logger.debug(f"Test point {test_point}")
        prediction = None

        if dummy_name in ("Adult Headform", "Child Headform"):
            criteria_type = CriteriaType.VRU_CRITERIA
        elif not pd.isna(row["LPL"]) and not pd.isna(row["HPL"]):
            if row["HPL"] < row["LPL"]:
                criteria_type = CriteriaType.CRITERIA
            elif row["HPL"] == row["LPL"]:
                criteria_type = CriteriaType.ZERO_SWITCH
            else:
                criteria_type = CriteriaType.INVERTED_CRITERIA
        elif not pd.isna(row["LPL"]) and pd.isna(row["HPL"]):
            criteria_type = CriteriaType.NONE_SWITCH
        elif pd.isna(row["LPL"]) and pd.isna(row["HPL"]):
            criteria_type = CriteriaType.CAPPING_CRITERIA
        else:
            criteria_type = CriteriaType.UNKNOWN

        logger.debug(f"Criteria Type: {criteria_type}")
        if isinstance(value, (int, float, np.int64)):
            value = round_half_up(value, 2)
        elif isinstance(value, str):
            try:
                value = round_half_up(float(value), 2)
            except ValueError:
                logger.warning(
                    f"Value '{value}' for criteria '{name}' is not a valid number."
                )
                value = 0.0

        if "OEM Prediction" in row and not pd.isna(row["OEM Prediction"]):
            prediction = row["OEM Prediction"].lower()
        capping_value = None
        if "Capping" in row and not pd.isna(row["Capping"]):
            capping_value = round_half_up(row["Capping"], 2)
        if pd.isna(hpl):
            hpl = None
        criteria = Criteria(
            name=name,
            hpl=hpl,
            lpl=lpl,
            capping_value=capping_value,
            value=value,
            prediction=prediction,
            criteria_type=criteria_type,
        )
        if "Test point" in row and not pd.isna(row["Test point"]):
            criteria.test_point = row["Test point"]

        criteria.calculate_color_score()
        return criteria


class ColorCriteria(BaseModel):
    name: str
    criteria_type: CriteriaType = CriteriaType.COLOR_CRITERIA
    color: Optional[str] = Field(None, pattern="^(green|yellow|orange|capping|red)$")
    countermeasure: bool
    redline_above_125mm: bool

    @property
    def score(self) -> float:
        if self.countermeasure:
            if self.color in ["green", "yellow"]:
                return 100.0
            elif self.color == "orange":
                return 75.0
            elif self.color == "red":
                return 50.0 if self.redline_above_125mm else 25.0
        else:
            if self.color == "green":
                return 100.0
            elif self.color == "yellow":
                return 50.0
            elif self.color == "orange":
                return 25.0
            else:
                return 0.0
