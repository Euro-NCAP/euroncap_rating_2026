# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from pydantic import BaseModel, field_validator
from typing import List, Optional
from euroncap_rating_2026.crash_protection.criteria import Criteria
from euroncap_rating_2026.crash_protection.measurement import Measurement
from euroncap_rating_2026.numeric import round_half_up
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BodyRegion(BaseModel):
    """
    Represents a body region in a dummy used in vehicle crash tests.

    Attributes:
        name (str): The name of the body region.
        _criteria (List[Criteria]): A list of criteria for the body region.
        bodyregion_score (Optional[float]): The score for the body region.
        score (Optional[float]): The overall score for the body region.
        inspection (Optional[float]): A inspection value for the body region.
        max_score (Optional[float]): The maximum score for the body region.
    """

    name: str
    _criteria: List[Criteria] = []
    _measurement: List[Measurement] = []
    bodyregion_score: Optional[float] = None
    score: Optional[float] = None
    inspection: Optional[float] = None
    max_score: Optional[float] = None

    @field_validator("inspection", mode="before")
    @classmethod
    def ensure_two_decimal_places(cls, v):
        """
        Ensures the inspection value has two decimal places.

        Args:
            v (float): The inspection value.

        Returns:
            float: The rounded inspection value.
        """
        if v is None:
            return v  # Allow None values
        if isinstance(v, (int, float, np.int64)):
            return round_half_up(float(v), 2)  # Ensure 2 decimal places
        raise ValueError("Value must be a float or int")

    @field_validator("score", mode="before")
    @classmethod
    def ensure_five_decimal_places(cls, v):
        """
        Ensures the score value has three decimal places.

        Args:
            v (float): The score value.

        Returns:
            float: The rounded score value.
        """
        if v is None:
            return v  # Allow None values
        if isinstance(v, (int, float, np.int64)):
            return round_half_up(float(v), 5)  # Ensure 5 decimal places
        raise ValueError("Value must be a float or int")

    @field_validator("bodyregion_score", mode="before")
    @classmethod
    def ensure_five_decimal_places_bodyregion_score(cls, v):
        """
        Ensures the body region score value has five decimal places.

        Args:
            v (float): The body region score value.

        Returns:
            float: The rounded body region score value.
        """
        if v is None:
            return v
        if isinstance(v, (int, float, np.int64)):
            return round_half_up(float(v), 5)
        raise ValueError("Value must be a float or int")

    def get_inspection(self) -> Optional[float]:
        """
        Retrieves the inspection value.

        Returns:
            Optional[float]: The inspection value.
        """
        return self.inspection

    def set_inspection(self, value: Optional[float]):
        """
        Sets the inspection value.

        Args:
            value (Optional[float]): The inspection value.
        """
        if value is None:
            self.inspection = None
            return
        self.inspection = round_half_up(value, 2)

    def get_max_score(self) -> Optional[float]:
        """
        Retrieves the maximum score.

        Returns:
            Optional[float]: The maximum score.
        """
        return self.max_score

    def set_max_score(self, value: Optional[float]):
        """
        Sets the maximum score.

        Args:
            value (Optional[float]): The maximum score.
        """
        if value is None:
            self.max_score = None
            return
        self.max_score = value

    def compute_bodyregion_score(self) -> Optional[float]:
        """
        Computes the score for the body region based on the criteria scores.

        Returns:
            Optional[float]: The computed body region score.
        """
        scores = [
            criteria.score for criteria in self._criteria if criteria.score is not None
        ]
        logger.debug(f"Body region name: {self.name}")
        logger.debug(f"Scores array: {scores}")

        if scores:
            self.bodyregion_score = round_half_up(min(scores), 5)
        else:
            self.bodyregion_score = None

    def get_bodyregion_score(self) -> Optional[float]:
        """
        Retrieves the body region score.

        Returns:
            Optional[float]: The body region score.
        """
        return self.bodyregion_score

    def set_bodyregion_score(self, value) -> Optional[float]:
        """
        Sets the body region score.

        Args:
            value (float): The body region score.
        """
        self.bodyregion_score = (
            round_half_up(float(value), 5) if value is not None else None
        )

    def compute_score(self) -> Optional[float]:
        """
        Computes the overall score for the body region based on the body region score, inspection, and max score.

        Returns:
            Optional[float]: The computed overall score.
        """
        if (
            self.bodyregion_score is not None
            and self.inspection is not None
            and self.max_score is not None
        ):
            measurements = [
                measurement.modifier
                for measurement in self._measurement
                if measurement.modifier is not None
            ]
            total_measurement_modifier = None
            if measurements:
                total_measurement_modifier = sum(measurements)
                bodyregion_score_modified = max(
                    self.bodyregion_score - total_measurement_modifier, 0
                )
            else:
                bodyregion_score_modified = self.bodyregion_score
            self.score = round_half_up(
                max(bodyregion_score_modified - self.inspection, 0)
                * self.max_score
                / 100,
                5,
            )
        else:
            self.score = None

    def get_score(self) -> Optional[float]:
        """
        Retrieves the overall score for the body region.

        Returns:
            Optional[float]: The overall score.
        """
        return self.score

    def set_criteria_list(self, criteria_list: List[Criteria]):
        """
        Sets the list of criteria for the body region and resolves dependencies.

        Args:
            criteria_list (List[Criteria]): The list of criteria.
        """
        self._criteria = criteria_list
        self.resolve_dependancies()
        self.compute_bodyregion_score()

    def set_measurement_list(self, measurement_list: List[Measurement]):
        """
        Sets the list of measurements for the body region.

        Args:
            measurement_list (List[Measurement]): The list of measurements.
        """
        self._measurement = measurement_list

    def get_measurement_list(self) -> List[Measurement]:
        """
        Retrieves the list of measurements for the body region.

        Returns:
            List[Measurement]: The list of measurements.
        """
        return self._measurement

    def set_criteria_value(self, criteria_name: str, value: float):
        """
        Sets the value for a specific criteria in the body region.

        Args:
            criteria_name (str): The name of the criteria.
            value (float): The value to set.
        """
        found = False
        for criteria in self._criteria:
            if criteria.name == criteria_name:
                found = True
                criteria.set_value(value)

        if not found:
            logger.error(
                f"Error. Criteria '{criteria_name}' not found in body region '{self.name}'"
            )
        else:
            self.resolve_dependancies()

    def set_criteria_capping_value(self, criteria_name: str, value: float):
        """
        Sets the capping value for a specific criteria in the body region.

        Args:
            criteria_name (str): The name of the criteria.
            value (float): The capping value to set.
        """
        for criteria in self._criteria:
            if criteria.name == criteria_name:
                criteria.set_capping_value(value)

    def contains_criteria(self, criteria_name):
        """
        Checks if the body region contains a specific criteria.

        Args:
            criteria_name (str): The name of the criteria.

        Returns:
            bool: True if the criteria is found, False otherwise.
        """
        found = False
        for criteria in self._criteria:
            if criteria.name == criteria_name:
                found = True
                break
        return found

    def get_criteria_lpl(self, criteria_name):
        """
        Retrieves the low-performance limit (LPL) for a specific criteria.

        Args:
            criteria_name (str): The name of the criteria.

        Returns:
            float: The LPL value.
        """
        for criteria in self._criteria:
            if criteria.name == criteria_name:
                return criteria.lpl
        return None

    def set_criteria_depends(self, criteria_name, depends_on_criteria):
        """
        Sets the dependency for a specific criteria.

        Args:
            criteria_name (str): The name of the criteria.
            depends_on_criteria (str): The criteria that this criteria depends on.
        """
        for criteria in self._criteria:
            if criteria.name == criteria_name:
                criteria.depends_on = depends_on_criteria

    def set_criteria_score(self, criteria_name, score_value):
        """
        Sets the score for a specific criteria.

        Args:
            criteria_name (str): The name of the criteria.
            score_value (float): The score value to set.
        """
        for criteria in self._criteria:
            if criteria.name == criteria_name:
                criteria.score = score_value

    def clear_criteria_assessment(self, criteria_name):
        """
        Clears the whole assessment of a specific criteria: score, colour and
        prediction result. Used when a criteria is not assessed at all (e.g.
        HIC15/Ares-3ms when head Ares stays below 80 g), so no rating colour
        or Prediction.Check survives in the report.

        Args:
            criteria_name (str): The name of the criteria.
        """
        for criteria in self._criteria:
            if criteria.name == criteria_name:
                criteria.score = None
                criteria.color = None
                criteria.prediction_result = None

    def get_criteria_score(self, criteria_name):
        """
        Retrieves the score for a specific criteria.

        Args:
            criteria_name (str): The name of the criteria.

        Returns:
            float: The score value.
        """
        for criteria in self._criteria:
            if criteria.name == criteria_name:
                return criteria.score
        return None

    def get_criteria_value(self, criteria_name: str) -> float:
        """
        Retrieves the value for a specific criteria.

        Args:
            criteria_name (str): The name of the criteria.

        Returns:
            float: The value of the criteria.
        """
        for criteria in self._criteria:
            if criteria.name == criteria_name:
                return criteria.value
        return None

    def set_criteria_prediction_color(self, criteria_name: str, prediction_color: str):
        """
        Sets the prediction color for a specific criteria.

        Args:
            criteria_name (str): The name of the criteria.
            prediction_color (str): The prediction color to set.
        """
        found = False
        for criteria in self._criteria:
            if criteria.name == criteria_name:
                found = True
                criteria.set_prediction(prediction_color)

        if not found:
            logger.error(
                f"Error. Criteria '{criteria_name}' not found in body region '{self.name}'"
            )
        else:
            self.resolve_dependancies()

    def get_criteria_capping_value(self, criteria_name: str) -> float:
        """
        Retrieves the capping value for a specific criteria.

        Args:
            criteria_name (str): The name of the criteria.

        Returns:
            float: The capping value.
        """
        for criteria in self._criteria:
            if criteria.name == criteria_name:
                return criteria.capping_value
        return None

    def resolve_dependancies(self):
        """
        Resolves dependencies between criteria in the body region.
        """
        ares_criteria_present = (
            self.contains_criteria("Ares")
            and self.contains_criteria("HIC15")
            and self.contains_criteria("Ares-3ms")
        )
        self.set_criteria_depends("HIC15", "Ares")
        self.set_criteria_depends("Ares-3ms", "Ares")

        if not ares_criteria_present:
            return
        else:
            logger.debug(f"Ares value: {self.get_criteria_value('Ares')}")
            logger.debug(f"HIC15 value: {self.get_criteria_value('HIC15')}")
            logger.debug(f"Ares-3ms value: {self.get_criteria_value('Ares-3ms')}")
            if self.get_criteria_value("Ares") < self.get_criteria_lpl("Ares"):
                logger.debug(
                    "Ares value is less than its LPL. HIC15 and Ares-3ms are "
                    "not assessed (score, colour and prediction result "
                    "cleared); Ares score set to 100.0"
                )
                self.clear_criteria_assessment("HIC15")
                self.clear_criteria_assessment("Ares-3ms")
                self.set_criteria_score("Ares", 100.0)
            else:
                logger.debug(
                    "Ares value is greater than or equal to its LPL. Setting Ares score to None and updating HIC15 and Ares-3ms values"
                )
                self.set_criteria_score("Ares", None)
                for criteria in self._criteria:
                    if criteria.name == "HIC15":
                        criteria.set_value(self.get_criteria_value("HIC15"))
                    if criteria.name == "Ares-3ms":
                        criteria.set_value(self.get_criteria_value("Ares-3ms"))

    def __repr__(self):
        """
        Returns a string representation of the BodyRegion object.

        Returns:
            str: A string representation of the BodyRegion object.
        """
        return f"{self.__class__.__name__}('{self.name}', {self._criteria})"
