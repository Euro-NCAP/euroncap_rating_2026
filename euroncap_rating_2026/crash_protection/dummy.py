# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from euroncap_rating_2026.crash_protection.body_region import BodyRegion
from euroncap_rating_2026.crash_protection.criteria import Criteria
from euroncap_rating_2026.crash_protection.measurement import Measurement
from euroncap_rating_2026.numeric import round_half_up
import pandas as pd
import logging
import numpy as np

logger = logging.getLogger(__name__)

_BIORID_COUPLED_CAPPING_CRITERIA = frozenset({"T-HRC start", "T1 acceleration"})


class Dummy(BaseModel):
    """
    Represents a dummy used in vehicle crash tests.

    Attributes:
        name (str): The name of the dummy.
        body_region_list (List[BodyRegion]): A list of body regions in the dummy.
        max_score (Optional[float]): The maximum score for the dummy.
        score (Optional[float]): The score for the dummy.
        capping (Optional[bool]): Indicates if capping is applied to the dummy.
    """

    name: str = Field(
        "HIII-50",
        pattern="^(HIII-50|HIII-05|HIII-95|THOR-50|Q6|Q10|WorldSID-50|Q6-Side|Q10-Side|BioRID-50|HPM|Adult Headform|Child Headform|aPLI|Upper legform|WorldSID-50-Farside)$",
    )
    body_region_list: List[BodyRegion] = []
    measurement_list: List[Measurement] = []
    max_score: Optional[float] = None
    score: Optional[float] = None
    capping: Optional[bool] = None

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
        self.max_score = round_half_up(value, 4)

    def compute_capping(self) -> None:
        """
        Computes the capping status based on the criteria of the body regions.
        """
        all_criteria = [
            criteria
            for body_region in self.body_region_list
            for criteria in body_region._criteria
        ]
        if self.name == "BioRID-50":
            coupled = {
                c.name: c.capping
                for c in all_criteria
                if c.name in _BIORID_COUPLED_CAPPING_CRITERIA
            }
            coupled_capping = len(coupled) == len(
                _BIORID_COUPLED_CAPPING_CRITERIA
            ) and all(coupled.values())
            other_capping = any(
                c.capping
                for c in all_criteria
                if c.name not in _BIORID_COUPLED_CAPPING_CRITERIA
            )
            self.capping = coupled_capping or other_capping
        else:
            self.capping = any(c.capping for c in all_criteria)

    def apply_coupled_color_rule(self) -> None:
        """
        For BioRID-50: T-HRC start and T1 acceleration must BOTH exceed their
        capping limit for either to be colored red. If only one does, both are green.
        """
        if self.name != "BioRID-50":
            return
        all_criteria = [
            criteria
            for body_region in self.body_region_list
            for criteria in body_region._criteria
        ]
        coupled = {
            c.name: c
            for c in all_criteria
            if c.name in _BIORID_COUPLED_CAPPING_CRITERIA
        }
        if len(coupled) < len(_BIORID_COUPLED_CAPPING_CRITERIA):
            return
        both_capping = all(c.capping for c in coupled.values())
        for c in coupled.values():
            c.color = "red" if both_capping else "green"

    def get_capping(self) -> Optional[bool]:
        """
        Retrieves the capping status.

        Returns:
            Optional[bool]: The capping status.
        """
        return self.capping

    def set_capping(self, value: Optional[bool]):
        """
        Sets the capping status.

        Args:
            value (Optional[bool]): The capping status.
        """
        self.capping = value

    def compute_score(self) -> Optional[float]:
        """
        Computes the score for the dummy based on the scores of the body regions.
        """
        scores = [
            body_region.get_score()
            for body_region in self.body_region_list
            if body_region.get_score() is not None
        ]
        if not scores:
            return None
        logger.debug(f"Body region scores: {scores}")
        total_score = sum(scores)
        if self.capping:
            total_score = 0.0
        self.score = round_half_up(total_score, 4)

    def get_score(self) -> Optional[float]:
        """
        Retrieves the score for the dummy.

        Returns:
            Optional[float]: The score for the dummy.
        """
        return self.score

    def set_score(self, value: Optional[float]):
        """
        Sets the score for the dummy.

        Args:
            value (Optional[float]): The score for the dummy.
        """
        if value is None:
            self.score = None
            return
        self.score = round_half_up(value, 4)

    def get_body_region(self, body_region_name: str) -> BodyRegion:
        """
        Retrieves a body region by its name.

        Args:
            body_region_name (str): The name of the body region.

        Returns:
            BodyRegion: The body region object.

        Raises:
            ValueError: If the body region is not found.
        """
        for body_region in self.body_region_list:
            if body_region.name == body_region_name:
                return body_region
        raise ValueError(
            f"Body region '{body_region_name}' not found in dummy '{self.name}'"
        )

    @staticmethod
    def get_dummy_from_row(dummy_name, df, row_index):
        """
         This function processes a DataFrame row by row to construct a Dummy object
        based on the provided model description. It validates the input data, extracts
        criteria for each body region, and resolves dependencies between criteria.
            dummy_name (str): The name of the dummy to be created.
            df (pd.DataFrame): The DataFrame containing the data for the dummy.
            row_index (int): The starting index of the row in the DataFrame.
            Dummy: The created Dummy object containing body regions and criteria.
            int: The updated row index after processing the rows for the dummy.
        Raises:
            ValueError: If the dummy name is not found in the model description.
            ValueError: If there are not enough rows in the DataFrame to load all criteria
                for the dummy.
        Notes:
            - The function handles special cases for specific dummy names (e.g., "Q6", "Q10",
              "Q6-Side", "Q10-Side", "WorldSID-50") to adjust criteria dependencies and values.
            - For dummies like "Q6" and "Q10", specific criteria such as "Ares-3ms" and "Ares"
              are handled differently for the "Head" body region.
            - For "WorldSID-50", additional adjustments are made for criteria like "Ares",
              "HIC15", and "Ares-3ms".
        Example:
            dummy, updated_row_index = get_dummy_from_row(
                dummy_name="Q10",
                df=dataframe,
                row_index=0,
        """

        logger.debug(f"Loading dummy: {dummy_name}")

        body_region_list = []
        body_region_measurement_dict = {}

        num_rows_to_inspect = 0
        body_region_criteria_map = {}
        body_region_test_points_map = {}
        dummy_capping_for_contact = False

        for i in range(row_index, len(df)):
            if not pd.isna(df.iloc[i]["Dummy"]) and i != row_index:
                break

            num_rows_to_inspect += 1
            if not pd.isna(df.iloc[i]["Body region"]):
                current_body_region = df.iloc[i]["Body region"]
                if "Test point" in df.columns and not pd.isna(df.iloc[i]["Test point"]):
                    current_body_region = (
                        f"{current_body_region}_{df.iloc[i]['Test point']}"
                    )
                    body_region_test_points_map[current_body_region] = df.iloc[i][
                        "Body region"
                    ]
                if current_body_region not in body_region_criteria_map:
                    body_region_criteria_map[current_body_region] = []
            if not pd.isna(df.iloc[i]["Criteria"]):
                if current_body_region is None:
                    logger.error(
                        f"Error: Criteria '{df.iloc[i]['Criteria']}' found without a corresponding body region at row {i}"
                    )
                    continue
                body_region_criteria_map[current_body_region].append(
                    df.iloc[i]["Criteria"]
                )

        logger.debug(f"Number of rows to inspect: {num_rows_to_inspect}")
        logger.debug(f"Body regions map: {body_region_criteria_map}")
        start_row_index = row_index
        start_row = df.iloc[start_row_index]

        if (
            pd.isna(start_row["Dummy"])
            or pd.isna(start_row["Body region"])
            or pd.isna(start_row["Criteria"])
        ):
            logger.error(
                f"Error: Indicated row {row_index} is not a valid row to start loading dummy {dummy_name}"
            )
            return None

        head_ares3ms_value = None
        head_hic15_value = None

        body_region_state = {}
        current_body_region_key = None
        row_index = start_row_index
        end_row_index = start_row_index + num_rows_to_inspect

        while row_index < end_row_index:
            row = df.iloc[row_index]

            if not pd.isna(row["Body region"]):
                current_body_region_key = row["Body region"]
                if "Test point" in df.columns and not pd.isna(row["Test point"]):
                    current_body_region_key = (
                        f"{current_body_region_key}_{row['Test point']}"
                    )

            if current_body_region_key is None:
                logger.error(
                    f"Error: Criteria '{row['Criteria']}' found without a corresponding body region at row {row_index}"
                )
                row_index += 1
                continue

            body_region_name = body_region_test_points_map.get(
                current_body_region_key, current_body_region_key
            )
            logger.debug(
                f"  Loading body region: {body_region_name} from row {row_index}"
            )

            if body_region_name not in body_region_state:
                body_region_state[body_region_name] = {
                    "criteria": [],
                    "measurements": [],
                }

            criteria_name = row["Criteria"]
            logger.debug(f"    Loading criteria: {criteria_name} at index {row_index}")

            is_capping = False
            is_measurement = False

            # If LPL and HPL are both NaN, check if it's a measurement:
            # - If Capping is present and not NaN, it's a capping criteria
            # - If Capping is present and NaN, it's a measurement
            # - If Capping is not present, it's a measurement
            if pd.isna(row["LPL"]) and pd.isna(row["HPL"]):
                if "Capping" in row:
                    if not pd.isna(row["Capping"]):
                        is_capping = True
                    else:
                        is_measurement = True
                else:
                    is_measurement = True

            if criteria_name == "Farside excursion":
                row_index += 1
                continue
            if is_measurement:
                logger.debug(f"Measurement {criteria_name} detected at row {row_index}")
                if criteria_name == "Direct contact with pole":
                    logger.debug(
                        f"Direct contact with pole measurement detected at row {row_index} with value {row['Value']}"
                    )
                    value = row["Value"]
                    dummy_capping_for_contact = str(value).lower() == "true" or (
                        isinstance(value, (int, float)) and value == 1.0
                    )
                    logger.debug(
                        f"Setting dummy_capping_for_contact to {dummy_capping_for_contact}"
                    )
                    row_index += 1
                    continue
                measurement_value = None if pd.isna(row["Value"]) else row["Value"]
                measurement = Measurement(name=criteria_name, value=measurement_value)
                body_region_state[body_region_name]["measurements"].append(measurement)
                row_index += 1
                continue

            criteria = Criteria.get_criteria_from_row(row, dummy_name=dummy_name)

            if body_region_name == "Head" and criteria.name == "Ares-3ms":
                head_ares3ms_value = criteria.value

            if body_region_name == "Head" and criteria.name == "HIC15":
                head_hic15_value = criteria.value

            body_region_state[body_region_name]["criteria"].append(criteria)
            row_index += 1

        for body_region_name, state in body_region_state.items():
            body_region_measurement_dict[body_region_name] = state["measurements"]
            body_region_obj = BodyRegion(name=body_region_name)
            logger.debug(
                f"Measurement list for body region '{body_region_name}': {body_region_measurement_dict[body_region_name]}"
            )
            for measurement in body_region_measurement_dict[body_region_name]:
                logger.debug(
                    f"Measurement '{measurement.name}' modifier: {getattr(measurement, 'modifier', None)}"
                )
            body_region_obj.set_measurement_list(
                body_region_measurement_dict[body_region_name]
            )
            body_region_obj.set_criteria_list(state["criteria"])
            body_region_list.append(body_region_obj)

        if row_index != start_row_index + num_rows_to_inspect:
            error_message = f"Error: Mismatch in row index calculation. Expected {start_row_index + num_rows_to_inspect}, but got {row_index}."
            logger.error(error_message)
            raise ValueError(error_message)

        if dummy_name in ["Q6", "Q10", "Q6-Side", "Q10-Side"]:
            for body_region in body_region_list:
                if body_region.name == "Head":
                    for criteria in body_region._criteria:
                        # For this dummies, Head Ares-3ms does not depend on Ares
                        if criteria.name == "Ares-3ms":
                            criteria.set_value(head_ares3ms_value)
                            criteria.depends_on = None
                        # For this dummies, Head Ares has always None score
                        if criteria.name == "Ares":
                            criteria.score = None
                # For this dummies, Neck My,extension depends on Head Ares
                elif body_region.name == "Neck":
                    for criteria in body_region._criteria:
                        if criteria.name == "My,extension":
                            head_ares_criteria = next(
                                (
                                    c
                                    for br in body_region_list
                                    if br.name == "Head"
                                    for c in br._criteria
                                    if c.name == "Ares"
                                ),
                                None,
                            )
                            if head_ares_criteria.value < head_ares_criteria.lpl:
                                # Not assessed below the 80 g hard-contact
                                # gate: clear the colour and prediction
                                # result along with the score, so no rating
                                # colour survives in the report.
                                body_region.clear_criteria_assessment("My,extension")
                            else:
                                criteria.set_value(criteria.value)

                            criteria.depends_on = "Ares"

        if dummy_name == "WorldSID-50":
            for body_region in body_region_list:
                if body_region.name == "Head":
                    for criteria in body_region._criteria:
                        if criteria.name == "Ares":
                            criteria.score = None
                            if criteria.capping_value:
                                if criteria.value >= criteria.capping_value:
                                    criteria.capping = True
                                else:
                                    criteria.capping = False
                        elif criteria.name == "HIC15":
                            criteria.depends_on = None
                            criteria.set_value(head_hic15_value)
                        elif criteria.name == "Ares-3ms":
                            criteria.depends_on = None
                            criteria.set_value(head_ares3ms_value)

        dummy = Dummy(name=dummy_name, body_region_list=body_region_list)
        dummy.set_capping(dummy_capping_for_contact)
        dummy.apply_coupled_color_rule()
        return (
            dummy,
            row_index,
        )

    def __repr__(self):
        return f"Dummy('{self.name}', {self.body_region_list})"
