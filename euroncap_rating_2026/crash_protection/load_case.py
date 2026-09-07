# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from pydantic import BaseModel, Field
from typing import List
from euroncap_rating_2026.crash_protection.seat import Seat
from euroncap_rating_2026.crash_protection.dummy import Dummy
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def _has_text(value):
    return pd.notna(value) and str(value).strip() != ""


class LoadCase(BaseModel):
    """
    Represents a load case in a vehicle crash test.

    Attributes:
        name (str): The name of the load case.
        seats (List[Seat]): A list of seats in the load case.
    """

    name: str
    seats: List[Seat] = Field(default_factory=list)
    raw_seats: List[Seat] = Field(default_factory=list)

    @property
    def id(self) -> str:
        """
        Generates a unique ID for the load case based on its name and seats.

        Returns:
            str: The unique ID of the load case.
        """
        id_parts = [self.name]
        for seat in self.seats:
            id_parts.append(seat.name)
            id_parts.append(seat.dummy.name)
        return "_".join(id_parts)

    def get_seat(self, seat_name: str) -> Seat:
        """
        Retrieves a seat by its name.

        Args:
            seat_name (str): The name of the seat.

        Returns:
            Seat: The seat object.

        Raises:
            ValueError: If the seat is not found.
        """
        for seat in self.seats:
            if seat.name == seat_name:
                return seat
        raise ValueError(f"Seat '{seat_name}' not found in load case '{self.name}'")

    @staticmethod
    def get_loadcase_from_row(row_index, df):
        """
        Creates a LoadCase object from a row in a DataFrame.

        Args:
            row_index (int): The index of the row.
            df (pd.DataFrame): The DataFrame containing the data.

        Returns:
            LoadCase: The created LoadCase object.
            int: The updated row index.
        """
        row = df.iloc[row_index]
        load_case_name = row["Loadcase"]

        seats_info = []
        for index in range(row_index, len(df)):
            row = df.iloc[index]
            if _has_text(row["Loadcase"]) and index != row_index:
                break
            seat = row["Seat position"]
            dummy = row["Dummy"]
            if _has_text(seat) and _has_text(dummy):
                seats_info.append((index, seat, dummy))
        logger.debug(f"seats_info: {seats_info}")

        seats = []
        last_row_index = row_index
        for seat_row_index, seat_name, dummy_name in seats_info:
            dummy, parsed_row_index = Dummy.get_dummy_from_row(
                dummy_name, df, seat_row_index
            )
            seat = Seat(name=seat_name, dummy=dummy)
            seats.append(seat)
            last_row_index = max(last_row_index, parsed_row_index)

        return LoadCase(name=load_case_name, seats=seats), last_row_index

    def __repr__(self):
        """
        Returns a string representation of the LoadCase object.

        Returns:
            str: A string representation of the LoadCase object.
        """
        return f"class LoadCase(BaseModel):('{self.name}', {self.seats})"
