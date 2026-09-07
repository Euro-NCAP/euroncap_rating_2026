# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from pydantic import BaseModel
from euroncap_rating_2026.crash_protection.dummy import Dummy


class Seat(BaseModel):
    """
    Represents a seat in a vehicle, containing a name and a dummy.

    Attributes:
        name (str): The name of the seat.
        dummy (Dummy): The dummy associated with the seat.
    """

    name: str
    dummy: Dummy

    def __repr__(self):
        """
        Returns a string representation of the Seat object.

        Returns:
            str: A string representation of the Seat object.
        """
        return f"Seat('{self.name}', {self.dummy})"
