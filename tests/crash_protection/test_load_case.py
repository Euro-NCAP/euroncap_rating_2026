# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import unittest
from euroncap_rating_2026.crash_protection.load_case import LoadCase
from euroncap_rating_2026.crash_protection.seat import Seat
from euroncap_rating_2026.crash_protection.dummy import Dummy
from euroncap_rating_2026.crash_protection.body_region import BodyRegion
from euroncap_rating_2026.crash_protection.criteria import Criteria
import pandas as pd


class TestLoadCase(unittest.TestCase):
    def setUp(self):
        self.criteria = Criteria(name="Ares", hpl=50.0, lpl=100.0, value=75.0)
        self.body_region = BodyRegion(name="Head & Neck", _criteria=[self.criteria])
        self.dummy = Dummy(
            name="HIII-50", body_region_list=[self.body_region.model_dump()]
        )
        self.seat = Seat(name="Driver", dummy=self.dummy.model_dump())
        self.load_case = LoadCase(name="Virtual-Low", seats=[self.seat.model_dump()])

    def test_get_seat(self):
        seat = self.load_case.get_seat("Driver")
        self.assertEqual(seat.name, "Driver")

    def test_get_loadcase_from_row(self):
        df = pd.DataFrame(
            {
                "Loadcase": ["Virtual-Low", "Virtual-Low"],
                "Seat position": ["Driver", "Driver"],
                "Dummy": ["HIII-50", "HIII-50"],
                "Body region": ["Head & Neck", "Head & Neck"],
                "Criteria": ["Ares", "HIC15"],
                "HPL": [50.0, 500.0],
                "LPL": [100.0, 700.0],
                "Capping": [90.0, None],
                "Value": [75.0, 600.0],
                "Modifier": [None, None],
            }
        )
        load_case, _ = LoadCase.get_loadcase_from_row(0, df)
        self.assertEqual(load_case.name, "Virtual-Low")
        self.assertEqual(len(load_case.seats), 1)
        self.assertEqual(load_case.seats[0].name, "Driver")


if __name__ == "__main__":
    unittest.main()
