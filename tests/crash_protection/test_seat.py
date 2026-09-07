# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import unittest
from euroncap_rating_2026.crash_protection.seat import Seat
from euroncap_rating_2026.crash_protection.dummy import Dummy
from euroncap_rating_2026.crash_protection.body_region import BodyRegion
from euroncap_rating_2026.crash_protection.criteria import Criteria


class TestSeat(unittest.TestCase):
    def setUp(self):
        self.criteria = Criteria(name="Ares", hpl=50.0, lpl=100.0, value=75.0)
        self.body_region = BodyRegion(name="Head & Neck")
        self.body_region.set_criteria_list([self.criteria])
        self.dummy = Dummy(
            name="HIII-50", body_region_list=[self.body_region.model_dump()]
        )
        self.seat = Seat(name="Driver", dummy=self.dummy)

    def test_initialization(self):
        self.assertEqual(self.seat.name, "Driver")
        self.assertEqual(self.seat.dummy.name, "HIII-50")

    def test_get_body_region(self):
        body_region = self.seat.dummy.get_body_region("Head & Neck")
        self.assertEqual(body_region.name, "Head & Neck")


if __name__ == "__main__":
    unittest.main()
