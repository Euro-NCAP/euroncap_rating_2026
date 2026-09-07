# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import unittest
from euroncap_rating_2026.crash_protection.body_region import BodyRegion
from euroncap_rating_2026.crash_protection.criteria import Criteria, CriteriaType
from euroncap_rating_2026.crash_protection.measurement import Measurement
import numpy as np
import pandas as pd
import logging


class TestBodyRegion(unittest.TestCase):
    def setUp(self):
        self.criteria1 = Criteria(
            name="Ares",
            hpl=None,
            lpl=80.0,
            value=75.0,
            criteria_type=CriteriaType.NONE_SWITCH,
        )
        self.criteria2 = Criteria(
            name="HIC15",
            hpl=500.0,
            lpl=700.0,
            value=600.0,
            criteria_type=CriteriaType.CRITERIA,
        )
        self.criteria3 = Criteria(
            name="Ares-3ms",
            hpl=72.0,
            lpl=80.0,
            value=0.0,
            criteria_type=CriteriaType.CRITERIA,
        )
        self.criteria4 = Criteria(
            name="Fx,shear",
            hpl=1.9,
            lpl=3.1,
            value=3.1,
            criteria_type=CriteriaType.CRITERIA,
        )
        self.body_region = BodyRegion(name="Head & Neck")
        self.body_region.set_criteria_list(
            [self.criteria1, self.criteria2, self.criteria3, self.criteria4]
        )

    def test_set_criteria_list(self):
        new_criteria = Criteria(name="Ares-3ms", hpl=72.0, lpl=80.0, value=0.0)
        self.body_region.set_criteria_list([new_criteria])
        self.assertEqual(len(self.body_region._criteria), 1)
        self.assertEqual(self.body_region._criteria[0].name, "Ares-3ms")

    def test_set_criteria_value(self):
        self.body_region.set_criteria_value("Ares", 90.0)
        self.assertEqual(self.body_region.get_criteria_value("Ares"), 90.0)

    def test_set_criteria_capping_value(self):
        self.body_region.set_criteria_capping_value("Ares", 80.0)
        self.assertEqual(self.body_region.get_criteria_capping_value("Ares"), 80.0)

    def test_resolve_dependancies(self):
        self.body_region.set_criteria_value("Ares", 60.0)
        self.body_region.resolve_dependancies()
        for criteria in self.body_region._criteria:
            if criteria.name == "HIC15":
                self.assertEqual(criteria.depends_on, "Ares")
            elif criteria.name == "Ares-3ms":
                self.assertEqual(criteria.depends_on, "Ares")

    def test_ares_below_80(self):
        self.body_region.set_criteria_value("Ares", 70.0)
        self.body_region.resolve_dependancies()
        self.assertEqual(self.body_region.get_criteria_score("Ares"), 100)
        self.assertIsNone(self.body_region.get_criteria_score("HIC15"))
        self.assertIsNone(self.body_region.get_criteria_score("Ares-3ms"))

    def test_ares_above_80(self):
        self.body_region.set_criteria_value("Ares", 90.0)
        self.assertIsNone(self.body_region.get_criteria_score("Ares"))
        self.assertIsNotNone(self.body_region.get_criteria_score("HIC15"))
        self.assertIsNotNone(self.body_region.get_criteria_score("Ares-3ms"))

    def test_ares_below_80_clears_colors_and_prediction_result(self):
        """Below the 80 g hard-contact gate HIC15 and Ares-3ms are not
        assessed at all: no score, no rating colour, no Prediction.Check.
        Ares itself keeps its colour."""
        self.body_region.set_criteria_value("HIC15", 650.0)
        self.body_region.set_criteria_value("Ares-3ms", 75.0)
        # Above the gate first, so both criteria carry a computed colour and
        # a stale prediction check the gate must then wipe.
        self.body_region.set_criteria_value("Ares", 90.0)
        for criteria in self.body_region._criteria:
            if criteria.name in ("HIC15", "Ares-3ms"):
                self.assertIsNotNone(criteria.color)
                criteria.prediction_result = "Correct"

        self.body_region.set_criteria_value("Ares", 70.0)

        for criteria in self.body_region._criteria:
            if criteria.name in ("HIC15", "Ares-3ms"):
                self.assertIsNone(criteria.score)
                self.assertIsNone(criteria.color)
                self.assertIsNone(criteria.prediction_result)
            elif criteria.name == "Ares":
                self.assertEqual(criteria.score, 100.0)
                self.assertIsNotNone(criteria.color)

    def test_ares_above_80_keeps_colors(self):
        """At or above 80 g HIC15 and Ares-3ms are assessed and coloured."""
        self.body_region.set_criteria_value("HIC15", 650.0)
        self.body_region.set_criteria_value("Ares-3ms", 75.0)
        self.body_region.set_criteria_value("Ares", 90.0)

        for criteria in self.body_region._criteria:
            if criteria.name in ("HIC15", "Ares-3ms"):
                self.assertIsNotNone(criteria.score)
                self.assertIsNotNone(criteria.color)

    def test_ares_80(self):
        self.body_region.set_criteria_value("Ares", 80.0)
        self.assertIsNone(self.body_region.get_criteria_score("Ares"))
        self.assertIsNotNone(self.body_region.get_criteria_score("HIC15"))
        self.assertIsNotNone(self.body_region.get_criteria_score("Ares-3ms"))

    def test_score_with_ares_below_threshold(self):
        self.body_region.set_criteria_value("Ares", 79.0)  # led to score 100
        self.body_region.set_criteria_value(
            "HIC15", 600.0
        )  # Ares is <80 so score is None
        self.body_region.set_criteria_value(
            "Ares-3ms", 75.0
        )  # Ares is <80 so score is None
        self.body_region.set_criteria_value("Fx,shear", 1.9)  # led to green: score 100
        self.body_region.compute_bodyregion_score()
        self.assertEqual(self.body_region.get_bodyregion_score(), 80.0)

    def test_score_with_ares_above_threshold(self):
        self.body_region.set_criteria_value("Ares", 81.0)  # led to score None
        self.body_region.set_criteria_value("HIC15", 600.0)  # led to orange: score 40
        self.body_region.set_criteria_value("Ares-3ms", 75.0)  # led to orange: score 40
        self.body_region.set_criteria_value("Fx,shear", 1.9)  # led to yellow: score 80
        self.body_region.compute_bodyregion_score()
        self.assertEqual(self.body_region.get_bodyregion_score(), 40.0)

    def test_score_with_ares_equals_threshold(self):
        self.body_region.set_criteria_value("Ares", 80.0)  # led to score None
        self.body_region.set_criteria_value("HIC15", 600.0)  # led to score orange: 40
        self.body_region.set_criteria_value("Ares-3ms", 75.0)  # led to score orange: 40
        self.body_region.set_criteria_value("Fx,shear", 1.9)  # led to yellow: score 80
        self.body_region.compute_bodyregion_score()
        self.assertEqual(self.body_region.get_bodyregion_score(), 40.0)

    def test_set_criteria_prediction_color(self):
        self.body_region.set_criteria_prediction_color("Ares", "red")
        self.assertEqual(self.criteria1.prediction, "red")

    def test_contains_criteria(self):
        self.assertTrue(self.body_region.contains_criteria("Ares"))
        self.assertFalse(self.body_region.contains_criteria("NonExistentCriteria"))

    def test_get_criteria_lpl(self):
        self.assertEqual(self.body_region.get_criteria_lpl("Ares"), 80.0)
        self.assertIsNone(self.body_region.get_criteria_lpl("NonExistentCriteria"))

    def test_set_criteria_depends(self):
        self.body_region.set_criteria_depends("Ares", "HIC15")
        self.assertEqual(self.criteria1.depends_on, "HIC15")

    def test_get_criteria_score(self):
        self.body_region.set_criteria_score("Ares", 95.0)
        self.assertEqual(self.body_region.get_criteria_score("Ares"), 95.0)
        self.assertIsNone(self.body_region.get_criteria_score("NonExistentCriteria"))

    def test_get_criteria_value(self):
        self.assertEqual(self.body_region.get_criteria_value("Ares"), 75.0)
        self.assertIsNone(self.body_region.get_criteria_value("NonExistentCriteria"))

    def test_get_criteria_capping_value(self):
        self.criteria1.set_capping_value(85.0)
        self.assertEqual(self.body_region.get_criteria_capping_value("Ares"), 85.0)
        self.assertIsNone(
            self.body_region.get_criteria_capping_value("NonExistentCriteria")
        )

    def test_default_modifier(self):
        self.assertEqual(self.body_region.get_inspection(), None)

    def test_set_inspection(self):
        self.body_region.set_inspection(1.5)
        self.assertEqual(self.body_region.get_inspection(), 1.5)

        self.body_region.set_inspection(None)
        self.assertIsNone(self.body_region.get_inspection())

    def test_default_max_score(self):
        self.assertIsNone(self.body_region.get_max_score())

    def test_set_max_score(self):
        self.body_region.set_max_score(100.0)
        self.assertEqual(self.body_region.get_max_score(), 100.0)

        self.body_region.set_max_score(None)
        self.assertIsNone(self.body_region.get_max_score())

    def test_default_score(self):
        self.assertIsNone(self.body_region.get_score())

    def test_compute_score_with_all_values(self):
        self.body_region.set_inspection(10.0)
        self.body_region.set_max_score(100.0)

        self.body_region.set_criteria_value("Ares", 81.0)  # led to score None
        self.body_region.set_criteria_value("HIC15", 600.0)  # led to orange: score 40
        self.body_region.set_criteria_value("Ares-3ms", 75.0)  # led to orange: score 40
        self.body_region.set_criteria_value("Fx,shear", 1.9)  # led to yellow: score 80
        self.body_region.compute_bodyregion_score()
        # Bodyregion score is 40
        self.assertEqual(self.body_region.get_bodyregion_score(), 40.0)
        self.body_region.compute_score()
        # 40-10*100/100
        self.assertEqual(self.body_region.get_score(), 30.0)

    def test_compute_score_without_modifier(self):
        self.body_region.set_max_score(100.0)
        self.body_region.set_criteria_value("Ares", 70.0)
        self.body_region.compute_bodyregion_score()
        self.body_region.compute_score()
        self.assertIsNone(self.body_region.get_score())

    def test_compute_score_without_max_score(self):
        self.body_region.set_inspection(10.0)
        self.body_region.set_criteria_value("Ares", 70.0)
        self.body_region.compute_bodyregion_score()
        self.body_region.compute_score()
        self.assertIsNone(self.body_region.get_score())

    def test_compute_score_without_bodyregion_score(self):
        self.body_region.set_inspection(10.0)
        self.body_region.set_max_score(100.0)
        # bodyregion_score is called when set_criteria_list method is called above
        self.body_region.compute_score()
        self.assertEqual(self.body_region.get_score(), 0.0)


class TestBodyRegionBlankMeasurement(unittest.TestCase):
    """A blank measurement (Measurement.value=None) must not contribute to
    the deduction total -- neither as a real 0 nor, for inverted bands like
    "Effective height modifier", as the maximum penalty."""

    def setUp(self):
        self.body_region = BodyRegion(name="Chest")
        self.body_region.set_criteria_list(
            [
                Criteria(
                    name="Ares",
                    hpl=None,
                    lpl=80.0,
                    value=75.0,
                    criteria_type=CriteriaType.NONE_SWITCH,
                )
            ]
        )
        self.body_region.set_inspection(0.0)
        self.body_region.set_max_score(100.0)

    def test_blank_measurement_excluded_from_deduction(self):
        self.body_region.set_measurement_list(
            [Measurement(name="Effective height modifier", value=None)]
        )
        self.body_region.compute_score()
        blank_score = self.body_region.get_score()

        self.body_region.set_measurement_list([])
        self.body_region.compute_score()
        no_measurement_score = self.body_region.get_score()

        self.assertEqual(blank_score, no_measurement_score)


if __name__ == "__main__":
    unittest.main()
