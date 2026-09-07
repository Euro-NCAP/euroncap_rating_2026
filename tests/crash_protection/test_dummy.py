# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import unittest
from euroncap_rating_2026.crash_protection.dummy import Dummy
from euroncap_rating_2026.crash_protection.body_region import BodyRegion
from euroncap_rating_2026.crash_protection.criteria import Criteria
from euroncap_rating_2026.crash_protection.measurement import Measurement
import pandas as pd
import numpy as np
import logging


class TestDummy(unittest.TestCase):
    def setUp(self):
        df = pd.DataFrame(
            {
                "Dummy": ["HIII-50", None],
                "Body region": ["Head & Neck", None],
                "Criteria": ["My,extension", "HIC15"],
                "HPL": [50.0, 500.0],
                "LPL": [100.0, 700.0],
                "Capping": [90.0, None],
                "Value": [75.0, 600.0],
                "Modifier": [None, None],
            }
        )
        self.dummy, _ = Dummy.get_dummy_from_row("HIII-50", df, 0)
        self.dummy.body_region_list[0].set_max_score(1.25)
        self.dummy.body_region_list[0].set_inspection(0.0)
        self.dummy.body_region_list[0].compute_bodyregion_score()
        self.dummy.body_region_list[0].compute_score()

    def test_get_body_region(self):
        body_region = self.dummy.get_body_region("Head & Neck")
        self.assertEqual(body_region.name, "Head & Neck")

    def test_get_dummy_from_row(self):
        df = pd.DataFrame(
            {
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
        dummy, _ = Dummy.get_dummy_from_row("HIII-50", df, 0)
        self.assertEqual(dummy.name, "HIII-50")
        self.assertEqual(len(dummy.body_region_list), 1)
        self.assertEqual(dummy.body_region_list[0].name, "Head & Neck")

    def test_max_score(self):
        self.dummy.set_max_score(100.0)
        self.assertEqual(self.dummy.get_max_score(), 100.0)
        self.dummy.set_max_score(None)
        self.assertIsNone(self.dummy.get_max_score())

    def test_score_precision(self):
        self.dummy.set_score(123.456789)
        self.assertEqual(self.dummy.get_score(), 123.4568)
        self.dummy.set_score(None)
        self.assertIsNone(self.dummy.get_score())

    def test_compute_score(self):
        self.dummy.compute_capping()
        self.dummy.compute_score()
        self.assertEqual(self.dummy.get_score(), 0.5)

    def test_capping(self):
        self.dummy.compute_capping()
        self.assertFalse(self.dummy.get_capping())
        self.dummy.body_region_list[0].set_criteria_value("My,extension", 1000.0)
        self.dummy.compute_capping()
        self.assertTrue(self.dummy.get_capping())

    def test_set_body_region_list(self):
        new_body_region = BodyRegion(name="Chest")
        self.dummy.body_region_list = [new_body_region]
        self.assertEqual(len(self.dummy.body_region_list), 1)
        self.assertEqual(self.dummy.body_region_list[0].name, "Chest")

    def test_get_set_max_score(self):
        self.dummy.set_max_score(200.0)
        self.assertEqual(self.dummy.get_max_score(), 200.0)
        self.dummy.set_max_score(None)
        self.assertIsNone(self.dummy.get_max_score())

    def test_get_set_score(self):
        self.dummy.set_score(456.789)
        self.assertEqual(self.dummy.get_score(), 456.789)
        self.dummy.set_score(None)
        self.assertIsNone(self.dummy.get_score())

    def test_get_set_capping(self):
        self.dummy.capping = True
        self.assertTrue(self.dummy.get_capping())
        self.dummy.capping = False
        self.assertFalse(self.dummy.get_capping())
        self.dummy.capping = None
        self.assertIsNone(self.dummy.get_capping())


class TestDummyDirectContactWithPole(unittest.TestCase):
    """Test cases for direct contact with pole measurement that triggers dummy capping."""

    def test_direct_contact_with_pole_true_string(self):
        """Dummy should be capped when 'Direct contact with pole' is 'true'."""
        df = pd.DataFrame(
            {
                "Dummy": ["WorldSID-50", None, None],
                "Body region": ["Head", None, None],
                "Criteria": ["Direct contact with pole", "HIC15", "Ares-3ms"],
                "HPL": [None, 500.0, 50.0],
                "LPL": [None, 700.0, 80.0],
                "Capping": [None, None, None],
                "Value": ["true", 600.0, 60.0],
            }
        )
        dummy, _ = Dummy.get_dummy_from_row("WorldSID-50", df, 0)
        self.assertTrue(dummy.get_capping())

    def test_direct_contact_with_pole_true_uppercase(self):
        """Dummy should be capped when 'Direct contact with pole' is 'True'."""
        df = pd.DataFrame(
            {
                "Dummy": ["WorldSID-50", None, None],
                "Body region": ["Head", None, None],
                "Criteria": ["Direct contact with pole", "HIC15", "Ares-3ms"],
                "HPL": [None, 500.0, 50.0],
                "LPL": [None, 700.0, 80.0],
                "Capping": [None, None, None],
                "Value": ["True", 600.0, 60.0],
            }
        )
        dummy, _ = Dummy.get_dummy_from_row("WorldSID-50", df, 0)
        self.assertTrue(dummy.get_capping())

    def test_direct_contact_with_pole_numeric_one(self):
        """Dummy should be capped when 'Direct contact with pole' is 1.0."""
        df = pd.DataFrame(
            {
                "Dummy": ["WorldSID-50", None, None],
                "Body region": ["Head", None, None],
                "Criteria": ["Direct contact with pole", "HIC15", "Ares-3ms"],
                "HPL": [None, 500.0, 50.0],
                "LPL": [None, 700.0, 80.0],
                "Capping": [None, None, None],
                "Value": [1.0, 600.0, 60.0],
            }
        )
        dummy, _ = Dummy.get_dummy_from_row("WorldSID-50", df, 0)
        self.assertTrue(dummy.get_capping())

    def test_direct_contact_with_pole_false_string(self):
        """Dummy should NOT be capped when 'Direct contact with pole' is 'false'."""
        df = pd.DataFrame(
            {
                "Dummy": ["WorldSID-50", None, None],
                "Body region": ["Head", None, None],
                "Criteria": ["Direct contact with pole", "HIC15", "Ares-3ms"],
                "HPL": [None, 500.0, 50.0],
                "LPL": [None, 700.0, 80.0],
                "Capping": [None, None, None],
                "Value": ["false", 600.0, 60.0],
            }
        )
        dummy, _ = Dummy.get_dummy_from_row("WorldSID-50", df, 0)
        self.assertFalse(dummy.get_capping())

    def test_direct_contact_with_pole_numeric_zero(self):
        """Dummy should NOT be capped when 'Direct contact with pole' is 0.0."""
        df = pd.DataFrame(
            {
                "Dummy": ["WorldSID-50", None, None],
                "Body region": ["Head", None, None],
                "Criteria": ["Direct contact with pole", "HIC15", "Ares-3ms"],
                "HPL": [None, 500.0, 50.0],
                "LPL": [None, 700.0, 80.0],
                "Capping": [None, None, None],
                "Value": [0.0, 600.0, 60.0],
            }
        )
        dummy, _ = Dummy.get_dummy_from_row("WorldSID-50", df, 0)
        self.assertFalse(dummy.get_capping())

    def test_direct_contact_capping_zeroes_score(self):
        """When dummy is capped due to direct contact, compute_score should return 0."""
        df = pd.DataFrame(
            {
                "Dummy": ["WorldSID-50", None, None],
                "Body region": ["Head", None, None],
                "Criteria": ["Direct contact with pole", "HIC15", "Ares-3ms"],
                "HPL": [None, 500.0, 50.0],
                "LPL": [None, 700.0, 80.0],
                "Capping": [None, None, None],
                "Value": ["true", 600.0, 60.0],
            }
        )
        dummy, _ = Dummy.get_dummy_from_row("WorldSID-50", df, 0)

        # Set up body region scores
        for body_region in dummy.body_region_list:
            body_region.set_max_score(1.0)
            body_region.set_inspection(0.0)
            body_region.compute_bodyregion_score()
            body_region.compute_score()

        dummy.compute_score()
        self.assertEqual(dummy.get_score(), 0.0)


class TestDummyNameValidation(unittest.TestCase):
    """Test cases for dummy name validation."""

    def test_valid_dummy_names(self):
        """All valid dummy names should be accepted."""
        valid_names = [
            "HIII-50",
            "HIII-05",
            "HIII-95",
            "THOR-50",
            "Q6",
            "Q10",
            "WorldSID-50",
            "Q6-Side",
            "Q10-Side",
            "BioRID-50",
            "HPM",
            "Adult Headform",
            "Child Headform",
            "aPLI",
            "Upper legform",
            "WorldSID-50-Farside",
        ]
        for name in valid_names:
            dummy = Dummy(name=name)
            self.assertEqual(dummy.name, name)

    def test_invalid_dummy_name_raises(self):
        """Invalid dummy names should raise validation error."""
        with self.assertRaises(Exception):
            Dummy(name="InvalidDummy")


class TestDummyScoreValidation(unittest.TestCase):
    """Test cases for score validation and rounding."""

    def test_score_accepts_int(self):
        """Score should accept integer values."""
        dummy = Dummy(name="HIII-50")
        dummy.score = 5
        self.assertEqual(dummy.score, 5.0)

    def test_score_accepts_numpy_int64(self):
        """Score should accept numpy int64 values."""
        dummy = Dummy(name="HIII-50")
        dummy.score = np.int64(10)
        self.assertEqual(dummy.score, 10.0)

    def test_score_accepts_none(self):
        """Score should accept None values."""
        dummy = Dummy(name="HIII-50")
        dummy.score = None
        self.assertIsNone(dummy.score)


class TestDummyComputeCapping(unittest.TestCase):
    """Test cases for compute_capping method."""

    def test_compute_capping_no_criteria_capped(self):
        """compute_capping should be False when no criteria are capped."""
        df = pd.DataFrame(
            {
                "Dummy": ["HIII-50", None],
                "Body region": ["Head & Neck", None],
                "Criteria": ["My,extension", "HIC15"],
                "HPL": [50.0, 500.0],
                "LPL": [100.0, 700.0],
                "Capping": [90.0, None],
                "Value": [60.0, 550.0],  # Values below capping threshold
            }
        )
        dummy, _ = Dummy.get_dummy_from_row("HIII-50", df, 0)
        dummy.compute_capping()
        self.assertFalse(dummy.get_capping())

    def test_compute_capping_criteria_capped(self):
        """compute_capping should be True when any criteria is capped."""
        df = pd.DataFrame(
            {
                "Dummy": ["HIII-50", None],
                "Body region": ["Head & Neck", None],
                "Criteria": ["My,extension", "HIC15"],
                "HPL": [50.0, 500.0],
                "LPL": [100.0, 700.0],
                "Capping": [90.0, None],
                "Value": [95.0, 550.0],  # First value exceeds capping threshold
            }
        )
        dummy, _ = Dummy.get_dummy_from_row("HIII-50", df, 0)
        dummy.compute_capping()
        self.assertTrue(dummy.get_capping())


class TestDummySpecialCases(unittest.TestCase):
    """Test cases for special dummy handling (Q6, Q10, WorldSID-50)."""

    def test_q6_head_ares_score_is_none(self):
        """For Q6 dummy, Head Ares criteria should have None score."""
        df = pd.DataFrame(
            {
                "Dummy": ["Q6", None, None],
                "Body region": ["Head", None, None],
                "Criteria": ["Ares", "HIC15", "Ares-3ms"],
                "HPL": [50.0, 500.0, 50.0],
                "LPL": [80.0, 700.0, 80.0],
                "Capping": [None, None, None],
                "Value": [60.0, 600.0, 55.0],
            }
        )
        dummy, _ = Dummy.get_dummy_from_row("Q6", df, 0)

        head_region = dummy.get_body_region("Head")
        ares_criteria = next(
            (c for c in head_region._criteria if c.name == "Ares"), None
        )
        self.assertIsNotNone(ares_criteria)
        self.assertIsNone(ares_criteria.score)

    def test_q10_head_ares_score_is_none(self):
        """For Q10 dummy, Head Ares criteria should have None score."""
        df = pd.DataFrame(
            {
                "Dummy": ["Q10", None, None],
                "Body region": ["Head", None, None],
                "Criteria": ["Ares", "HIC15", "Ares-3ms"],
                "HPL": [50.0, 500.0, 50.0],
                "LPL": [80.0, 700.0, 80.0],
                "Capping": [None, None, None],
                "Value": [60.0, 600.0, 55.0],
            }
        )
        dummy, _ = Dummy.get_dummy_from_row("Q10", df, 0)

        head_region = dummy.get_body_region("Head")
        ares_criteria = next(
            (c for c in head_region._criteria if c.name == "Ares"), None
        )
        self.assertIsNotNone(ares_criteria)
        self.assertIsNone(ares_criteria.score)

    def test_worldsid50_head_ares_score_is_none(self):
        """For WorldSID-50 dummy, Head Ares criteria should have None score."""
        df = pd.DataFrame(
            {
                "Dummy": ["WorldSID-50", None, None],
                "Body region": ["Head", None, None],
                "Criteria": ["Ares", "HIC15", "Ares-3ms"],
                "HPL": [50.0, 500.0, 50.0],
                "LPL": [80.0, 700.0, 80.0],
                "Capping": [None, None, None],
                "Value": [60.0, 600.0, 55.0],
            }
        )
        dummy, _ = Dummy.get_dummy_from_row("WorldSID-50", df, 0)

        head_region = dummy.get_body_region("Head")
        ares_criteria = next(
            (c for c in head_region._criteria if c.name == "Ares"), None
        )
        self.assertIsNotNone(ares_criteria)
        self.assertIsNone(ares_criteria.score)


class TestQDummyNeckMyExtensionGate(unittest.TestCase):
    """For Q dummies, Neck My,extension follows the head Ares 80 g gate:
    below it the criteria is not assessed at all -- no score, no rating
    colour, no Prediction.Check."""

    @staticmethod
    def _make_q6_df(head_ares_value):
        return pd.DataFrame(
            {
                "Dummy": ["Q6", None, None, None],
                "Body region": ["Head", None, None, "Neck"],
                "Criteria": ["Ares", "HIC15", "Ares-3ms", "My,extension"],
                "HPL": [50.0, 500.0, 50.0, 10.0],
                "LPL": [80.0, 700.0, 80.0, 20.0],
                "Capping": [None, None, None, None],
                "Value": [head_ares_value, 600.0, 55.0, 15.0],
            }
        )

    def test_head_ares_below_80_clears_neck_my_extension_assessment(self):
        dummy, _ = Dummy.get_dummy_from_row("Q6", self._make_q6_df(60.0), 0)

        neck_region = dummy.get_body_region("Neck")
        my_ext = next(
            (c for c in neck_region._criteria if c.name == "My,extension"), None
        )
        self.assertIsNotNone(my_ext)
        self.assertIsNone(my_ext.score)
        self.assertIsNone(my_ext.color)
        self.assertIsNone(my_ext.prediction_result)

        # HIC15 is gated too (via BodyRegion.resolve_dependancies)...
        head_region = dummy.get_body_region("Head")
        hic15 = next((c for c in head_region._criteria if c.name == "HIC15"), None)
        self.assertIsNone(hic15.score)
        self.assertIsNone(hic15.color)
        # ...but Ares-3ms is deliberately un-gated for Q dummies and stays
        # assessed and coloured.
        ares3ms = next((c for c in head_region._criteria if c.name == "Ares-3ms"), None)
        self.assertIsNotNone(ares3ms.score)
        self.assertIsNotNone(ares3ms.color)

    def test_head_ares_above_80_keeps_neck_my_extension_assessment(self):
        dummy, _ = Dummy.get_dummy_from_row("Q6", self._make_q6_df(90.0), 0)

        neck_region = dummy.get_body_region("Neck")
        my_ext = next(
            (c for c in neck_region._criteria if c.name == "My,extension"), None
        )
        self.assertIsNotNone(my_ext)
        self.assertIsNotNone(my_ext.score)
        self.assertIsNotNone(my_ext.color)


class TestDummyMeasurements(unittest.TestCase):
    """Test cases for measurement handling in dummy."""

    def test_measurement_added_to_body_region(self):
        """Measurements should be added to the correct body region."""
        df = pd.DataFrame(
            {
                "Dummy": ["HIII-50", None, None],
                "Body region": ["Chest", None, None],
                "Criteria": ["DAMAGE", "Ares", "HIC15"],
                "HPL": [None, 50.0, 500.0],
                "LPL": [None, 80.0, 700.0],
                "Capping": [None, None, None],
                "Value": [0.45, 60.0, 600.0],
            }
        )
        dummy, _ = Dummy.get_dummy_from_row("HIII-50", df, 0)

        chest_region = dummy.get_body_region("Chest")
        measurements = chest_region.get_measurement_list()

        self.assertEqual(len(measurements), 1)
        self.assertEqual(measurements[0].name, "DAMAGE")
        self.assertEqual(measurements[0].value, 0.45)

    def test_blank_measurement_value_stays_none_not_zero(self):
        """A measurement cell the OEM never filled in must stay unassessed
        (None), not be read as a real measured 0."""
        df = pd.DataFrame(
            {
                "Dummy": ["HIII-50", None, None],
                "Body region": ["Chest", None, None],
                "Criteria": ["DAMAGE", "Ares", "HIC15"],
                "HPL": [None, 50.0, 500.0],
                "LPL": [None, 80.0, 700.0],
                "Capping": [None, None, None],
                "Value": [None, 60.0, 600.0],
            }
        )
        dummy, _ = Dummy.get_dummy_from_row("HIII-50", df, 0)

        chest_region = dummy.get_body_region("Chest")
        measurements = chest_region.get_measurement_list()

        self.assertEqual(len(measurements), 1)
        self.assertIsNone(measurements[0].value)
        self.assertIsNone(measurements[0].modifier)


class TestDummyBodyRegionNotFound(unittest.TestCase):
    """Test cases for body region not found error."""

    def test_get_body_region_not_found_raises(self):
        """get_body_region should raise ValueError for non-existent body region."""
        dummy = Dummy(name="HIII-50")
        dummy.body_region_list = [BodyRegion(name="Head")]

        with self.assertRaises(ValueError) as context:
            dummy.get_body_region("NonExistent")

        self.assertIn("not found", str(context.exception))


class TestDummyRepr(unittest.TestCase):
    """Test cases for dummy string representation."""

    def test_repr(self):
        """__repr__ should return expected format."""
        dummy = Dummy(name="HIII-50")
        repr_str = repr(dummy)
        self.assertIn("Dummy", repr_str)
        self.assertIn("HIII-50", repr_str)


class TestBioRID50CoupledCapping(unittest.TestCase):
    """T-HRC start and T1 acceleration must BOTH exceed capping to cap the BioRID-50 dummy."""

    def _make_biorid_df(self, thrc_value, t1_value, thrc_cap=100.0, t1_cap=10.0):
        """Two coupled capping criteria, no regular criteria."""
        return pd.DataFrame(
            {
                "Dummy": ["BioRID-50", None],
                "Body region": ["Neck", None],
                "Criteria": ["T-HRC start", "T1 acceleration"],
                "HPL": [None, None],
                "LPL": [None, None],
                "Capping": [thrc_cap, t1_cap],
                "Value": [thrc_value, t1_value],
            }
        )

    def test_only_thrc_above_cap_does_not_cap_dummy(self):
        """Only T-HRC start exceeds capping → dummy should NOT be capped."""
        df = self._make_biorid_df(thrc_value=150.0, t1_value=5.0)
        dummy, _ = Dummy.get_dummy_from_row("BioRID-50", df, 0)
        dummy.compute_capping()
        self.assertFalse(dummy.get_capping())

    def test_only_t1_above_cap_does_not_cap_dummy(self):
        """Only T1 acceleration exceeds capping → dummy should NOT be capped."""
        df = self._make_biorid_df(thrc_value=50.0, t1_value=15.0)
        dummy, _ = Dummy.get_dummy_from_row("BioRID-50", df, 0)
        dummy.compute_capping()
        self.assertFalse(dummy.get_capping())

    def test_both_coupled_above_cap_caps_dummy(self):
        """Both T-HRC start and T1 acceleration exceed capping → dummy SHOULD be capped."""
        df = self._make_biorid_df(thrc_value=150.0, t1_value=15.0)
        dummy, _ = Dummy.get_dummy_from_row("BioRID-50", df, 0)
        dummy.compute_capping()
        self.assertTrue(dummy.get_capping())

    def test_neither_above_cap_does_not_cap_dummy(self):
        """Neither coupled criteria exceeds capping → dummy should NOT be capped."""
        df = self._make_biorid_df(thrc_value=50.0, t1_value=5.0)
        dummy, _ = Dummy.get_dummy_from_row("BioRID-50", df, 0)
        dummy.compute_capping()
        self.assertFalse(dummy.get_capping())

    def test_other_criteria_above_cap_caps_dummy(self):
        """A non-coupled criteria alone exceeding capping → dummy SHOULD be capped."""
        df = pd.DataFrame(
            {
                "Dummy": ["BioRID-50", None, None],
                "Body region": ["Neck", None, None],
                "Criteria": ["T-HRC start", "T1 acceleration", "Some other criteria"],
                "HPL": [None, None, None],
                "LPL": [None, None, None],
                "Capping": [100.0, 10.0, 50.0],
                "Value": [50.0, 5.0, 99.0],  # only "Some other criteria" exceeds cap
            }
        )
        dummy, _ = Dummy.get_dummy_from_row("BioRID-50", df, 0)
        dummy.compute_capping()
        self.assertTrue(dummy.get_capping())

    def test_coupled_rule_does_not_affect_other_dummies(self):
        """For non-BioRID-50 dummies, a single T-HRC start above cap is still sufficient."""
        df = pd.DataFrame(
            {
                "Dummy": ["HIII-50", None],
                "Body region": ["Neck", None],
                "Criteria": ["T-HRC start", "T1 acceleration"],
                "HPL": [None, None],
                "LPL": [None, None],
                "Capping": [100.0, 10.0],
                "Value": [150.0, 5.0],  # only T-HRC start exceeds cap
            }
        )
        dummy, _ = Dummy.get_dummy_from_row("HIII-50", df, 0)
        dummy.compute_capping()
        self.assertTrue(dummy.get_capping())


if __name__ == "__main__":
    unittest.main()
