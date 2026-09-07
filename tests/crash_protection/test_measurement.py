# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import unittest

from euroncap_rating_2026.crash_protection.measurement import Measurement


class TestMeasurementColour(unittest.TestCase):
    """
    Verify that Measurement.color returns 'green' when no penalty is applied
    (modifier == 0.0) and 'red' when any penalty is applied (modifier > 0.0).
    """

    # -- Shoulder belt load  (threshold 6.0 kN) --------------------------------

    def test_shoulder_belt_load_below_threshold_is_green(self):
        self.assertEqual(
            Measurement(name="Shoulder belt load", value=5.9).color, "green"
        )

    def test_shoulder_belt_load_at_threshold_is_red(self):
        self.assertEqual(Measurement(name="Shoulder belt load", value=6.0).color, "red")

    def test_shoulder_belt_load_above_threshold_is_red(self):
        self.assertEqual(Measurement(name="Shoulder belt load", value=6.1).color, "red")

    # -- Pedal blocking  (threshold 50.0 mm) -----------------------------------

    def test_pedal_blocking_below_threshold_is_green(self):
        self.assertEqual(Measurement(name="Pedal blocking", value=49.0).color, "green")

    def test_pedal_blocking_at_threshold_is_red(self):
        self.assertEqual(Measurement(name="Pedal blocking", value=50.0).color, "red")

    def test_pedal_blocking_above_threshold_is_red(self):
        self.assertEqual(Measurement(name="Pedal blocking", value=51.0).color, "red")

    # -- Pedal displacement - vertical  (threshold 72.0 mm) -------------------

    def test_pedal_displacement_vertical_below_threshold_is_green(self):
        self.assertEqual(
            Measurement(name="Pedal displacement - vertical", value=71.0).color, "green"
        )

    def test_pedal_displacement_vertical_at_threshold_is_red(self):
        self.assertEqual(
            Measurement(name="Pedal displacement - vertical", value=72.0).color, "red"
        )

    def test_pedal_displacement_vertical_above_threshold_is_red(self):
        self.assertEqual(
            Measurement(name="Pedal displacement - vertical", value=73.0).color, "red"
        )

    # -- Pedal displacement - rearward  (thresholds 100, 200 mm) --------------

    def test_pedal_displacement_rearward_below_lower_threshold_is_green(self):
        self.assertEqual(
            Measurement(name="Pedal displacement - rearward", value=99.0).color, "green"
        )

    def test_pedal_displacement_rearward_at_lower_threshold_is_red(self):
        self.assertEqual(
            Measurement(name="Pedal displacement - rearward", value=100.0).color, "red"
        )

    def test_pedal_displacement_rearward_between_thresholds_is_red(self):
        self.assertEqual(
            Measurement(name="Pedal displacement - rearward", value=150.0).color, "red"
        )

    def test_pedal_displacement_rearward_at_upper_threshold_is_red(self):
        self.assertEqual(
            Measurement(name="Pedal displacement - rearward", value=200.0).color, "red"
        )

    # -- Shoulder load  (threshold 3.0 kN) ------------------------------------

    def test_shoulder_load_below_threshold_is_green(self):
        self.assertEqual(Measurement(name="Shoulder load", value=2.9).color, "green")

    def test_shoulder_load_at_threshold_is_red(self):
        self.assertEqual(Measurement(name="Shoulder load", value=3.0).color, "red")

    def test_shoulder_load_above_threshold_is_red(self):
        self.assertEqual(Measurement(name="Shoulder load", value=3.1).color, "red")

    # -- Viscous Criterion  (threshold 1.0 m/s) --------------------------------

    def test_viscous_criterion_below_threshold_is_green(self):
        self.assertEqual(
            Measurement(name="Viscous Criterion", value=0.9).color, "green"
        )

    def test_viscous_criterion_at_threshold_is_red(self):
        self.assertEqual(Measurement(name="Viscous Criterion", value=1.0).color, "red")

    def test_viscous_criterion_above_threshold_is_red(self):
        self.assertEqual(Measurement(name="Viscous Criterion", value=1.1).color, "red")

    # -- Effective height modifier  (threshold 790.0 mm, inverted) -----------
    # modifier=100 (penalty) when value < 790; modifier=0 when value >= 790

    def test_effective_height_modifier_below_threshold_is_red(self):
        self.assertEqual(
            Measurement(name="Effective height modifier", value=789.0).color, "red"
        )

    def test_effective_height_modifier_at_threshold_is_green(self):
        self.assertEqual(
            Measurement(name="Effective height modifier", value=790.0).color, "green"
        )

    def test_effective_height_modifier_above_threshold_is_green(self):
        self.assertEqual(
            Measurement(name="Effective height modifier", value=791.0).color, "green"
        )

    # -- DAMAGE  (thresholds 0.42, 0.47) --------------------------------------

    def test_damage_below_lower_threshold_is_green(self):
        self.assertEqual(Measurement(name="DAMAGE", value=0.41).color, "green")

    def test_damage_at_lower_threshold_is_red(self):
        self.assertEqual(Measurement(name="DAMAGE", value=0.42).color, "red")

    def test_damage_between_thresholds_is_red(self):
        self.assertEqual(Measurement(name="DAMAGE", value=0.45).color, "red")

    def test_damage_at_upper_threshold_is_red(self):
        self.assertEqual(Measurement(name="DAMAGE", value=0.47).color, "red")

    # -- Excursion  (thresholds 450, 550 mm) -----------------------------------

    def test_excursion_below_lower_threshold_is_green(self):
        self.assertEqual(Measurement(name="Excursion", value=449.0).color, "green")

    def test_excursion_at_lower_threshold_is_red(self):
        self.assertEqual(Measurement(name="Excursion", value=450.0).color, "red")

    def test_excursion_between_thresholds_is_red(self):
        self.assertEqual(Measurement(name="Excursion", value=500.0).color, "red")

    def test_excursion_at_upper_threshold_is_red(self):
        self.assertEqual(Measurement(name="Excursion", value=550.0).color, "red")

    # -- Fpubic symphysis  (threshold 2.8 kN) ----------------------------------

    def test_fpubic_symphysis_below_threshold_is_green(self):
        self.assertEqual(
            Measurement(name="Fpubic symphysis", value=2.79).color, "green"
        )

    def test_fpubic_symphysis_at_threshold_is_red(self):
        self.assertEqual(Measurement(name="Fpubic symphysis", value=2.8).color, "red")

    def test_fpubic_symphysis_above_threshold_is_red(self):
        self.assertEqual(Measurement(name="Fpubic symphysis", value=2.81).color, "red")

    # -- Fy lumbar  (threshold 2.0 kN) ----------------------------------------

    def test_fy_lumbar_below_threshold_is_green(self):
        self.assertEqual(Measurement(name="Fy lumbar", value=1.99).color, "green")

    def test_fy_lumbar_at_threshold_is_red(self):
        self.assertEqual(Measurement(name="Fy lumbar", value=2.0).color, "red")

    def test_fy_lumbar_above_threshold_is_red(self):
        self.assertEqual(Measurement(name="Fy lumbar", value=2.01).color, "red")

    # -- Fz lumbar  (threshold 3.5 kN) ----------------------------------------

    def test_fz_lumbar_below_threshold_is_green(self):
        self.assertEqual(Measurement(name="Fz lumbar", value=3.49).color, "green")

    def test_fz_lumbar_at_threshold_is_red(self):
        self.assertEqual(Measurement(name="Fz lumbar", value=3.5).color, "red")

    def test_fz_lumbar_above_threshold_is_red(self):
        self.assertEqual(Measurement(name="Fz lumbar", value=3.51).color, "red")

    # -- Mx lumbar  (threshold 120.0 Nm) --------------------------------------

    def test_mx_lumbar_below_threshold_is_green(self):
        self.assertEqual(Measurement(name="Mx lumbar", value=119.9).color, "green")

    def test_mx_lumbar_at_threshold_is_red(self):
        self.assertEqual(Measurement(name="Mx lumbar", value=120.0).color, "red")

    def test_mx_lumbar_above_threshold_is_red(self):
        self.assertEqual(Measurement(name="Mx lumbar", value=120.1).color, "red")


class TestMeasurementBlankValue(unittest.TestCase):
    """A blank (unassessed) Measurement must not be scored as a real
    measurement of 0 -- neither silently passed (most measurements) nor,
    worse, forced to the maximum penalty (Effective height modifier, whose
    band is inverted: modifier=100.0 when value < 790.0, so an unfilled
    cell read as 0.0 used to score the harshest possible result)."""

    def test_effective_height_modifier_blank_is_not_max_penalty(self):
        self.assertIsNone(
            Measurement(name="Effective height modifier", value=None).modifier
        )

    def test_damage_blank_modifier_is_none_not_zero(self):
        self.assertIsNone(Measurement(name="DAMAGE", value=None).modifier)

    def test_shoulder_belt_load_blank_modifier_is_none(self):
        self.assertIsNone(Measurement(name="Shoulder belt load", value=None).modifier)

    def test_blank_value_color_is_none_not_red(self):
        self.assertIsNone(
            Measurement(name="Effective height modifier", value=None).color
        )


if __name__ == "__main__":
    unittest.main()
