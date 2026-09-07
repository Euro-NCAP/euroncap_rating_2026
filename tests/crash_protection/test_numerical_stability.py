# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Numerical stability tests for the rounding chain defined in the protocol:

  Input values      → 2 decimal places  (e.g. 354.25503  → 354.26)
  Criteria scores   → 2 decimal places  (e.g.  67.66667  →  67.67)
  Body region score → 5 decimal places  (e.g.   0.831333  →  0.83133)
  Dummy score       → 4 decimal places  (e.g.   0.83333  →  0.8333)
  Test scores       → 3 decimal places  (e.g.   8.86667  →   8.867)
  Stage scores      → 0, floored        (e.g.  79.879    →  79)

All rounding is decimal half-up (Excel ROUND semantics) via
euroncap_rating_2026.numeric, and stage floors go through floor_score,
which snaps 1-ulp-below-integer float sums before flooring.

Each class targets one level of the chain; TestRoundingChain covers the
propagation across levels.
"""

import math
import unittest

from euroncap_rating_2026.crash_protection.body_region import BodyRegion
from euroncap_rating_2026.crash_protection.criteria import Criteria, CriteriaType
from euroncap_rating_2026.crash_protection.dummy import Dummy
from euroncap_rating_2026.numeric import floor_score, round_half_up


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_body_region(name, *, bodyregion_score, max_score, inspection=0.0):
    """Return a BodyRegion whose score has been computed from pre-set inputs."""
    br = BodyRegion(name=name)
    br.set_bodyregion_score(bodyregion_score)
    br.set_max_score(max_score)
    br.set_inspection(inspection)
    br.compute_score()
    return br


def _make_dummy(body_regions):
    """Return a Dummy whose score is computed from the given body regions."""
    dummy = Dummy(name="HIII-50")
    dummy.body_region_list = body_regions
    dummy.compute_capping()
    dummy.compute_score()
    return dummy


def _decimal_places(value):
    """Count significant decimal places of a float (up to 10)."""
    s = f"{value:.10f}".rstrip("0")
    if "." not in s:
        return 0
    return len(s.split(".")[1])


# ---------------------------------------------------------------------------
# Criteria level — 2 decimal places
# ---------------------------------------------------------------------------


class TestCriteriaNumericalPrecision(unittest.TestCase):
    """Criteria input values are stored at exactly 2 decimal places."""

    def _criteria(self, value):
        return Criteria(
            name="HIC15",
            hpl=500.0,
            lpl=700.0,
            value=value,
            criteria_type=CriteriaType.CRITERIA,
        )

    def test_5_decimal_input_rounded_to_2(self):
        # Protocol example: 354.25503 → 354.26
        c = self._criteria(354.25503)
        self.assertEqual(c.value, 354.26)

    def test_6_decimal_input_rounded_to_2(self):
        # Protocol example: 67.66667 → 67.67
        c = self._criteria(67.66667)
        self.assertEqual(c.value, 67.67)

    def test_set_value_rounds_to_2(self):
        c = self._criteria(500.0)
        c.set_value(123.456789)
        self.assertEqual(c.value, 123.46)

    def test_value_already_at_2_decimals_is_unchanged(self):
        c = self._criteria(354.26)
        self.assertEqual(c.value, 354.26)

    def test_integer_value_accepted(self):
        c = self._criteria(600.0)
        self.assertEqual(c.value, 600.0)

    def test_value_at_most_2_decimal_places(self):
        for raw in (354.25503, 67.66667, 0.123456, 999.99999):
            with self.subTest(raw=raw):
                c = self._criteria(raw)
                self.assertLessEqual(_decimal_places(c.value), 2)


# ---------------------------------------------------------------------------
# Body region level — 5 decimal places
# ---------------------------------------------------------------------------


class TestBodyRegionNumericalPrecision(unittest.TestCase):
    """Body region scores (bodyregion_score and computed score) are at 5 dp."""

    def setUp(self):
        self.br = BodyRegion(name="Head & Neck")

    # --- bodyregion_score validator ---

    def test_bodyregion_score_5_decimals_unchanged(self):
        # 0.31333 already has 5 dp — stored as-is
        self.br.set_bodyregion_score(0.31333)
        self.assertEqual(self.br.get_bodyregion_score(), 0.31333)

    def test_bodyregion_score_6_decimals_rounded_to_5(self):
        self.br.set_bodyregion_score(0.831333)
        self.assertEqual(self.br.get_bodyregion_score(), 0.83133)

    def test_bodyregion_score_already_at_5_decimals_unchanged(self):
        self.br.set_bodyregion_score(0.31333)
        self.assertEqual(self.br.get_bodyregion_score(), 0.31333)

    def test_bodyregion_score_at_most_5_decimal_places(self):
        for raw in (0.31333, 0.12345678, 1.999995, 3.141592653):
            with self.subTest(raw=raw):
                self.br.set_bodyregion_score(raw)
                self.assertLessEqual(_decimal_places(self.br.get_bodyregion_score()), 5)

    # --- compute_score: (bodyregion_score - inspection) * max_score / 100 ---

    def test_computed_score_6_decimal_max_score_rounded_to_5(self):
        # 100.0 * 3.333333 / 100 = 3.333333 (6 dp) → 3.33333 (5 dp)
        self.br.set_bodyregion_score(100.0)
        self.br.set_max_score(3.333333)
        self.br.set_inspection(0.0)
        self.br.compute_score()
        self.assertEqual(self.br.get_score(), 3.33333)

    def test_computed_score_repeating_fraction_rounded_to_5(self):
        # 5/24 = 0.208333... (repeating) → at 5dp: 0.20833
        # This is the actual Sled & VT Virtual loadcase weight.
        self.br.set_bodyregion_score(100.0)
        self.br.set_max_score(5 / 24)
        self.br.set_inspection(0.0)
        self.br.compute_score()
        self.assertEqual(self.br.get_score(), 0.20833)

    def test_computed_score_with_inspection_rounded_to_5(self):
        # (100.0 - 13.33) * 1.234567 / 100 → round to 5 dp
        self.br.set_bodyregion_score(100.0)
        self.br.set_max_score(1.234567)
        self.br.set_inspection(13.33)
        self.br.compute_score()
        score = self.br.get_score()
        self.assertAlmostEqual(score, round(score, 5), places=10)
        self.assertLessEqual(_decimal_places(score), 5)

    def test_computed_score_at_most_5_decimal_places(self):
        for max_s in (3.333333, 1.234567, 2.718281, 1.414213):
            with self.subTest(max_score=max_s):
                self.br.set_bodyregion_score(100.0)
                self.br.set_max_score(max_s)
                self.br.set_inspection(0.0)
                self.br.compute_score()
                score = self.br.get_score()
                self.assertLessEqual(_decimal_places(score), 5)


# ---------------------------------------------------------------------------
# Dummy level — 4 decimal places, sum of body-region scores (4 dp each)
# ---------------------------------------------------------------------------


class TestDummyNumericalPrecision(unittest.TestCase):
    """Dummy score (sum of 4-dp body region scores) is stored at 4 decimal places."""

    def test_set_score_5_decimals_rounded_to_4(self):
        # Protocol example: 0.83333 → 0.8333
        d = Dummy(name="HIII-50")
        d.set_score(0.83333)
        self.assertEqual(d.get_score(), 0.8333)

    def test_set_score_6_decimals_rounded_to_4(self):
        d = Dummy(name="HIII-50")
        d.set_score(123.456789)
        self.assertEqual(d.get_score(), 123.4568)

    def test_sum_two_body_regions_floating_point_artifact(self):
        # Raw Python: 0.1 + 0.2 == 0.30000000000000004 (IEEE 754 artifact).
        # The dummy must round the sum to 4 dp, yielding exactly 0.3.
        br1 = _make_body_region("BR1", bodyregion_score=100.0, max_score=0.1)
        br2 = _make_body_region("BR2", bodyregion_score=100.0, max_score=0.2)
        self.assertEqual(br1.get_score(), 0.1)
        self.assertEqual(br2.get_score(), 0.2)

        self.assertNotEqual(0.1 + 0.2, 0.3)  # confirm the raw artifact exists

        dummy = _make_dummy([br1, br2])
        self.assertEqual(dummy.get_score(), 0.3)

    def test_sum_three_body_regions_repeating_fraction(self):
        # 1/3 ≈ 0.3333 at 4 dp. Three such body regions sum to 0.9999, not 1.0.
        # This documents the expected protocol behaviour: rounding before summing
        # differs from rounding after summing.
        br1 = _make_body_region("BR1", bodyregion_score=100.0, max_score=0.3333)
        br2 = _make_body_region("BR2", bodyregion_score=100.0, max_score=0.3333)
        br3 = _make_body_region("BR3", bodyregion_score=100.0, max_score=0.3333)
        self.assertEqual(br1.get_score(), 0.3333)
        self.assertEqual(br2.get_score(), 0.3333)
        self.assertEqual(br3.get_score(), 0.3333)

        dummy = _make_dummy([br1, br2, br3])
        self.assertEqual(dummy.get_score(), 0.9999)

    def test_sum_body_regions_with_6_decimal_max_scores(self):
        # max_score values with 6 decimal places produce 5-dp body region scores;
        # their sum is then rounded to 4 dp at the dummy level.
        br1 = _make_body_region("BR1", bodyregion_score=100.0, max_score=1.333333)
        br2 = _make_body_region("BR2", bodyregion_score=100.0, max_score=2.666667)
        # br1.score = round(1.333333, 5) = 1.33333
        # br2.score = round(2.666667, 5) = 2.66667
        self.assertEqual(br1.get_score(), 1.33333)
        self.assertEqual(br2.get_score(), 2.66667)

        dummy = _make_dummy([br1, br2])
        # 1.33333 + 2.66667 = 4.0 → round(4.0, 4) = 4.0
        self.assertEqual(dummy.get_score(), 4.0)

    def test_dummy_score_at_most_4_decimal_places(self):
        # Regardless of how many body regions or what their max_scores are,
        # the final dummy score must not exceed 4 decimal places.
        max_scores = [1.33333, 0.66667, 2.11111, 0.88889]
        body_regions = [
            _make_body_region(f"BR{i}", bodyregion_score=100.0, max_score=ms)
            for i, ms in enumerate(max_scores)
        ]
        dummy = _make_dummy(body_regions)
        score = dummy.get_score()
        self.assertLessEqual(_decimal_places(score), 4)


# ---------------------------------------------------------------------------
# Test / domain score level — 3 decimal places
# ---------------------------------------------------------------------------


class TestDomainScoreNumericalPrecision(unittest.TestCase):
    """
    Domain (test) scores are accumulated as round_half_up(float(total), 3).
    These tests verify the arithmetic directly, mirroring data_loader.py.
    """

    def test_domain_score_rounded_to_3(self):
        # Protocol example: 8.86667 → 8.867
        self.assertEqual(round_half_up(float(8.86667), 3), 8.867)

    def test_domain_score_sum_one_ulp_below_integer(self):
        # Perfectly ordinary 3-dp scores can sum 1 ulp below the integer;
        # the half-up guard must snap it back.
        total = 8.359 + 3.864 + 0.777  # 12.999999999999998
        self.assertEqual(round_half_up(float(total), 3), 13.0)

    def test_domain_score_sum_of_4_decimal_dummy_scores(self):
        # Four dummy scores at 4 dp summed and rounded to 3 dp.
        dummy_scores = [1.2346, 2.6667, 3.1416, 1.9271]
        total = round_half_up(float(sum(dummy_scores)), 3)
        self.assertEqual(total, round_half_up(total, 3))
        self.assertLessEqual(_decimal_places(total), 3)

    def test_domain_score_floating_point_sum_rounded_to_3(self):
        # 0.1 + 0.2 artifact at the dummy level carries into domain score.
        dummy_scores = [0.3, 0.1, 0.2]  # already rounded to 4 dp by dummy level
        total = round_half_up(float(sum(dummy_scores)), 3)
        self.assertEqual(total, 0.6)

    def test_domain_score_at_most_3_decimal_places(self):
        for raw_scores in (
            [1.3333, 2.6667],
            [0.1234, 5.6789, 3.1416],
            [8.8667, 0.0001],
        ):
            with self.subTest(scores=raw_scores):
                total = round_half_up(float(sum(raw_scores)), 3)
                self.assertLessEqual(_decimal_places(total), 3)


# ---------------------------------------------------------------------------
# Stage score level — floor (never rounds up)
# ---------------------------------------------------------------------------


class TestStageScoreFloor(unittest.TestCase):
    """Stage/overall scores use floor_score — fractions are always truncated,
    but 1-ulp-below-integer float artifacts are snapped before flooring."""

    def test_floor_protocol_example(self):
        # Protocol example: 79.879 → 79
        self.assertEqual(floor_score(79.879), 79)

    def test_floor_near_next_integer(self):
        # 79.999 must be 79, not 80
        self.assertEqual(floor_score(79.999), 79)

    def test_floor_on_exact_integer(self):
        self.assertEqual(floor_score(79.0), 79)

    def test_floor_differs_from_round_for_large_fraction(self):
        # round(4.9) == 5, but floor(4.9) == 4
        self.assertEqual(floor_score(4.9), 4)
        self.assertNotEqual(floor_score(4.9), round(4.9))

    def test_floor_on_sum_of_domain_scores(self):
        # Simulate: domain scores at 3 dp summing to a non-integer
        domain_scores = [26.867, 31.334, 21.678]
        overall = floor_score(sum(domain_scores))  # sum = 79.879
        self.assertEqual(overall, 79)

    def test_floor_on_sum_of_3_decimal_domain_scores_various(self):
        cases = [
            ([10.333, 20.667, 5.001], 36),  # sum = 36.001 → 36
            ([10.001, 20.001, 5.001], 35),  # sum = 35.003 → 35
            ([9.999, 9.999, 9.999], 29),  # sum = 29.997 → 29
        ]
        for domain_scores, expected in cases:
            with self.subTest(domain_scores=domain_scores):
                self.assertEqual(floor_score(sum(domain_scores)), expected)

    def test_floor_on_sum_one_ulp_below_integer(self):
        # math.floor silently eats a whole point on these sums.
        # NOTE: built with chained additions — since Python 3.12, sum() uses
        # compensated summation and would hide the artifact these document.
        cases = [
            (8.359 + 3.864 + 0.777, 13),  # 12.999999999999998
            (0.732 + 0.834 + 0.434, 2),  # 1.9999999999999998
        ]
        for raw_sum, expected in cases:
            with self.subTest(raw_sum=raw_sum):
                self.assertEqual(math.floor(raw_sum), expected - 1)
                self.assertEqual(floor_score(raw_sum), expected)


# ---------------------------------------------------------------------------
# End-to-end chain: criteria(2dp) → body_region(5dp) → dummy(4dp)
# ---------------------------------------------------------------------------


class TestRoundingChain(unittest.TestCase):
    """
    Integration: verifies precision is enforced at every level as values
    propagate from criteria input through to dummy score.
    """

    def setUp(self):
        # HIC15 hpl=500, lpl=700, value=600 → orange → score 40 (known from existing tests)
        self.criteria = Criteria(
            name="HIC15",
            hpl=500.0,
            lpl=700.0,
            value=600.0,
            criteria_type=CriteriaType.CRITERIA,
        )
        self.br = BodyRegion(name="Head & Neck")
        self.br.set_criteria_list([self.criteria])

    def test_criteria_value_precision_entering_chain(self):
        # Many-decimal input is truncated to 2 dp before any downstream computation.
        self.br.set_criteria_value("HIC15", 599.99999)
        self.assertEqual(self.br.get_criteria_value("HIC15"), 600.0)

    def test_bodyregion_score_5_decimal_after_compute(self):
        self.br.set_criteria_value("HIC15", 600.0)  # → score 40
        self.br.compute_bodyregion_score()
        score = self.br.get_bodyregion_score()
        self.assertAlmostEqual(score, round(score, 5), places=10)
        self.assertLessEqual(_decimal_places(score), 5)

    def test_body_region_computed_score_5_decimal_non_round_max(self):
        # 40 * 3.141592 / 100 = 1.2566368 → rounded to 5 dp = 1.25664
        self.br.set_criteria_value("HIC15", 600.0)  # → bodyregion_score 40
        self.br.compute_bodyregion_score()
        self.br.set_max_score(3.141592)
        self.br.set_inspection(0.0)
        self.br.compute_score()
        self.assertEqual(self.br.get_score(), 1.25664)

    def test_dummy_accumulates_body_region_scores_at_4dp(self):
        # Two body regions (scores at 5 dp); verify the dummy sum is rounded to 4 dp.
        br1 = _make_body_region("Head & Neck", bodyregion_score=100.0, max_score=0.1)
        br2 = _make_body_region("Chest", bodyregion_score=100.0, max_score=0.2)
        # br1.score = 0.1, br2.score = 0.2
        # raw 0.1 + 0.2 = 0.30000000000000004 → round to 4 dp = 0.3
        dummy = _make_dummy([br1, br2])
        self.assertEqual(dummy.get_score(), 0.3)
        self.assertLessEqual(_decimal_places(dummy.get_score()), 4)

    def test_full_chain_precision_at_each_level(self):
        # Use a max_score with 5 decimal places and verify precision at every level.
        self.br.set_criteria_value("HIC15", 354.25503)
        self.assertLessEqual(_decimal_places(self.br.get_criteria_value("HIC15")), 2)

        self.br.compute_bodyregion_score()
        br_score = self.br.get_bodyregion_score()
        self.assertLessEqual(
            _decimal_places(br_score) if br_score is not None else 0, 5
        )

        self.br.set_max_score(3.33333)
        self.br.set_inspection(0.0)
        self.br.compute_score()
        computed = self.br.get_score()
        self.assertLessEqual(
            _decimal_places(computed) if computed is not None else 0, 5
        )

        dummy = _make_dummy([self.br])
        dummy_score = dummy.get_score()
        self.assertLessEqual(
            _decimal_places(dummy_score) if dummy_score is not None else 0, 4
        )

        domain_total = round_half_up(float(dummy_score), 3)
        self.assertLessEqual(_decimal_places(domain_total), 3)

        stage_score = floor_score(domain_total)
        self.assertEqual(stage_score, int(stage_score))


if __name__ == "__main__":
    unittest.main()
