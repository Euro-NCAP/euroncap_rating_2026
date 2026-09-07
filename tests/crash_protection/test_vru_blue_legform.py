# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Full-coverage tests for the legform blue-point handling.

Features under test
-------------------
1. check_legform_blue_symmetry   – preprocess consistency check
2. select_blue_pattern           – SA→ST coin-flip promotion (rows 0/1/2)
3. _parse_test_point             – "(row, col)" string parser
4. _resolve_legform_blue_values  – value inference for T / ST / SA criteria
5. VruTestData.from_sheet_dict   – integration: warnings stored, values resolved
6. VruTestData.compute_vru_bodyregion_score – "Score successfully computed" message
7. report_writer Value-column styling – SA rows stay white, T/ST rows get grey
"""

import math
import unittest
from unittest.mock import patch

import openpyxl
import pandas as pd
from openpyxl.styles import PatternFill

from euroncap_rating_2026 import common
from euroncap_rating_2026.crash_protection import vru_processing
from euroncap_rating_2026.crash_protection.body_region import BodyRegion
from euroncap_rating_2026.crash_protection.criteria import Criteria, CriteriaType
from euroncap_rating_2026.crash_protection.dummy import Dummy
from euroncap_rating_2026.crash_protection.load_case import LoadCase
from euroncap_rating_2026.crash_protection.seat import Seat
from euroncap_rating_2026.crash_protection.vru_processing import (
    LEGFORMS_START_ROW_INDEX,
    LegformTestPoint,
    VruPredictionColor,
    VruTestData,
    _parse_test_point,
    _resolve_legform_blue_values,
    check_legform_blue_symmetry,
    get_legform_matrix,
    process_legform_criteria,
    process_legform_vru_loadcases,
    select_blue_pattern,
)

# ---------------------------------------------------------------------------
# Colour aliases for concise matrix construction
# ---------------------------------------------------------------------------
G = VruPredictionColor.GREY
T = VruPredictionColor.BLUE_T
ST = VruPredictionColor.BLUE_ST
SA = VruPredictionColor.BLUE_SA
GR = VruPredictionColor.GREEN
YL = VruPredictionColor.YELLOW


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------


def _make_matrix(rows):
    """Wrap a list-of-lists into a dict-accessible matrix (list form)."""
    return [list(row) for row in rows]


def _make_criteria(name, prediction, value, row=0, col=3, hpl=5.0, lpl=6.0):
    """Build a Criteria with an optional NaN value (simulates blank Excel cell)."""
    c = Criteria(name=name, hpl=hpl, lpl=lpl, criteria_type=CriteriaType.CRITERIA)
    c.test_point = common.format_test_point(row, col)
    c.set_prediction(prediction)
    if math.isnan(value):
        # Bypass validator to simulate a blank cell read from Excel
        c.__dict__["value"] = float("nan")
    else:
        c.set_value(value)
    return c


def _make_loadcase(criteria_list, loadcase_name="Upper Leg", br_name="Pelvis"):
    """Wrap criteria in the minimal LoadCase hierarchy."""
    br = BodyRegion(name=br_name)
    br._criteria = list(criteria_list)
    dummy = Dummy(name="Upper legform", body_region_list=[br])
    seat = Seat(name="Driver", dummy=dummy)
    return LoadCase(name=loadcase_name, seats=[seat])


# ===========================================================================
# 1 – _parse_test_point
# ===========================================================================


class TestParseTestPoint(unittest.TestCase):
    def test_standard_positive_coords(self):
        self.assertEqual(_parse_test_point("(1, 3)"), (1, 3))

    def test_negative_col(self):
        self.assertEqual(_parse_test_point("(0, -5)"), (0, -5))

    def test_zero_zero(self):
        self.assertEqual(_parse_test_point("(0, 0)"), (0, 0))

    def test_large_values(self):
        self.assertEqual(_parse_test_point("(18, 10)"), (18, 10))

    def test_none_input(self):
        self.assertEqual(_parse_test_point(None), (None, None))

    def test_empty_string(self):
        self.assertEqual(_parse_test_point(""), (None, None))

    def test_malformed_no_comma(self):
        self.assertEqual(_parse_test_point("bad"), (None, None))

    def test_too_many_parts(self):
        self.assertEqual(_parse_test_point("(1, 2, 3)"), (None, None))

    def test_non_integer_parts(self):
        self.assertEqual(_parse_test_point("(a, b)"), (None, None))

    def test_extra_whitespace_inside(self):
        self.assertEqual(_parse_test_point("( 2 ,  4 )"), (2, 4))


# ===========================================================================
# 2 – check_legform_blue_symmetry
# ===========================================================================


class TestCheckLegformBlueSymmetry(unittest.TestCase):
    """
    5-column matrix: col_center=2, col[j] = 2-j
      j=0 col= 2  ↔  j=4 col=-2
      j=1 col= 1  ↔  j=3 col=-1
      j=2 col= 0        (self-symmetric)
    """

    # ------------------------------------------------------------------
    # Valid matrices – must NOT raise
    # ------------------------------------------------------------------

    def test_all_grey_passes(self):
        m = _make_matrix([[G, G, G, G, G]] * 3)
        check_legform_blue_symmetry(m)  # should not raise

    def test_t_symmetric_t_passes(self):
        m = _make_matrix([[T, G, G, G, T], [G, G, G, G, G], [G, G, G, G, G]])
        check_legform_blue_symmetry(m)

    def test_t_symmetric_st_passes(self):
        # T/ST at mirror positions both count as valid
        m = _make_matrix([[T, G, G, G, ST], [G, G, G, G, G], [G, G, G, G, G]])
        check_legform_blue_symmetry(m)

    def test_st_symmetric_st_passes(self):
        m = _make_matrix([[ST, G, G, G, ST], [G, G, G, G, G], [G, G, G, G, G]])
        check_legform_blue_symmetry(m)

    def test_sa_symmetric_sa_passes(self):
        m = _make_matrix([[SA, G, G, G, SA], [G, G, G, G, G], [G, G, G, G, G]])
        check_legform_blue_symmetry(m)

    def test_t_at_center_col0_passes(self):
        # j=2 → col=0; symmetric maps to j_sym=2 (self) which IS T → valid
        m = _make_matrix([[G, G, T, G, G], [G, G, G, G, G], [G, G, G, G, G]])
        check_legform_blue_symmetry(m)

    def test_st_at_center_col0_passes(self):
        m = _make_matrix([[G, G, ST, G, G], [G, G, G, G, G], [G, G, G, G, G]])
        check_legform_blue_symmetry(m)

    def test_multiple_rows_all_valid(self):
        row = [SA, T, G, ST, SA]
        m = _make_matrix([row, row, row])
        check_legform_blue_symmetry(m)

    def test_full_symmetric_mixed_row(self):
        # j0=SA, j1=T, j2=grey, j3=ST, j4=SA
        m = _make_matrix([[SA, T, G, ST, SA]])
        check_legform_blue_symmetry(m)

    # ------------------------------------------------------------------
    # Invalid matrices – must raise ValueError
    # ------------------------------------------------------------------

    def test_t_without_symmetric_raises(self):
        # T at j=0 (col=2), symmetric j=4 is GREY
        m = _make_matrix([[T, G, G, G, G]])
        with self.assertRaises(ValueError) as ctx:
            check_legform_blue_symmetry(m)
        self.assertIn("col=2", str(ctx.exception))

    def test_t_with_sa_symmetric_raises(self):
        m = _make_matrix([[T, G, G, G, SA]])
        with self.assertRaises(ValueError):
            check_legform_blue_symmetry(m)

    def test_st_without_symmetric_raises(self):
        m = _make_matrix([[ST, G, G, G, G]])
        with self.assertRaises(ValueError):
            check_legform_blue_symmetry(m)

    def test_sa_at_col0_is_allowed(self):
        # SA at col=0 maps to j_sym=j (self-symmetric). Its own color is SA == SA,
        # so the check passes. The code intentionally allows this case.
        m = _make_matrix([[G, G, SA, G, G]])
        check_legform_blue_symmetry(m)  # must NOT raise

    def test_sa_without_symmetric_raises(self):
        # SA at j=0 (col=2), symmetric j=4 is GREY
        m = _make_matrix([[SA, G, G, G, G]])
        with self.assertRaises(ValueError):
            check_legform_blue_symmetry(m)

    def test_sa_with_t_symmetric_raises(self):
        # SA at j=0, symmetric j=4 is T (not SA)
        m = _make_matrix([[SA, G, G, G, T]])
        with self.assertRaises(ValueError) as ctx:
            check_legform_blue_symmetry(m)
        self.assertIn("expected SA", str(ctx.exception))

    def test_invalid_in_row1_raises(self):
        # Row 0 valid, row 1 invalid
        valid_row = [SA, G, G, G, SA]
        bad_row = [T, G, G, G, G]
        m = _make_matrix([valid_row, bad_row, [G, G, G, G, G]])
        with self.assertRaises(ValueError):
            check_legform_blue_symmetry(m)

    def test_error_message_contains_row_and_col(self):
        m = _make_matrix([[T, G, G, G, G]])
        with self.assertRaises(ValueError) as ctx:
            check_legform_blue_symmetry(m)
        msg = str(ctx.exception)
        self.assertIn("row=0", msg)
        self.assertIn("j=0", msg)


# ===========================================================================
# 3 – select_blue_pattern SA→ST promotion
# ===========================================================================


class TestSelectBluePatternSAPromotion(unittest.TestCase):
    """
    5-column rows.  valid_cols are all 5 indices when no grey is present.
    coin=0 → selected indices 0, 2, 4  (even in enumerate)
    coin=1 → selected indices 1, 3
    """

    def _all_sa_matrix(self, nrows=1, ncols=5):
        return _make_matrix([[SA] * ncols] * nrows)

    # ------------------------------------------------------------------
    # coin=0 promotes indices 0, 2, 4
    # ------------------------------------------------------------------

    def test_coin0_selected_cols_become_st(self):
        m = self._all_sa_matrix()
        with patch("random.randint", return_value=0):
            result = select_blue_pattern(0, m)
        # All 5 cells must always be returned (greenfield row: every SA is orphaned,
        # so the flip still applies, but no cell may be dropped from the result).
        result_cols = {col for _, col, _ in result}
        self.assertEqual(result_cols, {0, 1, 2, 3, 4})
        for col in (0, 2, 4):
            self.assertEqual(m[0][col], ST)

    def test_coin0_non_selected_cols_stay_sa(self):
        m = self._all_sa_matrix()
        with patch("random.randint", return_value=0):
            select_blue_pattern(0, m)
        for col in (1, 3):
            self.assertEqual(m[0][col], SA)

    def test_coin0_result_carries_st_color(self):
        m = self._all_sa_matrix()
        with patch("random.randint", return_value=0):
            result = select_blue_pattern(0, m)
        result_colors = {col: color for _, col, color in result}
        for col in (0, 2, 4):
            self.assertEqual(result_colors[col], ST)
        for col in (1, 3):
            self.assertEqual(result_colors[col], SA)

    # ------------------------------------------------------------------
    # coin=1 promotes indices 1, 3
    # ------------------------------------------------------------------

    def test_coin1_selected_cols_become_st(self):
        m = self._all_sa_matrix()
        with patch("random.randint", return_value=1):
            result = select_blue_pattern(0, m)
        result_cols = {col for _, col, _ in result}
        self.assertEqual(result_cols, {0, 1, 2, 3, 4})
        for col in (1, 3):
            self.assertEqual(m[0][col], ST)

    def test_coin1_non_selected_cols_stay_sa(self):
        m = self._all_sa_matrix()
        with patch("random.randint", return_value=1):
            select_blue_pattern(0, m)
        for col in (0, 2, 4):
            self.assertEqual(m[0][col], SA)

    # ------------------------------------------------------------------
    # T/ST cells at selected positions are NOT demoted
    # ------------------------------------------------------------------

    def test_t_at_selected_col_stays_t(self):
        # j=0 is T, rest SA – coin=0 selects j=0
        row = [T, SA, SA, SA, SA]
        m = _make_matrix([row])
        with patch("random.randint", return_value=0):
            select_blue_pattern(0, m)
        self.assertEqual(m[0][0], T)

    def test_st_at_selected_col_stays_st(self):
        row = [ST, SA, SA, SA, SA]
        m = _make_matrix([row])
        with patch("random.randint", return_value=0):
            select_blue_pattern(0, m)
        self.assertEqual(m[0][0], ST)

    # ------------------------------------------------------------------
    # Row 2 (Tibia/Knee) gets promoted when row_idx=1
    # ------------------------------------------------------------------

    def test_row2_also_promoted_when_row_idx_is_1(self):
        m = _make_matrix([[G] * 5, [SA] * 5, [SA] * 5])
        with patch("random.randint", return_value=0):
            select_blue_pattern(1, m)
        for col in (0, 2, 4):
            self.assertEqual(m[2][col], ST, f"row2 col={col} should be ST")
        for col in (1, 3):
            self.assertEqual(m[2][col], SA, f"row2 col={col} should stay SA")

    def test_row2_not_promoted_when_row_idx_is_0(self):
        m = _make_matrix([[SA] * 5, [SA] * 5, [SA] * 5])
        with patch("random.randint", return_value=0):
            select_blue_pattern(0, m)
        # Row 2 must not be touched when processing row 0
        for col in range(5):
            self.assertEqual(
                m[2][col], SA, f"row2 col={col} must not be promoted by row_idx=0"
            )

    # ------------------------------------------------------------------
    # Grey cells split a contiguous blue range
    # ------------------------------------------------------------------

    def test_grey_cells_inside_range_are_skipped(self):
        # j: 0=SA, 1=G, 2=SA, 3=G, 4=SA  →  valid_cols=[0,2,4]
        # coin=0: enumerate([0,2,4]) selects idx%2==0 → indices 0,2 → cols {0,4}
        row = [SA, G, SA, G, SA]
        m = _make_matrix([row])
        with patch("random.randint", return_value=0):
            result = select_blue_pattern(0, m)
        # All 3 blue cells must be returned (greenfield, all orphaned); only cols
        # 0 and 4 are promoted to ST by the coin, col 2 stays SA.
        result_cols = {col for _, col, _ in result}
        self.assertEqual(result_cols, {0, 2, 4})
        self.assertEqual(m[0][0], ST)
        self.assertEqual(m[0][4], ST)
        self.assertEqual(m[0][2], SA)

    # ------------------------------------------------------------------
    # Non-all-blue row: SA→ST promotion path is never triggered
    # ------------------------------------------------------------------

    def test_mixed_green_sa_row_uses_else_branch_no_promotion(self):
        # GREEN at j=0 breaks the all_blue condition
        row = [GR, SA, SA, SA, SA]
        m = _make_matrix([row])
        select_blue_pattern(0, m)
        # SA cells must remain SA (else branch never promotes)
        for j in (1, 2, 3, 4):
            self.assertEqual(m[0][j], SA)


# ===========================================================================
# 3b – select_blue_pattern: no-orphan rows must never flip and never drop cells
#
# Root-cause regression guard: select_blue_pattern used to run the coin flip
# unconditionally on any all-blue row and return only the coin-selected half,
# silently dropping any T/ST cell the coin didn't land on. These tests assert
# random.randint is not even called when no SA is orphaned, and that every
# blue cell in the row is always present in the return value.
# ===========================================================================


class TestSelectBluePatternNoOrphanPreservesAll(unittest.TestCase):

    def test_all_st_row_no_flip_all_kept(self):
        # From the real fixture allST_clean.xlsx (legform row 1, Femur): [ST]*21.
        row = [ST] * 21
        m = _make_matrix([row])
        with patch("random.randint") as mock_randint:
            result = select_blue_pattern(0, m)
        mock_randint.assert_not_called()
        self.assertEqual(len(result), 21)
        self.assertEqual(
            [color for _, _, color in sorted(result, key=lambda t: t[1])], row
        )

    def test_oem_alternating_plan_no_flip_reproduced_exactly(self):
        row = [T, SA, T, SA, T, SA, T]
        m = _make_matrix([row])
        with patch("random.randint") as mock_randint:
            result = select_blue_pattern(0, m)
        mock_randint.assert_not_called()
        self.assertEqual(len(result), 7)
        colors_by_col = {col: color for _, col, color in result}
        self.assertEqual([colors_by_col[j] for j in range(7)], row)

    def test_sa_pair_both_flanked_no_flip(self):
        # Both SAs already have a tested neighbor (idx0=T covers idx1, idx3=T covers idx2).
        row = [T, SA, SA, T]
        m = _make_matrix([row])
        with patch("random.randint") as mock_randint:
            result = select_blue_pattern(0, m)
        mock_randint.assert_not_called()
        colors_by_col = {col: color for _, col, color in result}
        self.assertEqual([colors_by_col[j] for j in range(4)], row)

    def test_no_orphaned_sa_in_longer_row_no_flip(self):
        row = [T, SA, T, SA, SA, T, SA, T]
        m = _make_matrix([row])
        with patch("random.randint") as mock_randint:
            result = select_blue_pattern(0, m)
        mock_randint.assert_not_called()
        colors_by_col = {col: color for _, col, color in result}
        self.assertEqual([colors_by_col[j] for j in range(len(row))], row)

    def test_real_mixed_no_orphan_row_from_fixture(self):
        # From the real fixture cp_template_legform_all_ST.xlsx (legform row 1,
        # Femur): despite the file's name, this row embeds two SA-SA pairs, each
        # flanked by ST on both sides -- every SA already has a tested neighbor,
        # so this must NOT flip. Today's bug flips it anyway and drops ~half the
        # declared ST cells.
        row = [ST] * 6 + [SA, SA] + [ST] * 5 + [SA, SA] + [ST] * 6
        self.assertEqual(len(row), 21)
        m = _make_matrix([row])
        with patch("random.randint") as mock_randint:
            result = select_blue_pattern(0, m)
        mock_randint.assert_not_called()
        self.assertEqual(len(result), 21)
        colors_by_col = {col: color for _, col, color in result}
        self.assertEqual([colors_by_col[j] for j in range(21)], row)

    def test_outermost_orphaned_sa_flips_and_leaves_no_orphan(self):
        # idx0's SA neighbor is idx1 (also SA) -> orphaned; same for idx5. Must flip.
        row = [SA, SA, T, T, SA, SA]
        for coin in (0, 1):
            with self.subTest(coin=coin):
                m = _make_matrix([row])
                with patch("random.randint", return_value=coin):
                    result = select_blue_pattern(0, m)
                # Every cell must still be present.
                self.assertEqual({col for _, col, _ in result}, set(range(6)))
                # The T cells (idx 2, 3) must never be relabeled.
                self.assertEqual(m[0][2], T)
                self.assertEqual(m[0][3], T)
                # No SA may remain orphaned after promotion.
                final = m[0]
                t_st = {T, ST}
                for i in range(6):
                    if final[i] == SA:
                        left = i > 0 and final[i - 1] in t_st
                        right = i < 5 and final[i + 1] in t_st
                        self.assertTrue(
                            left or right, f"col {i} left orphaned after coin={coin}"
                        )

    def test_greenfield_all_sa_reduces_to_alternating_pattern(self):
        # From the real fixture cp_template_legform_all_SA.xlsx (legform row 1): [SA]*21.
        row = [SA] * 21
        m = _make_matrix([row])
        with patch("random.randint", return_value=0):
            result = select_blue_pattern(0, m)
        self.assertEqual({col for _, col, _ in result}, set(range(21)))
        for col in range(21):
            expected = ST if col % 2 == 0 else SA
            self.assertEqual(m[0][col], expected)

    def test_t_st_never_relabeled_regardless_of_coin(self):
        row = [T, SA, ST, SA, T, SA]
        for coin in (0, 1):
            with self.subTest(coin=coin):
                m = _make_matrix([row])
                with patch("random.randint", return_value=coin):
                    result = select_blue_pattern(0, m)
                self.assertEqual({col for _, col, _ in result}, set(range(6)))
                self.assertEqual(m[0][0], T)
                self.assertEqual(m[0][2], ST)
                self.assertEqual(m[0][4], T)

    def test_parity_measured_over_valid_cols_not_raw_column(self):
        # j: 0=SA, 1=G, 2=SA, 3=G, 4=SA -> valid_cols=[0,2,4], greenfield (orphaned).
        # coin=0 selects valid_cols indices 0,2 (i.e. raw cols 0 and 4), not raw
        # columns 0 and 2 (which would be wrong since col 1 is grey, not blue).
        row = [SA, G, SA, G, SA]
        m = _make_matrix([row])
        with patch("random.randint", return_value=0):
            result = select_blue_pattern(0, m)
        self.assertEqual({col for _, col, _ in result}, {0, 2, 4})
        self.assertEqual(m[0][0], ST)
        self.assertEqual(m[0][4], ST)
        self.assertEqual(m[0][2], SA)


# ===========================================================================
# 4 – _resolve_legform_blue_values
# ===========================================================================

NAN = float("nan")


class TestResolveLegformBlueValues(unittest.TestCase):

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _loadcases(self, criteria_list, br_name="Pelvis"):
        return [_make_loadcase(criteria_list, br_name=br_name)]

    # ------------------------------------------------------------------
    # Return value: correct warning count
    # ------------------------------------------------------------------

    def test_empty_loadcases_returns_zero(self):
        self.assertEqual(_resolve_legform_blue_values([]), 0)

    def test_no_blue_criteria_returns_zero(self):
        c = _make_criteria("X", "green", 4.0, row=0, col=3)
        self.assertEqual(_resolve_legform_blue_values(self._loadcases([c])), 0)

    # ------------------------------------------------------------------
    # T – value provided: no warning
    # ------------------------------------------------------------------

    def test_t_with_value_no_warning(self):
        c = _make_criteria("Sum of forces", "t", 4.5, row=0, col=3)
        warnings = _resolve_legform_blue_values(self._loadcases([c]))
        self.assertEqual(warnings, 0)
        self.assertAlmostEqual(c.value, 4.5, places=2)

    # ------------------------------------------------------------------
    # T – no value: 1 warning, value stays NaN
    # ------------------------------------------------------------------

    def test_t_without_value_one_warning(self):
        c = _make_criteria("Sum of forces", "t", NAN, row=0, col=3)
        warnings = _resolve_legform_blue_values(self._loadcases([c]))
        self.assertEqual(warnings, 1)
        self.assertTrue(math.isnan(c.value))

    # ------------------------------------------------------------------
    # ST – no value but symmetric provided: value copied, no warning
    # ------------------------------------------------------------------

    def test_st_copies_value_from_symmetric(self):
        # ST at col=3, symmetric at col=-3
        st_no_val = _make_criteria("Sum of forces", "st", NAN, row=0, col=3)
        st_sym = _make_criteria("Sum of forces", "st", 4.2, row=0, col=-3)
        warnings = _resolve_legform_blue_values(self._loadcases([st_no_val, st_sym]))
        self.assertEqual(warnings, 0)
        self.assertAlmostEqual(st_no_val.value, 4.2, places=2)

    def test_st_symmetric_value_triggers_score_recalc(self):
        # After copying value the score should reflect the HPL/LPL (4.2 < 5.0 → green → 100)
        st_no_val = _make_criteria("Sum of forces", "st", NAN, row=0, col=3)
        st_sym = _make_criteria("Sum of forces", "st", 4.2, row=0, col=-3)
        _resolve_legform_blue_values(self._loadcases([st_no_val, st_sym]))
        # After set_value the color should be set (not None)
        self.assertIsNotNone(st_no_val.color)

    # ------------------------------------------------------------------
    # ST – no value AND symmetric also missing: 1 warning, both stay NaN
    # ------------------------------------------------------------------

    def test_st_both_missing_one_warning(self):
        st1 = _make_criteria("Sum of forces", "st", NAN, row=0, col=3)
        st2 = _make_criteria("Sum of forces", "st", NAN, row=0, col=-3)
        warnings = _resolve_legform_blue_values(self._loadcases([st1, st2]))
        self.assertEqual(warnings, 2)  # one per blank ST (both miss each other)
        self.assertTrue(math.isnan(st1.value))
        self.assertTrue(math.isnan(st2.value))

    # ------------------------------------------------------------------
    # SA – both adjacent have values: max is used, no warning
    # ------------------------------------------------------------------

    def test_sa_uses_max_of_adjacent(self):
        sa = _make_criteria("Sum of forces", "sa", NAN, row=0, col=2)
        adj_lo = _make_criteria(
            "Sum of forces", "t", 3.5, row=0, col=1
        )  # left  (lower)
        adj_hi = _make_criteria(
            "Sum of forces", "t", 5.8, row=0, col=3
        )  # right (higher → worst)
        warnings = _resolve_legform_blue_values(self._loadcases([sa, adj_lo, adj_hi]))
        self.assertEqual(warnings, 0)
        self.assertAlmostEqual(sa.value, 5.8, places=2)

    def test_sa_max_is_not_min(self):
        sa = _make_criteria("Sum of forces", "sa", NAN, row=0, col=2)
        adj_lo = _make_criteria("Sum of forces", "t", 3.5, row=0, col=1)
        adj_hi = _make_criteria("Sum of forces", "t", 5.8, row=0, col=3)
        _resolve_legform_blue_values(self._loadcases([sa, adj_lo, adj_hi]))
        self.assertNotAlmostEqual(sa.value, 3.5, places=2)

    # ------------------------------------------------------------------
    # SA – only one adjacent has value: that value is used
    # ------------------------------------------------------------------

    def test_sa_uses_single_adjacent_value(self):
        sa = _make_criteria("Sum of forces", "sa", NAN, row=0, col=2)
        adj = _make_criteria("Sum of forces", "t", 4.8, row=0, col=1)
        # No criterion at col=3
        warnings = _resolve_legform_blue_values(self._loadcases([sa, adj]))
        self.assertEqual(warnings, 0)
        self.assertAlmostEqual(sa.value, 4.8, places=2)

    # ------------------------------------------------------------------
    # SA – no adjacent values: 1 warning, value stays NaN
    # ------------------------------------------------------------------

    def test_sa_no_adjacent_one_warning(self):
        sa = _make_criteria("Sum of forces", "sa", NAN, row=0, col=2)
        warnings = _resolve_legform_blue_values(self._loadcases([sa]))
        self.assertEqual(warnings, 1)
        self.assertTrue(math.isnan(sa.value))

    # ------------------------------------------------------------------
    # Resolution order must not matter: a point resolved from its symmetric
    # partner is itself a valid source for an adjacent SA point, even when it
    # sits later in the sheet.
    # ------------------------------------------------------------------

    def test_sa_resolves_from_st_that_is_itself_mirror_resolved(self):
        # Sheet order runs +col..-col, so the edge SA at col=7 is visited
        # before the ST at col=6 has been copied from its partner at col=-6.
        sa = _make_criteria("Sum of forces", "sa", NAN, row=0, col=7)
        st = _make_criteria("Sum of forces", "st", NAN, row=0, col=6)
        st_sym = _make_criteria("Sum of forces", "st", 4.0, row=0, col=-6)
        warnings = _resolve_legform_blue_values(self._loadcases([sa, st, st_sym]))
        self.assertEqual(warnings, 0)
        self.assertAlmostEqual(st.value, 4.0, places=2)
        self.assertAlmostEqual(sa.value, 4.0, places=2)

    def test_resolution_independent_of_which_side_carries_the_value(self):
        # The prediction row is symmetric, so mirroring the measured value
        # must not change any resolved value or the warning count.
        for measured_col in (6, -6):
            with self.subTest(measured_col=measured_col):
                pts = {
                    col: _make_criteria(
                        "Sum of forces",
                        "sa" if abs(col) == 7 else "st",
                        4.0 if col == measured_col else NAN,
                        row=0,
                        col=col,
                    )
                    for col in (7, 6, -6, -7)
                }
                warnings = _resolve_legform_blue_values(
                    self._loadcases([pts[7], pts[6], pts[-6], pts[-7]])
                )
                self.assertEqual(warnings, 0)
                for col, criteria in pts.items():
                    self.assertAlmostEqual(
                        criteria.value, 4.0, places=2, msg=f"col={col}"
                    )

    # ------------------------------------------------------------------
    # An SA must honour "max of adjacent" even when a neighbour settles
    # after the SA is first resolvable, regardless of entry side.
    # ------------------------------------------------------------------

    def test_sa_takes_max_when_neighbour_resolves_later(self):
        # SA(-7) is visited while only ST(-6)=3.5 is available; ST(-8) is
        # mirror-resolved to 5.8 afterwards, so the SA max must be revised.
        sa = _make_criteria("Sum of forces", "sa", NAN, row=0, col=-7)
        st_near = _make_criteria("Sum of forces", "st", 3.5, row=0, col=-6)
        st_late = _make_criteria("Sum of forces", "st", NAN, row=0, col=-8)
        st_mirror = _make_criteria("Sum of forces", "st", 5.8, row=0, col=8)
        warnings = _resolve_legform_blue_values(
            self._loadcases([st_mirror, st_near, sa, st_late])
        )
        self.assertEqual(warnings, 0)
        self.assertAlmostEqual(st_late.value, 5.8, places=2)
        self.assertAlmostEqual(sa.value, 5.8, places=2)

    def test_sa_max_independent_of_which_side_carries_the_values(self):
        # Same fill mirrored: the SA on either side must end at the same max.
        for sign in (1, -1):
            with self.subTest(sign=sign):
                sa = _make_criteria("Sum of forces", "sa", NAN, row=0, col=sign * 7)
                st_near = _make_criteria(
                    "Sum of forces", "st", 3.5, row=0, col=sign * 6
                )
                st_late = _make_criteria(
                    "Sum of forces", "st", NAN, row=0, col=sign * 8
                )
                st_mirror = _make_criteria(
                    "Sum of forces", "st", 5.8, row=0, col=-sign * 8
                )
                warnings = _resolve_legform_blue_values(
                    self._loadcases(
                        sorted(
                            [sa, st_near, st_late, st_mirror],
                            key=lambda c: -_parse_test_point(c.test_point)[1],
                        )
                    )
                )
                self.assertEqual(warnings, 0)
                self.assertAlmostEqual(sa.value, 5.8, places=2, msg=f"sign={sign}")

    def test_measured_sa_value_is_never_revised(self):
        # A measured SA is authoritative: neighbours settling higher must not
        # overwrite it.
        sa = _make_criteria("Sum of forces", "sa", 2.0, row=0, col=-7)
        st_late = _make_criteria("Sum of forces", "st", NAN, row=0, col=-8)
        st_mirror = _make_criteria("Sum of forces", "st", 5.8, row=0, col=8)
        warnings = _resolve_legform_blue_values(
            self._loadcases([st_mirror, sa, st_late])
        )
        self.assertEqual(warnings, 0)
        self.assertAlmostEqual(sa.value, 2.0, places=2)

    # ------------------------------------------------------------------
    # Adjacent lookup is by (row, col, name) – different names don't match
    # ------------------------------------------------------------------

    def test_sa_adjacent_different_name_not_used(self):
        sa = _make_criteria("Sum of forces", "sa", NAN, row=0, col=2)
        # Adjacent criterion exists but with a different name
        adj = _make_criteria("OTHER criterion", "t", 4.8, row=0, col=1)
        warnings = _resolve_legform_blue_values(self._loadcases([sa, adj]))
        self.assertEqual(warnings, 1)

    # ------------------------------------------------------------------
    # Multiple criteria in one pass: warnings accumulate correctly
    # ------------------------------------------------------------------

    def test_multiple_missing_criteria_correct_warning_count(self):
        t_no_val = _make_criteria("F1", "t", NAN, row=0, col=3)
        sa_no_adj = _make_criteria("F2", "sa", NAN, row=0, col=5)
        st_no_sym = _make_criteria("F3", "st", NAN, row=0, col=4)
        # No symmetrics or adjacents provided → 3 warnings
        warnings = _resolve_legform_blue_values(
            self._loadcases([t_no_val, sa_no_adj, st_no_sym])
        )
        self.assertEqual(warnings, 3)

    def test_mixed_some_ok_some_not(self):
        t_ok = _make_criteria("F1", "t", 4.5, row=0, col=3)
        sa_ok_adj = _make_criteria("F2", "sa", NAN, row=0, col=2)
        adj = _make_criteria("F2", "t", 5.2, row=0, col=1)
        t_bad = _make_criteria("F3", "t", NAN, row=0, col=5)
        warnings = _resolve_legform_blue_values(
            self._loadcases([t_ok, sa_ok_adj, adj, t_bad])
        )
        self.assertEqual(warnings, 1)
        self.assertAlmostEqual(sa_ok_adj.value, 5.2, places=2)

    # ------------------------------------------------------------------
    # Criteria with no test_point are silently skipped (no crash)
    # ------------------------------------------------------------------

    def test_criteria_with_no_test_point_skipped(self):
        c = _make_criteria("F1", "t", NAN, row=0, col=3)
        c.test_point = None  # malformed
        # Should not raise
        warnings = _resolve_legform_blue_values(self._loadcases([c]))
        # Skipped, so no warning emitted for it
        self.assertEqual(warnings, 0)

    # ------------------------------------------------------------------
    # SA scoring: color and score set correctly; prediction_result stays None
    # ------------------------------------------------------------------

    def test_sa_resolve_sets_color(self):
        sa = _make_criteria("Sum of forces", "sa", NAN, row=0, col=2)
        adj = _make_criteria("Sum of forces", "t", 4.0, row=0, col=1)
        _resolve_legform_blue_values(self._loadcases([sa, adj]))
        self.assertIsNotNone(sa.color)

    def test_sa_prediction_result_not_incorrect(self):
        sa = _make_criteria("Sum of forces", "sa", NAN, row=0, col=2)
        adj = _make_criteria("Sum of forces", "t", 4.0, row=0, col=1)
        _resolve_legform_blue_values(self._loadcases([sa, adj]))
        self.assertIsNone(sa.prediction_result)

    def test_sa_score_computed_from_adjacent_value(self):
        # 4.0 < HPL (5.0) → green → score 100
        sa = _make_criteria("Sum of forces", "sa", NAN, row=0, col=2)
        adj = _make_criteria("Sum of forces", "t", 4.0, row=0, col=1)
        _resolve_legform_blue_values(self._loadcases([sa, adj]))
        self.assertAlmostEqual(sa.score, 100.0, places=1)

    def test_t_prediction_result_not_incorrect(self):
        t = _make_criteria("Sum of forces", "t", 4.0, row=0, col=3)
        _resolve_legform_blue_values(self._loadcases([t]))
        self.assertIsNone(t.prediction_result)

    def test_st_prediction_result_not_incorrect_after_resolve(self):
        st_blank = _make_criteria("Sum of forces", "st", NAN, row=0, col=3)
        st_sym = _make_criteria("Sum of forces", "st", 4.0, row=0, col=-3)
        _resolve_legform_blue_values(self._loadcases([st_blank, st_sym]))
        self.assertIsNone(st_blank.prediction_result)


# ===========================================================================
# 5 – VruTestData.from_sheet_dict integration
# ===========================================================================


class TestFromSheetDictIntegration(unittest.TestCase):
    """Verifies that from_sheet_dict calls _resolve_legform_blue_values and stores
    the warning count in legform_blue_warnings."""

    def _build_sheet_dict(self, criteria_list):
        br = BodyRegion(name="Pelvis")
        br._criteria = list(criteria_list)
        dummy = Dummy(name="Upper legform", body_region_list=[br])
        seat = Seat(name="Driver", dummy=dummy)
        lc = LoadCase(name="Upper Leg", seats=[seat])
        return {
            "CP - VRU Head Impact": [],
            "CP - VRU Pelvis & Leg Impact": [lc],
        }

    def test_no_warnings_when_all_t_values_provided(self):
        c = _make_criteria("Sum of forces", "t", 4.5)
        sheet_dict = self._build_sheet_dict([c])
        vtu = VruTestData.from_sheet_dict(sheet_dict)
        self.assertEqual(vtu.legform_blue_warnings, 0)

    def test_one_warning_stored_for_missing_t_value(self):
        c = _make_criteria("Sum of forces", "t", NAN)
        sheet_dict = self._build_sheet_dict([c])
        vtu = VruTestData.from_sheet_dict(sheet_dict)
        self.assertEqual(vtu.legform_blue_warnings, 1)

    def test_st_value_resolved_before_score_update(self):
        """ST with blank value and its symmetric already filled → value resolved."""
        st_blank = _make_criteria("Sum of forces", "st", NAN, row=0, col=3)
        st_sym = _make_criteria("Sum of forces", "st", 4.1, row=0, col=-3)
        sheet_dict = self._build_sheet_dict([st_blank, st_sym])
        vtu = VruTestData.from_sheet_dict(sheet_dict)
        self.assertEqual(vtu.legform_blue_warnings, 0)
        self.assertAlmostEqual(st_blank.value, 4.1, places=2)

    def test_sa_value_resolved_before_score_update(self):
        """SA uses max of adjacent; value propagates to criteria object."""
        sa = _make_criteria("Sum of forces", "sa", NAN, row=0, col=2)
        adj = _make_criteria("Sum of forces", "st", 5.7, row=0, col=1)
        sheet_dict = self._build_sheet_dict([sa, adj])
        vtu = VruTestData.from_sheet_dict(sheet_dict)
        self.assertEqual(vtu.legform_blue_warnings, 0)
        self.assertAlmostEqual(sa.value, 5.7, places=2)

    def test_legform_blue_warnings_default_is_zero(self):
        """Freshly constructed VruTestData has legform_blue_warnings=0."""
        self.assertEqual(VruTestData().legform_blue_warnings, 0)


# ===========================================================================
# 6 – compute_vru_bodyregion_score "Score successfully computed" message
# ===========================================================================

_VRU_LOGGER = "euroncap_rating_2026.crash_protection.vru_processing"


class TestScoreSuccessfullyComputedMessage(unittest.TestCase):

    _LEGFORM_SUCCESS = "Score successfully computed for all blue legform criteria"
    _APILLAR_SUCCESS = "Score successfully computed for all A-pillar headform criteria"

    def _minimal_vtu(self, blue_warnings, apillar_warnings=0):
        vtu = VruTestData()
        vtu.legform_blue_warnings = blue_warnings
        vtu.headform_apillar_warnings = apillar_warnings
        return vtu

    def test_info_message_logged_when_no_warnings(self):
        vtu = self._minimal_vtu(0)
        with self.assertLogs(_VRU_LOGGER, level="INFO") as cm:
            vtu.compute_vru_bodyregion_score()
        success_msgs = [m for m in cm.output if self._LEGFORM_SUCCESS in m]
        self.assertTrue(success_msgs, "Expected 'Score successfully computed' in logs")

    def test_no_message_when_warnings_present(self):
        vtu = self._minimal_vtu(1)
        with self.assertLogs(_VRU_LOGGER, level="INFO") as cm:
            vtu.compute_vru_bodyregion_score()
        success_msgs = [m for m in cm.output if self._LEGFORM_SUCCESS in m]
        self.assertEqual(success_msgs, [])

    def test_no_message_for_multiple_warnings(self):
        vtu = self._minimal_vtu(3)
        with self.assertLogs(_VRU_LOGGER, level="INFO") as cm:
            vtu.compute_vru_bodyregion_score()
        success_msgs = [m for m in cm.output if self._LEGFORM_SUCCESS in m]
        self.assertEqual(success_msgs, [])

    def test_apillar_message_logged_when_no_apillar_warnings(self):
        vtu = self._minimal_vtu(0, apillar_warnings=0)
        with self.assertLogs(_VRU_LOGGER, level="INFO") as cm:
            vtu.compute_vru_bodyregion_score()
        success_msgs = [m for m in cm.output if self._APILLAR_SUCCESS in m]
        self.assertTrue(success_msgs, "Expected A-pillar success message in logs")

    def test_no_apillar_message_when_apillar_warnings_present(self):
        vtu = self._minimal_vtu(0, apillar_warnings=2)
        with self.assertLogs(_VRU_LOGGER, level="INFO") as cm:
            vtu.compute_vru_bodyregion_score()
        success_msgs = [m for m in cm.output if self._APILLAR_SUCCESS in m]
        self.assertEqual(success_msgs, [])

    def test_legform_scores_populated_after_compute(self):
        """compute_vru_bodyregion_score always populates legform_scores."""
        vtu = self._minimal_vtu(0)
        with self.assertLogs(_VRU_LOGGER, level="INFO"):
            vtu.compute_vru_bodyregion_score()
        self.assertIn("Pelvis", vtu.legform_scores)
        self.assertIn("Femur", vtu.legform_scores)
        self.assertIn("Knee & Tibia", vtu.legform_scores)


# ===========================================================================
# 7 – report_writer Value-column SA styling
# ===========================================================================


class TestValueColumnSAStyling(unittest.TestCase):
    """
    Reproduce the styling logic from report_writer.write_report() inline so the
    assertion is deterministic without real Excel file I/O.
    """

    GREY_RGB = "00D9D9D9"  # openpyxl prepends 00 alpha when given a 6-char hex colour
    GREY_FILL = PatternFill(start_color="D9D9D9", end_color="D9D9D9", fill_type="solid")

    def _build_ws(self, prediction_values):
        """Build a minimal worksheet: col A = OEM Prediction, col B = Value."""
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.cell(row=1, column=1).value = "OEM Prediction"
        ws.cell(row=1, column=2).value = "Value"
        for i, pred in enumerate(prediction_values, start=2):
            ws.cell(row=i, column=1).value = pred
            ws.cell(row=i, column=2).value = ""
        return ws

    def _apply_styling(self, ws):
        """Apply the same logic as in report_writer for the Value column."""
        cols = list(ws.iter_cols(1, ws.max_column))
        oem_col = next((c for c in cols if c[0].value == "OEM Prediction"), None)
        val_col = next((c for c in cols if c[0].value == "Value"), None)

        oem_by_row = {}
        if oem_col is not None:
            for cell in oem_col[1:]:
                if cell.value and str(cell.value).strip():
                    oem_by_row[cell.row] = str(cell.value).strip().lower()

        if val_col is not None:
            for cell in val_col[1:]:
                prediction = oem_by_row.get(cell.row, "")
                if prediction != "sa":
                    cell.fill = self.GREY_FILL

    def _cell_is_grey(self, ws, row, col):
        fill = ws.cell(row=row, column=col).fill
        return fill.fill_type == "solid" and fill.fgColor.rgb == self.GREY_RGB

    # ------------------------------------------------------------------
    # T row → grey Value cell
    # ------------------------------------------------------------------

    def test_t_row_gets_grey_value_cell(self):
        ws = self._build_ws(["T"])
        self._apply_styling(ws)
        self.assertTrue(self._cell_is_grey(ws, 2, 2))

    # ------------------------------------------------------------------
    # ST row → grey Value cell
    # ------------------------------------------------------------------

    def test_st_row_gets_grey_value_cell(self):
        ws = self._build_ws(["St"])
        self._apply_styling(ws)
        self.assertTrue(self._cell_is_grey(ws, 2, 2))

    def test_st_lower_row_gets_grey_value_cell(self):
        ws = self._build_ws(["st"])
        self._apply_styling(ws)
        self.assertTrue(self._cell_is_grey(ws, 2, 2))

    # ------------------------------------------------------------------
    # SA row → NOT grey (white / no fill)
    # ------------------------------------------------------------------

    def test_sa_row_does_not_get_grey_value_cell(self):
        ws = self._build_ws(["Sa"])
        self._apply_styling(ws)
        self.assertFalse(self._cell_is_grey(ws, 2, 2))

    def test_sa_lower_row_does_not_get_grey_value_cell(self):
        ws = self._build_ws(["sa"])
        self._apply_styling(ws)
        self.assertFalse(self._cell_is_grey(ws, 2, 2))

    # ------------------------------------------------------------------
    # Mixed rows in correct order
    # ------------------------------------------------------------------

    def test_mixed_rows_correct_styling(self):
        ws = self._build_ws(["T", "St", "Sa", "Sa", "st"])
        self._apply_styling(ws)
        self.assertTrue(self._cell_is_grey(ws, 2, 2), "T → grey")
        self.assertTrue(self._cell_is_grey(ws, 3, 2), "St → grey")
        self.assertFalse(self._cell_is_grey(ws, 4, 2), "Sa → NOT grey")
        self.assertFalse(self._cell_is_grey(ws, 5, 2), "Sa → NOT grey")
        self.assertTrue(self._cell_is_grey(ws, 6, 2), "st → grey")

    # ------------------------------------------------------------------
    # Non-blue predictions (green, yellow) → grey (unchanged by new code)
    # ------------------------------------------------------------------

    def test_non_blue_prediction_still_gets_grey(self):
        ws = self._build_ws(["Green"])
        self._apply_styling(ws)
        self.assertTrue(self._cell_is_grey(ws, 2, 2))

    def test_empty_prediction_still_gets_grey(self):
        ws = self._build_ws([""])
        self._apply_styling(ws)
        self.assertTrue(self._cell_is_grey(ws, 2, 2))


# ===========================================================================
# 8 – End-to-end: process_legform_test consistency check is called
# ===========================================================================


class TestProcessLegformTestCallsSymmetryCheck(unittest.TestCase):
    """Verify that process_legform_test invokes check_legform_blue_symmetry
    and propagates any ValueError it raises."""

    def _minimal_df(self):
        """Minimal prediction DataFrame: 28 rows × 25 cols, all grey."""
        return pd.DataFrame("grey", index=range(28), columns=range(25))

    def _minimal_params(self):
        return pd.DataFrame(
            {
                "param_code": ["CP - VRU Pelvis Impact", "CP - VRU Leg Impact"],
                "Input parameter": [
                    "Number of verification tests",
                    "Number of verification tests",
                ],
                "Value": [1, 1],
            }
        )

    def test_symmetry_check_called_and_error_propagates(self):
        """Patching check_legform_blue_symmetry to raise ensures it is called."""
        pred_df = self._minimal_df()
        vru_params = self._minimal_params()
        sentinel = ValueError("injected asymmetry error")
        with patch.object(
            vru_processing, "check_legform_blue_symmetry", side_effect=sentinel
        ):
            with self.assertRaises(ValueError) as ctx:
                vru_processing.process_legform_test(pred_df, vru_params)
        self.assertIs(ctx.exception, sentinel)

    def test_asymmetric_t_raises_via_real_matrix(self):
        """T at j=0 without a symmetric T/ST partner → real ValueError."""
        pred_df = self._minimal_df()
        # 25 cols → after iloc[23:28].iloc[2:, 4:] → 3 rows × 21 cols
        # j=0 maps to df col 4; j=20 (symmetric) maps to df col 24
        # Set row 25 (legform matrix row 0) j=0 to "t", leave j=20 as "grey"
        for df_row in (25, 26, 27):
            pred_df.iat[df_row, 4] = "t"  # j=0, col=10 → no symmetric at j=20
        vru_params = self._minimal_params()
        with self.assertRaises(ValueError) as ctx:
            vru_processing.process_legform_test(pred_df, vru_params)
        self.assertIn("Asymmetric", str(ctx.exception))


# ===========================================================================
# 9 – process_legform_criteria: Value must be blank (NaN), not 0.0
#
# Root-cause regression guard for Issue 2 (SA resolution skipped).
# If process_legform_criteria writes 0.0 instead of NaN, pd.isna(0.0) = False
# and _resolve_legform_blue_values treats SA criteria as "already resolved",
# skipping the adjacent-column lookup and leaving SA with a wrong score.
# ===========================================================================


class TestProcessLegformCriteriaValueIsBlank(unittest.TestCase):

    def _make_bare_criteria(self, name="Sum of forces", hpl=5.0, lpl=6.0):
        return Criteria(
            name=name, hpl=hpl, lpl=lpl, criteria_type=CriteriaType.CRITERIA
        )

    def test_t_criteria_value_is_nan_after_process(self):
        c = self._make_bare_criteria()
        process_legform_criteria(c, common.format_test_point(0, 3), "T")
        self.assertTrue(math.isnan(c.value), f"Expected NaN, got {c.value}")

    def test_st_criteria_value_is_nan_after_process(self):
        c = self._make_bare_criteria()
        process_legform_criteria(c, common.format_test_point(0, 3), "St")
        self.assertTrue(math.isnan(c.value), f"Expected NaN, got {c.value}")

    def test_sa_criteria_value_is_nan_after_process(self):
        c = self._make_bare_criteria()
        process_legform_criteria(c, common.format_test_point(0, 2), "Sa")
        self.assertTrue(math.isnan(c.value), f"Expected NaN, got {c.value}")

    def test_score_defaults_to_zero_when_value_is_nan(self):
        c = self._make_bare_criteria()
        process_legform_criteria(c, common.format_test_point(0, 3), "T")
        self.assertEqual(c.score, 0.0)

    def test_sa_with_nan_value_gets_resolved_at_compute_time(self):
        """End-to-end: SA written as NaN → resolved from adjacent at from_sheet_dict."""
        sa = _make_criteria("Sum of forces", "sa", NAN, row=0, col=2)
        adj_lo = _make_criteria("Sum of forces", "st", 3.0, row=0, col=1)
        adj_hi = _make_criteria("Sum of forces", "st", 6.5, row=0, col=3)
        br = BodyRegion(name="Pelvis")
        br._criteria = [sa, adj_lo, adj_hi]
        dummy = Dummy(name="Upper legform", body_region_list=[br])
        seat = Seat(name="Driver", dummy=dummy)
        lc = LoadCase(name="Upper Leg", seats=[seat])
        sheet_dict = {"CP - VRU Head Impact": [], "CP - VRU Pelvis & Leg Impact": [lc]}
        vtu = VruTestData.from_sheet_dict(sheet_dict)
        self.assertEqual(vtu.legform_blue_warnings, 0)
        self.assertAlmostEqual(
            sa.value, 6.5, places=2, msg="SA should get max(3.0, 6.5)"
        )

    def test_sa_with_zero_value_is_not_re_resolved(self):
        """SA with a user-provided value of 0.0 must NOT be overwritten by adjacent lookup."""
        sa_with_val = _make_criteria("Sum of forces", "sa", 0.0, row=0, col=2)
        adj = _make_criteria("Sum of forces", "st", 4.8, row=0, col=1)
        warnings = _resolve_legform_blue_values([_make_loadcase([sa_with_val, adj])])
        self.assertEqual(warnings, 0)
        self.assertAlmostEqual(
            sa_with_val.value,
            0.0,
            places=2,
            msg="0.0 is a valid provided value; must not be overwritten",
        )


class TestProcessLegformTestWithManualInput(unittest.TestCase):
    """Tests for process_legform_test() when input_selected_points is supplied.

    Matrix layout
    -------------
    The legform slice of the prediction DataFrame is
    ``df.iloc[23:28, :]`` → reset_index → ``.iloc[2:, 4:]``, so:
        * legform matrix row r  ↔  df row  25 + r
        * legform matrix col j  ↔  df col   4 + j
    The matrix produced from a 28×15 "all-grey" df therefore has shape 3×11.
    Column j=5 (centre, N=11, col_center=5) is symmetric to itself, which
    makes it convenient for single-cell tests that need to pass the symmetry
    check.
    """

    _LEG_ROW = 25  # df row offset for legform matrix row 0
    _LEG_COL = 4  # df col offset for legform matrix col 0
    _N_COLS = 11  # number of legform matrix columns (df has 15 cols, cols 4–14)
    _CENTER = 5  # symmetric-to-itself column index in an 11-wide matrix

    def _build_df(self, cells=None):
        """28×15 all-grey DataFrame with optional legform cells set.

        cells: {(matrix_row, matrix_col): color_string}
        """
        df = pd.DataFrame("grey", index=range(28), columns=range(15))
        if cells:
            for (mrow, mcol), color in cells.items():
                df.iat[self._LEG_ROW + mrow, self._LEG_COL + mcol] = color
        return df

    def _call(self, df, input_selected_points):
        return vru_processing.process_legform_test(
            df, pd.DataFrame(), input_selected_points=input_selected_points
        )

    # ------------------------------------------------------------------
    # 1. Color forced to BLUE_ST
    # ------------------------------------------------------------------

    def test_matched_t_point_classified_as_blue_st(self):
        """A BLUE_T cell given via input_selected_points must come back as BLUE_ST."""
        df = self._build_df({(1, self._CENTER): "t"})
        result = self._call(
            df, {"Lower Leg": [{"Body region": "Femur", "row": 1, "col": 5}]}
        )
        self.assertEqual(len(result), 1)
        pt = result[0]
        self.assertEqual(pt.color, VruPredictionColor.BLUE_ST)
        self.assertEqual(pt.row, 1)
        self.assertEqual(pt.col, 5)

    def test_matched_sa_point_classified_as_blue_st(self):
        """A BLUE_SA cell explicitly selected via input_selected_points becomes BLUE_ST."""
        df = self._build_df({(0, self._CENTER): "sa"})
        result = self._call(
            df, {"Upper Leg": [{"Body region": "Pelvis", "row": 0, "col": 5}]}
        )
        matched = [p for p in result if p.row == 0 and p.col == 5]
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0].color, VruPredictionColor.BLUE_ST)

    # ------------------------------------------------------------------
    # 2. Consistency check still active
    # ------------------------------------------------------------------

    def test_symmetry_check_raises_for_asymmetric_blue_matrix(self):
        """Asymmetric T cell in the matrix must raise ValueError even when
        input_selected_points is provided (check runs before point matching)."""
        # j=3 has symmetric j_sym = 11-1-3 = 7; only j=3 is T → asymmetric
        df = self._build_df({(1, 3): "t"})
        with self.assertRaises(ValueError):
            self._call(
                df, {"Lower Leg": [{"Body region": "Femur", "row": 1, "col": 7}]}
            )

    # ------------------------------------------------------------------
    # 3. Row-2 mirror
    # ------------------------------------------------------------------

    def test_row2_mirror_added_when_non_grey(self):
        """Selecting a row-1 (Femur) point auto-adds the row-2 cell at the same
        column, provided that cell is not GREY."""
        df = self._build_df({(1, self._CENTER): "t", (2, self._CENTER): "sa"})
        result = self._call(
            df, {"Lower Leg": [{"Body region": "Femur", "row": 1, "col": 5}]}
        )
        rows_cols_colors = [(p.row, p.col, p.color) for p in result]
        self.assertIn((1, 5, VruPredictionColor.BLUE_ST), rows_cols_colors)
        self.assertIn((2, 5, VruPredictionColor.BLUE_SA), rows_cols_colors)

    def test_row2_mirror_skipped_when_grey(self):
        """If the row-2 cell at the mirrored column is GREY, no mirror is added."""
        df = self._build_df({(1, self._CENTER): "t"})  # row-2 stays grey
        result = self._call(
            df, {"Lower Leg": [{"Body region": "Femur", "row": 1, "col": 5}]}
        )
        row2_points = [p for p in result if p.row == 2]
        self.assertEqual(row2_points, [])

    # ------------------------------------------------------------------
    # 4. Unselected SA cells always appear in output
    # ------------------------------------------------------------------

    def test_unselected_sa_cells_always_included(self):
        """SA cells not covered by input_selected_points still appear in the output
        with BLUE_SA color (verification row without a fillable KPI value)."""
        df = self._build_df({(0, self._CENTER): "sa"})
        result = self._call(df, {})  # empty input → no matched points
        sa_points = [p for p in result if p.color == VruPredictionColor.BLUE_SA]
        self.assertEqual(len(sa_points), 1)
        self.assertEqual(sa_points[0].row, 0)
        self.assertEqual(sa_points[0].col, 5)

    def test_selected_sa_not_duplicated_as_unselected(self):
        """An SA cell that IS selected via input_selected_points must appear exactly
        once (as BLUE_ST), not also as an unselected BLUE_SA."""
        df = self._build_df({(0, self._CENTER): "sa"})
        result = self._call(
            df, {"Upper Leg": [{"Body region": "Pelvis", "row": 0, "col": 5}]}
        )
        at_center = [
            (p.row, p.col, p.color) for p in result if p.row == 0 and p.col == 5
        ]
        self.assertEqual(at_center, [(0, 5, VruPredictionColor.BLUE_ST)])


class TestProcessHeadformTestWithManualInput(unittest.TestCase):
    """Tests for process_headform_test() when input_selected_points is supplied.

    Matrix layout
    -------------
    The headform slice is ``df.iloc[0:21, :]`` → reset_index → ``.iloc[2:, 4:]``:
        * headform matrix row i  ↔  df row  2 + i
        * headform matrix col j  ↔  df col  4 + j
    VruTestPoint coordinates: row = 18 - i,  col = 10 - j.
    """

    _HEAD_ROW = 2  # df row offset for headform matrix row 0
    _HEAD_COL = 4  # df col offset for headform matrix col 0

    def _build_df(self, cells=None):
        """28×15 all-grey DataFrame with optional headform cells set.

        cells: {(matrix_row, matrix_col): color_string}
        """
        df = pd.DataFrame("grey", index=range(28), columns=range(15))
        if cells:
            for (mrow, mcol), color in cells.items():
                df.iat[self._HEAD_ROW + mrow, self._HEAD_COL + mcol] = color
        return df

    def _call(self, df, input_selected_points):
        return vru_processing.process_headform_test(
            df, pd.DataFrame(), input_selected_points=input_selected_points
        )

    def test_matched_blue_t_point_retains_blue_t_color(self):
        """A BLUE_T headform cell given via input_selected_points keeps its own color."""
        # matrix[0][0] → VruTestPoint(row=18, col=10)
        df = self._build_df({(0, 0): "t"})
        result = self._call(df, {"Headform": [{"row": 18, "col": 10}]})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].color, VruPredictionColor.BLUE_T)

    def test_non_blue_matched_point_retains_original_color(self):
        """A non-blue headform cell selected via input_selected_points must keep its
        original predicted color (not be overwritten with BLUE_ST) - otherwise the
        correction factor and blue-point bonus computed downstream are both wrong."""
        # matrix[0][3] → VruTestPoint(row=18, col=7)
        df = self._build_df({(0, 3): "green"})
        result = self._call(df, {"Headform": [{"row": 18, "col": 7}]})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].color, VruPredictionColor.GREEN)

    def test_unmatched_input_returns_empty(self):
        """If no candidate matches the requested attributes, an empty list is returned."""
        df = self._build_df()  # all grey — no candidates
        result = self._call(df, {"Headform": [{"row": 18, "col": 10}]})
        self.assertEqual(result, [])

    def test_multiple_selected_points_each_retain_own_color(self):
        """Several manually selected headform points must each keep their own
        original predicted color, guarding against any regression that relabels
        every matched point to a single uniform color."""
        df = self._build_df({(0, 0): "green", (0, 3): "yellow", (0, 6): "orange"})
        result = self._call(
            df,
            {
                "Headform": [
                    {"row": 18, "col": 10},
                    {"row": 18, "col": 7},
                    {"row": 18, "col": 4},
                ]
            },
        )
        colors_by_col = {p.col: p.color for p in result}
        self.assertEqual(len(result), 3)
        self.assertEqual(colors_by_col[10], VruPredictionColor.GREEN)
        self.assertEqual(colors_by_col[7], VruPredictionColor.YELLOW)
        self.assertEqual(colors_by_col[4], VruPredictionColor.ORANGE)

    def test_apillar_points_appended_when_not_selected(self):
        """Regression: green-20/30/40 cells absent from
        the explicit selection were silently dropped. They are mandatory on
        every path and must be appended."""
        # Full-width 21-column grid: matrix[9][2] → (9, 8) and
        # matrix[9][18] → (9, -8), a symmetric green-20 pair.
        df = pd.DataFrame("grey", index=range(28), columns=range(25))
        df.iat[self._HEAD_ROW + 0, self._HEAD_COL + 0] = "green"
        df.iat[self._HEAD_ROW + 9, self._HEAD_COL + 2] = "green-20"
        df.iat[self._HEAD_ROW + 9, self._HEAD_COL + 18] = "green-20"
        result = self._call(df, {"Headform": [{"row": 18, "col": 10}]})
        apillar = sorted(
            (p.row, p.col, p.color)
            for p in result
            if p.loadcase_name == "Headform Apillar"
        )
        self.assertEqual(
            apillar,
            [
                (9, -8, VruPredictionColor.GREEN_20),
                (9, 8, VruPredictionColor.GREEN_20),
            ],
        )
        self.assertEqual(
            [(p.row, p.col) for p in result if p.loadcase_name == "Headform"],
            [(18, 10)],
        )

    def test_apillar_point_selected_explicitly_is_not_duplicated(self):
        """An A-pillar point present in the explicit selection must appear once."""
        df = self._build_df({(9, 2): "green-30"})
        result = self._call(df, {"Headform Apillar": [{"row": 9, "col": 8}]})
        apillar = [
            (p.row, p.col) for p in result if p.loadcase_name == "Headform Apillar"
        ]
        self.assertEqual(apillar, [(9, 8)])

    def test_blue_cells_not_auto_appended_on_explicit_path(self):
        """Blue cells stay the caller's responsibility on the explicit path -
        only A-pillar cells are force-appended."""
        df = self._build_df({(0, 0): "blue", (0, 3): "green"})
        result = self._call(df, {"Headform": [{"row": 18, "col": 7}]})
        self.assertEqual([(p.row, p.col) for p in result], [(18, 7)])


# ===========================================================================
# get_vru_factors / VruScore.compute_blue_points
# ===========================================================================


def _make_vru_criteria(prediction, value, hpl=650.0, lpl=1700.0, row=10, col=4):
    """Build a VRU_CRITERIA criteria with a real color/score computed from thresholds."""
    c = Criteria(
        name="HIC15", hpl=hpl, lpl=lpl, criteria_type=CriteriaType.VRU_CRITERIA
    )
    c.test_point = common.format_test_point(row, col)
    c.set_prediction(prediction)
    c.set_value(value)
    return c


def _make_headform_loadcase(criteria_list, loadcase_name="Headform", br_name="Adult"):
    br = BodyRegion(name=br_name)
    br._criteria = list(criteria_list)
    dummy = Dummy(name="Adult Headform", body_region_list=[br])
    seat = Seat(name="Driver", dummy=dummy)
    return LoadCase(name=loadcase_name, seats=[seat])


class TestGetVruFactors(unittest.TestCase):
    def test_matching_green_predictions_yield_correction_factor_one(self):
        """When every tested point is Correct green, CF must be 1.0 - not clamped
        to 0.85 as happened when verification points were mislabeled blue."""
        criteria_list = [
            _make_vru_criteria("green", 100.0, row=1, col=1),  # green: value < hpl(650)
            _make_vru_criteria("green", 200.0, row=2, col=1),
            _make_vru_criteria("green", 300.0, row=3, col=1),
        ]
        loadcase = _make_headform_loadcase(criteria_list)
        factors = vru_processing.get_vru_factors([loadcase])
        self.assertAlmostEqual(factors["correction_factor"], 1.0, places=6)

    def test_tolerance_band_point_does_not_collapse_predicted_score(self):
        """A tested value within the ±10% VRU tolerance of its prediction should
        keep contributing to both numerator and denominator (not be dropped),
        and must not collapse the correction factor to the 0.85 floor."""
        # yellow-orange threshold = 650 + (1700-650)/3 = 1000; 10% tolerance upper = 1000/0.9 = 1111.1
        criteria = _make_vru_criteria("yellow", 1050.0)
        loadcase = _make_headform_loadcase([criteria])
        factors = vru_processing.get_vru_factors([loadcase])
        self.assertEqual(criteria.prediction_result, "In Tolerance")
        self.assertGreater(factors["predicted_score_verification"], 0)
        self.assertAlmostEqual(factors["correction_factor"], 80.0 / 75.0, places=6)

    def test_apillar_loadcase_does_not_contribute(self):
        """Criteria living in a 'Headform Apillar' loadcase must never leak into
        the correction factor - only the 'Headform' loadcase counts. Uses a plain
        'green' prediction (not green-40/30/20) to prove the loadcase-name filter
        itself is doing the exclusion, not just the color whitelist."""
        headform_criteria = [_make_vru_criteria("green", 500.0)]
        headform_lc = _make_headform_loadcase(headform_criteria)

        apillar_criteria = _make_vru_criteria("green", 500.0, row=1, col=1)
        apillar_lc = _make_headform_loadcase(
            [apillar_criteria], loadcase_name="Headform Apillar"
        )

        factors_without_apillar = vru_processing.get_vru_factors([headform_lc])
        factors_with_apillar = vru_processing.get_vru_factors([headform_lc, apillar_lc])
        self.assertEqual(factors_without_apillar, factors_with_apillar)

    def test_blue_prediction_excluded_from_correction_factor(self):
        """A genuine blue-zone prediction must not contribute to the CF."""
        criteria = _make_vru_criteria("blue", 500.0)
        loadcase = _make_headform_loadcase([criteria])
        factors = vru_processing.get_vru_factors([loadcase])
        self.assertEqual(factors["predicted_score_verification"], 0)
        self.assertEqual(factors["tested_score_verification"], 0)

    def test_out_of_band_correction_factor_is_still_clamped(self):
        """Regression guard: clamping behavior for out-of-band CF is unchanged
        (only a warning is added, the returned value is still clamped)."""
        # Predict green (weight 1.0) but test red (score 0) -> raw CF collapses towards 0
        criteria = _make_vru_criteria("green", 1800.0)
        loadcase = _make_headform_loadcase([criteria])
        factors = vru_processing.get_vru_factors([loadcase])
        self.assertGreaterEqual(factors["correction_factor"], 0.85)
        self.assertLessEqual(factors["correction_factor"], 1.15)


class TestComputeBluePoints(unittest.TestCase):
    def test_genuine_blue_prediction_counted_once(self):
        criteria = _make_vru_criteria("blue", 500.0)
        loadcase = _make_headform_loadcase([criteria], br_name="Adult")
        score = vru_processing.VruScore(name="Adult")
        score.compute_blue_points([loadcase])
        self.assertAlmostEqual(score.blue_points, criteria.get_score() / 100.0)

    def test_st_prediction_not_counted_as_blue_bonus(self):
        """A headform criteria mislabeled 'st' must not grant a blue bonus - only
        genuine 'blue' predictions should. Guards against the double-counting bug
        where verification points were both scored in the main grid and bonused here."""
        criteria = _make_vru_criteria("st", 500.0)
        loadcase = _make_headform_loadcase([criteria], br_name="Adult")
        score = vru_processing.VruScore(name="Adult")
        score.compute_blue_points([loadcase])
        self.assertEqual(score.blue_points, 0.0)

    def test_t_and_sa_predictions_not_counted_as_blue_bonus(self):
        """T/ST/SA are legform-only markers and must never grant a headform bonus."""
        for prediction in ("t", "sa"):
            with self.subTest(prediction=prediction):
                criteria = _make_vru_criteria(prediction, 500.0)
                loadcase = _make_headform_loadcase([criteria], br_name="Adult")
                score = vru_processing.VruScore(name="Adult")
                score.compute_blue_points([loadcase])
                self.assertEqual(score.blue_points, 0.0)


# ===========================================================================
# Scenario-level tests for partial vs full VRU data upload
# ===========================================================================

NAN = float("nan")


class TestPartialUploadScenario(unittest.TestCase):
    """End-to-end scenario tests for partial vs full VRU data upload.

    Verifies that VruTestData.from_sheet_dict correctly tracks
    legform_blue_warnings when only a subset of VRU test runs have been
    uploaded, and that the warning count is zero when all mandatory T-values
    are present.
    """

    def _sheet_dict(self, *criteria):
        lc = _make_loadcase(list(criteria))
        return {
            "CP - VRU Head Impact": [],
            "CP - VRU Pelvis & Leg Impact": [lc],
        }

    def test_partial_upload_returns_warnings_for_missing_t_values(self):
        """Partial upload: some T-values missing → legform_blue_warnings > 0.

        Simulates a user who only submitted test data for one position; the
        other position has a blank (NaN) Value cell.  The library must flag
        the missing mandatory value and leave it as NaN rather than coercing
        it to 0 or raising an exception.
        """
        uploaded = _make_criteria("Sum of forces", "t", 4.5, row=0, col=1)
        missing = _make_criteria("Sum of forces", "t", NAN, row=0, col=2)
        vtu = VruTestData.from_sheet_dict(self._sheet_dict(uploaded, missing))
        self.assertEqual(vtu.legform_blue_warnings, 1)
        self.assertAlmostEqual(uploaded.value, 4.5)
        self.assertTrue(math.isnan(missing.value))

    def test_full_upload_returns_zero_warnings(self):
        """Full upload: all T-values present → legform_blue_warnings == 0.

        Simulates a user who has uploaded all test data.  Callers may
        safely allow the upload-lock button because no warnings are present.
        """
        c1 = _make_criteria("Sum of forces", "t", 4.5, row=0, col=1)
        c2 = _make_criteria("Sum of forces", "t", 3.2, row=0, col=2)
        vtu = VruTestData.from_sheet_dict(self._sheet_dict(c1, c2))
        self.assertEqual(vtu.legform_blue_warnings, 0)
        self.assertAlmostEqual(c1.value, 4.5)
        self.assertAlmostEqual(c2.value, 3.2)


# ===========================================================================
# 10 – Uniform / mixed legform sheet fills on a real-template-shaped matrix
#
# Regression coverage for get_legform_matrix's column bound
# (VRU_MATRIX_COL_END_INDEX): the real "CP - VRU Prediction" sheet has a
# blank spacer column and a legend-text column ("T corresponds to ...")
# immediately after the real E:Y (21-column) legform data range. Before the
# fix those two columns were read as part of the matrix (defaulting to
# GREY), inflating it to 23 columns and pairing real data columns with the
# legend text instead of their true mirror column.
# ===========================================================================


class TestLegformRealTemplateShapedFill(unittest.TestCase):
    """Build a DataFrame shaped like the real template (spacer + legend-text
    columns trailing the 21 real data columns) and fill the 3 legform rows
    with all-T, all-ST, all-SA, and a symmetric mixed pattern."""

    _NUM_REAL_COLS = 21  # E:Y

    def _build_df(self, row_values):
        """row_values: list of 3 lists, each with _NUM_REAL_COLS colour strings."""
        ncols = 27  # matches the real template's sheet width (spacer + legend)
        df = pd.DataFrame("grey", index=range(28), columns=range(ncols))
        for i, colours in enumerate(row_values):
            assert len(colours) == self._NUM_REAL_COLS
            df_row = (
                LEGFORMS_START_ROW_INDEX + 2 + i
            )  # matrix rows 0,1,2 -> df rows 25,26,27
            for j, colour in enumerate(colours):
                df.iat[df_row, 4 + j] = colour
            df.iat[df_row, 25] = float("nan")  # blank spacer column (Z)
            df.iat[df_row, 26] = (
                f"{colours[0].upper()} legend text row {i}"  # legend column (AA)
            )
        return df

    def _matrix(self, row_values):
        df = self._build_df(row_values)
        return get_legform_matrix(df)

    # ------------------------------------------------------------------
    # Matrix shape must stay bounded to the 21 real data columns
    # ------------------------------------------------------------------

    def test_matrix_width_excludes_spacer_and_legend_columns(self):
        matrix = self._matrix([["t"] * self._NUM_REAL_COLS] * 3)
        self.assertEqual(len(matrix), 3)
        self.assertEqual(len(matrix[0]), self._NUM_REAL_COLS)

    # ------------------------------------------------------------------
    # All T
    # ------------------------------------------------------------------

    def test_all_t_passes_symmetry_check(self):
        matrix = self._matrix([["t"] * self._NUM_REAL_COLS] * 3)
        check_legform_blue_symmetry(matrix)  # must not raise
        self.assertTrue(all(c == T for row in matrix for c in row))

    # ------------------------------------------------------------------
    # All ST
    # ------------------------------------------------------------------

    def test_all_st_passes_symmetry_check(self):
        matrix = self._matrix([["st"] * self._NUM_REAL_COLS] * 3)
        check_legform_blue_symmetry(matrix)  # must not raise
        self.assertTrue(all(c == ST for row in matrix for c in row))

    # ------------------------------------------------------------------
    # All SA
    # ------------------------------------------------------------------

    def test_all_sa_passes_symmetry_check(self):
        matrix = self._matrix([["sa"] * self._NUM_REAL_COLS] * 3)
        check_legform_blue_symmetry(matrix)  # must not raise
        self.assertTrue(all(c == SA for row in matrix for c in row))

    # ------------------------------------------------------------------
    # Mixed T / ST / SA – symmetric about the centre column, must pass
    # ------------------------------------------------------------------

    def _symmetric_mixed_row(self):
        # 21 columns, centre index 10. Colour depends only on distance from
        # centre, so the row is symmetric by construction: outer -> T,
        # middle -> ST, centre 3 columns -> SA.
        row = []
        for j in range(self._NUM_REAL_COLS):
            distance = abs(j - 10)
            if distance >= 7:
                row.append("t")
            elif distance >= 3:
                row.append("st")
            else:
                row.append("sa")
        return row

    def test_mixed_symmetric_row_passes_symmetry_check(self):
        mixed_row = self._symmetric_mixed_row()
        matrix = self._matrix([mixed_row] * 3)
        check_legform_blue_symmetry(matrix)  # must not raise
        self.assertEqual(matrix[0][0], T)
        self.assertEqual(matrix[0][10], SA)

    # ------------------------------------------------------------------
    # Mixed T / ST / SA – asymmetric must still be caught
    # ------------------------------------------------------------------

    def test_mixed_asymmetric_row_still_raises(self):
        mixed_row = self._symmetric_mixed_row()
        mixed_row[0] = "t"
        mixed_row[-1] = "sa"  # breaks symmetry: T at j=0 mirrors SA at j=20
        matrix = self._matrix([mixed_row] * 3)
        with self.assertRaises(ValueError):
            check_legform_blue_symmetry(matrix)

    # ------------------------------------------------------------------
    # End-to-end: process_legform_test must not crash on a uniformly
    # filled real-shaped sheet (this is the exact scenario that used to
    # raise "Asymmetric T/ST prediction ... has color 'grey'").
    # ------------------------------------------------------------------

    def _vru_params(self):
        return pd.DataFrame(
            {
                "param_code": ["CP - VRU Pelvis Impact", "CP - VRU Leg Impact"],
                "Input parameter": [
                    "Number of verification tests",
                    "Number of verification tests",
                ],
                "Value": [1, 1],
            }
        )

    def test_process_legform_test_end_to_end_all_st(self):
        # Root-cause regression guard: an all-ST declared plan must round-trip
        # through process_legform_test with zero cells dropped (3 rows x 21 cols).
        df = self._build_df([["st"] * self._NUM_REAL_COLS] * 3)
        result = vru_processing.process_legform_test(df, self._vru_params())
        self.assertEqual(len(result), 63)
        self.assertTrue(all(p.color == ST for p in result))

    def test_process_legform_test_end_to_end_all_sa(self):
        # Greenfield: every cell needs promotion, but the total count must be
        # preserved -- no cell dropped, only SA -> ST relabeling for the coin's half.
        df = self._build_df([["sa"] * self._NUM_REAL_COLS] * 3)
        result = vru_processing.process_legform_test(df, self._vru_params())
        self.assertEqual(len(result), 63)
        self.assertTrue(all(p.color in (ST, SA) for p in result))

    def test_process_legform_test_end_to_end_no_orphan_mixed_row(self):
        # From the real fixture cp_template_legform_all_ST.xlsx (legform row,
        # Femur/Knee & Tibia): two SA-SA pairs each flanked by ST -- no orphan,
        # so nothing may be dropped or relabeled anywhere in the pipeline.
        mixed_row = ["st"] * 6 + ["sa", "sa"] + ["st"] * 5 + ["sa", "sa"] + ["st"] * 6
        self.assertEqual(len(mixed_row), self._NUM_REAL_COLS)
        df = self._build_df([mixed_row] * 3)
        result = vru_processing.process_legform_test(df, self._vru_params())
        self.assertEqual(len(result), 63)
        colors = [p.color for p in result]
        self.assertEqual(colors.count(ST), 3 * 17)
        self.assertEqual(colors.count(SA), 3 * 4)


# ===========================================================================
# _resolve_headform_apillar_values – symmetric A-pillar value propagation
# ===========================================================================


def _make_apillar_criteria(prediction, value, row, col):
    """Build an A-pillar HIC15 criteria with the protocol's pass/fail limits."""
    hpl = lpl = 1000.0 if prediction == "green-20" else 1700.0
    c = Criteria(
        name="HIC15", hpl=hpl, lpl=lpl, criteria_type=CriteriaType.VRU_CRITERIA
    )
    c.test_point = common.format_test_point(row, col)
    c.set_prediction(prediction)
    if value is not None and math.isnan(value):
        # Bypass validator to simulate a blank cell read from Excel
        c.__dict__["value"] = float("nan")
    else:
        c.set_value(value)
    return c


def _make_apillar_loadcase(criteria_list, br_name="Adult"):
    br = BodyRegion(name=br_name)
    br._criteria = list(criteria_list)
    dummy = Dummy(name="Adult Headform", body_region_list=[br])
    seat = Seat(name="Driver", dummy=dummy)
    return LoadCase(name="Headform Apillar", seats=[seat])


class TestResolveHeadformApillarValues(unittest.TestCase):
    def test_tested_value_copied_to_untested_same_shade_mirror(self):
        tested = _make_apillar_criteria("green-20", 950.0, row=9, col=8)
        untested = _make_apillar_criteria("green-20", 0.0, row=9, col=-8)
        lc = _make_apillar_loadcase([tested, untested])
        warnings = vru_processing._resolve_headform_apillar_values([lc])
        self.assertEqual(warnings, 0)
        self.assertEqual(untested.value, 950.0)
        self.assertEqual(untested.color, "green")

    def test_nan_value_treated_as_untested(self):
        tested = _make_apillar_criteria("green-30", 1500.0, row=10, col=5)
        untested = _make_apillar_criteria("green-30", float("nan"), row=10, col=-5)
        lc = _make_apillar_loadcase([tested, untested])
        warnings = vru_processing._resolve_headform_apillar_values([lc])
        self.assertEqual(warnings, 0)
        self.assertEqual(untested.value, 1500.0)

    def test_both_untested_pair_warns_and_scores_zero(self):
        a = _make_apillar_criteria("green-20", 0.0, row=9, col=8)
        b = _make_apillar_criteria("green-20", 0.0, row=9, col=-8)
        lc = _make_apillar_loadcase([a, b])
        warnings = vru_processing._resolve_headform_apillar_values([lc])
        self.assertEqual(warnings, 2)
        for criteria in (a, b):
            self.assertTrue(math.isnan(criteria.value))
            self.assertIsNone(criteria.color)
            self.assertEqual(criteria.score, 0.0)

    def test_untested_run_with_different_shade_mirror_warns_and_scores_zero(self):
        """A green-30 run is not symmetric to a green-20 run: no copy allowed,
        each must be tested individually."""
        tested = _make_apillar_criteria("green-20", 900.0, row=9, col=8)
        untested = _make_apillar_criteria("green-30", 0.0, row=9, col=-8)
        lc = _make_apillar_loadcase([tested, untested])
        warnings = vru_processing._resolve_headform_apillar_values([lc])
        self.assertEqual(warnings, 1)
        self.assertTrue(math.isnan(untested.value))
        self.assertEqual(untested.score, 0.0)
        self.assertEqual(tested.value, 900.0)

    def test_untested_centre_column_run_warns(self):
        """col == 0 is its own mirror: it must be tested itself."""
        centre = _make_apillar_criteria("green-40", 0.0, row=18, col=0)
        lc = _make_apillar_loadcase([centre])
        warnings = vru_processing._resolve_headform_apillar_values([lc])
        self.assertEqual(warnings, 1)
        self.assertTrue(math.isnan(centre.value))

    def test_both_tested_values_kept_independently(self):
        """A pair physically tested on both sides keeps both values."""
        a = _make_apillar_criteria("green-20", 900.0, row=9, col=8)
        b = _make_apillar_criteria("green-20", 990.0, row=9, col=-8)
        lc = _make_apillar_loadcase([a, b])
        warnings = vru_processing._resolve_headform_apillar_values([lc])
        self.assertEqual(warnings, 0)
        self.assertEqual(a.value, 900.0)
        self.assertEqual(b.value, 990.0)

    def test_headform_loadcase_not_touched(self):
        """Only 'Headform Apillar' loadcases are resolved: a regular Headform
        verification run with a 0.0 value is out of scope here."""
        criteria = _make_vru_criteria("green", 0.0, row=5, col=3)
        lc = _make_headform_loadcase([criteria], loadcase_name="Headform")
        warnings = vru_processing._resolve_headform_apillar_values([lc])
        self.assertEqual(warnings, 0)
        self.assertEqual(criteria.value, 0.0)

    def test_correction_factor_unaffected_by_mirror_copy(self):
        """The Apillar loadcase is excluded from get_vru_factors, so resolving
        mirror values must never move the correction factor."""
        headform_lc = _make_headform_loadcase(
            [_make_vru_criteria("green", 500.0, row=5, col=3)]
        )
        tested = _make_apillar_criteria("green-20", 950.0, row=9, col=8)
        untested = _make_apillar_criteria("green-20", 0.0, row=9, col=-8)
        apillar_lc = _make_apillar_loadcase([tested, untested])

        factors_before = vru_processing.get_vru_factors([headform_lc, apillar_lc])
        vru_processing._resolve_headform_apillar_values([headform_lc, apillar_lc])
        factors_after = vru_processing.get_vru_factors([headform_lc, apillar_lc])
        self.assertEqual(factors_before, factors_after)

    def test_from_sheet_dict_integration_scores_refreshed_with_copied_value(self):
        """The resolver runs before the score refresh in from_sheet_dict, so a
        copied passing HIC15 must come out green/100 and a must-test-missing
        run must come out score 0."""
        tested = _make_apillar_criteria("green-20", 950.0, row=9, col=8)
        untested = _make_apillar_criteria("green-20", 0.0, row=9, col=-8)
        orphan = _make_apillar_criteria("green-40", 0.0, row=18, col=0)
        lc = _make_apillar_loadcase([tested, untested, orphan])

        vtd = VruTestData.from_sheet_dict({"CP - VRU Head Impact": [lc]})

        self.assertEqual(vtd.headform_apillar_warnings, 1)
        self.assertEqual(untested.value, 950.0)
        self.assertEqual(untested.score, 100.0)
        self.assertEqual(tested.score, 100.0)
        self.assertEqual(orphan.score, 0.0)


# ===========================================================================
# Stratified headform sampling
# ===========================================================================


class TestAllocateBandQuotas(unittest.TestCase):
    def test_zero_total_returns_all_zero(self):
        alloc = vru_processing._allocate_band_quotas(
            {"Cyclist": 10, "Adult": 10, "Child": 10}, 0
        )
        self.assertEqual(sum(alloc.values()), 0)

    def test_min_one_per_band_when_total_allows(self):
        alloc = vru_processing._allocate_band_quotas(
            {"Cyclist": 100, "Adult": 1, "Child": 1}, 3
        )
        self.assertEqual(alloc, {"Cyclist": 1, "Adult": 1, "Child": 1})

    def test_proportional_split(self):
        alloc = vru_processing._allocate_band_quotas(
            {"Cyclist": 60, "Adult": 92, "Child": 74}, 10
        )
        self.assertEqual(sum(alloc.values()), 10)
        self.assertTrue(all(quota >= 1 for quota in alloc.values()))
        self.assertGreaterEqual(alloc["Adult"], alloc["Cyclist"])

    def test_fewer_points_than_bands_prefers_largest(self):
        alloc = vru_processing._allocate_band_quotas(
            {"Cyclist": 5, "Adult": 50, "Child": 20}, 2
        )
        self.assertEqual(sum(alloc.values()), 2)
        self.assertEqual(alloc["Adult"], 1)
        self.assertEqual(alloc["Child"], 1)
        self.assertEqual(alloc["Cyclist"], 0)

    def test_request_covering_everything_returns_sizes(self):
        sizes = {"Cyclist": 2, "Adult": 3}
        self.assertEqual(vru_processing._allocate_band_quotas(sizes, 5), sizes)

    def test_quota_never_exceeds_band_size(self):
        alloc = vru_processing._allocate_band_quotas(
            {"Cyclist": 2, "Adult": 100, "Child": 2}, 50
        )
        self.assertEqual(sum(alloc.values()), 50)
        self.assertLessEqual(alloc["Cyclist"], 2)
        self.assertLessEqual(alloc["Child"], 2)


class TestStratifiedHeadformSample(unittest.TestCase):
    def _cells(self, matrix_rows, per_row):
        """Build (i, j, color) cells: per_row green cells in each matrix row."""
        return [
            (i, j, VruPredictionColor.GREEN)
            for i in matrix_rows
            for j in range(per_row)
        ]

    def test_every_band_with_eligible_cells_is_sampled(self):
        # Cyclist rows i 0-6, Adult i 7-12, Child i 13-18
        cells = self._cells([0, 1], 8) + self._cells([8, 9], 8) + self._cells([15], 8)
        for _ in range(20):
            sampled = vru_processing._stratified_headform_sample(cells, 5)
            self.assertEqual(len(sampled), 5)
            bands = {vru_processing.headform_body_region(18 - i) for i, _, _ in sampled}
            self.assertEqual(bands, {"Cyclist", "Adult", "Child"})

    def test_zero_requested_returns_empty(self):
        self.assertEqual(
            vru_processing._stratified_headform_sample(self._cells([0], 5), 0), []
        )

    def test_request_covering_everything_returns_all(self):
        cells = self._cells([0, 8, 15], 3)
        sampled = vru_processing._stratified_headform_sample(cells, 100)
        self.assertEqual(sorted(sampled), sorted(cells))

    def test_single_band_grid(self):
        cells = self._cells([8, 9], 6)  # Adult only
        sampled = vru_processing._stratified_headform_sample(cells, 4)
        self.assertEqual(len(sampled), 4)
        self.assertTrue(
            all(
                vru_processing.headform_body_region(18 - i) == "Adult"
                for i, _, _ in sampled
            )
        )


# ===========================================================================
# Untested WAD bands keep their predicted score
# ===========================================================================


def _grid_points(spec):
    """Build vru_test_points_all from {(row, col): color} (point coordinates)."""
    return [
        vru_processing.VruTestPoint(row=row, col=col, color=color)
        for (row, col), color in spec.items()
    ]


class TestUntestedBandKeepsPredictedScore(unittest.TestCase):
    """compute_vru_bodyregion_score with a band that has coloured grid points
    but no verification row: per protocol §4.4.2.2 its predicted score stands
    and the correction factor of the tested regions applies."""

    def _build_vtd(self, grid, headform_criteria):
        vtd = VruTestData()
        vtd.headform_loadcases = [_make_headform_loadcase(headform_criteria)]
        vtd.vru_test_points_all = _grid_points(grid)
        return vtd

    def _grid_all_three_bands(self):
        # Cyclist rows 12-18, Adult 6-11, Child 0-5; green weight 1.0 each
        return {
            (14, 1): VruPredictionColor.GREEN,
            (14, 2): VruPredictionColor.GREEN,
            (8, 1): VruPredictionColor.GREEN,
            (8, 2): VruPredictionColor.GREEN,
            (2, 1): VruPredictionColor.GREEN,
            (2, 2): VruPredictionColor.GREEN,
        }

    def test_untested_bands_stay_in_numerator_and_denominator(self):
        # Only Adult was verified (CF = 1.0); Cyclist and Child have coloured
        # cells but no verification row.
        criteria = _make_vru_criteria("green", 100.0, row=8, col=1)
        vtd = self._build_vtd(self._grid_all_three_bands(), [criteria])

        vtd.compute_vru_bodyregion_score()

        headform_lc = next(lc for lc in vtd.headform_loadcases if lc.name == "Headform")
        regions = {
            br.name: br
            for seat in headform_lc.seats
            for br in seat.dummy.body_region_list
        }
        self.assertEqual(set(regions), {"Cyclist", "Adult", "Child"})
        for name in ("Cyclist", "Adult", "Child"):
            self.assertAlmostEqual(regions[name].bodyregion_score, 100.0, places=5)
            # 2 max points per band, 6 total, 10-point budget split evenly
            self.assertAlmostEqual(regions[name].max_score, 2 / 6 * 10.0, places=5)

    def test_band_without_coloured_cells_stays_excluded(self):
        # Child band entirely grey: not assessed - no score, no denominator share.
        grid = {
            (14, 1): VruPredictionColor.GREEN,
            (8, 1): VruPredictionColor.GREEN,
            (2, 1): VruPredictionColor.GREY,
        }
        criteria = _make_vru_criteria("green", 100.0, row=8, col=1)
        vtd = self._build_vtd(grid, [criteria])

        vtd.compute_vru_bodyregion_score()

        headform_lc = next(lc for lc in vtd.headform_loadcases if lc.name == "Headform")
        regions = {
            br.name: br
            for seat in headform_lc.seats
            for br in seat.dummy.body_region_list
        }
        self.assertIsNone(regions["Child"].bodyregion_score)
        self.assertIsNone(regions["Child"].max_score)
        # Cyclist and Adult split the full 10 points between them.
        self.assertAlmostEqual(regions["Cyclist"].max_score, 5.0, places=5)
        self.assertAlmostEqual(regions["Adult"].max_score, 5.0, places=5)

    def test_no_headform_rows_at_all_keeps_current_behaviour(self):
        vtd = VruTestData()
        vtd.vru_test_points_all = _grid_points(self._grid_all_three_bands())
        vtd.compute_vru_bodyregion_score()
        self.assertEqual(vtd.headform_loadcases, [])

    def test_original_seats_preserved_as_raw_seats(self):
        """The verification sheet keeps matching through the raw-seats id after
        the loadcase tree is extended."""
        criteria = _make_vru_criteria("green", 100.0, row=8, col=1)
        vtd = self._build_vtd(self._grid_all_three_bands(), [criteria])
        original_seats = list(vtd.headform_loadcases[0].seats)

        vtd.compute_vru_bodyregion_score()

        headform_lc = vtd.headform_loadcases[0]
        self.assertEqual(headform_lc.raw_seats, original_seats)
        self.assertEqual(
            [(s.name, s.dummy.name) for s in headform_lc.seats],
            [("Driver", "Adult Headform"), ("Driver", "Child Headform")],
        )


if __name__ == "__main__":
    unittest.main()
