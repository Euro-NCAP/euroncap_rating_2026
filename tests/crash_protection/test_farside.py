# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import unittest
import numpy as np
import pandas as pd

from euroncap_rating_2026.crash_protection.body_region import BodyRegion
from euroncap_rating_2026.crash_protection.criteria import (
    ColorCriteria,
    Criteria,
    CriteriaType,
)
from euroncap_rating_2026.crash_protection.compute_score import update_loadcase
from euroncap_rating_2026.crash_protection.data_loader import (
    apply_farside_capping_rule,
    load_data,
)
from euroncap_rating_2026.crash_protection.dummy import Dummy
from euroncap_rating_2026.crash_protection.load_case import LoadCase
from euroncap_rating_2026.crash_protection.seat import Seat


# ---------------------------------------------------------------------------
# Helpers shared by integration tests
# ---------------------------------------------------------------------------


def _lc_rows(lc_name: str, farside_color: str) -> list:
    """Two-row block representing one loadcase in the CP - Side Farside sheet."""
    return [
        {
            "Loadcase": lc_name,
            "Seat position": "Driver",
            "Dummy": "WorldSID-50-Farside",
            "Body region": "Pelvis",
            "Criteria": "HIC36",
            "HPL": 500.0,
            "LPL": 700.0,
            "Capping": np.nan,
            "Value": 600.0,
            "Modifier": np.nan,
        },
        {
            "Loadcase": np.nan,
            "Seat position": np.nan,
            "Dummy": np.nan,
            "Body region": np.nan,
            "Criteria": "Farside excursion",
            "HPL": np.nan,
            "LPL": np.nan,
            "Capping": np.nan,
            "Value": farside_color,
            "Modifier": np.nan,
        },
    ]


def _make_dfs(
    aemd_main_color: str = "green",
    aemd_robustness_colors: list = None,
    pole_main_color: str = "green",
    pole_robustness_colors: list = None,
    countermeasure: bool = False,
    redline: bool = False,
) -> dict:
    """
    Build the minimal dict of DataFrames that load_data() requires to exercise
    the CP - Side Farside logic.

    The helper always provides both AEMD and Pole variants because Stage 4
    reconstruction calls min() over each set unconditionally and would raise
    ValueError with an empty sequence.
    """
    if aemd_robustness_colors is None:
        aemd_robustness_colors = ["yellow"]
    if pole_robustness_colors is None:
        pole_robustness_colors = ["yellow"]

    # ---- CP - Side Farside ------------------------------------------------
    rows = []
    rows.extend(_lc_rows("Main-AEMD", aemd_main_color))

    aemd_lc_names = []
    for idx, color in enumerate(aemd_robustness_colors, start=1):
        name = (
            "Robustness-AEMDB"
            if len(aemd_robustness_colors) == 1
            else f"Robustness-AEMDB-{idx}"
        )
        aemd_lc_names.append(name)
        rows.extend(_lc_rows(name, color))

    rows.extend(_lc_rows("Main-Pole", pole_main_color))

    pole_lc_names = []
    for idx, color in enumerate(pole_robustness_colors, start=1):
        name = (
            "Robustness-Pole"
            if len(pole_robustness_colors) == 1
            else f"Robustness-Pole-{idx}"
        )
        pole_lc_names.append(name)
        rows.extend(_lc_rows(name, color))

    farside_df = pd.DataFrame(rows)

    # ---- Input parameters -------------------------------------------------
    input_params_df = pd.DataFrame(
        [
            {
                "Stage": "Crash Protection",
                "Stage element": "Side Impact",
                "Stage subelement": "Farside",
                "Input parameter": "Countermeasure?",
                "Value": countermeasure,
            },
            {
                "Stage": np.nan,
                "Stage element": np.nan,
                "Stage subelement": np.nan,
                "Input parameter": "Red line >125 mm outboard of the orange line",
                "Value": redline,
            },
        ]
    )

    # ---- CP - Body region scores ------------------------------------------
    # Rows for every parsed load case so the body-region-scores loop can match
    # them and set bodyregion_score / max_score / inspection.
    # For multi-variant robustness names (e.g. "Robustness-AEMDB-1"), an extra
    # exact-name row ("Robustness-AEMDB" / "Robustness-Pole") is added to
    # capture farside_virtual_aemb/pole_max_score and inspection needed by
    # the Stage-4 reconstruction.
    br_rows = []
    all_lc_pairs = (
        [("Main-AEMD", "WorldSID-50-Farside")]
        + [(n, "WorldSID-50-Farside") for n in aemd_lc_names]
        + [("Main-Pole", "WorldSID-50-Farside")]
        + [(n, "WorldSID-50-Farside") for n in pole_lc_names]
    )
    for i, (lc_name, dummy_name) in enumerate(all_lc_pairs):
        br_rows.append(
            {
                "Stage element": "Side" if i == 0 else np.nan,
                "Stage subelement": "Farside" if i == 0 else np.nan,
                "Loadcase": lc_name,
                "Seat position": "Driver",
                "Dummy": dummy_name,
                "Body region": "Pelvis",
                "Max score": 5.0,
                "Inspection [%]": 0.0,
            }
        )

    # If multi-variant names are used, the exact "Robustness-AEMDB" /
    # "Robustness-Pole" names won't appear in all_lc_pairs, so we add
    # extra rows purely to trigger farside_virtual_*_max_score capture.
    if "Robustness-AEMDB" not in aemd_lc_names:
        br_rows.append(
            {
                "Stage element": np.nan,
                "Stage subelement": np.nan,
                "Loadcase": "Robustness-AEMDB",
                "Seat position": "Driver",
                "Dummy": "WorldSID-50-Farside",
                "Body region": "Pelvis",
                "Max score": 5.0,
                "Inspection [%]": 0.0,
            }
        )
    if "Robustness-Pole" not in pole_lc_names:
        br_rows.append(
            {
                "Stage element": np.nan,
                "Stage subelement": np.nan,
                "Loadcase": "Robustness-Pole",
                "Seat position": "Driver",
                "Dummy": "WorldSID-50-Farside",
                "Body region": "Pelvis",
                "Max score": 5.0,
                "Inspection [%]": 0.0,
            }
        )

    br_scores_df = pd.DataFrame(br_rows)

    # ---- CP - Dummy Scores (empty – dummy-level scoring is not exercised) --
    dummy_scores_df = pd.DataFrame(
        columns=[
            "Stage element",
            "Stage subelement",
            "Loadcase",
            "Seat position",
            "Dummy",
            "Max score",
        ]
    )

    # ---- Test Scores – one row so test_score_inspections is never empty ----
    # inspection=0.0 means no points are deducted from the Farside sheet.
    test_scores_df = pd.DataFrame(
        [
            {
                "Stage": "Crash Protection",
                "Stage element": "Side Impact",
                "Stage subelement": "Farside",
                "Inspection [%]": 0.0,
            }
        ]
    )

    return {
        "CP - Side Farside": farside_df,
        "Input parameters": input_params_df,
        "CP - Body region scores": br_scores_df,
        "CP - Dummy Scores": dummy_scores_df,
        "Test Scores": test_scores_df,
    }


def _get_reconstructed_lc(sheet_dict: dict, lc_name: str):
    """Return the reconstructed load case from the farside sheet_dict."""
    for lc in sheet_dict["CP - Side Farside"]:
        if lc.name == lc_name:
            return lc
    raise AssertionError(f"Load case '{lc_name}' not found in sheet_dict")


def _get_head_br(sheet_dict: dict, lc_name: str):
    """Return the Head body region of the Driver/WorldSID-50 in a reconstructed LC."""
    lc = _get_reconstructed_lc(sheet_dict, lc_name)
    seat = lc.get_seat("Driver")
    return seat.dummy.get_body_region("Head")


def _make_dfs_with_pole32_front_pair(
    driver_value: float,
    front_passenger_value: float,
    lpl: float = 700.0,
) -> dict:
    dfs = _make_dfs()

    pole32_rows = [
        {
            "Loadcase": "Pole-32",
            "Seat position": "Driver",
            "Dummy": "WorldSID-50-Farside",
            "Body region": "Head",
            "Criteria": "HIC15",
            "HPL": 500.0,
            "LPL": lpl,
            "Capping": np.nan,
            "Value": driver_value,
            "Modifier": np.nan,
        },
        {
            "Loadcase": np.nan,
            "Seat position": "Front Passenger",
            "Dummy": "WorldSID-50-Farside",
            "Body region": "Head",
            "Criteria": "HIC15",
            "HPL": 500.0,
            "LPL": lpl,
            "Capping": np.nan,
            "Value": front_passenger_value,
            "Modifier": np.nan,
        },
    ]
    dfs["CP - Side Farside"] = pd.concat(
        [dfs["CP - Side Farside"], pd.DataFrame(pole32_rows)],
        ignore_index=True,
    )

    pole32_bodyregion_row = pd.DataFrame(
        [
            {
                "Stage element": np.nan,
                "Stage subelement": np.nan,
                "Loadcase": "Pole-32",
                "Seat position": "Driver and Front Passenger",
                "Dummy": "WorldSID-50-Farside",
                "Body region": "Head",
                "Max score": 2.0,
                "Inspection [%]": 0.0,
            }
        ]
    )
    dfs["CP - Body region scores"] = pd.concat(
        [dfs["CP - Body region scores"], pole32_bodyregion_row],
        ignore_index=True,
    )

    dfs["CP - Dummy Scores"] = pd.DataFrame(
        [
            {
                "Stage element": "Side",
                "Stage subelement": "Farside",
                "Loadcase": "Pole-32",
                "Seat position": "Driver and Front Passenger",
                "Dummy": "WorldSID-50-Farside",
                "Max score": 2.0,
            }
        ]
    )
    return dfs


# ===========================================================================
# 1. ColorCriteria – unit tests (no load_data dependency)
# ===========================================================================


class TestColorCriteria(unittest.TestCase):
    """Unit tests for the ColorCriteria color-to-score mapping."""

    # -- No countermeasure --------------------------------------------------

    def test_green_no_countermeasure(self):
        cc = ColorCriteria(
            name="Farside excursion",
            color="green",
            countermeasure=False,
            redline_above_125mm=False,
        )
        self.assertEqual(cc.score, 100.0)

    def test_yellow_no_countermeasure(self):
        cc = ColorCriteria(
            name="Farside excursion",
            color="yellow",
            countermeasure=False,
            redline_above_125mm=False,
        )
        self.assertEqual(cc.score, 50.0)

    def test_orange_no_countermeasure(self):
        cc = ColorCriteria(
            name="Farside excursion",
            color="orange",
            countermeasure=False,
            redline_above_125mm=False,
        )
        self.assertEqual(cc.score, 25.0)

    def test_red_no_countermeasure(self):
        cc = ColorCriteria(
            name="Farside excursion",
            color="red",
            countermeasure=False,
            redline_above_125mm=False,
        )
        self.assertEqual(cc.score, 0.0)

    def test_capping_no_countermeasure(self):
        # "capping" (the lowest severity level) falls through the else branch → 0.0
        cc = ColorCriteria(
            name="Farside excursion",
            color="capping",
            countermeasure=False,
            redline_above_125mm=False,
        )
        self.assertEqual(cc.score, 0.0)

    # -- With countermeasure, redline=False ---------------------------------

    def test_green_with_countermeasure(self):
        cc = ColorCriteria(
            name="Farside excursion",
            color="green",
            countermeasure=True,
            redline_above_125mm=False,
        )
        self.assertEqual(cc.score, 100.0)

    def test_yellow_with_countermeasure(self):
        cc = ColorCriteria(
            name="Farside excursion",
            color="yellow",
            countermeasure=True,
            redline_above_125mm=False,
        )
        self.assertEqual(cc.score, 100.0)

    def test_orange_with_countermeasure(self):
        cc = ColorCriteria(
            name="Farside excursion",
            color="orange",
            countermeasure=True,
            redline_above_125mm=False,
        )
        self.assertEqual(cc.score, 75.0)

    def test_red_with_countermeasure_no_redline(self):
        cc = ColorCriteria(
            name="Farside excursion",
            color="red",
            countermeasure=True,
            redline_above_125mm=False,
        )
        self.assertEqual(cc.score, 25.0)

    # -- With countermeasure AND redline ------------------------------------

    def test_red_with_countermeasure_with_redline(self):
        cc = ColorCriteria(
            name="Farside excursion",
            color="red",
            countermeasure=True,
            redline_above_125mm=True,
        )
        self.assertEqual(cc.score, 50.0)

    # -- Color stored correctly ---------------------------------------------

    def test_color_is_stored_as_given(self):
        for color in ("green", "yellow", "orange", "capping", "red"):
            with self.subTest(color=color):
                cc = ColorCriteria(
                    name="Farside excursion",
                    color=color,
                    countermeasure=False,
                    redline_above_125mm=False,
                )
                self.assertEqual(cc.color, color)


# ===========================================================================
# 2. Robustness color comparison – unit tests (logic replicated as helper)
# ===========================================================================


class TestFarsideRobustnessComparison(unittest.TestCase):
    """
    Tests for the PASS/FAIL robustness color-comparison logic embedded in
    load_data().  The comparison condition is replicated here so it can be
    tested as a pure function without DataFrame overhead.
    """

    PASS = 100.0
    FAIL = 0.0

    def _compare(self, main_color: str, robustness_color: str) -> float:
        """Mirror of the comparison block in load_data."""
        if main_color == robustness_color or (
            (main_color == "green" and robustness_color == "yellow")
            or (main_color == "yellow" and robustness_color == "green")
            or (main_color == "yellow" and robustness_color == "orange")
            or (main_color == "orange" and robustness_color == "yellow")
            or (main_color == "orange" and robustness_color == "red")
            or (main_color == "red" and robustness_color == "orange")
        ):
            return self.PASS
        return self.FAIL

    # -- Same-color pairs (always PASS) ------------------------------------

    def test_same_green(self):
        self.assertEqual(self._compare("green", "green"), self.PASS)

    def test_same_yellow(self):
        self.assertEqual(self._compare("yellow", "yellow"), self.PASS)

    def test_same_orange(self):
        self.assertEqual(self._compare("orange", "orange"), self.PASS)

    def test_same_red(self):
        self.assertEqual(self._compare("red", "red"), self.PASS)

    # -- Adjacent pairs → PASS ---------------------------------------------

    def test_green_to_yellow_passes(self):
        self.assertEqual(self._compare("green", "yellow"), self.PASS)

    def test_yellow_to_green_passes(self):
        self.assertEqual(self._compare("yellow", "green"), self.PASS)

    def test_yellow_to_orange_passes(self):
        self.assertEqual(self._compare("yellow", "orange"), self.PASS)

    def test_orange_to_yellow_passes(self):
        self.assertEqual(self._compare("orange", "yellow"), self.PASS)

    def test_orange_to_red_passes(self):
        self.assertEqual(self._compare("orange", "red"), self.PASS)

    def test_red_to_orange_passes(self):
        self.assertEqual(self._compare("red", "orange"), self.PASS)

    # -- Non-adjacent / skipped-step pairs → FAIL --------------------------

    def test_green_to_orange_fails(self):
        self.assertEqual(self._compare("green", "orange"), self.FAIL)

    def test_green_to_red_fails(self):
        self.assertEqual(self._compare("green", "red"), self.FAIL)

    def test_green_to_brown_fails(self):
        self.assertEqual(self._compare("green", "brown"), self.FAIL)

    def test_yellow_to_red_fails(self):
        self.assertEqual(self._compare("yellow", "red"), self.FAIL)

    def test_yellow_to_brown_fails(self):
        self.assertEqual(self._compare("yellow", "brown"), self.FAIL)

    def test_orange_to_green_fails(self):
        self.assertEqual(self._compare("orange", "green"), self.FAIL)

    def test_red_to_green_fails(self):
        self.assertEqual(self._compare("red", "green"), self.FAIL)

    def test_red_to_yellow_fails(self):
        self.assertEqual(self._compare("red", "yellow"), self.FAIL)


# ===========================================================================
# 3. load_data integration – farside-specific behaviour
# ===========================================================================


class TestFarsideLoadData(unittest.TestCase):
    """
    Integration tests that call load_data() with minimal DataFrames and
    verify farside-specific behaviour in the returned sheet_dict.

    All assertions target:
    - body_region.get_bodyregion_score() on parsed Main-* load cases
    - body_region.get_bodyregion_score() / get_score() on the Stage-4
      reconstructed Robustness-AEMDB / Robustness-Pole load cases
    - the structure (names, count) of load cases in sheet_dict after
      Stage-4 reconstruction
    """

    # -----------------------------------------------------------------------
    # Main-test load case scoring via ColorCriteria
    # -----------------------------------------------------------------------

    def test_main_aemd_green_no_countermeasure_sets_correct_bodyregion_score(self):
        dfs = _make_dfs(aemd_main_color="green", countermeasure=False)
        sheet_dict, _ = load_data(dfs)

        main_aemd = _get_reconstructed_lc(sheet_dict, "Main-AEMD")
        seat = main_aemd.get_seat("Driver")
        pelvis = seat.dummy.get_body_region("Pelvis")
        self.assertEqual(pelvis.get_bodyregion_score(), 100.0)

    def test_main_aemd_yellow_no_countermeasure_sets_correct_bodyregion_score(self):
        dfs = _make_dfs(aemd_main_color="yellow", countermeasure=False)
        sheet_dict, _ = load_data(dfs)

        main_aemd = _get_reconstructed_lc(sheet_dict, "Main-AEMD")
        pelvis = main_aemd.get_seat("Driver").dummy.get_body_region("Pelvis")
        self.assertEqual(pelvis.get_bodyregion_score(), 50.0)

    def test_main_aemd_orange_no_countermeasure_sets_correct_bodyregion_score(self):
        dfs = _make_dfs(aemd_main_color="orange", countermeasure=False)
        sheet_dict, _ = load_data(dfs)

        pelvis = (
            _get_reconstructed_lc(sheet_dict, "Main-AEMD")
            .get_seat("Driver")
            .dummy.get_body_region("Pelvis")
        )
        self.assertEqual(pelvis.get_bodyregion_score(), 25.0)

    def test_main_pole_yellow_with_countermeasure_sets_100(self):
        dfs = _make_dfs(pole_main_color="yellow", countermeasure=True)
        sheet_dict, _ = load_data(dfs)

        pelvis = (
            _get_reconstructed_lc(sheet_dict, "Main-Pole")
            .get_seat("Driver")
            .dummy.get_body_region("Pelvis")
        )
        self.assertEqual(pelvis.get_bodyregion_score(), 100.0)

    def test_main_aemd_red_with_countermeasure_and_redline_sets_50(self):
        dfs = _make_dfs(aemd_main_color="red", countermeasure=True, redline=True)
        sheet_dict, _ = load_data(dfs)

        pelvis = (
            _get_reconstructed_lc(sheet_dict, "Main-AEMD")
            .get_seat("Driver")
            .dummy.get_body_region("Pelvis")
        )
        self.assertEqual(pelvis.get_bodyregion_score(), 50.0)

    # -----------------------------------------------------------------------
    # Robustness PASS / FAIL – single variant
    # -----------------------------------------------------------------------

    def test_robustness_aemd_adjacent_color_pass(self):
        """Main=green, robustness=yellow → PASS → bodyregion_score 100.0."""
        dfs = _make_dfs(
            aemd_main_color="green",
            aemd_robustness_colors=["yellow"],
        )
        sheet_dict, _ = load_data(dfs)

        head = _get_head_br(sheet_dict, "Robustness-AEMDB")
        self.assertEqual(head.get_bodyregion_score(), 100.0)

    def test_robustness_aemd_non_adjacent_color_fail(self):
        """Main=green, robustness=orange → FAIL → bodyregion_score 0.0."""
        dfs = _make_dfs(
            aemd_main_color="green",
            aemd_robustness_colors=["orange"],
        )
        sheet_dict, _ = load_data(dfs)

        head = _get_head_br(sheet_dict, "Robustness-AEMDB")
        self.assertEqual(head.get_bodyregion_score(), 0.0)

    def test_robustness_aemd_same_color_pass(self):
        """Main=orange, robustness=orange → PASS."""
        dfs = _make_dfs(
            aemd_main_color="orange",
            aemd_robustness_colors=["orange"],
        )
        sheet_dict, _ = load_data(dfs)

        head = _get_head_br(sheet_dict, "Robustness-AEMDB")
        self.assertEqual(head.get_bodyregion_score(), 100.0)

    def test_robustness_pole_adjacent_color_pass(self):
        """Main=yellow, robustness=orange → PASS → bodyregion_score 100.0."""
        dfs = _make_dfs(
            pole_main_color="yellow",
            pole_robustness_colors=["orange"],
        )
        sheet_dict, _ = load_data(dfs)

        head = _get_head_br(sheet_dict, "Robustness-Pole")
        self.assertEqual(head.get_bodyregion_score(), 100.0)

    def test_robustness_pole_non_adjacent_color_fail(self):
        """Main=green, robustness=red → FAIL → bodyregion_score 0.0."""
        dfs = _make_dfs(
            pole_main_color="green",
            pole_robustness_colors=["red"],
        )
        sheet_dict, _ = load_data(dfs)

        head = _get_head_br(sheet_dict, "Robustness-Pole")
        self.assertEqual(head.get_bodyregion_score(), 0.0)

    # -----------------------------------------------------------------------
    # Final score computation on reconstructed load cases (with max/inspection)
    # -----------------------------------------------------------------------

    def test_robustness_aemd_pass_final_score(self):
        """PASS → bodyregion=100, max_score=5, inspection=0 → score=5.0."""
        dfs = _make_dfs(
            aemd_main_color="green",
            aemd_robustness_colors=["yellow"],
        )
        sheet_dict, _ = load_data(dfs)

        head = _get_head_br(sheet_dict, "Robustness-AEMDB")
        self.assertAlmostEqual(head.get_score(), 5.0)

    def test_robustness_aemd_fail_final_score(self):
        """FAIL → bodyregion=0, max_score=5, inspection=0 → score=0.0."""
        dfs = _make_dfs(
            aemd_main_color="green",
            aemd_robustness_colors=["orange"],
        )
        sheet_dict, _ = load_data(dfs)

        head = _get_head_br(sheet_dict, "Robustness-AEMDB")
        self.assertAlmostEqual(head.get_score(), 0.0)

    # -----------------------------------------------------------------------
    # Multi-variant robustness → min() applied
    # -----------------------------------------------------------------------

    def test_robustness_aemd_min_of_two_variants_when_one_fails(self):
        """
        Two AEMDB variants: one PASS (100) and one FAIL (0) → min=0 →
        reconstructed bodyregion_score must be 0.
        """
        dfs = _make_dfs(
            aemd_main_color="green",
            aemd_robustness_colors=["yellow", "orange"],  # PASS, FAIL
        )
        sheet_dict, _ = load_data(dfs)

        head = _get_head_br(sheet_dict, "Robustness-AEMDB")
        self.assertEqual(head.get_bodyregion_score(), 0.0)

    def test_robustness_aemd_min_of_two_variants_when_both_pass(self):
        """Two AEMDB variants both PASS → min=100 → bodyregion_score=100."""
        dfs = _make_dfs(
            aemd_main_color="green",
            aemd_robustness_colors=["yellow", "green"],  # PASS, PASS
        )
        sheet_dict, _ = load_data(dfs)

        head = _get_head_br(sheet_dict, "Robustness-AEMDB")
        self.assertEqual(head.get_bodyregion_score(), 100.0)

    def test_robustness_pole_min_of_two_variants_when_one_fails(self):
        dfs = _make_dfs(
            pole_main_color="orange",
            pole_robustness_colors=[
                "yellow",
                "green",
            ],  # PASS, FAIL (green is 2 steps from orange)
        )
        sheet_dict, _ = load_data(dfs)

        head = _get_head_br(sheet_dict, "Robustness-Pole")
        self.assertEqual(head.get_bodyregion_score(), 0.0)

    # -----------------------------------------------------------------------
    # Stage-4 reconstruction: old Robustness-* load cases are replaced
    # -----------------------------------------------------------------------

    def test_stage4_produces_exactly_one_robustness_aemdb_loadcase(self):
        """Regardless of how many variants in the sheet, Stage 4 always
        produces exactly one 'Robustness-AEMDB' load case."""
        dfs = _make_dfs(
            aemd_main_color="green",
            aemd_robustness_colors=["yellow", "orange"],
        )
        sheet_dict, _ = load_data(dfs)

        robustness_aemdb_lcs = [
            lc
            for lc in sheet_dict["CP - Side Farside"]
            if lc.name == "Robustness-AEMDB"
        ]
        self.assertEqual(len(robustness_aemdb_lcs), 1)

    def test_stage4_produces_exactly_one_robustness_pole_loadcase(self):
        dfs = _make_dfs(
            pole_main_color="yellow",
            pole_robustness_colors=["green", "orange"],
        )
        sheet_dict, _ = load_data(dfs)

        pole_lcs = [
            lc for lc in sheet_dict["CP - Side Farside"] if lc.name == "Robustness-Pole"
        ]
        self.assertEqual(len(pole_lcs), 1)

    def test_stage4_no_raw_robustness_lcs_remain(self):
        """Raw 'Robustness-AEMDB-1', 'Robustness-AEMDB-2', etc. are removed."""
        dfs = _make_dfs(
            aemd_main_color="green",
            aemd_robustness_colors=["yellow", "orange"],
            pole_robustness_colors=["yellow", "orange"],
        )
        sheet_dict, _ = load_data(dfs)

        lc_names = [lc.name for lc in sheet_dict["CP - Side Farside"]]
        self.assertNotIn("Robustness-AEMDB-1", lc_names)
        self.assertNotIn("Robustness-AEMDB-2", lc_names)
        self.assertNotIn("Robustness-Pole-1", lc_names)
        self.assertNotIn("Robustness-Pole-2", lc_names)

    def test_stage4_reconstructed_lcs_have_worldsid50_dummy(self):
        """Stage-4 builds new dummies named 'WorldSID-50' (not -Farside)."""
        dfs = _make_dfs()
        sheet_dict, _ = load_data(dfs)

        for lc_name in ("Robustness-AEMDB", "Robustness-Pole"):
            lc = _get_reconstructed_lc(sheet_dict, lc_name)
            seat = lc.get_seat("Driver")
            self.assertEqual(seat.dummy.name, "WorldSID-50")

    def test_stage4_reconstructed_lcs_have_head_body_region(self):
        """Stage-4 rebuilt load cases carry a single 'Head' body region."""
        dfs = _make_dfs()
        sheet_dict, _ = load_data(dfs)

        for lc_name in ("Robustness-AEMDB", "Robustness-Pole"):
            head = _get_head_br(sheet_dict, lc_name)
            self.assertEqual(head.name, "Head")

    # -----------------------------------------------------------------------
    # Color carry-forward: AEMD and Pole are independent
    # -----------------------------------------------------------------------

    def test_aemd_and_pole_main_colors_are_independent(self):
        """Different main colors for AEMD and Pole must produce different scores."""
        dfs = _make_dfs(
            aemd_main_color="green",  # → 100.0
            pole_main_color="orange",  # → 25.0
        )
        sheet_dict, _ = load_data(dfs)

        aemd_pelvis = (
            _get_reconstructed_lc(sheet_dict, "Main-AEMD")
            .get_seat("Driver")
            .dummy.get_body_region("Pelvis")
        )
        pole_pelvis = (
            _get_reconstructed_lc(sheet_dict, "Main-Pole")
            .get_seat("Driver")
            .dummy.get_body_region("Pelvis")
        )
        self.assertEqual(aemd_pelvis.get_bodyregion_score(), 100.0)
        self.assertEqual(pole_pelvis.get_bodyregion_score(), 25.0)

    def test_robustness_uses_correct_main_color_per_type(self):
        """Robustness-AEMDB compares against Main-AEMD color, not Main-Pole."""
        # Main-AEMD=green, robustness-AEMD=yellow → PASS (100)
        # Main-Pole=orange, robustness-Pole=green  → FAIL (orange is 2 steps from green)
        dfs = _make_dfs(
            aemd_main_color="green",
            aemd_robustness_colors=["yellow"],
            pole_main_color="orange",
            pole_robustness_colors=["green"],
        )
        sheet_dict, _ = load_data(dfs)

        head_aemd = _get_head_br(sheet_dict, "Robustness-AEMDB")
        head_pole = _get_head_br(sheet_dict, "Robustness-Pole")
        self.assertEqual(head_aemd.get_bodyregion_score(), 100.0)
        self.assertEqual(head_pole.get_bodyregion_score(), 0.0)

    def test_main_color_capping_through_load_data_caps_main_loadcases_only(self):
        """
        A 'capping' Farside excursion color in a main load case should cap
        Main-AEMD and Main-Pole, but not the robustness load cases.
        """
        dfs = _make_dfs(
            aemd_main_color="capping",
            pole_main_color="green",
            aemd_robustness_colors=["yellow"],
            pole_robustness_colors=["yellow"],
        )
        sheet_dict, _ = load_data(dfs)

        main_aemd_dummy = (
            _get_reconstructed_lc(sheet_dict, "Main-AEMD").get_seat("Driver").dummy
        )
        main_pole_dummy = (
            _get_reconstructed_lc(sheet_dict, "Main-Pole").get_seat("Driver").dummy
        )
        robustness_aemd_dummy = (
            _get_reconstructed_lc(sheet_dict, "Robustness-AEMDB")
            .get_seat("Driver")
            .dummy
        )
        robustness_pole_dummy = (
            _get_reconstructed_lc(sheet_dict, "Robustness-Pole")
            .get_seat("Driver")
            .dummy
        )

        self.assertTrue(main_aemd_dummy.get_capping())
        self.assertTrue(main_pole_dummy.get_capping())
        self.assertEqual(main_aemd_dummy.get_score(), 0.0)
        self.assertEqual(main_pole_dummy.get_score(), 0.0)
        self.assertNotEqual(robustness_aemd_dummy.get_capping(), True)
        self.assertNotEqual(robustness_pole_dummy.get_capping(), True)

    def test_pole32_merges_driver_and_front_passenger_when_both_below_lpl(self):
        dfs = _make_dfs_with_pole32_front_pair(0.0, 0.0)
        sheet_dict, _ = load_data(dfs)

        pole32 = _get_reconstructed_lc(sheet_dict, "Pole-32")
        self.assertEqual(len(pole32.seats), 1)

        merged_seat = pole32.get_seat("Driver and Front Passenger")
        self.assertEqual(merged_seat.dummy.get_body_region("Head").get_score(), 2.0)
        self.assertEqual(merged_seat.dummy.get_score(), 2.0)
        self.assertEqual(
            [seat.name for seat in pole32.raw_seats], ["Driver", "Front Passenger"]
        )

    def test_pole32_merges_driver_and_front_passenger_to_zero_when_any_value_not_below_lpl(
        self,
    ):
        dfs = _make_dfs_with_pole32_front_pair(710.0, 0.0)
        sheet_dict, _ = load_data(dfs)

        pole32 = _get_reconstructed_lc(sheet_dict, "Pole-32")
        merged_seat = pole32.get_seat("Driver and Front Passenger")
        self.assertEqual(merged_seat.dummy.get_body_region("Head").get_score(), 0.0)
        self.assertEqual(merged_seat.dummy.get_score(), 0.0)

    def test_pole32_update_loadcase_uses_raw_seats_for_source_sheet_rows(self):
        dfs = _make_dfs_with_pole32_front_pair(710.0, 720.0)
        sheet_dict, _ = load_data(dfs)

        pole32 = _get_reconstructed_lc(sheet_dict, "Pole-32")
        updated_df = update_loadcase(dfs["CP - Side Farside"].copy(), pole32)

        pole32_hic15_scores = updated_df.loc[
            (updated_df["Loadcase"] == "Pole-32")
            | (
                updated_df["Loadcase"].isna()
                & updated_df["Seat position"].isin(["Driver", "Front Passenger"])
            ),
            "Score",
        ].dropna()
        self.assertEqual(len(pole32_hic15_scores), 2)


# ===========================================================================
# 4. apply_farside_capping_rule – unit tests
# ===========================================================================


def _make_criteria(capping: bool = False) -> Criteria:
    """Return a simple CRITERIA-type with an optional capping flag."""
    c = Criteria(
        name="HIC15",
        hpl=500.0,
        lpl=700.0,
        value=600.0,
        criteria_type=CriteriaType.CRITERIA,
    )
    c.capping = capping
    return c


def _make_head_br(score: float = 2.0, capping: bool = False) -> BodyRegion:
    br = BodyRegion(name="Head")
    br.set_criteria_list([_make_criteria(capping=capping)])
    br.set_bodyregion_score(score)
    br.set_max_score(2.0)
    br.set_inspection(0.0)
    br.compute_score()
    return br


def _make_dummy_with_score(
    score_value: float = 2.0, capping: bool = False, name: str = "WorldSID-50"
) -> Dummy:
    br = _make_head_br(score=score_value, capping=capping)
    d = Dummy(name=name, body_region_list=[br])
    d.set_max_score(2.0)
    d.compute_score()
    return d


def _make_lc(
    name: str,
    score: float = 2.0,
    capping: bool = False,
    dummy_name: str = "WorldSID-50",
) -> LoadCase:
    dummy = _make_dummy_with_score(score_value=score, capping=capping, name=dummy_name)
    seat = Seat(name="Driver", dummy=dummy)
    return LoadCase(name=name, seats=[seat])


def _get_dummy(lc: LoadCase) -> Dummy:
    return lc.get_seat("Driver").dummy


class TestApplyCappingRule(unittest.TestCase):
    """
    Unit tests for apply_farside_capping_rule().

    Load cases are constructed directly (no load_data / DataFrame) to
    keep tests fast and isolated from the rest of the pipeline.

    The four affected names are: Main-AEMDB, Main-Pole, Robustness-AEMDB,
    Robustness-Pole.  Pole-32 is never affected.
    """

    AFFECTED = ["Main-AEMDB", "Main-Pole", "Robustness-AEMDB", "Robustness-Pole"]
    MAIN_ONLY = ["Main-AEMDB", "Main-Pole"]

    def _make_standard_load_cases(
        self, main_aemdb_capping=False, main_pole_capping=False
    ):
        """Return the standard five farside load cases used in most tests."""
        return [
            _make_lc("Main-AEMDB", capping=main_aemdb_capping),
            _make_lc("Main-Pole", capping=main_pole_capping),
            _make_lc("Robustness-AEMDB"),
            _make_lc("Robustness-Pole"),
            _make_lc("Pole-32", dummy_name="WorldSID-50-Farside"),
        ]

    # -----------------------------------------------------------------------
    # No capping – nothing changes
    # -----------------------------------------------------------------------

    def test_no_capping_in_main_lcs_leaves_scores_unchanged(self):
        load_cases = self._make_standard_load_cases()
        result = apply_farside_capping_rule(load_cases)

        for lc in result:
            dummy = _get_dummy(lc)
            self.assertNotEqual(
                dummy.get_capping(),
                True,
                msg=f"Load case '{lc.name}' should NOT be capped",
            )

    def test_no_capping_original_scores_preserved(self):
        load_cases = self._make_standard_load_cases()
        result = apply_farside_capping_rule(load_cases)

        for lc in result:
            dummy = _get_dummy(lc)
            self.assertIsNotNone(
                dummy.get_score(), msg=f"Score of '{lc.name}' should remain non-None"
            )
            self.assertGreater(
                dummy.get_score(), 0.0, msg=f"Score of '{lc.name}' should remain > 0"
            )

    # -----------------------------------------------------------------------
    # Capping triggered by Main-AEMDB
    # -----------------------------------------------------------------------

    def test_capping_from_main_aemdb_caps_all_four_affected_lcs(self):
        load_cases = self._make_standard_load_cases(main_aemdb_capping=True)
        result = apply_farside_capping_rule(load_cases)

        for lc in result:
            dummy = _get_dummy(lc)
            if lc.name in self.AFFECTED:
                self.assertTrue(
                    dummy.get_capping(),
                    msg=f"'{lc.name}' dummy should have capping=True",
                )
            else:
                self.assertNotEqual(
                    dummy.get_capping(), True, msg=f"'{lc.name}' should NOT be capped"
                )

    def test_capping_from_main_aemdb_zeroes_scores_of_affected_lcs(self):
        load_cases = self._make_standard_load_cases(main_aemdb_capping=True)
        result = apply_farside_capping_rule(load_cases)

        for lc in result:
            dummy = _get_dummy(lc)
            if lc.name in self.AFFECTED:
                self.assertEqual(
                    dummy.get_score(),
                    0.0,
                    msg=f"'{lc.name}' score should be 0.0 after capping",
                )

    # -----------------------------------------------------------------------
    # Capping triggered by Main-Pole
    # -----------------------------------------------------------------------

    def test_capping_from_main_pole_caps_all_four_affected_lcs(self):
        load_cases = self._make_standard_load_cases(main_pole_capping=True)
        result = apply_farside_capping_rule(load_cases)

        for lc in result:
            dummy = _get_dummy(lc)
            if lc.name in self.AFFECTED:
                self.assertTrue(
                    dummy.get_capping(),
                    msg=f"'{lc.name}' dummy should have capping=True",
                )

    def test_capping_from_main_pole_zeroes_scores_of_affected_lcs(self):
        load_cases = self._make_standard_load_cases(main_pole_capping=True)
        result = apply_farside_capping_rule(load_cases)

        for lc in result:
            dummy = _get_dummy(lc)
            if lc.name in self.AFFECTED:
                self.assertEqual(
                    dummy.get_score(),
                    0.0,
                    msg=f"'{lc.name}' score should be 0.0 after capping",
                )

    # -----------------------------------------------------------------------
    # Pole-32 always excluded
    # -----------------------------------------------------------------------

    def test_pole32_never_capped_when_main_aemdb_triggers_capping(self):
        load_cases = self._make_standard_load_cases(main_aemdb_capping=True)
        result = apply_farside_capping_rule(load_cases)

        pole32 = next(lc for lc in result if lc.name == "Pole-32")
        self.assertNotEqual(_get_dummy(pole32).get_capping(), True)

    def test_pole32_never_capped_when_main_pole_triggers_capping(self):
        load_cases = self._make_standard_load_cases(main_pole_capping=True)
        result = apply_farside_capping_rule(load_cases)

        pole32 = next(lc for lc in result if lc.name == "Pole-32")
        self.assertNotEqual(_get_dummy(pole32).get_capping(), True)

    def test_pole32_score_unchanged_when_capping_triggered(self):
        load_cases = self._make_standard_load_cases(main_aemdb_capping=True)
        pole32_before = _get_dummy(
            next(lc for lc in load_cases if lc.name == "Pole-32")
        ).get_score()
        result = apply_farside_capping_rule(load_cases)
        pole32_after = _get_dummy(
            next(lc for lc in result if lc.name == "Pole-32")
        ).get_score()
        self.assertEqual(pole32_before, pole32_after)

    # -----------------------------------------------------------------------
    # Both main load cases capping simultaneously
    # -----------------------------------------------------------------------

    def test_both_main_lcs_capping_caps_all_four_affected(self):
        load_cases = self._make_standard_load_cases(
            main_aemdb_capping=True, main_pole_capping=True
        )
        result = apply_farside_capping_rule(load_cases)

        for name in self.AFFECTED:
            lc = next(lc for lc in result if lc.name == name)
            self.assertTrue(
                _get_dummy(lc).get_capping(), msg=f"'{name}' should be capped"
            )
            self.assertEqual(
                _get_dummy(lc).get_score(), 0.0, msg=f"'{name}' score should be 0.0"
            )

    # -----------------------------------------------------------------------
    # Capping on robustness / Pole-32 criteria DOES NOT trigger the rule
    # -----------------------------------------------------------------------

    def test_capping_only_in_robustness_lc_does_not_trigger_rule(self):
        """Capping in Robustness-AEMDB body region must NOT spread to other LCs."""
        load_cases = [
            _make_lc("Main-AEMDB", capping=False),
            _make_lc("Main-Pole", capping=False),
            _make_lc("Robustness-AEMDB", capping=True),
            _make_lc("Robustness-Pole"),
            _make_lc("Pole-32", dummy_name="WorldSID-50-Farside"),
        ]
        result = apply_farside_capping_rule(load_cases)

        for lc in result:
            dummy = _get_dummy(lc)
            self.assertNotEqual(
                dummy.get_capping(),
                True,
                msg=f"'{lc.name}' should NOT be capped (trigger only from Main LCs)",
            )

    def test_capping_only_in_pole32_does_not_trigger_rule(self):
        """Capping in Pole-32 must NOT trigger the rule for any load case."""
        load_cases = [
            _make_lc("Main-AEMDB", capping=False),
            _make_lc("Main-Pole", capping=False),
            _make_lc("Robustness-AEMDB"),
            _make_lc("Robustness-Pole"),
            _make_lc("Pole-32", capping=True, dummy_name="WorldSID-50-Farside"),
        ]
        result = apply_farside_capping_rule(load_cases)

        for lc in result:
            dummy = _get_dummy(lc)
            self.assertNotEqual(
                dummy.get_capping(), True, msg=f"'{lc.name}' should NOT be capped"
            )

    # -----------------------------------------------------------------------
    # Return value
    # -----------------------------------------------------------------------

    def test_function_returns_same_list_object(self):
        load_cases = self._make_standard_load_cases()
        result = apply_farside_capping_rule(load_cases)
        self.assertIs(result, load_cases)

    def test_function_returns_same_number_of_load_cases(self):
        load_cases = self._make_standard_load_cases(main_aemdb_capping=True)
        result = apply_farside_capping_rule(load_cases)
        self.assertEqual(len(result), 5)

    # -----------------------------------------------------------------------
    # Main-color capping rule
    # -----------------------------------------------------------------------

    def test_main_aemdb_color_capping_caps_main_loadcases_only(self):
        load_cases = self._make_standard_load_cases()
        result = apply_farside_capping_rule(
            load_cases, {"AEMD": "capping", "Pole": "green"}
        )

        for lc in result:
            dummy = _get_dummy(lc)
            if lc.name in self.MAIN_ONLY:
                self.assertTrue(dummy.get_capping())
                self.assertEqual(dummy.get_score(), 0.0)
            else:
                self.assertNotEqual(dummy.get_capping(), True)

    def test_main_pole_color_capping_caps_main_loadcases_only(self):
        load_cases = self._make_standard_load_cases()
        result = apply_farside_capping_rule(
            load_cases, {"AEMD": "green", "Pole": "capping"}
        )

        for lc in result:
            dummy = _get_dummy(lc)
            if lc.name in self.MAIN_ONLY:
                self.assertTrue(dummy.get_capping())
                self.assertEqual(dummy.get_score(), 0.0)
            else:
                self.assertNotEqual(dummy.get_capping(), True)

    def test_main_color_capping_does_not_cap_robustness_loadcases(self):
        load_cases = self._make_standard_load_cases()
        result = apply_farside_capping_rule(load_cases, {"AEMD": "capping"})

        robustness_aemd = next(lc for lc in result if lc.name == "Robustness-AEMDB")
        robustness_pole = next(lc for lc in result if lc.name == "Robustness-Pole")
        self.assertNotEqual(_get_dummy(robustness_aemd).get_capping(), True)
        self.assertNotEqual(_get_dummy(robustness_pole).get_capping(), True)

    def test_main_color_capping_and_kpi_capping_still_caps_all_four(self):
        load_cases = self._make_standard_load_cases(main_aemdb_capping=True)
        result = apply_farside_capping_rule(load_cases, {"AEMD": "capping"})

        for lc in result:
            dummy = _get_dummy(lc)
            if lc.name in self.AFFECTED:
                self.assertTrue(dummy.get_capping())
                self.assertEqual(dummy.get_score(), 0.0)
            else:
                self.assertNotEqual(dummy.get_capping(), True)

    def test_main_color_not_capping_does_not_trigger_rule(self):
        load_cases = self._make_standard_load_cases()
        result = apply_farside_capping_rule(
            load_cases, {"AEMD": "green", "Pole": "yellow"}
        )

        for lc in result:
            dummy = _get_dummy(lc)
            self.assertNotEqual(dummy.get_capping(), True)


# ===========================================================================
# 5. CappingCriteria – colour emission tests
# ===========================================================================


class TestCappingCriteriaColour(unittest.TestCase):
    """
    Verify that CAPPING_CRITERIA emit 'red' / 'green' colours.

    Rule: value >= capping_value → red, else → green.
    Score stays None; capping flag matches colour.
    """

    def _make(self, name: str, capping_value: float) -> Criteria:
        return Criteria(
            name=name,
            criteria_type=CriteriaType.CAPPING_CRITERIA,
            capping_value=capping_value,
        )

    # -- Generic boundary behaviour ------------------------------------------

    def test_above_capping_value_is_red(self):
        c = self._make("HPC", 1000.0)
        c.set_value(1200.0)
        self.assertEqual(c.color, "red")
        self.assertTrue(c.capping)
        self.assertIsNone(c.score)

    def test_below_capping_value_is_green(self):
        c = self._make("HPC", 1000.0)
        c.set_value(800.0)
        self.assertEqual(c.color, "green")
        self.assertFalse(c.capping)
        self.assertIsNone(c.score)

    def test_exactly_at_capping_value_is_red(self):
        c = self._make("HPC", 1000.0)
        c.set_value(1000.0)
        self.assertEqual(c.color, "red")
        self.assertTrue(c.capping)
        self.assertIsNone(c.score)

    def test_no_capping_value_leaves_colour_none(self):
        c = Criteria(name="HPC", criteria_type=CriteriaType.CAPPING_CRITERIA)
        c.set_value(500.0)
        self.assertIsNone(c.color)

    # -- Pelvis: F pubic symphysis  (threshold 2.8 kN) -----------------------

    def test_pubic_symphysis_below_threshold_is_green(self):
        c = self._make("F pubic symphysis", 2.8)
        c.set_value(2.79)
        self.assertEqual(c.color, "green")

    def test_pubic_symphysis_at_threshold_is_red(self):
        c = self._make("F pubic symphysis", 2.8)
        c.set_value(2.8)
        self.assertEqual(c.color, "red")

    def test_pubic_symphysis_above_threshold_is_red(self):
        c = self._make("F pubic symphysis", 2.8)
        c.set_value(2.81)
        self.assertEqual(c.color, "red")

    # -- Pelvis: Fy lumbar  (threshold 2.0 kN) --------------------------------

    def test_fy_lumbar_below_threshold_is_green(self):
        c = self._make("Fy lumbar", 2.0)
        c.set_value(1.99)
        self.assertEqual(c.color, "green")

    def test_fy_lumbar_at_threshold_is_red(self):
        c = self._make("Fy lumbar", 2.0)
        c.set_value(2.0)
        self.assertEqual(c.color, "red")

    def test_fy_lumbar_above_threshold_is_red(self):
        c = self._make("Fy lumbar", 2.0)
        c.set_value(2.01)
        self.assertEqual(c.color, "red")

    # -- Pelvis: Fz lumbar  (threshold 3.5 kN) --------------------------------

    def test_fz_lumbar_below_threshold_is_green(self):
        c = self._make("Fz lumbar", 3.5)
        c.set_value(3.49)
        self.assertEqual(c.color, "green")

    def test_fz_lumbar_at_threshold_is_red(self):
        c = self._make("Fz lumbar", 3.5)
        c.set_value(3.5)
        self.assertEqual(c.color, "red")

    def test_fz_lumbar_above_threshold_is_red(self):
        c = self._make("Fz lumbar", 3.5)
        c.set_value(3.51)
        self.assertEqual(c.color, "red")

    # -- Pelvis: Mx lumbar  (threshold 120 Nm) --------------------------------

    def test_mx_lumbar_below_threshold_is_green(self):
        c = self._make("Mx lumbar", 120.0)
        c.set_value(119.9)
        self.assertEqual(c.color, "green")

    def test_mx_lumbar_at_threshold_is_red(self):
        c = self._make("Mx lumbar", 120.0)
        c.set_value(120.0)
        self.assertEqual(c.color, "red")

    def test_mx_lumbar_above_threshold_is_red(self):
        c = self._make("Mx lumbar", 120.0)
        c.set_value(120.1)
        self.assertEqual(c.color, "red")


if __name__ == "__main__":
    unittest.main()
