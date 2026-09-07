# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Regression tests for the UnboundLocalError bug in data_loader.load_data()
when processing CP - VRU Head Impact and CP - VRU Pelvis & Leg Impact sheets.

Before the fix, the unconditional logger.debug() at the end of the body-region
loop referenced `bodyregion_score` and `computed_score` which were never
assigned in the VRU branches, causing an UnboundLocalError regardless of the
active log level (f-strings are evaluated eagerly).

These tests call load_data() directly.  When no `CP - VRU Prediction Points`
sheet is present in `dfs`, compute_vru_bodyregion_score() now correctly skips
the headform scoring loop instead of raising a ZeroDivisionError.
"""

import unittest
from collections import Counter
from unittest.mock import patch

import pandas as pd

from euroncap_rating_2026 import common
from euroncap_rating_2026.crash_protection import data_loader
from euroncap_rating_2026.crash_protection import vru_processing
from euroncap_rating_2026.crash_protection.preprocess import add_vru_sheets_to_dfs


def _make_vru_dfs() -> dict:
    """Build a minimal inline dfs dict sufficient for data_loader.load_data()."""
    N = float("nan")
    return {
        "Input parameters": pd.DataFrame(
            {
                "Stage": ["Crash Protection", N, N, N, N, N],
                "Stage element": ["Frontal Impact", N, N, "Side Impact", N, N],
                "Stage subelement": ["Offset", N, N, "MDB", N, "Farside"],
                "Input parameter": [N, N, N, N, N, "Countermeasure?"],
                "Value": [N, N, N, N, N, "Yes"],
            }
        ),
        "CP - VRU Head Impact": pd.DataFrame(
            {
                "Loadcase": ["Headform", N],
                "Seat position": ["Driver", N],
                "Dummy": ["Adult Headform", N],
                "Body region": ["Adult", "Cyclist"],
                "Test point": ["(6, 9)", "(18, -6)"],
                "Criteria": ["HIC15", "HIC15"],
                "HPL": [650.0, 650.0],
                "LPL": [1700.0, 1700.0],
                "Capping": [N, N],
                "OEM Prediction": ["Green", "Green"],
                "Value": [0.0, 0.0],
            }
        ),
        "CP - VRU Pelvis & Leg Impact": pd.DataFrame(
            {
                "Loadcase": ["Upper Leg", N],
                "Seat position": ["Driver", N],
                "Dummy": ["Upper legform", N],
                "Body region": ["Pelvis", N],
                "Test point": ["(0, 10)", "(0, -9)"],
                "Criteria": ["Sum of forces", "Sum of forces"],
                "HPL": [5.0, 5.0],
                "LPL": [6.0, 6.0],
                "Capping": [N, N],
                "OEM Prediction": ["Green", "Green"],
                "Value": [0.0, 0.0],
            }
        ),
        "Test Scores": pd.DataFrame(
            {
                "Stage": ["Crash Protection"] + [N] * 8,
                "Stage element": [
                    "Frontal Impact",
                    N,
                    N,
                    "Side Impact",
                    N,
                    N,
                    "Rear Impact",
                    "VRU Impact",
                    N,
                ],
                "Stage subelement": [
                    "Offset",
                    "FW",
                    "Sled & VT",
                    "MDB",
                    "Pole",
                    "Farside",
                    "Whiplash",
                    "Head Impact",
                    "Pelvis & Leg Impact",
                ],
                "Inspection [%]": [0] * 9,
                "Score": [N] * 9,
                "Max score": [20, 10, 10, 15, 10, 10, 5, 10, 10],
            }
        ),
        "CP - Dummy Scores": pd.DataFrame(
            columns=[
                "Stage",
                "Stage element",
                "Stage subelement",
                "Loadcase",
                "Seat position",
                "Dummy",
                "Capping?",
                "Score",
                "Max score",
            ]
        ),
        "CP - Body region scores": pd.DataFrame(
            {
                "Stage element": ["VRU Impact", N, N, N, N],
                "Stage subelement": ["Head Impact", N, "Pelvis & Leg Impact", N, N],
                "Loadcase": ["Headform", N, "Upper Leg", "Lower Leg", N],
                "Seat position": ["Driver", N, "Driver", "Driver", N],
                "Dummy": [
                    "Adult Headform",
                    "Child Headform",
                    "Upper legform",
                    "aPLI",
                    N,
                ],
                "Body region": ["Adult", "Child", "Pelvis", "Femur", "Knee & Tibia"],
                "Body regionscore": [N, N, N, N, N],
                "Modifiers": [N, N, N, N, N],
                "Inspection [%]": [0, 0, 0, 0, 0],
                "Score": [N, N, N, N, N],
                "Max score": [N, N, 2.5, 2.5, 5.0],
            }
        ),
    }


class TestVruBranchesNoUnboundLocalError(unittest.TestCase):
    """
    Regression: load_data() must not raise UnboundLocalError for templates
    that include CP - VRU Head Impact and CP - VRU Pelvis & Leg Impact sheets.
    """

    def _assert_load_data_succeeds(self):
        dfs = _make_vru_dfs()

        # Confirm both VRU sheets are present so the test is meaningful.
        self.assertIn(
            "CP - VRU Head Impact",
            dfs,
            "'CP - VRU Head Impact' missing from inline dfs",
        )
        self.assertIn(
            "CP - VRU Pelvis & Leg Impact",
            dfs,
            "'CP - VRU Pelvis & Leg Impact' missing from inline dfs",
        )

        # Before the fix this raised:
        #   UnboundLocalError: cannot access local variable 'bodyregion_score'
        #   where it is not associated with a value
        try:
            sheet_dict, test_score_inspection = data_loader.load_data(dfs)
        except UnboundLocalError as exc:
            self.fail(
                f"load_data() raised UnboundLocalError – the VRU branch fix may "
                f"have been reverted. Details: {exc}"
            )

        # Both VRU sheets must appear in the returned sheet_dict.
        self.assertIn("CP - VRU Head Impact", sheet_dict)
        self.assertIn("CP - VRU Pelvis & Leg Impact", sheet_dict)

    def test_vru_branches_no_unbound_local_error_max_file(self):
        """Max-score template with VRU sheets: load_data() must complete cleanly."""
        self._assert_load_data_succeeds()

    def test_vru_branches_no_unbound_local_error_min_file(self):
        """Min-score template with VRU sheets: load_data() must complete cleanly."""
        self._assert_load_data_succeeds()


class TestLowerLegTestPointMapping(unittest.TestCase):
    def _build_lower_leg_points(self):
        return [
            vru_processing.LegformTestPoint(
                row=1,
                col=4,
                color=vru_processing.VruPredictionColor.GREEN,
                loadcase_name="Lower Leg",
            ),
            vru_processing.LegformTestPoint(
                row=1,
                col=5,
                color=vru_processing.VruPredictionColor.GREEN,
                loadcase_name="Lower Leg",
            ),
            vru_processing.LegformTestPoint(
                row=2,
                col=4,
                color=vru_processing.VruPredictionColor.GREEN,
                loadcase_name="Lower Leg",
            ),
            vru_processing.LegformTestPoint(
                row=2,
                col=5,
                color=vru_processing.VruPredictionColor.GREEN,
                loadcase_name="Lower Leg",
            ),
        ]

    def _format_test_point(self, row, col):
        return common.format_test_point(row, col)

    def test_lower_leg_loadcases_split_femur_from_knee_tibia_rows(self):
        loadcases = vru_processing.process_legform_vru_loadcases(
            self._build_lower_leg_points()
        )

        self.assertEqual(len(loadcases), 1)
        loadcase = loadcases[0]
        self.assertEqual(loadcase.name, "Lower Leg")
        self.assertEqual(len(loadcase.seats), 1)

        criteria_points = {}
        for seat in loadcase.seats:
            for body_region in seat.dummy.body_region_list:
                criteria_points[body_region.name] = {
                    criteria.test_point for criteria in body_region._criteria
                }

        self.assertEqual(
            criteria_points["Femur"],
            {self._format_test_point(1, 4), self._format_test_point(1, 5)},
        )
        self.assertEqual(
            criteria_points["Knee"],
            {self._format_test_point(2, 4), self._format_test_point(2, 5)},
        )
        self.assertEqual(
            criteria_points["Tibia"],
            {self._format_test_point(2, 4), self._format_test_point(2, 5)},
        )

    def test_generate_vru_pelvis_leg_impact_df_uses_split_lower_leg_rows(self):
        loadcases = vru_processing.process_legform_vru_loadcases(
            self._build_lower_leg_points()
        )

        df = vru_processing.generate_vru_pelvis_leg_impact_df(loadcases)
        femur_rows = df[df["Criteria"].str.startswith("Bending moment, F")]
        knee_rows = df[df["Criteria"] == "MCL elongation"]
        tibia_rows = df[df["Criteria"].str.startswith("Bending moment, T")]

        self.assertEqual(
            femur_rows["Test point"].tolist(),
            [
                self._format_test_point(1, 4),
                self._format_test_point(1, 4),
                self._format_test_point(1, 4),
                self._format_test_point(1, 5),
                self._format_test_point(1, 5),
                self._format_test_point(1, 5),
            ],
        )
        self.assertEqual(
            knee_rows["Test point"].tolist(),
            [self._format_test_point(2, 4), self._format_test_point(2, 5)],
        )
        self.assertEqual(
            tibia_rows["Test point"].tolist(),
            [
                self._format_test_point(2, 4),
                self._format_test_point(2, 4),
                self._format_test_point(2, 4),
                self._format_test_point(2, 4),
                self._format_test_point(2, 5),
                self._format_test_point(2, 5),
                self._format_test_point(2, 5),
                self._format_test_point(2, 5),
            ],
        )


class TestVruInputSelection(unittest.TestCase):
    def _build_prediction_df(self):
        df = pd.DataFrame("grey", index=range(28), columns=range(15))
        df.iat[12, 4] = "green"
        df.iat[11, 5] = "green-20"
        df.iat[26, 4] = "green"
        df.iat[27, 5] = "yellow"
        return df

    def _build_input_parameters_df(self):
        return pd.DataFrame(
            {
                "Stage": ["Crash Protection", None],
                "Stage element": ["VRU", None],
                "Stage subelement": ["Head Impact", "Leg Impact"],
                "Input parameter": ["Dummy parameter", "Dummy parameter"],
                "Value": ["X", "Y"],
            }
        )

    def test_select_vru_test_points_matches_semantic_input_and_omits_missing_groups(
        self,
    ):
        input_selected_points = {
            "Headform": {
                "Headform": [{"Body region": "Adult", "Color": "green"}],
            },
            "Legform": {
                "Lower Leg": [
                    {"Body region": "Femur", "Color": "green"},
                    {"Body region": "Knee/Tibia", "Color": "yellow"},
                ]
            },
        }

        headform_df, legform_df = data_loader.select_vru_test_points(
            self._build_prediction_df(),
            self._build_input_parameters_df(),
            input_selected_points=input_selected_points,
        )

        # A-pillar (green-20/30/40) points are mandatory on every path: the
        # grid's green-20 cell is appended even though the explicit selection
        # omitted it.
        self.assertEqual(
            headform_df["loadcase_name"].tolist(), ["Headform", "Headform Apillar"]
        )
        self.assertEqual(headform_df["body_region"].tolist(), ["Adult", "Adult"])
        apillar_rows = headform_df[headform_df["loadcase_name"] == "Headform Apillar"]
        self.assertEqual(apillar_rows["row"].tolist(), [9])
        self.assertEqual(apillar_rows["col"].tolist(), [9])

        self.assertEqual(
            legform_df["loadcase_name"].tolist(), ["Lower Leg", "Lower Leg"]
        )
        self.assertEqual(legform_df["row"].tolist(), [1, 2])
        self.assertEqual(legform_df["col"].tolist(), [10, 9])

    def test_select_vru_test_points_without_input_preserves_default_processing_path(
        self,
    ):
        headform_points = [
            vru_processing.VruTestPoint(
                row=8,
                col=10,
                color=vru_processing.VruPredictionColor.GREEN,
                loadcase_name="Headform",
            )
        ]
        legform_points = [
            vru_processing.LegformTestPoint(
                row=1,
                col=10,
                color=vru_processing.VruPredictionColor.GREEN,
                loadcase_name="Lower Leg",
            )
        ]

        with (
            patch.object(
                vru_processing,
                "process_headform_test",
                return_value=headform_points,
            ) as mock_headform,
            patch.object(
                vru_processing,
                "process_legform_test",
                return_value=legform_points,
            ) as mock_legform,
        ):
            headform_df, legform_df = data_loader.select_vru_test_points(
                self._build_prediction_df(),
                self._build_input_parameters_df(),
            )

        self.assertIsNone(mock_headform.call_args.kwargs["input_selected_points"])
        self.assertIsNone(mock_legform.call_args.kwargs["input_selected_points"])
        self.assertEqual(headform_df["loadcase_name"].tolist(), ["Headform"])
        self.assertEqual(legform_df["loadcase_name"].tolist(), ["Lower Leg"])


class TestAddVruSheetsToDfsDataFramePath(unittest.TestCase):
    """Some callers never run `crash_protection preprocess`; they call the
    dataframe-in/dataframe-out sibling
    crash_protection.preprocess.add_vru_sheets_to_dfs(dfs) directly, fed with
    DataFrames assembled from parquet storage. Both of the following fixes live
    in the shared selection code that entry point reaches
    (data_loader.select_vru_test_points -> vru_processing.process_headform_test
    / process_legform_test), so they must hold there too, not just through
    the file-based CLI:

    1. Graceful handling of missing VRU inputs
       (vru_processing.get_num_verification_tests): an unfilled 'Number of
       verification tests' cell used to reach
       common.frequency_proportional_sample as None and die with a
       TypeError; it now warns and samples 0 verification points.
    2. The blue-point selection fix (vru_processing.select_blue_pattern /
       _resolve_row_coin): an all-blue legform row keeps its full declared
       span -- the coin only decides which *orphaned* SA cells get promoted
       to ST, it never drops or relabels a declared T/ST cell.

    Every case round-trips dfs through parquet and back first, mirroring how
    such callers actually feed add_vru_sheets_to_dfs from parquet storage
    rather than an Excel file."""

    PREDICTION_SHEET = "CP - VRU Prediction"
    PARAMS_SHEET = "Input parameters"

    # df-absolute offsets of the two matrices (mirrors the iloc slicing in
    # process_headform_test / get_legform_matrix: slice start + 2 header
    # rows, columns from 4).
    HEAD_ROW0 = vru_processing.HEADFORMS_START_ROW_INDEX + 2
    LEG_ROW0 = vru_processing.LEGFORMS_START_ROW_INDEX + 2
    COL0 = 4
    LEG_WIDTH = vru_processing.VRU_MATRIX_COL_END_INDEX - COL0  # 21

    BLUE_VARIANTS = {
        vru_processing.VruPredictionColor.BLUE,
        vru_processing.VruPredictionColor.BLUE_T,
        vru_processing.VruPredictionColor.BLUE_ST,
        vru_processing.VruPredictionColor.BLUE_SA,
    }
    APILLAR = {
        vru_processing.VruPredictionColor.GREEN_40,
        vru_processing.VruPredictionColor.GREEN_30,
        vru_processing.VruPredictionColor.GREEN_20,
    }
    SAMPLED_POOL = {
        vru_processing.VruPredictionColor.GREEN,
        vru_processing.VruPredictionColor.YELLOW,
        vru_processing.VruPredictionColor.ORANGE,
        vru_processing.VruPredictionColor.RED,
        vru_processing.VruPredictionColor.BROWN,
    }

    # Headform pattern: 4 blue variants + 3 A-pillar greens + 8 sampleable colors.
    HEAD_PATTERN = {
        (0, 0): "blue",
        (0, 2): "t",
        (2, 0): "st",
        (2, 2): "sa",
        (4, 0): "green-40",
        (4, 2): "green-30",
        (4, 4): "green-20",
        (6, 0): "green",
        (6, 1): "yellow",
        (6, 2): "orange",
        (6, 3): "red",
        (6, 4): "brown",
        (7, 0): "green",
        (7, 1): "yellow",
        (7, 2): "orange",
    }
    HEAD_BLUES = 4
    HEAD_APILLAR = 3

    NUM_TESTS_HEAD = 3
    NUM_TESTS_PELVIS = 1
    NUM_TESTS_LEG = 2

    def _load_template_dfs(self):
        from importlib.resources import files

        template_path = str(files("data").joinpath("cp_template.xlsx"))
        return common.read_excel_file_to_dfs(template_path)

    def _parquet_roundtrip(self, dfs):
        import io

        def roundtrip(df):
            buf = io.BytesIO()
            df.to_parquet(buf)
            return pd.read_parquet(io.BytesIO(buf.getvalue()))

        out = {}
        for name, df in dfs.items():
            try:
                out[name] = roundtrip(df)
            except Exception:
                stringified = df.apply(
                    lambda col: col.map(lambda v: str(v) if pd.notna(v) else None)
                )
                out[name] = roundtrip(stringified)
        return out

    def _write_matrix(self, dfs, row0, values_by_offset, clear_rows, clear_cols):
        pred = dfs[self.PREDICTION_SHEET].astype(object)
        for i in range(clear_rows):
            for j in range(clear_cols):
                pred.iat[row0 + i, self.COL0 + j] = "grey"
        for (i, j), value in values_by_offset.items():
            pred.iat[row0 + i, self.COL0 + j] = value
        dfs[self.PREDICTION_SHEET] = pred

    def _set_num_verification_tests(self, dfs, head=None, pelvis=None, leg=None):
        params = dfs[self.PARAMS_SHEET].copy()
        wanted = {"Head Impact": head, "Pelvis Impact": pelvis, "Leg Impact": leg}
        current_subelement = None
        for idx, row in params.iterrows():
            if not pd.isna(row.get("Stage subelement")):
                current_subelement = row["Stage subelement"]
            input_param = row.get("Input parameter")
            if pd.isna(input_param) or "Number of verification tests" not in str(
                input_param
            ):
                continue
            if current_subelement in wanted and wanted[current_subelement] is not None:
                params.at[idx, "Value"] = wanted[current_subelement]
        dfs[self.PARAMS_SHEET] = params

    def _filled_dfs(self, leg_pattern):
        dfs = self._load_template_dfs()
        self._write_matrix(
            dfs, self.HEAD_ROW0, self.HEAD_PATTERN, clear_rows=19, clear_cols=11
        )
        self._write_matrix(
            dfs, self.LEG_ROW0, leg_pattern, clear_rows=3, clear_cols=self.LEG_WIDTH
        )
        return dfs

    def _leg_alternating_pattern(self):
        # Row 0 all ST; rows 1-2 alternating ST/SA -- every SA has a tested
        # neighbor, so no orphan, no coin: must be emitted exactly as declared.
        pattern = {}
        for j in range(self.LEG_WIDTH):
            pattern[(0, j)] = "st"
            pattern[(1, j)] = "st" if j % 2 == 0 else "sa"
            pattern[(2, j)] = "st" if j % 2 == 0 else "sa"
        return pattern

    @staticmethod
    def _color_of(point):
        raw = getattr(point.color, "value", point.color)
        return str(raw).lower()

    def _color_counts(self, points):
        return Counter(self._color_of(p) for p in points)

    def _run(self, dfs):
        dfs = self._parquet_roundtrip(dfs)
        return add_vru_sheets_to_dfs(dfs)

    def test_completely_unfilled_template_does_not_raise_and_selects_nothing(self):
        # Pre-fix: get_num_verification_tests reached
        # frequency_proportional_sample with None and raised TypeError.
        with self.assertLogs("euroncap_rating_2026", level="WARNING") as log_ctx:
            head_points, leg_points, result = self._run(self._load_template_dfs())

        self.assertEqual(len(head_points), 0)
        self.assertEqual(len(leg_points), 0)
        self.assertEqual(len(result["CP - VRU Head Impact"]), 0)
        self.assertEqual(len(result["CP - VRU Pelvis & Leg Impact"]), 0)
        missing_warnings = [
            r.getMessage()
            for r in log_ctx.records
            if "Number of verification tests" in r.getMessage()
            and "missing" in r.getMessage()
        ]
        self.assertEqual(len(missing_warnings), 3)

    def test_blank_verification_counts_still_keep_every_declared_blue_and_apillar_cell(
        self,
    ):
        dfs = self._filled_dfs(self._leg_alternating_pattern())

        head_points, leg_points, _ = self._run(dfs)

        head_counts = self._color_counts(head_points)
        sampled = sum(
            n
            for c, n in head_counts.items()
            if vru_processing.VruPredictionColor(c) in self.SAMPLED_POOL
        )
        blues = sum(
            n
            for c, n in head_counts.items()
            if vru_processing.VruPredictionColor(c) in self.BLUE_VARIANTS
        )
        apillar = sum(
            n
            for c, n in head_counts.items()
            if vru_processing.VruPredictionColor(c) in self.APILLAR
        )
        self.assertEqual(sampled, 0)
        self.assertEqual(blues, self.HEAD_BLUES)
        self.assertEqual(apillar, self.HEAD_APILLAR)

        leg_counts = self._color_counts(leg_points)
        st_per_alt_row = (self.LEG_WIDTH + 1) // 2
        sa_per_alt_row = self.LEG_WIDTH // 2
        self.assertEqual(len(leg_points), 3 * self.LEG_WIDTH)
        self.assertEqual(leg_counts.get("st", 0), self.LEG_WIDTH + 2 * st_per_alt_row)
        self.assertEqual(leg_counts.get("sa", 0), 2 * sa_per_alt_row)

    def test_fully_filled_with_known_counts_samples_exactly_that_many(self):
        dfs = self._filled_dfs(self._leg_alternating_pattern())
        self._set_num_verification_tests(
            dfs,
            head=self.NUM_TESTS_HEAD,
            pelvis=self.NUM_TESTS_PELVIS,
            leg=self.NUM_TESTS_LEG,
        )

        head_points, leg_points, _ = self._run(dfs)

        head_counts = self._color_counts(head_points)
        sampled = sum(
            n
            for c, n in head_counts.items()
            if vru_processing.VruPredictionColor(c) in self.SAMPLED_POOL
        )
        blues = sum(
            n
            for c, n in head_counts.items()
            if vru_processing.VruPredictionColor(c) in self.BLUE_VARIANTS
        )
        self.assertEqual(sampled, self.NUM_TESTS_HEAD)
        self.assertEqual(blues, self.HEAD_BLUES)

        leg_counts = self._color_counts(leg_points)
        st_per_alt_row = (self.LEG_WIDTH + 1) // 2
        sa_per_alt_row = self.LEG_WIDTH // 2
        self.assertEqual(len(leg_points), 3 * self.LEG_WIDTH)
        self.assertEqual(leg_counts.get("st", 0), self.LEG_WIDTH + 2 * st_per_alt_row)
        self.assertEqual(leg_counts.get("sa", 0), 2 * sa_per_alt_row)
        self.assertEqual(leg_counts.get("t", 0), 0)

    def test_stratified_sampling_covers_every_band_with_eligible_cells(self):
        """A single colour-proportional draw over the whole grid
        could leave a WAD band with no verification point purely by chance.
        The draw is stratified per band, so with head=3 each of the three
        bands must receive at least one sampled point - deterministically."""
        pattern = {
            # Cyclist (matrix rows 0-6), Adult (7-12), Child (13-18)
            (2, 0): "green",
            (2, 1): "yellow",
            (2, 2): "orange",
            (8, 0): "green",
            (8, 1): "brown",
            (15, 0): "green",
            (15, 1): "red",
            (15, 2): "green",
        }
        dfs = self._load_template_dfs()
        self._write_matrix(dfs, self.HEAD_ROW0, pattern, clear_rows=19, clear_cols=11)
        self._write_matrix(
            dfs,
            self.LEG_ROW0,
            self._leg_alternating_pattern(),
            clear_rows=3,
            clear_cols=self.LEG_WIDTH,
        )
        self._set_num_verification_tests(
            dfs, head=3, pelvis=self.NUM_TESTS_PELVIS, leg=self.NUM_TESTS_LEG
        )

        for _ in range(5):
            head_points, _, _ = self._run(dfs)
            bands = {p.body_region for p in head_points}
            self.assertEqual(bands, {"Cyclist", "Adult", "Child"})
            self.assertEqual(len(head_points), 3)

    def test_explicit_selection_without_apillar_still_emits_apillar_rows(self):
        """Regression: an explicit selection that omits
        the A-pillar points must not drop them - they are mandatory on every
        path and must land in the generated verification sheet."""
        dfs = self._filled_dfs(self._leg_alternating_pattern())
        input_selected_points = {
            "Headform": {
                "Headform": [{"row": 12, "col": 10, "Color": "green"}],
            },
            "Legform": {
                "Upper Leg": [{"row": 0, "col": 10}],
            },
        }

        dfs = self._parquet_roundtrip(dfs)
        head_points, _, result = add_vru_sheets_to_dfs(
            dfs, input_selected_points=input_selected_points
        )

        apillar_points = [
            p for p in head_points if p.loadcase_name == "Headform Apillar"
        ]
        self.assertEqual(len(apillar_points), self.HEAD_APILLAR)
        head_sheet = result["CP - VRU Head Impact"]
        self.assertIn("Headform Apillar", head_sheet["Loadcase"].tolist())
        apillar_predictions = {self._color_of(p) for p in apillar_points}
        self.assertEqual(apillar_predictions, {"green-40", "green-30", "green-20"})

    def test_all_sa_legform_rows_coin_flip_only_promotes_never_drops(self):
        all_sa = {(i, j): "sa" for i in range(3) for j in range(self.LEG_WIDTH)}
        dfs = self._filled_dfs(all_sa)
        self._set_num_verification_tests(
            dfs,
            head=self.NUM_TESTS_HEAD,
            pelvis=self.NUM_TESTS_PELVIS,
            leg=self.NUM_TESTS_LEG,
        )

        _, leg_points, _ = self._run(dfs)

        leg_counts = self._color_counts(leg_points)
        self.assertEqual(len(leg_points), 3 * self.LEG_WIDTH)
        self.assertLessEqual(set(leg_counts), {"st", "sa"})
        self.assertGreater(leg_counts.get("st", 0), 0)
        self.assertEqual(
            leg_counts.get("st", 0) + leg_counts.get("sa", 0), 3 * self.LEG_WIDTH
        )


if __name__ == "__main__":
    unittest.main()
