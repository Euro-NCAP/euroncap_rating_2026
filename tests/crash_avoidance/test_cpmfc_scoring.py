# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import math
import unittest
import pandas as pd

from euroncap_rating_2026 import common
from euroncap_rating_2026.crash_avoidance import matrix_processing
from euroncap_rating_2026.crash_avoidance.test_info import (
    _compute_cpmfc_color,
    _parse_optional_float,
    read_test_points,
)


def _make_test_point(color=common.PredictionColor.GREEN):
    return matrix_processing.TestPoint(
        row=1,
        col=2,
        color=color,
        test_range=matrix_processing.TestRange.STANDARD,
    )


class TestComputeCpmfcColor(unittest.TestCase):
    """Unit tests for _compute_cpmfc_color."""

    def test_pass_when_v_impact_zero(self):
        self.assertEqual(_compute_cpmfc_color(0.0, None), common.PredictionColor.GREEN)

    def test_pass_when_v_impact_zero_with_baseline(self):
        self.assertEqual(_compute_cpmfc_color(0.0, 10.0), common.PredictionColor.GREEN)

    def test_pass_when_diff_within_threshold(self):
        # diff = 0.4 <= 0.5
        self.assertEqual(_compute_cpmfc_color(5.0, 4.6), common.PredictionColor.GREEN)

    def test_pass_when_diff_exactly_threshold(self):
        # diff = 0.5 <= 0.5
        self.assertEqual(_compute_cpmfc_color(5.0, 4.5), common.PredictionColor.GREEN)

    def test_fail_when_diff_exceeds_threshold(self):
        # diff = 0.6 > 0.5
        self.assertEqual(_compute_cpmfc_color(5.0, 4.4), common.PredictionColor.RED)

    def test_grey_when_v_impact_none(self):
        self.assertEqual(_compute_cpmfc_color(None, 4.0), common.PredictionColor.GREY)

    def test_grey_when_v_impact_none_and_baseline_none(self):
        self.assertEqual(_compute_cpmfc_color(None, None), common.PredictionColor.GREY)

    def test_grey_when_baseline_none_and_impact_nonzero(self):
        self.assertEqual(_compute_cpmfc_color(3.0, None), common.PredictionColor.GREY)

    def test_fail_large_diff(self):
        self.assertEqual(_compute_cpmfc_color(20.0, 5.0), common.PredictionColor.RED)


class TestParseOptionalFloat(unittest.TestCase):
    """Unit tests for _parse_optional_float."""

    def test_returns_float_for_numeric(self):
        self.assertEqual(_parse_optional_float(3.5), 3.5)

    def test_returns_float_for_int(self):
        self.assertEqual(_parse_optional_float(3), 3.0)

    def test_returns_none_for_none(self):
        self.assertIsNone(_parse_optional_float(None))

    def test_returns_none_for_nan(self):
        self.assertIsNone(_parse_optional_float(float("nan")))

    def test_returns_none_for_numpy_nan(self):
        import numpy as np

        self.assertIsNone(_parse_optional_float(np.float64("nan")))

    def test_returns_none_for_invalid_string(self):
        self.assertIsNone(_parse_optional_float("abc"))

    def test_returns_float_for_numeric_string(self):
        self.assertEqual(_parse_optional_float("4.2"), 4.2)


class TestPreprocessBaselineColumns(unittest.TestCase):
    """Test that baseline columns appear correctly in preprocess output DataFrames."""

    def _make_rows(self):
        cpmfc_row = {
            "Scenario": "CPMFC",
            "Test point": "(1, 2)",
            "Range": "Standard",
            "OEM Prediction": "Green",
            "Expected value": "v_impact",
            "Value": float("nan"),
            "Expected value baseline": "v_impact_baseline",
            "Value baseline": float("nan"),
            "Impact location": "25%",
            "Distance": "d1",
        }
        cpmrcs_row = {
            "Scenario": "CPMRCs",
            "Test point": "(2, 1)",
            "Range": "Standard",
            "OEM Prediction": "Green",
            "Expected value": "v_impact",
            "Value": float("nan"),
            "Impact location": "25%",
            "VUT speed": "4 km/h",
        }
        return pd.DataFrame([cpmfc_row, cpmrcs_row])

    def test_baseline_columns_present(self):
        df = self._make_rows()
        self.assertIn("Expected value baseline", df.columns)
        self.assertIn("Value baseline", df.columns)

    def test_cpmfc_row_has_v_impact_baseline(self):
        df = self._make_rows()
        cpmfc_row = df[df["Scenario"] == "CPMFC"].iloc[0]
        self.assertEqual(cpmfc_row["Expected value baseline"], "v_impact_baseline")

    def test_cpmrcs_row_has_nan_baseline(self):
        df = self._make_rows()
        cpmrcs_row = df[df["Scenario"] == "CPMRCs"].iloc[0]
        self.assertTrue(math.isnan(cpmrcs_row["Value baseline"]))

    def test_expected_value_is_v_impact_for_cpmfc(self):
        df = self._make_rows()
        cpmfc_row = df[df["Scenario"] == "CPMFC"].iloc[0]
        self.assertEqual(cpmfc_row["Expected value"], "v_impact")


class TestReadTestPointsBaselineExtraction(unittest.TestCase):
    """Test that read_test_points correctly extracts value_baseline."""

    def _make_df(self, value, value_baseline):
        return pd.DataFrame(
            [
                {
                    "Scenario": "CPMFC",
                    "Test point": "(1, 2)",
                    "Range": "Standard",
                    "OEM Prediction": "Green",
                    "Expected value": "v_impact",
                    "Value": value,
                    "Expected value baseline": "v_impact_baseline",
                    "Value baseline": value_baseline,
                    "Impact location": "25%",
                    "Distance": "d1",
                }
            ]
        )

    def test_value_baseline_extracted_as_float(self):
        df = self._make_df(5.0, 4.6)
        _, computed_info = read_test_points(df, preserve_none=True)
        entry = computed_info["CPMFC_(1, 2)"]
        self.assertAlmostEqual(entry["value_baseline"], 4.6)

    def test_value_baseline_none_when_nan(self):
        df = self._make_df(float("nan"), float("nan"))
        _, computed_info = read_test_points(df, preserve_none=True)
        entry = computed_info["CPMFC_(1, 2)"]
        self.assertIsNone(entry["value_baseline"])

    def test_baseline_not_in_test_point_attributes(self):
        df = self._make_df(5.0, 4.6)
        test_points, _ = read_test_points(df, preserve_none=True)
        attrs = test_points[0].attributes
        self.assertNotIn("Value baseline", attrs)
        self.assertNotIn("Expected value baseline", attrs)

    def test_value_extracted(self):
        df = self._make_df(5.0, 4.6)
        _, computed_info = read_test_points(df, preserve_none=True)
        entry = computed_info["CPMFC_(1, 2)"]
        self.assertAlmostEqual(entry["value"], 5.0)

    def test_non_cpmfc_row_has_none_baseline_when_column_absent(self):
        df = pd.DataFrame(
            [
                {
                    "Scenario": "CPMRCs",
                    "Test point": "(2, 1)",
                    "Range": "Standard",
                    "OEM Prediction": "Green",
                    "Expected value": "v_impact",
                    "Value": float("nan"),
                    "Impact location": "25%",
                    "VUT speed": "4 km/h",
                }
            ]
        )
        _, computed_info = read_test_points(df, preserve_none=True)
        entry = computed_info["CPMRCs_(2, 1)"]
        self.assertIsNone(entry["value_baseline"])


if __name__ == "__main__":
    unittest.main()
