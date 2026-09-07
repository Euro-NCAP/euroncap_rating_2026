# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""N/A (and blank) handling for the crash_avoidance "Input parameters"
prediction-method values: an N/A or unfilled "Prediction - Standard"/
"Prediction - Extended" means the scenario is not claimed, so
get_prediction_methods returns the (None, None) sentinel and the caller
scores all three layers 0 instead of crashing."""

import unittest

import numpy as np
import pandas as pd

from euroncap_rating_2026.crash_avoidance import matrix_processing


def _input_params_df(standard, extended):
    return pd.DataFrame(
        {
            "Scenario": ["CCRs", np.nan],
            "Input parameter": ["Prediction - Standard", "Prediction - Extended"],
            "Value": [standard, extended],
        }
    )


class TestGetPredictionMethodsNotApplicable(unittest.TestCase):
    def test_both_real_values_unchanged(self):
        standard, extended = matrix_processing.get_prediction_methods(
            _input_params_df("VTA", "Self claimed"), "CCRs"
        )
        self.assertEqual(standard, matrix_processing.PredictionSource.VTA)
        self.assertEqual(extended, matrix_processing.PredictionSource.SELF_CLAIMED)

    def test_na_standard_returns_sentinel(self):
        self.assertEqual(
            matrix_processing.get_prediction_methods(
                _input_params_df("N/A", "VTA"), "CCRs"
            ),
            (None, None),
        )

    def test_na_extended_returns_sentinel(self):
        self.assertEqual(
            matrix_processing.get_prediction_methods(
                _input_params_df("VTA", "N/A"), "CCRs"
            ),
            (None, None),
        )

    def test_both_na_returns_sentinel(self):
        self.assertEqual(
            matrix_processing.get_prediction_methods(
                _input_params_df("N/A", "n/a"), "CCRs"
            ),
            (None, None),
        )

    def test_blank_returns_sentinel_instead_of_crashing(self):
        # Regression: NaN previously crashed with AttributeError on .strip().
        self.assertEqual(
            matrix_processing.get_prediction_methods(
                _input_params_df(np.nan, "VTA"), "CCRs"
            ),
            (None, None),
        )
        self.assertEqual(
            matrix_processing.get_prediction_methods(
                _input_params_df("", np.nan), "CCRs"
            ),
            (None, None),
        )

    def test_pandas_na_returns_sentinel(self):
        # pd.NA (nullable dtypes from parquet-sourced dfs) must classify as
        # blank, not fall through to the unknown-value ValueError.
        self.assertEqual(
            matrix_processing.get_prediction_methods(
                _input_params_df(pd.NA, "VTA"), "CCRs"
            ),
            (None, None),
        )

    def test_unknown_non_blank_value_still_raises(self):
        with self.assertRaises(ValueError):
            matrix_processing.get_prediction_methods(
                _input_params_df("Typo", "VTA"), "CCRs"
            )

    def test_missing_scenario_returns_sentinel(self):
        self.assertEqual(
            matrix_processing.get_prediction_methods(
                _input_params_df("VTA", "VTA"), "Nonexistent"
            ),
            (None, None),
        )

    def test_na_in_any_scenario_parameter_returns_sentinel(self):
        # ELK RE's slice carries a third parameter (Extended range
        # performance): an N/A there zeroes the whole scenario too, even
        # with both prediction methods filled.
        elk_re_df = pd.DataFrame(
            {
                "Scenario": ["ELK RE", np.nan, np.nan],
                "Input parameter": [
                    "Prediction - Standard",
                    "Prediction - Extended",
                    "Extended range performance",
                ],
                "Value": ["VTA", "VTA", "N/A"],
            }
        )
        self.assertEqual(
            matrix_processing.get_prediction_methods(elk_re_df, "ELK RE"),
            (None, None),
        )

    def test_erp_real_value_does_not_zero_elk_re(self):
        elk_re_df = pd.DataFrame(
            {
                "Scenario": ["ELK RE", np.nan, np.nan],
                "Input parameter": [
                    "Prediction - Standard",
                    "Prediction - Extended",
                    "Extended range performance",
                ],
                "Value": ["VTA", "VTA", "LDW"],
            }
        )
        standard, extended = matrix_processing.get_prediction_methods(
            elk_re_df, "ELK RE"
        )
        self.assertEqual(standard, matrix_processing.PredictionSource.VTA)
        self.assertEqual(extended, matrix_processing.PredictionSource.VTA)


if __name__ == "__main__":
    unittest.main()
