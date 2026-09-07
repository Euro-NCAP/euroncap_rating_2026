# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import unittest

import pandas as pd

from euroncap_rating_2026.post_crash import rescue_info


class TestRescueInfoComputeScore(unittest.TestCase):
    def test_missing_all_sheets_returns_unassessed_scores(self):
        """A missing sheet means nothing has been assessed yet -- unassessed
        (NaN), not a false 0."""
        result = rescue_info.compute_score({})

        self.assertTrue(pd.isna(result["Rescue sheets"]))
        self.assertTrue(pd.isna(result["Emergency response guide"]))

    def test_missing_erg_sheet_still_scores_rescue_sheets(self):
        dfs = {
            "RI - RS Verification": pd.DataFrame({"Value": ["PASS", "pass", "Pass"]}),
        }

        result = rescue_info.compute_score(dfs)

        self.assertEqual(result["Rescue sheets"], 35.0)
        self.assertTrue(pd.isna(result["Emergency response guide"]))

    def test_missing_rescue_sheet_still_scores_erg(self):
        dfs = {
            "RI - ERG Verification": pd.DataFrame({"Value": ["PASS", "PASS"]}),
        }

        result = rescue_info.compute_score(dfs)

        self.assertTrue(pd.isna(result["Rescue sheets"]))
        self.assertEqual(result["Emergency response guide"], 5.0)

    def test_blank_value_is_unassessed_not_zero(self):
        dfs = {
            "RI - RS Verification": pd.DataFrame({"Value": [None]}),
            "RI - ERG Verification": pd.DataFrame({"Value": ["PASS"]}),
        }

        result = rescue_info.compute_score(dfs)

        self.assertTrue(pd.isna(result["Rescue sheets"]))
        self.assertEqual(result["Emergency response guide"], 5.0)

    def test_explicit_fail_scores_zero_not_nan(self):
        dfs = {
            "RI - RS Verification": pd.DataFrame({"Value": ["FAIL"]}),
        }

        result = rescue_info.compute_score(dfs)

        self.assertEqual(result["Rescue sheets"], 0.0)

    def test_failed_sheet_only_affects_its_own_score(self):
        dfs = {
            "RI - RS Verification": pd.DataFrame({"Value": ["PASS", "FAIL"]}),
            "RI - ERG Verification": pd.DataFrame({"Value": ["PASS", "PASS"]}),
        }

        result = rescue_info.compute_score(dfs)

        self.assertEqual(result["Rescue sheets"], 0)
        self.assertEqual(result["Emergency response guide"], 5.0)


if __name__ == "__main__":
    unittest.main()
