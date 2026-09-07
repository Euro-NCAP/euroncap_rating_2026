# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import unittest

import pandas as pd

from euroncap_rating_2026.post_crash import mcb_hazard_lights


def _df(mcb_values=None, hazard_values=None):
    rows = []
    if mcb_values is not None:
        for i, value in enumerate(mcb_values):
            rows.append(
                {
                    "Category": "Advanced multi-collision brake" if i == 0 else None,
                    "Value": value,
                }
            )
    if hazard_values is not None:
        for i, value in enumerate(hazard_values):
            rows.append(
                {
                    "Category": (
                        "Automatic activation of hazard warning lights"
                        if i == 0
                        else None
                    ),
                    "Value": value,
                }
            )
    return pd.DataFrame(rows, columns=["Category", "Value"])


class TestMcbHazardLightsComputeScore(unittest.TestCase):
    def test_missing_sheet_returns_unassessed_scores(self):
        """A missing sheet means nothing has been assessed yet -- unassessed
        (NaN), not a false 0."""
        result = mcb_hazard_lights.compute_score({})

        self.assertTrue(pd.isna(result["Advanced multi-collision brake"]))
        self.assertTrue(
            pd.isna(result["Automatic activation of hazard warning lights"])
        )

    def test_blank_value_is_unassessed_not_zero(self):
        dfs = {
            "PCI - MCB & Hazard lights Verif": _df(
                mcb_values=[None], hazard_values=["PASS"]
            )
        }

        result = mcb_hazard_lights.compute_score(dfs)

        self.assertTrue(pd.isna(result["Advanced multi-collision brake"]))
        self.assertEqual(result["Automatic activation of hazard warning lights"], 1.0)

    def test_explicit_fail_scores_zero_not_nan(self):
        dfs = {
            "PCI - MCB & Hazard lights Verif": _df(
                mcb_values=["FAIL"], hazard_values=["PASS"]
            )
        }

        result = mcb_hazard_lights.compute_score(dfs)

        self.assertEqual(result["Advanced multi-collision brake"], 0.0)
        self.assertEqual(result["Automatic activation of hazard warning lights"], 1.0)

    def test_all_pass_scores_full_marks(self):
        dfs = {
            "PCI - MCB & Hazard lights Verif": _df(
                mcb_values=["PASS"], hazard_values=["PASS"]
            )
        }

        result = mcb_hazard_lights.compute_score(dfs)

        self.assertEqual(result["Advanced multi-collision brake"], 4.0)
        self.assertEqual(result["Automatic activation of hazard warning lights"], 1.0)

    def test_failed_section_only_affects_its_own_score(self):
        dfs = {
            "PCI - MCB & Hazard lights Verif": _df(
                mcb_values=["PASS", "FAIL"], hazard_values=["PASS"]
            )
        }

        result = mcb_hazard_lights.compute_score(dfs)

        self.assertEqual(result["Advanced multi-collision brake"], 0.0)
        self.assertEqual(result["Automatic activation of hazard warning lights"], 1.0)


if __name__ == "__main__":
    unittest.main()
