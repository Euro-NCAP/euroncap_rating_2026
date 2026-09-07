# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import unittest
import pandas as pd

from euroncap_rating_2026.crash_protection.dummy import Dummy


class TestBioRID50CoupledColor(unittest.TestCase):
    """
    For BioRID-50 dynamic whiplash tests, T-HRC start and T1 acceleration must
    BOTH exceed their capping limit to be colored red. If only one (or neither)
    exceeds its limit, both must be green.
    """

    def _make_biorid_df(self, thrc_value, t1_value, thrc_cap=100.0, t1_cap=10.0):
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

    def _get_criteria_colors(self, dummy):
        colors = {}
        for br in dummy.body_region_list:
            for c in br._criteria:
                colors[c.name] = c.color
        return colors

    def test_both_above_cap_both_red(self):
        """Both T-HRC start and T1 acceleration exceed capping → both red."""
        df = self._make_biorid_df(thrc_value=150.0, t1_value=15.0)
        dummy, _ = Dummy.get_dummy_from_row("BioRID-50", df, 0)
        colors = self._get_criteria_colors(dummy)
        self.assertEqual(colors["T-HRC start"], "red")
        self.assertEqual(colors["T1 acceleration"], "red")

    def test_only_thrc_above_cap_both_green(self):
        """Only T-HRC start exceeds capping → both must be green."""
        df = self._make_biorid_df(thrc_value=150.0, t1_value=5.0)
        dummy, _ = Dummy.get_dummy_from_row("BioRID-50", df, 0)
        colors = self._get_criteria_colors(dummy)
        self.assertEqual(colors["T-HRC start"], "green")
        self.assertEqual(colors["T1 acceleration"], "green")

    def test_only_t1_above_cap_both_green(self):
        """Only T1 acceleration exceeds capping → both must be green."""
        df = self._make_biorid_df(thrc_value=50.0, t1_value=15.0)
        dummy, _ = Dummy.get_dummy_from_row("BioRID-50", df, 0)
        colors = self._get_criteria_colors(dummy)
        self.assertEqual(colors["T-HRC start"], "green")
        self.assertEqual(colors["T1 acceleration"], "green")

    def test_neither_above_cap_both_green(self):
        """Neither coupled criterion exceeds capping → both green."""
        df = self._make_biorid_df(thrc_value=50.0, t1_value=5.0)
        dummy, _ = Dummy.get_dummy_from_row("BioRID-50", df, 0)
        colors = self._get_criteria_colors(dummy)
        self.assertEqual(colors["T-HRC start"], "green")
        self.assertEqual(colors["T1 acceleration"], "green")

    def test_other_dummy_colored_independently(self):
        """For non-BioRID-50 dummies (e.g. HIII-50), each criterion is colored independently."""
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
        colors = self._get_criteria_colors(dummy)
        self.assertEqual(colors["T-HRC start"], "red")
        self.assertEqual(colors["T1 acceleration"], "green")

    def test_third_criteria_unaffected(self):
        """A non-coupled third criterion is colored independently, even when both coupled are red."""
        df = pd.DataFrame(
            {
                "Dummy": ["BioRID-50", None, None],
                "Body region": ["Neck", None, None],
                "Criteria": ["T-HRC start", "T1 acceleration", "Some other criteria"],
                "HPL": [None, None, None],
                "LPL": [None, None, None],
                "Capping": [100.0, 10.0, 50.0],
                "Value": [150.0, 15.0, 30.0],  # coupled both exceed; third does not
            }
        )
        dummy, _ = Dummy.get_dummy_from_row("BioRID-50", df, 0)
        colors = self._get_criteria_colors(dummy)
        self.assertEqual(colors["T-HRC start"], "red")
        self.assertEqual(colors["T1 acceleration"], "red")
        self.assertEqual(colors["Some other criteria"], "green")

    def test_third_criteria_red_when_coupled_not_both_exceeded(self):
        """Third criterion turns red independently even when coupled criteria stay green."""
        df = pd.DataFrame(
            {
                "Dummy": ["BioRID-50", None, None],
                "Body region": ["Neck", None, None],
                "Criteria": ["T-HRC start", "T1 acceleration", "Some other criteria"],
                "HPL": [None, None, None],
                "LPL": [None, None, None],
                "Capping": [100.0, 10.0, 50.0],
                "Value": [150.0, 5.0, 99.0],  # only T-HRC and third exceed cap
            }
        )
        dummy, _ = Dummy.get_dummy_from_row("BioRID-50", df, 0)
        colors = self._get_criteria_colors(dummy)
        self.assertEqual(colors["T-HRC start"], "green")  # coupled rule forces green
        self.assertEqual(colors["T1 acceleration"], "green")
        self.assertEqual(colors["Some other criteria"], "red")


if __name__ == "__main__":
    unittest.main()
