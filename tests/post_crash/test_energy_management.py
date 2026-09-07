# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import unittest

import pandas as pd

from euroncap_rating_2026.post_crash import energy_management


SCENARIOS = [
    "Compliance with UN Regulation requirements",
    "Deactivation of high voltage energy - Automatic",
    "Deactivation of high voltage energy - First manual",
    "Deactivation of high voltage energy - Second manual",
    "Fulfilment of UN-R100.03 with predefined leadtime",
    "Thermal propagation detection communication inside the vehicle",
    "Thermal propagation detection communication to the car owner during charging",
    "Thermal propagation detection communication for the people around the car during charging",
]


def _rows_for_scenario(
    scenario: str, values: list[str], category: str | None = None
) -> list[dict]:
    rows = []
    for idx, value in enumerate(values):
        rows.append(
            {
                "Category": category if idx == 0 else None,
                "Scenario": scenario if idx == 0 else None,
                "Value": value,
            }
        )
    return rows


def _build_verification_df(values_per_scenario: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for idx, (scenario, values) in enumerate(values_per_scenario.items()):
        category = "Energy isolation" if idx == 0 else None
        rows.extend(_rows_for_scenario(scenario, values, category))
    return pd.DataFrame(rows, columns=["Category", "Scenario", "Value"])


class TestEnergyManagementPreprocess(unittest.TestCase):
    def test_preprocess_returns_expected_static_table(self):
        result_df = energy_management.preprocess({})

        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(list(result_df.columns), ["Category", "Scenario", "Value"])
        self.assertEqual(len(result_df), 8)

        self.assertIn(
            "Compliance with UN Regulation requirements", result_df["Scenario"].values
        )
        self.assertIn(
            "Deactivation of high voltage energy - Automatic",
            result_df["Scenario"].values,
        )
        self.assertIn(
            "Fulfilment of UN-R100.03 with predefined leadtime",
            result_df["Scenario"].values,
        )


class TestEnergyManagementComputeScore(unittest.TestCase):
    def test_missing_or_empty_verification_returns_all_unassessed(self):
        """A missing/empty sheet means nothing has been assessed yet --
        unassessed (NaN) for every scenario, not a false 0."""
        missing_result = energy_management.compute_score({})
        empty_result = energy_management.compute_score(
            {"VE - Energy Management Verif": pd.DataFrame()}
        )

        for scenario in SCENARIOS:
            self.assertTrue(pd.isna(missing_result[scenario]), scenario)
            self.assertTrue(pd.isna(empty_result[scenario]), scenario)

    def test_blank_value_is_unassessed_not_zero(self):
        df = _build_verification_df(
            {"Compliance with UN Regulation requirements": [None]}
        )
        result = energy_management.compute_score({"VE - Energy Management Verif": df})

        self.assertTrue(pd.isna(result["Compliance with UN Regulation requirements"]))
        for scenario in SCENARIOS:
            if scenario != "Compliance with UN Regulation requirements":
                self.assertTrue(pd.isna(result[scenario]), scenario)

    def test_explicit_fail_scores_zero_not_nan(self):
        df = _build_verification_df(
            {"Compliance with UN Regulation requirements": ["FAIL"]}
        )
        result = energy_management.compute_score({"VE - Energy Management Verif": df})

        self.assertEqual(result["Compliance with UN Regulation requirements"], 0.0)

    def test_leadtime_below_lowest_band_scores_zero_not_nan(self):
        df = _build_verification_df(
            {"Fulfilment of UN-R100.03 with predefined leadtime": ["≤20 min"]}
        )
        result = energy_management.compute_score({"VE - Energy Management Verif": df})

        self.assertEqual(
            result["Fulfilment of UN-R100.03 with predefined leadtime"], 0.0
        )

    def test_all_pass_with_90_min_gives_max_total(self):
        df = _build_verification_df(
            {
                "Compliance with UN Regulation requirements": ["PASS"],
                "Deactivation of high voltage energy - Automatic": ["PASS"],
                "Deactivation of high voltage energy - First manual": ["PASS"],
                "Deactivation of high voltage energy - Second manual": ["PASS"],
                "Fulfilment of UN-R100.03 with predefined leadtime": ["≥90 min"],
                "Thermal propagation detection communication inside the vehicle": [
                    "PASS"
                ],
                "Thermal propagation detection communication to the car owner during charging": [
                    "PASS"
                ],
                "Thermal propagation detection communication for the people around the car during charging": [
                    "PASS"
                ],
            }
        )

        result = energy_management.compute_score({"VE - Energy Management Verif": df})

        self.assertEqual(result["Compliance with UN Regulation requirements"], 3.0)
        self.assertEqual(result["Deactivation of high voltage energy - Automatic"], 5.0)
        self.assertEqual(
            result["Deactivation of high voltage energy - First manual"], 2.0
        )
        self.assertEqual(
            result["Deactivation of high voltage energy - Second manual"], 1.0
        )
        self.assertEqual(
            result["Fulfilment of UN-R100.03 with predefined leadtime"], 9.0
        )
        self.assertEqual(
            result["Thermal propagation detection communication inside the vehicle"],
            2.0,
        )
        self.assertEqual(
            result[
                "Thermal propagation detection communication to the car owner during charging"
            ],
            1.0,
        )
        self.assertEqual(
            result[
                "Thermal propagation detection communication for the people around the car during charging"
            ],
            1.0,
        )

    def test_fulfilment_tier_scoring(self):
        for value, expected in [
            ("≥90 min", 9.0),
            (">40 min", 6.0),
            (">20 min", 3.0),
            ("≤20 min", 0.0),
        ]:
            with self.subTest(value=value):
                df = _build_verification_df(
                    {
                        "Fulfilment of UN-R100.03 with predefined leadtime": [value],
                    }
                )
                result = energy_management.compute_score(
                    {"VE - Energy Management Verif": df}
                )
                self.assertEqual(
                    result["Fulfilment of UN-R100.03 with predefined leadtime"],
                    expected,
                )

    def test_pass_matching_is_case_insensitive(self):
        df = _build_verification_df(
            {
                "Compliance with UN Regulation requirements": ["pass"],
                "Deactivation of high voltage energy - Automatic": ["Pass"],
                "Deactivation of high voltage energy - First manual": ["pAsS"],
                "Deactivation of high voltage energy - Second manual": ["PASS"],
                "Thermal propagation detection communication inside the vehicle": [
                    "Pass"
                ],
                "Thermal propagation detection communication to the car owner during charging": [
                    "pass"
                ],
                "Thermal propagation detection communication for the people around the car during charging": [
                    "pAsS"
                ],
            }
        )

        result = energy_management.compute_score({"VE - Energy Management Verif": df})

        self.assertEqual(result["Compliance with UN Regulation requirements"], 3.0)
        self.assertEqual(result["Deactivation of high voltage energy - Automatic"], 5.0)
        self.assertEqual(
            result["Deactivation of high voltage energy - First manual"], 2.0
        )
        self.assertEqual(
            result["Deactivation of high voltage energy - Second manual"], 1.0
        )
        self.assertEqual(
            result["Thermal propagation detection communication inside the vehicle"],
            2.0,
        )
        self.assertEqual(
            result[
                "Thermal propagation detection communication to the car owner during charging"
            ],
            1.0,
        )
        self.assertEqual(
            result[
                "Thermal propagation detection communication for the people around the car during charging"
            ],
            1.0,
        )

    def test_multiple_rows_for_scenario_uses_first_row(self):
        df = _build_verification_df(
            {
                "Deactivation of high voltage energy - Automatic": ["PASS", "FAIL"],
                "Fulfilment of UN-R100.03 with predefined leadtime": [
                    ">40 min",
                    "≥90 min",
                ],
            }
        )

        result = energy_management.compute_score({"VE - Energy Management Verif": df})

        self.assertEqual(result["Deactivation of high voltage energy - Automatic"], 5.0)
        self.assertEqual(
            result["Fulfilment of UN-R100.03 with predefined leadtime"], 6.0
        )


if __name__ == "__main__":
    unittest.main()
