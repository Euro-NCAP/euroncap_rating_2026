# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import unittest

import pandas as pd

from euroncap_rating_2026.post_crash import occupant_extrication


EXPECTED_KEYS = [
    "Seat belt buckle unlatching",
    "Door opening - interior, post low voltage drop",
    "Door opening - exterior, post crash",
    "Door opening - exterior, post crash, post low voltage drop",
    "Tailgate opening",
    "Window opening",
    "Rescue tool / Emergency device",
]

MAX_SCORES = {
    "Seat belt buckle unlatching": 1.0,
    "Door opening - interior, post low voltage drop": 3.0,
    "Door opening - exterior, post crash": 4.0,
    "Door opening - exterior, post crash, post low voltage drop": 2.0,
    "Tailgate opening": 2.0,
    "Window opening": 3.0,
    "Rescue tool / Emergency device": 1.0,
}


def make_verif_df(
    seatbelt="PASS",
    door_interior="PASS",
    door_exterior="PASS",
    door_exterior_low="PASS",
    tailgate="PASS",
    window="PASS",
    rescue="PASS",
):
    rows = [
        {
            "Category": "Doors and belts",
            "Scenario": "Seat belt buckle unlatching",
            "Value": seatbelt,
        },
        {
            "Category": "Doors and belts",
            "Scenario": "Door opening - interior, post low voltage drop",
            "Value": door_interior,
        },
        {
            "Category": "Doors and belts",
            "Scenario": "Door opening - exterior, post crash",
            "Value": door_exterior,
        },
        {
            "Category": "Doors and belts",
            "Scenario": "Door opening - exterior, post crash, post low voltage drop",
            "Value": door_exterior_low,
        },
        {
            "Category": "Doors and belts",
            "Scenario": "Tailgate opening",
            "Value": tailgate,
        },
        {"Category": "Submergence", "Scenario": "Window opening", "Value": window},
        {
            "Category": "Submergence",
            "Scenario": "Rescue tool / Emergency device",
            "Value": rescue,
        },
    ]
    return pd.DataFrame(rows)


class TestOccupantExtricationPreprocess(unittest.TestCase):
    def test_returns_static_template(self):
        result_df = occupant_extrication.preprocess({"any": pd.DataFrame()})
        self.assertEqual(result_df.columns.tolist(), ["Category", "Scenario", "Value"])
        self.assertEqual(len(result_df), 7)
        # Default is blank ("not yet assessed"), not a false "PASS".
        self.assertTrue(result_df["Value"].isna().all())


class TestOccupantExtricationComputeScore(unittest.TestCase):
    def test_missing_verification_sheet_returns_all_unassessed(self):
        """A missing sheet means nothing has been assessed yet -- unassessed
        (NaN), not a false 0."""
        result = occupant_extrication.compute_score({})
        self.assertEqual(set(result.keys()), set(EXPECTED_KEYS))
        self.assertTrue(all(pd.isna(score) for score in result.values()))

    def test_blank_value_is_unassessed_not_zero(self):
        df = make_verif_df(seatbelt=None)
        result = occupant_extrication.compute_score(
            {"VE - Occupant Extrication Verif": df}
        )
        self.assertTrue(pd.isna(result["Seat belt buckle unlatching"]))

    def test_explicit_fail_scores_zero_not_nan(self):
        df = make_verif_df(seatbelt="FAIL")
        result = occupant_extrication.compute_score(
            {"VE - Occupant Extrication Verif": df}
        )
        self.assertEqual(result["Seat belt buckle unlatching"], 0.0)

    def test_all_pass_returns_max_scores(self):
        result = occupant_extrication.compute_score(
            {"VE - Occupant Extrication Verif": make_verif_df()}
        )
        for scenario, max_score in MAX_SCORES.items():
            self.assertEqual(result[scenario], max_score, msg=f"Failed for {scenario}")

    def test_all_fail_returns_all_zeros(self):
        result = occupant_extrication.compute_score(
            {
                "VE - Occupant Extrication Verif": make_verif_df(
                    seatbelt="FAIL",
                    door_interior="FAIL",
                    door_exterior="FAIL",
                    door_exterior_low="FAIL",
                    tailgate="FAIL",
                    window="FAIL",
                    rescue="FAIL",
                )
            }
        )
        self.assertTrue(all(score == 0.0 for score in result.values()))

    def test_seatbelt_fail_deducts_only_its_score(self):
        result = occupant_extrication.compute_score(
            {"VE - Occupant Extrication Verif": make_verif_df(seatbelt="FAIL")}
        )
        self.assertEqual(result["Seat belt buckle unlatching"], 0.0)
        self.assertEqual(result["Door opening - interior, post low voltage drop"], 3.0)
        self.assertEqual(result["Door opening - exterior, post crash"], 4.0)
        self.assertEqual(
            result["Door opening - exterior, post crash, post low voltage drop"], 2.0
        )
        self.assertEqual(result["Tailgate opening"], 2.0)
        self.assertEqual(result["Window opening"], 3.0)
        self.assertEqual(result["Rescue tool / Emergency device"], 1.0)

    def test_door_interior_fail_deducts_only_its_score(self):
        result = occupant_extrication.compute_score(
            {"VE - Occupant Extrication Verif": make_verif_df(door_interior="FAIL")}
        )
        self.assertEqual(result["Door opening - interior, post low voltage drop"], 0.0)
        self.assertEqual(result["Seat belt buckle unlatching"], 1.0)
        self.assertEqual(result["Door opening - exterior, post crash"], 4.0)

    def test_door_exterior_fail_deducts_only_its_score(self):
        result = occupant_extrication.compute_score(
            {"VE - Occupant Extrication Verif": make_verif_df(door_exterior="FAIL")}
        )
        self.assertEqual(result["Door opening - exterior, post crash"], 0.0)
        self.assertEqual(
            result["Door opening - exterior, post crash, post low voltage drop"], 2.0
        )

    def test_door_exterior_low_fail_deducts_only_its_score(self):
        result = occupant_extrication.compute_score(
            {"VE - Occupant Extrication Verif": make_verif_df(door_exterior_low="FAIL")}
        )
        self.assertEqual(
            result["Door opening - exterior, post crash, post low voltage drop"], 0.0
        )
        self.assertEqual(result["Door opening - exterior, post crash"], 4.0)

    def test_tailgate_fail_deducts_only_its_score(self):
        result = occupant_extrication.compute_score(
            {"VE - Occupant Extrication Verif": make_verif_df(tailgate="FAIL")}
        )
        self.assertEqual(result["Tailgate opening"], 0.0)
        self.assertEqual(result["Window opening"], 3.0)

    def test_window_fail_deducts_only_its_score(self):
        result = occupant_extrication.compute_score(
            {"VE - Occupant Extrication Verif": make_verif_df(window="FAIL")}
        )
        self.assertEqual(result["Window opening"], 0.0)
        self.assertEqual(result["Rescue tool / Emergency device"], 1.0)

    def test_rescue_fail_deducts_only_its_score(self):
        result = occupant_extrication.compute_score(
            {"VE - Occupant Extrication Verif": make_verif_df(rescue="FAIL")}
        )
        self.assertEqual(result["Rescue tool / Emergency device"], 0.0)
        self.assertEqual(result["Window opening"], 3.0)

    def test_input_parameters_have_no_effect(self):
        dfs_with_params = {
            "VE - Occupant Extrication Verif": make_verif_df(),
            "Input parameters": pd.DataFrame(
                [
                    {
                        "Stage subelement": "Occupant extrication",
                        "Input parameter": "Interior side door handles type",
                        "Value": "Electric",
                    }
                ]
            ),
        }
        dfs_without_params = {"VE - Occupant Extrication Verif": make_verif_df()}

        result_with = occupant_extrication.compute_score(dfs_with_params)
        result_without = occupant_extrication.compute_score(dfs_without_params)

        self.assertEqual(result_with, result_without)


if __name__ == "__main__":
    unittest.main()
