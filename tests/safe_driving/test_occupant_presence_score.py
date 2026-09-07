# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from euroncap_rating_2026.safe_driving import occupant_presence
from euroncap_rating_2026.safe_driving import data_model


class TestComputeClassificationScoreBasic(unittest.TestCase):
    """Test basic behavior of the compute_classification_score function."""

    def test_compute_classification_score_missing_dataframe(self):
        """Test that missing dataframe returns empty results."""
        dfs = {}
        result_df, score_dict = occupant_presence.compute_classification_score(dfs)

        self.assertTrue(result_df.empty)
        self.assertEqual(score_dict, {})

    def test_compute_classification_score_empty_dataframe(self):
        """Test that empty dataframe returns empty results."""
        dfs = {"OM - Occupant Presence": pd.DataFrame()}
        result_df, score_dict = occupant_presence.compute_classification_score(dfs)

        self.assertTrue(result_df.empty)
        self.assertEqual(score_dict, {})

    def test_compute_classification_score_returns_tuple(self):
        """Test that compute_classification_score returns a tuple of (DataFrame, dict)."""
        dfs = {"OM - Occupant Presence": pd.DataFrame()}
        result = occupant_presence.compute_classification_score(dfs)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], pd.DataFrame)
        self.assertIsInstance(result[1], dict)


class TestChildLeftBehindScoringLogic(unittest.TestCase):
    """Test the scoring logic for Child left behind based on system type and seat coverage."""

    def test_warning_rear_seats_scores_1_5(self):
        """Test that Warning + Rear seats gives 1.5 points (according to data_model)."""
        # Based on the scoring logic in occupant_presence_score.py
        type_of_system = data_model.OPTypeOfSystem.WARNING
        seat_coverage = data_model.OPSeatCoverage.REAR_SEATS

        # Score logic: Warning = 1.5 for rear seats, 3.0 for all passenger seats
        # Warning&Intervention = 3.0 for rear seats, 4.0 for all passenger seats
        if type_of_system == data_model.OPTypeOfSystem.WARNING:
            if seat_coverage == data_model.OPSeatCoverage.REAR_SEATS:
                expected_score = 1.5
            else:
                expected_score = 3.0
        else:
            if seat_coverage == data_model.OPSeatCoverage.REAR_SEATS:
                expected_score = 3.0
            else:
                expected_score = 4.0

        self.assertEqual(expected_score, 1.5)

    def test_warning_all_passenger_seats_scores_3_0(self):
        """Test that Warning + All passenger seats gives 3.0 points."""
        type_of_system = data_model.OPTypeOfSystem.WARNING
        seat_coverage = data_model.OPSeatCoverage.ALL_PASSENGER_SEATS

        if type_of_system == data_model.OPTypeOfSystem.WARNING:
            if seat_coverage == data_model.OPSeatCoverage.REAR_SEATS:
                expected_score = 1.5
            else:
                expected_score = 3.0
        else:
            if seat_coverage == data_model.OPSeatCoverage.REAR_SEATS:
                expected_score = 3.0
            else:
                expected_score = 4.0

        self.assertEqual(expected_score, 3.0)

    def test_warning_intervention_rear_seats_scores_3_0(self):
        """Test that Warning&Intervention + Rear seats gives 3.0 points."""
        type_of_system = data_model.OPTypeOfSystem.WARNING_AND_INTERVENTION
        seat_coverage = data_model.OPSeatCoverage.REAR_SEATS

        if type_of_system == data_model.OPTypeOfSystem.WARNING:
            if seat_coverage == data_model.OPSeatCoverage.REAR_SEATS:
                expected_score = 1.5
            else:
                expected_score = 3.0
        else:
            if seat_coverage == data_model.OPSeatCoverage.REAR_SEATS:
                expected_score = 3.0
            else:
                expected_score = 4.0

        self.assertEqual(expected_score, 3.0)

    def test_warning_intervention_all_passenger_seats_scores_4_0(self):
        """Test that Warning&Intervention + All passenger seats gives 4.0 points."""
        type_of_system = data_model.OPTypeOfSystem.WARNING_AND_INTERVENTION
        seat_coverage = data_model.OPSeatCoverage.ALL_PASSENGER_SEATS

        if type_of_system == data_model.OPTypeOfSystem.WARNING:
            if seat_coverage == data_model.OPSeatCoverage.REAR_SEATS:
                expected_score = 1.5
            else:
                expected_score = 3.0
        else:
            if seat_coverage == data_model.OPSeatCoverage.REAR_SEATS:
                expected_score = 3.0
            else:
                expected_score = 4.0

        self.assertEqual(expected_score, 4.0)


class TestChildEntersUnlockedVehicleScoringLogic(unittest.TestCase):
    """Test the scoring logic for Child enters unlocked vehicle."""

    def test_rear_seats_coverage_scores_0_5(self):
        """Test that Rear seats coverage gives 0.5 points."""
        seat_coverage = data_model.OPSeatCoverage.REAR_SEATS

        if seat_coverage == data_model.OPSeatCoverage.REAR_SEATS:
            expected_score = 0.5
        else:
            expected_score = 1.0

        self.assertEqual(expected_score, 0.5)

    def test_all_passenger_seats_coverage_scores_1_0(self):
        """Test that All passenger seats coverage gives 1.0 points."""
        seat_coverage = data_model.OPSeatCoverage.ALL_PASSENGER_SEATS

        if seat_coverage == data_model.OPSeatCoverage.REAR_SEATS:
            expected_score = 0.5
        else:
            expected_score = 1.0

        self.assertEqual(expected_score, 1.0)


class TestCrashOccupancyInformationScoringLogic(unittest.TestCase):
    """Test the scoring logic for Crash occupancy information."""

    def _make_dfs(self, adult_value: str, children_value: str) -> dict:
        """Build a minimal OM verification DataFrame for crash-occupancy tests.

        Child-presence-detection General requirements = FAIL so the complex CPD
        merge logic is bypassed; only the crash-occupancy branch is exercised.
        """
        om_df = pd.DataFrame(
            [
                {
                    "Category": "Child presence detection",
                    "Scenario": "General requirements",
                    "Value": "FAIL",
                    "Element": np.nan,
                },
                {
                    "Category": "Crash occupancy information",
                    "Scenario": "General requirements",
                    "Value": "PASS",
                    "Element": np.nan,
                },
                {
                    "Category": "Crash occupancy information",
                    "Scenario": "Adult occupants",
                    "Value": adult_value,
                    "Element": np.nan,
                },
                {
                    "Category": "Crash occupancy information",
                    "Scenario": "Children in all CRS",
                    "Value": children_value,
                    "Element": np.nan,
                },
            ]
        )
        return {"OM - Occ. presence verif.": om_df}

    def test_adult_occupants_pass_scores_4(self):
        _, score_dict = occupant_presence.compute_classification_score(
            self._make_dfs("PASS", "FAIL")
        )
        self.assertEqual(score_dict["Adult occupants"], 4.0)

    def test_children_in_crs_pass_scores_1(self):
        _, score_dict = occupant_presence.compute_classification_score(
            self._make_dfs("FAIL", "PASS")
        )
        self.assertEqual(score_dict["Children in all CRS"], 1.0)

    def test_adult_occupants_fail_scores_0(self):
        _, score_dict = occupant_presence.compute_classification_score(
            self._make_dfs("FAIL", "PASS")
        )
        self.assertEqual(score_dict["Adult occupants"], 0.0)

    def test_children_in_crs_fail_scores_0(self):
        _, score_dict = occupant_presence.compute_classification_score(
            self._make_dfs("PASS", "FAIL")
        )
        self.assertEqual(score_dict["Children in all CRS"], 0.0)

    def test_both_fail_scores_0_0(self):
        _, score_dict = occupant_presence.compute_classification_score(
            self._make_dfs("FAIL", "FAIL")
        )
        self.assertEqual(score_dict["Adult occupants"], 0.0)
        self.assertEqual(score_dict["Children in all CRS"], 0.0)

    def test_both_pass_scores_4_1(self):
        _, score_dict = occupant_presence.compute_classification_score(
            self._make_dfs("PASS", "PASS")
        )
        self.assertEqual(score_dict["Adult occupants"], 4.0)
        self.assertEqual(score_dict["Children in all CRS"], 1.0)


class TestDataFrameShapedEmptyStringForwardFill(unittest.TestCase):
    """Data reconstructed from external storage (e.g. parquet) may represent
    a collapsed/continuation Category cell as an empty string, not
    NaN/pd.NA -- unlike the file-based CLI round trip (openpyxl/read_excel),
    which always produces real NaN for a blank cell. .ffill() only
    propagates over actual nulls, so an empty string must be normalized
    first or the Category never gets reconstructed and the later lookup by
    Category finds nothing -- previously an uncaught IndexError from
    .iloc[0] on an empty result."""

    def _make_dfs(self):
        om_df = pd.DataFrame(
            [
                {
                    "Category": "Child presence detection",
                    "Scenario": "General requirements",
                    "Value": "FAIL",
                    "Element": "",
                },
                {
                    "Category": "Crash occupancy information",
                    "Scenario": "General requirements",
                    "Value": "PASS",
                    "Element": "",
                },
                {
                    "Category": "",
                    "Scenario": "Adult occupants",
                    "Value": "PASS",
                    "Element": "",
                },
                {
                    "Category": "",
                    "Scenario": "Children in all CRS",
                    "Value": "PASS",
                    "Element": "",
                },
            ]
        )
        return {"OM - Occ. presence verif.": om_df}

    def test_does_not_raise_and_scores_correctly(self):
        _, score_dict = occupant_presence.compute_classification_score(self._make_dfs())
        self.assertEqual(score_dict["Adult occupants"], 4.0)
        self.assertEqual(score_dict["Children in all CRS"], 1.0)


class TestCrashOccupancyInformationUnassessed(unittest.TestCase):
    """A blank Value must score NaN (unassessed), never 0.0 (fail)."""

    def _make_dfs(self, general_req_value, adult_value=None, children_value=None):
        om_df = pd.DataFrame(
            [
                {
                    "Category": "Child presence detection",
                    "Scenario": "General requirements",
                    "Value": "FAIL",
                    "Element": np.nan,
                },
                {
                    "Category": "Crash occupancy information",
                    "Scenario": "General requirements",
                    "Value": general_req_value,
                    "Element": np.nan,
                },
                {
                    "Category": "Crash occupancy information",
                    "Scenario": "Adult occupants",
                    "Value": adult_value,
                    "Element": np.nan,
                },
                {
                    "Category": "Crash occupancy information",
                    "Scenario": "Children in all CRS",
                    "Value": children_value,
                    "Element": np.nan,
                },
            ]
        )
        return {"OM - Occ. presence verif.": om_df}

    def test_general_req_blank_leaves_both_scores_unassessed(self):
        _, score_dict = occupant_presence.compute_classification_score(
            self._make_dfs(None, "PASS", "PASS")
        )
        self.assertTrue(pd.isna(score_dict["Adult occupants"]))
        self.assertTrue(pd.isna(score_dict["Children in all CRS"]))

    def test_general_req_pass_but_adult_occupants_blank_is_unassessed_not_zero(self):
        _, score_dict = occupant_presence.compute_classification_score(
            self._make_dfs("PASS", None, "PASS")
        )
        self.assertTrue(pd.isna(score_dict["Adult occupants"]))
        self.assertEqual(score_dict["Children in all CRS"], 1.0)


class TestChildPresenceDetectionUnassessed(unittest.TestCase):
    """CPD scoring must distinguish blank (unassessed) from an element that
    genuinely doesn't apply to this vehicle (missing row -> FAIL, unchanged)."""

    ALL_PASS_ELEMENTS = [
        ("Child left behind", "Front Seat - Warning"),
        ("Child left behind", "Front Seat - Intervention"),
        ("Child left behind", "Rear Seat - Warning"),
        ("Child left behind", "Rear Seat - Intervention"),
        ("Child enters unlocked vehicle", "Front compartment - Warning"),
        ("Child enters unlocked vehicle", "Front compartment - Intervention"),
        ("Child enters unlocked vehicle", "Rear Seat - Warning"),
        ("Child enters unlocked vehicle", "Rear Seat - Intervention"),
    ]

    def _make_dfs(self, general_req_value, element_values):
        rows = [
            {
                "Category": "Child presence detection",
                "Scenario": "General requirements",
                "Element": np.nan,
                "Value": general_req_value,
            },
            # Bypass the (separately-tested) Crash occupancy information
            # branch so these tests only exercise CPD scoring.
            {
                "Category": "Crash occupancy information",
                "Scenario": "General requirements",
                "Element": np.nan,
                "Value": "FAIL",
            },
        ]
        for scenario, element in self.ALL_PASS_ELEMENTS:
            if (scenario, element) in element_values:
                rows.append(
                    {
                        "Category": "Child presence detection",
                        "Scenario": scenario,
                        "Element": element,
                        "Value": element_values[(scenario, element)],
                    }
                )
        return {"OM - Occ. presence verif.": pd.DataFrame(rows)}

    def test_general_req_blank_leaves_cpd_scores_unassessed(self):
        dfs = self._make_dfs(None, {k: "PASS" for k in self.ALL_PASS_ELEMENTS})
        _, score_dict = occupant_presence.compute_classification_score(dfs)

        self.assertTrue(pd.isna(score_dict["Child left behind"]))
        self.assertTrue(pd.isna(score_dict["Child enters unlocked vehicle"]))

    def test_general_req_pass_all_elements_pass_scores_per_table(self):
        dfs = self._make_dfs("PASS", {k: "PASS" for k in self.ALL_PASS_ELEMENTS})
        _, score_dict = occupant_presence.compute_classification_score(dfs)

        self.assertEqual(score_dict["Child left behind"], 4.0)
        self.assertEqual(score_dict["Child enters unlocked vehicle"], 1.0)

    def test_general_req_pass_one_applicable_element_blank_is_unassessed(self):
        element_values = {k: "PASS" for k in self.ALL_PASS_ELEMENTS}
        element_values[("Child left behind", "Front Seat - Warning")] = None
        dfs = self._make_dfs("PASS", element_values)

        _, score_dict = occupant_presence.compute_classification_score(dfs)

        self.assertTrue(pd.isna(score_dict["Child left behind"]))
        self.assertTrue(pd.isna(score_dict["Child enters unlocked vehicle"]))

    def test_general_req_pass_element_not_applicable_still_scores_per_table(self):
        # "Front Seat - Intervention" is entirely absent (feature not offered
        # for this vehicle) -- this is the pre-existing "missing -> FAIL"
        # rule, not a blank/unassessed cell, so it must still be scored,
        # matching cpd_score_df row: PASS,FAIL,PASS,FAIL,PASS,PASS,PASS,PASS -> 3,1
        element_values = {k: "PASS" for k in self.ALL_PASS_ELEMENTS}
        del element_values[("Child left behind", "Front Seat - Intervention")]
        del element_values[("Child left behind", "Rear Seat - Intervention")]
        dfs = self._make_dfs("PASS", element_values)

        _, score_dict = occupant_presence.compute_classification_score(dfs)

        self.assertEqual(score_dict["Child left behind"], 3.0)
        self.assertEqual(score_dict["Child enters unlocked vehicle"], 1.0)


class TestOPTypeOfSystemEnum(unittest.TestCase):
    """Test OPTypeOfSystem enum values in scoring."""

    def test_warning_system_type(self):
        """Test Warning system type enum value."""
        self.assertEqual(data_model.OPTypeOfSystem.WARNING.value, "Warning")

    def test_warning_and_intervention_system_type(self):
        """Test Warning&Intervention system type enum value."""
        self.assertEqual(
            data_model.OPTypeOfSystem.WARNING_AND_INTERVENTION.value,
            "Warning and intervention",
        )

    def test_no_system_type(self):
        """Test No system type enum value."""
        self.assertEqual(data_model.OPTypeOfSystem.NO.value, "No")


class TestOPSeatCoverageEnum(unittest.TestCase):
    """Test OPSeatCoverage enum values in scoring."""

    def test_rear_seats_coverage(self):
        """Test Rear seats coverage enum value."""
        self.assertEqual(data_model.OPSeatCoverage.REAR_SEATS.value, "Rear seats")

    def test_all_passenger_seats_coverage(self):
        """Test All passenger seats coverage enum value."""
        self.assertEqual(
            data_model.OPSeatCoverage.ALL_PASSENGER_SEATS.value, "All seats"
        )


class TestChildPresenceNotApplicable(unittest.TestCase):
    """An "N/A" Type of system or Seat coverage input parameter means the
    child-presence system is not fitted: no verification rows are generated
    and the scenario scores a real 0.0, even when the General requirements
    row is left blank (no input is required from the user)."""

    def _input_params(
        self,
        clb_system="N/A",
        clb_coverage="N/A",
        ceuv_system="N/A",
        ceuv_coverage="N/A",
    ):
        return pd.DataFrame(
            {
                "Scenario": [
                    "Child left behind",
                    None,
                    "Child enters unlocked vehicle",
                    None,
                ],
                "Input parameter": [
                    "Type of system",
                    "Seat coverage",
                    "Type of system",
                    "Seat coverage",
                ],
                "Value": [clb_system, clb_coverage, ceuv_system, ceuv_coverage],
            }
        )

    def _make_dfs(self, input_params):
        om_df = occupant_presence.preprocess({"Input parameters": input_params})
        om_df["Value"] = om_df["Value"].astype(object)
        return {
            "Input parameters": input_params,
            "OM - Occ. presence verif.": om_df,
        }

    def test_na_generates_no_cpd_element_rows(self):
        dfs = self._make_dfs(self._input_params())
        om_df = dfs["OM - Occ. presence verif."]

        cpd_element_rows = om_df[
            om_df["Element"].astype(str).str.strip().replace("nan", "") != ""
        ]
        self.assertTrue(cpd_element_rows.empty)

    def test_both_scenarios_na_score_zero_even_with_blank_general_req(self):
        dfs = self._make_dfs(self._input_params())

        _, score_dict = occupant_presence.compute_classification_score(dfs)

        self.assertEqual(score_dict["Child left behind"], 0.0)
        self.assertEqual(score_dict["Child enters unlocked vehicle"], 0.0)

    def test_na_seat_coverage_alone_scores_zero(self):
        dfs = self._make_dfs(
            self._input_params(clb_system="Warning", clb_coverage="N/A")
        )

        _, score_dict = occupant_presence.compute_classification_score(dfs)

        self.assertEqual(score_dict["Child left behind"], 0.0)

    def test_one_scenario_na_leaves_the_other_unassessed(self):
        dfs = self._make_dfs(
            self._input_params(
                ceuv_system="Warning and intervention", ceuv_coverage="All seats"
            )
        )

        _, score_dict = occupant_presence.compute_classification_score(dfs)

        self.assertEqual(score_dict["Child left behind"], 0.0)
        self.assertTrue(pd.isna(score_dict["Child enters unlocked vehicle"]))

    def test_na_scores_zero_with_non_default_index(self):
        # Caller-supplied dfs may carry a non-default index: the scenario
        # parameter extraction must select rows by position, not by label
        # arithmetic.
        input_params = self._input_params()
        input_params.index = input_params.index + 10
        dfs = self._make_dfs(input_params)

        _, score_dict = occupant_presence.compute_classification_score(dfs)

        self.assertEqual(score_dict["Child left behind"], 0.0)
        self.assertEqual(score_dict["Child enters unlocked vehicle"], 0.0)


if __name__ == "__main__":
    unittest.main()
