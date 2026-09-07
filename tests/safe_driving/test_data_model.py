# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import unittest
from euroncap_rating_2026.safe_driving import data_model


class TestStageSubelements(unittest.TestCase):
    """Test cases for STAGE_SUBELEMENTS constant."""

    def test_stage_subelements_is_list(self):
        """Test that STAGE_SUBELEMENTS is a list."""
        self.assertIsInstance(data_model.STAGE_SUBELEMENTS, list)

    def test_stage_subelements_not_empty(self):
        """Test that STAGE_SUBELEMENTS is not empty."""
        self.assertGreater(len(data_model.STAGE_SUBELEMENTS), 0)

    def test_stage_subelements_have_required_keys(self):
        """Test that each element in STAGE_SUBELEMENTS has required keys."""
        required_keys = ["Stage ", "Stage element", "Stage subelement"]
        for item in data_model.STAGE_SUBELEMENTS:
            for key in required_keys:
                self.assertIn(key, item)

    def test_stage_subelements_all_safe_driving(self):
        """Test that all STAGE_SUBELEMENTS belong to Safe Driving stage."""
        for item in data_model.STAGE_SUBELEMENTS:
            self.assertEqual(item["Stage "], "Safe Driving")

    def test_stage_subelements_expected_elements(self):
        """Test that expected stage elements are present."""
        expected_elements = [
            "Occupant Monitoring",
            "Driver Engagement",
            "Vehicle Assistance",
        ]
        elements = set(item["Stage element"] for item in data_model.STAGE_SUBELEMENTS)
        for expected in expected_elements:
            self.assertIn(expected, elements)

    def test_stage_subelements_expected_subelements(self):
        """Test that expected stage subelements are present."""
        expected_subelements = [
            "Seatbelt usage",
            "Occupant classification",
            "Driver monitoring",
            "General vehicle controls",
            "Speed Assistance",
            "ACC Performance",
            "Steering Assistance",
        ]
        subelements = [
            item["Stage subelement"] for item in data_model.STAGE_SUBELEMENTS
        ]
        for expected in expected_subelements:
            self.assertIn(expected, subelements)


class TestAccMatrixIndices(unittest.TestCase):
    """Test cases for ACC_MATRIX_INDICES constant."""

    def test_acc_matrix_indices_is_dict(self):
        """Test that ACC_MATRIX_INDICES is a dictionary."""
        self.assertIsInstance(data_model.ACC_MATRIX_INDICES, dict)

    def test_acc_matrix_indices_not_empty(self):
        """Test that ACC_MATRIX_INDICES is not empty."""
        self.assertGreater(len(data_model.ACC_MATRIX_INDICES), 0)

    def test_acc_matrix_indices_have_required_keys(self):
        """Test that each matrix index entry has required keys."""
        required_keys = ["start_row", "n_rows", "start_col", "n_cols"]
        for matrix_name, indices in data_model.ACC_MATRIX_INDICES.items():
            for key in required_keys:
                self.assertIn(key, indices, f"Matrix {matrix_name} missing key {key}")

    def test_acc_matrix_indices_values_are_positive(self):
        """Test that all matrix index values are positive integers."""
        for matrix_name, indices in data_model.ACC_MATRIX_INDICES.items():
            self.assertGreater(
                indices["start_row"], 0, f"{matrix_name} start_row should be positive"
            )
            self.assertGreater(
                indices["n_rows"], 0, f"{matrix_name} n_rows should be positive"
            )
            self.assertGreater(
                indices["start_col"], 0, f"{matrix_name} start_col should be positive"
            )
            self.assertGreater(
                indices["n_cols"], 0, f"{matrix_name} n_cols should be positive"
            )

    def test_acc_matrix_indices_expected_matrices(self):
        """Test that expected matrices are present."""
        expected_matrices = [
            "CCRs straight",
            "CCRs curved",
            "CCRm",
            "CCRb",
            "CCR cut-in",
            "CCR cut-out",
            "CMRs straight",
            "CMRs curved",
            "CMRm",
            "CMRb",
            "CMR cut-in",
            "CMR cut-out",
            "CPLA",
            "CBLA",
        ]
        for expected in expected_matrices:
            self.assertIn(expected, data_model.ACC_MATRIX_INDICES)


class TestStageSubelementKey(unittest.TestCase):
    """Test cases for StageSubelementKey enum."""

    def test_stage_subelement_key_values(self):
        """Test that StageSubelementKey has expected values."""
        self.assertEqual(data_model.StageSubelementKey.OM, "OM")
        self.assertEqual(data_model.StageSubelementKey.DE, "DE")
        self.assertEqual(data_model.StageSubelementKey.VA, "VA")

    def test_stage_subelement_key_is_string_enum(self):
        """Test that StageSubelementKey values are strings."""
        for member in data_model.StageSubelementKey:
            self.assertIsInstance(member.value, str)


class TestOCTypeOfSystem(unittest.TestCase):
    """Test cases for OCTypeOfSystem enum."""

    def test_oc_type_of_system_values(self):
        """Test that OCTypeOfSystem has expected values."""
        self.assertEqual(data_model.OCTypeOfSystem.AUTOMATIC, "Automatic")
        self.assertEqual(data_model.OCTypeOfSystem.SYSTEM_ADVISED, "System advised")
        self.assertEqual(data_model.OCTypeOfSystem.MANUAL, "Manual")

    def test_oc_type_of_system_from_string(self):
        """Test that OCTypeOfSystem can be created from string."""
        self.assertEqual(
            data_model.OCTypeOfSystem("Automatic"), data_model.OCTypeOfSystem.AUTOMATIC
        )
        self.assertEqual(
            data_model.OCTypeOfSystem("System advised"),
            data_model.OCTypeOfSystem.SYSTEM_ADVISED,
        )
        self.assertEqual(
            data_model.OCTypeOfSystem("Manual"), data_model.OCTypeOfSystem.MANUAL
        )

    def test_oc_type_of_system_invalid_value(self):
        """Test that invalid string raises ValueError."""
        with self.assertRaises(ValueError):
            data_model.OCTypeOfSystem("Invalid")


class TestOCTypeOfSwitch(unittest.TestCase):
    """Test cases for OCTypeOfSwitch enum."""

    def test_oc_type_of_switch_values(self):
        """Test that OCTypeOfSwitch has expected values."""
        self.assertEqual(data_model.OCTypeOfSwitch.AUTOMATIC, "Automatic")
        self.assertEqual(data_model.OCTypeOfSwitch.HARDWARE, "Hardware")
        self.assertEqual(data_model.OCTypeOfSwitch.SOFTWARE, "Software")

    def test_oc_type_of_switch_from_string(self):
        """Test that OCTypeOfSwitch can be created from string."""
        self.assertEqual(
            data_model.OCTypeOfSwitch("Automatic"), data_model.OCTypeOfSwitch.AUTOMATIC
        )
        self.assertEqual(
            data_model.OCTypeOfSwitch("Hardware"), data_model.OCTypeOfSwitch.HARDWARE
        )
        self.assertEqual(
            data_model.OCTypeOfSwitch("Software"), data_model.OCTypeOfSwitch.SOFTWARE
        )

    def test_oc_type_of_switch_invalid_value(self):
        """Test that invalid string raises ValueError."""
        with self.assertRaises(ValueError):
            data_model.OCTypeOfSwitch("Invalid")


class TestOPTypeOfSystem(unittest.TestCase):
    """Test cases for OPTypeOfSystem enum."""

    def test_op_type_of_system_values(self):
        """Test that OPTypeOfSystem has expected values."""
        self.assertEqual(data_model.OPTypeOfSystem.WARNING, "Warning")
        self.assertEqual(
            data_model.OPTypeOfSystem.WARNING_AND_INTERVENTION,
            "Warning and intervention",
        )
        self.assertEqual(data_model.OPTypeOfSystem.NO, "No")

    def test_op_type_of_system_from_string(self):
        """Test that OPTypeOfSystem can be created from string."""
        self.assertEqual(
            data_model.OPTypeOfSystem("Warning"), data_model.OPTypeOfSystem.WARNING
        )
        self.assertEqual(
            data_model.OPTypeOfSystem("Warning and intervention"),
            data_model.OPTypeOfSystem.WARNING_AND_INTERVENTION,
        )
        self.assertEqual(data_model.OPTypeOfSystem("No"), data_model.OPTypeOfSystem.NO)

    def test_op_type_of_system_invalid_value(self):
        """Test that invalid string raises ValueError."""
        with self.assertRaises(ValueError):
            data_model.OPTypeOfSystem("Invalid")


class TestOPSeatCoverage(unittest.TestCase):
    """Test cases for OPSeatCoverage enum."""

    def test_op_seat_coverage_values(self):
        """Test that OPSeatCoverage has expected values."""
        self.assertEqual(data_model.OPSeatCoverage.REAR_SEATS, "Rear seats")
        self.assertEqual(data_model.OPSeatCoverage.ALL_PASSENGER_SEATS, "All seats")

    def test_op_seat_coverage_from_string(self):
        """Test that OPSeatCoverage can be created from string."""
        self.assertEqual(
            data_model.OPSeatCoverage("Rear seats"),
            data_model.OPSeatCoverage.REAR_SEATS,
        )
        self.assertEqual(
            data_model.OPSeatCoverage("All seats"),
            data_model.OPSeatCoverage.ALL_PASSENGER_SEATS,
        )

    def test_op_seat_coverage_invalid_value(self):
        """Test that invalid string raises ValueError."""
        with self.assertRaises(ValueError):
            data_model.OPSeatCoverage("Invalid")


class TestNotApplicable(unittest.TestCase):
    """Test cases for the "N/A" input parameter value."""

    def test_not_applicable_constant(self):
        self.assertEqual(data_model.NOT_APPLICABLE, "N/A")

    def test_oc_enums_accept_na(self):
        """The Input parameters dropdowns offer "N/A" for the passenger
        airbag Type of system/switch -- the enums must accept it."""
        self.assertEqual(
            data_model.OCTypeOfSystem("N/A"),
            data_model.OCTypeOfSystem.NOT_APPLICABLE,
        )
        self.assertEqual(
            data_model.OCTypeOfSwitch("N/A"),
            data_model.OCTypeOfSwitch.NOT_APPLICABLE,
        )

    def test_is_not_applicable_true_for_na_variants(self):
        import numpy as np

        self.assertTrue(data_model.is_not_applicable("N/A"))
        self.assertTrue(data_model.is_not_applicable("n/a"))
        self.assertTrue(data_model.is_not_applicable(" N/A "))
        self.assertTrue(
            data_model.is_not_applicable(data_model.OCTypeOfSystem.NOT_APPLICABLE)
        )

    def test_is_not_applicable_false_for_blank_and_real_values(self):
        import numpy as np

        self.assertFalse(data_model.is_not_applicable(None))
        self.assertFalse(data_model.is_not_applicable(np.nan))
        self.assertFalse(data_model.is_not_applicable(""))
        self.assertFalse(data_model.is_not_applicable("Automatic"))
        self.assertFalse(data_model.is_not_applicable(0))


if __name__ == "__main__":
    unittest.main()
