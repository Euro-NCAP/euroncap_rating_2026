# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import unittest

import pandas as pd

from euroncap_rating_2026.post_crash import advanced_ecall


TPS_CATEGORIES = [
    "Country coverage",
    "Multiple languages - EN, DE, FR, ES",
    "Multiple languages - 4 additional languages",
    "Hazard detection after crash",
    "Telephone pairing",
    "Vehicle information",
    "Vehicle attitude",
    "Any additional information",
    "AACN and OEM severity index",
]

TPS_DEPENDENT_CATEGORIES = [
    "Multiple languages - 4 additional languages",
    "Hazard detection after crash",
    "Telephone pairing",
    "Vehicle information",
    "Vehicle attitude",
    "Any additional information",
    "AACN and OEM severity index",
]

TPS_MAX = {category: 3.0 for category in TPS_CATEGORIES}

E112_CATEGORIES = [
    "Potential number of occupants",
    "Direction of impacts - Front impact",
    "Direction of impacts - Side impact",
    "Direction of impacts - Rear impact",
    "Direction of impacts - Rollovers as 1st impact",
    "Delta V - Front impact",
    "Delta V - Side impact",
    "Delta V - Rear impact",
]

E112_MAX = {
    "Potential number of occupants": 3.0,
    "Direction of impacts - Front impact": 2.0,
    "Direction of impacts - Side impact": 2.0,
    "Direction of impacts - Rear impact": 2.0,
    "Direction of impacts - Rollovers as 1st impact": 1.5,
    "Delta V - Front impact": 1.5,
    "Delta V - Side impact": 1.5,
    "Delta V - Rear impact": 1.5,
}


def _scenario_rows(
    scenario: str, values: list[str], category: str | None = None
) -> list[dict]:
    rows = []
    for idx, value in enumerate(values):
        rows.append(
            {
                "Category": category if idx == 0 else None,
                "Scenario": scenario if idx == 0 else None,
                "Requirement": f"Req {scenario} {idx + 1}",
                "Value": value,
            }
        )
    return rows


def _build_section_df(
    general_values: list[str] | None,
    scenario_values: dict[str, list[str]],
    category_header: str | None = None,
) -> pd.DataFrame:
    rows = []

    if general_values is not None:
        rows.extend(
            _scenario_rows("General requirements", general_values, category_header)
        )

    for idx, (scenario, values) in enumerate(scenario_values.items()):
        category = category_header if idx == 0 and general_values is None else None
        rows.extend(_scenario_rows(scenario, values, category))

    return pd.DataFrame(rows, columns=["Category", "Scenario", "Requirement", "Value"])


class TestAdvancedECall112(unittest.TestCase):
    def test_general_requirements_failed_returns_zeros(self):
        df = _build_section_df(
            general_values=["PASS", "FAIL"],
            scenario_values={scenario: ["PASS"] for scenario in E112_CATEGORIES},
        )

        result = advanced_ecall.compute_112_score(df)

        self.assertEqual(result, {scenario: 0.0 for scenario in E112_CATEGORIES})

    def test_general_requirements_missing_returns_unassessed(self):
        """A missing General requirements section means nothing has been
        assessed yet -- unassessed (NaN), not a false 0."""
        df = _build_section_df(
            general_values=None,
            scenario_values={scenario: ["PASS"] for scenario in E112_CATEGORIES},
        )

        result = advanced_ecall.compute_112_score(df)

        for scenario in E112_CATEGORIES:
            self.assertTrue(pd.isna(result[scenario]), scenario)

    def test_awards_points_only_for_fully_passing_categories(self):
        df = _build_section_df(
            general_values=["PASS"],
            scenario_values={
                "Potential number of occupants": ["PASS"],
                "Direction of impacts - Front impact": ["PASS", "FAIL"],
                "Direction of impacts - Side impact": ["pass"],
                "Direction of impacts - Rear impact": ["FAIL"],
                "Direction of impacts - Rollovers as 1st impact": ["PASS"],
                "Delta V - Front impact": ["PASS"],
                "Delta V - Side impact": ["PASS", "PASS"],
                "Delta V - Rear impact": ["fail"],
            },
        )

        result = advanced_ecall.compute_112_score(df)

        expected = {
            "Potential number of occupants": 3.0,
            "Direction of impacts - Front impact": 0.0,
            "Direction of impacts - Side impact": 2.0,
            "Direction of impacts - Rear impact": 0.0,
            "Direction of impacts - Rollovers as 1st impact": 1.5,
            "Delta V - Front impact": 1.5,
            "Delta V - Side impact": 1.5,
            "Delta V - Rear impact": 0.0,
        }
        self.assertEqual(result, expected)


class TestAdvancedECallTPS(unittest.TestCase):
    def test_general_requirements_failed_returns_zeros(self):
        df = _build_section_df(
            general_values=["FAIL"],
            scenario_values={scenario: ["PASS"] for scenario in TPS_CATEGORIES},
        )

        result = advanced_ecall.compute_tps_score(df)

        self.assertEqual(result, {scenario: 0 for scenario in TPS_CATEGORIES})

    def test_country_coverage_gates_everything_else(self):
        scenario_values = {scenario: ["PASS"] for scenario in TPS_CATEGORIES}
        scenario_values["Country coverage"] = ["FAIL"]

        df = _build_section_df(general_values=["PASS"], scenario_values=scenario_values)
        result = advanced_ecall.compute_tps_score(df)

        self.assertEqual(result["Country coverage"], 0)
        self.assertEqual(result["Multiple languages - EN, DE, FR, ES"], 0)
        for category in TPS_DEPENDENT_CATEGORIES:
            self.assertEqual(result[category], 0)

    def test_language_support_gates_remaining_categories(self):
        scenario_values = {scenario: ["PASS"] for scenario in TPS_CATEGORIES}
        scenario_values["Multiple languages - EN, DE, FR, ES"] = ["FAIL"]

        df = _build_section_df(general_values=["PASS"], scenario_values=scenario_values)
        result = advanced_ecall.compute_tps_score(df)

        self.assertEqual(result["Country coverage"], TPS_MAX["Country coverage"])
        self.assertEqual(result["Multiple languages - EN, DE, FR, ES"], 0)
        for category in TPS_DEPENDENT_CATEGORIES:
            self.assertEqual(result[category], 0)

    def test_when_country_and_language_pass_dependent_categories_score_independently(
        self,
    ):
        scenario_values = {
            "Country coverage": ["PASS"],
            "Multiple languages - EN, DE, FR, ES": ["PASS"],
            "Multiple languages - 4 additional languages": ["PASS"],
            "Hazard detection after crash": ["FAIL"],
            "Telephone pairing": ["PASS", "PASS"],
            "Vehicle information": ["FAIL", "PASS"],
            "Vehicle attitude": ["PASS"],
            "Any additional information": ["pass"],
            "AACN and OEM severity index": ["FAIL"],
        }

        df = _build_section_df(general_values=["PASS"], scenario_values=scenario_values)
        result = advanced_ecall.compute_tps_score(df)

        self.assertEqual(result["Country coverage"], 3.0)
        self.assertEqual(result["Multiple languages - EN, DE, FR, ES"], 3.0)
        self.assertEqual(result["Multiple languages - 4 additional languages"], 3.0)
        self.assertEqual(result["Hazard detection after crash"], 0)
        self.assertEqual(result["Telephone pairing"], 3.0)
        self.assertEqual(result["Vehicle information"], 0)
        self.assertEqual(result["Vehicle attitude"], 3.0)
        self.assertEqual(result["Any additional information"], 3.0)
        self.assertEqual(result["AACN and OEM severity index"], 0)

    def test_missing_country_coverage_section_keeps_everything_unassessed(self):
        """A missing Country coverage section means it hasn't been assessed
        yet -- unassessed (NaN), not a false 0, and the same for everything
        gated on it (we can't yet know if the gate will pass or fail)."""
        scenario_values = {
            category: ["PASS"]
            for category in TPS_CATEGORIES
            if category != "Country coverage"
        }

        df = _build_section_df(general_values=["PASS"], scenario_values=scenario_values)
        result = advanced_ecall.compute_tps_score(df)

        self.assertTrue(pd.isna(result["Country coverage"]))
        self.assertTrue(pd.isna(result["Multiple languages - EN, DE, FR, ES"]))
        for category in TPS_DEPENDENT_CATEGORIES:
            self.assertTrue(pd.isna(result[category]), category)

    def test_explicit_fail_country_coverage_zeroes_everything_gated(self):
        """An explicit FAIL (not missing/blank) on Country coverage is a
        deterministic gate failure -- unlike the blank case above, everything
        gated on it is a real 0, not unassessed."""
        scenario_values = {
            category: ["PASS"]
            for category in TPS_CATEGORIES
            if category != "Country coverage"
        }
        scenario_values["Country coverage"] = ["FAIL"]

        df = _build_section_df(general_values=["PASS"], scenario_values=scenario_values)
        result = advanced_ecall.compute_tps_score(df)

        self.assertEqual(result["Country coverage"], 0.0)
        self.assertEqual(result["Multiple languages - EN, DE, FR, ES"], 0.0)
        for category in TPS_DEPENDENT_CATEGORIES:
            self.assertEqual(result[category], 0.0)


class TestAdvancedECallComputeScore(unittest.TestCase):
    def test_empty_input_sheet_returns_empty_dict(self):
        dfs = {"PCI - Advanced eCall Verif": pd.DataFrame()}

        result = advanced_ecall.compute_score(dfs)

        self.assertEqual(result, {})

    def test_missing_input_sheet_returns_empty_dict(self):
        result = advanced_ecall.compute_score({})
        self.assertEqual(result, {})

    def test_compute_score_combines_112_and_tps_sections(self):
        df_112 = _build_section_df(
            general_values=["PASS"],
            scenario_values={
                "Potential number of occupants": ["PASS"],
                "Direction of impacts - Front impact": ["PASS"],
                "Direction of impacts - Side impact": ["FAIL"],
                "Direction of impacts - Rear impact": ["PASS"],
                "Direction of impacts - Rollovers as 1st impact": ["PASS"],
                "Delta V - Front impact": ["FAIL"],
                "Delta V - Side impact": ["PASS"],
                "Delta V - Rear impact": ["PASS"],
            },
            category_header="Advanced eCall - 112",
        )

        df_tps = _build_section_df(
            general_values=["PASS"],
            scenario_values={
                "Country coverage": ["PASS"],
                "Multiple languages - EN, DE, FR, ES": ["PASS"],
                "Multiple languages - 4 additional languages": ["PASS"],
                "Hazard detection after crash": ["PASS"],
                "Telephone pairing": ["FAIL"],
                "Vehicle information": ["PASS"],
                "Vehicle attitude": ["PASS"],
                "Any additional information": ["FAIL"],
                "AACN and OEM severity index": ["PASS"],
            },
            category_header="Advanced eCall - TPS",
        )

        ecall_verif_df = pd.concat([df_112, df_tps], ignore_index=True)
        dfs = {"PCI - Advanced eCall Verif": ecall_verif_df}

        result = advanced_ecall.compute_score(dfs)

        self.assertEqual(len(result), len(E112_CATEGORIES) + len(TPS_CATEGORIES))

        self.assertEqual(
            result["Potential number of occupants"],
            E112_MAX["Potential number of occupants"],
        )
        self.assertEqual(result["Direction of impacts - Side impact"], 0.0)
        self.assertEqual(result["Delta V - Front impact"], 0.0)

        self.assertEqual(result["Country coverage"], 3.0)
        self.assertEqual(result["Multiple languages - EN, DE, FR, ES"], 3.0)
        self.assertEqual(result["Telephone pairing"], 0)
        self.assertEqual(result["Any additional information"], 0)


if __name__ == "__main__":
    unittest.main()
