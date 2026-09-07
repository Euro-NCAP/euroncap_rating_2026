# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import unittest
import random
import re
from euroncap_rating_2026.safe_driving import driver_monitoring
from euroncap_rating_2026 import common
import numpy as np
import pandas as pd
import logging
from unittest.mock import patch


class TestDrivingMonitoring(unittest.TestCase):
    def setUp(self):
        # Create a DataFrame with extra columns, but ensure the section for 'Long Distraction' only has DM_COLUMNS
        self.full_df = pd.DataFrame(
            {
                "Scenario": [
                    "Long Distraction",
                    "Long Distraction",
                    "Short Distraction",
                    "Phone Use",
                    "Non-transient",
                ],
                "Long distraction": [
                    "Non-Driving task",
                    "Driving task",
                    "Basic phone use",
                    "Advanced phone use",
                    "Microsleep",
                ],
                "Movement Type": ["Head turn", "Blink", "Touch", "Swipe", "Sleep"],
                "Gaze Location": ["Road", "Mirror", "Phone", "Road", "Dashboard"],
                "Warning": ["GREEN", "RED", "GREEN", "RED", "GREEN"],
                "Forward Support": ["GREEN", "GREEN", "RED", "GREEN", "RED"],
                "Lane Support": ["RED", "GREEN", "GREEN", "RED", "GREEN"],
                "Extra1": [1, 2, 3, 4, 5],
                "Extra2": ["A", "B", "C", "D", "E"],
            }
        )
        # For TableProcessor, pass only DM_COLUMNS columns for 'Long Distraction' section
        self.dm_prediction_df = self.full_df[
            [
                "Long distraction",
                "Movement Type",
                "Gaze Location",
                "Warning",
                "Forward Support",
                "Lane Support",
            ]
        ].copy()
        # Noise Variables sheet for relevant tests
        self.noise_variables_df = pd.DataFrame(
            {
                "Elements": [
                    "Sunglasses (<15% transmittance)",
                    "Face-mask",
                    "Cap",
                    "Hats",
                ],
                "Performance": ["Functional", "Degraded", "Functional", "Functional"],
            }
        )
        self.table_processor = driver_monitoring.TableProcessor(
            self.dm_prediction_df, "Long distraction"
        )

    def test_pick_random_yes_cells(self):
        yes_cells_dict = {i: f"cell_{i}" for i in range(5)}
        picked = driver_monitoring.pick_random_yes_cells(yes_cells_dict, 3)
        self.assertEqual(len(picked), 3)
        self.assertTrue(all(k in yes_cells_dict for k in picked))

    def test_table_processor(self):
        tp = self.table_processor
        self.assertIsInstance(tp.cell_dict, dict)
        self.assertTrue(any(isinstance(v, dict) for v in tp.cell_dict.values()))

    def test_select_warning_test_points(self):
        valid_tasks = ["Non-Driving task", "Driving task"]
        result = driver_monitoring.select_warning_test_points(
            self.table_processor, valid_tasks
        )
        self.assertIsInstance(result, dict)

    def test_get_verification_rows(self):
        selected_cells = {
            k: v for k, v in list(self.table_processor.cell_dict.items())[:2]
        }
        rows = driver_monitoring.get_verification_rows(
            self.table_processor, selected_cells
        )
        self.assertTrue(isinstance(rows, list))
        self.assertTrue(all(isinstance(r, dict) for r in rows))

    def test_get_verification_rows_test_scenario_populated_when_provided(self):
        selected_cells = {
            (0, 4): {
                "Task": "Non-driving task",
                "Movement type": "Owl",
                "Gaze location": "Road",
                "Vehicle response": "Forward support",
                "Color": "GREEN",
            },
        }
        rows = driver_monitoring.get_verification_rows(
            self.table_processor,
            selected_cells,
            scenarios={"Forward support": "SCEN_FS"},
        )
        self.assertEqual(rows[0]["Test scenario"], "SCEN_FS")

    def test_get_verification_rows_test_scenario_none_when_not_in_scenarios_dict(self):
        selected_cells = {
            (0, 3): {
                "Task": "Non-driving task",
                "Movement type": "Owl",
                "Gaze location": "Road",
                "Vehicle response": "Warning",
                "Color": "GREEN",
            },
        }
        rows = driver_monitoring.get_verification_rows(
            self.table_processor,
            selected_cells,
            scenarios={"Forward support": "SCEN_FS"},
        )
        self.assertIsNone(rows[0]["Test scenario"])

    def test_get_verification_rows_test_scenario_none_when_scenarios_omitted(self):
        selected_cells = {
            (0, 4): {
                "Task": "Non-driving task",
                "Movement type": "Owl",
                "Gaze location": "Road",
                "Vehicle response": "Forward support",
                "Color": "GREEN",
            },
        }
        rows = driver_monitoring.get_verification_rows(
            self.table_processor, selected_cells
        )
        self.assertIsNone(rows[0]["Test scenario"])

    def test_sample_and_assign_noise_vars(self):
        warning_rows = [{"Requirement": "Warning", "Gaze Location": "Road"}]
        driving_rows = [{"Requirement": "Forward Support", "Gaze Location": "Mirror"}]
        noise_vars = ["Sunglasses", "Cap"]
        combined_rows = warning_rows + driving_rows
        driver_monitoring.sample_and_assign_noise_vars(
            combined_rows, noise_vars, sample_size=2
        )
        self.assertTrue(any("Noise variable" in r for r in combined_rows))

    def test_preprocess_selected_points_are_half_per_movement_type(self):
        class FakeTableProcessor:
            def __init__(self, name):
                self.name = name
                self.driver_state = "Transient"
                rows = []

                # Each (Task, Movement type) group has 4 rows, so expected selection is 2 per group.
                groups = [
                    ("Non-driving task", "Owl"),
                    ("Non-driving task", "Lizard"),
                    ("Driving task", "Owl"),
                    ("Basic phone use", "Owl and lizard"),
                    ("Advanced phone use", "Lizard"),
                    ("Multi-target", "Lizard"),
                ]
                for task, movement_type in groups:
                    for _ in range(4):
                        rows.append(
                            {
                                "Long distraction": task,
                                "Movement type": movement_type,
                                "Gaze location": "Road",
                                "Warning": "GREEN",
                                "Forward support": (
                                    "GREEN" if task == "Driving task" else "RED"
                                ),
                                "Lane support": (
                                    "GREEN" if task == "Driving task" else "RED"
                                ),
                            }
                        )

                self.processed_df = pd.DataFrame(rows)
                self.cell_dict = {}
                for row_idx, row in self.processed_df.iterrows():
                    for col_idx, col_name in enumerate(self.processed_df.columns):
                        cell_value = row[col_name]
                        if str(cell_value).strip().upper() in ["GREEN", "RED"]:
                            self.cell_dict[(row_idx, col_idx)] = {
                                "Task": row["Long distraction"],
                                "Movement type": row["Movement type"],
                                "Gaze location": row["Gaze location"],
                                "Vehicle response": col_name,
                                "Color": str(cell_value).strip().upper(),
                            }

            def deepcopy(self):
                return self

        noise_variables_df = pd.DataFrame(
            {
                "Element": ["Dark sunglasses", "Face-mask", "Cap", "Hat"],
                "Performance": ["Functional", "Functional", "Functional", "Functional"],
            }
        )

        def fake_table_processor_factory(_dm_prediction_df, df_name, **_kwargs):
            return FakeTableProcessor(df_name)

        with (
            patch.object(
                driver_monitoring,
                "TableProcessor",
                side_effect=fake_table_processor_factory,
            ),
            patch.object(
                driver_monitoring.common,
                "extract_table",
                return_value=noise_variables_df,
            ),
            patch.object(
                driver_monitoring, "get_mandatory_check_dict", return_value={"ok": True}
            ),
            patch.object(
                driver_monitoring, "get_non_transient_points", return_value={}
            ),
        ):
            _, selected_test_points, _ = driver_monitoring.preprocess(
                dm_prediction_df=pd.DataFrame(), param_df=pd.DataFrame()
            )

        # Warning points: warning_valid_tasks excludes "Driving task", leaving 5 valid
        # movement groups * ceil(4/2)=2 per group, for 3 processors = 30.
        expected_warning_points = 30
        # Intervention points: only ("Driving task", "Owl") has FS=GREEN/LS=GREEN in this
        # fixture (all other groups are RED for FS/LS); the new algorithm selects exactly
        # one forward-or-lane test per (processor, Task, Movement type) group, so this
        # yields 1 selection per processor * 3 processors = 3.
        expected_intervention_points = 3
        expected_total_points = expected_warning_points + expected_intervention_points

        self.assertEqual(len(selected_test_points), expected_total_points)

    def test_preprocess_uses_explicit_input_selection_and_skips_random_helpers(self):
        class FakeTableProcessor:
            def __init__(self, name, driver_state, cell_dict):
                self.name = name
                self.driver_state = driver_state
                self.processed_df = pd.DataFrame()
                self.cell_dict = cell_dict

            def deepcopy(self):
                return self

        processors = {
            "Long distraction": FakeTableProcessor(
                "Long distraction",
                "Transient",
                {
                    (0, 3): {
                        "Task": "Non-driving task",
                        "Movement type": "Owl",
                        "Gaze location": "Road",
                        "Vehicle response": "Warning",
                        "Color": "GREEN",
                    },
                    (1, 4): {
                        "Task": "Driving task",
                        "Movement type": "Owl",
                        "Gaze location": "Road",
                        "Vehicle response": "Forward support",
                        "Color": "GREEN",
                    },
                },
            ),
            "Short distraction": FakeTableProcessor(
                "Short distraction",
                "Transient",
                {
                    (2, 3): {
                        "Task": "Driving task",
                        "Movement type": "Lizard",
                        "Gaze location": "Mirror",
                        "Vehicle response": "Warning",
                        "Color": "GREEN",
                    }
                },
            ),
            "Phone use": FakeTableProcessor(
                "Phone use",
                "Transient",
                {
                    (3, 3): {
                        "Task": "Basic phone use",
                        "Movement type": "Owl and lizard",
                        "Gaze location": "Phone",
                        "Vehicle response": "Warning",
                        "Color": "GREEN",
                    }
                },
            ),
            "Non-transient": FakeTableProcessor(
                "Non-transient",
                "Non-transient",
                {
                    (4, 5): {
                        "Task": "Unresponsive",
                        "Movement type": "",
                        "Gaze location": "Road",
                        "Vehicle response": "Emergency function",
                        "Color": "GREEN",
                    }
                },
            ),
        }

        noise_variables_df = pd.DataFrame(
            {
                "Element": ["Dark sunglasses", "Face-mask", "Cap", "Hat"],
                "Performance": ["Functional", "Functional", "Functional", "Functional"],
            }
        )

        input_selected_points = {
            "Long distraction": [
                {
                    "Task": "Non-driving task",
                    "Movement type": "Owl",
                    "Vehicle response": "Warning",
                },
                {
                    "Task": "Driving task",
                    "Movement type": "Owl",
                    "Vehicle response": "Forward support",
                },
            ],
            "Non-transient": [
                {
                    "Task": "Unresponsive",
                    "Vehicle response": "Emergency function",
                }
            ],
        }

        with (
            patch.object(
                driver_monitoring,
                "TableProcessor",
                side_effect=lambda _df, name, **_kwargs: processors[name],
            ),
            patch.object(
                driver_monitoring.common,
                "extract_table",
                return_value=noise_variables_df,
            ),
            patch.object(
                driver_monitoring, "get_mandatory_check_dict", return_value={"ok": True}
            ),
            patch.object(
                driver_monitoring, "select_warning_test_points"
            ) as mock_warning,
            patch.object(
                driver_monitoring, "select_intervention_test_points"
            ) as mock_intervention,
            patch.object(
                driver_monitoring, "get_non_transient_points"
            ) as mock_non_transient,
            patch.object(
                driver_monitoring,
                "build_forward_support_scenario",
                return_value="SCEN_TEST",
            ),
            patch.object(driver_monitoring, "sample_and_assign_noise_vars"),
            patch.object(driver_monitoring, "assign_noise_vars_non_transient"),
        ):
            verification_df, selected_test_points, _ = driver_monitoring.preprocess(
                dm_prediction_df=pd.DataFrame(),
                param_df=pd.DataFrame(),
                input_selected_points=input_selected_points,
            )

        mock_warning.assert_not_called()
        mock_intervention.assert_not_called()
        mock_non_transient.assert_not_called()
        self.assertEqual(len(selected_test_points), 3)
        self.assertFalse((verification_df == "Phone use").any().any())
        self.assertTrue((verification_df == "Forward support").any().any())
        self.assertTrue((verification_df == "Emergency function").any().any())

        # Manually-selected Forward support point must still get a "Test scenario"
        # populated, even though it came through the "warning_points" input list
        # rather than the automatic intervention path.
        forward_support_row = verification_df[
            (verification_df == "Forward support").any(axis=1)
        ]
        self.assertTrue(
            (forward_support_row == "SCEN_TEST").any().any(),
            "Manually-selected Forward support row should carry the drawn scenario",
        )

    def test_clean_df_warning_score_phone_use_intervention_independent(self):
        """Phone use warning=0 must NOT zero out the intervention score (no cascade)."""
        df = pd.DataFrame(
            {
                "Category": ["Transient - Phone use", "Transient - Phone use"],
                "Scenario": [
                    "Basic phone use - Owl and lizard - Warning",
                    "Basic phone use - Owl and lizard - Intervention",
                ],
                "Score": [0.0, 1.25],
            }
        )
        result = driver_monitoring.clean_df_warning_score(df)
        intervention_score = result.loc[
            result["Scenario"].str.endswith("Intervention"), "Score"
        ].values[0]
        self.assertEqual(intervention_score, 1.25)

    def test_clean_df_warning_score_long_distraction_cascade_applies(self):
        """Long distraction warning=0 MUST zero out the intervention score (cascade applies)."""
        df = pd.DataFrame(
            {
                "Category": [
                    "Transient - Long distraction",
                    "Transient - Long distraction",
                ],
                "Scenario": [
                    "Non-driving task - Owl - Warning",
                    "Non-driving task - Owl - Intervention",
                ],
                "Score": [0.0, 0.5],
            }
        )
        result = driver_monitoring.clean_df_warning_score(df)
        intervention_score = result.loc[
            result["Scenario"].str.endswith("Intervention"), "Score"
        ].values[0]
        self.assertEqual(intervention_score, 0.0)

    # ------------------------------------------------------------------
    # Bug 1: Driving task Intervention — prediction-only scoring
    # ------------------------------------------------------------------

    def _make_driving_task_tp(self):
        """Return a fake TableProcessor whose processed_df has one Driving task / Owl row at index 0."""

        class FakeTP:
            pass

        fake = FakeTP()
        fake.processed_df = pd.DataFrame(
            {
                "Long distraction": ["Driving task"],
                "Movement type": ["Owl"],
                "Gaze location": ["Road"],
                "Warning": [None],
                "Forward support": ["GREEN"],
                "Lane support": ["GREEN"],
            }
        )
        return fake

    def _empty_tp(self):
        class FakeTP:
            pass

        fake = FakeTP()
        fake.processed_df = pd.DataFrame(
            columns=[
                "Long distraction",
                "Movement type",
                "Gaze location",
                "Warning",
                "Forward support",
                "Lane support",
            ]
        )
        return fake

    def test_compute_transient_score_driving_task_intervention_ignores_failed_verification(
        self,
    ):
        """Bug 1: Driving task Intervention score must be awarded based on prediction alone,
        even when the verification tests FAIL."""
        ld_tp = self._make_driving_task_tp()

        def tp_factory(dm_df, name):
            return ld_tp if name == "Long distraction" else self._empty_tp()

        transient_df = pd.DataFrame(
            {
                "Category": [
                    "Transient - Long distraction",
                    "Transient - Long distraction",
                ],
                "Scenario": [
                    "Driving task - Owl - Intervention",
                    "Driving task - Owl - Intervention",
                ],
                "Glance target type": ["Driving task", "Driving task"],
                "Movement type": ["Owl", "Owl"],
                "Requirement": ["Forward support", "Lane support"],
                "Value": ["FAIL", "FAIL"],
            }
        )
        transient_score_df = pd.DataFrame(
            {
                "Category": ["Transient - Long distraction"],
                "Scenario": ["Driving task - Owl - Intervention"],
                "Score": [0.0],
            }
        )
        dm_colors = {
            0: {
                "Forward support": driver_monitoring.common.PredictionColor.GREEN,
                "Lane support": driver_monitoring.common.PredictionColor.GREEN,
            }
        }

        with patch.object(driver_monitoring, "TableProcessor", side_effect=tp_factory):
            result = driver_monitoring.compute_transient_score(
                transient_df, pd.DataFrame(), dm_colors, transient_score_df
            )

        score = result.loc[
            result["Scenario"] == "Driving task - Owl - Intervention", "Score"
        ].values[0]
        self.assertGreater(
            score,
            0.0,
            "Driving task Intervention should score > 0 when prediction is GREEN, "
            "regardless of verification result",
        )

    def test_compute_transient_score_driving_task_intervention_zero_when_prediction_red(
        self,
    ):
        """Bug 1 (negative): Driving task Intervention must be 0 when prediction is RED,
        even though verification is irrelevant."""
        ld_tp = self._make_driving_task_tp()

        def tp_factory(dm_df, name):
            return ld_tp if name == "Long distraction" else self._empty_tp()

        transient_df = pd.DataFrame(
            {
                "Category": ["Transient - Long distraction"],
                "Scenario": ["Driving task - Owl - Intervention"],
                "Glance target type": ["Driving task"],
                "Movement type": ["Owl"],
                "Requirement": ["Forward support"],
                "Value": ["PASS"],
            }
        )
        transient_score_df = pd.DataFrame(
            {
                "Category": ["Transient - Long distraction"],
                "Scenario": ["Driving task - Owl - Intervention"],
                "Score": [0.0],
            }
        )
        dm_colors = {
            0: {
                "Forward support": driver_monitoring.common.PredictionColor.RED,
                "Lane support": driver_monitoring.common.PredictionColor.RED,
            }
        }

        with patch.object(driver_monitoring, "TableProcessor", side_effect=tp_factory):
            result = driver_monitoring.compute_transient_score(
                transient_df, pd.DataFrame(), dm_colors, transient_score_df
            )

        score = result.loc[
            result["Scenario"] == "Driving task - Owl - Intervention", "Score"
        ].values[0]
        self.assertEqual(
            score, 0.0, "Driving task Intervention must be 0 when prediction is RED"
        )

    # ------------------------------------------------------------------
    # Bug 2: Impairment — prediction-only scoring
    # ------------------------------------------------------------------

    def _make_non_transient_tp(self, impairment_type="Drowsiness"):
        """Return a fake TableProcessor whose processed_df mimics the Non-transient section."""

        class FakeTP:
            pass

        fake = FakeTP()
        fake.processed_df = pd.DataFrame(
            {
                "Non transient": ["Impairment"],
                "": [""],
                "Impairment type": [impairment_type],
                "Warning": ["GREEN"],
                "Forward and lane support": ["GREEN"],
                "Emergency function": [None],
            }
        )
        return fake

    def test_compute_non_transient_score_impairment_warning_ignores_verification(self):
        """Bug 2: Impairment Warning score must be awarded based on prediction alone,
        even when the verification test FAILS."""
        nt_tp = self._make_non_transient_tp("Drowsiness")

        non_transient_df = pd.DataFrame(
            {
                "Category": ["Non-transient - Impairment"],
                "Scenario": ["Warning"],
                "Glance target type": ["Drowsiness"],
                "Requirement": ["Warning"],
                "Value": ["FAIL"],
            }
        )
        non_transient_score_df = pd.DataFrame(
            {
                "Category": ["Non-transient - Impairment"],
                "Scenario": ["Drowsiness - Warning"],
                "Score": [0.0],
            }
        )
        dm_colors = {0: {"Warning": driver_monitoring.common.PredictionColor.GREEN}}

        with patch.object(driver_monitoring, "TableProcessor", return_value=nt_tp):
            result = driver_monitoring.compute_non_transient_score(
                non_transient_df, pd.DataFrame(), dm_colors, non_transient_score_df
            )

        score = result.loc[
            result["Scenario"] == "Drowsiness - Warning", "Score"
        ].values[0]
        self.assertGreater(
            score,
            0.0,
            "Impairment Warning should score > 0 when prediction is GREEN, "
            "regardless of verification result",
        )

    def test_compute_non_transient_score_impairment_intervention_ignores_verification(
        self,
    ):
        """Bug 2: Impairment Intervention score must be awarded based on prediction alone,
        even when the verification test FAILS."""
        nt_tp = self._make_non_transient_tp("Drowsiness")

        non_transient_df = pd.DataFrame(
            {
                "Category": ["Non-transient - Impairment"],
                "Scenario": ["Forward and lane support"],
                "Glance target type": ["Drowsiness"],
                "Requirement": ["Forward and lane support"],
                "Value": ["FAIL"],
            }
        )
        non_transient_score_df = pd.DataFrame(
            {
                "Category": ["Non-transient - Impairment"],
                "Scenario": ["Drowsiness - Intervention"],
                "Score": [0.0],
            }
        )
        dm_colors = {
            0: {
                "Forward and lane support": driver_monitoring.common.PredictionColor.GREEN
            }
        }

        with patch.object(driver_monitoring, "TableProcessor", return_value=nt_tp):
            result = driver_monitoring.compute_non_transient_score(
                non_transient_df, pd.DataFrame(), dm_colors, non_transient_score_df
            )

        score = result.loc[
            result["Scenario"] == "Drowsiness - Intervention", "Score"
        ].values[0]
        self.assertGreater(
            score,
            0.0,
            "Impairment Intervention should score > 0 when prediction is GREEN, "
            "regardless of verification result",
        )

    def test_compute_non_transient_score_microsleep_blank_scores_unassessed_not_zero(
        self,
    ):
        """A blank (not yet assessed) verification value for a non-Impairment
        task must leave the scenario unassessed (NaN), not scored as a FAIL
        (0.0)."""

        class FakeTP:
            pass

        fake = FakeTP()
        fake.processed_df = pd.DataFrame(
            {
                "Non transient": ["Microsleep"],
                "": [""],
                "Impairment type": [None],
                "Warning": ["GREEN"],
                "Forward and lane support": [None],
                "Emergency function": [None],
            }
        )

        non_transient_df = pd.DataFrame(
            {
                "Category": ["Non-transient - Microsleep"],
                "Scenario": ["Warning"],
                "Glance target type": ["Microsleep"],
                "Requirement": ["Warning"],
                "Value": [np.nan],
            }
        )
        non_transient_score_df = pd.DataFrame(
            {
                "Category": ["Non-transient - Microsleep"],
                "Scenario": ["Warning"],
                "Score": [0.0],
            }
        )
        dm_colors = {0: {"Warning": driver_monitoring.common.PredictionColor.GREEN}}

        with patch.object(driver_monitoring, "TableProcessor", return_value=fake):
            result = driver_monitoring.compute_non_transient_score(
                non_transient_df, pd.DataFrame(), dm_colors, non_transient_score_df
            )

        score = result.loc[result["Scenario"] == "Warning", "Score"].values[0]
        self.assertTrue(pd.isna(score))

    # ------------------------------------------------------------------
    # Bug 3: Long Distraction cascade — skip when Intervention is verified
    # ------------------------------------------------------------------

    def test_clean_df_warning_score_cascade_skipped_when_intervention_verified(self):
        """Bug 3: Long Distraction cascade must NOT zero out Intervention when it was
        independently verified (present in verified_scenarios)."""
        category = "Transient - Long distraction"
        intervention_scenario = "Non-driving task - Owl - Intervention"
        df = pd.DataFrame(
            {
                "Category": [category, category],
                "Scenario": ["Non-driving task - Owl - Warning", intervention_scenario],
                "Score": [0.0, 0.9],
            }
        )
        verified_scenarios = {(category, intervention_scenario)}
        result = driver_monitoring.clean_df_warning_score(df, verified_scenarios)
        score = result.loc[result["Scenario"] == intervention_scenario, "Score"].values[
            0
        ]
        self.assertEqual(
            score,
            0.9,
            "Cascade must be skipped when Intervention was independently verified",
        )

    def test_clean_df_warning_score_cascade_applies_when_intervention_not_verified(
        self,
    ):
        """Bug 3: Long Distraction cascade MUST zero out Intervention when it was NOT
        independently verified (absent from verified_scenarios)."""
        category = "Transient - Long distraction"
        intervention_scenario = "Non-driving task - Owl - Intervention"
        df = pd.DataFrame(
            {
                "Category": [category, category],
                "Scenario": ["Non-driving task - Owl - Warning", intervention_scenario],
                "Score": [0.0, 0.9],
            }
        )
        verified_scenarios = set()  # Intervention not in the set
        result = driver_monitoring.clean_df_warning_score(df, verified_scenarios)
        score = result.loc[result["Scenario"] == intervention_scenario, "Score"].values[
            0
        ]
        self.assertEqual(
            score,
            0.0,
            "Cascade must zero out Intervention when it was not independently verified",
        )

    def test_compute_non_transient_score_impairment_scores_zero_when_no_prediction_data(
        self,
    ):
        """Regression: all([]) == True; Impairment must not score when dm_colors is empty.
        This is the most critical case: all_test_pass=True for Impairment, so the only
        guard against a false positive is all_prediction_green."""
        nt_tp = self._make_non_transient_tp("Drowsiness")
        non_transient_df = pd.DataFrame(
            {
                "Category": ["Non-transient - Impairment"],
                "Scenario": ["Warning"],
                "Glance target type": ["Drowsiness"],
                "Requirement": ["Warning"],
                "Value": ["FAIL"],
            }
        )
        non_transient_score_df = pd.DataFrame(
            {
                "Category": ["Non-transient - Impairment"],
                "Scenario": ["Drowsiness - Warning"],
                "Score": [0.0],
            }
        )
        with patch.object(driver_monitoring, "TableProcessor", return_value=nt_tp):
            result = driver_monitoring.compute_non_transient_score(
                non_transient_df, pd.DataFrame(), {}, non_transient_score_df
            )
        score = result.loc[
            result["Scenario"] == "Drowsiness - Warning", "Score"
        ].values[0]
        self.assertEqual(
            score, 0.0, "Impairment must not score when dm_colors has no entries"
        )


class TestComputeTransientScoreIntervention(unittest.TestCase):
    """Fix 1: Forward and Lane support must be scored independently."""

    _DATA_ROW_IDX = 2  # index in dm_prediction_df where the test data row lands

    def _make_dm_prediction_df(self, glance_target_type, movement_type):
        """Minimal dm_prediction_df with all three sections so TableProcessor doesn't fail.

        Long distraction section occupies rows 0-2 (terminated by all-NaN at row 3).
        prepare_df slices self.df[2:] which keeps only the data row at index 2.
        """
        nan = None
        return pd.DataFrame(
            [
                # Long distraction: header rows + one data row, then all-NaN separator
                [
                    "Long distraction",
                    "Movement type",
                    "Gaze location",
                    "Warning",
                    "FS",
                    "LS",
                ],
                ["SubHeader", nan, nan, nan, nan, nan],
                [
                    glance_target_type,
                    movement_type,
                    "Road",
                    nan,
                    nan,
                    nan,
                ],  # idx 2 = data
                [
                    nan,
                    nan,
                    nan,
                    nan,
                    nan,
                    nan,
                ],  # all-NaN → ends Long distraction section
                # Short distraction: two rows (no data, prepare_df[2:] is empty)
                [
                    "Short distraction",
                    "Movement type",
                    "Gaze location",
                    "Warning",
                    "FS",
                    "LS",
                ],
                ["SubHeader", nan, nan, nan, nan, nan],
                [
                    nan,
                    nan,
                    nan,
                    nan,
                    nan,
                    nan,
                ],  # all-NaN → ends Short distraction section
                # Phone use: two rows (no data)
                ["Phone use", "Movement type", "Gaze location", "Warning", "FS", "LS"],
                ["SubHeader", nan, nan, nan, nan, nan],
            ]
        )

    def _score_df(self, glance_target_type, movement_type):
        return pd.DataFrame(
            [
                {
                    "Category": "Transient - Long distraction",
                    "Scenario": f"{glance_target_type} - {movement_type} - Intervention",
                    "Score": 0.0,
                }
            ]
        )

    def _empty_transient_df(self):
        return pd.DataFrame(
            columns=[
                "Category",
                "Scenario",
                "Glance target type",
                "Movement type",
                "Requirement",
                "Value",
            ]
        )

    def test_driving_task_fs_green_ls_red_awards_fs_only(self):
        """Driving task: Forward=GREEN, Lane=RED → FS score (0.8) only."""
        idx = self._DATA_ROW_IDX
        dm_colors = {
            idx: {
                "Forward support": common.PredictionColor.GREEN,
                "Lane support": common.PredictionColor.RED,
            }
        }
        result = driver_monitoring.compute_transient_score(
            self._empty_transient_df(),
            self._make_dm_prediction_df("Driving task", "Owl"),
            dm_colors,
            self._score_df("Driving task", "Owl"),
        )
        self.assertAlmostEqual(result.iloc[0]["Score"], 0.8)

    def test_driving_task_ls_green_fs_red_awards_ls_only(self):
        """Driving task: Lane=GREEN, Forward=RED → LS score (0.2) only."""
        idx = self._DATA_ROW_IDX
        dm_colors = {
            idx: {
                "Forward support": common.PredictionColor.RED,
                "Lane support": common.PredictionColor.GREEN,
            }
        }
        result = driver_monitoring.compute_transient_score(
            self._empty_transient_df(),
            self._make_dm_prediction_df("Driving task", "Owl"),
            dm_colors,
            self._score_df("Driving task", "Owl"),
        )
        self.assertAlmostEqual(result.iloc[0]["Score"], 0.2)

    def test_driving_task_both_green_awards_full_score(self):
        """Driving task: Forward=GREEN, Lane=GREEN → full score (1.0)."""
        idx = self._DATA_ROW_IDX
        dm_colors = {
            idx: {
                "Forward support": common.PredictionColor.GREEN,
                "Lane support": common.PredictionColor.GREEN,
            }
        }
        result = driver_monitoring.compute_transient_score(
            self._empty_transient_df(),
            self._make_dm_prediction_df("Driving task", "Owl"),
            dm_colors,
            self._score_df("Driving task", "Owl"),
        )
        self.assertAlmostEqual(result.iloc[0]["Score"], 1.0)

    def test_driving_task_both_red_scores_zero(self):
        """Driving task: Forward=RED, Lane=RED → 0."""
        idx = self._DATA_ROW_IDX
        dm_colors = {
            idx: {
                "Forward support": common.PredictionColor.RED,
                "Lane support": common.PredictionColor.RED,
            }
        }
        result = driver_monitoring.compute_transient_score(
            self._empty_transient_df(),
            self._make_dm_prediction_df("Driving task", "Owl"),
            dm_colors,
            self._score_df("Driving task", "Owl"),
        )
        self.assertAlmostEqual(result.iloc[0]["Score"], 0.0)

    def test_non_driving_task_fs_green_ls_red_with_all_pass(self):
        """Non-driving task: Forward=GREEN+PASS, Lane=RED → FS score (0.4) only."""
        idx = self._DATA_ROW_IDX
        dm_colors = {
            idx: {
                "Forward support": common.PredictionColor.GREEN,
                "Lane support": common.PredictionColor.RED,
            }
        }
        transient_df = pd.DataFrame(
            [
                {
                    "Category": "Transient - Long distraction",
                    "Scenario": "Non-driving task - Owl - Intervention",
                    "Glance target type": "Non-driving task",
                    "Movement type": "Owl",
                    "Requirement": "Forward support",
                    "Value": "PASS",
                },
                {
                    "Category": "Transient - Long distraction",
                    "Scenario": "Non-driving task - Owl - Intervention",
                    "Glance target type": "Non-driving task",
                    "Movement type": "Owl",
                    "Requirement": "Lane support",
                    "Value": "PASS",
                },
            ]
        )
        result = driver_monitoring.compute_transient_score(
            transient_df,
            self._make_dm_prediction_df("Non-driving task", "Owl"),
            dm_colors,
            self._score_df("Non-driving task", "Owl"),
        )
        # FS=0.4 (GREEN+PASS), LS=0 (RED) for Non-driving task Owl
        self.assertAlmostEqual(result.iloc[0]["Score"], 0.4)

    def test_non_driving_task_fs_fail_ls_green_awards_ls_only(self):
        """Non-driving task: Forward=GREEN+FAIL, Lane=GREEN+PASS → LS score (0.1) only."""
        idx = self._DATA_ROW_IDX
        dm_colors = {
            idx: {
                "Forward support": common.PredictionColor.GREEN,
                "Lane support": common.PredictionColor.GREEN,
            }
        }
        transient_df = pd.DataFrame(
            [
                {
                    "Category": "Transient - Long distraction",
                    "Scenario": "Non-driving task - Owl - Intervention",
                    "Glance target type": "Non-driving task",
                    "Movement type": "Owl",
                    "Requirement": "Forward support",
                    "Value": "FAIL",
                },
                {
                    "Category": "Transient - Long distraction",
                    "Scenario": "Non-driving task - Owl - Intervention",
                    "Glance target type": "Non-driving task",
                    "Movement type": "Owl",
                    "Requirement": "Lane support",
                    "Value": "PASS",
                },
            ]
        )
        result = driver_monitoring.compute_transient_score(
            transient_df,
            self._make_dm_prediction_df("Non-driving task", "Owl"),
            dm_colors,
            self._score_df("Non-driving task", "Owl"),
        )
        # FS=0 (FAIL), LS=0.1 (GREEN+PASS) for Non-driving task Owl
        self.assertAlmostEqual(result.iloc[0]["Score"], 0.1)

    def test_driving_task_intervention_scores_zero_when_no_prediction_data(self):
        """Regression: all([]) == True; Intervention must not score when dm_colors is empty."""
        result = driver_monitoring.compute_transient_score(
            self._empty_transient_df(),
            self._make_dm_prediction_df("Driving task", "Owl"),
            {},
            self._score_df("Driving task", "Owl"),
        )
        self.assertAlmostEqual(
            result.iloc[0]["Score"],
            0.0,
            msg="Intervention must not score when dm_colors has no entries",
        )

    def test_non_driving_task_warning_scores_zero_when_no_prediction_data(self):
        """Regression: all([]) == True; Warning must not score when dm_colors is empty."""
        score_df = pd.DataFrame(
            [
                {
                    "Category": "Transient - Long distraction",
                    "Scenario": "Non-driving task - Owl - Warning",
                    "Score": 0.0,
                }
            ]
        )
        result = driver_monitoring.compute_transient_score(
            self._empty_transient_df(),
            self._make_dm_prediction_df("Non-driving task", "Owl"),
            {},
            score_df,
        )
        self.assertAlmostEqual(
            result.iloc[0]["Score"],
            0.0,
            msg="Warning must not score when dm_colors has no entries",
        )

    # ------------------------------------------------------------------
    # New Euro NCAP requirement: Warning predictions all GREEN but a
    # verification test failed -> force FS/LS to 0 for that movement type
    # (Long distraction: both FS and LS; Short distraction: LS only).
    # ------------------------------------------------------------------

    def _make_dm_prediction_df_for_category(
        self, category, glance_target_type, movement_type
    ):
        """Like _make_dm_prediction_df, but the data row lands under `category`'s own
        section while still landing at absolute row index _DATA_ROW_IDX (2), by placing
        that category's section first."""
        nan = None
        rows = [
            [category, "Movement type", "Gaze location", "Warning", "FS", "LS"],
            ["SubHeader", nan, nan, nan, nan, nan],
            [glance_target_type, movement_type, "Road", nan, nan, nan],  # idx 2 = data
            [nan, nan, nan, nan, nan, nan],
        ]
        for other in ["Long distraction", "Short distraction", "Phone use"]:
            if other == category:
                continue
            rows.extend(
                [
                    [other, "Movement type", "Gaze location", "Warning", "FS", "LS"],
                    ["SubHeader", nan, nan, nan, nan, nan],
                    [nan, nan, nan, nan, nan, nan],
                ]
            )
        return pd.DataFrame(rows)

    def _score_df_with_warning(self, category, glance_target_type, movement_type):
        cat = f"Transient - {category}"
        return pd.DataFrame(
            [
                {
                    "Category": cat,
                    "Scenario": f"{glance_target_type} - {movement_type} - Warning",
                    "Score": 0.0,
                },
                {
                    "Category": cat,
                    "Scenario": f"{glance_target_type} - {movement_type} - Intervention",
                    "Score": 0.0,
                },
            ]
        )

    def test_long_distraction_warning_green_but_test_failed_zeroes_fs_and_ls(self):
        """Warning predictions GREEN but the Warning test FAILs -> Intervention forced to
        0 even though FS/LS have their own independently-passing verification."""
        idx = self._DATA_ROW_IDX
        dm_colors = {
            idx: {
                "Warning": common.PredictionColor.GREEN,
                "Forward support": common.PredictionColor.GREEN,
                "Lane support": common.PredictionColor.GREEN,
            }
        }
        transient_df = pd.DataFrame(
            [
                {
                    "Category": "Transient - Long distraction",
                    "Scenario": "Non-driving task - Owl - Warning",
                    "Glance target type": "Non-driving task",
                    "Movement type": "Owl",
                    "Requirement": "Warning",
                    "Value": "FAIL",
                },
                {
                    "Category": "Transient - Long distraction",
                    "Scenario": "Non-driving task - Owl - Intervention",
                    "Glance target type": "Non-driving task",
                    "Movement type": "Owl",
                    "Requirement": "Forward support",
                    "Value": "PASS",
                },
                {
                    "Category": "Transient - Long distraction",
                    "Scenario": "Non-driving task - Owl - Intervention",
                    "Glance target type": "Non-driving task",
                    "Movement type": "Owl",
                    "Requirement": "Lane support",
                    "Value": "PASS",
                },
            ]
        )
        result = driver_monitoring.compute_transient_score(
            transient_df,
            self._make_dm_prediction_df_for_category(
                "Long distraction", "Non-driving task", "Owl"
            ),
            dm_colors,
            self._score_df_with_warning("Long distraction", "Non-driving task", "Owl"),
        )
        intervention_score = result.loc[
            result["Scenario"] == "Non-driving task - Owl - Intervention", "Score"
        ].values[0]
        self.assertEqual(intervention_score, 0.0)

    def test_long_distraction_warning_green_and_test_passed_does_not_cascade(self):
        """Regression: Warning test PASSes -> no new cascade, full FS+LS sum awarded."""
        idx = self._DATA_ROW_IDX
        dm_colors = {
            idx: {
                "Warning": common.PredictionColor.GREEN,
                "Forward support": common.PredictionColor.GREEN,
                "Lane support": common.PredictionColor.GREEN,
            }
        }
        transient_df = pd.DataFrame(
            [
                {
                    "Category": "Transient - Long distraction",
                    "Scenario": "Non-driving task - Owl - Warning",
                    "Glance target type": "Non-driving task",
                    "Movement type": "Owl",
                    "Requirement": "Warning",
                    "Value": "PASS",
                },
                {
                    "Category": "Transient - Long distraction",
                    "Scenario": "Non-driving task - Owl - Intervention",
                    "Glance target type": "Non-driving task",
                    "Movement type": "Owl",
                    "Requirement": "Forward support",
                    "Value": "PASS",
                },
                {
                    "Category": "Transient - Long distraction",
                    "Scenario": "Non-driving task - Owl - Intervention",
                    "Glance target type": "Non-driving task",
                    "Movement type": "Owl",
                    "Requirement": "Lane support",
                    "Value": "PASS",
                },
            ]
        )
        result = driver_monitoring.compute_transient_score(
            transient_df,
            self._make_dm_prediction_df_for_category(
                "Long distraction", "Non-driving task", "Owl"
            ),
            dm_colors,
            self._score_df_with_warning("Long distraction", "Non-driving task", "Owl"),
        )
        intervention_score = result.loc[
            result["Scenario"] == "Non-driving task - Owl - Intervention", "Score"
        ].values[0]
        self.assertAlmostEqual(intervention_score, 0.5)

    def test_long_distraction_warning_no_verification_rows_does_not_cascade(self):
        """Regression: no Warning verification rows at all -> all_test_pass defaults True,
        so the new hard-fail rule must not fire."""
        idx = self._DATA_ROW_IDX
        dm_colors = {
            idx: {
                "Warning": common.PredictionColor.GREEN,
                "Forward support": common.PredictionColor.GREEN,
                "Lane support": common.PredictionColor.GREEN,
            }
        }
        transient_df = pd.DataFrame(
            columns=[
                "Category",
                "Scenario",
                "Glance target type",
                "Movement type",
                "Requirement",
                "Value",
            ]
        )
        result = driver_monitoring.compute_transient_score(
            transient_df,
            self._make_dm_prediction_df_for_category(
                "Long distraction", "Non-driving task", "Owl"
            ),
            dm_colors,
            self._score_df_with_warning("Long distraction", "Non-driving task", "Owl"),
        )
        intervention_score = result.loc[
            result["Scenario"] == "Non-driving task - Owl - Intervention", "Score"
        ].values[0]
        self.assertAlmostEqual(intervention_score, 0.5)

    def test_short_distraction_warning_green_but_test_failed_zeroes_ls_only(self):
        """Short distraction: same hard-fail condition zeroes LS only; FS keeps its own
        independently-passing score."""
        idx = self._DATA_ROW_IDX
        dm_colors = {
            idx: {
                "Warning": common.PredictionColor.GREEN,
                "Forward support": common.PredictionColor.GREEN,
                "Lane support": common.PredictionColor.GREEN,
            }
        }
        transient_df = pd.DataFrame(
            [
                {
                    "Category": "Transient - Short distraction",
                    "Scenario": "Non-driving task - Owl - Warning",
                    "Glance target type": "Non-driving task",
                    "Movement type": "Owl",
                    "Requirement": "Warning",
                    "Value": "FAIL",
                },
                {
                    "Category": "Transient - Short distraction",
                    "Scenario": "Non-driving task - Owl - Intervention",
                    "Glance target type": "Non-driving task",
                    "Movement type": "Owl",
                    "Requirement": "Forward support",
                    "Value": "PASS",
                },
                {
                    "Category": "Transient - Short distraction",
                    "Scenario": "Non-driving task - Owl - Intervention",
                    "Glance target type": "Non-driving task",
                    "Movement type": "Owl",
                    "Requirement": "Lane support",
                    "Value": "PASS",
                },
            ]
        )
        result = driver_monitoring.compute_transient_score(
            transient_df,
            self._make_dm_prediction_df_for_category(
                "Short distraction", "Non-driving task", "Owl"
            ),
            dm_colors,
            self._score_df_with_warning("Short distraction", "Non-driving task", "Owl"),
        )
        intervention_score = result.loc[
            result["Scenario"] == "Non-driving task - Owl - Intervention", "Score"
        ].values[0]
        # FS=0.4 kept, LS=0.1 forced to 0 -> 0.4 total (not the full 0.5 sum).
        self.assertAlmostEqual(intervention_score, 0.4)

    def test_short_distraction_warning_red_prediction_still_no_cascade(self):
        """Regression: Warning prediction RED (not green) must not trigger the new rule;
        Short distraction has no old cascade either, so FS+LS stay fully independent."""
        idx = self._DATA_ROW_IDX
        dm_colors = {
            idx: {
                "Warning": common.PredictionColor.RED,
                "Forward support": common.PredictionColor.GREEN,
                "Lane support": common.PredictionColor.GREEN,
            }
        }
        transient_df = pd.DataFrame(
            [
                {
                    "Category": "Transient - Short distraction",
                    "Scenario": "Non-driving task - Owl - Warning",
                    "Glance target type": "Non-driving task",
                    "Movement type": "Owl",
                    "Requirement": "Warning",
                    "Value": "FAIL",
                },
                {
                    "Category": "Transient - Short distraction",
                    "Scenario": "Non-driving task - Owl - Intervention",
                    "Glance target type": "Non-driving task",
                    "Movement type": "Owl",
                    "Requirement": "Forward support",
                    "Value": "PASS",
                },
                {
                    "Category": "Transient - Short distraction",
                    "Scenario": "Non-driving task - Owl - Intervention",
                    "Glance target type": "Non-driving task",
                    "Movement type": "Owl",
                    "Requirement": "Lane support",
                    "Value": "PASS",
                },
            ]
        )
        result = driver_monitoring.compute_transient_score(
            transient_df,
            self._make_dm_prediction_df_for_category(
                "Short distraction", "Non-driving task", "Owl"
            ),
            dm_colors,
            self._score_df_with_warning("Short distraction", "Non-driving task", "Owl"),
        )
        intervention_score = result.loc[
            result["Scenario"] == "Non-driving task - Owl - Intervention", "Score"
        ].values[0]
        self.assertAlmostEqual(intervention_score, 0.5)

    # ------------------------------------------------------------------
    # Blank (not yet assessed) verification values must score unassessed
    # (NaN), never a false FAIL (0.0) -- and must not trigger the
    # warning-hard-fail cascade, which is reserved for an explicit FAIL.
    # ------------------------------------------------------------------

    def test_non_driving_task_fs_blank_scores_unassessed_not_zero(self):
        """A blank Forward support verification value must leave the
        Intervention scenario unassessed (NaN), not scored as a FAIL (0.0)."""
        idx = self._DATA_ROW_IDX
        dm_colors = {
            idx: {
                "Forward support": common.PredictionColor.GREEN,
                "Lane support": common.PredictionColor.GREEN,
            }
        }
        transient_df = pd.DataFrame(
            [
                {
                    "Category": "Transient - Long distraction",
                    "Scenario": "Non-driving task - Owl - Intervention",
                    "Glance target type": "Non-driving task",
                    "Movement type": "Owl",
                    "Requirement": "Forward support",
                    "Value": np.nan,
                },
                {
                    "Category": "Transient - Long distraction",
                    "Scenario": "Non-driving task - Owl - Intervention",
                    "Glance target type": "Non-driving task",
                    "Movement type": "Owl",
                    "Requirement": "Lane support",
                    "Value": "PASS",
                },
            ]
        )
        result = driver_monitoring.compute_transient_score(
            transient_df,
            self._make_dm_prediction_df("Non-driving task", "Owl"),
            dm_colors,
            self._score_df("Non-driving task", "Owl"),
        )
        self.assertTrue(pd.isna(result.iloc[0]["Score"]))

    def test_non_driving_task_warning_blank_scores_unassessed_not_zero(self):
        """A blank Warning verification value must leave the scenario
        unassessed (NaN), not scored as a FAIL (0.0)."""
        idx = self._DATA_ROW_IDX
        dm_colors = {idx: {"Warning": common.PredictionColor.GREEN}}
        transient_df = pd.DataFrame(
            [
                {
                    "Category": "Transient - Long distraction",
                    "Scenario": "Non-driving task - Owl - Warning",
                    "Glance target type": "Non-driving task",
                    "Movement type": "Owl",
                    "Requirement": "Warning",
                    "Value": np.nan,
                },
            ]
        )
        score_df = pd.DataFrame(
            [
                {
                    "Category": "Transient - Long distraction",
                    "Scenario": "Non-driving task - Owl - Warning",
                    "Score": 0.0,
                }
            ]
        )
        result = driver_monitoring.compute_transient_score(
            transient_df,
            self._make_dm_prediction_df("Non-driving task", "Owl"),
            dm_colors,
            score_df,
        )
        self.assertTrue(pd.isna(result.iloc[0]["Score"]))

    def test_long_distraction_warning_blank_does_not_cascade(self):
        """A Warning verification row that's simply unassessed (blank) must
        NOT trigger the hard-fail cascade onto Intervention -- only an
        explicit FAIL should. (Prevents every untouched submission's
        Warning row from silently zeroing out its Intervention score.)"""
        idx = self._DATA_ROW_IDX
        dm_colors = {
            idx: {
                "Warning": common.PredictionColor.GREEN,
                "Forward support": common.PredictionColor.GREEN,
                "Lane support": common.PredictionColor.GREEN,
            }
        }
        transient_df = pd.DataFrame(
            [
                {
                    "Category": "Transient - Long distraction",
                    "Scenario": "Non-driving task - Owl - Warning",
                    "Glance target type": "Non-driving task",
                    "Movement type": "Owl",
                    "Requirement": "Warning",
                    "Value": np.nan,
                },
                {
                    "Category": "Transient - Long distraction",
                    "Scenario": "Non-driving task - Owl - Intervention",
                    "Glance target type": "Non-driving task",
                    "Movement type": "Owl",
                    "Requirement": "Forward support",
                    "Value": "PASS",
                },
                {
                    "Category": "Transient - Long distraction",
                    "Scenario": "Non-driving task - Owl - Intervention",
                    "Glance target type": "Non-driving task",
                    "Movement type": "Owl",
                    "Requirement": "Lane support",
                    "Value": "PASS",
                },
            ]
        )
        result = driver_monitoring.compute_transient_score(
            transient_df,
            self._make_dm_prediction_df_for_category(
                "Long distraction", "Non-driving task", "Owl"
            ),
            dm_colors,
            self._score_df_with_warning("Long distraction", "Non-driving task", "Owl"),
        )
        warning_score = result.loc[
            result["Scenario"] == "Non-driving task - Owl - Warning", "Score"
        ].values[0]
        intervention_score = result.loc[
            result["Scenario"] == "Non-driving task - Owl - Intervention", "Score"
        ].values[0]
        self.assertTrue(pd.isna(warning_score))
        self.assertAlmostEqual(intervention_score, 0.5)


class TestGetDmColorsFromText(unittest.TestCase):
    """Fix 2: dm_colors must be built from cell text, not fill color."""

    def _make_df(self, rows):
        """Create a DataFrame where each item in rows is a list of 6 cell values."""
        return pd.DataFrame(rows)

    def test_green_text_maps_to_green_color(self):
        noise_len = 4  # min_row = 4 + 3 = 7
        rows = [[None] * 6 for _ in range(20)]
        rows[7] = ["Non-driving task", "Owl", "Road", "GREEN", "GREEN", "RED"]
        df = self._make_df(rows)
        dm_colors = driver_monitoring.get_dm_colors_from_text(df, noise_len)
        self.assertIn(7, dm_colors)
        self.assertEqual(dm_colors[7]["Warning"], common.PredictionColor.GREEN)
        self.assertEqual(dm_colors[7]["Forward support"], common.PredictionColor.GREEN)
        self.assertEqual(dm_colors[7]["Lane support"], common.PredictionColor.RED)

    def test_red_text_maps_to_red_color(self):
        noise_len = 0  # min_row = 3
        rows = [[None] * 6 for _ in range(10)]
        rows[5] = [None, None, None, "RED", "RED", "RED"]
        df = self._make_df(rows)
        dm_colors = driver_monitoring.get_dm_colors_from_text(df, noise_len)
        self.assertIn(5, dm_colors)
        self.assertEqual(dm_colors[5]["Warning"], common.PredictionColor.RED)
        self.assertEqual(dm_colors[5]["Forward support"], common.PredictionColor.RED)
        self.assertEqual(dm_colors[5]["Lane support"], common.PredictionColor.RED)

    def test_rows_below_threshold_excluded(self):
        noise_len = 4  # min_row = 7
        rows = [[None] * 6 for _ in range(10)]
        rows[6] = [None, None, None, "GREEN", "GREEN", "GREEN"]  # below threshold
        df = self._make_df(rows)
        dm_colors = driver_monitoring.get_dm_colors_from_text(df, noise_len)
        self.assertNotIn(6, dm_colors)

    def test_non_transient_rows_use_different_column_mapping(self):
        noise_len = 0  # min_row = 3
        rows = [[None] * 6 for _ in range(70)]
        rows[66] = [None, None, None, "GREEN", "RED", "GREEN"]
        df = self._make_df(rows)
        dm_colors = driver_monitoring.get_dm_colors_from_text(df, noise_len)
        self.assertIn(66, dm_colors)
        self.assertEqual(dm_colors[66]["Warning"], common.PredictionColor.GREEN)
        self.assertEqual(
            dm_colors[66]["Forward and lane support"], common.PredictionColor.RED
        )
        self.assertEqual(
            dm_colors[66]["Emergency function"], common.PredictionColor.GREEN
        )

    def test_non_string_cells_not_included(self):
        noise_len = 0  # min_row = 3
        rows = [[None] * 6 for _ in range(10)]
        rows[3] = [None, None, None, None, None, None]  # no text values
        df = self._make_df(rows)
        dm_colors = driver_monitoring.get_dm_colors_from_text(df, noise_len)
        self.assertNotIn(3, dm_colors)

    def test_case_insensitive_text(self):
        noise_len = 0  # min_row = 3
        rows = [[None] * 6 for _ in range(10)]
        rows[3] = [None, None, None, "Green", " red ", "GREEN"]
        df = self._make_df(rows)
        dm_colors = driver_monitoring.get_dm_colors_from_text(df, noise_len)
        self.assertIn(3, dm_colors)
        self.assertEqual(dm_colors[3]["Warning"], common.PredictionColor.GREEN)
        self.assertEqual(dm_colors[3]["Forward support"], common.PredictionColor.RED)
        self.assertEqual(dm_colors[3]["Lane support"], common.PredictionColor.GREEN)


class TestGetCellDictGreyDetection(unittest.TestCase):
    """GREY predictions from literal N/A/GREY/GRAY text; required-but-blank cells map to RED."""

    def _build(self, prediction_rows, required_flags_rows=None):
        columns = driver_monitoring.DM_COLUMNS
        padding = [["Long distraction", "Owl", "Road", "GREEN", "GREEN", "GREEN"]] * 2
        dm_prediction_df = pd.DataFrame(padding + prediction_rows, columns=columns)

        required_mask_df = None
        if required_flags_rows is not None:
            required_mask_df = dm_prediction_df.copy()
            all_flags = [
                [False, False, False],
                [False, False, False],
            ] + required_flags_rows
            for row_idx, flags in enumerate(all_flags):
                required_mask_df.loc[row_idx, columns[3:6]] = flags
        return dm_prediction_df, required_mask_df

    def test_na_text_maps_to_grey(self):
        dm_prediction_df, _ = self._build(
            [["Non-driving task", "Owl", "Road", "N/A", "GREEN", "GREEN"]]
        )
        tp = driver_monitoring.TableProcessor(dm_prediction_df, "Long distraction")
        self.assertEqual(tp.cell_dict[(2, 3)]["Color"], "GREY")

    def test_grey_and_gray_text_case_insensitive_maps_to_grey(self):
        dm_prediction_df, _ = self._build(
            [["Non-driving task", "Owl", "Road", "Grey", "gray", "GREEN"]]
        )
        tp = driver_monitoring.TableProcessor(dm_prediction_df, "Long distraction")
        self.assertEqual(tp.cell_dict[(2, 3)]["Color"], "GREY")
        self.assertEqual(tp.cell_dict[(2, 4)]["Color"], "GREY")

    def test_blank_cell_without_required_mask_stays_excluded(self):
        dm_prediction_df, _ = self._build(
            [["Non-driving task", "Owl", "Road", None, "GREEN", "GREEN"]]
        )
        tp = driver_monitoring.TableProcessor(dm_prediction_df, "Long distraction")
        self.assertNotIn((2, 3), tp.cell_dict)

    def test_required_blank_cell_maps_to_red(self):
        dm_prediction_df, required_mask_df = self._build(
            prediction_rows=[
                ["Non-driving task", "Owl", "Road", None, "GREEN", "GREEN"]
            ],
            required_flags_rows=[[True, False, False]],
        )
        tp = driver_monitoring.TableProcessor(
            dm_prediction_df, "Long distraction", required_mask_df=required_mask_df
        )
        self.assertEqual(tp.cell_dict[(2, 3)]["Color"], "RED")

    def test_not_required_blank_cell_stays_excluded(self):
        dm_prediction_df, required_mask_df = self._build(
            prediction_rows=[
                ["Non-driving task", "Owl", "Road", None, "GREEN", "GREEN"]
            ],
            required_flags_rows=[[False, False, False]],
        )
        tp = driver_monitoring.TableProcessor(
            dm_prediction_df, "Long distraction", required_mask_df=required_mask_df
        )
        self.assertNotIn((2, 3), tp.cell_dict)


class TestPreprocessRequiredMaskFallback(unittest.TestCase):
    def _run_preprocess(self, required_mask_df, default_mask_df):
        captured_masks = []

        class FakeTableProcessor:
            def __init__(self, _dm_prediction_df, df_name, required_mask_df=None):
                captured_masks.append(required_mask_df)
                self.name = df_name
                self.driver_state = "Transient"
                rows = []
                groups = [
                    ("Non-driving task", "Owl"),
                    ("Non-driving task", "Lizard"),
                    ("Driving task", "Owl"),
                    ("Basic phone use", "Owl and lizard"),
                    ("Advanced phone use", "Lizard"),
                    ("Multi-target", "Lizard"),
                ]
                for task, movement_type in groups:
                    for _ in range(4):
                        rows.append(
                            {
                                "Long distraction": task,
                                "Movement type": movement_type,
                                "Gaze location": "Road",
                                "Warning": "GREEN",
                                "Forward support": (
                                    "GREEN" if task == "Driving task" else "RED"
                                ),
                                "Lane support": (
                                    "GREEN" if task == "Driving task" else "RED"
                                ),
                            }
                        )
                self.processed_df = pd.DataFrame(rows)
                self.cell_dict = {}

            def deepcopy(self):
                return self

        noise_variables_df = pd.DataFrame(
            {
                "Element": ["Dark sunglasses", "Face-mask", "Cap", "Hat"],
                "Performance": ["Functional", "Functional", "Functional", "Functional"],
            }
        )

        with (
            patch.object(
                driver_monitoring, "TableProcessor", side_effect=FakeTableProcessor
            ),
            patch.object(
                driver_monitoring.common,
                "extract_table",
                return_value=noise_variables_df,
            ),
            patch.object(
                driver_monitoring, "get_mandatory_check_dict", return_value={"ok": True}
            ),
            patch.object(
                driver_monitoring, "get_non_transient_points", return_value={}
            ),
            patch.object(
                driver_monitoring.common,
                "get_default_dm_prediction_required_mask",
                return_value=default_mask_df,
            ) as mock_default_mask,
        ):
            driver_monitoring.preprocess(
                dm_prediction_df=pd.DataFrame(),
                param_df=pd.DataFrame(),
                required_mask_df=required_mask_df,
            )

        return captured_masks, mock_default_mask

    def test_falls_back_to_default_required_mask_when_none_passed(self):
        sentinel_mask = pd.DataFrame({"sentinel": [True]})
        captured_masks, mock_default_mask = self._run_preprocess(
            required_mask_df=None, default_mask_df=sentinel_mask
        )
        mock_default_mask.assert_called_once()
        self.assertTrue(all(mask is sentinel_mask for mask in captured_masks))

    def test_does_not_call_default_mask_when_required_mask_df_given(self):
        explicit_mask = pd.DataFrame({"explicit": [True]})
        sentinel_mask = pd.DataFrame({"sentinel": [True]})
        captured_masks, mock_default_mask = self._run_preprocess(
            required_mask_df=explicit_mask, default_mask_df=sentinel_mask
        )
        mock_default_mask.assert_not_called()
        self.assertTrue(all(mask is explicit_mask for mask in captured_masks))


class _FakeTP:
    """Minimal TableProcessor stand-in: only .name/.cell_dict are needed by the
    selection functions under test here."""

    def __init__(self, name, cell_dict):
        self.name = name
        self.cell_dict = cell_dict


class TestApplyPredictionConsistency(unittest.TestCase):
    def test_forces_mixed_group_to_red(self):
        tp = _FakeTP(
            "Long distraction",
            {
                (0, 4): {
                    "Task": "Non-driving task",
                    "Movement type": "Owl",
                    "Gaze location": "A",
                    "Vehicle response": "Forward support",
                    "Color": "GREEN",
                },
                (1, 4): {
                    "Task": "Non-driving task",
                    "Movement type": "Owl",
                    "Gaze location": "B",
                    "Vehicle response": "Forward support",
                    "Color": "RED",
                },
            },
        )
        driver_monitoring.apply_prediction_consistency({"Long distraction": tp})
        self.assertEqual(tp.cell_dict[(0, 4)]["Color"], "RED")
        self.assertEqual(tp.cell_dict[(1, 4)]["Color"], "RED")

    def test_leaves_uniform_green_group_untouched(self):
        tp = _FakeTP(
            "Long distraction",
            {
                (0, 4): {
                    "Task": "Non-driving task",
                    "Movement type": "Owl",
                    "Gaze location": "A",
                    "Vehicle response": "Forward support",
                    "Color": "GREEN",
                },
                (1, 4): {
                    "Task": "Non-driving task",
                    "Movement type": "Owl",
                    "Gaze location": "B",
                    "Vehicle response": "Forward support",
                    "Color": "GREEN",
                },
            },
        )
        driver_monitoring.apply_prediction_consistency({"Long distraction": tp})
        self.assertEqual(tp.cell_dict[(0, 4)]["Color"], "GREEN")
        self.assertEqual(tp.cell_dict[(1, 4)]["Color"], "GREEN")

    def test_leaves_uniform_red_group_untouched(self):
        tp = _FakeTP(
            "Long distraction",
            {
                (0, 4): {
                    "Task": "Non-driving task",
                    "Movement type": "Owl",
                    "Gaze location": "A",
                    "Vehicle response": "Forward support",
                    "Color": "RED",
                },
                (1, 4): {
                    "Task": "Non-driving task",
                    "Movement type": "Owl",
                    "Gaze location": "B",
                    "Vehicle response": "Forward support",
                    "Color": "RED",
                },
            },
        )
        driver_monitoring.apply_prediction_consistency({"Long distraction": tp})
        self.assertEqual(tp.cell_dict[(0, 4)]["Color"], "RED")
        self.assertEqual(tp.cell_dict[(1, 4)]["Color"], "RED")

    def test_skips_non_transient_table(self):
        nt = _FakeTP(
            "Non-transient",
            {
                (0, 4): {
                    "Task": "Microsleep",
                    "Movement type": "",
                    "Gaze location": "A",
                    "Vehicle response": "Warning",
                    "Color": "GREEN",
                },
                (1, 4): {
                    "Task": "Microsleep",
                    "Movement type": "",
                    "Gaze location": "B",
                    "Vehicle response": "Warning",
                    "Color": "RED",
                },
            },
        )
        driver_monitoring.apply_prediction_consistency({"Non-transient": nt})
        self.assertEqual(nt.cell_dict[(0, 4)]["Color"], "GREEN")
        self.assertEqual(nt.cell_dict[(1, 4)]["Color"], "RED")

    def test_scopes_by_task(self):
        tp = _FakeTP(
            "Long distraction",
            {
                (0, 4): {
                    "Task": "Non-driving task",
                    "Movement type": "Owl",
                    "Gaze location": "A",
                    "Vehicle response": "Forward support",
                    "Color": "RED",
                },
                (1, 4): {
                    "Task": "Driving task",
                    "Movement type": "Owl",
                    "Gaze location": "B",
                    "Vehicle response": "Forward support",
                    "Color": "GREEN",
                },
            },
        )
        driver_monitoring.apply_prediction_consistency({"Long distraction": tp})
        self.assertEqual(tp.cell_dict[(0, 4)]["Color"], "RED")
        self.assertEqual(tp.cell_dict[(1, 4)]["Color"], "GREEN")

    def test_scopes_by_vehicle_response(self):
        tp = _FakeTP(
            "Long distraction",
            {
                (0, 3): {
                    "Task": "Non-driving task",
                    "Movement type": "Owl",
                    "Gaze location": "A",
                    "Vehicle response": "Warning",
                    "Color": "RED",
                },
                (0, 4): {
                    "Task": "Non-driving task",
                    "Movement type": "Owl",
                    "Gaze location": "A",
                    "Vehicle response": "Forward support",
                    "Color": "GREEN",
                },
            },
        )
        driver_monitoring.apply_prediction_consistency({"Long distraction": tp})
        self.assertEqual(tp.cell_dict[(0, 3)]["Color"], "RED")
        self.assertEqual(tp.cell_dict[(0, 4)]["Color"], "GREEN")

    def test_forces_grey_and_red_mixed_group_to_red(self):
        tp = _FakeTP(
            "Long distraction",
            {
                (0, 4): {
                    "Task": "Non-driving task",
                    "Movement type": "Owl",
                    "Gaze location": "A",
                    "Vehicle response": "Forward support",
                    "Color": "GREY",
                },
                (1, 4): {
                    "Task": "Non-driving task",
                    "Movement type": "Owl",
                    "Gaze location": "B",
                    "Vehicle response": "Forward support",
                    "Color": "RED",
                },
            },
        )
        driver_monitoring.apply_prediction_consistency({"Long distraction": tp})
        self.assertEqual(tp.cell_dict[(0, 4)]["Color"], "RED")
        self.assertEqual(tp.cell_dict[(1, 4)]["Color"], "RED")

    def test_forces_green_grey_red_mixed_group_to_red(self):
        tp = _FakeTP(
            "Long distraction",
            {
                (0, 4): {
                    "Task": "Non-driving task",
                    "Movement type": "Owl",
                    "Gaze location": "A",
                    "Vehicle response": "Forward support",
                    "Color": "GREEN",
                },
                (1, 4): {
                    "Task": "Non-driving task",
                    "Movement type": "Owl",
                    "Gaze location": "B",
                    "Vehicle response": "Forward support",
                    "Color": "GREY",
                },
                (2, 4): {
                    "Task": "Non-driving task",
                    "Movement type": "Owl",
                    "Gaze location": "C",
                    "Vehicle response": "Forward support",
                    "Color": "RED",
                },
            },
        )
        driver_monitoring.apply_prediction_consistency({"Long distraction": tp})
        self.assertEqual(tp.cell_dict[(0, 4)]["Color"], "RED")
        self.assertEqual(tp.cell_dict[(1, 4)]["Color"], "RED")
        self.assertEqual(tp.cell_dict[(2, 4)]["Color"], "RED")

    def test_leaves_green_grey_group_without_red_untouched(self):
        tp = _FakeTP(
            "Long distraction",
            {
                (0, 4): {
                    "Task": "Non-driving task",
                    "Movement type": "Owl",
                    "Gaze location": "A",
                    "Vehicle response": "Forward support",
                    "Color": "GREEN",
                },
                (1, 4): {
                    "Task": "Non-driving task",
                    "Movement type": "Owl",
                    "Gaze location": "B",
                    "Vehicle response": "Forward support",
                    "Color": "GREY",
                },
            },
        )
        driver_monitoring.apply_prediction_consistency({"Long distraction": tp})
        self.assertEqual(tp.cell_dict[(0, 4)]["Color"], "GREEN")
        self.assertEqual(tp.cell_dict[(1, 4)]["Color"], "GREY")


class TestSelectInterventionTestPoints(unittest.TestCase):
    def _make_tp(self, name, entries):
        """entries: list of (task, movement_type, gaze, vehicle_response, color)."""
        cell_dict = {}
        for row_idx, (task, movement_type, gaze, vehicle_response, color) in enumerate(
            entries
        ):
            col_idx = 4 if vehicle_response == driver_monitoring.FORWARD_SUPPORT else 5
            cell_dict[(row_idx, col_idx)] = {
                "Task": task,
                "Movement type": movement_type,
                "Gaze location": gaze,
                "Vehicle response": vehicle_response,
                "Color": color,
            }
        return _FakeTP(name, cell_dict)

    def _selected_groups(self, result):
        """Flatten result into a list of (Task, Movement type) for every selected cell."""
        groups = []
        for cells in result.values():
            for info in cells.values():
                groups.append((info["Task"], info["Movement type"]))
        return groups

    def test_respects_max_total(self):
        entries = []
        for i in range(10):
            entries.append(
                (
                    "Non-driving task",
                    f"M{i}",
                    f"G{i}",
                    driver_monitoring.FORWARD_SUPPORT,
                    "GREEN",
                )
            )
            entries.append(
                (
                    "Non-driving task",
                    f"M{i}",
                    f"G{i}",
                    driver_monitoring.LANE_SUPPORT,
                    "GREEN",
                )
            )
        tp = self._make_tp("Long distraction", entries)
        for seed in range(20):
            random.seed(seed)
            result = driver_monitoring.select_intervention_test_points(
                [tp], max_total=8
            )
            total = sum(len(v) for v in result.values())
            self.assertLessEqual(total, 8)

    def test_reaches_max_total_when_oversupplied(self):
        entries = []
        for i in range(10):
            entries.append(
                (
                    "Non-driving task",
                    f"M{i}",
                    f"G{i}",
                    driver_monitoring.FORWARD_SUPPORT,
                    "GREEN",
                )
            )
            entries.append(
                (
                    "Non-driving task",
                    f"M{i}",
                    f"G{i}",
                    driver_monitoring.LANE_SUPPORT,
                    "GREEN",
                )
            )
        tp = self._make_tp("Long distraction", entries)
        random.seed(1)
        result = driver_monitoring.select_intervention_test_points([tp], max_total=8)
        total = sum(len(v) for v in result.values())
        self.assertEqual(total, 8)

    def test_at_most_one_per_group(self):
        entries = []
        for i in range(5):
            entries.append(
                (
                    "Non-driving task",
                    f"M{i}",
                    f"G{i}",
                    driver_monitoring.FORWARD_SUPPORT,
                    "GREEN",
                )
            )
            entries.append(
                (
                    "Non-driving task",
                    f"M{i}",
                    f"G{i}",
                    driver_monitoring.LANE_SUPPORT,
                    "GREEN",
                )
            )
        tp = self._make_tp("Long distraction", entries)
        for seed in range(20):
            random.seed(seed)
            result = driver_monitoring.select_intervention_test_points(
                [tp], max_total=8
            )
            groups = self._selected_groups(result)
            self.assertEqual(len(groups), len(set(groups)))

    def test_forced_group_uses_only_available_strategy_driving_task(self):
        # Driving task, only Forward support is GREEN for this movement type.
        entries = [
            ("Driving task", "Owl", "G0", driver_monitoring.FORWARD_SUPPORT, "GREEN")
        ]
        tp = self._make_tp("Long distraction", entries)
        for seed in range(20):
            random.seed(seed)
            result = driver_monitoring.select_intervention_test_points(
                [tp], max_total=8
            )
            selected = result["Long distraction"]
            self.assertEqual(len(selected), 1)
            (info,) = selected.values()
            self.assertEqual(
                info["Vehicle response"], driver_monitoring.FORWARD_SUPPORT
            )

    def test_forced_group_uses_only_available_strategy_other_task(self):
        entries = [
            ("Non-driving task", "Owl", "G0", driver_monitoring.LANE_SUPPORT, "GREEN")
        ]
        tp = self._make_tp("Long distraction", entries)
        for seed in range(20):
            random.seed(seed)
            result = driver_monitoring.select_intervention_test_points(
                [tp], max_total=8
            )
            selected = result["Long distraction"]
            self.assertEqual(len(selected), 1)
            (info,) = selected.values()
            self.assertEqual(info["Vehicle response"], driver_monitoring.LANE_SUPPORT)

    def test_pool_reuse_fallback_does_not_crash(self):
        # All groups share the same single gaze location -> pool exhausts immediately.
        entries = []
        for i in range(6):
            entries.append(
                (
                    "Non-driving task",
                    f"M{i}",
                    "SHARED",
                    driver_monitoring.FORWARD_SUPPORT,
                    "GREEN",
                )
            )
            entries.append(
                (
                    "Non-driving task",
                    f"M{i}",
                    "SHARED",
                    driver_monitoring.LANE_SUPPORT,
                    "GREEN",
                )
            )
        tp = self._make_tp("Long distraction", entries)
        for seed in range(20):
            random.seed(seed)
            result = driver_monitoring.select_intervention_test_points(
                [tp], max_total=8
            )
            total = sum(len(v) for v in result.values())
            self.assertLessEqual(total, 8)
            groups = self._selected_groups(result)
            self.assertEqual(len(groups), len(set(groups)))

    def test_balance_within_one_on_symmetric_fixture(self):
        entries = []
        for i in range(6):
            entries.append(
                (
                    "Non-driving task",
                    f"FS{i}",
                    f"G_fs{i}",
                    driver_monitoring.FORWARD_SUPPORT,
                    "GREEN",
                )
            )
        for i in range(6):
            entries.append(
                (
                    "Non-driving task",
                    f"LS{i}",
                    f"G_ls{i}",
                    driver_monitoring.LANE_SUPPORT,
                    "GREEN",
                )
            )
        tp = self._make_tp("Long distraction", entries)
        for seed in range(10):
            random.seed(seed)
            result = driver_monitoring.select_intervention_test_points(
                [tp], max_total=12
            )
            fs_count = sum(
                1
                for cells in result.values()
                for info in cells.values()
                if info["Vehicle response"] == driver_monitoring.FORWARD_SUPPORT
            )
            ls_count = sum(
                1
                for cells in result.values()
                for info in cells.values()
                if info["Vehicle response"] == driver_monitoring.LANE_SUPPORT
            )
            self.assertEqual(fs_count, 6)
            self.assertEqual(ls_count, 6)

    def test_returns_all_processor_keys_even_when_empty(self):
        empty_tp = _FakeTP("Phone use", {})
        entries = [
            ("Driving task", "Owl", "G0", driver_monitoring.FORWARD_SUPPORT, "GREEN")
        ]
        tp = self._make_tp("Long distraction", entries)
        result = driver_monitoring.select_intervention_test_points(
            [tp, empty_tp], max_total=8
        )
        self.assertIn("Long distraction", result)
        self.assertIn("Phone use", result)
        self.assertEqual(result["Phone use"], {})

    def test_no_selection_when_all_red(self):
        entries = [
            ("Driving task", "Owl", "G0", driver_monitoring.FORWARD_SUPPORT, "RED"),
            ("Driving task", "Owl", "G0", driver_monitoring.LANE_SUPPORT, "RED"),
        ]
        tp = self._make_tp("Long distraction", entries)
        result = driver_monitoring.select_intervention_test_points([tp], max_total=8)
        self.assertEqual(result["Long distraction"], {})

    def test_matching_avoids_avoidable_duplicate(self):
        # Reproduces the structural pattern found in the dsm_test fixtures
        # (07_random_red_seed3 / 10_random_red_seed6): two groups whose candidate
        # gaze sets overlap on one value but each also has a distinct alternative,
        # so a fully-distinct assignment is always possible.
        entries = [
            (
                "Non-driving task",
                "Body lean",
                "G1",
                driver_monitoring.LANE_SUPPORT,
                "GREEN",
            ),
            (
                "Non-driving task",
                "Body lean",
                "G2",
                driver_monitoring.LANE_SUPPORT,
                "GREEN",
            ),
            (
                "Non-driving task",
                "Lizard",
                "G2",
                driver_monitoring.LANE_SUPPORT,
                "GREEN",
            ),
            (
                "Non-driving task",
                "Lizard",
                "G3",
                driver_monitoring.LANE_SUPPORT,
                "GREEN",
            ),
        ]
        tp = self._make_tp("Long distraction", entries)
        for seed in range(50):
            random.seed(seed)
            result = driver_monitoring.select_intervention_test_points(
                [tp], max_total=8
            )
            gazes = [
                info["Gaze location"] for info in result["Long distraction"].values()
            ]
            self.assertEqual(len(gazes), 2)
            self.assertEqual(len(gazes), len(set(gazes)))

    def test_matching_maximizes_distinct_count_when_impossible_to_avoid_all_duplicates(
        self,
    ):
        # 3 groups whose combined candidates only span 2 distinct gaze values ->
        # a duplicate is mathematically unavoidable, but only one, not more.
        entries = [
            ("Non-driving task", "M0", "G1", driver_monitoring.LANE_SUPPORT, "GREEN"),
            ("Non-driving task", "M1", "G1", driver_monitoring.LANE_SUPPORT, "GREEN"),
            ("Non-driving task", "M1", "G2", driver_monitoring.LANE_SUPPORT, "GREEN"),
            ("Non-driving task", "M2", "G2", driver_monitoring.LANE_SUPPORT, "GREEN"),
        ]
        tp = self._make_tp("Long distraction", entries)
        for seed in range(50):
            random.seed(seed)
            result = driver_monitoring.select_intervention_test_points(
                [tp], max_total=8
            )
            gazes = [
                info["Gaze location"] for info in result["Long distraction"].values()
            ]
            self.assertEqual(len(gazes), 3)
            self.assertEqual(len(set(gazes)), 2)

    def test_nan_gaze_location_does_not_crash_matching(self):
        # A GREEN candidate with a NaN/blank gaze location (as seen in filled DM
        # prediction sheets) must not be treated as a matchable value and must
        # not raise StopIteration during gaze assignment.
        entries = [
            (
                "Non-driving task",
                "M0",
                float("nan"),
                driver_monitoring.LANE_SUPPORT,
                "GREEN",
            ),
            ("Non-driving task", "M1", "G1", driver_monitoring.LANE_SUPPORT, "GREEN"),
        ]
        tp = self._make_tp("Long distraction", entries)
        for seed in range(20):
            random.seed(seed)
            result = driver_monitoring.select_intervention_test_points(
                [tp], max_total=8
            )
            self.assertEqual(len(result["Long distraction"]), 2)


class TestMaxBipartiteMatching(unittest.TestCase):
    def test_matching_basic(self):
        candidate_sets = [{"a", "b"}, {"a"}, {"b", "c"}]
        for seed in range(20):
            random.seed(seed)
            matching = driver_monitoring._max_bipartite_matching(candidate_sets)
            self.assertEqual(len(matching), 3)
            values = list(matching.values())
            self.assertEqual(len(values), len(set(values)))
            for pick_idx, value in matching.items():
                self.assertIn(value, candidate_sets[pick_idx])

    def test_matching_returns_maximum_when_not_fully_matchable(self):
        candidate_sets = [{"a"}, {"a"}, {"a"}]
        for seed in range(20):
            random.seed(seed)
            matching = driver_monitoring._max_bipartite_matching(candidate_sets)
            self.assertEqual(len(matching), 1)
            (value,) = matching.values()
            self.assertEqual(value, "a")


class TestBuildForwardSupportScenario(unittest.TestCase):
    _PATTERN = re.compile(
        r"^(CCRs|CMRs|CCRm), IL (\d+)%, VUT (\d+) km/h, Target (\d+) km/h$"
    )

    def test_format_and_speed_range(self):
        for _ in range(50):
            scenario = driver_monitoring.build_forward_support_scenario()
            match = self._PATTERN.match(scenario)
            self.assertIsNotNone(match, scenario)
            scenario_type = match.group(1)
            impact_location = int(match.group(2))
            vut_speed = int(match.group(3))
            target_speed = int(match.group(4))
            self.assertIn(
                impact_location, driver_monitoring.FORWARD_SUPPORT_IMPACT_LOCATIONS
            )
            self.assertIn(vut_speed, driver_monitoring.FORWARD_SUPPORT_VUT_SPEEDS)
            self.assertEqual(
                target_speed,
                driver_monitoring.FORWARD_SUPPORT_TARGET_SPEEDS[scenario_type],
            )


class TestSelectDmTestPoints(unittest.TestCase):
    class _FakeTP:
        def __init__(self, name, driver_state, cell_dict, processed_df=None):
            self.name = name
            self.driver_state = driver_state
            self.cell_dict = cell_dict
            self.processed_df = (
                processed_df
                if processed_df is not None
                else pd.DataFrame(
                    columns=[
                        "Long distraction",
                        "Movement type",
                        "Gaze location",
                        "Warning",
                        "Forward support",
                        "Lane support",
                    ]
                )
            )

    def _processors(self):
        return {
            "Long distraction": self._FakeTP("Long distraction", "Transient", {}),
            "Short distraction": self._FakeTP("Short distraction", "Transient", {}),
            "Phone use": self._FakeTP("Phone use", "Transient", {}),
            "Non-transient": self._FakeTP("Non-transient", "Non-transient", {}),
        }

    def test_return_shape(self):
        processors = self._processors()
        with patch.object(
            driver_monitoring,
            "TableProcessor",
            side_effect=lambda _df, name, **_kwargs: processors[name],
        ):
            result = driver_monitoring.select_dm_test_points(pd.DataFrame())
        self.assertEqual(
            set(result.keys()),
            {
                "table_processors",
                "warning_points",
                "intervention_points",
                "non_transient_points",
                "transient_scenarios",
                "non_transient_scenarios",
            },
        )
        self.assertEqual(
            set(result["transient_scenarios"].keys()),
            {"Forward support", "Lane support"},
        )
        self.assertEqual(
            set(result["non_transient_scenarios"].keys()), {"Forward and lane support"}
        )

    def test_uses_input_points_skips_intervention_selection(self):
        processors = self._processors()
        with patch.object(
            driver_monitoring,
            "TableProcessor",
            side_effect=lambda _df, name, **_kwargs: processors[name],
        ):
            result = driver_monitoring.select_dm_test_points(
                pd.DataFrame(), input_selected_points={}
            )
        for name in ["Long distraction", "Short distraction", "Phone use"]:
            self.assertEqual(result["intervention_points"][name], {})

    def test_draws_independent_scenarios(self):
        processors = self._processors()
        with (
            patch.object(
                driver_monitoring,
                "TableProcessor",
                side_effect=lambda _df, name, **_kwargs: processors[name],
            ),
            patch.object(
                driver_monitoring,
                "build_forward_support_scenario",
                side_effect=["SCEN_A", "SCEN_B"],
            ) as mock_build,
        ):
            result = driver_monitoring.select_dm_test_points(pd.DataFrame())
        self.assertEqual(mock_build.call_count, 2)
        self.assertEqual(result["transient_scenarios"]["Forward support"], "SCEN_A")
        self.assertEqual(
            result["non_transient_scenarios"]["Forward and lane support"], "SCEN_B"
        )


class TestPreprocessTestScenarioColumn(unittest.TestCase):
    class _FakeTP:
        def __init__(self, name, driver_state, cell_dict):
            self.name = name
            self.driver_state = driver_state
            self.processed_df = pd.DataFrame()
            self.cell_dict = cell_dict

    @staticmethod
    def _rebuild_verification_df(final_df):
        """Reconstruct the real verification DataFrame the way compute_score does:
        the actual column names live as a DATA row (after the general-requirements
        row and an empty row), not as final_df's own .columns."""
        header_row_idx = 2
        cols = final_df.iloc[header_row_idx].tolist()
        data = final_df.values[header_row_idx + 1 :]
        return pd.DataFrame(data, columns=cols)

    def test_transient_and_non_transient_scenarios_are_independent_and_survive_ffill(
        self,
    ):
        processors = {
            "Long distraction": self._FakeTP(
                "Long distraction",
                "Transient",
                {
                    (0, 4): {
                        "Task": "Driving task",
                        "Movement type": "Owl",
                        "Gaze location": "Road",
                        "Vehicle response": "Forward support",
                        "Color": "GREEN",
                    },
                    (1, 4): {
                        "Task": "Driving task",
                        "Movement type": "Lizard",
                        "Gaze location": "Road",
                        "Vehicle response": "Forward support",
                        "Color": "GREEN",
                    },
                    (2, 5): {
                        "Task": "Driving task",
                        "Movement type": "Owl",
                        "Gaze location": "Mirror",
                        "Vehicle response": "Lane support",
                        "Color": "GREEN",
                    },
                    (3, 3): {
                        "Task": "Non-driving task",
                        "Movement type": "Owl",
                        "Gaze location": "Road",
                        "Vehicle response": "Warning",
                        "Color": "GREEN",
                    },
                },
            ),
            "Short distraction": self._FakeTP("Short distraction", "Transient", {}),
            "Phone use": self._FakeTP("Phone use", "Transient", {}),
            "Non-transient": self._FakeTP(
                "Non-transient",
                "Non-transient",
                {
                    (4, 4): {
                        "Task": "Sleep",
                        "Movement type": "",
                        "Gaze location": "Road",
                        "Vehicle response": "Forward and lane support",
                        "Color": "GREEN",
                    },
                },
            ),
        }
        input_selected_points = {
            "Long distraction": [
                {
                    "Task": "Driving task",
                    "Movement type": "Owl",
                    "Vehicle response": "Forward support",
                },
                {
                    "Task": "Driving task",
                    "Movement type": "Lizard",
                    "Vehicle response": "Forward support",
                },
                {
                    "Task": "Driving task",
                    "Movement type": "Owl",
                    "Vehicle response": "Lane support",
                },
                {
                    "Task": "Non-driving task",
                    "Movement type": "Owl",
                    "Vehicle response": "Warning",
                },
            ],
            "Non-transient": [
                {"Task": "Sleep", "Vehicle response": "Forward and lane support"},
            ],
        }
        noise_variables_df = pd.DataFrame(
            {
                "Element": ["Dark sunglasses", "Face-mask", "Cap", "Hat"],
                "Performance": ["Functional"] * 4,
            }
        )

        with (
            patch.object(
                driver_monitoring,
                "TableProcessor",
                side_effect=lambda _df, name, **_kwargs: processors[name],
            ),
            patch.object(
                driver_monitoring.common,
                "extract_table",
                return_value=noise_variables_df,
            ),
            patch.object(
                driver_monitoring, "get_mandatory_check_dict", return_value={"ok": True}
            ),
            patch.object(
                driver_monitoring,
                "build_forward_support_scenario",
                side_effect=["SCEN_TRANSIENT", "SCEN_NON_TRANSIENT"],
            ),
            patch.object(driver_monitoring, "sample_and_assign_noise_vars"),
            patch.object(driver_monitoring, "assign_noise_vars_non_transient"),
        ):
            final_df, _, _ = driver_monitoring.preprocess(
                dm_prediction_df=pd.DataFrame(),
                param_df=pd.DataFrame(),
                input_selected_points=input_selected_points,
            )

        real_df = self._rebuild_verification_df(final_df)

        fs_rows = real_df[real_df["Requirement"] == "Forward support"]
        self.assertEqual(len(fs_rows), 2)
        self.assertTrue((fs_rows["Test scenario"] == "SCEN_TRANSIENT").all())

        ls_rows = real_df[real_df["Requirement"] == "Lane support"]
        self.assertEqual(len(ls_rows), 1)
        self.assertTrue(
            (ls_rows["Test scenario"] == driver_monitoring.LANE_SUPPORT_SCENARIO).all()
        )

        warning_rows = real_df[real_df["Requirement"] == "Warning"]
        self.assertEqual(len(warning_rows), 1)
        self.assertTrue(warning_rows["Test scenario"].isna().all())

        fl_rows = real_df[real_df["Requirement"] == "Forward and lane support"]
        self.assertEqual(len(fl_rows), 1)
        self.assertTrue((fl_rows["Test scenario"] == "SCEN_NON_TRANSIENT").all())


class TestMandatoryNoiseVariables(unittest.TestCase):
    """Mandatory noise variable list correctness and preprocess() message behavior."""

    _REAL_MANDATORY_ELEMENTS = [
        "Daytime - Night-time",
        "Clear sunglasses",
        "Short facial hair",
        "Long facial hair",
        "Face-mask",
    ]

    def _functional_noise_variables_df(
        self, non_functional_elements=None, non_functional_categories=None
    ):
        non_functional_elements = non_functional_elements or []
        non_functional_categories = non_functional_categories or []
        rows = [
            {
                "Element": element,
                "Category": "",
                "Performance": (
                    "Non functional"
                    if element in non_functional_elements
                    else "Functional"
                ),
            }
            for element in self._REAL_MANDATORY_ELEMENTS
        ]
        rows.extend(
            {
                "Element": "",
                "Category": category,
                "Performance": (
                    "Non functional"
                    if category in non_functional_categories
                    else "Functional"
                ),
            }
            for category in [
                "Talking",
                "Long head hair fringe obscuring eyes",
                "Thick eyelash makeup",
            ]
        )
        return pd.DataFrame(rows)

    def test_mandatory_lists_content(self):
        self.assertEqual(
            driver_monitoring.MANDATORY_FUNCTIONAL_ELEMENTS,
            self._REAL_MANDATORY_ELEMENTS,
        )
        self.assertEqual(driver_monitoring.MANDATORY_FUNCTIONAL_CATEGORIES, [])

    def test_get_mandatory_check_dict_ignores_removed_elements(self):
        noise_variables_df = self._functional_noise_variables_df(
            non_functional_elements=[
                "Long head hair fringe obscuring eyes",
                "Thick eyelash makeup",
            ],
            non_functional_categories=["Talking"],
        )
        result = driver_monitoring.get_mandatory_check_dict(noise_variables_df)

        self.assertTrue(all(result.values()))
        self.assertNotIn("Long head hair fringe obscuring eyes", result)
        self.assertNotIn("Thick eyelash makeup", result)
        self.assertNotIn("Talking", result)

    def test_get_mandatory_check_dict_fails_on_real_mandatory_element(self):
        noise_variables_df = self._functional_noise_variables_df(
            non_functional_elements=["Face-mask"],
        )
        result = driver_monitoring.get_mandatory_check_dict(noise_variables_df)

        self.assertFalse(result["Face-mask"])
        self.assertFalse(all(result.values()))

    def test_preprocess_returns_message_df_for_single_failed_mandatory_element(self):
        noise_variables_df = self._functional_noise_variables_df(
            non_functional_elements=["Face-mask"],
        )
        with patch.object(
            driver_monitoring.common, "extract_table", return_value=noise_variables_df
        ):
            verification_df, selected_test_points, _ = driver_monitoring.preprocess(
                dm_prediction_df=pd.DataFrame(), param_df=pd.DataFrame()
            )

        self.assertEqual(selected_test_points, [])
        self.assertEqual(list(verification_df.columns), ["Category"])
        self.assertEqual(len(verification_df), 1)
        self.assertEqual(
            verification_df["Category"].iloc[0],
            "Mandatory noise variable Face-mask was set to Non functional. "
            "No verification tests should be performed since the score will be 0.",
        )

    def test_preprocess_returns_one_message_row_per_failed_mandatory_element(self):
        noise_variables_df = self._functional_noise_variables_df(
            non_functional_elements=["Face-mask", "Clear sunglasses"],
        )
        with patch.object(
            driver_monitoring.common, "extract_table", return_value=noise_variables_df
        ):
            verification_df, selected_test_points, _ = driver_monitoring.preprocess(
                dm_prediction_df=pd.DataFrame(), param_df=pd.DataFrame()
            )

        self.assertEqual(selected_test_points, [])
        self.assertEqual(len(verification_df), 2)
        expected_messages = {
            "Mandatory noise variable Face-mask was set to Non functional. "
            "No verification tests should be performed since the score will be 0.",
            "Mandatory noise variable Clear sunglasses was set to Non functional. "
            "No verification tests should be performed since the score will be 0.",
        }
        self.assertEqual(set(verification_df["Category"]), expected_messages)

    def test_preprocess_does_not_early_return_for_removed_elements(self):
        class FakeTableProcessor:
            def __init__(self, name, driver_state, cell_dict):
                self.name = name
                self.driver_state = driver_state
                self.processed_df = pd.DataFrame()
                self.cell_dict = cell_dict

            def deepcopy(self):
                return self

        processors = {
            "Long distraction": FakeTableProcessor(
                "Long distraction",
                "Transient",
                {
                    (0, 3): {
                        "Task": "Non-driving task",
                        "Movement type": "Owl",
                        "Gaze location": "Road",
                        "Vehicle response": "Warning",
                        "Color": "GREEN",
                    },
                },
            ),
            "Short distraction": FakeTableProcessor(
                "Short distraction", "Transient", {}
            ),
            "Phone use": FakeTableProcessor("Phone use", "Transient", {}),
            "Non-transient": FakeTableProcessor("Non-transient", "Non-transient", {}),
        }

        noise_variables_df = self._functional_noise_variables_df(
            non_functional_elements=[
                "Long head hair fringe obscuring eyes",
                "Thick eyelash makeup",
            ],
            non_functional_categories=["Talking"],
        )

        input_selected_points = {
            "Long distraction": [
                {
                    "Task": "Non-driving task",
                    "Movement type": "Owl",
                    "Vehicle response": "Warning",
                },
            ],
        }

        with (
            patch.object(
                driver_monitoring,
                "TableProcessor",
                side_effect=lambda _df, name, **_kwargs: processors[name],
            ),
            patch.object(
                driver_monitoring.common,
                "extract_table",
                return_value=noise_variables_df,
            ),
            patch.object(driver_monitoring, "select_warning_test_points"),
            patch.object(driver_monitoring, "select_intervention_test_points"),
            patch.object(
                driver_monitoring, "get_non_transient_points", return_value={}
            ),
            patch.object(
                driver_monitoring,
                "build_forward_support_scenario",
                return_value="SCEN_TEST",
            ),
            patch.object(driver_monitoring, "sample_and_assign_noise_vars"),
            patch.object(driver_monitoring, "assign_noise_vars_non_transient"),
        ):
            verification_df, selected_test_points, _ = driver_monitoring.preprocess(
                dm_prediction_df=pd.DataFrame(),
                param_df=pd.DataFrame(),
                input_selected_points=input_selected_points,
            )

        self.assertEqual(len(selected_test_points), 1)
        self.assertFalse(
            verification_df.astype(str)
            .apply(lambda col: col.str.contains("Mandatory noise variable"))
            .any()
            .any()
        )


if __name__ == "__main__":
    unittest.main()
