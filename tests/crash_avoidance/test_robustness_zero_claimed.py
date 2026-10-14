# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Robustness layer score when the OEM claims no robustness layers.

The constant part counts the claimed layers that were not tested
(n_claimed - 1). With nothing claimed that went to -1, so a failed layer
(total = constant part) gave a negative Robustness score. With nothing
claimed the score must be 0 whether the tested layer passes or fails.
"""

import unittest

import pandas as pd

from euroncap_rating_2026 import common
from euroncap_rating_2026.crash_avoidance import (
    data_model,
    matrix_processing,
    test_info,
)

LOADCASE = "CCRm"
N_APPLICABLE = 7
STANDARD_TOTAL = data_model.TOTAL_SCORES[LOADCASE]["Standard"]
ROBUSTNESS_TOTAL = data_model.TOTAL_SCORES[LOADCASE]["Robustness"]


def robustness_df(n_claimed):
    claims = ["YES"] * n_claimed + ["NO"] * (N_APPLICABLE - n_claimed)
    return pd.DataFrame(
        {
            "Robustness layer": [f"Layer {i}" for i in range(N_APPLICABLE)],
            LOADCASE: claims,
        }
    )


def correct_standard_point():
    test_point = matrix_processing.TestPoint(
        row=0,
        col=0,
        color=common.PredictionColor.GREEN,
        test_range=matrix_processing.TestRange.STANDARD,
        attributes={"VUT speed": "50 km/h"},
    )
    return matrix_processing.ComputedTestPoint(
        test_point, test_name=LOADCASE, value=0.0
    )


def robustness_score(n_claimed, layer_failed):
    layer_score = test_info.compute_robustness_layer_score(
        robustness_df(n_claimed), LOADCASE, LOADCASE
    )
    # A correct Standard point keeps the Standard score at its full total, so
    # the 50% Standard gate never zeroes the robustness score here; a failure
    # comes only from has_robustness_failure.
    result = test_info.compute_loadcase_score(
        LOADCASE,
        LOADCASE,
        [correct_standard_point()],
        matrix_processing.PredictionSource.VTA,
        matrix_processing.PredictionSource.VTA,
        STANDARD_TOTAL,
        0.0,
        layer_score,
        has_robustness_failure=layer_failed,
    )
    assert result.standard_score == STANDARD_TOTAL
    return result.robustness_layer_score


class TestZeroClaimedRobustnessLayers(unittest.TestCase):
    def test_parts_are_zero(self):
        layer_score = test_info.compute_robustness_layer_score(
            robustness_df(0), LOADCASE, LOADCASE
        )
        self.assertEqual(layer_score.tested_score, 0.0)
        self.assertEqual(layer_score.constant_score, 0.0)
        self.assertEqual(layer_score.total_score, 0.0)

    def test_passing_layer_scores_zero(self):
        self.assertEqual(robustness_score(0, layer_failed=False), 0.0)

    def test_failing_layer_scores_zero(self):
        self.assertEqual(robustness_score(0, layer_failed=True), 0.0)


class TestClaimedRobustnessLayersUnchanged(unittest.TestCase):
    def test_one_claimed_passing(self):
        self.assertAlmostEqual(
            robustness_score(1, layer_failed=False), ROBUSTNESS_TOTAL / N_APPLICABLE
        )

    def test_one_claimed_failing(self):
        self.assertEqual(robustness_score(1, layer_failed=True), 0.0)

    def test_three_claimed_failing_keeps_constant_part(self):
        self.assertAlmostEqual(
            robustness_score(3, layer_failed=True), ROBUSTNESS_TOTAL / N_APPLICABLE * 2
        )


if __name__ == "__main__":
    unittest.main()
