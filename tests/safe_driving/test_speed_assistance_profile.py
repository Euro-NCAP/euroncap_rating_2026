# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""speed_assistance.SpeedAssistanceProfile as an extension point.

test_speed_assistance.py covers the built-in Euro NCAP profile. These tests
register a small synthetic profile for a second protocol and cover what a
profile can change: the prediction-grid geometry, the all-rows pass rule,
independently scored implicit rows, the absence of coupling rules, and a
sub-cap applied at Category Scores level. They also pin that the Euro NCAP
profile is unchanged by the refactor."""

import unittest

import numpy as np
import pandas as pd

from euroncap_rating_2026 import common
from euroncap_rating_2026.safe_driving import speed_assistance as sa

TEST_PROTOCOL = "test-26-28"
ZONE_HEADERS = ("Zone A", "Zone B", "Zone C")
ZONE_SCENARIOS = tuple(f"{sa.CONDITIONAL_SPEED_LIMIT_STR} - {h}" for h in ZONE_HEADERS)
ZONE_CAP = 0.2
TEST_MAPPING = {
    "Rain / Wetness": sa.CONDITIONAL_SPEED_LIMIT_STR,
    **{h: sa.CONDITIONAL_SPEED_LIMIT_STR for h in ZONE_HEADERS},
    "Highway / Motorway": sa.IMPLICIT_SPEED_LIMIT_STR,
    "Residential zones": sa.IMPLICIT_SPEED_LIMIT_STR,
    "Dynamic speed signs": sa.DYNAMIC_SPEED_LIMIT_STR,
}
TEST_POINTS = {
    "Conditional speed limits - Rain / Wetness": 0.4,
    **{s: 0.1 for s in ZONE_SCENARIOS},
    "Implicit speed limits - Highway / Motorway": 0.3,
    "Implicit speed limits - Residential zones": 0.3,
    "Dynamic speed limits - Dynamic speed signs": 0.4,
}
HEADERS = list(TEST_MAPPING)
MARKETS = ["Market 1", "Market 2"]
N_COLS = 2 + len(HEADERS)
TEST_PROFILE = sa.SpeedAssistanceProfile(
    protocol=TEST_PROTOCOL,
    slif_adv_mapping=TEST_MAPPING,
    advanced_points=TEST_POINTS,
    collapse_implicit=False,
    implicit_collapsed_scenario=None,
    coupling_rules=(),
    sub_caps=((frozenset(ZONE_SCENARIOS), ZONE_CAP),),
    adv_table_n_rows=3,
    adv_table_n_cols=N_COLS,
    hazard_table_start_row=4,
    expected_row_count=2,
    group_label="test markets",
    expected_group_row_count=2,
    pass_rule=sa.SlifAdvancedPassRule.ALL_ROWS,
)
# Column order in the verification sheet is the profile's scenario order.
ENCAP_CSV_LITERAL = """
Category;Scenario;Value
SLIF - General requirements;;
SLIF - Accuracy;Distance based KPI;
SLIF - Accuracy;Event based KPI;
SLIF - Advanced speed limit;Conditional speed limits - Rain / Wetness;FAIL
SLIF - Advanced speed limit;Conditional speed limits - Snow / Icy;FAIL
SLIF - Advanced speed limit;Conditional speed limits - Time / Season;FAIL
SLIF - Advanced speed limit;Conditional speed limits - Distance for / Distance in;FAIL
SLIF - Advanced speed limit;Conditional speed limits - Non-lane relevant arrows;FAIL
SLIF - Advanced speed limit;Conditional speed limits - Lane relevant arrows;FAIL
SLIF - Advanced speed limit;Conditional speed limits - Vehicle categories;FAIL
SLIF - Advanced speed limit;Implicit speed limits - Highway, city and residential zones;FAIL
SLIF - Advanced speed limit;Dynamic speed limits - Non-lane relevant;FAIL
SLIF - Advanced speed limit;Dynamic speed limits - Lane relevant;FAIL
SLIF - Local hazards;Construction zones - Sending;FAIL
SLIF - Local hazards;Construction zones - Receiving & informing;FAIL
SLIF - Local hazards;Items on road - Sending;FAIL
SLIF - Local hazards;Items on road - Receiving & informing;FAIL
SLIF - Local hazards;Stopped vehicle - Sending;FAIL
SLIF - Local hazards;Stopped vehicle - Receiving & informing;FAIL
SLIF - Local hazards;Broken down vehicle - Sending;FAIL
SLIF - Local hazards;Broken down vehicle - Receiving & informing;FAIL
SLIF - Local hazards;Post crash - Sending;FAIL
SLIF - Local hazards;Post crash - Receiving & informing;FAIL
SLIF - Local hazards;Poor weather - Sending;FAIL
SLIF - Local hazards;Poor weather - Receiving & informing;FAIL
SLIF - Local hazards;Poor road - Sending;FAIL
SLIF - Local hazards;Poor road - Receiving & informing;FAIL
SLIF - Local hazards;Wrong way driver - Sending;FAIL
SLIF - Local hazards;Wrong way driver - Receiving & informing;FAIL
SLIF - Local hazards;Amber & blue lights - Receiving & informing;FAIL
SLIF - Local hazards;Traffic jam - Receiving & informing;FAIL
Speed control function;Speedometer accuracy;≥-3 km/h
"""


def register_test_profile():
    common.register_protocol(
        common.ProtocolSpec(TEST_PROTOCOL, "Test NCAP", "Test NCAP Ltd")
    )
    sa.register_profile(TEST_PROFILE)


def unregister_test_profile():
    sa.PROFILES.pop(TEST_PROTOCOL, None)
    common.PROTOCOLS.pop(TEST_PROTOCOL, None)


def make_prediction_df(scenario_values=None, hazard_values=None):
    """'VA - Speed assist. pred.' in the test profile's layout as
    read_excel_file_to_dfs would produce it (sheet row 1 is the header)."""
    scenario_values = scenario_values or {}
    hazard_values = hazard_values or {}
    rows = [[None, None] + HEADERS]
    for i, market in enumerate(MARKETS):
        group = "Test markets" if i == 0 else None
        rows.append(
            [group, market]
            + [scenario_values.get(h, ["Green", "Green"])[i] for h in HEADERS]
        )
    rows.append([None] * N_COLS)  # blank buffer row
    hazard_header = ["Element", "Sub"] + sa.LOCAL_HAZARD_COLUMNS
    rows.append(hazard_header + [None] * (N_COLS - len(hazard_header)))
    for row_idx in range(4):
        row = ["Channel", "Direction"]
        for hazard in sa.LOCAL_HAZARD_COLUMNS:
            values = hazard_values.get(hazard)
            row.append(values[row_idx] if values else "Green")
        rows.append(row + [None] * (N_COLS - len(row)))
    return pd.DataFrame(rows)


def make_version_df(protocol=TEST_PROTOCOL):
    return pd.DataFrame(
        [[None, None], [None, None], [common.PROTOCOL_LABEL, protocol]],
        columns=["Template version", "5.4.7"],
    )


def make_scenario_scores_df():
    rows = [("SLIF - Accuracy", "Distance based"), (None, "Event based")]
    for i, scenario in enumerate(TEST_PROFILE.advanced_scenarios):
        rows.append(
            (sa.SLIF_ADVANCED_SPEED_LIMIT_CATEGORY if i == 0 else None, scenario)
        )
    for i, scenario in enumerate(sa.SLIF_LOCAL_HAZARD_VERIFICATION_ROWS):
        rows.append(("SLIF - Local hazards" if i == 0 else None, scenario))
    rows.append(("SLIF - System updates", None))
    rows.append(("Speed control function", "Speedometer accuracy"))
    rows.append((None, None))
    return pd.DataFrame(
        {
            "Category": [r[0] for r in rows],
            "Scenario": [r[1] for r in rows],
            "Score": [np.nan] * len(rows),
        }
    )


def make_dfs(scenario_values=None, hazard_values=None, protocol=TEST_PROTOCOL):
    return {
        "VA - Speed assist. pred.": make_prediction_df(scenario_values, hazard_values),
        "Input parameters": pd.DataFrame(
            {
                "Category": ["Speed control function", "SLIF - System updates"],
                "Input parameter": ["Type of system", "System updates type"],
                "Value": ["iacc", "continuous"],
            }
        ),
        "Scenario Scores": make_scenario_scores_df(),
        "Version": make_version_df(protocol),
    }


def value_for(verification_df, scenario):
    return verification_df.loc[verification_df["Scenario"] == scenario, "Value"].iloc[0]


class ProfileTestCase(unittest.TestCase):
    def setUp(self):
        register_test_profile()

    def tearDown(self):
        unregister_test_profile()


class TestRegistry(ProfileTestCase):
    def test_marker_selects_the_registered_profile(self):
        self.assertIs(sa.resolve_profile(None, make_dfs()), TEST_PROFILE)
        self.assertIs(sa.get_profile(TEST_PROTOCOL), TEST_PROFILE)
        self.assertIs(sa.resolve_profile(None, {}), sa.ENCAP_26_28_PROFILE)
        self.assertIs(
            sa.resolve_profile("encap-23-25", make_dfs()), sa.ENCAP_26_28_PROFILE
        )

    def test_identical_registration_is_a_noop_and_conflict_raises(self):
        sa.register_profile(TEST_PROFILE)
        other = sa.SpeedAssistanceProfile(
            **{**TEST_PROFILE.__dict__, "group_label": "different"}
        )
        with self.assertRaises(ValueError):
            sa.register_profile(other)
        self.assertIs(sa.PROFILES[TEST_PROTOCOL], TEST_PROFILE)

    def test_profile_referencing_unknown_scenarios_is_rejected(self):
        bad = sa.SpeedAssistanceProfile(
            **{
                **TEST_PROFILE.__dict__,
                "protocol": TEST_PROTOCOL,
                "sub_caps": ((frozenset({"Conditional speed limits - Nope"}), 0.1),),
            }
        )
        sa.PROFILES.pop(TEST_PROTOCOL)
        with self.assertRaises(ValueError):
            sa.register_profile(bad)

    def test_points_and_mapping_must_name_the_same_scenarios(self):
        """preprocess fills "<category> - <header>" rows; compute_score awards
        advanced_points by those names. A key preprocess never fills would
        silently score 0, so registration refuses the mismatch."""
        sa.PROFILES.pop(TEST_PROTOCOL)
        typo_points = dict(TEST_POINTS)
        typo_points["Dynamic speed limits - Dynamic speed sign"] = typo_points.pop(
            "Dynamic speed limits - Dynamic speed signs"
        )
        typo = sa.SpeedAssistanceProfile(
            **{**TEST_PROFILE.__dict__, "advanced_points": typo_points}
        )
        with self.assertRaises(ValueError) as ctx:
            sa.register_profile(typo)
        self.assertIn("Dynamic speed sign'", str(ctx.exception))
        self.assertIn("Dynamic speed signs'", str(ctx.exception))
        extra_column = sa.SpeedAssistanceProfile(
            **{
                **TEST_PROFILE.__dict__,
                "slif_adv_mapping": {
                    **TEST_MAPPING,
                    "Snow / Icy": sa.CONDITIONAL_SPEED_LIMIT_STR,
                },
            }
        )
        with self.assertRaises(ValueError):
            sa.register_profile(extra_column)
        self.assertNotIn(TEST_PROTOCOL, sa.PROFILES)

    def test_collapsing_profile_needs_an_implicit_column(self):
        sa.PROFILES.pop(TEST_PROTOCOL)
        collapsed = "Implicit speed limits - all"
        points = {k: v for k, v in TEST_POINTS.items() if not k.startswith("Implicit")}
        points[collapsed] = 0.5
        mapping = {
            k: v for k, v in TEST_MAPPING.items() if v != sa.IMPLICIT_SPEED_LIMIT_STR
        }
        no_implicit = sa.SpeedAssistanceProfile(
            **{
                **TEST_PROFILE.__dict__,
                "collapse_implicit": True,
                "implicit_collapsed_scenario": collapsed,
                "advanced_points": points,
                "slif_adv_mapping": mapping,
            }
        )
        with self.assertRaises(ValueError) as ctx:
            sa.register_profile(no_implicit)
        self.assertIn("maps no implicit column", str(ctx.exception))
        # With the implicit columns kept, the collapsed row is the one scored.
        with_implicit = sa.SpeedAssistanceProfile(
            **{**no_implicit.__dict__, "slif_adv_mapping": TEST_MAPPING}
        )
        self.assertIs(sa.register_profile(with_implicit), with_implicit)

    def test_built_in_euro_ncap_profile_passes_registration_checks(self):
        self.assertIs(
            sa.register_profile(sa.ENCAP_26_28_PROFILE), sa.ENCAP_26_28_PROFILE
        )

    def test_profile_for_unregistered_protocol_is_rejected(self):
        common.PROTOCOLS.pop(TEST_PROTOCOL)
        sa.PROFILES.pop(TEST_PROTOCOL)
        with self.assertRaises(ValueError):
            sa.register_profile(TEST_PROFILE)

    def test_registered_protocol_without_profile_raises_clearly(self):
        sa.PROFILES.pop(TEST_PROTOCOL)
        with self.assertRaises(ValueError) as ctx:
            sa.preprocess(make_dfs())
        self.assertIn("No Speed Assistance profile", str(ctx.exception))

    def test_grid_of_another_protocol_with_euro_ncap_profile_raises(self):
        with self.assertRaises(ValueError):
            sa.preprocess(make_dfs(), protocol="encap-26-28")


class TestExplicitProtocolThroughEntryPoints(ProfileTestCase):
    """The DataFrame entry points a consuming application calls take an
    explicit ``protocol``, so it can pass its own package id instead of
    relying on the workbook marker (which stays the fallback)."""

    def test_preprocess_stage_subelement_forwards_the_protocol(self):
        from euroncap_rating_2026.safe_driving import test_info

        # No Version sheet at all: only the explicit protocol can select
        # the profile here.
        dfs = make_dfs()
        del dfs["Version"]
        test_info.preprocess_stage_subelement(
            dfs,
            "Vehicle Assistance",
            "Speed Assistance",
            protocol=TEST_PROTOCOL,
        )
        verification_df = dfs["VA - Speed assist. verif."]
        asl = verification_df[
            verification_df["Category"].ffill() == sa.SLIF_ADVANCED_SPEED_LIMIT_CATEGORY
        ]
        self.assertEqual(tuple(asl["Scenario"]), TEST_PROFILE.advanced_scenarios)

    def test_marker_is_still_the_fallback(self):
        from euroncap_rating_2026.safe_driving import test_info

        dfs = make_dfs()  # carries the marker, no explicit protocol
        test_info.preprocess_stage_subelement(
            dfs, "Vehicle Assistance", "Speed Assistance"
        )
        verification_df = dfs["VA - Speed assist. verif."]
        asl = verification_df[
            verification_df["Category"].ffill() == sa.SLIF_ADVANCED_SPEED_LIMIT_CATEGORY
        ]
        self.assertEqual(tuple(asl["Scenario"]), TEST_PROFILE.advanced_scenarios)

    def test_explicit_protocol_beats_a_contradicting_marker(self):
        from euroncap_rating_2026.safe_driving import test_info

        # Marker says the test protocol, the caller says Euro NCAP: the
        # Euro NCAP profile is used, and its geometry rejects this grid.
        with self.assertRaises(ValueError):
            test_info.preprocess_stage_subelement(
                make_dfs(),
                "Vehicle Assistance",
                "Speed Assistance",
                protocol="encap-26-28",
            )

    def test_entry_points_accept_the_protocol_keyword(self):
        import importlib
        import inspect

        from euroncap_rating_2026.safe_driving import integrity, test_info

        # Imported by module path: the package re-exports the click commands
        # under the same names as these modules.
        sd_compute_score = importlib.import_module(
            "euroncap_rating_2026.safe_driving.compute_score"
        )
        sd_pre = importlib.import_module("euroncap_rating_2026.safe_driving.preprocess")

        for func in (
            sa.preprocess,
            sa.compute_score,
            integrity.rebuild_trusted_input,
            integrity.rebuild_trusted_input_dfs,
            test_info.preprocess_stage_subelement,
            sd_compute_score.get_updated_dfs,
            sd_compute_score.calculate_score,
            sd_pre.add_verification_sheets_to_dfs,
        ):
            with self.subTest(func=func.__name__):
                parameter = inspect.signature(func).parameters.get("protocol")
                self.assertIsNotNone(parameter, f"{func.__name__} takes no protocol")
                self.assertIsNone(
                    parameter.default,
                    f"{func.__name__}'s protocol must default to None",
                )


class TestEuroNcapProfileUnchanged(unittest.TestCase):
    def test_aliases_and_verification_csv(self):
        self.assertEqual(sa.SLIF_ADV_MAPPING, sa.ENCAP_26_28_PROFILE.slif_adv_mapping)
        self.assertEqual(sa.EXPECTED_APPLICATION_AREA_ROW_COUNT, 29)
        self.assertEqual(sa.EXPECTED_MEMBER_STATE_ROW_COUNT, 10)
        self.assertEqual(sa.EURO_NCAP_MEMBER_STATES_LABEL, "euro ncap member states")
        self.assertEqual(sa.ENCAP_26_28_PROFILE.sub_caps, ())
        self.assertEqual(sa.SLIF_SCORING_CSV_STRING, ENCAP_CSV_LITERAL)
        self.assertEqual(
            sa.SLIF_SCORING_CSV_STRING, sa.ENCAP_26_28_PROFILE.verification_csv
        )
        self.assertEqual(len(sa.ENCAP_ADVANCED_SPEED_LIMIT_SCENARIOS), 10)
        self.assertAlmostEqual(sum(sa.ENCAP_ADVANCED_SPEED_LIMIT_POINTS.values()), 3.0)
        self.assertIs(sa.PROFILES["encap-26-28"], sa.ENCAP_26_28_PROFILE)


class TestPreprocessGrid(ProfileTestCase):
    def _verif(self, scenario_values=None, bg=None):
        dfs = make_dfs(scenario_values)
        if bg is not None:
            dfs["VA - Speed assist. pred. (bg)"] = bg
        verification_df, _ = sa.preprocess(dfs)
        return verification_df

    def test_two_row_grid_is_read_with_the_profile_geometry(self):
        verification_df = self._verif()
        asl = verification_df[
            verification_df["Category"].ffill() == sa.SLIF_ADVANCED_SPEED_LIMIT_CATEGORY
        ]
        self.assertEqual(tuple(asl["Scenario"]), TEST_PROFILE.advanced_scenarios)
        self.assertTrue((asl["Value"] == "PASS").all())
        hazards = verification_df[
            verification_df["Category"].ffill() == "SLIF - Local hazards"
        ]
        self.assertEqual(len(hazards), 18)
        self.assertTrue((hazards["Value"] == "PASS").all())

    def test_any_red_row_fails_the_scenario(self):
        for values in (["Red", "Green"], ["Green", "Red"]):
            verification_df = self._verif({"Rain / Wetness": values})
            self.assertEqual(
                value_for(verification_df, "Conditional speed limits - Rain / Wetness"),
                "FAIL",
            )

    def test_unknown_text_row_fails_the_scenario(self):
        verification_df = self._verif({"Rain / Wetness": ["Green", "Amber"]})
        self.assertEqual(
            value_for(verification_df, "Conditional speed limits - Rain / Wetness"),
            "FAIL",
        )

    def test_blank_row_is_not_applicable_and_passes(self):
        verification_df = self._verif({"Zone B": ["Green", None]})
        self.assertEqual(
            value_for(verification_df, "Conditional speed limits - Zone B"), "PASS"
        )

    def test_grey_filled_blank_row_fails(self):
        bg = pd.DataFrame([[None] * N_COLS for _ in range(3)])
        bg.iloc[1, HEADERS.index("Dynamic speed signs") + 2] = (
            common.PredictionColor.GREY
        )
        verification_df = self._verif({"Dynamic speed signs": [None, "Green"]}, bg)
        self.assertEqual(
            value_for(verification_df, "Dynamic speed limits - Dynamic speed signs"),
            "FAIL",
        )

    def test_implicit_rows_are_independent_when_not_collapsed(self):
        verification_df = self._verif({"Residential zones": ["Red", "Red"]})
        self.assertEqual(
            value_for(verification_df, "Implicit speed limits - Residential zones"),
            "FAIL",
        )
        self.assertEqual(
            value_for(verification_df, "Implicit speed limits - Highway / Motorway"),
            "PASS",
        )
        self.assertNotIn(
            sa.ENCAP_IMPLICIT_COLLAPSED_SCENARIO, set(verification_df["Scenario"])
        )

    def test_local_hazards_block_is_read_from_the_profile_position(self):
        dfs = make_dfs(hazard_values={"Post crash": ["Red", "Green", "Red", "Green"]})
        verification_df, _ = sa.preprocess(dfs)
        self.assertEqual(value_for(verification_df, "Post crash - Sending"), "FAIL")
        self.assertEqual(
            value_for(verification_df, "Post crash - Receiving & informing"), "PASS"
        )

    def test_group_label_is_matched_case_insensitively(self):
        sa.PROFILES.pop(TEST_PROTOCOL)
        natural_case = sa.SpeedAssistanceProfile(
            **{**TEST_PROFILE.__dict__, "group_label": "Test Markets"}
        )
        sa.register_profile(natural_case)
        verification_df, _ = sa.preprocess(make_dfs())
        self.assertEqual(
            value_for(verification_df, "Conditional speed limits - Rain / Wetness"),
            "PASS",
        )

    def test_wrong_group_label_raises(self):
        dfs = make_dfs()
        dfs["VA - Speed assist. pred."].iloc[1, 0] = "Euro NCAP member states"
        with self.assertRaises(ValueError):
            sa.preprocess(dfs)


class TestComputeScoreAndSubCap(ProfileTestCase):
    def _score(self, scenario_values=None):
        dfs = make_dfs(scenario_values)
        verification_df, scenario_scores_df = sa.preprocess(dfs)
        for scenario in ("Distance based KPI", "Event based KPI"):
            verification_df.loc[verification_df["Scenario"] == scenario, "Value"] = (
                "PASS"
            )
        verification_df.loc[
            verification_df["Category"] == "SLIF - General requirements", "Value"
        ] = "PASS"
        verification_df.loc[
            verification_df["Scenario"] == "Intelligent adaptive cruise control",
            "Value",
        ] = "PASS"
        dfs["VA - Speed assist. verif."] = verification_df
        dfs["Scenario Scores"] = scenario_scores_df
        _, scenario_scores_df, score_dict, hazard_cap = sa.compute_score(dfs)
        scenario_scores_df = common.update_scoring_sheet(
            scenario_scores_df, score_dict, key_column="Scenario"
        )
        category_scores_df = pd.DataFrame(
            {
                "Category": [
                    sa.SLIF_ADVANCED_SPEED_LIMIT_CATEGORY,
                    "SLIF - Local hazards",
                ],
                "Score": [np.nan, np.nan],
            }
        )
        category_scores_df = common.update_score_sum_interval(
            category_scores_df,
            scenario_scores_df,
            key_column="Category",
            interval_column="Category",
        )
        raw = category_scores_df.loc[0, "Score"]
        category_scores_df = sa.apply_advanced_speed_limit_sub_caps(
            category_scores_df, scenario_scores_df, dfs=dfs
        )
        return score_dict, raw, category_scores_df.loc[0, "Score"], hazard_cap

    def test_all_green_raw_points_exceed_the_capped_category(self):
        score_dict, raw, capped, hazard_cap = self._score()
        # 0.4 + 3 x 0.1 + 0.3 + 0.3 + 0.4 = 1.7 raw; the zone group counts 0.2.
        self.assertAlmostEqual(raw, 1.7)
        self.assertAlmostEqual(capped, 1.6)
        self.assertEqual(hazard_cap, 3.0)
        for scenario in ZONE_SCENARIOS:
            self.assertAlmostEqual(score_dict[scenario], 0.1)

    def test_capped_group_at_the_cap_loses_nothing(self):
        _, raw, capped, _ = self._score({"Zone C": ["Red", "Green"]})
        self.assertAlmostEqual(raw, 1.6)
        self.assertAlmostEqual(capped, 1.6)

    def test_below_the_cap_every_lost_scenario_counts(self):
        _, raw, capped, _ = self._score(
            {"Zone B": ["Red", "Green"], "Zone C": ["Green", "Red"]}
        )
        self.assertAlmostEqual(raw, 1.5)
        self.assertAlmostEqual(capped, 1.5)

    def test_scenarios_outside_the_group_are_not_capped(self):
        score_dict, _, capped, _ = self._score({"Residential zones": ["Red", "Green"]})
        self.assertAlmostEqual(capped, 1.3)
        self.assertEqual(score_dict["Implicit speed limits - Residential zones"], 0.0)
        self.assertAlmostEqual(
            score_dict["Implicit speed limits - Highway / Motorway"], 0.3
        )

    def test_score_dict_carries_only_the_profiles_scenarios(self):
        score_dict, *_ = self._score()
        asl_keys = {
            k
            for k in score_dict
            if k.split(" - ")[0]
            in (
                sa.CONDITIONAL_SPEED_LIMIT_STR,
                sa.IMPLICIT_SPEED_LIMIT_STR,
                sa.DYNAMIC_SPEED_LIMIT_STR,
            )
        }
        self.assertEqual(asl_keys, set(TEST_PROFILE.advanced_scenarios))


class TestApplySubCapsDirectly(ProfileTestCase):
    def _frames(self, zone_score, category_score):
        scenario_scores_df = pd.DataFrame(
            {
                "Scenario": list(ZONE_SCENARIOS) + ["Other"],
                "Score": [zone_score] * 3 + [1.0],
            }
        )
        category_scores_df = pd.DataFrame(
            {
                "Category": [
                    sa.SLIF_ADVANCED_SPEED_LIMIT_CATEGORY,
                    "SLIF - Local hazards",
                ],
                "Score": [category_score, 2.5],
            }
        )
        return scenario_scores_df, category_scores_df

    def test_euro_ncap_profile_is_a_no_op(self):
        scenario_scores_df, category_scores_df = self._frames(0.1, 1.3)
        out = sa.apply_advanced_speed_limit_sub_caps(
            category_scores_df.copy(), scenario_scores_df, protocol="encap-26-28"
        )
        self.assertAlmostEqual(out.loc[0, "Score"], 1.3)

    def test_unassessed_group_member_leaves_category_alone(self):
        scenario_scores_df, category_scores_df = self._frames(0.1, np.nan)
        scenario_scores_df.loc[0, "Score"] = np.nan
        out = sa.apply_advanced_speed_limit_sub_caps(
            category_scores_df, scenario_scores_df, protocol=TEST_PROTOCOL
        )
        self.assertTrue(pd.isna(out.loc[0, "Score"]))

    def test_other_categories_are_untouched(self):
        scenario_scores_df, category_scores_df = self._frames(0.1, 1.3)
        out = sa.apply_advanced_speed_limit_sub_caps(
            category_scores_df, scenario_scores_df, protocol=TEST_PROTOCOL
        )
        self.assertAlmostEqual(out.loc[0, "Score"], 1.2)
        self.assertEqual(out.loc[1, "Score"], 2.5)


if __name__ == "__main__":
    unittest.main()
