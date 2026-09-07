# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import random
import pandas as pd
import logging
from dataclasses import dataclass
import openpyxl
from euroncap_rating_2026.crash_avoidance import matrix_processing
from euroncap_rating_2026.crash_avoidance import data_model
from euroncap_rating_2026.crash_avoidance import robustness_layer
from euroncap_rating_2026 import common
from euroncap_rating_2026.numeric import round_half_up
import euroncap_rating_2026.crash_avoidance.matrix_processing as matrix_processing


logger = logging.getLogger(__name__)


ATTRIBUTE_KEY_ALIASES = {
    "Target speed": ("Target speed", "GVT speed"),
    "GVT speed": ("Target speed", "GVT speed"),
    "Impact location": ("Impact location", "Impact Location"),
    "Impact Location": ("Impact location", "Impact Location"),
}


FC_ROBUSTNESS_LAYERS_POOL = [
    "Driver input pre-crash",
    "Speed",
    "Acceleration",
    "Initial position offset",
    "Trajectory/Heading",
]

LDC_ROBUSTNESS_LAYERS_POOL = [
    "Impact location",
    "Initial position offset",
]


@dataclass
class StageSubelement:
    name: str
    selected_points_dict: dict
    selected_robustness_dict: dict
    test_points_df: None
    removed_prediction_rows: dict = None


@dataclass
class SelectionInfo:
    test_name: str
    standard_points: list
    extended_points: list
    selected_robustness_layer: robustness_layer.RobustnessLayer = None
    assessment_criteria: robustness_layer.AssessmentCriteria = None
    verification_condition: str = None


@dataclass
class LoadcaseInfo:
    test_name: str
    test_points: list
    robustness_layers: dict
    num_extended_tests: int = 2
    num_robustness_tests: int = 1
    score_extended_range: float = 0.0
    score_standard_range: float = 0.0
    robustness_layer_score: float = 0.0
    stage_subelement_key: data_model.StageSubelementKey = None

    @property
    def num_standard_tests(self):
        return data_model.STANDARD_RANGE_VERIFICATION_TEST_NUM.get(self.test_name, 0)

    def select_test_points(self):
        """
        Select test points based on the defined criteria.
        Randomly selects:
        - num_standard_tests from non-extended range test points
        - num_extended_tests from extended range test points
        - num_robustness_tests from robustness layers (if available)
        """

        # Handle subtest case: if self.test_name is in SUBTEST_TO_TEST_DICT, update num_standard_tests using _day and _night
        is_subtest = False
        if f"{self.test_name}_day" in data_model.SUBTEST_TO_TEST_DICT:
            is_subtest = True
            day_test = f"{self.test_name}_day"
            night_test = f"{self.test_name}_night"
            num_day_test = data_model.STANDARD_RANGE_VERIFICATION_TEST_NUM.get(
                day_test, 0
            )
            num_night_test = data_model.STANDARD_RANGE_VERIFICATION_TEST_NUM.get(
                night_test, 0
            )
        # Filter out test points that are red or grey
        filtered_test_points = [
            tp
            for tp in self.test_points
            if tp.color
            not in (
                common.PredictionColor.RED,
                common.PredictionColor.GREY,
            )
        ]

        # CC/CM ELK On/OvU/OvI: only GREEN predictions are eligible for selection -
        # yellow, orange, and brown points must be excluded in addition to red/grey.
        if self.test_name.startswith(("CC ELK", "CM ELK")):
            filtered_test_points = [
                tp
                for tp in filtered_test_points
                if tp.color == common.PredictionColor.GREEN
            ]

        # Remove from filtered test points some corner cases referring to the protocol:
        # CCRb with target speed >80 km/h
        # CCFhos: every cell except the pinned CCFHOS_PINNED_TEST_POINT
        # CCFhol: every cell
        # CMRb with target speed >80 km/h
        # CCFtap and CMFtap with target speed >60 km/h
        # CCCscp and CMCscp with target speed >60 km/h
        # CM ELK On and CM ELK Ov with target speed >80 km/h
        # ELK RE with VUT speed >80 km/h
        # CC ELK On and CC ELK Ov with target speed >80 km/h

        def get_speed(attr, key):
            val = attr.get(key, 0)
            try:
                val = val.replace(" km/h", "") if isinstance(val, str) else val
                return float(val)
            except (TypeError, ValueError):
                return 0.0

        for tp in filtered_test_points:
            logger.debug(
                f"Test point at row {tp.row}, col {tp.col} has attributes: {tp.attributes}"
            )
        filtered_test_points = [
            tp
            for tp in filtered_test_points
            if not (
                (
                    self.test_name.startswith("CCRb")
                    and get_speed(tp.attributes, "Target speed") > 80
                )
                or (
                    # CCFhos is a Corner Case in full bar one cell (CA Frontal
                    # 4.1.3 + the 4.2.1 footnote), so the assessor tests only
                    # CCFHOS_PINNED_TEST_POINT. That cell is always selected
                    # without any pinning code: the red/grey filter above has
                    # already dropped it when it isn't a scored colour, it is
                    # not in EXTENDED_RANGE_CELLS["CCFhos"] so it classifies
                    # as Standard, and random.sample over a single remaining
                    # candidate is deterministic on every seed.
                    self.test_name.startswith("CCFhos")
                    and (tp.row, tp.col) != data_model.CCFHOS_PINNED_TEST_POINT
                )
                # CCFhol is a Corner Case in full with no exception: it never
                # gets a verification test, Standard or Extended.
                or self.test_name.startswith("CCFhol")
                or (
                    self.test_name.startswith("CMRb")
                    and get_speed(tp.attributes, "Target speed") > 80
                )
                or (
                    self.test_name.startswith(("CCFtap", "CMFtap"))
                    and get_speed(tp.attributes, "Target speed") > 60
                )
                or (
                    self.test_name.startswith(("CCCscp", "CMCscp"))
                    and get_speed(tp.attributes, "Target speed") > 60
                )
                or (
                    self.test_name.startswith("CM ELK")
                    and get_speed(tp.attributes, "Target speed") > 80
                )
                or (
                    self.test_name.startswith("ELK RE")
                    and get_speed(tp.attributes, "VUT speed") > 80
                )
                or (
                    self.test_name.startswith("CC ELK")
                    and get_speed(tp.attributes, "Target speed") > 80
                )
            )
        ]

        # Select one random robustness layer with value "YES".
        # This must happen before point sampling so that robustness-layer-dependent
        # exclusions can be applied to the candidate pool first.
        yes_layers = [
            layer for layer, val in self.robustness_layers.items() if val == "YES"
        ]

        if self.stage_subelement_key == data_model.StageSubelementKey.FC:
            robustness_layer_pool = FC_ROBUSTNESS_LAYERS_POOL
        elif self.stage_subelement_key == data_model.StageSubelementKey.LDC:
            robustness_layer_pool = LDC_ROBUSTNESS_LAYERS_POOL
        else:
            robustness_layer_pool = []

        yes_pool_layers = [
            layer for layer in yes_layers if layer in robustness_layer_pool
        ]

        selected_robustness = random.sample(
            yes_pool_layers, min(self.num_robustness_tests, len(yes_pool_layers))
        )

        if selected_robustness:
            # Get the verification condition for the selected robustness layer
            selected_robustness = selected_robustness[0]
            verification_condition, assessment_criteria = (
                robustness_layer.get_robustness_layer_verification_condition(
                    selected_robustness, self.test_name
                )
            )
        else:
            selected_robustness = None
            verification_condition = None
            assessment_criteria = None

        logger.debug(
            f"Selected robustness layer for {self.test_name}: {selected_robustness}, "
            f"Verification condition: {verification_condition}, "
            f"Assessment criteria: {assessment_criteria}"
        )

        # Apply robustness-layer-dependent point exclusions before sampling
        filtered_test_points = _apply_robustness_layer_point_exclusions(
            self.test_name, selected_robustness, filtered_test_points
        )

        # Select standard range test points
        if is_subtest:
            # Split standard points into day and night based on row index
            day_points = [
                tp
                for tp in filtered_test_points
                if tp.test_range == matrix_processing.TestRange.STANDARD
                and 0 <= tp.row <= 5
            ]
            night_points = [
                tp
                for tp in filtered_test_points
                if tp.test_range == matrix_processing.TestRange.STANDARD
                and 6 <= tp.row <= 11
            ]
            selected_day = random.sample(day_points, min(num_day_test, len(day_points)))
            selected_night = random.sample(
                night_points, min(num_night_test, len(night_points))
            )
            selected_standard = selected_day + selected_night
        else:
            standard_points = [
                tp
                for tp in filtered_test_points
                if tp.test_range == matrix_processing.TestRange.STANDARD
            ]
            selected_standard = random.sample(
                standard_points, min(self.num_standard_tests, len(standard_points))
            )

        # Select extended range test points
        extended_points = [
            tp
            for tp in filtered_test_points
            if tp.test_range == matrix_processing.TestRange.EXTENDED
        ]
        selected_extended = random.sample(
            extended_points, min(self.num_extended_tests, len(extended_points))
        )

        return SelectionInfo(
            test_name=self.test_name,
            standard_points=selected_standard,
            extended_points=selected_extended,
            selected_robustness_layer=selected_robustness,
            assessment_criteria=assessment_criteria,
            verification_condition=verification_condition,
        )

    def select_points_from_input(self, input_points: list) -> "SelectionInfo":
        """
        Select test points by matching each dict in input_points against the available
        TestPoint objects using attribute comparison, rather than random selection.

        Keys present in an input dict but absent from tp.attributes (e.g. 'Door selected',
        'Function') are injected into the matched TestPoint's attributes after matching so
        they propagate to the output rows.

        For CCCscp and CMCscp, 'Target approach' is excluded from matching (the matrix
        stores a randomly-assigned value) and is instead always taken from the input dict.

        Returns a SelectionInfo with no robustness layer selected.
        """
        import copy

        # Keys to ignore during attribute matching for specific test names
        if self.test_name in ("CCCscp", "CMCscp"):
            ignore_keys = frozenset(["Target approach"])
        else:
            ignore_keys = frozenset()

        # Keys to always overwrite from the input dict after matching
        overwrite_keys = ignore_keys

        standard_points = []
        extended_points = []

        for input_attrs in input_points:
            matched = None
            for tp in self.test_points:
                if attributes_match(
                    tp.attributes, input_attrs, ignore_keys=ignore_keys
                ):
                    matched = copy.copy(tp)
                    matched.attributes = dict(tp.attributes)
                    break
            if matched is None:
                logger.warning(
                    f"[{self.test_name}] No TestPoint found matching attributes {input_attrs}. Skipping."
                )
                continue
            # Inject keys from the input dict that are absent from tp.attributes
            # (under any alias spelling, stored under the canonical one), and
            # overwrite keys that were excluded from matching.
            for key, val in input_attrs.items():
                target_key = canonical_attribute_key(matched.attributes, key)
                if target_key not in matched.attributes or key in overwrite_keys:
                    matched.attributes[target_key] = val

            if matched.test_range == matrix_processing.TestRange.EXTENDED:
                extended_points.append(matched)
            else:
                standard_points.append(matched)

        # Select robustness layer the same way as the matrix flow.
        yes_layers = [
            layer for layer, val in self.robustness_layers.items() if val == "YES"
        ]

        if self.stage_subelement_key == data_model.StageSubelementKey.FC:
            robustness_layer_pool = FC_ROBUSTNESS_LAYERS_POOL
        elif self.stage_subelement_key == data_model.StageSubelementKey.LDC:
            robustness_layer_pool = LDC_ROBUSTNESS_LAYERS_POOL
        else:
            robustness_layer_pool = []

        yes_pool_layers = [
            layer for layer in yes_layers if layer in robustness_layer_pool
        ]

        selected_robustness_candidates = random.sample(
            yes_pool_layers, min(self.num_robustness_tests, len(yes_pool_layers))
        )

        if selected_robustness_candidates:
            selected_robustness_layer = selected_robustness_candidates[0]
            verification_condition, assessment_criteria = (
                robustness_layer.get_robustness_layer_verification_condition(
                    selected_robustness_layer, self.test_name
                )
            )
        else:
            selected_robustness_layer = None
            verification_condition = None
            assessment_criteria = None

        return SelectionInfo(
            test_name=self.test_name,
            standard_points=standard_points,
            extended_points=extended_points,
            selected_robustness_layer=selected_robustness_layer,
            assessment_criteria=assessment_criteria,
            verification_condition=verification_condition,
        )


_ELK_ON_TESTS = frozenset({"CC ELK On", "CM ELK On"})
_ELK_OV_TESTS = frozenset({"CC ELK OvU", "CC ELK OvI", "CM ELK OvU", "CM ELK OvI"})


def _apply_robustness_layer_point_exclusions(
    test_name: str, selected_robustness, filtered_test_points: list
) -> list:
    """
    Filters out test points that are forbidden for a given (test, robustness layer) pair.

    - CC/CM ELK On + "Impact location": exclude any point whose
      'Lateral velocity' attribute equals "0.3 m/s".
    - CC/CM ELK OvU/OvI + "Initial position offset": exclude any point whose
      attributes match an entry in
      data_model.ELK_OV_INITIAL_POSITION_OFFSET_EXCLUDED_CELLS for this scenario.
    """
    if selected_robustness is None:
        return filtered_test_points

    if test_name in _ELK_ON_TESTS and selected_robustness == "Impact location":
        before = len(filtered_test_points)
        filtered_test_points = [
            tp
            for tp in filtered_test_points
            if str(tp.attributes.get("Lateral velocity", "")).strip() != "0.3 m/s"
        ]
        logger.debug(
            "[%s] Excluded %d point(s) with Lateral velocity 0.3 m/s "
            "due to 'Impact location' robustness layer.",
            test_name,
            before - len(filtered_test_points),
        )

    elif (
        test_name in _ELK_OV_TESTS and selected_robustness == "Initial position offset"
    ):
        excluded_cells = [
            {k: v for k, v in cell.items() if k != "Scenario"}
            for cell in data_model.ELK_OV_INITIAL_POSITION_OFFSET_EXCLUDED_CELLS
            if cell.get("Scenario") == test_name
        ]
        if excluded_cells:
            before = len(filtered_test_points)
            filtered_test_points = [
                tp
                for tp in filtered_test_points
                if not any(
                    attributes_match(tp.attributes, excl) for excl in excluded_cells
                )
            ]
            logger.debug(
                "[%s] Excluded %d point(s) matching purple cells "
                "due to 'Initial position offset' robustness layer.",
                test_name,
                before - len(filtered_test_points),
            )

    return filtered_test_points


def attributes_match(
    tp_attrs: dict, input_attrs: dict, ignore_keys: frozenset = frozenset()
) -> bool:
    """
    Return True if every key in input_attrs that also exists in tp_attrs has the same
    value (after stripping surrounding whitespace from both sides).
    Keys present in input_attrs but absent from tp_attrs are ignored — they are typically
    attributes added post-match (e.g. 'Door selected', 'Function').
    Keys listed in ignore_keys are skipped entirely (e.g. 'Target approach' for CCCscp/CMCscp
    where the matrix stores a randomly-assigned value that should not be used for matching).
    """

    def _get_attr_value(attrs: dict, key: str):
        for candidate_key in ATTRIBUTE_KEY_ALIASES.get(key, (key,)):
            if candidate_key in attrs:
                return attrs[candidate_key]
        return None

    def _normalize_value(key: str, value):
        if value is None:
            return None

        normalized_key = key.lower()
        normalized_value = str(value).strip()

        if normalized_key in {"impact location", "impact location"}:
            if normalized_value.endswith("%"):
                try:
                    return f"{int(float(normalized_value[:-1]))}%"
                except ValueError:
                    return normalized_value
            try:
                numeric_value = float(normalized_value)
            except ValueError:
                return normalized_value
            return f"{int(numeric_value * 100)}%"

        return normalized_value

    for key, input_val in input_attrs.items():
        if key in ignore_keys:
            continue
        tp_val = _get_attr_value(tp_attrs, key)
        if tp_val is None:
            continue
        if _normalize_value(key, tp_val) != _normalize_value(key, input_val):
            return False
    return True


# Legacy alias kept for any external caller that imported the private name.
_attributes_match = attributes_match


def canonical_attribute_key(attrs: dict, key: str) -> str:
    """Return the spelling of *key* to use when writing into *attrs*: the
    alias spelling already present in *attrs* if any, otherwise the canonical
    (first-listed) alias — so merging input attributes into a matched point
    never adds a duplicate under an alternative spelling, and a key the point
    doesn't carry is stored under its canonical form (e.g. an input
    "Impact Location" lands as "Impact location")."""
    aliases = ATTRIBUTE_KEY_ALIASES.get(key, (key,))
    for candidate_key in aliases:
        if candidate_key in attrs:
            return candidate_key
    return aliases[0]


def get_all_lsc_points(loadcase_info):
    """
    Low speed collision point selection criteria: select all non-grey points.
    """
    test_name = loadcase_info.test_name
    test_points = loadcase_info.test_points

    filtered_test_points = [
        tp
        for tp in test_points
        if tp.color not in (common.PredictionColor.GREY, common.PredictionColor.RED)
    ]
    return SelectionInfo(
        test_name=test_name,
        standard_points=filtered_test_points,
        extended_points=[],
        selected_robustness_layer=None,
        assessment_criteria=None,
        verification_condition=None,
    )


def get_lsc_selected_points(loadcase_info):
    """
    Low speed collision point selection criteria:

    • Car & PTW Scenarios: test lowest and highest target speed, and 1 random target speed in between. In case of impact, test adjacent cases and keep testing in +10km/h increments of target speed until prediction is met.
    • CPMRCm: Test 1.00 and 2.00m gaps for all EPTc speeds. In case of impact, test 1.50m gap.
    • CPMRCs, CPMFC: Test all cases for the 25 and 75% impact location. In case of impact, test the 50% case.
    • CBNAO SfS: Test the highest and lowest target speed, in all ‘d’ cases. In case of impact, test the mid target speed (if applicable).
    • CBDA: Test the largest and shortest gap in combination with the highest and lowest target speed. In case of the prediction not being met, test adjacent grid cells in all directions until prediction is met.
    """
    standard_points = []

    test_name = loadcase_info.test_name
    test_points = loadcase_info.test_points

    filtered_test_points = [
        tp
        for tp in test_points
        if tp.color
        not in (
            common.PredictionColor.RED,
            common.PredictionColor.GREY,
        )
    ]

    # CPMRCm: Test 1.00 and 2.00m gaps for all EPTc speeds.
    if test_name.startswith("CPMRCm"):
        # Test 1.00 and 2.00m gaps for all EPTc speeds.
        # For each EPTc speed, select points with Gap 1.00 and 2.00
        eptc_speeds = sorted(
            set(
                tp.attributes.get("Target speed")
                for tp in filtered_test_points
                if tp.test_range == matrix_processing.TestRange.STANDARD
                and tp.attributes.get("Target speed") is not None
            )
        )
        for speed in eptc_speeds:
            for gap in ["1.00 m", "2.00 m"]:
                for tp in filtered_test_points:
                    if (
                        tp.test_range == matrix_processing.TestRange.STANDARD
                        and tp.attributes.get("Target speed") == speed
                        and tp.attributes.get("Gap") == gap
                    ):
                        standard_points.append(tp)
                        break  # Only one point per speed-gap combination

    # CPMRCs, CPMFC: Test all cases for the 25 and 75% impact location.
    elif test_name.startswith("CPMRCs") or test_name.startswith("CPMFC"):
        # Test all cases for 25% and 75% impact location.
        for tp in filtered_test_points:
            if (
                tp.test_range == matrix_processing.TestRange.STANDARD
                and tp.attributes.get("Impact location") in ["25%", "75%"]
            ):
                standard_points.append(tp)

    # CBNAO: Test the highest and lowest target speed, in all ‘d’ cases.
    elif test_name.startswith("CBNAO SfS"):
        # Test highest, lowest, and all available target speeds, in all 'd' cases.
        speeds = sorted(
            set(
                tp.attributes.get("Target speed")
                for tp in filtered_test_points
                if tp.test_range == matrix_processing.TestRange.STANDARD
                and tp.attributes.get("d") is not None
            )
        )
        if speeds:
            lowest, highest = speeds[0], speeds[-1]
            for tp in filtered_test_points:
                if (
                    tp.test_range == matrix_processing.TestRange.STANDARD
                    and tp.attributes.get("d") is not None
                    and tp.attributes.get("Target speed") in [lowest, highest]
                ):
                    standard_points.append(tp)

    # CBDA: Test the largest and shortest gap in combination with the highest and lowest target speed.
    elif test_name.startswith("CBDA"):
        # Test largest and shortest gap with highest and lowest target speed.
        gaps = sorted(
            set(
                tp.attributes.get("Gap")
                for tp in filtered_test_points
                if tp.test_range == matrix_processing.TestRange.STANDARD
            )
        )
        speeds = sorted(
            set(
                tp.attributes.get("Target speed")
                for tp in filtered_test_points
                if tp.test_range == matrix_processing.TestRange.STANDARD
            )
        )
        if gaps and speeds:
            # Select combinations of shortest/longest gap and lowest/highest EBT speed
            for gap in (gaps[0], gaps[-1]):
                for speed in (speeds[0], speeds[-1]):
                    for tp in filtered_test_points:
                        if (
                            tp.test_range == matrix_processing.TestRange.STANDARD
                            and tp.attributes.get("Gap") == gap
                            and tp.attributes.get("Target speed") == speed
                        ):
                            standard_points.append(tp)
                            break  # Stop after first match

    else:
        # Car & PTW Scenarios: test lowest and highest target speed, and 1 random target speed in between.
        # Handle cases where the speed attribute can be "GVT speed", or "Target speed"
        speed_keys = ["GVT speed", "Target speed"]
        found_speed_key = None
        for key in speed_keys:
            if any(
                tp.test_range == matrix_processing.TestRange.STANDARD
                and tp.attributes.get(key) is not None
                for tp in filtered_test_points
            ):
                found_speed_key = key
                break
        if found_speed_key:
            speeds = sorted(
                set(
                    tp.attributes.get(found_speed_key)
                    for tp in filtered_test_points
                    if tp.test_range == matrix_processing.TestRange.STANDARD
                    and tp.attributes.get(found_speed_key) is not None
                )
            )
        else:
            speeds = []
        if speeds:
            lowest, highest = speeds[0], speeds[-1]
            mid_speeds = [s for s in speeds if s != lowest and s != highest]
            mid = random.choice(mid_speeds) if mid_speeds else None
            for tp in filtered_test_points:
                if (
                    tp.test_range == matrix_processing.TestRange.STANDARD
                    and tp.attributes.get(found_speed_key) in [lowest, highest]
                ):
                    standard_points.append(tp)
            if mid is not None:
                for tp in filtered_test_points:
                    if (
                        tp.test_range == matrix_processing.TestRange.STANDARD
                        and tp.attributes.get(found_speed_key) == mid
                    ):
                        standard_points.append(tp)

    return SelectionInfo(
        test_name=loadcase_info.test_name,
        standard_points=standard_points,
        extended_points=[],
        selected_robustness_layer=None,
        assessment_criteria=None,
        verification_condition=None,
    )


def compute_robustness_layer_score(robustness_layers_df, loadcase_key, subtest_name):
    if robustness_layers_df is None:
        logger.debug("No Robustness Layers sheet provided.")
        return 0.0

    if loadcase_key not in robustness_layers_df.columns:
        raise ValueError(
            f"Column '{loadcase_key}' not found in Robustness Layers sheet."
        )

    test_robustness_df = robustness_layers_df[[loadcase_key]]

    number_of_claimed_robustness_layers = float(
        (test_robustness_df[loadcase_key] == "YES").sum()
    )
    number_of_applicable_robustness_layers = float(
        test_robustness_df[loadcase_key].notna().sum()
    )
    logger.debug(
        f"Number of claimed robustness layers: {number_of_claimed_robustness_layers}"
    )
    logger.debug(
        f"Number of applicable robustness layers: {number_of_applicable_robustness_layers}"
    )

    total_robustness_score = data_model.TOTAL_SCORES.get(subtest_name, {}).get(
        "Robustness", 0
    )

    if total_robustness_score is None:
        total_robustness_score = 0.0

    logger.debug(
        f"total_robustness_score: {total_robustness_score}, number_of_applicable_robustness_layers: {number_of_applicable_robustness_layers}"
    )
    robustness_layer_tested_score = (
        total_robustness_score / number_of_applicable_robustness_layers * 1
    )

    logger.debug(
        f"total_robustness_score: {total_robustness_score}, number_of_applicable_robustness_layers: {number_of_applicable_robustness_layers}, not_tested_claimed_robustness_layers_num: {number_of_claimed_robustness_layers - 1}"
    )
    # Number of claimed robustness layers is decreased by 1 because 1 has been selected for testing
    not_tested_claimed_robustness_layers_num = number_of_claimed_robustness_layers - 1
    robustness_layer_contant_score = (
        total_robustness_score
        / number_of_applicable_robustness_layers
        * not_tested_claimed_robustness_layers_num
    )

    logger.debug(
        f"Calculated robustness score for {subtest_name}: {robustness_layer_contant_score}"
    )
    claimed_layers = set(
        robustness_layers_df.loc[
            robustness_layers_df[loadcase_key] == "YES", "Robustness layer"
        ]
        .dropna()
        .tolist()
    )
    robustness_layer_score = robustness_layer.RobustnessLayerScore(
        robustness_layer_tested_score,
        robustness_layer_contant_score,
        n_applicable=number_of_applicable_robustness_layers,
        claimed_layers=claimed_layers,
    )
    logger.info(
        f"Robustness layer total: {robustness_layer_score.total_score}, "
        f"constant part: {robustness_layer_score.constant_score}, "
        f"tested part: {robustness_layer_score.tested_score}, "
    )
    return robustness_layer_score


def apply_extended_range_bands(score_extended_range, total_extended_score):
    """Snap an extended-range score to its protocol band.

    Bands: [0, 50%) -> 0, [50%, 75%) -> 50%, [75%, 100%) -> 75%,
    otherwise the full total. Score and band edges are quantized at 6 dp
    before comparing so that a score that is "really" exactly on an edge
    but computed 1 ulp off lands in the right band; the
    final else also clamps a score 1 ulp above the total, which the old
    exact == comparison passed through unsnapped.
    """
    score = round_half_up(score_extended_range, 6)
    half_score = round_half_up(0.5 * total_extended_score, 6)
    three_quarter_score = round_half_up(0.75 * total_extended_score, 6)
    if score < half_score:
        return 0
    elif half_score <= score < three_quarter_score:
        return 0.5 * total_extended_score
    elif three_quarter_score <= score < total_extended_score:
        return 0.75 * total_extended_score
    else:
        return total_extended_score


def meets_half_total(score, total_score):
    """True when score reaches 50% of total_score.

    Both sides are quantized at 6 dp so a score that is exactly half the
    total but computed as 0.4999999...*total still passes.
    """
    return round_half_up(score, 6) >= round_half_up(0.5 * total_score, 6)


def compute_predicted_score(test_name, test_points):

    extended_score_sum = sum(
        tp.predicted_score
        for tp in test_points
        if tp.test_range == matrix_processing.TestRange.EXTENDED
    )
    standard_score_sum = sum(
        tp.predicted_score
        for tp in test_points
        if tp.test_range == matrix_processing.TestRange.STANDARD
    )
    extended_count = sum(
        1 for tp in test_points if tp.test_range == matrix_processing.TestRange.EXTENDED
    )
    standard_count = sum(
        1 for tp in test_points if tp.test_range == matrix_processing.TestRange.STANDARD
    )
    logger.debug(f"Number of test points (is_extended_range=True): {extended_count}")
    logger.debug(f"Number of test points (is_extended_range=False): {standard_count}")
    logger.debug(
        f"Sum of predicted scores (is_extended_range=True): {extended_score_sum}"
    )
    logger.debug(
        f"Sum of predicted scores (is_extended_range=False): {standard_score_sum}"
    )

    total_standard_score = data_model.TOTAL_SCORES[test_name]["Standard"]
    if "Extended" in data_model.TOTAL_SCORES[test_name]:
        total_extended_score = data_model.TOTAL_SCORES[test_name]["Extended"]
    else:
        total_extended_score = 0.0

    logger.debug(
        f"standard_score_sum: {standard_score_sum}, standard_count: {standard_count}, total_standard_score: {total_standard_score}, "
    )
    score_standard_range = (
        standard_score_sum / standard_count * total_standard_score
        if standard_count > 0
        else 0
    )

    logger.debug(
        f"extended_score_sum: {extended_score_sum}, extended_count: {extended_count}, total_extended_score: {total_extended_score}, "
    )
    score_extended_range = (
        extended_score_sum / extended_count * total_extended_score
        if extended_count > 0
        else 0
    )

    logger.debug(f"Prefiltered extended score: {score_extended_range}")

    # Adjust scores based on the defined ranges
    score_extended_range = apply_extended_range_bands(
        score_extended_range, total_extended_score
    )

    logger.debug(f"Calculated score for standard range: {score_standard_range}")
    logger.debug(f"Calculated score for extended range: {score_extended_range}")

    return score_standard_range, score_extended_range


def strip_suffix(test_name):
    suffixes = ("fs", "ns", "fo", "no")
    for suffix in suffixes:
        if test_name.endswith(suffix):
            return test_name[: -len(suffix)]
    return test_name


def read_loadcase_info(
    prediction_df, robustness_layers_df, loadcase_name, stage_subelement_key
):

    test_points = matrix_processing.get_test_matrix(
        prediction_df, loadcase_name, stage_subelement_key
    )
    if test_points is None:
        logger.warning(
            "No prediction matrix available for scenario '%s'. Using zero test points.",
            loadcase_name,
        )
        test_points = []
    subtest_name = None
    if loadcase_name in data_model.SUBTEST_TO_TEST_DICT.keys():
        logger.debug(
            f"Scenario {loadcase_name} is a subtest, mapping to main test: {data_model.SUBTEST_TO_TEST_DICT[loadcase_name]}"
        )
        subtest_name = loadcase_name
        loadcase_name = data_model.SUBTEST_TO_TEST_DICT[loadcase_name]

    if robustness_layers_df is not None:
        robustness_key = loadcase_name

        if robustness_key not in robustness_layers_df.columns:
            raise ValueError(
                f"Column '{robustness_key}' not found in Robustness Layers sheet."
            )
        # Create a dictionary with robustness layer names as keys and their values for the given loadcase_name
        robustness_dict = {
            robustness_layers_df.at[idx, "Robustness layer"]: val
            for idx, val in robustness_layers_df[robustness_key].dropna().items()
        }
        logger.debug(f"Robustness layers for {robustness_key}: {robustness_dict}")
    else:
        robustness_dict = {}
    return LoadcaseInfo(
        test_name=subtest_name if subtest_name else loadcase_name,
        test_points=test_points,
        robustness_layers=robustness_dict,
        stage_subelement_key=stage_subelement_key,
    )


def get_sheet_prefix(stage_element_name, subelement_name):
    logger.debug(
        f"get_sheet_prefix called with stage_element_name='{stage_element_name}', subelement_name='{subelement_name}'"
    )
    element_initials = get_stage_subelement_key(stage_element_name).value
    if subelement_name == "Pedestrian & cyclist":
        subelement_name = "Ped & Cyc"
    if subelement_name == "Single vehicle":
        subelement_name = "Single Veh"
    return f"{element_initials} - {subelement_name}"


def check_robustness_scoring_enabled(subtest_name, test_points):
    predicted_standard_score, _ = compute_predicted_score(subtest_name, test_points)
    total_standard_score = data_model.TOTAL_SCORES[subtest_name]["Standard"]

    return meets_half_total(predicted_standard_score, total_standard_score)


def get_stage_subelement_key(stage_element_name):
    element_initials = "".join([word[0].upper() for word in stage_element_name.split()])
    if element_initials not in data_model.StageSubelementKey.__members__:
        raise ValueError(f"Invalid stage element name: {stage_element_name}")
    return data_model.StageSubelementKey[element_initials]


def find_duplicate_rows(test_points):
    """
    Groups test points that share the same attributes (VUT speed, target
    speed, Function, Day/Night, impact location, etc.) -- i.e. test points
    that represent the same real-world scenario declared twice, which
    happens when a grey-cell Function dropdown makes a second sub-block
    resolve to the same attributes as a fixed sub-block above it.

    Returns a list of groups (each a list of TestPoint objects), one group
    per distinct attribute tuple that has 2 or more test points. Each group
    keeps every member (not just the first two), and tracks the actual
    TestPoint objects -- not bare row numbers -- since a CPLA/CBLA matrix is
    several columns wide (one per impact location) and a bare row number is
    shared by unrelated test points in other columns.
    """
    attr_to_test_points = {}
    for tp in test_points:
        # Convert attributes dict to a tuple of sorted items for hashability
        attr_tuple = tuple(sorted(tp.attributes.items()))
        attr_to_test_points.setdefault(attr_tuple, []).append(tp)
    return [group for group in attr_to_test_points.values() if len(group) > 1]


def check_cpla_cbla_coherence(test_points):
    """
    Checks CPLA/CBLA coherence: when two test points share the same
    attributes (a duplicate declaration, e.g. both sub-blocks resolving to
    Function=AEB at the same speed), their colors must agree with each
    other -- ignoring any member colored GREY (no prediction entered).

    Returns a tuple (coherent: bool, mismatches: list[str]); mismatches is
    empty when coherent is True, and otherwise contains one human-readable
    description per disagreeing duplicate group, naming the block-relative
    (row, col) of every disagreeing test point, its color, and the shared
    attributes -- for use in an actionable error message upstream.
    """
    mismatches = []
    for non_grey, attributes in collect_cpla_cbla_coherence_mismatches(test_points):
        points_desc = ", ".join(
            f"(row={tp.row}, col={tp.col})={tp.color.value}" for tp in non_grey
        )
        mismatch = (
            f"test points at {points_desc} share attributes {attributes} "
            "but have different colors"
        )
        logger.error("CPLA/CBLA coherence check: %s.", mismatch)
        mismatches.append(mismatch)

    return not mismatches, mismatches


def collect_cpla_cbla_coherence_mismatches(test_points):
    """The structured form of check_cpla_cbla_coherence's rule: one
    (disagreeing_points, shared_attributes) tuple per duplicate-declaration
    group whose non-grey members predict more than one colour.
    disagreeing_points keeps every non-grey member, in matrix order. Shared
    by the blocking preprocess check above and the report-only collector
    (consistency.find_prediction_inconsistencies)."""
    duplicate_groups = find_duplicate_rows(test_points)
    if not duplicate_groups:
        logger.debug("No duplicate rows found for CPLA/CBLA coherence check.")
        return []

    mismatches = []
    for group in duplicate_groups:
        non_grey = [tp for tp in group if tp.color != common.PredictionColor.GREY]
        colors = {tp.color for tp in non_grey}
        if len(colors) > 1:
            mismatches.append((non_grey, dict(group[0].attributes)))
    return mismatches


# CPLA/CBLA's fixed original-AEB band and choice band (grey-cell Function
# dropdown) each have a row at these two VUT speeds -- the
# only speeds where both bands are physically present at once.
_AEB_DUPLICATE_OVERLAP_SPEEDS = {"50 km/h", "60 km/h"}


def collapse_cpla_cbla_aeb_duplicate_rows(prediction_df, test_points, test_name):
    """
    Per the collapse-and-promote rule: for each of 50
    and 60 km/h independently, if the choice band's row (grey-cell Function
    dropdown) declares AEB, it duplicates the fixed band's original-AEB row
    at that speed -- delete the choice-band row entirely. The fixed-band
    row's own extended-range 25% IL cell is promoted to standard range to
    compensate (see matrix_processing.classify_cpla_rows /
    classify_cbla_rows, which derive that promotion from the resulting row
    shape, not from anything this function writes).

    A choice-band row left on FCW is a genuine warning test and is kept
    untouched -- the decision is per speed, not all-or-nothing, so a CPLA/
    CBLA matrix can end up with 0, 1, or 2 rows fewer.

    Args:
        prediction_df: the full prediction sheet DataFrame (spans multiple
            matrices; only this matrix's rows are dropped).
        test_points: this matrix's *pre-collapse* TestPoints, as produced by
            matrix_processing.get_test_matrix/get_test_matrix_from_region --
            already carries "Function"/"VUT speed" attributes, so this
            reuses them instead of re-reading the raw sheet.
        test_name: "CPLA day", "CPLA night", or "CBLA".

    Returns:
        (updated_df, removed_matrix_rows, matrix_start_row): updated_df is
        prediction_df with the matched rows actually dropped and the index
        reset (a blanked row would be misread by common.get_n_rows as the
        end of the matrix). removed_matrix_rows is a sorted list of the
        0-based matrix-relative row indices that were removed -- empty if
        nothing was collapsed. matrix_start_row is this matrix's first data
        row's 0-based index in prediction_df (None if the matrix wasn't
        found), returned so the caller can translate removed_matrix_rows to
        worksheet coordinates without re-locating the matrix itself.
    """
    matrix_indices = matrix_processing.get_matrix_indices(prediction_df, test_name)
    if not matrix_indices:
        return prediction_df, [], None
    start_row = matrix_indices["start_row"]

    rows_to_drop = set()
    for tp in test_points:
        if tp.color == common.PredictionColor.GREY:
            continue  # not selected by the OEM, nothing to collapse

        function = str(tp.attributes.get("Function", "")).strip().upper()
        if function != "AEB":
            continue  # FCW (or unset): kept as-is

        if tp.attributes.get("VUT speed") not in _AEB_DUPLICATE_OVERLAP_SPEEDS:
            continue

        if not matrix_processing.is_fcw_declared_as_aeb(tp, test_name):
            continue  # this is the fixed-band row itself, not the duplicate

        rows_to_drop.add(tp.row)

    if not rows_to_drop:
        return prediction_df, [], start_row

    absolute_rows_to_drop = [start_row + row for row in rows_to_drop]
    updated_df = prediction_df.drop(index=absolute_rows_to_drop).reset_index(drop=True)
    logger.info(
        "Collapsed %d duplicate AEB row(s) for %s: matrix rows %s.",
        len(rows_to_drop),
        test_name,
        sorted(rows_to_drop),
    )
    return updated_df, sorted(rows_to_drop), start_row


def _get_raw_input_parameter(input_parameters_df, parameter_name):
    """The raw Value of the first "Input parameters" row whose
    "Input parameter" equals *parameter_name*, or None when absent."""
    values = input_parameters_df.loc[
        input_parameters_df["Input parameter"] == parameter_name, "Value"
    ]
    return values.iloc[0] if not values.empty else None


def get_vehicle_response(dfs):
    # Read "Vehicle response" from input_parameters_df
    vehicle_response_value = _get_raw_input_parameter(
        dfs["Input parameters"], "Vehicle response"
    )

    if vehicle_response_value is None:
        logger.warning(f"Vehicle response not found in input parameters")
        vehicle_response_enum = None
    else:
        try:
            vehicle_response_enum = data_model.VehicleResponse(vehicle_response_value)
            logger.debug(f"Vehicle response: {vehicle_response_enum}")
        except ValueError as e:
            logger.error(
                f"Invalid vehicle response value '{vehicle_response_value}': {e}"
            )
            vehicle_response_enum = None
    return vehicle_response_enum


def get_extended_range_performance(dfs):
    """Normalized (stripped, lowercase) value of the "Extended range
    performance" input parameter, or None when the sheet/row is absent.
    Shared by the ELK RE expected-value override below and the report-only
    LDW colour rule (consistency.find_prediction_inconsistencies), so the
    two read the parameter identically."""
    if "Input parameters" not in dfs:
        return None
    raw = _get_raw_input_parameter(
        dfs["Input parameters"], "Extended range performance"
    )
    return None if raw is None else str(raw).strip().lower()


# The LSC dooring rule (CBDA): which predicted colours the workbook's
# "Vehicle response" input parameter allows. Shared by the blocking
# preprocess check (check_lsc_consistency) and the report-only collector
# (consistency.find_prediction_inconsistencies), so the rule cannot drift
# between the two.
DOORING_ALLOWED_COLORS = {
    data_model.VehicleResponse.INFORMATION: (common.PredictionColor.BROWN,),
    data_model.VehicleResponse.WARNING: (
        common.PredictionColor.ORANGE,
        common.PredictionColor.YELLOW,
    ),
    data_model.VehicleResponse.RETENTION: (
        common.PredictionColor.GREEN,
        common.PredictionColor.YELLOW,
    ),
}


def collect_lsc_consistency_violations(test_points, vehicle_response_enum):
    """Every dooring test point whose colour DOORING_ALLOWED_COLORS forbids
    for *vehicle_response_enum* -- all of them, not just the first. Returns
    an empty list for an unknown/None vehicle response (nothing to compare
    against; the missing-input check owns reporting the empty parameter)."""
    allowed_colors = DOORING_ALLOWED_COLORS.get(vehicle_response_enum)
    if allowed_colors is None:
        logger.warning("Unknown vehicle response: %s", vehicle_response_enum)
        return []
    return [tp for tp in test_points if tp.color not in allowed_colors]


def check_lsc_consistency(test_points, vehicle_response_enum):
    if not test_points:
        return True
    if vehicle_response_enum not in DOORING_ALLOWED_COLORS:
        logger.warning("Unknown vehicle response: %s", vehicle_response_enum)
        return False

    violations = collect_lsc_consistency_violations(test_points, vehicle_response_enum)
    for tp in violations:
        logger.error(
            f"Test point at row {tp.row}, col {tp.col} has color {tp.color}, which is not allowed for vehicle response {vehicle_response_enum}."
        )
    return not violations


def add_door_attribute(test_points, vehicle_response_enum):
    if vehicle_response_enum == data_model.VehicleResponse.INFORMATION:
        for tp in test_points:
            tp.attributes["Door selected"] = "Driver"
    elif vehicle_response_enum == data_model.VehicleResponse.WARNING:
        for tp in test_points:
            if tp.color == common.PredictionColor.ORANGE:
                tp.attributes["Door selected"] = "Driver"
            elif tp.color == common.PredictionColor.YELLOW:
                tp.attributes["Door selected"] = random.choice(
                    ["Driver", "Passenger", "Behind driver", "Behind passenger"]
                )
    elif vehicle_response_enum == data_model.VehicleResponse.RETENTION:
        for tp in test_points:
            if tp.color == common.PredictionColor.GREEN:
                tp.attributes["Door selected"] = random.choice(
                    ["Driver", "Passenger", "Behind driver", "Behind passenger"]
                )
            elif tp.color == common.PredictionColor.YELLOW:
                tp.attributes["Door selected"] = "Driver"

    for tp in test_points:
        tp.attributes["Function"] = vehicle_response_enum.value.capitalize()

    return test_points


def preprocess_stage_subelement(
    dfs, stage_element_name, subelement_name, input_selected_points=None
):
    sheet_prefix = get_sheet_prefix(stage_element_name, subelement_name)
    robustness_sheet_name = f"{sheet_prefix} robust. pred."
    prediction_sheet_name = f"{sheet_prefix} pred."
    # Key is needed to access STAGE_SUBELEMENT_TO_LOADCASES with the right prefix (LDC, FC, ..)
    stage_subelement_key = get_stage_subelement_key(stage_element_name)

    if stage_subelement_key == data_model.StageSubelementKey.LDC:
        robustness_sheet_name = f"{stage_subelement_key.value} - robust. pred."
    elif stage_subelement_key == data_model.StageSubelementKey.LSC:
        robustness_sheet_name = None

    if robustness_sheet_name is not None and robustness_sheet_name not in dfs:
        raise ValueError(f"{robustness_sheet_name} sheet not found in the input data.")

    test_points_df = pd.DataFrame()

    test_point_rows = []

    if subelement_name == "Pedestrian & cyclist":
        subelement_name = "Ped & Cyc"
    if subelement_name == "Single vehicle":
        subelement_name = "Single Veh"
    stage_subelement_dict = data_model.STAGE_SUBELEMENT_TO_LOADCASES[
        stage_subelement_key
    ]

    if subelement_name not in stage_subelement_dict:
        raise ValueError(
            f"Test prefix '{subelement_name}' not found in STAGE_SUBELEMENT_TO_LOADCASES."
        )

    selected_points_dict = {}
    selected_robustness_dict = {}
    selected_day_robustness = None
    # Cumulative count of rows already collapsed out of prediction_sheet_name
    # earlier in this loop (e.g. by CPLA day, before CPLA night/CBLA are
    # reached) -- CPLA day/night/CBLA share one sheet, top to bottom, so a
    # matrix processed later must add back every row removed above it to
    # recover its position in the *original* (pre-collapse) worksheet.
    prediction_row_offset = 0
    removed_prediction_worksheet_rows = []
    # stage_subelement_dict contains matching test names for the given test prefix
    for loadcase_name in stage_subelement_dict[subelement_name]:
        prediction_df = dfs[prediction_sheet_name]
        if robustness_sheet_name is not None:
            robustness_layers_df = dfs[robustness_sheet_name]
        else:
            robustness_layers_df = None

        if loadcase_name in ["Driveability", "Driver state link"]:
            logger.debug(
                f"Skipping scenario {loadcase_name} as it does not require test point selection."
            )
            continue

        logger.debug(f"Preprocessing scenario: {loadcase_name}")

        loadcase_info = read_loadcase_info(
            prediction_df, robustness_layers_df, loadcase_name, stage_subelement_key
        )

        if stage_subelement_key == data_model.StageSubelementKey.FC and (
            "CPLA" in loadcase_name or "CBLA" in loadcase_name
        ):
            coherence_check, mismatches = check_cpla_cbla_coherence(
                loadcase_info.test_points
            )
            if not coherence_check:
                raise ValueError(
                    f"CPLA and CBLA coherence check failed for scenario "
                    f"{loadcase_name}: " + "; ".join(mismatches) + "."
                )
            else:
                logger.info(
                    f"CPLA and CBLA coherence check passed for scenario {loadcase_name}."
                )

            # Collapse must run -- and loadcase_info must be re-read -- before
            # test-point selection below, so selection sees the collapsed
            # matrix (fewer rows, promoted standard-range cells) rather than
            # the pre-collapse one.
            updated_prediction_df, removed_matrix_rows, matrix_start_row = (
                collapse_cpla_cbla_aeb_duplicate_rows(
                    prediction_df, loadcase_info.test_points, loadcase_info.test_name
                )
            )
            if removed_matrix_rows:
                for matrix_row in removed_matrix_rows:
                    # +2: pandas header=0 offset from 0-based df row index to
                    # 1-based worksheet row (see common.get_start_row).
                    # +prediction_row_offset: rows already removed above
                    # this matrix earlier in this same loop (CPLA day/night/
                    # CBLA share one sheet, top to bottom).
                    original_worksheet_row = (
                        matrix_start_row + matrix_row + prediction_row_offset + 2
                    )
                    removed_prediction_worksheet_rows.append(original_worksheet_row)
                prediction_row_offset += len(removed_matrix_rows)
                dfs[prediction_sheet_name] = updated_prediction_df
                prediction_df = updated_prediction_df
                loadcase_info = read_loadcase_info(
                    prediction_df,
                    robustness_layers_df,
                    loadcase_name,
                    stage_subelement_key,
                )

        current_test_criteria = data_model.get_criteria_for_test(
            loadcase_info.test_name
        )
        is_dooring = current_test_criteria == data_model.NcapTestCriteria.LSC_DOORING
        # Get vehicle response enum
        vehicle_response_enum = get_vehicle_response(dfs)

        # ELK RE only: with Extended range performance
        # = LDW the extended-range verification KPI is the DTLE at the
        # warning instant, not at the end of the excursion.
        is_road_edge = current_test_criteria == data_model.NcapTestCriteria.ROAD_EDGE
        extended_range_performance = (
            get_extended_range_performance(dfs) if is_road_edge else None
        )

        # When input_selected_points is provided, always use input-based selection.
        # Scenarios absent from the dict receive an empty list → zero test points.
        using_input_points = input_selected_points is not None

        if stage_subelement_key == data_model.StageSubelementKey.LSC:
            if using_input_points:
                selected_info = loadcase_info.select_points_from_input(
                    input_selected_points.get(loadcase_name, [])
                )
            else:
                selected_info = get_all_lsc_points(loadcase_info)
            selected_standard_points = selected_info.standard_points
            selected_extended_points = selected_info.extended_points

            if is_dooring and not using_input_points and selected_standard_points:
                if common.is_not_applicable(
                    _get_raw_input_parameter(
                        dfs["Input parameters"], "Vehicle response"
                    )
                ):
                    # An explicit "N/A" Vehicle response means no dooring
                    # system is fitted: the scenario is not claimed (it
                    # scores 0 at compute time), so the dooring colour rule
                    # has nothing to block on.
                    logger.info(
                        f"Vehicle response is N/A; skipping the dooring "
                        f"consistency check for scenario {loadcase_name}."
                    )
                else:
                    check_result = check_lsc_consistency(
                        selected_standard_points, vehicle_response_enum
                    )
                    if not check_result:
                        raise ValueError(
                            f"LSC consistency check failed for scenario {loadcase_name} with vehicle response {vehicle_response_enum}. Please update color specified according to LSC protocol"
                        )
                    else:
                        logger.info(
                            f"LSC consistency check passed for scenario {loadcase_name} with vehicle response {vehicle_response_enum}."
                        )
                    selected_standard_points = add_door_attribute(
                        selected_standard_points, vehicle_response_enum
                    )

            is_robustness_scoring_enabled = False
        else:
            if using_input_points:
                selected_info = loadcase_info.select_points_from_input(
                    input_selected_points.get(loadcase_name, [])
                )
            else:
                selected_info = loadcase_info.select_test_points()
            is_robustness_scoring_enabled = check_robustness_scoring_enabled(
                loadcase_info.test_name, selected_info.standard_points
            )
            selected_standard_points = selected_info.standard_points
            selected_extended_points = selected_info.extended_points

        if not is_robustness_scoring_enabled:
            selected_robustness = "N/A"
            selected_info.verification_condition = "N/A"
        else:
            if "_day" in loadcase_info.test_name:
                selected_day_robustness = selected_info.selected_robustness_layer
                selected_robustness = selected_info.selected_robustness_layer
            if "_night" in loadcase_info.test_name:
                selected_robustness = selected_day_robustness
                selected_day_robustness = None
            else:
                selected_robustness = selected_info.selected_robustness_layer

        selected_robustness_dict[loadcase_info.test_name] = selected_robustness

        for point in selected_standard_points:
            row = {
                "Scenario": loadcase_info.test_name,
                "Test point": f"({point.row}, {point.col})",
                "Range": point.test_range.value,
                "Robustness layer": (
                    selected_robustness
                    if selected_robustness is not None
                    else "Not Applicable"
                ),
                "Verification condition": (
                    selected_info.verification_condition
                    if selected_info.verification_condition
                    else None
                ),
            }

            row["Robustness"] = (
                "PASS"
                if selected_robustness not in [None, "N/A"]
                else ("Not Applicable" if selected_robustness is None else "")
            )

            for attr_name, attr_value in point.attributes.items():
                row[attr_name] = attr_value

            expected_value = data_model.get_score_parameter_for_test(
                loadcase_info.test_name
            )

            if (
                loadcase_info.test_name in ["CPLA day", "CPLA night", "CPLA", "CBLA"]
                and "Function" in point.attributes
                and str(point.attributes["Function"]).upper() == "FCW"
            ):
                expected_value = "fcw_ttc"

            if is_dooring:
                if vehicle_response_enum == data_model.VehicleResponse.RETENTION:
                    expected_value = "TTC @ t_door_opening"
                elif vehicle_response_enum == data_model.VehicleResponse.WARNING:
                    expected_value = "TTC @ t_warning"
                elif vehicle_response_enum == data_model.VehicleResponse.INFORMATION:
                    expected_value = "TTC @ t_information"

            # Add OEM Prediction and Value at the end
            row["OEM Prediction"] = str(point.color.value).capitalize()
            row["Value"] = float("nan")
            row["Expected value"] = expected_value
            if loadcase_info.test_name == "CPMFC":
                row["Expected value baseline"] = "v_impact_baseline"
                row["Value baseline"] = float("nan")
            test_point_rows.append(row)

        for point in selected_extended_points:
            row = {
                "Scenario": loadcase_info.test_name,
                "Test point": f"({point.row}, {point.col})",
                "Range": point.test_range.value,
            }
            expected_value = data_model.get_score_parameter_for_test(
                loadcase_info.test_name
            )

            if (
                loadcase_info.test_name in ["CPLA day", "CPLA night", "CPLA", "CBLA"]
                and "Function" in point.attributes
                and str(point.attributes["Function"]).upper() == "FCW"
            ):
                expected_value = "fcw_ttc"

            if is_dooring:
                if vehicle_response_enum == data_model.VehicleResponse.RETENTION:
                    expected_value = "TTC @ t_door_opening"
                elif vehicle_response_enum == data_model.VehicleResponse.WARNING:
                    expected_value = "TTC @ t_warning"
                elif vehicle_response_enum == data_model.VehicleResponse.INFORMATION:
                    expected_value = "TTC @ t_information"

            if is_road_edge and extended_range_performance == "ldw":
                expected_value = "DTLE @ T_LDW"

            for attr_name, attr_value in point.attributes.items():
                row[attr_name] = attr_value
            # Add OEM Prediction and Value at the end
            row["OEM Prediction"] = str(point.color.value).capitalize()
            row["Value"] = float("nan")
            row["Expected value"] = expected_value
            if loadcase_info.test_name == "CPMFC":
                row["Expected value baseline"] = "v_impact_baseline"
                row["Value baseline"] = float("nan")
            test_point_rows.append(row)

        # Handle verification conditions and robustness layers for test point rows
        for row in test_point_rows:
            # Remove Verification condition if it's NaN or "N/A"
            if row.get("Verification condition") in [None, "N/A"] or pd.isna(
                row.get("Verification condition")
            ):
                row.pop("Verification condition", None)

            # Remove Robustness layer if it's NaN or "N/A"
            if row.get("Robustness layer") in [None, "N/A"] or pd.isna(
                row.get("Robustness layer")
            ):
                row.pop("Robustness layer", None)

        selected_points_dict[loadcase_info.test_name] = (
            selected_standard_points + selected_extended_points
        )

    test_points_df = pd.DataFrame(test_point_rows)
    # Remove "Verification condition" columns if all values are NaN
    for col in ["Verification condition"]:
        if col in test_points_df.columns and test_points_df[col].isna().all():
            test_points_df = test_points_df.drop(columns=[col])
    removed_prediction_rows = (
        {prediction_sheet_name: removed_prediction_worksheet_rows}
        if removed_prediction_worksheet_rows
        else {}
    )
    return StageSubelement(
        name=subelement_name,
        selected_points_dict=selected_points_dict,
        selected_robustness_dict=selected_robustness_dict,
        test_points_df=test_points_df,
        removed_prediction_rows=removed_prediction_rows,
    )


def _parse_optional_float(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        result = float(v)
    except (ValueError, TypeError):
        return None
    if pd.isna(result):
        return None
    return result


def read_test_points(df, preserve_none=False, total_rows=None):
    test_points = []
    computed_info_dict = {}
    for _, row in df.iterrows():
        # Parse "Test point" from string like "(4, 2)" to row and col integers
        row_str = row["Test point"]
        row_idx, col_idx = map(int, row_str[1:-1].split(","))
        normalized_test_point = f"({row_idx}, {col_idx})"

        current_verification_condition = row.get("Verification condition", None)
        value = row.get("Value", None)
        if value is None or pd.isna(value):
            value = None if preserve_none else 0.0
        else:
            try:
                value = float(value)
            except (ValueError, TypeError):
                value = None if preserve_none else 0.0
        computed_info_dict[f"{row['Scenario']}_{normalized_test_point}"] = {
            "value": value,
            "value_baseline": _parse_optional_float(row.get("Value baseline")),
            "verification_condition": current_verification_condition,
        }
        test_point = matrix_processing.TestPoint(
            row=row_idx,
            col=col_idx,
            test_range=matrix_processing.TestRange(row["Range"]),
            color=common.PredictionColor(row["OEM Prediction"].lower()),
            robustness_layer=row.get("Robustness layer", None),
            total_rows=total_rows,
            attributes={
                k: v
                for k, v in row.items()
                if k
                not in [
                    "Scenario",
                    "Robustness layer",
                    "Robustness",
                    "Test point",
                    "Range",
                    "OEM Prediction",
                    "Value",
                    "Value baseline",
                    "Verification condition",
                    "Test Run",
                    "Expected value",
                    "Expected value baseline",
                ]
                and pd.notna(v)
            },
        )
        test_points.append(test_point)

    return test_points, computed_info_dict


def _has_robustness_column_failures(scenario_df: pd.DataFrame) -> bool:
    if "Robustness" not in scenario_df.columns:
        return False

    current_df = scenario_df
    if "Range" in current_df.columns:
        standard_mask = (
            current_df["Range"].astype(str).str.strip().str.lower()
            == matrix_processing.TestRange.STANDARD.value.lower()
        )
        current_df = current_df.loc[standard_mask]

    if current_df.empty:
        return False

    # If no "Robustness layer" column exists, no layer was designated for testing in the
    # OEM template. Any FAIL in the "Robustness" column is spurious (e.g. ELK RE where
    # none of the YES layers appear in the LDC pool so the template has no layer column).
    if "Robustness layer" not in current_df.columns:
        return False

    # Ignore rows where the robustness layer is "Not Applicable" — no layer could be
    # selected for this test (e.g. ELK RE new-format templates).
    applicable_mask = (
        current_df["Robustness layer"].astype(str).str.strip().str.lower()
        != "not applicable"
    )
    current_df = current_df.loc[applicable_mask]

    if current_df.empty:
        return False

    robustness_values = (
        current_df["Robustness"].dropna().astype(str).str.strip().str.upper()
    )
    return (robustness_values == "FAIL").any()


@dataclass
class LoadcaseScore:
    loadcase_name: str
    standard_score: float
    extended_score: float
    robustness_layer_score: float
    total_score: float
    tested_robustness_layer: str = None
    robustness_layer_failed: bool = False
    n_applicable: float = 0.0
    claimed_layers: set = None
    robustness_score_per_layer: float = 0.0


def _set_stage_subelement_scores_to_zero(
    loadcase_score_dict, stage_element, stage_subelement, loadcase_names
):
    stage_scores = loadcase_score_dict.setdefault(stage_element, {})
    subelement_scores = stage_scores.setdefault(stage_subelement, {})

    for loadcase_name in loadcase_names:
        subelement_scores[loadcase_name] = LoadcaseScore(
            loadcase_name=loadcase_name,
            standard_score=0.0,
            extended_score=0.0,
            robustness_layer_score=0.0,
            total_score=0.0,
        )


def apply_cross_scenario_robustness_rule(loadcase_score_dict):
    """
    Implements protocol section 4.2.2:
    If a specific Robustness Layer fails in >=2 scenarios involving the same collision
    partner within the same stage subelement, it is considered failed for all scenarios
    of that collision partner within that stage subelement.
    """
    from collections import defaultdict

    failure_counts = defaultdict(int)
    for stage_element, subelement_dict in loadcase_score_dict.items():
        for stage_subelement, scenario_dict in subelement_dict.items():
            for loadcase_name, score in scenario_dict.items():
                if score.robustness_layer_failed and score.tested_robustness_layer:
                    partner = data_model.SUBTEST_TO_COLLISION_PARTNER.get(loadcase_name)
                    if partner:
                        failure_counts[
                            (
                                stage_element,
                                stage_subelement,
                                partner,
                                score.tested_robustness_layer,
                            )
                        ] += 1

    for (
        stage_element,
        stage_subelement,
        partner,
        layer,
    ), count in failure_counts.items():
        if count < 2:
            continue
        logger.info(
            "[!!!] Robustness layer '%s' fails for all '%s' scenarios in '%s / %s' "
            "(failed in %d scenarios). Applying cross-scenario propagation.",
            layer,
            partner,
            stage_element,
            stage_subelement,
            count,
        )
        scenario_dict = loadcase_score_dict.get(stage_element, {}).get(
            stage_subelement, {}
        )
        for loadcase_name, score in scenario_dict.items():
            if data_model.SUBTEST_TO_COLLISION_PARTNER.get(loadcase_name) != partner:
                continue
            if score.n_applicable == 0:
                continue
            # Already accounted for: this scenario's own tested layer is the same one
            if score.robustness_layer_failed and score.tested_robustness_layer == layer:
                continue
            # No reduction if the layer was not claimed in this scenario
            if score.claimed_layers is None or layer not in score.claimed_layers:
                continue
            # Each layer is worth total_robustness_score/n_applicable (stored as robustness_score_per_layer).
            reduction = score.robustness_score_per_layer
            old_rob = score.robustness_layer_score
            new_rob = max(0.0, old_rob - reduction)
            delta = old_rob - new_rob
            score.robustness_layer_score = new_rob
            score.total_score -= delta
            logger.info(
                "  -> Reduced robustness score for '%s' by %.4f "
                "(layer '%s' failed via cross-scenario rule). New score: %.4f",
                loadcase_name,
                delta,
                layer,
                score.robustness_layer_score,
            )


def compute_loadcase_score(
    score_test_name,
    subtest_name,
    computed_test_points,
    standard_oem_prediction_method,
    extended_oem_prediction_method,
    predicted_standard_score,
    predicted_extended_score,
    robustness_layer_score,
    has_robustness_failure=False,
):
    # Colour severity ranking, lower is better:
    # green < yellow < orange < brown < red.
    # Same order as TestPoint.predicted_score / color_score_map
    # (green 1.0 > yellow 0.75 > orange 0.5 > brown 0.25 > red 0.0), which
    # the library already applies regardless of the row's VUT speed.
    # Must stay strictly ordered -- equal ranks silently break the
    # "computed is better than predicted" check below: with ORANGE and BROWN
    # both at rank 2, a measured orange against a predicted brown was counted
    # INCORRECT.
    color_rank = {
        common.PredictionColor.GREEN: 0,
        common.PredictionColor.YELLOW: 1,
        common.PredictionColor.ORANGE: 2,
        common.PredictionColor.BROWN: 3,
        common.PredictionColor.RED: 4,
    }

    correct_count_dict = {"Standard": 0, "Extended": 0}
    eligible_count_dict = {"Standard": 0, "Extended": 0}
    for ctp in computed_test_points:

        pred_color = ctp.test_point.color
        comp_color = ctp.computed_color

        logger.debug(
            "Assessing test point: %s | pred_color: %s | comp_color: %s | assessment_criteria: %s | prediction_result: %s",
            f"{ctp.test_point.row}, {ctp.test_point.col}",
            pred_color,
            comp_color,
            ctp.assessment_criteria,
            ctp.prediction_result,
        )

        if (
            comp_color == common.PredictionColor.GREY
            or pred_color == common.PredictionColor.GREY
        ):
            logger.debug(f"---> Skip grey test point")
            continue

        if ctp.test_point.test_range == matrix_processing.TestRange.STANDARD:
            eligible_count_dict["Standard"] += 1
        elif ctp.test_point.test_range == matrix_processing.TestRange.EXTENDED:
            eligible_count_dict["Extended"] += 1
        else:
            logger.debug(f"---> Skip unknown test range")
            continue

        if ctp.assessment_criteria == robustness_layer.AssessmentCriteria.NOT_RED:
            # An absolute performance bar, so it reads the point's true colour
            # -- the measured band, no tolerance applied. FC v1.2 sec 4.2.4
            # scopes the 2 km/h tolerance to verifying whether the *prediction*
            # is correct, so the colour it applies to a within-tolerance point
            # (comp_color, the OEM's prediction) must not decide "not red":
            # that would let a measured red pass because brown was predicted.
            # measured_color is None only when there is nothing measured to
            # place in a band, where comp_color is the prediction itself.
            outcome_color = (
                ctp.measured_color if ctp.measured_color is not None else comp_color
            )
            if outcome_color != common.PredictionColor.RED:
                logger.debug(f"---> Correct! Not red following assessment criteria")
                if ctp.test_point.test_range == matrix_processing.TestRange.STANDARD:
                    correct_count_dict["Standard"] += 1
                elif ctp.test_point.test_range == matrix_processing.TestRange.EXTENDED:
                    correct_count_dict["Extended"] += 1
        elif (
            not ctp.assessment_criteria
            or ctp.assessment_criteria
            == robustness_layer.AssessmentCriteria.SAME_OR_BETTER_THAN_PREDICTED
        ):
            # Correct if computed == predicted or in tolerance
            if (
                ctp.prediction_result == matrix_processing.PredictionResult.CORRECT
                or ctp.prediction_result
                == matrix_processing.PredictionResult.IN_TOLERANCE
            ):
                logger.debug(f"---> Correct! Computed matches predicted")
                if ctp.test_point.test_range == matrix_processing.TestRange.STANDARD:
                    correct_count_dict["Standard"] += 1
                elif ctp.test_point.test_range == matrix_processing.TestRange.EXTENDED:
                    correct_count_dict["Extended"] += 1
            # Correct if computed is better (lower rank) than predicted
            elif color_rank[comp_color] < color_rank[pred_color]:
                logger.debug(f"---> Correct! Computed is better than predicted")
                if ctp.test_point.test_range == matrix_processing.TestRange.STANDARD:
                    correct_count_dict["Standard"] += 1
                elif ctp.test_point.test_range == matrix_processing.TestRange.EXTENDED:
                    correct_count_dict["Extended"] += 1

    logger.info(
        f"Number of correct or better-than-predicted test points for {subtest_name}: {correct_count_dict}"
    )

    n_standard_actual = eligible_count_dict["Standard"]
    n_extended_actual = eligible_count_dict["Extended"]

    standard_lookup = data_model.SCORE_FROM_VERIFICATION_TEST_OUTCOME["Standard"][
        standard_oem_prediction_method.value
    ]
    if n_standard_actual not in standard_lookup:
        logger.warning(
            f"Unexpected number of Standard verification test points: {n_standard_actual}. "
            f"Expected one of {sorted(standard_lookup.keys())}. Defaulting discount factor to 1.0."
        )
        standard_discount_factor = 1.0
        correct_count_standard = 0
    else:
        standard_discount_factor_thresholds = standard_lookup[n_standard_actual]
        correct_count_standard = min(correct_count_dict["Standard"], n_standard_actual)
        standard_discount_factor = (
            standard_discount_factor_thresholds[correct_count_standard] / 100
        )

    extended_lookup = data_model.SCORE_FROM_VERIFICATION_TEST_OUTCOME["Extended"][
        extended_oem_prediction_method.value
    ]
    if n_extended_actual not in extended_lookup:
        logger.warning(
            f"Unexpected number of Extended verification test points: {n_extended_actual}. "
            f"Expected one of {sorted(extended_lookup.keys())}. Defaulting discount factor to 1.0."
        )
        extended_discount_factor = 1.0
        correct_count_extended = 0
    else:
        extended_discount_factor_thresholds = extended_lookup[n_extended_actual]
        correct_count_extended = min(correct_count_dict["Extended"], n_extended_actual)
        extended_discount_factor = (
            extended_discount_factor_thresholds[correct_count_extended] / 100
        )

    some_standard_failed = correct_count_standard < n_standard_actual
    layer_failed = some_standard_failed or has_robustness_failure

    # Determine which robustness layer was tested (for cross-scenario propagation tracking)
    tested_robustness_layer = None
    for ctp in computed_test_points:
        if (
            ctp.test_point.test_range == matrix_processing.TestRange.STANDARD
            and ctp.test_point.robustness_layer
        ):
            tested_robustness_layer = ctp.test_point.robustness_layer
            break

    if layer_failed:
        robustness_layer_score.total_score = robustness_layer_score.constant_score
        logger.info(
            (
                f"[!!!] Robustness layer failed for {score_test_name}-{subtest_name} "
                f"(some_standard_failed={some_standard_failed}, has_robustness_failure={has_robustness_failure}). "
                f"Setting robustness layer score to constant_score: {robustness_layer_score.constant_score}"
            )
        )
    standard_final_score = predicted_standard_score * standard_discount_factor
    extended_final_score = predicted_extended_score * extended_discount_factor
    logger.debug(
        f"Final scores for {score_test_name}-{subtest_name}: "
        f"Standard Range: {standard_final_score} "
        f"(initial: {predicted_standard_score}, discount: {standard_discount_factor}), "
        f"Extended Range: {extended_final_score} "
        f"(initial: {predicted_extended_score}, discount: {extended_discount_factor})"
    )

    # Thresholding robustness layer score
    # It is counted only if predicted_standard_score is ≥50% of the total available score in
    # the Standard Range of that category
    total_standard_score = data_model.TOTAL_SCORES[subtest_name]["Standard"]

    if not meets_half_total(standard_final_score, total_standard_score):
        robustness_layer_score.total_score = 0.0
        logger.info(
            f"[!!!] Standard final score {standard_final_score} is below the threshold of 50% of the total standard score {total_standard_score}. Setting the robustness layer score to 0."
        )

    total_test_score = (
        standard_final_score + extended_final_score + robustness_layer_score.total_score
    )

    logger.info(
        f"Total score for {score_test_name}-{subtest_name}: {total_test_score} "
        f"(Standard: {standard_final_score}, Extended: {extended_final_score}, Robustness layer: {robustness_layer_score.total_score})"
    )
    return LoadcaseScore(
        loadcase_name=score_test_name,
        standard_score=standard_final_score,
        extended_score=extended_final_score,
        robustness_layer_score=robustness_layer_score.total_score,
        total_score=total_test_score,
        tested_robustness_layer=tested_robustness_layer,
        robustness_layer_failed=layer_failed,
        n_applicable=robustness_layer_score.n_applicable,
        claimed_layers=robustness_layer_score.claimed_layers,
        robustness_score_per_layer=robustness_layer_score.tested_score,
    )


def _compute_cpmfc_color(v_impact, v_impact_baseline):
    """CPMFC pass logic: pass if v_impact == 0,
    else pass if v_impact - v_impact_baseline <= 0.5 km/h."""
    if v_impact is None:
        return common.PredictionColor.GREY
    if v_impact == 0.0:
        return common.PredictionColor.GREEN
    if v_impact_baseline is None:
        return common.PredictionColor.GREY
    return (
        common.PredictionColor.GREEN
        if (v_impact - v_impact_baseline) <= 0.5
        else common.PredictionColor.RED
    )


def compute_lsc_loadcase_score(
    loadcase_name,
    test_points_df_dict,
    input_parameters_df,
    all_test_points,
    matrix_total_cells,
):

    computed_color_dict = {}

    if loadcase_name == "CBDA" and common.is_not_applicable(
        _get_raw_input_parameter(input_parameters_df, "Vehicle response")
    ):
        # An explicit "N/A" Vehicle response means no dooring system is
        # fitted: CBDA scores 0. Checked on the raw value because
        # get_vehicle_response collapses N/A into None (same as blank), and
        # blank must keep its current behavior (scored from OEM colors).
        logger.info("Vehicle response is N/A; CBDA scores 0.")
        return (
            LoadcaseScore(
                loadcase_name=loadcase_name,
                standard_score=0.0,
                extended_score=0.0,
                robustness_layer_score=0.0,
                total_score=0.0,
            ),
            computed_color_dict,
        )

    vehicle_response_enum = get_vehicle_response(
        {"Input parameters": input_parameters_df}
    )

    if loadcase_name not in test_points_df_dict:
        # No verification rows were selected for this scenario: nothing was
        # verified, so the OEM prediction stands. Score the prediction
        # matrix directly -- each standard cell contributes its OEM colour
        # weight, exactly as a verification row with a blank Value does.
        # Grey cells are skipped, so an all-Grey matrix
        # still scores 0.
        if not all_test_points:
            return (
                LoadcaseScore(
                    loadcase_name=loadcase_name,
                    standard_score=0.0,
                    extended_score=0.0,
                    robustness_layer_score=0.0,
                    total_score=0.0,
                ),
                computed_color_dict,
            )
        test_points, computed_info_dict = list(all_test_points), {}
        # There are no verification rows to write a Colour back into, so
        # the computed colors must not reach the caller's write-back.
        write_back_colors = False
    else:
        test_points, computed_info_dict = read_test_points(
            test_points_df_dict[loadcase_name], preserve_none=True
        )
        write_back_colors = True

    color_score_map = {
        common.PredictionColor.GREEN: 1.0,
        common.PredictionColor.YELLOW: 0.75,
        common.PredictionColor.ORANGE: 0.5,
        common.PredictionColor.BROWN: 0.25,
        common.PredictionColor.RED: 0.0,
    }

    standard_test_points = [
        tp
        for tp in test_points
        if tp.test_range == matrix_processing.TestRange.STANDARD
    ]
    standard_count = len(standard_test_points)

    lsc_final_score = 0.0
    total_standard_score = data_model.TOTAL_SCORES[loadcase_name]["Standard"]

    if standard_count == 0:
        logger.warning(
            "No STANDARD test points found for loadcase '%s'. Setting lsc_final_score to 0.0.",
            loadcase_name,
        )
        return (
            LoadcaseScore(
                loadcase_name=loadcase_name,
                standard_score=0.0,
                extended_score=0.0,
                robustness_layer_score=0.0,
                total_score=0.0,
            ),
            computed_color_dict,
        )

    if matrix_total_cells <= 0:
        logger.warning(
            "Invalid matrix size (%d cells) for loadcase '%s'. Setting lsc_final_score to 0.0.",
            matrix_total_cells,
            loadcase_name,
        )
        return (
            LoadcaseScore(
                loadcase_name=loadcase_name,
                standard_score=0.0,
                extended_score=0.0,
                robustness_layer_score=0.0,
                total_score=0.0,
            ),
            computed_color_dict,
        )

    cell_weight = 1.0 / matrix_total_cells

    for test_point in standard_test_points:
        row_key = f"{loadcase_name}_({test_point.row}, {test_point.col})"
        current_computed_dict = computed_info_dict.get(row_key, {})
        value = current_computed_dict.get("value", None)
        oem_color = test_point.color

        if oem_color == common.PredictionColor.GREY:
            logger.debug("---> Skip grey test point at %s", row_key)
            computed_color_dict[row_key] = common.PredictionColor.GREY
            continue

        if value is None:
            # Value not specified: use OEM prediction color weight directly
            scaling = color_score_map.get(oem_color, 0.0)
            computed_color_dict[row_key] = oem_color
            logger.debug(
                "LSC %s | no value | oem_color: %s | scaling: %.2f",
                row_key,
                oem_color,
                scaling,
            )
        elif loadcase_name == "CPMFC":
            value_baseline = current_computed_dict.get("value_baseline", None)
            computed_color = _compute_cpmfc_color(value, value_baseline)
            computed_color_dict[row_key] = computed_color
            scaling = color_score_map.get(computed_color, 0.0)
            logger.debug(
                "LSC %s | CPMFC | value: %s | baseline: %s | computed_color: %s | scaling: %.2f",
                row_key,
                value,
                value_baseline,
                computed_color,
                scaling,
            )
        else:
            # Value specified: evaluate via threshold logic, use computed color weight
            computed_test_point = matrix_processing.ComputedTestPoint(
                test_name=loadcase_name,
                test_point=test_point,
                value=round_half_up(value, 2),
            )
            computed_color = computed_test_point.get_lsc_computed_color(
                vehicle_response_enum
            )
            computed_color_dict[row_key] = computed_color
            scaling = color_score_map.get(computed_color, 0.0)
            logger.debug(
                "LSC %s | value: %.2f | computed_color: %s | scaling: %.2f",
                row_key,
                value,
                computed_color,
                scaling,
            )

        lsc_final_score += cell_weight * scaling

    lsc_final_score = lsc_final_score * total_standard_score

    return (
        LoadcaseScore(
            loadcase_name=loadcase_name,
            standard_score=lsc_final_score,
            extended_score=0.0,
            robustness_layer_score=0.0,
            total_score=lsc_final_score,
        ),
        computed_color_dict if write_back_colors else {},
    )


# Convert bg_color to PredictionColor
def rgb_tuple_from_hex(hex_str):
    if hex_str is None or len(hex_str) != 8:
        return None
    r = int(hex_str[2:4], 16) / 255
    g = int(hex_str[4:6], 16) / 255
    b = int(hex_str[6:8], 16) / 255
    return (r, g, b)


def get_prediction_color_from_bgcolor(bg_color_hex):
    rgb = rgb_tuple_from_hex(bg_color_hex)
    if rgb is None:
        return None
    for color, rgb_val in common.PREDICTION_COLOR_MAP.items():
        if all(abs(a - b) < 0.02 for a, b in zip(rgb, rgb_val)):
            return color
    return None


def load_verification_sheet_with_bg_colors(input_file, sheet_name):
    wb = openpyxl.load_workbook(input_file, data_only=True)
    df_combined = common.load_sheet_with_bg_colors(wb, sheet_name)
    wb.close()
    return df_combined


def compute_stage_score(dfs, stage_info, loadcase_score_dict):
    stage_element = stage_info["Stage element"]
    stage_subelement = stage_info["Stage subelement"]
    logger.debug(f"Stage info: {stage_info}")

    sheet_prefix = get_sheet_prefix(stage_element, stage_subelement)
    robustness_sheet_name = f"{sheet_prefix} robust. pred."
    verification_sheet_name = f"{sheet_prefix} verif."
    prediction_sheet_name = f"{sheet_prefix} pred."
    stage_subelement_key = get_stage_subelement_key(stage_element)

    if stage_subelement_key == data_model.StageSubelementKey.LDC:
        robustness_sheet_name = f"{stage_subelement_key.value} - robust. pred."
    elif stage_subelement_key == data_model.StageSubelementKey.LSC:
        robustness_sheet_name = None

    if stage_subelement == "Pedestrian & cyclist":
        stage_subelement = "Ped & Cyc"
    if stage_subelement == "Single vehicle":
        stage_subelement = "Single Veh"

    stage_subelement_dict = data_model.STAGE_SUBELEMENT_TO_LOADCASES[
        stage_subelement_key
    ]

    if stage_subelement not in stage_subelement_dict:
        raise ValueError(
            f"Test prefix '{stage_subelement}' not found in STAGE_SUBELEMENT_TO_LOADCASES."
        )

    all_verification_df = dfs.get(verification_sheet_name, None)
    logger.debug(
        f"verif. sheet '{verification_sheet_name}' loaded: {all_verification_df is not None}"
    )
    logger.debug(f"All verification DataFrame:\n{all_verification_df}")
    if all_verification_df is None:
        logger.warning(
            "Skipping score computation for %s - %s because sheet '%s' is missing. "
            "All affected loadcase scores will be set to 0.",
            stage_element,
            stage_subelement,
            verification_sheet_name,
        )
        _set_stage_subelement_scores_to_zero(
            loadcase_score_dict,
            stage_element,
            stage_subelement,
            stage_subelement_dict[stage_subelement],
        )
        return

    if robustness_sheet_name is not None and robustness_sheet_name not in dfs:
        raise ValueError(f"{robustness_sheet_name} sheet not found in the input data.")

    if prediction_sheet_name not in dfs:
        raise ValueError(f"{prediction_sheet_name} sheet not found in the input data.")

    # Load the verification sheet as a DataFrame, including cell background colors
    prediction_df_bg_pred = dfs[f"{prediction_sheet_name} (bg)"]
    prediction_df = dfs[prediction_sheet_name]
    logger.debug(f"Prediction sheet with bg and color: {prediction_df_bg_pred}")

    verification_test_df = data_model.get_test_point_df_from_verification_df(
        all_verification_df, sheet_prefix
    )
    logger.debug(f" sheet_prefix: {sheet_prefix}")
    logger.debug(
        f"Columns in verification_test_df: {verification_test_df.columns.tolist()}"
    )
    logger.debug(
        f"verif. test points DataFrame for {stage_subelement}:\n{verification_test_df}"
    )
    verification_test_df["Colour"] = verification_test_df["Colour"].astype(str)
    robustness_layers_df = (
        dfs[robustness_sheet_name] if robustness_sheet_name in dfs else None
    )

    input_parameters_df = dfs["Input parameters"]
    start_row_lookup = matrix_processing.build_start_row_lookup(prediction_df)
    matrix_indices_cache = {}

    # Split test_points_df into a dictionary of DataFrames based on the "Scenario" column
    test_points_df_dict = {}
    if "Scenario" in verification_test_df.columns:
        for loadcase_name in verification_test_df["Scenario"].unique():
            test_points_df_dict[loadcase_name] = verification_test_df[
                verification_test_df["Scenario"] == loadcase_name
            ].copy()

    score_set = False
    driveability_score = 0.0
    for loadcase_name in stage_subelement_dict[stage_subelement]:
        matrix_indices = matrix_processing.get_matrix_indices(
            prediction_df,
            loadcase_name,
            start_row_lookup=start_row_lookup,
            matrix_indices_cache=matrix_indices_cache,
        )
        if matrix_indices is None and not (
            stage_subelement_key == data_model.StageSubelementKey.LDC
            and loadcase_name in {"Driveability", "Driver state link"}
        ):
            if loadcase_name in data_model.SUBTEST_TO_TEST_DICT:
                score_test_name = data_model.SUBTEST_TO_TEST_DICT[loadcase_name]
            else:
                score_test_name = loadcase_name

            logger.warning(
                "No matrix available while scoring scenario '%s'. Using zero score.",
                loadcase_name,
            )
            if (
                score_test_name
                not in loadcase_score_dict[stage_element][stage_subelement]
            ):
                loadcase_score_dict[stage_element][stage_subelement][
                    score_test_name
                ] = LoadcaseScore(
                    loadcase_name=score_test_name,
                    standard_score=0.0,
                    extended_score=0.0,
                    robustness_layer_score=0.0,
                    total_score=0.0,
                )
            continue

        if stage_subelement_key == data_model.StageSubelementKey.LSC:
            robustness_layers_df = None

            all_test_points = matrix_processing.get_test_matrix_from_region(
                prediction_df_bg_pred,
                matrix_indices["start_row"],
                matrix_indices["n_rows"],
                matrix_indices["start_col"],
                matrix_indices["n_cols"],
                loadcase_name,
                stage_subelement_key,
                attributes_df=prediction_df,
            )

            loadcase_score, lsc_computed_color_dict = compute_lsc_loadcase_score(
                loadcase_name,
                test_points_df_dict,
                input_parameters_df,
                all_test_points,
                matrix_total_cells=matrix_indices["n_rows"] * matrix_indices["n_cols"],
            )
            score_test_name = loadcase_name
            for row_key, computed_color in lsc_computed_color_dict.items():
                loadcase_name_in_key, test_point_str = row_key.split("_")
                test_point_row, test_point_col = map(
                    int, test_point_str[1:-1].split(",")
                )
                logger.debug(
                    f"Updating LSC color for loadcase '{loadcase_name_in_key}', test point '{test_point_str}' to: {computed_color}"
                )
                verification_test_df.loc[
                    (verification_test_df["Scenario"] == loadcase_name)
                    & (
                        verification_test_df["Test point"]
                        == f"({test_point_row}, {test_point_col})"
                    ),
                    "Colour",
                ] = computed_color.value.capitalize()
            score_set = True

        if stage_subelement_key == data_model.StageSubelementKey.LDC:
            if loadcase_name == "Driveability":
                logger.info("Handling non-loadcase scenario: Driveability")
                # Extract relevant input parameters and assign scores
                driveability_input_param = "Heading correction"
                driveability_score = 0.0
                value = input_parameters_df.loc[
                    input_parameters_df["Input parameter"] == driveability_input_param,
                    "Value",
                ]
                input_param_check_passed = (
                    not value.empty
                    and str(value.iloc[0]).strip().lower() != "differential braking"
                )
                # Get the "Value" of the row in all_verification_df where "Scenario" is "Driveability"
                driveability_value = None
                if (
                    "Scenario" in all_verification_df.columns
                    and "Value" in all_verification_df.columns
                ):
                    driveability_row = all_verification_df[
                        all_verification_df["Scenario"] == "Driveability"
                    ]
                    if not driveability_row.empty:
                        driveability_value = driveability_row.iloc[0]["Value"]
                logger.debug(
                    f"Driveability value from verification sheet: {driveability_value}"
                )
                driveability_row_passed = (
                    driveability_value is not None
                    and str(driveability_value).strip().lower() == "pass"
                )

                if input_param_check_passed and driveability_row_passed:
                    driveability_score = 2.0

                # Store the score in loadcase_score_dict
                if stage_element not in loadcase_score_dict:
                    loadcase_score_dict[stage_element] = {}
                if stage_subelement not in loadcase_score_dict[stage_element]:
                    loadcase_score_dict[stage_element][stage_subelement] = {}

                loadcase_score_dict[stage_element][stage_subelement][loadcase_name] = (
                    LoadcaseScore(
                        loadcase_name=loadcase_name,
                        standard_score=driveability_score,
                        extended_score=0.0,
                        robustness_layer_score=0.0,
                        total_score=driveability_score,
                    )
                )
                continue
            elif loadcase_name == "Driver state link":
                logger.info("Handling non-loadcase scenario: Driver state link")
                # Get the "Value" of the row in all_verification_df where "Scenario" is "Driver state link"
                driver_state_link_value = None
                if (
                    "Scenario" in all_verification_df.columns
                    and "Value" in all_verification_df.columns
                ):
                    driver_state_link_row = all_verification_df[
                        all_verification_df["Scenario"] == "Driver state link"
                    ]
                    if not driver_state_link_row.empty:
                        driver_state_link_value = driver_state_link_row.iloc[0]["Value"]
                logger.debug(
                    f"Driver state link value from verification sheet: {driver_state_link_value}"
                )
                driver_state_link_row_passed = (
                    driver_state_link_value is not None
                    and str(driver_state_link_value).strip().lower() == "pass"
                )
                driver_state_link_score = 0.0
                if driveability_score > 0.0 and driver_state_link_row_passed:
                    driver_state_link_score = 3.0

                if stage_element not in loadcase_score_dict:
                    loadcase_score_dict[stage_element] = {}
                if stage_subelement not in loadcase_score_dict[stage_element]:
                    loadcase_score_dict[stage_element][stage_subelement] = {}

                loadcase_score_dict[stage_element][stage_subelement][loadcase_name] = (
                    LoadcaseScore(
                        loadcase_name=loadcase_name,
                        standard_score=driver_state_link_score,
                        extended_score=0.0,
                        robustness_layer_score=0.0,
                        total_score=driver_state_link_score,
                    )
                )
                continue

        if not score_set:

            if loadcase_name in data_model.SUBTEST_TO_TEST_DICT.keys():
                logger.debug(
                    f"Scenario {loadcase_name} is a subtest, mapping to main test: {data_model.SUBTEST_TO_TEST_DICT[loadcase_name]}"
                )
                subtest_name = loadcase_name
                score_test_name = data_model.SUBTEST_TO_TEST_DICT[loadcase_name]
            else:
                subtest_name = loadcase_name
                score_test_name = loadcase_name

            standard_oem_prediction_method, extended_oem_prediction_method = (
                matrix_processing.get_prediction_methods(
                    input_parameters_df, score_test_name
                )
            )
            if (
                standard_oem_prediction_method is None
                or extended_oem_prediction_method is None
            ):
                # N/A (or blank) prediction method: the scenario is not
                # claimed, so all three layers (Standard/Extended/Robustness)
                # score 0. The matrix processing and the verification Colour
                # write-back are skipped -- nothing was verified. This is the
                # only gate that excludes a claimed scenario: whether
                # verification rows exist does not decide assessment
                # membership.
                loadcase_score = LoadcaseScore(
                    loadcase_name=score_test_name,
                    standard_score=0.0,
                    extended_score=0.0,
                    robustness_layer_score=0.0,
                    total_score=0.0,
                )
            else:
                if subtest_name not in test_points_df_dict:
                    # No verification rows were selected for this scenario:
                    # nothing was verified, so the OEM prediction stands.
                    # Score through the normal path with an empty test-point
                    # set -- compute_loadcase_score then keeps the full
                    # predicted Standard/Extended score (discount factor 1.0)
                    # plus the robustness layer, instead of zeroing the
                    # scenario.
                    has_robustness_failure = False
                    test_points, computed_info_dict = [], {}
                else:
                    has_robustness_failure = _has_robustness_column_failures(
                        test_points_df_dict[subtest_name]
                    )
                    test_points, computed_info_dict = read_test_points(
                        test_points_df_dict[subtest_name],
                        preserve_none=True,
                        total_rows=matrix_indices["n_rows"],
                    )
                all_test_points = matrix_processing.get_test_matrix_from_region(
                    prediction_df_bg_pred,
                    matrix_indices["start_row"],
                    matrix_indices["n_rows"],
                    matrix_indices["start_col"],
                    matrix_indices["n_cols"],
                    loadcase_name,
                    stage_subelement_key,
                    attributes_df=prediction_df,
                )

                robustness_layer_score = compute_robustness_layer_score(
                    robustness_layers_df, score_test_name, subtest_name
                )

                predicted_standard_score, predicted_extended_score = (
                    compute_predicted_score(subtest_name, all_test_points)
                )

                logger.debug(
                    f"Scenario: {score_test_name}-{subtest_name}, "
                    f"Predicted Standard Score: {predicted_standard_score}, "
                    f"Predicted Extended Score: {predicted_extended_score}, "
                    f"Predicted Robustness Score: {robustness_layer_score}, "
                )

                computed_test_points = []
                for test_point in test_points:
                    row_key = f"{loadcase_name}_({test_point.row}, {test_point.col})"
                    current_computed_dict = computed_info_dict.get(row_key)
                    if current_computed_dict is None:
                        logger.warning(
                            f"No computed info found for key '{row_key}'. "
                            f"Available keys: {list(computed_info_dict.keys())}. "
                            f"Treating value as None."
                        )
                        current_computed_dict = {}
                    current_verification_condition = current_computed_dict.get(
                        "verification_condition", None
                    )
                    logger.debug(f"test_point.attributes: {test_point.attributes}")
                    raw_value = current_computed_dict.get("value")
                    computed_test_point = matrix_processing.ComputedTestPoint(
                        test_name=loadcase_name,
                        test_point=test_point,
                        value=(
                            round_half_up(raw_value, 2)
                            if raw_value is not None
                            else None
                        ),
                    )
                    if current_verification_condition is None:
                        computed_test_point.assessment_criteria = None
                    else:
                        computed_test_point.assessment_criteria = (
                            robustness_layer.get_assessment_criteria(
                                computed_test_point.test_point.robustness_layer,
                                loadcase_name,
                                current_verification_condition,
                            )
                        )
                    logger.debug(f"ComputedTestPoint: {computed_test_point}")
                    logger.debug(
                        f"--> computed color: {computed_test_point.computed_color}"
                    )
                    logger.debug(
                        f"--> prediction result: {computed_test_point.prediction_result}"
                    )
                    logger.debug(
                        f"--> assessment criteria: {computed_test_point.assessment_criteria}"
                    )
                    computed_test_points.append(computed_test_point)

                if (
                    stage_subelement_key == data_model.StageSubelementKey.LDC
                    and loadcase_name == "ELK RE"
                ):
                    extended_range_performance = input_parameters_df.loc[
                        input_parameters_df["Input parameter"]
                        == "Extended range performance",
                        "Value",
                    ]
                    extended_range_performance = (
                        str(extended_range_performance.iloc[0]).strip().lower()
                    )
                    # An "N/A" Extended range performance never reaches this
                    # point: get_prediction_methods zeroes the whole ELK RE
                    # scenario (its slice includes this parameter) before the
                    # matrix processing runs.
                    if extended_range_performance == "ldw":
                        predicted_extended_score = 0.5 * predicted_extended_score

                # Using computed test dict to populate Colour column in the verification sheet
                for computed_test_point in computed_test_points:
                    logger.debug(
                        f"Original prediction color: {computed_test_point.test_point.color}, Computed color: {computed_test_point.computed_color}"
                    )
                    verification_test_df.loc[
                        (verification_test_df["Scenario"] == loadcase_name)
                        & (
                            verification_test_df["Test point"]
                            == f"({computed_test_point.test_point.row}, {computed_test_point.test_point.col})"
                        ),
                        "Colour",
                    ] = computed_test_point.computed_color.value.capitalize()

                loadcase_score = compute_loadcase_score(
                    score_test_name,
                    subtest_name,
                    computed_test_points,
                    standard_oem_prediction_method,
                    extended_oem_prediction_method,
                    predicted_standard_score,
                    predicted_extended_score,
                    robustness_layer_score,
                    has_robustness_failure=has_robustness_failure,
                )
        # If already present, update the score fields; otherwise, set the entry
        if (
            stage_element in loadcase_score_dict
            and stage_subelement in loadcase_score_dict[stage_element]
            and score_test_name in loadcase_score_dict[stage_element][stage_subelement]
        ):
            logger.debug(
                f"Updating existing scenario score for {stage_element} - {stage_subelement} - {score_test_name}"
            )
            logger.debug(
                f"Previous score: {loadcase_score_dict[stage_element][stage_subelement][score_test_name]}"
            )
            loadcase_score_dict[stage_element][stage_subelement][
                score_test_name
            ].standard_score += loadcase_score.standard_score
            loadcase_score_dict[stage_element][stage_subelement][
                score_test_name
            ].extended_score += loadcase_score.extended_score
            loadcase_score_dict[stage_element][stage_subelement][
                score_test_name
            ].robustness_layer_score += loadcase_score.robustness_layer_score
            logger.debug(
                f"Updated score: {loadcase_score_dict[stage_element][stage_subelement][score_test_name]}"
            )
        else:
            logger.debug(
                f"Adding new scenario score for {stage_element} - {stage_subelement} - {score_test_name}"
            )
            logger.debug(f"New score: {loadcase_score}")
            loadcase_score_dict[stage_element][stage_subelement][
                score_test_name
            ] = loadcase_score

    all_verification_col_names = [
        col if "Unnamed" not in str(col) else ""
        for col in all_verification_df.columns.tolist()
    ]
    all_verification_df.columns = all_verification_col_names

    # Copy Colour column from verification_test_df to all_verification_df for matching Scenario and Test point
    colour_col_idx = None
    for i, col_name in enumerate(verification_test_df.columns):
        if col_name == "Colour":
            colour_col_idx = i
            break
    for idx, row in verification_test_df.iterrows():
        scenario = row.get("Scenario")
        test_point = row.get("Test point")
        color_value = row.get("Colour")
        all_verification_df.iloc[idx, colour_col_idx] = color_value

    dfs[verification_sheet_name] = all_verification_df
