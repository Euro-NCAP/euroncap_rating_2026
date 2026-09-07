# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import logging
import math
import pandas as pd
from euroncap_rating_2026 import common
import random
import io
import numpy as np
from dataclasses import dataclass
import copy
from typing import Dict, List


@dataclass
class SelectedTestPoint:
    row: int
    col: int


@dataclass
class DmCandidateRecord:
    row: int
    col: int
    attributes: dict


logger = logging.getLogger(__name__)
pd.set_option("future.no_silent_downcasting", True)

DM_COLUMNS = [
    "Long distraction",
    "Movement type",
    "Gaze location",
    "Warning",
    "Forward support",
    "Lane support",
]

NON_TRANSIENT_COLUMNS = [
    "Non transient",
    "",
    "Impairment type",
    "Warning",
    "Forward and lane support",
    "Emergency function",
]

GREY_TEXT_VALUES = {"N/A", "GREY", "GRAY"}

FORWARD_SUPPORT = "Forward support"
LANE_SUPPORT = "Lane support"
FORWARD_SUPPORT_SCENARIO_OPTIONS = ["CCRs", "CMRs", "CCRm"]
FORWARD_SUPPORT_IMPACT_LOCATIONS = [25, 50, 75]
FORWARD_SUPPORT_VUT_SPEEDS = [30, 40, 50, 60, 70, 80]
FORWARD_SUPPORT_TARGET_SPEEDS = {"CCRs": 0, "CMRs": 0, "CCRm": 20}
LANE_SUPPORT_SCENARIO = "LKA Dashed Line"


# Randomly pick n yes_cells from long_distraction.yes_cells_dict
def pick_random_yes_cells(yes_cells_dict, n):
    keys = list(yes_cells_dict.keys())
    if n > len(keys):
        n = len(keys)
    picked_keys = random.sample(keys, n)
    return {k: yes_cells_dict[k] for k in picked_keys}


GENERAL_REQUIREMENT_CSV_STRING = """Category;temp0;temp1;temp2;temp3;temp4;temp5;temp6;Value
General requirements;;;;;;;;
"""

TRANSIENT_DM_SCORING_CSV_STRING = """
Driver state,Distraction category,Glance target type,Movement type,Warning,Forward support,Lane support
Transient,Long distraction,Non-driving task,Owl,0.5,0.4,0.1
Transient,Long distraction,Non-driving task,Lizard,0.5,0.4,0.1
Transient,Long distraction,Non-driving task,Body lean,0.5,0.4,0.1
Transient,Long distraction,Driving task,Owl,,0.8,0.2
Transient,Long distraction,Driving task,Lizard,,0.8,0.2
Transient,Short distraction,Non-driving task,Owl,0.5,0.4,0.1
Transient,Short distraction,Non-driving task,Lizard,0.5,0.4,0.1
Transient,Short distraction,Driving task,Owl,,0.8,0.2
Transient,Short distraction,Driving task,Lizard,,0.8,0.2
Transient,Short distraction,Multi-target,Lizard,0.5,0.4,0.1
Transient,Phone use,Basic phone use,Owl and lizard,1.25,1.0,0.25
Transient,Phone use,Advanced phone use,Lizard,1.25,1.0,0.25
"""

NON_TRANSIENT_DM_SCORING_CSV_STRING = """
Task,Impairment type,Warning,Forward and lane support,Emergency function
Impairment,Drowsiness,0.5,1.5,
Impairment,Non-fatigue,0.5,1.5,
Microsleep,,0.5,1.5,
Sleep,,0.5,1.5,
Unresponsive,,,,2.0
"""

NOISE_VAR_TO_REQUIREMENT_STRING = {}
NOISE_VAR_TO_REQUIREMENT_STRING["Dark sunglasses"] = (
    "Inform of non functional system within 10 seconds after the driver wears dark sunglasses"
)
NOISE_VAR_TO_REQUIREMENT_STRING["Hand on wheel at 12 o'clock position"] = (
    "Inform of non functional system within 10 seconds after the driver obstructs the sensor by gripping the upper portion of the steering wheel"
)
NOISE_VAR_TO_REQUIREMENT_STRING["Cap"] = (
    "Inform of non functional system within 10 seconds after the driver wears a cap"
)
NOISE_VAR_TO_REQUIREMENT_STRING["Hat"] = (
    "Inform of non functional system within 10 seconds after the driver wears a hat"
)

ASSISTED_DRIVING_REQUIREMENT_STRING = "The system shall meet a minimum score of 50% of Driving Collaboration and  50% of Driver Monitoring in accordance to the 2026 Assisted Driving Grading Protocol"


class TableProcessor:
    def __init__(self, dm_prediction_df, df_name, required_mask_df=None):
        self.name = df_name
        self.df = common.extract_table(dm_prediction_df, self.name, 0)
        self.processed_df = self.prepare_df()
        if self.name in ["Long distraction", "Short distraction", "Phone use"]:
            self.driver_state = "Transient"
        else:
            self.driver_state = "Non-transient"
            self.processed_df.columns = NON_TRANSIENT_COLUMNS
        self.required_mask = self._build_required_mask(required_mask_df)
        self.cell_dict = self.get_cell_dict()

    def _build_required_mask(self, required_mask_df):
        if required_mask_df is None:
            return None
        mask_table = common.extract_table(required_mask_df, self.name, 0)
        if mask_table.empty:
            return None
        mask_table.columns = (
            DM_COLUMNS if self.driver_state == "Transient" else NON_TRANSIENT_COLUMNS
        )
        return mask_table[2:].copy()

    def deepcopy(self):
        """
        Returns a deep copy of the TableProcessor instance.
        """
        new_instance = TableProcessor.__new__(TableProcessor)
        new_instance.name = copy.deepcopy(self.name)
        new_instance.df = copy.deepcopy(self.df)
        new_instance.processed_df = copy.deepcopy(self.processed_df)
        new_instance.driver_state = copy.deepcopy(self.driver_state)
        new_instance.required_mask = copy.deepcopy(self.required_mask)
        new_instance.cell_dict = copy.deepcopy(self.cell_dict)
        return new_instance

    def prepare_df(self):
        self.df.columns = DM_COLUMNS
        df = self.df[2:].copy()
        # Only forward-fill the specified columns if they exist in the DataFrame
        columns_to_ffill = [
            "Noise variables",
            "Category",
            "Long distraction",
            "Movement type",
            "Short distraction",
            "Phone use",
            "Non-transient",
        ]
        existing_cols = [col for col in columns_to_ffill if col in df.columns]
        df.loc[:, existing_cols] = df[existing_cols].ffill(axis=0)
        return df

    def get_cell_dict(self):
        cells_dict = {}
        for row_idx, row in self.processed_df.iterrows():
            for col_idx, col_name in enumerate(self.processed_df.columns):
                cell_value = row[col_name]
                normalized = str(cell_value).strip().upper()
                is_required_blank = (
                    normalized in ("", "NAN", "NONE")
                    and self.required_mask is not None
                    and row_idx in self.required_mask.index
                    and bool(self.required_mask.loc[row_idx, col_name])
                )
                if normalized in ("GREEN", "RED"):
                    color = normalized
                elif is_required_blank:
                    color = "RED"
                elif normalized in GREY_TEXT_VALUES:
                    color = "GREY"
                else:
                    continue
                header_row_keys = ["Task", "Movement type", "Gaze location"]
                header_row_values = [
                    row[self.processed_df.columns[i]] for i in range(3)
                ]
                header_dict = dict(zip(header_row_keys, header_row_values))
                cells_dict[(row_idx, col_idx)] = {
                    **header_dict,
                    "Vehicle response": self.processed_df.columns[col_idx],
                    "Color": color,
                }

        return cells_dict


def _attributes_match(candidate_attrs: dict, input_attrs: dict) -> bool:
    for key, input_val in input_attrs.items():
        if key not in candidate_attrs:
            continue
        if str(candidate_attrs[key]).strip() != str(input_val).strip():
            return False
    return True


def get_candidate_records(table_processor: TableProcessor) -> List[DmCandidateRecord]:
    return [
        DmCandidateRecord(
            row=row_idx,
            col=col_idx,
            attributes=dict(cell_info),
        )
        for (row_idx, col_idx), cell_info in table_processor.cell_dict.items()
    ]


def select_cells_from_input(
    table_processor: TableProcessor, input_points: List[dict], group_name: str
) -> Dict:
    selected_cells = {}
    candidate_records = get_candidate_records(table_processor)

    for input_attrs in input_points:
        matched_record = None
        for candidate in candidate_records:
            if _attributes_match(candidate.attributes, input_attrs):
                matched_attrs = dict(candidate.attributes)
                for key, value in input_attrs.items():
                    if key not in matched_attrs:
                        matched_attrs[key] = value
                matched_record = ((candidate.row, candidate.col), matched_attrs)
                break

        if matched_record is None:
            logger.warning(
                "[DM %s] No candidate cell found matching attributes %s. Skipping.",
                group_name,
                input_attrs,
            )
            continue

        selected_cells[matched_record[0]] = matched_record[1]

    return dict(sorted(selected_cells.items(), key=lambda item: item[0][0]))


def select_warning_test_points(table_processor, valid_tasks=List) -> Dict:
    task_movement_test_num = (
        table_processor.processed_df.groupby(
            [
                table_processor.processed_df.columns[0],
                table_processor.processed_df.columns[1],
            ]
        )
        .size()
        .reset_index(name="Count")
    )
    task_movement_test_num["warning_test_points_num"] = task_movement_test_num[
        "Count"
    ].apply(lambda x: math.ceil(x / 2))

    logger.debug(
        f"Task movement test num with warning_test_points_num:\n{task_movement_test_num}"
    )
    logger.debug(f"Table processor processed_df:\n{table_processor.processed_df}")
    warning_test_points = []
    for _, row in task_movement_test_num.iterrows():
        if row["Long distraction"] not in valid_tasks:
            continue
        task = row["Long distraction"]
        movement_type = row[table_processor.processed_df.columns[1]]
        # Find all cell_dict entries matching current task and movement type, saving coords and cell_info
        matching_points = [
            ((coords, cell_info))
            for coords, cell_info in table_processor.cell_dict.items()
            if cell_info.get("Task") == task
            and cell_info.get("Movement type") == movement_type
            and cell_info.get("Vehicle response") == "Warning"
            and cell_info.get("Color") == "GREEN"
        ]

        warning_test_points_num = row["warning_test_points_num"]
        # Randomly sample warning_test_points_num from matching_points
        sampled_points = random.sample(
            matching_points, min(warning_test_points_num, len(matching_points))
        )
        warning_test_points.extend(sampled_points)

    warning_test_dict = {point[0]: point[1] for point in warning_test_points}

    # Sort warning_test_dict by the first element in the key tuple
    sorted_warning_test_dict = dict(
        sorted(warning_test_dict.items(), key=lambda item: item[0][0])
    )
    logger.debug(
        f"Selected warning test points for {table_processor.name}: {sorted_warning_test_dict}"
    )
    return sorted_warning_test_dict


def iter_prediction_groups(table_processors: Dict[str, "TableProcessor"]):
    """Yield every transient (Task, Movement type, Vehicle response) group as
    ``(table_name, table_processor, (task, movement_type, vehicle_response),
    coords_list)``. The grouping the consistency rule is defined over --
    shared by the enforcing apply_prediction_consistency below and the
    report-only collector (consistency.find_prediction_inconsistencies).
    Non-transient has no movement type and is intentionally not yielded."""
    for name in ("Long distraction", "Short distraction", "Phone use"):
        tp = table_processors.get(name)
        if tp is None:
            continue
        groups = {}
        for coords, info in tp.cell_dict.items():
            key = (
                info.get("Task"),
                info.get("Movement type"),
                info.get("Vehicle response"),
            )
            groups.setdefault(key, []).append(coords)
        for key, coords_list in groups.items():
            yield name, tp, key, coords_list


def apply_prediction_consistency(table_processors: Dict[str, "TableProcessor"]) -> None:
    """Enforce prediction consistency per movement type / intervention strategy.

    For each transient table (Long distraction, Short distraction, Phone use), group the
    prediction cells by (Task, Movement type, Vehicle response) and, if ANY cell in a group
    is RED, force ALL cells in that group to RED (this includes GREEN and GREY cells alike).
    Because selection filters on ``Color == "GREEN"``, a forced-RED group is simply never
    selected (by warning or intervention selection alike).

    Mutates each TableProcessor's ``cell_dict`` in place. Must run before any selection.
    Non-transient has no movement type and is intentionally left untouched.
    """
    total_groups = 0
    flipped_groups = 0
    for (
        name,
        tp,
        (task, movement_type, vehicle_response),
        coords_list,
    ) in iter_prediction_groups(table_processors):
        total_groups += 1
        red_coords = [c for c in coords_list if tp.cell_dict[c].get("Color") == "RED"]
        if red_coords and len(red_coords) < len(coords_list):
            flipped_groups += 1
            logger.info(
                "[DM consistency] %s: forcing group (Task=%s, Movement type=%s, "
                "Vehicle response=%s) to RED — %d/%d cells were already RED",
                name,
                task,
                movement_type,
                vehicle_response,
                len(red_coords),
                len(coords_list),
            )
            for c in coords_list:
                tp.cell_dict[c]["Color"] = "RED"
    logger.info(
        "[DM consistency] Inspected %d (Task, Movement type, Vehicle response) groups "
        "across transient tables, forced %d mixed group(s) to RED",
        total_groups,
        flipped_groups,
    )


def get_prediction_forced_red_coordinates(
    table_processors: Dict[str, "TableProcessor"],
) -> List:
    """Excel (row, col) coordinates (1-based) of every "DE - DM pred." cell whose
    final Color is RED after ``apply_prediction_consistency`` has run.

    This covers cells the OEM explicitly marked Red, required-but-blank cells, and
    cells forced RED because they share a (Task, Movement type, Vehicle response)
    group with one of those -- i.e. every cell that must render as a failure in
    the prediction sheet, not just the one that originally triggered it.

    ``TableProcessor.cell_dict`` keys are (row_idx, col_idx) into the table as read
    by pandas (header=0, columns A-F): row_idx 0 is Excel row 2, col_idx 0 is
    Excel column A.
    """
    coordinates = []
    for tp in table_processors.values():
        for (row_idx, col_idx), info in tp.cell_dict.items():
            if info.get("Color") == "RED":
                coordinates.append((row_idx + 2, col_idx + 1))
    return coordinates


def _max_bipartite_matching(candidate_sets):
    """Maximum bipartite matching between pick indices and candidate values.

    ``candidate_sets[i]`` is the set of values pick ``i`` could take. Returns
    ``{pick_index: value}`` for the largest possible set of picks that can
    each be assigned a mutually distinct value (Kuhn's algorithm).
    """
    match_of_value = {}  # value -> pick index

    def augment(pick_idx, visited):
        candidates = list(candidate_sets[pick_idx])
        random.shuffle(candidates)
        for value in candidates:
            if value in visited:
                continue
            visited.add(value)
            if value not in match_of_value or augment(match_of_value[value], visited):
                match_of_value[value] = pick_idx
                return True
        return False

    pick_order = list(range(len(candidate_sets)))
    random.shuffle(pick_order)
    for pick_idx in pick_order:
        augment(pick_idx, set())

    return {pick_idx: value for value, pick_idx in match_of_value.items()}


def select_intervention_test_points(
    table_processors: List["TableProcessor"], max_total: int = 8
) -> Dict[str, Dict[tuple, dict]]:
    """Select the forward / lane support (intervention) test points.

    Two-phase selection:

      Phase A (selection): decides WHICH (Task, Movement type) groups are picked
      and with WHICH strategy (forward/lane support), via the same rules as before:
      2. Driving-task movement types with a mixed prediction (exactly one of
         forward/lane GREEN, the other RED) are handled first and forced to their
         only-GREEN strategy.
      3/4. The remaining (all-GREEN) driving movement types are filled with a
         balance-alternating choice (the strategy with fewer selections so far,
         coin-flip tie) -- one test per movement type, covering all driving types.
      5/6. Non-driving + phone movement types are then filled with the same
         balance-alternating rule, PER STRATEGY (a movement type is eligible for a
         strategy only if that strategy is GREEN), until max_total total or the
         needed strategy has no eligible movement type left.
      7. If still under max_total, fill from any remaining GREEN movement
         type/strategy (relaxed) until max_total is reached or no GREEN groups
         remain.
      A movement type yields at most one intervention test.

      Phase B (gaze assignment): once every pick's (group, strategy) is finalized,
      a global maximum bipartite matching (picks <-> gaze locations) assigns each
      pick a gaze location, maximizing the number of distinct gaze locations used
      across the whole selection. Picks left unmatched (only possible when a
      duplicate is mathematically unavoidable given the picks' candidates) fall
      back to a random candidate of their own.

    Returns ``{table_processor.name: {coords: cell_info}}``.
    """
    # Groups keyed by (tp.name, Task, Movement type); only GREEN forward/lane cells are
    # collected, so both-red movement types never appear.
    driving_groups = {}
    other_groups = {}
    for tp in table_processors:
        for coords, cell_info in tp.cell_dict.items():
            support = cell_info.get("Vehicle response")
            if support not in (FORWARD_SUPPORT, LANE_SUPPORT):
                continue
            if cell_info.get("Color") != "GREEN":
                continue
            key = (tp.name, cell_info.get("Task"), cell_info.get("Movement type"))
            bucket = (
                driving_groups
                if cell_info.get("Task") == "Driving task"
                else other_groups
            )
            group = bucket.setdefault(key, {FORWARD_SUPPORT: [], LANE_SUPPORT: []})
            group[support].append(
                (cell_info.get("Gaze location"), tp, coords, cell_info)
            )

    selected = {tp.name: {} for tp in table_processors}
    counts = {FORWARD_SUPPORT: 0, LANE_SUPPORT: 0}
    total = {"n": 0}
    picks = []  # list of (tp_name, candidates) where candidates = group[strategy]

    def reserve(group, strategy, tp_name):
        picks.append((tp_name, group[strategy]))
        counts[strategy] += 1
        total["n"] += 1

    def target_strategy(available):
        return min(available, key=lambda s: (counts[s], random.random()))

    # --- Driving-task selection (steps 2-4) ---
    driving_keys = list(driving_groups.keys())
    random.shuffle(driving_keys)
    used = set()

    def assign_driving(keys, forced):
        remaining = list(keys)
        while remaining:
            key = remaining.pop()
            g = driving_groups[key]
            fs, ls = bool(g[FORWARD_SUPPORT]), bool(g[LANE_SUPPORT])
            strategy = (
                (FORWARD_SUPPORT if fs else LANE_SUPPORT)
                if forced
                else target_strategy([FORWARD_SUPPORT, LANE_SUPPORT])
            )
            reserve(g, strategy, key[0])
            used.add(key)

    forced_keys = [
        k
        for k in driving_keys
        if bool(driving_groups[k][FORWARD_SUPPORT])
        != bool(driving_groups[k][LANE_SUPPORT])
    ]
    assign_driving(forced_keys, forced=True)
    assign_driving([k for k in driving_keys if k not in used], forced=False)

    # --- Steps 5/6: non-driving + phone, balance-alternating, per strategy ---
    other_keys = list(other_groups.keys())
    random.shuffle(other_keys)
    while total["n"] < max_total:
        strategy = target_strategy([FORWARD_SUPPORT, LANE_SUPPORT])
        cands = [k for k in other_keys if k not in used and other_groups[k][strategy]]
        if not cands:
            logger.info(
                "[DM intervention] Strict alternation stopped at %d/%d selections — "
                "no eligible group left for %s",
                total["n"],
                max_total,
                strategy,
            )
            break
        key = random.choice(cands)
        reserve(other_groups[key], strategy, key[0])
        used.add(key)

    # --- Step 7: relaxed fill toward max_total from any remaining GREEN group ---
    relaxed_added = 0
    while total["n"] < max_total:
        cands = [
            k
            for k in other_keys
            if k not in used
            and (other_groups[k][FORWARD_SUPPORT] or other_groups[k][LANE_SUPPORT])
        ]
        if not cands:
            break
        key = random.choice(cands)
        g = other_groups[key]
        avail = [s for s in (FORWARD_SUPPORT, LANE_SUPPORT) if g[s]]
        reserve(g, target_strategy(avail), key[0])
        used.add(key)
        relaxed_added += 1
    if relaxed_added:
        logger.info(
            "[DM intervention] Relaxed fill added %d more selection(s) to reach %d/%d total",
            relaxed_added,
            total["n"],
            max_total,
        )

    # --- Phase B: globally-optimal gaze assignment across all finalized picks ---
    def is_valid_gaze(gaze):
        return gaze is not None and str(gaze).strip() and str(gaze).lower() != "nan"

    candidate_sets = [
        {gaze for gaze, _tp, _coords, _info in candidates if is_valid_gaze(gaze)}
        for _tp_name, candidates in picks
    ]
    matching = _max_bipartite_matching(candidate_sets)

    for i, (_tp_name, candidates) in enumerate(picks):
        if i in matching:
            gaze, tp, coords, cell_info = next(
                c for c in candidates if c[0] == matching[i]
            )
        else:
            gaze, tp, coords, cell_info = random.choice(candidates)
        selected[tp.name][coords] = cell_info
        logger.debug(
            "[DM intervention] Assigned gaze=%s to %s (Task=%s, Movement type=%s)",
            gaze,
            tp.name,
            cell_info.get("Task"),
            cell_info.get("Movement type"),
        )

    for name in selected:
        selected[name] = dict(
            sorted(selected[name].items(), key=lambda item: item[0][0])
        )

    distinct_gaze_used = len(
        {
            cell_info.get("Gaze location")
            for cells in selected.values()
            for cell_info in cells.values()
        }
    )
    logger.info(
        "[DM intervention] Selection summary: total=%d (max=%d), FS=%d, LS=%d, "
        "distinct gaze locations used=%d/%d, per-table counts=%s",
        total["n"],
        max_total,
        counts[FORWARD_SUPPORT],
        counts[LANE_SUPPORT],
        distinct_gaze_used,
        total["n"],
        {name: len(cells) for name, cells in selected.items()},
    )
    return selected


def build_forward_support_scenario() -> str:
    """Return one randomly selected forward-support scenario string."""
    scenario_type = random.choice(FORWARD_SUPPORT_SCENARIO_OPTIONS)
    impact_location = random.choice(FORWARD_SUPPORT_IMPACT_LOCATIONS)
    vut_speed = random.choice(FORWARD_SUPPORT_VUT_SPEEDS)
    target_speed = FORWARD_SUPPORT_TARGET_SPEEDS[scenario_type]
    scenario = (
        f"{scenario_type}, IL {impact_location}%, "
        f"VUT {vut_speed} km/h, Target {target_speed} km/h"
    )
    logger.debug(f"[DM scenario] Drew forward-support scenario: {scenario}")
    return scenario


def get_verification_rows(
    table_processor: TableProcessor,
    selected_cells: Dict,
    non_transient=False,
    scenarios: Dict = None,
) -> List[Dict]:
    verification_rows = []
    for yes_cell in selected_cells.items():
        coords, cell_info = yes_cell

        is_warning = cell_info.get("Vehicle response").lower() == "warning"

        driver_state = table_processor.driver_state
        task = cell_info.get("Task")
        movement_type = cell_info.get("Movement type")
        processor_name = table_processor.name

        if non_transient:
            category_str = f"{driver_state} - {task}"
            if is_warning:
                scenario_str = "Warning"
            else:
                scenario_str = "Intervention"
        else:
            # For Phone use, task is either Basic phone use or Advanced phone use,
            # Only first part is needed for scenario string (e.g. Basic phone use -> Basic)
            task_scenario_str = task
            movement_type_scenario_str = movement_type
            if processor_name in ["Phone use"]:
                task_scenario_str = str(task).split()[0]
                if movement_type in ["Owl", "Lizard"] and task_scenario_str == "Basic":
                    movement_type_scenario_str = "Owl and lizard"

            category_str = f"{driver_state} - {processor_name}"
            if is_warning:
                scenario_str = (
                    f"{task_scenario_str} - {movement_type_scenario_str} - Warning"
                )
            else:
                scenario_str = (
                    f"{task_scenario_str} - {movement_type_scenario_str} - Intervention"
                )

        # Create one row using verification_df_cols mapping elements from cell_info
        verification_row = {
            "Category": category_str,
            "Scenario": scenario_str,
            "Test point": common.format_test_point(coords[0] + 2, coords[1] + 1),
            # "Driver state": table_processor.driver_state,
            # "Distraction category": table_processor.name,
            "Glance target type": task,
            "Movement type": movement_type,
            "Gaze location": cell_info.get("Gaze location"),
            "Noise variable": None,
            "Requirement": cell_info.get("Vehicle response"),
            "Value": np.nan,
            "Test scenario": (scenarios or {}).get(cell_info.get("Vehicle response")),
        }
        verification_rows.append(verification_row)
    return verification_rows


def get_non_transient_points(table_processor):
    non_transient_points = []
    for coords, cell_info in table_processor.cell_dict.items():
        if (
            cell_info.get("Task") == "Microsleep"
            and cell_info.get("Vehicle response") == "Warning"
            and cell_info.get("Color") == "GREEN"
        ):
            non_transient_points.append((coords, cell_info))
        if (
            cell_info.get("Task") == "Sleep"
            and cell_info.get("Vehicle response") == "Warning"
            and cell_info.get("Color") == "GREEN"
        ):
            non_transient_points.append((coords, cell_info))
        if (
            cell_info.get("Task") == "Unresponsive"
            and cell_info.get("Vehicle response") == "Emergency function"
            and cell_info.get("Color") == "GREEN"
        ):
            non_transient_points.append((coords, cell_info))

    sleep_tasks = ["Microsleep", "Sleep"]
    matching_points = [
        ((coords, cell_info))
        for coords, cell_info in table_processor.cell_dict.items()
        if cell_info.get("Task") in sleep_tasks
        and cell_info.get("Vehicle response") == "Forward and lane support"
        and cell_info.get("Color") == "GREEN"
    ]

    # Randomly sample 1 from matching_points if 2 are available otherwise use the sole point
    sampled_points = random.sample(matching_points, min(1, len(matching_points)))
    non_transient_points.extend(sampled_points)

    non_transient_test_dict = {point[0]: point[1] for point in non_transient_points}

    # Sort warning_test_dict by the first element in the key tuple
    sorted_non_transient_test_dict = dict(
        sorted(non_transient_test_dict.items(), key=lambda item: item[0][0])
    )

    return sorted_non_transient_test_dict


def select_dm_test_points(
    dm_prediction_df, input_selected_points=None, required_mask_df=None
) -> Dict:
    """Standalone driver-monitoring test-point selection.

    Performs the complete selection from the prediction, for all transient and
    non-transient runs, so it can be plugged into ``preprocess`` (or exercised on
    its own for testing). It only SELECTS test points -- it does not touch
    scoring, occlusion assignment or report formatting.

    Selection steps (reusing the existing, unchanged helpers where applicable):
      * transient warning  -> ``select_warning_test_points`` per distraction
        (or ``select_cells_from_input`` when explicit points are provided);
      * transient forward/lane intervention -> ``select_intervention_test_points``
        (skipped -- returns ``{}`` per table -- when explicit points are provided,
        since ``select_cells_from_input`` already picks up any explicitly-listed
        Forward/Lane support cells alongside the warning cells for that table);
      * non-transient -> ``get_non_transient_points``
        (or ``select_cells_from_input`` when explicit points are provided).

    It also draws the forward-support scenario once for the transient runs and
    once (independently) for the non-transient run.

    Returns a dict::

        {
          "table_processors": {name: TableProcessor},
          "warning_points": {name: {coords: cell_info}},        # transient
          "intervention_points": {name: {coords: cell_info}},   # transient
          "non_transient_points": {coords: cell_info},
          "transient_scenarios": {"Forward support": str, "Lane support": str},
          "non_transient_scenarios": {"Forward and lane support": str},
        }
    """
    long_distraction = TableProcessor(
        dm_prediction_df, "Long distraction", required_mask_df=required_mask_df
    )
    short_distraction = TableProcessor(
        dm_prediction_df, "Short distraction", required_mask_df=required_mask_df
    )
    phone_use = TableProcessor(
        dm_prediction_df, "Phone use", required_mask_df=required_mask_df
    )
    non_transient = TableProcessor(
        dm_prediction_df, "Non-transient", required_mask_df=required_mask_df
    )
    table_processors = {
        "Long distraction": long_distraction,
        "Short distraction": short_distraction,
        "Phone use": phone_use,
        "Non-transient": non_transient,
    }

    # Consistency check BEFORE any selection: a movement type / strategy group that
    # is not fully GREEN is forced fully RED (so it is never selected).
    apply_prediction_consistency(table_processors)

    using_input_points = input_selected_points is not None
    warning_valid_tasks = [
        "Non-driving task",
        "Basic phone use",
        "Advanced phone use",
        "Multi-target",
    ]

    # --- transient warning selection ---
    warning_points = {}
    for name in ["Long distraction", "Short distraction", "Phone use"]:
        tp = table_processors[name]
        if using_input_points:
            warning_points[name] = select_cells_from_input(
                tp, input_selected_points.get(name, []), name
            )
        else:
            warning_points[name] = select_warning_test_points(
                tp, valid_tasks=warning_valid_tasks
            )

    # --- transient forward/lane intervention selection ---
    if using_input_points:
        intervention_points = {
            "Long distraction": {},
            "Short distraction": {},
            "Phone use": {},
        }
    else:
        intervention_points = select_intervention_test_points(
            [long_distraction, short_distraction, phone_use],
            max_total=8,
        )

    # --- non-transient selection ---
    if using_input_points:
        non_transient_points = select_cells_from_input(
            non_transient,
            input_selected_points.get("Non-transient", []),
            "Non-transient",
        )
    else:
        non_transient_points = get_non_transient_points(non_transient)

    # --- scenarios: one draw for transient, one (independent) for non-transient ---
    transient_scenarios = {
        "Forward support": build_forward_support_scenario(),
        "Lane support": LANE_SUPPORT_SCENARIO,
    }
    non_transient_scenarios = {
        "Forward and lane support": build_forward_support_scenario(),
    }

    return {
        "table_processors": table_processors,
        "warning_points": warning_points,
        "intervention_points": intervention_points,
        "non_transient_points": non_transient_points,
        "transient_scenarios": transient_scenarios,
        "non_transient_scenarios": non_transient_scenarios,
    }


def sample_and_assign_noise_vars(combined_rows, noise_vars_permuted, sample_size=3):
    """
    Randomly sample sample_size rows from the combination of warning_rows and driving_rows,
    and assign noise variables in a round-robin fashion to the existing rows.
    """
    # Filter out rows that match the specific exclusion criteria
    combined_rows_to_consider = [
        row
        for row in combined_rows
        if not (
            row.get("Distraction category") == "Long distraction"
            and row.get("Glance target type") == "Non-driving task"
            and row.get("Movement type") == "Body lean"
        )
        and row.get("Noise variable") is None
    ]

    sampled_rows = []
    if len(combined_rows_to_consider) > 0:
        sampled_rows = random.sample(
            combined_rows_to_consider, min(sample_size, len(combined_rows_to_consider))
        )

    noise_vars_permuted = random.sample(noise_vars_permuted, len(noise_vars_permuted))
    for idx, row in enumerate(sampled_rows):
        row["Noise variable"] = noise_vars_permuted[idx % len(noise_vars_permuted)]

    logger.info(f"Sampled {sample_size} rows and assigned noise variables")
    reorder_list(combined_rows)


def assign_noise_vars_non_transient(non_transient_rows, noise_vars_permuted):
    if len(non_transient_rows) == 0:
        return

    sleep_microsleep_rows = [
        row
        for row in non_transient_rows
        if row.get("Glance target type") in ["Microsleep", "Sleep"]
    ]
    unresponsive_rows = [
        row
        for row in non_transient_rows
        if row.get("Glance target type") == "Unresponsive"
    ]

    noise_vars_permuted = random.sample(noise_vars_permuted, len(noise_vars_permuted))
    sampled_sleep_rows = random.sample(
        sleep_microsleep_rows, min(2, len(sleep_microsleep_rows))
    )
    for idx, row in enumerate(sampled_sleep_rows):
        row["Noise variable"] = noise_vars_permuted[idx % len(noise_vars_permuted)]

    noise_vars_permuted = random.sample(noise_vars_permuted, len(noise_vars_permuted))
    for idx, row in enumerate(unresponsive_rows):
        row["Noise variable"] = noise_vars_permuted[idx % len(noise_vars_permuted)]


def reorder_list(rows):
    # Move all rows with Requirement == 'Warning' to the top
    warning_rows = [row for row in rows if row.get("Requirement") == "Warning"]
    other_rows = [row for row in rows if row.get("Requirement") != "Warning"]
    rows = warning_rows + other_rows
    return rows


MANDATORY_FUNCTIONAL_ELEMENTS = [
    "Daytime - Night-time",
    "Clear sunglasses",
    "Short facial hair",
    "Long facial hair",
    "Face-mask",
]

MANDATORY_FUNCTIONAL_CATEGORIES = []


def get_mandatory_performance_states(noise_variables_df):
    """
    Classify each mandatory element/category's Performance cell into
    "functional", "non_functional", or "blank" (not yet assessed by the
    OEM). Blank must stay distinct from an explicit "Non functional" --
    callers scoring the result should treat blank as unassessed (NaN), not
    as a confirmed fail (0).

    Args:
        noise_variables_df: DataFrame containing noise variables data

    Returns:
        dict: Mandatory element/category names to "functional"/"non_functional"/"blank"
    """
    mandatory_states = {}

    for mandatory_element in MANDATORY_FUNCTIONAL_ELEMENTS:
        mandatory_element_df = noise_variables_df[
            noise_variables_df["Element"] == mandatory_element
        ]

        if len(mandatory_element_df) > 0:
            current_performance = mandatory_element_df["Performance"].values[0]
            if common.is_empty_cell(current_performance):
                state = "blank"
            elif current_performance.strip().lower() == "functional":
                state = "functional"
            else:
                state = "non_functional"
        else:
            state = "non_functional"
        logger.debug(f"Mandatory element '{mandatory_element}' state: {state}")
        mandatory_states[mandatory_element] = state

    for mandatory_category in MANDATORY_FUNCTIONAL_CATEGORIES:
        mandatory_category_df = noise_variables_df[
            noise_variables_df["Category"] == mandatory_category
        ]

        if len(mandatory_category_df) > 0:
            current_performance = mandatory_category_df["Performance"].values[0]
            if common.is_empty_cell(current_performance):
                state = "blank"
            elif current_performance.strip().lower() == "functional":
                state = "functional"
            else:
                state = "non_functional"
        else:
            state = "non_functional"
        logger.debug(f"Mandatory category '{mandatory_category}' state: {state}")
        mandatory_states[mandatory_category] = state

    return mandatory_states


def get_mandatory_check_dict(noise_variables_df):
    """
    Check if mandatory elements and categories have functional performance.

    Args:
        noise_variables_df: DataFrame containing noise variables data

    Returns:
        dict: Dictionary with mandatory element/category names as keys and functional status as values
    """
    mandatory_states = get_mandatory_performance_states(noise_variables_df)
    return {name: state == "functional" for name, state in mandatory_states.items()}


def preprocess(
    dm_prediction_df, param_df, input_selected_points=None, required_mask_df=None
):
    common.validate_input_selected_points(input_selected_points, context="DM")

    noise_variables_df = common.extract_table(dm_prediction_df, "Noise variables", 0)

    mandatory_check_dict = get_mandatory_check_dict(noise_variables_df)

    if not all(mandatory_check_dict.values()):
        failed_variables = [name for name, ok in mandatory_check_dict.items() if not ok]
        logger.info(
            f"Mandatory noise variable(s) non functional: {failed_variables}; skipping test point selection."
        )
        message_df = pd.DataFrame(
            {
                "Category": [
                    f"Mandatory noise variable {name} was set to Non functional. "
                    "No verification tests should be performed since the score will be 0."
                    for name in failed_variables
                ]
            }
        )
        return message_df, [], []

    noise_var_subset = ["Dark sunglasses", "Face-mask", "Cap", "Hat"]
    noise_variables_subset_df = noise_variables_df[
        noise_variables_df["Element"].isin(noise_var_subset)
    ]
    functional_df = noise_variables_subset_df[
        noise_variables_subset_df["Performance"].str.lower() == "functional"
    ]

    additional_general_requirements_subset = [
        "Dark sunglasses",
        "Hand on wheel at 12 o'clock position",
        "Cap",
        "Hat",
    ]
    additional_general_requirements_subset_df = noise_variables_df[
        noise_variables_df["Element"].isin(additional_general_requirements_subset)
    ]
    non_functional_df = additional_general_requirements_subset_df[
        additional_general_requirements_subset_df["Performance"].str.lower()
        == "non functional"
    ]
    logger.info(f"Functional rows in subset:\n{functional_df}")
    logger.info(f"Non-functional rows in subset:\n{non_functional_df}")

    non_functional_noise_vars = []
    if len(non_functional_df) > 0:
        logger.info(
            "There are non-functional rows among selected noise variables. New general requirements rows need to be generated"
        )
        non_functional_noise_vars = non_functional_df["Element"].tolist()
        logger.info(f"Non-functional noise variables: {non_functional_noise_vars}")

    functional_noise_vars = functional_df["Element"].tolist()
    # Create a random permutation of functional_noise_vars
    functional_noise_vars_permuted = random.sample(
        functional_noise_vars, len(functional_noise_vars)
    )
    logger.info(f"Functional noise variables: {functional_noise_vars_permuted}")

    if required_mask_df is None:
        required_mask_df = common.get_default_dm_prediction_required_mask()

    dm_selection = select_dm_test_points(
        dm_prediction_df,
        input_selected_points=input_selected_points,
        required_mask_df=required_mask_df,
    )
    table_processors = dm_selection["table_processors"]
    long_distraction = table_processors["Long distraction"]
    short_distraction = table_processors["Short distraction"]
    phone_use = table_processors["Phone use"]
    non_transient = table_processors["Non-transient"]

    long_distraction_warning_points = dm_selection["warning_points"]["Long distraction"]
    short_distraction_warning_points = dm_selection["warning_points"][
        "Short distraction"
    ]
    phone_use_warning_points = dm_selection["warning_points"]["Phone use"]
    non_transient_points = dm_selection["non_transient_points"]

    long_distraction_intervention_points = dm_selection["intervention_points"][
        "Long distraction"
    ]
    short_distraction_intervention_points = dm_selection["intervention_points"][
        "Short distraction"
    ]
    phone_use_intervention_points = dm_selection["intervention_points"]["Phone use"]

    transient_scenarios = dm_selection["transient_scenarios"]
    non_transient_scenarios = dm_selection["non_transient_scenarios"]

    # Passed to BOTH warning and intervention verification-row calls: when explicit
    # input_selected_points are used, select_cells_from_input may return a mix of
    # Warning/Forward support/Lane support cells inside the same "warning_points" dict
    # (intervention_points is {} in that mode), so the scenario lookup must apply there too.
    # For genuine Warning cells this is a no-op since "Warning" is never a key in the dict.
    intervention_scenarios = {
        "Forward support": transient_scenarios["Forward support"],
        "Lane support": transient_scenarios["Lane support"],
    }

    long_distraction_warning_rows = get_verification_rows(
        long_distraction,
        long_distraction_warning_points,
        scenarios=intervention_scenarios,
    )
    short_distraction_warning_rows = get_verification_rows(
        short_distraction,
        short_distraction_warning_points,
        scenarios=intervention_scenarios,
    )
    phone_use_warning_rows = get_verification_rows(
        phone_use, phone_use_warning_points, scenarios=intervention_scenarios
    )

    long_distraction_intervention_rows = get_verification_rows(
        long_distraction,
        long_distraction_intervention_points,
        scenarios=intervention_scenarios,
    )
    short_distraction_intervention_rows = get_verification_rows(
        short_distraction,
        short_distraction_intervention_points,
        scenarios=intervention_scenarios,
    )
    phone_use_intervention_rows = get_verification_rows(
        phone_use, phone_use_intervention_points, scenarios=intervention_scenarios
    )

    non_transient_rows = get_verification_rows(
        non_transient,
        non_transient_points,
        non_transient=True,
        scenarios={
            "Forward and lane support": non_transient_scenarios[
                "Forward and lane support"
            ]
        },
    )

    # Collect all keys from the *_points dicts and create dataclass list
    selected_test_points = []
    for points_dict in [
        long_distraction_warning_points,
        short_distraction_warning_points,
        phone_use_warning_points,
        long_distraction_intervention_points,
        short_distraction_intervention_points,
        phone_use_intervention_points,
        non_transient_points,
    ]:
        for row_idx, col_idx in points_dict.keys():
            selected_test_points.append(SelectedTestPoint(row=row_idx, col=col_idx))

    for functional_noise_var in functional_noise_vars_permuted:
        logger.info(f"Considering functional noise variable: {functional_noise_var}")

    long_distraction_rows = (
        long_distraction_warning_rows + long_distraction_intervention_rows
    )
    short_distraction_rows = (
        short_distraction_warning_rows + short_distraction_intervention_rows
    )
    phone_use_rows = phone_use_warning_rows + phone_use_intervention_rows
    # Call the function
    sample_and_assign_noise_vars(
        long_distraction_rows,
        functional_noise_vars_permuted,
        sample_size=3,
    )
    sample_and_assign_noise_vars(
        short_distraction_rows,
        functional_noise_vars_permuted,
        sample_size=3,
    )
    sample_and_assign_noise_vars(
        phone_use_rows,
        functional_noise_vars_permuted,
        sample_size=3,
    )
    assign_noise_vars_non_transient(non_transient_rows, functional_noise_vars_permuted)

    verification_rows = []

    verification_rows.extend(long_distraction_rows)
    verification_rows.extend(short_distraction_rows)
    verification_rows.extend(phone_use_rows)
    verification_rows.extend(non_transient_rows)

    verification_df = pd.DataFrame(verification_rows)
    verification_df = common.opposite_ffill(
        verification_df,
        exclude_columns=[
            "Value",
            "Requirement",
            "Noise variable",
            "Gaze location",
            "Test scenario",
        ],
    )
    logger.info(f"Generated verification DataFrame:\n{verification_df}")

    general_requirement_df = pd.read_csv(
        io.StringIO(GENERAL_REQUIREMENT_CSV_STRING), sep=";"
    )

    # Convert to numpy arrays and stack
    verification_columns = verification_df.columns.tolist()
    if general_requirement_df.shape[1] < len(verification_columns):
        for column_index in range(
            general_requirement_df.shape[1], len(verification_columns)
        ):
            general_requirement_df[f"temp{column_index}"] = ""
    general_requirement_df = general_requirement_df.iloc[:, : len(verification_columns)]

    gen_cols = general_requirement_df.columns.tolist()
    for idx, col in enumerate(gen_cols):
        if "temp" in col:
            gen_cols[idx] = ""

    # Insert an empty row between the two DataFrames
    empty_row = pd.DataFrame(
        [[""] * len(verification_columns)], columns=verification_columns
    )

    # Create one row with verification_df column names
    verification_df_col_row = pd.DataFrame(
        [verification_columns], columns=verification_columns
    )

    # Convert to numpy arrays and stack
    final_df_verification_df = pd.DataFrame(
        np.vstack(
            [
                general_requirement_df.values,
                empty_row.values,
                verification_df_col_row.values,
                verification_df.values,
            ]
        ),
        columns=gen_cols,
    )
    forced_red_coordinates = get_prediction_forced_red_coordinates(table_processors)
    return final_df_verification_df, selected_test_points, forced_red_coordinates


def clean_df_warning_score(df, verified_scenarios=None):
    # For Long distraction only: if Warning score is 0, zero out the corresponding
    # Intervention scores. Per WG agreement, this cascade does NOT apply to other
    # distraction categories (e.g. Phone use), where warning and intervention are
    # scored independently.
    # The cascade is also skipped for any Intervention scenario that was independently
    # verified (i.e. present in verified_scenarios). When verified_scenarios is None
    # the old behaviour (always cascade) is preserved for backward compatibility.
    for idx, row in df.iterrows():
        scenario = row.get("Scenario", "")
        category = row.get("Category", "")
        score = row.get("Score", 0)
        if "Long distraction" not in category:
            continue
        if (
            isinstance(scenario, str)
            and scenario.strip().endswith("Warning")
            and score == 0
        ):
            scenario_prefix = " - ".join(scenario.strip().split(" - ")[:-1])
            # Find all matching rows with same Category and Scenario prefix, but not "Warning"
            mask = (
                (df["Category"] == category)
                & (df["Scenario"].str.startswith(scenario_prefix))
                & (~df["Scenario"].str.endswith("Warning"))
            )
            for other_idx in df[mask].index:
                if verified_scenarios is not None:
                    other_key = (
                        df.at[other_idx, "Category"],
                        df.at[other_idx, "Scenario"],
                    )
                    if other_key in verified_scenarios:
                        logger.debug(
                            f"Skipping cascade for '{other_key}' — Intervention was independently verified"
                        )
                        continue
                if df.at[other_idx, "Score"] != 0:
                    logger.debug(
                        f"Setting Score to 0 for Category='{category}', Scenario='{df.at[other_idx, 'Scenario']}' because corresponding Warning is 0"
                    )
                df.at[other_idx, "Score"] = 0
    return df


def compute_transient_score(
    transient_df, dm_prediction_df, dm_colors, transient_score_df
):
    ##################################################################################
    # Transient scoring
    ##################################################################################
    # If Glance target type is 'Basic phone use', set Movement type to 'Owl and lizard'
    # As in NCAP protocol scoring it is indicated that for phone use tasks,
    # the movement type is always Owl and lizard
    transient_df.loc[
        transient_df["Glance target type"] == "Basic phone use", "Movement type"
    ] = "Owl and lizard"

    transient_grouped = transient_df.groupby(["Category", "Scenario"])

    long_distraction_table_processor = TableProcessor(
        dm_prediction_df, "Long distraction"
    )
    short_distraction_table_processor = TableProcessor(
        dm_prediction_df, "Short distraction"
    )
    phone_use_table_processor = TableProcessor(dm_prediction_df, "Phone use")
    table_processor_dict = {
        "Long distraction": long_distraction_table_processor,
        "Short distraction": short_distraction_table_processor,
        "Phone use": phone_use_table_processor,
    }

    transient_dm_scoring_df = pd.read_csv(io.StringIO(TRANSIENT_DM_SCORING_CSV_STRING))

    """
    For each pair of (glance target type, movement type) e.g. (Non-Driving Task - Owl)
    For each vehicle response vs among columns [Warning, FS, LS]
    If all cells of vs are green
    AND
    All verification rows for vs (if any) are PASS
    Then assign score (usually 0.5, 0.4, 0.1)
    Warning and Intervention are scored independently EXCEPT for Long distraction:
    for Long distraction only, if warning score is 0 then FS and LS scores are also
    zeroed out (per WG agreement, see clean_df_warning_score). For Phone use, warning
    and intervention are fully independent.

    On top of that (per updated Euro NCAP requirements): for Long and Short distraction
    specifically, if a movement type's Warning predictions are all GREEN but at least one
    Warning verification test still failed, the corresponding intervention is forced to 0
    as well -- FS+LS for Long distraction, LS only for Short distraction -- unconditionally,
    even if FS/LS had their own independently-passing verification. This is tracked via
    warning_hard_fail and applied in the Intervention branch below; it is independent of
    (and unaffected by) the separate red-prediction cascade in clean_df_warning_score.
    """

    verified_scenarios = set()
    warning_hard_fail = {}

    for score_idx, score_row in transient_score_df.iterrows():

        group_key = score_row["Scenario"]
        logger.info(f"Processing group: {group_key}")

        scenario_substrings = group_key.split(" - ")

        glance_target_type = scenario_substrings[0]
        movement_type = scenario_substrings[1]
        requirement_str = scenario_substrings[2]

        category_substrings = score_row["Category"].split(" - ")
        transient_str = category_substrings[0]
        task_str = category_substrings[1]

        logger.debug(f"#### Glance target type: {glance_target_type}")
        logger.debug(f"#### Movement type: {movement_type}")
        logger.debug(f"#### Requirement: {requirement_str}")
        logger.debug(f"#### Transient string: {transient_str}")
        logger.debug(f"#### Task string for table processor lookup: {task_str}")
        group_df = (
            transient_grouped.get_group(
                (
                    f"{transient_str} - {task_str}",
                    f"{glance_target_type} - {movement_type} - {requirement_str}",
                )
            )
            if (
                f"{transient_str} - {task_str}",
                f"{glance_target_type} - {movement_type} - {requirement_str}",
            )
            in transient_grouped.groups
            else pd.DataFrame()
        )

        if len(group_df) > 0:
            verified_scenarios.add((score_row["Category"], score_row["Scenario"]))

        if task_str == "Phone use":
            glance_target_type = glance_target_type + " phone use"

        current_table_processor = table_processor_dict.get(task_str)

        current_table_processor.processed_df.ffill(inplace=True)
        current_prediction_df = current_table_processor.processed_df
        current_prediction_df.rename(
            columns={"Long distraction": "Glance target type"}, inplace=True
        )

        # For phone use tasks with Basic phone use glance target type, movement type is set to Owl and lizard,
        # so we need to filter prediction df with an OR condition on movement type to get the relevant rows for scoring
        if movement_type == "Owl and lizard":
            current_prediction_df = current_prediction_df[
                (current_prediction_df["Glance target type"] == glance_target_type)
                & (
                    (current_prediction_df["Movement type"] == "Owl")
                    | (current_prediction_df["Movement type"] == "Lizard")
                )
            ]
        else:
            current_prediction_df = current_prediction_df[
                (current_prediction_df["Glance target type"] == glance_target_type)
                & (current_prediction_df["Movement type"] == movement_type)
            ]

        if requirement_str == "Intervention":
            # Forward and Lane support are scored independently per Step 7 protocol.
            fs_colors = []
            ls_colors = []
            for idx in current_prediction_df.index:
                if idx in dm_colors:
                    fs_colors.append(dm_colors[idx].get("Forward support"))
                    ls_colors.append(dm_colors[idx].get("Lane support"))

            fs_green = bool(fs_colors) and all(
                c == common.PredictionColor.GREEN for c in fs_colors
            )
            ls_green = bool(ls_colors) and all(
                c == common.PredictionColor.GREEN for c in ls_colors
            )

            movement_prefix = f"{glance_target_type} - {movement_type}"
            hard_fail = warning_hard_fail.get(
                (score_row["Category"], movement_prefix), False
            )
            if hard_fail and task_str == "Long distraction":
                fs_green = False
                ls_green = False
                logger.info(
                    f"  - Forcing FS and LS to 0 for {score_row['Category']} / "
                    f"{movement_prefix}: Warning hard-fail cascade "
                    "(predictions GREEN but a verification test failed)"
                )
            elif hard_fail and task_str == "Short distraction":
                ls_green = False
                logger.info(
                    f"  - Forcing LS to 0 for {score_row['Category']} / "
                    f"{movement_prefix}: Warning hard-fail cascade "
                    "(predictions GREEN but a verification test failed)"
                )

            logger.debug(
                f"Forward support green: {fs_green}, Lane support green: {ls_green}"
            )

            # For driving-task gaze locations (and empty verification groups), test pass
            # status is independent of prediction color per Step 7 scoring rules.
            if glance_target_type == "Driving task" or len(group_df) == 0:
                fs_test_pass = True
                ls_test_pass = True
                fs_test_blank = False
                ls_test_blank = False
            else:
                fs_states = group_df[group_df["Requirement"] == "Forward support"][
                    "Value"
                ].apply(common.classify_pass_fail_value)
                ls_states = group_df[group_df["Requirement"] == "Lane support"][
                    "Value"
                ].apply(common.classify_pass_fail_value)
                fs_test_pass = (fs_states == "pass").all()
                ls_test_pass = (ls_states == "pass").all()
                fs_test_blank = (fs_states == "blank").any()
                ls_test_blank = (ls_states == "blank").any()

            if fs_test_blank or ls_test_blank:
                # At least one Forward/Lane support verification row hasn't
                # been assessed by the OEM yet -- leave this scenario
                # unassessed (NaN) rather than scoring it as a FAIL (0.0).
                transient_score_df.at[score_idx, "Score"] = np.nan
                continue

            scoring_filter = (
                transient_dm_scoring_df["Glance target type"] == glance_target_type
            ) & (transient_dm_scoring_df["Movement type"] == movement_type)
            matching = transient_dm_scoring_df[scoring_filter]
            fs_max = (
                float(matching["Forward support"].values[0])
                if len(matching) > 0
                else 0.0
            )
            ls_max = (
                float(matching["Lane support"].values[0]) if len(matching) > 0 else 0.0
            )
            if pd.isna(fs_max):
                fs_max = 0.0
            if pd.isna(ls_max):
                ls_max = 0.0

            candidate_score = (fs_max if (fs_green and fs_test_pass) else 0.0) + (
                ls_max if (ls_green and ls_test_pass) else 0.0
            )
            logger.info(
                f"  - Intervention: fs_green={fs_green}, ls_green={ls_green}, "
                f"fs_pass={fs_test_pass}, ls_pass={ls_test_pass}, score={candidate_score} "
                f"for {score_row['Category']}-{score_row['Scenario']}"
            )
            transient_score_df.at[score_idx, "Score"] = candidate_score
        else:
            # Warning: all prediction cells must be GREEN and all verification tests must PASS.
            prediction_colors = []
            for idx in current_prediction_df.index:
                if idx in dm_colors:
                    prediction_colors.append(dm_colors[idx].get(requirement_str))

            all_prediction_green = bool(prediction_colors) and all(
                color == common.PredictionColor.GREEN for color in prediction_colors
            )
            logger.debug(f"All prediction green: {all_prediction_green}")

            if len(group_df) == 0:
                logger.debug(
                    "No verification rows for this group, setting all_test_pass to True by default"
                )
                all_test_pass = True
                any_test_blank = False
                any_test_failed = False
            else:
                test_states = group_df[group_df["Requirement"] == requirement_str][
                    "Value"
                ].apply(common.classify_pass_fail_value)
                all_test_pass = (test_states == "pass").all()
                any_test_blank = (test_states == "blank").any()
                any_test_failed = (test_states == "fail").any()

            if any_test_blank:
                # At least one Warning verification row hasn't been assessed
                # by the OEM yet -- leave this scenario unassessed (NaN)
                # rather than scoring it as a FAIL (0.0), and don't cascade a
                # hard-fail onto Intervention over a mere "not yet assessed".
                transient_score_df.at[score_idx, "Score"] = np.nan
                continue

            if (
                task_str in ("Long distraction", "Short distraction")
                and all_prediction_green
                and any_test_failed
            ):
                warning_hard_fail[
                    (score_row["Category"], f"{glance_target_type} - {movement_type}")
                ] = True
                logger.info(
                    f"  - Warning hard-fail: predictions all GREEN but a verification "
                    f"test failed for {score_row['Category']} / {glance_target_type} - "
                    f"{movement_type} — forcing FS/LS to 0 for this movement type"
                )

            candidate_score = float(
                transient_dm_scoring_df[
                    (
                        transient_dm_scoring_df["Glance target type"]
                        == glance_target_type
                    )
                    & (transient_dm_scoring_df["Movement type"] == movement_type)
                ][requirement_str].values[0]
            )
            if pd.isna(candidate_score):
                candidate_score = 0.0

            if all_prediction_green and all_test_pass:
                logger.info(
                    f"  - all cells GREEN and all tests PASS, setting {candidate_score} to {score_row['Category']}-{score_row['Scenario']}"
                )
                transient_score_df.at[score_idx, "Score"] = float(candidate_score)

    transient_score_df = clean_df_warning_score(transient_score_df, verified_scenarios)

    return transient_score_df


def compute_non_transient_score(
    non_transient_df, dm_prediction_df, dm_colors, non_transient_score_df
):
    ##################################################################################
    # Non Transient scoring
    ##################################################################################
    non_transient_dm_scoring_df = pd.read_csv(
        io.StringIO(NON_TRANSIENT_DM_SCORING_CSV_STRING)
    )

    # # Add Drowsiness row to non_transient_df if it doesn't exist
    drowsiness_row = pd.DataFrame([{"Glance target type": "Drowsiness"}])
    non_fatigue_row = pd.DataFrame([{"Glance target type": "Non-fatigue"}])
    if "Drowsiness" not in non_transient_df["Glance target type"].values:
        non_transient_df = pd.concat(
            [drowsiness_row, non_transient_df], ignore_index=True
        )
    if "Non-fatigue" not in non_transient_df["Glance target type"].values:
        non_transient_df = pd.concat(
            [non_fatigue_row, non_transient_df], ignore_index=True
        )

    non_transient_grouped = non_transient_df.groupby(["Category", "Scenario"])
    non_transient_table_processor = TableProcessor(dm_prediction_df, "Non-transient")

    non_transient_table_processor.processed_df.ffill(inplace=True)
    non_transient_processed_df = non_transient_table_processor.processed_df
    non_transient_processed_df.rename(columns={"Non transient": "Task"}, inplace=True)
    logger.debug(f"non_transient_processed_df:\n{non_transient_processed_df}")

    for score_idx, score_row in non_transient_score_df.iterrows():

        group_key = score_row["Scenario"]
        logger.info(f"Processing group: {group_key}")

        category_substrings = score_row["Category"].split(" - ")
        transient_str = category_substrings[0]
        task_str = category_substrings[1]
        impairment_type = None
        if task_str == "Impairment":
            sceario_substrings = group_key.split(" - ")
            impairment_type = sceario_substrings[0]
            requirement_str = sceario_substrings[1]
        else:
            requirement_str = group_key

        if requirement_str == "Intervention":
            if task_str == "Unresponsive":
                requirement_str = "Emergency function"
            else:
                requirement_str = "Forward and lane support"

        logger.debug(f"#### Transient string: {transient_str}")
        logger.debug(f"#### Task string for table processor lookup: {task_str}")
        logger.debug(f"#### Impairment type: {impairment_type}")
        logger.debug(f"#### Requirement: {requirement_str}")

        group_df = (
            non_transient_grouped.get_group(
                (f"{transient_str} - {task_str}", requirement_str)
            )
            if (f"{transient_str} - {task_str}", requirement_str)
            in non_transient_grouped.groups
            else pd.DataFrame()
        )
        non_transient_table_processor = TableProcessor(
            dm_prediction_df, "Non-transient"
        )

        ###
        if impairment_type in ["Drowsiness", "Non-fatigue"]:
            current_prediction_df = non_transient_processed_df[
                (non_transient_processed_df["Task"] == "Impairment")
                & (non_transient_processed_df["Impairment type"] == impairment_type)
            ]
        else:
            current_prediction_df = non_transient_processed_df[
                (non_transient_processed_df["Task"] == task_str)
            ]

        prediction_colors = []
        for idx in current_prediction_df.index:
            if idx in dm_colors:
                current_color = dm_colors[idx].get(requirement_str)
                prediction_colors.append(current_color)

        all_prediction_green = bool(prediction_colors) and all(
            color == common.PredictionColor.GREEN for color in prediction_colors
        )
        logger.debug(f"All prediction green: {all_prediction_green}")

        all_test_pass = False
        any_test_blank = False

        if task_str == "Impairment":
            # Impairment scoring is based on prediction only (Step 7 scoring rules).
            all_test_pass = True
        elif len(group_df) == 0:
            logger.debug(
                "No verification rows for this group, setting all_test_pass to True by default"
            )
            all_test_pass = True
        else:
            test_states = group_df[group_df["Requirement"] == requirement_str][
                "Value"
            ].apply(common.classify_pass_fail_value)
            all_test_pass = (test_states == "pass").all()
            any_test_blank = (test_states == "blank").any()

        if any_test_blank:
            # At least one verification row hasn't been assessed by the OEM
            # yet -- leave this scenario unassessed (NaN) rather than
            # scoring it as a FAIL (0.0).
            non_transient_score_df.at[score_idx, "Score"] = np.nan
            continue

        if task_str == "Impairment":
            candidate_score = float(
                non_transient_dm_scoring_df[
                    (non_transient_dm_scoring_df["Task"] == task_str)
                    & (
                        non_transient_dm_scoring_df["Impairment type"]
                        == impairment_type
                    )
                ][requirement_str].values[0]
            )
        else:
            candidate_score = float(
                non_transient_dm_scoring_df[
                    (non_transient_dm_scoring_df["Task"] == task_str)
                ][requirement_str].values[0]
            )

        if pd.isna(candidate_score):
            candidate_score = 0.0
        if all_prediction_green and all_test_pass:
            logger.info(
                f"  - all cells GREEN and all tests PASS, setting {candidate_score} to {score_row['Category']}-{score_row['Scenario']}"
            )
            non_transient_score_df.at[score_idx, "Score"] = float(candidate_score)

    non_transient_score_df = clean_df_warning_score(non_transient_score_df)

    return non_transient_score_df


def get_dm_colors(cell_colors_raw, len_noise_variables_df):
    dm_score_colors_raw = dict(cell_colors_raw)
    dm_score_colors_raw = {
        k: v for k, v in dm_score_colors_raw.items() if not k[0] in ["A", "B", "C"]
    }
    dm_score_colors_raw = {
        k: v
        for k, v in dm_score_colors_raw.items()
        if int(k[1:]) >= len_noise_variables_df + 5
    }
    dm_colors = {}
    for k, v in dm_score_colors_raw.items():
        col_letter = k[0]
        row_number = int(k[1:])
        # Adjust row number to match driver_state_df indexing (subtract header rows and noise variable rows)
        row_number = row_number - 2
        if row_number < 66:
            if col_letter == "D":
                response_type = "Warning"
            elif col_letter == "E":
                response_type = "Forward support"
            elif col_letter == "F":
                response_type = "Lane support"
        else:
            if col_letter == "D":
                response_type = "Warning"
            elif col_letter == "E":
                response_type = "Forward and lane support"
            elif col_letter == "F":
                response_type = "Emergency function"

        if row_number not in dm_colors:
            dm_colors[row_number] = {}
        dm_colors[row_number][response_type] = v
    return dm_colors


def get_dm_colors_from_text(dm_prediction_df, len_noise_variables_df):
    """Build dm_colors from cell text (GREEN/RED) — the same source used by preprocess.

    Matches the row/column mapping of get_dm_colors so the returned dict is
    interchangeable with it.  Templates that store predictions as text on a
    neutral fill (e.g. gray) are handled correctly because we read the string
    value rather than the background fill color.
    """
    min_row = len_noise_variables_df + 3
    text_to_color = {
        "GREEN": common.PredictionColor.GREEN,
        "RED": common.PredictionColor.RED,
    }
    dm_colors = {}
    for row_number, row in dm_prediction_df.iterrows():
        if row_number < min_row:
            continue
        if row_number < 66:
            col_map = {3: "Warning", 4: "Forward support", 5: "Lane support"}
        else:
            col_map = {
                3: "Warning",
                4: "Forward and lane support",
                5: "Emergency function",
            }
        row_data = {}
        for col_idx, response_type in col_map.items():
            if col_idx < len(row):
                cell_val = row.iloc[col_idx]
                if isinstance(cell_val, str):
                    color = text_to_color.get(cell_val.strip().upper())
                    if color is not None:
                        row_data[response_type] = color
        if row_data:
            dm_colors[row_number] = row_data
    return dm_colors


def get_dm_colors_combined(
    dm_prediction_df, len_noise_variables_df, cell_colors_raw=None
):
    """Build dm_colors from whichever source actually carries the prediction color.

    A freshly-authored input template marks predictions with literal GREEN/RED text
    on a neutral fill, so ``get_dm_colors_from_text`` finds everything. But the
    ``sd_preprocessed_template.xlsx`` produced by the ``preprocess`` step (the file the
    ``compute-score`` CLI command is required to run against) has already been through
    report_writer's ``format_prediction_cells`` formatting, which paints the real fill
    color and blanks the cell text (leaving only an "X" on selected test points) — so
    only the background fill still carries the color there. Combine both sources,
    preferring text when present, so either input shape scores correctly.
    """
    dm_colors = {}
    if cell_colors_raw:
        dm_colors = get_dm_colors(cell_colors_raw, len_noise_variables_df)
    for row_number, row_colors in get_dm_colors_from_text(
        dm_prediction_df, len_noise_variables_df
    ).items():
        dm_colors.setdefault(row_number, {}).update(row_colors)
    return dm_colors


def compute_score(dfs):

    transient_score_data = [
        ("Transient - Long distraction", "Non-driving task - Owl - Warning"),
        ("Transient - Long distraction", "Non-driving task - Owl - Intervention"),
        ("Transient - Long distraction", "Non-driving task - Lizard - Warning"),
        ("Transient - Long distraction", "Non-driving task - Lizard - Intervention"),
        ("Transient - Long distraction", "Non-driving task - Body lean - Warning"),
        ("Transient - Long distraction", "Non-driving task - Body lean - Intervention"),
        ("Transient - Long distraction", "Driving task - Owl - Intervention"),
        ("Transient - Long distraction", "Driving task - Lizard - Intervention"),
        ("Transient - Short distraction", "Non-driving task - Owl - Warning"),
        ("Transient - Short distraction", "Non-driving task - Owl - Intervention"),
        ("Transient - Short distraction", "Non-driving task - Lizard - Warning"),
        ("Transient - Short distraction", "Non-driving task - Lizard - Intervention"),
        ("Transient - Short distraction", "Driving task - Owl - Intervention"),
        ("Transient - Short distraction", "Driving task - Lizard - Intervention"),
        ("Transient - Short distraction", "Multi-target - Lizard - Warning"),
        ("Transient - Short distraction", "Multi-target - Lizard - Intervention"),
        ("Transient - Phone use", "Basic - Owl and lizard - Warning"),
        ("Transient - Phone use", "Basic - Owl and lizard - Intervention"),
        ("Transient - Phone use", "Advanced - Lizard - Warning"),
        ("Transient - Phone use", "Advanced - Lizard - Intervention"),
    ]
    non_transient_score_data = [
        ("Non-transient - Impairment", "Drowsiness - Warning"),
        ("Non-transient - Impairment", "Drowsiness - Intervention"),
        ("Non-transient - Impairment", "Non-fatigue - Warning"),
        ("Non-transient - Impairment", "Non-fatigue - Intervention"),
        ("Non-transient - Microsleep", "Warning"),
        ("Non-transient - Microsleep", "Intervention"),
        ("Non-transient - Sleep", "Warning"),
        ("Non-transient - Sleep", "Intervention"),
        ("Non-transient - Unresponsive", "Intervention"),
    ]

    transient_score_df = pd.DataFrame(
        transient_score_data, columns=["Category", "Scenario"]
    )
    transient_score_df["Score"] = 0.0

    non_transient_score_df = pd.DataFrame(
        non_transient_score_data, columns=["Category", "Scenario"]
    )
    non_transient_score_df["Score"] = 0.0

    score_df = pd.concat(
        [transient_score_df, non_transient_score_df], ignore_index=True
    )

    if "DE - DM pred." not in dfs or "DE - DM verif." not in dfs:
        logger.warning(
            "DE - DM pred. or DE - DM verif. sheet is missing, cannot compute driver monitoring score."
        )
        return score_df
    dm_prediction_df = dfs["DE - DM pred."]
    dm_verification_df = dfs["DE - DM verif."]

    noise_variables_df = common.extract_table(dm_prediction_df, "Noise variables", 0)

    mandatory_states = get_mandatory_performance_states(noise_variables_df)

    # Not yet assessed by the OEM -- unassessed (NaN), not a silent 0/FAIL.
    if any(state == "blank" for state in mandatory_states.values()):
        logger.info(
            "Mandatory noise variable(s) not yet assessed. Driver Monitoring score remains unassessed."
        )
        logger.info(f"Mandatory check results: {mandatory_states}")
        score_df["Score"] = np.nan
        return score_df

    if not all(state == "functional" for state in mandatory_states.values()):
        logger.info(
            "There are non functional performance rows, returning score dict with all zeros."
        )
        logger.info(f"Mandatory check results: {mandatory_states}")
        return score_df

    logger.info("Computing driver monitoring score as noise variable checks passed.")

    # Extract General requirements section
    general_req_rows = common.extract_section(
        dm_verification_df, "General requirements", col_name="Category"
    )
    # Remove all rows where every value is NaN
    general_req_rows = general_req_rows.dropna(how="all")

    if general_req_rows.empty:
        logging.warning("General requirements section is missing.")

    general_req_states = general_req_rows["Value"].apply(
        common.classify_pass_fail_value
    )

    if (general_req_states == "blank").any():
        logging.info(
            "General requirements not yet assessed. Leaving Driver Monitoring unassessed."
        )
        score_df["Score"] = np.nan
        return score_df

    # If any general requirement failed, return score dict with all zeros
    if (general_req_states == "fail").any():
        logging.info("One or more General requirements failed. Keeping score to 0.")
        return score_df

    driver_state_header_idx = 2
    # Create driver_state_df from dm_verification_df starting from row 9 (header)
    driver_state_df = pd.DataFrame(
        dm_verification_df.values[driver_state_header_idx + 1 :],
        columns=dm_verification_df.iloc[driver_state_header_idx].tolist(),
    )
    driver_state_df = driver_state_df.dropna(how="all")

    driver_state_df.ffill(inplace=True)

    # Separate two groups based on Glance target type as they have different scoring dict
    transient_glance_target_types = [
        "Non-driving task",
        "Driving task",
        "Multi-target",
        "Basic phone use",
        "Advanced phone use",
    ]
    non_transient_glance_target_types = [
        "Microsleep",
        "Sleep",
        "Unresponsive",
    ]

    transient_df = driver_state_df[
        driver_state_df["Glance target type"].isin(transient_glance_target_types)
    ].copy()
    non_transient_df = driver_state_df[
        driver_state_df["Glance target type"].isin(non_transient_glance_target_types)
    ].copy()

    dm_colors = get_dm_colors_combined(
        dm_prediction_df,
        len(noise_variables_df),
        cell_colors_raw=dfs.get("DE - DM pred. (cell_colors)"),
    )

    transient_score_df = compute_transient_score(
        transient_df, dm_prediction_df, dm_colors, transient_score_df
    )

    non_transient_score_df = compute_non_transient_score(
        non_transient_df, dm_prediction_df, dm_colors, non_transient_score_df
    )

    score_df = pd.concat(
        [transient_score_df, non_transient_score_df], ignore_index=True
    )
    logger.debug(f"Final transient scoring df:\n{score_df}")
    return score_df
