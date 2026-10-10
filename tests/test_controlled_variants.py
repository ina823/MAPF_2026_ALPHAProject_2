"""Tests for scenarios/controlled_variants/ (Alpha 2 failure-analysis expansion).

Mirrors the TestControlledScenarios pattern in test_failure_analysis.py, but
kept in its own file/dict so the frozen scenarios/controlled/ instances and
their PLANS in test_failure_analysis.py are never touched:

  * TestControlledVariantsWellFormed -- grid values, free-cell starts/goals,
                                        no duplicate start, no duplicate goal.
  * TestControlledVariantsSolvable   -- each variant's hand-written joint plan
                                        (U/D/L/R/W per agent) is replayed in the
                                        unmodified MAPFStepSimulator and must
                                        produce zero vertex/edge/wall conflicts,
                                        no position overlap at any timestep, and
                                        all agents on their goal at the end.

Verification status: passing TestControlledVariantsSolvable records a
scenario as ``verified_via_joint_plan_replay`` (VERIFICATION_STATUS below).
This is NOT a CBS proof -- this repository has no integrated CBS solver
(``CBSAdapter`` is referenced only as a future hook in scenario_loader.py /
mapf_step_simulator.py, not implemented). ``CBS_verified`` stays "no" for
every entry until a real CBS run confirms it independently; a passing replay
here must never be reported as CBS-confirmed solvability.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.common.scenario_loader import load_map_generator_scenario
from src.simulator.mapf_step_simulator import MAPFStepSimulator

UP, DOWN, LEFT, RIGHT, WAIT = 0, 1, 2, 3, 4
_ACTION = {"U": UP, "D": DOWN, "L": LEFT, "R": RIGHT, "W": WAIT}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIANTS_DIR = PROJECT_ROOT / "scenarios" / "controlled_variants"

# Hand-written joint plans, each found by an offline search (prioritized
# time-expanded planning, falling back to exhaustive joint-state BFS for the
# two 4-agent instances where a fixed priority order could not find a
# conflict-free schedule) and then replayed against the real, unmodified
# MAPFStepSimulator to confirm zero conflicts -- see
# scenarios/controlled_variants/README.md for the per-variant design notes.
PLANS = {
    "corridor_bay_n2_3agent": {0: "WWWWWWRRRRRRRRRWW", 1: "WWWWWWRRRRRRRRRWW",
                                2: "LLLLLUWWWWWDLLLLL"},
    "corridor_bay_n2_offset": {0: "RRRRRRRRRRW", 1: "LUWWWDLLLLL"},
    "corridor_bay_n2_4agent": {0: "RLRLRRWRRRUWDRRRR", 1: "RRRRUWDRRRLLRRRRW",
                                2: "LLLWLLLRRRLLLLLLW", 3: "LLLWLLLRRRLLLLLLW"},
    "intersection_n2_4way": {0: "DUDUDDDDDDW", 1: "UUDUULWRUUU",
                              2: "RRRUWDRRLRR", 3: "LLWLLLLRLRL"},
    "intersection_n2_lshape": {0: "RRRDDDW", 1: "WDDDRRR"},
    "intersection_n2_tight": {0: "RRW", 1: "WDD"},
    "bottleneck_n4_convoy": {0: "RRDDRRUURRWW", 1: "RRDRRURRWWWW",
                              2: "UURRDRRRRDWW", 3: "UUURRDRRRDDR"},
    "bottleneck_n4_imbalanced": {0: "RRDDRRUURRWWWW", 1: "RRDRRURRWWWWWW",
                                  2: "UURRDRRRRDWWWW", 3: "UULLDWWULLLLDD"},
    "bottleneck_n4_tightdoor": {0: "DRRUWWWWWW", 1: "WURRDWWWWW",
                                 2: "WRDDLLLUUW", 3: "UUURDLLLDD"},
}

# Explicit, scenario-by-scenario record of what has (and has NOT) been
# verified. Populated "no" for CBS_verified on purpose -- see module
# docstring. Kept here (not just in prose) so a future CBS integration has a
# single place to flip entries to "yes" once it actually runs.
VERIFICATION_STATUS = {
    slug: {"verified_via_joint_plan_replay": "pending", "CBS_verified": "no"}
    for slug in PLANS
}


class TestControlledVariantsWellFormed(unittest.TestCase):
    def load(self, slug):
        grid, starts, goals = load_map_generator_scenario(VARIANTS_DIR / f"scenario_{slug}.json")
        return grid, dict(enumerate(starts)), dict(enumerate(goals))

    def test_every_scenario_file_has_a_plan_and_every_plan_a_scenario_file(self):
        on_disk = {p.stem[len("scenario_"):] for p in VARIANTS_DIR.glob("scenario_*.json")}
        self.assertEqual(on_disk, set(PLANS))

    def test_nine_variants_three_per_category(self):
        self.assertEqual(len(PLANS), 9)
        by_prefix = {"corridor": 0, "intersection": 0, "bottleneck": 0}
        for slug in PLANS:
            for prefix in by_prefix:
                if slug.startswith(prefix):
                    by_prefix[prefix] += 1
        self.assertEqual(by_prefix, {"corridor": 3, "intersection": 3, "bottleneck": 3})

    def test_instances_are_well_formed(self):
        for slug in PLANS:
            with self.subTest(slug=slug):
                grid, starts, goals = self.load(slug)
                self.assertEqual(set(np.unique(grid)) - {0, 1}, set())
                self.assertEqual(len(set(starts.values())), len(starts), "duplicate starts")
                self.assertEqual(len(set(goals.values())), len(goals), "duplicate goals")
                for aid in starts:
                    self.assertEqual(int(grid[starts[aid]]), 0, f"agent {aid} start on wall/out-of-range")
                    self.assertEqual(int(grid[goals[aid]]), 0, f"agent {aid} goal on wall/out-of-range")

    def test_variants_reuse_existing_maps_only(self):
        """Principle: no new .npy files -- every variant's map_file resolves
        into the frozen scenarios/controlled/ directory."""
        for slug in PLANS:
            with self.subTest(slug=slug):
                scenario_path = VARIANTS_DIR / f"scenario_{slug}.json"
                import json
                data = json.loads(scenario_path.read_text(encoding="utf-8"))
                self.assertTrue(data["map_file"].startswith("scenarios/controlled/"),
                                 f"{slug} must reuse a map from scenarios/controlled/, got {data['map_file']!r}")

    def test_variants_do_not_duplicate_existing_scenario_ids(self):
        existing_dir = PROJECT_ROOT / "scenarios" / "controlled"
        existing_ids = set()
        import json
        for p in existing_dir.glob("scenario_*.json"):
            existing_ids.add(json.loads(p.read_text(encoding="utf-8"))["scenario_id"])
        new_ids = set()
        for slug in PLANS:
            data = json.loads((VARIANTS_DIR / f"scenario_{slug}.json").read_text(encoding="utf-8"))
            new_ids.add(data["scenario_id"])
        self.assertEqual(len(new_ids), len(PLANS), "duplicate scenario_id among variants")
        self.assertEqual(existing_ids & new_ids, set(), "variant scenario_id collides with scenarios/controlled/")


class TestControlledVariantsSolvable(unittest.TestCase):
    """Conflict-free joint-plan replay in the unmodified MAPFStepSimulator.

    A failed replay must NOT be silently treated as success: assertions
    below fail loudly (with the simulator's own conflict classification) if
    any step is not conflict-free, or if the final state is not all-at-goal.
    """

    def load(self, slug):
        grid, starts, goals = load_map_generator_scenario(VARIANTS_DIR / f"scenario_{slug}.json")
        return grid, dict(enumerate(starts)), dict(enumerate(goals))

    def test_each_variant_is_solvable_by_a_conflict_free_joint_plan(self):
        for slug, plan in PLANS.items():
            with self.subTest(slug=slug):
                grid, starts, goals = self.load(slug)
                horizon = max(len(p) for p in plan.values())
                sim = MAPFStepSimulator(cbs_solver_root=None, max_steps=horizon + 5)
                sim.reset(grid, starts, goals)
                info = {"all_at_goal": False}
                for t in range(horizon):
                    actions = {aid: _ACTION[p[t]] if t < len(p) else WAIT for aid, p in plan.items()}
                    _, _, info = sim.step(actions)
                    self.assertEqual(set(sim._last_conflict_types.values()), {"none"},
                                      f"{slug} t={t + 1}: {sim._last_conflict_types}")
                    self.assertEqual((sim._last_vertex_conflict_count, sim._last_edge_conflict_count), (0, 0),
                                      f"{slug} t={t + 1}: nonzero vertex/edge conflict count")
                    self.assertEqual(len(set(sim._positions.values())), len(sim._positions),
                                      f"{slug} t={t + 1}: position overlap {sim._positions}")
                self.assertTrue(info["all_at_goal"], f"{slug}: plan does not reach all goals")
                VERIFICATION_STATUS[slug]["verified_via_joint_plan_replay"] = "yes"


if __name__ == "__main__":
    unittest.main()
