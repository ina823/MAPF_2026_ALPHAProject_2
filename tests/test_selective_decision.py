"""Mock tests for the Selective routing decision (src/hybrid/selective_decision.py).

No CBS, no simulator, no file I/O. WFG inputs come from the real pure
function ``build_wfg`` on either the saved-log fixtures of tests/test_wfg.py
(FACT: positions/intents from the DEV logs) or small SYNTHETIC grids (marked).
``candidate_paths`` are hand-written MOCKS of a group CBS result.

Expected decisions encode the v0 rule: every route is GLOBAL. P1-P5 stay
PENDING and are reported as fixed blockers, never switched on here.
"""

import copy
import dataclasses
import itertools
import unittest

from src.hybrid import selective_decision as sd
from src.hybrid.selective_decision import (
    DECISION_RULE_VERSION,
    PENDING_BLOCKERS,
    ROUTE_GLOBAL,
    SelectiveDecision,
    decide_selective,
)
from src.hybrid.wfg import WfgEdge, WfgResult, build_wfg
from tests.test_wfg import CASES, _grid

# Scenario goals in runtime (row, col), from scenarios/controlled/*.json via
# scenario_loader.xy_to_rowcol (checked against result.json "goals").
GOALS = {
    "corridor_t5": {0: (2, 10), 1: (2, 0)},
    "intersection_t2": {0: (3, 5), 1: (5, 3)},
    "bottleneck_t2": {0: (1, 7), 1: (3, 7), 2: (0, 1), 3: (4, 1)},
    "bottleneck_t7": {0: (1, 7), 1: (3, 7), 2: (0, 1), 3: (4, 1)},
}

OPEN = _grid("""
.......
.......
.......
""")


def _log_case(name):
    c = CASES[name]
    wfg = build_wfg(c["current"], c["intended"], c["blocked"], c["grid"])
    return sorted(c["current"]), dict(c["current"]), GOALS[name], wfg


def _partial_case():
    """SYNTHETIC: 0 and 1 contest (0,2); 2 moves freely far away."""
    cur = {0: (0, 1), 1: (0, 3), 2: (2, 6)}
    want = {0: (0, 2), 1: (0, 2), 2: (2, 5)}
    goals = {0: (0, 6), 1: (0, 0), 2: (2, 0)}
    wfg = build_wfg(cur, want, {0, 1}, OPEN)
    return [0, 1, 2], cur, goals, wfg


class TestDecisionOutcomes(unittest.TestCase):
    def assertGlobal(self, d):
        self.assertIsInstance(d, SelectiveDecision)
        self.assertEqual(d.route, ROUTE_GLOBAL)
        self.assertEqual(d.rule_version, DECISION_RULE_VERSION)

    # -- full group: Global is the natural path, not a fallback -------------
    def test_full_group_uses_global_for_all_three_log_states(self):
        for name in ("corridor_t5", "intersection_t2", "bottleneck_t7"):
            with self.subTest(case=name):
                ids, pos, goals, wfg = _log_case(name)
                d = decide_selective(ids, pos, goals, wfg)
                self.assertGlobal(d)
                self.assertEqual(d.reason, sd.FULL_GROUP)
                self.assertFalse(d.fallback)
                self.assertFalse(d.selective_candidate)
                self.assertEqual(d.candidate_groups, (tuple(ids),))
                self.assertEqual(d.outside_agents, ())
                self.assertEqual(d.blockers, ())

    def test_full_group_does_not_need_goals(self):
        ids, pos, _, wfg = _log_case("corridor_t5")
        self.assertEqual(decide_selective(ids, pos, None, wfg).reason, sd.FULL_GROUP)

    # -- normal partial group: candidate only, Global fallback --------------
    def test_partial_group_is_candidate_only_and_blocked(self):
        ids, pos, goals, wfg = _partial_case()
        d = decide_selective(ids, pos, goals, wfg)
        self.assertGlobal(d)
        self.assertEqual(d.reason, sd.SELECTIVE_CANDIDATE_BLOCKED)
        self.assertTrue(d.fallback)
        self.assertTrue(d.selective_candidate)
        self.assertEqual(d.candidate_groups, ((0, 1),))
        self.assertEqual(d.outside_agents, (2,))
        self.assertEqual(d.findings, ())
        self.assertEqual(d.blockers, PENDING_BLOCKERS)

    def test_clean_mock_paths_still_do_not_enable_selective(self):
        # Mock group plan: 1 steps aside, 0 passes. No collision with the
        # (assumed still) outside agent 2 -- yet execution stays blocked,
        # because the hold-still assumption itself is unverified (P2).
        ids, pos, goals, wfg = _partial_case()
        paths = {0: [(0, 1), (0, 1), (0, 2), (0, 3)], 1: [(0, 3), (1, 3), (1, 2), (1, 1)]}
        d = decide_selective(ids, pos, goals, wfg, candidate_paths=paths)
        self.assertGlobal(d)
        self.assertEqual(d.reason, sd.SELECTIVE_CANDIDATE_BLOCKED)
        self.assertEqual(d.findings, ())
        self.assertIn("P2_OUTSIDE_AGENT_CONTROL_UNVERIFIED", d.blockers)

    # -- independent groups (saved log: Bottleneck t=2) ----------------------
    def test_independent_groups_bottleneck_t2(self):
        ids, pos, goals, wfg = _log_case("bottleneck_t2")
        d = decide_selective(ids, pos, goals, wfg)
        self.assertGlobal(d)
        self.assertEqual(d.reason, sd.SELECTIVE_CANDIDATE_BLOCKED)
        self.assertTrue(d.fallback)
        self.assertEqual(d.candidate_groups, ((0, 1), (2, 3)))
        self.assertEqual(d.outside_agents, ())
        # Facts from S_2 and scenario goals: agents 2/3 stand on the goals of
        # 0/1. A {0,1}-only plan with 2/3 frozen cannot reach those goals.
        self.assertEqual(
            d.findings,
            (
                ("GROUPS_COVER_ALL_AGENTS",),
                ("GROUP_OCCUPIES_OUTSIDE_GOAL", 2, 0),
                ("GROUP_OCCUPIES_OUTSIDE_GOAL", 3, 1),
                ("MULTIPLE_GROUPS", 2),
                ("OUTSIDE_OCCUPIES_GROUP_GOAL", 0, 2),
                ("OUTSIDE_OCCUPIES_GROUP_GOAL", 1, 3),
            ),
        )

    def test_cross_group_conflict_in_mock_paths(self):
        # SYNTHETIC two independent groups whose mock plans meet at (1,3).
        cur = {0: (0, 1), 1: (0, 3), 2: (2, 1), 3: (2, 3)}
        want = {0: (0, 2), 1: (0, 2), 2: (2, 2), 3: (2, 2)}
        wfg = build_wfg(cur, want, {0, 1, 2, 3}, OPEN)
        self.assertEqual(wfg.groups, ((0, 1), (2, 3)))
        goals = {0: (0, 6), 1: (0, 0), 2: (2, 6), 3: (2, 0)}
        paths = {
            0: [(0, 1), (0, 1)], 1: [(0, 3), (1, 3)],
            2: [(2, 1), (2, 1)], 3: [(2, 3), (1, 3)],
        }
        d = decide_selective([0, 1, 2, 3], cur, goals, wfg, candidate_paths=paths)
        self.assertGlobal(d)
        self.assertIn(("CROSS_GROUP_VERTEX_CONFLICT", 1, 3, 1), d.findings)

    # -- no group ------------------------------------------------------------
    def test_no_group_falls_back_to_global(self):
        cur = {0: (0, 0), 1: (2, 6)}
        wfg = build_wfg(cur, {0: (0, 1), 1: (2, 5)}, set(), OPEN)
        self.assertEqual(wfg.groups, ())
        d = decide_selective([0, 1], cur, {0: (0, 6), 1: (2, 0)}, wfg)
        self.assertGlobal(d)
        self.assertEqual(d.reason, sd.NO_GROUP)
        self.assertTrue(d.fallback)
        self.assertFalse(d.selective_candidate)

    # -- outside-agent risk --------------------------------------------------
    def test_outside_agent_on_group_goal(self):
        ids, pos, goals, wfg = _partial_case()
        goals = {**goals, 0: (2, 6)}  # 0's goal is where outside 2 stands
        d = decide_selective(ids, pos, goals, wfg)
        self.assertEqual(d.reason, sd.SELECTIVE_CANDIDATE_BLOCKED)
        self.assertIn(("OUTSIDE_OCCUPIES_GROUP_GOAL", 0, 2), d.findings)

    def test_group_agent_on_outside_goal(self):
        ids, pos, goals, wfg = _partial_case()
        goals = {**goals, 2: (0, 1)}  # outside 2's goal is under group agent 0
        d = decide_selective(ids, pos, goals, wfg)
        self.assertIn(("GROUP_OCCUPIES_OUTSIDE_GOAL", 0, 2), d.findings)

    def test_mock_path_through_outside_agent_is_flagged(self):
        ids, pos, goals, wfg = _partial_case()
        # Outside 2 sits at (2,6); mock plan drives 0 into it at k=6.
        paths = {
            0: [(0, 1), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 5), (2, 6)],
            1: [(0, 3), (0, 2), (0, 1), (0, 0)],
        }
        d = decide_selective(ids, pos, goals, wfg, candidate_paths=paths)
        self.assertGlobal(d)
        self.assertEqual(d.reason, sd.SELECTIVE_CANDIDATE_BLOCKED)
        self.assertIn(("OUTSIDE_VERTEX_CONFLICT", 0, 2, 7), d.findings)

    def test_group_internal_conflicts_in_mock_paths(self):
        ids, pos, goals, wfg = _partial_case()
        vertex = {0: [(0, 1), (0, 2)], 1: [(0, 3), (0, 2)]}
        d = decide_selective(ids, pos, goals, wfg, candidate_paths=vertex)
        self.assertIn(("GROUP_INTERNAL_VERTEX_CONFLICT", 0, 1, 1), d.findings)
        swap = {0: [(0, 1), (0, 2), (0, 3)], 1: [(0, 3), (0, 3), (0, 2)]}
        d = decide_selective(ids, pos, goals, wfg, candidate_paths=swap)
        self.assertIn(("GROUP_INTERNAL_EDGE_CONFLICT", 0, 1, 1), d.findings)

    def test_group_depending_on_pending_t7_edge_is_flagged(self):
        # SYNTHETIC: 0 follows into 1's cell while 1 moves out (T7, PENDING);
        # 1 and 2 contest (0,2). 3 is outside. Without the T7 edge, 0 would
        # not be in the group, so the finding marks that dependency. It does
        # not decide T7.
        cur = {0: (0, 0), 1: (0, 1), 2: (0, 3), 3: (2, 6)}
        want = {0: (0, 1), 1: (0, 2), 2: (0, 2), 3: (2, 5)}
        wfg = build_wfg(cur, want, {1, 2}, OPEN)
        self.assertEqual(wfg.groups, ((0, 1, 2),))
        goals = {0: (0, 6), 1: (1, 6), 2: (0, 0), 3: (2, 0)}
        d = decide_selective([0, 1, 2, 3], cur, goals, wfg)
        self.assertIn(("GROUP_DEPENDS_ON_PENDING_T7", (0, 1, 2)), d.findings)
        self.assertEqual(d.route, ROUTE_GLOBAL)

    def test_bottleneck_t7_full_group_is_not_t7_dependent(self):
        ids, pos, goals, wfg = _log_case("bottleneck_t7")
        self.assertEqual(sd._depends_on_t7(wfg, (0, 1, 2, 3)), False)


class TestInsufficientAndInvalid(unittest.TestCase):
    def assertReason(self, d, reason):
        self.assertIsInstance(d, SelectiveDecision)
        self.assertEqual(d.route, ROUTE_GLOBAL)
        self.assertTrue(d.fallback)
        self.assertEqual(d.reason, reason)

    def test_insufficient_input(self):
        ids, pos, goals, wfg = _partial_case()
        self.assertReason(decide_selective(ids, pos, goals, None), "INSUFFICIENT_INPUT:WFG_MISSING")
        self.assertReason(decide_selective(None, pos, goals, wfg), "INSUFFICIENT_INPUT:AGENT_IDS_MISSING")
        self.assertReason(decide_selective([], pos, goals, wfg), "INSUFFICIENT_INPUT:AGENT_IDS_MISSING")
        self.assertReason(decide_selective(ids, None, goals, wfg), "INSUFFICIENT_INPUT:POSITIONS_MISSING")
        d = decide_selective(ids, pos, None, wfg)
        self.assertReason(d, "INSUFFICIENT_INPUT:GOALS_MISSING")
        self.assertTrue(d.selective_candidate)  # group known, safety not checkable
        self.assertEqual(d.blockers, PENDING_BLOCKERS)

    def test_invalid_inputs_are_classified_and_never_raise(self):
        ids, pos, goals, wfg = _partial_case()
        bad_edge = WfgEdge(0, 2, "contest", (0, 2))
        cases = {
            "INVALID_INPUT:DUPLICATE_AGENT_ID": ([0, 0, 1, 2], pos, goals, wfg),
            "INVALID_INPUT:AGENT_ID_TYPE": ([0, 1, "2"], pos, goals, wfg),
            "INVALID_INPUT:AGENT_IDS_TYPE": ("012", pos, goals, wfg),
            "INVALID_INPUT:POSITIONS_TYPE": (ids, [(0, 1)], goals, wfg),
            "INVALID_INPUT:POSITIONS_AGENT_SET": (ids, {0: (0, 1), 1: (0, 3)}, goals, wfg),
            "INVALID_INPUT:POSITIONS_OVERLAP": (ids, {0: (0, 1), 1: (0, 1), 2: (2, 6)}, goals, wfg),
            "INVALID_INPUT:POSITION_SHAPE": (ids, {0: (0, 1, 0), 1: (0, 3), 2: (2, 6)}, goals, wfg),
            "INVALID_INPUT:WFG_TYPE": (ids, pos, goals, {"groups": ((0, 1),)}),
            "INVALID_INPUT:WFG_RULE_VERSION": (ids, pos, goals, dataclasses.replace(wfg, rule_version="x")),
            "INVALID_INPUT:WFG_UNKNOWN_AGENT": (ids, pos, goals, dataclasses.replace(wfg, groups=((0, 9),))),
            "INVALID_INPUT:WFG_GROUPS_OVERLAP": (ids, pos, goals, dataclasses.replace(wfg, groups=((0, 1), (1, 2)))),
            "INVALID_INPUT:WFG_GROUP_SHAPE": (ids, pos, goals, dataclasses.replace(wfg, groups=((1, 0),))),
            "INVALID_INPUT:WFG_GROUP_ORDER": (
                [0, 1, 2, 3], {**pos, 3: (1, 0)}, {**goals, 3: (1, 6)},
                dataclasses.replace(wfg, groups=((2, 3), (0, 1)))),
            "INVALID_INPUT:WFG_EDGE_CROSSES_GROUP": (
                ids, pos, goals, dataclasses.replace(wfg, edges=wfg.edges + (bad_edge,))),
            "INVALID_INPUT:GOALS_TYPE": (ids, pos, [(0, 6)], wfg),
            "INVALID_INPUT:GOALS_AGENT_SET": (ids, pos, {0: (0, 6)}, wfg),
            "INVALID_INPUT:GOAL_SHAPE": (ids, pos, {**goals, 2: "x"}, wfg),
        }
        for reason, args in cases.items():
            with self.subTest(reason):
                self.assertReason(decide_selective(*args), reason)

    def test_invalid_mock_paths_are_classified(self):
        ids, pos, goals, wfg = _partial_case()
        ok1 = [(0, 3), (0, 3)]
        cases = {
            "INVALID_CANDIDATE_PATHS:TYPE": [(0, 1)],
            "INVALID_CANDIDATE_PATHS:AGENT_SET": {0: [(0, 1)]},
            "INVALID_CANDIDATE_PATHS:EMPTY_OR_MALFORMED": {0: [], 1: ok1},
            "INVALID_CANDIDATE_PATHS:START_MISMATCH": {0: [(1, 1)], 1: ok1},
            "INVALID_CANDIDATE_PATHS:NON_ADJACENT": {0: [(0, 1), (0, 3)], 1: ok1},
        }
        for reason, paths in cases.items():
            with self.subTest(reason):
                d = decide_selective(ids, pos, goals, wfg, candidate_paths=paths)
                self.assertReason(d, reason)
                self.assertTrue(d.selective_candidate)


class TestDeterminismAndPurity(unittest.TestCase):
    def test_same_input_same_decision_under_reordering(self):
        for name in ("bottleneck_t2", "bottleneck_t7"):
            ids, pos, goals, wfg = _log_case(name)
            expected = decide_selective(ids, pos, goals, wfg)
            for perm in itertools.permutations(ids):
                with self.subTest(case=name, order=perm):
                    got = decide_selective(
                        list(reversed(perm)),
                        {a: pos[a] for a in perm},
                        {a: list(goals[a]) for a in reversed(perm)},
                        wfg,
                    )
                    self.assertEqual(got, expected)
            self.assertEqual(decide_selective(ids, pos, goals, wfg), expected)

    def test_inputs_are_not_mutated(self):
        ids, pos, goals, wfg = _partial_case()
        paths = {0: [(0, 1), (0, 1)], 1: [(0, 3), (1, 3)]}
        snapshot = copy.deepcopy((ids, pos, goals, wfg, paths))
        decide_selective(ids, pos, goals, wfg, candidate_paths=paths)
        self.assertEqual((ids, pos, goals, wfg, paths), snapshot)

    def test_v0_has_no_selective_route(self):
        # Guard: v0 must not expose a route other than GLOBAL.
        routes = {v for k, v in vars(sd).items() if k.startswith("ROUTE_")}
        self.assertEqual(routes, {"GLOBAL"})
        self.assertEqual(len(PENDING_BLOCKERS), 5)
        self.assertEqual([b[:2] for b in PENDING_BLOCKERS], ["P1", "P2", "P3", "P4", "P5"])


if __name__ == "__main__":
    unittest.main()
