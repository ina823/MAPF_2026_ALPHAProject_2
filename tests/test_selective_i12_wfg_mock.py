"""I12-2: WFG / time axis / group selection, Mock-verified. NO Selective execution.

What runs for real (all unmodified): MAPFStepSimulator, DeadlockMonitor,
run_global_hybrid_episode, build_wfg, decide_selective, l0_union_group,
expansion_candidates, validate_full_robot, Path Guard, accept_replan_result,
ShadowMonitor / ShadowSelectivePlanner.
What is Mock: every planner here is a TEST-ONLY stand-in (``test_only = True``)
and no CBS solver is called. A Mock result is never evidence that a real
subset (group) CBS call succeeds -- no such call exists in the code yet.

FACT vs RULE OUTPUT:
- Fact: positions / intended cells / blocked flags / trigger timesteps read
  from saved DEV records (sha256-pinned, read-only, skipped if absent).
- Rule output: WFG edges and groups, decisions and expansion hits are what the
  DRAFT rules (wfg-candidate-v0, selective-decision-v0, selective-safety-v0)
  compute from those facts. They are not values any Selective run recorded.

Time axis (contract P1-c/d): the WFG is built from the Trigger step's S_{t-1}
(positions before the step) and intended cells; decision, group, expansion
and the CBS start use S_t (positions after the step).

Items marked UNIMPLEMENTED are skipped on purpose and are NOT passes.
"""

import csv
import hashlib
import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.common import spec
from src.common.deadlock_monitor import DeadlockMonitor
from src.common.scenario_loader import load_map_generator_scenario
from src.hybrid.cbs_runtime_bridge import PathGuardValidator
from src.hybrid.global_hybrid import GlobalHybridConfig, run_global_hybrid_episode
from src.hybrid.replan_boundary import accept_replan_result
from src.hybrid.selective_decision import (
    FULL_GROUP,
    NO_GROUP,
    PENDING_BLOCKERS,
    ROUTE_GLOBAL,
    SELECTIVE_CANDIDATE_BLOCKED,
    decide_selective,
)
from src.hybrid.selective_safety import (
    ACTION_CALL,
    LADDER_UNTIL_OUTSIDE_POLICY_VERIFIED,
    LEVEL_GLOBAL,
    ExpansionHit,
    TriggerBudget,
    expansion_candidates,
    l0_union_group,
    next_ladder_step,
    validate_full_robot,
)
from src.hybrid.selective_shadow import ShadowMonitor, ShadowSelectivePlanner
from src.hybrid.wfg import RULE_VERSION, WfgEdge, build_wfg
from src.path_guard import InvalidPathError, validate_path_guard
from src.simulator.mapf_step_simulator import MAPFStepSimulator
from tests.hybrid_mocks import (
    DOWN, LEFT, RIGHT, UP, WAIT,
    AcceptAllValidator, AllowAllRecalls, AlwaysResumeIL, MockReplanResult,
    RecordingPlanner, RecordSafeStop, ScriptedPolicy,
)

ROOT = Path(__file__).resolve().parents[1]

# -- saved DEV records (read-only) ---------------------------------------------------

CORRIDOR_SCENARIO = ROOT / "scenarios" / "controlled" / "scenario_corridor_bay_n2.json"
CORRIDOR_RUN = (ROOT / "outputs" / "selective_shadow_dev"
                / "20261111_180253_corridor_bay_n2_GLOBAL_HYBRID_DEV_H3_SHADOW")
CORRIDOR_FILES = {
    "shadow": (CORRIDOR_RUN / "shadow.json",
               "3ac7e2eb1278727998a50b8017c8d2c1d458b98e3396863fc6d9bd40c1ab01cb"),
    "comparison": (CORRIDOR_RUN / "comparison.json",
                   "7580c7af1939f34c83cf9d3b8b104c40c3d78c04bb7a86e69e1def4cb7585609"),
    "off_steps": (CORRIDOR_RUN / "off" / "20261111_180253_corridor_bay_n2_GLOBAL_HYBRID_DEV_H3_SHADOW_steps.csv",
                  "12a05326e84041ec849fbdd947aa824b0eba378167bde901e58fd76cbc5091eb"),
}

WH8A_SCENARIO = ROOT / "scenarios" / "warehouse_v3_dev" / "scenario_wh8a_9x8.json"
WH8A_RUN = ROOT / "outputs" / "warehouse_v3_ilonly" / "20261110_110923_wh8a_9x8_IL_ONLY"
WH8A_FILES = {
    "result": (WH8A_RUN / "result.json",
               "5f37b5c999a37a6d52628f4eac9747b989f53a333e332fd78df815012bc5441b"),
    "steps": (WH8A_RUN / "20261110_110923_wh8a_9x8_IL_ONLY_steps.csv",
              "76c5a750fa0de550640e35cbe2c82acaf6cd2c7217fb945e87880f4c3fb93e6b"),
}


def _pinned(test, files):
    """Skip if a record is absent (outputs are untracked); fail if it changed."""
    for name, (path, sha) in files.items():
        if not path.is_file():
            test.skipTest(f"saved DEV record absent: {path}")
        test.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), sha,
                         f"saved DEV record {name} changed: {path}")
    return {name: path for name, (path, _) in files.items()}


def _load_scenario(path):
    grid, starts, goals = load_map_generator_scenario(path)
    return grid, dict(enumerate(starts)), dict(enumerate(goals))


def _step_rows(steps_csv, t):
    """One step's rows -> (S_{t-1}, intended, S_t, blocked, trigger codes)."""
    with open(steps_csv, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if int(r["timestep"]) == t]
    prev = {int(r["agent_id"]): (int(r["current_row"]), int(r["current_col"])) for r in rows}
    want = {int(r["agent_id"]): (int(r["intended_row"]), int(r["intended_col"])) for r in rows}
    s_t = {int(r["agent_id"]): (int(r["next_row"]), int(r["next_col"])) for r in rows}
    blocked = tuple(sorted(int(r["agent_id"]) for r in rows if r["blocked"] == "True"))
    triggers = {r["trigger_reason"] for r in rows}
    return prev, want, s_t, blocked, triggers


def _cells(d):
    return {int(a): (int(p[0]), int(p[1])) for a, p in d.items()}


# -- one real simulator step, observed exactly as the runner observes it ----------

def _observe(grid, prev, actions):
    """Run ONE MAPFStepSimulator step from ``prev``. Returns S_{t-1}, intended
    (raw ACTION_DELTA target, as global_hybrid.run_step computes it), S_t,
    blocked (conflict_type != "none", as ShadowMonitor derives it)."""
    sim = MAPFStepSimulator(cbs_solver_root=None, max_steps=10)
    sim.reset(np.asarray(grid), dict(prev), dict(prev))
    intended = {a: (p[0] + spec.ACTION_DELTA[actions[a]][0], p[1] + spec.ACTION_DELTA[actions[a]][1])
                for a, p in prev.items()}
    sim.step(dict(actions))
    s_t = {a: tuple(sim._positions[a]) for a in sim._agent_ids}
    blocked = tuple(sorted(a for a, c in sim._last_conflict_types.items() if c != "none"))
    return dict(prev), intended, s_t, blocked


def _config():
    return GlobalHybridConfig(h_exec=1, resume_policy=AlwaysResumeIL(),
                              recall_policy=AllowAllRecalls(), safe_stop_handler=RecordSafeStop())


def _shadow_episode(grid, starts, goals, actions, max_steps=6):
    """Real Global Hybrid episode with the shadow wrappers and a TEST-ONLY
    planner that answers no_solution (so nothing is executed from CBS)."""
    inner = RecordingPlanner(MockReplanResult(status="no_solution", latency_sec=0.01))
    mon = ShadowMonitor(DeadlockMonitor())
    planner = ShadowSelectivePlanner(inner, mon)
    result = run_global_hybrid_episode(
        ScriptedPolicy({a: list(v) for a, v in actions.items()}), grid, starts, goals, max_steps,
        planner=planner, path_validator=AcceptAllValidator(), config=_config(), monitor=mon)
    return result, planner, inner


LINE = np.zeros((1, 5), dtype=np.int64)
GRID_3x7 = np.zeros((3, 7), dtype=np.int64)


# == A. time axis ========================================================================

class TestA_TimeAxis(unittest.TestCase):

    def test_a1_moving_case_wfg_uses_s_t_minus_1_cbs_uses_s_t(self):
        """S_{t-1} != S_t at the Trigger (agent 7 moved), non-contiguous ids."""
        starts = {7: (1, 6), 10: (0, 1), 20: (2, 1)}
        goals = {7: (1, 0), 10: (1, 6), 20: (0, 0)}           # 10's goal = 7's cell at S_t
        actions = {10: [DOWN] * 6, 20: [UP] * 6, 7: [LEFT, RIGHT] * 3}
        with mock.patch("src.hybrid.selective_shadow.build_wfg", wraps=build_wfg) as spy:
            result, planner, inner = _shadow_episode(GRID_3x7, starts, goals, actions)

        (rec,) = planner.records
        (call,) = spy.call_args_list
        self.assertEqual(call.args[0], {7: (1, 5), 10: (0, 1), 20: (2, 1)})   # S_{t-1}, not S_t
        self.assertEqual(call.args[1], rec.intended)
        self.assertIsNone(rec.shadow_error)
        self.assertEqual(rec.trigger_t, 2)
        s_prev = {7: (1, 5), 10: (0, 1), 20: (2, 1)}
        s_t = {7: (1, 6), 10: (0, 1), 20: (2, 1)}
        self.assertEqual(rec.s_t_minus_1, s_prev)
        self.assertEqual(rec.s_t, s_t)
        self.assertNotEqual(rec.s_t_minus_1, rec.s_t)
        self.assertEqual(rec.intended, {7: (1, 6), 10: (1, 1), 20: (1, 1)})

        # WFG input = S_{t-1} + intended
        self.assertEqual(rec.wfg, build_wfg(s_prev, rec.intended, rec.blocked, GRID_3x7))
        self.assertEqual(rec.wfg.groups, ((10, 20),))
        # CBS start = S_t: the (Global) planner got S_t, in sorted-id order.
        self.assertEqual(inner.calls[0]["current_positions"], [s_t[7], s_t[10], s_t[20]])
        self.assertEqual(result["cbs_calls"][0]["s_t"], s_t)
        # Group / expansion / precheck use S_t: 7 stands on 10's goal only at S_t.
        self.assertEqual((rec.l0_group, rec.outside), ((10, 20), (7,)))
        self.assertEqual(rec.expansion_hits, (ExpansionHit("GOAL_BLOCKED", 10, 7, (1, 6)),))
        self.assertEqual(rec.expansion_add, (7,))
        self.assertEqual(expansion_candidates((10, 20), s_prev, goals), ((), ()))
        self.assertEqual(rec.precheck.goal_blocked, (10,))
        self.assertEqual(rec.precheck.unreachable, (10,))

    def test_a2_full_stall_case_s_t_minus_1_equals_s_t(self):
        starts = {3: (0, 1), 8: (2, 1)}
        goals = {3: (2, 1), 8: (0, 1)}
        result, planner, inner = _shadow_episode(np.zeros((3, 3), dtype=np.int64), starts, goals,
                                                 {3: [DOWN] * 6, 8: [UP] * 6})
        (rec,) = planner.records
        self.assertEqual(rec.trigger_t, 1)
        self.assertEqual(rec.s_t_minus_1, rec.s_t)
        self.assertEqual(rec.s_t, starts)
        self.assertEqual(inner.calls[0]["current_positions"], [starts[3], starts[8]])
        self.assertEqual(rec.decision.reason, FULL_GROUP)

    def test_a3_wrong_axis_changes_the_group(self):
        """Agent 2 leaves (0,2) while 1 and 3 contest it. From S_{t-1} the rule
        links 2 (occupy edges, occupant moves -- PENDING T7); built wrongly from
        S_t, 2 is no longer there and drops out. The inputs are not
        interchangeable, so the caller must pass S_{t-1}."""
        grid = np.zeros((3, 5), dtype=np.int64)
        prev = {1: (0, 1), 2: (0, 2), 3: (1, 2)}
        s_prev, want, s_t, blocked = _observe(grid, prev, {1: RIGHT, 2: RIGHT, 3: UP})
        self.assertEqual(want, {1: (0, 2), 2: (0, 3), 3: (0, 2)})
        self.assertEqual(s_t, {1: (0, 1), 2: (0, 3), 3: (1, 2)})   # 1/3 cancelled, 2 moved
        self.assertEqual(blocked, (1, 3))
        self.assertNotEqual(s_prev, s_t)
        right = build_wfg(s_prev, want, blocked, grid)
        self.assertEqual(right.edges, (WfgEdge(1, 2, "occupy", (0, 2)),
                                       WfgEdge(1, 3, "contest", (0, 2)),
                                       WfgEdge(3, 2, "occupy", (0, 2))))
        self.assertEqual(right.groups, ((1, 2, 3),))
        wrong = build_wfg(s_t, want, blocked, grid)
        self.assertEqual(wrong.groups, ((1, 3),))


# == B. WFG relations ====================================================================

class TestB_WfgRelations(unittest.TestCase):

    def test_b1_occupy_into_waiting_agent(self):
        s_prev, want, s_t, blocked = _observe(LINE, {1: (0, 0), 2: (0, 1)}, {1: RIGHT, 2: WAIT})
        self.assertEqual(blocked, (1,))
        self.assertEqual(s_t, s_prev)
        wfg = build_wfg(s_prev, want, blocked, LINE)
        self.assertEqual(wfg.edges, (WfgEdge(1, 2, "occupy", (0, 1)),))
        self.assertIs(wfg.edges[0].occupant_moves, False)
        self.assertEqual(wfg.groups, ((1, 2),))

    def test_b2_occupy_with_moving_occupant_is_not_a_group(self):
        """PENDING T7: the edge is emitted (occupant_moves=True) but nobody was
        blocked, so no group forms -- it is not treated as a wait."""
        s_prev, want, s_t, blocked = _observe(LINE, {1: (0, 0), 2: (0, 1)}, {1: RIGHT, 2: RIGHT})
        self.assertEqual(blocked, ())
        self.assertEqual(s_t, {1: (0, 1), 2: (0, 2)})
        wfg = build_wfg(s_prev, want, blocked, LINE)
        self.assertEqual(wfg.edges, (WfgEdge(1, 2, "occupy", (0, 1)),))
        self.assertIs(wfg.edges[0].occupant_moves, True)
        self.assertEqual(wfg.groups, ())

    def test_b3_contest(self):
        s_prev, want, s_t, blocked = _observe(LINE, {1: (0, 0), 2: (0, 2)}, {1: RIGHT, 2: LEFT})
        self.assertEqual(blocked, (1, 2))
        wfg = build_wfg(s_prev, want, blocked, LINE)
        self.assertEqual(wfg.edges, (WfgEdge(1, 2, "contest", (0, 1)),))
        self.assertEqual(wfg.groups, ((1, 2),))

    def test_b4_contest_is_not_a_definite_wait(self):
        """A contest edge is undirected (nobody waits FOR a specific agent) and
        alone it does not make a group: grouping needs an observed block."""
        prev, want = {1: (0, 0), 2: (0, 2)}, {1: (0, 1), 2: (0, 1)}
        no_block = build_wfg(prev, want, (), LINE)
        self.assertEqual(no_block.edges, (WfgEdge(1, 2, "contest", (0, 1)),))
        self.assertEqual(no_block.groups, ())
        self.assertFalse(any(e.kind == "occupy" for e in no_block.edges))
        relabelled = build_wfg({2: (0, 0), 1: (0, 2)}, {2: (0, 1), 1: (0, 1)}, (), LINE)
        self.assertEqual(relabelled.edges, no_block.edges)        # src < dst, no direction

    def test_b5_swap(self):
        s_prev, want, s_t, blocked = _observe(LINE, {1: (0, 0), 2: (0, 1)}, {1: RIGHT, 2: LEFT})
        self.assertEqual(blocked, (1, 2))
        self.assertEqual(s_t, s_prev)
        wfg = build_wfg(s_prev, want, blocked, LINE)
        self.assertEqual(wfg.edges, (WfgEdge(1, 2, "swap", (0, 1)),))
        self.assertEqual(wfg.groups, ((1, 2),))
        self.assertEqual(wfg.rule_version, RULE_VERSION)


# == C. group selection ==================================================================

# Two independent contests (2,5) and (9,11) + outside 40 and 41; ids non-contiguous.
C_PREV = {2: (0, 1), 5: (2, 1), 9: (0, 4), 11: (2, 4), 40: (1, 6), 41: (0, 6)}
C_ACT = {2: DOWN, 5: UP, 9: DOWN, 11: UP, 40: WAIT, 41: WAIT}
C_GOALS = {2: (2, 0), 5: (1, 6), 9: (2, 5), 11: (0, 3), 40: (1, 0), 41: (0, 0)}   # 5's goal = 40's cell


class TestC_GroupSelection(unittest.TestCase):

    def setUp(self):
        self.s_prev, self.want, self.s_t, self.blocked = _observe(GRID_3x7, C_PREV, C_ACT)
        self.wfg = build_wfg(self.s_prev, self.want, self.blocked, GRID_3x7)
        self.ids = sorted(C_PREV)

    def test_c1_full_group(self):
        s_prev, want, s_t, blocked = _observe(LINE, {4: (0, 0), 9: (0, 2)}, {4: RIGHT, 9: LEFT})
        d = decide_selective([4, 9], s_t, {4: (0, 4), 9: (0, 0)}, build_wfg(s_prev, want, blocked, LINE))
        self.assertEqual((d.route, d.reason, d.fallback, d.selective_candidate), (ROUTE_GLOBAL, FULL_GROUP, False, False))
        self.assertEqual(d.candidate_groups, ((4, 9),))
        self.assertEqual(d.outside_agents, ())

    def test_c2_independent_groups_union(self):
        self.assertEqual(self.wfg.groups, ((2, 5), (9, 11)))
        self.assertEqual(l0_union_group(self.wfg.groups), (2, 5, 9, 11))
        d = decide_selective(self.ids, self.s_t, C_GOALS, self.wfg)
        self.assertEqual((d.route, d.reason), (ROUTE_GLOBAL, SELECTIVE_CANDIDATE_BLOCKED))
        self.assertEqual(d.outside_agents, (40, 41))
        self.assertIn(("MULTIPLE_GROUPS", 2), d.findings)
        self.assertEqual(d.blockers, PENDING_BLOCKERS)

    def test_c3_goal_blocked_and_exact_expansion_ids(self):
        l0 = l0_union_group(self.wfg.groups)
        hits, add = expansion_candidates(l0, self.s_t, C_GOALS)
        self.assertEqual(hits, (ExpansionHit("GOAL_BLOCKED", 5, 40, (1, 6)),))
        self.assertEqual(add, (40,))                      # 41 is outside but blocks nothing
        self.assertTrue(set(add).isdisjoint(l0))
        d = decide_selective(self.ids, self.s_t, C_GOALS, self.wfg)
        self.assertIn(("OUTSIDE_OCCUPIES_GROUP_GOAL", 5, 40), d.findings)

    def test_c4_path_blocked_with_mock_group_paths(self):
        """MOCK group paths (no CBS): 41 stands on a cell of 9's path."""
        mock_paths = {9: [(0, 4), (0, 5), (0, 6), (1, 6)]}
        hits, add = expansion_candidates((9, 11), self.s_t, C_GOALS, group_paths=mock_paths)
        self.assertEqual(hits, (ExpansionHit("PATH_BLOCKED", 9, 40, (1, 6)),
                                ExpansionHit("PATH_BLOCKED", 9, 41, (0, 6))))
        self.assertEqual(add, (40, 41))
        self.assertTrue(set(add).isdisjoint({2, 5}))      # other group never "added"

    def test_c5_non_contiguous_ids_are_never_renumbered(self):
        ids = set(C_PREV)
        for e in self.wfg.edges:
            self.assertTrue({e.src, e.dst} <= ids)
        d = decide_selective(self.ids, self.s_t, C_GOALS, self.wfg)
        self.assertEqual(set(a for g in d.candidate_groups for a in g) | set(d.outside_agents), ids)

    def test_c6_no_group(self):
        s_prev, want, s_t, blocked = _observe(LINE, {1: (0, 0), 2: (0, 3)}, {1: RIGHT, 2: RIGHT})
        d = decide_selective([1, 2], s_t, {1: (0, 1), 2: (0, 4)}, build_wfg(s_prev, want, blocked, LINE))
        self.assertEqual((d.route, d.reason), (ROUTE_GLOBAL, NO_GROUP))


# == D. saved DEV records (read-only) ====================================================

class TestD_DevRecords(unittest.TestCase):

    def test_d1_corridor_1111_t5(self):
        p = _pinned(self, CORRIDOR_FILES)
        (rec,) = json.loads(p["shadow"].read_text(encoding="utf-8"))["records"]
        cmp_ = json.loads(p["comparison"].read_text(encoding="utf-8"))
        grid, _, goals = _load_scenario(CORRIDOR_SCENARIO)

        # Facts: recorded shadow inputs == the independent step CSV row t=5.
        prev, want, s_t, blocked, triggers = _step_rows(p["off_steps"], 5)
        self.assertEqual(rec["trigger_t"], 5)
        self.assertEqual(triggers, {"TRIGGER_FULL_REPEAT"})
        self.assertEqual(_cells(rec["s_t_minus_1"]), prev)
        self.assertEqual(_cells(rec["intended"]), want)
        self.assertEqual(_cells(rec["s_t"]), s_t)
        self.assertEqual(tuple(rec["blocked"]), blocked)
        self.assertEqual(prev, s_t)                         # full stall: S_{t-1} == S_t
        # CBS start (real atb033 call, OFF and ON) == recorded S_t.
        for side in ("off", "on"):
            call = cmp_[side]["cbs_calls"][0]
            self.assertEqual(call["trigger_timestep"], 5)
            self.assertEqual(_cells(call["s_t"]), s_t)

        # Rule output recomputed from the facts == what the shadow recorded.
        wfg = build_wfg(prev, want, blocked, grid)
        self.assertEqual([(e.src, e.dst, e.kind, list(e.cell)) for e in wfg.edges],
                         [(e["src"], e["dst"], e["kind"], e["cell"]) for e in rec["wfg"]["edges"]])
        self.assertEqual([list(g) for g in wfg.groups], rec["wfg"]["groups"])
        d = decide_selective(sorted(s_t), s_t, goals, wfg)
        self.assertEqual((d.route, d.reason), (rec["decision"]["route"], rec["decision"]["reason"]))
        self.assertEqual(d.reason, FULL_GROUP)

    def test_d2_wh8a_il_only_t10(self):
        p = _pinned(self, WH8A_FILES)
        res = json.loads(p["result"].read_text(encoding="utf-8"))
        grid, _, goals = _load_scenario(WH8A_SCENARIO)
        self.assertEqual(_cells(res["goals"]), goals)
        mon = res["monitor"]
        self.assertEqual((mon["trigger_reason_code"], mon["trigger_timestep"], mon["first_seen"], mon["period"]),
                         ("TRIGGER_FULL_REPEAT", 10, 9, 1))

        # Facts: t=9 moved some agents (S_8 != S_9); t=10 is the stall (S_9 == S_10).
        prev9, _, s9, _, trig9 = _step_rows(p["steps"], 9)
        self.assertEqual(trig9, {"TRIGGER_NONE"})
        self.assertEqual(sorted(a for a in prev9 if prev9[a] != s9[a]), [0, 3, 4, 6])
        prev, want, s_t, blocked, triggers = _step_rows(p["steps"], 10)
        self.assertEqual(triggers, {"TRIGGER_FULL_REPEAT"})
        self.assertEqual(prev, s9)
        self.assertEqual(prev, s_t)
        self.assertEqual(blocked, (0, 1, 2, 3, 4, 5, 6))

        # Rule output (wfg-candidate-v0), not an observation.
        wfg = build_wfg(prev, want, blocked, grid)
        self.assertEqual(wfg.edges, (
            WfgEdge(0, 3, "swap", (8, 3)),
            WfgEdge(1, 2, "occupy", (3, 3)),
            WfgEdge(2, 5, "contest", (4, 3)),
            WfgEdge(4, 6, "swap", (0, 4)),
        ))
        self.assertIs(wfg.edges[1].occupant_moves, True)   # 2 is itself a mover (T7)
        self.assertEqual(wfg.groups, ((0, 3), (1, 2, 5), (4, 6)))
        l0 = l0_union_group(wfg.groups)
        self.assertEqual(l0, (0, 1, 2, 3, 4, 5, 6))
        d = decide_selective(sorted(s_t), s_t, goals, wfg)
        self.assertEqual((d.route, d.reason, d.outside_agents), (ROUTE_GLOBAL, SELECTIVE_CANDIDATE_BLOCKED, (7,)))
        self.assertIn(("MULTIPLE_GROUPS", 3), d.findings)
        self.assertIn(("GROUP_DEPENDS_ON_PENDING_T7", (1, 2, 5)), d.findings)
        self.assertEqual(d.blockers, PENDING_BLOCKERS)
        self.assertEqual(expansion_candidates(l0, s_t, goals), ((), ()))   # 7 sits on its own goal
        # IL-only run: no CBS was called, so no CBS start/plan is asserted here.
        self.assertNotIn("CBS", res.get("action_sources", []))


# == E. safety boundary ==================================================================

class TestE_SafetyBoundary(unittest.TestCase):

    def test_e1_v0_routes_global_only_with_all_blockers(self):
        self.assertEqual(PENDING_BLOCKERS, (
            "P1_TEMPORAL_RESERVATION_UNVERIFIED",
            "P2_OUTSIDE_AGENT_CONTROL_UNVERIFIED",
            "P3_FULL_ROBOT_VALIDATION_UNAVAILABLE",
            "P4_BUDGET_UNDEFINED",
            "P5_GROUP_RULE_UNAGREED",
        ))
        s_prev, want, s_t, blocked = _observe(GRID_3x7, C_PREV, C_ACT)
        wfg = build_wfg(s_prev, want, blocked, GRID_3x7)
        cases = [
            decide_selective(sorted(C_PREV), s_t, C_GOALS, wfg),
            decide_selective(sorted(C_PREV), s_t, C_GOALS, wfg,
                             candidate_paths={2: [(0, 1)], 5: [(2, 1)], 9: [(0, 4)], 11: [(2, 4)]}),
            decide_selective(sorted(C_PREV), s_t, None, wfg),
            decide_selective(sorted(C_PREV), s_t, C_GOALS, None),
        ]
        for d in cases:
            self.assertEqual(d.route, ROUTE_GLOBAL)
            if d.selective_candidate:
                self.assertEqual(d.blockers, PENDING_BLOCKERS)

    def test_e2_runtime_ladder_is_global_only(self):
        self.assertEqual(LADDER_UNTIL_OUTSIDE_POLICY_VERIFIED, (LEVEL_GLOBAL,))
        step = next_ladder_step(LADDER_UNTIL_OUTSIDE_POLICY_VERIFIED, (),
                                TriggerBudget(total_time_ms=1000.0, max_attempts=1,
                                              level_timeout_ms={"G": 1000.0}))
        self.assertEqual((step.action, step.level), (ACTION_CALL, LEVEL_GLOBAL))

    def _composed(self):
        """Group (1,2) reaches its goals in a MOCK plan; outside 3 holds S_t."""
        grid = np.zeros((3, 5), dtype=np.int64)
        s_t = {1: (0, 0), 2: (2, 0), 3: (1, 4)}
        goals = {1: (0, 2), 2: (2, 2), 3: (1, 0)}
        group = {1: [(0, 0), (0, 1), (0, 2)], 2: [(2, 0), (2, 1), (2, 2)]}
        return grid, s_t, goals, group

    def test_e3_existing_path_guard_keeps_all_agent_goal_check(self):
        grid, s_t, goals, group = self._composed()
        full = validate_full_robot(group, s_t, sorted(s_t), grid)
        self.assertTrue(full.ok)                            # vertex/edge/move clean ...
        self.assertEqual(full.full_paths[3], [(1, 4)] * 3)  # ... with 3 WAITing at S_t
        with self.assertRaisesRegex(InvalidPathError, "Agent 3: incorrect goal position"):
            validate_path_guard(full.full_paths, s_t, goals, grid)
        ok, why = PathGuardValidator()(grid, s_t, goals, full.full_paths)
        self.assertFalse(ok)
        self.assertIn("incorrect goal position", why)

    def test_e4_existing_boundary_refuses_partial_and_composed_plans(self):
        grid, s_t, goals, group = self._composed()
        order = tuple(sorted(s_t))
        partial = accept_replan_result(MockReplanResult("success", group, 0.01), agent_order=order,
                                       current_positions=s_t, goals=goals, grid=grid,
                                       path_validator=PathGuardValidator(), paths_keying="agent_id")
        self.assertEqual((partial.accepted, partial.reason), (False, "BOUNDARY_PATH_COUNT"))
        full = validate_full_robot(group, s_t, order, grid).full_paths
        composed = accept_replan_result(MockReplanResult("success", full, 0.01), agent_order=order,
                                        current_positions=s_t, goals=goals, grid=grid,
                                        path_validator=PathGuardValidator(), paths_keying="agent_id")
        self.assertFalse(composed.accepted)
        self.assertTrue(composed.reason.startswith("VALIDATOR_REJECT:"))

    def test_e5_shadow_never_executes_selective(self):
        """Even with a partial-group candidate, the planner is asked for ALL agents."""
        starts = {7: (1, 6), 10: (0, 1), 20: (2, 1)}
        goals = {7: (1, 0), 10: (1, 6), 20: (0, 0)}
        result, planner, inner = _shadow_episode(GRID_3x7, starts, goals,
                                                 {10: [DOWN] * 6, 20: [UP] * 6, 7: [LEFT, RIGHT] * 3})
        (rec,) = planner.records
        self.assertTrue(rec.decision.selective_candidate)
        self.assertEqual(len(inner.calls), 1)
        self.assertEqual(len(inner.calls[0]["current_positions"]), 3)
        self.assertEqual(result["cbs_calls"][0]["agent_order"], (7, 10, 20))
        self.assertNotIn("CBS", result["action_sources"])

    @unittest.skip("UNIMPLEMENTED (I12): Selective full-agent validator -- outside agents WAIT at S_t, "
                   "group agents must end at their goals. Not in src; the existing Path Guard "
                   "all-agent goal check must stay unchanged (see test_e3/e4).")
    def test_e6_selective_full_agent_validator(self):
        raise AssertionError("not implemented")

    @unittest.skip("UNIMPLEMENTED (Yunji): real subset/group CBS call with outside agents as obstacles "
                   "(P1 temporal reservation unverified). Mock results in this file are NOT CBS success.")
    def test_e7_real_subset_cbs_success(self):
        raise AssertionError("not implemented")


if __name__ == "__main__":
    unittest.main()
