"""Selective Hybrid on the REAL Global Hybrid Runner, MOCK planners (src/hybrid/selective_planner.py).

Real and unmodified: run_global_hybrid_episode, MAPFStepSimulator, DeadlockMonitor,
replan_boundary, Path Guard, PathGuardValidator, selective_compose / safety /
decision / wfg / shadow. MOCK: every group ("subset") planner and every opt-in
Global planner (``test_only``). The real-IL tests use the real CbsRuntimePlanner
with ``replan_with_status`` patched to the recorded atb033 answer -- no solver runs.

No test here shows that a real subset CBS call succeeds; none exists
(PENDING:REAL_SUBSET_CBS). Budget numbers are TEST values (PENDING P4).
"""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import run_selective_mock_dev as dev
from scripts.run_selective_mock_dev import (
    ACTIONS_A, GLOBAL_PATHS_A, GOALS_A, GRID_A, GROUP_PATHS_A, STARTS_A, TEST_BUDGET,
    ScriptedGlobalPlanner, ScriptedSubsetPlanner, run_global, run_selective, success,
)
from src.hybrid.cbs_runtime_bridge import PathGuardValidator
from src.hybrid.global_hybrid import MonitorRearmConfig
from src.hybrid.replan_boundary import check_validator_pairing
from src.hybrid.selective_compose import GROUP_REACHES_GOAL, OUTSIDE_HELD
from src.hybrid.selective_decision import PENDING_BLOCKERS, ROUTE_GLOBAL
from src.hybrid.selective_planner import (
    PENDING,
    SKIP_EQUALS_ALL,
    SKIP_GOAL_BLOCKED,
    SKIP_NO_EXPANSION,
    SKIP_UNREACHABLE,
    SelectiveConfig,
    SelectivePathValidator,
    SelectivePlanner,
)
from src.hybrid.selective_safety import LADDER_UNTIL_OUTSIDE_POLICY_VERIFIED, TriggerBudget
from src.hybrid.selective_shadow import ShadowMonitor
from src.common.deadlock_monitor import DeadlockMonitor
from tests.hybrid_mocks import DOWN, UP, WAIT, LEFT, RIGHT, MockReplanResult, perturbing_simulator

ROOT = Path(__file__).resolve().parents[1]
OPT_IN = SelectiveConfig(ladder=("L0", "L1", "G"), budget=TEST_BUDGET, acknowledge_unverified_outside_policy=True)


def status(s, latency_sec):
    return MockReplanResult(status=s, paths=None, latency_sec=latency_sec, error=f"mock {s}")


def cbs_moves(result):
    return [i for i, s in enumerate(result["action_sources"]) if s == "CBS"]


# -- fixture B: 4 agents, L0 goal blocked by outside 7 -> L1 = (7, 10, 20), outside 30 ----------
STARTS_B = {7: (1, 6), 10: (0, 1), 20: (2, 1), 30: (0, 6)}
GOALS_B = {7: (2, 6), 10: (1, 6), 20: (0, 1), 30: (0, 6)}
ACTIONS_B = {10: [DOWN] * 40, 20: [UP] * 40, 7: [LEFT, RIGHT] + [WAIT] * 40, 30: [WAIT] * 40}
L1_PATHS_B = {7: [(1, 6), (2, 6)],
              10: [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 5), (1, 6)],
              20: [(2, 1), (1, 1), (0, 1)]}

# -- fixture C: outside 30 holds the only passage (1,3) -> 10's goal unreachable --------------
GRID_C = np.zeros((3, 7), dtype=np.int64)
GRID_C[0, 3] = GRID_C[2, 3] = 1
STARTS_C = {10: (0, 1), 20: (2, 1), 30: (1, 3)}
GOALS_C = {10: (1, 5), 20: (0, 0), 30: (1, 3)}
ACTIONS_C = {10: [DOWN] * 40, 20: [UP] * 40, 30: [WAIT] * 40}


# == 1. default G == plain Global ======================================================

class TestDefaultGIsGlobal(unittest.TestCase):

    def _same(self, make_global, **kw):
        g0, g1 = make_global(), make_global()
        base = run_global(g0, **kw)
        sel, planner, validator = run_selective(g1, **kw)
        self.assertEqual(sel, base)                                  # whole result dict
        self.assertEqual(g1.calls, g0.calls)                         # same args, same count
        for rec in planner.records:
            self.assertEqual((rec.mode, rec.outcome, rec.attempts), ("DEFAULT_G", "PASS_THROUGH", []))
        return base, planner, validator

    def test_success_h4_and_h2(self):
        for h in (4, 2, 1):
            with self.subTest(h_exec=h):
                base, planner, validator = self._same(lambda: ScriptedGlobalPlanner(success(GLOBAL_PATHS_A, 0.2)),
                                                      h_exec=h)
                self.assertEqual(validator.calls, [("G", (True, ""))])  # real PathGuardValidator verdict
                (rec,) = planner.records
                # analysis recorded only: v0 decision still GLOBAL with all blockers
                self.assertEqual(rec.decision.route, ROUTE_GLOBAL)
                self.assertEqual(rec.decision.blockers, PENDING_BLOCKERS)

    def test_returns_the_global_result_object_itself(self):
        res = success(GLOBAL_PATHS_A, 0.2)
        mon = ShadowMonitor(DeadlockMonitor())
        p = SelectivePlanner(ScriptedGlobalPlanner(res), mon, SelectiveConfig())
        mon.reset(STARTS_A, GOALS_A)
        self.assertIs(p(dict(STARTS_A), dict(GOALS_A), GRID_A), res)

    def test_non_success_and_exception_and_rearm(self):
        self._same(lambda: ScriptedGlobalPlanner(status("no_solution", 0.1)))
        self._same(lambda: ScriptedGlobalPlanner(status("timeout", 0.1)))
        self._same(lambda: ScriptedGlobalPlanner(RuntimeError("boom")))
        self._same(lambda: ScriptedGlobalPlanner(success(GLOBAL_PATHS_A, 0.2)), h_exec=1,
                   rearm=MonitorRearmConfig(max_cbs_calls=3, cooldown_steps=0, cbs_time_budget_ms=10_000.0))

    def test_global_plan_failing_path_guard_is_refused_like_global(self):
        bad = {**GLOBAL_PATHS_A, 7: [(1, 6)] * 5}                     # 7 never reaches its goal
        base, _, validator = self._same(lambda: ScriptedGlobalPlanner(success(bad, 0.2)))
        self.assertEqual(base["termination"], "SAFE_STOP")
        self.assertTrue(base["stop_reason"].startswith("VALIDATOR_REJECT:"))
        self.assertFalse(validator.calls[0][1][0])


try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class TestDefaultGRealIL(unittest.TestCase):
    """Real IL + real CbsRuntimePlanner/PathGuardValidator; solver answer = recorded atb033 file."""

    def _episode(self, scenario, solver, selective, tmp):
        from scripts.run_hybrid_resume_dev import dev_resume_config
        from scripts.run_il import DEFAULT_CKPT, load_scenario
        from src.hybrid.cbs_runtime_bridge import CbsRuntimePlanner
        from src.hybrid.global_hybrid import run_global_hybrid_episode
        from src.il.policy import NavHintPolicy
        grid, starts, goals = load_scenario(scenario)
        root = tmp / "solver"
        root.mkdir(exist_ok=True)
        (root / "cbs.py").write_text("raise SystemExit('fake solver must not run')\n", encoding="utf-8")
        inner = CbsRuntimePlanner(solver_root=root, work_root=tmp / f"cbs_{selective}")
        mon, planner, validator = DeadlockMonitor(), inner, PathGuardValidator()
        if selective:
            mon = ShadowMonitor(mon)
            planner = SelectivePlanner(inner, mon, SelectiveConfig())
            validator = SelectivePathValidator(planner)
        calls = []
        with patch("src.cbs_runtime_adapter.replan_with_status",
                   side_effect=lambda **kw: calls.append(kw["current_positions"]) or solver(**kw)):
            res = run_global_hybrid_episode(NavHintPolicy(DEFAULT_CKPT), grid, starts, goals, 160,
                                            planner=planner, path_validator=validator,
                                            config=dev_resume_config(3), monitor=mon)
        return res, calls, planner

    def test_corridor_and_wh8a_identical(self):
        from tests.test_common_runner import CORRIDOR, recorded_solver
        from tests.test_selective_shadow_e2e import WH8A, _status_solver
        expected = {"corridor": ("SUCCESS", None, 13, ["CBS"] * 3, ((0, 1),)),
                    "wh8a": ("SAFE_STOP", "CBS_NO_SOLUTION", 10, [], ((0, 3), (1, 2, 5), (4, 6)))}
        for name, scenario, solver in (("corridor", CORRIDOR, recorded_solver),
                                       ("wh8a", WH8A, _status_solver("no_solution", 0.2))):
            with self.subTest(name), tempfile.TemporaryDirectory() as d:
                off, off_calls, _ = self._episode(scenario, solver, False, Path(d))
                on, on_calls, planner = self._episode(scenario, solver, True, Path(d))
                self.assertEqual(on, off)
                self.assertEqual(on_calls, off_calls)
                self.assertEqual(len(off_calls), 1)
                (rec,) = planner.records
                term, stop, steps, cbs, groups = expected[name]
                self.assertEqual((off["termination"], off["stop_reason"], off["steps"]), (term, stop, steps))
                self.assertEqual([s for s in off["action_sources"] if s == "CBS"], cbs)
                self.assertEqual((rec.outcome, rec.wfg.groups), ("PASS_THROUGH", groups))


# == 2. opt-in refusals ===================================================================

class TestOptInGuards(unittest.TestCase):

    def test_config_requires_budget_and_acknowledgement(self):
        with self.assertRaisesRegex(ValueError, "TriggerBudget"):
            SelectiveConfig(ladder=("L0", "G"), acknowledge_unverified_outside_policy=True)
        with self.assertRaisesRegex(ValueError, "acknowledge"):
            SelectiveConfig(ladder=("L0", "G"), budget=TEST_BUDGET)
        with self.assertRaisesRegex(ValueError, "level_timeout_ms"):
            SelectiveConfig(ladder=("L0", "G"), acknowledge_unverified_outside_policy=True,
                            budget=TriggerBudget(1000.0, 2, {"G": 500.0}))
        for lad in (("L0",), ("G", "L0"), ("L1", "L0", "G"), ("L0", "L0", "G"), ("X", "G"), ()):
            with self.subTest(ladder=lad), self.assertRaises(ValueError):
                SelectiveConfig(ladder=lad, budget=TEST_BUDGET, acknowledge_unverified_outside_policy=True)
        self.assertEqual(SelectiveConfig().ladder, LADDER_UNTIL_OUTSIDE_POLICY_VERIFIED)
        self.assertFalse(SelectiveConfig().opt_in)

    def test_real_subset_and_real_global_are_refused_in_opt_in(self):
        mon = ShadowMonitor(DeadlockMonitor())

        class RealLookingSubset:
            def __call__(self, *a, **k):
                raise AssertionError("must not be called")

        with self.assertRaisesRegex(ValueError, "PENDING:REAL_SUBSET_CBS"):
            SelectivePlanner(ScriptedGlobalPlanner(), mon, OPT_IN, subset_planner=RealLookingSubset())
        with self.assertRaisesRegex(ValueError, "subset_planner"):
            SelectivePlanner(ScriptedGlobalPlanner(), mon, OPT_IN)

        class NoTimeoutGlobal:  # like the real CbsRuntimePlanner: no per-call timeout
            paths_keying = "agent_id"
        with self.assertRaisesRegex(ValueError, "timeout_ms"):
            SelectivePlanner(NoTimeoutGlobal(), mon, OPT_IN, subset_planner=ScriptedSubsetPlanner())

    def test_wiring_guards(self):
        with self.assertRaisesRegex(ValueError, "ShadowMonitor"):
            SelectivePlanner(ScriptedGlobalPlanner(), DeadlockMonitor(), SelectiveConfig())

        class IndexKeyed:
            pass
        with self.assertRaisesRegex(ValueError, "agent_id"):
            SelectivePlanner(IndexKeyed(), ShadowMonitor(DeadlockMonitor()), SelectiveConfig())
        p = SelectivePlanner(ScriptedGlobalPlanner(), ShadowMonitor(DeadlockMonitor()), SelectiveConfig())
        v = SelectivePathValidator(p)
        self.assertFalse(getattr(v, "test_only", False))
        check_validator_pairing(object(), v)


# == 3. opt-in partial group: compose, execute, IL_RESUME, expected/actual ================

class TestPartialGroupExecution(unittest.TestCase):

    def _run(self, *subset, glob=None, h_exec=4, config=OPT_IN, **kw):
        g = glob or ScriptedGlobalPlanner(success(GLOBAL_PATHS_A, 0.2))
        s = ScriptedSubsetPlanner(*subset)
        result, planner, validator = run_selective(g, config, s, h_exec=h_exec, **kw)
        return result, planner, validator, g, s

    def test_l0_group_executes_with_outside_held(self):
        result, planner, validator, g, s = self._run(success(GROUP_PATHS_A, 0.1))
        (rec,) = planner.records
        # time axis: WFG from S_{t-1} (7 at (1,5)), CBS from S_t (7 at (1,6))
        self.assertEqual(rec.trigger_t, 2)
        self.assertEqual(rec.s_t_minus_1, {7: (1, 5), 10: (0, 1), 20: (2, 1)})
        self.assertEqual(rec.s_t, STARTS_A)
        self.assertEqual(rec.wfg.groups, ((10, 20),))
        (call,) = s.calls
        self.assertEqual(call["current_positions"], {10: (0, 1), 20: (2, 1)})   # S_t, original ids
        self.assertEqual(call["outside"], {7: (1, 6)})
        self.assertEqual(call["goals"], {10: (2, 1), 20: (0, 1)})
        self.assertEqual(call["timeout_ms"], 400.0)
        self.assertEqual(g.calls, [])                                            # Global not called
        # executed by the unchanged Runner, all-agent expected == actual, 7 never moved
        self.assertEqual(rec.outcome, "EXECUTE:L0")
        self.assertEqual((rec.executed_group, rec.outside_held), ((10, 20), (7,)))
        self.assertEqual(rec.goal_status, {10: GROUP_REACHES_GOAL, 20: GROUP_REACHES_GOAL, 7: OUTSIDE_HELD})
        self.assertEqual(validator.calls, [("SELECTIVE", (True, ""))])
        self.assertEqual(cbs_moves(result), [2, 3, 4, 5])
        for e in result["cbs_exec_trace"]:
            self.assertTrue(e["match"])
            self.assertEqual(set(e["expected"]), {7, 10, 20})
            self.assertEqual(e["actual"][7], (1, 6))
        for k in range(4):
            self.assertEqual(result["actions"][2 + k][7], WAIT)
        self.assertEqual([t["to"] for t in result["transitions"]][-2:], ["IL_RESUME", "IL"])
        self.assertEqual(result["termination"], "SUCCESS")                       # 7 reaches (1,5) by IL
        self.assertEqual(result["trajectory"][6][7], (1, 6))                    # not at goal after CBS
        self.assertEqual(result["cbs_calls"][0]["agent_order"], (7, 10, 20))
        self.assertAlmostEqual(result["cbs_calls"][0]["latency_ms"], 100.0)
        # Mock, not real
        self.assertEqual(rec.attempts[0].planner_kind, "MOCK")
        self.assertFalse(rec.real_subset_cbs_called)
        self.assertIn("PENDING:REAL_SUBSET_CBS", rec.pending)

    def test_il_resume_mid_plan(self):
        result, planner, *_ = self._run(success(GROUP_PATHS_A, 0.1), h_exec=2)
        self.assertEqual(cbs_moves(result), [2, 3])
        self.assertEqual(len(result["cbs_exec_trace"]), 2)
        self.assertTrue(all(e["match"] for e in result["cbs_exec_trace"]))
        resume = [t for t in result["transitions"] if t["to"] == "IL_RESUME"]
        self.assertEqual([(t["t"], t["reason"]) for t in resume], [(4, "EXEC_WINDOW_END")])
        self.assertEqual(result["action_sources"][4], "IL")

    def test_outside_agent_displaced_during_exec_stops(self):
        result, *_ = self._run(success(GROUP_PATHS_A, 0.1), simulator_cls=perturbing_simulator(3, 7, (1, 5)))
        self.assertEqual((result["termination"], result["stop_reason"]), ("SAFE_STOP", "EXEC_MISMATCH"))
        self.assertEqual(result["exec_mismatch"]["actual"][7], (1, 5))
        self.assertEqual(result["exec_mismatch"]["expected"][7], (1, 6))
        self.assertEqual(cbs_moves(result), [2])                                 # no further move
        self.assertEqual(result["steps"], 3)

    def test_group_agent_displaced_during_exec_stops(self):
        result, *_ = self._run(success(GROUP_PATHS_A, 0.1), simulator_cls=perturbing_simulator(4, 20, (2, 2)))
        self.assertEqual(result["stop_reason"], "EXEC_MISMATCH")
        self.assertEqual(result["steps"], 4)


# == 4. fallback ladder, budget, SAFE_STOP ==================================================

class TestLadder(unittest.TestCase):

    def _run(self, *subset, glob=None, config=OPT_IN):
        g = glob or ScriptedGlobalPlanner(success(GLOBAL_PATHS_A, 0.2))
        s = ScriptedSubsetPlanner(*subset)
        result, planner, validator = run_selective(g, config, s)
        return result, planner.records[0], g, s, validator

    def test_timeout_and_no_solution_fall_back_to_global(self):
        for st in ("timeout", "no_solution"):
            with self.subTest(st):
                result, rec, g, s, v = self._run(status(st, 0.3))
                self.assertEqual([(a.level, a.status) for a in rec.attempts], [("L0", st), ("G", "success")])
                self.assertEqual(len(s.calls), 1)
                (gc,) = g.calls
                self.assertEqual(gc["current_positions"], STARTS_A)                 # all agents, S_t
                self.assertEqual(gc["timeout_ms"], 700.0)                          # min(800, 1000-300)
                self.assertEqual(rec.outcome, "EXECUTE:G")
                self.assertEqual(v.calls, [("G", (True, ""))])                     # real Path Guard
                self.assertEqual(result["termination"], "SUCCESS")
                self.assertAlmostEqual(result["cbs_calls"][0]["latency_ms"], 500.0)  # failed L0 counted

    def test_invalid_path_and_solver_error_safe_stop(self):
        for st, reason in (("invalid_path", "CBS_INVALID_PATH"), ("solver_error", "UNAGREED_STATUS:solver_error"),
                           ("input_error", "UNAGREED_STATUS:input_error")):
            with self.subTest(st):
                result, rec, g, s, v = self._run(status(st, 0.1))
                self.assertEqual(g.calls, [])
                self.assertEqual(rec.outcome, f"SAFE_STOP:{reason}")
                self.assertEqual((result["termination"], result["stop_reason"]), ("SAFE_STOP", f"CBS_{st.upper()}"))
                self.assertEqual(cbs_moves(result), [])
                self.assertEqual(v.calls, [])

    def test_ladder_exhausted(self):
        result, rec, g, s, _ = self._run(status("timeout", 0.2), glob=ScriptedGlobalPlanner(status("no_solution", 0.2)))
        self.assertEqual(rec.outcome, "SAFE_STOP:LADDER_EXHAUSTED:CBS_NO_SOLUTION")
        self.assertEqual(result["stop_reason"], "CBS_NO_SOLUTION")
        self.assertAlmostEqual(result["cbs_calls"][0]["latency_ms"], 400.0)

    def test_budget_exhausted_by_failed_attempt(self):
        result, rec, g, *_ = self._run(status("timeout", 1.2))
        self.assertEqual(rec.outcome, "SAFE_STOP:TIME_BUDGET_EXHAUSTED")
        self.assertEqual(g.calls, [])
        self.assertEqual(result["stop_reason"], "CBS_TIMEOUT")

    def test_attempt_limit_and_unknown_latency(self):
        one = replace(OPT_IN, budget=TriggerBudget(1000.0, 1, dict(TEST_BUDGET.level_timeout_ms)))
        _, rec, g, *_ = self._run(status("timeout", 0.1), config=one)
        self.assertEqual((rec.outcome, g.calls), ("SAFE_STOP:ATTEMPT_LIMIT", []))
        result, rec, g, *_ = self._run(status("no_solution", None))
        self.assertEqual((rec.outcome, g.calls), ("SAFE_STOP:TIME_BUDGET_EXHAUSTED", []))
        self.assertIsNone(result["cbs_calls"][0]["latency_ms"])

    def test_full_check_failures_safe_stop_never_execute(self):
        cases = {
            "vertex with held outside": {10: GROUP_PATHS_A[10], 20: [(2, 1), (2, 2), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
                                                                     (1, 5), (1, 4), (1, 3), (1, 2), (0, 2), (0, 1)]},
            "group edge swap": {10: [(0, 1), (1, 1), (2, 1)], 20: [(2, 1), (2, 1), (1, 1), (0, 1)]},
            "group goal not reached": {10: [(0, 1), (0, 1)], 20: GROUP_PATHS_A[20]},
            "static obstacle / bounds": {10: [(0, 1), (-1, 1), (0, 1), (1, 1), (2, 1)], 20: GROUP_PATHS_A[20]},
            "wrong agent set": {10: GROUP_PATHS_A[10]},
            "start != S_t": {10: [(1, 1), (2, 1)], 20: GROUP_PATHS_A[20]},
        }
        for name, paths in cases.items():
            with self.subTest(name):
                result, rec, g, s, v = self._run(success(paths, 0.1))
                self.assertEqual(rec.attempts[0].full_check_ok, False, rec.attempts[0].full_check_reason)
                self.assertEqual(rec.outcome, "SAFE_STOP:FULL_CHECK_FAILED")
                self.assertEqual(g.calls, [])
                self.assertEqual((result["termination"], result["stop_reason"]), ("SAFE_STOP", "CBS_INVALID_PATH"))
                self.assertEqual(cbs_moves(result), [])

    def test_global_success_failing_path_guard_safe_stop(self):
        bad = {**GLOBAL_PATHS_A, 7: [(1, 6)] * 5}
        result, rec, g, s, v = self._run(status("timeout", 0.1), glob=ScriptedGlobalPlanner(success(bad, 0.2)))
        self.assertEqual(rec.outcome, "SAFE_STOP:FULL_CHECK_FAILED")
        self.assertIn("incorrect goal position", rec.attempts[-1].full_check_reason)
        self.assertEqual(result["stop_reason"], "CBS_INVALID_PATH")

    def test_planner_exception_propagates_to_runner(self):
        result, rec, *_ = self._run(RuntimeError("subset boom"))
        self.assertEqual((result["termination"], result["stop_reason"]), ("SAFE_STOP", "PLANNER_EXCEPTION"))


# == 5. prechecks before any CBS call ========================================================

class TestPrecheck(unittest.TestCase):

    def test_goal_blocked_expands_to_l1(self):
        g = ScriptedGlobalPlanner(status("no_solution", 0.1))
        s = ScriptedSubsetPlanner(success(L1_PATHS_B, 0.1))
        result, planner, v = run_selective(g, OPT_IN, s, starts=STARTS_B, goals=GOALS_B, actions=ACTIONS_B, h_exec=8)
        (rec,) = planner.records
        l0, l1, _ = rec.levels
        self.assertEqual((l0.group, l0.viable, l0.skip_reason, l0.goal_blocked), ((10, 20), False, SKIP_GOAL_BLOCKED, (10,)))
        self.assertEqual((l1.group, l1.viable), ((7, 10, 20), True))
        (call,) = s.calls
        self.assertEqual(set(call["current_positions"]), {7, 10, 20})
        self.assertEqual(call["outside"], {30: (0, 6)})
        self.assertEqual(rec.outcome, "EXECUTE:L1")
        self.assertEqual(rec.outside_held, (30,))
        self.assertEqual(g.calls, [])
        self.assertTrue(all(e["match"] and e["actual"][30] == (0, 6) for e in result["cbs_exec_trace"]))
        self.assertEqual(result["termination"], "SUCCESS")

    def test_unreachable_with_outside_held_goes_global_without_subset_call(self):
        g = ScriptedGlobalPlanner(status("no_solution", 0.1))
        s = ScriptedSubsetPlanner(AssertionError("subset must not be called"))
        result, planner, _ = run_selective(g, OPT_IN, s, grid=GRID_C, starts=STARTS_C, goals=GOALS_C,
                                           actions=ACTIONS_C)
        (rec,) = planner.records
        l0, l1, _ = rec.levels
        self.assertEqual((l0.skip_reason, l0.unreachable), (SKIP_UNREACHABLE, (10,)))
        self.assertEqual(l1.skip_reason, SKIP_NO_EXPANSION)
        self.assertEqual(s.calls, [])
        self.assertEqual(len(g.calls), 1)
        self.assertEqual(result["stop_reason"], "CBS_NO_SOLUTION")

    def test_full_group_goes_global(self):
        grid = np.zeros((3, 3), dtype=np.int64)
        starts, goals = {4: (0, 1), 9: (2, 1)}, {4: (2, 1), 9: (0, 1)}
        g = ScriptedGlobalPlanner(status("no_solution", 0.1))
        s = ScriptedSubsetPlanner(AssertionError("subset must not be called"))
        _, planner, _ = run_selective(g, OPT_IN, s, grid=grid, starts=starts, goals=goals,
                                      actions={4: [DOWN] * 9, 9: [UP] * 9})
        (rec,) = planner.records
        self.assertEqual([lv.skip_reason for lv in rec.levels[:2]], [SKIP_EQUALS_ALL, SKIP_EQUALS_ALL])
        self.assertEqual((s.calls, len(g.calls)), ([], 1))


# == 6. validator binding =====================================================================

class TestValidatorBinding(unittest.TestCase):

    def setUp(self):
        self.p = SelectivePlanner(ScriptedGlobalPlanner(), ShadowMonitor(DeadlockMonitor()), SelectiveConfig())
        self.v = SelectivePathValidator(self.p)
        from src.hybrid.selective_compose import compose_selective_plan
        self.full = compose_selective_plan(GROUP_PATHS_A, STARTS_A, GOALS_A, sorted(STARTS_A), GRID_A).full_paths

    def test_nothing_pending_is_rejected(self):
        self.assertEqual(self.v(GRID_A, STARTS_A, GOALS_A, self.full), (False, "SELECTIVE_NO_PENDING_PLAN"))

    def test_g_pending_keeps_all_agent_goal_check(self):
        self.p._pending = ("G",)
        ok, why = self.v(GRID_A, STARTS_A, GOALS_A, self.full)      # 7 held off goal
        self.assertFalse(ok)
        self.assertIn("incorrect goal position", why)

    def test_selective_pending_must_match_composed_plan(self):
        self.p._pending = ("SELECTIVE", (10, 20), self.full)
        self.assertEqual(self.v(GRID_A, STARTS_A, GOALS_A, self.full), (True, ""))
        self.assertIsNone(self.p._pending)                           # consumed
        # Defense in depth: even a pending plan that is itself bad (outside 7 moves) is rejected.
        bad = {**self.full, 7: [(1, 6), (1, 5), (1, 5), (1, 5), (1, 5)]}
        self.p._pending = ("SELECTIVE", (10, 20), bad)
        self.assertEqual(self.v(GRID_A, STARTS_A, GOALS_A, bad), (False, "SELECTIVE_OUTSIDE_NOT_HELD:7"))
        other = {a: q + [q[-1]] for a, q in self.full.items()}      # valid but not the composed plan
        self.p._pending = ("SELECTIVE", (10, 20), self.full)
        self.assertEqual(self.v(GRID_A, STARTS_A, GOALS_A, other), (False, "SELECTIVE_PLAN_NOT_THE_COMPOSED_ONE"))


# == 7. scope + DEV script ====================================================================

class TestScopeAndScript(unittest.TestCase):

    def test_default_runtime_does_not_import_selective_planner(self):
        for rel in ("src/hybrid/global_hybrid.py", "src/compare/common_runner.py",
                    "src/hybrid/cbs_runtime_bridge.py", "src/hybrid/selective_shadow.py"):
            self.assertNotIn("selective_planner", (ROOT / rel).read_text(encoding="utf-8"), rel)
        from src.compare import common_runner
        self.assertIn("SELECTIVE_HYBRID", common_runner.NOT_IMPLEMENTED)
        self.assertEqual(set(PENDING_BLOCKERS), {b for b in PENDING_BLOCKERS})
        self.assertEqual(len(PENDING_BLOCKERS), 5)

    def test_dev_script_writes_only_a_new_dir(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(dev.main(["--out-root", d]), 0)
            (out,) = list(Path(d).iterdir())
            self.assertEqual([p.name for p in out.iterdir()], ["summary.json"])
            s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(s["mock_only"])
            self.assertFalse(s["real_cbs_called"])
            self.assertTrue(s["default_g_equals_global"])
            rec = s["runs"]["SELECTIVE_OPT_IN_MOCK"]["records"][0]
            self.assertEqual(rec["outcome"], "EXECUTE:L0")
            self.assertFalse(rec["real_subset_cbs_called"])
            self.assertEqual(rec["attempts"][0]["planner_kind"], "MOCK")
            with patch.object(dev, "datetime") as dt:             # same timestamp -> refused
                dt.now.return_value = __import__("datetime").datetime.strptime(out.name[:15], "%Y%m%d_%H%M%S")
                with self.assertRaises(FileExistsError):
                    dev.main(["--out-root", d])

    def test_dev_script_refuses_real(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(dev.main(["--real", "--out-root", d]), 2)
            self.assertEqual(list(Path(d).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
