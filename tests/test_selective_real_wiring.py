"""1st connection of the REAL subset / bounded Global CBS adapters to SelectivePlanner.

Layers (kept apart on purpose; a lower layer is never reported as a higher one):

  A. Gate          constructor opt-in rules. No solver.
  B. Runner, MOCK solver
                   the unchanged Global Hybrid Runner + SelectivePlanner + the REAL
                   SubsetCbsPlanner / BoundedGlobalCbsPlanner, with only
                   ``run_cbs_runtime`` replaced (no atb033 process). Shows wiring,
                   ID/coord preservation, outside WAIT, budget and failure handling.
  C. Mock planners budget rule (ScriptedSubsetPlanner / ScriptedGlobalPlanner).
  D. REAL SOLVER   atb033 ``cbs.py`` really runs, planner level only (no Runner,
                   no Simulator). Skipped unless ALPHA2_CBS_SOLVER_ROOT points to
                   the atb033 cbs directory. Run with PYTHONHASHSEED set.

No real Runner E2E scenario is run here. Budget numbers are TEST values (PENDING P4).
"""

import math
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.run_selective_mock_dev import (
    ACTIONS_A, GLOBAL_PATHS_A, GOALS_A, GRID_A, GROUP_PATHS_A, STARTS_A, TEST_BUDGET,
    ScriptedGlobalPlanner, ScriptedSubsetPlanner, hybrid_config, run_global, success,
)
from src.cbs_runtime_adapter import CBSRuntimeResult
from src.common.deadlock_monitor import DeadlockMonitor
from src.hybrid import selective_planner as sp
from src.hybrid import subset_cbs_adapter as adapter_mod
from src.hybrid.cbs_runtime_bridge import CbsRuntimePlanner, PathGuardValidator
from src.hybrid.global_hybrid import run_global_hybrid_episode
from src.hybrid.selective_planner import (
    BUDGET_OVERRUN,
    BUDGET_UNVERIFIABLE,
    AttemptRecord,
    SelectiveConfig,
    SelectivePathValidator,
    SelectivePlanner,
)
from src.hybrid.selective_safety import TriggerBudget
from src.hybrid.selective_shadow import ShadowMonitor
from src.hybrid.subset_cbs_adapter import (
    GLOBAL_CALL_DIR_PREFIX,
    BoundedGlobalCbsPlanner,
    SubsetCbsPlanner,
)
from tests.hybrid_mocks import MockReplanResult, ScriptedPolicy

WAIT = 4
OPT_IN = SelectiveConfig(ladder=("L0", "L1", "G"), budget=TEST_BUDGET, acknowledge_unverified_outside_policy=True)
RESERVE_MS = 50.0


def status(s, latency_sec):
    return MockReplanResult(status=s, paths=None, latency_sec=latency_sec, error=f"mock {s}")


class FakeRuntime:
    """Replaces run_cbs_runtime (shared by both adapters). Answers by agent set."""

    def __init__(self, answers, sleep=None):
        self.answers = answers            # {frozenset(ids): CBSRuntimeResult}
        self.sleep = sleep or {}          # {frozenset(ids): seconds}
        self.calls = []

    def __call__(self, current_positions, goals, grid_map, timeout_sec=30, solver_root=None):
        key = frozenset(current_positions)
        self.calls.append(dict(ids=key, current_positions=dict(current_positions), goals=dict(goals),
                               grid=np.array(grid_map, copy=True), timeout_sec=timeout_sec, cwd=Path.cwd()))
        if key in self.sleep:
            time.sleep(self.sleep[key])
        return self.answers[key]


def ok(paths):
    return CBSRuntimeResult("success", {a: list(p) for a, p in paths.items()}, 0.01)


GROUP = frozenset((10, 20))
ALL = frozenset(STARTS_A)


class _Tmp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.solver = self.tmp / "solver"
        self.solver.mkdir()
        (self.solver / "cbs.py").write_text("raise SystemExit('fake solver must never run')\n", encoding="utf-8")
        self.cwd0 = Path.cwd()

    def tearDown(self):
        self.assertEqual(Path.cwd(), self.cwd0, "cwd must be restored")
        self._tmp.cleanup()

    def _root(self, name):
        self._n = getattr(self, "_n", 0) + 1            # a fresh work root per planner (dirs are never reused)
        return self.tmp / f"{name}_{self._n:02d}"

    def subset(self, reserve=RESERVE_MS):
        return SubsetCbsPlanner(solver_root=self.solver, work_root=self._root("subset"), overhead_reserve_ms=reserve)

    def glob(self, reserve=RESERVE_MS):
        return BoundedGlobalCbsPlanner(solver_root=self.solver, work_root=self._root("global"),
                                       overhead_reserve_ms=reserve)


def run_runner(global_planner, subset_planner, config=OPT_IN, *, allow=True, h_exec=4, max_steps=20):
    mon = ShadowMonitor(DeadlockMonitor())
    planner = SelectivePlanner(global_planner, mon, config, subset_planner=subset_planner,
                               allow_real_subset_cbs=allow)
    validator = SelectivePathValidator(planner)
    result = run_global_hybrid_episode(
        ScriptedPolicy({a: list(v) for a, v in ACTIONS_A.items()}), GRID_A, STARTS_A, GOALS_A, max_steps,
        planner=planner, path_validator=validator, config=hybrid_config(h_exec), monitor=mon)
    return result, planner, validator


def cbs_moves(result):
    return [i for i, s in enumerate(result["action_sources"]) if s == "CBS"]


# == A. gate ==================================================================================

class TestGate(_Tmp):

    def test_real_subset_needs_explicit_flag_and_exact_class(self):
        mon = ShadowMonitor(DeadlockMonitor())
        g = ScriptedGlobalPlanner()
        real = self.subset()
        for flag in (False, None, 1, "yes"):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "PENDING:REAL_SUBSET_CBS"):
                SelectivePlanner(g, mon, OPT_IN, subset_planner=real, allow_real_subset_cbs=flag)
        with self.assertRaisesRegex(ValueError, "PENDING:REAL_SUBSET_CBS"):
            SelectivePlanner(g, mon, OPT_IN, subset_planner=real)                    # default = refused

        class Sub(SubsetCbsPlanner):
            pass

        class RealLooking:
            calls = []

            def __call__(self, *a, **k):
                raise AssertionError("must not be called")

        for other in (Sub(solver_root=self.solver, work_root=self.tmp / "x", overhead_reserve_ms=0.0),
                      RealLooking(), self.glob()):
            with self.subTest(type(other).__name__), self.assertRaisesRegex(ValueError, "PENDING:REAL_SUBSET_CBS"):
                SelectivePlanner(g, mon, OPT_IN, subset_planner=other, allow_real_subset_cbs=True)
        p = SelectivePlanner(g, mon, OPT_IN, subset_planner=real, allow_real_subset_cbs=True)
        self.assertTrue(p.allow_real_subset_cbs)
        # Mock subset planners stay accepted without the flag
        SelectivePlanner(g, mon, OPT_IN, subset_planner=ScriptedSubsetPlanner())

    def test_real_global_must_be_bounded(self):
        mon = ShadowMonitor(DeadlockMonitor())
        s = ScriptedSubsetPlanner()
        with self.assertRaisesRegex(ValueError, "timeout_ms"):
            SelectivePlanner(CbsRuntimePlanner(solver_root=self.solver, work_root=self.tmp / "c"), mon, OPT_IN,
                             subset_planner=s)

        class Claims:
            paths_keying = "agent_id"
            accepts_timeout_ms = True

        with self.assertRaisesRegex(ValueError, "BoundedGlobalCbsPlanner"):
            SelectivePlanner(Claims(), mon, OPT_IN, subset_planner=s)
        SelectivePlanner(self.glob(), mon, OPT_IN, subset_planner=s)
        SelectivePlanner(self.glob(), mon, OPT_IN, subset_planner=self.subset(), allow_real_subset_cbs=True)
        self.assertTrue(BoundedGlobalCbsPlanner.accepts_timeout_ms)
        self.assertFalse(BoundedGlobalCbsPlanner.test_only)

    def test_default_mode_is_plain_global_even_with_real_subset_given(self):
        real = self.subset()
        base = run_global(ScriptedGlobalPlanner(success(GLOBAL_PATHS_A, 0.2)))
        mon = ShadowMonitor(DeadlockMonitor())
        g = ScriptedGlobalPlanner(success(GLOBAL_PATHS_A, 0.2))
        planner = SelectivePlanner(g, mon, SelectiveConfig(), subset_planner=real, allow_real_subset_cbs=True)
        res = run_global_hybrid_episode(
            ScriptedPolicy({a: list(v) for a, v in ACTIONS_A.items()}), GRID_A, STARTS_A, GOALS_A, 20,
            planner=planner, path_validator=SelectivePathValidator(planner), config=hybrid_config(4), monitor=mon)
        self.assertEqual(res, base)
        self.assertEqual(real.calls, [])
        self.assertEqual([(r.outcome, r.attempts, r.real_subset_cbs_called) for r in planner.records],
                         [("PASS_THROUGH", [], False)])


# == B. Runner + real adapters, run_cbs_runtime mocked ============================================

class TestRunnerRealAdaptersMockSolver(_Tmp):

    def _run(self, fake, *, reserve=RESERVE_MS, config=OPT_IN, **kw):
        s, g = self.subset(reserve), self.glob(reserve)
        with patch.object(adapter_mod, "run_cbs_runtime", fake):
            result, planner, validator = run_runner(g, s, config, **kw)
        (rec,) = planner.records
        return result, rec, s, g, validator

    def test_l0_real_adapter_ids_outside_wait_full_guard(self):
        fake = FakeRuntime({GROUP: ok(GROUP_PATHS_A)})
        result, rec, s, g, v = self._run(fake)
        self.assertEqual(rec.outcome, "EXECUTE:L0")
        (att,) = rec.attempts
        self.assertEqual((att.planner_kind, att.solver_invoked, att.status, att.raw_status), ("REAL", True,
                                                                                            "success", "success"))
        self.assertTrue(rec.real_subset_cbs_called)
        self.assertFalse(rec.real_global_cbs_called)
        (call,) = fake.calls
        self.assertEqual(call["current_positions"], {10: (0, 1), 20: (2, 1)})        # original ids, S_t
        self.assertEqual(call["goals"], {10: (2, 1), 20: (0, 1)})
        self.assertEqual(int(call["grid"][1, 6]), 1)                                # outside 7 = obstacle
        self.assertEqual(int(call["grid"].sum()), 1)
        self.assertEqual(int(GRID_A.sum()), 0)                                      # caller grid untouched
        self.assertIsInstance(call["timeout_sec"], float)
        self.assertEqual(call["timeout_sec"], (400.0 - RESERVE_MS) / 1000.0)        # never rounded up
        self.assertEqual(call["cwd"], s.calls[0]["work_dir"])
        self.assertTrue(s.calls[0]["work_dir"].is_relative_to(self.tmp))
        self.assertEqual(g.calls, [])
        self.assertEqual(v.calls, [("SELECTIVE", (True, ""))])
        self.assertEqual((rec.executed_group, rec.outside_held), ((10, 20), (7,)))
        self.assertEqual(cbs_moves(result), [2, 3, 4, 5])
        for k in range(4):
            self.assertEqual(result["actions"][2 + k][7], WAIT)
        for e in result["cbs_exec_trace"]:
            self.assertTrue(e["match"])
            self.assertEqual(set(e["expected"]), {7, 10, 20})
        self.assertEqual(result["termination"], "SUCCESS")
        self.assertLessEqual(att.latency_ms, att.timeout_ms)

    def test_l0_no_solution_falls_back_to_real_bounded_global(self):
        fake = FakeRuntime({GROUP: CBSRuntimeResult("no_solution", None, 0.01, "x"),
                            ALL: ok(GLOBAL_PATHS_A)})
        result, rec, s, g, v = self._run(fake)
        self.assertEqual([(a.level, a.status, a.solver_invoked) for a in rec.attempts],
                         [("L0", "no_solution", True), ("G", "success", True)])
        self.assertEqual(rec.outcome, "EXECUTE:G")
        self.assertTrue(rec.real_subset_cbs_called and rec.real_global_cbs_called)
        g_att = rec.attempts[1]
        self.assertAlmostEqual(g_att.timeout_ms, min(800.0, 1000.0 - rec.attempts[0].latency_ms))
        g_call = fake.calls[1]
        self.assertEqual(set(g_call["current_positions"]), {7, 10, 20})
        self.assertEqual(int(g_call["grid"].sum()), 0)                              # no outside obstacles in G
        self.assertAlmostEqual(g_call["timeout_sec"], (g_att.timeout_ms - RESERVE_MS) / 1000.0)
        self.assertLess(g_call["timeout_sec"] * 1000.0, g_att.timeout_ms)
        self.assertTrue(g.calls[0]["work_dir"].name.startswith(GLOBAL_CALL_DIR_PREFIX))
        self.assertEqual(v.calls, [("G", (True, ""))])
        self.assertEqual(result["termination"], "SUCCESS")
        total = sum(a.latency_ms for a in rec.attempts)
        self.assertAlmostEqual(result["cbs_calls"][0]["latency_ms"], total)          # failed L0 charged

    def test_invalid_results_never_executed(self):
        through_outside = {10: GROUP_PATHS_A[10],
                           20: [(2, 1), (2, 2), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 5), (1, 4), (1, 3),
                                (1, 2), (0, 2), (0, 1)]}
        cases = {
            "path through held outside": through_outside,
            "wrong agent id": {10: GROUP_PATHS_A[10], 21: GROUP_PATHS_A[20]},
            "start != S_t": {10: [(1, 1), (2, 1)], 20: GROUP_PATHS_A[20]},
            "goal not reached": {10: [(0, 1), (0, 1)], 20: GROUP_PATHS_A[20]},
            "edge swap": {10: [(0, 1), (1, 1), (2, 1)], 20: [(2, 1), (2, 1), (1, 1), (0, 1)]},
        }
        for name, paths in cases.items():
            with self.subTest(name):
                fake = FakeRuntime({GROUP: CBSRuntimeResult("success", paths, 0.01), ALL: ok(GLOBAL_PATHS_A)})
                result, rec, s, g, v = self._run(fake)
                self.assertEqual(rec.attempts[0].status, "invalid_path")
                self.assertEqual(rec.outcome, "SAFE_STOP:CBS_INVALID_PATH")
                self.assertEqual(g.calls, [])
                self.assertEqual((result["termination"], result["stop_reason"]), ("SAFE_STOP", "CBS_INVALID_PATH"))
                self.assertEqual(cbs_moves(result), [])
                self.assertEqual(v.calls, [])

    def test_failure_statuses_keep_policy(self):
        for st, outcome in (("timeout", "EXECUTE:G"), ("no_solution", "EXECUTE:G"),
                            ("invalid_path", "SAFE_STOP:CBS_INVALID_PATH"),
                            ("solver_error", "SAFE_STOP:UNAGREED_STATUS:solver_error")):
            with self.subTest(st):
                fake = FakeRuntime({GROUP: CBSRuntimeResult(st, None, 0.01, "x"), ALL: ok(GLOBAL_PATHS_A)})
                result, rec, *_ = self._run(fake)
                self.assertEqual(rec.outcome, outcome)

    def test_blocked_before_call_is_not_a_real_call(self):
        fake = FakeRuntime({})
        result, rec, s, g, _ = self._run(fake, reserve=450.0)                        # L0 400 ms <= reserve
        (att,) = rec.attempts
        self.assertEqual((att.planner_kind, att.solver_invoked, att.status), ("REAL", False, "input_error"))
        self.assertFalse(rec.real_subset_cbs_called)
        self.assertEqual(fake.calls, [])
        self.assertEqual(list(s.work_root.glob("*")) if s.work_root.exists() else [], [])
        self.assertEqual(rec.outcome, "SAFE_STOP:UNAGREED_STATUS:input_error")
        self.assertEqual(result["stop_reason"], "CBS_INPUT_ERROR")

    def test_late_subset_success_is_not_executed(self):
        fake = FakeRuntime({GROUP: ok(GROUP_PATHS_A), ALL: ok(GLOBAL_PATHS_A)}, sleep={GROUP: 0.45})
        result, rec, s, g, v = self._run(fake)
        l0, gl = rec.attempts
        self.assertEqual((l0.raw_status, l0.status, l0.full_check_ok), ("success", "timeout", None))
        self.assertTrue(l0.full_check_reason.startswith(BUDGET_OVERRUN))
        self.assertGreater(l0.budget_overrun_ms, 0.0)
        self.assertTrue(rec.real_subset_cbs_called)                              # it ran, and was refused
        self.assertEqual(rec.outcome, "EXECUTE:G")
        self.assertLessEqual(gl.timeout_ms, 1000.0 - l0.latency_ms + 1e-9)
        self.assertEqual(v.calls, [("G", (True, ""))])

    def test_late_global_success_is_not_executed(self):
        fake = FakeRuntime({GROUP: CBSRuntimeResult("no_solution", None, 0.01, "x"), ALL: ok(GLOBAL_PATHS_A)},
                           sleep={ALL: 0.85})
        result, rec, s, g, v = self._run(fake)
        self.assertEqual([(a.level, a.raw_status, a.status) for a in rec.attempts],
                         [("L0", "no_solution", "no_solution"), ("G", "success", "timeout")])
        self.assertEqual(rec.outcome, "SAFE_STOP:LADDER_EXHAUSTED:CBS_TIMEOUT")
        self.assertEqual((result["termination"], result["stop_reason"]), ("SAFE_STOP", "CBS_TIMEOUT"))
        self.assertEqual(cbs_moves(result), [])
        self.assertEqual(v.calls, [])

    def test_bounded_global_refuses_unbounded_call(self):
        g = self.glob()
        fake = FakeRuntime({ALL: ok(GLOBAL_PATHS_A)})
        with patch.object(adapter_mod, "run_cbs_runtime", fake):
            r = g(dict(STARTS_A), dict(GOALS_A), GRID_A)
        self.assertEqual((r.status, r.paths), ("input_error", None))
        self.assertIn("TIMEOUT_MS", r.error)
        self.assertEqual(fake.calls, [])
        self.assertFalse(g.calls[0]["called"])
        with patch.object(adapter_mod, "run_cbs_runtime", fake):
            r = g(dict(STARTS_A), dict(GOALS_A), GRID_A, timeout_ms=999)
        self.assertEqual(r.status, "success")
        self.assertEqual(fake.calls[0]["timeout_sec"], (999 - RESERVE_MS) / 1000.0)
        self.assertEqual(set(r.paths), {7, 10, 20})
        # a Global answer that leaves an agent off its goal is refused
        bad = {**GLOBAL_PATHS_A, 7: [(1, 6)] * 5}
        with patch.object(adapter_mod, "run_cbs_runtime", FakeRuntime({ALL: ok(bad)})):
            r = g(dict(STARTS_A), dict(GOALS_A), GRID_A, timeout_ms=999)
        self.assertEqual((r.status, r.paths), ("invalid_path", None))


# == C. Mock planners: the same budget rule, still labelled MOCK ===============================

class TestMockBudgetRule(unittest.TestCase):

    def _run(self, *subset, glob=None):
        g = glob or ScriptedGlobalPlanner(success(GLOBAL_PATHS_A, 0.2))
        s = ScriptedSubsetPlanner(*subset)
        result, planner, validator = run_runner(g, s, allow=False)
        (rec,) = planner.records
        return result, rec, g, s, validator

    def test_mock_success_stays_mock(self):
        _, rec, *_ = self._run(success(GROUP_PATHS_A, 0.1))
        self.assertEqual(rec.outcome, "EXECUTE:L0")
        self.assertEqual((rec.attempts[0].planner_kind, rec.attempts[0].solver_invoked), ("MOCK", None))
        self.assertFalse(rec.real_subset_cbs_called or rec.real_global_cbs_called)

    def test_mock_late_success_falls_to_global(self):
        result, rec, g, *_ = self._run(success(GROUP_PATHS_A, 0.5))                 # 500 > 400 granted
        self.assertEqual((rec.attempts[0].raw_status, rec.attempts[0].status), ("success", "timeout"))
        self.assertEqual(g.calls[0]["timeout_ms"], 500.0)                           # overrun charged
        self.assertEqual(rec.outcome, "EXECUTE:G")

    def test_mock_unknown_or_bad_latency_success_never_executes(self):
        for lat in (None, float("nan"), float("inf"), -0.1, "0.1x"):
            with self.subTest(lat=lat):
                result, rec, g, *_ = self._run(success(GROUP_PATHS_A, lat))
                att = rec.attempts[0]
                self.assertEqual((att.raw_status, att.status, att.latency_ms), ("success", "timeout", None))
                self.assertTrue(att.full_check_reason.startswith(BUDGET_UNVERIFIABLE))
                self.assertEqual(rec.outcome, "SAFE_STOP:TIME_BUDGET_EXHAUSTED")
                self.assertEqual(g.calls, [])
                self.assertEqual(cbs_moves(result), [])

    def test_second_guard_refuses_execute_over_total(self):
        real_call = SelectivePlanner._call_level

        def inflated(self, rec, plan, *a):
            res, att, extra = real_call(self, rec, plan, *a)
            return res, AttemptRecord(att.level, att.group, att.timeout_ms, att.status, 1200.0,
                                      att.full_check_ok, att.full_check_reason, att.planner_kind), extra

        with patch.object(SelectivePlanner, "_call_level", inflated):
            result, rec, g, *_ = self._run(success(GROUP_PATHS_A, 0.1))
        self.assertEqual(rec.outcome, "SAFE_STOP:TIME_BUDGET_OVERRUN")
        self.assertEqual(result["stop_reason"], "CBS_TIMEOUT")
        self.assertEqual(cbs_moves(result), [])


# == D. REAL SOLVER, planner level only ==========================================================

SOLVER_ROOT = os.environ.get("ALPHA2_CBS_SOLVER_ROOT")
HAVE_SOLVER = bool(SOLVER_ROOT) and (Path(SOLVER_ROOT) / "cbs.py").is_file()


def fed_monitor():
    """ShadowMonitor holding fixture A's Trigger step (S_1 -> S_2 == S_0), fed by hand -- no Runner."""
    mon = ShadowMonitor(DeadlockMonitor())
    mon.reset(dict(STARTS_A), dict(GOALS_A))
    s1 = {7: (1, 5), 10: (0, 1), 20: (2, 1)}
    ct = {7: "none", 10: "vertex", 20: "vertex"}
    mon.observe_step(1, s1, ct, current_positions=dict(STARTS_A), intended_positions={7: (1, 5), 10: (1, 1),
                                                                                       20: (1, 1)})
    mon.observe_step(2, dict(STARTS_A), ct, current_positions=s1, intended_positions={7: (1, 6), 10: (1, 1),
                                                                                       20: (1, 1)})
    return mon


@unittest.skipUnless(HAVE_SOLVER, "REAL SOLVER: set ALPHA2_CBS_SOLVER_ROOT to the atb033 cbs directory")
class TestRealSolverPlannerLevel(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def mk(self, cls, name, reserve=100.0):
        return cls(solver_root=SOLVER_ROOT, work_root=self.tmp / name, overhead_reserve_ms=reserve)

    def test_real_subset_ids_and_outside_held(self):
        p = self.mk(SubsetCbsPlanner, "s")
        r = p({10: (0, 1), 20: (2, 1)}, {10: (2, 1), 20: (0, 1)}, GRID_A, outside={7: (1, 6)}, timeout_ms=15000.0)
        self.assertEqual(r.status, "success", r.error)
        self.assertEqual(set(r.paths), {10, 20})
        self.assertTrue(p.calls[0]["called"])
        self.assertTrue(all((1, 6) not in path for path in r.paths.values()))
        self.assertLessEqual(r.latency_sec * 1000.0, 15000.0)

    def test_real_bounded_global(self):
        g = self.mk(BoundedGlobalCbsPlanner, "g")
        r = g(dict(STARTS_A), dict(GOALS_A), GRID_A, timeout_ms=15000.0)
        self.assertEqual(r.status, "success", r.error)
        self.assertEqual(set(r.paths), {7, 10, 20})
        self.assertEqual(PathGuardValidator()(GRID_A, STARTS_A, GOALS_A, r.paths), (True, ""))
        self.assertEqual(g.calls[0]["solver_timeout_sec"], (15000.0 - 100.0) / 1000.0)

    def test_real_timeout_stops_solver(self):
        n, k = 10, 10                                    # needs ~2.7 s on the dev machine
        grid = np.zeros((n, n), dtype=int)
        starts = {100 + i: (i, 0) for i in range(k)}
        goals = {100 + i: (k - 1 - i, n - 1) for i in range(k)}
        g = self.mk(BoundedGlobalCbsPlanner, "t", reserve=0.0)
        r = g(starts, goals, grid, timeout_ms=300.0)
        self.assertEqual((r.status, r.paths), ("timeout", None), r.error)
        self.assertTrue(g.calls[0]["called"])
        # NOT a hard bound: kill + pipe drain overrun is recorded (measured ~55-105 ms). Sanity limit only.
        self.assertGreaterEqual(g.calls[0]["budget_overrun_ms"], 0.0)
        self.assertLess(r.latency_sec * 1000.0, 300.0 + 2000.0)

    def test_selective_planner_l0_real(self):
        mon = fed_monitor()
        budget = TriggerBudget(20000.0, 3, {"L0": 15000.0, "L1": 15000.0, "G": 15000.0})
        cfg = SelectiveConfig(ladder=("L0", "L1", "G"), budget=budget, acknowledge_unverified_outside_policy=True)
        s, g = self.mk(SubsetCbsPlanner, "s"), self.mk(BoundedGlobalCbsPlanner, "g")
        planner = SelectivePlanner(g, mon, cfg, subset_planner=s, allow_real_subset_cbs=True)
        out = planner(dict(STARTS_A), dict(GOALS_A), GRID_A)
        (rec,) = planner.records
        self.assertEqual(rec.wfg.groups, ((10, 20),))
        self.assertEqual(rec.outcome, "EXECUTE:L0", rec.attempts)
        self.assertEqual((out.status, out.level, out.group), ("success", "L0", (10, 20)))
        self.assertTrue(rec.real_subset_cbs_called)
        self.assertFalse(rec.real_global_cbs_called)
        self.assertEqual(rec.attempts[0].planner_kind, "REAL")
        self.assertEqual(set(out.paths), {7, 10, 20})
        self.assertTrue(all(c == (1, 6) for c in out.paths[7]))                    # outside held (WAIT)
        self.assertEqual(SelectivePathValidator(planner)(GRID_A, STARTS_A, GOALS_A, out.paths), (True, ""))
        self.assertEqual(g.calls, [])

    def test_selective_planner_real_l0_timeout_then_real_g(self):
        mon = fed_monitor()
        budget = TriggerBudget(20000.0, 3, {"L0": 60.0, "L1": 60.0, "G": 15000.0})
        cfg = SelectiveConfig(ladder=("L0", "G"), budget=budget, acknowledge_unverified_outside_policy=True)
        s, g = self.mk(SubsetCbsPlanner, "s", reserve=0.0), self.mk(BoundedGlobalCbsPlanner, "g")
        planner = SelectivePlanner(g, mon, cfg, subset_planner=s, allow_real_subset_cbs=True)
        out = planner(dict(STARTS_A), dict(GOALS_A), GRID_A)
        (rec,) = planner.records
        l0, gl = rec.attempts
        self.assertEqual(l0.status, "timeout")                                     # 60 ms < interpreter start
        self.assertTrue(rec.real_subset_cbs_called and rec.real_global_cbs_called)
        self.assertEqual(rec.outcome, "EXECUTE:G", rec.attempts)
        self.assertAlmostEqual(gl.timeout_ms, min(15000.0, 20000.0 - l0.latency_ms))
        self.assertAlmostEqual(out.latency_sec * 1000.0, l0.latency_ms + gl.latency_ms, places=6)  # L0 charged
        self.assertEqual(SelectivePathValidator(planner)(GRID_A, STARTS_A, GOALS_A, out.paths), (True, ""))


if __name__ == "__main__":
    unittest.main()
