"""src/compare/common_runner.py: one interface, five existing runners.

Each method's EpisodeRecord must equal a direct call of the existing runner
on the same inputs (real NavHint checkpoint, real Corridor scenario). CBS
methods use the real CbsRuntimePlanner + PathGuardValidator; only the solver
output is injected (replan_with_status patched with the paths atb033 returned
in the saved Corridor runs), so no solver runs and results are deterministic.
The last class runs the real atb033 solver (skipped when absent).
"""

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_corridor_hybrid_e2e import dev_config
from scripts.run_hybrid_resume_dev import dev_resume_config
from scripts.run_il import DEFAULT_CKPT, PROJECT_ROOT, load_scenario, run_episode
from src.common.deadlock_monitor import DeadlockMonitor
from src.common.safety_only import SafetyOnlyV0
from src.compare.common_runner import (
    CBS_METHODS,
    DEV_SETTINGS,
    GLOBAL_HYBRID_DEV_FULL,
    GLOBAL_HYBRID_DEV_H3,
    IL_ONLY,
    METHODS,
    PURE_CBS_DEV_SINGLE_PLAN,
    SAFETY_ONLY_DEV_K2,
    RunSpec,
    run_method,
)
from src.compare.pure_cbs import run_pure_cbs_episode
from src.hybrid.cbs_runtime_bridge import CbsRuntimePlanner, PathGuardValidator
from src.hybrid.global_hybrid import run_global_hybrid_episode
from src.replan_status import ReplanResult
from src.simulator.mapf_step_simulator import MAPFStepSimulator

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

CORRIDOR = PROJECT_ROOT / "scenarios" / "controlled" / "scenario_corridor_bay_n2.json"
REAL_SOLVER_ROOT = Path(os.environ.get(
    "ALPHA2_CBS_SOLVER_ROOT",
    PROJECT_ROOT.parent / "MAPF_ALPHA2_CBS_CHECK/third_party/multi_agent_path_planning/centralized/cbs",
))
HAVE_REAL_SOLVER = (REAL_SOLVER_ROOT / "cbs.py").is_file()


def _row(*cols_and_rows):
    return [tuple(c) for c in cols_and_rows]


# atb033 output recorded in the saved Corridor runs (solver index order = agent id order).
S_0 = ((2, 0), (2, 10))
PURE_PLAN = {
    0: _row((2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (1, 5), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10)),
    1: _row((2, 10), (2, 9), (2, 8), (2, 7), (2, 6), (2, 6), (2, 5), (2, 4), (2, 3), (2, 2), (2, 1), (2, 0), (2, 0)),
}
S_T5 = ((2, 4), (2, 6))
GLOBAL_PLAN = {
    0: _row((2, 4), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 10)),
    1: _row((2, 6), (2, 5), (1, 5), (2, 5), (2, 4), (2, 3), (2, 2), (2, 1), (2, 0)),
}


def recorded_solver(*, current_positions, goals, **kwargs):
    """Stand-in for replan_with_status: the recorded atb033 plan for that exact state."""
    plans = {S_0: PURE_PLAN, S_T5: GLOBAL_PLAN}
    plan = plans.get(tuple(current_positions))
    if plan is None:
        return ReplanResult(status="solver_error", paths=None, latency_sec=0.01,
                            error=f"no recorded plan for {current_positions}")
    return ReplanResult(status="success", paths=plan, latency_sec=0.05)


class _Case(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.solver_root = self.tmp / "solver"
        self.solver_root.mkdir()
        (self.solver_root / "cbs.py").write_text("raise SystemExit('fake solver must not run')\n",
                                                 encoding="utf-8")
        self.out = self.tmp / "runs"

    def tearDown(self):
        self._tmp.cleanup()

    def spec(self, method, **kw):
        kw.setdefault("solver_root", self.solver_root)
        return RunSpec(method=method, scenario=CORRIDOR, out_dir=self.out, **kw)

    def run_recorded(self, method, **kw):
        with patch("src.cbs_runtime_adapter.replan_with_status", side_effect=recorded_solver):
            return run_method(self.spec(method, **kw))

    def direct_planner(self, name):
        return CbsRuntimePlanner(solver_root=self.solver_root, work_root=self.tmp / name)


COMPARED = ("steps", "done", "all_at_goal", "timed_out", "final_positions", "goals", "trajectory", "actions")


# -- Same result as the existing runners ----------------------------------------

@unittest.skipIf(torch is None, "torch is not installed")
class TestSameResultAsExistingRunners(_Case):

    def setUp(self):
        super().setUp()
        from src.il.policy import NavHintPolicy
        self.policy = lambda: NavHintPolicy(DEFAULT_CKPT)
        self.grid, self.starts, self.goals = load_scenario(CORRIDOR)

    def test_il_only(self):
        record = run_method(self.spec(IL_ONLY))
        direct = run_episode(self.policy(), self.grid, self.starts, self.goals, 160, monitor=DeadlockMonitor())
        self.assertEqual(record.raw, direct)
        self.assertEqual((record.termination, record.termination_source), ("TIMEOUT", "DERIVED_FROM_FLAGS"))
        self.assertEqual((record.steps, record.all_at_goal, record.max_steps), (160, False, 160))
        self.assertIs(record.trajectory, record.raw["trajectory"])
        self.assertEqual(record.action_sources, ["IL"] * 160)
        self.assertEqual(record.monitor["trigger_timestep"], 5)

    def test_safety_only_dev_k2(self):
        record = run_method(self.spec(SAFETY_ONLY_DEV_K2))
        direct = run_episode(self.policy(), self.grid, self.starts, self.goals, 160,
                             monitor=DeadlockMonitor(), safety=SafetyOnlyV0(hold_steps=2))
        self.assertEqual(record.raw, direct)    # the recorder never changes an action
        self.assertEqual(record.termination, "TIMEOUT")
        self.assertEqual(record.safety_armed_t, 5)
        self.assertEqual(record.safety_holds, [(6, 0), (7, 0)])
        self.assertEqual(record.action_sources[5:7], ["SAFETY", "SAFETY"])

    def test_pure_cbs(self):
        record = self.run_recorded(PURE_CBS_DEV_SINGLE_PLAN)
        with patch("src.cbs_runtime_adapter.replan_with_status", side_effect=recorded_solver):
            direct = run_pure_cbs_episode(self.grid, self.starts, self.goals, 160,
                                          planner=self.direct_planner("direct"),
                                          path_validator=PathGuardValidator(), monitor=DeadlockMonitor())
        for key in COMPARED + ("termination", "stop_reason", "action_sources", "cbs_exec_trace",
                               "transitions", "plan_horizon"):
            self.assertEqual(record.raw[key], direct[key], key)
        self.assertEqual((record.termination, record.termination_source, record.steps), ("SUCCESS", "RUNNER", 12))
        self.assertEqual(record.cbs_calls[0]["s_0"], self.starts)
        self.assertEqual(record.cbs_timeout_sec, 30)

    def test_global_full_and_h3(self):
        for method, config, steps in ((GLOBAL_HYBRID_DEV_FULL, dev_config(160), 13),
                                      (GLOBAL_HYBRID_DEV_H3, dev_resume_config(3), 13)):
            with self.subTest(method):
                record = self.run_recorded(method)
                with patch("src.cbs_runtime_adapter.replan_with_status", side_effect=recorded_solver):
                    direct = run_global_hybrid_episode(
                        self.policy(), self.grid, self.starts, self.goals, 160,
                        planner=self.direct_planner(f"direct_{method}"), path_validator=PathGuardValidator(),
                        config=config, monitor=DeadlockMonitor())
                for key in COMPARED + ("termination", "stop_reason", "action_sources", "cbs_exec_trace",
                                       "transitions", "cbs_calls"):
                    self.assertEqual(record.raw[key], direct[key], key)
                self.assertEqual((record.termination, record.steps), ("SUCCESS", steps))
        self.assertIn(("CBS_EXEC", "IL_RESUME"), [(t["from"], t["to"]) for t in record.transitions])


# -- Methods stay separate --------------------------------------------------------

@unittest.skipIf(torch is None, "torch is not installed")
class TestMethodsDoNotMix(_Case):

    def test_all_five_methods_same_inputs_own_outputs(self):
        records = {m: self.run_recorded(m) for m in METHODS}
        self.assertEqual(len({r.run_dir for r in records.values()}), 5)
        self.assertEqual(len({r.scenario_sha256 for r in records.values()}), 1)
        self.assertEqual(len({r.grid_sha256 for r in records.values()}), 1)
        self.assertEqual({r.max_steps for r in records.values()}, {160})
        self.assertEqual({r.agent_ids for r in records.values()}, {(0, 1)})
        for method, r in records.items():
            with self.subTest(method):
                self.assertEqual(r.method, method)
                self.assertEqual(r.dev_settings, DEV_SETTINGS[method])
                run_dir = Path(r.run_dir)
                self.assertTrue(run_dir.name.endswith(method))
                with (run_dir / f"{r.run_id}_steps.csv").open(newline="", encoding="utf-8") as fh:
                    rows = list(csv.DictReader(fh))
                self.assertEqual({row["mode"] for row in rows}, {method})
                self.assertEqual(len(rows), r.steps * 2)
                result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
                self.assertEqual((result["method"], result["termination"]), (method, r.termination))
                for name in ("summary", "monitor"):
                    self.assertTrue((run_dir / f"{r.run_id}_{name}.csv").is_file())
                self.assertEqual((run_dir / "cbs").is_dir(), method in CBS_METHODS)
                if method in CBS_METHODS:
                    self.assertEqual(r.cbs_timeout_sec, 30)
                    self.assertEqual(len(r.cbs_calls), 1)
                else:
                    self.assertIsNone(r.cbs_timeout_sec)
                    self.assertEqual(r.cbs_calls, [])
                    self.assertNotIn("CBS", r.action_sources)
        self.assertIsNone(records[PURE_CBS_DEV_SINGLE_PLAN].checkpoint_sha256)
        self.assertEqual(set(records[PURE_CBS_DEV_SINGLE_PLAN].action_sources), {"CBS"})
        il_ckpt = {records[m].checkpoint_sha256 for m in METHODS if m != PURE_CBS_DEV_SINGLE_PLAN}
        self.assertEqual(len(il_ckpt), 1)
        self.assertEqual(records[SAFETY_ONLY_DEV_K2].termination, "TIMEOUT")
        self.assertEqual(records[IL_ONLY].safety_holds, [])

    def test_planner_calls_recorded_for_cbs_methods_only(self):
        expected = {PURE_CBS_DEV_SINGLE_PLAN: PURE_PLAN, GLOBAL_HYBRID_DEV_FULL: GLOBAL_PLAN,
                    GLOBAL_HYBRID_DEV_H3: GLOBAL_PLAN}
        for method in METHODS:
            with self.subTest(method):
                r = self.run_recorded(method)
                if method not in CBS_METHODS:
                    self.assertEqual(r.planner_calls, [])
                    continue
                (call,) = r.planner_calls
                self.assertEqual(call["paths"], expected[method])   # solver index order == agent ids 0, 1
                self.assertEqual(call["solver_root"], str(self.solver_root.resolve()))
                self.assertEqual(call["timeout_sec"], 30)
                self.assertTrue(call["work_dir"].startswith(r.run_dir))

    def test_existing_run_dir_is_refused(self):
        with patch("src.common.episode_logger.make_run_id", return_value="fixed_run"):
            first = run_method(self.spec(IL_ONLY, max_steps=3))
            (Path(first.run_dir) / "result.json").write_text("SENTINEL\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run_method(self.spec(IL_ONLY, max_steps=3))
        self.assertEqual((Path(first.run_dir) / "result.json").read_text(encoding="utf-8"), "SENTINEL\n")


# -- Refused specs ------------------------------------------------------------------

class TestRefusedSpecs(_Case):

    def test_selective_wfg_not_implemented(self):
        for method in ("SELECTIVE_WFG", "SELECTIVE_HYBRID", "WFG"):
            with self.subTest(method), self.assertRaises(NotImplementedError):
                run_method(self.spec(method))
        self.assertFalse(self.out.exists())

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            run_method(self.spec("PURE_CBS"))
        self.assertFalse(self.out.exists())

    def test_cbs_method_needs_solver_root(self):
        for method in CBS_METHODS:
            with self.subTest(method), self.assertRaises(ValueError):
                run_method(self.spec(method, solver_root=None))
        self.assertFalse(self.out.exists())

    def test_bad_limits(self):
        with self.assertRaises(ValueError):
            run_method(self.spec(IL_ONLY, max_steps=0))
        with self.assertRaises(ValueError):
            run_method(self.spec(PURE_CBS_DEV_SINGLE_PLAN, cbs_timeout_sec=0))
        self.assertFalse(self.out.exists())


# -- Failures: no extra moves; overlap visibility ------------------------------------

def _bad_start_solver(*, current_positions, goals, **kwargs):
    plan = {i: [(p[0], p[1] + 1)] + list(PURE_PLAN[i][1:]) for i, p in enumerate(current_positions)}
    return ReplanResult(status="success", paths=plan, latency_sec=0.05)


def _timeout_solver(**kwargs):
    return ReplanResult(status="timeout", paths=None, latency_sec=30.0, error="atb033 CBS timed out")


class _OverlapAtT1(MAPFStepSimulator):
    """After step 1, agent 1 is forced onto agent 0's cell."""

    def step(self, actions):
        obs, done, info = super().step(actions)
        if info["t"] == 1:
            self._positions[1] = self._positions[0]
        return obs, done, info


@unittest.skipIf(torch is None, "torch is not installed")
class TestFailuresThroughCommonRunner(_Case):

    def test_pure_cbs_refused_plan_makes_no_move(self):
        with patch("src.cbs_runtime_adapter.replan_with_status", side_effect=_bad_start_solver):
            r = run_method(self.spec(PURE_CBS_DEV_SINGLE_PLAN))
        self.assertEqual((r.termination, r.stop_reason, r.steps), ("SAFE_STOP", "CBS_INVALID_PATH", 0))
        self.assertIn("incorrect start position", r.cbs_calls[0]["error"])
        self.assertEqual(r.trajectory, [r.starts])

    def test_global_solver_timeout_stops_at_trigger(self):
        with patch("src.cbs_runtime_adapter.replan_with_status", side_effect=_timeout_solver):
            r = run_method(self.spec(GLOBAL_HYBRID_DEV_FULL))
        self.assertEqual((r.termination, r.stop_reason, r.steps), ("SAFE_STOP", "CBS_TIMEOUT", 5))
        self.assertNotIn("CBS", r.action_sources)
        self.assertEqual(r.cbs_calls[0]["latency_ms"], 30000.0)

    def test_il_only_overlap_is_visible_not_hidden(self):
        with patch("scripts.run_il.MAPFStepSimulator", _OverlapAtT1):
            il = run_method(self.spec(IL_ONLY, max_steps=4))
            safety = run_method(self.spec(SAFETY_ONLY_DEV_K2, max_steps=4))
        # IL-only does not check overlaps at runtime: the run continues, the record says so.
        self.assertFalse(il.overlap_checked_at_runtime)
        self.assertEqual((il.overlap_any_step, il.first_overlap_t, il.integrity_unhandled), (True, 1, True))
        self.assertEqual((il.termination, il.steps), ("TIMEOUT", 4))
        # Safety-only's runner does check: it aborts at the same step.
        self.assertTrue(safety.overlap_checked_at_runtime)
        self.assertEqual((safety.termination, safety.steps, safety.integrity_unhandled),
                         ("INTEGRITY_ABORT", 1, False))


# -- Real atb033 solver ------------------------------------------------------------

@unittest.skipUnless(HAVE_REAL_SOLVER, f"atb033 solver not found at {REAL_SOLVER_ROOT}")
class TestRealSolverThroughCommonRunner(_Case):

    def test_pure_cbs_corridor(self):
        r = run_method(self.spec(PURE_CBS_DEV_SINGLE_PLAN, solver_root=REAL_SOLVER_ROOT))
        self.assertEqual((r.termination, r.all_at_goal, r.overlap_any_step), ("SUCCESS", True, False))
        self.assertEqual(r.cbs_calls[0]["status"], "success")
        self.assertTrue(all(e["match"] for e in r.cbs_exec_trace))
        self.assertTrue(r.environment["solver_git"]["commit"])
        self.assertTrue((Path(r.run_dir) / "cbs" / "cbs_call_01").is_dir())


if __name__ == "__main__":
    unittest.main()
