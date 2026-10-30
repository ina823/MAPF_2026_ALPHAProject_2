"""Corridor Global Hybrid E2E failure paths: real IL checkpoint, real
MAPFStepSimulator, real DeadlockMonitor, real run_cbs_runtime() + Path Guard.

  * timeout / solver_error / no_solution: a stand-in cbs.py is really run as
    a subprocess by src/cbs_adapter.py (sleeps / exits 3 / prints
    "Solution not found");
  * invalid_path: replan_with_status is patched to return a jump, so the CBS
    adapter's own Path Guard refuses it;
  * expected/actual mismatch and duplicate Trigger need a valid CBS plan and
    use the real atb033 solver (ALPHA2_CBS_SOLVER_ROOT, else the sibling
    MAPF_ALPHA2_CBS_CHECK clone); they are skipped when it is not found.

Every failure must end in SAFE_STOP with zero sim.step() calls after the
Trigger step. All solver files go to a temporary directory.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.run_corridor_hybrid_e2e import (
    DEFAULT_SCENARIO,
    DevRecordSafeStop,
    evaluate,
    run_corridor_e2e,
)
from scripts.run_il import DEFAULT_CKPT, PROJECT_ROOT, load_scenario
from src.common.deadlock_monitor import DeadlockMonitor
from src.hybrid.cbs_runtime_bridge import CbsRuntimePlanner, PathGuardValidator
from src.hybrid.global_hybrid import GlobalHybridConfig, run_global_hybrid_episode
from src.replan_status import ReplanResult
from src.simulator.mapf_step_simulator import MAPFStepSimulator
from tests.hybrid_mocks import AllowAllRecalls, AlwaysResumeIL

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

REAL_SOLVER_ROOT = Path(os.environ.get(
    "ALPHA2_CBS_SOLVER_ROOT",
    PROJECT_ROOT.parent / "MAPF_ALPHA2_CBS_CHECK/third_party/multi_agent_path_planning/centralized/cbs",
))
HAVE_REAL_SOLVER = (REAL_SOLVER_ROOT / "cbs.py").is_file()

FAKE_SOLVERS = {
    "sleep": "import time\ntime.sleep(30)\n",
    "crash": "import sys\nsys.exit(3)\n",
    "no_solution": "print(' Solution not found')\n",
}


def counting_simulator(base=MAPFStepSimulator):
    class _CountingSimulator(base):
        step_ts = []

        def step(self, actions):
            obs, done, info = super().step(actions)
            type(self).step_ts.append(info["t"])
            return obs, done, info

    _CountingSimulator.step_ts = []
    return _CountingSimulator


@unittest.skipIf(torch is None, "torch is not installed")
class _CorridorCase(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def fake_solver(self, kind):
        root = self.tmp / f"solver_{kind}"
        root.mkdir()
        (root / "cbs.py").write_text(FAKE_SOLVERS[kind], encoding="utf-8")
        return root

    def planner(self, solver_root, timeout_sec=30):
        self._runs = getattr(self, "_runs", 0) + 1
        return CbsRuntimePlanner(solver_root=solver_root, work_root=self.tmp / f"run_{self._runs}",
                                 timeout_sec=timeout_sec)


class TestCorridorCbsFailureNoMoves(_CorridorCase):

    def _assert_safe_stop_no_moves(self, result, sim_cls, status, reason):
        call = result["cbs_calls"][0]
        trigger_t = call["trigger_timestep"]
        self.assertEqual(len(result["cbs_calls"]), 1)
        self.assertEqual(call["status"], status)
        self.assertFalse(call["accepted"])
        self.assertEqual(result["termination"], "SAFE_STOP")
        self.assertEqual(result["stop_reason"], reason)
        self.assertTrue(call["error"], "failure reason must be recorded")
        # Zero simulator steps after the Trigger step.
        self.assertEqual(sim_cls.step_ts[-1], trigger_t)
        self.assertEqual(result["steps"], trigger_t)
        self.assertNotIn("CBS", result["action_sources"])
        self.assertEqual(result["final_positions"], call["s_t"])
        self.assertEqual(result["safe_stop"]["reason"], reason)
        self.assertEqual(result["safe_stop"]["t"], trigger_t)
        # S_t is the real simulator state at the Trigger.
        self.assertEqual(call["s_t"], result["trajectory"][trigger_t])
        return call

    def test_real_subprocess_timeout(self):
        sim_cls = counting_simulator()
        planner = self.planner(self.fake_solver("sleep"), timeout_sec=1)
        result = run_corridor_e2e(planner, simulator_cls=sim_cls)
        call = self._assert_safe_stop_no_moves(result, sim_cls, "timeout", "CBS_TIMEOUT")
        self.assertIn("timed out", call["error"])

    def test_real_subprocess_solver_error(self):
        sim_cls = counting_simulator()
        result = run_corridor_e2e(self.planner(self.fake_solver("crash")), simulator_cls=sim_cls)
        call = self._assert_safe_stop_no_moves(result, sim_cls, "solver_error", "CBS_SOLVER_ERROR")
        self.assertIn("returncode: 3", call["error"])

    def test_real_subprocess_no_solution(self):
        sim_cls = counting_simulator()
        result = run_corridor_e2e(self.planner(self.fake_solver("no_solution")), simulator_cls=sim_cls)
        self._assert_safe_stop_no_moves(result, sim_cls, "no_solution", "CBS_NO_SOLUTION")

    def test_invalid_path_refused_by_adapter_path_guard(self):
        def jump_paths(*, current_positions, goals, **kwargs):
            # One-step teleport S -> G for every agent: illegal movement.
            return ReplanResult(
                status="success",
                paths={i: [tuple(s), tuple(g)] for i, (s, g) in enumerate(zip(current_positions, goals))},
                latency_sec=0.01,
            )

        sim_cls = counting_simulator()
        with patch("src.cbs_runtime_adapter.replan_with_status", side_effect=jump_paths):
            result = run_corridor_e2e(self.planner(self.fake_solver("crash")), simulator_cls=sim_cls)
        call = self._assert_safe_stop_no_moves(result, sim_cls, "invalid_path", "CBS_INVALID_PATH")
        self.assertIn("illegal movement", call["error"])

    def test_s_t_is_the_live_state_not_the_scenario_start(self):
        sim_cls = counting_simulator()
        result = run_corridor_e2e(self.planner(self.fake_solver("crash")), simulator_cls=sim_cls)
        _, starts, _ = load_scenario(DEFAULT_SCENARIO)
        self.assertNotEqual(result["cbs_calls"][0]["s_t"], starts)


@unittest.skipUnless(HAVE_REAL_SOLVER, f"atb033 solver not found at {REAL_SOLVER_ROOT}")
class TestCorridorWithRealSolver(_CorridorCase):

    def _trigger_t(self):
        result = run_corridor_e2e(self.planner(self.fake_solver("crash")))
        return result["cbs_calls"][0]["trigger_timestep"]

    def test_exec_mismatch_stops_next_move(self):
        trigger_t = self._trigger_t()
        grid, _, _ = load_scenario(DEFAULT_SCENARIO)

        class _Perturbing(MAPFStepSimulator):
            def step(self, actions):
                obs, done, info = super().step(actions)
                if info["t"] == trigger_t + 1:
                    # Teleport agent 0 to a free, unoccupied cell: never Path[1].
                    taken = set(self._positions.values())
                    free = next((int(r), int(c)) for r, c in np.argwhere(grid == 0)
                                if (int(r), int(c)) not in taken)
                    self._positions[0] = free
                return obs, done, info

        sim_cls = counting_simulator(_Perturbing)
        planner = self.planner(REAL_SOLVER_ROOT)
        result = run_corridor_e2e(planner, simulator_cls=sim_cls)
        self.assertEqual(result["cbs_calls"][0]["status"], "success")
        self.assertEqual(result["termination"], "SAFE_STOP")
        self.assertEqual(result["stop_reason"], "EXEC_MISMATCH")
        self.assertEqual(result["action_sources"].count("CBS"), 1)
        self.assertEqual(sim_cls.step_ts[-1], trigger_t + 1)   # no move after the mismatch
        mismatch = result["exec_mismatch"]
        self.assertEqual(mismatch["t"], trigger_t + 1)
        self.assertNotEqual(mismatch["expected"], mismatch["actual"])
        # The perturbation hits the first CBS step, so the verdict names Path[1].
        self.assertEqual(evaluate(result, planner.calls)["first_failed_check"],
                         "first_cbs_step_reaches_path1")

    def test_stale_trigger_never_recalls_cbs(self):
        # Resume IL after every single CBS step with an allow-all recall policy:
        # the Monitor's stale TRIGGER_FULL_REPEAT must not cause a second call.
        from src.il.policy import NavHintPolicy

        grid, starts, goals = load_scenario(DEFAULT_SCENARIO)
        planner = self.planner(REAL_SOLVER_ROOT)
        recall = AllowAllRecalls()
        result = run_global_hybrid_episode(
            NavHintPolicy(DEFAULT_CKPT), grid, starts, goals, 40,
            planner=planner,
            path_validator=PathGuardValidator(),
            config=GlobalHybridConfig(h_exec=1, resume_policy=AlwaysResumeIL(),
                                      recall_policy=recall, safe_stop_handler=DevRecordSafeStop()),
            monitor=DeadlockMonitor(),
        )
        self.assertEqual(len(planner.calls), 1)
        self.assertEqual(len(result["cbs_calls"]), 1)
        self.assertEqual(len(recall.asked), 1)
        self.assertEqual(result["action_sources"].count("CBS"), 1)
        self.assertIn(("CBS_EXEC", "IL_RESUME"), [(t["from"], t["to"]) for t in result["transitions"]])


if __name__ == "__main__":
    unittest.main()
