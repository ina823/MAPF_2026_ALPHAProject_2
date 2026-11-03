"""src/compare/pure_cbs.py: one CBS plan from S_0, executed to the end.

Three kinds of tests, kept apart:
  * MOCK: test-only planner + AcceptAllValidator against the runner's own
    state machine (execution, mismatch, overlap, max_steps, exceptions);
  * REAL ADAPTER: the real CbsRuntimePlanner -> run_cbs_runtime() (agent-id
    conversion + its Path Guard) and the real PathGuardValidator. Solver
    output is injected by patching replan_with_status / run_cbs_runtime, or a
    stand-in cbs.py really runs as a subprocess (timeout / crash / no
    solution). Work dirs live in a temporary directory;
  * REAL SOLVER: atb033 CBS on the Corridor scenario (skipped when absent).
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.cbs_runtime_adapter import CBSRuntimeResult
from src.common.deadlock_monitor import DeadlockMonitor
from src.common.episode_logger import EpisodeLogger
from src.compare.pure_cbs import run_pure_cbs_episode
from src.hybrid.cbs_runtime_bridge import CbsRuntimePlanner, PathGuardValidator
from src.replan_status import ReplanResult
from src.simulator.mapf_step_simulator import MAPFStepSimulator
from scripts.run_il import PROJECT_ROOT, load_scenario
from tests.hybrid_mocks import (
    GOALS as MOCK_GOALS,
    PATHS_TO_GOALS,
    STARTS as MOCK_STARTS,
    RIGHT, UP,
    AcceptAllValidator,
    MockReplanResult,
    RaisingPlanner,
    RecordingPlanner,
    open_grid,
    perturbing_simulator,
    success,
)

REAL_SOLVER_ROOT = Path(os.environ.get(
    "ALPHA2_CBS_SOLVER_ROOT",
    PROJECT_ROOT.parent / "MAPF_ALPHA2_CBS_CHECK/third_party/multi_agent_path_planning/centralized/cbs",
))
HAVE_REAL_SOLVER = (REAL_SOLVER_ROOT / "cbs.py").is_file()
CORRIDOR = PROJECT_ROOT / "scenarios" / "controlled" / "scenario_corridor_bay_n2.json"

# Non-contiguous ids; solver side is indexed in sorted(agent_id) order.
STARTS = {3: (0, 0), 7: (2, 0)}
GOALS = {3: (0, 2), 7: (1, 0)}
INDEX_PATHS = {0: [(0, 0), (0, 1), (0, 2)], 1: [(2, 0), (1, 0), (1, 0)]}
ID_PATHS = {3: INDEX_PATHS[0], 7: INDEX_PATHS[1]}
SWAP_STARTS = {3: (0, 0), 7: (0, 1)}
SWAP_GOALS = {3: (0, 1), 7: (0, 0)}

FAKE_SOLVERS = {
    "sleep": "import time\ntime.sleep(30)\n",
    "crash": "import sys\nsys.exit(3)\n",
    "no_solution": "print(' Solution not found')\n",
}


def counting_simulator(base=MAPFStepSimulator):
    class _Counting(base):
        step_ts = []

        def step(self, actions):
            obs, done, info = super().step(actions)
            type(self).step_ts.append(info["t"])
            return obs, done, info

    _Counting.step_ts = []
    return _Counting


def run(planner, validator, *, starts=MOCK_STARTS, goals=MOCK_GOALS, max_steps=10,
        sim_base=MAPFStepSimulator, grid=None, **kw):
    sim_cls = counting_simulator(sim_base)
    result = run_pure_cbs_episode(grid if grid is not None else open_grid(), starts, goals, max_steps,
                                  planner=planner, path_validator=validator,
                                  simulator_cls=sim_cls, **kw)
    return result, sim_cls


class _Asserts(unittest.TestCase):

    def assert_refused_without_moves(self, result, sim_cls, *, reason, status, starts):
        call = result["cbs_calls"][0]
        self.assertEqual((call["request_timestep"], call["s_0"]), (0, starts))
        self.assertEqual((call["status"], call["accepted"], call["reason"]), (status, False, reason))
        self.assertEqual((result["termination"], result["stop_reason"]), ("SAFE_STOP", reason))
        self.assertEqual(result["safe_stop"], {"reason": reason, "t": 0, "positions": starts})
        self.assertEqual(sim_cls.step_ts, [])          # not a single simulator step
        self.assertEqual((result["steps"], result["actions"], result["cbs_exec_trace"]), (0, [], []))
        self.assertEqual(result["trajectory"], [starts])
        self.assertIsNotNone(call["planner_wall_ms"])


# -- MOCK: runner state machine ------------------------------------------------

class TestMockExecution(_Asserts):

    def test_whole_plan_executed_from_s0(self):
        planner = RecordingPlanner(success(PATHS_TO_GOALS))
        result, sim_cls = run(planner, AcceptAllValidator())
        self.assertEqual(planner.calls[0]["current_positions"], [(0, 0), (2, 0)])  # S_0
        self.assertEqual((result["termination"], result["all_at_goal"], result["steps"]), ("SUCCESS", True, 2))
        self.assertEqual(result["action_sources"], ["CBS", "CBS"])
        self.assertEqual(result["trajectory"][1], {0: (0, 1), 1: (2, 1)})   # Path[1]
        self.assertTrue(all(e["match"] and not e["overlap"] for e in result["cbs_exec_trace"]))
        self.assertEqual(sim_cls.step_ts, [1, 2])
        self.assertEqual(result["transitions"], [{"t": 0, "from": "PLAN", "to": "CBS_EXEC", "reason": ""}])
        self.assertEqual((result["plan_horizon"], result["plan_exceeds_max_steps"]), (2, False))
        self.assertEqual(result["cbs_calls"][0]["latency_ms"], 250.0)
        self.assertIsNone(result["stop_reason"])

    def test_mismatch_at_k2_stops_all_moves(self):
        goals = {0: (0, 3), 1: (2, 3)}
        paths = [[(0, 0), (0, 1), (0, 2), (0, 3)], [(2, 0), (2, 1), (2, 2), (2, 3)]]
        result, sim_cls = run(RecordingPlanner(success(paths)), AcceptAllValidator(), goals=goals,
                              sim_base=perturbing_simulator(inject_t=2, agent_id=0, new_pos=(1, 1)))
        self.assertEqual((result["termination"], result["stop_reason"]), ("SAFE_STOP", "EXEC_MISMATCH"))
        mismatch = result["exec_mismatch"]
        self.assertEqual((mismatch["t"], mismatch["k"]), (2, 2))
        self.assertEqual(mismatch["expected"], {0: (0, 2), 1: (2, 2)})
        self.assertEqual(mismatch["actual"], {0: (1, 1), 1: (2, 2)})
        self.assertEqual(sim_cls.step_ts, [1, 2])        # no step after detection
        self.assertEqual(result["safe_stop"]["t"], 2)

    def test_overlap_aborts(self):
        result, sim_cls = run(RecordingPlanner(success(PATHS_TO_GOALS)), AcceptAllValidator(),
                              sim_base=perturbing_simulator(inject_t=1, agent_id=1, new_pos=(0, 1)))
        self.assertEqual(result["termination"], "INTEGRITY_ABORT")
        self.assertEqual((result["integrity_abort"], result["integrity_abort_timestep"]), (True, 1))
        self.assertTrue(result["cbs_exec_trace"][0]["overlap"])
        self.assertEqual(sim_cls.step_ts, [1])
        self.assertIsNone(result["safe_stop"])

    def test_plan_longer_than_max_steps_ends_timeout(self):
        # max_steps (episode limit) is separate from the solver timeout.
        result, sim_cls = run(RecordingPlanner(success(PATHS_TO_GOALS)), AcceptAllValidator(), max_steps=1)
        self.assertEqual((result["termination"], result["all_at_goal"], result["timed_out"]),
                         ("TIMEOUT", False, True))
        self.assertEqual((result["steps"], result["plan_horizon"], result["plan_exceeds_max_steps"]), (1, 2, True))
        self.assertEqual(sim_cls.step_ts, [1])
        self.assertIsNone(result["stop_reason"])

    def test_planner_exception(self):
        planner = RaisingPlanner()
        result, sim_cls = run(planner, AcceptAllValidator())
        self.assert_refused_without_moves(result, sim_cls, reason="PLANNER_EXCEPTION", status=None,
                                          starts=MOCK_STARTS)
        self.assertIn("mock solver crashed", result["cbs_calls"][0]["error"])
        self.assertIsNone(result["cbs_calls"][0]["latency_ms"])

    def test_already_at_goal(self):
        paths = [[(0, 0)], [(2, 0)]]
        result, sim_cls = run(RecordingPlanner(success(paths)), AcceptAllValidator(), goals=MOCK_STARTS)
        self.assertEqual((result["termination"], result["all_at_goal"], result["steps"]), ("SUCCESS", True, 0))
        self.assertEqual(sim_cls.step_ts, [])

    def test_monitor_and_logger_only_observe(self):
        baseline, _ = run(RecordingPlanner(success(PATHS_TO_GOALS)), AcceptAllValidator())
        with tempfile.TemporaryDirectory() as tmp:
            logger = EpisodeLogger("pure", scenario="synthetic", mode="PURE_CBS", out_dir=tmp)
            observed, _ = run(RecordingPlanner(success(PATHS_TO_GOALS)), AcceptAllValidator(),
                              monitor=DeadlockMonitor(), logger=logger)
        for key in ("trajectory", "actions", "termination", "steps"):
            self.assertEqual(observed[key], baseline[key], key)
        rows = logger.step_rows
        self.assertEqual(len(rows), observed["steps"] * len(MOCK_STARTS))
        self.assertEqual({r["mode"] for r in rows}, {"PURE_CBS"})
        self.assertEqual({r["trigger_reason"] for r in rows}, {"TRIGGER_NONE"})
        self.assertEqual(rows[-1]["success"], True)

    def test_test_only_validator_refused_with_real_planner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "solver"
            root.mkdir()
            (root / "cbs.py").write_text("raise SystemExit(1)\n", encoding="utf-8")
            planner = CbsRuntimePlanner(solver_root=root, work_root=Path(tmp) / "run")
            with self.assertRaises(ValueError):
                run(planner, AcceptAllValidator())
            self.assertEqual(planner.calls, [])


# -- REAL ADAPTER + real Path Guard ---------------------------------------------

class _RealCase(_Asserts):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self._runs = 0

    def tearDown(self):
        self._tmp.cleanup()

    def solver(self, body="raise SystemExit('fake solver must not run')\n", name="solver"):
        root = self.tmp / name
        root.mkdir()
        (root / "cbs.py").write_text(body, encoding="utf-8")
        return root

    def planner(self, root=None, timeout_sec=30):
        self._runs += 1
        return CbsRuntimePlanner(solver_root=root or self.solver(name=f"solver_{self._runs}"),
                                 work_root=self.tmp / f"run_{self._runs}", timeout_sec=timeout_sec)

    def run_injected(self, solver_paths, *, starts=STARTS, goals=GOALS):
        seen = []

        def fake_replan(**kwargs):
            seen.append(kwargs)
            return ReplanResult(status="success", paths=solver_paths, latency_sec=0.05)

        planner = self.planner()
        with patch("src.cbs_runtime_adapter.replan_with_status", side_effect=fake_replan):
            result, sim_cls = run(planner, PathGuardValidator(), starts=starts, goals=goals)
        self.assertEqual(len(seen), 1)
        return result, sim_cls, seen[0], planner

    def run_bypassing_adapter(self, id_paths, *, starts=STARTS, goals=GOALS):
        def fake_runtime(current_positions, goals, grid_map, **kwargs):
            return CBSRuntimeResult(status="success", paths=id_paths, latency_sec=0.05)

        with patch("src.hybrid.cbs_runtime_bridge.run_cbs_runtime", side_effect=fake_runtime):
            return run(self.planner(), PathGuardValidator(), starts=starts, goals=goals)


class TestRealAdapterExecution(_RealCase):

    def test_ids_3_7_executed_with_single_conversion(self):
        result, sim_cls, seen, planner = self.run_injected(INDEX_PATHS)
        self.assertEqual(seen["current_positions"], ((0, 0), (2, 0)))   # S_0, sorted ids
        self.assertEqual(result["actions"][0], {3: RIGHT, 7: UP})       # Path[1]
        self.assertEqual((result["termination"], result["final_positions"]), ("SUCCESS", GOALS))
        self.assertEqual(result["cbs_calls"][0]["agent_order"], (3, 7))
        self.assertTrue((self.tmp / "run_1" / "cbs_call_01").is_dir())   # isolated work dir
        self.assertEqual(planner.calls[0]["paths"], ID_PATHS)


class TestRealAdapterRefusals(_RealCase):

    def test_path0_not_s0(self):
        result, sim_cls, _, _ = self.run_injected({0: [(0, 1), (0, 1), (0, 2)], 1: INDEX_PATHS[1]})
        self.assert_refused_without_moves(result, sim_cls, reason="CBS_INVALID_PATH",
                                          status="invalid_path", starts=STARTS)
        self.assertIn("Agent 3: incorrect start position", result["cbs_calls"][0]["error"])

    def test_agent_ids_missing_extra_swapped(self):
        cases = {
            "missing": ({0: INDEX_PATHS[0]}, "CBS Agent IDs do not match"),
            "extra": ({**INDEX_PATHS, 2: [(1, 3), (1, 3), (1, 3)]}, "CBS Agent IDs do not match"),
            "swapped": ({0: INDEX_PATHS[1], 1: INDEX_PATHS[0]}, "Agent 3: incorrect start position"),
        }
        for name, (paths, error) in cases.items():
            with self.subTest(name):
                result, sim_cls, _, _ = self.run_injected(paths)
                self.assert_refused_without_moves(result, sim_cls, reason="CBS_INVALID_PATH",
                                                  status="invalid_path", starts=STARTS)
                self.assertIn(error, result["cbs_calls"][0]["error"])

    def test_boundary_refuses_bypassed_start_mismatch(self):
        result, sim_cls = self.run_bypassing_adapter({3: [(0, 1), (0, 1), (0, 2)], 7: ID_PATHS[7]})
        self.assert_refused_without_moves(result, sim_cls, reason="BOUNDARY_START_MISMATCH",
                                          status="success", starts=STARTS)

    def test_boundary_path_guard_refuses_bypassed_edge_conflict(self):
        swap = {3: [(0, 0), (0, 1)], 7: [(0, 1), (0, 0)]}
        result, sim_cls = self.run_bypassing_adapter(swap, starts=SWAP_STARTS, goals=SWAP_GOALS)
        self.assert_refused_without_moves(
            result, sim_cls, reason="VALIDATOR_REJECT:InvalidPathError: Edge conflict at t=0->1: Agent 3 and 7",
            status="success", starts=SWAP_STARTS)


class TestRealSubprocessFailures(_RealCase):

    def _check(self, kind, status, reason, error_part, timeout_sec=30):
        planner = self.planner(self.solver(FAKE_SOLVERS[kind], name=f"fake_{kind}"), timeout_sec=timeout_sec)
        result, sim_cls = run(planner, PathGuardValidator(), starts=STARTS, goals=GOALS)
        self.assert_refused_without_moves(result, sim_cls, reason=reason, status=status, starts=STARTS)
        call = result["cbs_calls"][0]
        self.assertIn(error_part, call["error"])
        self.assertIsNotNone(call["latency_ms"])
        return call

    def test_solver_timeout(self):
        # Solver timeout (1 s here) is a different limit from max_steps.
        self._check("sleep", "timeout", "CBS_TIMEOUT", "timed out", timeout_sec=1)

    def test_solver_error(self):
        self._check("crash", "solver_error", "CBS_SOLVER_ERROR", "returncode: 3")

    def test_no_solution(self):
        call = self._check("no_solution", "no_solution", "CBS_NO_SOLUTION", "")
        self.assertTrue(call["error"])


# -- REAL SOLVER: atb033 on Corridor ----------------------------------------------

@unittest.skipUnless(HAVE_REAL_SOLVER, f"atb033 solver not found at {REAL_SOLVER_ROOT}")
class TestCorridorRealSolver(_RealCase):

    def test_corridor_from_s0(self):
        grid, starts, goals = load_scenario(CORRIDOR)
        planner = self.planner(REAL_SOLVER_ROOT)
        result, sim_cls = run(planner, PathGuardValidator(), starts=starts, goals=goals, max_steps=160,
                              grid=grid, monitor=DeadlockMonitor())
        call = result["cbs_calls"][0]
        paths = planner.calls[0]["paths"]
        self.assertEqual((call["status"], call["accepted"]), ("success", True))
        self.assertEqual({aid: p[0] for aid, p in paths.items()}, starts)          # Path[0] == S_0
        self.assertEqual(result["trajectory"][1], {aid: p[1] for aid, p in paths.items()})  # Path[1]
        self.assertTrue(all(e["match"] and not e["overlap"] for e in result["cbs_exec_trace"]))
        self.assertEqual((result["termination"], result["all_at_goal"]), ("SUCCESS", True))
        self.assertEqual(result["steps"], result["plan_horizon"])
        self.assertEqual(sim_cls.step_ts, list(range(1, result["steps"] + 1)))


if __name__ == "__main__":
    unittest.main()
