"""Failure injection for the Global Hybrid Runner v0 safety contract
(docs/global_hybrid_runner_v0.md, invariants I-1..I-4). Only cases the
existing suites do not already cover.

Two kinds of tests, kept apart:

  * REAL (F1-F7): the real src/cbs_runtime_adapter.run_cbs_runtime() (agent-id
    conversion + its own Path Guard) and the real PathGuardValidator at the
    boundary. Only the solver output is injected -- by patching
    replan_with_status (below the adapter) or run_cbs_runtime (adapter
    bypassed, so the boundary is the last line of defence). No solver runs;
    CbsRuntimePlanner work dirs live in a temporary directory.
  * MOCK (F8-F11): deterministic Mock planners / simulators from
    tests/hybrid_mocks.py against the runner state machine. F11 uses a STUB
    monitor: the real DeadlockMonitor never reports TRIGGER_FULL_REPEAT
    together with INTEGRITY_ISSUE without an overlap (the overlap aborts the
    episode first), so F11 checks the I-2 branch only.

Every failure must end with the exact stop_reason/status/error recorded and
no simulator step or IL act() after the point of failure. The policy stubs
(AlwaysResumeIL, AllowAllRecalls, RecordSafeStop) are temporary test values,
not agreed rules.
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.cbs_runtime_adapter import CBSRuntimeResult
from src.common.deadlock_monitor import DeadlockMonitor
from src.hybrid.cbs_runtime_bridge import CbsRuntimePlanner, PathGuardValidator
from src.hybrid.global_hybrid import GlobalHybridConfig, run_global_hybrid_episode
from src.replan_status import ReplanResult
from src.simulator.mapf_step_simulator import MAPFStepSimulator
from tests.hybrid_mocks import (
    PATHS_TO_GOALS,
    STARTS as MOCK_STARTS,
    AcceptAllValidator,
    AllowAllRecalls,
    AlwaysResumeIL,
    MockReplanResult,
    RecordSafeStop,
    RecordingPlanner,
    ScriptedPolicy,
    open_grid,
    perturbing_simulator,
    success,
)

# Non-contiguous ids: an index-keyed and an id-keyed dict cannot look alike.
STARTS = {3: (0, 0), 7: (2, 0)}
GOALS = {3: (0, 2), 7: (1, 0)}
INDEX_PATHS = {  # solver side, sorted(agent_id) order: 0 -> 3, 1 -> 7
    0: [(0, 0), (0, 1), (0, 2)],
    1: [(2, 0), (1, 0), (1, 0)],
}
ID_PATHS = {3: INDEX_PATHS[0], 7: INDEX_PATHS[1]}

# Swap neighbours: valid endpoints and adjacent moves, but an edge conflict.
SWAP_STARTS = {3: (0, 0), 7: (0, 1)}
SWAP_GOALS = {3: (0, 1), 7: (0, 0)}
SWAP_INDEX_PATHS = {0: [(0, 0), (0, 1)], 1: [(0, 1), (0, 0)]}
SWAP_ID_PATHS = {3: SWAP_INDEX_PATHS[0], 7: SWAP_INDEX_PATHS[1]}

TRIGGER_T = 1  # all-WAIT IL: S_1 == S_0 -> TRIGGER_FULL_REPEAT at t=1


# -- Recorders ----------------------------------------------------------------

def counting_simulator(base=MAPFStepSimulator):
    """Fresh subclass recording the t of every step() call."""

    class _Counting(base):
        step_ts = []

        def step(self, actions):
            obs, done, info = super().step(actions)
            type(self).step_ts.append(info["t"])
            return obs, done, info

    _Counting.step_ts = []
    return _Counting


class CountingPolicy(ScriptedPolicy):
    def __init__(self, actions_per_agent=None):
        super().__init__(actions_per_agent)
        self.act_calls = 0

    def act(self, obs, positions):
        self.act_calls += 1
        return super().act(obs, positions)


class CountingPathGuard(PathGuardValidator):
    """The real Path Guard validator, counting how often the boundary used it."""

    def __init__(self):
        self.calls = 0

    def __call__(self, grid, starts, goals, paths):
        self.calls += 1
        return super().__call__(grid, starts, goals, paths)


class RaisingValidator:
    test_only = True

    def __call__(self, grid, starts, goals, paths):
        raise RuntimeError("mock validator crashed")


def run(planner, validator, *, starts, goals, h_exec=10, max_steps=10, policy=None,
        sim_base=MAPFStepSimulator, monitor=None, recall=None):
    policy = policy or CountingPolicy()
    sim_cls = counting_simulator(sim_base)
    config = GlobalHybridConfig(
        h_exec=h_exec,
        resume_policy=AlwaysResumeIL(),
        recall_policy=recall or AllowAllRecalls(),
        safe_stop_handler=RecordSafeStop(),
    )
    result = run_global_hybrid_episode(
        policy, open_grid(), starts, goals, max_steps,
        planner=planner,
        path_validator=validator,
        config=config,
        monitor=monitor or DeadlockMonitor(),
        simulator_cls=sim_cls,
    )
    return result, sim_cls, policy


class _Asserts(unittest.TestCase):

    def assert_refused_before_any_cbs_move(self, run_out, *, stop_reason, status, starts,
                                           error_contains=None, error_is_none=False):
        result, sim_cls, policy = run_out
        call = result["cbs_calls"][0]
        self.assertEqual(len(result["cbs_calls"]), 1)
        self.assertEqual(call["trigger_timestep"], TRIGGER_T)
        self.assertEqual(call["s_t"], starts)
        self.assertEqual(call["status"], status)
        self.assertFalse(call["accepted"])
        self.assertEqual(call["reason"], stop_reason)
        self.assertEqual(result["termination"], "SAFE_STOP")
        self.assertEqual(result["stop_reason"], stop_reason)
        self.assertEqual(result["safe_stop"], {"reason": stop_reason, "t": TRIGGER_T})
        if error_contains is not None:
            self.assertIn(error_contains, call["error"])
        if error_is_none:
            self.assertIsNone(call["error"])
        # Zero simulator steps and zero IL act() calls after the Trigger step.
        self.assertEqual(sim_cls.step_ts, [TRIGGER_T])
        self.assertEqual(policy.act_calls, TRIGGER_T)
        self.assertNotIn("CBS", result["action_sources"])
        self.assertEqual(result["cbs_exec_trace"], [])
        self.assertEqual(result["final_positions"], starts)


class _RealCase(_Asserts):
    """Real CbsRuntimePlanner whose solver must never run (replan is patched)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.solver_root = self.tmp / "solver"
        self.solver_root.mkdir()
        (self.solver_root / "cbs.py").write_text(
            "raise SystemExit('fake solver must not run')\n", encoding="utf-8")
        self._runs = 0

    def tearDown(self):
        self._tmp.cleanup()

    def planner(self):
        self._runs += 1
        return CbsRuntimePlanner(solver_root=self.solver_root, work_root=self.tmp / f"run_{self._runs}")

    def run_real_adapter(self, solver_paths, *, starts=STARTS, goals=GOALS):
        """Inject solver output below run_cbs_runtime(); adapter + Path Guard are real."""
        seen = []

        def fake_replan(**kwargs):
            seen.append(kwargs)
            return ReplanResult(status="success", paths=solver_paths, latency_sec=0.05)

        planner = self.planner()
        with patch("src.cbs_runtime_adapter.replan_with_status", side_effect=fake_replan):
            out = run(planner, PathGuardValidator(), starts=starts, goals=goals)
        self.assertEqual(len(seen), 1)
        self.assertEqual(len(planner.calls), 1)
        return out, seen[0]

    def run_bypassing_adapter(self, id_paths, *, starts=STARTS, goals=GOALS):
        """Inject an already-converted result: only the boundary guards it."""
        guard = CountingPathGuard()

        def fake_runtime(current_positions, goals, grid_map, **kwargs):
            return CBSRuntimeResult(status="success", paths=id_paths, latency_sec=0.05)

        with patch("src.hybrid.cbs_runtime_bridge.run_cbs_runtime", side_effect=fake_runtime):
            out = run(self.planner(), guard, starts=starts, goals=goals)
        return out, guard


# -- REAL: adapter + Path Guard ------------------------------------------------

class TestRealAdapterRefusesBadSolverOutput(_RealCase):

    def test_f1_path0_not_s_t(self):
        bad = {0: [(0, 1), (0, 1), (0, 2)], 1: INDEX_PATHS[1]}
        out, seen = self.run_real_adapter(bad)
        self.assertEqual(seen["current_positions"], ((0, 0), (2, 0)))  # solver got the real S_t
        self.assert_refused_before_any_cbs_move(
            out, stop_reason="CBS_INVALID_PATH", status="invalid_path", starts=STARTS,
            error_contains="Agent 3: incorrect start position")

    def test_f2_missing_agent_index(self):
        out, _ = self.run_real_adapter({0: INDEX_PATHS[0]})
        self.assert_refused_before_any_cbs_move(
            out, stop_reason="CBS_INVALID_PATH", status="invalid_path", starts=STARTS,
            error_contains="CBS Agent IDs do not match")

    def test_f3_extra_agent_index(self):
        out, _ = self.run_real_adapter({**INDEX_PATHS, 2: [(1, 3), (1, 3), (1, 3)]})
        self.assert_refused_before_any_cbs_move(
            out, stop_reason="CBS_INVALID_PATH", status="invalid_path", starts=STARTS,
            error_contains="CBS Agent IDs do not match")

    def test_f4_swapped_index_order(self):
        # Index 0 now carries agent 7's path: mapped onto id 3 it starts at (2, 0).
        out, _ = self.run_real_adapter({0: INDEX_PATHS[1], 1: INDEX_PATHS[0]})
        self.assert_refused_before_any_cbs_move(
            out, stop_reason="CBS_INVALID_PATH", status="invalid_path", starts=STARTS,
            error_contains="Agent 3: incorrect start position")

    def test_f7a_edge_conflict_refused_by_adapter_path_guard(self):
        out, _ = self.run_real_adapter(SWAP_INDEX_PATHS, starts=SWAP_STARTS, goals=SWAP_GOALS)
        self.assert_refused_before_any_cbs_move(
            out, stop_reason="CBS_INVALID_PATH", status="invalid_path", starts=SWAP_STARTS,
            error_contains="Edge conflict at t=0->1: Agent 3 and 7")


# -- REAL: boundary Path Guard with the adapter bypassed ------------------------

class TestBoundaryRefusesBadIdKeyedPaths(_RealCase):

    def test_f5_missing_or_extra_agent_id(self):
        cases = {
            "missing_7": {3: ID_PATHS[3]},
            "extra_9": {**ID_PATHS, 9: [(1, 3), (1, 3), (1, 3)]},
        }
        for name, paths in cases.items():
            with self.subTest(name):
                out, guard = self.run_bypassing_adapter(paths)
                self.assert_refused_before_any_cbs_move(
                    out, stop_reason="BOUNDARY_PATH_COUNT", status="success", starts=STARTS,
                    error_is_none=True)
                self.assertEqual(guard.calls, 0)  # refused before the validator

    def test_f6_path0_not_s_t(self):
        out, guard = self.run_bypassing_adapter({3: [(0, 1), (0, 1), (0, 2)], 7: ID_PATHS[7]})
        self.assert_refused_before_any_cbs_move(
            out, stop_reason="BOUNDARY_START_MISMATCH", status="success", starts=STARTS,
            error_is_none=True)
        self.assertEqual(guard.calls, 0)

    def test_f7b_edge_conflict_refused_by_boundary_path_guard(self):
        out, guard = self.run_bypassing_adapter(SWAP_ID_PATHS, starts=SWAP_STARTS, goals=SWAP_GOALS)
        self.assert_refused_before_any_cbs_move(
            out, stop_reason="VALIDATOR_REJECT:InvalidPathError: Edge conflict at t=0->1: Agent 3 and 7",
            status="success", starts=SWAP_STARTS, error_is_none=True)
        self.assertEqual(guard.calls, 1)


# -- MOCK: runner state machine ------------------------------------------------

class TestMockBoundaryReasonsWithoutMoves(_Asserts):

    def test_f8_remaining_refusal_reasons(self):
        cases = [
            ("CBS_UNKNOWN_STATUS", MockReplanResult(status="weird", paths=PATHS_TO_GOALS),
             AcceptAllValidator(), "weird"),
            ("BOUNDARY_EMPTY_PATH", success([[], PATHS_TO_GOALS[1]]), AcceptAllValidator(), "success"),
            ("BOUNDARY_MALFORMED_PATH", success([[(0, 0), (0,)], PATHS_TO_GOALS[1]]),
             AcceptAllValidator(), "success"),
            ("VALIDATOR_ERROR", success(PATHS_TO_GOALS), RaisingValidator(), "success"),
        ]
        for reason, planner_result, validator, status in cases:
            with self.subTest(reason):
                planner = RecordingPlanner(planner_result)
                out = run(planner, validator, starts=MOCK_STARTS, goals={0: (0, 2), 1: (2, 2)})
                self.assertEqual(len(planner.calls), 1)
                self.assert_refused_before_any_cbs_move(
                    out, stop_reason=reason, status=status, starts=MOCK_STARTS)


class TestMockMismatchMidPlan(unittest.TestCase):

    def test_f9_mismatch_at_k2_stops_all_moves(self):
        goals = {0: (0, 3), 1: (2, 3)}
        paths = [[(0, 0), (0, 1), (0, 2), (0, 3)], [(2, 0), (2, 1), (2, 2), (2, 3)]]
        # CBS step k=1 -> t=2, k=2 -> t=3: push agent 0 to free (1, 2) instead of (0, 2).
        planner = RecordingPlanner(success(paths))
        result, sim_cls, policy = run(
            planner, AcceptAllValidator(), starts=MOCK_STARTS, goals=goals, h_exec=3,
            sim_base=perturbing_simulator(inject_t=3, agent_id=0, new_pos=(1, 2)))
        self.assertEqual(result["termination"], "SAFE_STOP")
        self.assertEqual(result["stop_reason"], "EXEC_MISMATCH")
        self.assertEqual(result["safe_stop"], {"reason": "EXEC_MISMATCH", "t": 3})
        mismatch = result["exec_mismatch"]
        self.assertEqual((mismatch["t"], mismatch["k"]), (3, 2))
        self.assertEqual(mismatch["expected"], {0: (0, 2), 1: (2, 2)})
        self.assertEqual(mismatch["actual"], {0: (1, 2), 1: (2, 2)})
        self.assertEqual([e["match"] for e in result["cbs_exec_trace"]], [True, False])
        # After detection at t=3: no simulator step, no IL act(), no CBS step k=3.
        self.assertEqual(sim_cls.step_ts, [1, 2, 3])
        self.assertEqual(policy.act_calls, 1)
        self.assertEqual(result["action_sources"], ["IL", "CBS", "CBS"])
        self.assertNotIn("IL_RESUME", [tr["to"] for tr in result["transitions"]])
        self.assertEqual(len(planner.calls), 1)


class TestMockIlPhaseOverlap(unittest.TestCase):

    def test_f10_overlap_during_il_aborts_without_cbs(self):
        planner = RecordingPlanner(success(PATHS_TO_GOALS))
        result, sim_cls, policy = run(
            planner, AcceptAllValidator(), starts=MOCK_STARTS, goals={0: (0, 2), 1: (2, 2)},
            sim_base=perturbing_simulator(inject_t=1, agent_id=1, new_pos=(0, 0)))
        self.assertEqual(result["termination"], "INTEGRITY_ABORT")
        self.assertTrue(result["integrity_abort"])
        self.assertEqual(result["integrity_abort_timestep"], 1)
        self.assertEqual(planner.calls, [])
        self.assertEqual(result["cbs_calls"], [])
        self.assertEqual(result["transitions"], [])
        self.assertEqual(sim_cls.step_ts, [1])
        self.assertEqual(policy.act_calls, 1)
        self.assertIsNone(result["safe_stop"])


class StubMonitor:
    """STUB, not the real DeadlockMonitor: reports a fresh TRIGGER_FULL_REPEAT
    on every step together with a fixed overlap_integrity_status."""

    def __init__(self, integrity_status):
        self.integrity_status = integrity_status

    def reset(self, positions, goals):
        pass

    def observe_step(self, timestep, positions, conflict_types, **kwargs):
        return SimpleNamespace(
            trigger_reason_code="TRIGGER_FULL_REPEAT",
            trigger_timestep=timestep,
            overlap_integrity_status=self.integrity_status,
        )


class TestStubMonitorIntegrityGate(unittest.TestCase):
    """I-2 branch only (stub monitor) -- not reproduced with the real Monitor."""

    def _run(self, integrity_status):
        planner = RecordingPlanner(success(PATHS_TO_GOALS))
        recall = AllowAllRecalls()
        result, sim_cls, policy = run(
            planner, AcceptAllValidator(), starts=MOCK_STARTS, goals={0: (0, 2), 1: (2, 2)},
            max_steps=4, monitor=StubMonitor(integrity_status), recall=recall)
        return result, planner, recall, sim_cls

    def test_f11_integrity_issue_blocks_cbs_request(self):
        result, planner, recall, sim_cls = self._run("INTEGRITY_ISSUE")
        self.assertEqual(planner.calls, [])
        self.assertEqual(result["cbs_calls"], [])
        self.assertEqual(recall.asked, [])       # gate checked before the recall policy
        self.assertEqual(result["transitions"], [])
        self.assertEqual(result["action_sources"], ["IL"] * 4)
        self.assertEqual(sim_cls.step_ts, [1, 2, 3, 4])
        self.assertEqual(result["termination"], "TIMEOUT")

    def test_f11_control_clean_status_does_request_cbs(self):
        # Same stub with CLEAN: the Trigger path is live, so the test above is not vacuous.
        result, planner, recall, _ = self._run("CLEAN")
        self.assertEqual(len(planner.calls), 1)
        self.assertEqual(recall.asked, [(0, 1)])
        self.assertEqual(result["cbs_calls"][0]["trigger_timestep"], 1)


if __name__ == "__main__":
    unittest.main()
