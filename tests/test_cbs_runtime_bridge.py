"""Agent-id contract between the real CBS runtime adapter and the Global
Hybrid Runner (src/hybrid/cbs_runtime_bridge.py).

The only index <-> agent_id conversion lives inside the CBS branch's
src/cbs_runtime_adapter.py (build_cbs_request / enumerate(agent_ids)). These
tests run that real adapter and the real src/path_guard.py and patch only the
layer below it (replan_with_status / CBSAdapter.plan), so no solver runs.
Non-contiguous ids {3, 7} make a second conversion visible: with ids {0, 1}
an index-keyed and an id-keyed dict look the same.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.common.deadlock_monitor import DeadlockMonitor
from src.cbs_runtime_adapter import CBSRuntimeResult
from src.hybrid.cbs_runtime_bridge import CbsRuntimePlanner, PathGuardValidator
from src.hybrid.global_hybrid import GlobalHybridConfig, run_global_hybrid_episode
from src.hybrid.replan_boundary import accept_replan_result, check_validator_pairing
from src.replan_status import ReplanResult
from tests.hybrid_mocks import (
    RIGHT, UP,
    AcceptAllValidator,
    AlwaysResumeIL,
    AllowAllRecalls,
    MockReplanResult,
    RecordSafeStop,
    ScriptedPolicy,
    open_grid,
    success,
)

STARTS = {3: (0, 0), 7: (2, 0)}
GOALS = {3: (0, 2), 7: (1, 0)}
# CBS-side paths, indexed in sorted(agent_id) order: index 0 -> 3, index 1 -> 7.
INDEX_PATHS = {
    0: [(0, 0), (0, 1), (0, 2)],
    1: [(2, 0), (1, 0), (1, 0)],
}
ID_PATHS = {3: INDEX_PATHS[0], 7: INDEX_PATHS[1]}


def _fake_solver_root(tmp):
    root = Path(tmp) / "solver"
    root.mkdir()
    (root / "cbs.py").write_text("raise SystemExit('fake solver must not run')\n", encoding="utf-8")
    return root


def _run(planner, *, starts=STARTS, goals=GOALS, validator=None, max_steps=10):
    config = GlobalHybridConfig(
        h_exec=10,
        resume_policy=AlwaysResumeIL(),
        recall_policy=AllowAllRecalls(),
        safe_stop_handler=RecordSafeStop(),
    )
    return run_global_hybrid_episode(
        ScriptedPolicy(), open_grid(), starts, goals, max_steps,
        planner=planner,
        path_validator=validator or PathGuardValidator(),
        config=config,
        monitor=DeadlockMonitor(),
    )


class _TmpCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.solver_root = _fake_solver_root(self.tmp)
        self.work_root = self.tmp / "run"

    def tearDown(self):
        self._tmp.cleanup()

    def planner(self, **kw):
        return CbsRuntimePlanner(solver_root=self.solver_root, work_root=self.work_root, **kw)


class TestNonContiguousAgentIds(_TmpCase):

    def _run_with_fake_replan(self, starts):
        calls = []

        def fake_replan(**kwargs):
            calls.append(kwargs)
            return ReplanResult(status="success", paths=INDEX_PATHS, latency_sec=0.05)

        with patch("src.cbs_runtime_adapter.replan_with_status", side_effect=fake_replan):
            result = _run(self.planner(), starts=starts)
        return result, calls

    def test_ids_3_7_reach_goals_with_single_conversion(self):
        result, calls = self._run_with_fake_replan(STARTS)
        self.assertEqual(len(calls), 1)
        # The adapter (not the runner) sorts ids and builds the solver lists.
        self.assertEqual(calls[0]["current_positions"], ((0, 0), (2, 0)))
        self.assertEqual(calls[0]["goals"], ((0, 2), (1, 0)))
        self.assertEqual(result["actions"][1], {3: RIGHT, 7: UP})   # Path[1]
        self.assertEqual(result["termination"], "SUCCESS")
        self.assertEqual(result["final_positions"], GOALS)
        self.assertTrue(result["cbs_calls"][0]["accepted"])

    def test_dict_insertion_order_does_not_matter(self):
        result, calls = self._run_with_fake_replan({7: (2, 0), 3: (0, 0)})
        self.assertEqual(calls[0]["current_positions"], ((0, 0), (2, 0)))
        self.assertEqual(result["actions"][1], {3: RIGHT, 7: UP})
        self.assertEqual(result["termination"], "SUCCESS")

    def test_runner_passes_agent_id_dicts_to_planner(self):
        seen = []

        def fake_runtime(current_positions, goals, grid_map, **kwargs):
            seen.append((dict(current_positions), dict(goals)))
            return CBSRuntimeResult(status="timeout", paths=None, latency_sec=1.0, error="t/o")

        with patch("src.hybrid.cbs_runtime_bridge.run_cbs_runtime", side_effect=fake_runtime):
            result = _run(self.planner())
        self.assertEqual(seen, [(STARTS, GOALS)])
        self.assertEqual(result["cbs_calls"][0]["error"], "t/o")


class TestBoundaryKeying(unittest.TestCase):

    def _accept(self, result, keying):
        return accept_replan_result(
            result, agent_order=(3, 7), current_positions=STARTS, goals=GOALS,
            grid=open_grid(), path_validator=PathGuardValidator(), paths_keying=keying,
        )

    def test_agent_id_keyed_paths_accepted(self):
        outcome = self._accept(success(ID_PATHS), "agent_id")
        self.assertTrue(outcome.accepted, outcome.reason)
        self.assertEqual(outcome.plan.expected[0], {3: (0, 1), 7: (1, 0)})

    def test_id_keyed_paths_in_index_mode_refused(self):
        # An id-keyed result re-mapped as indexes = double conversion.
        self.assertEqual(self._accept(success(ID_PATHS), "index").reason, "BOUNDARY_PATH_COUNT")

    def test_index_keyed_paths_in_agent_id_mode_refused(self):
        self.assertEqual(self._accept(success(INDEX_PATHS), "agent_id").reason, "BOUNDARY_PATH_COUNT")

    def test_agent_id_mode_requires_mapping(self):
        paths = [INDEX_PATHS[0], INDEX_PATHS[1]]
        self.assertEqual(self._accept(success(paths), "agent_id").reason, "BOUNDARY_PATHS_TYPE")

    def test_unknown_keying_raises(self):
        with self.assertRaises(ValueError):
            self._accept(success(ID_PATHS), "name")

    def test_index_is_still_the_default(self):
        outcome = accept_replan_result(
            success([INDEX_PATHS[0], INDEX_PATHS[1]]), agent_order=(3, 7),
            current_positions=STARTS, goals=GOALS, grid=open_grid(),
            path_validator=AcceptAllValidator(),
        )
        self.assertTrue(outcome.accepted)


class TestRealPathGuardAtBoundary(_TmpCase):

    def test_validator_is_not_test_only(self):
        self.assertFalse(getattr(PathGuardValidator(), "test_only", False))
        check_validator_pairing(self.planner(), PathGuardValidator())

    def test_accept_all_validator_refused_with_runtime_planner(self):
        with self.assertRaises(ValueError):
            check_validator_pairing(self.planner(), AcceptAllValidator())

    def test_validator_verdicts(self):
        v = PathGuardValidator()
        self.assertEqual(v(open_grid(), STARTS, GOALS, ID_PATHS), (True, ""))
        ok, why = v(open_grid(), STARTS, GOALS, {3: ID_PATHS[3], 7: [(2, 0), (2, 1), (2, 1)]})
        self.assertFalse(ok)
        self.assertIn("goal", why)

    def test_boundary_guard_catches_conflict_that_bypassed_adapter(self):
        # Adjacent moves (so not BOUNDARY_NON_ADJACENT) but both land on (1, 0)
        # at t=1: only the Path Guard can refuse this.
        bad = {3: [(0, 0), (1, 0), (0, 0), (0, 1), (0, 2)],
               7: [(2, 0), (1, 0), (1, 0), (1, 0), (1, 0)]}

        def fake_runtime(current_positions, goals, grid_map, **kwargs):
            return CBSRuntimeResult(status="success", paths=bad, latency_sec=0.1)

        with patch("src.hybrid.cbs_runtime_bridge.run_cbs_runtime", side_effect=fake_runtime):
            result = _run(self.planner())
        self.assertEqual(result["termination"], "SAFE_STOP")
        self.assertTrue(result["stop_reason"].startswith("VALIDATOR_REJECT:"), result["stop_reason"])
        self.assertIn("Vertex conflict", result["stop_reason"])
        self.assertNotIn("CBS", result["action_sources"])


class TestPlannerConstructionAndWorkDir(_TmpCase):

    def test_solver_root_must_contain_cbs_py(self):
        with self.assertRaises(FileNotFoundError):
            CbsRuntimePlanner(solver_root=self.tmp, work_root=self.work_root)

    def test_work_root_is_required(self):
        with self.assertRaises(TypeError):
            CbsRuntimePlanner(solver_root=self.solver_root)

    def test_each_call_gets_its_own_new_work_dir(self):
        seen_work_dirs = []

        def fake_plan(adapter_self, starts, goals, grid):
            seen_work_dirs.append(Path(adapter_self.config.work_dir).resolve())
            seen_work_dirs.append(Path(adapter_self.config.solver_root))
            return {0: [(0, 0), (0, 1), (0, 2)], 1: [(2, 0), (1, 0), (1, 0)]}

        cwd = os.getcwd()
        planner = self.planner()
        with patch("src.replanner.CBSAdapter.plan", autospec=True, side_effect=fake_plan):
            first = planner(STARTS, GOALS, open_grid())
            second = planner(STARTS, GOALS, open_grid())
        self.assertEqual(os.getcwd(), cwd)
        self.assertEqual((first.status, second.status), ("success", "success"))
        self.assertEqual(seen_work_dirs[0], (self.work_root / "cbs_call_01" / "outputs/paths/replan").resolve())
        self.assertEqual(seen_work_dirs[2], (self.work_root / "cbs_call_02" / "outputs/paths/replan").resolve())
        self.assertEqual(seen_work_dirs[1], self.solver_root.resolve())
        self.assertEqual([c["work_dir"] for c in planner.calls],
                         [self.work_root / "cbs_call_01", self.work_root / "cbs_call_02"])

    def test_existing_call_dir_is_never_reused(self):
        (self.work_root / "cbs_call_01").mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            self.planner()(STARTS, GOALS, open_grid())


if __name__ == "__main__":
    unittest.main()
