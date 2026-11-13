"""Mock tests for src/hybrid/subset_cbs_adapter.SubsetCbsPlanner.

No real CBS runs here. Layer A replaces ``run_cbs_runtime``; layer B keeps
the whole existing chain (run_cbs_runtime -> replan_with_status -> replan ->
CBSAdapter -> run_atb033_cbs) and replaces only ``subprocess.run`` so the
YAML, coordinate conversion and the seconds timeout handed to the solver are
checked without executing atb033. Paths in the fixture are the ones Yunji's
standalone real run printed (docs/evidence/selective_subset_cbs_real_20261113.log);
using them here is NOT a real CBS result.
"""

import math
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from src.cbs_runtime_adapter import CBSRuntimeResult
from src.common.deadlock_monitor import DeadlockMonitor
from src.hybrid import subset_cbs_adapter as mod
from src.hybrid.selective_compose import (
    OUTSIDE_HELD,
    SelectiveFullPathValidator,
    compose_selective_plan,
    selective_boundary_check,
)
from src.hybrid.selective_planner import SelectiveConfig, SelectivePlanner
from src.hybrid.selective_safety import TriggerBudget
from src.hybrid.selective_shadow import ShadowMonitor
from src.hybrid.subset_cbs_adapter import BLOCKED, REJECTED, SubsetCbsPlanner

WAIT = 4

GRID = np.zeros((3, 7), dtype=int)
S_T = {7: (1, 5), 10: (0, 1), 20: (2, 1)}
GOALS = {7: (1, 6), 10: (2, 1), 20: (0, 1)}
GROUP = (10, 20)
OUTSIDE = {7: (1, 5)}
G_START = {a: S_T[a] for a in GROUP}
G_GOAL = {a: GOALS[a] for a in GROUP}
PATHS = {
    10: [(0, 1), (1, 1), (2, 1), (2, 1), (2, 1)],
    20: [(2, 1), (2, 2), (1, 2), (1, 1), (0, 1)],
}


class FakeRuntime:
    """Stands in for run_cbs_runtime; records every call."""

    def __init__(self, result=None, raise_exc=None):
        self.result = result or CBSRuntimeResult("success", {a: list(p) for a, p in PATHS.items()}, 0.25)
        self.raise_exc = raise_exc
        self.calls = []

    def __call__(self, current_positions, goals, grid_map, timeout_sec=30, solver_root=None):
        self.calls.append(dict(current_positions=dict(current_positions), goals=dict(goals),
                               grid=grid_map, grid_copy=np.array(grid_map, copy=True),
                               timeout_sec=timeout_sec, solver_root=solver_root, cwd=Path.cwd()))
        if self.raise_exc:
            raise self.raise_exc
        return self.result


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.solver = self.tmp / "solver"
        self.solver.mkdir()
        (self.solver / "cbs.py").write_text("raise SystemExit('must never run in these tests')\n")
        self.work = self.tmp / "work"
        self.cwd0 = Path.cwd()

    def tearDown(self):
        self.assertEqual(Path.cwd(), self.cwd0, "cwd must be restored")
        self._tmp.cleanup()

    def planner(self, reserve=0.0, work_root=None):
        self._n = getattr(self, "_n", 0) + 1
        root = work_root if work_root is not None else self.work / f"planner_{self._n:02d}"
        return SubsetCbsPlanner(solver_root=self.solver, work_root=root, overhead_reserve_ms=reserve)

    def call(self, fake, *, start=None, goal=None, grid=None, outside=None, timeout_ms=1000.0, reserve=0.0,
             planner=None):
        p = planner or self.planner(reserve)
        with patch.object(mod, "run_cbs_runtime", fake):
            res = p(G_START if start is None else start, G_GOAL if goal is None else goal,
                    GRID if grid is None else grid,
                    outside=OUTSIDE if outside is None else outside, timeout_ms=timeout_ms)
        return res, p


# -- A. contract with run_cbs_runtime mocked ---------------------------------

class TestSuccessContract(_Base):
    def test_group_only_ids_and_coords_kept(self):
        fake = FakeRuntime()
        res, p = self.call(fake)
        self.assertEqual(res.status, "success")
        self.assertIsNone(res.error)
        self.assertEqual(res.paths, PATHS)
        self.assertEqual(set(res.paths), set(GROUP))
        for path in res.paths.values():
            self.assertTrue(all(type(c) is tuple and len(c) == 2 for c in path))
        call = fake.calls[0]
        self.assertEqual(call["current_positions"], G_START)
        self.assertEqual(call["goals"], G_GOAL)
        self.assertNotIn(7, call["current_positions"])
        self.assertEqual(call["solver_root"], self.solver.resolve())
        self.assertIsInstance(res.latency_sec, float)
        self.assertGreaterEqual(res.latency_sec, 0.0)

    def test_outside_is_obstacle_on_a_copy_only(self):
        grid = GRID.copy()
        before = grid.copy()
        fake = FakeRuntime()
        self.call(fake, grid=grid)
        np.testing.assert_array_equal(grid, before)                 # caller grid untouched
        sent = fake.calls[0]["grid_copy"]
        self.assertIsNot(fake.calls[0]["grid"], grid)
        self.assertEqual(int(sent[1, 5]), 1)                         # outside agent 7
        expected = before.copy()
        expected[1, 5] = 1
        np.testing.assert_array_equal(sent, expected)                # nothing else changed

    def test_list_grid_accepted_and_not_modified(self):
        grid = GRID.tolist()
        before = [row[:] for row in grid]
        res, _ = self.call(FakeRuntime(), grid=grid)
        self.assertEqual(res.status, "success")
        self.assertEqual(grid, before)

    def test_signature_matches_selective_planner_call(self):
        # SelectivePlanner._call_level: planner(starts, goals, np.array(grid, copy=True), outside=..., timeout_ms=...)
        res, _ = self.call(FakeRuntime(), grid=np.array(GRID, copy=True))
        self.assertEqual(res.status, "success")
        self.assertEqual(SubsetCbsPlanner.paths_keying, "agent_id")

    def test_result_has_exactly_the_four_fields(self):
        res, _ = self.call(FakeRuntime())
        self.assertIsInstance(res, CBSRuntimeResult)
        self.assertEqual(set(vars(res)), {"status", "paths", "latency_sec", "error"})


class TestTimeout(_Base):
    def test_seconds_never_rounded_up(self):
        for ms, reserve in ((1000.0, 0.0), (1234.5, 0.0), (1.0, 0.0), (800.0, 150.0), (999, 0)):
            fake = FakeRuntime()
            res, _ = self.call(fake, timeout_ms=ms, reserve=reserve)
            sent = fake.calls[0]["timeout_sec"]
            self.assertIsInstance(sent, float)
            self.assertEqual(sent, (ms - reserve) / 1000.0)
            self.assertLessEqual(sent * 1000.0, ms)
            self.assertEqual(res.status, "success")

    def test_bad_timeout_blocks_before_call(self):
        for ms in (None, 0, 0.0, -5.0, math.nan, math.inf, True, "500", [500]):
            fake = FakeRuntime()
            res, p = self.call(fake, timeout_ms=ms)
            self.assertEqual(res.status, "input_error", ms)
            self.assertIn(f"{BLOCKED}:TIMEOUT_MS", res.error)
            self.assertIsNone(res.paths)
            self.assertEqual(fake.calls, [])
            self.assertFalse(p.calls[-1]["called"])
            self.assertFalse(p.work_root.exists())

    def test_reserve_consuming_budget_blocks(self):
        for ms, reserve in ((100.0, 100.0), (100.0, 250.0)):
            fake = FakeRuntime()
            res, _ = self.call(fake, timeout_ms=ms, reserve=reserve)
            self.assertEqual(res.status, "input_error")
            self.assertIn("TIMEOUT_BUDGET", res.error)
            self.assertEqual(fake.calls, [])

    def test_reserve_must_be_explicit_and_valid(self):
        with self.assertRaises(TypeError):
            SubsetCbsPlanner(solver_root=self.solver, work_root=self.work)
        for bad in (-1.0, math.nan, math.inf, True, None, "0"):
            with self.assertRaises(ValueError):
                SubsetCbsPlanner(solver_root=self.solver, work_root=self.work, overhead_reserve_ms=bad)

    def test_missing_solver_refused(self):
        with self.assertRaises(FileNotFoundError):
            SubsetCbsPlanner(solver_root=self.tmp / "nope", work_root=self.work, overhead_reserve_ms=0)

    def test_overrun_recorded_on_slow_success(self):
        clock = iter([0.0, 2.5])
        fake = FakeRuntime()
        with patch.object(mod, "perf_counter", lambda: next(clock)):
            res, p = self.call(fake, timeout_ms=1000.0)
        self.assertEqual(res.status, "success")
        self.assertEqual(res.latency_sec, 2.5)                      # whole call is charged
        self.assertEqual(p.calls[0]["budget_overrun_ms"], 1500.0)

    def test_latency_is_outer_wall_time(self):
        fake = FakeRuntime(CBSRuntimeResult("timeout", None, 0.9, "x"))
        clock = iter([10.0, 11.25])
        with patch.object(mod, "perf_counter", lambda: next(clock)):
            res, p = self.call(fake)
        self.assertEqual(res.latency_sec, 1.25)
        self.assertEqual(p.calls[0]["solver_latency_sec"], 0.9)
        self.assertEqual(p.calls[0]["budget_overrun_ms"], 250.0)


class TestBlockedInputs(_Base):
    def assertBlocked(self, tag, **kw):
        fake = FakeRuntime()
        res, p = self.call(fake, **kw)
        self.assertEqual(res.status, "input_error", tag)
        self.assertIsNone(res.paths)
        self.assertTrue(res.error.startswith(BLOCKED), res.error)
        self.assertIn(tag, res.error)
        self.assertEqual(fake.calls, [], tag)
        self.assertFalse(p.calls[-1]["called"])
        self.assertFalse(p.work_root.exists(), "no work dir for a blocked call")

    def test_goal_blocked_by_outside(self):
        # outside agent 7 stands on 20's goal (1,5)
        self.assertBlocked("GOAL_BLOCKED", start={10: (0, 1), 20: (0, 2)}, goal={10: (2, 1), 20: (1, 5)})
        self.assertBlocked("GOAL_BLOCKED", outside={7: (1, 5), 8: (2, 3)},
                           start={10: (0, 1), 20: (0, 2)}, goal={10: (2, 3), 20: (2, 4)})

    def test_group_outside_id_overlap(self):
        self.assertBlocked("GROUP_OUTSIDE_OVERLAP", outside={10: (1, 5)})

    def test_overlapping_cells(self):
        self.assertBlocked("OVERLAPPING_STARTS", outside={7: (0, 1)})
        self.assertBlocked("OVERLAPPING_STARTS", start={10: (0, 1), 20: (0, 1)})

    def test_duplicate_goals(self):
        self.assertBlocked("DUPLICATE_GROUP_GOALS", goal={10: (2, 1), 20: (2, 1)})

    def test_bounds_and_obstacles(self):
        self.assertBlocked("OUT_OF_BOUNDS", outside={7: (3, 0)})
        self.assertBlocked("OUT_OF_BOUNDS", goal={10: (2, 7), 20: (0, 1)})
        g = GRID.copy()
        g[1, 5] = 1
        self.assertBlocked("ON_OBSTACLE", grid=g)
        g = GRID.copy()
        g[2, 1] = 1
        self.assertBlocked("ON_OBSTACLE", grid=g)

    def test_agent_set(self):
        self.assertBlocked("EMPTY_GROUP", start={}, goal={})
        self.assertBlocked("GROUP_ID_MISMATCH", goal={10: (2, 1)})
        self.assertBlocked("AGENT_ID", start={True: (0, 1), 20: (2, 1)}, goal={True: (2, 1), 20: (0, 1)})
        self.assertBlocked("AGENT_ID", outside={"7": (1, 5)})
        self.assertBlocked("NOT_A_MAPPING", outside=[(1, 5)])

    def test_bad_cells_and_grid(self):
        self.assertBlocked("CELL", outside={7: (1.0, 5)})
        self.assertBlocked("CELL", start={10: "01", 20: (2, 1)})
        self.assertBlocked("CELL", goal={10: (2, 1, 0), 20: (0, 1)})
        self.assertBlocked("GRID", grid=np.zeros((3,), dtype=int))
        self.assertBlocked("GRID", grid=np.full((3, 7), 2))
        g = GRID.astype(float)
        g[0, 6] = 0.5                                               # must not be truncated to 0
        self.assertBlocked("GRID", grid=g)
        self.assertBlocked("GRID", grid=[[0, 0], [0]])              # ragged
        self.assertBlocked("GRID", grid=np.array([["0"] * 7] * 3))  # strings


class TestWorkDirs(_Base):
    def test_fresh_dir_per_call_and_cwd(self):
        fake = FakeRuntime()
        p = self.planner()
        self.call(fake, planner=p)
        self.call(fake, planner=p)
        dirs = [c["cwd"] for c in fake.calls]
        self.assertEqual([d.name for d in dirs], ["subset_cbs_call_01", "subset_cbs_call_02"])
        self.assertEqual(dirs[0].parent, p.work_root)
        self.assertEqual([c["work_dir"] for c in p.calls], dirs)

    def test_existing_dir_never_reused(self):
        d = self.work / "subset_cbs_call_01"
        (d / "outputs/paths/replan").mkdir(parents=True)
        old = d / "outputs/paths/replan/input.yaml"
        old.write_text("keep: me\n")
        fake = FakeRuntime()
        res, p = self.call(fake, planner=self.planner(work_root=self.work))
        self.assertEqual(res.status, "input_error")
        self.assertIn("WORK_DIR", res.error)
        self.assertEqual(fake.calls, [])
        self.assertEqual(old.read_text(), "keep: me\n")

    def test_cwd_restored_after_exception(self):
        res, _ = self.call(FakeRuntime(raise_exc=RuntimeError("boom")))
        self.assertEqual(res.status, "solver_error")
        self.assertIsNone(res.paths)
        self.assertEqual(Path.cwd(), self.cwd0)


class TestFailureStatuses(_Base):
    def test_non_success_passes_through_without_paths(self):
        for st in ("timeout", "no_solution", "invalid_path", "input_error", "solver_error"):
            res, _ = self.call(FakeRuntime(CBSRuntimeResult(st, None, 0.1, f"injected {st}")))
            self.assertEqual(res.status, st)
            self.assertIsNone(res.paths)
            self.assertEqual(res.error, f"injected {st}")

    def test_non_success_with_paths_still_drops_paths(self):
        res, _ = self.call(FakeRuntime(CBSRuntimeResult("timeout", dict(PATHS), 0.1, "t")))
        self.assertIsNone(res.paths)

    def test_unknown_status(self):
        res, _ = self.call(FakeRuntime(CBSRuntimeResult("weird", dict(PATHS), 0.1)))
        self.assertEqual(res.status, "solver_error")
        self.assertIsNone(res.paths)


class TestSuccessRejected(_Base):
    def assertRejected(self, paths, tag="", **kw):
        res, _ = self.call(FakeRuntime(CBSRuntimeResult("success", paths, 0.1)), **kw)
        self.assertEqual(res.status, "invalid_path", paths)
        self.assertIsNone(res.paths)
        self.assertTrue(res.error.startswith(REJECTED), res.error)
        self.assertIn(tag, res.error)

    def test_agent_set(self):
        self.assertRejected({**PATHS, 7: [(1, 5)] * 5}, "AGENT_SET")          # outside agent returned
        self.assertRejected({10: PATHS[10]}, "AGENT_SET")                     # agent lost
        self.assertRejected({0: PATHS[10], 1: PATHS[20]}, "AGENT_SET")        # renumbered to indexes
        self.assertRejected(None, "PATHS_NOT_A_MAPPING")
        self.assertRejected([PATHS[10], PATHS[20]], "PATHS_NOT_A_MAPPING")

    def test_endpoints_and_moves(self):
        self.assertRejected({**PATHS, 10: [(0, 0), (1, 0), (2, 0), (2, 1), (2, 1)]}, "GROUP_PATH")  # wrong start
        self.assertRejected({**PATHS, 10: [(0, 1), (1, 1), (1, 1), (1, 1), (1, 1)]}, "GROUP_PATH")  # wrong goal
        self.assertRejected({**PATHS, 10: [(0, 1), (2, 1), (2, 1), (2, 1), (2, 1)]}, "GROUP_PATH")  # jump
        self.assertRejected({**PATHS, 10: [(0, 1), (2, 1)]}, "UNEQUAL_LENGTH")
        self.assertRejected({**PATHS, 10: []}, "MALFORMED")
        self.assertRejected({**PATHS, 10: "abc"}, "MALFORMED")

    def test_conflicts(self):
        vertex = {10: [(0, 1), (1, 1), (2, 1)], 20: [(2, 1), (1, 1), (0, 1)]}
        self.assertRejected(vertex, "Vertex")
        swap = {10: [(1, 1), (1, 2)], 20: [(1, 2), (1, 1)]}                  # edge swap
        self.assertRejected(swap, "GROUP_PATH", start={10: (1, 1), 20: (1, 2)}, goal={10: (1, 2), 20: (1, 1)})

    def test_malformed_mapping_refused_not_raised(self):
        class Exploding(dict):
            def items(self):
                raise RuntimeError("boom")
        self.assertRejected(Exploding(PATHS), "CHECK_EXCEPTION")

    def test_full_robot_layer_rejects_on_its_own(self):
        """Two independent layers stop a path into a held outside agent: the map
        check on the copied grid (outside = obstacle) and validate_full_robot on
        the original grid. With the first layer disabled the second must still refuse."""
        start = {10: (1, 4), 20: (0, 0)}
        goal = {10: (1, 6), 20: (0, 1)}
        through = {10: [(1, 4), (1, 5), (1, 6)], 20: [(0, 0), (0, 1), (0, 1)]}
        with patch.object(mod, "validate_path_map_and_conflicts", lambda paths, grid: None):
            res, _ = self.call(FakeRuntime(CBSRuntimeResult("success", through, 0.1)), start=start, goal=goal)
        self.assertEqual(res.status, "invalid_path")
        self.assertIn(f"{REJECTED}:FULL:FULL_CHECK_REJECT", res.error)
        self.assertIsNone(res.paths)
        # and the copy-grid map layer alone (full check disabled) also refuses
        fake_ok = type("F", (), {"ok": True, "reason": "", "full_paths": None})()
        with patch.object(mod, "validate_full_robot", lambda *a: fake_ok):
            res, _ = self.call(FakeRuntime(CBSRuntimeResult("success", through, 0.1)), start=start, goal=goal)
        self.assertEqual(res.status, "invalid_path")
        self.assertIn(f"{REJECTED}:GROUP_PATH", res.error)

    def test_path_through_outside_agent(self):
        start = {10: (1, 4), 20: (0, 0)}
        goal = {10: (1, 6), 20: (0, 1)}
        through = {10: [(1, 4), (1, 5), (1, 6)], 20: [(0, 0), (0, 1), (0, 1)]}  # crosses held agent 7
        res, _ = self.call(FakeRuntime(CBSRuntimeResult("success", through, 0.1)), start=start, goal=goal)
        self.assertEqual(res.status, "invalid_path")
        self.assertIn(REJECTED, res.error)
        self.assertIsNone(res.paths)


# -- compatibility with compose / validator / boundary / SelectivePlanner -----

class TestSelectiveCompatibility(_Base):
    def test_compose_validator_boundary(self):
        res, _ = self.call(FakeRuntime())
        plan = compose_selective_plan(res.paths, S_T, GOALS, sorted(S_T), GRID)
        self.assertTrue(plan.ok, plan.reason)
        self.assertEqual(plan.group, GROUP)
        self.assertEqual(plan.goal_status[7], OUTSIDE_HELD)
        self.assertTrue(all(c == S_T[7] for c in plan.full_paths[7]))
        ok, why = SelectiveFullPathValidator(GROUP)(GRID, S_T, GOALS, plan.full_paths)
        self.assertTrue(ok, why)
        outcome, _ = selective_boundary_check(res, s_t=S_T, goals=GOALS, grid=GRID)
        self.assertTrue(outcome.accepted, outcome.reason)
        self.assertTrue(all(a[7] == WAIT for a in outcome.plan.actions))

    def test_outside_wait_violation_rejected_downstream(self):
        res, _ = self.call(FakeRuntime())
        plan = compose_selective_plan(res.paths, S_T, GOALS, sorted(S_T), GRID)
        tampered = {a: list(p) for a, p in plan.full_paths.items()}
        tampered[7][1] = (1, 6)
        ok, why = SelectiveFullPathValidator(GROUP)(GRID, S_T, GOALS, tampered)
        self.assertFalse(ok)
        self.assertIn("OUTSIDE_NOT_HELD", why)

    def test_failures_refused_by_boundary(self):
        for st in ("timeout", "no_solution", "invalid_path", "input_error", "solver_error"):
            res, _ = self.call(FakeRuntime(CBSRuntimeResult(st, None, 0.1, "x")))
            outcome, _ = selective_boundary_check(res, s_t=S_T, goals=GOALS, grid=GRID)
            self.assertFalse(outcome.accepted)
            self.assertIsNone(outcome.plan)
        res, _ = self.call(FakeRuntime(), outside={7: (2, 1)})               # blocked
        outcome, _ = selective_boundary_check(res, s_t=S_T, goals=GOALS, grid=GRID)
        self.assertFalse(outcome.accepted)

    def test_selective_planner_still_refuses_real_adapter(self):
        """PENDING:REAL_SUBSET_CBS is NOT lifted in this step."""
        p = self.planner()
        self.assertFalse(getattr(p, "test_only", True))

        class G:
            paths_keying = "agent_id"
            accepts_timeout_ms = True
            test_only = True

            def __call__(self, *a, **k):
                raise AssertionError("not called")

        cfg = SelectiveConfig(ladder=("L0", "G"), budget=TriggerBudget(1000.0, 2, {"L0": 400.0, "G": 600.0}),
                              acknowledge_unverified_outside_policy=True)
        with self.assertRaisesRegex(ValueError, "PENDING:REAL_SUBSET_CBS"):
            SelectivePlanner(G(), ShadowMonitor(DeadlockMonitor()), cfg, subset_planner=p)


# -- B. existing chain, only subprocess.run replaced -------------------------

class TestExistingChainSubprocessMocked(_Base):
    """run_cbs_runtime / replan / CBSAdapter run for real; atb033 does not."""

    def _fake_run(self, *, write_schedule=None, raise_timeout=False):
        seen = {}

        def fake(command, cwd=None, capture_output=None, text=None, timeout=None, check=None):
            seen.update(command=command, cwd=Path(cwd), timeout=timeout, proc_cwd=Path.cwd())
            seen["solver_input"] = yaml.unsafe_load(Path(command[2]).read_text(encoding="utf-8"))
            if raise_timeout:
                raise subprocess.TimeoutExpired(command, timeout)
            if write_schedule is not None:
                Path(command[3]).write_text(yaml.safe_dump({"cost": 0, "schedule": write_schedule}),
                                            encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        return fake, seen

    @staticmethod
    def _schedule(paths_by_index):
        return {f"agent{i}": [{"t": t, "x": c, "y": r} for t, (r, c) in enumerate(p)]
                for i, p in paths_by_index.items()}

    def call_chain(self, fake, timeout_ms=1500.0):
        p = self.planner()
        with patch("src.cbs_adapter.subprocess.run", fake):
            res = p(G_START, G_GOAL, GRID, outside=OUTSIDE, timeout_ms=timeout_ms)
        return res, p

    def test_full_chain_success_ids_restored(self):
        fake, seen = self._fake_run(write_schedule=self._schedule({0: PATHS[10], 1: PATHS[20]}))
        res, p = self.call_chain(fake, timeout_ms=1500.0)
        self.assertEqual(res.status, "success", res.error)
        self.assertEqual(res.paths, PATHS)                         # index 0/1 -> ids 10/20
        self.assertEqual(seen["timeout"], 1.5)                     # exactly timeout_ms / 1000
        self.assertEqual(seen["cwd"], self.solver.resolve())
        inp = seen["solver_input"]
        self.assertEqual([a["name"] for a in inp["agents"]], ["agent0", "agent1"])   # group only
        self.assertEqual(inp["agents"][0]["start"], [1, 0])        # (row 0, col 1) -> [x=1, y=0]
        self.assertEqual(inp["agents"][1]["goal"], [1, 0])
        self.assertIn((5, 1), inp["map"]["obstacles"])             # outside 7 at (1,5) -> (x=5, y=1)
        self.assertEqual(len(inp["map"]["obstacles"]), 1)
        self.assertEqual(inp["map"]["dimensions"], [7, 3])
        work = p.calls[0]["work_dir"]
        self.assertTrue((work / "outputs/paths/replan/input.yaml").is_file())
        self.assertTrue((work / "outputs/paths/replan/output.yaml").is_file())
        self.assertEqual(sorted(x.name for x in p.work_root.iterdir()), ["subset_cbs_call_01"])

    def test_full_chain_timeout(self):
        fake, seen = self._fake_run(raise_timeout=True)
        res, _ = self.call_chain(fake, timeout_ms=250.0)
        self.assertEqual(res.status, "timeout")
        self.assertIsNone(res.paths)
        self.assertEqual(seen["timeout"], 0.25)
        self.assertIn("0.25s", res.error)

    def test_full_chain_no_solution(self):
        def fake(command, **kw):
            return subprocess.CompletedProcess(command, 0, stdout=" Solution not found", stderr="")
        res, _ = self.call_chain(fake)
        self.assertEqual(res.status, "no_solution")
        self.assertIsNone(res.paths)

    def test_full_chain_path_through_outside_rejected(self):
        # solver "ignores" the obstacle: agent 0 (id 10) goes through (1,5) -- caught by Path Guard
        start = {10: (1, 4), 20: (0, 0)}
        goal = {10: (1, 6), 20: (0, 1)}
        bad = self._schedule({0: [(1, 4), (1, 5), (1, 6)], 1: [(0, 0), (0, 1), (0, 1)]})
        fake, _ = self._fake_run(write_schedule=bad)
        p = self.planner()
        with patch("src.cbs_adapter.subprocess.run", fake):
            res = p(start, goal, GRID, outside=OUTSIDE, timeout_ms=1000.0)
        self.assertEqual(res.status, "invalid_path")
        self.assertIsNone(res.paths)

    def test_project_outputs_untouched(self):
        root = Path(__file__).resolve().parents[1]
        replan_dir = root / "outputs/paths/replan"
        before = {f: f.stat().st_mtime_ns for f in replan_dir.glob("*")} if replan_dir.exists() else None
        fake, _ = self._fake_run(write_schedule=self._schedule({0: PATHS[10], 1: PATHS[20]}))
        self.call_chain(fake)
        after = {f: f.stat().st_mtime_ns for f in replan_dir.glob("*")} if replan_dir.exists() else None
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
