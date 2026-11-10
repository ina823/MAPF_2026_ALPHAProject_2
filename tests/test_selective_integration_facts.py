"""Integration FACTS for a future Selective Hybrid wiring — no policy decided here.

These tests pin down behaviour that already exists in the code (WFG, decision,
safety scaffold, CBS Adapter input path, Path Guard, Simulator, replan
boundary). They do NOT resolve any PENDING item of tests/test_selective_safety.py
and they assert no unagreed policy.

No real CBS: ``src.cbs_adapter.run_atb033_cbs`` is patched to a recorder that
fails if reached, the solver root points at a nonexistent path, and every
adapter call runs with cwd in a fresh temporary directory (the replanner writes
to the relative path outputs/paths/replan). Inputs are small synthetic grids.

NOT verifiable here (reported, not asserted): how the real atb033 solver
behaves with outside agents encoded as static obstacles (expected timeout for
an unreachable goal, per code reading only), and any Runner path that injects
WAIT for outside agents (none exists).
"""

import contextlib
import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml

from src.cbs_adapter import create_atb033_input
from src.cbs_runtime_adapter import run_cbs_runtime
from src.common import spec
from src.hybrid.cbs_runtime_bridge import PathGuardValidator
from src.hybrid.replan_boundary import accept_replan_result
from src.hybrid.selective_decision import ROUTE_GLOBAL, decide_selective
from src.hybrid.selective_safety import expansion_candidates, validate_full_robot
from src.hybrid.wfg import build_wfg
from src.il.nav_hint import bfs_dist
from src.path_guard import InvalidPathError, validate_path_guard
from src.simulator.mapf_step_simulator import MAPFStepSimulator
from tests.hybrid_mocks import LEFT, RIGHT, MockReplanResult

NONEXISTENT_SOLVER = Path(tempfile.gettempdir()) / "alpha2_no_such_cbs_solver_root"
WAIT_ACTION = spec.delta_to_action(0, 0)


def _open(h=3, w=7):
    return np.zeros((h, w), dtype=np.int64)


@contextlib.contextmanager
def _no_solver():
    """Run in a temp cwd with the solver subprocess entry point patched out."""
    with tempfile.TemporaryDirectory() as tmp, contextlib.chdir(tmp):
        with mock.patch("src.cbs_adapter.run_atb033_cbs",
                        side_effect=RuntimeError("solver must not be called")) as solver:
            yield Path(tmp), solver


def _mark_outside(grid, s_t, outside):
    aug = np.array(grid, dtype=np.int64, copy=True)
    for a in outside:
        r, c = s_t[a]
        aug[r, c] = 1
    return aug


def _step(sim, actions):
    """One simulator step, recording S_{t-1}, intended, S_t and blocked the way
    global_hybrid.run_step feeds the Monitor."""
    before = {a: tuple(sim._positions[a]) for a in sim._agent_ids}
    intended = {}
    for a in sim._agent_ids:
        dr, dc = spec.ACTION_DELTA[actions[a]]
        intended[a] = (before[a][0] + dr, before[a][1] + dc)
    sim.step(actions)
    after = {a: tuple(sim._positions[a]) for a in sim._agent_ids}
    blocked = {a for a, c in sim._last_conflict_types.items() if c != "none"}
    return before, intended, after, blocked


# Shared synthetic state: {0,1} contest (0,2) and are cancelled; 2 moves freely.
STARTS = {0: (0, 1), 1: (0, 3), 2: (2, 6)}
GOALS = {0: (1, 2), 1: (0, 1), 2: (2, 0)}
ACTIONS = {0: RIGHT, 1: LEFT, 2: LEFT}


def _trigger_state():
    grid = _open()
    sim = MAPFStepSimulator(cbs_solver_root=None, max_steps=10)
    sim.reset(grid, STARTS, GOALS)
    before, intended, s_t, blocked = _step(sim, ACTIONS)
    return grid, before, intended, s_t, blocked


# -- 1. time axis: WFG on S_{t-1}+intended, CBS from S_t ----------------------

class TestTimeAxis(unittest.TestCase):

    def test_simulator_step_separates_pre_and_post_move_states(self):
        grid, before, intended, s_t, blocked = _trigger_state()
        self.assertEqual(before, STARTS)
        self.assertEqual(blocked, {0, 1})
        self.assertEqual(s_t, {0: (0, 1), 1: (0, 3), 2: (2, 5)})
        # Blocked agents: S_{t-1} == S_t. The mover: they differ.
        self.assertNotEqual(before[2], s_t[2])
        # intended is a one-step offset from S_{t-1}, not from S_t.
        self.assertEqual(intended[2], s_t[2])
        self.assertEqual(intended, {0: (0, 2), 1: (0, 2), 2: (2, 5)})

    def test_wfg_input_is_pre_move_state(self):
        grid, before, intended, s_t, blocked = _trigger_state()
        wfg = build_wfg(before, intended, blocked, grid)
        self.assertEqual(wfg.groups, ((0, 1),))
        self.assertEqual([(e.src, e.dst, e.kind, e.cell) for e in wfg.edges],
                         [(0, 1, "contest", (0, 2))])
        # Decision consumes S_t positions with that WFG; still GLOBAL in v0.
        d = decide_selective(sorted(s_t), s_t, GOALS, wfg)
        self.assertEqual(d.route, ROUTE_GLOBAL)
        self.assertEqual(d.outside_agents, (2,))

    def test_cbs_start_must_be_post_move_state(self):
        grid, before, intended, s_t, blocked = _trigger_state()
        stale = {0: [s_t[0]], 1: [s_t[1]], 2: [before[2], (2, 4)]}
        fresh = {0: [s_t[0]], 1: [s_t[1]], 2: [s_t[2], (2, 4)]}
        self.assertEqual(validate_full_robot(stale, s_t, [0, 1, 2], grid).reason,
                         "FULL_CHECK_START_MISMATCH:2")
        self.assertTrue(validate_full_robot(fresh, s_t, [0, 1, 2], grid).ok)
        accept_all = lambda g, st, gl, p: (True, "")  # noqa: E731 (boundary-only probe)
        for paths, want in ((stale, "BOUNDARY_START_MISMATCH"), (fresh, "")):
            out = accept_replan_result(
                MockReplanResult("success", paths, 0.0), agent_order=(0, 1, 2),
                current_positions=s_t, goals=GOALS, grid=grid,
                path_validator=accept_all, paths_keying="agent_id")
            self.assertEqual(out.reason, want)


# -- 2. outside agents as obstacles on a COPY of the grid ----------------------

class TestObstacleCopy(unittest.TestCase):

    def test_numpy_copy_leaves_original_untouched(self):
        grid, _, _, s_t, _ = _trigger_state()
        snapshot = grid.copy()
        aug = _mark_outside(grid, s_t, (2,))
        self.assertEqual(int(aug[2, 5]), 1)
        self.assertTrue(np.array_equal(grid, snapshot))
        self.assertEqual(int(grid[2, 5]), 0)

    def test_shallow_list_copy_would_mutate_original(self):
        # FACT about list-of-lists grids: a shallow copy shares rows. Any
        # future obstacle marking must deep-copy (or use a numpy copy).
        grid = [[0] * 7 for _ in range(3)]
        shallow = list(grid)
        shallow[2][5] = 1
        self.assertEqual(grid[2][5], 1)
        grid = [[0] * 7 for _ in range(3)]
        deep = copy.deepcopy(grid)
        deep[2][5] = 1
        self.assertEqual(grid[2][5], 0)

    def test_adapter_encodes_marked_cell_as_xy_obstacle(self):
        grid, _, _, s_t, _ = _trigger_state()
        aug = _mark_outside(grid, s_t, (2,))
        starts = [s_t[0], s_t[1]]
        goals = [GOALS[0], GOALS[1]]
        with tempfile.TemporaryDirectory() as tmp:
            orig = yaml.safe_load(create_atb033_input(grid, starts, goals, Path(tmp) / "a.yaml").read_text())
            marked = yaml.safe_load(create_atb033_input(aug, starts, goals, Path(tmp) / "b.yaml").read_text())
        self.assertEqual(orig["map"]["obstacles"], [])
        # (row, col) = (2, 5) -> atb033 [x, y] = [5, 2]; conversion only in the adapter.
        self.assertEqual(marked["map"]["obstacles"], [[5, 2]])
        self.assertEqual(marked["map"]["dimensions"], [7, 3])
        # Only group agents are planned; the outside agent appears only as a cell.
        self.assertEqual([a["name"] for a in marked["agents"]], ["agent0", "agent1"])
        self.assertEqual(marked["agents"][0]["start"], [1, 0])


# -- 3. group goal blocked: detected before any solver run ---------------------

class TestBlockedGoalDetectedBeforeSolver(unittest.TestCase):

    def setUp(self):
        self.grid = _open()
        # Outside 2 stands on group agent 0's goal.
        self.s_t = {0: (0, 1), 1: (0, 3), 2: (2, 5)}
        self.goals = {0: (2, 5), 1: (0, 1), 2: (2, 0)}
        self.aug = _mark_outside(self.grid, self.s_t, (2,))

    def test_precheck_flags_goal_blocked(self):
        hits, add = expansion_candidates((0, 1), self.s_t, self.goals)
        self.assertEqual([(h.kind, h.group_agent, h.outside_agent) for h in hits],
                         [("GOAL_BLOCKED", 0, 2)])
        self.assertEqual(add, (2,))
        dist = bfs_dist(self.aug, self.goals[0])
        self.assertEqual(int(dist[self.s_t[0]]), -1)

    def test_runtime_returns_input_error_without_calling_solver(self):
        with _no_solver() as (tmp, solver):
            r = run_cbs_runtime({0: self.s_t[0], 1: self.s_t[1]},
                                {0: self.goals[0], 1: self.goals[1]},
                                self.aug, timeout_sec=1, solver_root=NONEXISTENT_SOLVER)
            wrote = list(tmp.rglob("*"))
        self.assertEqual(r.status, "input_error")
        self.assertIn("goal (2, 5) is blocked", r.error)
        self.assertIsNone(r.paths)
        solver.assert_not_called()
        self.assertEqual(wrote, [])            # rejected before any YAML is written

    def test_control_valid_input_reaches_solver_boundary(self):
        # Same group on the unmarked grid: validation passes, so the call gets
        # as far as the (patched) solver. Shows input_error above is the gate.
        with _no_solver() as (tmp, solver):
            r = run_cbs_runtime({0: self.s_t[0], 1: self.s_t[1]},
                                {0: self.goals[0], 1: self.goals[1]},
                                self.grid, timeout_sec=1, solver_root=NONEXISTENT_SOLVER)
        self.assertEqual(solver.call_count, 1)
        self.assertEqual(r.status, "solver_error")

    def test_walled_goal_is_free_but_unreachable(self):
        # Goal cell free, only access cell held by outside agent 2. Adapter
        # validation would PASS (goal not blocked), so only a BFS precheck
        # catches it. What the real solver then does (timeout vs no_solution)
        # is NOT verified here.
        grid = np.zeros((1, 5), dtype=np.int64)          # 1x5 corridor
        s_t = {0: (0, 0), 1: (0, 2)}                      # outside 1 in the middle
        goals = {0: (0, 4), 1: (0, 2)}
        aug = _mark_outside(grid, s_t, (1,))
        self.assertEqual(int(bfs_dist(grid, goals[0])[s_t[0]]), 4)          # reachable before
        self.assertEqual(int(bfs_dist(aug, goals[0])[s_t[0]]), -1)          # walled off after
        self.assertEqual(int(aug[goals[0]]), 0)                              # goal itself free
        hits, _ = expansion_candidates((0,), {0: s_t[0], 1: s_t[1]}, goals)
        self.assertEqual(hits, ())                                           # no GOAL_BLOCKED


# -- 4. outside WAIT vs Path Guard's all-agent goal check -----------------------

class TestPathGuardGoalCheckConflictsWithOutsideWait(unittest.TestCase):

    def setUp(self):
        self.grid = _open()
        self.s_t = {0: (0, 1), 1: (0, 3), 2: (1, 5)}
        self.goals = {0: (1, 2), 1: (0, 1), 2: (2, 0)}
        self.group = {0: [(0, 1), (1, 1), (1, 2)], 1: [(0, 3), (0, 2), (0, 1)]}

    def test_group_only_path_guard_passes(self):
        validate_path_guard(self.group, {0: self.s_t[0], 1: self.s_t[1]},
                            {0: self.goals[0], 1: self.goals[1]}, self.grid)

    def test_full_plan_passes_full_check_but_fails_path_guard_goal(self):
        r = validate_full_robot(self.group, self.s_t, [0, 1, 2], self.grid)
        self.assertTrue(r.ok, r.reason)
        self.assertEqual(r.full_paths[2], [(1, 5)] * 3)
        with self.assertRaises(InvalidPathError) as ctx:
            validate_path_guard(r.full_paths, self.s_t, self.goals, self.grid)
        self.assertIn("Agent 2: incorrect goal position", str(ctx.exception))
        ok, why = PathGuardValidator()(self.grid, self.s_t, self.goals, r.full_paths)
        self.assertFalse(ok)
        out = accept_replan_result(
            MockReplanResult("success", r.full_paths, 0.0), agent_order=(0, 1, 2),
            current_positions=self.s_t, goals=self.goals, grid=self.grid,
            path_validator=PathGuardValidator(), paths_keying="agent_id")
        self.assertFalse(out.accepted)
        self.assertTrue(out.reason.startswith("VALIDATOR_REJECT:"), out.reason)
        self.assertIn("incorrect goal position", out.reason)

    def test_rejection_is_only_the_goal_endpoint(self):
        # Same composed plan, but the outside agent's goal is its S_t cell:
        # Path Guard now accepts, so the conflict is exactly the goal check.
        r = validate_full_robot(self.group, self.s_t, [0, 1, 2], self.grid)
        validate_path_guard(r.full_paths, self.s_t, {**self.goals, 2: self.s_t[2]}, self.grid)


# -- 5. Runner/Simulator need an action for EVERY agent -------------------------

class TestEveryAgentNeedsAnAction(unittest.TestCase):

    def test_simulator_raises_on_missing_agent_action(self):
        sim = MAPFStepSimulator(cbs_solver_root=None, max_steps=10)
        sim.reset(_open(), STARTS, GOALS)
        with self.assertRaises(KeyError):
            sim.step({0: WAIT_ACTION, 1: WAIT_ACTION})          # agent 2 missing

    def test_boundary_refuses_group_only_paths_for_full_agent_order(self):
        s_t = {0: (0, 1), 1: (0, 3), 2: (1, 5)}
        group = {0: [(0, 1), (1, 1)], 1: [(0, 3), (0, 2)]}
        out = accept_replan_result(
            MockReplanResult("success", group, 0.0), agent_order=(0, 1, 2),
            current_positions=s_t, goals=GOALS, grid=_open(),
            path_validator=lambda *a: (True, ""), paths_keying="agent_id")
        self.assertEqual(out.reason, "BOUNDARY_PATH_COUNT")

    def test_composed_plan_yields_wait_for_outside_and_executes_cleanly(self):
        grid = _open()
        s_t = {0: (0, 1), 1: (0, 3), 2: (1, 5)}
        goals = {0: (1, 2), 1: (0, 1), 2: (2, 0)}
        r = validate_full_robot({0: [(0, 1), (1, 1), (1, 2)], 1: [(0, 3), (0, 2), (0, 1)]},
                                s_t, [0, 1, 2], grid)
        self.assertTrue(r.ok, r.reason)

        def full_check(g, st, gl, paths):     # test-local stand-in, NOT a proposed validator
            v = validate_full_robot(paths, st, sorted(st), g)
            return v.ok, v.reason

        out = accept_replan_result(
            MockReplanResult("success", r.full_paths, 0.0), agent_order=(0, 1, 2),
            current_positions=s_t, goals=goals, grid=grid,
            path_validator=full_check, paths_keying="agent_id")
        self.assertTrue(out.accepted, out.reason)
        for step_actions in out.plan.actions:
            self.assertEqual(set(step_actions), {0, 1, 2})
            self.assertEqual(step_actions[2], WAIT_ACTION)

        sim = MAPFStepSimulator(cbs_solver_root=None, max_steps=10)
        sim.reset(grid, s_t, goals)
        for k, step_actions in enumerate(out.plan.actions):
            sim.step(dict(step_actions))
            self.assertEqual({a: tuple(p) for a, p in sim._positions.items()}, out.plan.expected[k])
            self.assertEqual(set(sim._last_conflict_types.values()), {"none"})


if __name__ == "__main__":
    unittest.main()
