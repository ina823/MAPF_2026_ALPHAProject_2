"""I12-3: Selective full-agent plan composition + pre-execution check (Mock).

Module under test: src/hybrid/selective_compose.py (new, not wired anywhere).
Group paths below are MOCK group-CBS answers -- no solver is called and a
pass here is NOT evidence that a real subset CBS call succeeds (that, and the
outside-agent WAIT policy itself, remain PENDING P1-P5).

Real, unmodified code exercised: Path Guard (endpoints / movements /
map-and-conflicts / validate_path_guard), replan_boundary.accept_replan_result,
selective_safety.validate_full_robot / exec_mismatches, MAPFStepSimulator.
"""

import ast
import inspect
import unittest
from pathlib import Path

import numpy as np

from src.hybrid import selective_compose as sc
from src.hybrid.cbs_runtime_bridge import PathGuardValidator
from src.hybrid.replan_boundary import accept_replan_result, check_validator_pairing
from src.hybrid.selective_compose import (
    GROUP_REACHES_GOAL,
    OUTSIDE_HELD,
    SelectiveFullPathValidator,
    compose_selective_plan,
    exec_step_mismatches,
    selective_boundary_check,
)
from src.hybrid.selective_decision import PENDING_BLOCKERS
from src.hybrid.selective_safety import LADDER_UNTIL_OUTSIDE_POLICY_VERIFIED, exec_mismatches
from src.path_guard import InvalidPathError, validate_path_guard
from src.simulator.mapf_step_simulator import MAPFStepSimulator
from tests.hybrid_mocks import WAIT, MockReplanResult, perturbing_simulator

ROOT = Path(__file__).resolve().parents[1]

#   col 0 1 2 3 4 5
#   r0  . . . . . .
#   r1  . # . . . .
#   r2  . . . . . .
GRID = np.zeros((3, 6), dtype=np.int64)
GRID[1, 1] = 1

# group {3, 12}, outside {7, 40}; ids non-contiguous on purpose
S_T = {3: (0, 0), 12: (2, 0), 7: (1, 3), 40: (2, 5)}
GOALS = {3: (0, 3), 12: (2, 2), 7: (1, 0), 40: (2, 5)}     # 7 off goal, 40 already on goal
GROUP = {3: [(0, 0), (0, 1), (0, 2), (0, 3)],               # 3 moves
         12: [(2, 0), (2, 1), (2, 2)]}                      # shorter -> padded with its goal
IDS = sorted(S_T)


def compose(group=None, s_t=None, goals=None, ids=None, grid=None):
    return compose_selective_plan(GROUP if group is None else group, S_T if s_t is None else s_t,
                                  GOALS if goals is None else goals, IDS if ids is None else ids,
                                  GRID if grid is None else grid)


def boundary(group=None, status="success", s_t=None, goals=None, latency=0.05):
    res = MockReplanResult(status=status, paths=GROUP if group is None else group, latency_sec=latency)
    return selective_boundary_check(res, s_t=S_T if s_t is None else s_t,
                                    goals=GOALS if goals is None else goals, grid=GRID)


# -- 1. agent IDs ---------------------------------------------------------------------

class TestAgentIds(unittest.TestCase):

    def test_every_runtime_id_kept_none_added_none_renumbered(self):
        p = compose()
        self.assertTrue(p.ok, p.reason)
        self.assertEqual(set(p.full_paths), set(S_T))
        self.assertEqual((p.group, p.outside), ((3, 12), (7, 40)))
        self.assertEqual(set(p.goal_status), set(S_T))

    def test_agent_set_mismatches_are_refused(self):
        cases = {
            "group agent unknown": dict(group={**GROUP, 99: [(1, 5)]}),
            "s_t missing agent": dict(s_t={a: c for a, c in S_T.items() if a != 7}),
            "goals missing agent": dict(goals={a: c for a, c in GOALS.items() if a != 40}),
            "agent_ids missing": dict(ids=[3, 12, 7]),
            "agent_ids extra": dict(ids=IDS + [41]),
            "agent_ids duplicate": dict(ids=[3, 3, 12, 7, 40]),
            "agent_ids bool": dict(ids=[True, 3, 12, 7, 40]),
            "empty group": dict(group={}),
            "group key str": dict(group={"3": GROUP[3], 12: GROUP[12]}),
        }
        for name, kw in cases.items():
            with self.subTest(name):
                p = compose(**kw)
                self.assertFalse(p.ok)
                self.assertTrue(p.reason.startswith(sc.R_AGENT_SET), p.reason)
                self.assertIsNone(p.full_paths)

    def test_s_t_overlap_refused(self):
        p = compose(s_t={**S_T, 7: (2, 5)})
        self.assertEqual((p.ok, p.reason), (False, sc.R_S_T_OVERLAP))


# -- 2. Path[0] = S_t and group goal rule ---------------------------------------------

class TestStartAndGoal(unittest.TestCase):

    def test_every_path_starts_at_s_t(self):
        p = compose()
        for a, path in p.full_paths.items():
            self.assertEqual(path[0], S_T[a])

    def test_group_start_mismatch_refused(self):
        p = compose(group={**GROUP, 3: [(0, 1), (0, 2), (0, 3)]})
        self.assertFalse(p.ok)
        self.assertEqual(p.reason, f"{sc.R_GROUP_PATH}:Agent 3: incorrect start position")

    def test_group_must_reach_its_goal_existing_rule(self):
        p = compose(group={**GROUP, 3: [(0, 0), (0, 1), (0, 2)]})
        self.assertFalse(p.ok)
        self.assertEqual(p.reason, f"{sc.R_GROUP_PATH}:Agent 3: incorrect goal position")

    def test_outside_agent_is_never_reported_at_goal(self):
        p = compose()
        self.assertEqual(p.goal_status, {3: GROUP_REACHES_GOAL, 12: GROUP_REACHES_GOAL,
                                         7: OUTSIDE_HELD, 40: OUTSIDE_HELD})
        self.assertNotIn(GROUP_REACHES_GOAL, {p.goal_status[7], p.goal_status[40]})
        self.assertNotEqual(p.full_paths[7][-1], GOALS[7])
        self.assertFalse(p.ends_all_at_goal)           # 7 is held off its goal

    def test_ends_all_at_goal_only_when_outside_already_on_goal(self):
        s_t, goals = {3: (0, 0), 40: (2, 5)}, {3: (0, 3), 40: (2, 5)}
        p = compose_selective_plan({3: GROUP[3]}, s_t, goals, [3, 40], GRID)
        self.assertTrue(p.ok, p.reason)
        self.assertEqual(p.goal_status[40], OUTSIDE_HELD)  # still "held", not "reached"
        self.assertTrue(p.ends_all_at_goal)

    def test_malformed_group_paths_refused(self):
        for bad in ([], "ab", [(0, 0, 0)], [(0.0, 0)], [(True, 0)], None):
            with self.subTest(bad=bad):
                p = compose(group={**GROUP, 3: bad})
                self.assertFalse(p.ok)
                self.assertTrue(p.reason.startswith(sc.R_GROUP_PATH), p.reason)


# -- 3-5. obstacles, conflicts, length --------------------------------------------------

class TestMapConflictsLength(unittest.TestCase):

    def test_group_path_through_static_obstacle(self):
        p = compose(group={3: [(0, 0), (0, 1), (1, 1), (0, 1), (0, 2), (0, 3)], 12: GROUP[12]})
        self.assertFalse(p.ok)
        self.assertIn("obstacle at (1, 1)", p.reason)

    def test_outside_agent_on_static_obstacle(self):
        p = compose(s_t={**S_T, 7: (1, 1)})
        self.assertFalse(p.ok)
        self.assertIn("Agent 7, t=0: obstacle", p.reason)

    def test_out_of_bounds(self):
        p = compose(group={3: [(0, 0), (-1, 0), (0, 0), (0, 1), (0, 2), (0, 3)], 12: GROUP[12]})
        self.assertFalse(p.ok)
        self.assertIn("out of bounds", p.reason)

    def test_non_adjacent_move(self):
        p = compose(group={3: [(0, 0), (0, 2), (0, 3)], 12: GROUP[12]})
        self.assertFalse(p.ok)
        self.assertIn("illegal movement", p.reason)

    def test_group_into_held_outside_agent_vertex(self):
        """Group goal behind an outside agent: 7 holds (1,3), 3 must pass it."""
        goals = {**GOALS, 3: (1, 4)}
        path = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (1, 4)]
        p = compose(group={3: path, 12: GROUP[12]}, goals=goals)
        self.assertFalse(p.ok)
        self.assertIn("Vertex conflict at t=4: Agent 3 and 7", p.reason)

    def test_group_goal_occupied_by_outside_agent(self):
        goals = {**GOALS, 3: (1, 3)}                     # 7 stands there at S_t
        p = compose(group={3: [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3)], 12: GROUP[12]}, goals=goals)
        self.assertFalse(p.ok)
        self.assertIn("Vertex conflict", p.reason)
        self.assertIn("Agent 3 and 7", p.reason)

    def test_group_internal_vertex(self):
        s_t = {**S_T, 12: (0, 2)}
        goals = {**GOALS, 12: (0, 1)}
        p = compose(group={3: GROUP[3], 12: [(0, 2), (0, 1)]}, s_t=s_t, goals=goals)
        self.assertFalse(p.ok)
        self.assertIn("Vertex conflict at t=1", p.reason)

    def test_group_internal_edge_swap(self):
        s_t = {**S_T, 3: (0, 2), 12: (0, 3)}
        goals = {**GOALS, 3: (0, 3), 12: (0, 2)}
        p = compose(group={3: [(0, 2), (0, 3)], 12: [(0, 3), (0, 2)]}, s_t=s_t, goals=goals)
        self.assertFalse(p.ok)
        self.assertIn("Edge conflict at t=0->1", p.reason)

    def test_equal_length_padding(self):
        p = compose()
        self.assertEqual({len(path) for path in p.full_paths.values()}, {4})
        self.assertEqual(p.full_paths[12], [(2, 0), (2, 1), (2, 2), (2, 2)])   # padded with goal
        self.assertEqual(p.full_paths[7], [(1, 3)] * 4)
        self.assertEqual(p.full_paths[40], [(2, 5)] * 4)

    def test_padded_tail_conflict_is_caught(self):
        """12 parks at its goal (0,2) after t=1, 3 passes through (0,2) at t=2."""
        s_t = {**S_T, 12: (1, 2)}
        goals = {**GOALS, 12: (0, 2)}
        p = compose(group={3: GROUP[3], 12: [(1, 2), (0, 2)]}, s_t=s_t, goals=goals)
        self.assertFalse(p.ok)
        self.assertIn("Vertex conflict at t=2", p.reason)


# -- 6. the existing Path Guard is not weakened -------------------------------------------

class TestExistingPathGuardUnchanged(unittest.TestCase):

    def test_validate_path_guard_still_rejects_held_outside_agent(self):
        p = compose()
        with self.assertRaisesRegex(InvalidPathError, "Agent 7: incorrect goal position"):
            validate_path_guard(p.full_paths, S_T, GOALS, GRID)
        ok, why = PathGuardValidator()(GRID, S_T, GOALS, p.full_paths)
        self.assertFalse(ok)

    def test_module_does_not_call_or_patch_validate_path_guard(self):
        tree = ast.parse(inspect.getsource(sc))
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                    and n.module == "src.path_guard" for a in n.names}
        self.assertEqual(imported, {"InvalidPathError", "validate_path_endpoints", "validate_path_movements"})
        called = {getattr(n.func, "id", getattr(n.func, "attr", None))
                  for n in ast.walk(tree) if isinstance(n, ast.Call)}
        self.assertNotIn("validate_path_guard", called)
        foreign = [t for n in ast.walk(tree) if isinstance(n, (ast.Assign, ast.AugAssign))
                   for t in getattr(n, "targets", [getattr(n, "target", None)])
                   if isinstance(t, ast.Attribute) and getattr(t.value, "id", None) != "self"]
        self.assertEqual(foreign, [])                  # no monkeypatching of other modules

    def test_selective_validator_is_not_test_only(self):
        v = SelectiveFullPathValidator((3, 12))
        self.assertFalse(getattr(v, "test_only", False))
        check_validator_pairing(object(), v)            # usable with a real planner


# -- 7. refusals before execution, never repaired ---------------------------------------

class TestBoundaryRefusals(unittest.TestCase):

    def test_success_goes_through_existing_boundary(self):
        outcome, plan = boundary()
        self.assertTrue(outcome.accepted, outcome.reason)
        ap = outcome.plan
        self.assertEqual(ap.agent_order, tuple(IDS))
        self.assertEqual(ap.paths, plan.full_paths)
        self.assertEqual(len(ap.actions), 3)
        for k, step in enumerate(ap.actions):
            self.assertEqual(step[7], WAIT, k)
            self.assertEqual(step[40], WAIT, k)
            self.assertEqual(set(step), set(IDS))
        self.assertAlmostEqual(ap.latency_ms, 50.0)

    def test_non_success_group_status_refused(self):
        for status in ("timeout", "no_solution", "invalid_path", "input_error", "solver_error"):
            with self.subTest(status):
                outcome, plan = boundary(status=status)
                self.assertEqual((outcome.accepted, outcome.reason, outcome.plan, plan),
                                 (False, f"CBS_{status.upper()}", None, None))
        outcome, _ = boundary(status="weird")
        self.assertEqual((outcome.accepted, outcome.reason), (False, "CBS_UNKNOWN_STATUS"))

    def test_index_keyed_group_paths_refused(self):
        outcome, _ = boundary(group=[GROUP[3], GROUP[12]])
        self.assertEqual((outcome.accepted, outcome.plan), (False, None))
        self.assertTrue(outcome.reason.startswith(sc.R_GROUP_RESULT))

    def test_every_compose_failure_is_a_boundary_refusal(self):
        for name, group in {"wall": {3: [(0, 0), (0, 1), (1, 1)], 12: GROUP[12]},
                            "goal": {3: [(0, 0), (0, 1)], 12: GROUP[12]},
                            "unknown agent": {**GROUP, 99: [(0, 5)]}}.items():
            with self.subTest(name):
                outcome, plan = boundary(group=group)
                self.assertFalse(outcome.accepted)
                self.assertIsNone(outcome.plan)
                self.assertFalse(plan.ok)
                self.assertEqual(outcome.reason, plan.reason)

    def test_validator_rejects_tampered_full_paths(self):
        p = compose()
        v = SelectiveFullPathValidator(p.group)
        self.assertEqual(v(GRID, S_T, GOALS, p.full_paths), (True, ""))
        moved = {**p.full_paths, 7: [(1, 3), (1, 2), (1, 2), (1, 2)]}
        self.assertEqual(v(GRID, S_T, GOALS, moved), (False, f"{sc.R_OUTSIDE_NOT_HELD}:7"))
        missing = {a: q for a, q in p.full_paths.items() if a != 40}
        self.assertFalse(v(GRID, S_T, GOALS, missing)[0])
        # Every agent one WAIT longer is still a consistent composition -> accepted;
        # only the outside agents longer than the group horizon is not.
        all_longer = {a: q + [q[-1]] for a, q in p.full_paths.items()}
        self.assertEqual(v(GRID, S_T, GOALS, all_longer), (True, ""))
        outside_longer = {a: (q + [q[-1]] if a in p.outside else q) for a, q in p.full_paths.items()}
        self.assertEqual(v(GRID, S_T, GOALS, outside_longer), (False, sc.R_PATHS_MISMATCH))
        short_goal = {**p.full_paths, 3: [(0, 0), (0, 1), (0, 2), (0, 2)]}
        self.assertFalse(v(GRID, S_T, GOALS, short_goal)[0])
        self.assertEqual(v(GRID, S_T, GOALS, "nope"), (False, f"{sc.R_AGENT_SET}:paths"))

    def test_boundary_with_selective_validator_refuses_outside_motion(self):
        """Even straight through accept_replan_result, a plan moving an outside agent is refused."""
        full = {**compose().full_paths, 7: [(1, 3), (1, 4), (1, 4), (1, 4)]}
        out = accept_replan_result(MockReplanResult("success", full, 0.01), agent_order=tuple(IDS),
                                   current_positions=S_T, goals=GOALS, grid=GRID,
                                   path_validator=SelectiveFullPathValidator((3, 12)),
                                   paths_keying="agent_id")
        self.assertEqual((out.accepted, out.reason), (False, f"VALIDATOR_REJECT:{sc.R_OUTSIDE_NOT_HELD}:7"))

    def test_garbage_never_raises(self):
        for args in ((None, S_T, GOALS, IDS, GRID), (GROUP, None, GOALS, IDS, GRID),
                     (GROUP, S_T, None, IDS, GRID), (GROUP, S_T, GOALS, None, GRID),
                     (GROUP, S_T, GOALS, "3", GRID), (GROUP, S_T, GOALS, IDS, None),
                     (GROUP, S_T, GOALS, IDS, [[0]])):
            p = compose_selective_plan(*args)
            self.assertFalse(p.ok)
            self.assertIsNone(p.full_paths)


# -- 8. execution window expected / actual ------------------------------------------------

class TestExecution(unittest.TestCase):

    def _run(self, sim_cls, h):
        outcome, _ = boundary()
        ap = outcome.plan
        sim = sim_cls(cbs_solver_root=None, max_steps=50)
        sim.reset(GRID, dict(S_T), dict(GOALS))
        trace = []
        for k in range(1, h + 1):
            sim.step(dict(ap.actions[k - 1]))
            actual = {a: tuple(sim._positions[a]) for a in sim._agent_ids}
            bad = exec_step_mismatches(ap, k, actual)
            self.assertEqual(bad, exec_mismatches(ap.paths, k, actual))   # same verdict as selective_safety
            trace.append((k, actual, bad))
            if bad:
                break
        return ap, trace

    def test_real_simulator_follows_expected_for_every_agent(self):
        ap, trace = self._run(MAPFStepSimulator, 3)
        for k, actual, bad in trace:
            self.assertEqual(bad, ())
            self.assertEqual(actual, ap.expected[k - 1])
            self.assertEqual((actual[7], actual[40]), (S_T[7], S_T[40]))   # outside never moved

    def test_h_exec_window_prefix(self):
        ap, trace = self._run(MAPFStepSimulator, 2)
        self.assertEqual(len(trace), 2)
        self.assertEqual(trace[-1][1], ap.expected[1])

    def test_outside_agent_displaced_is_detected(self):
        _, trace = self._run(perturbing_simulator(1, 7, (1, 2)), 3)
        self.assertEqual(len(trace), 1)                  # stops before the next move
        self.assertEqual(trace[0][2], (7,))

    def test_group_agent_displaced_is_detected(self):
        _, trace = self._run(perturbing_simulator(2, 12, (2, 0)), 3)
        self.assertEqual([k for k, _, bad in trace if bad], [2])
        self.assertEqual(trace[-1][2], (12,))

    def test_missing_extra_and_bad_k(self):
        ap = boundary()[0].plan
        exp = dict(ap.expected[0])
        self.assertEqual(exec_step_mismatches(ap, 1, exp), ())
        self.assertEqual(exec_step_mismatches(ap, 1, {a: c for a, c in exp.items() if a != 40}), (40,))
        self.assertEqual(exec_step_mismatches(ap, 1, {**exp, 99: (0, 5)}), (99,))
        for k in (0, 4, -1, True, 1.0):
            with self.assertRaises(ValueError):
                exec_step_mismatches(ap, k, exp)


# -- 9. scope: not wired, nothing agreed by this module ------------------------------------

class TestScope(unittest.TestCase):

    def test_not_imported_by_runtime(self):
        for rel in ("src/hybrid/global_hybrid.py", "src/compare/common_runner.py",
                    "src/hybrid/selective_shadow.py", "src/hybrid/cbs_runtime_bridge.py"):
            self.assertNotIn("selective_compose", (ROOT / rel).read_text(encoding="utf-8"), rel)
        for script in (ROOT / "scripts").glob("*.py"):
            self.assertNotIn("selective_compose", script.read_text(encoding="utf-8"), script.name)

    def test_pending_policies_untouched(self):
        self.assertEqual(len(PENDING_BLOCKERS), 5)
        self.assertEqual(LADDER_UNTIL_OUTSIDE_POLICY_VERIFIED, ("G",))


if __name__ == "__main__":
    unittest.main()
