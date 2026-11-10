"""Mock safety tests for Selective Hybrid v0 (src/hybrid/selective_safety.py).

No CBS, no solver, no file output. WFG groups come from the real
``build_wfg``; routing from the real ``decide_selective``; full-robot checks
from the real (unmodified) Path Guard functions. Group paths and CBS attempts
are hand-written MOCKS. Budget numbers below are TEST-ONLY values, not agreed
defaults.

Sections map to the request's items 1-10. Policies that are not agreed are
``@unittest.skip("PENDING: ...")`` and assert nothing.
"""

import ast
import dataclasses
import inspect
import unittest
from pathlib import Path

from src.hybrid import selective_safety as ss
from src.hybrid.selective_decision import ROUTE_GLOBAL, SELECTIVE_CANDIDATE_BLOCKED, decide_selective
from src.hybrid.selective_safety import (
    ACTION_CALL,
    ACTION_EXECUTE,
    ACTION_SAFE_STOP,
    LADDER_UNTIL_OUTSIDE_POLICY_VERIFIED,
    Attempt,
    ExpansionHit,
    TriggerBudget,
    exec_mismatches,
    expansion_candidates,
    l0_union_group,
    next_ladder_step,
    validate_full_robot,
)
from src.hybrid.wfg import build_wfg
from src.path_guard import validate_path_map_and_conflicts
from tests.test_selective_decision import OPEN
from tests.test_wfg import CASES, _grid

ROOT = Path(__file__).resolve().parents[1]

GRID5 = _grid("""
.......
.......
.......
.......
.......
""")


def _pad(paths):
    n = max(len(p) for p in paths.values())
    return {a: list(p) + [p[-1]] * (n - len(p)) for a, p in paths.items()}


def _two_groups_plus_outside():
    """SYNTHETIC (OPEN 3x7). Step t: 0/1 contest (0,2), 2/3 contest (2,2),
    agent 4 moves freely (1,6)->(1,5). WFG uses S_{t-1}=cur + intended (P1);
    planning uses S_t, where only agent 4 has moved."""
    cur = {0: (0, 1), 1: (0, 3), 2: (2, 1), 3: (2, 3), 4: (1, 6)}
    want = {0: (0, 2), 1: (0, 2), 2: (2, 2), 3: (2, 2), 4: (1, 5)}
    s_t = {**cur, 4: (1, 5)}
    goals = {0: (0, 6), 1: (0, 0), 2: (2, 6), 3: (2, 0), 4: (1, 0)}
    wfg = build_wfg(cur, want, {0, 1, 2, 3}, OPEN)
    return cur, s_t, goals, wfg


def _budget(total=1000.0, attempts=3, **levels):
    # TEST-ONLY numbers. The real allocation is an OPEN decision.
    return TriggerBudget(total_time_ms=total, max_attempts=attempts,
                         level_timeout_ms=levels or {"L0": 800.0, "L1": 800.0, "G": 800.0})


LADDER = ("L0", "L1", "G")


# -- 1. two independent groups -> union ---------------------------------------

class TestUnionOfIndependentGroups(unittest.TestCase):

    def test_bottleneck_t2_union_equals_all_agents(self):
        c = CASES["bottleneck_t2"]
        wfg = build_wfg(c["current"], c["intended"], c["blocked"], c["grid"])
        self.assertEqual(wfg.groups, ((0, 1), (2, 3)))
        # Union of the two independent groups is every agent: L0 == G input.
        self.assertEqual(l0_union_group(wfg.groups), (0, 1, 2, 3))

    def test_synthetic_union_leaves_outside_agent(self):
        cur, s_t, goals, wfg = _two_groups_plus_outside()
        self.assertEqual(wfg.groups, ((0, 1), (2, 3)))
        self.assertEqual(l0_union_group(wfg.groups), (0, 1, 2, 3))
        d = decide_selective(sorted(s_t), s_t, goals, wfg)
        # Selection is not execution: the decision module still routes GLOBAL.
        self.assertEqual(d.route, ROUTE_GLOBAL)
        self.assertEqual(d.reason, SELECTIVE_CANDIDATE_BLOCKED)
        self.assertEqual(d.outside_agents, (4,))

    def test_union_plan_checked_with_outside_wait(self):
        cur, s_t, goals, wfg = _two_groups_plus_outside()
        paths = {0: [(0, 1)], 1: [(0, 3), (0, 2)], 2: [(2, 1)], 3: [(2, 3), (2, 2)]}
        r = validate_full_robot(paths, s_t, sorted(s_t), OPEN)
        self.assertTrue(r.ok, r.reason)
        # Outside agent 4 is held at S_t (not S_{t-1}) for the whole horizon.
        self.assertEqual(r.full_paths[4], [(1, 5), (1, 5)])
        self.assertEqual(r.full_paths[0], [(0, 1), (0, 1)])

    def test_union_rejects_non_int_ids(self):
        with self.assertRaises(ValueError):
            l0_union_group(((0, "1"),))


# -- 2. outside agent occupies a group goal ------------------------------------

class TestOutsideOccupiesGroupGoal(unittest.TestCase):

    def setUp(self):
        # SYNTHETIC: {0,1} contest (0,2); outside 2 moved (2,6)->(2,5) at step t.
        self.cur = {0: (0, 1), 1: (0, 3), 2: (2, 6)}
        self.want = {0: (0, 2), 1: (0, 2), 2: (2, 5)}
        self.s_t = {0: (0, 1), 1: (0, 3), 2: (2, 5)}
        self.goals = {0: (2, 5), 1: (0, 0), 2: (2, 0)}   # 0's goal = 2's S_t cell
        self.wfg = build_wfg(self.cur, self.want, {0, 1}, OPEN)
        self.paths = {
            0: [(0, 1), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 5)],
            1: [(0, 3), (0, 2), (0, 1), (0, 0)],
        }

    def test_goal_blocker_is_an_expansion_candidate(self):
        hits, add = expansion_candidates((0, 1), self.s_t, self.goals)
        self.assertEqual(hits, (ExpansionHit("GOAL_BLOCKED", 0, 2, (2, 5)),))
        self.assertEqual(add, (2,))

    def test_decision_module_reports_same_fact(self):
        d = decide_selective([0, 1, 2], self.s_t, self.goals, self.wfg)
        self.assertEqual(d.route, ROUTE_GLOBAL)
        self.assertIn(("OUTSIDE_OCCUPIES_GROUP_GOAL", 0, 2), d.findings)

    def test_full_check_rejects_plan_into_held_agent(self):
        group_only = _pad(self.paths)
        validate_path_map_and_conflicts(group_only, OPEN)   # group-only check passes
        r = validate_full_robot(self.paths, self.s_t, [0, 1, 2], OPEN)
        self.assertFalse(r.ok)
        self.assertTrue(r.reason.startswith("FULL_CHECK_REJECT:Vertex conflict at t=6"), r.reason)
        self.assertIn("Agent 0 and 2", r.reason)


# -- 3. group path crosses an outside agent's static position ------------------

class TestGroupPathThroughOutsideAgent(unittest.TestCase):

    def test_path_blocker_detected_and_full_check_rejects(self):
        s_t = {0: (0, 1), 1: (0, 3), 2: (1, 3)}
        goals = {0: (1, 6), 1: (0, 0), 2: (2, 0)}
        paths = {
            0: [(0, 1), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6)],
            1: [(0, 3), (0, 2), (0, 1), (0, 0)],
        }
        hits, add = expansion_candidates((0, 1), s_t, goals, paths)
        self.assertEqual(hits, (ExpansionHit("PATH_BLOCKED", 0, 2, (1, 3)),))
        self.assertEqual(add, (2,))
        validate_path_map_and_conflicts(_pad(paths), OPEN)   # group-only passes
        r = validate_full_robot(paths, s_t, [0, 1, 2], OPEN)
        self.assertFalse(r.ok)
        self.assertIn("Vertex conflict at t=3", r.reason)
        self.assertIn("Agent 0 and 2", r.reason)

    def test_no_hit_when_path_avoids_outside(self):
        s_t = {0: (0, 1), 1: (0, 3), 2: (2, 6)}
        goals = {0: (0, 6), 1: (0, 0), 2: (2, 0)}
        hits, add = expansion_candidates((0, 1), s_t, goals, {0: [(0, 1), (1, 1)]})
        self.assertEqual((hits, add), ((), ()))


# -- 4. vertex / edge conflict between groups ---------------------------------

class TestCrossGroupConflicts(unittest.TestCase):

    def test_cross_group_vertex(self):
        cur, s_t, goals, wfg = _two_groups_plus_outside()
        a = {0: [(0, 1)], 1: [(0, 3), (1, 3)]}
        b = {2: [(2, 1)], 3: [(2, 3), (1, 3)]}
        validate_path_map_and_conflicts(_pad(a), OPEN)
        validate_path_map_and_conflicts(_pad(b), OPEN)
        r = validate_full_robot({**a, **b}, s_t, sorted(s_t), OPEN)
        self.assertFalse(r.ok)
        self.assertIn("Vertex conflict at t=1: Agent 1 and 3", r.reason)

    def test_cross_group_edge(self):
        # SYNTHETIC: groups {0,1} and {2,3}; 1 sits right above 2.
        cur = {0: (0, 0), 1: (0, 2), 2: (1, 2), 3: (1, 4)}
        want = {0: (0, 1), 1: (0, 1), 2: (1, 3), 3: (1, 3)}
        wfg = build_wfg(cur, want, {0, 1, 2, 3}, OPEN)
        self.assertEqual(wfg.groups, ((0, 1), (2, 3)))
        a = {0: [(0, 0), (0, 0)], 1: [(0, 2), (1, 2)]}
        b = {2: [(1, 2), (0, 2)], 3: [(1, 4), (1, 4)]}
        validate_path_map_and_conflicts(a, OPEN)
        validate_path_map_and_conflicts(b, OPEN)
        r = validate_full_robot({**a, **b}, cur, [0, 1, 2, 3], OPEN)
        self.assertFalse(r.ok)
        self.assertIn("Edge conflict at t=0->1: Agent 1 and 2", r.reason)


# -- 5. Path[0] vs actual S_t ---------------------------------------------------

class TestStartStateIsSt(unittest.TestCase):

    def test_path_from_s_t_minus_1_is_rejected(self):
        cur, s_t, goals, wfg = _two_groups_plus_outside()
        # Expanded group includes mover 4, but its path starts at S_{t-1}.
        paths = {0: [(0, 1)], 1: [(0, 3)], 4: [cur[4], (1, 4)]}
        r = validate_full_robot(paths, s_t, sorted(s_t), OPEN)
        self.assertEqual((r.ok, r.reason), (False, "FULL_CHECK_START_MISMATCH:4"))
        paths[4] = [s_t[4], (1, 4)]
        self.assertTrue(validate_full_robot(paths, s_t, sorted(s_t), OPEN).ok)

    def test_exec_mismatch_covers_outside_agents(self):
        cur, s_t, goals, wfg = _two_groups_plus_outside()
        r = validate_full_robot({0: [(0, 1)], 1: [(0, 3), (0, 2)]}, s_t, sorted(s_t), OPEN)
        self.assertTrue(r.ok, r.reason)
        actual = {a: r.full_paths[a][1] for a in r.full_paths}
        self.assertEqual(exec_mismatches(r.full_paths, 1, actual), ())
        actual[4] = (1, 4)            # outside agent drifted during execution
        self.assertEqual(exec_mismatches(r.full_paths, 1, actual), (4,))


# -- 6. agent ID loss / coordinate confusion ----------------------------------

class TestAgentIdAndCoordinates(unittest.TestCase):

    def setUp(self):
        _, self.s_t, _, _ = _two_groups_plus_outside()
        self.paths = {0: [(0, 1)], 1: [(0, 3), (0, 2)]}

    def test_agent_set_mismatches(self):
        ids = sorted(self.s_t)
        no4 = {a: p for a, p in self.s_t.items() if a != 4}
        cases = {
            "FULL_CHECK_AGENT_SET:s_t": (self.paths, no4, ids),
            "FULL_CHECK_AGENT_SET:group_paths": ({**self.paths, 9: [(1, 1)]}, self.s_t, ids),
            "FULL_CHECK_AGENT_SET:agent_ids": (self.paths, self.s_t, ids + [ids[0]]),
        }
        for reason, args in cases.items():
            with self.subTest(reason):
                self.assertEqual(validate_full_robot(*args, OPEN).reason, reason)
        # agent_ids drops 4 while s_t keeps it: S_t no longer matches.
        self.assertEqual(validate_full_robot(self.paths, self.s_t, [0, 1, 2, 3], OPEN).reason,
                         "FULL_CHECK_AGENT_SET:s_t")

    def test_exec_mismatch_flags_lost_agent(self):
        r = validate_full_robot(self.paths, self.s_t, sorted(self.s_t), OPEN)
        actual = {a: r.full_paths[a][1] for a in r.full_paths if a != 3}
        self.assertEqual(exec_mismatches(r.full_paths, 1, actual), (3,))

    def test_xy_path_is_rejected_as_start_mismatch(self):
        xy = {a: [(c, r) for r, c in p] for a, p in self.paths.items()}   # [x,y] leak
        r = validate_full_robot(xy, self.s_t, sorted(self.s_t), OPEN)
        self.assertEqual((r.ok, r.reason), (False, "FULL_CHECK_START_MISMATCH:0"))

    def test_module_does_no_coordinate_conversion(self):
        src = inspect.getsource(ss)
        tree = ast.parse(src)
        imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertNotIn("src.cbs_adapter", imported)
        self.assertNotIn("src.common.scenario_loader", imported)
        for name in ("internal_to_atb033", "atb033_to_internal", "xy_to_rowcol"):
            self.assertNotIn(name, src)


# -- 7. timeout then insufficient budget --------------------------------------

class TestTimeoutAndBudget(unittest.TestCase):

    def test_timeout_consumes_all_time_then_safe_stop(self):
        at = [Attempt("L0", (0, 1), "timeout", 1000.0)]
        s = next_ladder_step(LADDER, at, _budget())
        self.assertEqual((s.action, s.reason), (ACTION_SAFE_STOP, "TIME_BUDGET_EXHAUSTED"))

    def test_next_timeout_never_exceeds_remaining(self):
        at = [Attempt("L0", (0, 1), "timeout", 700.0)]
        s = next_ladder_step(LADDER, at, _budget())
        self.assertEqual((s.action, s.level), (ACTION_CALL, "L1"))
        self.assertEqual(s.remaining_ms, 300.0)
        self.assertEqual(s.timeout_ms, 300.0)        # requested 800, clamped

    def test_failed_latency_is_counted_across_levels(self):
        at = [Attempt("L0", (0, 1), "timeout", 400.0), Attempt("L1", (0, 1, 2), "no_solution", 450.0)]
        s = next_ladder_step(LADDER, at, _budget())
        self.assertEqual((s.action, s.level, s.timeout_ms), (ACTION_CALL, "G", 150.0))

    def test_unknown_latency_means_no_time_left(self):
        at = [Attempt("L0", (0, 1), "timeout", None)]
        s = next_ladder_step(LADDER, at, _budget())
        self.assertEqual((s.action, s.reason), (ACTION_SAFE_STOP, "TIME_BUDGET_EXHAUSTED"))

    def test_attempt_limit_is_shared_by_levels(self):
        at = [Attempt("L0", (0, 1), "timeout", 10.0)]
        s = next_ladder_step(LADDER, at, _budget(attempts=1))
        self.assertEqual((s.action, s.reason), (ACTION_SAFE_STOP, "ATTEMPT_LIMIT"))

    def test_first_call_is_clamped_too(self):
        s = next_ladder_step(LADDER, [], _budget(total=500.0))
        self.assertEqual((s.action, s.level, s.timeout_ms), (ACTION_CALL, "L0", 500.0))

    def test_budget_has_no_defaults_and_rejects_bad_values(self):
        with self.assertRaises(TypeError):
            TriggerBudget()  # noqa: no default numbers exist
        for kw in ({"total_time_ms": 0}, {"max_attempts": 0}, {"total_time_ms": float("inf")}):
            with self.subTest(kw):
                with self.assertRaises(ValueError):
                    TriggerBudget(**{"total_time_ms": 1.0, "max_attempts": 1,
                                     "level_timeout_ms": {"G": 1.0}, **kw})

    def test_ladder_shape_is_enforced(self):
        for bad in ((), ("L0",), ("G", "L0"), ("L0", "L0", "G")):
            with self.subTest(bad):
                with self.assertRaises(ValueError):
                    next_ladder_step(bad, [], _budget())
        with self.assertRaises(ValueError):   # level without a requested timeout
            next_ladder_step(("L0", "G"), [], _budget(G=1.0))

    @unittest.skip("PENDING OPEN_DECISION: per-level time split (L0/L1/G ms), total per-Trigger "
                   "budget and max_attempts values; and how this per-Trigger ledger relates to "
                   "the existing episode-level MonitorRearmConfig budget")
    def test_pending_budget_numbers(self):
        pass

    @unittest.skip("PENDING OPEN_DECISION: minimum useful timeout -- whether a tiny remaining "
                   "time should still be spent on a call or end in SAFE_STOP / jump to G")
    def test_pending_minimum_timeout(self):
        pass


# -- 8. no_solution -> expand or Global ----------------------------------------

class TestNoSolutionLadder(unittest.TestCase):

    def test_walks_ladder_then_stops_after_global(self):
        b = _budget(total=10000.0, attempts=5)
        at = [Attempt("L0", (0, 1), "no_solution", 100.0)]
        s = next_ladder_step(LADDER, at, b)
        self.assertEqual((s.action, s.level, s.reason), (ACTION_CALL, "L1", "AFTER_CBS_NO_SOLUTION"))
        at.append(Attempt("L1", (0, 1, 2), "no_solution", 100.0))
        s = next_ladder_step(LADDER, at, b)
        self.assertEqual((s.action, s.level), (ACTION_CALL, "G"))
        at.append(Attempt("G", (0, 1, 2, 3), "no_solution", 100.0))
        s = next_ladder_step(LADDER, at, b)
        self.assertEqual((s.action, s.reason), (ACTION_SAFE_STOP, "LADDER_EXHAUSTED:CBS_NO_SOLUTION"))

    def test_without_l1_goes_to_global(self):
        at = [Attempt("L0", (0, 1), "no_solution", 100.0)]
        s = next_ladder_step(("L0", "G"), at, _budget())
        self.assertEqual((s.action, s.level), (ACTION_CALL, "G"))

    def test_attempts_out_of_ladder_order_are_refused(self):
        with self.assertRaises(ValueError):
            next_ladder_step(LADDER, [Attempt("L1", (0,), "no_solution", 1.0)], _budget())

    def test_expansion_from_goal_blocker_feeds_l1_group(self):
        # Bottleneck t=2 log state, group {0,1} alone: 2/3 stand on 0/1's goals.
        c = CASES["bottleneck_t2"]
        goals = {0: (1, 7), 1: (3, 7), 2: (0, 1), 3: (4, 1)}
        hits, add = expansion_candidates((0, 1), c["current"], goals)
        self.assertEqual(add, (2, 3))
        self.assertEqual({h.kind for h in hits}, {"GOAL_BLOCKED"})

    @unittest.skip("PENDING OPEN_DECISION: when remaining budget should skip L1 and jump "
                   "straight to G (current function always takes the next ladder entry)")
    def test_pending_skip_to_global(self):
        pass

    @unittest.skip("PENDING OPEN_DECISION: whether L1 expansion repeats to a closure "
                   "(expansion_candidates is one level only) and its size limit")
    def test_pending_expansion_closure(self):
        pass


# -- 9. invalid_path / full-check failure -> SAFE_STOP --------------------------

class TestImmediateSafeStop(unittest.TestCase):

    def test_invalid_path_stops_despite_budget_and_levels(self):
        at = [Attempt("L0", (0, 1), "invalid_path", 1.0)]
        s = next_ladder_step(LADDER, at, _budget(total=1e6, attempts=10))
        self.assertEqual((s.action, s.reason), (ACTION_SAFE_STOP, "CBS_INVALID_PATH"))

    def test_full_check_failure_stops(self):
        s_t = {0: (0, 1), 1: (0, 3), 2: (1, 3)}
        paths = {0: [(0, 1), (1, 1), (1, 2), (1, 3)], 1: [(0, 3), (0, 2)]}
        check = validate_full_robot(paths, s_t, [0, 1, 2], OPEN)
        self.assertFalse(check.ok)
        at = [Attempt("L0", (0, 1), "success", 5.0, full_check_ok=check.ok)]
        s = next_ladder_step(LADDER, at, _budget(total=1e6, attempts=10))
        self.assertEqual((s.action, s.reason), (ACTION_SAFE_STOP, "FULL_CHECK_FAILED"))

    def test_success_needs_full_check_result(self):
        with self.assertRaises(ValueError):
            next_ladder_step(LADDER, [Attempt("L0", (0, 1), "success", 5.0)], _budget())

    def test_success_with_full_check_executes(self):
        at = [Attempt("L0", (0, 1), "success", 5.0, full_check_ok=True)]
        s = next_ladder_step(LADDER, at, _budget())
        self.assertEqual((s.action, s.level), (ACTION_EXECUTE, "L0"))

    def test_obstacle_and_jump_are_full_check_failures(self):
        s_t = {0: (0, 0), 1: (4, 6)}
        grid = [row[:] for row in GRID5]
        grid[0][1] = 1
        r = validate_full_robot({0: [(0, 0), (0, 1)]}, s_t, [0, 1], grid)
        self.assertIn("obstacle", r.reason)
        r = validate_full_robot({0: [(0, 0), (0, 2)]}, s_t, [0, 1], GRID5)
        self.assertIn("illegal movement", r.reason)

    def test_unagreed_statuses_never_continue_the_ladder(self):
        # Provisional: same as current Global (SAFE_STOP). The final rule is OPEN.
        for status in ("input_error", "solver_error", "weird"):
            with self.subTest(status):
                s = next_ladder_step(LADDER, [Attempt("L0", (0, 1), status, 1.0)], _budget())
                self.assertNotEqual(s.action, ACTION_CALL)
                self.assertTrue(s.reason.startswith("UNAGREED_STATUS:"))

    @unittest.skip("PENDING OPEN_DECISION: final policy for input_error / solver_error "
                   "(retry, expand, Global, or SAFE_STOP)")
    def test_pending_input_and_solver_error_policy(self):
        pass


# -- 10. Global result preserved when Selective is not used --------------------

class TestGlobalPreserved(unittest.TestCase):

    def test_current_ladder_is_global_only(self):
        self.assertEqual(LADDER_UNTIL_OUTSIDE_POLICY_VERIFIED, ("G",))
        b = TriggerBudget(total_time_ms=1.0, max_attempts=1, level_timeout_ms={"G": 1.0})
        s = next_ladder_step(LADDER_UNTIL_OUTSIDE_POLICY_VERIFIED, [], b)
        self.assertEqual((s.action, s.level), (ACTION_CALL, "G"))
        at = [Attempt("G", (0, 1, 2), "no_solution", 0.5)]
        s = next_ladder_step(LADDER_UNTIL_OUTSIDE_POLICY_VERIFIED, at, b)
        self.assertEqual(s.action, ACTION_SAFE_STOP)    # same as Global: non-success -> stop

    def test_full_group_check_adds_nothing(self):
        s_t = {0: (0, 0), 1: (2, 0)}
        paths = {0: [(0, 0), (0, 1), (0, 2)], 1: [(2, 0), (2, 1), (2, 2)]}
        r = validate_full_robot(paths, s_t, [0, 1], OPEN)
        self.assertTrue(r.ok)
        self.assertEqual(r.full_paths, paths)          # no WAIT injected for G

    def test_global_runner_untouched_and_not_wired(self):
        from src.compare import common_runner
        from src.hybrid import global_hybrid
        from src.hybrid.global_hybrid import GlobalHybridConfig
        fields = [f.name for f in dataclasses.fields(GlobalHybridConfig)]
        self.assertEqual(fields, ["h_exec", "resume_policy", "recall_policy", "safe_stop_handler", "rearm"])
        self.assertNotIn("group_selector", fields)
        src = inspect.getsource(global_hybrid)
        for name in ("selective", "wfg"):
            self.assertNotIn(f"src.hybrid.{name}", src)
        self.assertIn("SELECTIVE_HYBRID", common_runner.NOT_IMPLEMENTED)

    def test_mock_global_episode_identical_with_selective_helpers_called(self):
        from tests.test_global_hybrid_runner import _run
        from tests.hybrid_mocks import GOALS, PATHS_TO_GOALS, STARTS, RecordingPlanner, success

        base = _run(RecordingPlanner(success(PATHS_TO_GOALS)), h_exec=2)
        # Exercise every new helper on the same Trigger state, then rerun.
        paths = {0: PATHS_TO_GOALS[0], 1: PATHS_TO_GOALS[1]}
        validate_full_robot(paths, STARTS, [0, 1], [[0] * 4 for _ in range(3)])
        expansion_candidates((0,), STARTS, GOALS, {0: PATHS_TO_GOALS[0]})
        next_ladder_step(("G",), [], _budget(G=1.0))
        again = _run(RecordingPlanner(success(PATHS_TO_GOALS)), h_exec=2)
        self.assertEqual(again, base)
        self.assertEqual(base["termination"], "SUCCESS")
        self.assertEqual(base["action_sources"], ["IL", "CBS", "CBS"])


# -- PENDING policies that must not be asserted yet ---------------------------

class TestPendingOutsidePolicy(unittest.TestCase):

    @unittest.skip("PENDING_VALIDATION P3: WAIT + static-obstacle handling of outside agents "
                   "is a candidate only; the runner cannot hold outside agents today and atb033 "
                   "has no time-indexed reservation input")
    def test_pending_static_obstacle_policy_is_safe(self):
        pass

    @unittest.skip("PENDING OPEN_DECISION: full-check horizon -- whole padded plan (current) "
                   "vs execution window H = min(h_exec, horizon)")
    def test_pending_full_check_window(self):
        pass


if __name__ == "__main__":
    unittest.main()
