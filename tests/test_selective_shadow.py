"""Mock tests for Selective SHADOW mode (src/hybrid/selective_shadow.py).

Every episode runs the real ``run_global_hybrid_episode`` with the real
``DeadlockMonitor`` and MAPFStepSimulator, and test-only Mock planners from
tests/hybrid_mocks.py (or small test-local test_only planners). No CBS solver,
no file output outside a temporary directory. Rearm / H_exec / recall values
are TEST-ONLY.

The core assertion everywhere: the episode result with the shadow wrappers
injected is EQUAL to the result without them.
"""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.common.deadlock_monitor import DeadlockMonitor
from src.common.episode_logger import EpisodeLogger
from src.hybrid import selective_shadow as sh
from src.hybrid.global_hybrid import GlobalHybridConfig, MonitorRearmConfig, run_global_hybrid_episode
from src.hybrid.replan_boundary import check_validator_pairing
from src.hybrid.selective_decision import ROUTE_GLOBAL
from src.hybrid.selective_shadow import ShadowMonitor, ShadowSelectivePlanner
from tests.hybrid_mocks import (
    GOALS, PATHS_TO_GOALS, STARTS,
    LEFT, RIGHT,
    AcceptAllValidator, AllowAllRecalls, AlwaysResumeIL, MockReplanResult,
    RaisingPlanner, RecordSafeStop, RecordingPlanner, ScriptedPolicy, open_grid, success,
)


# -- helpers --------------------------------------------------------------------

class StayPlanner:
    """TEST ONLY. Index-keyed: every agent holds its S_t for one step."""
    test_only = True

    def __init__(self):
        self.calls = []

    def __call__(self, current_positions, goals, grid):
        self.calls.append((list(current_positions), list(goals)))
        return success([[tuple(p), tuple(p)] for p in current_positions], latency_sec=0.01)


class StepRightPlanner:
    """TEST ONLY. Index-keyed: each agent steps one cell right (if free)."""
    test_only = True

    def __init__(self):
        self.calls = []

    def __call__(self, current_positions, goals, grid):
        self.calls.append(list(current_positions))
        g = np.asarray(grid)
        paths = []
        for r, c in current_positions:
            nxt = (r, c + 1) if c + 1 < g.shape[1] and g[r, c + 1] == 0 else (r, c)
            paths.append([(r, c), nxt])
        return success(paths, latency_sec=0.02)


class AgentIdStayPlanner:
    """TEST ONLY. agent_id-keyed, like CbsRuntimePlanner."""
    test_only = True
    paths_keying = "agent_id"

    def __init__(self):
        self.calls = []

    def __call__(self, current_positions, goals, grid):
        self.calls.append(dict(current_positions))
        return success({a: [p, p] for a, p in current_positions.items()}, latency_sec=0.01)


def _config(h_exec=1, rearm=None):
    return GlobalHybridConfig(h_exec=h_exec, resume_policy=AlwaysResumeIL(),
                              recall_policy=AllowAllRecalls(), safe_stop_handler=RecordSafeStop(),
                              rearm=rearm)


def _episode(planner, grid, starts, goals, policy_actions, *, shadow, max_steps=6,
             h_exec=1, rearm=None, logger=None, monitor=None):
    """Run once; with shadow=True the planner and monitor are wrapped."""
    mon = monitor if monitor is not None else DeadlockMonitor()
    used_planner = planner
    if shadow:
        if not isinstance(mon, ShadowMonitor):
            mon = ShadowMonitor(mon)
        used_planner = ShadowSelectivePlanner(planner, mon)
    result = run_global_hybrid_episode(
        ScriptedPolicy(copy.deepcopy(policy_actions)), grid, starts, goals, max_steps,
        planner=used_planner, path_validator=AcceptAllValidator(),
        config=_config(h_exec, rearm), monitor=mon, logger=logger)
    return result, used_planner, mon


def _compare(test, make_planner, grid, starts, goals, actions, **kw):
    """Run with and without shadow on fresh planners; assert equal outcomes."""
    p0, p1 = make_planner(), make_planner()
    base, _, mon0 = _episode(p0, grid, starts, goals, actions, shadow=False, **kw)
    shad, wrapper, mon = _episode(p1, grid, starts, goals, actions, shadow=True, **kw)
    test.assertEqual(shad, base)                                # whole result dict
    test.assertEqual(shad.keys(), base.keys())
    for key in ("trajectory", "actions", "termination", "stop_reason", "transitions",
                "cbs_calls", "action_sources"):
        test.assertEqual(shad[key], base[key], key)
    if "monitor_trigger_history" in base:                       # rearm mode only
        test.assertEqual(shad["monitor_trigger_history"], base["monitor_trigger_history"])
    test.assertEqual(mon.trigger_history, mon0.trigger_history)
    test.assertEqual(mon.result, mon0.result)
    test.assertEqual(p1.calls, p0.calls)
    return base, wrapper, mon, p1


# Synthetic layouts (all open grids; ids are runtime ids).
GRID_3x7 = np.zeros((3, 7), dtype=np.int64)

# Two agents contest (1,1) forever -> Trigger at t=1, one FULL group.
FULL_STARTS = {0: (0, 1), 1: (2, 1)}
FULL_GOALS = {0: (2, 1), 1: (0, 1)}
FULL_ACTIONS = {0: [1] * 10, 1: [0] * 10}                 # DOWN / UP

# Two independent contests + outside agent 4 oscillating (1,6)<->(1,5):
# S_1 != S_0 (4 moved), S_2 == S_0 -> Trigger at t=2 where agent 4 moved.
IND_STARTS = {0: (0, 1), 1: (0, 3), 2: (2, 1), 3: (2, 3), 4: (1, 6)}
IND_GOALS = {0: (1, 6), 1: (0, 0), 2: (2, 6), 3: (2, 0), 4: (1, 0)}   # 0's goal = 4's cell
IND_ACTIONS = {0: [RIGHT] * 10, 1: [LEFT] * 10, 2: [RIGHT] * 10, 3: [LEFT] * 10,
               4: [LEFT, RIGHT] * 5}


# -- 1-3, 8. outcome identity -----------------------------------------------------

class TestGlobalOutcomeUnchanged(unittest.TestCase):

    def test_existing_mock_success_episode_identical(self):
        base, wrapper, mon, p1 = _compare(
            self, lambda: RecordingPlanner(success(PATHS_TO_GOALS)),
            open_grid(), STARTS, GOALS, {}, h_exec=2, max_steps=10)
        self.assertEqual(base["termination"], "SUCCESS")
        self.assertEqual(len(p1.calls), 1)                 # exactly one Global call
        self.assertEqual(len(wrapper.records), 1)
        rec = wrapper.records[0]
        self.assertIsNone(rec.shadow_error)
        self.assertEqual(rec.decision.reason, "NO_GROUP")  # all-WAIT: no WFG edges
        self.assertEqual(rec.global_status, "success")

    def test_full_group_recorded_path_unchanged(self):
        base, wrapper, mon, p1 = _compare(self, StayPlanner, GRID_3x7, FULL_STARTS, FULL_GOALS,
                                          FULL_ACTIONS)
        rec = wrapper.records[0]
        self.assertEqual(rec.trigger_t, 1)
        self.assertEqual(rec.wfg.groups, ((0, 1),))
        self.assertEqual((rec.decision.route, rec.decision.reason), (ROUTE_GLOBAL, "FULL_GROUP"))
        self.assertEqual((rec.l0_group, rec.outside, rec.precheck), ((0, 1), (), None))
        self.assertEqual(len(p1.calls), 1)

    def test_independent_groups_recorded_path_unchanged(self):
        base, wrapper, mon, p1 = _compare(self, StayPlanner, GRID_3x7, IND_STARTS, IND_GOALS,
                                          IND_ACTIONS)
        rec = wrapper.records[0]
        self.assertEqual(rec.wfg.groups, ((0, 1), (2, 3)))
        self.assertEqual(rec.decision.route, ROUTE_GLOBAL)
        self.assertEqual(rec.decision.reason, "SELECTIVE_CANDIDATE_BLOCKED")
        self.assertEqual((rec.l0_group, rec.outside), ((0, 1, 2, 3), (4,)))
        self.assertEqual(rec.expansion_add, (4,))
        self.assertEqual(rec.precheck.goal_blocked, (0,))
        self.assertEqual(rec.precheck.unreachable, (0,))
        # The planner still got ALL agents (Global request), exactly once.
        self.assertEqual(len(p1.calls), 1)
        self.assertEqual(len(p1.calls[0][0]), 5)
        self.assertEqual(base["cbs_calls"][0]["agent_order"], (0, 1, 2, 3, 4))

    def test_agent_id_keyed_planner_identical(self):
        base, wrapper, mon, p1 = _compare(self, AgentIdStayPlanner, GRID_3x7, IND_STARTS,
                                          IND_GOALS, IND_ACTIONS)
        self.assertIsNone(wrapper.records[0].shadow_error)
        self.assertEqual(len(p1.calls), 1)

    def test_non_success_status_unchanged(self):
        for status in ("timeout", "no_solution", "invalid_path", "input_error", "solver_error"):
            with self.subTest(status):
                mk = lambda: RecordingPlanner(MockReplanResult(status, None, 1.5))  # noqa: E731
                base, wrapper, mon, p1 = _compare(self, mk, GRID_3x7, FULL_STARTS, FULL_GOALS,
                                                  FULL_ACTIONS)
                self.assertEqual(base["termination"], "SAFE_STOP")
                self.assertEqual(base["stop_reason"], f"CBS_{status.upper()}")
                self.assertEqual(base["cbs_calls"][0]["latency_ms"], 1500.0)
                self.assertEqual(wrapper.records[0].global_status, status)

    def test_wrapper_returns_inner_result_object_and_args(self):
        seen = {}

        class Probe:
            test_only = True

            def __call__(self, cur, goals, grid):
                seen.update(cur=cur, goals=goals, grid=grid)
                return seen.setdefault("result", success([[(0, 0)]]))

        mon = ShadowMonitor(DeadlockMonitor())
        w = ShadowSelectivePlanner(Probe(), mon)
        cur, goals, grid = [(0, 0)], [(0, 1)], open_grid()
        snapshot = grid.copy()
        out = w(cur, goals, grid)
        self.assertIs(out, seen["result"])
        self.assertIs(seen["cur"], cur)
        self.assertIs(seen["goals"], goals)
        self.assertIs(seen["grid"], grid)
        self.assertTrue(np.array_equal(grid, snapshot))
        self.assertEqual(w.records[0].shadow_error, sh.ERR_NO_STEP)   # no step observed yet


# -- 4. shadow failure vs Global failure ----------------------------------------

class TestFailures(unittest.TestCase):

    def test_shadow_exception_recorded_global_unchanged(self):
        with mock.patch("src.hybrid.selective_shadow.build_wfg", side_effect=RuntimeError("boom")):
            base, wrapper, mon, p1 = _compare(self, StayPlanner, GRID_3x7, IND_STARTS,
                                              IND_GOALS, IND_ACTIONS)
        rec = wrapper.records[0]
        self.assertEqual(rec.shadow_error, "EXCEPTION:RuntimeError:boom")
        self.assertIsNone(rec.decision)
        self.assertEqual(rec.global_status, "success")
        self.assertEqual(len(p1.calls), 1)

    def test_monitor_recording_failure_does_not_touch_monitor_path(self):
        with mock.patch("src.hybrid.selective_shadow._cells", side_effect=RuntimeError("rec")):
            base, wrapper, mon, p1 = _compare(self, StayPlanner, GRID_3x7, FULL_STARTS,
                                              FULL_GOALS, FULL_ACTIONS)
        self.assertTrue(mon.record_errors)
        self.assertTrue(all(e.startswith("EXCEPTION:RuntimeError") for e in mon.record_errors))

    def test_global_planner_exception_propagates(self):
        base, _, _ = _episode(RaisingPlanner(), GRID_3x7, FULL_STARTS, FULL_GOALS,
                              FULL_ACTIONS, shadow=False)
        inner = RaisingPlanner()
        shad, wrapper, _ = _episode(inner, GRID_3x7, FULL_STARTS, FULL_GOALS,
                                    FULL_ACTIONS, shadow=True)
        self.assertEqual(shad, base)
        self.assertEqual(shad["stop_reason"], "PLANNER_EXCEPTION")
        self.assertEqual(inner.calls, 1)
        self.assertIsNone(wrapper.records[0].global_status)
        # Direct call: the inner exception is not swallowed.
        w = ShadowSelectivePlanner(RaisingPlanner(), ShadowMonitor(DeadlockMonitor()))
        with self.assertRaises(RuntimeError):
            w([(0, 0)], [(0, 1)], open_grid())

    def test_paths_keying_and_test_only_are_inner_values(self):
        class RealLike:                      # no test_only, agent_id keying
            paths_keying = "agent_id"

            def __call__(self, *a):
                raise AssertionError("not called")

        class Bare:
            def __call__(self, *a):
                raise AssertionError("not called")

        mon = ShadowMonitor(DeadlockMonitor())
        w = ShadowSelectivePlanner(RealLike(), mon)
        self.assertEqual((w.paths_keying, w.test_only), ("agent_id", False))
        with self.assertRaises(ValueError):
            check_validator_pairing(w, AcceptAllValidator())       # same refusal as inner
        with self.assertRaises(ValueError):
            check_validator_pairing(RealLike(), AcceptAllValidator())
        b = ShadowSelectivePlanner(Bare(), mon)
        self.assertEqual((b.paths_keying, b.test_only), ("index", False))
        t = ShadowSelectivePlanner(StayPlanner(), mon)
        self.assertEqual((t.paths_keying, t.test_only), ("index", True))


# -- 5. time axis and Trigger timestep -------------------------------------------

class TestTimeAxisRecording(unittest.TestCase):

    def test_s_t_minus_1_and_s_t_are_distinct_at_trigger(self):
        base, wrapper, mon, p1 = _compare(self, StayPlanner, GRID_3x7, IND_STARTS, IND_GOALS,
                                          IND_ACTIONS)
        rec = wrapper.records[0]
        self.assertEqual(rec.trigger_t, 2)
        self.assertEqual(base["cbs_calls"][0]["trigger_timestep"], 2)
        self.assertEqual(rec.s_t_minus_1[4], (1, 5))      # before step 2
        self.assertEqual(rec.s_t[4], (1, 6))              # after step 2 (= CBS start)
        self.assertEqual(rec.intended[4], (1, 6))
        self.assertEqual(rec.blocked, (0, 1, 2, 3))
        self.assertEqual(rec.s_t, base["cbs_calls"][0]["s_t"])
        trig = [s for s in mon.steps if s.fresh_trigger]
        self.assertEqual([s.t for s in trig], [2])
        self.assertEqual(rec.wfg.groups, ((0, 1), (2, 3)))  # built from S_1 + intended

    def test_args_mismatching_last_step_are_not_analysed(self):
        mon = ShadowMonitor(DeadlockMonitor())
        mon.reset(FULL_STARTS, FULL_GOALS)
        mon.observe_step(1, dict(FULL_STARTS), {0: "vertex", 1: "vertex"},
                         current_positions=dict(FULL_STARTS),
                         intended_positions={0: (1, 1), 1: (1, 1)})
        w = ShadowSelectivePlanner(StayPlanner(), mon)
        w([(0, 2), (2, 1)], [(2, 1), (0, 1)], GRID_3x7)
        self.assertEqual(w.records[0].shadow_error, sh.ERR_S_T_MISMATCH)
        self.assertEqual(len(w.inner.calls), 1)

    def test_missing_intent_is_reported(self):
        mon = ShadowMonitor(DeadlockMonitor())
        mon.reset(FULL_STARTS, FULL_GOALS)
        mon.observe_step(1, dict(FULL_STARTS), {0: "none", 1: "none"})
        w = ShadowSelectivePlanner(StayPlanner(), mon)
        w([(0, 1), (2, 1)], [(2, 1), (0, 1)], GRID_3x7)
        self.assertEqual(w.records[0].shadow_error, sh.ERR_NO_INTENT)


# -- 7. reset / rearm / episode boundaries ----------------------------------------

class TestBoundaries(unittest.TestCase):

    REARM = MonitorRearmConfig(max_cbs_calls=3, cooldown_steps=0, cbs_time_budget_ms=1e9)  # TEST-ONLY

    def test_rearm_produces_one_fresh_record_per_call(self):
        base, wrapper, mon, p1 = _compare(self, StepRightPlanner, open_grid(), STARTS, GOALS, {},
                                          rearm=self.REARM, max_steps=10)
        self.assertEqual(len(p1.calls), len(base["cbs_calls"]))
        self.assertGreaterEqual(len(base["cbs_calls"]), 2)
        self.assertEqual([r.trigger_t for r in wrapper.records],
                         [c["trigger_timestep"] for c in base["cbs_calls"]])
        self.assertEqual([r.s_t for r in wrapper.records], [c["s_t"] for c in base["cbs_calls"]])
        self.assertEqual([r.call_index for r in wrapper.records],
                         list(range(1, len(wrapper.records) + 1)))
        self.assertTrue(all(r.shadow_error is None for r in wrapper.records))
        self.assertEqual(len(mon.rearms), len(base["monitor_rearms"]))

    def test_second_episode_never_reuses_first_episode_state(self):
        mon = ShadowMonitor(DeadlockMonitor())
        inner = StayPlanner()
        _episode(inner, GRID_3x7, IND_STARTS, IND_GOALS, IND_ACTIONS, shadow=True, monitor=mon)
        first = list(mon.steps)
        w2 = ShadowSelectivePlanner(StayPlanner(), mon)
        # New episode on the same monitor: reset clears steps and goals.
        mon.reset(FULL_STARTS, FULL_GOALS)
        self.assertEqual((mon.episode, mon.steps, mon.rearms), (2, [], []))
        self.assertEqual(mon.goals, FULL_GOALS)
        w2([(0, 1), (2, 1)], [(2, 1), (0, 1)], GRID_3x7)          # no step yet in episode 2
        self.assertEqual(w2.records[0].shadow_error, sh.ERR_NO_STEP)
        self.assertTrue(first and all(s.episode == 1 for s in first))

    def test_same_wrapper_across_two_episodes(self):
        mon = ShadowMonitor(DeadlockMonitor())
        inner = StayPlanner()
        wrapper = ShadowSelectivePlanner(inner, mon)
        for _ in range(2):
            run_global_hybrid_episode(
                ScriptedPolicy(copy.deepcopy(FULL_ACTIONS)), GRID_3x7, FULL_STARTS, FULL_GOALS, 6,
                planner=wrapper, path_validator=AcceptAllValidator(), config=_config(),
                monitor=mon)
        self.assertEqual([(r.episode, r.call_index, r.trigger_t) for r in wrapper.records],
                         [(1, 1, 1), (2, 1, 1)])
        self.assertTrue(all(r.shadow_error is None for r in wrapper.records))

    def test_stale_step_is_not_reanalysed(self):
        mon = ShadowMonitor(DeadlockMonitor())
        mon.reset(FULL_STARTS, FULL_GOALS)
        mon.observe_step(1, dict(FULL_STARTS), {0: "vertex", 1: "vertex"},
                         current_positions=dict(FULL_STARTS),
                         intended_positions={0: (1, 1), 1: (1, 1)})
        w = ShadowSelectivePlanner(StayPlanner(), mon)
        args = ([(0, 1), (2, 1)], [(2, 1), (0, 1)], GRID_3x7)
        w(*args)
        w(*args)
        self.assertEqual([r.shadow_error for r in w.records], [None, sh.ERR_STALE_STEP])
        self.assertEqual(len(w.inner.calls), 2)          # inner still called every time

    def test_observe_step_called_exactly_once_with_same_args(self):
        class CountingMonitor(DeadlockMonitor):
            def __init__(self):
                super().__init__()
                self.observed = []

            def observe_step(self, *args, **kwargs):
                out = super().observe_step(*args, **kwargs)
                self.observed.append((args, kwargs, out))
                return out

        inner = CountingMonitor()
        mon = ShadowMonitor(inner)
        mon.reset(FULL_STARTS, FULL_GOALS)
        pos, ct = dict(FULL_STARTS), {0: "vertex", 1: "vertex"}
        kw = dict(current_positions=dict(FULL_STARTS), intended_positions={0: (1, 1), 1: (1, 1)})
        out = mon.observe_step(1, pos, ct, **kw)
        self.assertEqual(len(inner.observed), 1)
        args, kwargs, inner_out = inner.observed[0]
        self.assertIs(args[1], pos)
        self.assertIs(args[2], ct)
        self.assertEqual(kwargs.keys(), kw.keys())
        self.assertIs(out, inner_out)                     # same object, not a copy
        # Delegated attributes are the inner monitor's own.
        self.assertEqual(mon.trigger_history, inner.trigger_history)
        self.assertEqual(mon.rearm_count, inner.rearm_count)
        self.assertEqual(mon.result, inner.result)


# -- 9. files: only a temporary directory -----------------------------------------

class TestFilesOnlyInTempDir(unittest.TestCase):

    def test_logger_rows_and_written_csv_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            outs = []
            for shadow, name in ((False, "base"), (True, "shadow")):
                d = Path(tmp) / name
                logger = EpisodeLogger("shadow_cmp", scenario="synthetic", mode="GLOBAL_HYBRID", out_dir=d)
                _episode(StayPlanner(), GRID_3x7, IND_STARTS, IND_GOALS, IND_ACTIONS,
                         shadow=shadow, logger=logger)
                path = logger.write_steps()
                self.assertTrue(Path(path).resolve().is_relative_to(Path(tmp).resolve()))
                outs.append((logger.step_rows, Path(path).read_bytes()))
            self.assertEqual(outs[0], outs[1])

    def test_module_writes_no_files(self):
        import inspect
        src = inspect.getsource(sh)
        for token in ("open(", "write_", "mkdir", "outputs/"):
            self.assertNotIn(token, src)


if __name__ == "__main__":
    unittest.main()
