"""Global Hybrid Runner: Monitor rearm + bounded CBS recall (Mock CBS).

Synthetic setup (no claim of real IL behaviour): 3x4 open grid, agents 0 and
1 start at (0,0)/(2,0) with goals at (0,3)/(2,3), all-WAIT IL policy unless
scripted otherwise. The real DeadlockMonitor fires at t=1 (S_1 == S_0); a
Mock CBS plan moves both agents right; after the CBS window IL WAITs again,
so the rearmed Monitor fires a second, fresh Trigger.

The MonitorRearmConfig values here are test-only, not agreed policy.
"""

import unittest

from scripts.run_corridor_hybrid_e2e import dev_config
from scripts.run_hybrid_resume_dev import (
    DEV_REARM_CBS_TIME_BUDGET_MS,
    DEV_REARM_COOLDOWN_STEPS,
    DEV_REARM_MAX_CBS_CALLS,
    dev_rearm_config,
    dev_resume_config,
)
from src.common.deadlock_monitor import DeadlockMonitor
from src.compare.common_runner import (
    ALL_METHODS,
    CBS_METHODS,
    DEV_SETTINGS,
    GLOBAL_HYBRID_DEV_H3_REARM,
    METHODS,
)
from src.compare.compare_csv import EXEC_POLICY, REQUEST_KIND
from src.hybrid.global_hybrid import (
    CBS_TIME_BUDGET_EXCEEDED,
    RECALL_COOLDOWN_ACTIVE,
    RECALL_LIMIT_EXCEEDED,
    RECALL_SAME_STATE,
    GlobalHybridConfig,
    MonitorRearmConfig,
    run_global_hybrid_episode,
)
from src.simulator.mapf_step_simulator import MAPFStepSimulator
from tests.hybrid_mocks import (
    RIGHT, WAIT,
    AcceptAllValidator,
    AllowAllRecalls,
    AlwaysResumeIL,
    MockReplanResult,
    RecordSafeStop,
    ScriptedPolicy,
    open_grid,
    perturbing_simulator,
    success,
)

STARTS = {0: (0, 0), 1: (2, 0)}
GOALS = {0: (0, 3), 1: (2, 3)}
P1 = [[(0, 0), (0, 1), (0, 2), (0, 3)], [(2, 0), (2, 1), (2, 2), (2, 3)]]   # from S_0
P2 = [[(0, 1), (0, 2), (0, 3)], [(2, 1), (2, 2), (2, 3)]]                   # from (0,1)/(2,1)
S_AFTER_1 = {0: (0, 1), 1: (2, 1)}

BASE_RESULT_KEYS = {
    "steps", "done", "all_at_goal", "timed_out", "final_positions", "goals", "trajectory",
    "actions", "integrity_abort", "integrity_abort_timestep", "termination", "stop_reason",
    "safe_stop", "transitions", "cbs_calls", "action_sources", "cbs_exec_trace", "exec_mismatch",
}
REARM_KEYS = {"rearm_config", "monitor_rearms", "triggers", "monitor_trigger_history"}


class SequencePlanner:
    """TEST ONLY: returns results[i] on the i-th call (index-keyed paths)."""

    test_only = True

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, current_positions, goals, grid):
        self.calls.append({"current_positions": list(current_positions), "goals": list(goals)})
        item = self.results[len(self.calls) - 1]
        if isinstance(item, Exception):
            raise item
        return item


class RejectNthValidator(AcceptAllValidator):
    """TEST ONLY: rejects the n-th validation (1-based), accepts the rest."""

    def __init__(self, n):
        super().__init__()
        self.n = n

    def __call__(self, grid, starts, goals, paths):
        super().__call__(grid, starts, goals, paths)
        return (False, "mock_reject_nth") if len(self.calls) == self.n else (True, "")


class SpyMonitor(DeadlockMonitor):
    """Real DeadlockMonitor that logs the order of observe_step / rearm calls."""

    def __init__(self):
        super().__init__()
        self.events = []

    def observe_step(self, timestep, positions, conflict_types, **kw):
        self.events.append(("observe", timestep))
        return super().observe_step(timestep, positions, conflict_types, **kw)

    def rearm(self, current_positions, current_timestep):
        self.events.append(("rearm", current_timestep, dict(current_positions)))
        return super().rearm(current_positions, current_timestep)


def counting_simulator(base=MAPFStepSimulator):
    class _Counting(base):
        step_calls = 0

        def step(self, actions):
            type(self).step_calls += 1
            return super().step(actions)

    return _Counting


def rearm_cfg(max_calls=2, cooldown=1, budget_ms=10_000.0):
    return MonitorRearmConfig(max_cbs_calls=max_calls, cooldown_steps=cooldown,
                              cbs_time_budget_ms=budget_ms)


def _run(planner, *, rearm=None, h_exec=1, recall=None, validator=None, policy=None,
         starts=STARTS, goals=GOALS, max_steps=12, simulator_cls=None, monitor=None):
    config = GlobalHybridConfig(
        h_exec=h_exec,
        resume_policy=AlwaysResumeIL(),
        recall_policy=recall or AllowAllRecalls(),
        safe_stop_handler=RecordSafeStop(),
        rearm=rearm,
    )
    sim_cls = simulator_cls or counting_simulator()
    result = run_global_hybrid_episode(
        policy or ScriptedPolicy(), open_grid(), starts, goals, max_steps,
        planner=planner, path_validator=validator or AcceptAllValidator(),
        config=config, monitor=monitor or DeadlockMonitor(), simulator_cls=sim_cls,
    )
    return result, sim_cls


def _route(result):
    return [(tr["from"], tr["to"], tr["reason"]) for tr in result["transitions"]]


class TestBaselineUnchanged(unittest.TestCase):

    def test_rearm_none_keeps_single_trigger_behaviour(self):
        planner = SequencePlanner(success(P1), success(P2))
        spy = SpyMonitor()
        result, sim = _run(planner, rearm=None, max_steps=8, monitor=spy)
        self.assertEqual(len(planner.calls), 1)               # Monitor latched, no recall
        self.assertEqual(result["termination"], "TIMEOUT")
        self.assertEqual(set(result), BASE_RESULT_KEYS)        # no rearm-mode keys
        self.assertFalse(any(e[0] == "rearm" for e in spy.events))
        self.assertEqual(spy.rearm_count, 0)

    def test_dev_configs_of_existing_methods_have_no_rearm(self):
        self.assertIsNone(dev_resume_config(3).rearm)
        self.assertIsNone(dev_config(160).rearm)


class TestSecondCbsCall(unittest.TestCase):

    def test_restall_after_resume_triggers_second_call(self):
        p2_from_s3 = [[(0, 2), (0, 3)], [(2, 2), (2, 3)]]
        planner = SequencePlanner(success(P1), success(p2_from_s3, latency_sec=0.5))
        result, sim = _run(planner, rearm=rearm_cfg(), h_exec=2)
        calls = result["cbs_calls"]
        self.assertEqual(len(calls), 2)
        first, second = calls
        self.assertEqual(first["trigger_timestep"], 1)
        # h_exec=2: CBS t=2,3 -> (0,2)/(2,2); resume+rearm at t=3; IL WAIT t=4 repeats S_3.
        self.assertEqual(second["trigger_timestep"], 4)
        self.assertEqual(second["s_t"], {0: (0, 2), 1: (2, 2)})
        self.assertEqual(second["s_t"], result["trajectory"][4])
        self.assertEqual(second["status"], "success")
        self.assertTrue(second["accepted"])
        self.assertEqual(second["latency_ms"], 500.0)
        self.assertEqual(planner.calls[1]["current_positions"], [(0, 2), (2, 2)])

    def test_second_plan_executes_from_path1_and_succeeds(self):
        planner = SequencePlanner(success(P1), success(P2))
        result, sim = _run(planner, rearm=rearm_cfg(), h_exec=1)
        # t=1 Trigger, CBS t=2, resume t=2, IL t=3 (re-stall), second CBS window t=4
        calls = result["cbs_calls"]
        self.assertEqual([c["trigger_timestep"] for c in calls], [1, 3])
        self.assertEqual(calls[1]["s_t"], S_AFTER_1)
        second_trace = [e for e in result["cbs_exec_trace"] if e["t"] > 3]
        self.assertEqual(second_trace[0]["k"], 1)
        self.assertEqual(second_trace[0]["expected"], {0: P2[0][1], 1: P2[1][1]})   # Path[1]
        self.assertTrue(all(e["match"] and not e["overlap"] for e in result["cbs_exec_trace"]))
        self.assertEqual(result["triggers"][1]["first_seen"], 2)
        self.assertEqual(result["triggers"][1]["period"], 1)
        self.assertEqual([t["outcome"] for t in result["triggers"]][:2], ["CBS_REQUEST", "CBS_REQUEST"])

    def test_full_success_with_second_window_reaching_goal(self):
        planner = SequencePlanner(success(P1), success([[(0, 2), (0, 3)], [(2, 2), (2, 3)]]))
        result, sim = _run(planner, rearm=rearm_cfg(), h_exec=2)
        self.assertEqual(result["termination"], "SUCCESS")
        self.assertTrue(result["all_at_goal"])
        self.assertEqual(result["action_sources"], ["IL", "CBS", "CBS", "IL", "CBS"])
        self.assertEqual(result["monitor_rearms"], [{
            "t": 3, "positions": {0: (0, 2), 1: (2, 2)}, "after_call": 1,
            "resume_reason": "EXEC_WINDOW_END"}])
        self.assertEqual(len(result["monitor_trigger_history"]), 1)
        self.assertEqual(result["monitor_trigger_history"][0]["trigger_timestep"], 1)
        self.assertEqual(result["rearm_config"],
                         {"max_cbs_calls": 2, "cooldown_steps": 1, "cbs_time_budget_ms": 10_000.0})
        self.assertEqual(sim.step_calls, result["steps"])


class TestRearmTiming(unittest.TestCase):

    def test_rearm_once_per_resume_never_during_cbs_exec(self):
        spy = SpyMonitor()
        planner = SequencePlanner(success(P1), success([[(0, 2), (0, 3)], [(2, 2), (2, 3)]]))
        result, _ = _run(planner, rearm=rearm_cfg(), h_exec=2, monitor=spy)
        rearms = [e for e in spy.events if e[0] == "rearm"]
        self.assertEqual(len(rearms), 1)
        idx = spy.events.index(rearms[0])
        # immediately after the last CBS step's observation, before the first IL step
        self.assertEqual(spy.events[idx - 1], ("observe", 3))
        self.assertEqual(spy.events[idx + 1], ("observe", 4))
        self.assertEqual(result["action_sources"][3 - 1], "CBS")
        self.assertEqual(result["action_sources"][4 - 1], "IL")
        self.assertEqual(rearms[0][1], 3)
        self.assertEqual(rearms[0][2], result["trajectory"][3])        # actual positions
        resume = [tr for tr in result["transitions"] if tr["from"] == "IL_RESUME"]
        self.assertEqual([tr["t"] for tr in resume], [3])


class TestRecallRefusalsSafeStop(unittest.TestCase):

    def _assert_stopped(self, result, sim, reason, t, planner_calls, planner):
        self.assertEqual(result["termination"], "SAFE_STOP")
        self.assertEqual(result["stop_reason"], reason)
        self.assertEqual(result["steps"], t)
        self.assertEqual(sim.step_calls, t)                 # no sim.step after the refusal
        self.assertEqual(len(result["action_sources"]), t)
        self.assertEqual(len(planner.calls), planner_calls)
        self.assertEqual(_route(result)[-1], ("IL", "SAFE_STOP", reason))
        self.assertEqual(result["triggers"][-1]["outcome"], reason)
        self.assertEqual(result["safe_stop"]["reason"], reason)

    def test_recall_limit_exceeded(self):
        p2_partial = [[(0, 1), (0, 2), (0, 3)], [(2, 1), (2, 2), (2, 3)]]
        planner = SequencePlanner(success(P1), success(p2_partial), success(P1))
        result, sim = _run(planner, rearm=rearm_cfg(max_calls=2), h_exec=1)
        # calls at t=1, t=3; resume t=2, t=4; third stall at t=5 -> limit
        self.assertEqual([c["trigger_timestep"] for c in result["cbs_calls"]], [1, 3])
        self.assertEqual([r["t"] for r in result["monitor_rearms"]], [2, 4])
        self._assert_stopped(result, sim, RECALL_LIMIT_EXCEEDED, 5, 2, planner)

    def test_cbs_time_budget_exceeded(self):
        planner = SequencePlanner(success(P1, latency_sec=0.25), success(P2))
        result, sim = _run(planner, rearm=rearm_cfg(budget_ms=200.0), h_exec=1)
        self._assert_stopped(result, sim, CBS_TIME_BUDGET_EXCEEDED, 3, 1, planner)

    def test_unknown_latency_counts_as_budget_exceeded(self):
        planner = SequencePlanner(MockReplanResult("success", paths=P1, latency_sec=None), success(P2))
        result, sim = _run(planner, rearm=rearm_cfg(), h_exec=1)
        self._assert_stopped(result, sim, CBS_TIME_BUDGET_EXCEEDED, 3, 1, planner)

    def test_cooldown_active(self):
        planner = SequencePlanner(success(P1), success(P2))
        result, sim = _run(planner, rearm=rearm_cfg(cooldown=2), h_exec=1)
        # resume t=2, re-stall t=3: 3 - 2 = 1 < 2
        self._assert_stopped(result, sim, RECALL_COOLDOWN_ACTIVE, 3, 1, planner)

    def test_cooldown_satisfied_allows_recall(self):
        # IL: WAIT at t=1 (Trigger), after resume agent 0 steps RIGHT once, then WAITs.
        policy = ScriptedPolicy({0: [WAIT, RIGHT], 1: [WAIT, WAIT]})
        planner = SequencePlanner(success(P1), success([[(0, 3)], [(2, 2), (2, 3)]]))
        result, sim = _run(planner, rearm=rearm_cfg(cooldown=2), h_exec=2, policy=policy)
        # CBS t=2,3 -> resume t=3; IL t=4 agent 0 -> (0,3); t=5 repeat -> 5 - 3 = 2 >= 2
        self.assertEqual([c["trigger_timestep"] for c in result["cbs_calls"]], [1, 5])
        self.assertEqual(result["termination"], "SUCCESS")

    def test_same_state_as_earlier_call_refused(self):
        there_and_back = [[(0, 0), (0, 1), (0, 0)], [(2, 0), (2, 1), (2, 0)]]
        planner = SequencePlanner(success(there_and_back), success(P1))
        result, sim = _run(planner, rearm=rearm_cfg(), h_exec=2)
        # CBS t=2,3 returns to S_0; resume t=3; IL t=4 repeats S_0 == call 1's S_t
        self.assertEqual(result["triggers"][-1]["positions"], result["cbs_calls"][0]["s_t"])
        self._assert_stopped(result, sim, RECALL_SAME_STATE, 4, 1, planner)


class TestSecondCallFailuresStopMoves(unittest.TestCase):
    """A failed second call never moves an agent; contracts of the first call hold."""

    def _assert_no_move_after(self, result, sim, t, reason):
        self.assertEqual(result["termination"], "SAFE_STOP")
        self.assertEqual(result["stop_reason"], reason)
        self.assertEqual(result["steps"], t)
        self.assertEqual(sim.step_calls, t)
        self.assertEqual(result["trajectory"][-1], result["trajectory"][t])

    def test_second_call_solver_status_failures(self):
        for status in ("timeout", "no_solution", "invalid_path", "input_error", "solver_error"):
            with self.subTest(status=status):
                planner = SequencePlanner(success(P1),
                                          MockReplanResult(status, paths=P2, latency_sec=0.1, error="x"))
                result, sim = _run(planner, rearm=rearm_cfg(), h_exec=1)
                self._assert_no_move_after(result, sim, 3, f"CBS_{status.upper()}")
                self.assertEqual(result["cbs_calls"][1]["status"], status)
                self.assertAlmostEqual(result["cbs_calls"][1]["latency_ms"], 100.0)

    def test_second_call_stale_path0_refused(self):
        planner = SequencePlanner(success(P1), success(P1))       # still starts at S_0
        result, sim = _run(planner, rearm=rearm_cfg(), h_exec=1)
        self._assert_no_move_after(result, sim, 3, "BOUNDARY_START_MISMATCH")

    def test_second_call_validator_reject(self):
        planner = SequencePlanner(success(P1), success(P2))
        result, sim = _run(planner, rearm=rearm_cfg(), h_exec=1, validator=RejectNthValidator(2))
        self._assert_no_move_after(result, sim, 3, "VALIDATOR_REJECT:mock_reject_nth")

    def test_second_call_planner_exception(self):
        planner = SequencePlanner(success(P1), RuntimeError("mock crash"))
        result, sim = _run(planner, rearm=rearm_cfg(), h_exec=1)
        self._assert_no_move_after(result, sim, 3, "PLANNER_EXCEPTION")
        self.assertIn("mock crash", result["cbs_calls"][1]["error"])

    def test_second_window_mismatch_stops(self):
        # h_exec=1: CBS t=2, resume t=2, re-stall t=3, second CBS step t=4 perturbed
        sim_cls = counting_simulator(perturbing_simulator(4, 1, (1, 1)))
        planner = SequencePlanner(success(P1), success(P2))
        result, sim = _run(planner, rearm=rearm_cfg(), h_exec=1, simulator_cls=sim_cls)
        self._assert_no_move_after(result, sim, 4, "EXEC_MISMATCH")
        self.assertEqual(result["exec_mismatch"]["t"], 4)
        self.assertEqual(result["exec_mismatch"]["k"], 1)

    def test_second_window_overlap_aborts(self):
        sim_cls = counting_simulator(perturbing_simulator(4, 1, (0, 2)))   # onto agent 0's cell
        planner = SequencePlanner(success(P1), success(P2))
        result, sim = _run(planner, rearm=rearm_cfg(), h_exec=1, simulator_cls=sim_cls)
        self.assertEqual(result["termination"], "INTEGRITY_ABORT")
        self.assertEqual(result["integrity_abort_timestep"], 4)
        self.assertEqual(sim.step_calls, 4)


class TestAgentIdsPreserved(unittest.TestCase):

    def test_non_contiguous_ids_on_second_call(self):
        starts = {3: (0, 0), 7: (2, 0)}
        goals = {3: (0, 3), 7: (2, 3)}
        planner = SequencePlanner(success(P1), success(P2))
        result, _ = _run(planner, rearm=rearm_cfg(), h_exec=1, starts=starts, goals=goals)
        second = result["cbs_calls"][1]
        self.assertEqual(second["agent_order"], (3, 7))
        self.assertEqual(second["s_t"], {3: (0, 1), 7: (2, 1)})
        self.assertEqual(planner.calls[1]["current_positions"], [(0, 1), (2, 1)])
        self.assertEqual(result["monitor_rearms"][0]["positions"], {3: (0, 1), 7: (2, 1)})


class TestConfigAndRegistration(unittest.TestCase):

    def test_rearm_config_has_no_defaults_and_validates(self):
        with self.assertRaises(TypeError):
            MonitorRearmConfig()
        for kwargs in ({"max_cbs_calls": 0}, {"max_cbs_calls": True}, {"max_cbs_calls": 1.0},
                       {"cooldown_steps": -1}, {"cooldown_steps": 1.5},
                       {"cbs_time_budget_ms": 0}, {"cbs_time_budget_ms": -5.0},
                       {"cbs_time_budget_ms": True}, {"cbs_time_budget_ms": "1"}):
            args = {"max_cbs_calls": 2, "cooldown_steps": 0, "cbs_time_budget_ms": 1.0, **kwargs}
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                MonitorRearmConfig(**args)
        with self.assertRaises(ValueError):
            GlobalHybridConfig(h_exec=1, resume_policy=AlwaysResumeIL(), recall_policy=AllowAllRecalls(),
                               safe_stop_handler=RecordSafeStop(), rearm={"max_cbs_calls": 2})

    def test_dev_rearm_config_values_are_the_dev_candidates(self):
        cfg = dev_rearm_config(3)
        self.assertEqual(cfg.h_exec, 3)
        self.assertEqual(cfg.rearm, MonitorRearmConfig(
            max_cbs_calls=DEV_REARM_MAX_CBS_CALLS, cooldown_steps=DEV_REARM_COOLDOWN_STEPS,
            cbs_time_budget_ms=DEV_REARM_CBS_TIME_BUDGET_MS))
        self.assertEqual((DEV_REARM_MAX_CBS_CALLS, DEV_REARM_COOLDOWN_STEPS), (2, 2))

    def test_rearm_method_is_opt_in_only(self):
        self.assertNotIn(GLOBAL_HYBRID_DEV_H3_REARM, METHODS)     # default batch stays 15 runs
        self.assertEqual(len(METHODS), 5)
        self.assertIn(GLOBAL_HYBRID_DEV_H3_REARM, ALL_METHODS)
        self.assertIn(GLOBAL_HYBRID_DEV_H3_REARM, CBS_METHODS)
        self.assertEqual(DEV_SETTINGS[GLOBAL_HYBRID_DEV_H3_REARM]["status"],
                         "development setting, not an agreed project policy")
        self.assertEqual(REQUEST_KIND[GLOBAL_HYBRID_DEV_H3_REARM], "IL_STALL_TRIGGER")
        self.assertEqual(EXEC_POLICY[GLOBAL_HYBRID_DEV_H3_REARM], "DEV_FIXED_H_EXEC")


if __name__ == "__main__":
    unittest.main()
