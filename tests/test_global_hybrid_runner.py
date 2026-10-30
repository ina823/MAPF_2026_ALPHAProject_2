"""Global Hybrid Runner v0 state-machine tests with Mock CBS
(docs/global_hybrid_runner_v0.md).

Synthetic setup (tests/hybrid_mocks.py): 3x4 open grid, all-WAIT IL policy,
so the real DeadlockMonitor fires TRIGGER_FULL_REPEAT at t=1 with S_t ==
starts. H_exec / resume / recall / SAFE_STOP values below are temporary
test-only choices, not agreed project rules.
"""

import unittest

from scripts.run_il import DEFAULT_CKPT, load_scenario, run_episode
from src.common.deadlock_monitor import DeadlockMonitor
from src.common.episode_logger import EpisodeLogger
from src.hybrid.global_hybrid import GlobalHybridConfig, run_global_hybrid_episode
from src.simulator.mapf_step_simulator import MAPFStepSimulator
from tests.hybrid_mocks import (
    GOALS, PATHS_TO_GOALS, STARTS,
    RIGHT, UP, WAIT,
    AcceptAllValidator,
    AllowAllRecalls,
    AlwaysResumeIL,
    AlwaysSafeStopAfterExec,
    DenyAllRecalls,
    FailIfCalledPlanner,
    MockReplanResult,
    RaisingPlanner,
    RecordSafeStop,
    RecordingPlanner,
    RejectAllValidator,
    ReversedOrderPlanner,
    ScriptedPolicy,
    open_grid,
    perturbing_simulator,
    success,
)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _run(planner, *, h_exec=2, validator=None, resume=None, recall=None, stop=None,
         policy=None, starts=STARTS, goals=GOALS, max_steps=10,
         simulator_cls=MAPFStepSimulator, logger=None):
    config = GlobalHybridConfig(
        h_exec=h_exec,
        resume_policy=resume or AlwaysResumeIL(),
        recall_policy=recall or AllowAllRecalls(),
        safe_stop_handler=stop or RecordSafeStop(),
    )
    return run_global_hybrid_episode(
        policy or ScriptedPolicy(),
        open_grid(), starts, goals, max_steps,
        planner=planner,
        path_validator=validator or AcceptAllValidator(),
        config=config,
        monitor=DeadlockMonitor(),
        logger=logger,
        simulator_cls=simulator_cls,
    )


def _route(result):
    return [(tr["from"], tr["to"]) for tr in result["transitions"]]


class TestSuccess(unittest.TestCase):

    def test_cbs_exec_reaches_goals(self):
        planner = RecordingPlanner(success(PATHS_TO_GOALS))
        result = _run(planner, h_exec=2)
        self.assertEqual(result["termination"], "SUCCESS")
        self.assertTrue(result["all_at_goal"])
        self.assertEqual(result["steps"], 3)
        self.assertEqual(result["action_sources"], ["IL", "CBS", "CBS"])
        self.assertEqual(_route(result), [
            ("IL", "TRIGGER"), ("TRIGGER", "CBS_REQUEST"), ("CBS_REQUEST", "CBS_EXEC"),
        ])
        self.assertEqual(len(planner.calls), 1)
        self.assertEqual(result["cbs_calls"][0]["latency_ms"], 250.0)
        self.assertTrue(result["cbs_calls"][0]["accepted"])

    def test_h_exec_window_then_il_resume(self):
        result = _run(RecordingPlanner(success(PATHS_TO_GOALS)), h_exec=1, max_steps=4)
        self.assertEqual(result["action_sources"], ["IL", "CBS", "IL", "IL"])
        self.assertEqual(_route(result)[-2:], [("CBS_EXEC", "IL_RESUME"), ("IL_RESUME", "IL")])
        self.assertEqual(result["termination"], "TIMEOUT")
        # Only Path[1] was executed; IL (all WAIT) holds it afterwards.
        self.assertEqual(result["final_positions"], {0: (0, 1), 1: (2, 1)})


class TestTriggerStateAndFirstMove(unittest.TestCase):

    def test_planner_receives_actual_positions_at_trigger(self):
        # Agent 0 moves RIGHT once, then everyone WAITs: S_1 repeats at t=2,
        # so S_t = {0: (0, 1), 1: (2, 0)} -- not the episode starts.
        paths = [[(0, 1), (0, 2)], [(2, 0), (2, 1), (2, 2)]]
        planner = RecordingPlanner(success(paths))
        result = _run(planner, policy=ScriptedPolicy({0: [RIGHT]}))
        self.assertEqual(result["cbs_calls"][0]["trigger_timestep"], 2)
        self.assertEqual(planner.calls[0]["current_positions"], [(0, 1), (2, 0)])
        self.assertEqual(planner.calls[0]["goals"], [(0, 2), (2, 2)])
        self.assertEqual(result["termination"], "SUCCESS")

    def test_first_cbs_move_targets_path1(self):
        goals = {0: (0, 2), 1: (1, 0)}
        paths = [[(0, 0), (0, 1), (0, 2)], [(2, 0), (1, 0)]]
        result = _run(RecordingPlanner(success(paths)), goals=goals, h_exec=1, max_steps=2)
        self.assertEqual(result["actions"][1], {0: RIGHT, 1: UP})
        self.assertEqual(result["trajectory"][2], {0: (0, 1), 1: (1, 0)})


class TestAgentIdMapping(unittest.TestCase):

    STARTS = {3: (0, 0), 7: (2, 0)}
    GOALS = {3: (0, 2), 7: (1, 0)}
    PATHS = [[(0, 0), (0, 1), (0, 2)], [(2, 0), (1, 0)]]

    def test_non_contiguous_ids_follow_input_order(self):
        planner = RecordingPlanner(success(self.PATHS))
        result = _run(planner, starts=self.STARTS, goals=self.GOALS)
        self.assertEqual(planner.calls[0]["current_positions"], [(0, 0), (2, 0)])
        self.assertEqual(result["actions"][1], {3: RIGHT, 7: UP})
        self.assertEqual(result["termination"], "SUCCESS")
        self.assertEqual(result["final_positions"], self.GOALS)

    def test_swapped_index_order_refused_before_any_move(self):
        result = _run(ReversedOrderPlanner(success(self.PATHS)),
                      starts=self.STARTS, goals=self.GOALS)
        self.assertEqual(result["termination"], "SAFE_STOP")
        self.assertEqual(result["stop_reason"], "BOUNDARY_START_MISMATCH")
        self.assertNotIn("CBS", result["action_sources"])


class TestCbsFailureNoMoves(unittest.TestCase):

    def _assert_stopped_without_cbs_moves(self, result, reason, stop):
        self.assertEqual(result["termination"], "SAFE_STOP")
        self.assertEqual(result["stop_reason"], reason)
        self.assertEqual(result["steps"], 1)               # only the IL step that triggered
        self.assertEqual(len(result["actions"]), 1)
        self.assertNotIn("CBS", result["action_sources"])
        self.assertEqual(result["final_positions"], STARTS)
        self.assertEqual([c[0] for c in stop.calls], [reason])
        self.assertEqual(result["safe_stop"], {"reason": reason, "t": 1})

    def test_five_failure_statuses_with_valid_looking_paths(self):
        for status in ("timeout", "no_solution", "invalid_path", "input_error", "solver_error"):
            with self.subTest(status=status):
                stop = RecordSafeStop()
                planner = RecordingPlanner(MockReplanResult(status=status, paths=PATHS_TO_GOALS))
                result = _run(planner, stop=stop)
                self._assert_stopped_without_cbs_moves(result, f"CBS_{status.upper()}", stop)
                self.assertEqual(result["cbs_calls"][0]["status"], status)

    def test_planner_exception(self):
        stop = RecordSafeStop()
        planner = RaisingPlanner()
        result = _run(planner, stop=stop)
        self._assert_stopped_without_cbs_moves(result, "PLANNER_EXCEPTION", stop)
        self.assertEqual(planner.calls, 1)

    def test_path_guard_reject(self):
        stop = RecordSafeStop()
        validator = RejectAllValidator("mock_reject")
        result = _run(RecordingPlanner(success(PATHS_TO_GOALS)), validator=validator, stop=stop)
        self._assert_stopped_without_cbs_moves(result, "VALIDATOR_REJECT:mock_reject", stop)
        self.assertEqual(validator.calls, 1)

    def test_illegal_jump_refused_not_turned_into_wait(self):
        stop = RecordSafeStop()
        paths = [[(0, 0), (0, 2)], [(2, 0), (2, 1)]]
        result = _run(RecordingPlanner(success(paths)), stop=stop)
        self._assert_stopped_without_cbs_moves(result, "BOUNDARY_NON_ADJACENT", stop)


class TestExecutionMismatchAndOverlap(unittest.TestCase):

    def test_mismatch_stops_next_move(self):
        # After CBS step 1 (t=2) agent 0 is pushed to (1,1) instead of Path[1]=(0,1).
        sim_cls = perturbing_simulator(inject_t=2, agent_id=0, new_pos=(1, 1))
        result = _run(RecordingPlanner(success(PATHS_TO_GOALS)), h_exec=2, simulator_cls=sim_cls)
        self.assertEqual(result["termination"], "SAFE_STOP")
        self.assertEqual(result["stop_reason"], "EXEC_MISMATCH")
        self.assertEqual(result["action_sources"], ["IL", "CBS"])   # CBS step 2 never issued
        self.assertEqual(result["steps"], 2)
        self.assertEqual(result["final_positions"], {0: (1, 1), 1: (2, 1)})

    def test_overlap_during_cbs_exec_aborts(self):
        sim_cls = perturbing_simulator(inject_t=2, agent_id=1, new_pos=(0, 1))
        logger = EpisodeLogger("hybrid_overlap", scenario="synthetic", mode="GLOBAL_HYBRID")
        result = _run(RecordingPlanner(success(PATHS_TO_GOALS)), h_exec=2,
                      simulator_cls=sim_cls, logger=logger)
        self.assertEqual(result["termination"], "INTEGRITY_ABORT")
        self.assertTrue(result["integrity_abort"])
        self.assertEqual(result["integrity_abort_timestep"], 2)
        self.assertEqual(result["action_sources"], ["IL", "CBS"])
        last_rows = [r for r in logger.step_rows if r["timestep"] == 2]
        self.assertEqual({r["failure_reason"] for r in last_rows}, {"integrity_overlap"})


class TestDuplicateTriggerPrevention(unittest.TestCase):

    def test_stale_trigger_code_never_recalls_cbs(self):
        recall = AllowAllRecalls()
        planner = RecordingPlanner(success(PATHS_TO_GOALS))
        logger = EpisodeLogger("hybrid_recall", scenario="synthetic", mode="GLOBAL_HYBRID")
        result = _run(planner, h_exec=1, recall=recall, max_steps=8, logger=logger)
        # The Monitor keeps reporting TRIGGER_FULL_REPEAT after t=1 ...
        later = {r["trigger_reason"] for r in logger.step_rows if r["timestep"] > 1}
        self.assertEqual(later, {"TRIGGER_FULL_REPEAT"})
        # ... yet CBS was requested exactly once, for the t=1 Trigger.
        self.assertEqual(len(planner.calls), 1)
        self.assertEqual(recall.asked, [(0, 1)])
        self.assertEqual(result["action_sources"].count("CBS"), 1)
        self.assertEqual(result["termination"], "TIMEOUT")

    def test_recall_policy_can_deny(self):
        result = _run(FailIfCalledPlanner(), recall=DenyAllRecalls(), max_steps=5)
        self.assertEqual(result["cbs_calls"], [])
        self.assertEqual(result["action_sources"], ["IL"] * 5)
        self.assertEqual(result["termination"], "TIMEOUT")


class TestSafeStopIssuesNoMoves(unittest.TestCase):

    def test_resume_policy_safe_stop(self):
        stop = RecordSafeStop()
        result = _run(RecordingPlanner(success(PATHS_TO_GOALS)), h_exec=1,
                      resume=AlwaysSafeStopAfterExec(), stop=stop, max_steps=10)
        self.assertEqual(result["termination"], "SAFE_STOP")
        self.assertEqual(result["stop_reason"], "RESUME_POLICY_SAFE_STOP")
        self.assertEqual(result["action_sources"], ["IL", "CBS"])
        self.assertEqual(len(result["trajectory"]), 3)
        self.assertEqual(len(stop.calls), 1)

    def test_safe_stop_makes_no_further_simulator_steps(self):
        step_calls = []

        class _CountingSimulator(MAPFStepSimulator):
            def step(self, actions):
                step_calls.append(dict(actions))
                return super().step(actions)

        result = _run(RecordingPlanner(MockReplanResult(status="timeout")),
                      simulator_cls=_CountingSimulator)
        self.assertEqual(result["termination"], "SAFE_STOP")
        self.assertEqual(step_calls, [{0: WAIT, 1: WAIT}])


class TestConfigAndGuards(unittest.TestCase):

    def test_config_has_no_defaults(self):
        with self.assertRaises(TypeError):
            GlobalHybridConfig()
        with self.assertRaises(TypeError):
            GlobalHybridConfig(h_exec=1)

    def test_h_exec_must_be_positive_int(self):
        for bad in (0, -1, 1.5, None):
            with self.subTest(h_exec=bad), self.assertRaises(ValueError):
                GlobalHybridConfig(h_exec=bad, resume_policy=AlwaysResumeIL(),
                                   recall_policy=DenyAllRecalls(),
                                   safe_stop_handler=RecordSafeStop())

    def test_monitor_required(self):
        config = GlobalHybridConfig(h_exec=1, resume_policy=AlwaysResumeIL(),
                                    recall_policy=DenyAllRecalls(),
                                    safe_stop_handler=RecordSafeStop())
        with self.assertRaises(ValueError):
            run_global_hybrid_episode(
                ScriptedPolicy(), open_grid(), STARTS, GOALS, 3,
                planner=FailIfCalledPlanner(), path_validator=AcceptAllValidator(),
                config=config, monitor=None,
            )

    def test_accept_all_validator_refused_with_real_planner(self):
        def real_like_planner(current_positions, goals, grid):
            raise AssertionError("not called")
        with self.assertRaises(ValueError):
            _run(real_like_planner, validator=AcceptAllValidator())

    def test_validator_required(self):
        config = GlobalHybridConfig(h_exec=1, resume_policy=AlwaysResumeIL(),
                                    recall_policy=DenyAllRecalls(),
                                    safe_stop_handler=RecordSafeStop())
        with self.assertRaises(ValueError):
            run_global_hybrid_episode(
                ScriptedPolicy(), open_grid(), STARTS, GOALS, 3,
                planner=FailIfCalledPlanner(), path_validator=None,
                config=config, monitor=DeadlockMonitor(),
            )


class TestBaselineIlTrajectoryPreserved(unittest.TestCase):
    """Without a CBS request the hybrid runner must reproduce run_il.run_episode."""

    KEYS = ("steps", "done", "all_at_goal", "timed_out", "final_positions", "trajectory", "actions")

    def _compare(self, make_policy, grid, starts, goals, max_steps):
        baseline = run_episode(make_policy(), grid, starts, goals, max_steps, monitor=DeadlockMonitor())
        config = GlobalHybridConfig(h_exec=1, resume_policy=AlwaysResumeIL(),
                                    recall_policy=DenyAllRecalls(),
                                    safe_stop_handler=RecordSafeStop())
        hybrid = run_global_hybrid_episode(
            make_policy(), grid, starts, goals, max_steps,
            planner=FailIfCalledPlanner(), path_validator=AcceptAllValidator(),
            config=config, monitor=DeadlockMonitor(),
        )
        for key in self.KEYS:
            self.assertEqual(hybrid[key], baseline[key], key)
        self.assertEqual(hybrid["cbs_calls"], [])
        return hybrid

    def test_no_trigger_success(self):
        hybrid = self._compare(lambda: ScriptedPolicy({0: [RIGHT, RIGHT], 1: [RIGHT, RIGHT]}),
                               open_grid(), STARTS, GOALS, 10)
        self.assertEqual(hybrid["termination"], "SUCCESS")

    def test_trigger_ignored_when_recall_denied(self):
        hybrid = self._compare(ScriptedPolicy, open_grid(), STARTS, GOALS, 6)
        self.assertEqual(hybrid["termination"], "TIMEOUT")

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_real_checkpoint_scenarios(self):
        from src.il.policy import NavHintPolicy
        for scenario, max_steps in (
            ("scenarios/smoke/scenario_empty_s11_n3.json", 160),
            ("scenarios/controlled/scenario_corridor_bay_n2.json", 20),
            ("scenarios/controlled/scenario_bottleneck_n4.json", 20),
        ):
            with self.subTest(scenario=scenario):
                grid, starts, goals = load_scenario(scenario)
                self._compare(lambda: NavHintPolicy(DEFAULT_CKPT), grid, starts, goals, max_steps)


if __name__ == "__main__":
    unittest.main()
