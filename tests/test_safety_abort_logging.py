
"""Regression tests for Safety-only integrity abort and logging.

Before the KI-1 fix (32e2158) the real Bottleneck run overlapped at t=9 and
was the live evidence for integrity_abort (docs/safety_only_v0_results.md).
After the fix no real scenario overlaps, so the abort path is exercised by
fault injection (_OverlapInjectingSimulator) instead; the run_il.py abort
check itself is unchanged.
"""

import unittest
from unittest import mock

from scripts.run_il import (
    DEFAULT_CKPT,
    load_scenario,
    run_episode,
)
from src.common.deadlock_monitor import DeadlockMonitor
from src.common.episode_logger import EpisodeLogger
from src.common.safety_only import SafetyOnlyV0
from src.il.nav_hint import bfs_dist
from src.il.policy import NavHintPolicy
from src.simulator.mapf_step_simulator import MAPFStepSimulator

UP, DOWN, LEFT, RIGHT, WAIT = 0, 1, 2, 3, 4


class _FixedActionPolicy:
    """Torch-free stand-in for NavHintPolicy: replays fixed actions and
    builds `_dist` with bfs_dist(), which run_episode() reads when a
    monitor is attached."""

    def __init__(self, actions_per_agent):
        self._actions_per_agent = actions_per_agent
        self._t = 0

    def reset(self, grid, goals):
        self._dist = {aid: bfs_dist(grid, goal) for aid, goal in goals.items()}
        self._t = 0

    def act(self, obs, positions):
        actions = {
            aid: (seq[self._t] if self._t < len(seq) else WAIT)
            for aid, seq in self._actions_per_agent.items()
        }
        self._t += 1
        return actions


def _overlap_injecting_simulator(inject_t):
    """Fault injection: after the real step() at timestep `inject_t`, force
    agent 1 onto agent 0's cell. inject_t=None injects nothing."""

    class _OverlapInjectingSimulator(MAPFStepSimulator):
        def step(self, actions):
            obs, done, info = super().step(actions)
            if inject_t is not None and info["t"] == inject_t:
                self._positions[1] = self._positions[0]
            return obs, done, info

    return _OverlapInjectingSimulator


class TestSafetyAbortLogging(unittest.TestCase):

    def test_bottleneck_no_integrity_abort_after_ki1_fix(self):
        """Formerly test_bottleneck_integrity_abort_and_csv_rows (abort at
        t=9, agents 1 and 3 on (3,7)). Since 32e2158 agent 1's cancelled
        RIGHT also cancels agent 3's move, so there is no overlap: the run
        now deadlocks and times out at t=20 without an integrity abort."""
        grid, starts, goals = load_scenario(
            "scenarios/controlled/scenario_bottleneck_n4.json"
        )

        policy = NavHintPolicy(DEFAULT_CKPT)
        monitor = DeadlockMonitor()
        safety = SafetyOnlyV0(hold_steps=2)

        logger = EpisodeLogger(
            "test_safety_abort",
            scenario="scenario_bottleneck_n4.json",
            mode="SAFETY_V0",
        )

        result = run_episode(
            policy,
            grid,
            starts,
            goals,
            max_steps=20,
            logger=logger,
            monitor=monitor,
            safety=safety,
        )

        self.assertFalse(result["integrity_abort"])
        self.assertIsNone(result["integrity_abort_timestep"])
        self.assertEqual(result["steps"], 20)
        self.assertFalse(result["all_at_goal"])
        self.assertTrue(result["timed_out"])
        self.assertEqual(monitor.result.overlap_integrity_status, "CLEAN")

        # Safety-only still armed at the t=2 Trigger (agent 0 WAITs t=3,4).
        self.assertEqual(monitor.result.trigger_timestep, 2)
        self.assertTrue(safety.triggered)
        self.assertEqual(safety.selected_agent, 0)
        self.assertEqual(result["actions"][2][0], WAIT)
        self.assertEqual(result["actions"][3][0], WAIT)

        # The former t=9 overlap is gone.
        t9 = result["trajectory"][9]
        self.assertEqual(t9[1], (3, 6))
        self.assertEqual(t9[3], (3, 7))

        self.assertTrue(all(
            row["failure_reason"] == ("timeout" if row["timestep"] == 20 else "")
            for row in logger.step_rows
        ))


class TestSafetyIntegrityAbortFaultInjection(unittest.TestCase):
    """Fault injection keeps the run_il.py integrity_abort path covered now
    that the real simulator no longer produces overlaps (KI-1 fixed)."""

    SCENARIO = "scenarios/controlled/scenario_corridor_bay_n2.json"
    ACTIONS = {0: [RIGHT] * 8, 1: [LEFT] * 8}  # Full-repeat Trigger at t=5

    def _run(self, inject_t):
        grid, starts, goals = load_scenario(self.SCENARIO)
        monitor = DeadlockMonitor()
        safety = SafetyOnlyV0(hold_steps=2)
        logger = EpisodeLogger(
            "test_fault_injection",
            scenario="scenario_corridor_bay_n2.json",
            mode="SAFETY_V0",
        )
        with mock.patch(
            "scripts.run_il.MAPFStepSimulator",
            _overlap_injecting_simulator(inject_t),
        ):
            result = run_episode(
                _FixedActionPolicy(self.ACTIONS),
                grid,
                starts,
                goals,
                max_steps=8,
                logger=logger,
                monitor=monitor,
                safety=safety,
            )
        return result, monitor, safety, logger

    def test_injected_overlap_triggers_integrity_abort_and_csv_rows(self):
        result, monitor, safety, logger = self._run(inject_t=3)

        self.assertTrue(result["integrity_abort"])
        self.assertEqual(result["integrity_abort_timestep"], 3)
        self.assertEqual(result["steps"], 3)
        self.assertFalse(result["all_at_goal"])
        self.assertFalse(result["timed_out"])
        self.assertEqual(monitor.result.overlap_integrity_status, "INTEGRITY_ISSUE")
        self.assertEqual(monitor.result.overlap_timesteps, 1)

        final_rows = [row for row in logger.step_rows if row["timestep"] == 3]
        self.assertEqual(len(final_rows), 2)
        self.assertTrue(all(
            row["failure_reason"] == "integrity_overlap" for row in final_rows
        ))

        earlier_rows = [row for row in logger.step_rows if row["timestep"] < 3]
        self.assertTrue(earlier_rows)
        self.assertTrue(all(row["failure_reason"] == "" for row in earlier_rows))
        self.assertEqual(max(row["timestep"] for row in logger.step_rows), 3)

    def test_control_without_injection_does_not_abort(self):
        result, monitor, safety, logger = self._run(inject_t=None)

        self.assertFalse(result["integrity_abort"])
        self.assertIsNone(result["integrity_abort_timestep"])
        self.assertEqual(result["steps"], 8)
        self.assertEqual(monitor.result.overlap_integrity_status, "CLEAN")
        self.assertEqual(monitor.result.trigger_timestep, 5)
        self.assertTrue(safety.triggered)
        self.assertFalse(any(
            row["failure_reason"] == "integrity_overlap"
            for row in logger.step_rows
        ))

    def test_overlap_at_would_be_trigger_step_aborts_before_safety_arms(self):
        """An injected overlap at t=5 (the control run's Trigger step)
        changes the joint state, so no Trigger fires there; the run must
        abort at t=5 and Safety-only must never arm or override actions."""
        result, monitor, safety, logger = self._run(inject_t=5)

        self.assertTrue(result["integrity_abort"])
        self.assertEqual(result["integrity_abort_timestep"], 5)
        self.assertEqual(monitor.result.overlap_integrity_status, "INTEGRITY_ISSUE")
        self.assertFalse(safety.triggered)
        self.assertTrue(all(
            actions[0] == RIGHT and actions[1] == LEFT
            for actions in result["actions"]
        ))


if __name__ == "__main__":
    unittest.main()
