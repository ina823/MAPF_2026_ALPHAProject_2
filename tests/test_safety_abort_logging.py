
"""Regression tests for Safety-only integrity abort and logging."""

import unittest

from scripts.run_il import (
    DEFAULT_CKPT,
    load_scenario,
    run_episode,
)
from src.common.deadlock_monitor import DeadlockMonitor
from src.common.episode_logger import EpisodeLogger
from src.common.safety_only import SafetyOnlyV0
from src.il.policy import NavHintPolicy


class TestSafetyAbortLogging(unittest.TestCase):

    def test_bottleneck_integrity_abort_and_csv_rows(self):
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

        # The known KI-1 case must stop at t=9.
        self.assertTrue(result["integrity_abort"])
        self.assertEqual(
            result["integrity_abort_timestep"], 9
        )
        self.assertEqual(result["steps"], 9)
        self.assertFalse(result["all_at_goal"])
        self.assertFalse(result["timed_out"])

        # All four agent rows at t=9 record the reason.
        final_rows = [
            row for row in logger.step_rows
            if row["timestep"] == 9
        ]

        self.assertEqual(len(final_rows), 4)
        self.assertTrue(all(
            row["failure_reason"] == "integrity_overlap"
            for row in final_rows
        ))

        # Earlier steps must not be mislabeled.
        earlier_rows = [
            row for row in logger.step_rows
            if row["timestep"] < 9
        ]

        self.assertTrue(all(
            row["failure_reason"] == ""
            for row in earlier_rows
        ))


if __name__ == "__main__":
    unittest.main()
