
"""Integration tests for Safety-only v0 and DeadlockMonitor."""

import unittest

from scripts.run_il import load_scenario, run_episode
from src.common.deadlock_monitor import DeadlockMonitor
from src.common.safety_only import SafetyOnlyV0
from src.il.nav_hint import bfs_dist


class FixedCorridorPolicy:
    """Fixed actions for controlled trigger-timing verification."""

    def reset(self, grid, goals):
        self._dist = {
            aid: bfs_dist(grid, goal)
            for aid, goal in goals.items()
        }

    def act(self, obs, positions):
        return {0: 3, 1: 2}  # RIGHT, LEFT


class TestSafetyIntegration(unittest.TestCase):

    def setUp(self):
        self.grid, self.starts, self.goals = load_scenario(
            "scenarios/controlled/scenario_corridor_bay_n2.json"
        )

    def test_wait_starts_after_trigger(self):
        monitor = DeadlockMonitor()
        safety = SafetyOnlyV0(hold_steps=2)

        result = run_episode(
            FixedCorridorPolicy(),
            self.grid,
            self.starts,
            self.goals,
            max_steps=8,
            monitor=monitor,
            safety=safety,
        )

        # First full-repeat should occur at t=5.
        self.assertEqual(monitor.result.trigger_timestep, 5)
        self.assertTrue(safety.triggered)
        self.assertEqual(safety.selected_agent, 0)

        # actions[4] corresponds to t=5.
        self.assertEqual(result["actions"][4][0], 3)

        # Intervention starts at the NEXT timestep.
        self.assertEqual(result["actions"][5][0], 4)
        self.assertEqual(result["actions"][6][0], 4)

        # After two WAIT steps, original action returns.
        self.assertEqual(result["actions"][7][0], 3)

        # Other robot keeps its original IL action.
        self.assertEqual(result["actions"][5][1], 2)

    def test_requires_monitor(self):
        with self.assertRaises(ValueError):
            run_episode(
                FixedCorridorPolicy(),
                self.grid,
                self.starts,
                self.goals,
                max_steps=3,
                safety=SafetyOnlyV0(),
            )


if __name__ == "__main__":
    unittest.main()
