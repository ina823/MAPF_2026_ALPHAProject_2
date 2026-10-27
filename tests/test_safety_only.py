
"""Unit tests for deterministic Safety-only v0."""

import unittest

from src.common.safety_only import SafetyOnlyV0


class TestSafetyOnlyV0(unittest.TestCase):

    def setUp(self):
        self.safety = SafetyOnlyV0(hold_steps=2)
        self.positions = {0: (2, 4), 1: (2, 6)}
        self.goals = {0: (2, 10), 1: (2, 0)}
        self.actions = {0: 3, 1: 2}

    def test_select_lowest_active_agent(self):
        self.assertTrue(
            self.safety.arm(self.positions, self.goals)
        )
        self.assertEqual(self.safety.selected_agent, 0)

    def test_wait_exactly_two_steps(self):
        self.safety.arm(self.positions, self.goals)

        self.assertEqual(
            self.safety.apply(self.actions),
            {0: 4, 1: 2}
        )
        self.assertEqual(
            self.safety.apply(self.actions),
            {0: 4, 1: 2}
        )
        self.assertEqual(
            self.safety.apply(self.actions),
            {0: 3, 1: 2}
        )

    def test_original_actions_unchanged(self):
        self.safety.arm(self.positions, self.goals)
        self.safety.apply(self.actions)

        self.assertEqual(self.actions, {0: 3, 1: 2})

    def test_skip_agent_at_goal(self):
        positions = {0: (2, 10), 1: (2, 6)}

        self.assertTrue(
            self.safety.arm(positions, self.goals)
        )
        self.assertEqual(self.safety.selected_agent, 1)

    def test_no_active_agent(self):
        self.assertFalse(
            self.safety.arm(self.goals, self.goals)
        )

    def test_trigger_only_once(self):
        self.assertTrue(
            self.safety.arm(self.positions, self.goals)
        )
        self.assertFalse(
            self.safety.arm(self.positions, self.goals)
        )

    def test_reset(self):
        self.safety.arm(self.positions, self.goals)
        self.safety.reset()

        self.assertIsNone(self.safety.selected_agent)
        self.assertEqual(self.safety.remaining, 0)
        self.assertFalse(self.safety.triggered)

    def test_invalid_hold_steps(self):
        with self.assertRaises(ValueError):
            SafetyOnlyV0(hold_steps=0)


if __name__ == "__main__":
    unittest.main()
