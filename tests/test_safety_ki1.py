
"""Reproduce the known KI-1 collision-resolution limitation."""

import unittest

from scripts.run_il import load_scenario
from src.simulator.mapf_step_simulator import MAPFStepSimulator


class TestSafetyKI1Characterization(unittest.TestCase):

    def test_bottleneck_cascading_vertex_overlap(self):
        grid, _, goals = load_scenario(
            "scenarios/controlled/scenario_bottleneck_n4.json"
        )

        # Positions recorded at t=8 in Safety-only v0.
        positions = {
            0: (2, 5),
            1: (3, 6),
            2: (1, 7),
            3: (3, 7),
        }

        # Actions applied at t=9.
        actions = {
            0: 0,
            1: 3,
            2: 1,
            3: 0,
        }

        sim = MAPFStepSimulator(
            cbs_solver_root=None,
            max_steps=20,
        )
        sim.reset(grid, positions, goals)
        sim.step(actions)

        # Reproduce existing KI-1 behavior.
        self.assertEqual(sim._positions[1], (3, 7))
        self.assertEqual(sim._positions[3], (3, 7))

        self.assertLess(
            len(set(sim._positions.values())),
            len(sim._positions),
        )


if __name__ == "__main__":
    unittest.main()
