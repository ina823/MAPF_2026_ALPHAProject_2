"""KI-1 regression (RED): Bottleneck t=8 -> t=9 must preserve unique positions.

Run from the MAPF_2026_ALPHAProject_2 repository root:
    python -m pytest tests/test_ki1_regression_red.py -q

Unfixed simulator: expected FAIL (Agent 1 and Agent 3 overlap at (3,7)).
After a correct simulator fix: expected PASS.

Important: sim.reset() makes the LOCAL timestep start at 0. The one call to
sim.step() below recreates the historical t=8->9 transition, but info['t']=1.
"""

import unittest
from pathlib import Path

from src.common.scenario_loader import load_map_generator_scenario
from src.simulator.mapf_step_simulator import MAPFStepSimulator


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios" / "controlled" / "scenario_bottleneck_n4.json"


class TestKI1RegressionRed(unittest.TestCase):
    def test_bottleneck_t8_to_t9_no_final_overlap(self):
        grid, _starts, goal_list = load_map_generator_scenario(SCENARIO)
        goals = dict(enumerate(goal_list))

        # Actual Safety-only v0 snapshot at original t=8.
        positions_t8 = {
            0: (2, 5),
            1: (3, 6),
            2: (1, 7),
            3: (3, 7),
        }
        actions_at_t9 = {
            0: 0,  # UP
            1: 3,  # RIGHT: (3,6) -> (3,7)
            2: 1,  # DOWN:  (1,7) -> (2,7)
            3: 0,  # UP:    (3,7) -> (2,7)
        }
        self.assertEqual(len(set(positions_t8.values())), 4)

        sim = MAPFStepSimulator(cbs_solver_root=None, max_steps=20)
        sim.reset(grid, positions_t8, goals)
        _obs, _done, info = sim.step(actions_at_t9)

        # This is one isolated step, not a replay of the preceding 8 steps.
        self.assertEqual(info["t"], 1)
        final = sim._positions

        # Safety invariant: distinct agents must never occupy the same cell.
        self.assertEqual(
            len(set(final.values())),
            len(final),
            f"KI-1: Bottleneck t=8->9 final positions overlap: {final}",
        )


if __name__ == "__main__":
    unittest.main()
