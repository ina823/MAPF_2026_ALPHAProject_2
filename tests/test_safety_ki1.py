
"""KI-1 Bottleneck regression after collision-resolution fix."""

import unittest
from pathlib import Path

from src.common.scenario_loader import load_map_generator_scenario
from src.simulator.mapf_step_simulator import MAPFStepSimulator


class TestSafetyKI1Regression(unittest.TestCase):

    def test_bottleneck_cascading_vertex_no_overlap(self):
        scenario = (
            Path(__file__).resolve().parents[1]
            / "scenarios"
            / "controlled"
            / "scenario_bottleneck_n4.json"
        )

        grid, _, goal_list = load_map_generator_scenario(scenario)
        goals = dict(enumerate(goal_list))

        # Bottleneck 원래 실행의 t=8 위치
        positions = {
            0: (2, 5),
            1: (3, 6),
            2: (1, 7),
            3: (3, 7),
        }

        # t=9에서 제안된 Action
        actions = {
            0: 0,  # UP
            1: 3,  # RIGHT
            2: 1,  # DOWN
            3: 0,  # UP
        }

        sim = MAPFStepSimulator(
            cbs_solver_root=None,
            max_steps=20,
        )

        sim.reset(grid, positions, goals)
        sim.step(actions)

        # KI-1 수정 후 기대 위치
        self.assertEqual(sim._positions[1], (3, 6))
        self.assertEqual(sim._positions[3], (3, 7))

        # 모든 로봇의 최종 위치는 서로 달라야 함
        self.assertEqual(
            len(set(sim._positions.values())),
            len(sim._positions),
        )


if __name__ == "__main__":
    unittest.main()
