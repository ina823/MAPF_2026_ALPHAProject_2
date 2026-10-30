import unittest
import numpy as np

from src.simulator.mapf_step_simulator import MAPFStepSimulator


class TestSimulatorKI1(unittest.TestCase):

    def test_no_overlap_after_collision_cancel(self):

        # 1. 테스트용 3x3 지도 생성
        grid = np.zeros((3, 3), dtype=int)

        starts = {
            0: (1, 0),
            1: (1, 1),
            2: (0, 2),
        }

        goals = {
            0: (2, 2),
            1: (2, 1),
            2: (0, 0),
        }

        # 2. Simulator 초기화
        sim = MAPFStepSimulator(
            cbs_solver_root=None,
            max_steps=10,
        )

        sim.reset(grid, starts, goals)

        # 3. 충돌을 유발하는 Action 실행
        actions = {
            0: 3,  # RIGHT
            1: 3,  # RIGHT
            2: 1,  # DOWN
        }

        sim.step(actions)

        # 4. 실제 최종 위치 확인
        positions = dict(sim._positions)

        # 5. 위치 중복 검사
        self.assertEqual(
            len(positions),
            len(set(positions.values())),
            f"KI-1 overlap detected: {positions}",
        )


if __name__ == "__main__":
    unittest.main()
