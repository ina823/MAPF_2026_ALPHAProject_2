
import unittest
from unittest.mock import patch

import numpy as np

from src.cbs_runtime_adapter import run_cbs_runtime
from src.replan_status import ReplanResult


class TestCBSRuntimeAdapter(unittest.TestCase):

    def setUp(self):
        # 모든 테스트에서 공통으로 사용하는 환경
        self.grid = np.zeros((3, 3), dtype=int)

        self.starts = {
            7: (1, 2),
            2: (1, 0),
        }

        self.goals = {
            7: (1, 0),
            2: (1, 2),
        }

    def run_with_fake_result(self, fake_result):
        # 실제 CBS 호출을 가짜 결과로 교체
        with patch(
            "src.cbs_runtime_adapter.replan_with_status",
            return_value=fake_result,
        ):
            return run_cbs_runtime(
                self.starts,
                self.goals,
                self.grid,
            )

    # TEST 1: 정상 경로
    def test_success(self):
        fake = ReplanResult(
            status="success",
            paths={
                0: [
                    (1, 0), (1, 1), (1, 2),
                    (1, 2), (1, 2),
                ],
                1: [
                    (1, 2), (0, 2), (0, 1),
                    (1, 1), (1, 0),
                ],
            },
            latency_sec=0.1,
        )

        result = self.run_with_fake_result(fake)

        self.assertEqual(result.status, "success")
        self.assertEqual(set(result.paths), {2, 7})
        self.assertEqual(result.paths[2][0], (1, 0))
        self.assertEqual(result.paths[7][0], (1, 2))

    # TEST 2: CBS 시간초과
    def test_timeout(self):
        fake = ReplanResult(
            status="timeout",
            paths=None,
            latency_sec=30.0,
            error="Simulated timeout",
        )

        result = self.run_with_fake_result(fake)

        self.assertEqual(result.status, "timeout")
        self.assertIsNone(result.paths)

    # TEST 3: 잘못된 점프 경로
    def test_invalid_jump(self):
        fake = ReplanResult(
            status="success",
            paths={
                0: [(1, 0), (1, 2)],
                1: [(1, 2), (1, 0)],
            },
            latency_sec=0.1,
        )

        result = self.run_with_fake_result(fake)

        self.assertEqual(result.status, "invalid_path")
        self.assertIsNone(result.paths)

    # TEST 4: Agent ID 누락
    def test_missing_agent_id(self):
        fake = ReplanResult(
            status="success",
            paths={
                0: [(1, 0), (1, 1), (1, 2)],
            },
            latency_sec=0.1,
        )

        result = self.run_with_fake_result(fake)

        self.assertEqual(result.status, "invalid_path")
        self.assertIsNone(result.paths)

    # TEST 5: 로봇 위치 중복
    def test_overlapping_starts(self):
        starts = {
            2: (1, 0),
            7: (1, 0),
        }

        result = run_cbs_runtime(
            starts,
            self.goals,
            self.grid,
        )

        self.assertEqual(result.status, "input_error")
        self.assertIsNone(result.paths)


if __name__ == "__main__":
    unittest.main()
