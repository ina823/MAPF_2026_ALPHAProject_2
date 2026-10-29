
import unittest
from unittest.mock import patch

from src.replanner import replan
from src.path_guard import InvalidPathError


class TestReplannerGuard(unittest.TestCase):

    def test_valid_cbs_result(self):
        grid = [[0] * 3 for _ in range(3)]

        valid_paths = {
            0: [(0, 0), (0, 1), (0, 2)]
        }

        with patch(
            "src.replanner.CBSAdapter.plan",
            return_value=valid_paths
        ):
            result = replan(
                [(0, 0)],
                [(0, 2)],
                grid
            )

        self.assertEqual(result, valid_paths)

    def test_invalid_cbs_result_blocked(self):
        grid = [[0] * 3 for _ in range(3)]

        # 한 timestep에 두 칸 점프하는 잘못된 CBS 결과
        invalid_paths = {
            0: [(0, 0), (0, 2)]
        }

        with patch(
            "src.replanner.CBSAdapter.plan",
            return_value=invalid_paths
        ):
            with self.assertRaises(InvalidPathError):
                replan(
                    [(0, 0)],
                    [(0, 2)],
                    grid
                )


if __name__ == "__main__":
    unittest.main()
