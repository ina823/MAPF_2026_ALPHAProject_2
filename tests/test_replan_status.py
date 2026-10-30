
import unittest
from unittest.mock import patch

from src.replan_status import replan_with_status
from src.cbs_adapter import (
    CBSTimeoutError,
    CBSNoSolutionError,
)
from src.path_guard import InvalidPathError


class TestReplanStatus(unittest.TestCase):

    def setUp(self):
        self.grid = [[0] * 3 for _ in range(3)]
        self.starts = [(0, 0)]
        self.goals = [(0, 2)]

    def run_with_error(self, error):
        with patch(
            "src.replan_status.replan",
            side_effect=error
        ):
            return replan_with_status(
                self.starts,
                self.goals,
                self.grid
            )

    def test_success(self):
        valid_paths = {
            0: [(0, 0), (0, 1), (0, 2)]
        }

        with patch(
            "src.replan_status.replan",
            return_value=valid_paths
        ):
            result = replan_with_status(
                self.starts,
                self.goals,
                self.grid
            )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.paths, valid_paths)
        self.assertIsNone(result.error)
        self.assertGreaterEqual(result.latency_sec, 0)

    def test_timeout(self):
        result = self.run_with_error(
            CBSTimeoutError("Timeout")
        )
        self.assertEqual(result.status, "timeout")
        self.assertIsNone(result.paths)

    def test_no_solution(self):
        result = self.run_with_error(
            CBSNoSolutionError("No solution")
        )
        self.assertEqual(result.status, "no_solution")
        self.assertIsNone(result.paths)

    def test_invalid_path(self):
        result = self.run_with_error(
            InvalidPathError("Illegal movement")
        )
        self.assertEqual(result.status, "invalid_path")
        self.assertIsNone(result.paths)

    def test_solver_error(self):
        result = self.run_with_error(
            RuntimeError("Solver failed")
        )
        self.assertEqual(result.status, "solver_error")
        self.assertIsNone(result.paths)

    def test_input_error(self):
        result = self.run_with_error(
            ValueError("Invalid input")
        )
        self.assertEqual(result.status, "input_error")
        self.assertIsNone(result.paths)


if __name__ == "__main__":
    unittest.main()
