
import unittest

from src.path_guard import (
    validate_path_movements,
    validate_path_endpoints,
    validate_path_map_and_conflicts,
    validate_path_guard,
    InvalidPathError,
)


class TestPathGuard(unittest.TestCase):

    def test_valid_path(self):
        paths = {
            0: [(0, 0), (0, 1), (0, 2), (0, 2)]
        }
        validate_path_movements(paths)

    def test_invalid_jump(self):
        paths = {
            0: [(0, 0), (0, 2)]
        }
        with self.assertRaises(InvalidPathError):
            validate_path_movements(paths)

    def test_invalid_diagonal(self):
        paths = {
            0: [(0, 0), (1, 1)]
        }
        with self.assertRaises(InvalidPathError):
            validate_path_movements(paths)

    def test_valid_endpoints(self):
        paths = {0: [(0, 0), (0, 1)]}
        starts = {0: (0, 0)}
        goals = {0: (0, 1)}

        validate_path_endpoints(paths, starts, goals)

    def test_invalid_start(self):
        paths = {0: [(1, 0), (0, 1)]}
        starts = {0: (0, 0)}
        goals = {0: (0, 1)}

        with self.assertRaises(InvalidPathError):
            validate_path_endpoints(paths, starts, goals)

    def test_invalid_goal(self):
        paths = {0: [(0, 0), (0, 1)]}
        starts = {0: (0, 0)}
        goals = {0: (2, 2)}

        with self.assertRaises(InvalidPathError):
            validate_path_endpoints(paths, starts, goals)

    def test_missing_agent(self):
        paths = {0: [(0, 0), (0, 1)]}
        starts = {0: (0, 0), 1: (2, 0)}
        goals = {0: (0, 1), 1: (2, 1)}

        with self.assertRaises(InvalidPathError):
            validate_path_endpoints(paths, starts, goals)

    def test_extra_agent(self):
        paths = {
            0: [(0, 0), (0, 1)],
            1: [(2, 0), (2, 1)],
        }
        starts = {0: (0, 0)}
        goals = {0: (0, 1)}

        with self.assertRaises(InvalidPathError):
            validate_path_endpoints(paths, starts, goals)

    def test_valid_map_and_conflicts(self):
        grid = [[0] * 3 for _ in range(3)]

        paths = {
            0: [(1,0), (1,1), (1,2), (1,2), (1,2)],
            1: [(1,2), (0,2), (0,1), (1,1), (1,0)]
        }

        validate_path_map_and_conflicts(paths, grid)

    def test_out_of_bounds(self):
        grid = [[0] * 3 for _ in range(3)]
        paths = {0: [(0,0), (-1,0)]}

        with self.assertRaises(InvalidPathError):
            validate_path_map_and_conflicts(paths, grid)

    def test_obstacle(self):
        grid = [[0,1], [0,0]]
        paths = {0: [(0,0), (0,1)]}

        with self.assertRaises(InvalidPathError):
            validate_path_map_and_conflicts(paths, grid)

    def test_vertex_conflict(self):
        grid = [[0] * 3 for _ in range(3)]

        paths = {
            0: [(0,0), (0,1)],
            1: [(0,2), (0,1)]
        }

        with self.assertRaises(InvalidPathError):
            validate_path_map_and_conflicts(paths, grid)

    def test_edge_conflict(self):
        grid = [[0] * 3 for _ in range(3)]

        paths = {
            0: [(1,0), (1,1)],
            1: [(1,1), (1,0)]
        }

        with self.assertRaises(InvalidPathError):
            validate_path_map_and_conflicts(paths, grid)

    def test_path_length_mismatch(self):
        grid = [[0] * 3 for _ in range(3)]

        paths = {
            0: [(0,0), (0,1)],
            1: [(2,0), (2,1), (2,2)]
        }

        with self.assertRaises(InvalidPathError):
            validate_path_map_and_conflicts(paths, grid)


    def test_integrated_valid_path(self):
        grid = [[0] * 3 for _ in range(3)]

        paths = {
            0: [(1,0), (1,1), (1,2), (1,2), (1,2)],
            1: [(1,2), (0,2), (0,1), (1,1), (1,0)]
        }

        starts = {0: (1,0), 1: (1,2)}
        goals = {0: (1,2), 1: (1,0)}

        validate_path_guard(paths, starts, goals, grid)

    def test_integrated_invalid_jump(self):
        grid = [[0] * 3 for _ in range(3)]

        paths = {0: [(0,0), (0,2)]}
        starts = {0: (0,0)}
        goals = {0: (0,2)}

        with self.assertRaises(InvalidPathError):
            validate_path_guard(paths, starts, goals, grid)

    def test_integrated_vertex_conflict(self):
        grid = [[0] * 3 for _ in range(3)]

        paths = {
            0: [(0,0), (0,1)],
            1: [(0,2), (0,1)]
        }

        starts = {0: (0,0), 1: (0,2)}
        goals = {0: (0,1), 1: (0,1)}

        with self.assertRaises(InvalidPathError):
            validate_path_guard(paths, starts, goals, grid)


if __name__ == "__main__":
    unittest.main()
