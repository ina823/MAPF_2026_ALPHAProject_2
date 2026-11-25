import unittest

from src.ros_bridge.grid_frame import (
    grid_to_world,
    world_to_grid,
    is_at_target,
)


class TestGridFrame(unittest.TestCase):

    def test_grid_to_world(self):
        self.assertEqual(
            grid_to_world(2, 3),
            (3.0, -2.0)
        )

    def test_world_to_grid(self):
        self.assertEqual(
            world_to_grid(3.0, -2.0),
            (2, 3)
        )

    def test_round_trip(self):
        world = grid_to_world(2, 3)
        self.assertEqual(
            world_to_grid(*world),
            (2, 3)
        )

    def test_custom_origin_and_size(self):
        world = grid_to_world(2, 3, 0.5, 10.0, 5.0)
        self.assertEqual(
            world_to_grid(*world, 0.5, 10.0, 5.0),
            (2, 3)
        )

    def test_target_reached(self):
        self.assertTrue(
            is_at_target(3.02, -1.98, 3.0, -2.0)
        )

    def test_target_not_reached(self):
        self.assertFalse(
            is_at_target(2.9, -2.0, 3.0, -2.0)
        )

    def test_invalid_cell_size(self):
        for size in (0, -1, float("inf"), float("nan")):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    grid_to_world(2, 3, size)

                with self.assertRaises(ValueError):
                    world_to_grid(3, -2, size)

    def test_invalid_tolerance(self):
        for tolerance in (-1, float("inf"), float("nan")):
            with self.subTest(tolerance=tolerance):
                with self.assertRaises(ValueError):
                    is_at_target(0, 0, 0, 0, tolerance)


if __name__ == "__main__":
    unittest.main()
