import math
import unittest

from src.ros_bridge.cell_motion import (
    quaternion_to_yaw,
    target_yaw,
    normalize_angle,
    angle_error,
    compute_velocity,
)


class TestCellMotion(unittest.TestCase):

    def test_quaternion_zero(self):
        self.assertAlmostEqual(
            quaternion_to_yaw(0, 0, 0, 1), 0.0
        )

    def test_quaternion_90(self):
        q = math.sqrt(0.5)
        self.assertAlmostEqual(
            quaternion_to_yaw(0, 0, q, q),
            math.pi / 2
        )

    def test_target_yaw(self):
        self.assertAlmostEqual(
            target_yaw(0, 0, 1, 1),
            math.pi / 4
        )

    def test_normalize_angle(self):
        self.assertAlmostEqual(
            normalize_angle(math.radians(270)),
            math.radians(-90)
        )

    def test_angle_error_positive(self):
        self.assertAlmostEqual(
            angle_error(
                math.radians(170),
                math.radians(-170)
            ),
            math.radians(20)
        )

    def test_angle_error_negative(self):
        self.assertAlmostEqual(
            angle_error(
                math.radians(-170),
                math.radians(170)
            ),
            math.radians(-20)
        )

    def test_move_forward(self):
        self.assertEqual(
            compute_velocity(0, 0, 0, 1, 0),
            (0.3, 0.0, False)
        )

    def test_rotate_left(self):
        self.assertEqual(
            compute_velocity(0, 0, 0, 0, 1),
            (0.0, 0.5, False)
        )

    def test_rotate_right(self):
        self.assertEqual(
            compute_velocity(0, 0, 0, 0, -1),
            (0.0, -0.5, False)
        )

    def test_slow_down(self):
        linear, angular, reached = compute_velocity(
            0, 0, 0, 0.1, 0
        )

        self.assertAlmostEqual(linear, 0.08)
        self.assertEqual(angular, 0.0)
        self.assertFalse(reached)

    def test_stop_at_target(self):
        self.assertEqual(
            compute_velocity(0, 0, 0, 0.03, 0),
            (0.0, 0.0, True)
        )

    def test_invalid_pose(self):
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    compute_velocity(value, 0, 0, 1, 0)


if __name__ == "__main__":
    unittest.main()
