import math
import unittest

from src.ros_bridge.odom_safety import (
    is_odom_fresh,
    is_stamp_fresh,
    is_stamp_newer,
    is_frame_valid,
    is_position_jump,
    is_quaternion_valid,
)

NAN = float("nan")
INF = float("inf")
BAD_NUMBERS = (NAN, INF, -INF, None, True, "1.0")


class TestIsOdomFresh(unittest.TestCase):

    def test_normal(self):
        self.assertTrue(is_odom_fresh(10.0, 9.9, max_age=0.25))

    def test_same_time(self):
        self.assertTrue(is_odom_fresh(10.0, 10.0, max_age=0.25))

    def test_stale(self):
        self.assertFalse(is_odom_fresh(10.0, 9.0, max_age=0.25))

    def test_not_received(self):
        self.assertFalse(is_odom_fresh(10.0, None, max_age=0.25))

    def test_future_rejected_by_default(self):
        self.assertFalse(is_odom_fresh(10.0, 10.001, max_age=0.25))

    def test_future_within_tolerance(self):
        self.assertTrue(
            is_odom_fresh(10.0, 10.05, max_age=0.25, max_future=0.1)
        )

    def test_future_beyond_tolerance(self):
        self.assertFalse(
            is_odom_fresh(10.0, 10.2, max_age=0.25, max_future=0.1)
        )

    def test_boundary_inclusive(self):
        # 0.5, 0.25 는 이진수로 정확히 표현된다
        self.assertTrue(is_odom_fresh(1.0, 0.75, max_age=0.25))
        self.assertTrue(
            is_odom_fresh(1.0, 1.5, max_age=0.25, max_future=0.5)
        )

    def test_just_past_boundary(self):
        self.assertFalse(
            is_odom_fresh(1.0, 0.75 - 1e-9, max_age=0.25)
        )
        self.assertFalse(
            is_odom_fresh(1.0, 1.5 + 1e-9, max_age=0.25, max_future=0.5)
        )

    def test_invalid_time_values(self):
        for bad in BAD_NUMBERS:
            with self.subTest(now=bad):
                self.assertFalse(is_odom_fresh(bad, 1.0, max_age=1.0))
            if bad is None:
                continue
            with self.subTest(received_at=bad):
                self.assertFalse(is_odom_fresh(1.0, bad, max_age=1.0))

    def test_invalid_max_age_raises(self):
        for bad in (0, 0.0, -0.1, NAN, INF, -INF, None, True, "0.25"):
            with self.subTest(max_age=bad):
                with self.assertRaises(ValueError):
                    is_odom_fresh(1.0, 1.0, max_age=bad)

    def test_invalid_max_age_raises_even_without_message(self):
        # 설정 오류는 데이터 상태와 관계없이 항상 드러나야 한다
        with self.assertRaises(ValueError):
            is_odom_fresh(1.0, None, max_age=-1.0)
        with self.assertRaises(ValueError):
            is_odom_fresh(NAN, 1.0, max_age=NAN)

    def test_invalid_max_future_raises(self):
        for bad in (-0.1, NAN, INF, None):
            with self.subTest(max_future=bad):
                with self.assertRaises(ValueError):
                    is_odom_fresh(1.0, 1.0, max_age=1.0, max_future=bad)


class TestIsStampFresh(unittest.TestCase):

    def test_normal_header_stamp(self):
        self.assertTrue(is_stamp_fresh(100.0, 99.5, max_age=1.5))

    def test_stale_header_stamp(self):
        self.assertFalse(is_stamp_fresh(100.0, 98.0, max_age=1.5))

    def test_no_stamp(self):
        self.assertFalse(is_stamp_fresh(100.0, None, max_age=1.5))

    def test_future_stamp(self):
        self.assertFalse(is_stamp_fresh(100.0, 100.5, max_age=1.5))
        self.assertTrue(
            is_stamp_fresh(100.0, 100.05, max_age=1.5, max_future=0.1)
        )

    def test_stamp_zero_at_sim_start(self):
        self.assertTrue(is_stamp_fresh(0.0, 0.0, max_age=1.5))

    def test_nan_inf(self):
        for bad in (NAN, INF, -INF):
            with self.subTest(bad=bad):
                self.assertFalse(is_stamp_fresh(bad, 1.0, max_age=1.5))
                self.assertFalse(is_stamp_fresh(1.0, bad, max_age=1.5))

    def test_invalid_max_age(self):
        with self.assertRaises(ValueError):
            is_stamp_fresh(1.0, 1.0, max_age=0)


class TestIsStampNewer(unittest.TestCase):

    def test_first_message(self):
        self.assertTrue(is_stamp_newer(0.0, None))
        self.assertTrue(is_stamp_newer(5.0, None))

    def test_newer(self):
        self.assertTrue(is_stamp_newer(2.0, 1.0))

    def test_duplicate(self):
        self.assertFalse(is_stamp_newer(1.0, 1.0))

    def test_reverse(self):
        self.assertFalse(is_stamp_newer(0.5, 1.0))

    def test_negative_stamp(self):
        self.assertFalse(is_stamp_newer(-1.0, None))

    def test_invalid_stamp(self):
        for bad in BAD_NUMBERS:
            with self.subTest(stamp=bad):
                self.assertFalse(is_stamp_newer(bad, 1.0))
                self.assertFalse(is_stamp_newer(bad, None))

    def test_invalid_previous(self):
        for bad in (NAN, INF, -INF, True, "1.0", -1.0):
            with self.subTest(previous=bad):
                self.assertFalse(is_stamp_newer(2.0, bad))


class TestIsFrameValid(unittest.TestCase):

    ODOM = "vehicle_blue/odom"
    CHILD = "vehicle_blue/chassis"

    def test_correct(self):
        self.assertTrue(
            is_frame_valid(self.ODOM, self.CHILD, self.ODOM, self.CHILD)
        )

    def test_other_robot(self):
        self.assertFalse(is_frame_valid(
            "vehicle_green/odom", "vehicle_green/chassis",
            self.ODOM, self.CHILD
        ))
        self.assertFalse(is_frame_valid(
            self.ODOM, "vehicle_green/chassis", self.ODOM, self.CHILD
        ))

    def test_swapped(self):
        self.assertFalse(
            is_frame_valid(self.CHILD, self.ODOM, self.ODOM, self.CHILD)
        )

    def test_empty(self):
        self.assertFalse(is_frame_valid("", self.CHILD, self.ODOM, self.CHILD))
        self.assertFalse(is_frame_valid(self.ODOM, "", self.ODOM, self.CHILD))

    def test_not_exact(self):
        self.assertFalse(
            is_frame_valid("/" + self.ODOM, self.CHILD, self.ODOM, self.CHILD)
        )
        self.assertFalse(
            is_frame_valid(self.ODOM + " ", self.CHILD, self.ODOM, self.CHILD)
        )

    def test_non_string(self):
        for bad in (None, 1, b"vehicle_blue/odom"):
            with self.subTest(bad=bad):
                self.assertFalse(
                    is_frame_valid(bad, self.CHILD, self.ODOM, self.CHILD)
                )

    def test_invalid_expected_raises(self):
        for bad in ("", None, 1):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    is_frame_valid(self.ODOM, self.CHILD, bad, self.CHILD)
                with self.assertRaises(ValueError):
                    is_frame_valid(self.ODOM, self.CHILD, self.ODOM, bad)

    def test_same_expected_raises(self):
        with self.assertRaises(ValueError):
            is_frame_valid("a", "a", "a", "a")


class TestIsPositionJump(unittest.TestCase):

    def test_normal_motion(self):
        # 0.3 m/s 로 1 s → 0.3 m, 허용 0.3 + 0.05
        self.assertFalse(is_position_jump(0, 0, 0.3, 0, 1.0, 0.3, 0.05))

    def test_diagonal_normal(self):
        self.assertFalse(is_position_jump(0, 0, 0.2, 0.2, 1.0, 0.3, 0.05))

    def test_jump(self):
        self.assertTrue(is_position_jump(0, 0, 1.0, 0, 1.0, 0.3, 0.05))

    def test_jump_in_short_dt(self):
        # 50 Hz 에서 0.1 m 이동은 5 m/s 에 해당
        self.assertTrue(is_position_jump(0, 0, 0.1, 0, 0.02, 0.6, 0.02))

    def test_reset_to_origin(self):
        self.assertTrue(is_position_jump(3.0, 2.0, 0.0, 0.0, 1.0, 0.6, 0.05))

    def test_stationary(self):
        self.assertFalse(is_position_jump(1.0, 2.0, 1.0, 2.0, 1.0, 0.3, 0.05))

    def test_stationary_noise_within_margin(self):
        self.assertFalse(is_position_jump(0, 0, 0.0, 0.04, 0.02, 0.3, 0.05))

    def test_boundary_inclusive(self):
        # max_distance = 0.5 * 1.0 + 0.25 = 0.75 (이진수로 정확)
        self.assertFalse(is_position_jump(0, 0, 0.75, 0, 1.0, 0.5, 0.25))
        self.assertTrue(
            is_position_jump(0, 0, 0.75 + 1e-9, 0, 1.0, 0.5, 0.25)
        )

    def test_zero_margin(self):
        self.assertFalse(is_position_jump(0, 0, 0.5, 0, 1.0, 0.5, 0.0))
        self.assertTrue(is_position_jump(0, 0, 0.51, 0, 1.0, 0.5, 0.0))

    def test_invalid_dt(self):
        for dt in (0, 0.0, -0.1, NAN, INF, -INF, None, True):
            with self.subTest(dt=dt):
                self.assertTrue(is_position_jump(0, 0, 0, 0, dt))

    def test_invalid_position(self):
        for bad in BAD_NUMBERS:
            for index in range(4):
                args = [0.0, 0.0, 0.0, 0.0]
                args[index] = bad
                with self.subTest(index=index, bad=bad):
                    self.assertTrue(is_position_jump(*args, 1.0))

    def test_invalid_max_speed_raises(self):
        for bad in (0, -0.1, NAN, INF, None):
            with self.subTest(max_speed=bad):
                with self.assertRaises(ValueError):
                    is_position_jump(0, 0, 0, 0, 1.0, max_speed=bad)

    def test_invalid_margin_raises(self):
        for bad in (-0.01, NAN, INF, None):
            with self.subTest(margin=bad):
                with self.assertRaises(ValueError):
                    is_position_jump(0, 0, 0, 0, 1.0, margin=bad)

    def test_config_error_raised_before_data_check(self):
        with self.assertRaises(ValueError):
            is_position_jump(NAN, 0, 0, 0, -1.0, max_speed=0)


class TestIsQuaternionValid(unittest.TestCase):

    def test_identity(self):
        self.assertTrue(is_quaternion_valid(0, 0, 0, 1))

    def test_yaw_90(self):
        q = math.sqrt(0.5)
        self.assertTrue(is_quaternion_valid(0, 0, q, q))

    def test_small_norm_error_allowed(self):
        self.assertTrue(is_quaternion_valid(0, 0, 0, 1.04))

    def test_zero(self):
        self.assertFalse(is_quaternion_valid(0, 0, 0, 0))

    def test_not_normalized(self):
        self.assertFalse(is_quaternion_valid(0, 0, 0, 2.0))
        self.assertFalse(is_quaternion_valid(0, 0, 0, 0.5))

    def test_nan_inf(self):
        for bad in BAD_NUMBERS:
            with self.subTest(bad=bad):
                self.assertFalse(is_quaternion_valid(0, 0, bad, 1))

    def test_huge_values(self):
        self.assertFalse(is_quaternion_valid(1e200, 0, 0, 1e200))

    def test_invalid_tolerance(self):
        for bad in (-0.1, 1.0, NAN, None):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    is_quaternion_valid(0, 0, 0, 1, norm_tolerance=bad)


if __name__ == "__main__":
    unittest.main()
