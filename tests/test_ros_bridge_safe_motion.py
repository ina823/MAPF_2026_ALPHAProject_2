"""ROS2 없이 Executor 안전성 로직(OdomSafetyMonitor + run_move)을 검증한다.

FakeWorld 는 monotonic 시간, 시뮬레이션 시간, /clock, Odometry, 단순 차동구동 운동을
흉내 낸다. 여기서 검증하는 것은 "정지 명령 발행"이며, 실제 물리적 정지는 아니다.
"""

import math
import unittest
from dataclasses import replace

from src.ros_bridge import odom_monitor as om
from src.ros_bridge.odom_monitor import OdomSafetyMonitor
from src.ros_bridge.safe_motion import (
    run_move,
    wait_until_ready,
    MOVE_TIMEOUT,
    TARGET_REACHED,
)
from src.ros_bridge.safety_config import (
    OdomSafetyConfig,
    RobotInterface,
    DEMO_1HZ_CONFIG,
    MAPF_50HZ_CONFIG,
    DEMO_VEHICLE_BLUE,
)

ODOM = DEMO_VEHICLE_BLUE.odom_frame_id
BASE = DEMO_VEHICLE_BLUE.base_frame_id
STOP = (0.0, 0.0)


class FakeWorld:

    def __init__(self, config=DEMO_1HZ_CONFIG):
        self.config = config
        self.monitor = OdomSafetyMonitor(config, ODOM, BASE)

        self.mono = 1000.0          # monotonic 시간 (sim 과 다른 원점)
        self.sim = 0.0              # 시뮬레이션 시간
        self.sim_rate = 1.0         # 0 이면 일시정지

        self.clock_enabled = True
        self.odom_enabled = True
        self.odom_period = config.odom_period
        self.next_odom_sim = 0.0
        self.odom_override = {}

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.cmd = STOP
        self.actuator_enabled = True

        self.commands = []
        self.errors = []
        self.infos = []
        self.hooks = []
        self.ok_flag = True

    # --- safe_motion io ---
    def ok(self):
        return self.ok_flag

    def monotonic(self):
        return self.mono

    def sleep(self, seconds):
        self.mono += seconds
        dt_sim = seconds * self.sim_rate
        self.sim += dt_sim
        if self.actuator_enabled:
            linear, angular = self.cmd
            self.yaw += angular * dt_sim
            self.x += linear * math.cos(self.yaw) * dt_sim
            self.y += linear * math.sin(self.yaw) * dt_sim

    def spin(self):
        for hook in list(self.hooks):
            hook(self)
        if self.clock_enabled:
            self.monitor.on_clock(self.sim, self.mono)
        if self.odom_enabled and self.sim >= self.next_odom_sim - 1e-9:
            self.emit_odom(**self.odom_override)
            self.next_odom_sim = self.sim + self.odom_period

    def publish_velocity(self, linear, angular):
        self.commands.append((linear, angular))
        self.cmd = (linear, angular)

    def stop_robot(self):
        self.publish_velocity(0.0, 0.0)

    def log_info(self, text):
        self.infos.append(text)

    def log_error(self, text):
        self.errors.append(text)

    # --- helpers ---
    def emit_odom(self, **override):
        fields = dict(
            stamp_sim=self.sim,
            frame_id=ODOM,
            child_frame_id=BASE,
            x=self.x,
            y=self.y,
            qx=0.0,
            qy=0.0,
            qz=math.sin(self.yaw / 2),
            qw=math.cos(self.yaw / 2),
            received_mono=self.mono,
        )
        fields.update(override)
        return self.monitor.on_odom(**fields)

    def advance(self, seconds, step=0.05):
        for _ in range(int(round(seconds / step))):
            self.sleep(step)
            self.spin()

    def at_sim(self, t, action):
        def hook(world):
            if world.sim >= t - 1e-9:
                world.hooks.remove(hook)
                action(world)
        self.hooks.append(hook)

    def moving_commands(self):
        return [c for c in self.commands if c != STOP]


def ready_world(config=DEMO_1HZ_CONFIG):
    world = FakeWorld(config)
    world.spin()
    assert world.monitor.check(world.mono).ok
    return world


class TestConfig(unittest.TestCase):

    def test_presets_valid(self):
        self.assertEqual(DEMO_1HZ_CONFIG.odom_period, 1.0)
        self.assertAlmostEqual(MAPF_50HZ_CONFIG.odom_period, 0.02)

    def test_presets_differ(self):
        self.assertGreater(
            DEMO_1HZ_CONFIG.odom_receive_timeout_mono,
            MAPF_50HZ_CONFIG.odom_receive_timeout_mono,
        )
        self.assertGreater(
            DEMO_1HZ_CONFIG.odom_stamp_max_age_sim,
            MAPF_50HZ_CONFIG.odom_stamp_max_age_sim,
        )

    def test_quarter_second_rejected_for_1hz(self):
        with self.assertRaises(ValueError):
            replace(DEMO_1HZ_CONFIG, odom_receive_timeout_mono=0.25)
        with self.assertRaises(ValueError):
            replace(DEMO_1HZ_CONFIG, odom_stamp_max_age_sim=0.25)

    def test_quarter_second_allowed_for_50hz(self):
        self.assertEqual(MAPF_50HZ_CONFIG.odom_receive_timeout_mono, 0.25)

    def test_invalid_values_rejected(self):
        for field, bad in (
            ("jump_max_speed", 0.0),
            ("jump_margin", -0.01),
            ("clock_receive_timeout_mono", float("nan")),
            ("sim_stall_timeout_mono", float("inf")),
            ("odom_rate_hz", 0.0),
            ("quaternion_norm_tolerance", 1.0),
            ("odom_stamp_future_tolerance_sim", None),
            ("move_timeout_mono", True),
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    replace(DEMO_1HZ_CONFIG, **{field: bad})

    def test_interface_validation(self):
        with self.assertRaises(ValueError):
            replace(DEMO_VEHICLE_BLUE, odom_frame_id="")
        with self.assertRaises(ValueError):
            replace(DEMO_VEHICLE_BLUE, base_frame_id=ODOM)
        self.assertIsInstance(DEMO_VEHICLE_BLUE, RobotInterface)

    def test_config_type(self):
        self.assertIsInstance(MAPF_50HZ_CONFIG, OdomSafetyConfig)


class TestMonitorTimeBases(unittest.TestCase):

    def test_receive_time_and_stamp_are_separate(self):
        world = FakeWorld()
        world.monitor.on_clock(10.0, 5000.0)
        # stamp 가 오래됐지만(sim) 방금 도착(mono)
        world.monitor.on_odom(8.0, ODOM, BASE, 0, 0, 0, 0, 0, 1, 5000.0)
        status = world.monitor.check(5000.0)
        self.assertFalse(status.ok)
        self.assertEqual(status.reason, om.ODOM_STAMP_STALE)

    def test_sim_and_mono_offsets_do_not_matter(self):
        # mono 원점이 sim 과 1e6 s 차이 나도 정상 판정
        monitor = OdomSafetyMonitor(DEMO_1HZ_CONFIG, ODOM, BASE)
        monitor.on_clock(3.0, 1_000_000.0)
        monitor.on_odom(3.0, ODOM, BASE, 0, 0, 0, 0, 0, 1, 1_000_000.0)
        self.assertTrue(monitor.check(1_000_000.1).ok)

    def test_stamp_in_wall_epoch_rejected(self):
        world = FakeWorld()
        world.monitor.on_clock(10.0, 5000.0)
        result = world.monitor.on_odom(
            1.7e9, ODOM, BASE, 0, 0, 0, 0, 0, 1, 5000.0
        )
        self.assertEqual(result, om.TIME_BASE_MISMATCH)
        self.assertTrue(world.monitor.faulted)

    def test_stamp_moderately_ahead_blocks_without_latch(self):
        world = FakeWorld()
        world.monitor.on_clock(10.0, 5000.0)
        result = world.monitor.on_odom(
            10.5, ODOM, BASE, 0, 0, 0, 0, 0, 1, 5000.0
        )
        self.assertEqual(result, om.ACCEPTED)
        self.assertEqual(
            world.monitor.check(5000.0).reason,
            om.ODOM_STAMP_AHEAD_OF_CLOCK,
        )
        self.assertFalse(world.monitor.faulted)
        # /clock 이 따라오면 해소
        world.monitor.on_clock(10.5, 5000.5)
        self.assertTrue(world.monitor.check(5000.5).ok)

    def test_small_future_stamp_tolerated(self):
        world = FakeWorld()
        world.monitor.on_clock(10.0, 5000.0)
        result = world.monitor.on_odom(
            10.1, ODOM, BASE, 0, 0, 0, 0, 0, 1, 5000.0
        )
        self.assertEqual(result, om.ACCEPTED)
        self.assertTrue(world.monitor.check(5000.0).ok)

    def test_clock_regression_latched(self):
        world = ready_world()
        world.advance(2.0)
        self.assertEqual(
            world.monitor.on_clock(0.5, world.mono),
            om.SIM_TIME_REGRESSION,
        )
        world.advance(1.0)
        self.assertEqual(
            world.monitor.check(world.mono).reason, om.SIM_TIME_REGRESSION
        )

    def test_clock_equal_is_not_regression(self):
        world = ready_world()
        self.assertIsNone(world.monitor.on_clock(world.sim, world.mono))
        self.assertFalse(world.monitor.faulted)

    def test_invalid_clock(self):
        for bad in (float("nan"), float("inf"), -1.0, None):
            with self.subTest(bad=bad):
                monitor = OdomSafetyMonitor(DEMO_1HZ_CONFIG, ODOM, BASE)
                self.assertEqual(
                    monitor.on_clock(bad, 1.0), om.INVALID_CLOCK
                )


class TestMonitorDuplicates(unittest.TestCase):

    def test_duplicate_does_not_refresh(self):
        world = ready_world()
        first = world.monitor.last_odom_received_mono
        world.mono += 0.5
        self.assertEqual(
            world.emit_odom(stamp_sim=0.0), om.IGNORED_DUPLICATE
        )
        self.assertEqual(world.monitor.last_odom_received_mono, first)
        self.assertEqual(world.monitor.duplicate_count, 1)

    def test_out_of_order_does_not_refresh_or_move_pose(self):
        world = ready_world()
        world.advance(2.0)
        stamp = world.monitor.last_odom.stamp_sim
        received = world.monitor.last_odom_received_mono
        result = world.emit_odom(stamp_sim=stamp - 0.5, x=9.0)
        self.assertEqual(result, om.IGNORED_OUT_OF_ORDER)
        self.assertEqual(world.monitor.last_odom_received_mono, received)
        self.assertEqual(world.monitor.pose[0], 0.0)
        self.assertEqual(world.monitor.out_of_order_count, 1)
        self.assertFalse(world.monitor.faulted)


class TestRunMoveBlocked(unittest.TestCase):

    def assert_blocked(self, world, reason):
        result = run_move(1.0, 0.0, world.monitor, world)
        self.assertFalse(result.success)
        self.assertTrue(result.blocked_before_start)
        self.assertEqual(result.reason, reason)
        self.assertEqual(world.moving_commands(), [])
        self.assertEqual(world.commands[-1], STOP)
        return result

    def test_no_odom(self):
        world = FakeWorld()
        world.odom_enabled = False
        world.spin()
        self.assert_blocked(world, om.NO_ODOM)

    def test_no_clock(self):
        world = FakeWorld()
        world.clock_enabled = False
        world.spin()
        self.assert_blocked(world, om.NO_CLOCK)

    def test_stale_odom(self):
        world = ready_world()
        world.odom_enabled = False
        world.advance(2.5)
        self.assert_blocked(world, om.ODOM_RECEIVE_TIMEOUT)

    def test_stale_stamp(self):
        world = FakeWorld()
        world.monitor.on_clock(10.0, world.mono)
        world.emit_odom(stamp_sim=8.0)
        self.assert_blocked(world, om.ODOM_STAMP_STALE)

    def test_invalid_frame(self):
        for frame, child in (
            ("vehicle_green/odom", BASE),
            (ODOM, "vehicle_green/chassis"),
            ("", BASE),
            (ODOM, ""),
        ):
            with self.subTest(frame=frame, child=child):
                world = FakeWorld()
                world.odom_override = dict(
                    frame_id=frame, child_frame_id=child
                )
                world.spin()
                self.assertEqual(world.monitor.fault, om.INVALID_FRAME)
                self.assert_blocked(world, om.INVALID_FRAME)

    def test_invalid_quaternion(self):
        for q in (
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 2.0),
            (float("nan"), 0.0, 0.0, 1.0),
            (0.0, 0.0, float("inf"), 1.0),
        ):
            with self.subTest(q=q):
                world = FakeWorld()
                world.odom_override = dict(
                    qx=q[0], qy=q[1], qz=q[2], qw=q[3]
                )
                world.spin()
                self.assert_blocked(world, om.INVALID_QUATERNION)

    def test_invalid_position(self):
        for bad in (float("nan"), float("inf")):
            with self.subTest(bad=bad):
                world = FakeWorld()
                world.odom_override = dict(x=bad)
                world.spin()
                self.assert_blocked(world, om.INVALID_POSITION)

    def test_invalid_stamp(self):
        world = FakeWorld()
        world.odom_override = dict(stamp_sim=float("nan"))
        world.spin()
        self.assert_blocked(world, om.INVALID_STAMP)

    def test_sim_paused_before_start(self):
        world = ready_world()
        world.sim_rate = 0.0
        world.advance(1.5)
        self.assert_blocked(world, om.SIM_TIME_STALLED)

    def test_blocked_start_is_not_latched(self):
        world = FakeWorld()
        world.odom_enabled = False
        world.spin()
        self.assert_blocked(world, om.NO_ODOM)
        self.assertFalse(world.monitor.faulted)


class TestRunMoveAborts(unittest.TestCase):

    def run_and_assert_abort(self, world, reason):
        result = run_move(1.0, 0.0, world.monitor, world)
        self.assertFalse(result.success)
        self.assertFalse(result.blocked_before_start)
        self.assertEqual(result.reason, reason)
        # 이동 명령을 낸 뒤 마지막 명령은 정지
        self.assertTrue(world.moving_commands())
        self.assertEqual(world.commands[-1], STOP)
        self.assertTrue(world.monitor.faulted)
        self.assertTrue(any(reason in e for e in world.errors))
        return result

    # RTF 1.0: sim 기준 stamp 나이(1.5 s)가 mono 수신 공백(2.0 s)보다 먼저 걸린다.
    # RTF 0.6: 같은 1.5 s(sim)가 2.5 s(mono)이므로 수신 Timeout 이 먼저 걸린다.
    TIMEOUT_CASES = (
        (1.0, om.ODOM_STAMP_STALE),
        (0.6, om.ODOM_RECEIVE_TIMEOUT),
    )

    def test_odom_timeout_during_move(self):
        cfg = DEMO_1HZ_CONFIG
        for rate, expected in self.TIMEOUT_CASES:
            with self.subTest(rtf=rate):
                world = ready_world()
                world.sim_rate = rate
                world.at_sim(
                    1.0, lambda w: setattr(w, "odom_enabled", False)
                )
                self.run_and_assert_abort(world, expected)
                # 마지막 수락 odom 후 각 한도 + 제어 주기 안에 정지 명령
                gap_mono = world.mono - world.monitor.last_odom_received_mono
                self.assertLessEqual(
                    gap_mono,
                    cfg.odom_receive_timeout_mono
                    + 2 * cfg.control_period_mono,
                )
                gap_sim = world.sim - world.monitor.last_odom.stamp_sim
                self.assertLessEqual(
                    gap_sim,
                    cfg.odom_stamp_max_age_sim
                    + 2 * cfg.control_period_mono,
                )

    def test_duplicates_during_move_do_not_extend(self):
        for rate, expected in self.TIMEOUT_CASES:
            with self.subTest(rtf=rate):
                world = ready_world()
                world.sim_rate = rate

                def freeze(w):
                    w.odom_override = dict(
                        stamp_sim=w.monitor.last_odom.stamp_sim
                    )

                world.at_sim(1.0, freeze)
                self.run_and_assert_abort(world, expected)
                self.assertGreater(world.monitor.duplicate_count, 0)

    def test_out_of_order_during_move_do_not_extend(self):
        for rate, expected in self.TIMEOUT_CASES:
            with self.subTest(rtf=rate):
                world = ready_world()
                world.sim_rate = rate

                def rewind(w):
                    # 마지막 수락 stamp(1.0)보다 과거
                    w.odom_override = dict(stamp_sim=0.5)

                world.at_sim(1.5, rewind)
                self.run_and_assert_abort(world, expected)
                self.assertGreater(world.monitor.out_of_order_count, 0)

    def test_stamp_stale_during_move(self):
        # 수신은 계속되지만 stamp 가 sim 보다 점점 뒤처짐 (지연 누적)
        world = ready_world()

        def lag(w):
            w.odom_override = dict(stamp_sim=max(w.sim - 1.6, 0.001))

        world.at_sim(1.0, lambda w: w.hooks.append(lag))
        self.run_and_assert_abort(world, om.ODOM_STAMP_STALE)

    def test_position_jump_during_move(self):
        world = ready_world()
        world.at_sim(1.0, lambda w: setattr(w, "x", w.x + 2.0))
        self.run_and_assert_abort(world, om.POSITION_JUMP)

    def test_odom_reset_to_origin_is_jump(self):
        world = ready_world()
        world.x, world.y = 5.0, 5.0
        world.monitor.reset()
        world.spin()
        world.emit_odom()

        def reset_odom(w):
            w.odom_override = dict(x=0.0, y=0.0)

        world.at_sim(1.0, reset_odom)
        result = run_move(6.0, 5.0, world.monitor, world)
        self.assertEqual(result.reason, om.POSITION_JUMP)
        self.assertEqual(world.commands[-1], STOP)

    def test_invalid_frame_during_move(self):
        world = ready_world()
        world.at_sim(
            1.0,
            lambda w: w.odom_override.update(frame_id="vehicle_green/odom"),
        )
        self.run_and_assert_abort(world, om.INVALID_FRAME)

    def test_invalid_quaternion_during_move(self):
        world = ready_world()
        world.at_sim(
            1.0, lambda w: w.odom_override.update(qw=float("nan"))
        )
        self.run_and_assert_abort(world, om.INVALID_QUATERNION)

    def test_sim_time_regression_during_move(self):
        world = ready_world()
        world.at_sim(1.0, lambda w: setattr(w, "sim", 0.0))
        self.run_and_assert_abort(world, om.SIM_TIME_REGRESSION)

    def test_sim_pause_during_move(self):
        world = ready_world()
        world.at_sim(0.5, lambda w: setattr(w, "sim_rate", 0.0))
        self.run_and_assert_abort(world, om.SIM_TIME_STALLED)

    def test_bridge_lost_during_move(self):
        # /clock 과 Odometry 가 함께 끊김 (Bridge 종료 등)
        world = ready_world()

        def lose(w):
            w.clock_enabled = False
            w.odom_enabled = False

        world.at_sim(0.5, lose)
        self.run_and_assert_abort(world, om.CLOCK_TIMEOUT)

    def test_clock_lost_odom_continues(self):
        # /clock 만 끊김: odom stamp 가 마지막 clock 을 앞서 나가며 차단
        world = ready_world()
        world.at_sim(0.5, lambda w: setattr(w, "clock_enabled", False))
        self.run_and_assert_abort(world, om.ODOM_STAMP_AHEAD_OF_CLOCK)

    def test_move_timeout(self):
        # 바퀴가 헛돌아 위치가 변하지 않지만 피드백은 정상
        world = ready_world()
        world.actuator_enabled = False
        result = self.run_and_assert_abort(world, MOVE_TIMEOUT)
        self.assertEqual(result.reason, MOVE_TIMEOUT)

    def test_shutdown_returns_false(self):
        world = ready_world()
        world.at_sim(0.5, lambda w: setattr(w, "ok_flag", False))
        result = run_move(1.0, 0.0, world.monitor, world)
        self.assertFalse(result.success)


class TestNoAutomaticResume(unittest.TestCase):

    def test_fault_blocks_later_moves_even_with_good_data(self):
        world = ready_world()
        world.at_sim(1.0, lambda w: setattr(w, "x", w.x + 2.0))
        first = run_move(1.0, 0.0, world.monitor, world)
        self.assertEqual(first.reason, om.POSITION_JUMP)

        # 피드백이 다시 정상이어도 잠금 유지
        world.odom_override = {}
        world.advance(3.0)
        self.assertFalse(world.monitor.check(world.mono).ok)
        self.assertIsNone(world.monitor.pose)

        count = len(world.moving_commands())
        second = run_move(1.0, 0.0, world.monitor, world)
        self.assertFalse(second.success)
        self.assertTrue(second.blocked_before_start)
        self.assertEqual(second.reason, om.POSITION_JUMP)
        self.assertEqual(len(world.moving_commands()), count)

    def test_timeout_during_move_is_latched(self):
        world = ready_world()
        world.at_sim(1.0, lambda w: setattr(w, "odom_enabled", False))
        first = run_move(1.0, 0.0, world.monitor, world)
        world.odom_enabled = True
        world.advance(2.0)
        self.assertEqual(
            world.monitor.check(world.mono).reason, first.reason
        )
        self.assertTrue(world.monitor.faulted)

    def test_messages_ignored_while_faulted(self):
        world = ready_world()
        world.monitor.latch(om.POSITION_JUMP)
        self.assertEqual(world.emit_odom(stamp_sim=5.0), om.IGNORED_FAULTED)

    def test_first_fault_reason_kept(self):
        world = ready_world()
        world.monitor.latch(om.INVALID_FRAME)
        world.monitor.latch(om.POSITION_JUMP)
        self.assertEqual(world.monitor.fault, om.INVALID_FRAME)

    def test_explicit_reset_only(self):
        world = ready_world()
        world.monitor.latch(om.POSITION_JUMP)
        world.monitor.reset()
        world.spin()
        world.emit_odom()
        self.assertTrue(world.monitor.check(world.mono).ok)


class TestStopCommandVsPhysicalStop(unittest.TestCase):

    def test_stop_published_but_robot_may_keep_moving(self):
        world = ready_world()

        def fail(w):
            w.odom_enabled = False
            w.actuator_stuck = True

        world.at_sim(1.0, fail)
        result = run_move(1.0, 0.0, world.monitor, world)
        self.assertEqual(world.commands[-1], STOP)

        # 정지 명령 이후 액추에이터가 이전 속도로 계속 움직이는 상황
        world.cmd = (0.3, 0.0)
        x_before = world.x
        world.sleep(1.0)
        self.assertGreater(world.x, x_before)
        # run_move 는 물리적 정지를 주장하지 않는다
        self.assertFalse(result.success)
        self.assertNotIn("STOPPED", result.reason)


class TestRunMoveSuccess(unittest.TestCase):

    def test_reach_target_1hz(self):
        world = ready_world()
        result = run_move(1.0, 0.0, world.monitor, world)
        self.assertTrue(result.success, result)
        self.assertEqual(result.reason, TARGET_REACHED)
        self.assertEqual(world.commands[-1], STOP)
        self.assertFalse(world.monitor.faulted)
        self.assertLess(math.hypot(1.0 - world.x, world.y), 0.1)

    def test_reach_target_50hz(self):
        world = ready_world(MAPF_50HZ_CONFIG)
        result = run_move(1.0, 0.0, world.monitor, world)
        self.assertTrue(result.success, result)
        self.assertFalse(world.monitor.faulted)

    def test_rotate_then_move(self):
        world = ready_world()
        result = run_move(0.0, 1.0, world.monitor, world)
        self.assertTrue(result.success, result)
        self.assertFalse(world.monitor.faulted)


class TestWaitUntilReady(unittest.TestCase):

    def test_ready(self):
        world = FakeWorld()
        status = wait_until_ready(world.monitor, world, 5.0)
        self.assertTrue(status.ok)
        self.assertEqual(world.commands, [])

    def test_no_clock_times_out(self):
        world = FakeWorld()
        world.clock_enabled = False
        start = world.mono
        status = wait_until_ready(world.monitor, world, 5.0)
        self.assertEqual(status.reason, om.NO_CLOCK)
        self.assertLess(world.mono - start, 5.2)
        self.assertEqual(world.commands, [])

    def test_fault_returns_immediately(self):
        world = FakeWorld()
        world.odom_override = dict(frame_id="vehicle_green/odom")
        start = world.mono
        status = wait_until_ready(world.monitor, world, 5.0)
        self.assertEqual(status.reason, om.INVALID_FRAME)
        self.assertLess(world.mono - start, 0.1)


if __name__ == "__main__":
    unittest.main()
