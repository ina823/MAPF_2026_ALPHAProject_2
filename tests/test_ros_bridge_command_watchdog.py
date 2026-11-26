"""독립 Command Watchdog 순수 로직 검증 (ROS2 불필요)."""

import json
import math
import unittest
from dataclasses import replace

from src.ros_bridge import command_watchdog as cw
from src.ros_bridge import odom_monitor as om
from src.ros_bridge import stop_check as sc
from src.ros_bridge import watchdog_protocol as wp
from src.ros_bridge.command_watchdog import CommandWatchdog, STOP
from src.ros_bridge.watchdog_protocol import (
    WatchdogClient,
    encode_request_id,
    decode_request_id,
    parse_status,
    status_to_json,
)
from src.ros_bridge.safety_config import (
    DEMO_1HZ_CONFIG,
    DEMO_1HZ_WATCHDOG,
    DEMO_1HZ_STOP_CHECK,
    MAPF_50HZ_CONFIG,
    MAPF_50HZ_WATCHDOG,
    DEMO_VEHICLE_BLUE,
    make_robot_interface,
)

ODOM = DEMO_VEHICLE_BLUE.odom_frame_id
BASE = DEMO_VEHICLE_BLUE.base_frame_id
WD = DEMO_1HZ_WATCHDOG
TICK = WD.tick_period_mono


class Rig:
    """Watchdog 와 시간(mono/sim), /clock, Odometry, Executor 요청을 흉내 낸다."""

    def __init__(self):
        self.wd = CommandWatchdog(
            "vehicle_blue", WD, DEMO_1HZ_CONFIG, DEMO_1HZ_STOP_CHECK,
            ODOM, BASE, boot_id="boot",
        )
        self.mono = 500.0
        self.sim = 0.0
        self.sim_rate = 1.0
        self.clock_on = True
        self.odom_on = True
        self.next_odom = 0.0
        self.x = 0.0
        self.vx = 0.0
        self.sender = "exec1"
        self.seq = 0
        self.outputs = []

    def feed(self):
        if self.clock_on:
            self.wd.on_clock(self.sim, self.mono)
        if self.odom_on and self.sim >= self.next_odom - 1e-9:
            self.odom()
            self.next_odom = self.sim + 1.0

    def odom(self, **override):
        fields = dict(
            stamp_sim=self.sim, frame_id=ODOM, child_frame_id=BASE,
            x=self.x, y=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0,
            received_mono=self.mono,
            linear_x=self.vx, linear_y=0.0, angular_z=0.0,
        )
        fields.update(override)
        return self.wd.on_odom(**fields)

    def step(self, dt=TICK):
        self.mono += dt
        self.sim += dt * self.sim_rate
        self.x += self.vx * dt * self.sim_rate
        self.feed()
        out = self.wd.tick(self.mono)
        self.outputs.append(out)
        self.vx = out[0]
        return out

    def run(self, seconds, request=None):
        for _ in range(int(round(seconds / TICK))):
            if request is not None:
                self.request(*request)
            self.step()

    def ready(self):
        self.feed()
        self.wd.tick(self.mono)
        assert self.wd.state == wp.READY, self.wd.health
        return self

    def request(self, linear=0.0, angular=0.0, token=None, sender=None,
                seq=None, stamp=None, extra=(0.0, 0.0, 0.0, 0.0)):
        if seq is None:
            self.seq += 1
            seq = self.seq
        rid = encode_request_id(
            token or self.wd.token, sender or self.sender, seq
        )
        ly, lz, ax, ay = extra
        result = self.wd.on_request(
            rid, self.sim if stamp is None else stamp,
            linear, ly, lz, ax, ay, angular, self.mono,
        )
        if result.forwarded:
            self.outputs.append(result.output)
            self.vx = result.output[0]
        return result


class TestProtocol(unittest.TestCase):

    def test_request_id_roundtrip(self):
        rid = encode_request_id("boot-3", "abc", 17)
        self.assertEqual(decode_request_id(rid), ("boot-3", "abc", 17))

    def test_bad_request_ids(self):
        for bad in (None, "", "tok", "tok#1", "tok#s#", "tok#s#-1",
                    "tok#s#1.5", "#s#1", "tok##1", "a#b#c#1", "t#s#١"):
            with self.subTest(bad=bad):
                self.assertIsNone(decode_request_id(bad))

    def test_encode_rejects_separator(self):
        with self.assertRaises(ValueError):
            encode_request_id("a#b", "s", 1)
        with self.assertRaises(ValueError):
            encode_request_id("a", "s", -1)

    def test_status_json_roundtrip(self):
        rig = Rig().ready()
        text = status_to_json(rig.wd.status(rig.mono))
        data = parse_status(text)
        self.assertEqual(data["state"], wp.READY)
        self.assertEqual(data["token"], "boot-0")

    def test_bad_status(self):
        for bad in ("", "{", "[]", json.dumps({"state": "READY"}),
                    None, 3):
            with self.subTest(bad=bad):
                self.assertIsNone(parse_status(bad))

    def test_robot_interface_for_multi_robot(self):
        green = make_robot_interface("vehicle_green")
        self.assertEqual(
            green.cmd_request_topic, "/mapf/vehicle_green/cmd_vel_request"
        )
        self.assertEqual(green.cmd_vel_topic, "/model/vehicle_green/cmd_vel")
        self.assertEqual(green.odom_frame_id, "vehicle_green/odom")
        self.assertEqual(
            DEMO_VEHICLE_BLUE.cmd_request_topic,
            "/mapf/vehicle_blue/cmd_vel_request",
        )
        self.assertEqual(
            DEMO_VEHICLE_BLUE.cmd_vel_topic, "/model/vehicle_blue/cmd_vel"
        )
        for bad in ("", "a/b", None):
            with self.assertRaises(ValueError):
                make_robot_interface(bad)

    def test_timeout_config_checked_against_executor(self):
        DEMO_1HZ_WATCHDOG.validate_against(DEMO_1HZ_CONFIG)
        MAPF_50HZ_WATCHDOG.validate_against(MAPF_50HZ_CONFIG)
        with self.assertRaises(ValueError):
            MAPF_50HZ_WATCHDOG.validate_against(DEMO_1HZ_CONFIG)
        with self.assertRaises(ValueError):
            replace(WD, command_timeout_mono=0.05)


class TestInitialStop(unittest.TestCase):

    def test_initial_state_outputs_stop(self):
        rig = Rig()
        self.assertEqual(rig.wd.state, wp.WAITING)
        self.assertEqual(rig.wd.output, STOP)
        self.assertEqual(rig.wd.tick(rig.mono), STOP)

    def test_moving_request_rejected_before_ready(self):
        rig = Rig()
        rig.clock_on = False
        rig.feed()
        rig.wd.tick(rig.mono)
        rig.wd.monitor.on_clock(0.0, rig.mono)  # stamp 검사 통과용 clock 1회
        result = rig.request(0.2)
        self.assertEqual(result.result, cw.REJECTED_NOT_READY)
        self.assertEqual(rig.wd.tick(rig.mono), STOP)

    def test_stop_request_harmless_before_ready(self):
        rig = Rig()
        rig.feed()
        result = rig.request(0.0)
        self.assertTrue(result.forwarded)
        self.assertEqual(result.output, STOP)


class TestForwarding(unittest.TestCase):

    def test_valid_request_forwarded(self):
        rig = Rig().ready()
        result = rig.request(0.2, 0.1)
        self.assertTrue(result.forwarded)
        self.assertEqual(result.output, (0.2, 0.1))
        self.assertEqual(rig.wd.state, wp.ACTIVE)
        self.assertEqual(rig.step(), (0.2, 0.1))

    def test_last_valid_command_time_recorded(self):
        rig = Rig().ready()
        rig.request(0.2)
        self.assertEqual(rig.wd.last_command_mono, rig.mono)

    def test_continuous_stream_keeps_active(self):
        rig = Rig().ready()
        rig.run(3.0, request=(0.2, 0.0))
        self.assertEqual(rig.wd.state, wp.ACTIVE)
        self.assertFalse(rig.wd.monitor.faulted)

    def test_stop_request_returns_to_ready(self):
        rig = Rig().ready()
        rig.request(0.2)
        rig.step()
        rig.request(0.0)
        self.assertEqual(rig.wd.state, wp.READY)
        rig.run(2.0)  # 정지 상태에서는 요청이 없어도 Fault 아님
        self.assertEqual(rig.wd.state, wp.READY)
        self.assertFalse(rig.wd.monitor.faulted)


class TestCommandTimeout(unittest.TestCase):

    def test_timeout_latches_and_stops(self):
        rig = Rig().ready()
        rig.request(0.2)
        start = rig.mono  # 500.0: 0.5 를 더해도 이진수로 정확
        for _ in range(9):
            self.assertEqual(rig.step(), (0.2, 0.0))
        # 경계값(정확히 Timeout)은 아직 허용
        rig.wd.on_clock(rig.sim, start + WD.command_timeout_mono)
        self.assertEqual(
            rig.wd.tick(start + WD.command_timeout_mono), (0.2, 0.0)
        )
        out = rig.wd.tick(start + WD.command_timeout_mono + 1e-6)
        self.assertEqual(out, STOP)
        self.assertEqual(rig.wd.state, wp.FAULT)
        self.assertEqual(rig.wd.fault, cw.COMMAND_TIMEOUT)

    def test_executor_stream_stops(self):
        # Executor 가 20 Hz 로 보내다가 갑자기 중단 (프로세스 종료/SIGKILL 과 같은 관측)
        rig = Rig().ready()
        rig.run(1.0, request=(0.3, 0.0))
        last_request = rig.mono
        rig.run(2.0)
        self.assertEqual(rig.wd.fault, cw.COMMAND_TIMEOUT)
        # 마지막 요청 이후 이동 출력은 Timeout + 1 tick 을 넘지 않는다
        moving_after = sum(1 for o in rig.outputs[-40:] if o != STOP)
        self.assertLessEqual(
            moving_after * TICK, WD.command_timeout_mono + TICK + 1e-9
        )
        self.assertTrue(all(o == STOP for o in rig.outputs[-20:]))
        self.assertGreater(rig.mono - last_request, WD.command_timeout_mono)

    def test_rejected_requests_do_not_extend_timeout(self):
        rig = Rig().ready()
        rig.request(0.2)
        for _ in range(20):
            rig.request(0.2, seq=1)  # 중복 seq: 거부
            rig.step()
        self.assertEqual(rig.wd.fault, cw.COMMAND_TIMEOUT)
        self.assertGreater(rig.wd.rejected[cw.REJECTED_OLD_SEQ], 0)


class TestInvalidVelocity(unittest.TestCase):

    def assert_latched(self, reason, **kwargs):
        rig = Rig().ready()
        rig.request(0.1)
        result = rig.request(**kwargs)
        self.assertEqual(result.result, reason)
        self.assertFalse(result.forwarded)
        self.assertEqual(rig.wd.output, STOP)
        self.assertEqual(rig.wd.fault, reason)
        self.assertEqual(rig.step(), STOP)

    def test_nan_inf(self):
        for bad in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(bad=bad):
                self.assert_latched(cw.INVALID_VELOCITY, linear=bad)
                self.assert_latched(cw.INVALID_VELOCITY, angular=bad)
                self.assert_latched(
                    cw.INVALID_VELOCITY, extra=(bad, 0.0, 0.0, 0.0)
                )

    def test_unused_components(self):
        for index in range(4):
            extra = [0.0, 0.0, 0.0, 0.0]
            extra[index] = 0.1
            with self.subTest(index=index):
                self.assert_latched(
                    cw.UNUSED_TWIST_COMPONENT, linear=0.1, extra=tuple(extra)
                )

    def test_speed_limit(self):
        self.assert_latched(cw.SPEED_LIMIT, linear=0.31)
        self.assert_latched(cw.SPEED_LIMIT, linear=-0.31)
        self.assert_latched(cw.SPEED_LIMIT, angular=0.51)
        self.assert_latched(cw.SPEED_LIMIT, angular=-0.51)

    def test_speed_limit_boundary_allowed(self):
        rig = Rig().ready()
        self.assertTrue(rig.request(0.3, 0.0).forwarded)
        self.assertTrue(rig.request(0.0, -0.5).forwarded)
        self.assertFalse(rig.wd.monitor.faulted)


class TestOdomAndClock(unittest.TestCase):

    def test_no_odom_waiting(self):
        rig = Rig()
        rig.odom_on = False
        rig.run(1.0)
        self.assertEqual(rig.wd.state, wp.WAITING)
        self.assertEqual(rig.wd.health, om.NO_ODOM)
        self.assertEqual(rig.request(0.2).result, cw.REJECTED_NOT_READY)
        self.assertTrue(all(o == STOP for o in rig.outputs))

    def test_no_clock_waiting(self):
        rig = Rig()
        rig.clock_on = False
        rig.run(1.0)
        self.assertEqual(rig.wd.health, om.NO_CLOCK)
        # clock 이 없으면 요청 stamp 도 검증할 수 없다
        self.assertEqual(rig.request(0.2).result, cw.REJECTED_STALE_REQUEST)
        self.assertTrue(all(o == STOP for o in rig.outputs))

    def test_odom_timeout_during_motion(self):
        rig = Rig().ready()
        rig.odom_on = False
        rig.run(3.0, request=(0.2, 0.0))
        self.assertIn(
            rig.wd.fault, (om.ODOM_STAMP_STALE, om.ODOM_RECEIVE_TIMEOUT)
        )
        self.assertEqual(rig.wd.output, STOP)

    def test_odom_timeout_while_idle_not_latched(self):
        rig = Rig().ready()
        rig.odom_on = False
        rig.run(3.0)
        self.assertEqual(rig.wd.state, wp.WAITING)
        self.assertFalse(rig.wd.monitor.faulted)

    def test_clock_lost_during_motion(self):
        rig = Rig().ready()
        rig.clock_on = False
        rig.odom_on = False
        for _ in range(40):
            rig.wd.on_request(
                encode_request_id(rig.wd.token, "exec1", 100 + _),
                0.0, 0.2, 0, 0, 0, 0, 0, rig.mono,
            )
            rig.step()
        self.assertTrue(rig.wd.monitor.faulted)
        self.assertEqual(rig.wd.output, STOP)

    def test_sim_paused_during_motion(self):
        rig = Rig().ready()
        rig.request(0.2)
        rig.sim_rate = 0.0
        for _ in range(30):
            rig.request(0.2)
            rig.step()
        self.assertEqual(rig.wd.fault, om.SIM_TIME_STALLED)
        self.assertEqual(rig.wd.output, STOP)

    def test_sim_regression(self):
        rig = Rig().ready()
        rig.run(2.0)
        rig.sim = 0.1
        rig.step()
        self.assertEqual(rig.wd.fault, om.SIM_TIME_REGRESSION)
        self.assertEqual(rig.wd.output, STOP)

    def test_odom_fault_immediately_stops(self):
        rig = Rig().ready()
        rig.request(0.2)
        rig.mono += 0.01
        rig.sim += 1.0
        rig.wd.on_clock(rig.sim, rig.mono)
        rig.odom(x=5.0)
        self.assertEqual(rig.wd.fault, om.POSITION_JUMP)
        self.assertEqual(rig.wd.output, STOP)


class TestFaultLatchAndReset(unittest.TestCase):

    def faulted_rig(self):
        rig = Rig().ready()
        rig.run(0.5, request=(0.2, 0.0))
        rig.run(1.0)  # Executor 중단 → COMMAND_TIMEOUT
        assert rig.wd.fault == cw.COMMAND_TIMEOUT
        return rig

    def test_fault_persists_with_good_data(self):
        rig = self.faulted_rig()
        rig.run(5.0)
        self.assertEqual(rig.wd.state, wp.FAULT)
        self.assertTrue(all(o == STOP for o in rig.outputs[-100:]))

    def test_requests_rejected_while_faulted(self):
        rig = self.faulted_rig()
        result = rig.request(0.2)
        self.assertEqual(result.result, cw.REJECTED_FAULTED)
        self.assertEqual(rig.step(), STOP)

    def test_fault_recorded(self):
        rig = self.faulted_rig()
        status = rig.wd.status(rig.mono)
        self.assertEqual(status["state"], wp.FAULT)
        self.assertEqual(status["fault"], cw.COMMAND_TIMEOUT)
        self.assertEqual(
            status["fault_history"][-1]["reason"], cw.COMMAND_TIMEOUT
        )
        json.loads(status_to_json(status))

    def test_reset_refused_while_requests_arrive(self):
        rig = self.faulted_rig()
        rig.request(0.2)  # 거부되지만 아직 들어오는 중
        ok, message = rig.wd.reset(rig.mono)
        self.assertFalse(ok)
        self.assertEqual(rig.wd.state, wp.FAULT)

    def test_explicit_reset(self):
        rig = self.faulted_rig()
        old_token = rig.wd.token
        ok, _ = rig.wd.reset(rig.mono)
        self.assertTrue(ok)
        self.assertNotEqual(rig.wd.token, old_token)
        self.assertIsNone(rig.wd.fault)
        self.assertEqual(rig.wd.state, wp.WAITING)
        self.assertEqual(rig.wd.output, STOP)
        rig.next_odom = rig.sim
        rig.step()
        self.assertEqual(rig.wd.state, wp.READY)

    def test_reset_does_not_reuse_old_command(self):
        rig = self.faulted_rig()
        rig.wd.reset(rig.mono)
        self.assertIsNone(rig.wd.last_command)
        rig.next_odom = rig.sim
        rig.run(2.0)
        self.assertTrue(all(o == STOP for o in rig.outputs[-40:]))

    def test_old_session_request_rejected_after_reset(self):
        rig = self.faulted_rig()
        old_token = rig.wd.token
        rig.wd.reset(rig.mono)
        rig.next_odom = rig.sim
        rig.step()
        # Reset 전에 큐에 남아 있던 요청 (이전 token)
        result = rig.request(0.2, token=old_token, seq=10_000)
        self.assertEqual(result.result, cw.REJECTED_STALE_SESSION)
        self.assertEqual(rig.step(), STOP)

    def test_delayed_request_with_old_stamp_rejected(self):
        rig = Rig().ready()
        rig.run(2.0)
        result = rig.request(0.2, stamp=rig.sim - 1.0)
        self.assertEqual(result.result, cw.REJECTED_STALE_REQUEST)
        self.assertEqual(rig.step(), STOP)

    def test_out_of_order_and_duplicate_seq_rejected(self):
        rig = Rig().ready()
        rig.request(0.2, seq=5)
        self.assertEqual(rig.request(0.2, seq=5).result, cw.REJECTED_OLD_SEQ)
        self.assertEqual(rig.request(0.2, seq=4).result, cw.REJECTED_OLD_SEQ)

    def test_bad_request_id_rejected(self):
        rig = Rig().ready()
        result = rig.wd.on_request("garbage", rig.sim, 0.2, 0, 0, 0, 0, 0,
                                   rig.mono)
        self.assertEqual(result.result, cw.REJECTED_BAD_ID)
        self.assertEqual(rig.step(), STOP)


class TestOwnership(unittest.TestCase):

    def test_second_sender_rejected_while_owner_active(self):
        rig = Rig().ready()
        rig.request(0.2)
        result = rig.request(0.1, sender="exec2", seq=1)
        self.assertEqual(result.result, cw.REJECTED_NOT_OWNER)

    def test_owner_released_after_idle_quiet(self):
        # 정지 상태에서 Executor 가 종료되면 새 Executor 가 Reset 없이 이어받을 수 있다
        rig = Rig().ready()
        rig.request(0.2)
        rig.step()
        rig.request(0.0)
        rig.run(1.0)
        self.assertIsNone(rig.wd.owner)
        self.assertTrue(rig.request(0.2, sender="exec2", seq=1).forwarded)

    def test_new_executor_needs_reset_after_motion_kill(self):
        rig = Rig().ready()
        rig.run(0.5, request=(0.2, 0.0))
        rig.run(1.0)  # 이동 중 SIGKILL 과 같은 관측
        self.assertEqual(rig.wd.fault, cw.COMMAND_TIMEOUT)
        result = rig.request(0.2, sender="exec2", seq=1)
        self.assertEqual(result.result, cw.REJECTED_FAULTED)


class TestShutdown(unittest.TestCase):

    def test_shutdown_outputs_stop(self):
        rig = Rig().ready()
        rig.request(0.2)
        rig.wd.shutdown()
        self.assertEqual(rig.wd.state, wp.SHUTDOWN)
        self.assertEqual(rig.wd.output, STOP)
        self.assertEqual(rig.wd.tick(rig.mono), STOP)
        self.assertEqual(rig.request(0.2).result, cw.REJECTED_SHUTDOWN)
        self.assertFalse(rig.wd.reset(rig.mono + 10)[0])


class TestStopConfirmation(unittest.TestCase):

    def test_stop_confirmed_after_motion(self):
        rig = Rig().ready()
        rig.run(1.0, request=(0.2, 0.0))
        rig.request(0.0)
        rig.run(4.0)
        status = rig.wd.status(rig.mono)
        self.assertEqual(status["stop_check"], sc.CONFIRMED)

    def test_stop_not_confirmed_without_fresh_odom(self):
        rig = Rig().ready()
        rig.run(1.0, request=(0.2, 0.0))
        rig.request(0.0)
        rig.odom_on = False
        rig.run(1.0)
        self.assertEqual(rig.wd.status(rig.mono)["stop_check"], sc.PENDING)
        rig.run(4.0)
        self.assertEqual(
            rig.wd.status(rig.mono)["stop_check"], sc.NO_FRESH_ODOM
        )

    def test_stop_not_confirmed_if_still_moving(self):
        rig = Rig().ready()
        rig.run(1.0, request=(0.2, 0.0))
        rig.request(0.0)
        for _ in range(100):
            rig.step()
            rig.vx = 0.2  # 정지 명령 후에도 바퀴가 계속 도는 상황
        self.assertEqual(rig.wd.status(rig.mono)["stop_check"], sc.NOT_STOPPED)

    def test_stop_confirmed_after_command_timeout_fault(self):
        # Fault 로 잠긴 뒤에도 Odometry 로 실제 정지를 확인한다
        rig = Rig().ready()
        rig.run(1.0, request=(0.2, 0.0))
        rig.run(5.0)
        self.assertEqual(rig.wd.fault, cw.COMMAND_TIMEOUT)
        self.assertEqual(rig.wd.status(rig.mono)["stop_check"], sc.CONFIRMED)

    def test_stop_unverifiable_after_odom_integrity_fault(self):
        rig = Rig().ready()
        rig.run(0.5, request=(0.2, 0.0))
        rig.x += 5.0  # 위치 급변 → Odometry 신뢰 불가
        rig.next_odom = rig.sim  # 이동 중 바로 다음 Odometry 로 전달
        rig.run(0.1, request=(0.2, 0.0))
        rig.run(5.0)
        self.assertEqual(rig.wd.fault, om.POSITION_JUMP)
        self.assertEqual(
            rig.wd.status(rig.mono)["stop_check"], sc.UNVERIFIABLE
        )

    def test_reset_clears_stop_clock(self):
        rig = Rig().ready()
        rig.run(2.0)
        rig.sim = 0.1
        rig.step()
        self.assertEqual(rig.wd.fault, om.SIM_TIME_REGRESSION)
        rig.run(1.0)
        self.assertTrue(rig.wd.reset(rig.mono)[0])
        self.assertIsNone(rig.wd.clock_sim)
        rig.next_odom = rig.sim
        rig.step()
        self.assertAlmostEqual(rig.wd.clock_sim, rig.sim)

    def test_never_moved(self):
        rig = Rig().ready()
        self.assertEqual(rig.wd.status(rig.mono)["stop_check"], sc.NOT_STARTED)

    def _first_terminal(self, rig, seconds):
        """(경과 mono, stop_check) — 처음으로 확정 결과가 나온 시점."""
        start = rig.mono
        for _ in range(int(round(seconds / TICK))):
            rig.step()
            state = rig.wd.status(rig.mono)["stop_check"]
            if state not in (sc.PENDING, sc.NOT_STARTED):
                return rig.mono - start, state
        return None, sc.PENDING

    def test_clock_lost_during_motion_stop_check_bounded(self):
        # 사용자 실험 재현: 이동 중 Clock Bridge 종료, Gazebo 와 Odometry 는 계속
        rig = Rig().ready()
        rig.run(1.0, request=(0.3, 0.0))
        self.assertEqual(rig.wd.state, wp.ACTIVE)
        rig.clock_on = False
        frozen = rig.wd.clock_sim
        start = rig.mono
        # Executor 는 계속 이동 요청을 보낸다 (자신이 받은 마지막 /clock 으로 stamp)
        for _ in range(int(round(3.0 / TICK))):
            rig.request(0.3, 0.0, stamp=frozen)
            rig.step()
            if rig.wd.state == wp.FAULT:
                break
        self.assertEqual(rig.wd.state, wp.FAULT)
        self.assertIn(
            rig.wd.fault,
            (om.ODOM_STAMP_AHEAD_OF_CLOCK, om.CLOCK_TIMEOUT,
             om.SIM_TIME_STALLED),
        )
        self.assertEqual(rig.wd.output, STOP)
        _, state = self._first_terminal(rig, 20.0)
        elapsed = rig.mono - start
        self.assertEqual(state, sc.UNVERIFIABLE)
        self.assertIsNotNone(elapsed)
        # fault 감지 + monotonic 제한 이내 (sim Timeout 으로는 도달할 수 없음)
        self.assertLessEqual(
            elapsed,
            DEMO_1HZ_CONFIG.clock_receive_timeout_mono
            + DEMO_1HZ_STOP_CHECK.timeout_mono + 1.0,
        )
        # Clock 이 복구되어도 결과는 바뀌지 않는다 (Reset 전까지)
        rig.clock_on = True
        rig.run(5.0)
        self.assertEqual(rig.wd.status(rig.mono)["stop_check"], sc.UNVERIFIABLE)

    def test_clock_and_odom_lost_after_stop_bounded(self):
        # 정지 명령 직후 시뮬레이션 전체가 멈춤 (Clock, Odometry 모두 중단)
        rig = Rig().ready()
        rig.run(1.0, request=(0.3, 0.0))
        rig.request(0.0)
        rig.clock_on = False
        rig.odom_on = False
        elapsed, state = self._first_terminal(rig, 20.0)
        self.assertIn(state, (sc.UNVERIFIABLE,))
        self.assertLessEqual(elapsed, DEMO_1HZ_STOP_CHECK.timeout_mono + 0.1)
        self.assertNotEqual(state, sc.CONFIRMED)

    def test_normal_stop_still_confirmed_with_monotonic_limit(self):
        # monotonic 제한이 정상 경로의 CONFIRMED 를 막지 않는다
        rig = Rig().ready()
        rig.run(1.0, request=(0.2, 0.0))
        rig.request(0.0)
        elapsed, state = self._first_terminal(rig, 10.0)
        self.assertEqual(state, sc.CONFIRMED)
        self.assertLess(elapsed, DEMO_1HZ_STOP_CHECK.timeout_sim)

    def test_stop_check_uses_last_mono_without_argument(self):
        rig = Rig().ready()
        rig.run(1.0, request=(0.3, 0.0))
        rig.request(0.0)
        rig.clock_on = False
        rig.odom_on = False
        rig.run(DEMO_1HZ_STOP_CHECK.timeout_mono + 1.0)
        self.assertEqual(rig.wd.stop_check().state, sc.UNVERIFIABLE)


class TestWatchdogClient(unittest.TestCase):

    def make(self, state=wp.READY, **override):
        rig = Rig().ready()
        status = rig.wd.status(rig.mono)
        status["state"] = state
        status.update(override)
        client = WatchdogClient("vehicle_blue", WD.status_timeout_mono,
                                sender_id="exec1")
        client.on_status(status_to_json(status), 10.0)
        return client

    def test_unavailable(self):
        client = WatchdogClient("vehicle_blue", 0.5)
        self.assertEqual(client.check(1.0).reason, wp.WATCHDOG_UNAVAILABLE)

    def test_ok(self):
        self.assertTrue(self.make().check(10.1).ok)

    def test_status_timeout(self):
        self.assertEqual(
            self.make().check(10.6).reason, wp.WATCHDOG_STATUS_TIMEOUT
        )

    def test_fault(self):
        status = self.make(wp.FAULT, fault=cw.COMMAND_TIMEOUT).check(10.1)
        self.assertFalse(status.ok)
        self.assertEqual(status.reason, "WATCHDOG_FAULT:COMMAND_TIMEOUT")

    def test_not_ready(self):
        for state in (wp.WAITING, wp.SHUTDOWN):
            with self.subTest(state=state):
                status = self.make(state).check(10.1)
                self.assertTrue(status.reason.startswith(wp.WATCHDOG_NOT_READY))

    def test_robot_mismatch(self):
        status = self.make(robot_id="vehicle_green").check(10.1)
        self.assertEqual(status.reason, wp.WATCHDOG_ROBOT_MISMATCH)

    def test_session_changed(self):
        client = self.make()
        self.assertEqual(
            client.check(10.1, pinned_token="other").reason,
            wp.WATCHDOG_SESSION_CHANGED,
        )

    def test_invalid_status_ignored(self):
        client = self.make()
        self.assertFalse(client.on_status("{bad", 10.2))
        self.assertEqual(client.invalid_status_count, 1)

    def test_seq_strictly_increasing_across_sessions(self):
        client = WatchdogClient("vehicle_blue", 0.5, sender_id="e")
        a = decode_request_id(client.next_request_id("t0"))
        b = decode_request_id(client.next_request_id("t1"))
        self.assertLess(a[2], b[2])
        self.assertEqual(b[1], "e")

    def test_sender_ids_differ_per_client(self):
        a = WatchdogClient("vehicle_blue", 0.5)
        b = WatchdogClient("vehicle_blue", 0.5)
        self.assertNotEqual(a.sender_id, b.sender_id)


class TestEndToEndPure(unittest.TestCase):

    def test_client_and_watchdog(self):
        rig = Rig().ready()
        client = WatchdogClient("vehicle_blue", WD.status_timeout_mono)
        client.on_status(status_to_json(rig.wd.status(rig.mono)), rig.mono)
        self.assertTrue(client.check(rig.mono).ok)
        rid = client.next_request_id(client.token)
        result = rig.wd.on_request(rid, rig.sim, 0.2, 0, 0, 0, 0, 0, rig.mono)
        self.assertTrue(result.forwarded)
        self.assertTrue(math.isclose(rig.wd.output[0], 0.2))


if __name__ == "__main__":
    unittest.main()
