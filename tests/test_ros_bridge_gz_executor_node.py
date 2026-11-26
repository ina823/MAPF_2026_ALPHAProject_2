"""GazeboExecutorNode 의 rclpy 연결부 검증 (Gazebo 불필요, rclpy 없으면 SKIP).

실제 로봇에 명령이 나가지 않도록
- 테스트 전용 Topic 이름과 별도 ROS_DOMAIN_ID 를 쓰고,
- 이동 요청 Publisher 를 기록용 가짜 객체로 교체한다.
메시지는 DDS 를 거치지 않고 콜백을 직접 호출해 주입한다.
Watchdog 상태도 콜백으로 직접 주입한다 (Watchdog 실행 상태를 흉내 냄).
"""

import os
import time
import unittest
from dataclasses import replace

try:
    import rclpy
    from builtin_interfaces.msg import Time
    from nav_msgs.msg import Odometry
    from rosgraph_msgs.msg import Clock
    from std_msgs.msg import String
    from src.ros_bridge.gz_executor_node import (
        GazeboExecutorNode,
        ODOM_QOS,
        CLOCK_QOS,
        stamp_to_seconds,
    )
    from rclpy.qos import QoSHistoryPolicy, QoSReliabilityPolicy
    HAVE_RCLPY = True
except ImportError:  # pragma: no cover - ROS 미설치 환경
    HAVE_RCLPY = False

from src.ros_bridge import odom_monitor as om
from src.ros_bridge import watchdog_protocol as wp
from src.ros_bridge.safety_config import (
    DEMO_1HZ_CONFIG,
    DEMO_1HZ_WATCHDOG,
    MAPF_50HZ_CONFIG,
    MAPF_50HZ_WATCHDOG,
    RobotInterface,
)
from src.ros_bridge.watchdog_protocol import (
    decode_request_id,
    status_to_json,
)

PREFIX = f"/odom_safety_test_{os.getpid()}"
INTERFACE = RobotInterface(
    robot_id="vehicle_blue",
    cmd_request_topic=PREFIX + "/cmd_vel_request",
    cmd_vel_topic=PREFIX + "/cmd_vel",
    odom_topic=PREFIX + "/odometry",
    clock_topic=PREFIX + "/clock",
    watchdog_status_topic=PREFIX + "/watchdog_status",
    watchdog_reset_service=PREFIX + "/watchdog_reset",
    odom_frame_id="vehicle_blue/odom",
    base_frame_id="vehicle_blue/chassis",
)
TEST_DOMAIN_ID = 200 + os.getpid() % 30

# 기존 Odometry 안전성 테스트가 Watchdog 상태 Timeout 에 가려지지 않도록
# 상태 신뢰 시간을 길게 둔다. Watchdog 상태 Timeout 자체는 별도 테스트에서 기본값으로 검증.
LONG_STATUS_WATCHDOG = replace(DEMO_1HZ_WATCHDOG, status_timeout_mono=30.0)


def watchdog_status_msg(state=wp.READY, token="boot-0", fault=None,
                        robot_id="vehicle_blue"):
    msg = String()
    msg.data = status_to_json({
        "robot_id": robot_id, "state": state, "fault": fault,
        "fault_detail": None, "token": token, "epoch": 0,
        "output": [0.0, 0.0], "stop_check": "NOT_STARTED",
        "stop_detail": "", "health": "OK",
    })
    return msg


class RecordingPublisher:

    def __init__(self):
        self.messages = []
        self.raw = []

    def publish(self, msg):
        self.raw.append(msg)
        self.messages.append((msg.twist.linear.x, msg.twist.angular.z))


def make_time(seconds):
    t = Time()
    t.sec = int(seconds)
    t.nanosec = int(round((seconds - int(seconds)) * 1e9))
    return t


def clock_msg(seconds):
    msg = Clock()
    msg.clock = make_time(seconds)
    return msg


def odom_msg(stamp, x=0.0, y=0.0, frame="vehicle_blue/odom",
             child="vehicle_blue/chassis", q=(0.0, 0.0, 0.0, 1.0)):
    msg = Odometry()
    msg.header.stamp = make_time(stamp)
    msg.header.frame_id = frame
    msg.child_frame_id = child
    msg.pose.pose.position.x = float(x)
    msg.pose.pose.position.y = float(y)
    (msg.pose.pose.orientation.x,
     msg.pose.pose.orientation.y,
     msg.pose.pose.orientation.z,
     msg.pose.pose.orientation.w) = (float(v) for v in q)
    return msg


@unittest.skipUnless(HAVE_RCLPY, "rclpy not available")
class TestGazeboExecutorNode(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init(domain_id=TEST_DOMAIN_ID)

    @classmethod
    def tearDownClass(cls):
        rclpy.try_shutdown()

    def setUp(self):
        self.node = GazeboExecutorNode(
            config=DEMO_1HZ_CONFIG,
            interface=INTERFACE,
            node_name=f"odom_safety_test_{os.getpid()}",
            watchdog_config=LONG_STATUS_WATCHDOG,
        )
        self.pub = RecordingPublisher()
        self.node.cmd_pub = self.pub
        self.node.watchdog_status_callback(watchdog_status_msg())

    def tearDown(self):
        self.node.destroy_node()

    def feed_ready(self, sim=1.0):
        self.node.clock_callback(clock_msg(sim))
        self.node.odom_callback(odom_msg(sim))

    # --- 설정 / QoS ---

    def test_qos_keeps_latest_only(self):
        for qos in (ODOM_QOS, CLOCK_QOS):
            self.assertEqual(qos.history, QoSHistoryPolicy.KEEP_LAST)
            self.assertEqual(qos.depth, 1)
        self.assertEqual(
            CLOCK_QOS.reliability, QoSReliabilityPolicy.BEST_EFFORT
        )

    def test_uses_given_topics(self):
        self.assertEqual(self.node.odom_sub.topic_name, INTERFACE.odom_topic)
        self.assertEqual(
            self.node.clock_sub.topic_name, INTERFACE.clock_topic
        )

    def test_stamp_conversion(self):
        self.assertAlmostEqual(stamp_to_seconds(make_time(12.25)), 12.25)

    def test_config_selectable(self):
        node = GazeboExecutorNode(
            config=MAPF_50HZ_CONFIG,
            interface=INTERFACE,
            node_name=f"odom_safety_test_50hz_{os.getpid()}",
            watchdog_config=MAPF_50HZ_WATCHDOG,
        )
        try:
            self.assertIs(node.monitor.config, MAPF_50HZ_CONFIG)
        finally:
            node.destroy_node()

    # --- 콜백 ---

    def test_valid_messages_ready(self):
        self.feed_ready()
        status = self.node.safety_status()
        self.assertTrue(status.ok, status)
        self.assertEqual(self.node.pose, (0.0, 0.0, 0.0))
        self.assertIsNotNone(self.node.last_odom_time)

    def test_header_stamp_is_used_not_receive_time(self):
        self.node.clock_callback(clock_msg(10.0))
        self.node.odom_callback(odom_msg(8.0))
        self.assertEqual(
            self.node.safety_status().reason, om.ODOM_STAMP_STALE
        )

    def test_duplicate_does_not_refresh(self):
        self.feed_ready()
        first = self.node.last_odom_time
        time.sleep(0.01)
        self.node.odom_callback(odom_msg(1.0))
        self.assertEqual(self.node.last_odom_time, first)
        self.assertEqual(self.node.monitor.duplicate_count, 1)
        self.assertEqual(self.pub.messages, [])

    def test_invalid_frame_publishes_stop(self):
        self.node.clock_callback(clock_msg(1.0))
        self.node.odom_callback(odom_msg(1.0, frame="vehicle_green/odom"))
        self.assertEqual(self.node.monitor.fault, om.INVALID_FRAME)
        self.assertEqual(self.pub.messages[-1], (0.0, 0.0))
        self.assertIsNone(self.node.pose)

    def test_invalid_quaternion_publishes_stop(self):
        self.node.clock_callback(clock_msg(1.0))
        self.node.odom_callback(odom_msg(1.0, q=(0.0, 0.0, 0.0, 0.0)))
        self.assertEqual(self.node.monitor.fault, om.INVALID_QUATERNION)
        self.assertEqual(self.pub.messages[-1], (0.0, 0.0))

    def test_nan_position_publishes_stop(self):
        self.node.clock_callback(clock_msg(1.0))
        self.node.odom_callback(odom_msg(1.0, x=float("nan")))
        self.assertEqual(self.node.monitor.fault, om.INVALID_POSITION)
        self.assertEqual(self.pub.messages[-1], (0.0, 0.0))

    def test_position_jump_publishes_stop(self):
        self.feed_ready(1.0)
        self.node.clock_callback(clock_msg(2.0))
        self.node.odom_callback(odom_msg(2.0, x=3.0))
        self.assertEqual(self.node.monitor.fault, om.POSITION_JUMP)
        self.assertEqual(self.pub.messages[-1], (0.0, 0.0))

    def test_clock_regression_publishes_stop(self):
        self.feed_ready(5.0)
        self.node.clock_callback(clock_msg(0.1))
        self.assertEqual(self.node.monitor.fault, om.SIM_TIME_REGRESSION)
        self.assertEqual(self.pub.messages[-1], (0.0, 0.0))

    # --- move_to_target ---

    def test_move_blocked_without_odom(self):
        self.node.clock_callback(clock_msg(1.0))
        self.assertFalse(self.node.move_to_target(1.0, 0.0))
        self.assertEqual(self.node.last_move_result.reason, om.NO_ODOM)
        self.assertTrue(self.node.last_move_result.blocked_before_start)
        self.assertTrue(self.pub.messages)
        self.assertTrue(all(m == (0.0, 0.0) for m in self.pub.messages))

    def test_move_blocked_without_clock(self):
        self.node.odom_callback(odom_msg(1.0))
        self.assertFalse(self.node.move_to_target(1.0, 0.0))
        self.assertEqual(self.node.last_move_result.reason, om.NO_CLOCK)
        self.assertTrue(all(m == (0.0, 0.0) for m in self.pub.messages))

    def test_move_blocked_after_fault(self):
        self.node.clock_callback(clock_msg(1.0))
        self.node.odom_callback(odom_msg(1.0, frame="vehicle_green/odom"))
        self.node.odom_callback(odom_msg(2.0))  # 정상 메시지여도 잠금 유지
        self.assertFalse(self.node.move_to_target(1.0, 0.0))
        self.assertEqual(
            self.node.last_move_result.reason, om.INVALID_FRAME
        )
        self.assertTrue(all(m == (0.0, 0.0) for m in self.pub.messages))

    def test_move_aborts_when_feedback_stops(self):
        # 실제 rclpy.spin_once 를 쓰되 아무도 발행하지 않으므로
        # /clock 공백 → CLOCK_TIMEOUT 으로 중단되어야 한다 (약 1 s).
        self.feed_ready()
        start = time.monotonic()
        self.assertFalse(self.node.move_to_target(1.0, 0.0))
        elapsed = time.monotonic() - start
        result = self.node.last_move_result
        self.assertEqual(result.reason, om.CLOCK_TIMEOUT)
        self.assertFalse(result.blocked_before_start)
        self.assertTrue(any(m != (0.0, 0.0) for m in self.pub.messages))
        self.assertEqual(self.pub.messages[-1], (0.0, 0.0))
        self.assertLess(
            elapsed, DEMO_1HZ_CONFIG.clock_receive_timeout_mono + 0.5
        )
        self.assertTrue(self.node.monitor.faulted)



@unittest.skipUnless(HAVE_RCLPY, "rclpy not available")
class TestExecutorWatchdogLink(unittest.TestCase):
    """Executor ↔ Watchdog 연결: 우회 금지, 미실행/Fault 시 출발 차단."""

    @classmethod
    def setUpClass(cls):
        rclpy.init(domain_id=TEST_DOMAIN_ID)

    @classmethod
    def tearDownClass(cls):
        rclpy.try_shutdown()

    def setUp(self):
        self.node = GazeboExecutorNode(
            config=DEMO_1HZ_CONFIG,
            interface=INTERFACE,
            node_name=f"executor_link_test_{os.getpid()}",
        )
        self.pub = RecordingPublisher()
        self.node.cmd_pub = self.pub

    def tearDown(self):
        self.node.destroy_node()

    def feed_ready(self, sim=1.0):
        self.node.clock_callback(clock_msg(sim))
        self.node.odom_callback(odom_msg(sim))

    def test_no_direct_cmd_vel_publisher(self):
        node = GazeboExecutorNode(
            config=DEMO_1HZ_CONFIG, interface=INTERFACE,
            node_name=f"executor_pub_check_{os.getpid()}",
        )
        try:
            topics = [p.topic_name for p in node.publishers]
            self.assertIn(INTERFACE.cmd_request_topic, topics)
            self.assertNotIn(INTERFACE.cmd_vel_topic, topics)
        finally:
            node.destroy_node()

    def test_blocked_when_watchdog_not_running(self):
        self.feed_ready()
        self.assertFalse(self.node.move_to_target(1.0, 0.0))
        result = self.node.last_move_result
        self.assertEqual(result.reason, wp.WATCHDOG_UNAVAILABLE)
        self.assertTrue(result.blocked_before_start)
        # 세션을 모르므로 요청 자체를 보내지 않는다 (이동 명령 0건)
        self.assertEqual(self.pub.messages, [])
        self.assertFalse(self.node.monitor.faulted)

    def test_moving_request_without_session_raises(self):
        with self.assertRaises(RuntimeError):
            self.node.publish_velocity(0.1, 0.0)
        self.assertFalse(self.node.stop_robot())

    def test_blocked_when_watchdog_faulted(self):
        self.feed_ready()
        self.node.watchdog_status_callback(
            watchdog_status_msg(wp.FAULT, fault="COMMAND_TIMEOUT")
        )
        self.assertFalse(self.node.move_to_target(1.0, 0.0))
        self.assertEqual(
            self.node.last_move_result.reason,
            "WATCHDOG_FAULT:COMMAND_TIMEOUT",
        )
        self.assertTrue(all(m == (0.0, 0.0) for m in self.pub.messages))

    def test_blocked_when_watchdog_waiting(self):
        self.feed_ready()
        self.node.watchdog_status_callback(watchdog_status_msg(wp.WAITING))
        self.assertFalse(self.node.move_to_target(1.0, 0.0))
        self.assertTrue(
            self.node.last_move_result.reason.startswith(
                wp.WATCHDOG_NOT_READY
            )
        )

    def test_watchdog_status_visible(self):
        self.node.watchdog_status_callback(
            watchdog_status_msg(wp.FAULT, fault="SPEED_LIMIT")
        )
        link, status = self.node.watchdog_status()
        self.assertFalse(link.ok)
        self.assertEqual(status["fault"], "SPEED_LIMIT")

    def test_request_format(self):
        self.feed_ready(3.5)
        self.node.watchdog_status_callback(watchdog_status_msg(token="tok-7"))
        self.node.publish_velocity(0.1, 0.2)
        self.node.publish_velocity(0.0, 0.0)
        first, second = self.pub.raw
        token, sender, seq1 = decode_request_id(first.header.frame_id)
        _, sender2, seq2 = decode_request_id(second.header.frame_id)
        self.assertEqual(token, "tok-7")
        self.assertEqual(sender, sender2)
        self.assertLess(seq1, seq2)
        self.assertAlmostEqual(stamp_to_seconds(first.header.stamp), 3.5)
        tw = first.twist
        self.assertEqual(
            (tw.linear.y, tw.linear.z, tw.angular.x, tw.angular.y),
            (0.0, 0.0, 0.0, 0.0),
        )

    def test_abort_when_watchdog_status_stops(self):
        # 기본 status_timeout (0.5 s): 이동 중 Watchdog 상태가 끊기면 중단
        self.feed_ready()
        self.node.watchdog_status_callback(watchdog_status_msg())
        start = time.monotonic()
        self.assertFalse(self.node.move_to_target(1.0, 0.0))
        elapsed = time.monotonic() - start
        result = self.node.last_move_result
        self.assertEqual(result.reason, wp.WATCHDOG_STATUS_TIMEOUT)
        self.assertFalse(result.blocked_before_start)
        self.assertEqual(self.pub.messages[-1], (0.0, 0.0))
        self.assertLess(elapsed, DEMO_1HZ_WATCHDOG.status_timeout_mono + 0.5)
        self.assertTrue(self.node.monitor.faulted)

    def test_abort_when_session_changes(self):
        self.feed_ready()
        self.node.watchdog_status_callback(watchdog_status_msg(token="t-0"))
        gate = self.node._watchdog_gate()
        now = time.monotonic()
        self.assertTrue(gate(now).ok)
        self.node.watchdog_status_callback(watchdog_status_msg(token="t-1"))
        self.assertEqual(
            gate(time.monotonic()).reason, wp.WATCHDOG_SESSION_CHANGED
        )

    def test_stale_stop_feedback_rejected(self):
        # 정지 명령 이전에 받은 twist=0 Odometry 로는 정지를 확인하지 않는다
        self.node.watchdog_status_callback(watchdog_status_msg())
        self.feed_ready(5.0)
        result = self.node.verify_stop(timeout_mono=0.3)
        self.assertFalse(result.confirmed)
        self.assertEqual(self.node.stop_verifier.post_settle_count, 0)
        # 정지 요청 자체는 발행됨 (발행 성공 ≠ 정지 확인)
        self.assertTrue(self.pub.messages)
        self.assertTrue(all(m == (0.0, 0.0) for m in self.pub.messages))

    def test_shutdown_stop_repeats(self):
        self.node.watchdog_status_callback(watchdog_status_msg())
        count = self.node.shutdown_stop()
        self.assertEqual(count, DEMO_1HZ_WATCHDOG.shutdown_stop_repeats)
        self.assertEqual(len(self.pub.messages), count)
        self.assertTrue(all(m == (0.0, 0.0) for m in self.pub.messages))


if __name__ == "__main__":
    unittest.main()
