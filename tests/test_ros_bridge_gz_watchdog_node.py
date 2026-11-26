"""GazeboWatchdogNode 검증 (Gazebo 불필요, rclpy 없으면 SKIP).

- TestGazeboWatchdogNode: 콜백을 직접 호출. cmd_vel/상태 Publisher 는 기록용 가짜 객체.
- TestDdsIntegration: 실제 DDS 로 Executor → Watchdog → cmd_vel → 가짜 시뮬레이터 연결.
  테스트 전용 ROS_DOMAIN_ID 와 Topic 접두어를 써서 실제 로봇/Gazebo 와 분리한다.
  가짜 시뮬레이터는 명령 속도를 그대로 적분하는 단순 모델이며 Gazebo 물리 검증이 아니다.
"""

import json
import math
import os
import threading
import time
import unittest

try:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from builtin_interfaces.msg import Time
    from geometry_msgs.msg import Twist, TwistStamped
    from nav_msgs.msg import Odometry
    from rosgraph_msgs.msg import Clock
    from std_srvs.srv import Trigger
    from src.ros_bridge.gz_watchdog_node import GazeboWatchdogNode
    from src.ros_bridge.gz_executor_node import GazeboExecutorNode
    from src.ros_bridge.qos import CLOCK_QOS, ODOM_QOS, CMD_QOS
    HAVE_RCLPY = True
except ImportError:  # pragma: no cover - ROS 미설치 환경
    HAVE_RCLPY = False

from src.ros_bridge import command_watchdog as cw
from src.ros_bridge import stop_check as sc
from src.ros_bridge import watchdog_protocol as wp
from src.ros_bridge.safety_config import (
    DEMO_1HZ_CONFIG,
    DEMO_1HZ_WATCHDOG,
    DEMO_1HZ_STOP_CHECK,
    MAPF_50HZ_CONFIG,
    MAPF_50HZ_WATCHDOG,
    MAPF_50HZ_STOP_CHECK,
    make_robot_interface,
)
from src.ros_bridge.watchdog_protocol import encode_request_id

TEST_DOMAIN_ID = 200 + (os.getpid() + 7) % 30
ROBOT = f"wdtest{os.getpid()}"
INTERFACE = make_robot_interface(
    ROBOT, request_prefix=f"/wdtest_{os.getpid()}",
    model_prefix=f"/wdtest_model_{os.getpid()}",
    clock_topic=f"/wdtest_{os.getpid()}/clock",
)


class Recorder:

    def __init__(self, kind):
        self.kind = kind
        self.messages = []

    def publish(self, msg):
        if self.kind == "twist":
            self.messages.append((msg.linear.x, msg.angular.z))
        else:
            self.messages.append(json.loads(msg.data))


def make_time(seconds):
    t = Time()
    t.sec = int(seconds)
    t.nanosec = int(round((seconds - int(seconds)) * 1e9))
    return t


def clock_msg(seconds):
    msg = Clock()
    msg.clock = make_time(seconds)
    return msg


def odom_msg(stamp, x=0.0, vx=0.0, wz=0.0):
    msg = Odometry()
    msg.header.stamp = make_time(stamp)
    msg.header.frame_id = INTERFACE.odom_frame_id
    msg.child_frame_id = INTERFACE.base_frame_id
    msg.pose.pose.position.x = float(x)
    msg.pose.pose.orientation.w = 1.0
    msg.twist.twist.linear.x = float(vx)
    msg.twist.twist.angular.z = float(wz)
    return msg


def request_msg(token, seq, stamp, linear=0.0, angular=0.0, sender="exec1",
                linear_y=0.0):
    msg = TwistStamped()
    msg.header.frame_id = encode_request_id(token, sender, seq)
    msg.header.stamp = make_time(stamp)
    msg.twist.linear.x = float(linear)
    msg.twist.linear.y = float(linear_y)
    msg.twist.angular.z = float(angular)
    return msg


@unittest.skipUnless(HAVE_RCLPY, "rclpy not available")
class TestGazeboWatchdogNode(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init(domain_id=TEST_DOMAIN_ID)

    @classmethod
    def tearDownClass(cls):
        rclpy.try_shutdown()

    def setUp(self):
        self.node = GazeboWatchdogNode(
            INTERFACE, DEMO_1HZ_WATCHDOG, DEMO_1HZ_CONFIG, DEMO_1HZ_STOP_CHECK,
            node_name=f"watchdog_test_{os.getpid()}",
        )
        self.initial_stops = self.node.stop_published_count
        self.cmd = Recorder("twist")
        self.status = Recorder("status")
        self.node.cmd_pub = self.cmd
        self.node.status_pub = self.status
        self.sim = 1.0
        self.seq = 0

    def tearDown(self):
        self.node.destroy_node()

    def feed(self, advance=0.0):
        self.sim += advance
        self.node.clock_callback(clock_msg(self.sim))
        self.node.odom_callback(odom_msg(self.sim))

    def request(self, linear, angular=0.0, **kw):
        self.seq += 1
        self.node.request_callback(request_msg(
            kw.pop("token", self.node.core.token), self.seq, self.sim,
            linear, angular, **kw
        ))

    def test_only_cmd_vel_publisher_is_watchdog(self):
        node = GazeboWatchdogNode(
            INTERFACE, node_name=f"watchdog_pub_check_{os.getpid()}"
        )
        try:
            topics = sorted(p.topic_name for p in node.publishers)
            self.assertIn(INTERFACE.cmd_vel_topic, topics)
            self.assertIn(INTERFACE.watchdog_status_topic, topics)
            self.assertNotIn(INTERFACE.cmd_request_topic, topics)
        finally:
            node.destroy_node()

    def test_initial_stop(self):
        self.assertGreaterEqual(self.initial_stops, 1)
        self.node.tick()
        self.assertEqual(self.cmd.messages, [(0.0, 0.0)])
        self.assertEqual(self.status.messages[-1]["state"], wp.WAITING)

    def test_forward_valid_request(self):
        self.feed()
        self.node.tick()
        self.assertEqual(self.status.messages[-1]["state"], wp.READY)
        self.request(0.2, 0.1)
        self.assertEqual(self.cmd.messages[-1], (0.2, 0.1))

    def test_command_timeout(self):
        self.feed()
        self.node.tick()
        self.request(0.2)
        time.sleep(DEMO_1HZ_WATCHDOG.command_timeout_mono + 0.1)
        self.feed(advance=0.6)
        self.node.tick()
        self.assertEqual(self.cmd.messages[-1], (0.0, 0.0))
        status = self.status.messages[-1]
        self.assertEqual(status["state"], wp.FAULT)
        self.assertEqual(status["fault"], cw.COMMAND_TIMEOUT)

    def test_invalid_request_latches(self):
        self.feed()
        self.node.tick()
        self.request(0.1, linear_y=0.2)
        self.assertEqual(self.cmd.messages[-1], (0.0, 0.0))
        self.assertEqual(self.node.core.fault, cw.UNUSED_TWIST_COMPONENT)
        self.request(0.31)
        self.assertEqual(self.cmd.messages[-1], (0.0, 0.0))

    def test_reset_service_and_old_session(self):
        self.feed()
        self.node.tick()
        self.request(0.2)
        old_token = self.node.core.token
        self.node.core.monitor.latch("TEST_FAULT")

        # 요청이 아직 들어오는 중이면 Reset 거부
        response = self.node.reset_callback(
            Trigger.Request(), Trigger.Response()
        )
        self.assertFalse(response.success)

        time.sleep(DEMO_1HZ_WATCHDOG.command_timeout_mono + 0.1)
        response = self.node.reset_callback(
            Trigger.Request(), Trigger.Response()
        )
        self.assertTrue(response.success, response.message)
        self.assertNotEqual(self.node.core.token, old_token)

        self.feed(advance=0.1)
        self.node.tick()
        self.assertEqual(self.status.messages[-1]["state"], wp.READY)
        before = len(self.cmd.messages)
        self.request(0.2, token=old_token)  # Reset 전 세션의 지연 요청
        self.node.tick()
        self.assertTrue(
            all(m == (0.0, 0.0) for m in self.cmd.messages[before:])
        )

    def test_odom_fault_stops(self):
        self.feed()
        self.node.tick()
        self.request(0.2)
        self.node.clock_callback(clock_msg(self.sim + 1.0))
        self.node.odom_callback(odom_msg(self.sim + 1.0, x=5.0))
        self.assertEqual(self.cmd.messages[-1], (0.0, 0.0))
        self.assertEqual(self.status.messages[-1]["fault"], "POSITION_JUMP")

    def test_shutdown_stop(self):
        self.feed()
        self.node.tick()
        self.request(0.2)
        count = self.node.shutdown_stop()
        self.assertEqual(count, DEMO_1HZ_WATCHDOG.shutdown_stop_repeats)
        self.assertEqual(self.cmd.messages[-count:], [(0.0, 0.0)] * count)
        self.assertEqual(self.status.messages[-1]["state"], wp.SHUTDOWN)


class FakeSim(Node if HAVE_RCLPY else object):
    """cmd_vel 을 적분하는 단순 차동구동 모델. /clock 100 Hz, Odometry 50 Hz."""

    def __init__(self, interface):
        super().__init__(f"fake_sim_{os.getpid()}")
        self.interface = interface
        self.lock = threading.Lock()
        self.sim = 1.0
        self.x = self.y = self.yaw = 0.0
        self.cmd = (0.0, 0.0)
        self.cmd_count = 0
        self.ticks = 0
        self.ignore_commands = False
        self.create_subscription(
            Twist, interface.cmd_vel_topic, self.on_cmd, CMD_QOS
        )
        self.clock_pub = self.create_publisher(
            Clock, interface.clock_topic, CLOCK_QOS
        )
        self.odom_pub = self.create_publisher(
            Odometry, interface.odom_topic, ODOM_QOS
        )
        self.create_timer(0.01, self.step)

    def on_cmd(self, msg):
        with self.lock:
            self.cmd_count += 1
            if not self.ignore_commands:
                self.cmd = (msg.linear.x, msg.angular.z)

    def step(self):
        with self.lock:
            dt = 0.01
            self.sim += dt
            linear, angular = self.cmd
            self.yaw += angular * dt
            self.x += linear * math.cos(self.yaw) * dt
            self.y += linear * math.sin(self.yaw) * dt
            self.ticks += 1
            self.clock_pub.publish(clock_msg(self.sim))
            if self.ticks % 2 == 0:
                msg = Odometry()
                msg.header.stamp = make_time(self.sim)
                msg.header.frame_id = self.interface.odom_frame_id
                msg.child_frame_id = self.interface.base_frame_id
                msg.pose.pose.position.x = self.x
                msg.pose.pose.position.y = self.y
                msg.pose.pose.orientation.z = math.sin(self.yaw / 2)
                msg.pose.pose.orientation.w = math.cos(self.yaw / 2)
                msg.twist.twist.linear.x = linear
                msg.twist.twist.angular.z = angular
                self.odom_pub.publish(msg)


@unittest.skipUnless(HAVE_RCLPY, "rclpy not available")
class TestDdsIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init(domain_id=TEST_DOMAIN_ID)

    @classmethod
    def tearDownClass(cls):
        rclpy.try_shutdown()

    def setUp(self):
        self.interface = make_robot_interface(
            f"dds{os.getpid()}{self.id().rsplit('_', 1)[-1]}",
            request_prefix=f"/wdint_{os.getpid()}",
            model_prefix=f"/wdint_model_{os.getpid()}",
            clock_topic=f"/wdint_{os.getpid()}/clock_{self.id().rsplit('_', 1)[-1]}",
        )
        self.watchdog = GazeboWatchdogNode(
            self.interface, MAPF_50HZ_WATCHDOG, MAPF_50HZ_CONFIG,
            MAPF_50HZ_STOP_CHECK,
        )
        self.sim = FakeSim(self.interface)
        self.background = SingleThreadedExecutor()
        self.background.add_node(self.watchdog)
        self.background.add_node(self.sim)
        self.thread = threading.Thread(
            target=self.background.spin, daemon=True
        )
        self.thread.start()
        self.executor_node = GazeboExecutorNode(
            config=MAPF_50HZ_CONFIG,
            interface=self.interface,
            node_name=f"executor_int_{os.getpid()}",
            watchdog_config=MAPF_50HZ_WATCHDOG,
            stop_config=MAPF_50HZ_STOP_CHECK,
        )

    def tearDown(self):
        self.background.shutdown(timeout_sec=2.0)
        self.thread.join(timeout=2.0)
        self.executor_node.destroy_node()
        self.watchdog.destroy_node()
        self.sim.destroy_node()

    def test_move_through_watchdog_and_confirm_stop(self):
        status = self.executor_node.wait_until_ready(timeout_mono=10.0)
        self.assertTrue(status.ok, status)

        self.assertTrue(
            self.executor_node.move_to_target(0.3, 0.0),
            self.executor_node.last_move_result,
        )
        with self.sim.lock:
            self.assertGreater(self.sim.x, 0.2)

        stop = self.executor_node.verify_stop(timeout_mono=10.0)
        self.assertTrue(stop.confirmed, stop)
        self.assertNotEqual(self.watchdog.core.state, wp.FAULT)

    def test_executor_silence_triggers_watchdog_stop(self):
        # Executor 가 이동 명령을 보낸 뒤 사라지는 상황 (SIGKILL 과 같은 관측)
        status = self.executor_node.wait_until_ready(timeout_mono=10.0)
        self.assertTrue(status.ok, status)
        self.executor_node.publish_velocity(0.2, 0.0)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if self.watchdog.core.fault == cw.COMMAND_TIMEOUT:
                break
            time.sleep(0.02)
        self.assertEqual(self.watchdog.core.fault, cw.COMMAND_TIMEOUT)
        time.sleep(0.1)
        with self.sim.lock:
            self.assertEqual(self.sim.cmd, (0.0, 0.0))

        # Watchdog 가 실제 정지를 Odometry 로 확인
        deadline = time.monotonic() + 5.0
        state = None
        while time.monotonic() < deadline:
            state = self.watchdog.core.status(time.monotonic())["stop_check"]
            if state != sc.PENDING:
                break
            time.sleep(0.05)
        self.assertEqual(state, sc.CONFIRMED)


if __name__ == "__main__":
    unittest.main()
