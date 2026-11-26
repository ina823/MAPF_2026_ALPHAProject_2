"""MAPF Executor Node.

실제 cmd_vel 을 직접 발행하지 않는다. 모든 속도 명령은 독립 Watchdog 의
이동 요청 Topic 으로만 보낸다 (Executor → 요청 → Watchdog → cmd_vel).
"""

import math
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String

from src.ros_bridge.odom_monitor import (
    OdomSafetyMonitor,
    ACCEPTED,
    IGNORED_DUPLICATE,
    IGNORED_OUT_OF_ORDER,
    IGNORED_FAULTED,
)
from src.ros_bridge.qos import (  # noqa: F401  (기존 import 경로 호환)
    ODOM_QOS,
    CLOCK_QOS,
    CMD_QOS,
    REQUEST_QOS,
    STATUS_QOS,
)
from src.ros_bridge.safe_motion import run_move, wait_until_ready
from src.ros_bridge.safety_config import (
    DEMO_1HZ_CONFIG,
    DEMO_1HZ_WATCHDOG,
    DEMO_1HZ_STOP_CHECK,
    DEMO_VEHICLE_BLUE,
)
from src.ros_bridge.shutdown import repeat_stop
from src.ros_bridge.stop_check import StopVerifier, StopCheck, NO_FRESH_ODOM
from src.ros_bridge.watchdog_protocol import WatchdogClient


# 한 제어 주기에 처리할 최대 콜백 수 (구독 3개 × 큐 깊이 1 + 여유)
MAX_CALLBACKS_PER_SPIN = 8

_IGNORED = (IGNORED_DUPLICATE, IGNORED_OUT_OF_ORDER, IGNORED_FAULTED)

NO_SESSION = "NO_WATCHDOG_SESSION"


def stamp_to_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def seconds_to_stamp(seconds, stamp):
    sec = int(math.floor(seconds))
    stamp.sec = sec
    stamp.nanosec = min(int(round((seconds - sec) * 1e9)), 999_999_999)
    return stamp


class GazeboExecutorNode(Node):

    def __init__(
        self,
        config=DEMO_1HZ_CONFIG,
        interface=DEMO_VEHICLE_BLUE,
        node_name="mapf_gz_executor",
        watchdog_config=DEMO_1HZ_WATCHDOG,
        stop_config=DEMO_1HZ_STOP_CHECK,
    ):
        super().__init__(node_name)

        watchdog_config.validate_against(config)

        self.config = config
        self.interface = interface
        self.watchdog_config = watchdog_config
        self.stop_config = stop_config
        self.monitor = OdomSafetyMonitor(
            config,
            interface.odom_frame_id,
            interface.base_frame_id,
        )
        self.watchdog_client = WatchdogClient(
            interface.robot_id, watchdog_config.status_timeout_mono
        )
        self.stop_verifier = StopVerifier(stop_config)
        self.last_move_result = None
        self.unsent_command_count = 0

        # 이동 요청 Publisher 하나뿐. 실제 cmd_vel Publisher 는 만들지 않는다.
        self.cmd_pub = self.create_publisher(
            TwistStamped,
            interface.cmd_request_topic,
            REQUEST_QOS
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            interface.odom_topic,
            self.odom_callback,
            ODOM_QOS
        )

        self.clock_sub = self.create_subscription(
            Clock,
            interface.clock_topic,
            self.clock_callback,
            CLOCK_QOS
        )

        self.status_sub = self.create_subscription(
            String,
            interface.watchdog_status_topic,
            self.watchdog_status_callback,
            STATUS_QOS
        )

        self.get_logger().info(
            f"Gazebo Executor initialized (safety config: {config.name}, "
            f"request topic: {interface.cmd_request_topic})"
        )

    # ------------------------------------------------------------------
    # 기존 인터페이스 호환
    # ------------------------------------------------------------------

    @property
    def pose(self):
        """마지막 수락 pose (fault 시 None). 신선도는 safety_status() 로 확인."""
        return self.monitor.pose

    @property
    def last_odom_time(self):
        """마지막으로 '수락한' Odometry 의 수신 시각 (time.monotonic)."""
        return self.monitor.last_odom_received_mono

    def safety_status(self):
        return self.monitor.check(time.monotonic())

    def watchdog_status(self):
        """(LinkStatus, 마지막 Watchdog 상태 dict 또는 None)."""
        return (
            self.watchdog_client.check(time.monotonic()),
            self.watchdog_client.status,
        )

    def reset_safety(self):
        """Executor 측 잠금 해제. Watchdog Reset 과 별개이며 자동 호출하지 않는다."""
        self.monitor.reset()
        self.stop_verifier.cancel()

    # ------------------------------------------------------------------
    # 콜백
    # ------------------------------------------------------------------

    def clock_callback(self, msg):
        received_mono = time.monotonic()
        fault = self.monitor.on_clock(
            stamp_to_seconds(msg.clock), received_mono
        )
        if fault is not None:
            self._on_fault(fault)

    def odom_callback(self, msg):
        received_mono = time.monotonic()

        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        twist = msg.twist.twist
        stamp = stamp_to_seconds(msg.header.stamp)

        result = self.monitor.on_odom(
            stamp,
            msg.header.frame_id,
            msg.child_frame_id,
            position.x,
            position.y,
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
            received_mono,
            linear_x=twist.linear.x,
            linear_y=twist.linear.y,
            angular_z=twist.angular.z,
        )

        if result == ACCEPTED:
            self.stop_verifier.add_sample(
                stamp, twist.linear.x, twist.linear.y, twist.angular.z
            )
            return

        if result in _IGNORED:
            if result != IGNORED_FAULTED:
                self.get_logger().warning(f"{result} (stamp={stamp:.6f})")
            return

        self._on_fault(result)

    def watchdog_status_callback(self, msg):
        if not self.watchdog_client.on_status(msg.data, time.monotonic()):
            self.get_logger().warning("INVALID_WATCHDOG_STATUS")

    def _on_fault(self, reason):
        self.get_logger().error(
            f"ODOM_SAFETY_FAULT: {reason} ({self.monitor.fault_detail})"
        )
        if rclpy.ok():
            self.stop_robot()

    # ------------------------------------------------------------------
    # 명령 (모두 Watchdog 요청 Topic 으로)
    # ------------------------------------------------------------------

    def publish_velocity(self, linear, angular):
        """이동 요청 발행. 성공해도 Watchdog 가 전달했다는 보장은 없다."""
        if not all(math.isfinite(v) for v in (linear, angular)):
            raise ValueError("Invalid velocity")

        token = self.watchdog_client.token
        moving = linear != 0.0 or angular != 0.0

        if token is None:
            if moving:
                raise RuntimeError(f"{NO_SESSION}: watchdog session unknown")
            # 세션을 모르면 정지 요청도 Watchdog 가 받아 주지 않는다.
            # Watchdog 는 요청이 없으면 기본적으로 정지를 출력한다.
            self.unsent_command_count += 1
            return False

        msg = TwistStamped()
        msg.header.frame_id = self.watchdog_client.next_request_id(token)
        clock = self.monitor.last_clock_sim
        if clock is not None:
            seconds_to_stamp(clock, msg.header.stamp)
        msg.twist.linear.x = float(linear)
        msg.twist.angular.z = float(angular)

        self.cmd_pub.publish(msg)
        return True

    def stop_robot(self):
        """정지 요청 발행. 실제 물리적 정지는 확인하지 않는다 (verify_stop 사용)."""
        return self.publish_velocity(0.0, 0.0)

    def _watchdog_gate(self):
        pinned = {}

        def gate(now_mono):
            if "token" not in pinned:
                status = self.watchdog_client.check(now_mono)
                if status.ok:
                    pinned["token"] = self.watchdog_client.token
                return status
            return self.watchdog_client.check(
                now_mono, pinned_token=pinned["token"]
            )

        return gate

    def wait_until_ready(self, timeout_mono=None):
        """clock + 신선한 Odometry + Watchdog READY 까지 대기."""
        if timeout_mono is None:
            timeout_mono = self.config.startup_timeout_mono
        return wait_until_ready(
            self.monitor, _NodeMotionIO(self), timeout_mono,
            gate=lambda now: self.watchdog_client.check(now),
        )

    def move_to_target(self, target_x, target_y):
        result = run_move(
            target_x, target_y,
            self.monitor,
            _NodeMotionIO(self),
            gate=self._watchdog_gate(),
        )
        self.last_move_result = result
        return result.success

    # ------------------------------------------------------------------
    # 실제 정지 확인
    # ------------------------------------------------------------------

    def verify_stop(self, timeout_mono=None):
        """정지 요청을 보내며 Odometry twist 로 실제 정지를 확인한다.

        새 Odometry(stamp > 정지 시점 + settle)만 사용한다. 오래된 Odometry 로는
        확인하지 않는다. 반환: StopCheck (CONFIRMED / NOT_STOPPED / NO_FRESH_ODOM /
        UNVERIFIABLE / PENDING). /clock 이 멈추면 StopVerifier 의 monotonic 제한
        (``timeout_mono``) 으로 UNVERIFIABLE 이 된다.
        """
        cfg = self.stop_config
        if timeout_mono is None:
            timeout_mono = max(cfg.timeout_sim * 1.5, cfg.timeout_mono) + 1.0

        clock = self.monitor.last_clock_sim
        if clock is None:
            return StopCheck(NO_FRESH_ODOM, "no clock")

        self.stop_robot()
        io = _NodeMotionIO(self)
        start = time.monotonic()
        self.stop_verifier.start(clock, start)
        result = self.stop_verifier.evaluate(clock, start)

        while rclpy.ok() and time.monotonic() - start <= timeout_mono:
            io.spin()
            result = self.stop_verifier.evaluate(
                self.monitor.last_clock_sim, time.monotonic()
            )
            if result.terminal:
                return result
            self.stop_robot()
            time.sleep(self.config.control_period_mono)

        return result

    def shutdown_stop(self):
        """종료 시 정지 요청 반복 발행. 발행 횟수 (실제 정지 확인 아님)."""
        return repeat_stop(
            self.stop_robot,
            self.watchdog_config.shutdown_stop_repeats,
            self.watchdog_config.shutdown_stop_interval_mono,
        )


class _NodeMotionIO:
    """safe_motion 이 요구하는 입출력을 rclpy Node 로 구현한다."""

    def __init__(self, node):
        self.node = node

    def ok(self):
        return rclpy.ok()

    def spin(self):
        for _ in range(MAX_CALLBACKS_PER_SPIN):
            rclpy.spin_once(self.node, timeout_sec=0.0)

    def monotonic(self):
        return time.monotonic()

    def sleep(self, seconds):
        time.sleep(seconds)

    def publish_velocity(self, linear, angular):
        self.node.publish_velocity(linear, angular)

    def stop_robot(self):
        self.node.stop_robot()

    def log_info(self, text):
        self.node.get_logger().info(text)

    def log_error(self, text):
        self.node.get_logger().error(text)
