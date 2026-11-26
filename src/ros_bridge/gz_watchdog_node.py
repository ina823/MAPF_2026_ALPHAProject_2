"""독립 Command Watchdog ROS2 Node.

실제 cmd_vel 을 발행하는 유일한 Node 다. 판단은 모두 CommandWatchdog(순수 Python)가 한다.

구독: 이동 요청(TwistStamped), Odometry, /clock
발행: 실제 cmd_vel(Twist), 상태(std_msgs/String JSON)
서비스: Reset (std_srvs/Trigger)
"""

import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String
from std_srvs.srv import Trigger

from src.ros_bridge.command_watchdog import (
    CommandWatchdog,
    FORWARDED,
    INVALID_VELOCITY,
    SPEED_LIMIT,
    UNUSED_TWIST_COMPONENT,
    STOP,
)
from src.ros_bridge.qos import (
    ODOM_QOS, CLOCK_QOS, CMD_QOS, REQUEST_QOS, STATUS_QOS,
)
from src.ros_bridge.safety_config import (
    DEMO_1HZ_CONFIG,
    DEMO_1HZ_WATCHDOG,
    DEMO_1HZ_STOP_CHECK,
    DEMO_VEHICLE_BLUE,
)
from src.ros_bridge.shutdown import repeat_stop
from src.ros_bridge.watchdog_protocol import status_to_json


def stamp_to_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


_REQUEST_FAULTS = (INVALID_VELOCITY, SPEED_LIMIT, UNUSED_TWIST_COMPONENT)


class GazeboWatchdogNode(Node):

    def __init__(
        self,
        interface=DEMO_VEHICLE_BLUE,
        watchdog_config=DEMO_1HZ_WATCHDOG,
        odom_config=DEMO_1HZ_CONFIG,
        stop_config=DEMO_1HZ_STOP_CHECK,
        node_name=None,
    ):
        super().__init__(node_name or f"mapf_watchdog_{interface.robot_id}")

        self.interface = interface
        self.config = watchdog_config
        self.core = CommandWatchdog(
            interface.robot_id,
            watchdog_config,
            odom_config,
            stop_config,
            interface.odom_frame_id,
            interface.base_frame_id,
        )
        self._last_state = None
        self._last_fault = None
        self._last_stop_check = None
        self.stop_published_count = 0

        self.cmd_pub = self.create_publisher(
            Twist, interface.cmd_vel_topic, CMD_QOS
        )
        self.status_pub = self.create_publisher(
            String, interface.watchdog_status_topic, STATUS_QOS
        )
        self.request_sub = self.create_subscription(
            TwistStamped, interface.cmd_request_topic,
            self.request_callback, REQUEST_QOS
        )
        self.odom_sub = self.create_subscription(
            Odometry, interface.odom_topic, self.odom_callback, ODOM_QOS
        )
        self.clock_sub = self.create_subscription(
            Clock, interface.clock_topic, self.clock_callback, CLOCK_QOS
        )
        self.reset_srv = self.create_service(
            Trigger, interface.watchdog_reset_service, self.reset_callback
        )
        self.timer = self.create_timer(
            watchdog_config.tick_period_mono, self.tick
        )

        # 초기 상태: 정지만 발행
        self._publish_cmd(STOP)
        self.get_logger().info(
            f"Watchdog initialized robot={interface.robot_id} "
            f"request={interface.cmd_request_topic} "
            f"cmd_vel={interface.cmd_vel_topic} "
            f"timeout={watchdog_config.command_timeout_mono}s "
            f"session={self.core.token}"
        )

    # ------------------------------------------------------------------
    # 콜백
    # ------------------------------------------------------------------

    def clock_callback(self, msg):
        if self.core.on_clock(stamp_to_seconds(msg.clock), time.monotonic()):
            self._publish_cmd(STOP)
            self._report()

    def odom_callback(self, msg):
        twist = msg.twist.twist
        pose = msg.pose.pose
        self.core.on_odom(
            stamp_to_seconds(msg.header.stamp),
            msg.header.frame_id,
            msg.child_frame_id,
            pose.position.x, pose.position.y,
            pose.orientation.x, pose.orientation.y,
            pose.orientation.z, pose.orientation.w,
            time.monotonic(),
            linear_x=twist.linear.x,
            linear_y=twist.linear.y,
            angular_z=twist.angular.z,
        )
        if self.core.monitor.faulted:
            self._publish_cmd(STOP)
            self._report()

    def request_callback(self, msg):
        twist = msg.twist
        result = self.core.on_request(
            msg.header.frame_id,
            stamp_to_seconds(msg.header.stamp),
            twist.linear.x, twist.linear.y, twist.linear.z,
            twist.angular.x, twist.angular.y, twist.angular.z,
            time.monotonic(),
        )
        if result.result == FORWARDED:
            self._publish_cmd(result.output)
        elif result.result in _REQUEST_FAULTS:
            self._publish_cmd(STOP)
            self._report()

    def tick(self):
        output = self.core.tick(time.monotonic())
        self._publish_cmd(output)
        self._report()

    def reset_callback(self, request, response):
        ok, message = self.core.reset(time.monotonic())
        response.success = ok
        response.message = message
        if ok:
            self.get_logger().warning(f"WATCHDOG_RESET: {message}")
        else:
            self.get_logger().error(f"WATCHDOG_RESET_REFUSED: {message}")
        self._publish_cmd(STOP)
        self._report()
        return response

    # ------------------------------------------------------------------
    # 출력
    # ------------------------------------------------------------------

    def _publish_cmd(self, command):
        linear, angular = command
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.cmd_pub.publish(msg)
        if command == STOP:
            self.stop_published_count += 1

    def _report(self):
        status = self.core.status(time.monotonic())
        text = String()
        text.data = status_to_json(status)
        self.status_pub.publish(text)

        if status["state"] != self._last_state:
            self.get_logger().info(
                f"WATCHDOG_STATE: {self._last_state} -> {status['state']} "
                f"(health={status['health']})"
            )
            self._last_state = status["state"]
        if status["fault"] != self._last_fault:
            if status["fault"] is not None:
                self.get_logger().error(
                    f"WATCHDOG_FAULT: {status['fault']} "
                    f"({status['fault_detail']}) -> STOP published"
                )
            self._last_fault = status["fault"]
        if status["stop_check"] != self._last_stop_check:
            self.get_logger().info(
                f"STOP_CHECK: {status['stop_check']} {status['stop_detail']}"
            )
            self._last_stop_check = status["stop_check"]

    def shutdown_stop(self):
        """정상 종료/SIGINT/SIGTERM: 정지 명령 반복 발행 (실제 정지 확인 아님)."""
        self.core.shutdown()
        count = repeat_stop(
            lambda: self._publish_cmd(STOP),
            self.config.shutdown_stop_repeats,
            self.config.shutdown_stop_interval_mono,
        )
        try:
            self._report()
        except Exception:
            pass
        return count
