import math
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from src.ros_bridge.cell_motion import (
    quaternion_to_yaw,
    compute_velocity,
)

class GazeboExecutorNode(Node):

    def __init__(self):
        super().__init__("mapf_gz_executor")

        self.pose = None
        self.last_odom_time = None

        self.cmd_pub = self.create_publisher(
            Twist,
            "/model/vehicle_blue/cmd_vel",
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            "/model/vehicle_blue/odometry",
            self.odom_callback,
            10
        )

        self.get_logger().info("Gazebo Executor initialized")

    def odom_callback(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        q = (
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        )

        norm = math.hypot(*q)

        if not math.isfinite(norm) or norm < 1e-9:
            self.pose = None
            return

        yaw = quaternion_to_yaw(
            *(value / norm for value in q)
        )

        if not all(math.isfinite(v) for v in (
            position.x, position.y, yaw
        )):
            self.pose = None
            return

        self.pose = (
            position.x,
            position.y,
            yaw
        )

        self.last_odom_time = time.monotonic()

    def publish_velocity(self, linear, angular):
        if not all(math.isfinite(v) for v in (linear, angular)):
            raise ValueError("Invalid velocity")

        msg = Twist()

        msg.linear.x = float(linear)
        msg.angular.z = float(angular)

        self.cmd_pub.publish(msg)

    def stop_robot(self):
        self.publish_velocity(0.0, 0.0)

    def move_to_target(self, target_x, target_y):
        start_time = time.monotonic()

        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.1)

                now = time.monotonic()

                if now - start_time > 60.0:
                    self.get_logger().error("MOVE_TIMEOUT")
                    return False

                if self.pose is None:
                    self.stop_robot()

                    if self.last_odom_time is not None:
                        self.get_logger().error("INVALID_ODOM")
                        return False

                    if now - start_time > 5.0:
                        self.get_logger().error("NO_VALID_ODOM")
                        return False

                    continue

                if now - self.last_odom_time > 2.5:
                    self.get_logger().error("ODOM_TIMEOUT")
                    return False

                x, y, yaw = self.pose

                linear, angular, reached = compute_velocity(
                    x, y, yaw,
                    target_x, target_y
                )

                if reached:
                    self.stop_robot()
                    self.get_logger().info("TARGET_REACHED")
                    return True

                self.publish_velocity(linear, angular)

                time.sleep(0.05)

            return False

        finally:
            if rclpy.ok():
                self.stop_robot()
