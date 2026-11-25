import math
import time

import rclpy
from rclpy.signals import SignalHandlerOptions

from src.ros_bridge.grid_frame import grid_to_world
from src.ros_bridge.gz_executor_node import GazeboExecutorNode


def main():
    rclpy.init(
        signal_handler_options=SignalHandlerOptions.NO
    )

    node = GazeboExecutorNode()

    try:
        print("Waiting for odometry...")

        start = time.monotonic()

        while node.pose is None:
            rclpy.spin_once(node, timeout_sec=0.1)

            if time.monotonic() - start > 5.0:
                print("FAIL: No valid odometry")
                return 1

        x0, y0, yaw0 = node.pose

        print(f"Start: ({x0:.3f}, {y0:.3f})")
        print(f"Start yaw: {math.degrees(yaw0):.2f} deg")

        target_x, target_y = grid_to_world(
            0, 1,
            cell_size=1.0,
            x0=x0,
            y0=y0
        )

        print(f"Target: ({target_x:.3f}, {target_y:.3f})")

        success = node.move_to_target(
            target_x, target_y
        )

        if not success:
            print("FAIL: Movement failed")
            return 1

        print("Checking final position...")

        check_start = time.monotonic()

        while time.monotonic() - check_start < 3.0:
            node.stop_robot()
            rclpy.spin_once(node, timeout_sec=0.1)
            time.sleep(0.05)

        if (
            node.pose is None
            or node.last_odom_time <= check_start
            or time.monotonic() - node.last_odom_time > 2.5
        ):
            print("FAIL: No fresh final odometry")
            return 1

        x, y, yaw = node.pose

        error = math.hypot(
            target_x - x,
            target_y - y
        )

        print(f"Final: ({x:.3f}, {y:.3f})")
        print(f"Position error: {error:.4f} m")

        if error <= 0.05:
            print("PASS: Target reached")
            return 0

        print("FAIL: Position error too large")
        return 1

    except KeyboardInterrupt:
        print("Interrupted")
        return 130

    finally:
        if rclpy.ok():
            for _ in range(10):
                node.stop_robot()
                time.sleep(0.05)

        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
