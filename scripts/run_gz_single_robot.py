"""단일 로봇 1셀 이동 (Executor). 독립 Watchdog 가 먼저 실행 중이어야 한다.

    python3 -m scripts.run_gz_watchdog --robot-id vehicle_blue      # 터미널 A
    python3 -m scripts.run_gz_single_robot                          # 터미널 B

종료 코드: 0 성공, 1 이동 실패, 2 정지 미확인, 3 예외, 130 SIGINT, 143 SIGTERM
"""

import math

import rclpy
from rclpy.signals import SignalHandlerOptions

from src.ros_bridge.grid_frame import grid_to_world
from src.ros_bridge.gz_executor_node import GazeboExecutorNode
from src.ros_bridge.shutdown import (
    EXIT_OK,
    EXIT_MOVE_FAILED,
    EXIT_STOP_UNCONFIRMED,
    failure_category,
    run_guarded,
)


def run(node):
    print("Waiting for /clock, odometry and watchdog...")

    status = node.wait_until_ready()

    if not status.ok:
        print(f"FAIL: Not ready ({status.reason})")
        return EXIT_MOVE_FAILED

    x0, y0, yaw0 = status.pose

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
        reason = node.last_move_result.reason
        print(f"FAIL: {failure_category(reason)} ({reason})")
        return EXIT_MOVE_FAILED

    print("STOP_COMMAND_PUBLISHED (via watchdog request)")
    print("Verifying physical stop from fresh odometry twist...")

    stop = node.verify_stop()

    if not stop.confirmed:
        print(f"FAIL: Stop not confirmed ({stop.state}: {stop.detail})")
        return EXIT_STOP_UNCONFIRMED

    print(f"STOP_CONFIRMED ({stop.detail})")

    status = node.safety_status()

    if not status.ok:
        print(f"FAIL: No fresh final odometry ({status.reason})")
        return EXIT_MOVE_FAILED

    x, y, yaw = status.pose

    error = math.hypot(
        target_x - x,
        target_y - y
    )

    print(f"Final: ({x:.3f}, {y:.3f})")
    print(f"Position error: {error:.4f} m (odometry, not ground truth)")

    if error <= 0.05:
        print("PASS: Target reached")
        return EXIT_OK

    print("FAIL: Position error too large")
    return EXIT_MOVE_FAILED


def main():
    rclpy.init(
        signal_handler_options=SignalHandlerOptions.NO
    )

    node = GazeboExecutorNode()

    def cleanup():
        if rclpy.ok():
            count = node.shutdown_stop()
            print(
                f"EXECUTOR_STOP_REQUESTS_PUBLISHED: {count} "
                "(physical stop not verified here)"
            )
        node.destroy_node()
        rclpy.try_shutdown()

    return run_guarded(lambda: run(node), cleanup)


if __name__ == "__main__":
    raise SystemExit(main())
