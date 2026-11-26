"""독립 Command Watchdog 실행.

프로젝트 루트에서 실행:
    python3 -m scripts.run_gz_watchdog --robot-id vehicle_blue --profile demo

Reset (Fault 해제, Executor 종료 후):
    ros2 service call /mapf/vehicle_blue/watchdog_reset std_srvs/srv/Trigger
"""

import argparse

import rclpy
from rclpy.signals import SignalHandlerOptions

from src.ros_bridge.gz_watchdog_node import GazeboWatchdogNode
from src.ros_bridge.safety_config import (
    DEMO_1HZ_CONFIG,
    DEMO_1HZ_WATCHDOG,
    DEMO_1HZ_STOP_CHECK,
    MAPF_50HZ_CONFIG,
    MAPF_50HZ_WATCHDOG,
    MAPF_50HZ_STOP_CHECK,
    make_robot_interface,
)
from src.ros_bridge.shutdown import EXIT_OK, run_guarded

PROFILES = {
    "demo": (DEMO_1HZ_WATCHDOG, DEMO_1HZ_CONFIG, DEMO_1HZ_STOP_CHECK),
    "mapf50": (MAPF_50HZ_WATCHDOG, MAPF_50HZ_CONFIG, MAPF_50HZ_STOP_CHECK),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-id", default="vehicle_blue")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="demo")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    watchdog_config, odom_config, stop_config = PROFILES[args.profile]
    interface = make_robot_interface(args.robot_id)

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = GazeboWatchdogNode(
        interface, watchdog_config, odom_config, stop_config
    )

    def body():
        rclpy.spin(node)
        return EXIT_OK

    def cleanup():
        if rclpy.ok():
            count = node.shutdown_stop()
            print(f"WATCHDOG_STOP_PUBLISHED: {count} (physical stop not verified)")
        node.destroy_node()
        rclpy.try_shutdown()

    return run_guarded(body, cleanup)


if __name__ == "__main__":
    raise SystemExit(main())
