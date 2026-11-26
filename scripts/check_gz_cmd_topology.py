"""실제 cmd_vel 발행자가 Watchdog 하나뿐인지 확인한다 (분리 Bridge 실험 환경용).

프로젝트 루트에서 실행 (Gazebo, 분리 Bridge, Watchdog 실행 후):
    python3 -m scripts.check_gz_cmd_topology --robot-id vehicle_blue

검사
- /model/<robot>/cmd_vel ROS 발행자 = [mapf_watchdog_<robot>] 정확히 하나
  (기존 양방향 Bridge 가 남아 있으면 Bridge 도 발행자로 보여 실패한다)
- cmd_vel 구독자에 mapf_cmd_vel_bridge_<robot> 존재
- /model/<robot>/odometry ROS 발행자 = [mapf_odom_bridge_<robot>] 정확히 하나

Gazebo 쪽 발행자는 ROS Graph 에 보이지 않는다. 별도로 확인:
    gz topic -i -t /model/vehicle_blue/cmd_vel

종료 코드: 0 통과, 1 실패
"""

import argparse
import json

import rclpy

from src.ros_bridge.gz_topology import wait_for_topology
from src.ros_bridge.safety_config import make_robot_interface


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-id", default="vehicle_blue")
    parser.add_argument(
        "--timeout", type=float, default=5.0,
        help="Graph 발견 대기 상한 (monotonic s)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    interface = make_robot_interface(args.robot_id)
    rclpy.init()
    node = rclpy.create_node(f"mapf_topology_check_{args.robot_id}")
    try:
        result, graph = wait_for_topology(node, interface, args.timeout)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

    print(json.dumps(graph, indent=2, sort_keys=True))
    if result.ok:
        print("TOPOLOGY_OK: watchdog is the only cmd_vel publisher; "
              "odometry has a single dedicated bridge")
        return 0
    for problem in result.problems:
        print(f"TOPOLOGY_FAIL: {problem}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
