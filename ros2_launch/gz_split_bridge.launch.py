"""Odometry 독립 장애 주입용 Gazebo 실행 환경 (ros_gz_sim_demos diff_drive.sdf).

기존 ``ros_gz_sim_demos/diff_drive.launch.py`` 는 하나의 parameter_bridge 가
cmd_vel 과 Odometry 를 양방향(@)으로 함께 연결한다. 이 Launch 는 Bridge 를 셋으로 나눈다.

- cmd_vel  : ROS → Gazebo 단방향 (``]``)  Node ``mapf_cmd_vel_bridge_<robot>``
- Odometry : Gazebo → ROS 단방향 (``[``)  Node ``mapf_odom_bridge_<robot>``
- Clock    : Gazebo → ROS 단방향 (``[``)  Node ``mapf_clock_bridge``

각 Bridge 는 별도 프로세스라서 Odometry Bridge 만 종료해도 나머지는 유지된다.
cmd_vel 은 단방향이므로 ROS 쪽 cmd_vel 발행자는 Watchdog 하나만 남는다
(양방향 Bridge 는 Gazebo→ROS 방향으로 같은 Topic 을 다시 발행해 발행자로 보인다).
RViz 와 vehicle_green Bridge 는 실행하지 않는다.

실행 (프로젝트 루트):
    ros2 launch ros2_launch/gz_split_bridge.launch.py
    ros2 launch ros2_launch/gz_split_bridge.launch.py gui:=false     # 서버만
    ros2 launch ros2_launch/gz_split_bridge.launch.py odom_bridge:=false
        # Odometry Bridge 는 별도 터미널에서 실행 (장애 주입 후 그 터미널에서만 재시작 가능)

Launch 대신 터미널별로 직접 실행할 때 (같은 Node 이름을 써야 장애 주입 도구가 찾는다):
    ros2 run ros_gz_bridge parameter_bridge \\
        '/model/vehicle_blue/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist' \\
        --ros-args -r __node:=mapf_cmd_vel_bridge_vehicle_blue
    ros2 run ros_gz_bridge parameter_bridge \\
        '/model/vehicle_blue/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry' \\
        --ros-args -r __node:=mapf_odom_bridge_vehicle_blue
    ros2 run ros_gz_bridge parameter_bridge \\
        '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock' \\
        --ros-args -r __node:=mapf_clock_bridge
"""

import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.ros_bridge.fault_injection import (  # noqa: E402
    CLOCK_BRIDGE_NODE,
    clock_bridge_spec,
    cmd_vel_bridge_node,
    cmd_vel_bridge_spec,
    odom_bridge_node,
    odom_bridge_spec,
)
from src.ros_bridge.safety_config import make_robot_interface  # noqa: E402


def _bridge(spec, name):
    return Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name=name,
        arguments=[spec],
        output="screen",
    )


def _setup(context):
    robot_id = LaunchConfiguration("robot_id").perform(context)
    world = LaunchConfiguration("world").perform(context)
    gui = LaunchConfiguration("gui").perform(context).lower() in ("1", "true", "yes")
    with_odom = LaunchConfiguration("odom_bridge").perform(context).lower() in (
        "1", "true", "yes",
    )
    interface = make_robot_interface(robot_id)

    gz_args = f"-r {world}" if gui else f"-r -s {world}"
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py"
        )),
        launch_arguments={"gz_args": gz_args}.items(),
    )

    actions = [
        gz_sim,
        _bridge(cmd_vel_bridge_spec(interface), cmd_vel_bridge_node(robot_id)),
        _bridge(clock_bridge_spec(interface), CLOCK_BRIDGE_NODE),
    ]
    if with_odom:
        actions.append(
            _bridge(odom_bridge_spec(interface), odom_bridge_node(robot_id))
        )
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("robot_id", default_value="vehicle_blue"),
        DeclareLaunchArgument("world", default_value="diff_drive.sdf"),
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("odom_bridge", default_value="true"),
        OpaqueFunction(function=_setup),
    ])
