"""ROS Graph 에서 cmd_vel / Odometry 발행·구독 Node 를 조회한다 (rclpy 필요).

판정은 fault_injection.check_topology (순수 Python) 가 한다.
"""

import time

import rclpy

from src.ros_bridge.fault_injection import check_topology


def _full_name(info):
    ns = info.node_namespace or "/"
    return f"{ns.rstrip('/')}/{info.node_name}"


def endpoint_names(node, topic, publishers):
    infos = (
        node.get_publishers_info_by_topic(topic)
        if publishers else node.get_subscriptions_info_by_topic(topic)
    )
    return [_full_name(info) for info in infos]


def collect_topology(node, interface):
    return {
        "cmd_vel_publishers": endpoint_names(node, interface.cmd_vel_topic, True),
        "cmd_vel_subscribers": endpoint_names(node, interface.cmd_vel_topic, False),
        "odom_publishers": endpoint_names(node, interface.odom_topic, True),
        "request_publishers": endpoint_names(node, interface.cmd_request_topic, True),
        "clock_publishers": endpoint_names(node, interface.clock_topic, True),
    }


def evaluate_topology(node, interface):
    graph = collect_topology(node, interface)
    result = check_topology(
        interface.robot_id,
        graph["cmd_vel_publishers"],
        graph["cmd_vel_subscribers"],
        graph["odom_publishers"],
    )
    return result, graph


def wait_for_topology(node, interface, timeout_mono, period=0.2):
    """검증을 통과할 때까지 Graph 를 다시 조회한다. (TopologyCheck, graph)."""
    deadline = time.monotonic() + timeout_mono
    result, graph = evaluate_topology(node, interface)
    while not result.ok and time.monotonic() < deadline and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=period)
        result, graph = evaluate_topology(node, interface)
    return result, graph
