"""ROS Bridge 공용 QoS (rclpy 필요)."""

from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
)


def _latest_only(reliability):
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=reliability,
        durability=QoSDurabilityPolicy.VOLATILE,
    )


# Odometry: 최신 1개만 보관. Bridge 발행자(기본 RELIABLE)와 호환.
ODOM_QOS = _latest_only(QoSReliabilityPolicy.RELIABLE)

# /clock: 고주기. 최신 값만 필요하고 재전송은 의미가 없다.
CLOCK_QOS = _latest_only(QoSReliabilityPolicy.BEST_EFFORT)

# 실제 cmd_vel 과 이동 요청: 최신 명령만 의미가 있다. 지연된 옛 요청을 쌓지 않는다.
CMD_QOS = _latest_only(QoSReliabilityPolicy.RELIABLE)
REQUEST_QOS = _latest_only(QoSReliabilityPolicy.RELIABLE)

# Watchdog 상태
STATUS_QOS = _latest_only(QoSReliabilityPolicy.RELIABLE)
