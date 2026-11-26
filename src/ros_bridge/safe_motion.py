"""안전성 검사를 거치는 단일 목표 이동 루프 (rclpy 의존 없음).

ROS 입출력은 ``io`` 객체로 주입한다. 필요한 메서드:

- ``ok() -> bool``           : 실행 환경이 살아 있는지 (rclpy.ok)
- ``spin() -> None``         : 대기 중인 콜백 처리
- ``monotonic() -> float``   : time.monotonic
- ``sleep(seconds) -> None``
- ``publish_velocity(linear, angular) -> None``
- ``stop_robot() -> None``   : 정지 명령 발행 (실제 정지 여부는 확인하지 않는다)
- ``log_info(text)``, ``log_error(text)``

정지 "명령 발행"만 보장한다. 로봇이 물리적으로 멈췄는지는 이 모듈이 판단하지 않는다.

``gate(now_mono) -> (ok, reason)`` 를 주면 Odometry 검사에 더해 매 주기 확인한다
(예: 독립 Watchdog 상태). gate 실패도 Odometry 이상과 같게 처리한다.
"""

from dataclasses import dataclass

from src.ros_bridge.cell_motion import compute_velocity
from src.ros_bridge.odom_monitor import SafetyStatus


MOVE_TIMEOUT = "MOVE_TIMEOUT"
SHUTDOWN = "SHUTDOWN"
TARGET_REACHED = "TARGET_REACHED"
NOT_READY = "NOT_READY"
READY = "READY"


@dataclass(frozen=True)
class MoveResult:
    success: bool
    reason: str
    # 이동 시작 전에 차단됐는지 (True 면 이동 명령을 한 번도 내지 않음)
    blocked_before_start: bool = False


def _combined(monitor, gate, now):
    status = monitor.check(now)
    if not status.ok or gate is None:
        return status
    link = gate(now)
    if not link.ok:
        return SafetyStatus(False, link.reason)
    return status


def wait_until_ready(monitor, io, timeout_mono, gate=None):
    """clock + 신선한 odom (+ gate) 이 갖춰질 때까지 대기. 이동 명령은 내지 않는다."""
    start = io.monotonic()
    status = _combined(monitor, gate, start)

    while io.ok():
        io.spin()
        now = io.monotonic()
        status = _combined(monitor, gate, now)

        if status.ok:
            return status

        if monitor.faulted or now - start > timeout_mono:
            return status

        io.sleep(monitor.config.control_period_mono)

    return status


def run_move(
    target_x, target_y, monitor, io,
    controller=compute_velocity, gate=None,
):
    cfg = monitor.config
    start = io.monotonic()

    status = _combined(monitor, gate, start)

    if not status.ok:
        # 시작 전 차단: 잠그지 않는다 (아직 움직이지 않았으므로).
        # 단, 기존 fault 는 이미 잠겨 있으므로 계속 차단된다.
        io.stop_robot()
        io.log_error(f"MOVE_BLOCKED: {status.reason}")
        return MoveResult(False, status.reason, blocked_before_start=True)

    try:
        while io.ok():
            io.spin()

            now = io.monotonic()

            if now - start > cfg.move_timeout_mono:
                reason = monitor.latch(MOVE_TIMEOUT)
                io.stop_robot()
                io.log_error(f"MOVE_ABORTED: {reason}")
                return MoveResult(False, reason)

            status = _combined(monitor, gate, now)

            if not status.ok:
                # 이동 중 이상: 정지 명령 + 잠금. 자동 재개 없음.
                reason = monitor.latch(status.reason)
                io.stop_robot()
                io.log_error(f"MOVE_ABORTED: {reason}")
                return MoveResult(False, reason)

            x, y, yaw = status.pose

            linear, angular, reached = controller(
                x, y, yaw, target_x, target_y
            )

            if reached:
                io.stop_robot()
                io.log_info(TARGET_REACHED)
                return MoveResult(True, TARGET_REACHED)

            io.publish_velocity(linear, angular)

            io.sleep(cfg.control_period_mono)

        return MoveResult(False, SHUTDOWN)

    finally:
        if io.ok():
            io.stop_robot()
