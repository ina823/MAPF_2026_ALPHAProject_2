"""Odometry / Clock 안전성 상태 관리 (rclpy 의존 없음).

시간 기준
---------
- ``*_mono`` : 수신 시각, ``time.monotonic()``. 메시지가 "언제 도착했는지".
- ``*_sim``  : /clock 및 ``header.stamp``. 메시지가 "언제 생성됐는지".

두 기준은 같은 식 안에서 서로 빼거나 비교하지 않는다.
수신 공백은 mono 끼리, 생성 시각 나이는 sim 끼리만 비교한다.

Fault 처리
----------
비정상 피드백(Frame, stamp, NaN/Inf, Quaternion, 위치 급변, 시뮬레이션 시간 역행)은
즉시 ``fault`` 로 잠긴다(latched). 잠긴 fault 는 ``reset()`` 을 명시적으로 호출하기
전에는 풀리지 않으며, 이 모듈과 Executor 는 ``reset()`` 을 스스로 호출하지 않는다.

Timeout 류(Clock/Odometry 공백, 오래된 stamp, 시뮬레이션 정지)는 ``check()`` 결과로만
보고된다. 이동 중에 발생하면 이동 루프가 ``latch()`` 로 잠근다.
"""

import math
from dataclasses import dataclass

from src.ros_bridge.cell_motion import quaternion_to_yaw
from src.ros_bridge.odom_safety import (
    is_odom_fresh,
    is_stamp_fresh,
    is_stamp_newer,
    is_frame_valid,
    is_position_jump,
    is_quaternion_valid,
)


# check() 결과 사유
OK = "OK"
NO_CLOCK = "NO_CLOCK"
CLOCK_TIMEOUT = "CLOCK_TIMEOUT"
SIM_TIME_STALLED = "SIM_TIME_STALLED"
NO_ODOM = "NO_ODOM"
ODOM_RECEIVE_TIMEOUT = "ODOM_RECEIVE_TIMEOUT"
ODOM_STAMP_STALE = "ODOM_STAMP_STALE"
ODOM_STAMP_AHEAD_OF_CLOCK = "ODOM_STAMP_AHEAD_OF_CLOCK"

# 잠기는(latched) fault
INVALID_CLOCK = "INVALID_CLOCK"
SIM_TIME_REGRESSION = "SIM_TIME_REGRESSION"
INVALID_FRAME = "INVALID_FRAME"
INVALID_STAMP = "INVALID_STAMP"
TIME_BASE_MISMATCH = "TIME_BASE_MISMATCH"
INVALID_POSITION = "INVALID_POSITION"
INVALID_QUATERNION = "INVALID_QUATERNION"
INVALID_TWIST = "INVALID_TWIST"
POSITION_JUMP = "POSITION_JUMP"

# on_odom() 결과
ACCEPTED = "ACCEPTED"
IGNORED_DUPLICATE = "IGNORED_DUPLICATE"
IGNORED_OUT_OF_ORDER = "IGNORED_OUT_OF_ORDER"
IGNORED_FAULTED = "IGNORED_FAULTED"


@dataclass(frozen=True)
class SafetyStatus:
    ok: bool
    reason: str
    pose: tuple = None  # (x, y, yaw), ok 일 때만


@dataclass(frozen=True)
class OdomSample:
    stamp_sim: float
    received_mono: float
    x: float
    y: float
    yaw: float
    # twist (child frame). 제공되지 않으면 None — 실제 정지 판정에 쓸 수 없다.
    linear_x: float = None
    linear_y: float = None
    angular_z: float = None


def _finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


class OdomSafetyMonitor:

    def __init__(self, config, odom_frame_id, base_frame_id):
        # 설정 오류는 생성 시점에 드러낸다
        is_frame_valid(odom_frame_id, base_frame_id,
                       odom_frame_id, base_frame_id)

        self.config = config
        self.odom_frame_id = odom_frame_id
        self.base_frame_id = base_frame_id
        self.reset()

    # ------------------------------------------------------------------
    # 상태
    # ------------------------------------------------------------------

    def reset(self):
        """모든 상태와 fault 를 지운다. 운영자 판단으로만 호출해야 한다."""
        self.fault = None
        self.fault_detail = None

        self.last_clock_sim = None
        self.last_clock_received_mono = None
        self.last_sim_advance_mono = None

        self.last_odom = None

        self.accepted_count = 0
        self.duplicate_count = 0
        self.out_of_order_count = 0

    @property
    def faulted(self):
        return self.fault is not None

    @property
    def pose(self):
        """마지막으로 수락한 pose. fault 상태면 None. 신선도는 check() 로 확인."""
        if self.faulted or self.last_odom is None:
            return None
        s = self.last_odom
        return (s.x, s.y, s.yaw)

    @property
    def last_odom_received_mono(self):
        if self.last_odom is None:
            return None
        return self.last_odom.received_mono

    def latch(self, reason, detail=None):
        """fault 를 잠근다. 이미 잠겨 있으면 최초 사유를 유지한다."""
        if self.fault is None:
            self.fault = reason
            self.fault_detail = detail
        return self.fault

    # ------------------------------------------------------------------
    # 입력
    # ------------------------------------------------------------------

    def on_clock(self, clock_sim, received_mono):
        """/clock 수신. 새로 잠긴 fault 가 있으면 그 사유를, 아니면 None."""
        if self.faulted:
            return None

        if not _finite(clock_sim) or clock_sim < 0:
            return self.latch(INVALID_CLOCK, f"clock={clock_sim!r}")

        previous = self.last_clock_sim

        if previous is not None and clock_sim < previous:
            return self.latch(
                SIM_TIME_REGRESSION,
                f"clock {previous:.6f} -> {clock_sim:.6f}"
            )

        if previous is None or clock_sim > previous:
            self.last_sim_advance_mono = received_mono

        self.last_clock_sim = clock_sim
        self.last_clock_received_mono = received_mono
        return None

    def on_odom(
        self,
        stamp_sim,
        frame_id,
        child_frame_id,
        x, y,
        qx, qy, qz, qw,
        received_mono,
        linear_x=None,
        linear_y=None,
        angular_z=None,
    ):
        """Odometry 수신. ACCEPTED / IGNORED_* / 새로 잠긴 fault 사유 중 하나.

        twist(linear_x, linear_y, angular_z)는 선택이다. 주면 모두 유한해야 한다.
        """
        cfg = self.config

        if self.faulted:
            return IGNORED_FAULTED

        if not is_frame_valid(
            frame_id, child_frame_id,
            self.odom_frame_id, self.base_frame_id
        ):
            return self.latch(
                INVALID_FRAME,
                f"frame_id={frame_id!r} child_frame_id={child_frame_id!r}"
            )

        previous = self.last_odom
        previous_stamp = None if previous is None else previous.stamp_sim

        if not is_stamp_newer(stamp_sim, previous_stamp):
            if not _finite(stamp_sim) or stamp_sim < 0:
                return self.latch(INVALID_STAMP, f"stamp={stamp_sim!r}")
            # 중복·역순: 수신 시각/pose 를 갱신하지 않는다
            if stamp_sim == previous_stamp:
                self.duplicate_count += 1
                return IGNORED_DUPLICATE
            self.out_of_order_count += 1
            return IGNORED_OUT_OF_ORDER

        # 시간 기준 불일치 (예: stamp 가 벽시계 epoch).
        # /clock 이 조금 늦게 도착하는 정상 지연은 check() 에서 비잠금으로 다룬다.
        # stamp_max_age 이상 앞서면 지연으로 설명할 수 없으므로 잠근다.
        if self.last_clock_sim is not None and (
            stamp_sim - self.last_clock_sim
            > cfg.odom_stamp_max_age_sim
        ):
            return self.latch(
                TIME_BASE_MISMATCH,
                f"stamp={stamp_sim:.6f} clock={self.last_clock_sim:.6f}"
            )

        if not (_finite(x) and _finite(y)):
            return self.latch(INVALID_POSITION, f"x={x!r} y={y!r}")

        if not is_quaternion_valid(
            qx, qy, qz, qw, cfg.quaternion_norm_tolerance
        ):
            return self.latch(
                INVALID_QUATERNION, f"q=({qx!r}, {qy!r}, {qz!r}, {qw!r})"
            )

        if previous is not None:
            dt = stamp_sim - previous.stamp_sim
            if is_position_jump(
                previous.x, previous.y, x, y, dt,
                cfg.jump_max_speed, cfg.jump_margin
            ):
                distance = math.hypot(x - previous.x, y - previous.y)
                return self.latch(
                    POSITION_JUMP,
                    f"{distance:.3f} m in {dt:.3f} s"
                )

        twist = (linear_x, linear_y, angular_z)
        if any(v is not None for v in twist) and not all(
            _finite(v) for v in twist
        ):
            return self.latch(INVALID_TWIST, f"twist={twist!r}")

        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        yaw = quaternion_to_yaw(qx / norm, qy / norm, qz / norm, qw / norm)

        if not _finite(yaw):
            return self.latch(INVALID_QUATERNION, f"yaw={yaw!r}")

        self.last_odom = OdomSample(
            stamp_sim=float(stamp_sim),
            received_mono=received_mono,
            x=float(x),
            y=float(y),
            yaw=yaw,
            linear_x=None if linear_x is None else float(linear_x),
            linear_y=None if linear_y is None else float(linear_y),
            angular_z=None if angular_z is None else float(angular_z),
        )
        self.accepted_count += 1
        return ACCEPTED

    # ------------------------------------------------------------------
    # 판정
    # ------------------------------------------------------------------

    def check(self, now_mono):
        """지금 이 pose 로 이동 명령을 내도 되는지 판정한다."""
        cfg = self.config

        if self.faulted:
            return SafetyStatus(False, self.fault)

        # --- Clock (mono 끼리 비교) ---
        if self.last_clock_received_mono is None:
            return SafetyStatus(False, NO_CLOCK)

        if not is_odom_fresh(
            now_mono, self.last_clock_received_mono,
            cfg.clock_receive_timeout_mono
        ):
            return SafetyStatus(False, CLOCK_TIMEOUT)

        if not is_odom_fresh(
            now_mono, self.last_sim_advance_mono,
            cfg.sim_stall_timeout_mono
        ):
            return SafetyStatus(False, SIM_TIME_STALLED)

        # --- Odometry 수신 시각 (mono 끼리 비교) ---
        sample = self.last_odom

        if sample is None:
            return SafetyStatus(False, NO_ODOM)

        if not is_odom_fresh(
            now_mono, sample.received_mono,
            cfg.odom_receive_timeout_mono
        ):
            return SafetyStatus(False, ODOM_RECEIVE_TIMEOUT)

        # --- Odometry 생성 시각 (sim 끼리 비교) ---
        if (
            sample.stamp_sim - self.last_clock_sim
            > cfg.odom_stamp_future_tolerance_sim
        ):
            return SafetyStatus(False, ODOM_STAMP_AHEAD_OF_CLOCK)

        if not is_stamp_fresh(
            self.last_clock_sim, sample.stamp_sim,
            cfg.odom_stamp_max_age_sim,
            cfg.odom_stamp_future_tolerance_sim
        ):
            return SafetyStatus(False, ODOM_STAMP_STALE)

        return SafetyStatus(True, OK, (sample.x, sample.y, sample.yaw))
