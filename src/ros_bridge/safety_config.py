"""Odometry 안전성 설정 (rclpy 의존 없음).

시간 단위는 모두 초다. 각 필드의 시간 기준을 이름 끝에 표시한다.

- ``*_mono`` : 수신 측 ``time.monotonic()`` (벽시계 기반, 단조 증가)
- ``*_sim``  : /clock 및 Odometry ``header.stamp`` 의 시뮬레이션 시간

두 기준은 서로 직접 비교하지 않는다.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class OdomSafetyConfig:
    name: str

    # Odometry 발행 주기 (검증용 정보; 신선도 허용 시간 하한 계산에 쓴다)
    odom_rate_hz: float

    # 마지막으로 "수락한" Odometry 를 받은 뒤 허용하는 최대 공백 (monotonic)
    odom_receive_timeout_mono: float

    # 최신 /clock 시간 대비 header.stamp 최대 나이 (sim)
    odom_stamp_max_age_sim: float

    # header.stamp 가 최신 /clock 보다 앞서도 되는 최대 폭 (sim).
    # /clock 과 Odometry 는 별도 Topic 이라 도착 순서가 바뀔 수 있다.
    odom_stamp_future_tolerance_sim: float

    # 마지막 /clock 수신 후 허용 공백 (monotonic)
    clock_receive_timeout_mono: float

    # 시뮬레이션 시간이 증가하지 않아도 되는 최대 시간 (monotonic). 일시정지 감지.
    sim_stall_timeout_mono: float

    # 위치 급변 판정: 연속 두 메시지의 stamp 차이 dt 동안
    # jump_max_speed * dt + jump_margin 을 넘게 움직이면 급변
    jump_max_speed: float
    jump_margin: float

    # |‖q‖ - 1| 허용 오차
    quaternion_norm_tolerance: float

    # 이동 시작 전 준비 상태(clock + 신선한 odom) 대기 한도 (monotonic)
    startup_timeout_mono: float

    # 한 번의 이동 전체 한도 (monotonic)
    move_timeout_mono: float

    # 제어 루프 주기 (monotonic)
    control_period_mono: float

    def __post_init__(self):
        positive = (
            "odom_rate_hz",
            "odom_receive_timeout_mono",
            "odom_stamp_max_age_sim",
            "clock_receive_timeout_mono",
            "sim_stall_timeout_mono",
            "jump_max_speed",
            "startup_timeout_mono",
            "move_timeout_mono",
            "control_period_mono",
        )
        non_negative = (
            "odom_stamp_future_tolerance_sim",
            "jump_margin",
            "quaternion_norm_tolerance",
        )

        for field_name in positive + non_negative:
            value = getattr(self, field_name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{field_name} must be finite")
            if field_name in positive and value <= 0:
                raise ValueError(f"{field_name} must be positive")
            if field_name in non_negative and value < 0:
                raise ValueError(f"{field_name} must be non-negative")

        if self.quaternion_norm_tolerance >= 1:
            raise ValueError("quaternion_norm_tolerance must be < 1")

        period = 1.0 / self.odom_rate_hz

        # 허용 시간이 발행 주기 이하이면 정상 상황에서도 매 주기 Timeout 이 난다.
        # (예: 0.25 s 를 1 Hz 데모에 적용하는 실수를 여기서 막는다)
        if self.odom_stamp_max_age_sim <= period:
            raise ValueError(
                "odom_stamp_max_age_sim must exceed one odom period "
                f"({period:.3f} s)"
            )
        if self.odom_receive_timeout_mono <= period:
            raise ValueError(
                "odom_receive_timeout_mono must exceed one odom period "
                f"({period:.3f} s)"
            )
        if self.move_timeout_mono <= self.control_period_mono:
            raise ValueError("move_timeout_mono must exceed control period")

    @property
    def odom_period(self):
        return 1.0 / self.odom_rate_hz


def _require_finite(owner, field_name, value, positive):
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{owner}.{field_name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{owner}.{field_name} must be positive")
    if not positive and value < 0:
        raise ValueError(f"{owner}.{field_name} must be non-negative")


@dataclass(frozen=True)
class RobotInterface:
    """로봇 1대의 Topic/Frame 구성.

    - ``cmd_request_topic``: Executor → Watchdog 이동 요청 (TwistStamped)
    - ``cmd_vel_topic``: Watchdog → Gazebo 실제 제어 (Twist). Watchdog 만 발행한다.
    - ``watchdog_status_topic``: Watchdog 상태 (std_msgs/String, JSON)
    - ``watchdog_reset_service``: Fault Reset (std_srvs/Trigger)
    """

    robot_id: str
    cmd_request_topic: str
    cmd_vel_topic: str
    odom_topic: str
    clock_topic: str
    watchdog_status_topic: str
    watchdog_reset_service: str
    odom_frame_id: str
    base_frame_id: str

    def __post_init__(self):
        for field_name in (
            "robot_id", "cmd_request_topic", "cmd_vel_topic",
            "odom_topic", "clock_topic", "watchdog_status_topic",
            "watchdog_reset_service", "odom_frame_id", "base_frame_id",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.odom_frame_id == self.base_frame_id:
            raise ValueError("odom_frame_id and base_frame_id must differ")
        topics = (
            self.cmd_request_topic, self.cmd_vel_topic, self.odom_topic,
            self.clock_topic, self.watchdog_status_topic,
        )
        if len(set(topics)) != len(topics):
            raise ValueError("topics must be distinct")


def make_robot_interface(
    robot_id,
    request_prefix="/mapf",
    model_prefix="/model",
    clock_topic="/clock",
    base_link="chassis",
):
    """Robot ID 로부터 Topic/Frame 을 만든다 (다중 로봇 확장용)."""
    if not isinstance(robot_id, str) or not robot_id or "/" in robot_id:
        raise ValueError("robot_id must be a non-empty name without '/'")
    return RobotInterface(
        robot_id=robot_id,
        cmd_request_topic=f"{request_prefix}/{robot_id}/cmd_vel_request",
        cmd_vel_topic=f"{model_prefix}/{robot_id}/cmd_vel",
        odom_topic=f"{model_prefix}/{robot_id}/odometry",
        clock_topic=clock_topic,
        watchdog_status_topic=f"{request_prefix}/{robot_id}/watchdog_status",
        watchdog_reset_service=f"{request_prefix}/{robot_id}/watchdog_reset",
        odom_frame_id=f"{robot_id}/odom",
        base_frame_id=f"{robot_id}/{base_link}",
    )


@dataclass(frozen=True)
class StopCheckConfig:
    """정지 명령 이후 Odometry twist 로 실제 정지를 확인하는 기준 (sim 시간)."""

    name: str
    linear_tolerance: float      # |v| 허용 (m/s)
    angular_tolerance: float     # |w| 허용 (rad/s)
    settle_sim: float            # 정지 명령 후 감속 대기. 이 구간 샘플은 판정에 쓰지 않는다
    window_sim: float            # 연속 정지 상태가 유지되어야 하는 구간
    min_samples: int             # window 안 최소 샘플 수 (새 메시지 수신 확인)
    sample_max_age_sim: float    # 최신 샘플이 /clock 대비 이보다 오래되면 판정 보류
    timeout_sim: float           # 정지 명령 후 이 시간 안에 확인 못하면 실패
    # 판정 시작 후 monotonic 대기 상한. /clock 이 멈추면 timeout_sim 에 도달하지 못하므로
    # 이 시간이 지나도 결론이 없으면 UNVERIFIABLE 로 확정한다.
    # 정상(실시간 배율 1 근처)에서는 sim 판정이 먼저 나도록 timeout_sim 이상으로 둔다.
    timeout_mono: float

    def __post_init__(self):
        for field_name in (
            "linear_tolerance", "angular_tolerance", "window_sim",
            "sample_max_age_sim", "timeout_sim", "timeout_mono",
        ):
            _require_finite("StopCheckConfig", field_name,
                            getattr(self, field_name), True)
        _require_finite("StopCheckConfig", "settle_sim",
                        self.settle_sim, False)
        if (
            not isinstance(self.min_samples, int)
            or isinstance(self.min_samples, bool)
            or self.min_samples < 2
        ):
            raise ValueError("min_samples must be an int >= 2")
        if self.timeout_sim <= self.settle_sim + self.window_sim:
            raise ValueError("timeout_sim must exceed settle_sim + window_sim")
        if self.timeout_mono < self.timeout_sim:
            raise ValueError("timeout_mono must be >= timeout_sim")


@dataclass(frozen=True)
class WatchdogConfig:
    name: str

    # 마지막 유효 이동 요청 수신 후 허용 공백 (monotonic)
    command_timeout_mono: float

    # Watchdog 판정/출력 주기 (monotonic)
    tick_period_mono: float

    # Executor 가 Watchdog 상태 메시지를 신뢰하는 최대 나이 (monotonic)
    status_timeout_mono: float

    # 요청 header.stamp 와 Watchdog 의 /clock 차이 허용 (sim). 큐 지연 요청 차단.
    request_max_age_sim: float

    # 허용 속도 (Executor 제어 상한과 같게 둔다. 초과 요청은 차단)
    max_linear_speed: float
    max_angular_speed: float

    # 사용하지 않는 Twist 성분(linear.y/z, angular.x/y) 허용 절대값
    unused_component_tolerance: float

    # 종료 시 정지 명령 반복
    shutdown_stop_repeats: int
    shutdown_stop_interval_mono: float

    def __post_init__(self):
        for field_name in (
            "command_timeout_mono", "tick_period_mono", "status_timeout_mono",
            "request_max_age_sim", "max_linear_speed", "max_angular_speed",
            "shutdown_stop_interval_mono",
        ):
            _require_finite("WatchdogConfig", field_name,
                            getattr(self, field_name), True)
        _require_finite("WatchdogConfig", "unused_component_tolerance",
                        self.unused_component_tolerance, False)
        if (
            not isinstance(self.shutdown_stop_repeats, int)
            or isinstance(self.shutdown_stop_repeats, bool)
            or self.shutdown_stop_repeats < 1
        ):
            raise ValueError("shutdown_stop_repeats must be an int >= 1")
        if self.command_timeout_mono <= 2 * self.tick_period_mono:
            raise ValueError("command_timeout_mono must exceed 2 ticks")
        if self.status_timeout_mono <= 2 * self.tick_period_mono:
            raise ValueError("status_timeout_mono must exceed 2 ticks")

    def validate_against(self, odom_config):
        """Executor 제어 주기와 맞지 않는 Timeout 을 거부한다."""
        if self.command_timeout_mono <= 4 * odom_config.control_period_mono:
            raise ValueError(
                "command_timeout_mono must exceed 4 executor control periods"
            )


# ros_gz_sim_demos diff_drive.sdf 의 vehicle_blue: odom_publish_frequency 1 Hz,
# DiffDrive 속도 상한 0.5 m/s (제어 상한 0.3 m/s).
DEMO_1HZ_CONFIG = OdomSafetyConfig(
    name="demo_vehicle_blue_1hz",
    odom_rate_hz=1.0,
    odom_receive_timeout_mono=2.0,
    odom_stamp_max_age_sim=1.5,
    odom_stamp_future_tolerance_sim=0.2,
    clock_receive_timeout_mono=1.0,
    sim_stall_timeout_mono=1.0,
    jump_max_speed=0.6,
    jump_margin=0.05,
    quaternion_norm_tolerance=0.05,
    startup_timeout_mono=5.0,
    move_timeout_mono=60.0,
    control_period_mono=0.05,
)

# 자체 MAPF 로봇 목표 사양 (50 Hz). 로봇 속도 상한이 확정되면 jump_max_speed 를 갱신한다.
MAPF_50HZ_CONFIG = OdomSafetyConfig(
    name="mapf_robot_50hz",
    odom_rate_hz=50.0,
    odom_receive_timeout_mono=0.25,
    odom_stamp_max_age_sim=0.1,
    odom_stamp_future_tolerance_sim=0.05,
    clock_receive_timeout_mono=0.5,
    sim_stall_timeout_mono=0.5,
    jump_max_speed=0.6,
    jump_margin=0.02,
    quaternion_norm_tolerance=0.05,
    startup_timeout_mono=5.0,
    move_timeout_mono=60.0,
    control_period_mono=0.02,
)

# Frame ID 는 gz-sim DiffDrive 기본값 (<model>/odom, <model>/<canonical link>).
# 실제 값은 Gazebo 실행 시 `ros2 topic echo --once` 로 확인해야 한다 (미검증).
DEMO_VEHICLE_BLUE = make_robot_interface("vehicle_blue")

# Executor 실제 명령 주기: control_period 0.05 s (+ spin/처리 시간) → 약 20 Hz.
# Timeout 0.5 s = 약 10 주기. Executor 소멸 시 최대 이동 ≈ 0.3 m/s × (0.5 + tick 0.05) s
# ≈ 0.17 m (+ DiffDrive 감속 1 m/s² 구간 약 0.045 m).
DEMO_1HZ_WATCHDOG = WatchdogConfig(
    name="demo_vehicle_blue_watchdog",
    command_timeout_mono=0.5,
    tick_period_mono=0.05,
    status_timeout_mono=0.5,
    request_max_age_sim=0.5,
    max_linear_speed=0.3,
    max_angular_speed=0.5,
    unused_component_tolerance=1e-9,
    shutdown_stop_repeats=10,
    shutdown_stop_interval_mono=0.05,
)

# 50 Hz 로봇: control_period 0.02 s → Timeout 0.2 s (10 주기). 속도 상한은 사양 확정 전 임시값.
MAPF_50HZ_WATCHDOG = WatchdogConfig(
    name="mapf_robot_50hz_watchdog",
    command_timeout_mono=0.2,
    tick_period_mono=0.02,
    status_timeout_mono=0.25,
    request_max_age_sim=0.2,
    max_linear_speed=0.3,
    max_angular_speed=0.5,
    unused_component_tolerance=1e-9,
    shutdown_stop_repeats=10,
    shutdown_stop_interval_mono=0.02,
)

# 1 Hz Odometry: settle 0.5 s 이후 1.0 s 이상 떨어진 새 샘플 2개가 모두 정지여야 확인.
DEMO_1HZ_STOP_CHECK = StopCheckConfig(
    name="demo_vehicle_blue_stop_check",
    linear_tolerance=0.01,
    angular_tolerance=0.02,
    settle_sim=0.5,
    window_sim=1.0,
    min_samples=2,
    sample_max_age_sim=1.5,
    timeout_sim=4.0,
    # 실시간 배율 약 0.67 까지는 sim 판정이 먼저 난다. 그보다 느리거나 /clock 이 멈추면
    # 6 s 뒤 UNVERIFIABLE.
    timeout_mono=6.0,
)

MAPF_50HZ_STOP_CHECK = StopCheckConfig(
    name="mapf_robot_50hz_stop_check",
    linear_tolerance=0.01,
    angular_tolerance=0.02,
    settle_sim=0.5,
    window_sim=0.2,
    min_samples=5,
    sample_max_age_sim=0.1,
    timeout_sim=2.0,
    timeout_mono=3.0,
)
