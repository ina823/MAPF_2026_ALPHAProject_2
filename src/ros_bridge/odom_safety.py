"""Odometry 안전성 검사용 순수 함수 (rclpy 의존 없음).

입력 처리 규칙 (모든 함수 공통):

- **설정값**(max_age, max_future, max_speed, margin, 기대 Frame ID,
  quaternion 허용 오차)이 잘못되면 ``ValueError`` 를 던진다.
  설정 오류는 실행 중에 조용히 넘어가면 안 되므로 데이터 검사보다 먼저 확인한다.
- **데이터**(시각, 위치, Frame ID, quaternion)가 잘못되면
  예외 없이 "안전하지 않음" 쪽 결과를 돌려준다 (fail-closed).
  - 신선도/순서/Frame/quaternion 검사: ``False``
  - 위치 급변 검사: ``True``
- 숫자 데이터는 ``int``/``float`` 만 허용한다. ``None``, ``bool``, 문자열,
  NaN, ±Inf 는 잘못된 데이터다.
"""

import math


def _is_finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _require_positive(name, value):
    if not _is_finite_number(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _require_non_negative(name, value):
    if not _is_finite_number(value) or value < 0:
        raise ValueError(f"{name} must be non-negative and finite")


def is_odom_fresh(
    now,
    received_at,
    max_age=0.25,
    max_future=0.0
):
    """``now - received_at`` 이 ``[-max_future, max_age]`` 안이면 True.

    두 시각은 반드시 같은 시간 기준이어야 한다
    (예: 둘 다 time.monotonic(), 또는 둘 다 시뮬레이션 시간).
    경계값은 포함한다. 미래 시각은 ``max_future`` 까지만 허용한다
    (기본 0 → 미래 시각은 모두 거부).
    """
    _require_positive("max_age", max_age)
    _require_non_negative("max_future", max_future)

    if received_at is None:
        return False

    if not (_is_finite_number(now) and _is_finite_number(received_at)):
        return False

    age = now - received_at

    return -max_future <= age <= max_age


def is_stamp_fresh(
    sim_now,
    stamp,
    max_age=0.25,
    max_future=0.0
):
    """메시지 생성 시각(header.stamp)의 신선도.

    ``sim_now`` 는 /clock 기준 시뮬레이션 시간이어야 하며
    monotonic 시간과 섞어 쓰면 안 된다.
    """
    return is_odom_fresh(
        sim_now,
        stamp,
        max_age,
        max_future
    )


def is_stamp_newer(stamp, previous_stamp):
    """stamp 가 직전에 수락한 stamp 보다 엄격히 새로우면 True.

    같은 값(중복)과 더 작은 값(역순)은 False.
    음수/비유한 stamp 는 False. previous_stamp 가 None 이면 첫 메시지로 본다.
    """
    if not _is_finite_number(stamp) or stamp < 0:
        return False

    if previous_stamp is None:
        return True

    if not _is_finite_number(previous_stamp) or previous_stamp < 0:
        return False

    return stamp > previous_stamp


def is_frame_valid(
    frame_id,
    child_frame_id,
    expected_frame,
    expected_child
):
    """header.frame_id 와 child_frame_id 가 기대값과 정확히 같으면 True."""
    for name, value in (
        ("expected_frame", expected_frame),
        ("expected_child", expected_child),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")

    if expected_frame == expected_child:
        raise ValueError("expected_frame and expected_child must differ")

    if not isinstance(frame_id, str) or not isinstance(child_frame_id, str):
        return False

    return (
        bool(frame_id)
        and bool(child_frame_id)
        and frame_id == expected_frame
        and child_frame_id == expected_child
    )


def is_position_jump(
    previous_x,
    previous_y,
    current_x,
    current_y,
    dt,
    max_speed=0.3,
    margin=0.05
):
    """두 위치 사이 거리가 ``max_speed * dt + margin`` 을 넘으면 True.

    경계값(정확히 같은 거리)은 급변이 아니다.
    dt 는 두 메시지의 header.stamp 차이(시뮬레이션 시간)이며,
    0 이하이거나 비유한이면 판단할 수 없으므로 True(급변)로 본다.
    """
    _require_positive("max_speed", max_speed)
    _require_non_negative("margin", margin)

    values = (
        previous_x, previous_y,
        current_x, current_y,
        dt
    )

    if not all(_is_finite_number(v) for v in values):
        return True

    if dt <= 0:
        return True

    distance = math.hypot(
        current_x - previous_x,
        current_y - previous_y
    )

    max_distance = max_speed * dt + margin

    return distance > max_distance


def is_quaternion_valid(x, y, z, w, norm_tolerance=0.05):
    """유한하고 norm 이 1 ± norm_tolerance 안이면 True."""
    _require_non_negative("norm_tolerance", norm_tolerance)

    if norm_tolerance >= 1:
        raise ValueError("norm_tolerance must be < 1")

    values = (x, y, z, w)

    if not all(_is_finite_number(v) for v in values):
        return False

    norm = math.sqrt(x * x + y * y + z * z + w * w)

    return abs(norm - 1.0) <= norm_tolerance
