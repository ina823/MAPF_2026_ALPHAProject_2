import math


def quaternion_to_yaw(x, y, z, w):
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)

    return math.atan2(sin_yaw, cos_yaw)

def target_yaw(x, y, target_x, target_y):
    return math.atan2(target_y - y, target_x - x)


def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def angle_error(current_yaw, desired_yaw):
    return normalize_angle(desired_yaw - current_yaw)

def compute_velocity(x, y, yaw, target_x, target_y):
    values = (x, y, yaw, target_x, target_y)

    if not all(math.isfinite(v) for v in values):
        raise ValueError("Invalid pose or target")

    distance = math.hypot(target_x - x, target_y - y)

    position_tolerance = 0.05
    angle_tolerance = math.radians(5)

    max_linear = 0.3
    max_angular = 0.5

    kp_linear = 0.8
    kp_angular = 1.5

    # 1. 목표 위치 도착
    if distance <= position_tolerance:
        return 0.0, 0.0, True

    desired_yaw = target_yaw(x, y, target_x, target_y)
    error = angle_error(yaw, desired_yaw)

    # 2. 목표 방향으로 회전
    if abs(error) > angle_tolerance:
        angular = kp_angular * error
        angular = max(-max_angular, min(max_angular, angular))

        return 0.0, angular, False

    # 3. 목표 위치로 직진
    linear = min(max_linear, kp_linear * distance)

    return linear, 0.0, False
