"""Executor ↔ Watchdog 통신 계약 (rclpy 의존 없음).

표준 메시지만 사용한다 (사용자 정의 msg 패키지 빌드 불필요).

이동 요청: geometry_msgs/TwistStamped on ``RobotInterface.cmd_request_topic``
- ``header.frame_id`` = ``"<session_token>#<sender_id>#<seq>"``
  - session_token: Watchdog 가 상태 메시지로 알려 준 현재 세션. Reset 마다 바뀐다.
  - sender_id: Executor 프로세스마다 새로 만드는 임의 ID.
    한 세션에서는 한 sender 만 명령할 수 있다 (첫 유효 요청이 소유권을 가진다).
    소유 sender 가 정지 상태로 Command Timeout 이상 조용하면 소유권이 풀린다.
    이동 중 끊기면 Fault 로 잠기고 Reset 으로만 새 세션이 열린다.
  - seq: sender 별로 엄격히 증가하는 정수. 중복·역순 요청은 거부된다.
- ``header.stamp`` = Executor 가 마지막으로 받은 /clock (sim). 큐 지연 요청 차단용.
- ``twist.linear.x``, ``twist.angular.z`` 만 사용. 나머지 성분은 0 이어야 한다.

Watchdog 상태: std_msgs/String (JSON) on ``RobotInterface.watchdog_status_topic``.
"""

import json
import os
from dataclasses import dataclass


# Watchdog 상태
WAITING = "WAITING"          # Odometry/Clock 미준비. 정지만 출력
READY = "READY"              # 정상, 이동 명령 없음. 정지 출력
ACTIVE = "ACTIVE"            # 유효 이동 명령 전달 중
FAULT = "FAULT"              # 잠김. 정지만 출력. 명시적 Reset 필요
SHUTDOWN = "SHUTDOWN"        # 종료 중. 정지만 출력

STATES = (WAITING, READY, ACTIVE, FAULT, SHUTDOWN)

# Executor 측 판정 사유
WATCHDOG_UNAVAILABLE = "WATCHDOG_UNAVAILABLE"
WATCHDOG_STATUS_TIMEOUT = "WATCHDOG_STATUS_TIMEOUT"
WATCHDOG_NOT_READY = "WATCHDOG_NOT_READY"
WATCHDOG_FAULT = "WATCHDOG_FAULT"
WATCHDOG_ROBOT_MISMATCH = "WATCHDOG_ROBOT_MISMATCH"
WATCHDOG_SESSION_CHANGED = "WATCHDOG_SESSION_CHANGED"
WATCHDOG_OK = "OK"

STATUS_KEYS = (
    "robot_id", "state", "fault", "fault_detail", "token", "epoch",
    "output", "stop_check", "stop_detail", "health",
)

_SEPARATOR = "#"


def _valid_part(value):
    return isinstance(value, str) and bool(value) and _SEPARATOR not in value


def encode_request_id(token, sender_id, seq):
    if not _valid_part(token):
        raise ValueError("token must be a non-empty string without '#'")
    if not _valid_part(sender_id):
        raise ValueError("sender_id must be a non-empty string without '#'")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
        raise ValueError("seq must be a non-negative int")
    return f"{token}{_SEPARATOR}{sender_id}{_SEPARATOR}{seq}"


def decode_request_id(request_id):
    """``(token, sender_id, seq)`` 또는 형식이 잘못되면 None."""
    if not isinstance(request_id, str):
        return None
    parts = request_id.split(_SEPARATOR)
    if len(parts) != 3:
        return None
    token, sender_id, seq_text = parts
    if not token or not sender_id:
        return None
    if not seq_text.isascii() or not seq_text.isdigit():
        return None
    return token, sender_id, int(seq_text)


def new_random_id(nbytes=6):
    return os.urandom(nbytes).hex()


def status_to_json(status):
    return json.dumps(status, sort_keys=True, allow_nan=False)


def parse_status(text):
    """JSON 상태를 dict 로. 형식이 잘못되면 None."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    if any(key not in data for key in STATUS_KEYS):
        return None
    if data["state"] not in STATES:
        return None
    if not isinstance(data["token"], str) or not data["token"]:
        return None
    if not isinstance(data["robot_id"], str):
        return None
    return data


@dataclass(frozen=True)
class LinkStatus:
    ok: bool
    reason: str
    detail: str = ""


class WatchdogClient:
    """Executor 가 Watchdog 상태를 추적하고 요청 ID 를 만든다."""

    def __init__(self, robot_id, status_timeout_mono, sender_id=None):
        self.robot_id = robot_id
        self.status_timeout_mono = status_timeout_mono
        self.sender_id = sender_id or new_random_id()
        self.status = None
        self.status_received_mono = None
        self.invalid_status_count = 0
        self._seq = 0

    def on_status(self, text, received_mono):
        data = parse_status(text)
        if data is None:
            self.invalid_status_count += 1
            return False
        self.status = data
        self.status_received_mono = received_mono
        return True

    @property
    def token(self):
        return None if self.status is None else self.status["token"]

    def check(self, now_mono, pinned_token=None):
        """이동 명령을 보내도 되는지. pinned_token 은 이동 시작 시 세션."""
        status = self.status

        if status is None:
            return LinkStatus(False, WATCHDOG_UNAVAILABLE)

        age = now_mono - self.status_received_mono
        if not (0 <= age <= self.status_timeout_mono):
            return LinkStatus(False, WATCHDOG_STATUS_TIMEOUT, f"age={age:.3f}")

        if status["robot_id"] != self.robot_id:
            return LinkStatus(
                False, WATCHDOG_ROBOT_MISMATCH, status["robot_id"]
            )

        if status["state"] == FAULT:
            return LinkStatus(
                False, f"{WATCHDOG_FAULT}:{status['fault']}",
                str(status["fault_detail"]),
            )

        if status["state"] not in (READY, ACTIVE):
            return LinkStatus(
                False, f"{WATCHDOG_NOT_READY}:{status['state']}",
                str(status["health"]),
            )

        if pinned_token is not None and status["token"] != pinned_token:
            return LinkStatus(False, WATCHDOG_SESSION_CHANGED)

        return LinkStatus(True, WATCHDOG_OK)

    def next_request_id(self, token):
        """이 sender 에서 엄격히 증가하는 seq 로 요청 ID 를 만든다."""
        self._seq += 1
        return encode_request_id(token, self.sender_id, self._seq)
