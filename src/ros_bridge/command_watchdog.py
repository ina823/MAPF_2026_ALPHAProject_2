"""독립 Command Watchdog 의 순수 Python 로직 (rclpy 의존 없음).

Executor → (이동 요청) → CommandWatchdog → (실제 cmd_vel) → 로봇

출력 규칙
---------
- 실제 cmd_vel 은 이 Watchdog 만 결정한다.
- ACTIVE 이고 마지막 유효 요청이 Command Timeout 이내일 때만 이동 속도를 출력한다.
  그 외(WAITING/READY/FAULT/SHUTDOWN)에는 항상 정지 (0, 0).
- 새 유효 요청 없이 이동 명령을 Command Timeout 이상 반복하지 않는다.

Fault (잠김, 명시적 reset() 전까지 유지)
-----------------------------------------
- 이동 중 Command Timeout (Executor 중단/종료/SIGKILL 포함)
- 이동 중 Odometry/Clock 이상 (Timeout, 시뮬레이션 정지 등)
- Odometry 감시기가 잠근 fault (Frame, 급변, Quaternion, 시간 역행 등)
- NaN/Inf, 속도 상한 초과, 사용하지 않는 Twist 성분이 0 이 아닌 요청

Reset
-----
- 요청이 Command Timeout 안에 계속 들어오는 중이면 거부한다.
- 새 세션 token 을 만들고 이전 이동 명령/소유권/seq/Odometry 상태를 모두 버린다.
  따라서 Reset 전에 만들어졌거나 큐에 지연된 요청은 이전 token 이라 거부된다.
- Reset 직후는 WAITING 이며 새 Odometry/Clock 이 정상이어야 READY 가 된다.
"""

import math
from dataclasses import dataclass

from src.ros_bridge import odom_monitor as om
from src.ros_bridge.odom_monitor import (
    OdomSafetyMonitor,
    ACCEPTED,
    IGNORED_FAULTED,
)
from src.ros_bridge.odom_safety import is_frame_valid
from src.ros_bridge.stop_check import StopVerifier, StopCheck, UNVERIFIABLE
from src.ros_bridge.watchdog_protocol import (
    WAITING, READY, ACTIVE, FAULT, SHUTDOWN,
    decode_request_id,
    new_random_id,
)


STOP = (0.0, 0.0)

# 요청 처리 결과
FORWARDED = "FORWARDED"
REJECTED_BAD_ID = "REJECTED_BAD_ID"
REJECTED_STALE_SESSION = "REJECTED_STALE_SESSION"
REJECTED_NOT_OWNER = "REJECTED_NOT_OWNER"
REJECTED_OLD_SEQ = "REJECTED_OLD_SEQ"
REJECTED_STALE_REQUEST = "REJECTED_STALE_REQUEST"
REJECTED_NOT_READY = "REJECTED_NOT_READY"
REJECTED_FAULTED = "REJECTED_FAULTED"
REJECTED_SHUTDOWN = "REJECTED_SHUTDOWN"

# Watchdog 가 잠그는 fault
COMMAND_TIMEOUT = "COMMAND_TIMEOUT"
INVALID_VELOCITY = "INVALID_VELOCITY"
SPEED_LIMIT = "SPEED_LIMIT"
UNUSED_TWIST_COMPONENT = "UNUSED_TWIST_COMPONENT"

_LATCHING_REQUEST_FAULTS = (INVALID_VELOCITY, SPEED_LIMIT, UNUSED_TWIST_COMPONENT)

# Odometry/Clock 자체를 믿을 수 없는 fault: 이 상태에서는 Odometry 로 정지를 확인하지 않는다.
ODOM_INTEGRITY_FAULTS = frozenset((
    om.INVALID_FRAME, om.INVALID_STAMP, om.TIME_BASE_MISMATCH,
    om.INVALID_POSITION, om.INVALID_QUATERNION, om.INVALID_TWIST,
    om.POSITION_JUMP, om.SIM_TIME_REGRESSION, om.INVALID_CLOCK,
))


@dataclass(frozen=True)
class RequestResult:
    result: str
    output: tuple = None   # FORWARDED 일 때 즉시 발행할 (linear, angular)
    detail: str = ""

    @property
    def forwarded(self):
        return self.result == FORWARDED


def _finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _is_moving(command):
    return command is not None and command != STOP


class CommandWatchdog:

    def __init__(
        self,
        robot_id,
        watchdog_config,
        odom_config,
        stop_config,
        odom_frame_id,
        base_frame_id,
        boot_id=None,
    ):
        watchdog_config.validate_against(odom_config)

        self.robot_id = robot_id
        self.config = watchdog_config
        self.monitor = OdomSafetyMonitor(
            odom_config, odom_frame_id, base_frame_id
        )
        self.stop_verifier = StopVerifier(stop_config)
        self.boot_id = boot_id or new_random_id()
        self.epoch = 0

        self.shutting_down = False
        # Fault 후에도 갱신되는 Watchdog 자체 /clock (정지 확인용).
        # Odometry 감시기의 clock 은 Fault 시 갱신을 멈춘다.
        self.clock_sim = None
        # 마지막으로 관측한 monotonic 시각 (정지 확인 monotonic 제한의 시작 시각용)
        self.last_mono = None
        self.last_any_request_mono = None
        self.reset_count = 0
        self.forwarded_count = 0
        self.rejected = {}
        self.fault_history = []

        self._new_session()
        self._output = STOP
        self.health = "NOT_STARTED"

    # ------------------------------------------------------------------
    # 세션
    # ------------------------------------------------------------------

    @property
    def token(self):
        return f"{self.boot_id}-{self.epoch}"

    def _new_session(self):
        self.owner = None
        self.sender_last_seq = {}
        self.last_command = None
        self.last_command_mono = None
        self.last_request_mono = None

    # ------------------------------------------------------------------
    # 상태
    # ------------------------------------------------------------------

    @property
    def fault(self):
        return self.monitor.fault

    @property
    def fault_detail(self):
        return self.monitor.fault_detail

    def _latch(self, reason, detail, now_mono):
        first = self.monitor.fault is None
        self.monitor.latch(reason, detail)
        if first:
            self.fault_history.append({
                "reason": reason,
                "detail": detail,
                "mono": now_mono,
                "epoch": self.epoch,
            })
        return self.monitor.fault

    def _record_monitor_fault(self, now_mono):
        if self.monitor.fault is not None and (
            not self.fault_history
            or self.fault_history[-1]["epoch"] != self.epoch
        ):
            self.fault_history.append({
                "reason": self.monitor.fault,
                "detail": self.monitor.fault_detail,
                "mono": now_mono,
                "epoch": self.epoch,
            })

    def _reject(self, result, detail=""):
        self.rejected[result] = self.rejected.get(result, 0) + 1
        return RequestResult(result, None, detail)

    @property
    def state(self):
        if self.shutting_down:
            return SHUTDOWN
        if self.monitor.faulted:
            return FAULT
        if _is_moving(self.last_command):
            return ACTIVE
        if self.health != "OK":
            return WAITING
        return READY

    @property
    def output(self):
        return self._output

    # ------------------------------------------------------------------
    # 입력
    # ------------------------------------------------------------------

    def _note_mono(self, now_mono):
        if _finite(now_mono) and (
            self.last_mono is None or now_mono > self.last_mono
        ):
            self.last_mono = float(now_mono)

    def on_clock(self, clock_sim, received_mono):
        self._note_mono(received_mono)
        if _finite(clock_sim) and clock_sim >= 0 and (
            self.clock_sim is None or clock_sim >= self.clock_sim
        ):
            self.clock_sim = float(clock_sim)
        fault = self.monitor.on_clock(clock_sim, received_mono)
        if fault is not None:
            self._record_monitor_fault(received_mono)
            self._set_output(STOP)
        return fault

    def on_odom(
        self, stamp_sim, frame_id, child_frame_id, x, y,
        qx, qy, qz, qw, received_mono,
        linear_x=None, linear_y=None, angular_z=None,
    ):
        self._note_mono(received_mono)
        result = self.monitor.on_odom(
            stamp_sim, frame_id, child_frame_id, x, y,
            qx, qy, qz, qw, received_mono,
            linear_x=linear_x, linear_y=linear_y, angular_z=angular_z,
        )
        if result == ACCEPTED:
            self.stop_verifier.add_sample(
                stamp_sim, linear_x, linear_y, angular_z
            )
        elif result == IGNORED_FAULTED:
            # 명령/Timeout 계열 Fault 후에도 실제 정지 확인은 계속한다
            if self._odom_trusted_for_stop_check(
                stamp_sim, frame_id, child_frame_id
            ):
                self.stop_verifier.add_sample(
                    stamp_sim, linear_x, linear_y, angular_z
                )
        elif self.monitor.faulted:
            self._record_monitor_fault(received_mono)
            self._set_output(STOP)
        return result

    def _odom_trusted_for_stop_check(self, stamp_sim, frame_id, child_frame_id):
        if self.monitor.fault in ODOM_INTEGRITY_FAULTS:
            return False
        if not is_frame_valid(
            frame_id, child_frame_id,
            self.monitor.odom_frame_id, self.monitor.base_frame_id,
        ):
            return False
        if not _finite(stamp_sim) or self.clock_sim is None:
            return False
        # /clock 보다 크게 앞선 stamp 는 쓰지 않는다
        return (
            stamp_sim - self.clock_sim
            <= self.monitor.config.odom_stamp_future_tolerance_sim
        )

    def stop_check(self, now_mono=None):
        """now_mono 생략 시 마지막 관측 monotonic 시각으로 monotonic 제한을 판정한다."""
        if now_mono is not None:
            self._note_mono(now_mono)
        if (
            self.monitor.fault in ODOM_INTEGRITY_FAULTS
            and self.stop_verifier.started
        ):
            return StopCheck(UNVERIFIABLE, f"odometry untrusted ({self.fault})")
        return self.stop_verifier.evaluate(
            self.clock_sim,
            self.last_mono if now_mono is None else now_mono,
        )

    def on_request(
        self, request_id, stamp_sim,
        linear_x, linear_y, linear_z,
        angular_x, angular_y, angular_z,
        received_mono,
    ):
        cfg = self.config
        self._note_mono(received_mono)
        self.last_any_request_mono = received_mono

        if self.shutting_down:
            return self._reject(REJECTED_SHUTDOWN)

        decoded = decode_request_id(request_id)
        if decoded is None:
            return self._reject(REJECTED_BAD_ID, repr(request_id))
        token, sender, seq = decoded

        # 이전 세션(Reset 전, 다른 Watchdog 실행)의 요청: 큐 지연 요청 포함
        if token != self.token:
            return self._reject(REJECTED_STALE_SESSION, token)

        if self.monitor.faulted:
            return self._reject(REJECTED_FAULTED, self.monitor.fault)

        if self.owner is not None and sender != self.owner:
            return self._reject(REJECTED_NOT_OWNER, sender)

        last_seq = self.sender_last_seq.get(sender)
        if last_seq is not None and seq <= last_seq:
            return self._reject(REJECTED_OLD_SEQ, f"{seq} <= {last_seq}")

        clock_sim = self.monitor.last_clock_sim
        if clock_sim is None or not _finite(stamp_sim) or (
            abs(clock_sim - stamp_sim) > cfg.request_max_age_sim
        ):
            return self._reject(
                REJECTED_STALE_REQUEST,
                f"stamp={stamp_sim!r} clock={clock_sim!r}",
            )

        # 속도 값 검사: 잘못된 값은 Executor 결함으로 보고 잠근다
        values = (linear_x, linear_y, linear_z, angular_x, angular_y, angular_z)
        if not all(_finite(v) for v in values):
            self._latch(INVALID_VELOCITY, repr(values), received_mono)
            self._set_output(STOP)
            return RequestResult(INVALID_VELOCITY)

        unused = (linear_y, linear_z, angular_x, angular_y)
        if any(abs(v) > cfg.unused_component_tolerance for v in unused):
            self._latch(UNUSED_TWIST_COMPONENT, repr(unused), received_mono)
            self._set_output(STOP)
            return RequestResult(UNUSED_TWIST_COMPONENT)

        if (
            abs(linear_x) > cfg.max_linear_speed
            or abs(angular_z) > cfg.max_angular_speed
        ):
            self._latch(
                SPEED_LIMIT, f"linear={linear_x!r} angular={angular_z!r}",
                received_mono,
            )
            self._set_output(STOP)
            return RequestResult(SPEED_LIMIT)

        command = (float(linear_x), float(angular_z))

        # 준비되지 않은 상태에서는 이동 요청을 받지 않는다 (정지 요청은 무해하므로 수락)
        if _is_moving(command) and self.health != "OK":
            return self._reject(REJECTED_NOT_READY, self.health)

        self.owner = sender
        self.sender_last_seq[sender] = seq
        self.last_request_mono = received_mono
        self.last_command = command
        self.last_command_mono = received_mono

        self._set_output(command)
        self.forwarded_count += 1
        return RequestResult(FORWARDED, command)

    # ------------------------------------------------------------------
    # 주기 판정
    # ------------------------------------------------------------------

    def tick(self, now_mono):
        """주기적으로 호출. 지금 발행해야 할 실제 cmd_vel (linear, angular)."""
        cfg = self.config
        self._note_mono(now_mono)

        if self.shutting_down or self.monitor.faulted:
            self.health = self.monitor.fault or SHUTDOWN
            self._set_output(STOP)
            return STOP

        status = self.monitor.check(now_mono)
        self.health = status.reason

        moving = _is_moving(self.last_command)

        if moving:
            if not status.ok:
                self._latch(status.reason, "during motion", now_mono)
                self._set_output(STOP)
                return STOP

            age = now_mono - self.last_command_mono
            if not (0 <= age <= cfg.command_timeout_mono):
                self._latch(
                    COMMAND_TIMEOUT, f"no valid request for {age:.3f} s",
                    now_mono,
                )
                self._set_output(STOP)
                return STOP

            self._set_output(self.last_command)
            return self.last_command

        # 정지 상태: 소유 sender 가 조용하면 소유권 해제
        if self.owner is not None and (
            now_mono - self.last_request_mono > cfg.command_timeout_mono
        ):
            self.owner = None

        self._set_output(STOP)
        return STOP

    def _set_output(self, command):
        was_moving = _is_moving(self._output)
        self._output = command
        if not _is_moving(command):
            if was_moving:
                # 이동 → 정지 전환: 실제 정지 확인 시작
                clock = self.clock_sim
                if clock is not None:
                    self.stop_verifier.start(clock, self.last_mono)
                else:
                    self.stop_verifier.cancel()
        else:
            self.stop_verifier.cancel()
        if not _is_moving(command) and _is_moving(self.last_command) and (
            self.monitor.faulted or self.shutting_down
        ):
            self.last_command = None

    # ------------------------------------------------------------------
    # Reset / 종료
    # ------------------------------------------------------------------

    def reset(self, now_mono):
        """명시적 Reset. (성공 여부, 메시지)."""
        self._note_mono(now_mono)
        if self.shutting_down:
            return False, "shutting down"

        # 거부된 요청 포함, 어떤 요청이든 아직 들어오는 중이면 거부
        if self.last_any_request_mono is not None and (
            now_mono - self.last_any_request_mono
            <= self.config.command_timeout_mono
        ):
            return False, "requests still arriving; stop the executor first"

        previous = self.monitor.fault
        self.monitor.reset()
        self.stop_verifier.cancel()
        self.clock_sim = None
        self.epoch += 1
        self._new_session()
        self._output = STOP
        self.health = "RESET"
        self.reset_count += 1
        return True, f"reset from {previous}; new session {self.token}"

    def shutdown(self):
        self.shutting_down = True
        self.last_command = None
        self._set_output(STOP)

    # ------------------------------------------------------------------
    # 보고
    # ------------------------------------------------------------------

    def status(self, now_mono):
        stop = self.stop_check(now_mono)
        return {
            "robot_id": self.robot_id,
            "state": self.state,
            "fault": self.monitor.fault,
            "fault_detail": self.monitor.fault_detail,
            "token": self.token,
            "epoch": self.epoch,
            "output": list(self._output),
            "stop_check": stop.state,
            "stop_detail": stop.detail,
            "health": self.health,
            "owner": self.owner,
            "forwarded": self.forwarded_count,
            "rejected": dict(self.rejected),
            "resets": self.reset_count,
            "fault_history": self.fault_history[-5:],
        }
