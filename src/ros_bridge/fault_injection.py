"""Odometry Bridge 장애 주입 실험의 순수 Python 로직 (rclpy 의존 없음).

구성
----
- Bridge 사양 문자열: ros_gz_bridge ``parameter_bridge`` 인자.
  ``@`` 양방향, ``[`` Gazebo→ROS 단방향, ``]`` ROS→Gazebo 단방향.
- 대상 프로세스 검증: /proc 의 실행 인자와 시작 시각으로 Odometry 전용 Bridge 만 고른다.
  종료 직전에 다시 읽어 같은 프로세스인지 확인한다 (PID 재사용 방지).
- Topic 구성 검증: 실제 cmd_vel 발행자가 Watchdog 하나뿐인지,
  Odometry 발행자가 Odometry 전용 Bridge 하나뿐인지.
- 주입 조건: Watchdog 상태가 ACTIVE 이고 출력 속도가 0 이 아닌 상태가 연속으로 관측될 때만.
  고정 sleep 으로 주입하지 않는다.
- 기록과 판정: 장애 감지 시간, Fault 원인, 정지 명령 관측, Executor 종료 코드, STOP_CHECK.
  Odometry 가 끊긴 상태이므로 실제 정지 확인은 주장하지 않는다.
"""

import math
import os
from dataclasses import dataclass, field


PARAMETER_BRIDGE = "parameter_bridge"
ROS_ARGS = "--ros-args"

# 방향 기호 (ros_gz_bridge)
BIDIRECTIONAL = "@"
GZ_TO_ROS = "["
ROS_TO_GZ = "]"


# ----------------------------------------------------------------------
# Bridge 사양 / Node 이름
# ----------------------------------------------------------------------

def bridge_spec(topic, ros_type, gz_type, direction):
    if direction not in (BIDIRECTIONAL, GZ_TO_ROS, ROS_TO_GZ):
        raise ValueError(f"unknown bridge direction {direction!r}")
    for value in (topic, ros_type, gz_type):
        if not isinstance(value, str) or not value:
            raise ValueError("bridge spec parts must be non-empty strings")
        if any(ch in value for ch in "@[] "):
            raise ValueError(f"invalid character in bridge spec part {value!r}")
    return f"{topic}@{ros_type}{direction}{gz_type}"


def cmd_vel_bridge_spec(interface):
    """cmd_vel 전용 ROS→Gazebo 단방향."""
    return bridge_spec(
        interface.cmd_vel_topic, "geometry_msgs/msg/Twist", "gz.msgs.Twist",
        ROS_TO_GZ,
    )


def odom_bridge_spec(interface):
    """Odometry 전용 Gazebo→ROS 단방향."""
    return bridge_spec(
        interface.odom_topic, "nav_msgs/msg/Odometry", "gz.msgs.Odometry",
        GZ_TO_ROS,
    )


def clock_bridge_spec(interface):
    """Clock 전용 Gazebo→ROS 단방향."""
    return bridge_spec(
        interface.clock_topic, "rosgraph_msgs/msg/Clock", "gz.msgs.Clock",
        GZ_TO_ROS,
    )


def cmd_vel_bridge_node(robot_id):
    return f"mapf_cmd_vel_bridge_{robot_id}"


def odom_bridge_node(robot_id):
    return f"mapf_odom_bridge_{robot_id}"


CLOCK_BRIDGE_NODE = "mapf_clock_bridge"


def watchdog_node(robot_id):
    """gz_watchdog_node.GazeboWatchdogNode 의 기본 Node 이름과 같아야 한다."""
    return f"mapf_watchdog_{robot_id}"


def spec_topic(spec):
    """사양 문자열의 Topic 부분. 사양이 아니면 None."""
    if not isinstance(spec, str) or BIDIRECTIONAL not in spec:
        return None
    topic = spec.split(BIDIRECTIONAL, 1)[0]
    return topic or None


# ----------------------------------------------------------------------
# 프로세스 식별 / 검증
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    argv: tuple
    start_ticks: int      # /proc/<pid>/stat 22번째 필드 (부팅 후 clock tick)
    exe_name: str = None  # /proc/<pid>/exe 의 basename (읽지 못하면 None)


def _read_start_ticks(stat_text):
    # comm 에 공백/괄호가 들어갈 수 있으므로 마지막 ')' 뒤에서 센다.
    tail = stat_text.rsplit(")", 1)[1].split()
    # tail[0] 은 3번째 필드(state). starttime 은 22번째 필드 → tail[19]
    return int(tail[19])


def read_process(pid, proc_root="/proc"):
    """``ProcessIdentity`` 또는 프로세스가 없거나 읽을 수 없으면 None."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return None
    base = os.path.join(proc_root, str(pid))
    try:
        with open(os.path.join(base, "cmdline"), "rb") as f:
            raw = f.read()
        with open(os.path.join(base, "stat"), "r") as f:
            stat_text = f.read()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    if not raw:
        return None  # 좀비 또는 커널 스레드
    argv = tuple(
        part.decode("utf-8", "replace") for part in raw.rstrip(b"\0").split(b"\0")
    )
    try:
        start_ticks = _read_start_ticks(stat_text)
    except (IndexError, ValueError):
        return None
    try:
        exe_name = os.path.basename(os.readlink(os.path.join(base, "exe")))
        if exe_name.endswith(" (deleted)"):
            exe_name = exe_name[: -len(" (deleted)")]
    except OSError:
        exe_name = None
    return ProcessIdentity(pid, argv, start_ticks, exe_name)


def _ros_remaps(argv):
    """``--ros-args`` 뒤의 ``-r a:=b`` / ``--remap a:=b`` 를 dict 로."""
    remaps = {}
    in_ros = False
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == ROS_ARGS:
            in_ros = True
        elif arg == "--":
            in_ros = False
        elif in_ros and arg in ("-r", "--remap") and i + 1 < len(argv):
            key, sep, value = argv[i + 1].partition(":=")
            if sep:
                remaps[key] = value
            i += 1
        i += 1
    return remaps


def bridge_positional_args(argv):
    """argv[1:] 중 ``--ros-args`` 이전 인자 (Bridge 사양 목록)."""
    args = list(argv[1:])
    if ROS_ARGS in args:
        args = args[: args.index(ROS_ARGS)]
    return tuple(args)


def validate_bridge_process(
    identity, expected_spec, expected_node,
    allowed_exe_names=(PARAMETER_BRIDGE,), self_pid=None,
):
    """(ok, reason). 지정된 단일 사양만 가진 parameter_bridge 인지 확인한다.

    - argv[0] 의 basename 이 ``parameter_bridge``
    - 실행 파일(/proc/<pid>/exe) basename 이 허용 목록 안 (읽을 수 없으면 거부)
    - ``--ros-args`` 이전 인자가 정확히 ``[expected_spec]`` 하나
      (cmd_vel/clock 등 다른 사양이 섞인 공통 Bridge 는 거부)
    - Node 이름 remap ``__node:=expected_node``
    - 자기 자신, PID 1 은 거부
    """
    if identity is None:
        return False, "process not found"
    if identity.pid <= 1:
        return False, f"refusing pid {identity.pid}"
    if identity.pid == (os.getpid() if self_pid is None else self_pid):
        return False, "refusing own pid"
    if not identity.argv or os.path.basename(identity.argv[0]) != PARAMETER_BRIDGE:
        return False, f"argv[0] is not {PARAMETER_BRIDGE}: {identity.argv[:1]!r}"
    if identity.exe_name is None:
        return False, "executable path unreadable"
    if identity.exe_name not in allowed_exe_names:
        return False, f"unexpected executable {identity.exe_name!r}"
    positional = bridge_positional_args(identity.argv)
    if positional != (expected_spec,):
        return False, f"bridge args {positional!r} != ({expected_spec!r},)"
    node = _ros_remaps(identity.argv).get("__node")
    if node != expected_node:
        return False, f"node name {node!r} != {expected_node!r}"
    return True, "ok"


def find_bridge_processes(
    expected_spec, expected_node, proc_root="/proc",
    allowed_exe_names=(PARAMETER_BRIDGE,), self_pid=None,
):
    """검증을 통과한 프로세스 목록과, 사양 문자열은 포함하지만 거부된 후보 목록.

    ``ros2 run`` 의 상위 Python 프로세스도 같은 인자를 갖지만 argv[0] 이
    parameter_bridge 가 아니므로 거부 목록으로 간다.
    """
    accepted, rejected = [], []
    try:
        names = os.listdir(proc_root)
    except OSError:
        return accepted, rejected
    for name in names:
        if not name.isdigit():
            continue
        identity = read_process(int(name), proc_root)
        if identity is None or expected_spec not in identity.argv:
            continue
        ok, reason = validate_bridge_process(
            identity, expected_spec, expected_node,
            allowed_exe_names, self_pid,
        )
        (accepted if ok else rejected).append((identity, reason))
    accepted.sort(key=lambda item: item[0].pid)
    rejected.sort(key=lambda item: item[0].pid)
    return accepted, rejected


def same_process(before, after):
    """종료 직전 재확인: PID·시작 시각·실행 인자가 모두 같아야 같은 프로세스."""
    return (
        before is not None and after is not None
        and before.pid == after.pid
        and before.start_ticks == after.start_ticks
        and before.argv == after.argv
        and before.exe_name == after.exe_name
    )


def open_verified_pidfd(
    identity, expected_spec, expected_node, proc_root="/proc",
    allowed_exe_names=(PARAMETER_BRIDGE,), self_pid=None,
):
    """(pidfd, reason). pidfd 를 먼저 열고 그 뒤에 다시 읽어 같은 프로세스인지 확인한다.

    pidfd 는 열린 시점의 프로세스에 고정되므로, 재확인이 통과하면 이후 PID 가
    재사용되어도 다른 프로세스에 신호가 가지 않는다.
    """
    try:
        fd = os.pidfd_open(identity.pid)
    except (ProcessLookupError, OSError) as exc:
        return None, f"pidfd_open failed: {exc}"
    current = read_process(identity.pid, proc_root)
    if not same_process(identity, current):
        os.close(fd)
        return None, "process identity changed since discovery"
    ok, reason = validate_bridge_process(
        current, expected_spec, expected_node, allowed_exe_names, self_pid
    )
    if not ok:
        os.close(fd)
        return None, reason
    return fd, "ok"


def pidfd_exited(fd, timeout_s=0.0):
    """pidfd 가 가리키는 프로세스가 종료했으면 True (최대 timeout_s 대기)."""
    import select

    poller = select.poll()
    poller.register(fd, select.POLLIN)
    return bool(poller.poll(max(0, int(timeout_s * 1000))))


# ----------------------------------------------------------------------
# Topic 구성 검증
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class TopologyCheck:
    ok: bool
    problems: tuple
    cmd_vel_publishers: tuple
    cmd_vel_subscribers: tuple
    odom_publishers: tuple


def _names(endpoints):
    return tuple(sorted(endpoints))


def check_topology(
    robot_id, cmd_vel_publishers, cmd_vel_subscribers, odom_publishers,
):
    """각 인자는 Node 전체 이름(``/ns/name`` 또는 ``name``) 목록.

    - cmd_vel 발행자는 Watchdog 하나뿐이어야 한다 (양방향 Bridge 가 남아 있으면 Bridge 도 발행자로 보인다)
    - cmd_vel 구독자에 cmd_vel 전용 Bridge 가 있어야 한다
    - Odometry 발행자는 Odometry 전용 Bridge 하나뿐이어야 한다
    """
    def bare(name):
        return name.rsplit("/", 1)[-1]

    problems = []
    wd = watchdog_node(robot_id)
    cmd_bridge = cmd_vel_bridge_node(robot_id)
    odom_bridge = odom_bridge_node(robot_id)

    pubs = [bare(n) for n in cmd_vel_publishers]
    if pubs != [wd]:
        problems.append(
            f"cmd_vel publishers must be exactly [{wd}], got {sorted(pubs)}"
        )
    subs = [bare(n) for n in cmd_vel_subscribers]
    if cmd_bridge not in subs:
        problems.append(f"cmd_vel bridge {cmd_bridge} is not subscribed")
    if subs.count(cmd_bridge) > 1:
        problems.append(f"duplicate cmd_vel bridge {cmd_bridge}")
    opubs = [bare(n) for n in odom_publishers]
    if opubs != [odom_bridge]:
        problems.append(
            f"odometry publishers must be exactly [{odom_bridge}], "
            f"got {sorted(opubs)}"
        )
    return TopologyCheck(
        not problems, tuple(problems),
        _names(cmd_vel_publishers), _names(cmd_vel_subscribers),
        _names(odom_publishers),
    )


# ----------------------------------------------------------------------
# 주입 조건
# ----------------------------------------------------------------------

def _finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def moving_output(output, epsilon=1e-9):
    """Watchdog 상태의 output [linear, angular] 이 유한하고 0 이 아니면 True."""
    if not isinstance(output, (list, tuple)) or len(output) != 2:
        return False
    if not all(_finite(v) for v in output):
        return False
    return abs(output[0]) > epsilon or abs(output[1]) > epsilon


class InjectionTrigger:
    """ACTIVE + 출력 속도 ≠ 0 상태가 같은 세션에서 연속 ``min_consecutive`` 번 관측되면 발동.

    한 번 발동하면 다시 발동하지 않는다 (``fired``).
    """

    def __init__(self, robot_id, min_consecutive=2):
        if not isinstance(min_consecutive, int) or min_consecutive < 1:
            raise ValueError("min_consecutive must be an int >= 1")
        self.robot_id = robot_id
        self.min_consecutive = min_consecutive
        self.consecutive = 0
        self.token = None
        self.fired = False
        self.fired_status = None

    def on_status(self, status):
        """status: parse_status() 결과 dict. 이번 상태로 주입해야 하면 True."""
        if self.fired:
            return False
        qualifies = (
            isinstance(status, dict)
            and status.get("robot_id") == self.robot_id
            and status.get("state") == "ACTIVE"
            and status.get("fault") is None
            and moving_output(status.get("output"))
        )
        if not qualifies:
            self.consecutive = 0
            self.token = None
            return False
        if self.token is not None and status.get("token") != self.token:
            self.consecutive = 0
        self.token = status.get("token")
        self.consecutive += 1
        if self.consecutive >= self.min_consecutive:
            self.fired = True
            self.fired_status = dict(status)
            return True
        return False


# ----------------------------------------------------------------------
# 기록 / 판정
# ----------------------------------------------------------------------

# 실험 결과 판정
PASS = "PASS"
FAIL = "FAIL"
ABORTED = "ABORTED"

# Odometry 가 끊긴 상태에서 물리적 정지에 대한 유일한 허용 결론
PHYSICAL_STOP_NOT_VERIFIED = "NOT_VERIFIED_ODOMETRY_LOST"

_TERMINAL_STOP_CHECKS = ("CONFIRMED", "NOT_STOPPED", "NO_FRESH_ODOM", "UNVERIFIABLE")


@dataclass
class InjectionRecord:
    """시각은 모두 injector 의 time.monotonic() (``*_mono``)."""

    robot_id: str
    target_pid: int = None
    target_argv: tuple = None
    signal_name: str = None

    trigger_status: dict = None
    kill_mono: float = None
    bridge_exit_mono: float = None

    # Watchdog 상태 관측
    first_non_active_mono: float = None
    first_non_active_status: dict = None
    first_fault_mono: float = None
    fault: str = None
    fault_detail: str = None
    stop_check_transitions: list = field(default_factory=list)
    last_status: dict = None

    # 실제 cmd_vel 관측 (injector 의 구독; 물리 정지가 아니라 명령 발행 관측)
    cmd_zero_after_leave: int = 0
    cmd_nonzero_after_leave: int = 0           # grace 이후
    cmd_nonzero_in_grace: int = 0              # 전환 직후 이미 전송 중이던 메시지
    first_stop_cmd_after_leave_mono: float = None

    # 장애 격리 관측
    odom_msgs_after_exit: int = 0
    clock_msgs_after_kill: int = 0
    cmd_bridge_alive_after: bool = None
    cmd_bridge_subscribed_after: bool = None

    executor_exit_code: int = None
    executor_spawned: bool = False
    executor_terminated_by_injector: bool = False

    notes: list = field(default_factory=list)

    cmd_grace_mono: float = 0.2

    def on_status(self, status, now_mono):
        self.last_status = dict(status)
        stop = status.get("stop_check")
        if not self.stop_check_transitions or (
            self.stop_check_transitions[-1]["stop_check"] != stop
        ):
            self.stop_check_transitions.append({
                "mono": now_mono,
                "since_kill": self._since_kill(now_mono),
                "stop_check": stop,
                "detail": status.get("stop_detail"),
                "state": status.get("state"),
            })
        if self.kill_mono is None:
            return
        if self.first_non_active_mono is None and status.get("state") != "ACTIVE":
            self.first_non_active_mono = now_mono
            self.first_non_active_status = dict(status)
        if self.first_fault_mono is None and status.get("fault") is not None:
            self.first_fault_mono = now_mono
            self.fault = status.get("fault")
            self.fault_detail = status.get("fault_detail")

    def on_cmd_vel(self, linear, angular, now_mono):
        if self.first_non_active_mono is None:
            return
        if moving_output((linear, angular)) or not (
            _finite(linear) and _finite(angular)
        ):
            if now_mono - self.first_non_active_mono <= self.cmd_grace_mono:
                self.cmd_nonzero_in_grace += 1
            else:
                self.cmd_nonzero_after_leave += 1
        else:
            self.cmd_zero_after_leave += 1
            if self.first_stop_cmd_after_leave_mono is None:
                self.first_stop_cmd_after_leave_mono = now_mono

    def on_odom(self, now_mono):
        if self.bridge_exit_mono is not None and now_mono > self.bridge_exit_mono:
            self.odom_msgs_after_exit += 1

    def on_clock(self, now_mono):
        if self.kill_mono is not None and now_mono > self.kill_mono:
            self.clock_msgs_after_kill += 1

    def _since_kill(self, mono):
        if mono is None or self.kill_mono is None:
            return None
        return round(mono - self.kill_mono, 4)

    @property
    def final_stop_check(self):
        if self.last_status is None:
            return None
        return self.last_status.get("stop_check")

    def stop_check_settled(self):
        return self.final_stop_check in _TERMINAL_STOP_CHECKS

    def evaluate(self, detection_bound_mono):
        """(verdict, checks) — checks 는 [(이름, 통과 여부, 설명)]."""
        checks = []

        def add(name, ok, detail):
            checks.append({"check": name, "ok": bool(ok), "detail": detail})

        if self.kill_mono is None:
            return ABORTED, checks

        latency = self._since_kill(self.first_non_active_mono)
        add(
            "watchdog_left_active",
            latency is not None and latency <= detection_bound_mono,
            f"latency={latency} s bound={detection_bound_mono} s "
            f"(state={None if self.first_non_active_status is None else self.first_non_active_status.get('state')})",
        )
        add(
            "no_motion_command_after_leave",
            self.cmd_nonzero_after_leave == 0,
            f"nonzero_after_grace={self.cmd_nonzero_after_leave} "
            f"in_grace={self.cmd_nonzero_in_grace} zero={self.cmd_zero_after_leave}",
        )
        add(
            "stop_command_observed",
            self.cmd_zero_after_leave > 0,
            f"zero cmd_vel messages after leaving ACTIVE: {self.cmd_zero_after_leave}",
        )
        add(
            "odometry_isolated",
            self.bridge_exit_mono is not None and self.odom_msgs_after_exit == 0,
            f"bridge_exited={self.bridge_exit_mono is not None} "
            f"odom_msgs_after_exit={self.odom_msgs_after_exit}",
        )
        add(
            "cmd_vel_bridge_kept",
            bool(self.cmd_bridge_alive_after) and bool(self.cmd_bridge_subscribed_after),
            f"process_alive={self.cmd_bridge_alive_after} "
            f"subscribed={self.cmd_bridge_subscribed_after}",
        )
        add(
            "stop_check_not_confirmed",
            self.final_stop_check != "CONFIRMED",
            f"final={self.final_stop_check} (CONFIRMED would be inconsistent: no odometry)",
        )
        add(
            "stop_check_settled",
            self.stop_check_settled(),
            f"final={self.final_stop_check}",
        )
        if self.executor_spawned:
            add(
                "executor_reported_failure",
                self.executor_exit_code not in (None, 0),
                f"exit_code={self.executor_exit_code}",
            )
        verdict = PASS if all(c["ok"] for c in checks) else FAIL
        return verdict, checks

    def to_dict(self, detection_bound_mono):
        verdict, checks = self.evaluate(detection_bound_mono)
        return {
            "robot_id": self.robot_id,
            "verdict": verdict,
            "checks": checks,
            "physical_stop": PHYSICAL_STOP_NOT_VERIFIED,
            "target": {
                "pid": self.target_pid,
                "argv": list(self.target_argv or ()),
                "signal": self.signal_name,
            },
            "trigger_status": self.trigger_status,
            "timeline_since_kill_s": {
                "bridge_exit": self._since_kill(self.bridge_exit_mono),
                "watchdog_left_active": self._since_kill(self.first_non_active_mono),
                "first_fault_status": self._since_kill(self.first_fault_mono),
                "first_stop_cmd_after_leave": self._since_kill(
                    self.first_stop_cmd_after_leave_mono
                ),
            },
            "fault": self.fault,
            "fault_detail": self.fault_detail,
            "first_non_active_status": self.first_non_active_status,
            "stop_check_final": self.final_stop_check,
            "stop_check_transitions": self.stop_check_transitions,
            "cmd_vel": {
                "zero_after_leave": self.cmd_zero_after_leave,
                "nonzero_after_leave": self.cmd_nonzero_after_leave,
                "nonzero_in_grace": self.cmd_nonzero_in_grace,
                "grace_s": self.cmd_grace_mono,
            },
            "isolation": {
                "odom_msgs_after_exit": self.odom_msgs_after_exit,
                "clock_msgs_after_kill": self.clock_msgs_after_kill,
                "cmd_bridge_alive_after": self.cmd_bridge_alive_after,
                "cmd_bridge_subscribed_after": self.cmd_bridge_subscribed_after,
            },
            "executor": {
                "spawned": self.executor_spawned,
                "exit_code": self.executor_exit_code,
                "terminated_by_injector": self.executor_terminated_by_injector,
            },
            "last_status": self.last_status,
            "notes": list(self.notes),
        }
