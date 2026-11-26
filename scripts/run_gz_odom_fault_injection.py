"""이동 중 Odometry Bridge 장애 주입 실험 (분리 Bridge 환경 전용).

전제 (각각 별도 터미널, 프로젝트 루트):
    ros2 launch ros2_launch/gz_split_bridge.launch.py          # Gazebo + 분리 Bridge 3개
    python3 -m scripts.run_gz_watchdog --robot-id vehicle_blue  # 독립 Watchdog

실행:
    python3 -m scripts.run_gz_odom_fault_injection --robot-id vehicle_blue

절차
1. Odometry 전용 Bridge 프로세스를 /proc 에서 찾고 실행 인자를 검증한다
   (argv[0]=parameter_bridge, 사양이 Odometry 단방향 하나뿐, Node 이름 일치).
   cmd_vel 전용 Bridge 프로세스도 찾아 둔다 (종료 후 생존 확인용).
2. ROS Graph 검증: cmd_vel 발행자 = Watchdog 하나, Odometry 발행자 = Odometry Bridge 하나,
   이동 요청 발행자 없음 (다른 Executor 미실행).
3. Watchdog READY 확인 후 Executor (scripts.run_gz_single_robot) 를 자식 프로세스로 실행.
4. Watchdog 상태가 ACTIVE 이고 출력 속도가 0 이 아닌 상태를 연속 관측하면
   (고정 sleep 이 아님) pidfd 로 같은 프로세스임을 재확인한 뒤 신호를 보낸다.
   ARM 제한 시간 안에 조건이 오지 않으면 아무것도 종료하지 않고 중단한다.
5. 관측: 장애 감지 시간, Fault 원인, 실제 cmd_vel 정지 명령, STOP_CHECK,
   Executor 종료 코드, Odometry 중단 여부, cmd_vel Bridge 생존 여부.
6. JSON 기록: outputs/logs/gz_fault_injection/<시각>_odom_bridge_kill.json

이 도구는 물리적 정지를 확인하지 않는다. Odometry 가 끊긴 상태이므로
결과의 physical_stop 은 항상 NOT_VERIFIED_ODOMETRY_LOST 이다.

종료 코드: 0 PASS, 1 FAIL(기대 조건 위반), 2 ABORTED(주입 전 중단), 3 예외
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String

from src.ros_bridge import fault_injection as fi
from src.ros_bridge.gz_topology import endpoint_names, wait_for_topology
from src.ros_bridge.qos import CLOCK_QOS, CMD_QOS, ODOM_QOS, STATUS_QOS
from src.ros_bridge.safety_config import (
    DEMO_1HZ_CONFIG,
    DEMO_1HZ_STOP_CHECK,
    DEMO_1HZ_WATCHDOG,
    MAPF_50HZ_CONFIG,
    MAPF_50HZ_STOP_CHECK,
    MAPF_50HZ_WATCHDOG,
    make_robot_interface,
)
from src.ros_bridge.watchdog_protocol import READY, parse_status

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "outputs", "logs", "gz_fault_injection")

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_ABORTED = 2
EXIT_EXCEPTION = 3

PROFILES = {
    "demo": (DEMO_1HZ_WATCHDOG, DEMO_1HZ_CONFIG, DEMO_1HZ_STOP_CHECK),
    "mapf50": (MAPF_50HZ_WATCHDOG, MAPF_50HZ_CONFIG, MAPF_50HZ_STOP_CHECK),
}

SIGNALS = {"KILL": signal.SIGKILL, "TERM": signal.SIGTERM, "INT": signal.SIGINT}


class Aborted(Exception):
    pass


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot-id", default="vehicle_blue")
    p.add_argument("--profile", choices=sorted(PROFILES), default="demo",
                   help="Watchdog 실행 시 쓴 profile 과 같아야 한다")
    p.add_argument("--signal", choices=sorted(SIGNALS), default="KILL",
                   help="Odometry Bridge 에 보낼 신호 (기본 KILL: Bridge 비정상 종료 모사)")
    p.add_argument("--target-pid", type=int, default=None,
                   help="Odometry Bridge PID 직접 지정 (지정해도 같은 검증을 거친다)")
    p.add_argument("--min-active-statuses", type=int, default=2,
                   help="ACTIVE+출력≠0 연속 관측 횟수")
    p.add_argument("--ready-timeout", type=float, default=30.0)
    p.add_argument("--arm-timeout", type=float, default=20.0,
                   help="Executor 시작 후 ACTIVE 이동 관측 대기 상한")
    p.add_argument("--observe-timeout", type=float, default=25.0,
                   help="주입 후 관측 상한 (STOP_CHECK 확정 + Executor 종료 대기)")
    p.add_argument("--no-spawn-executor", action="store_true",
                   help="Executor 를 직접 띄우지 않는다 (다른 터미널에서 수동 실행)")
    p.add_argument("--out-dir", default=LOG_DIR)
    return p.parse_args(argv)


class InjectorNode(Node):

    def __init__(self, interface, record, trigger):
        super().__init__(f"mapf_odom_fault_injector_{interface.robot_id}")
        self.interface = interface
        self.record = record
        self.trigger = trigger
        self.fire = False
        self.last_status = None
        self.invalid_status = 0
        # 이 Node 는 아무것도 발행하지 않는다 (구독만).
        self.create_subscription(String, interface.watchdog_status_topic,
                                 self._on_status, STATUS_QOS)
        self.create_subscription(Twist, interface.cmd_vel_topic,
                                 self._on_cmd, CMD_QOS)
        self.create_subscription(Odometry, interface.odom_topic,
                                 self._on_odom, ODOM_QOS)
        self.create_subscription(Clock, interface.clock_topic,
                                 self._on_clock, CLOCK_QOS)

    def _on_status(self, msg):
        now = time.monotonic()
        status = parse_status(msg.data)
        if status is None:
            self.invalid_status += 1
            return
        self.last_status = status
        self.record.on_status(status, now)
        if self.trigger.on_status(status):
            self.fire = True

    def _on_cmd(self, msg):
        self.record.on_cmd_vel(msg.linear.x, msg.angular.z, time.monotonic())

    def _on_odom(self, msg):
        self.record.on_odom(time.monotonic())

    def _on_clock(self, msg):
        self.record.on_clock(time.monotonic())


def spin_for(node, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=min(0.01, max(0.0, end - time.monotonic())))


def discover(spec, node_name, target_pid=None):
    if target_pid is not None:
        identity = fi.read_process(target_pid)
        ok, reason = fi.validate_bridge_process(identity, spec, node_name)
        if not ok:
            raise Aborted(f"pid {target_pid} rejected: {reason}")
        return identity
    accepted, rejected = fi.find_bridge_processes(spec, node_name)
    for identity, reason in rejected:
        print(f"  candidate pid={identity.pid} rejected: {reason}")
    if len(accepted) != 1:
        raise Aborted(
            f"expected exactly one process for {spec!r} node={node_name}, "
            f"found {[i.pid for i, _ in accepted]}"
        )
    return accepted[0][0]


def git_state():
    def run(*cmd):
        try:
            return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                                  timeout=5).stdout.strip()
        except Exception:
            return None
    return {"head": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain"))}


def stop_executor(proc, record):
    if proc is None or proc.poll() is not None:
        return
    record.executor_terminated_by_injector = True
    record.notes.append("executor still running at end; sent SIGTERM")
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        record.notes.append("executor ignored SIGTERM for 5 s; sent SIGKILL")
        proc.kill()
        proc.wait(timeout=5.0)


def experiment(args, node, record, trigger, interface, configs, log):
    watchdog_cfg, odom_cfg, stop_cfg = configs
    odom_spec = fi.odom_bridge_spec(interface)
    odom_node = fi.odom_bridge_node(interface.robot_id)
    cmd_spec = fi.cmd_vel_bridge_spec(interface)
    cmd_node = fi.cmd_vel_bridge_node(interface.robot_id)

    # 1. 대상 프로세스
    log("Discovering bridge processes...")
    target = discover(odom_spec, odom_node, args.target_pid)
    cmd_bridge = discover(cmd_spec, cmd_node)
    record.target_pid = target.pid
    record.target_argv = target.argv
    log(f"  odometry bridge pid={target.pid} argv={list(target.argv)}")
    log(f"  cmd_vel  bridge pid={cmd_bridge.pid}")

    # 2. Graph
    log("Checking ROS graph...")
    topo, graph = wait_for_topology(node, interface, args.ready_timeout)
    record.notes.append(f"graph_before={graph}")
    if not topo.ok:
        raise Aborted(f"topology check failed: {list(topo.problems)}")
    if graph["request_publishers"]:
        raise Aborted(
            f"another publisher on {interface.cmd_request_topic}: "
            f"{graph['request_publishers']} (stop other executors first)"
        )
    log("  TOPOLOGY_OK: watchdog is the only cmd_vel publisher")

    # 3. Watchdog READY
    deadline = time.monotonic() + args.ready_timeout
    while rclpy.ok() and time.monotonic() < deadline:
        spin_for(node, 0.05)
        s = node.last_status
        if s is not None and s["robot_id"] == interface.robot_id and s["state"] == READY:
            break
    else:
        s = node.last_status
        raise Aborted(f"watchdog not READY within {args.ready_timeout}s "
                      f"(last={None if s is None else (s['state'], s['fault'], s['health'])})")
    log(f"  watchdog READY session={node.last_status['token']}")

    # 4. Executor
    proc = None
    exec_log_path = None
    if not args.no_spawn_executor:
        exec_log_path = os.path.join(args.out_dir, f"{args.stamp}_executor.log")
        exec_log = open(exec_log_path, "w")
        proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "scripts.run_gz_single_robot"],
            cwd=ROOT, stdout=exec_log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        exec_log.close()
        record.executor_spawned = True
        log(f"Executor started pid={proc.pid} (log: {exec_log_path})")
    else:
        log("Start the executor now in another terminal "
            "(python3 -m scripts.run_gz_single_robot)")

    # 5. ARM: 조건이 올 때까지 대기 (고정 sleep 없음)
    log(f"ARMED: waiting for ACTIVE with nonzero output "
        f"x{args.min_active_statuses} (limit {args.arm_timeout}s)")
    arm_deadline = time.monotonic() + args.arm_timeout
    while rclpy.ok() and not node.fire:
        if time.monotonic() > arm_deadline:
            stop_executor(proc, record)
            raise Aborted("ACTIVE motion not observed within arm timeout; nothing killed")
        if proc is not None and proc.poll() is not None:
            record.executor_exit_code = proc.returncode
            raise Aborted(f"executor exited ({proc.returncode}) before injection; nothing killed")
        rclpy.spin_once(node, timeout_sec=0.01)

    # 6. 주입: pidfd 로 고정 후 재검증, 신호
    fd, reason = fi.open_verified_pidfd(target, odom_spec, odom_node)
    if fd is None:
        stop_executor(proc, record)
        raise Aborted(f"target re-verification failed: {reason}; nothing killed")
    sig = SIGNALS[args.signal]
    record.signal_name = signal.Signals(sig).name
    record.trigger_status = trigger.fired_status
    try:
        signal.pidfd_send_signal(fd, sig)
        record.kill_mono = time.monotonic()
        log(f"INJECTED: {record.signal_name} -> odometry bridge pid={target.pid} "
            f"(watchdog output={trigger.fired_status['output']})")

        # 7. 관측
        obs_deadline = record.kill_mono + args.observe_timeout
        last_report = None
        while rclpy.ok() and time.monotonic() < obs_deadline:
            rclpy.spin_once(node, timeout_sec=0.01)
            now = time.monotonic()
            if record.bridge_exit_mono is None and fi.pidfd_exited(fd):
                record.bridge_exit_mono = now
                log(f"  odometry bridge exited (+{now - record.kill_mono:.3f}s)")
            s = node.last_status
            if s is not None:
                key = (s["state"], s["fault"], s["stop_check"])
                if key != last_report:
                    log(f"  +{now - record.kill_mono:6.3f}s state={s['state']} "
                        f"fault={s['fault']} output={s['output']} "
                        f"stop_check={s['stop_check']} {s['stop_detail']}")
                    last_report = key
            if proc is not None and record.executor_exit_code is None \
                    and proc.poll() is not None:
                record.executor_exit_code = proc.returncode
                log(f"  executor exited code={proc.returncode} "
                    f"(+{now - record.kill_mono:.3f}s)")
            executor_done = proc is None or record.executor_exit_code is not None
            if executor_done and record.stop_check_settled() \
                    and record.bridge_exit_mono is not None:
                # 마지막 정지 명령/상태를 조금 더 관측
                spin_for(node, 1.0)
                break
    finally:
        os.close(fd)

    stop_executor(proc, record)
    if proc is not None and record.executor_exit_code is None:
        record.executor_exit_code = proc.returncode

    # 8. 격리 확인
    current = fi.read_process(cmd_bridge.pid)
    record.cmd_bridge_alive_after = fi.same_process(cmd_bridge, current)
    subs = endpoint_names(node, interface.cmd_vel_topic, False)
    record.cmd_bridge_subscribed_after = any(
        n.rsplit("/", 1)[-1] == cmd_node for n in subs
    )
    return exec_log_path


def main(argv=None):
    args = parse_args(argv)
    args.stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.out_dir, exist_ok=True)
    configs = PROFILES[args.profile]
    watchdog_cfg, odom_cfg, stop_cfg = configs
    interface = make_robot_interface(args.robot_id)
    detection_bound = (
        odom_cfg.odom_receive_timeout_mono + watchdog_cfg.tick_period_mono + 1.0
    )

    record = fi.InjectionRecord(robot_id=args.robot_id)
    trigger = fi.InjectionTrigger(args.robot_id, args.min_active_statuses)

    def log(text):
        print(text, flush=True)

    def on_sigterm(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_sigterm)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = InjectorNode(interface, record, trigger)

    exit_code = EXIT_EXCEPTION
    exec_log_path = None
    error = None
    try:
        exec_log_path = experiment(args, node, record, trigger, interface, configs, log)
        verdict, _ = record.evaluate(detection_bound)
        exit_code = EXIT_PASS if verdict == fi.PASS else EXIT_FAIL
    except Aborted as exc:
        error = f"ABORTED: {exc}"
        exit_code = EXIT_ABORTED
    except KeyboardInterrupt:
        error = "interrupted"
        exit_code = EXIT_ABORTED if record.kill_mono is None else EXIT_FAIL
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        exit_code = EXIT_EXCEPTION
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

    result = record.to_dict(detection_bound)
    if error:
        result["error"] = error
        if record.kill_mono is None:
            result["verdict"] = fi.ABORTED
    result.update({
        "experiment": "odom_bridge_kill_during_motion",
        "timestamp": args.stamp,
        "profile": args.profile,
        "configs": {
            "watchdog": watchdog_cfg.name, "odom": odom_cfg.name,
            "stop_check": stop_cfg.name,
            "odom_receive_timeout_mono": odom_cfg.odom_receive_timeout_mono,
            "command_timeout_mono": watchdog_cfg.command_timeout_mono,
            "stop_check_timeout_sim": stop_cfg.timeout_sim,
            "stop_check_timeout_mono": stop_cfg.timeout_mono,
            "detection_bound_mono": detection_bound,
        },
        "trigger": {"min_active_statuses": args.min_active_statuses},
        "executor_log": exec_log_path,
        "git": git_state(),
        "exit_code": exit_code,
    })
    path = os.path.join(args.out_dir, f"{args.stamp}_odom_bridge_kill.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True, default=str)

    log("")
    log(f"VERDICT: {result['verdict']}" + (f" ({error})" if error else ""))
    for check in result["checks"]:
        log(f"  [{'PASS' if check['ok'] else 'FAIL'}] {check['check']}: {check['detail']}")
    t = result["timeline_since_kill_s"]
    first = result["first_non_active_status"] or {}
    log(f"  fault={result['fault']} detail={result['fault_detail']} "
        f"(first non-ACTIVE: state={first.get('state')} health={first.get('health')})")
    if result["fault"] is None and first:
        log("  note: watchdog left ACTIVE without FAULT "
            "(executor detected first and requested stop)")
    log(f"  watchdog_left_active=+{t['watchdog_left_active']}s "
        f"first_fault_status=+{t['first_fault_status']}s "
        f"first_stop_cmd=+{t['first_stop_cmd_after_leave']}s")
    if record.executor_spawned:
        log(f"  executor_exit_code={record.executor_exit_code}")
    else:
        log("  executor_exit_code=NOT_OBSERVED (manual executor: check `echo $?`)")
    log(f"  stop_check_final={result['stop_check_final']}")
    log(f"  physical_stop={result['physical_stop']} "
        "(odometry bridge terminated; physical stop is NOT verified)")
    log(f"RECORD: {path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
