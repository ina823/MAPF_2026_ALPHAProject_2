"""Odometry Bridge 장애 주입 도구 검증 (Gazebo 불필요).

- 순수 로직: Bridge 사양, 실행 인자 검증, 주입 조건, 기록/판정, Topic 구성 판정.
- 실제 프로세스: 가짜 프로세스로 /proc 읽기와 pidfd 신호를 확인한다.
- 실제 parameter_bridge (설치된 경우): 테스트 전용 ROS_DOMAIN_ID / GZ_PARTITION /
  Topic 으로 분리 Bridge 3개를 띄워 탐색·검증·Odometry Bridge 단독 종료를 확인한다.
  Gazebo 서버 없이 Bridge 프로세스만 확인하며, 실제 Gazebo 데이터 흐름 검증은 아니다.
"""

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from src.ros_bridge import fault_injection as fi
from src.ros_bridge.safety_config import DEMO_VEHICLE_BLUE, make_robot_interface

ROBOT = "vehicle_blue"
ODOM_SPEC = "/model/vehicle_blue/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry"
CMD_SPEC = "/model/vehicle_blue/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist"
CLOCK_SPEC = "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"
ODOM_NODE = "mapf_odom_bridge_vehicle_blue"
BRIDGE_BIN = "/opt/ros/jazzy/lib/ros_gz_bridge/parameter_bridge"

try:  # ROS 환경(source 된 라이브러리 경로)이 있어야 실제 Bridge 가 실행된다
    import rclpy  # noqa: F401
    HAVE_ROS_ENV = bool(os.environ.get("AMENT_PREFIX_PATH"))
except ImportError:  # pragma: no cover - ROS 미설치 환경
    HAVE_ROS_ENV = False


def identity(argv, pid=4242, exe="parameter_bridge", start=1000):
    return fi.ProcessIdentity(pid, tuple(argv), start, exe)


def bridge_argv(spec=ODOM_SPEC, node=ODOM_NODE, prefix=BRIDGE_BIN):
    return [prefix, spec, "--ros-args", "-r", f"__node:={node}"]


class TestSpecs(unittest.TestCase):

    def test_specs_match_interface(self):
        self.assertEqual(fi.odom_bridge_spec(DEMO_VEHICLE_BLUE), ODOM_SPEC)
        self.assertEqual(fi.cmd_vel_bridge_spec(DEMO_VEHICLE_BLUE), CMD_SPEC)
        self.assertEqual(fi.clock_bridge_spec(DEMO_VEHICLE_BLUE), CLOCK_SPEC)

    def test_directions_are_one_way(self):
        # 분리 환경에는 양방향(@ 두 번) 사양이 없어야 한다
        for spec in (ODOM_SPEC, CMD_SPEC, CLOCK_SPEC):
            self.assertEqual(spec.count("@"), 1, spec)
        self.assertIn("]gz.msgs.Twist", CMD_SPEC)        # ROS → Gazebo
        self.assertIn("[gz.msgs.Odometry", ODOM_SPEC)    # Gazebo → ROS
        self.assertIn("[gz.msgs.Clock", CLOCK_SPEC)

    def test_bad_spec(self):
        with self.assertRaises(ValueError):
            fi.bridge_spec("/a", "b", "c", "?")
        with self.assertRaises(ValueError):
            fi.bridge_spec("/a b", "b", "c", fi.GZ_TO_ROS)
        with self.assertRaises(ValueError):
            fi.bridge_spec("", "b", "c", fi.GZ_TO_ROS)

    def test_node_names(self):
        self.assertEqual(fi.odom_bridge_node(ROBOT), ODOM_NODE)
        self.assertEqual(fi.watchdog_node(ROBOT), "mapf_watchdog_vehicle_blue")
        self.assertEqual(fi.spec_topic(ODOM_SPEC), "/model/vehicle_blue/odometry")
        self.assertIsNone(fi.spec_topic("--ros-args"))

    def test_watchdog_node_name_matches_node_class(self):
        # gz_watchdog_node 의 기본 Node 이름 규칙과 같아야 Topic 검사가 맞다
        path = os.path.join(os.path.dirname(__file__), os.pardir,
                            "src", "ros_bridge", "gz_watchdog_node.py")
        with open(path) as f:
            self.assertIn('f"mapf_watchdog_{interface.robot_id}"', f.read())


class TestValidateBridgeProcess(unittest.TestCase):

    def check(self, ident, spec=ODOM_SPEC, node=ODOM_NODE, **kw):
        return fi.validate_bridge_process(ident, spec, node, self_pid=1, **kw)

    def test_accepts_dedicated_odom_bridge(self):
        ok, reason = self.check(identity(bridge_argv()))
        self.assertTrue(ok, reason)

    def test_rejects_combined_bidirectional_bridge(self):
        # 기존 diff_drive.launch.py 공통 Bridge
        argv = [BRIDGE_BIN,
                "/model/vehicle_blue/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",
                "/model/vehicle_blue/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry"]
        ok, _ = self.check(identity(argv), node="parameter_bridge")
        self.assertFalse(ok)

    def test_rejects_odom_spec_mixed_with_cmd_vel(self):
        argv = [BRIDGE_BIN, ODOM_SPEC, CMD_SPEC, "--ros-args", "-r",
                f"__node:={ODOM_NODE}"]
        ok, reason = self.check(identity(argv))
        self.assertFalse(ok)
        self.assertIn("bridge args", reason)

    def test_rejects_cmd_and_clock_bridges(self):
        for spec, node in ((CMD_SPEC, "mapf_cmd_vel_bridge_vehicle_blue"),
                           (CLOCK_SPEC, "mapf_clock_bridge")):
            with self.subTest(spec=spec):
                ok, _ = self.check(identity(bridge_argv(spec, node)))
                self.assertFalse(ok)

    def test_rejects_wrong_node_name(self):
        ok, reason = self.check(identity(bridge_argv(node="other")))
        self.assertFalse(ok)
        self.assertIn("node name", reason)
        argv = [BRIDGE_BIN, ODOM_SPEC]
        self.assertFalse(self.check(identity(argv))[0])

    def test_rejects_ros2_run_parent(self):
        # `ros2 run` 상위 Python 프로세스는 같은 인자를 갖지만 대상이 아니다
        argv = ["/usr/bin/python3", "/opt/ros/jazzy/bin/ros2", "run",
                "ros_gz_bridge", "parameter_bridge", ODOM_SPEC,
                "--ros-args", "-r", f"__node:={ODOM_NODE}"]
        ok, reason = self.check(identity(argv, exe="python3.12"))
        self.assertFalse(ok)
        self.assertIn("argv[0]", reason)

    def test_rejects_wrong_executable(self):
        ok, reason = self.check(identity(bridge_argv(), exe="bash"))
        self.assertFalse(ok)
        self.assertIn("executable", reason)
        ok, reason = self.check(identity(bridge_argv(), exe=None))
        self.assertFalse(ok)

    def test_rejects_pid_1_and_self(self):
        self.assertFalse(self.check(identity(bridge_argv(), pid=1))[0])
        ok, reason = fi.validate_bridge_process(
            identity(bridge_argv(), pid=99), ODOM_SPEC, ODOM_NODE, self_pid=99
        )
        self.assertFalse(ok)
        self.assertFalse(self.check(None)[0])

    def test_remap_forms(self):
        argv = [BRIDGE_BIN, ODOM_SPEC, "--ros-args", "--remap",
                f"__node:={ODOM_NODE}", "--log-level", "info"]
        self.assertTrue(self.check(identity(argv))[0])

    def test_same_process(self):
        a = identity(bridge_argv())
        self.assertTrue(fi.same_process(a, identity(bridge_argv())))
        self.assertFalse(fi.same_process(a, identity(bridge_argv(), start=1001)))
        self.assertFalse(fi.same_process(a, identity(bridge_argv(node="x"))))
        self.assertFalse(fi.same_process(a, None))


class TestFakeProc(unittest.TestCase):
    """가짜 /proc 디렉터리로 탐색을 검증한다."""

    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root)

    def add(self, pid, argv, exe="parameter_bridge", start=500, comm="x) y"):
        d = os.path.join(self.root, str(pid))
        os.makedirs(d)
        with open(os.path.join(d, "cmdline"), "wb") as f:
            f.write(b"\0".join(a.encode() for a in argv) + b"\0")
        fields = ["S"] + ["0"] * 18 + [str(start)] + ["0"] * 10
        with open(os.path.join(d, "stat"), "w") as f:
            f.write(f"{pid} ({comm}) " + " ".join(fields))
        if exe is not None:
            os.symlink(f"/opt/bin/{exe}", os.path.join(d, "exe"))

    def test_read_process(self):
        self.add(10, bridge_argv(), start=777)
        ident = fi.read_process(10, self.root)
        self.assertEqual(ident.argv, tuple(bridge_argv()))
        self.assertEqual(ident.start_ticks, 777)
        self.assertEqual(ident.exe_name, "parameter_bridge")
        self.assertIsNone(fi.read_process(11, self.root))
        self.assertIsNone(fi.read_process(0, self.root))

    def test_find_exactly_one(self):
        self.add(10, bridge_argv())
        self.add(11, bridge_argv(CMD_SPEC, "mapf_cmd_vel_bridge_vehicle_blue"))
        self.add(12, bridge_argv(CLOCK_SPEC, "mapf_clock_bridge"))
        self.add(13, ["/usr/bin/python3", "/opt/ros/jazzy/bin/ros2", "run",
                      "ros_gz_bridge", "parameter_bridge", ODOM_SPEC,
                      "--ros-args", "-r", f"__node:={ODOM_NODE}"],
                 exe="python3.12")
        self.add(14, ["bash"], exe="bash")
        accepted, rejected = fi.find_bridge_processes(
            ODOM_SPEC, ODOM_NODE, self.root, self_pid=1
        )
        self.assertEqual([i.pid for i, _ in accepted], [10])
        self.assertEqual([i.pid for i, _ in rejected], [13])

    def test_two_odom_bridges_found(self):
        self.add(10, bridge_argv())
        self.add(20, bridge_argv())
        accepted, _ = fi.find_bridge_processes(
            ODOM_SPEC, ODOM_NODE, self.root, self_pid=1
        )
        self.assertEqual(len(accepted), 2)  # 호출 측이 "정확히 하나" 가 아니면 중단

    def test_zombie_or_unreadable(self):
        d = os.path.join(self.root, "30")
        os.makedirs(d)
        open(os.path.join(d, "cmdline"), "wb").close()
        with open(os.path.join(d, "stat"), "w") as f:
            f.write("30 (z) Z 0")
        self.assertIsNone(fi.read_process(30, self.root))


class TestTopology(unittest.TestCase):

    def test_ok(self):
        r = fi.check_topology(
            ROBOT, ["/mapf_watchdog_vehicle_blue"],
            ["/mapf_cmd_vel_bridge_vehicle_blue", "/injector"],
            ["/mapf_odom_bridge_vehicle_blue"],
        )
        self.assertTrue(r.ok, r.problems)

    def test_bidirectional_bridge_is_second_publisher(self):
        r = fi.check_topology(
            ROBOT, ["/mapf_watchdog_vehicle_blue", "/ros_gz_bridge"],
            ["/ros_gz_bridge"], ["/ros_gz_bridge"],
        )
        self.assertFalse(r.ok)
        self.assertEqual(len(r.problems), 3)

    def test_executor_publishing_cmd_vel_directly(self):
        r = fi.check_topology(
            ROBOT, ["/mapf_watchdog_vehicle_blue", "/mapf_gz_executor"],
            ["/mapf_cmd_vel_bridge_vehicle_blue"],
            ["/mapf_odom_bridge_vehicle_blue"],
        )
        self.assertFalse(r.ok)

    def test_no_watchdog(self):
        r = fi.check_topology(
            ROBOT, [], ["/mapf_cmd_vel_bridge_vehicle_blue"],
            ["/mapf_odom_bridge_vehicle_blue"],
        )
        self.assertFalse(r.ok)

    def test_two_odom_publishers(self):
        r = fi.check_topology(
            ROBOT, ["/mapf_watchdog_vehicle_blue"],
            ["/mapf_cmd_vel_bridge_vehicle_blue"],
            ["/mapf_odom_bridge_vehicle_blue", "/ros_gz_bridge"],
        )
        self.assertFalse(r.ok)


def status(state="ACTIVE", output=(0.3, 0.0), fault=None, token="b-0",
           stop_check="NOT_STARTED", robot=ROBOT):
    return {
        "robot_id": robot, "state": state, "fault": fault,
        "fault_detail": None if fault is None else "during motion",
        "token": token, "epoch": 0, "output": list(output),
        "stop_check": stop_check, "stop_detail": "", "health": "OK",
    }


class TestTrigger(unittest.TestCase):

    def test_fires_after_consecutive_active_moving(self):
        t = fi.InjectionTrigger(ROBOT, 2)
        self.assertFalse(t.on_status(status()))
        self.assertTrue(t.on_status(status()))
        self.assertTrue(t.fired)
        self.assertFalse(t.on_status(status()))  # 한 번만

    def test_requires_nonzero_output(self):
        t = fi.InjectionTrigger(ROBOT, 1)
        for out in ((0.0, 0.0), (float("nan"), 0.0), (float("inf"), 0.0),
                    None, (0.3,), "x"):
            with self.subTest(out=out):
                self.assertFalse(t.on_status(status(output=out) if out is not None
                                             else {**status(), "output": None}))
        self.assertTrue(t.on_status(status(output=(0.0, 0.5))))

    def test_requires_active_state_and_robot(self):
        t = fi.InjectionTrigger(ROBOT, 1)
        for s in (status(state="READY"), status(state="FAULT", fault="X"),
                  status(fault="X"), status(robot="vehicle_green"),
                  None, "ACTIVE"):
            with self.subTest(s=s):
                self.assertFalse(t.on_status(s))
        self.assertFalse(t.fired)

    def test_interruption_resets_count(self):
        t = fi.InjectionTrigger(ROBOT, 3)
        t.on_status(status())
        t.on_status(status())
        t.on_status(status(output=(0.0, 0.0)))
        t.on_status(status())
        self.assertFalse(t.on_status(status()))
        self.assertTrue(t.on_status(status()))

    def test_session_change_resets_count(self):
        t = fi.InjectionTrigger(ROBOT, 2)
        t.on_status(status(token="b-0"))
        self.assertFalse(t.on_status(status(token="b-1")))
        self.assertTrue(t.on_status(status(token="b-1")))

    def test_bad_min(self):
        with self.assertRaises(ValueError):
            fi.InjectionTrigger(ROBOT, 0)


class TestRecord(unittest.TestCase):

    BOUND = 3.05

    def run_scenario(self, fault="ODOM_RECEIVE_TIMEOUT", final="NO_FRESH_ODOM",
                     odom_after=0, nonzero_late=0, exit_code=1,
                     leave_after=1.6, cmd_alive=True):
        r = fi.InjectionRecord(robot_id=ROBOT, executor_spawned=True)
        r.on_status(status(), 99.9)
        r.kill_mono = 100.0
        r.bridge_exit_mono = 100.01
        r.on_cmd_vel(0.3, 0.0, 100.5)          # 아직 ACTIVE: 기록 안 함
        for _ in range(odom_after):
            r.on_odom(100.5)
        r.on_clock(100.5)
        r.on_status(status(), 101.0)
        t = 100.0 + leave_after
        r.on_status(status("FAULT", (0, 0), fault, stop_check="PENDING"), t)
        r.on_cmd_vel(0.3, 0.0, t + 0.05)       # grace 안
        for i in range(nonzero_late):
            r.on_cmd_vel(0.3, 0.0, t + 0.5 + i)
        r.on_cmd_vel(0.0, 0.0, t + 0.06)
        r.on_status(status("FAULT", (0, 0), fault, stop_check=final), t + 6.0)
        r.executor_exit_code = exit_code
        r.cmd_bridge_alive_after = cmd_alive
        r.cmd_bridge_subscribed_after = cmd_alive
        return r

    def checks(self, r):
        verdict, checks = r.evaluate(self.BOUND)
        return verdict, {c["check"]: c["ok"] for c in checks}

    def test_expected_outcome_passes(self):
        r = self.run_scenario()
        verdict, c = self.checks(r)
        self.assertEqual(verdict, fi.PASS, c)
        d = r.to_dict(self.BOUND)
        self.assertEqual(d["physical_stop"], fi.PHYSICAL_STOP_NOT_VERIFIED)
        self.assertEqual(d["fault"], "ODOM_RECEIVE_TIMEOUT")
        self.assertAlmostEqual(d["timeline_since_kill_s"]["watchdog_left_active"], 1.6)
        self.assertEqual(d["cmd_vel"]["nonzero_in_grace"], 1)
        self.assertEqual(d["isolation"]["clock_msgs_after_kill"], 1)
        self.assertEqual([t["stop_check"] for t in d["stop_check_transitions"]],
                         ["NOT_STARTED", "PENDING", "NO_FRESH_ODOM"])

    def test_confirmed_stop_check_is_inconsistent(self):
        verdict, c = self.checks(self.run_scenario(final="CONFIRMED"))
        self.assertEqual(verdict, fi.FAIL)
        self.assertFalse(c["stop_check_not_confirmed"])

    def test_pending_stop_check_not_settled(self):
        verdict, c = self.checks(self.run_scenario(final="PENDING"))
        self.assertEqual(verdict, fi.FAIL)
        self.assertFalse(c["stop_check_settled"])

    def test_unverifiable_is_settled(self):
        verdict, _ = self.checks(self.run_scenario(final="UNVERIFIABLE"))
        self.assertEqual(verdict, fi.PASS)

    def test_odometry_from_other_publisher_fails_isolation(self):
        verdict, c = self.checks(self.run_scenario(odom_after=1))
        self.assertEqual(verdict, fi.FAIL)
        self.assertFalse(c["odometry_isolated"])

    def test_motion_command_after_fault_fails(self):
        verdict, c = self.checks(self.run_scenario(nonzero_late=1))
        self.assertFalse(c["no_motion_command_after_leave"])
        self.assertEqual(verdict, fi.FAIL)

    def test_slow_detection_fails(self):
        verdict, c = self.checks(self.run_scenario(leave_after=4.0))
        self.assertFalse(c["watchdog_left_active"])

    def test_executor_success_fails(self):
        verdict, c = self.checks(self.run_scenario(exit_code=0))
        self.assertFalse(c["executor_reported_failure"])

    def test_cmd_bridge_lost_fails(self):
        verdict, c = self.checks(self.run_scenario(cmd_alive=False))
        self.assertFalse(c["cmd_vel_bridge_kept"])

    def test_executor_stopped_first_without_fault(self):
        # Executor 가 먼저 감지해 정지 요청 → Watchdog 는 FAULT 없이 WAITING
        r = fi.InjectionRecord(robot_id=ROBOT)
        r.kill_mono = 100.0
        r.bridge_exit_mono = 100.0
        r.on_status(status("WAITING", (0, 0), stop_check="PENDING"), 101.8)
        r.on_cmd_vel(0.0, 0.0, 101.85)
        r.on_status(status("WAITING", (0, 0), stop_check="NO_FRESH_ODOM"), 106.0)
        r.cmd_bridge_alive_after = r.cmd_bridge_subscribed_after = True
        verdict, c = self.checks(r)
        self.assertEqual(verdict, fi.PASS, c)
        self.assertIsNone(r.fault)

    def test_not_injected_is_aborted(self):
        r = fi.InjectionRecord(robot_id=ROBOT)
        self.assertEqual(r.evaluate(self.BOUND)[0], fi.ABORTED)


_DECOY = "import time\nwhile True:\n    time.sleep(0.05)\n"


class TestRealProcess(unittest.TestCase):
    """가짜 Bridge 프로세스 (argv[0]=parameter_bridge, Python 실행) 로 pidfd 경로 확인."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # 스크립트 경로 자체를 사양 문자열로 만든다 → argv 가 실제 Bridge 와 같은 형태
        self.spec = os.path.join(
            self.tmp, "model", "decoy", "odometry@nav_msgs", "msg",
            "Odometry[gz.msgs.Odometry",
        )
        os.makedirs(os.path.dirname(self.spec))
        with open(self.spec, "w") as f:
            f.write(_DECOY)
        self.node = f"decoy_odom_bridge_{os.getpid()}"
        self.exe = os.path.basename(os.path.realpath(sys.executable))
        self.procs = []

    def tearDown(self):
        for p in self.procs:
            if p.poll() is None:
                p.kill()
            p.wait(timeout=5)
        shutil.rmtree(self.tmp)

    def spawn(self, node=None):
        p = subprocess.Popen(
            ["parameter_bridge", self.spec, "--ros-args", "-r",
             f"__node:={node or self.node}"],
            executable=sys.executable,
        )
        self.procs.append(p)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            ident = fi.read_process(p.pid)
            if ident is not None and ident.argv and ident.argv[0] == "parameter_bridge":
                return p, ident
            time.sleep(0.01)
        self.fail("decoy did not start")

    def test_discover_and_kill_only_target(self):
        target, _ = self.spawn()
        other, other_ident = self.spawn(node="some_other_node")
        accepted, rejected = fi.find_bridge_processes(
            self.spec, self.node, allowed_exe_names=(self.exe,)
        )
        self.assertEqual([i.pid for i, _ in accepted], [target.pid])
        self.assertIn(other.pid, [i.pid for i, _ in rejected])

        fd, reason = fi.open_verified_pidfd(
            accepted[0][0], self.spec, self.node, allowed_exe_names=(self.exe,)
        )
        self.assertIsNotNone(fd, reason)
        try:
            self.assertFalse(fi.pidfd_exited(fd))
            signal.pidfd_send_signal(fd, signal.SIGKILL)
            self.assertTrue(fi.pidfd_exited(fd, 5.0))
        finally:
            os.close(fd)
        self.assertEqual(target.wait(timeout=5), -signal.SIGKILL)
        self.assertIsNone(other.poll())  # 다른 프로세스는 그대로
        self.assertTrue(fi.same_process(other_ident, fi.read_process(other.pid)))

    def test_refuses_when_identity_changed(self):
        target, ident = self.spawn()
        forged = fi.ProcessIdentity(ident.pid, ident.argv, ident.start_ticks + 1,
                                    ident.exe_name)
        fd, reason = fi.open_verified_pidfd(
            forged, self.spec, self.node, allowed_exe_names=(self.exe,)
        )
        self.assertIsNone(fd)
        self.assertIn("identity changed", reason)
        self.assertIsNone(target.poll())

    def test_refuses_exited_process(self):
        target, ident = self.spawn()
        target.kill()
        target.wait(timeout=5)
        fd, _ = fi.open_verified_pidfd(
            ident, self.spec, self.node, allowed_exe_names=(self.exe,)
        )
        self.assertIsNone(fd)

    def test_production_exe_check_rejects_decoy(self):
        _, ident = self.spawn()
        ok, reason = fi.validate_bridge_process(ident, self.spec, self.node)
        self.assertFalse(ok)
        self.assertIn("executable", reason)


@unittest.skipUnless(
    HAVE_ROS_ENV and os.access(BRIDGE_BIN, os.X_OK),
    "ROS environment or ros_gz_bridge not available",
)
class TestRealParameterBridge(unittest.TestCase):
    """실제 parameter_bridge 3개를 분리 실행 (Gazebo 서버 없음, 격리된 Domain/Partition)."""

    def setUp(self):
        self.robot = f"fitest{os.getpid()}"
        self.interface = make_robot_interface(
            self.robot, clock_topic=f"/fitest_{os.getpid()}/clock"
        )
        env = dict(os.environ)
        env["ROS_DOMAIN_ID"] = str(170 + os.getpid() % 30)
        env["GZ_PARTITION"] = f"mapf_fitest_{os.getpid()}"
        env["GZ_IP"] = "127.0.0.1"
        self.procs = {}
        for key, spec, node in (
            ("cmd", fi.cmd_vel_bridge_spec(self.interface),
             fi.cmd_vel_bridge_node(self.robot)),
            ("odom", fi.odom_bridge_spec(self.interface),
             fi.odom_bridge_node(self.robot)),
            ("clock", fi.clock_bridge_spec(self.interface), f"fitest_clock_{os.getpid()}"),
        ):
            self.procs[key] = subprocess.Popen(
                [BRIDGE_BIN, spec, "--ros-args", "-r", f"__node:={node}"],
                env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

    def tearDown(self):
        for p in self.procs.values():
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(timeout=5)

    def find(self, spec, node):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            accepted, _ = fi.find_bridge_processes(spec, node)
            if accepted:
                return accepted
            time.sleep(0.05)
        return []

    def test_kill_odom_bridge_keeps_cmd_and_clock(self):
        odom_spec = fi.odom_bridge_spec(self.interface)
        odom_node = fi.odom_bridge_node(self.robot)
        accepted = self.find(odom_spec, odom_node)
        self.assertEqual([i.pid for i, _ in accepted], [self.procs["odom"].pid])
        ident = accepted[0][0]
        self.assertEqual(ident.exe_name, fi.PARAMETER_BRIDGE)

        # cmd_vel Bridge 는 Odometry 대상으로 거부된다
        cmd_ident = fi.read_process(self.procs["cmd"].pid)
        self.assertFalse(fi.validate_bridge_process(cmd_ident, odom_spec, odom_node)[0])
        clock_ident = fi.read_process(self.procs["clock"].pid)

        fd, reason = fi.open_verified_pidfd(ident, odom_spec, odom_node)
        self.assertIsNotNone(fd, reason)
        try:
            signal.pidfd_send_signal(fd, signal.SIGKILL)
            self.assertTrue(fi.pidfd_exited(fd, 5.0))
        finally:
            os.close(fd)
        self.assertEqual(self.procs["odom"].wait(timeout=5), -signal.SIGKILL)

        time.sleep(0.3)
        self.assertIsNone(self.procs["cmd"].poll())
        self.assertIsNone(self.procs["clock"].poll())
        self.assertTrue(fi.same_process(cmd_ident, fi.read_process(cmd_ident.pid)))
        self.assertTrue(fi.same_process(clock_ident, fi.read_process(clock_ident.pid)))


if __name__ == "__main__":
    unittest.main()
