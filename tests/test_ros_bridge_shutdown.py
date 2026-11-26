"""종료 처리 검증 (ROS2 불필요).

실제 하위 프로세스에 SIGINT/SIGTERM/SIGKILL 을 보내 run_guarded 의 정리 경로를 확인한다.
여기서 확인하는 것은 "정지 명령 발행 함수 호출"이며 실제 로봇 정지는 아니다.
"""

import os
import signal
import subprocess
import sys
import textwrap
import unittest

from src.ros_bridge import shutdown as sd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHILD = textwrap.dedent("""
    import sys, time
    sys.path.insert(0, {root!r})
    from src.ros_bridge.shutdown import run_guarded, repeat_stop

    mode = sys.argv[1]
    interval = float(sys.argv[2])

    def body():
        print("READY", flush=True)
        if mode == "exception":
            raise RuntimeError("boom")
        if mode == "normal":
            return 0
        if mode == "fail":
            return 1
        while True:
            time.sleep(0.01)

    def cleanup():
        n = repeat_stop(lambda: print("STOP", flush=True), 10, interval)
        print(f"CLEANUP {{n}}", flush=True)

    def log(text):
        print(text.splitlines()[0], flush=True)

    sys.exit(run_guarded(body, cleanup, log=log))
""").format(root=ROOT)


def start_child(mode, interval=0.001):
    proc = subprocess.Popen(
        [sys.executable, "-c", CHILD, mode, str(interval)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=ROOT,
    )
    line = proc.stdout.readline().strip()
    assert line == "READY", (line, proc.stderr.read())
    return proc


def finish(proc):
    # readline() 으로 이미 버퍼에 들어온 출력까지 읽기 위해 communicate() 대신 read()
    out = proc.stdout.read()
    err = proc.stderr.read()
    proc.wait(timeout=20)
    return proc.returncode, out.splitlines(), err


class TestRunGuardedProcess(unittest.TestCase):

    def test_normal_exit(self):
        code, lines, _ = finish(start_child("normal"))
        self.assertEqual(code, sd.EXIT_OK)
        self.assertEqual(lines.count("STOP"), 10)
        self.assertIn("CLEANUP 10", lines)

    def test_failure_exit_still_cleans_up(self):
        code, lines, _ = finish(start_child("fail"))
        self.assertEqual(code, sd.EXIT_MOVE_FAILED)
        self.assertIn("CLEANUP 10", lines)

    def test_python_exception(self):
        code, lines, _ = finish(start_child("exception"))
        self.assertEqual(code, sd.EXIT_EXCEPTION)
        self.assertIn("UNHANDLED_EXCEPTION:", lines)
        self.assertIn("CLEANUP 10", lines)

    def test_sigint(self):
        proc = start_child("loop")
        proc.send_signal(signal.SIGINT)
        code, lines, _ = finish(proc)
        self.assertEqual(code, sd.EXIT_SIGINT)
        self.assertIn("SHUTDOWN_REQUESTED: SIGINT", lines)
        self.assertIn("CLEANUP 10", lines)

    def test_sigterm(self):
        proc = start_child("loop")
        proc.send_signal(signal.SIGTERM)
        code, lines, _ = finish(proc)
        self.assertEqual(code, sd.EXIT_SIGTERM)
        self.assertIn("SHUTDOWN_REQUESTED: SIGTERM", lines)
        self.assertEqual(lines.count("STOP"), 10)
        self.assertIn("CLEANUP 10", lines)

    def test_second_ctrl_c_does_not_cut_cleanup(self):
        proc = start_child("loop", interval=0.05)
        proc.send_signal(signal.SIGINT)
        self.assertEqual(proc.stdout.readline().strip(),
                         "SHUTDOWN_REQUESTED: SIGINT")
        self.assertEqual(proc.stdout.readline().strip(), "STOP")
        proc.send_signal(signal.SIGINT)
        proc.send_signal(signal.SIGTERM)
        code, lines, _ = finish(proc)
        self.assertEqual(code, sd.EXIT_SIGINT)
        self.assertIn("CLEANUP 10", lines)

    def test_sigkill_cannot_clean_up(self):
        # SIGKILL 은 프로세스 안에서 처리할 수 없다 → Watchdog Command Timeout 이 대응
        proc = start_child("loop")
        proc.kill()
        code, lines, _ = finish(proc)
        self.assertEqual(code, -signal.SIGKILL)
        self.assertNotIn("STOP", lines)


class TestHelpers(unittest.TestCase):

    def test_repeat_stop_counts(self):
        calls = []
        count = sd.repeat_stop(lambda: calls.append(1), 5, 0.0,
                               sleep=lambda s: None)
        self.assertEqual(count, 5)
        self.assertEqual(len(calls), 5)

    def test_repeat_stop_stops_on_error(self):
        calls = []

        def publish():
            calls.append(1)
            if len(calls) == 3:
                raise RuntimeError("context invalid")

        count = sd.repeat_stop(publish, 10, 0.0, sleep=lambda s: None)
        self.assertEqual(count, 2)

    def test_exit_codes(self):
        self.assertEqual(
            sd.exit_code_for(sd.ShutdownRequested(signal.SIGTERM)),
            sd.EXIT_SIGTERM,
        )
        self.assertEqual(
            sd.exit_code_for(sd.ShutdownRequested(signal.SIGINT)),
            sd.EXIT_SIGINT,
        )
        self.assertEqual(sd.exit_code_for(KeyboardInterrupt()), sd.EXIT_SIGINT)
        self.assertEqual(sd.exit_code_for(ValueError()), sd.EXIT_EXCEPTION)

    def test_shutdown_not_swallowed_by_except_exception(self):
        with self.assertRaises(sd.ShutdownRequested):
            try:
                raise sd.ShutdownRequested(signal.SIGTERM)
            except Exception:  # noqa: BLE001 - 의도적으로 삼키지 못함을 확인
                self.fail("swallowed")

    def test_handlers_restored(self):
        before = signal.getsignal(signal.SIGTERM)
        code = sd.run_guarded(lambda: 0, lambda: None, log=lambda m: None)
        self.assertEqual(code, 0)
        self.assertEqual(signal.getsignal(signal.SIGTERM), before)

    def test_cleanup_error_reported(self):
        logs = []

        def cleanup():
            raise RuntimeError("cleanup failed")

        code = sd.run_guarded(lambda: 0, cleanup, log=logs.append)
        self.assertEqual(code, 0)
        self.assertTrue(any("CLEANUP_ERROR" in m for m in logs))

    def test_failure_categories(self):
        cases = {
            "MOVE_TIMEOUT": "MOVE_TIMEOUT",
            "ODOM_RECEIVE_TIMEOUT": "ODOM_TIMEOUT",
            "ODOM_STAMP_STALE": "ODOM_TIMEOUT",
            "NO_ODOM": "ODOM_TIMEOUT",
            "CLOCK_TIMEOUT": "CLOCK",
            "SIM_TIME_STALLED": "CLOCK",
            "SIM_TIME_REGRESSION": "CLOCK",
            "WATCHDOG_UNAVAILABLE": "WATCHDOG",
            "WATCHDOG_FAULT:COMMAND_TIMEOUT": "WATCHDOG",
            "POSITION_JUMP": "FEEDBACK_FAULT",
            "SHUTDOWN": "SHUTDOWN",
        }
        for reason, category in cases.items():
            with self.subTest(reason=reason):
                self.assertEqual(sd.failure_category(reason), category)


if __name__ == "__main__":
    unittest.main()
