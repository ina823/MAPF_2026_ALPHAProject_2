"""종료 처리 공용 로직 (rclpy 의존 없음).

- SIGINT(Ctrl+C), SIGTERM: ``ShutdownRequested`` 로 바꿔 finally 의 정리 코드가 실행되게 한다.
- 정리(정지 명령 반복 발행) 중에는 SIGINT/SIGTERM 을 무시해 정리가 중간에 끊기지 않게 한다.
- SIGKILL 은 프로세스 안에서 처리할 수 없다. 독립 Watchdog 의 Command Timeout 이 대응한다.
"""

import signal
import sys
import time
import traceback


EXIT_OK = 0
EXIT_MOVE_FAILED = 1
EXIT_STOP_UNCONFIRMED = 2
EXIT_EXCEPTION = 3
EXIT_SIGINT = 130
EXIT_SIGTERM = 143

_HANDLED_SIGNALS = (signal.SIGINT, signal.SIGTERM)


class ShutdownRequested(BaseException):
    """SIGINT/SIGTERM 수신. ``except Exception`` 에 삼켜지지 않도록 BaseException."""

    def __init__(self, signum):
        super().__init__(signal.Signals(signum).name)
        self.signum = signum


def _raise_shutdown(signum, frame):
    raise ShutdownRequested(signum)


def install_signal_handlers():
    """SIGINT/SIGTERM → ShutdownRequested. 이전 handler 복원 함수를 돌려준다."""
    previous = {s: signal.getsignal(s) for s in _HANDLED_SIGNALS}
    for s in _HANDLED_SIGNALS:
        signal.signal(s, _raise_shutdown)

    def restore():
        for s, handler in previous.items():
            signal.signal(s, handler)

    return restore


def ignore_signals():
    for s in _HANDLED_SIGNALS:
        signal.signal(s, signal.SIG_IGN)


def exit_code_for(exc):
    if isinstance(exc, ShutdownRequested):
        return EXIT_SIGTERM if exc.signum == signal.SIGTERM else EXIT_SIGINT
    if isinstance(exc, KeyboardInterrupt):
        return EXIT_SIGINT
    return EXIT_EXCEPTION


def repeat_stop(publish_stop, repeats, interval, sleep=time.sleep):
    """정지 명령을 반복 발행. 발행 성공 횟수 (실제 정지 여부와 무관)."""
    published = 0
    for index in range(repeats):
        try:
            publish_stop()
        except Exception as exc:  # context 종료 등: 더 보낼 수 없음
            print(f"STOP_PUBLISH_ERROR: {exc!r}", file=sys.stderr)
            break
        published += 1
        if index + 1 < repeats:
            sleep(interval)
    return published


def run_guarded(body, cleanup, log=print):
    """body() 를 실행하고 어떤 종료 경로에서도 cleanup() 을 한 번 실행한다.

    반환값: 종료 코드. body() 의 반환값이 정상 경로의 종료 코드다.
    """
    restore = install_signal_handlers()
    code = EXIT_EXCEPTION
    try:
        code = body()
    except (ShutdownRequested, KeyboardInterrupt) as exc:
        ignore_signals()  # 두 번째 Ctrl+C 가 정리를 끊지 않도록 즉시 무시
        code = exit_code_for(exc)
        log(f"SHUTDOWN_REQUESTED: {exc}")
    except Exception:
        ignore_signals()
        code = EXIT_EXCEPTION
        log("UNHANDLED_EXCEPTION:\n" + traceback.format_exc())
    finally:
        ignore_signals()
        try:
            cleanup()
        except Exception:
            log("CLEANUP_ERROR:\n" + traceback.format_exc())
        restore()
    return code


_ODOM_TIMEOUT_REASONS = (
    "NO_ODOM", "ODOM_RECEIVE_TIMEOUT", "ODOM_STAMP_STALE",
    "ODOM_STAMP_AHEAD_OF_CLOCK",
)
_CLOCK_REASONS = (
    "NO_CLOCK", "CLOCK_TIMEOUT", "SIM_TIME_STALLED", "SIM_TIME_REGRESSION",
    "INVALID_CLOCK",
)


def failure_category(reason):
    """이동 실패 사유를 종료 보고용 분류로 바꾼다."""
    if reason == "MOVE_TIMEOUT":
        return "MOVE_TIMEOUT"
    if reason == "SHUTDOWN":
        return "SHUTDOWN"
    if reason in _ODOM_TIMEOUT_REASONS:
        return "ODOM_TIMEOUT"
    if reason in _CLOCK_REASONS:
        return "CLOCK"
    if isinstance(reason, str) and reason.startswith("WATCHDOG_"):
        return "WATCHDOG"
    return "FEEDBACK_FAULT"
