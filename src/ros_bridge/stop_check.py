"""정지 명령 이후 Odometry twist 로 실제 정지를 확인한다 (rclpy 의존 없음).

"정지 명령 발행 성공"과 "실제 정지 확인 성공"은 다른 사건이다.
이 모듈은 후자만 판정한다.

판정 규칙 (모든 시각은 sim = header.stamp / /clock 기준)
- 정지 명령 시각 ``stop_sim`` + ``settle_sim`` 이전 stamp 의 샘플은 쓰지 않는다
  (정지 명령 전에 만들어진 오래된 Odometry 와 감속 구간 배제).
- 이후 새 샘플(stamp 가 엄격히 증가)이 연속으로 허용 오차 안에 있어야 한다.
  허용 오차를 넘는 샘플이 오면 구간을 처음부터 다시 센다.
- 연속 구간이 ``window_sim`` 이상, 샘플 ``min_samples`` 개 이상이고
  최신 샘플이 현재 /clock 대비 ``sample_max_age_sim`` 이내이면 CONFIRMED.
- ``stop_sim + timeout_sim`` 까지 확인하지 못하면 실패
  (새 샘플이 없으면 NO_FRESH_ODOM, 있으면 NOT_STOPPED).

monotonic 대기 제한 (Clock 장애 대비)
- 위 sim 기준 Timeout 은 /clock 이 멈추면 영원히 도달하지 않는다 (PENDING 고착).
- 그래서 판정 시작 후 ``timeout_mono`` (수신 측 time.monotonic) 가 지나도록
  결론이 나지 않으면 UNVERIFIABLE 로 확정한다. 정지 확인이 아니며 실패로 취급한다.
- monotonic 시작 시각은 ``start(stop_sim, start_mono)`` 로 주거나,
  생략하면 ``evaluate(now_sim, now_mono)`` 가 처음 now_mono 를 받은 시각으로 정한다.
- 확정된 결과는 다시 ``start`` 하기 전까지 바뀌지 않는다
  (Clock 이 복구된 뒤 들어온 샘플로 UNVERIFIABLE 을 CONFIRMED 로 뒤집지 않는다).
"""

import math
from dataclasses import dataclass


NOT_STARTED = "NOT_STARTED"
PENDING = "PENDING"
CONFIRMED = "CONFIRMED"
NOT_STOPPED = "NOT_STOPPED"
NO_FRESH_ODOM = "NO_FRESH_ODOM"
# Odometry 자체를 믿을 수 없어 정지 여부를 판정할 수 없음 (Frame/급변/시간 역행 등)
UNVERIFIABLE = "UNVERIFIABLE"

_WINDOW_EPSILON = 1e-6


@dataclass(frozen=True)
class StopCheck:
    state: str
    detail: str = ""

    @property
    def confirmed(self):
        return self.state == CONFIRMED

    @property
    def failed(self):
        """확정된 실패 (정지를 확인하지 못함). UNVERIFIABLE 포함."""
        return self.state in (NOT_STOPPED, NO_FRESH_ODOM, UNVERIFIABLE)

    @property
    def terminal(self):
        return self.confirmed or self.failed


def _finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


class StopVerifier:

    def __init__(self, config):
        self.config = config
        self.stop_sim = None
        self.start_mono = None
        self._clear()

    def _clear(self):
        self.window_start_sim = None
        self.window_count = 0
        self.latest_stamp_sim = None
        self.post_settle_count = 0
        self.ignored_count = 0
        self.last_moving_detail = None
        self.result = None

    @property
    def started(self):
        return self.stop_sim is not None

    def start(self, stop_sim, start_mono=None):
        """정지 명령을 발행한 시점의 /clock 시간으로 판정을 시작한다.

        start_mono: 같은 시점의 time.monotonic(). None 이면 첫 evaluate 의 now_mono.
        """
        if not _finite(stop_sim) or stop_sim < 0:
            raise ValueError("stop_sim must be finite and non-negative")
        if start_mono is not None and not _finite(start_mono):
            raise ValueError("start_mono must be finite")
        self.stop_sim = float(stop_sim)
        self.start_mono = None if start_mono is None else float(start_mono)
        self._clear()

    def cancel(self):
        self.stop_sim = None
        self.start_mono = None
        self._clear()

    def add_sample(self, stamp_sim, linear_x, linear_y, angular_z):
        if not self.started or self.result is not None:
            return

        cfg = self.config

        if not _finite(stamp_sim):
            self.ignored_count += 1
            return

        # 정지 명령 전/감속 구간 또는 중복·역순 샘플
        if stamp_sim <= self.stop_sim + cfg.settle_sim or (
            self.latest_stamp_sim is not None
            and stamp_sim <= self.latest_stamp_sim
        ):
            self.ignored_count += 1
            return

        self.latest_stamp_sim = float(stamp_sim)
        self.post_settle_count += 1

        twist = (linear_x, linear_y, angular_z)
        stopped = all(_finite(v) for v in twist) and (
            math.hypot(linear_x, linear_y) <= cfg.linear_tolerance
            and abs(angular_z) <= cfg.angular_tolerance
        )

        if stopped:
            if self.window_start_sim is None:
                self.window_start_sim = float(stamp_sim)
                self.window_count = 0
            self.window_count += 1
        else:
            self.window_start_sim = None
            self.window_count = 0
            self.last_moving_detail = (
                f"stamp={stamp_sim:.3f} v=({linear_x!r}, {linear_y!r}) "
                f"w={angular_z!r}"
            )

    def evaluate(self, now_sim, now_mono=None):
        """now_mono 를 주지 않으면 monotonic 제한은 적용되지 않는다 (sim 판정만)."""
        if not self.started:
            return StopCheck(NOT_STARTED)

        if self.result is not None:
            return self.result

        cfg = self.config

        if now_mono is not None and _finite(now_mono) and self.start_mono is None:
            self.start_mono = float(now_mono)

        if not _finite(now_sim):
            return self._mono_deadline(now_sim, now_mono) or StopCheck(
                PENDING, "no valid clock"
            )

        window_ok = (
            self.window_start_sim is not None
            and self.window_count >= cfg.min_samples
            and self.latest_stamp_sim - self.window_start_sim
            >= cfg.window_sim - _WINDOW_EPSILON
        )
        fresh = (
            self.latest_stamp_sim is not None
            and now_sim - self.latest_stamp_sim <= cfg.sample_max_age_sim
        )

        if window_ok and fresh:
            self.result = StopCheck(
                CONFIRMED,
                f"{self.window_count} samples over "
                f"{self.latest_stamp_sim - self.window_start_sim:.3f} s",
            )
            return self.result

        if now_sim - self.stop_sim > cfg.timeout_sim:
            if self.post_settle_count == 0:
                self.result = StopCheck(NO_FRESH_ODOM, "no odometry after stop")
            elif not fresh:
                self.result = StopCheck(NO_FRESH_ODOM, "latest odometry stale")
            else:
                self.result = StopCheck(
                    NOT_STOPPED, self.last_moving_detail or "window too short"
                )
            return self.result

        expired = self._mono_deadline(now_sim, now_mono)
        if expired is not None:
            return expired

        if self.latest_stamp_sim is not None and not fresh:
            return StopCheck(PENDING, "latest odometry stale")

        return StopCheck(PENDING, f"{self.window_count} stopped samples")

    def _mono_deadline(self, now_sim, now_mono):
        """monotonic 제한을 넘었으면 UNVERIFIABLE 로 확정한 결과, 아니면 None."""
        if (
            now_mono is None or not _finite(now_mono)
            or self.start_mono is None
        ):
            return None
        elapsed_mono = now_mono - self.start_mono
        if elapsed_mono <= self.config.timeout_mono:
            return None
        if _finite(now_sim):
            sim_text = f"sim elapsed {now_sim - self.stop_sim:.3f} s"
        else:
            sim_text = "no valid clock"
        self.result = StopCheck(
            UNVERIFIABLE,
            f"STOP_CHECK_DEADLINE: {elapsed_mono:.3f} s monotonic, {sim_text} "
            f"(clock stalled or too slow), post-settle samples "
            f"{self.post_settle_count}, stopped window {self.window_count}",
        )
        return self.result
