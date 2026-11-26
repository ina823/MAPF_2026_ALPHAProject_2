import unittest
from dataclasses import replace

from src.ros_bridge import stop_check as sc
from src.ros_bridge.stop_check import StopVerifier
from src.ros_bridge.safety_config import (
    DEMO_1HZ_STOP_CHECK,
    MAPF_50HZ_STOP_CHECK,
)

CFG = DEMO_1HZ_STOP_CHECK  # settle 0.5, window 1.0, min 2, max_age 1.5, timeout 4.0


def started(stop_sim=10.0, cfg=CFG):
    v = StopVerifier(cfg)
    v.start(stop_sim)
    return v


class TestStopVerifier(unittest.TestCase):

    def test_not_started(self):
        v = StopVerifier(CFG)
        self.assertEqual(v.evaluate(1.0).state, sc.NOT_STARTED)
        v.add_sample(1.0, 0, 0, 0)
        self.assertEqual(v.evaluate(1.0).state, sc.NOT_STARTED)

    def test_confirmed_with_fresh_zero_twist(self):
        v = started()
        v.add_sample(11.0, 0.0, 0.0, 0.0)
        self.assertEqual(v.evaluate(11.0).state, sc.PENDING)
        v.add_sample(12.0, 0.001, 0.0, -0.001)
        result = v.evaluate(12.0)
        self.assertTrue(result.confirmed, result)

    def test_old_odometry_before_stop_is_not_used(self):
        v = started()
        # 정지 명령 이전/감속 구간 stamp: 판정에 쓰지 않음
        for stamp in (8.0, 9.0, 10.0, 10.5):
            v.add_sample(stamp, 0.0, 0.0, 0.0)
        self.assertEqual(v.post_settle_count, 0)
        self.assertEqual(v.evaluate(14.0).state, sc.PENDING)
        result = v.evaluate(14.1)
        self.assertEqual(result.state, sc.NO_FRESH_ODOM)
        self.assertFalse(result.confirmed)

    def test_duplicate_stamp_not_counted(self):
        v = started()
        v.add_sample(11.0, 0, 0, 0)
        v.add_sample(11.0, 0, 0, 0)
        v.add_sample(10.9, 0, 0, 0)
        self.assertEqual(v.window_count, 1)
        self.assertEqual(v.evaluate(12.0).state, sc.PENDING)

    def test_single_sample_not_enough(self):
        v = started()
        v.add_sample(11.0, 0, 0, 0)
        self.assertFalse(v.evaluate(11.5).confirmed)

    def test_short_window_not_enough(self):
        v = started()
        v.add_sample(11.0, 0, 0, 0)
        v.add_sample(11.5, 0, 0, 0)
        self.assertFalse(v.evaluate(11.5).confirmed)

    def test_still_moving_fails(self):
        v = started()
        for t in (11.0, 12.0, 13.0, 14.0):
            v.add_sample(t, 0.2, 0.0, 0.0)
        self.assertEqual(v.evaluate(14.0).state, sc.PENDING)
        result = v.evaluate(14.1)
        self.assertEqual(result.state, sc.NOT_STOPPED)
        self.assertTrue(result.failed)

    def test_rotating_fails(self):
        v = started()
        for t in (11.0, 12.0, 13.0, 14.0):
            v.add_sample(t, 0.0, 0.0, 0.1)
        self.assertEqual(v.evaluate(14.1).state, sc.NOT_STOPPED)

    def test_lateral_velocity_counted(self):
        v = started()
        for t in (11.0, 12.0):
            v.add_sample(t, 0.0, 0.05, 0.0)
        self.assertFalse(v.evaluate(12.0).confirmed)

    def test_motion_resets_window(self):
        v = started()
        v.add_sample(11.0, 0, 0, 0)
        v.add_sample(12.0, 0.1, 0, 0)
        v.add_sample(13.0, 0, 0, 0)
        self.assertFalse(v.evaluate(13.0).confirmed)
        v.add_sample(14.0, 0, 0, 0)
        self.assertTrue(v.evaluate(14.0).confirmed)

    def test_tolerance_boundary(self):
        v = started()
        v.add_sample(11.0, CFG.linear_tolerance, 0, CFG.angular_tolerance)
        v.add_sample(12.0, -CFG.linear_tolerance, 0, -CFG.angular_tolerance)
        self.assertTrue(v.evaluate(12.0).confirmed)
        v = started()
        v.add_sample(11.0, CFG.linear_tolerance * 1.01, 0, 0)
        v.add_sample(12.0, 0, 0, 0)
        self.assertFalse(v.evaluate(12.0).confirmed)

    def test_stale_latest_sample_rejected(self):
        v = started()
        v.add_sample(11.0, 0, 0, 0)
        v.add_sample(12.0, 0, 0, 0)
        # /clock 이 1.5 s 넘게 앞서면 정지 확인하지 않음
        result = v.evaluate(13.6)
        self.assertEqual(result.state, sc.PENDING)
        self.assertIn("stale", result.detail)
        self.assertEqual(v.evaluate(14.1).state, sc.NO_FRESH_ODOM)

    def test_missing_or_invalid_twist_is_not_stop(self):
        v = started()
        v.add_sample(11.0, None, None, None)
        v.add_sample(12.0, 0, 0, float("nan"))
        self.assertFalse(v.evaluate(12.0).confirmed)
        self.assertEqual(v.window_count, 0)

    def test_result_sticky_until_restart(self):
        v = started()
        v.add_sample(11.0, 0, 0, 0)
        v.add_sample(12.0, 0, 0, 0)
        self.assertTrue(v.evaluate(12.0).confirmed)
        v.add_sample(13.0, 0.3, 0, 0)
        self.assertTrue(v.evaluate(13.0).confirmed)
        v.start(20.0)
        self.assertEqual(v.evaluate(20.0).state, sc.PENDING)

    def test_cancel(self):
        v = started()
        v.cancel()
        self.assertEqual(v.evaluate(12.0).state, sc.NOT_STARTED)

    def test_invalid_start(self):
        v = StopVerifier(CFG)
        for bad in (None, float("nan"), -1.0):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    v.start(bad)

    def test_invalid_clock_pending(self):
        v = started()
        self.assertEqual(v.evaluate(None).state, sc.PENDING)

    def test_50hz(self):
        cfg = MAPF_50HZ_STOP_CHECK
        v = started(1.0, cfg)
        t = 1.0 + cfg.settle_sim
        for _ in range(12):
            t += 0.02
            v.add_sample(t, 0.0, 0.0, 0.0)
        self.assertTrue(v.evaluate(t).confirmed)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            replace(CFG, timeout_mono=CFG.timeout_sim - 0.01)
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(timeout_mono=bad):
                with self.assertRaises(ValueError):
                    replace(CFG, timeout_mono=bad)
        with self.assertRaises(ValueError):
            replace(CFG, min_samples=1)
        with self.assertRaises(ValueError):
            replace(CFG, timeout_sim=1.0)
        with self.assertRaises(ValueError):
            replace(CFG, linear_tolerance=0.0)



class TestStopVerifierMonotonicDeadline(unittest.TestCase):
    """/clock 이 멈추면 sim Timeout 에 도달하지 못한다. monotonic 제한으로 확정한다."""

    def test_frozen_clock_becomes_unverifiable(self):
        v = StopVerifier(CFG)
        v.start(10.0, 100.0)
        # /clock 이 10.0 에 멈춤: sim 기준으로는 영원히 PENDING
        result = v.evaluate(10.0, 100.0 + CFG.timeout_mono)
        self.assertEqual(result.state, sc.PENDING)
        self.assertFalse(result.terminal)
        result = v.evaluate(10.0, 100.0 + CFG.timeout_mono + 0.01)
        self.assertEqual(result.state, sc.UNVERIFIABLE)
        self.assertFalse(result.confirmed)
        self.assertTrue(result.failed)
        self.assertTrue(result.terminal)
        self.assertIn("STOP_CHECK_DEADLINE", result.detail)
        self.assertIn("sim elapsed 0.000", result.detail)

    def test_no_valid_clock_becomes_unverifiable(self):
        v = StopVerifier(CFG)
        v.start(10.0, 100.0)
        self.assertEqual(v.evaluate(None, 105.0).state, sc.PENDING)
        result = v.evaluate(float("nan"), 106.5)
        self.assertEqual(result.state, sc.UNVERIFIABLE)
        self.assertIn("no valid clock", result.detail)

    def test_lazy_arm_on_first_evaluate(self):
        v = StopVerifier(CFG)
        v.start(10.0)
        self.assertEqual(v.evaluate(10.0, 200.0).state, sc.PENDING)
        self.assertEqual(v.start_mono, 200.0)
        self.assertEqual(v.evaluate(10.0, 205.9).state, sc.PENDING)
        self.assertEqual(v.evaluate(10.0, 206.1).state, sc.UNVERIFIABLE)

    def test_without_mono_sim_only(self):
        # now_mono 를 주지 않으면 이전과 같이 sim 기준만 적용된다
        v = StopVerifier(CFG)
        v.start(10.0, 100.0)
        self.assertEqual(v.evaluate(10.0).state, sc.PENDING)
        self.assertEqual(v.evaluate(14.1).state, sc.NO_FRESH_ODOM)

    def test_unverifiable_is_sticky_after_clock_recovers(self):
        v = StopVerifier(CFG)
        v.start(10.0, 100.0)
        self.assertEqual(v.evaluate(10.0, 107.0).state, sc.UNVERIFIABLE)
        # Clock 복구 후 정지 샘플이 들어와도 결과를 바꾸지 않는다
        v.add_sample(11.0, 0, 0, 0)
        v.add_sample(12.0, 0, 0, 0)
        self.assertEqual(v.evaluate(12.0, 108.0).state, sc.UNVERIFIABLE)
        # 새 판정은 다시 시작해야 한다
        v.start(12.0, 108.0)
        self.assertEqual(v.evaluate(12.0, 108.0).state, sc.PENDING)

    def test_sim_verdict_takes_precedence(self):
        v = StopVerifier(CFG)
        v.start(10.0, 100.0)
        for t in (11.0, 12.0, 13.0, 14.0):
            v.add_sample(t, 0.2, 0.0, 0.0)
        self.assertEqual(v.evaluate(14.1, 107.0).state, sc.NOT_STOPPED)

        v = StopVerifier(CFG)
        v.start(10.0, 100.0)
        v.add_sample(11.0, 0, 0, 0)
        v.add_sample(12.0, 0, 0, 0)
        self.assertTrue(v.evaluate(12.0, 107.0).confirmed)

    def test_no_confirmation_from_samples_before_clock_freeze(self):
        # 정지 직후 /clock 이 멈추고 그 뒤 샘플이 없다: 확인 불가
        v = StopVerifier(CFG)
        v.start(10.0, 100.0)
        v.add_sample(10.4, 0, 0, 0)   # settle 구간
        for mono in (101.0, 103.0, 105.0, 106.0):
            self.assertEqual(v.evaluate(10.4, mono).state, sc.PENDING)
        self.assertEqual(v.evaluate(10.4, 106.01).state, sc.UNVERIFIABLE)

    def test_slow_sim_hits_monotonic_deadline(self):
        # 실시간 배율 0.5: mono 6 s 동안 sim 3 s → sim Timeout(4 s) 전 monotonic 제한
        v = StopVerifier(CFG)
        v.start(10.0, 100.0)
        result = None
        for i in range(1, 70):
            mono = 100.0 + i * 0.1
            result = v.evaluate(10.0 + i * 0.05, mono)
            if result.terminal:
                break
        self.assertEqual(result.state, sc.UNVERIFIABLE)
        self.assertGreater(mono - 100.0, CFG.timeout_mono)

    def test_invalid_start_mono(self):
        v = StopVerifier(CFG)
        for bad in (float("nan"), float("inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    v.start(10.0, bad)

    def test_cancel_clears_start_mono(self):
        v = StopVerifier(CFG)
        v.start(10.0, 100.0)
        v.cancel()
        self.assertIsNone(v.start_mono)
        self.assertEqual(v.evaluate(10.0, 1000.0).state, sc.NOT_STARTED)

    def test_failed_states(self):
        self.assertTrue(sc.StopCheck(sc.UNVERIFIABLE).failed)
        self.assertTrue(sc.StopCheck(sc.NO_FRESH_ODOM).failed)
        self.assertTrue(sc.StopCheck(sc.NOT_STOPPED).failed)
        self.assertFalse(sc.StopCheck(sc.PENDING).failed)
        self.assertFalse(sc.StopCheck(sc.PENDING).terminal)
        self.assertFalse(sc.StopCheck(sc.CONFIRMED).failed)
        self.assertTrue(sc.StopCheck(sc.CONFIRMED).terminal)


if __name__ == "__main__":
    unittest.main()
