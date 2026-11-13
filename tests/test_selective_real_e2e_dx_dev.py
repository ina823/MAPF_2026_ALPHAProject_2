"""Tests for scripts/run_selective_real_e2e_dx_dev.py: scenario construction and the section-4
classification (synthetic records -- no solver, no IL). The real D1-D3 evidence is the script's
own run output, not these tests."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_selective_real_e2e_dx_dev as dx

CHECKS = ["1_trigger_in_real_IL", "2_wfg_uses_s_t_minus_1_and_intended", "3_cbs_start_is_s_t",
          "4_real_subset_cbs_at_L0", "5_real_subset_cbs_called_flag", "6_group_smaller_than_all",
          "7_outside_is_static_obstacle", "8_full_compose_path_guard", "9_outside_wait_every_exec_step",
          "10_expected_equals_actual_all_agents", "11_no_collision_no_unplanned_move", "12_il_resume_then_safe"]


def rec(outcome="EXECUTE:L0", termination="SUCCESS", stop_reason=None, generic="PASS", failed=(), trigger=True,
        cbs_steps=(2, 3, 4)):
    return {"transitions": [{"t": 2, "to": "TRIGGER"}] if trigger else [],
            "selective_records": [{"outcome": outcome}] if trigger else [],
            "generic_checks": {"verdict": generic},
            "partial_checks": {"checks": [{"name": n, "ok": n not in failed} for n in CHECKS]},
            "termination": termination, "stop_reason": stop_reason, "cbs_step_indices": list(cbs_steps)}


class TestClassify(unittest.TestCase):
    def test_classes(self):
        self.assertEqual(dx.classify(rec()), "SELECTIVE_E2E_SUCCESS")
        self.assertEqual(dx.classify(rec(termination="TIMEOUT", failed=("12_il_resume_then_safe",))),
                         "SELECTIVE_EXEC_THEN_EPISODE_TIMEOUT")
        self.assertEqual(dx.classify(rec(outcome="EXECUTE:G")), "L0L1_NOT_EXECUTED_THEN_GLOBAL:SUCCESS")
        self.assertEqual(dx.classify(rec(outcome="SAFE_STOP:LADDER_EXHAUSTED:CBS_TIMEOUT", termination="SAFE_STOP",
                                         stop_reason="CBS_TIMEOUT", cbs_steps=())), "SAFE_STOP:CBS_TIMEOUT")
        self.assertEqual(dx.classify(rec(trigger=False)), "NO_TRIGGER")

    def test_unsafe_or_incomplete_is_never_success(self):
        self.assertEqual(dx.classify(rec(generic="FAIL")), "FAIL")
        for item in CHECKS[:-1]:
            with self.subTest(item):
                self.assertEqual(dx.classify(rec(failed=(item,))), "FAIL")
        # a SAFE_STOP that still moved by CBS is not a clean stop
        self.assertEqual(dx.classify(rec(outcome="SAFE_STOP:X", termination="SAFE_STOP")), "FAIL")
        # a safe stop is never reported as reaching the goal
        self.assertNotIn("SUCCESS", dx.classify(rec(outcome="SAFE_STOP:X", termination="SAFE_STOP", cbs_steps=())))


class TestScenarios(unittest.TestCase):
    def test_originals_only_read_and_d1_adds_one_parked_agent(self):
        files = [dx.SCEN / "controlled" / "scenario_intersection_n2.json",
                 dx.SCEN / "pilot_v2" / "scenario_pilot2_staggered_blocks_9x9.json",
                 dx.SCEN / "pilot_v2" / "scenario_pilot2_loop_w2_7x9.json"]
        before = {p: p.read_bytes() for p in files}
        for n in ("D1", "D2", "D3"):
            grid, starts, goals, _ = dx.scenario(n)
            self.assertEqual(set(starts), set(goals))
            grid[0, 0] = 1
        self.assertEqual(before, {p: p.read_bytes() for p in files})
        _, s1, g1, _ = dx.scenario("D1")
        self.assertEqual((s1[2], g1[2], len(s1)), ((0, 3), (0, 3), 3))
        self.assertEqual(len(dx.scenario("D2")[1]), 4)
        self.assertEqual(len(dx.scenario("D3")[1]), 4)

    def test_refuses_without_solver_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.dict(os.environ, {"PYTHONHASHSEED": "0"}):
                self.assertEqual(dx.main(["--solver-root", str(Path(d) / "none"), "--out-root", d]), 2)
            self.assertEqual(list(Path(d).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
