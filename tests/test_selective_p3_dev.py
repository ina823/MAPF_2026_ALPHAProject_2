"""Tests for scripts/run_selective_p3_dev.py -- the P3 A-E classifier on SYNTHETIC records (no
solver, no IL) and the scenario builders. Real P3 evidence is only the script's own real run."""

import copy
import unittest

import numpy as np

from scripts import run_selective_p3_dev as p3

ALL_OK = ["1_trigger_in_real_IL", "2_wfg_uses_s_t_minus_1_and_intended", "3_cbs_start_is_s_t",
          "4_real_subset_cbs_at_L0", "5_real_subset_cbs_called_flag", "6_group_smaller_than_all",
          "7_outside_is_static_obstacle", "8_full_compose_path_guard", "9_outside_wait_every_exec_step",
          "10_expected_equals_actual_all_agents", "11_no_collision_no_unplanned_move", "12_il_resume_then_safe"]


def p3_case():
    """Agent 2 (outside) moves at t=0->1, stands at (0,1) (goal (0,3)) when the Trigger fires at t=2,
    WAITs through the 3-step window, then IL moves it to (0,2), (0,3)."""
    g0, g1 = (5, 0), (5, 4)
    T = [{0: (4, 0), 1: (4, 4), 2: (0, 0)},
         {0: (4, 0), 1: (4, 4), 2: (0, 1)},
         {0: (4, 0), 1: (4, 4), 2: (0, 1)},           # t=2 Trigger, S_t
         {0: (5, 0), 1: (4, 4), 2: (0, 1)},
         {0: (5, 0), 1: (5, 4), 2: (0, 1)},
         {0: (5, 0), 1: (5, 4), 2: (0, 1)},           # t=5 IL_RESUME
         {0: (5, 0), 1: (5, 4), 2: (0, 2)},
         {0: (5, 0), 1: (5, 4), 2: (0, 3)}]
    A = [{0: 4, 1: 4, 2: 3}, {0: 1, 1: 1, 2: 3}, {0: 1, 1: 4, 2: 4}, {0: 4, 1: 1, 2: 4}, {0: 4, 1: 4, 2: 4},
         {0: 4, 1: 4, 2: 3}, {0: 4, 1: 4, 2: 3}]
    src = ["IL", "IL", "CBS", "CBS", "CBS", "IL", "IL"]
    trace = [{"t": k + 1, "expected": T[k + 1], "actual": T[k + 1], "match": True} for k in (2, 3, 4)]
    return {
        "transitions": [{"t": 2, "to": "TRIGGER"}, {"t": 2, "to": "CBS_REQUEST"}, {"t": 2, "to": "CBS_EXEC"},
                        {"t": 5, "to": "IL_RESUME", "reason": "EXEC_WINDOW_END"}, {"t": 5, "to": "IL"}],
        "trajectory": T, "actions": A, "action_sources": src,
        "goals": {0: g0, 1: g1, 2: (0, 3)},
        "selective_records": [{"trigger_t": 2, "outcome": "EXECUTE:L0", "outside_held": (2,),
                               "executed_group": (0, 1)}],
        "runner_cbs_calls": [{"accepted": True}],
        "generic_checks": {"verdict": "PASS", "first_failed_check": None}, "integrity_abort": False,
        "termination": "SUCCESS", "stop_reason": None, "all_at_goal": True, "cbs_step_indices": [2, 3, 4],
        "partial_checks": {"checks": [{"name": n, "ok": True} for n in ALL_OK]},
        "exec_window": [{"t_after": k + 1, "actions": A[k], "moved": []} for k in (2, 3, 4)],
        "adapter_calls": [{"planner": "subset", "called": True, "solver_input_obstacles_xy": [[1, 0]]}],
        "cbs_exec_trace": trace,
    }


class TestP3Classes(unittest.TestCase):
    def test_full_success_is_A(self):
        out = p3.p3_checks(p3_case())
        self.assertEqual(out["class"], "A", out)
        self.assertEqual(out["moving_outside"], [2])
        self.assertEqual(out["outside_first_move_after_resume"], {2: 6})
        self.assertEqual(out["outside_detail"][2]["moves_before_trigger"], 1)

    def test_no_resume_move_is_B_not_A(self):
        r = p3_case()
        for k in (5, 6):
            r["trajectory"][k + 1][2] = (0, 1)
        r["termination"], r["all_at_goal"] = "TIMEOUT", False
        self.assertEqual(p3.p3_checks(r)["class"], "B")

    def test_outside_at_goal_or_never_moved_is_D(self):
        r = p3_case()
        r["goals"][2] = (0, 1)
        self.assertEqual(p3.p3_checks(r)["class"], "D")
        r = p3_case()
        r["trajectory"][0][2] = (0, 1)                       # never moved before the Trigger
        self.assertEqual(p3.p3_checks(r)["class"], "D")

    def test_window_violations_are_E(self):
        for name, mutate in {
            "outside not WAIT": lambda r: r["exec_window"][1]["actions"].__setitem__(2, 3),
            "outside actual moved": lambda r: r["cbs_exec_trace"][1]["actual"].__setitem__(2, (0, 2)),
            "obstacle cell wrong": lambda r: r["adapter_calls"][0].__setitem__("solver_input_obstacles_xy", [[2, 0]]),
            "path guard": lambda r: r["partial_checks"]["checks"][7].__setitem__("ok", False),
            "generic": lambda r: r["generic_checks"].__setitem__("verdict", "FAIL"),
        }.items():
            with self.subTest(name):
                r = copy.deepcopy(p3_case())
                mutate(r)
                self.assertEqual(p3.p3_checks(r)["class"], "E")

    def test_safe_stop_is_C_and_no_trigger_or_global_is_D(self):
        r = p3_case()
        r["selective_records"][0]["outcome"] = "SAFE_STOP:LADDER_EXHAUSTED:CBS_TIMEOUT"
        r["termination"], r["cbs_step_indices"] = "SAFE_STOP", []
        self.assertEqual(p3.p3_checks(r)["class"], "C")
        r = p3_case()
        r["selective_records"][0]["outcome"] = "EXECUTE:G"
        self.assertEqual(p3.p3_checks(r)["class"], "D")
        r = p3_case()
        r["transitions"], r["selective_records"] = [], []
        self.assertEqual(p3.p3_checks(r)["class"], "D")


class TestScenarios(unittest.TestCase):
    def test_p3f_map_keeps_the_original_cross(self):
        orig = p3.scenario("D1")[0]
        g, s, gl, _ = p3.scenario("P3F")
        np.testing.assert_array_equal(g[:, :7], orig)
        self.assertTrue((g[:, 10] == 0).all() and (g[3, 7:] == 0).all())
        self.assertEqual((s[2], gl[2]), ((0, 10), (6, 10)))
        g, s, gl, _ = p3.scenario("P3Q")
        np.testing.assert_array_equal(g, orig)
        self.assertEqual((s[2], gl[2]), ((0, 3), (6, 3)))

    def test_detour_wrapper_produces_a_legal_detour_onto_the_cell(self):
        from src.cbs_runtime_adapter import CBSRuntimeResult
        grid = p3.scenario("D1")[0]
        real = lambda *a, **k: CBSRuntimeResult("success", {0: [(3, 2), (3, 3), (3, 4), (3, 5)],
                                                            1: [(2, 3), (2, 3), (2, 3), (3, 3), (4, 3), (5, 3)][:4]}, 0.1)
        w = p3.detour_wrapper(real, grid, (3, 6))
        r = w({0: (3, 2), 1: (2, 3)}, {0: (3, 5), 1: (4, 3)}, grid)
        self.assertIn((3, 6), r.paths[0])
        self.assertEqual(r.paths[0][-1], (3, 5))
        self.assertEqual(len({len(p) for p in r.paths.values()}), 1)
        for p in r.paths.values():
            self.assertTrue(all(abs(a[0] - b[0]) + abs(a[1] - b[1]) <= 1 for a, b in zip(p, p[1:])))


if __name__ == "__main__":
    unittest.main()
