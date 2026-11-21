"""scripts/run_partial_trigger_p3_dev.py: refusal without a solver and the A-E verdict on SYNTHETIC
records (no solver, no IL). Real P3 evidence is only the script's own real run output."""

import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_partial_trigger_p3_dev as dev
from src.hybrid.partial_trigger import PARTIAL_TRIGGER_REASON


def rec_a():
    """Outside 2 moves t0->1, Partial Trigger at t=2, held 3 steps, IL moves it at t=6, all at goal."""
    T = [{0: (4, 0), 1: (4, 4), 2: (0, 0)}, {0: (4, 0), 1: (4, 4), 2: (0, 1)}, {0: (4, 0), 1: (4, 4), 2: (0, 1)},
         {0: (5, 0), 1: (4, 4), 2: (0, 1)}, {0: (5, 0), 1: (5, 4), 2: (0, 1)}, {0: (5, 0), 1: (5, 4), 2: (0, 1)},
         {0: (5, 0), 1: (5, 4), 2: (0, 2)}, {0: (5, 0), 1: (5, 4), 2: (0, 3)}]
    A = [{0: 4, 1: 4, 2: 3}, {0: 1, 1: 1, 2: 3}, {0: 1, 1: 4, 2: 4}, {0: 4, 1: 1, 2: 4}, {0: 4, 1: 4, 2: 4},
         {0: 4, 1: 4, 2: 3}, {0: 4, 1: 4, 2: 3}]
    att = {"level": "L0", "group": (0, 1), "planner_kind": "REAL", "solver_invoked": True, "raw_status": "success",
           "status": "success", "timeout_ms": 1.0, "latency_ms": 0.5, "budget_overrun_ms": None, "full_check_ok": True}
    return {
        "transitions": [{"t": 2, "to": "TRIGGER", "reason": PARTIAL_TRIGGER_REASON}, {"t": 2, "to": "CBS_REQUEST", "reason": ""},
                        {"t": 2, "to": "CBS_EXEC", "reason": ""}, {"t": 5, "to": "IL_RESUME", "reason": "EXEC_WINDOW_END"},
                        {"t": 5, "to": "IL", "reason": ""}],
        "trajectory": T, "actions": A, "action_sources": ["IL", "IL", "CBS", "CBS", "CBS", "IL", "IL"],
        "goals": {0: (5, 0), 1: (5, 4), 2: (0, 3)}, "n_agents": 3,
        "selective_records": [{"trigger_t": 2, "outcome": "EXECUTE:L0", "outside_held": (2,), "executed_group": (0, 1),
                               "wfg_groups": [[0, 1]], "levels": [], "attempts": [att],
                               "real_subset_cbs_called": True}],
        "partial_source": {"fired": [{"t": 2, "group": (0, 1)}], "refused": []},
        "adapter_calls": [{"planner": "subset", "called": True, "status": "success",
                           "solver_input_obstacles_xy": [[1, 0]]}],
        "validator_calls": [("SELECTIVE", (True, ""))], "runner_cbs_calls": [{"accepted": True}],
        "cbs_exec_trace": [{"match": True}] * 3, "safety_scan": {"vertex_overlap_steps": [], "edge_swaps": []},
        "integrity_abort": False, "termination": "SUCCESS", "stop_reason": None, "all_at_goal": True,
        "cbs_step_indices": [2, 3, 4], "runner_triggers": [{"t": 2, "reason": PARTIAL_TRIGGER_REASON,
                                                          "outcome": "CBS_REQUEST"}], "steps": 7,
    }


class TestVerdict(unittest.TestCase):
    def test_a(self):
        out = dev.p3_verdict(rec_a())
        self.assertEqual(out["class"], "A", out)
        self.assertEqual(out["info"]["outside_first_il_move_after_resume"], {2: 6})

    def test_full_first_or_no_trigger_is_d(self):
        r = rec_a()
        r["transitions"][0]["reason"] = "TRIGGER_FULL_REPEAT"
        self.assertEqual(dev.p3_verdict(r)["class"], "D")
        r = rec_a()
        r["transitions"] = []
        self.assertEqual(dev.p3_verdict(r)["class"], "D")

    def test_safe_stop_is_c_and_safety_failures_are_e(self):
        r = rec_a()
        r["selective_records"][0]["outcome"] = "SAFE_STOP:CBS_TIMEOUT"
        r["termination"], r["cbs_step_indices"] = "SAFE_STOP", []
        self.assertEqual(dev.p3_verdict(r)["class"], "C")
        for name, mutate in {
            "outside not WAIT": lambda r: r["actions"][3].__setitem__(2, 3),
            "obstacle not S_t": lambda r: r["adapter_calls"][0].__setitem__("solver_input_obstacles_xy", [[9, 9]]),
            "path guard": lambda r: r.__setitem__("validator_calls", [("SELECTIVE", (False, "x"))]),
            "mismatch": lambda r: r.__setitem__("cbs_exec_trace", [{"match": False}]),
            "mock planner": lambda r: r["selective_records"][0]["attempts"][0].__setitem__("planner_kind", "MOCK"),
        }.items():
            with self.subTest(name):
                r = copy.deepcopy(rec_a())
                mutate(r)
                self.assertEqual(dev.p3_verdict(r)["class"], "E")

    def test_no_resume_move_is_b(self):
        r = rec_a()
        r["trajectory"][6][2] = r["trajectory"][7][2] = (0, 1)
        r["termination"], r["all_at_goal"] = "TIMEOUT", False
        self.assertEqual(dev.p3_verdict(r)["class"], "B")


class TestScript(unittest.TestCase):
    def test_refuses_without_solver_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.dict(os.environ, {"PYTHONHASHSEED": "0"}):
                self.assertEqual(dev.main(["--solver-root", str(Path(d) / "none"), "--out-root", d]), 2)
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_dev_values_are_explicit(self):
        self.assertEqual((dev.K, dev.H_EXEC, dev.SCENARIOS), (2, 3, ("P3F", "D2")))


if __name__ == "__main__":
    unittest.main()
