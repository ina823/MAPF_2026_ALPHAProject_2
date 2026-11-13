"""Tests for scripts/run_selective_real_e2e_dev.py (the DEV real Selective E2E runner).

MOCK SOLVER here: only ``subprocess.run`` in src.cbs_adapter is replaced; it answers with
the recorded atb033 Corridor plan (tests.test_common_runner.GLOBAL_PLAN, agents 0/1 at S_5).
Real IL, real Simulator, real Runner, real SelectivePlanner/validators. This only checks
that the script's verdicts catch faults; it is NOT evidence that the real E2E passes
(that is the script's own real run).
"""

import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_selective_real_e2e_dev as dev

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class TestPreflight(unittest.TestCase):
    def test_refuses_without_solver_or_seed_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.dict(os.environ, {"PYTHONHASHSEED": "random"}):
                self.assertEqual(dev.main(["--solver-root", str(Path(d) / "nope"), "--out-root", d]), 2)
            self.assertEqual(list(Path(d).iterdir()), [])
            pre = dev.preflight(Path(d))
            self.assertTrue(any("cbs.py" in p for p in pre["problems"]))

    def test_original_scenarios_only_read(self):
        before = {p: p.read_bytes() for p in (dev.CORRIDOR, dev.BOTTLENECK)}
        for name in ("S1", "S2", "S1B", "S3", "CORRIDOR"):
            grid, starts, goals, _ = dev.scenario(name)
            grid[0, 0] = 1                                    # copies only
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        g1, s1, _, _ = dev.scenario("S1")
        self.assertEqual((s1[2], len(s1)), ((2, 8), 3))
        g, _, _, _ = dev.scenario("CORRIDOR")
        self.assertEqual(int(g[3, 1]), 1)                     # the S1B extra cell is not in the original


def _recorded_subprocess(command, cwd=None, capture_output=None, text=None, timeout=None, check=None):
    """Replaces ONLY subprocess.run in src.cbs_adapter: the rest of the chain (input YAML,
    coordinate conversion, id restoration, Path Guard) runs for real. Answers with the
    recorded atb033 plan when the solver input is agents 0/1 at S_5, else exits 1."""
    import subprocess
    import yaml
    from tests.test_common_runner import GLOBAL_PLAN
    inp = yaml.unsafe_load(Path(command[2]).read_text(encoding="utf-8"))
    starts = [tuple(a["start"]) for a in inp["agents"]]
    if starts != [(4, 2), (6, 2)]:                       # [x, y] of (2,4), (2,6)
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="no recorded plan")
    sched = {f"agent{i}": [{"t": t, "x": c, "y": r} for t, (r, c) in enumerate(GLOBAL_PLAN[i])] for i in (0, 1)}
    Path(command[3]).write_text(yaml.safe_dump({"cost": 0, "schedule": sched}), encoding="utf-8")
    return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


@unittest.skipIf(torch is None, "torch is not installed")
class TestVerdictsCatchFaults(unittest.TestCase):
    """S1B with the recorded answer -> partial verdict PASS; each tampering must FAIL it."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        solver = Path(cls._tmp.name) / "solver"
        solver.mkdir()
        (solver / "cbs.py").write_text("raise SystemExit('must not run')\n", encoding="utf-8")
        with patch("src.cbs_adapter.subprocess.run", _recorded_subprocess):
            run = dev.run_selective("S1B", solver, Path(cls._tmp.name) / "run")
        cls.rec = dev.record(run)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_untampered_passes(self):
        self.assertEqual(dev.partial_checks(self.rec)["verdict"], "PASS", dev.partial_checks(self.rec))
        rec = copy.deepcopy(self.rec)
        rec["generic_checks"] = dev.generic_checks(rec)               # as main() does before fallback_checks
        self.assertEqual(rec["generic_checks"]["verdict"], "PASS")
        self.assertEqual(dev.fallback_checks(rec)["verdict"], "PASS")
        self.assertTrue(dev.fallback_checks(rec)["partial_group_replan_observed"])

    def _fails(self, mutate, check):
        rec = copy.deepcopy(self.rec)
        mutate(rec)
        out = dev.partial_checks(rec)
        self.assertEqual(out["verdict"], "FAIL")
        self.assertFalse(next(c["ok"] for c in out["checks"] if c["name"] == check), out)

    def test_outside_not_waiting_is_caught(self):
        def m(rec):
            rec["exec_window"][0]["actions"][2] = 3
        self._fails(m, "9_outside_wait_every_exec_step")

        def moved(rec):
            rec["exec_window"][1]["moved"] = [2]
        self._fails(moved, "9_outside_wait_every_exec_step")

    def test_collision_and_mismatch_are_caught(self):
        def overlap(rec):
            rec["safety_scan"]["vertex_overlap_steps"] = [7]
        self._fails(overlap, "11_no_collision_no_unplanned_move")

        def mismatch(rec):
            rec["cbs_exec_trace"][0]["match"] = False
        self._fails(mismatch, "10_expected_equals_actual_all_agents")

    def test_wrong_start_mock_and_full_group_are_caught(self):
        def start(rec):
            c = next(c for c in rec["adapter_calls"] if c["planner"] == "subset")
            c["solver_input_agents"][0] = (c["solver_input_agents"][0][0], [9, 9], c["solver_input_agents"][0][2])
        self._fails(start, "3_cbs_start_is_s_t")

        def mock(rec):
            rec["selective_records"][0]["attempts"][0]["planner_kind"] = "MOCK"
        self._fails(mock, "4_real_subset_cbs_at_L0")

        def whole(rec):
            rec["selective_records"][0]["executed_group"] = (0, 1, 2)
        self._fails(whole, "6_group_smaller_than_all")

    def test_safety_scan_detects_swap_and_overlap(self):
        res = {"trajectory": [{0: (0, 0), 1: (0, 1)}, {0: (0, 1), 1: (0, 0)}, {0: (0, 1), 1: (0, 1)}]}
        self.assertEqual(dev.safety_scan(res), {"vertex_overlap_steps": [2], "edge_swaps": [(1, 0, 1)]})


if __name__ == "__main__":
    unittest.main()
