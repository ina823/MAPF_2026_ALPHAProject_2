"""scripts/run_selective_shadow_dev.py -- no real CBS.

Only ``src.cbs_runtime_adapter.replan_with_status`` is replaced (recorded
atb033 Corridor plans from tests/test_common_runner.py, or a fixed status);
the fake solver root's cbs.py exits if it is ever run. Every output goes to a
TemporaryDirectory. The solver-git and fixed-seed preflight gates are turned
off only where the fake solver root makes them impossible; their refusal
paths are tested separately.
"""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_selective_shadow_dev as dev
from src.compare import provenance
from src.hybrid import selective_shadow as sh
from src.replan_status import ReplanResult
from tests.test_common_runner import CORRIDOR, _bad_start_solver, recorded_solver

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

SOLVER = "src.cbs_runtime_adapter.replan_with_status"


class _Env(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.solver_root = self.tmp / "solver"
        self.solver_root.mkdir()
        (self.solver_root / "cbs.py").write_text("raise SystemExit('fake solver must not run')\n",
                                                 encoding="utf-8")
        (self.solver_root / "a_star.py").write_text("", encoding="utf-8")
        self.out = self.tmp / "selective_shadow_dev"

    def tearDown(self):
        self._tmp.cleanup()

    def run_dev(self, solver=recorded_solver, **kw):
        kw.setdefault("require_solver_git", False)
        kw.setdefault("require_fixed_seed", False)
        kw.setdefault("run_id", "shadow_dev_test")
        with patch(SOLVER, side_effect=solver) as m:
            comp = dev.run_shadow_dev(solver_root=self.solver_root, out_root=self.out,
                                      scenario=CORRIDOR, **kw)
        return comp, m

    def all_files(self):
        return sorted(p for p in self.tmp.rglob("*") if p.is_file())


# -- PASS path ------------------------------------------------------------------------

@unittest.skipIf(torch is None, "torch is not installed")
class TestPass(_Env):

    def test_identical_global_and_shadow_recorded(self):
        comp, solver = self.run_dev()
        self.assertEqual((comp["verdict"], comp["exit_code"], comp["problems"]), (dev.PASS, 0, []))
        self.assertEqual(solver.call_count, 2)                     # OFF 1 + ON 1, nothing else
        for side in ("off", "on"):
            s = comp[side]
            self.assertEqual((s["termination"], s["steps"], s["planner_call_count"]), ("SUCCESS", 13, 1))
            self.assertEqual([(c["trigger_timestep"], c["status"], c["accepted"]) for c in s["cbs_calls"]],
                             [(5, "success", True)])
            self.assertEqual([c["ok"] for c in s["path_guard_recheck"]], [True])
        self.assertEqual(comp["off"]["cbs_calls"], comp["on"]["cbs_calls"])
        self.assertEqual(comp["latency_ms_reported_not_compared"], {"off": [50.0], "on": [50.0]})
        (rec,) = comp["shadow_records_summary"]
        self.assertEqual((rec["trigger_t"], rec["decision_reason"], rec["route"], rec["shadow_error"]),
                         (5, "FULL_GROUP", "GLOBAL", None))
        self.assertIs(comp["selective_group_cbs_executed"], False)

        run_dir = self.out / "shadow_dev_test"
        written = json.loads((run_dir / "comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(written["verdict"], dev.PASS)
        shadow = json.loads((run_dir / "shadow.json").read_text(encoding="utf-8"))
        self.assertEqual(len(shadow["records"]), 1)
        self.assertEqual(shadow["records"][0]["wfg"]["groups"], [[0, 1]])
        for side in ("off", "on"):
            for name in ("steps", "summary", "monitor"):
                self.assertTrue((run_dir / side / f"shadow_dev_test_{name}.csv").is_file())
        # Same CSV bytes OFF vs ON.
        for name in ("steps", "summary", "monitor"):
            self.assertEqual((run_dir / "off" / f"shadow_dev_test_{name}.csv").read_bytes(),
                             (run_dir / "on" / f"shadow_dev_test_{name}.csv").read_bytes())
        # Everything stayed inside the temp out_root (fake solver dir aside).
        for p in self.all_files():
            self.assertTrue(p.is_relative_to(self.out) or p.is_relative_to(self.solver_root), p)

    def test_latency_difference_is_reported_not_a_diff(self):
        n = {"i": 0}

        def drifting(**kw):
            n["i"] += 1
            r = recorded_solver(**kw)
            return ReplanResult(status=r.status, paths=r.paths, latency_sec=0.05 * n["i"], error=r.error)

        comp, _ = self.run_dev(solver=drifting)
        self.assertEqual(comp["verdict"], dev.PASS)
        self.assertEqual(comp["latency_ms_reported_not_compared"], {"off": [50.0], "on": [100.0]})


# -- detection: any Global change stops with a non-PASS verdict ----------------------------

@unittest.skipIf(torch is None, "torch is not installed")
class TestDetection(_Env):

    def test_planner_called_twice_is_global_diff(self):
        orig = sh.ShadowSelectivePlanner.__call__

        def twice(self, cur, goals, grid):
            orig(self, cur, goals, grid)
            return self.inner(cur, goals, grid)

        with patch.object(sh.ShadowSelectivePlanner, "__call__", twice):
            comp, solver = self.run_dev()
        self.assertEqual((comp["verdict"], comp["exit_code"]), (dev.DIFF, 1))
        self.assertIn("planner call count 1 != 2", comp["global_diffs"])

    def test_altered_planner_args_is_global_diff(self):
        orig = sh.ShadowSelectivePlanner.__call__

        def swapped(self, cur, goals, grid):
            cur = dict(cur)
            cur[0], cur[1] = cur[1], cur[0]
            return orig(self, cur, goals, grid)

        with patch.object(sh.ShadowSelectivePlanner, "__call__", swapped):
            comp, _ = self.run_dev()
        self.assertEqual(comp["verdict"], dev.DIFF)
        self.assertTrue(comp["global_diffs"])

    def test_shadow_internal_error_is_not_hidden(self):
        with patch("src.hybrid.selective_shadow.build_wfg", side_effect=RuntimeError("boom")):
            comp, _ = self.run_dev()
        self.assertEqual(comp["global_diffs"], [])                  # Global unchanged ...
        self.assertEqual(comp["verdict"], dev.SHADOW_RECORD_FAIL)   # ... but the run is not a PASS
        self.assertIn("record 1 shadow_error=EXCEPTION:RuntimeError:boom", comp["shadow_record_problems"])

    def test_off_safety_problem_stops_before_on(self):
        comp, solver = self.run_dev(solver=_bad_start_solver)
        self.assertEqual((comp["verdict"], comp["exit_code"]), (dev.STOPPED_OFF_SAFETY, 1))
        self.assertTrue(any("not accepted" in p for p in comp["problems"]))
        self.assertEqual(solver.call_count, 1)                      # ON never ran
        self.assertIsNone(comp["on"])
        run_dir = self.out / "shadow_dev_test"
        self.assertFalse((run_dir / "on").exists())
        self.assertFalse((run_dir / "shadow.json").exists())
        self.assertTrue((run_dir / "comparison.json").is_file())

    def test_wall_limit_after_off_stops_before_on(self):
        ticks = iter([0.0] + [1000.0] * 10)
        comp, solver = self.run_dev(wall_limit_sec=10, clock=lambda: next(ticks))
        self.assertEqual(comp["verdict"], dev.STOPPED_WALL_LIMIT)
        self.assertEqual(solver.call_count, 1)
        self.assertFalse((self.out / "shadow_dev_test" / "on").exists())


# -- refusals: nothing written, no solver call -----------------------------------------

class TestRefusals(_Env):

    def assert_refused(self, comp, solver, fragment):
        self.assertEqual((comp["verdict"], comp["exit_code"]), ("REFUSED", dev.EXIT_REFUSED))
        self.assertTrue(any(fragment in e for e in comp["errors"]), comp["errors"])
        self.assertEqual(solver.call_count, 0)
        self.assertFalse(self.out.exists())

    def test_cbs_timeout_must_stay_within_30s(self):
        for bad in (31, 0, True, 30.0):
            with self.subTest(bad):
                comp, m = self.run_dev(cbs_timeout_sec=bad)
                self.assert_refused(comp, m, "cbs_timeout_sec")

    def test_wall_limit_required(self):
        for bad in (0, -1, None):
            with self.subTest(bad):
                comp, m = self.run_dev(wall_limit_sec=bad)
                self.assert_refused(comp, m, "wall_limit_sec")

    def test_missing_solver_files(self):
        (self.solver_root / "a_star.py").unlink()
        comp, m = self.run_dev()
        self.assert_refused(comp, m, "a_star.py not found")

    def test_solver_must_be_clean_git_checkout(self):
        comp, m = self.run_dev(require_solver_git=True)
        self.assert_refused(comp, m, "not a git checkout")

    def test_hash_seed_must_be_fixed_at_startup(self):
        with patch.object(provenance, "STARTUP_ENV_VALUE", None):
            comp, m = self.run_dev(require_fixed_seed=True)
        self.assert_refused(comp, m, "PYTHONHASHSEED not fixed")

    def test_existing_run_dir_is_refused_and_kept(self):
        run_dir = self.out / "shadow_dev_test"
        run_dir.mkdir(parents=True)
        (run_dir / "comparison.json").write_text("SENTINEL\n", encoding="utf-8")
        comp, m = self.run_dev()
        self.assertEqual(comp["verdict"], "REFUSED")
        self.assertEqual(m.call_count, 0)
        self.assertEqual((run_dir / "comparison.json").read_text(encoding="utf-8"), "SENTINEL\n")

    def test_cli_refuses_timeout_over_30_and_writes_nothing(self):
        code = dev.main(["--solver-root", str(self.solver_root), "--cbs-timeout-sec", "31",
                         "--out-root", str(self.out), "--wall-limit-sec", "60"])
        self.assertEqual(code, dev.EXIT_REFUSED)
        self.assertFalse(self.out.exists())

    def test_cli_preflight_only_writes_nothing(self):
        with patch(SOLVER, side_effect=AssertionError("no CBS in preflight")):
            code = dev.main(["--solver-root", str(self.solver_root), "--preflight-only",
                             "--out-root", str(self.out)])
        self.assertEqual(code, dev.EXIT_REFUSED)            # fake root is not a git checkout
        self.assertFalse(self.out.exists())

    def test_defaults(self):
        self.assertEqual(dev.MAX_CBS_TIMEOUT_SEC, 30)
        self.assertEqual(dev.DEFAULT_OUT_ROOT, dev.PROJECT_ROOT / "outputs" / "selective_shadow_dev")
        self.assertEqual(dev.H_EXEC, 3)
        self.assertGreater(dev.DEFAULT_WALL_LIMIT_SEC, 2 * dev.MAX_CBS_TIMEOUT_SEC)


class TestWatchdog(_Env):

    def test_watchdog_ends_process_and_leaves_marker(self):
        (self.out / "rid").mkdir(parents=True)
        fired = threading.Event()
        with patch.object(dev.os, "_exit", side_effect=lambda code: fired.set()) as ex:
            timer = dev._arm_watchdog(0.05, self.out, "rid")
            self.assertTrue(fired.wait(5))
            timer.join(5)
        ex.assert_called_once_with(dev.EXIT_WALL)
        self.assertIn("wall limit", (self.out / "rid" / "ABORTED_WALL_LIMIT.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
