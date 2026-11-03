"""scripts/run_compare_batch.py: refusals before running, and a small batch.

The small batch is Corridor x 5 methods with the recorded atb033 output
injected below the real CbsRuntimePlanner / PathGuardValidator (no solver
runs). Git state is injected as clean/dirty because the test tree is usually
uncommitted. Everything is written into a temporary directory.
"""

import contextlib
import csv
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_compare_batch import main
from scripts.run_il import DEFAULT_CKPT
from src.compare.common_runner import METHODS
from tests.test_common_runner import CORRIDOR, recorded_solver

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

CLEAN = {"commit": "c0ffee", "dirty": False}
DIRTY = {"commit": "c0ffee", "dirty": True}


class _Case(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.solver_root = self.tmp / "solver"
        self.solver_root.mkdir()
        (self.solver_root / "cbs.py").write_text("raise SystemExit('fake solver must not run')\n",
                                                 encoding="utf-8")
        self.out = self.tmp / "compare_test"

    def tearDown(self):
        self._tmp.cleanup()

    def run_cli(self, *extra, git=CLEAN, git_fn=None):
        argv = ["--solver-root", str(self.solver_root), "--scenario", str(CORRIDOR),
                "--out-dir", str(self.out), *extra]
        buf = io.StringIO()
        git_patch = (patch("src.compare.common_runner._git", side_effect=git_fn) if git_fn
                     else patch("src.compare.common_runner._git", return_value=git))
        with git_patch, \
                patch("src.cbs_runtime_adapter.replan_with_status", side_effect=recorded_solver), \
                contextlib.redirect_stdout(buf):
            code = main(argv)
        return code, buf.getvalue()


class TestRefusedBeforeRunning(_Case):

    def test_unimplemented_method(self):
        code, out = self.run_cli("--method", "IL_ONLY", "--method", "SELECTIVE_WFG")
        self.assertEqual(code, 2)
        self.assertIn("SELECTIVE_WFG is not implemented", out)
        self.assertFalse(self.out.exists())

    def test_unknown_and_duplicate_method(self):
        code, out = self.run_cli("--method", "IL_ONLY", "--method", "IL_ONLY", "--method", "PURE_CBS")
        self.assertEqual(code, 2)
        self.assertIn("duplicate method", out)
        self.assertIn("unknown method 'PURE_CBS'", out)

    def test_dirty_project(self):
        code, out = self.run_cli(git=DIRTY)
        self.assertEqual(code, 2)
        self.assertIn("dirty", out)
        self.assertFalse(self.out.exists())

    def test_existing_output_dir(self):
        self.out.mkdir()
        (self.out / "keep.txt").write_text("SENTINEL\n", encoding="utf-8")
        code, out = self.run_cli()
        self.assertEqual(code, 2)
        self.assertEqual((self.out / "keep.txt").read_text(encoding="utf-8"), "SENTINEL\n")
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ["keep.txt"])

    def test_checkpoint_not_in_manifest(self):
        other = self.tmp / "ckpt"
        other.mkdir()
        shutil.copy(DEFAULT_CKPT, other / "cnn_navhint.pt")
        code, out = self.run_cli("--ckpt", str(other / "cnn_navhint.pt"))
        self.assertEqual(code, 2)
        self.assertIn("checkpoint check failed", out)


@unittest.skipIf(torch is None, "torch is not installed")
class TestSmallBatch(_Case):

    def test_corridor_five_methods(self):
        code, out = self.run_cli()
        self.assertEqual(code, 0, out)
        runs = sorted(p.name for p in (self.out / "runs").iterdir())
        self.assertEqual(len(runs), 5)
        with (self.out / "compare_episodes.csv").open(newline="", encoding="utf-8") as fh:
            episodes = list(csv.DictReader(fh))
        with (self.out / "compare_cbs_calls.csv").open(newline="", encoding="utf-8") as fh:
            calls = list(csv.DictReader(fh))
        self.assertEqual([e["method"] for e in episodes], list(METHODS))
        self.assertEqual({e["max_steps"] for e in episodes}, {"160"})
        self.assertEqual({e["batch_id"] for e in episodes}, {"compare_test"})
        kinds = {c["method"]: (c["cbs_request_kind"], c["request_timestep"]) for c in calls}
        self.assertEqual(kinds, {"PURE_CBS_DEV_SINGLE_PLAN": ("INITIAL_STATE", "0"),
                                 "GLOBAL_HYBRID_DEV_FULL": ("IL_STALL_TRIGGER", "5"),
                                 "GLOBAL_HYBRID_DEV_H3": ("IL_STALL_TRIGGER", "5")})
        manifest = json.loads((self.out / "batch_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual((manifest["fair_batch"], manifest["fairness_reasons"]), (True, []))
        self.assertEqual(manifest["run_order"], [e["run_id"] for e in episodes])
        self.assertIn("Development", manifest["data_role"])
        self.assertGreater(manifest["csv_vs_runner_checks"], 0)

    def test_non_standard_max_steps_is_unfair(self):
        code, out = self.run_cli("--max-steps", "20", "--method", "IL_ONLY", "--method",
                                 "PURE_CBS_DEV_SINGLE_PLAN")
        self.assertEqual(code, 1)
        manifest = json.loads((self.out / "batch_manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(manifest["fair_batch"])
        self.assertTrue(any("max_steps is not 160" in r for r in manifest["fairness_reasons"]))

    def test_allow_dirty_runs_but_is_unfair(self):
        from scripts.run_compare_batch import PROJECT_ROOT

        def project_dirty_only(path):   # the solver clone stays clean
            return DIRTY if Path(path).resolve() == PROJECT_ROOT.resolve() else CLEAN

        code, _ = self.run_cli("--allow-dirty", "--method", "IL_ONLY", git=None, git_fn=project_dirty_only)
        self.assertEqual(code, 1)
        manifest = json.loads((self.out / "batch_manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(any("dirty" in r for r in manifest["fairness_reasons"]))


if __name__ == "__main__":
    unittest.main()
