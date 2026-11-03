"""scripts/render_trajectory.py: validated HTML animations of saved runs.

Runs are produced by common_runner.run_method on the real Corridor scenario
(real checkpoint; CBS methods with the recorded atb033 output injected below
the real CbsRuntimePlanner / PathGuardValidator) into a temporary directory.
The last class reads the saved Development batch read-only (skipped when it
is not present).
"""

import contextlib
import hashlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.render_trajectory import RenderError, load_run_data, main, render_html, render_runs
from scripts.run_il import PROJECT_ROOT
from src.compare.common_runner import (
    GLOBAL_HYBRID_DEV_H3,
    IL_ONLY,
    PURE_CBS_DEV_SINGLE_PLAN,
    SAFETY_ONLY_DEV_K2,
    RunSpec,
    run_method,
)
from tests.test_common_runner import CORRIDOR, _OverlapAtT1, recorded_solver

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

BATCH = PROJECT_ROOT / "outputs" / "logs" / "compare_20261103_215010"
PRIORITY = ("corridor_bay_n2_PURE_CBS_DEV_SINGLE_PLAN", "corridor_bay_n2_GLOBAL_HYBRID_DEV_FULL",
            "corridor_bay_n2_GLOBAL_HYBRID_DEV_H3", "bottleneck_n4_GLOBAL_HYBRID_DEV_H3")


def embedded(html_text):
    m = re.search(r'<script type="application/json" id="run-data">(.*?)</script>', html_text, re.S)
    return json.loads(m.group(1))


class _Case(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.solver_root = self.tmp / "solver"
        self.solver_root.mkdir()
        (self.solver_root / "cbs.py").write_text("raise SystemExit('fake solver must not run')\n",
                                                 encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def run_dir(self, method, **kw):
        # A fresh parent per run: run_ids have 1 s resolution and run_method never reuses a dir.
        self._n = getattr(self, "_n", 0) + 1
        spec = RunSpec(method=method, scenario=CORRIDOR, out_dir=self.tmp / f"runs_{self._n}",
                       solver_root=self.solver_root, **kw)
        with patch("src.cbs_runtime_adapter.replan_with_status", side_effect=recorded_solver):
            return Path(run_method(spec).run_dir)

    def edit(self, run_dir, change):
        path = run_dir / "result.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        change(result)
        path.write_text(json.dumps(result), encoding="utf-8")


@unittest.skipIf(torch is None, "torch is not installed")
class TestRunData(_Case):

    def test_pure_cbs_preserved(self):
        run_dir = self.run_dir(PURE_CBS_DEV_SINGLE_PLAN)
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        d = load_run_data(run_dir)
        self.assertEqual(d["agents"], [0, 1])
        self.assertEqual(d["positions"], [[s[str(a)] for a in (0, 1)] for s in result["trajectory"]])
        self.assertEqual((d["starts"], d["goals"]), ([[2, 0], [2, 10]], [[2, 10], [2, 0]]))
        self.assertEqual((d["steps"], d["termination"], d["all_at_goal"]), (12, "SUCCESS", True))
        self.assertEqual(set(d["sources"]), {"CBS"})
        self.assertEqual(d["at_goal"][-1], [True, True])
        self.assertEqual(d["at_goal"][0], [False, False])
        # Step 5: agent 1 waits at (2,6) while agent 0 enters (2,5) (recorded plan).
        self.assertEqual((d["actions"][4], d["kinds"][4]), (["RIGHT", "WAIT"], ["MOVE", "WAIT"]))
        for step_actions, step_kinds in zip(d["actions"], d["kinds"]):
            for a, k in zip(step_actions, step_kinds):
                self.assertEqual(k == "WAIT", a == "WAIT")
        self.assertEqual([(e["t"], e["kind"]) for e in d["events"]], [(0, "CBS"), (12, "END")])
        self.assertFalse(any(d["overlap"]))

    def test_global_h3_events_and_sources(self):
        d = load_run_data(self.run_dir(GLOBAL_HYBRID_DEV_H3))
        self.assertEqual(d["sources"], ["IL"] * 5 + ["CBS"] * 3 + ["IL"] * 5)
        self.assertEqual([(e["t"], e["kind"]) for e in d["events"]],
                         [(5, "TRIGGER"), (5, "CBS"), (8, "RESUME"), (13, "END")])   # causal order
        self.assertTrue(d["is_dev_setting"])

    def test_safety_holds_and_monitor_trigger(self):
        d = load_run_data(self.run_dir(SAFETY_ONLY_DEV_K2))
        self.assertEqual((d["sources"][5], d["sources"][6]), ("SAFETY", "SAFETY"))
        kinds = [(e["t"], e["kind"]) for e in d["events"]]
        self.assertIn((5, "MONITOR"), kinds)
        self.assertIn((6, "SAFETY"), kinds)
        self.assertIn((7, "SAFETY"), kinds)
        self.assertEqual(d["termination"], "TIMEOUT")

    def test_il_only_is_not_a_dev_setting(self):
        d = load_run_data(self.run_dir(IL_ONLY, max_steps=8))
        self.assertFalse(d["is_dev_setting"])
        self.assertEqual(set(d["sources"]), {"IL"})


@unittest.skipIf(torch is None, "torch is not installed")
class TestRefusesInconsistentRuns(_Case):

    def test_position_change_must_match_action(self):
        # Injected teleport at t=1: the recorded action cannot explain it.
        with patch("scripts.run_il.MAPFStepSimulator", _OverlapAtT1):
            run_dir = self.run_dir(IL_ONLY, max_steps=4)
        with self.assertRaisesRegex(RenderError, "is not action"):
            load_run_data(run_dir)

    def test_tampered_inputs(self):
        cases = {
            "grid_sha256": (lambda r: r.update(grid_sha256="0" * 64), "grid differs"),
            "start": (lambda r: r["trajectory"][0].update({"0": [2, 1]}), "not the start state"),
            "wall": (lambda r: r["trajectory"][3].update({"0": [0, 0]}), "on a wall"),
            "goal flag": (lambda r: r.update(all_at_goal=False), "all_at_goal disagrees"),
            "missing": (lambda r: r.pop("action_sources"), "missing field"),
            "source": (lambda r: r["action_sources"].__setitem__(0, "MAGIC"), "unknown action source"),
        }
        for name, (change, message) in cases.items():
            with self.subTest(name):
                run_dir = self.run_dir(PURE_CBS_DEV_SINGLE_PLAN)
                self.edit(run_dir, change)
                with self.assertRaisesRegex(RenderError, message):
                    load_run_data(run_dir)


@unittest.skipIf(torch is None, "torch is not installed")
class TestHtmlOutput(_Case):

    def test_html_embeds_exact_data_and_loads_nothing_external(self):
        run_dir = self.run_dir(PURE_CBS_DEV_SINGLE_PLAN)
        data = load_run_data(run_dir)
        text = render_html(data)
        self.assertEqual(embedded(text), data)
        self.assertNotIn("<link", text)
        self.assertNotIn("<script src", text)
        self.assertEqual(set(re.findall(r"https?://[^'\"\s)]+", text)), {"http://www.w3.org/2000/svg"})
        for control in ('id="play"', 'id="slider"', 'id="timeline"', 'id="events"'):
            self.assertIn(control, text)

    def test_script_closing_tag_cannot_break_out(self):
        data = load_run_data(self.run_dir(PURE_CBS_DEV_SINGLE_PLAN))
        data["stop_reason"] = "</script><b>x</b>"
        self.assertEqual(embedded(render_html(data))["stop_reason"], "</script><b>x</b>")

    def test_render_runs_new_dir_only(self):
        dirs = [self.run_dir(PURE_CBS_DEV_SINGLE_PLAN), self.run_dir(GLOBAL_HYBRID_DEV_H3)]
        before = {p: hashlib.sha256((p / "result.json").read_bytes()).hexdigest() for p in dirs}
        out = self.tmp / "videos" / "render_test"
        written = render_runs(dirs, out)
        self.assertEqual(sorted(p.name for p in out.iterdir()),
                         sorted([p.name for p in written] + ["render_manifest.json"]))
        self.assertEqual({p: hashlib.sha256((p / "result.json").read_bytes()).hexdigest() for p in dirs}, before)
        manifest = json.loads((out / "render_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual([r["source_result_sha256"] for r in manifest["runs"]], list(before.values()))
        with self.assertRaises(FileExistsError):
            render_runs(dirs, out)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--run-dir", str(dirs[0]), "--out-dir", str(out)]), 2)

    def test_duplicate_or_invalid_run_writes_nothing(self):
        good = self.run_dir(PURE_CBS_DEV_SINGLE_PLAN)
        with self.assertRaisesRegex(RenderError, "duplicate run_id"):
            render_runs([good, good], self.tmp / "out1")
        self.assertFalse((self.tmp / "out1").exists())
        bad = self.run_dir(GLOBAL_HYBRID_DEV_H3)
        self.edit(bad, lambda r: r.update(grid_sha256="0" * 64))
        with self.assertRaises(RenderError):
            render_runs([good, bad], self.tmp / "out2")
        self.assertFalse((self.tmp / "out2").exists())


@unittest.skipUnless((BATCH / "runs").is_dir(), "saved Development batch not present")
class TestSavedBatch(unittest.TestCase):

    def test_all_fifteen_validate_and_priority_four_render(self):
        dirs = sorted((BATCH / "runs").iterdir())
        self.assertEqual(len(dirs), 15)
        before = {p: hashlib.sha256((p / "result.json").read_bytes()).hexdigest() for p in dirs}
        for p in dirs:
            load_run_data(p)
        priority = [next(p for p in dirs if p.name.endswith(name)) for name in PRIORITY]
        with tempfile.TemporaryDirectory() as tmp:
            written = render_runs(priority, Path(tmp) / "render")
            self.assertEqual(len(written), 4)
            h3 = embedded(written[3].read_text(encoding="utf-8"))
            self.assertEqual((h3["termination"], h3["steps"], h3["all_at_goal"]), ("TIMEOUT", 160, False))
        self.assertEqual({p: hashlib.sha256((p / "result.json").read_bytes()).hexdigest() for p in dirs}, before)


if __name__ == "__main__":
    unittest.main()
