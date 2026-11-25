"""batch_manifest.json records for SELECTIVE_HYBRID_DEV (DEV, step 4).

Unit part: ``selective_manifest_entry`` on applied settings read back from
real built Selective objects (no solver). CLI part: scripts/run_compare_batch
on the Corridor with a MOCK SOLVER -- default methods get the recorded atb033
plan through ``replan_with_status`` (as in tests/test_run_compare_batch), and
Selective through ``subprocess.run`` in src.cbs_adapter only (as in
tests/test_selective_real_e2e_dev). This checks the manifest wiring, not the
real solver.
"""

import copy
import json
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.compare import selective_bridge as sb
from src.compare.common_runner import (DEV_SETTINGS, METHODS, SELECTIVE_HYBRID_DEV, method_dev_settings)
from tests.test_run_compare_batch import CLEAN, _Case as _BatchCase
from tests.test_selective_settings_record import _Case as _BuildCase

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

_REAL_RUN = subprocess.run


def _cbs_only(command, *args, **kwargs):
    """MOCK SOLVER for the atb033 ``cbs.py`` call only. Patching src.cbs_adapter.subprocess.run
    replaces ``subprocess.run`` process-wide, so every other call (e.g. platform.platform()
    probing the processor) is passed to the real function."""
    from tests.test_selective_real_e2e_dev import _recorded_subprocess
    if isinstance(command, (list, tuple)) and len(command) > 3 and str(command[1]).endswith("cbs.py") \
            and "stdout" not in kwargs:
        return _recorded_subprocess(command, *args, **kwargs)
    return _REAL_RUN(command, *args, **kwargs)


REQUIRED_VIEW = ("h_exec", "partial_trigger_k", "ladder", "total_budget_ms", "max_attempts",
                 "level_timeout_ms", "reserve_ms", "detectors", "rearm_limits", "cooldown_steps",
                 "outside_wait_policy")


def _rec(run_id, applied):
    return SimpleNamespace(run_id=run_id, dev_settings={} if applied is None else {"applied": applied})


class TestManifestEntry(_BuildCase):

    def applied(self, **kw):
        return sb.selective_applied_settings(self.build(**kw), self.raw())

    def entry(self, *applied_list):
        return sb.selective_manifest_entry(method_dev_settings(SELECTIVE_HYBRID_DEV),
                                           [_rec(f"r{i}", a) for i, a in enumerate(applied_list)],
                                           batch_cbs_timeout_sec=30)

    def test_every_required_setting_recorded_from_applied_objects(self):
        e = self.entry(self.applied(), self.applied())
        self.assertEqual(e["applied_settings_mismatches"], [])
        self.assertEqual(e["applied_settings_runs"], ["r0", "r1"])
        self.assertNotIn("applied_settings_by_run", e)
        v = e["applied_settings"]
        self.assertTrue(set(REQUIRED_VIEW) <= set(v))
        self.assertEqual(v["unobserved"], [])
        self.assertEqual((v["h_exec"], v["partial_trigger_k"], v["ladder"]), (3, sb.PARTIAL_TRIGGER_K, ["L0", "L1", "G"]))
        self.assertEqual((v["total_budget_ms"], v["max_attempts"]), (30000.0, 3))
        self.assertEqual(v["level_timeout_ms"], sb.LEVEL_TIMEOUT_MS)
        self.assertEqual(v["reserve_ms"], {"subset": sb.RESERVE_MS, "global": sb.RESERVE_MS})
        self.assertEqual(v["detectors"]["partial_trigger"], {"K2": {"k_conflict": 2, "k_no_progress": 2}})
        self.assertEqual(v["detectors"]["deadlock_monitor"]["class"], "DeadlockMonitor")
        self.assertEqual(v["rearm_limits"], {"max_cbs_calls": 2, "cbs_time_budget_ms": 60000.0})
        self.assertEqual(v["cooldown_steps"], 2)
        self.assertEqual((v["outside_wait_policy"], v["acknowledge_unverified_outside_policy"]),
                         ("DEV_UNVERIFIED", True))
        json.dumps(e)

    def test_timeouts_kept_apart_from_cbs_timeout_sec(self):
        t = self.entry(self.applied())["timeouts"]
        self.assertEqual((t["batch_cbs_timeout_sec"], t["batch_cbs_timeout_sec_applied"]), (30, False))
        for word in ("level_timeout_ms", "total_budget_ms", "reserve_ms", "cbs_timeout_sec"):
            self.assertIn(word, t["note"])

    def test_divergence_from_recorded_reported(self):
        self.assertTrue(any("level_timeout_ms" in r for r in
                            self.entry(self.applied(levels={"L0": 1.0, "L1": 1.0, "G": 1.0}))
                            ["applied_settings_mismatches"]))
        self.assertTrue(any("partial_trigger_k" in r for r in
                            self.entry(self.applied(partial_k=3))["applied_settings_mismatches"]))

    def test_rearm_and_cooldown_divergence_reported_against_planned(self):
        a = self.applied()
        a["rearm"] = dict(a["rearm"], cooldown_steps=9)
        reasons = self.entry(a)["applied_settings_mismatches"]
        self.assertTrue(any(r.startswith("r0: rearm") for r in reasons), reasons)

    def test_unobserved_value_is_none_not_defaulted(self):
        planner = self.build()
        planner.subset_planner = None   # reserve unreadable
        a = sb.selective_applied_settings(planner, self.raw(rearm=None))
        v = self.entry(a)["applied_settings"]
        self.assertIsNone(v["reserve_ms"])
        self.assertIsNone(v["rearm_limits"])
        self.assertIsNone(v["cooldown_steps"])
        for key in ("reserve_ms", "rearm", "rearm_limits", "cooldown_steps"):
            self.assertIn(key, v["unobserved"])

    def test_runs_that_differ_are_listed_per_run(self):
        b = copy.deepcopy(self.applied())
        b["deadlock_monitor"]["config"]["no_progress_window"] = 99   # not in planned / recorded
        e = self.entry(self.applied(), b)
        self.assertIsNone(e["applied_settings"])
        self.assertEqual(set(e["applied_settings_by_run"]), {"r0", "r1"})
        self.assertIn("applied Selective settings differ between runs", e["applied_settings_mismatches"])

    def test_run_without_applied_settings_reported(self):
        e = self.entry(None)
        self.assertIsNone(e["applied_settings"])
        self.assertEqual(e["applied_settings_mismatches"], ["r0: no applied Selective settings recorded"])


@unittest.skipIf(torch is None, "torch is not installed")
class TestBatchManifest(_BatchCase):

    def selective_cli(self, *extra):
        with patch("src.cbs_adapter.subprocess.run", side_effect=_cbs_only):
            code, out = self.run_cli("--method", "GLOBAL_HYBRID_DEV_H3", "--method", SELECTIVE_HYBRID_DEV,
                                     *extra)
        return code, out, json.loads((self.out / "batch_manifest.json").read_text(encoding="utf-8"))

    def test_default_methods_manifest_unchanged(self):
        code, out = self.run_cli()
        self.assertEqual(code, 0, out)
        manifest = json.loads((self.out / "batch_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(list(manifest["methods"]), list(METHODS))
        for m, entry in manifest["methods"].items():
            self.assertEqual(entry, {"dev_settings": json.loads(json.dumps(DEV_SETTINGS[m])),
                                     "cbs_request_kind": entry["cbs_request_kind"]})
        self.assertEqual(list(manifest["cbs_request_timing"]), ["INITIAL_STATE", "IL_STALL_TRIGGER"])

    def test_selective_manifest_matches_each_run(self):
        code, out, manifest = self.selective_cli()
        self.assertEqual(code, 0, out)
        self.assertEqual((manifest["fair_batch"], manifest["fairness_reasons"]), (True, []))
        self.assertIn("Development", manifest["data_role"])
        timing = manifest["cbs_request_timing"]
        self.assertEqual(list(timing), ["INITIAL_STATE", "IL_STALL_TRIGGER", "SELECTIVE_TRIGGER"])
        self.assertEqual(manifest["methods"]["GLOBAL_HYBRID_DEV_H3"],
                         {"dev_settings": DEV_SETTINGS["GLOBAL_HYBRID_DEV_H3"], "cbs_request_kind": "IL_STALL_TRIGGER"})

        sel = manifest["methods"][SELECTIVE_HYBRID_DEV]
        self.assertEqual(sel["cbs_request_kind"], "SELECTIVE_TRIGGER")
        self.assertEqual(sel["dev_settings"], json.loads(json.dumps(method_dev_settings(SELECTIVE_HYBRID_DEV))))
        self.assertEqual(sel["applied_settings_mismatches"], [])
        self.assertEqual(sel["timeouts"]["batch_cbs_timeout_sec_applied"], False)
        self.assertEqual(manifest["conditions"]["cbs_timeout_sec"], 30)
        (run_id,) = sel["applied_settings_runs"]
        result = json.loads((self.out / "runs" / run_id / "result.json").read_text(encoding="utf-8"))
        applied = result["dev_settings"]["applied"]
        v = sel["applied_settings"]
        self.assertEqual(v, json.loads(json.dumps(sb.selective_manifest_settings(sel["dev_settings"], applied))))
        self.assertEqual({**v["rearm_limits"], "cooldown_steps": v["cooldown_steps"]}, applied["rearm"])
        self.assertEqual(v["detectors"]["partial_trigger"], applied["partial_trigger_detectors"])
        self.assertEqual(v["h_exec"], 3)
        self.assertTrue(result["planner_calls"])
        for call in result["planner_calls"]:   # per-call limit is the ladder's, not cbs_timeout_sec
            self.assertLessEqual(call["timeout_ms"], v["level_timeout_ms"][call["level"]])
            self.assertNotEqual(call["timeout_sec"], 30)

    def test_settings_mismatch_makes_batch_unfair(self):
        with patch.object(sb, "_check_against_planned", return_value=["rearm: planned x != applied y"]):
            code, _, manifest = self.selective_cli()
        self.assertEqual(code, 1)
        self.assertFalse(manifest["fair_batch"])
        self.assertTrue(any(r.startswith("Selective settings:") and "rearm" in r
                            for r in manifest["fairness_reasons"]), manifest["fairness_reasons"])

    def test_dirty_tree_stays_unfair_with_selective(self):
        from pathlib import Path
        from scripts.run_compare_batch import PROJECT_ROOT
        from tests.test_run_compare_batch import DIRTY

        def project_dirty_only(path):
            return DIRTY if Path(path).resolve() == PROJECT_ROOT.resolve() else CLEAN

        with patch("src.cbs_adapter.subprocess.run", side_effect=_cbs_only):
            code, _ = self.run_cli("--allow-dirty", "--method", SELECTIVE_HYBRID_DEV, git=None,
                                   git_fn=project_dirty_only)
        manifest = json.loads((self.out / "batch_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 1)
        self.assertFalse(manifest["fair_batch"])
        self.assertTrue(any("dirty" in r for r in manifest["fairness_reasons"]))


if __name__ == "__main__":
    unittest.main()
