"""Selective Hybrid applied-settings record (DEV): values are read back from
the real component objects the bridge builds. No solver is run (a stand-in
``cbs.py`` only satisfies the adapters' existence check)."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.deadlock_monitor import DeadlockMonitor
from src.compare import common_runner as cr
from src.compare import selective_bridge as sb
from src.compare.common_runner import DEV_SETTINGS, SELECTIVE_HYBRID_DEV
from src.hybrid.partial_trigger import PartialTriggerConfig, PartialTriggerDetector, PartialTriggerShadow
from src.hybrid.selective_safety import TriggerBudget

REARM = {"max_cbs_calls": 2, "cooldown_steps": 2, "cbs_time_budget_ms": 60000.0}


class _Case(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / "cbs.py").write_text("# stand-in, never executed\n")

    def build(self, *, partial_k=sb.PARTIAL_TRIGGER_K, total=30000.0, attempts=3,
              levels=None, reserve=sb.RESERVE_MS):
        monitor = DeadlockMonitor()
        if partial_k is not None:
            monitor = PartialTriggerShadow(
                monitor, {f"K{partial_k}": PartialTriggerDetector(PartialTriggerConfig(partial_k, partial_k))})
        budget = TriggerBudget(total_time_ms=total, max_attempts=attempts,
                               level_timeout_ms=dict(levels or sb.LEVEL_TIMEOUT_MS))
        _, planner, _ = sb.build_selective_dev_components(
            solver_root=self.root, work_dir=self.root / "w", monitor=monitor, budget=budget, reserve_ms=reserve)
        return planner

    @staticmethod
    def recorded():
        rec = dict(DEV_SETTINGS[SELECTIVE_HYBRID_DEV])
        rec.update(sb.selective_extra_settings())
        return rec

    @staticmethod
    def raw(h_exec=3, rearm=REARM):
        raw = {"h_exec_applied": h_exec}
        if rearm is not None:
            raw["rearm_config"] = dict(rearm)
        return raw


class TestAppliedSettings(_Case):

    def test_values_read_from_built_objects(self):
        applied = sb.selective_applied_settings(self.build(), self.raw())
        self.assertEqual(applied, {
            "h_exec": 3,
            "ladder": ["L0", "L1", "G"],
            "total_budget_ms": 30000.0,
            "max_attempts": 3,
            "level_timeout_ms": {"L0": 10000.0, "L1": 10000.0, "G": 20000.0},
            "reserve_ms": {"subset": 500.0, "global": 500.0},
            "acknowledge_unverified_outside_policy": True,
            "partial_trigger_detectors": {"K2": {"k_conflict": 2, "k_no_progress": 2}},
            "deadlock_monitor": {"class": "DeadlockMonitor",
                                 "config": {"no_progress_window": None, "k_label_window": None}},
            "rearm": REARM,
            "unobserved": [],
        })

    def test_object_values_not_module_constants(self):
        applied = sb.selective_applied_settings(
            self.build(partial_k=4, total=1234.0, attempts=1, reserve=7.0), self.raw(h_exec=5))
        self.assertEqual((applied["h_exec"], applied["total_budget_ms"], applied["max_attempts"]), (5, 1234.0, 1))
        self.assertEqual(applied["reserve_ms"], {"subset": 7.0, "global": 7.0})
        self.assertEqual(applied["partial_trigger_detectors"], {"K4": {"k_conflict": 4, "k_no_progress": 4}})

    def test_unreadable_values_listed_not_filled(self):
        applied = sb.selective_applied_settings(self.build(partial_k=None), {})
        self.assertEqual(applied["unobserved"], ["h_exec", "partial_trigger_detectors", "rearm"])
        for key in applied["unobserved"]:
            self.assertNotIn(key, applied)
        self.assertEqual(applied["deadlock_monitor"]["class"], "DeadlockMonitor")   # no PartialTriggerShadow layer

    def test_record_is_json_serialisable(self):
        import json
        json.dumps(sb.selective_applied_settings(self.build(), self.raw()), sort_keys=True)


class TestRecordedVersusApplied(_Case):

    def test_current_dev_settings_match_applied(self):
        applied = sb.selective_applied_settings(self.build(), self.raw())
        self.assertEqual(sb.check_selective_settings(self.recorded(), applied), [])

    def test_each_divergence_reported(self):
        applied = sb.selective_applied_settings(
            self.build(partial_k=3, total=9.0, attempts=2, reserve=1.0,
                       levels={"L0": 1.0, "L1": 1.0, "G": 2.0}),
            self.raw(h_exec=4))
        reasons = sb.check_selective_settings(self.recorded(), applied)
        for name in ("h_exec", "total_budget_ms", "max_attempts", "level_timeout_ms",
                     "reserve_ms[subset]", "reserve_ms[global]", "partial_trigger_k"):
            self.assertTrue(any(r.startswith(name + ":") for r in reasons), (name, reasons))
        self.assertEqual(len(reasons), 7, reasons)

    def test_ladder_and_outside_policy_divergence_reported(self):
        applied = sb.selective_applied_settings(self.build(), self.raw())
        rec = self.recorded()
        rec["ladder"] = ["G"]
        rec["outside_wait_policy"] = "VERIFIED"
        reasons = sb.check_selective_settings(rec, applied)
        self.assertEqual([r.split(":")[0] for r in reasons], ["ladder", "outside_wait_policy"])

    def test_unobserved_keys_are_not_compared(self):
        applied = sb.selective_applied_settings(self.build(), {})
        self.assertEqual(sb.check_selective_settings(self.recorded(), applied), [])


class TestDevSettingsUnchanged(unittest.TestCase):

    def test_selective_static_settings_and_extras(self):
        self.assertEqual(DEV_SETTINGS[SELECTIVE_HYBRID_DEV]["ladder"], ["L0", "L1", "G"])
        self.assertNotIn("applied", DEV_SETTINGS[SELECTIVE_HYBRID_DEV])
        self.assertEqual(sb.selective_extra_settings(),
                         {"partial_trigger_k": 2, "level_timeout_ms": {"L0": 10000.0, "L1": 10000.0, "G": 20000.0},
                          "reserve_ms": 500.0})


class TestMethodDevSettings(_Case):
    """Stage 3: Common Runner validates Selective settings before running."""

    def test_default_methods_unchanged_copies(self):
        from src.compare.common_runner import METHODS, method_dev_settings
        for method in METHODS + (cr.GLOBAL_HYBRID_DEV_H3_REARM,):
            with self.subTest(method):
                got = method_dev_settings(method)
                self.assertEqual(got, DEV_SETTINGS[method])
                self.assertIsNot(got, DEV_SETTINGS[method])

    def test_selective_settings_are_static_plus_bridge_extras(self):
        got = cr.method_dev_settings(SELECTIVE_HYBRID_DEV)
        self.assertEqual(got, self.recorded())
        self.assertEqual(got["h_exec"], cr.H_EXEC_DEV)
        self.assertEqual(got["h_exec"], 3)
        self.assertNotIn("applied", got)
        self.assertNotIn("partial_trigger_k", DEV_SETTINGS[SELECTIVE_HYBRID_DEV])   # caller's dict not mutated

    def test_planned_settings_match_what_built_objects_apply(self):
        planned = sb.selective_planned_settings(h_exec=3, total_budget_ms=30000.0, max_attempts=3)
        applied = sb.selective_applied_settings(self.build(), self.raw())
        for key, value in planned.items():
            self.assertEqual(applied[key], value, key)

    def test_missing_setting_refused_not_defaulted(self):
        for key in sb.REQUIRED_SETTINGS:
            if key in sb.selective_extra_settings():
                continue
            with self.subTest(key), mock.patch.dict(DEV_SETTINGS[SELECTIVE_HYBRID_DEV]):
                del DEV_SETTINGS[SELECTIVE_HYBRID_DEV][key]
                with self.assertRaisesRegex(ValueError, rf"missing \['{key}'\]"):
                    cr.method_dev_settings(SELECTIVE_HYBRID_DEV)
        extras = {**sb.selective_extra_settings(), "reserve_ms": None}
        with mock.patch.object(sb, "selective_extra_settings", return_value=extras):
            with self.assertRaisesRegex(ValueError, r"missing \['reserve_ms'\]"):
                cr.method_dev_settings(SELECTIVE_HYBRID_DEV)

    def test_recorded_ladder_differing_from_bridge_refused(self):
        with mock.patch.dict(DEV_SETTINGS[SELECTIVE_HYBRID_DEV], ladder=["L0", "G"]):
            with self.assertRaisesRegex(ValueError, r"ladder: recorded \['L0', 'G'\]"):
                cr.method_dev_settings(SELECTIVE_HYBRID_DEV)

    def test_recorded_extras_differing_from_bridge_refused(self):
        cases = {"partial_trigger_k": 3, "reserve_ms": 400.0,
                 "level_timeout_ms": {"L0": 1.0, "L1": 1.0, "G": 2.0}}
        for key, value in cases.items():
            extras = {**sb.selective_extra_settings(), key: value}
            with self.subTest(key), mock.patch.object(sb, "selective_extra_settings", return_value=extras):
                with self.assertRaisesRegex(ValueError, key):
                    cr.method_dev_settings(SELECTIVE_HYBRID_DEV)

    def test_invalid_values_refused_by_the_config_objects(self):
        for key, value in {"h_exec": 0, "total_budget_ms": -1.0, "max_attempts": 0}.items():
            with self.subTest(key), mock.patch.dict(DEV_SETTINGS[SELECTIVE_HYBRID_DEV], {key: value}):
                with self.assertRaisesRegex(ValueError, key.replace("total_budget_ms", "total_time_ms")):
                    cr.method_dev_settings(SELECTIVE_HYBRID_DEV)

    def test_run_method_refuses_before_creating_a_run_dir(self):
        out = self.root / "out"
        spec = cr.RunSpec(method=SELECTIVE_HYBRID_DEV, scenario=self.root / "never_read.yaml",
                          out_dir=out, solver_root=self.root)
        with mock.patch.dict(DEV_SETTINGS[SELECTIVE_HYBRID_DEV], ladder=["G"]):
            with self.assertRaisesRegex(ValueError, "ladder"):
                cr.run_method(spec)
        self.assertFalse(out.exists())


class TestSelectiveImportedOnlyWhenNamed(unittest.TestCase):

    def test_default_methods_do_not_import_selective_modules(self):
        code = (
            "import sys\n"
            "from src.compare import common_runner as cr\n"
            "for m in cr.METHODS + (cr.GLOBAL_HYBRID_DEV_H3_REARM,):\n"
            "    cr.method_dev_settings(m)\n"
            "bad = [n for n in ('src.compare.selective_bridge', 'src.hybrid.selective_planner',\n"
            "                   'src.hybrid.subset_cbs_adapter', 'src.hybrid.partial_trigger') if n in sys.modules]\n"
            "print(bad)\n"
            "cr.method_dev_settings(cr.SELECTIVE_HYBRID_DEV)\n"
            "print('src.compare.selective_bridge' in sys.modules)\n"
        )
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.split(), ["[]", "True"], out.stderr)


if __name__ == "__main__":
    unittest.main()
