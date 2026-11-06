"""scripts/register_dev_scenarios.py + scripts/check_registry.py on the real 16 files."""

import io
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np

from scripts import check_registry, register_dev_scenarios as rds
from src.data import registry as rg
from src.data.fingerprint import compute_fingerprints, grid_from_rows

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = PROJECT_ROOT / "registry"

EXPECTED_FAMILIES = {
    "fcorr01": {"scenario_corridor_bay_n2", "scenario_corridor_bay_n2_offset",
                "scenario_corridor_bay_n2_3agent", "scenario_corridor_bay_n2_4agent",
                "scenario_corridor_bay_n2_3rdagent_perturbation"},
    "fint01": {"scenario_intersection_n2", "scenario_intersection_n2_4way",
               "scenario_intersection_n2_lshape", "scenario_intersection_n2_tight"},
    "fbot01": {"scenario_bottleneck_n4", "scenario_bottleneck_n4_tightdoor",
               "scenario_bottleneck_n4_convoy", "scenario_bottleneck_n4_imbalanced",
               "scenario_bottleneck_n4_tightdoor_1agent_control"},
    "fcyc01": {"scenario_cycle_ring_n4"},
    "femp01": {"scenario_empty_s11_n3"},
}


def quiet(fn, *args):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return fn(*args)


class TestDevRegistration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = rds.scenario_paths()
        cls.before = rds.source_hashes(cls.paths)
        cls.reg = rds.build_dev_registry("2026-11-05")

    def test_sources_read_only(self):
        self.assertEqual(rds.source_hashes(self.paths), self.before)
        self.assertEqual(len(self.before), 21)          # 16 scenario JSON + 5 maps

    def test_families(self):
        got = {}
        for rec in self.reg.records:
            got.setdefault(rec["family_id"], set()).add(Path(rec["source_path"]).stem)
        self.assertEqual(got, EXPECTED_FAMILIES)

    def test_only_same_map_relations_and_no_review(self):
        summary = self.reg.family_summary()
        self.assertEqual(summary["relation_counts"], {"SAME_MAP": 26})
        self.assertEqual(summary["review_items"], [])
        self.assertEqual(summary["split_counts"], {"DEV": 16})
        self.assertTrue(all(r["agent_id_matches_index"] for r in self.reg.records))
        self.assertEqual(self.reg.consistency_errors(), [])

    def test_roles_recorded(self):
        roles = {Path(r["source_path"]).stem: (r["dev_role"], r["usage_status"]) for r in self.reg.records}
        self.assertEqual(roles["scenario_bottleneck_n4_convoy"], ("variant_ki1", "HOLD"))
        self.assertEqual(roles["scenario_empty_s11_n3"], ("regression_anchor", "REGRESSION"))

    def test_deterministic(self):
        again = rds.build_dev_registry("2026-11-05")
        self.assertEqual(again.to_jsonl(), self.reg.to_jsonl())
        self.assertEqual(again.summary_json(), self.reg.summary_json())

    def test_dev_family_blocks_held_out_reuse(self):
        corridor = next(r for r in self.reg.records if r["source_path"].endswith("corridor_bay_n2.json"))
        # the same corridor map with a new start/goal -> SAME_MAP with a DEV record
        grid = np.pad(grid_from_rows(corridor["grid_trim_rows"]), ((1, 2), (0, 0)), constant_values=1)
        c = compute_fingerprints(grid, [(2, 3)], [(2, 8)])
        c.update(source_path="new/scenario.json", scenario_json_sha256="new",
                 structure_type="corridor", agent_id_matches_index=True)
        for split in ("VAL", "TEST"):
            d = self.reg.evaluate(c, split)
            self.assertEqual(d.status, rg.REJECT, split)
            self.assertTrue(any(x.startswith("FAMILY_LEAK: SAME_MAP") for x in d.reasons))


class TestCommittedRegistry(unittest.TestCase):
    def test_registry_files_match_a_fresh_build(self):
        instances = REGISTRY / rds.INSTANCES
        if not instances.is_file():
            self.skipTest("registry/ not generated yet")
        loaded = rg.Registry.load(instances)
        dev = [r for r in loaded.records if r["split"] == "DEV"]
        self.assertEqual(loaded.records[:16], dev)           # the 16 DEV lines come first
        dates = {r["registered_at"] for r in dev}
        self.assertEqual(len(dates), 1)
        fresh = rds.build_dev_registry(dates.pop())
        # DEV lines are byte-identical to a fresh read-only import (append-only registry)
        self.assertTrue(instances.read_bytes().startswith(fresh.to_jsonl()))
        self.assertEqual((REGISTRY / rds.FAMILIES).read_bytes(), loaded.summary_json())

    def test_check_registry_passes(self):
        if not (REGISTRY / rds.INSTANCES).is_file():
            self.skipTest("registry/ not generated yet")
        _, problems = check_registry.check(REGISTRY)
        self.assertEqual(problems, [])


class TestScripts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_register_then_check_then_refuse_rerun(self):
        out = self.dir / "reg"
        self.assertEqual(quiet(rds.main, ["--registry-dir", str(out), "--registered-at", "2026-11-05"]), 0)
        self.assertEqual(quiet(check_registry.main, ["--registry-dir", str(out)]), 0)
        before = (out / rds.INSTANCES).read_bytes()
        self.assertEqual(quiet(rds.main, ["--registry-dir", str(out)]), 2)
        self.assertEqual((out / rds.INSTANCES).read_bytes(), before)
        self.assertEqual(sorted(p.name for p in out.iterdir()), [rds.FAMILIES, rds.INSTANCES])

    def test_check_detects_tampering(self):
        out = self.dir / "reg"
        quiet(rds.main, ["--registry-dir", str(out), "--registered-at", "2026-11-05"])
        fam = out / rds.FAMILIES
        fam.write_bytes(fam.read_bytes().replace(b'"count": 5', b'"count": 6', 1))
        self.assertEqual(quiet(check_registry.main, ["--registry-dir", str(out)]), 1)

    def test_check_detects_changed_source(self):
        root = self.dir / "root"
        shutil.copytree(PROJECT_ROOT / "scenarios", root / "scenarios")
        reg = rds.build_dev_registry("2026-11-05", root=root)
        out = self.dir / "reg"
        rg.create_files_exclusive([(out / rds.INSTANCES, reg.to_jsonl()),
                                   (out / rds.FAMILIES, reg.summary_json())])
        self.assertEqual(check_registry.check(out, root=root)[1], [])
        victim = root / "scenarios" / "controlled_variants" / "scenario_intersection_n2_tight.json"
        victim.write_text(victim.read_text().replace("[2, 3]", "[1, 3]"))   # copy only
        problems = check_registry.check(out, root=root)[1]
        self.assertTrue(any("scenario_json_sha256 differs" in p for p in problems))

    def test_extra_scenario_file_refused(self):
        root = self.dir / "root"
        shutil.copytree(PROJECT_ROOT / "scenarios", root / "scenarios")
        (root / "scenarios" / "smoke" / "scenario_extra.json").write_text("{}")
        with self.assertRaises(rg.RegistryError):
            rds.scenario_paths(root)


if __name__ == "__main__":
    unittest.main()
