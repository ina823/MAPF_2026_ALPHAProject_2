"""scripts/generate_pilot_instances.py + scripts/check_pilot_instances.py (no solver run)."""

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts import check_pilot_instances as cpi
from scripts import check_registry, generate_pilot_instances as gpi
from scripts import register_dev_scenarios as rds
from src.data import registry as rg

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = PROJECT_ROOT / "registry"


def quiet(fn, *args):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return fn(*args)


class TestCommittedPilots(unittest.TestCase):
    def setUp(self):
        self.reg = rg.Registry.load(REGISTRY / rds.INSTANCES)
        self.pilots = [r for r in self.reg.records if r["split"] == "PILOT"]
        if not self.pilots:
            self.skipTest("pilots not generated yet")

    def test_files_and_fingerprints_regenerate_identically(self):
        self.assertEqual(cpi.regeneration_checks(self.reg), [])

    def test_three_new_families_never_held_out(self):
        self.assertEqual(len(self.pilots), 3)
        self.assertEqual({r["structure_type"] for r in self.pilots}, {"corridor", "bottleneck", "intersection"})
        self.assertEqual(len({r["family_id"] for r in self.pilots}), 3)
        dev_families = {r["family_id"] for r in self.reg.records if r["split"] == "DEV"}
        self.assertFalse(dev_families & {r["family_id"] for r in self.pilots})
        for r in self.pilots:
            self.assertTrue(all(x["status"] == rg.REVIEW for x in r["relations"]), r["instance_id"])
            self.assertEqual(r["generator"]["seed"], gpi.PILOT_SPECS[Path(r["source_path"]).stem[9:]][1]["seed"])
            self.assertTrue(r["agent_id_matches_index"])

    def test_pilot_family_blocks_held_out(self):
        rec = self.pilots[0]
        cand = dict(rec, source_path="elsewhere.json", scenario_json_sha256="other")
        for split in ("VAL", "TEST"):
            d = self.reg.evaluate(cand, split)
            self.assertEqual(d.status, rg.REJECT)
            self.assertTrue(any(x.startswith("DUPLICATE_INSTANCE") for x in d.reasons))

    def test_registry_consistent(self):
        self.assertEqual(check_registry.check(REGISTRY)[1], [])


class TestGenerateScript(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.reg_dir = self.dir / "registry"
        dev = rds.build_dev_registry("2026-11-05")
        rg.create_files_exclusive([(self.reg_dir / rds.INSTANCES, dev.to_jsonl()),
                                   (self.reg_dir / rds.FAMILIES, dev.summary_json())])
        self.dev_bytes = (self.reg_dir / rds.INSTANCES).read_bytes()
        self.args = ["--registry-dir", str(self.reg_dir), "--root", str(self.dir), "--registered-at", "2026-11-05"]

    def tearDown(self):
        self.tmp.cleanup()

    def test_dry_run_writes_nothing(self):
        self.assertEqual(quiet(gpi.main, self.args + ["--dry-run"]), 0)
        self.assertFalse((self.dir / gpi.PILOT_DIR).exists())
        self.assertEqual((self.reg_dir / rds.INSTANCES).read_bytes(), self.dev_bytes)

    def test_generate_append_check_and_refuse_rerun(self):
        self.assertEqual(quiet(gpi.main, self.args), 0)
        files = sorted(p.name for p in (self.dir / gpi.PILOT_DIR).iterdir())
        self.assertEqual(len(files), 6)
        data = (self.reg_dir / rds.INSTANCES).read_bytes()
        self.assertTrue(data.startswith(self.dev_bytes))        # DEV lines unchanged
        reg = rg.Registry.load(self.reg_dir / rds.INSTANCES)
        self.assertEqual([r["split"] for r in reg.records].count("PILOT"), 3)
        # DEV sources are repo-relative, the temporary pilots are absolute paths
        self.assertEqual(check_registry.check(self.reg_dir, root=PROJECT_ROOT)[1], [])
        # same pilots as the committed ones (same seeds -> same fingerprints)
        committed = {r["instance_id"]: r["grid_fp"] + r["agents_ordered_fp"]
                     for r in rg.Registry.load(REGISTRY / rds.INSTANCES).records if r["split"] == "PILOT"}
        if committed:
            self.assertEqual({r["instance_id"]: r["grid_fp"] + r["agents_ordered_fp"]
                              for r in reg.records if r["split"] == "PILOT"}, committed)
        self.assertEqual(quiet(gpi.main, self.args), 2)

    def test_failure_after_writing_rolls_back(self):
        fam_before = (self.reg_dir / rds.FAMILIES).read_bytes()
        with patch.object(gpi, "fingerprint_scenario", side_effect=RuntimeError("boom")):
            self.assertEqual(quiet(gpi.main, self.args), 1)
        self.assertFalse((self.dir / gpi.PILOT_DIR).exists())
        self.assertEqual((self.reg_dir / rds.INSTANCES).read_bytes(), self.dev_bytes)
        self.assertEqual((self.reg_dir / rds.FAMILIES).read_bytes(), fam_before)


class TestSolverPlan(unittest.TestCase):
    def test_plan_counts_and_paths(self):
        plan = cpi.solver_plan(PROJECT_ROOT / "outputs" / "pilot_checks_v1" / "20990101_000000",
                               cpi.DEFAULT_SOLVER_ROOT, 120)
        self.assertEqual(plan["expected_cbs_calls"], 3)
        self.assertEqual(plan["worst_case_wall_sec"], 360)
        self.assertTrue(plan["max_path_ok"])
        self.assertTrue(all(c["request"].startswith("INITIAL_STATE") for c in plan["calls"]))


if __name__ == "__main__":
    unittest.main()
