"""src/data/generator_v2.py + scripts/{generate,check}_pilot_v2_candidates.py."""

import copy
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np

from scripts import check_pilot_v2_candidates as chk
from scripts import generate_pilot_v2_candidates as genv2
from src.common.scenario_loader import load_map_generator_scenario
from src.data import generator_v2 as g2
from src.data.fingerprint import compute_fingerprints, grid_rows, normalize_grid
from src.data.generator import GeneratorError
from src.data.quality_gate import independence_check
from src.data.registry import REVIEW, SUBMAP, Registry, RegistryError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REG_DIR = PROJECT_ROOT / "registry"
INSTANCES = REG_DIR / "instances_v1.jsonl"
GENERATOR_V1_SHA = "a2e103190800d0d84d5e479ee998bbc98d2ac5cce2b7803ce54bd0a408f5837d"


def specs():
    return copy.deepcopy(genv2.load_prereg()["candidates"])


def spec_by_id(cid):
    return next(s for s in specs() if s["candidate_id"] == cid)


def quiet(fn, *args):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return fn(*args)


class TestPrereg(unittest.TestCase):
    def test_sha_is_pinned_and_a_changed_file_is_refused(self):
        self.assertEqual(hashlib.sha256(genv2.PREREG_PATH.read_bytes()).hexdigest(), genv2.PREREG_SHA256)
        with tempfile.TemporaryDirectory() as d:
            changed = Path(d) / "prereg.json"
            changed.write_bytes(genv2.PREREG_PATH.read_bytes() + b" ")
            with self.assertRaises(RegistryError):
                genv2.load_prereg(changed)

    def test_candidates_are_the_approved_set(self):
        got = [(s["candidate_id"], s["seed"], s["grade"], s["num_agents"]) for s in specs()]
        self.assertEqual(got, [("PV2C-001", 21, "NORMAL", 4), ("PV2C-002", 22, "NORMAL", 4),
                               ("PV2C-003", 23, "STRESS", 6)])
        for s in specs():
            self.assertNotIn(s["seed"], (11, 12, 13))
            self.assertLess(s["seed"], 100)

    def test_every_candidate_has_a_prediction_and_manual_holds_exist(self):
        prereg = genv2.load_prereg()
        for s in prereg["candidates"]:
            self.assertIn("ood_class", s["predicted"])
            self.assertIn("target_met", s["predicted"])
        self.assertEqual({h["candidate_id"] for h in prereg["manual_similarity_holds"]},
                         {"PV2C-001", "PV2C-002", "PV2C-003"})


class TestBuilders(unittest.TestCase):
    def test_built_grid_matches_prereg_and_is_read_only(self):
        for s in specs():
            inst = g2.build(s["builder"], s["params"], s["seed"])
            self.assertEqual(grid_rows(normalize_grid(inst.grid)), s["grid_rows"], s["slug"])
            self.assertEqual(int((inst.grid == 0).sum()), s["free_cells"], s["slug"])
            self.assertEqual(len(inst.starts), s["num_agents"], s["slug"])
            self.assertEqual(inst.grid.dtype, np.int64)
            self.assertFalse(inst.grid.flags.writeable)
            self.assertEqual(inst.descriptor["generator"], g2.GENERATOR_NAME)

    def test_same_seed_same_instance_other_seed_only_moves_agents(self):
        for s in specs():
            a = g2.build(s["builder"], s["params"], s["seed"])
            b = g2.build(s["builder"], s["params"], s["seed"])
            self.assertEqual((a.starts, a.goals), (b.starts, b.goals))
            self.assertTrue(np.array_equal(a.grid, b.grid))
            others = [g2.build(s["builder"], s["params"], s["seed"] + k) for k in (1, 2, 3)]
            self.assertTrue(all(np.array_equal(a.grid, o.grid) for o in others))
            self.assertTrue(any((a.starts, a.goals) != (o.starts, o.goals) for o in others), s["slug"])

    def test_hub_fixed_starts_follow_prereg(self):
        s = spec_by_id("PV2C-003")
        inst = g2.build(s["builder"], s["params"], s["seed"])
        self.assertEqual(list(inst.starts[4]), s["fixed_starts"]["S->N"])
        self.assertEqual(list(inst.starts[5]), s["fixed_starts"]["E->W"])

    def test_bad_parameters_raise(self):
        s = spec_by_id("PV2C-002")
        bad = copy.deepcopy(s["params"])
        bad["zones"]["west"] = {"rows": [3, 3], "cols": [2, 3]}          # central wall
        with self.assertRaises(GeneratorError):
            g2.build(s["builder"], bad, s["seed"])
        s = spec_by_id("PV2C-001")
        bad = copy.deepcopy(s["params"])
        bad["block_h"] = 1                                               # single-cell pillars
        with self.assertRaises(GeneratorError):
            g2.build(s["builder"], bad, s["seed"])
        with self.assertRaises(GeneratorError):
            g2.build("corridor_bays", {}, 1)

    def test_scenario_records_provenance_in_xy(self):
        s = spec_by_id("PV2C-002")
        inst = g2.build(s["builder"], s["params"], s["seed"])
        _, js = g2.scenario_files(inst, s["slug"], code_sha256="a", base_generator_sha256="b",
                                  prereg_sha256="c", candidate_id=s["candidate_id"])
        doc = json.loads(js)
        self.assertEqual(doc["generator"]["prereg_sha256"], "c")
        self.assertEqual(doc["generator"]["base_generator_sha256"], "b")
        self.assertEqual(doc["agents"][0]["start"], [inst.starts[0][1], inst.starts[0][0]])


class TestGenerateScript(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _run(self):
        return quiet(genv2.main, ["--root", str(self.tmp), "--out-root", str(self.tmp / "out")])

    def test_writes_six_files_keeps_registry_and_refuses_a_second_run(self):
        before = INSTANCES.read_bytes()
        self.assertEqual(self._run(), 0)
        files = sorted(p.name for p in (self.tmp / genv2.PILOT_V2_DIR).iterdir())
        self.assertEqual(len(files), 6)
        self.assertEqual(self._run(), 2)
        self.assertEqual(INSTANCES.read_bytes(), before)
        for s in specs():
            grid, starts, _ = load_map_generator_scenario(genv2.candidate_paths(s, self.tmp)[1])
            self.assertEqual(grid_rows(normalize_grid(grid)), s["grid_rows"])
            self.assertEqual(len(starts), s["num_agents"])

    def test_failed_candidate_is_recorded_and_others_are_written(self):
        sp = specs()
        sp[0]["grid_rows"][0] = "111111111"                              # no longer matches the build
        records = genv2.generate_all(sp, self.tmp)
        self.assertEqual([r["generated"] for r in records], [False, True, True])
        self.assertIn("grid differs", records[0]["error"])
        self.assertFalse(genv2.candidate_paths(sp[0], self.tmp)[1].exists())
        self.assertTrue(genv2.candidate_paths(sp[1], self.tmp)[1].exists())

    def test_bytes_do_not_depend_on_pythonhashseed(self):
        outs = []
        for seed in ("0", "1"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            r = subprocess.run([sys.executable, "-m", "scripts.generate_pilot_v2_candidates", "--dry-run"],
                               cwd=str(PROJECT_ROOT), env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            outs.append(r.stdout)
        self.assertEqual(outs[0], outs[1])


class TestMetrics(unittest.TestCase):
    def test_width1_cells(self):
        grid = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]])
        # outside the map counts as blocked: each corner has it above or below and a wall on the other side
        self.assertEqual(chk.width1_cells(grid), [[0, 0], [0, 2], [1, 1], [2, 0], [2, 2]])
        self.assertEqual(chk.width1_cells(np.zeros((3, 3), dtype=int)), [])

    def test_articulation_cells(self):
        self.assertEqual(chk.articulation_cells(np.array([[0, 0, 0]])), [[0, 1]])
        self.assertEqual(chk.articulation_cells(np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])), [])

    def test_agent_metrics_unreachable_goal(self):
        grid = np.array([[0, 1, 0]])
        m = chk.agent_metrics(grid, [(0, 0)], [(0, 2)])
        self.assertFalse(m["all_goals_reachable"])
        self.assertIsNone(m["soc_lower_bound"])

    def test_agent_metrics_bounds(self):
        grid = np.zeros((3, 3), dtype=int)
        m = chk.agent_metrics(grid, [(0, 0), (2, 2)], [(0, 2), (0, 0)])
        self.assertEqual((m["soc_lower_bound"], m["makespan_lower_bound"]), (6, 4))
        self.assertEqual(m["agents_starting_on_another_goal"], [0])


class TestNoSelection(unittest.TestCase):
    def test_normal_missing_its_target_keeps_its_grade(self):
        for ood, wide1, agents in (("OOD_TAIL", 0, 4), ("IN_RANGE", 8, 4), ("IN_RANGE", 0, 5)):
            t = chk.evaluate_target("NORMAL", ood, wide1, agents, 5)
            self.assertEqual((t["grade"], t["target_met"]), ("NORMAL", False), (ood, wide1, agents))
        self.assertTrue(chk.evaluate_target("NORMAL", "IN_RANGE", 0, 4, 5)["target_met"])

    def test_stress_target(self):
        self.assertTrue(chk.evaluate_target("STRESS", "OOD_TAIL", 5, 6, 13)["target_met"])
        t = chk.evaluate_target("STRESS", "OOD_TAIL", 5, 6, 12)
        self.assertEqual((t["grade"], t["target_met"]), ("STRESS", False))

    def test_independence_is_never_approved(self):
        for existing, pairs, manual in (([], [], []), ([{}], [], []), ([], [{}], []), ([], [], [{}])):
            s = chk.independence_summary(existing, pairs, manual)
            self.assertFalse(s["independence_approved"])
        self.assertEqual(chk.independence_summary([], [], [])["status"], chk.NO_RELATION_NOT_APPROVED)
        self.assertEqual(chk.independence_summary([], [], [{}])["status"], chk.HOLD)

    def test_scripts_cannot_reach_the_solver(self):
        for path in (PROJECT_ROOT / "scripts" / "check_pilot_v2_candidates.py",
                     PROJECT_ROOT / "scripts" / "generate_pilot_v2_candidates.py"):
            text = path.read_text(encoding="utf-8")
            for word in ("cbs_runtime_bridge", "CbsRuntimePlanner", "--run-solver", "solver_root"):
                self.assertNotIn(word, text, path.name)


class TestRelationsAndHolds(unittest.TestCase):
    def setUp(self):
        self.reg = Registry.load(INSTANCES)

    def _record(self, grid, starts, goals):
        rec = compute_fingerprints(grid, starts, goals)
        rec.update(structure_type="test", structure_descriptor=None)
        return rec

    def test_isolated_pillar_is_caught_as_submap_of_dev_ring(self):
        grid = np.zeros((5, 5), dtype=int)
        grid[2, 2] = 1
        res = independence_check(self.reg, self._record(grid, [(0, 0)], [(4, 4)]))
        self.assertFalse(res["independent"])
        self.assertIn(("DEV_fcyc01_001", SUBMAP, REVIEW),
                      [(r["other"], r["relation"], r["status"]) for r in res["relations"]])

    def test_pairwise_relation_between_candidates_is_found(self):
        small = np.zeros((3, 3), dtype=int)
        small[1, 1] = 1
        big = np.zeros((5, 5), dtype=int)
        big[2, 2] = 1
        pairs = chk.pairwise_candidate_relations({
            "A": self._record(small, [(0, 0)], [(2, 2)]), "B": self._record(big, [(0, 0)], [(4, 4)])})
        self.assertEqual([(p["a"], p["b"], p["relation"]) for p in pairs], [("A", "B", SUBMAP)])

    def test_same_instance_between_candidates_is_exact(self):
        rec = self._record(np.zeros((4, 4), dtype=int), [(0, 0)], [(3, 3)])
        pairs = chk.pairwise_candidate_relations({"A": rec, "B": copy.deepcopy(rec)})
        self.assertEqual([p["relation"] for p in pairs], ["EXACT"])

    def test_manual_hold_stays_without_any_automatic_relation(self):
        prereg = genv2.load_prereg()
        rec = self._record(np.zeros((12, 13), dtype=int), [(0, 0)], [(11, 12)])
        entries = chk.review_entries(prereg, {"PV2C-002": rec}, self.reg, {"PV2C-002": []}, [])
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual((e["relation"], e["source"], e["status"], e["final_decision"], e["decision_owner"]),
                         (chk.MANUAL_SIMILARITY, "MANUAL", chk.HOLD, None, None))
        self.assertEqual(e["other"], ["DEV_fcyc01_001"])

    def test_family_scope_hold_lists_family_members(self):
        prereg = genv2.load_prereg()
        rec = self._record(np.zeros((12, 13), dtype=int), [(0, 0)], [(11, 12)])
        entries = chk.review_entries(prereg, {"PV2C-003": rec}, self.reg, {"PV2C-003": []}, [])
        self.assertEqual(len(entries), 2)
        self.assertEqual(len(entries[0]["evidence"]["members"]), 5)     # fint01 4 + fint02 1
        self.assertEqual(len(entries[1]["evidence"]["members"]), 6)     # fbot01 5 + fbot02 1


class TestCheckWithMissingCandidate(unittest.TestCase):
    def test_missing_files_are_problems_and_the_record_is_kept(self):
        with tempfile.TemporaryDirectory() as d:
            result = chk.check_candidate(specs()[0], Path(d), chk.load_reference(REG_DIR / chk.REFERENCE))
        self.assertEqual(result["candidate_id"], "PV2C-001")
        self.assertTrue(result["problems"])
        self.assertIsNone(result["fingerprint"])


class TestExistingDataUntouched(unittest.TestCase):
    def test_generator_v1_sha(self):
        self.assertEqual(hashlib.sha256((PROJECT_ROOT / "src/data/generator.py").read_bytes()).hexdigest(),
                         GENERATOR_V1_SHA)

    def test_registry_still_has_only_the_19_records(self):
        reg = Registry.load(INSTANCES)
        self.assertEqual(len(reg.records), 19)
        self.assertFalse([r for r in reg.records if "pilot_v2" in r["source_path"]])


class TestCommittedCandidates(unittest.TestCase):
    """Against scenarios/pilot_v2/ as generated in the repository."""

    @classmethod
    def setUpClass(cls):
        cls.report = chk.build_report(PROJECT_ROOT, REG_DIR)

    def test_all_three_records_are_kept_without_integrity_problems(self):
        self.assertEqual([c["candidate_id"] for c in self.report["candidates"]], ["PV2C-001", "PV2C-002", "PV2C-003"])
        self.assertEqual(self.report["problems"], [])
        self.assertFalse(self.report["cbs_run"])

    def test_static_checks_hold(self):
        for c in self.report["candidates"]:
            self.assertTrue(c["seed_reproducible"], c["slug"])
            self.assertTrue(c["cells_valid"], c["slug"])
            self.assertTrue(c["all_goals_reachable"], c["slug"])
            self.assertTrue(c["agent_id_matches_index"], c["slug"])
            self.assertTrue(c["coordinates_ok"], c["slug"])

    def test_grade_is_unchanged_and_independence_is_not_approved(self):
        prereg = {s["candidate_id"]: s["grade"] for s in specs()}
        for c in self.report["candidates"]:
            self.assertEqual(c["grade"], prereg[c["candidate_id"]])
            self.assertEqual(c["target"]["grade"], prereg[c["candidate_id"]])
            self.assertFalse(c["independence"]["independence_approved"])
            self.assertEqual(c["independence"]["status"], chk.HOLD)
            self.assertFalse(c["ood"]["auto_fail"])

    def test_all_review_entries_are_undecided_holds(self):
        self.assertGreaterEqual(len(self.report["review_entries"]), 4)
        for e in self.report["review_entries"]:
            self.assertEqual((e["status"], e["final_decision"], e["decision_owner"]), (chk.HOLD, None, None))

    def test_review_file_matches_the_recomputed_entries(self):
        path = REG_DIR / chk.REVIEW_FILE
        self.assertEqual(path.read_bytes(), chk.review_bytes(self.report["review_entries"]))


if __name__ == "__main__":
    unittest.main()
