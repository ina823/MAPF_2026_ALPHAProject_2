"""Data split inventory, leakage report, usage rules and held-out freeze.

Synthetic cases run in a temporary repository root (scenario files in the
real JSON + .npy format, a registry written with src.data.registry). The last
class reads the REAL repository read-only and checks the current split state
and that the registry file is not changed.
"""

import contextlib
import hashlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data import inventory as inv
from src.data import split_policy as sp
from src.data.fingerprint import fingerprint_scenario
from src.data.registry import Registry

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GRID = np.array([[0, 0, 0, 0, 0],
                 [0, 1, 1, 0, 0],
                 [0, 0, 0, 0, 1]], dtype=np.int64)
AGENTS = [((0, 0), (2, 3)), ((2, 0), (0, 4))]            # (row, col) start, goal


def _builder(name):
    """In-memory builder used through inventory.in_memory_records (module path of this file)."""
    if name == "BAD":
        raise RuntimeError("cannot build")
    return GRID.copy(), {0: AGENTS[0][0], 1: AGENTS[1][0]}, {0: AGENTS[0][1], 1: AGENTS[1][1]}, "copy"


class Repo:
    """A temporary repository root with scenarios/, registry/, outputs/."""

    def __init__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "registry").mkdir()
        self.registry_records = []

    def cleanup(self):
        self._tmp.cleanup()

    def scenario(self, rel, grid=GRID, agents=AGENTS, *, extra=None):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        map_name = path.stem.replace("scenario_", "map_") + ".npy"
        np.save(path.parent / map_name, np.asarray(grid, dtype=np.int64))
        data = {"scenario_id": path.stem, "map_id": path.stem, "map_file": map_name,
                "agents": [{"agent_id": i, "start": [s[1], s[0]], "goal": [g[1], g[0]]}
                           for i, (s, g) in enumerate(agents)], **(extra or {})}
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return rel

    def register(self, rel, split, family):
        fp = fingerprint_scenario(self.root / rel)
        n = 1 + sum(1 for r in self.registry_records if r["split"] == split and r["family_id"] == family)
        rec = dict(fp, source_path=rel,
                   map_file_path=str(Path(rel).parent / Path(fp["map_file_path"]).name),
                   instance_id=f"{split}_{family}_{n:03d}", family_id=family, split=split,
                   structure_type="corridor", structure_descriptor=None, relations=[])
        self.registry_records.append(rec)
        self.write_registry()
        return rec

    def write_registry(self):
        (self.root / "registry" / "instances_v1.jsonl").write_bytes(Registry(self.registry_records).to_jsonl())

    def report(self, builders=()):
        return inv.build_report(self.root, builders=builders)


def rot90(grid, agents):
    g = np.rot90(grid)
    h = grid.shape[1]          # rot90 ccw: (r, c) -> (W-1-c, r)
    t = lambda p: (h - 1 - p[1], p[0])
    return g, [(t(s), t(gl)) for s, gl in agents]


class _Case(unittest.TestCase):
    def setUp(self):
        self.repo = Repo()
        self.addCleanup(self.repo.cleanup)

    def relations(self, report, a, b):
        return sorted(p["relation"] for p in report["pairs"] if {p["a"], p["b"]} == {a, b})


class TestRoles(_Case):

    def test_role_from_registry_prereg_or_unknown(self):
        r = self.repo
        dev = r.scenario("scenarios/dev/scenario_a.json")
        r.register(dev, "DEV", "fcorr01")
        r.scenario("scenarios/pilot_x/scenario_b.json", grid=np.zeros((4, 6), dtype=np.int64))
        (r.root / "scenarios/pilot_x/prereg.json").write_text(json.dumps({"split_intent": "PILOT (development side)"}))
        r.scenario("scenarios/misc/scenario_c.json", grid=np.zeros((5, 5), dtype=np.int64))
        prereg = (("scenarios/pilot_x/", "scenarios/pilot_x/prereg.json"),)
        orig = inv.PREREGISTRATIONS
        inv.PREREGISTRATIONS = prereg
        try:
            rep = r.report()
        finally:
            inv.PREREGISTRATIONS = orig
        roles = {x["source_path"]: (x["data_role"], x["split"]) for x in rep["records"]}
        self.assertEqual(roles, {"scenarios/dev/scenario_a.json": ("DEV", "DEV"),
                                 "scenarios/pilot_x/scenario_b.json": ("DEV", "PILOT"),
                                 "scenarios/misc/scenario_c.json": ("UNKNOWN", None)})
        unknown = next(x for x in rep["records"] if x["data_role"] == "UNKNOWN")
        self.assertIsNone(unknown["role_evidence"])
        self.assertEqual(rep["role_counts"], {"DEV": 2, "VALIDATION": 0, "TEST": 0, "UNKNOWN": 1})

    def test_record_carries_the_required_fields(self):
        r = self.repo
        r.register(r.scenario("scenarios/dev/scenario_a.json"), "DEV", "fcorr01")
        (rec,) = r.report()["records"]
        for key in ("scenario_id", "data_role", "map_id", "grid_fp", "map_file_sha256", "scenario_json_sha256",
                    "num_agents", "starts_rc", "goals_rc", "family_id", "generation_method",
                    "used_in_development"):
            self.assertIn(key, rec)
        self.assertEqual(rec["scenario_json_sha256"],
                         hashlib.sha256((r.root / "scenarios/dev/scenario_a.json").read_bytes()).hexdigest())
        self.assertEqual((rec["starts_rc"], rec["goals_rc"]), ([[0, 0], [2, 0]], [[2, 3], [0, 4]]))

    def test_in_memory_instance_and_build_error(self):
        rep = self.repo.report(builders=(("tests.test_data_split", "_builder", "X"),
                                         ("tests.test_data_split", "_builder", "BAD")))
        ok, bad = rep["records"]
        self.assertEqual((ok["kind"], ok["data_role"], ok["used_in_development"]), ("IN_MEMORY", "DEV", "USED"))
        self.assertIsNone(ok["scenario_json_sha256"])
        self.assertIn("cannot build", bad["build_error"])
        self.assertTrue(any("cannot build" in e for e in rep["errors"]))

    def test_usage_found_or_not_recorded(self):
        r = self.repo
        r.scenario("scenarios/dev/scenario_used.json")
        r.scenario("scenarios/dev/scenario_quiet.json", grid=np.zeros((4, 6), dtype=np.int64))
        (r.root / "outputs/run1").mkdir(parents=True)
        (r.root / "outputs/run1/summary.json").write_text('{"scenario": "scenario_used.json"}')
        use = {x["source_path"]: (x["used_in_development"], x["usage_evidence"]) for x in r.report()["records"]}
        self.assertEqual(use["scenarios/dev/scenario_used.json"], ("USED", ["outputs/run1"]))
        self.assertEqual(use["scenarios/dev/scenario_quiet.json"], ("NO_RECORD_FOUND", []))


class TestDuplicatesAndLeaks(_Case):

    def test_same_file_bytes(self):
        r = self.repo
        a = r.scenario("scenarios/dev/scenario_a.json")
        (r.root / "scenarios/copy").mkdir(parents=True)
        for f in (r.root / "scenarios/dev").iterdir():
            shutil.copy(f, r.root / "scenarios/copy" / f.name)
        self.assertEqual(self.relations(r.report(), a, "scenarios/copy/scenario_a.json"),
                         ["EXACT", "SAME_FILE_BYTES"])

    def test_same_grid_and_agents_in_another_file(self):
        r = self.repo
        a = r.scenario("scenarios/dev/scenario_a.json")
        b = r.scenario("scenarios/dev/scenario_b.json", extra={"note": "generator seed 7"})
        self.assertEqual(self.relations(r.report(), a, b), ["EXACT"])   # a seed field alone is not new data

    def test_agent_ids_swapped(self):
        r = self.repo
        a = r.scenario("scenarios/dev/scenario_a.json")
        b = r.scenario("scenarios/dev/scenario_b.json", agents=list(reversed(AGENTS)))
        self.assertEqual(self.relations(r.report(), a, b), ["RELABEL_EQUIV"])

    def test_rotated_map_and_agents(self):
        r = self.repo
        a = r.scenario("scenarios/dev/scenario_a.json")
        g, ag = rot90(GRID, AGENTS)
        b = r.scenario("scenarios/dev/scenario_b.json", grid=g, agents=ag)
        rep = r.report()
        self.assertEqual(self.relations(rep, a, b), ["SYMMETRY_EQUIV"])
        (p,) = [p for p in rep["pairs"] if p["relation"] == "SYMMETRY_EQUIV"]
        self.assertEqual(p["status"], "CONFIRMED")

    def test_near_map_needs_review_not_confirmed(self):
        r = self.repo
        a = r.scenario("scenarios/dev/scenario_a.json")
        g = GRID.copy()
        g[0, 2] = 1
        b = r.scenario("scenarios/dev/scenario_b.json", grid=g, agents=[((0, 0), (2, 3)), ((2, 0), (1, 4))])
        rep = r.report()
        self.assertEqual(self.relations(rep, a, b), ["NEAR_MAP"])
        self.assertEqual({p["status"] for p in rep["pairs"]}, {"REVIEW"})

    def test_dev_validation_leak(self):
        r = self.repo
        r.register(r.scenario("scenarios/dev/scenario_a.json"), "DEV", "fcorr01")
        r.register(r.scenario("scenarios/val/scenario_a2.json"), "VAL", "fcorr09")
        rep = r.report()
        self.assertEqual(rep["role_pair_counts"]["DEV-VALIDATION"], 1)   # EXACT (ids differ: not the same bytes)
        self.assertTrue(any("same instance in two roles" in e for e in rep["errors"]))
        self.assertEqual({l["role_pair"] for l in rep["leaks"]}, {"DEV-VALIDATION"})

    def test_dev_test_leak_by_same_map(self):
        r = self.repo
        r.register(r.scenario("scenarios/dev/scenario_a.json"), "DEV", "fcorr01")
        r.register(r.scenario("scenarios/test/scenario_t.json", agents=[((0, 3), (2, 2))]), "TEST", "fcorr09")
        (leak,) = r.report()["leaks"]
        self.assertEqual((leak["role_pair"], leak["relation"]), ("DEV-TEST", "SAME_MAP"))

    def test_validation_test_leak_by_relabel(self):
        r = self.repo
        r.register(r.scenario("scenarios/val/scenario_v.json"), "VAL", "fcorr08")
        r.register(r.scenario("scenarios/test/scenario_t.json", agents=list(reversed(AGENTS))), "TEST", "fcorr09")
        (leak,) = r.report()["leaks"]
        self.assertEqual((leak["role_pair"], leak["relation"]), ("VALIDATION-TEST", "RELABEL_EQUIV"))

    def test_unrelated_maps_no_pairs(self):
        r = self.repo
        r.register(r.scenario("scenarios/dev/scenario_a.json"), "DEV", "fcorr01")
        r.register(r.scenario("scenarios/test/scenario_t.json", grid=np.zeros((6, 9), dtype=np.int64),
                              agents=[((0, 0), (5, 8))]), "TEST", "femp09")
        rep = r.report()
        self.assertEqual((rep["pairs"], rep["leaks"], rep["errors"]), ([], [], []))

    def test_seed_only_runs_are_one_instance(self):
        runs = [{"instance_fp": "x", "seed": "0", "source": "b1"}, {"instance_fp": "x", "seed": "1", "source": "b2"},
                {"instance_fp": "x", "seed": "2", "source": "b3"}, {"instance_fp": "y", "source": "b1"}]
        groups = {g["instance_fp"]: g for g in inv.seed_sweeps(runs)}
        self.assertEqual((groups["x"]["runs"], groups["x"]["counts_as_instances"], groups["x"]["seed_sweep"]),
                         (3, 1, True))
        self.assertEqual((groups["y"]["seeds"], groups["y"]["seed_sweep"]), (["NOT_RECORDED"], False))

    def test_manifest_runs_mapped_to_instances(self):
        r = self.repo
        a = r.scenario("scenarios/dev/scenario_a.json")
        sha = hashlib.sha256((r.root / a).read_bytes()).hexdigest()
        for name, seed in (("b1", "0"), ("b2", "5")):
            d = r.root / "outputs" / name
            d.mkdir(parents=True)
            (d / "batch_manifest.json").write_text(json.dumps(
                {"conditions": {"scenarios": {"scenario_a.json": sha}}, "reproducibility": {"seed_value": seed}}))
        (g,) = r.report()["seed_sweeps"]
        self.assertEqual((g["seeds"], g["seed_sweep"], g["counts_as_instances"]), (["0", "5"], True, 1))


class TestUsageRules(unittest.TestCase):

    def test_role_purpose_matrix(self):
        allowed = {(role, p) for role in sp.ALLOWED_USE for p in sp.PURPOSES
                   if not sp.check_use(role, p, freeze={}, freeze_problems=[])}
        self.assertEqual(allowed, {("DEV", "development"), ("DEV", "debugging"),
                                   ("VALIDATION", "parameter_selection"), ("TEST", "final_evaluation")})
        self.assertTrue(sp.check_use("UNKNOWN", "development"))
        self.assertTrue(sp.check_use("DEV", "final_evaluation"))

    def test_test_needs_a_verified_freeze(self):
        self.assertIn("freeze record", sp.check_use("TEST", "final_evaluation")[0])
        self.assertIn("not verified", sp.check_use("TEST", "final_evaluation", freeze={})[0])
        self.assertIn("does not verify", sp.check_use("TEST", "final_evaluation", freeze={},
                                                      freeze_problems=["x changed"])[0])

    def test_dev_results_never_labelled_test(self):
        self.assertEqual(sp.result_label(["DEV", "DEV"]), "DEV")
        self.assertEqual(sp.result_label(["DEV", "TEST"]), "UNLABELLED")
        self.assertEqual(sp.result_label(["UNKNOWN"]), "UNLABELLED")
        self.assertTrue(sp.check_result_claim(["DEV"], "TEST"))
        self.assertTrue(sp.check_result_claim(["DEV"], "VALIDATION"))
        self.assertEqual(sp.check_result_claim(["TEST", "TEST"], "TEST"), [])

    def test_shortfall(self):
        self.assertEqual(sp.held_out_shortfall({"VALIDATION": 0, "TEST": 0}, {"VALIDATION": 45, "TEST": 75}),
                         {"VALIDATION": {"have": 0, "target": 45, "missing": 45},
                          "TEST": {"have": 0, "target": 75, "missing": 75}})


class TestFreeze(_Case):

    def frozen(self):
        r = self.repo
        r.register(r.scenario("scenarios/dev/scenario_a.json"), "DEV", "fcorr01")
        r.register(r.scenario("scenarios/test/scenario_t1.json", grid=np.zeros((6, 9), dtype=np.int64),
                              agents=[((0, 0), (5, 8))]), "TEST", "femp09")
        path = r.root / "registry" / "instances_v1.jsonl"
        reg = Registry.load(path)
        return sp.freeze_record(reg, path, "TEST", commit="c0ffee", frozen_at="2026-11-25", root=r.root), path

    def test_freeze_verifies_and_allows_final_evaluation(self):
        freeze, path = self.frozen()
        self.assertEqual((freeze["instance_count"], freeze["data_role"]), (1, "TEST"))
        problems = sp.verify_freeze(freeze, self.repo.root, Registry.load(path))
        self.assertEqual(problems, [])
        self.assertEqual(sp.check_use("TEST", "final_evaluation", freeze=freeze, freeze_problems=problems), [])

    def test_changed_file_breaks_the_freeze(self):
        freeze, path = self.frozen()
        f = self.repo.root / "scenarios/test/scenario_t1.json"
        f.write_text(f.read_text() + " ")
        self.assertTrue(any("scenario file changed" in p
                            for p in sp.verify_freeze(freeze, self.repo.root, Registry.load(path))))
        np.save(self.repo.root / "scenarios/test/map_t1.npy", np.ones((6, 9), dtype=np.int64))
        self.assertTrue(any("map file changed" in p
                            for p in sp.verify_freeze(freeze, self.repo.root, Registry.load(path))))

    def test_membership_change_breaks_the_freeze(self):
        freeze, path = self.frozen()
        self.repo.register(self.repo.scenario("scenarios/test/scenario_t2.json",
                                              grid=np.zeros((7, 9), dtype=np.int64), agents=[((0, 0), (6, 8))]),
                           "TEST", "femp10")
        self.assertTrue(any("registered in TEST after the freeze" in p
                            for p in sp.verify_freeze(freeze, self.repo.root, Registry.load(path))))

    def test_split_change_or_edited_record_breaks_the_freeze(self):
        freeze, path = self.frozen()
        self.repo.registry_records[1]["split"] = "VAL"
        self.repo.registry_records[1]["instance_id"] = "VAL_femp09_001"
        self.repo.write_registry()
        self.assertTrue(any("no longer in the registry" in p
                            for p in sp.verify_freeze(freeze, self.repo.root, Registry.load(path))))
        edited = dict(freeze, instances=[dict(freeze["instances"][0], scenario_json_sha256="0" * 64)])
        self.assertTrue(any("set_sha256" in p for p in sp.verify_freeze(edited, self.repo.root, Registry.load(path))))

    def test_only_held_out_non_empty_splits_are_frozen(self):
        _, path = self.frozen()
        reg = Registry.load(path)
        with self.assertRaises(ValueError):
            sp.freeze_record(reg, path, "DEV", commit="c", frozen_at="d", root=self.repo.root)
        with self.assertRaises(ValueError):
            sp.freeze_record(reg, path, "VAL", commit="c", frozen_at="d", root=self.repo.root)
        with self.assertRaises(ValueError):
            sp.freeze_record(reg, path, "TEST", commit="", frozen_at="d", root=self.repo.root)


class TestCli(_Case):

    def run_cli(self):
        from scripts.check_data_split import main
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["--root", str(self.repo.root), "--out-root", str(self.repo.root / "out")])
        (report,) = list((self.repo.root / "out").rglob("data_split_report.json"))
        return code, json.loads(report.read_text(encoding="utf-8")), buf.getvalue()

    def test_clean_split_exit_0(self):
        self.repo.register(self.repo.scenario("scenarios/dev/scenario_a.json"), "DEV", "fcorr01")
        code, report, out = self.run_cli()
        self.assertEqual(code, 0, out)
        self.assertEqual(report["held_out_shortfall"]["TEST"], {"have": 0, "target": 75, "missing": 75})

    def test_leak_exit_1(self):
        self.repo.register(self.repo.scenario("scenarios/dev/scenario_a.json"), "DEV", "fcorr01")
        self.repo.register(self.repo.scenario("scenarios/test/scenario_t.json"), "TEST", "fcorr09")
        code, report, out = self.run_cli()
        self.assertEqual(code, 1)
        self.assertIn("LEAK: DEV-TEST EXACT", out)


class TestRealRepository(unittest.TestCase):
    """Read-only checks on the actual scenarios/ and registry/."""

    @classmethod
    def setUpClass(cls):
        cls.registry_path = PROJECT_ROOT / "registry" / "instances_v1.jsonl"
        cls.before = hashlib.sha256(cls.registry_path.read_bytes()).hexdigest()
        cls.report = inv.build_report(PROJECT_ROOT)

    def test_registry_file_untouched(self):
        self.assertEqual(hashlib.sha256(self.registry_path.read_bytes()).hexdigest(), self.before)
        self.assertEqual(self.report["registry_sha256"], self.before)

    def test_no_held_out_data_and_no_unknown(self):
        self.assertEqual({k: self.report["role_counts"][k] for k in ("VALIDATION", "TEST", "UNKNOWN")},
                         {"VALIDATION": 0, "TEST": 0, "UNKNOWN": 0})
        self.assertEqual((self.report["leaks"], self.report["errors"]), ([], []))

    def test_every_registered_dev_file_stays_dev(self):
        reg = Registry.load(self.registry_path)
        roles = {x["source_path"]: x["data_role"] for x in self.report["records"] if x["kind"] == "FILE"}
        for rec in reg.records:
            self.assertEqual(roles[rec["source_path"]], "DEV", rec["instance_id"])
        self.assertEqual(len(roles), len(list((PROJECT_ROOT / "scenarios").rglob("scenario_*.json"))))

    def test_unregistered_files_have_prereg_evidence(self):
        for x in self.report["records"]:
            if x["kind"] == "FILE" and x["registry_instance_id"] is None:
                self.assertEqual(x["data_role"], "DEV", x["source_path"])
                self.assertIn("split_intent", x["role_evidence"])

    def test_loader_agrees_with_fingerprints(self):
        from scripts.run_il import load_scenario
        for x in self.report["records"]:
            if x["kind"] != "FILE":
                continue
            grid, starts, goals = load_scenario(PROJECT_ROOT / x["source_path"])
            ids = sorted(starts)
            self.assertEqual([list(starts[a]) for a in ids], x["starts_rc"], x["source_path"])
            self.assertEqual([list(goals[a]) for a in ids], x["goals_rc"], x["source_path"])


if __name__ == "__main__":
    unittest.main()
