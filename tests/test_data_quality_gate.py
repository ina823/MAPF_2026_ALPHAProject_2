"""src/data/quality_gate.py + scripts/check_pilot_gate.py (PILOT data-quality gate v0)."""

import copy
import hashlib
import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np

from scripts import check_pilot_gate as cpg
from src.data import quality_gate as qg
from src.data.fingerprint import compute_fingerprints
from src.data.registry import REVIEW, Registry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REG_DIR = PROJECT_ROOT / "registry"


def load_all():
    reg = Registry.load(REG_DIR / "instances_v1.jsonl")
    ledger = qg.load_jsonl(REG_DIR / cpg.LEDGER)
    solv = json.loads((REG_DIR / cpg.SOLVABILITY).read_text(encoding="utf-8"))
    return reg, ledger, solv


class TestReviewLedger(unittest.TestCase):
    def setUp(self):
        self.reg, self.ledger, _ = load_all()

    def test_committed_ledger_is_five_holds_without_decision(self):
        self.assertEqual(qg.validate_review_ledger(self.ledger, self.reg), [])
        self.assertEqual(len(self.ledger), 5)
        for e in self.ledger:
            self.assertEqual((e["relation"], e["status"], e["final_decision"], e["decision_owner"]),
                             ("SUBMAP", qg.HOLD, None, None))
            self.assertEqual(e["b"], "PILOT_fint02_001")

    def test_a_decision_is_refused_in_v0(self):
        for field, value in (("final_decision", "UNRELATED"), ("final_decision", "RELATED"),
                             ("status", "RELATED"), ("decision_owner", "someone")):
            led = copy.deepcopy(self.ledger)
            led[0][field] = value
            self.assertTrue(qg.validate_review_ledger(led, self.reg), (field, value))

    def test_entry_must_match_registry_and_recomputed_relation(self):
        led = copy.deepcopy(self.ledger)
        led[0]["a_instance_fp"] = "0" * 64
        self.assertTrue(any("a_instance_fp" in p for p in qg.validate_review_ledger(led, self.reg)))
        led = copy.deepcopy(self.ledger)
        led[0]["b"] = "PILOT_fbot02_001"                    # no SUBMAP with this record
        self.assertTrue(any("not recomputed" in p for p in qg.validate_review_ledger(led, self.reg)))
        led = copy.deepcopy(self.ledger)
        led[0]["evidence"]["submap_placements"] = 8
        self.assertTrue(any("submap_placements" in p for p in qg.validate_review_ledger(led, self.reg)))

    def test_every_unresolved_review_needs_an_entry(self):
        problems = qg.validate_review_ledger(self.ledger[1:], self.reg)
        self.assertEqual(len(problems), 1)
        self.assertIn("DEV_fcorr01_001", problems[0])

    def test_ledger_does_not_resolve_or_merge(self):
        before = {p: (REG_DIR / p).read_bytes() for p in ("instances_v1.jsonl", "families_v1.json")}
        _, _, solv = load_all()
        qg.gate_report(self.reg, self.ledger, solv, {})
        reviews = [r for r in self.reg.all_relations() if r["status"] == REVIEW]
        self.assertEqual(len(reviews), 5)
        self.assertTrue(all(r["review_decision"] is None for r in reviews))
        self.assertEqual(self.reg.by_id()["PILOT_fint02_001"]["family_id"], "fint02")
        self.assertEqual(self.reg.summary_json(), before["families_v1.json"])
        for p, data in before.items():
            self.assertEqual((REG_DIR / p).read_bytes(), data)

    def test_submap_placements_are_distinct(self):
        ids = self.reg.by_id()
        ev = qg.submap_evidence(ids["DEV_fcorr01_001"], ids["PILOT_fint02_001"])
        self.assertEqual(ev, {"smaller_shape": [2, 11], "larger_shape": [9, 11], "submap_placements": 4})


class TestClassifyOod(unittest.TestCase):
    REF = {"source": {"sha256": "x"}, "features": {
        "wall_count": {"p0.5": 0.0, "p99.5": 17.0, "min": 0.0, "max": 20.0},
        "goal_col": {"p0.5": -12.0, "p99.5": 13.0, "min": -38.0, "max": 44.0}}}

    def check(self, *warnings):
        return {"warnings": list(warnings), "auto_fail": False}

    def test_in_range(self):
        res = qg.classify_ood(self.check(), self.REF)
        self.assertEqual((res["ood_class"], res["items"], res["auto_fail"]), ("IN_RANGE", [], False))

    def test_tail_versus_beyond_observed(self):
        tail = {"kind": "START", "agent": 0, "feature": "wall_count", "value": 20.0, "bounds": [0, 17]}
        res = qg.classify_ood(self.check(tail), self.REF)
        self.assertEqual((res["ood_class"], res["items"][0]["severity"]), (qg.OOD_TAIL, qg.TAIL))
        self.assertEqual(res["causes"], ["DENSE_WALL_VIEW"])
        beyond = dict(tail, value=21.0)
        rng = {"kind": "REACHABLE_RANGE", "agent": 1, "feature": "goal_col", "value": [-14.0, 14.0],
               "bounds": [-12, 13]}
        res = qg.classify_ood(self.check(beyond, rng), self.REF)
        self.assertEqual(res["ood_class"], qg.OOD_BEYOND_OBSERVED)
        self.assertEqual([i["severity"] for i in res["items"]], [qg.BEYOND_OBSERVED, qg.TAIL])
        self.assertEqual(res["items"][1]["training_observed"], [-38.0, 44.0])
        self.assertFalse(res["auto_fail"])
        self.assertIn("not the Validation exclusion policy", res["threshold_status"])


class TestSolvability(unittest.TestCase):
    def setUp(self):
        self.reg, _, self.solv = load_all()
        self.recs = {r["instance_id"]: r for r in self.solv["records"]}

    def test_committed_record_is_consistent(self):
        self.assertEqual(qg.validate_solvability(self.solv, self.reg), [])

    def test_pilot2_keeps_cbs_timeout_beside_other_witness(self):
        d = qg.derive_solvability(self.recs["PILOT_fbot02_001"])
        self.assertEqual(d, {"cbs_status": "timeout", "cbs_solved": False,
                             "solution_existence": qg.CONFIRMED_BY_OTHER_METHOD,
                             "witness_evidence": "UNARCHIVED"})
        self.assertEqual(self.recs["PILOT_fbot02_001"]["cbs"]["status"], "timeout")
        for iid in ("PILOT_fcorr02_001", "PILOT_fint02_001"):
            self.assertEqual(qg.derive_solvability(self.recs[iid])["solution_existence"], qg.CONFIRMED_BY_CBS)

    def test_derivation_without_witness(self):
        rec = copy.deepcopy(self.recs["PILOT_fbot02_001"])
        rec["other_methods"] = []
        self.assertEqual(qg.derive_solvability(rec)["solution_existence"], qg.UNKNOWN)
        rec["cbs"]["status"] = "no_solution"
        self.assertEqual(qg.derive_solvability(rec)["solution_existence"], qg.NO_SOLUTION_REPORTED_BY_CBS)
        rec = copy.deepcopy(self.recs["PILOT_fbot02_001"])
        rec["other_methods"][0]["path_guard_ok"] = False
        self.assertEqual(qg.derive_solvability(rec)["solution_existence"], qg.UNKNOWN)

    def test_other_method_cannot_become_cbs_success(self):
        def problems_after(mutate):
            doc = copy.deepcopy(self.solv)
            mutate({r["instance_id"]: r for r in doc["records"]}["PILOT_fbot02_001"])
            return qg.validate_solvability(doc, self.reg)
        self.assertTrue(problems_after(lambda r: r.__setitem__("cbs_solved", True)))
        self.assertTrue(problems_after(lambda r: r["cbs"].__setitem__("solution_existence", "CONFIRMED_BY_CBS")))
        self.assertTrue(problems_after(lambda r: r["other_methods"][0].__setitem__("method", "atb033_CBS_longer")))
        self.assertTrue(problems_after(lambda r: r["cbs"].__setitem__("status", "solved")))
        self.assertTrue(problems_after(lambda r: r["other_methods"][0].__setitem__("path_guard_ok", None)))

    def test_every_pilot_needs_a_record(self):
        doc = copy.deepcopy(self.solv)
        doc["records"] = doc["records"][:2]
        self.assertTrue(any("PILOT_fint02_001" in p for p in qg.validate_solvability(doc, self.reg)))

    def test_statuses_match_adapter_contract(self):
        from src.hybrid.replan_boundary import CBS_STATUSES
        self.assertEqual(qg.CBS_STATUSES, CBS_STATUSES)

    def test_source_cross_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rec = copy.deepcopy(self.recs["PILOT_fbot02_001"])
            problems, notes = qg.check_cbs_source(rec, root)
            self.assertEqual(problems, [])
            self.assertEqual(len(notes), 1)                  # missing source -> note only
            src = root / rec["cbs"]["source_file"]
            src.parent.mkdir(parents=True)
            raw = json.dumps({"cbs": [{"pilot": "pilot_rooms_door_6x14", "status": "success",
                                       "latency_sec": 120.07, "path_guard_ok": None}]}).encode()
            src.write_bytes(raw)
            self.assertTrue(qg.check_cbs_source(rec, root)[0])                 # sha mismatch
            rec["cbs"]["source_sha256"] = hashlib.sha256(raw).hexdigest()
            problems, _ = qg.check_cbs_source(rec, root)
            self.assertTrue(any("status" in p for p in problems))              # success != timeout


class TestIndependence(unittest.TestCase):
    def test_existing_submap_is_not_independent(self):
        reg, _, _ = load_all()
        others = Registry([r for r in reg.records if r["instance_id"] != "PILOT_fint02_001"])
        res = qg.independence_check(others, reg.by_id()["PILOT_fint02_001"])
        self.assertFalse(res["independent"])
        self.assertEqual({r["relation"] for r in res["relations"]}, {"SUBMAP"})

    def test_unrelated_open_map_is_independent(self):
        reg, _, _ = load_all()
        grid = np.zeros((7, 9), dtype=np.int64)
        grid[3, 2:7] = 1
        cand = compute_fingerprints(grid, [(0, 0), (6, 8)], [(6, 0), (0, 8)])
        self.assertEqual(qg.independence_check(reg, cand), {"independent": True, "relations": []})


class TestScript(unittest.TestCase):
    def run_main(self, args):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = cpg.main(args)
        return rc, out.getvalue(), err.getvalue()

    def test_committed_records_pass_and_report_flags(self):
        rc, out, _ = self.run_main(["--json"])
        self.assertEqual(rc, 0)
        rep = json.loads(out[:out.rindex("}") + 1])
        by = {i["instance_id"]: i for i in rep["instances"]}
        self.assertEqual(set(by), {"PILOT_fcorr02_001", "PILOT_fbot02_001", "PILOT_fint02_001"})
        self.assertIn("CBS_TIMEOUT", by["PILOT_fbot02_001"]["flags"])
        self.assertIn("WITNESS_UNARCHIVED", by["PILOT_fbot02_001"]["flags"])
        self.assertFalse(by["PILOT_fbot02_001"]["solvability"]["cbs_solved"])
        self.assertIn("REVIEW_HOLD", by["PILOT_fint02_001"]["flags"])
        self.assertTrue(all(i["auto_reject"] is False and i["ood"]["auto_fail"] is False
                            for i in rep["instances"]))

    def test_decision_in_ledger_fails_the_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for f in ("instances_v1.jsonl", "families_v1.json", cpg.LEDGER, cpg.SOLVABILITY, cpg.REFERENCE):
                shutil.copy(REG_DIR / f, d / f)
            lines = (d / cpg.LEDGER).read_text(encoding="utf-8").splitlines(keepends=True)
            e = json.loads(lines[0])
            e["final_decision"] = "UNRELATED"
            lines[0] = json.dumps(e, ensure_ascii=False) + "\n"
            (d / cpg.LEDGER).write_text("".join(lines), encoding="utf-8")
            rc, _, err = self.run_main(["--registry-dir", str(d)])
        self.assertEqual(rc, 1)
        self.assertIn("final_decision", err)


if __name__ == "__main__":
    unittest.main()
