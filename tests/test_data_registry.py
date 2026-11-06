"""src/data/registry.py: relations, registration decisions, safe JSONL writes."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.data import fingerprint as fp
from src.data import registry as rg

G = np.array([[0, 0, 0, 1, 0],
              [0, 1, 0, 0, 0],
              [0, 1, 1, 0, 1],
              [0, 0, 0, 0, 0]], dtype=np.int64)
STARTS = [(0, 0), (3, 4)]
GOALS = [(3, 0), (0, 4)]
_counter = [0]


def cand(grid=G, starts=STARTS, goals=GOALS, *, source=None, structure="corridor",
         descriptor=None, ids_ok=True):
    _counter[0] += 1
    rec = fp.compute_fingerprints(grid, starts, goals)
    rec.update({
        "source_path": source or f"synthetic/scenario_{_counter[0]}.json",
        "scenario_json_sha256": f"sha-{source or _counter[0]}",
        "structure_type": structure,
        "structure_descriptor": descriptor,
        "agent_id_matches_index": ids_ok,
    })
    return rec


def rotated(k=1, flip=False):
    g = fp.normalize_grid(G)
    return (fp.transform_grid(g, k, flip), [fp.transform_cell(s, g.shape, k, flip) for s in STARTS],
            [fp.transform_cell(s, g.shape, k, flip) for s in GOALS])


def near_grid():
    g = G.copy()
    g[2, 4] = 0                       # one wall cell opened
    return g


def relations(a, b):
    return [r["relation"] for r in rg.relate(a, b)]


class TestRelations(unittest.TestCase):
    def test_exact(self):
        self.assertEqual(relations(cand(), cand()), [rg.EXACT])

    def test_relabel_is_not_exact(self):
        rels = rg.relate(cand(), cand(G, STARTS[::-1], GOALS[::-1]))
        self.assertEqual([r["relation"] for r in rels], [rg.RELABEL_EQUIV])
        self.assertFalse(rels[0]["details"]["id_order_equal"])

    def test_trim_equiv(self):
        padded = np.pad(G, ((2, 0), (0, 1)), constant_values=1)
        rels = rg.relate(cand(), cand(padded, [(r + 2, c) for r, c in STARTS],
                                      [(r + 2, c) for r, c in GOALS]))
        self.assertEqual([r["relation"] for r in rels], [rg.TRIM_EQUIV])
        self.assertTrue(rels[0]["details"]["id_order_equal"])

    def test_rotation_with_agents_is_symmetry_not_same_instance(self):
        for k, flip in ((1, False), (2, False), (0, True), (3, True)):
            rels = rg.relate(cand(), cand(*rotated(k, flip)))
            self.assertEqual([r["relation"] for r in rels], [rg.SYMMETRY_EQUIV], (k, flip))
            self.assertEqual(rels[0]["status"], rg.CONFIRMED)

    def test_rotated_map_with_untransformed_agents_is_same_map(self):
        g, _, _ = rotated(2)
        self.assertEqual(relations(cand(), cand(g, [(0, 4)], [(3, 4)])), [rg.SAME_MAP])

    def test_same_map_other_start_goal(self):
        rels = rg.relate(cand(), cand(G, [(0, 0)], [(3, 0)]))
        self.assertEqual([r["relation"] for r in rels], [rg.SAME_MAP])
        self.assertEqual(rels[0]["details"]["shared_start_goal_pairs"], 1)
        self.assertEqual(rels[0]["details"]["num_agents"], [1, 2])

    def test_review_relations(self):
        big = np.ones((7, 8), dtype=int)
        big[1:5, 2:7] = G
        big[5, 2:7] = 0
        r = rg.relate(cand(), cand(big, [(1, 2)], [(4, 2)]))
        self.assertEqual([x["relation"] for x in r], [rg.SUBMAP])
        self.assertEqual(r[0]["status"], rg.REVIEW)
        self.assertEqual(relations(cand(), cand(near_grid(), STARTS, GOALS)), [rg.NEAR_MAP])
        other = np.zeros((3, 3), dtype=int)
        d = {"type": "corridor", "length": 5}
        self.assertEqual(relations(cand(descriptor=d), cand(other, [(0, 0)], [(2, 2)], descriptor=d)),
                         [rg.STRUCT_MATCH])

    def test_small_window_is_not_submap(self):
        tiny = np.zeros((2, 2), dtype=int)
        self.assertEqual(relations(cand(), cand(tiny, [(0, 0)], [(1, 1)])), [])

    def test_unrelated(self):
        other = np.zeros((6, 6), dtype=int)
        other[2, 1:5] = 1
        self.assertEqual(relations(cand(), cand(other, [(0, 0)], [(5, 5)])), [])


class TestDecisions(unittest.TestCase):
    def setUp(self):
        self.reg = rg.Registry()
        self.dev = self.reg.register(cand(source="dev/a.json"), "DEV", registered_at="2026-11-05")

    def test_new_family_and_ids(self):
        self.assertEqual(self.dev["family_id"], "fcorr01")
        self.assertEqual(self.dev["instance_id"], "DEV_fcorr01_001")
        other = np.zeros((6, 6), dtype=int)
        rec = self.reg.register(cand(other, [(0, 0)], [(5, 5)], structure="empty"), "DEV",
                                registered_at="d")
        self.assertEqual(rec["family_id"], "femp01")

    def test_same_file_again(self):
        d = self.reg.evaluate(cand(source="dev/a.json"), "DEV")
        self.assertEqual(d.status, rg.REJECT)
        self.assertTrue(any(r.startswith("ALREADY_REGISTERED") for r in d.reasons))

    def test_split_change(self):
        d = self.reg.evaluate(cand(source="dev/a.json"), "TEST")
        self.assertEqual(d.status, rg.REJECT)
        self.assertTrue(any(r.startswith("SPLIT_CHANGE") for r in d.reasons))

    def test_same_content_other_file_is_duplicate(self):
        d = self.reg.evaluate(cand(source="dev/copy.json"), "DEV")
        self.assertEqual(d.status, rg.REJECT)
        self.assertTrue(any(r.startswith("DUPLICATE_INSTANCE") for r in d.reasons))
        with self.assertRaises(rg.RegistryError):
            self.reg.register(cand(source="dev/copy.json"), "DEV", registered_at="d")

    def test_relabel_in_dev_keeps_its_own_id(self):
        rec = self.reg.register(cand(G, STARTS[::-1], GOALS[::-1]), "DEV", registered_at="d")
        self.assertEqual(rec["family_id"], self.dev["family_id"])
        self.assertEqual(rec["instance_id"], "DEV_fcorr01_002")
        self.assertEqual(rec["relations"][0]["relation"], rg.RELABEL_EQUIV)

    def test_family_leak_refused(self):
        padded = np.pad(G, 1, constant_values=1)
        leaks = {
            "SAME_MAP": cand(G, [(0, 0)], [(3, 0)]),
            "SYMMETRY": cand(*rotated(1)),
            "RELABEL": cand(G, STARTS[::-1], GOALS[::-1]),
            "TRIM": cand(padded, [(r + 1, c + 1) for r, c in STARTS], [(r + 1, c + 1) for r, c in GOALS]),
        }
        for split in ("VAL", "TEST"):
            for name, c in leaks.items():
                d = self.reg.evaluate(c, split)
                self.assertEqual(d.status, rg.REJECT, (split, name))
                self.assertTrue(any(r.startswith("FAMILY_LEAK") for r in d.reasons), (split, name))

    def test_pilot_shares_development_domain(self):
        d = self.reg.evaluate(cand(G, [(0, 0)], [(3, 0)]), "PILOT")
        self.assertEqual((d.status, d.family_id), (rg.ACCEPT, "fcorr01"))

    def test_unresolved_review_holds_held_out(self):
        c = cand(near_grid(), STARTS, GOALS)
        for split in ("VAL", "TEST"):
            d = self.reg.evaluate(c, split)
            self.assertEqual(d.status, rg.HOLD, split)
            self.assertTrue(d.reasons[0].startswith("UNRESOLVED_REVIEW: NEAR_MAP"))
        with self.assertRaises(rg.RegistryError):
            self.reg.register(c, "VAL", registered_at="d")

    def test_review_in_dev_is_recorded_but_not_merged(self):
        rec = self.reg.register(cand(near_grid(), STARTS, GOALS), "DEV", registered_at="d")
        self.assertNotEqual(rec["family_id"], self.dev["family_id"])
        self.assertEqual([(r["relation"], r["status"]) for r in rec["relations"]],
                         [(rg.NEAR_MAP, rg.REVIEW)])

    def test_review_decisions(self):
        c = cand(near_grid(), STARTS, GOALS)
        self.reg.reviews.add(rg.instance_key(self.dev), rg.instance_key(c), rg.NEAR_MAP,
                             rg.UNRELATED, "reviewer", "door differs on purpose")
        d = self.reg.evaluate(c, "VAL")
        self.assertEqual(d.status, rg.ACCEPT)
        self.assertNotEqual(d.family_id, self.dev["family_id"])
        self.reg.reviews.add(rg.instance_key(self.dev), rg.instance_key(c), rg.NEAR_MAP,
                             rg.RELATED, "reviewer", "same room layout")
        d = self.reg.evaluate(c, "VAL")
        self.assertEqual(d.status, rg.REJECT)
        self.assertTrue(any(r.startswith("FAMILY_LEAK: NEAR_MAP") for r in d.reasons))
        with self.assertRaises(rg.RegistryError):
            self.reg.reviews.add("a", "b", rg.SAME_MAP, rg.RELATED, "r", "x")

    def test_family_conflict(self):
        other = np.zeros((6, 6), dtype=int)
        b = self.reg.register(cand(other, [(0, 0)], [(5, 5)], structure="empty"), "DEV", registered_at="d")
        c = cand(other, [(1, 1)], [(4, 4)])                 # SAME_MAP with b
        c_near = dict(c)
        self.reg.reviews.add(rg.instance_key(self.dev), rg.instance_key(c), rg.NEAR_MAP,
                             rg.RELATED, "r", "x")
        with patch.object(rg, "relate", side_effect=lambda x, y: (
                [rg._rel(rg.NEAR_MAP, rg.REVIEW)] if x is self.dev else
                [rg._rel(rg.SAME_MAP, rg.CONFIRMED)] if x is b else [])):
            d = self.reg.evaluate(c_near, "DEV")
        self.assertEqual(d.status, rg.REJECT)
        self.assertTrue(any(r.startswith("FAMILY_CONFLICT") for r in d.reasons))

    def test_held_out_requires_agent_id_index(self):
        other = np.zeros((6, 6), dtype=int)
        d = self.reg.evaluate(cand(other, [(0, 0)], [(5, 5)], ids_ok=False), "TEST")
        self.assertEqual(d.status, rg.REJECT)
        self.assertTrue(any(r.startswith("AGENT_ID_NOT_INDEX") for r in d.reasons))

    def test_new_held_out_family_accepted(self):
        other = np.zeros((6, 6), dtype=int)
        other[3, 1:5] = 1
        rec = self.reg.register(cand(other, [(0, 0)], [(5, 5)], structure="bottleneck"), "VAL",
                                registered_at="d")
        self.assertEqual(rec["instance_id"], "VAL_fbot01_001")


def open_map(i):
    """A family-free 6x6 map, distinct per i (one wall cell moved)."""
    g = np.zeros((6, 6), dtype=int)
    g[2, 1:5] = 1
    g[4, i % 6] = 1
    return g


class TestRulesV2(unittest.TestCase):
    def setUp(self):
        self.reg = rg.Registry()
        self.dev = self.reg.register(cand(source="dev/a.json"), "DEV", registered_at="d")

    def test_instance_fp_is_exact_identity_only(self):
        a = cand()
        self.assertEqual(rg.instance_fp(a), rg.instance_fp(cand()))
        self.assertNotEqual(rg.instance_fp(a), rg.instance_fp(cand(G, STARTS[::-1], GOALS[::-1])))
        self.assertNotEqual(rg.instance_fp(a), rg.instance_fp(cand(*rotated(1))))
        padded = np.pad(G, 1, constant_values=1)
        self.assertNotEqual(rg.instance_fp(a), rg.instance_fp(
            cand(padded, [(r + 1, c + 1) for r, c in STARTS], [(r + 1, c + 1) for r, c in GOALS])))

    def test_family_cap_for_new_candidates(self):
        g = open_map(0)
        for n, (s, t) in enumerate((((0, 0), (5, 5)), ((0, 1), (5, 4)), ((0, 2), (5, 3)))):
            rec = self.reg.register(cand(g, [s], [t], structure="bottleneck"), "VAL", registered_at="d")
            self.assertEqual(rec["instance_id"], f"VAL_fbot01_{n + 1:03d}")
        d = self.reg.evaluate(cand(g, [(0, 3)], [(5, 2)]), "VAL")
        self.assertEqual(d.status, rg.REJECT)
        self.assertTrue(any(r.startswith("FAMILY_CAP") for r in d.reasons))

    def test_family_cap_not_retroactive_for_legacy_dev(self):
        for i in range(3):
            self.reg.register(cand(G, [(0, 0)], [(3, i)]), "DEV", registered_at="d", apply_family_cap=False)
        self.assertEqual(sum(r["family_id"] == "fcorr01" for r in self.reg.records), 4)
        self.assertEqual(self.reg.evaluate(cand(G, [(0, 0)], [(1, 0)]), "DEV").status, rg.REJECT)
        self.assertEqual(self.reg.consistency_errors(), [])        # legacy DEV over cap is fine

    def test_consistency_flags_held_out_over_cap(self):
        g = open_map(1)
        recs = list(self.reg.records)
        for i in range(4):
            c = cand(g, [(0, i)], [(5, 5 - i)], structure="bottleneck")
            c.update(instance_id=f"VAL_fbot01_{i + 1:03d}", family_id="fbot01", split="VAL",
                     relations=[{"other": f"VAL_fbot01_{j + 1:03d}", "relation": rg.SAME_MAP,
                                 "status": rg.CONFIRMED, "details": {}} for j in range(i)])
            recs.append(c)
        errors = rg.Registry(recs).consistency_errors()
        self.assertTrue(any(e.startswith("FAMILY_CAP exceeded: fbot01") for e in errors))

    def test_struct_match_never_auto_enters_test(self):
        d = {"structure": "corridor_bays", "length": 15}
        self.reg = rg.Registry()
        dev = self.reg.register(cand(descriptor=d), "DEV", registered_at="d")
        c = cand(open_map(2), [(0, 0)], [(5, 5)], descriptor=d)
        self.assertEqual(self.reg.evaluate(c, "VAL").status, rg.HOLD)       # unresolved
        d_test = self.reg.evaluate(c, "TEST")
        self.assertEqual(d_test.status, rg.REJECT)
        self.reg.reviews.add(rg.instance_fp(dev), rg.instance_fp(c), rg.STRUCT_MATCH,
                             rg.UNRELATED, "reviewer", "different layout")
        self.assertEqual(self.reg.evaluate(c, "VAL").status, rg.ACCEPT)     # approved for VAL only
        d_test = self.reg.evaluate(c, "TEST")
        self.assertEqual(d_test.status, rg.REJECT)
        self.assertTrue(any(r.startswith("STRUCT_MATCH_TEST_FORBIDDEN") for r in d_test.reasons))

    def test_unresolved_review_with_pilot_blocks_test(self):
        pilot = self.reg.register(cand(open_map(3), [(0, 0)], [(5, 5)], structure="empty"), "PILOT",
                                  registered_at="d")
        g = open_map(3).copy()
        g[0, 5] = 1                                                          # NEAR_MAP of the pilot
        d = self.reg.evaluate(cand(g, [(0, 0)], [(5, 5)]), "TEST")
        self.assertEqual(d.status, rg.HOLD)
        self.assertIn(pilot["instance_id"], d.reasons[0])


class TestFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.reg = rg.Registry()
        self.reg.register(cand(source="dev/a.json"), "DEV", registered_at="d")
        self.reg.register(cand(G, [(0, 0)], [(3, 0)], source="dev/b.json"), "DEV", registered_at="d")

    def tearDown(self):
        self.tmp.cleanup()

    def test_roundtrip_and_consistency(self):
        path = self.dir / "instances_v1.jsonl"
        rg.create_files_exclusive([(path, self.reg.to_jsonl())])
        loaded = rg.Registry.load(path)
        self.assertEqual(loaded.records, self.reg.records)
        self.assertEqual(loaded.to_jsonl(), path.read_bytes())
        self.assertEqual(loaded.consistency_errors(), [])
        self.assertEqual(loaded.family_summary()["relation_counts"], {"SAME_MAP": 1})

    def test_load_rejects_bad_files(self):
        good = self.reg.to_jsonl()
        lines = good.decode().splitlines()
        cases = {
            "partial": good[:-5],
            "invalid": (lines[0] + "\n{not json\n").encode(),
            "empty_line": (lines[0] + "\n\n").encode(),
            "duplicate": (lines[0] + "\n" + lines[0] + "\n").encode(),
        }
        bad_schema = json.loads(lines[0])
        bad_schema["fp_schema"] = 99
        cases["schema"] = (json.dumps(bad_schema) + "\n").encode()
        moved = json.loads(lines[0])
        moved["split"] = "TEST"                     # split changed but id kept
        cases["split_edit"] = (json.dumps(moved) + "\n").encode()
        for name, data in cases.items():
            p = self.dir / f"{name}.jsonl"
            p.write_bytes(data)
            with self.assertRaises(rg.RegistryError, msg=name):
                rg.Registry.load(p)

    def test_tampered_relation_detected(self):
        recs = [dict(r) for r in self.reg.records]
        recs[1] = dict(recs[1], relations=[])
        self.assertTrue(rg.Registry(recs).consistency_errors())
        recs = [dict(r) for r in self.reg.records]
        recs[1] = dict(recs[1], family_id="fcorr09", instance_id="DEV_fcorr09_001")
        errors = rg.Registry(recs).consistency_errors()
        self.assertTrue(any("different families" in e for e in errors))

    def test_create_refuses_existing_and_writes_nothing(self):
        a, b = self.dir / "a.jsonl", self.dir / "b.json"
        b.write_bytes(b"keep")
        with self.assertRaises(rg.RegistryError):
            rg.create_files_exclusive([(a, b"x\n"), (b, b"y\n")])
        self.assertFalse(a.exists())
        self.assertEqual(b.read_bytes(), b"keep")
        self.assertEqual([p.name for p in self.dir.iterdir()], ["b.json"])

    def test_create_rolls_back_on_second_failure(self):
        a, b = self.dir / "a.jsonl", self.dir / "b.json"
        real_link = os.link
        calls = []

        def flaky(src, dst):
            calls.append(dst)
            if len(calls) == 2:
                raise OSError("simulated failure")
            real_link(src, dst)

        with patch.object(rg.os, "link", side_effect=flaky):
            with self.assertRaises(OSError):
                rg.create_files_exclusive([(a, b"x\n"), (b, b"y\n")])
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_append_to_file(self):
        path = self.dir / "instances_v1.jsonl"
        rg.create_files_exclusive([(path, self.reg.to_jsonl())])
        loaded = rg.Registry.load(path)
        n = len(loaded.records)
        loaded.register(cand(open_map(4), [(0, 0)], [(5, 5)], structure="empty"), "PILOT", registered_at="d")
        loaded.append_to_file(path, n)
        self.assertTrue(path.read_bytes().startswith(self.reg.to_jsonl()))
        self.assertEqual(len(rg.Registry.load(path).records), n + 1)
        # editing an existing record in memory is refused
        again = rg.Registry.load(path)
        again.records[0] = dict(again.records[0], usage_status="EDITED")
        with self.assertRaises(rg.RegistryError):
            again.append_to_file(path, len(again.records))

    def test_replace_checked(self):
        path = self.dir / "instances_v1.jsonl"
        rg.create_files_exclusive([(path, self.reg.to_jsonl())])
        loaded = rg.Registry.load(path)
        rg.replace_file_checked(path, loaded.to_jsonl(), loaded.source_sha256)
        path.write_bytes(b"changed by someone else\n")
        with self.assertRaises(rg.RegistryError):
            rg.replace_file_checked(path, loaded.to_jsonl(), loaded.source_sha256)
        self.assertEqual(path.read_bytes(), b"changed by someone else\n")
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ["instances_v1.jsonl"])


if __name__ == "__main__":
    unittest.main()
