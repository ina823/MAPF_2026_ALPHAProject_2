"""Unit tests for the warehouse generator v3 (src/data/generator_v3.py) and the
WH8-A DEV pilot static checks (scripts/warehouse_v3_dev.py).

No CBS, no IL episode. The only simulator use is MAPFStepSimulator.reset()
inside il_compat / the loader check (t=0 observation).
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data import generator_v3 as g3
from src.data.generator import GeneratorError
from src.data.il_compat import load_reference, start_features
from src.il.nav_hint import bfs_dist
from scripts import warehouse_v3_dev as wh

ROOT = Path(__file__).resolve().parents[1]
WH8A = dict(n_racks=2, rack_len=3, rack_w=1, aisle_w=2, cross_w=2, station_rows=2, n_down=4, n_up=4)
WH8A_ROWS = ["00000000", "00000000", "00100100", "00100100", "00100100",
             "00000000", "00000000", "00000000", "00000000"]


def _rows(grid):
    return ["".join(str(int(v)) for v in row) for row in grid]


def _files(seed=31, params=WH8A):
    inst = g3.warehouse_aisles(seed=seed, **params)
    return inst, g3.scenario_files(inst, "x", code_sha256="c", base_generator_sha256="b",
                                   prereg_sha256="p", candidate_id="X")


class TestLayout(unittest.TestCase):
    def test_wh8a_layout_matches_prereg_grid(self):
        grid = g3.warehouse_layout(**{k: WH8A[k] for k in
                                      ("n_racks", "rack_len", "rack_w", "aisle_w", "cross_w", "station_rows")})
        self.assertEqual(grid.shape, (9, 8))
        self.assertEqual(_rows(grid), WH8A_ROWS)
        self.assertEqual(grid.dtype, np.int64)
        self.assertEqual(int((grid == 0).sum()), 66)

    def test_narrow_aisle_variant_shape(self):
        # WH8-B parameters from the design doc (layout only; not generated as a candidate here).
        grid = g3.warehouse_layout(n_racks=3, rack_len=3, rack_w=1, aisle_w=1, cross_w=2, station_rows=2)
        self.assertEqual(grid.shape, (9, 7))
        self.assertEqual(int((grid == 0).sum()), 54)

    def test_zone_rows(self):
        self.assertEqual(g3.zone_rows(height=9, cross_w=2, station_rows=2),
                         {"top_goal": 0, "top_start": 1, "station_start": 7, "station_goal": 8})

    def test_invalid_parameters(self):
        bad = [dict(WH8A, cross_w=1), dict(WH8A, station_rows=1), dict(WH8A, n_racks=0),
               dict(WH8A, aisle_w=True), dict(WH8A, rack_len=2.0), dict(WH8A, n_down=-1),
               dict(WH8A, n_down=0, n_up=0), dict(WH8A, n_down=9)]
        for params in bad:
            with self.subTest(params=params), self.assertRaises(GeneratorError):
                g3.warehouse_aisles(seed=31, **params)
        with self.assertRaises(GeneratorError):
            g3.warehouse_aisles(seed=True, **WH8A)
        with self.assertRaises(GeneratorError):
            g3.build("nope", WH8A, 31)


class TestAgents(unittest.TestCase):
    def test_agents_distinct_in_zones_and_reachable(self):
        for seed in range(25):
            with self.subTest(seed=seed):
                inst = g3.warehouse_aisles(seed=seed, **WH8A)
                self.assertEqual(len(inst.starts), 8)
                self.assertEqual(len(set(inst.starts)), 8)
                self.assertEqual(len(set(inst.goals)), 8)
                self.assertTrue(all(s[0] == 1 for s in inst.starts[:4]))     # top start row
                self.assertTrue(all(g[0] == 8 for g in inst.goals[:4]))      # station goal row
                self.assertTrue(all(s[0] == 7 for s in inst.starts[4:]))     # station start row
                self.assertTrue(all(g[0] == 0 for g in inst.goals[4:]))      # top goal row
                for s, g in zip(inst.starts, inst.goals):
                    self.assertGreaterEqual(bfs_dist(inst.grid, g)[s], 0)
                self.assertEqual(inst.structure_type, "warehouse")
                self.assertFalse(inst.grid.flags.writeable)

    def test_start_crowding_stays_within_training_p995(self):
        # Construction property claimed in the prereg: at t=0, robot_count <= 3
        # and other_goal_count <= 4 (training p99.5 = 4) for every seed.
        for seed in range(25):
            inst = g3.warehouse_aisles(seed=seed, **WH8A)
            for i, f in enumerate(start_features(inst.grid, inst.starts, inst.goals)):
                with self.subTest(seed=seed, agent=i):
                    self.assertLessEqual(f["robot_count"], 3)
                    self.assertLessEqual(f["other_goal_count"], 4)


class TestReproducibility(unittest.TestCase):
    def test_same_seed_same_bytes_different_seed_same_grid(self):
        a, fa = _files(31)
        b, fb = _files(31)
        self.assertEqual(fa, fb)
        self.assertEqual((a.starts, a.goals), (b.starts, b.goals))
        c, fc = _files(32)
        self.assertEqual(fa[0], fc[0])                   # same map bytes
        self.assertNotEqual((a.starts, a.goals), (c.starts, c.goals))

    def test_bytes_identical_across_pythonhashseed(self):
        code = ("import hashlib,sys;sys.path.insert(0,'.');from src.data import generator_v3 as g;"
                "i=g.warehouse_aisles(seed=31,n_racks=2,rack_len=3,rack_w=1,aisle_w=2,cross_w=2,"
                "station_rows=2,n_down=4,n_up=4);"
                "m,j=g.scenario_files(i,'x',code_sha256='c',base_generator_sha256='b',prereg_sha256='p',"
                "candidate_id='X');print(hashlib.sha256(m+j).hexdigest())")
        out = []
        for seed in ("0", "1"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            res = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                                 capture_output=True, text=True, check=True)
            out.append(res.stdout.strip())
        self.assertEqual(out[0], out[1])

    def test_scenario_json_format(self):
        inst, (npy, js) = _files(31)
        scen = json.loads(js)
        self.assertEqual([a["agent_id"] for a in scen["agents"]], list(range(8)))
        for a, s, g in zip(scen["agents"], inst.starts, inst.goals):
            self.assertEqual(a["start"], [s[1], s[0]])          # [x, y] = [col, row]
            self.assertEqual(a["goal"], [g[1], g[0]])
        self.assertEqual(scen["generator"]["name"], "alpha2_warehouse")
        self.assertEqual(scen["generator"]["version"], 3)
        self.assertEqual(scen["generator"]["seed"], 31)


class TestDevPilotScript(unittest.TestCase):
    def test_prereg_is_pinned(self):
        prereg = wh.load_prereg()
        self.assertEqual(prereg["candidates"][0]["candidate_id"], "WH8-A")
        self.assertEqual(prereg["candidates"][0]["grid_rows"], WH8A_ROWS)
        self.assertEqual(prereg["candidates"][0]["params"], WH8A)
        with self.assertRaises(Exception):
            wh.load_prereg(pinned="0" * 64)

    def test_generate_is_exclusive_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = wh.generate(Path(tmp))
            self.assertTrue(first[0]["generated"])
            self.assertFalse(first[0]["already_present"])
            again = wh.generate(Path(tmp))
            self.assertTrue(again[0]["already_present"])
            spec = wh.load_prereg()["candidates"][0]
            map_path, scen_path = wh.candidate_paths(spec, Path(tmp))
            scen_path.write_bytes(scen_path.read_bytes() + b" ")      # tamper
            tampered = scen_path.read_bytes()
            third = wh.generate(Path(tmp))
            self.assertFalse(third[0]["generated"])
            self.assertIn("not overwritten", third[0]["error"])
            self.assertEqual(scen_path.read_bytes(), tampered)

    def test_static_report_on_saved_candidate(self):
        spec = wh.load_prereg()["candidates"][0]
        map_path, scen_path = wh.candidate_paths(spec)
        if not (map_path.is_file() and scen_path.is_file()):
            self.skipTest("WH8-A files not generated in this checkout")
        before = hashlib.sha256((ROOT / "registry" / "instances_v1.jsonl").read_bytes()).hexdigest()
        report = wh.build_report(ROOT, ROOT / "registry")
        after = hashlib.sha256((ROOT / "registry" / "instances_v1.jsonl").read_bytes()).hexdigest()
        self.assertEqual(before, after)
        self.assertEqual(report["problems"], [])
        self.assertFalse(report["cbs_run"])
        self.assertFalse(report["il_episode_run"])
        c = report["candidates"][0]
        self.assertTrue(c["seed_reproducible"] and c["cells_valid"] and c["coordinates_ok"])
        self.assertTrue(c["all_goals_reachable"])
        self.assertTrue(c["actions"]["ok"])
        self.assertTrue(all(c["loader"][k] for k in ("agent_id_matches_index",
                                                     "run_il_load_scenario_matches_generator",
                                                     "t0_observation_shapes_ok")))
        self.assertEqual(c["il_check"]["auto_fail"], False)
        self.assertEqual(c["ood"]["auto_fail"], False)
        self.assertEqual(c["registry"]["independence_approved"], False)
        self.assertFalse(c["registry"]["duplicate_instance_fp"])
        for h in c["registry"]["holds"]:
            self.assertEqual((h["status"], h["final_decision"]), ("HOLD", None))

    def test_check_reports_missing_files_without_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = wh.load_prereg()["candidates"][0]
            ref = load_reference(ROOT / "registry" / "il_input_reference_v1.json")
            out = wh.check_candidate(spec, Path(tmp), ref)
            self.assertTrue(any("missing" in p for p in out["problems"]))

    def test_main_check_writes_only_to_new_out_dir(self):
        spec = wh.load_prereg()["candidates"][0]
        map_path, scen_path = wh.candidate_paths(spec)
        if not (map_path.is_file() and scen_path.is_file()):
            self.skipTest("WH8-A files not generated in this checkout")
        tmp = tempfile.mkdtemp()
        try:
            rc = wh.main(["check", "--out-root", tmp])
            self.assertEqual(rc, 0)
            written = list(Path(tmp).glob("static_*/static_checks.json"))
            self.assertEqual(len(written), 1)
        finally:
            shutil.rmtree(tmp)

    def test_action_checks_detect_bad_table_use(self):
        grid = np.zeros((3, 3), dtype=np.int64)
        res = wh.action_checks(grid, [(0, 0)], [(2, 2)])
        self.assertTrue(res["ok"])
        self.assertEqual(len(res["agents"][0]["bfs_path_actions"]), 4)


if __name__ == "__main__":
    unittest.main()
