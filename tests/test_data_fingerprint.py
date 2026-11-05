"""src/data/fingerprint.py: normalisation levels, coordinates, D4 transforms."""

import unittest
from pathlib import Path

import numpy as np

from src.common.scenario_loader import load_map_generator_scenario
from src.data import fingerprint as fp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCEN = PROJECT_ROOT / "scenarios"

# asymmetric 4x5 grid: no rotation/reflection maps it onto itself
G = np.array([[0, 0, 0, 1, 0],
              [0, 1, 0, 0, 0],
              [0, 1, 1, 0, 1],
              [0, 0, 0, 0, 0]], dtype=np.int64)
STARTS = [(0, 0), (3, 4)]
GOALS = [(3, 0), (0, 4)]


def transformed(grid, starts, goals, k, flip):
    g = fp.normalize_grid(grid)
    return (fp.transform_grid(g, k, flip),
            [fp.transform_cell(s, g.shape, k, flip) for s in starts],
            [fp.transform_cell(s, g.shape, k, flip) for s in goals])


class TestCoordinates(unittest.TestCase):
    def test_uses_loader_row_col(self):
        rec = fp.fingerprint_scenario(SCEN / "smoke" / "scenario_empty_s11_n3.json")
        _, starts, goals = load_map_generator_scenario(SCEN / "smoke" / "scenario_empty_s11_n3.json")
        self.assertEqual(rec["starts_rc"], [list(s) for s in starts])
        self.assertEqual(rec["starts_rc"][0], [4, 3])          # JSON [x=3, y=4]
        self.assertEqual(rec["goals_rc"], [list(g) for g in goals])

    def test_source_info_and_agent_id_check(self):
        rec = fp.fingerprint_scenario(SCEN / "controlled_variants" / "scenario_bottleneck_n4_convoy.json")
        self.assertEqual(rec["source_path"], "scenarios/controlled_variants/scenario_bottleneck_n4_convoy.json")
        self.assertEqual(rec["map_file_path"], "scenarios/controlled/map_bottleneck_n4.npy")
        self.assertEqual(rec["agent_ids_in_file"], [0, 1, 2, 3])
        self.assertTrue(rec["agent_id_matches_index"])
        self.assertEqual(rec["fp_schema"], fp.FP_SCHEMA)


class TestGrid(unittest.TestCase):
    def test_dtype_does_not_change_grid_fp(self):
        a = fp.compute_fingerprints(G, STARTS, GOALS)
        for dt in (np.uint8, np.int32, np.bool_):
            self.assertEqual(fp.compute_fingerprints(G.astype(dt), STARTS, GOALS)["grid_fp"], a["grid_fp"])

    def test_rejects_malformed(self):
        with self.assertRaises(fp.FingerprintError):
            fp.normalize_grid(np.array([[0, 2]]))
        with self.assertRaises(fp.FingerprintError):
            fp.normalize_grid(np.zeros((2, 2, 2), dtype=int))
        with self.assertRaises(fp.FingerprintError):
            fp.normalize_grid(np.zeros((2, 2), dtype=float))
        with self.assertRaises(fp.FingerprintError):
            fp.compute_fingerprints(G, [(1, 1)], [(0, 0)])          # start on a wall
        with self.assertRaises(fp.FingerprintError):
            fp.compute_fingerprints(G, [(9, 0)], [(0, 0)])          # outside
        with self.assertRaises(fp.FingerprintError):
            fp.compute_fingerprints(G, [(0, 0), (0, 0)], GOALS)     # shared start

    def test_trim_walls(self):
        padded = np.pad(G, 1, constant_values=1)
        a = fp.compute_fingerprints(G, STARTS, GOALS)
        b = fp.compute_fingerprints(padded, [(r + 1, c + 1) for r, c in STARTS],
                                    [(r + 1, c + 1) for r, c in GOALS])
        self.assertNotEqual(a["grid_fp"], b["grid_fp"])
        self.assertEqual(a["grid_trim_fp"], b["grid_trim_fp"])
        self.assertEqual(b["grid_trim_offset"], [1, 1])
        self.assertEqual(a["agents_trim_ordered_fp"], b["agents_trim_ordered_fp"])

    def test_corridor_map_trims_to_two_rows(self):
        rec = fp.fingerprint_scenario(SCEN / "controlled" / "scenario_corridor_bay_n2.json")
        self.assertEqual(rec["grid_shape"], [5, 11])
        self.assertEqual(rec["grid_trim_shape"], [2, 11])
        self.assertEqual(rec["grid_trim_offset"], [1, 0])


class TestTransforms(unittest.TestCase):
    def test_cell_mapping_matches_grid_mapping(self):
        g = fp.normalize_grid(np.zeros((3, 5), dtype=int))
        for k, flip in fp.TRANSFORMS:
            for r in range(3):
                for c in range(5):
                    marked = np.zeros((3, 5), dtype=np.uint8)
                    marked[r, c] = 1
                    t = fp.transform_grid(marked, k, flip)
                    self.assertEqual(tuple(int(v) for v in np.argwhere(t == 1)[0]),
                                     fp.transform_cell((r, c), g.shape, k, flip))

    def test_canonical_invariant_under_joint_transform(self):
        base = fp.compute_fingerprints(G, STARTS, GOALS)
        grid_fps = set()
        for k, flip in fp.TRANSFORMS:
            tg, ts, tgo = transformed(G, STARTS, GOALS, k, flip)
            rec = fp.compute_fingerprints(tg, ts, tgo)
            grid_fps.add(rec["grid_fp"])
            self.assertEqual(rec["map_canon_fp"], base["map_canon_fp"])
            self.assertEqual(rec["agents_canon_set_fp"], base["agents_canon_set_fp"])
            self.assertEqual(rec["agents_canon_ordered_fp"], base["agents_canon_ordered_fp"])
        self.assertEqual(len(grid_fps), 8)          # strict fingerprint distinguishes all 8

    def test_symmetric_maps_list_all_tying_transforms(self):
        for name in ("controlled/scenario_intersection_n2.json", "controlled/scenario_cycle_ring_n4.json"):
            rec = fp.fingerprint_scenario(SCEN / name)
            self.assertEqual(len(rec["map_canon_transforms"]), 8, name)
        rec = fp.compute_fingerprints(G, STARTS, GOALS)
        self.assertEqual(len(rec["map_canon_transforms"]), 1)


class TestAgents(unittest.TestCase):
    def test_ordered_and_id_free_fingerprints(self):
        a = fp.compute_fingerprints(G, STARTS, GOALS)
        b = fp.compute_fingerprints(G, STARTS[::-1], GOALS[::-1])
        self.assertNotEqual(a["agents_ordered_fp"], b["agents_ordered_fp"])
        self.assertEqual(a["agents_set_fp"], b["agents_set_fp"])

    def test_start_goal_pairs_not_mixed(self):
        a = fp.compute_fingerprints(G, STARTS, GOALS)
        b = fp.compute_fingerprints(G, STARTS, GOALS[::-1])     # same cells, other pairing
        self.assertNotEqual(a["agents_set_fp"], b["agents_set_fp"])


if __name__ == "__main__":
    unittest.main()
