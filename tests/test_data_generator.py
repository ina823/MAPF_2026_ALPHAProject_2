"""src/data/generator.py: determinism, validity, JSON coordinate round trip."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.common.scenario_loader import load_map_generator_scenario, xy_to_rowcol
from src.data import generator as gen

CORRIDOR = dict(length=15, bays_up=(5, 9), bays_down=(), n_east=2, n_west=1)
ROOMS = dict(room_h=6, room_w=6, door_len=2, door_row=2, n_east=2, n_west=2)
CROSS = dict(height=9, width=11, corridor_rows=(2, 6), corridor_cols=(5,))


class TestStructures(unittest.TestCase):
    def test_same_seed_same_bytes(self):
        for build, kw in ((gen.corridor_bays, CORRIDOR), (gen.rooms_door, ROOMS), (gen.crossing, CROSS)):
            a = gen.scenario_files(build(**kw, seed=5), "x", "sha")
            b = gen.scenario_files(build(**kw, seed=5), "x", "sha")
            self.assertEqual(a, b, build.__name__)

    def test_seed_moves_agents_not_layout(self):
        a, b = gen.rooms_door(**ROOMS, seed=1), gen.rooms_door(**ROOMS, seed=2)
        self.assertTrue(np.array_equal(a.grid, b.grid))
        self.assertNotEqual((a.starts, a.goals), (b.starts, b.goals))

    def test_corridor_layout_and_order(self):
        inst = gen.corridor_bays(**CORRIDOR, seed=11)
        self.assertEqual(inst.grid.shape, (2, 15))
        self.assertEqual(inst.grid[0].tolist().count(0), 2)          # two bays
        self.assertTrue((inst.grid[1] == 0).all())                    # the corridor
        east = list(zip(inst.starts[:2], inst.goals[:2]))
        self.assertEqual(sorted(east), east)                          # order kept along the corridor
        self.assertTrue(all(s[1] < 3 and g[1] >= 12 for s, g in east))

    def test_rooms_door_layout(self):
        inst = gen.rooms_door(**ROOMS, seed=12)
        self.assertEqual(inst.grid.shape, (6, 14))
        self.assertEqual(inst.grid[:, 6:8].sum(), 10)                 # passage: 2 free cells in 12
        self.assertEqual(inst.grid[2, 6:8].tolist(), [0, 0])
        self.assertEqual(inst.structure_type, "bottleneck")

    def test_crossing_layout(self):
        inst = gen.crossing(**CROSS, seed=13)
        free = set(zip(*np.where(inst.grid == 0)))
        self.assertEqual(len(free), 2 * 11 + 9 - 2)
        self.assertEqual(inst.descriptor["flows"], ["S@col5", "E@row2", "W@row6"])

    def test_validation(self):
        grid = np.array([[0, 1, 0]])
        with self.assertRaises(gen.GeneratorError):
            gen.validate_instance(grid, [(0, 0)], [(0, 2)])           # unreachable
        with self.assertRaises(gen.GeneratorError):
            gen.validate_instance(grid, [(0, 1)], [(0, 0)])           # start on a wall
        with self.assertRaises(gen.GeneratorError):
            gen.validate_instance(np.zeros((1, 3), int), [(0, 0), (0, 0)], [(0, 1), (0, 2)])
        with self.assertRaises(gen.GeneratorError):
            gen.corridor_bays(length=4, bays_up=(1,), bays_down=(), n_east=3, n_west=2, seed=0)


class TestFiles(unittest.TestCase):
    def test_json_round_trip_through_loader(self):
        inst = gen.crossing(**CROSS, seed=13)
        npy, js = gen.scenario_files(inst, "pilot_x", "codesha")
        scen = json.loads(js)
        self.assertEqual(scen["map_file"], "map_pilot_x.npy")
        self.assertEqual([a["agent_id"] for a in scen["agents"]], [0, 1, 2])
        self.assertEqual(scen["generator"]["seed"], 13)
        for a, s in zip(scen["agents"], inst.starts):
            self.assertEqual(xy_to_rowcol(a["start"]), s)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "map_pilot_x.npy").write_bytes(npy)
            (Path(tmp) / "scenario_pilot_x.json").write_bytes(js)
            grid, starts, goals = load_map_generator_scenario(Path(tmp) / "scenario_pilot_x.json")
        self.assertTrue(np.array_equal(grid, inst.grid))
        self.assertEqual(grid.dtype, np.int64)
        self.assertEqual((tuple(starts), tuple(goals)), (inst.starts, inst.goals))


if __name__ == "__main__":
    unittest.main()
