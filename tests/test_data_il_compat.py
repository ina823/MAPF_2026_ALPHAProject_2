"""src/data/il_compat.py + scripts/build_il_input_reference.py."""

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np

from scripts import build_il_input_reference as bir
from src.data import il_compat as ic
from src.il.nav_hint import bfs_dist, make_goal_dir

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE = PROJECT_ROOT / "registry" / "il_input_reference_v1.json"


def synthetic_npz(path: Path) -> None:
    rng = np.random.default_rng(0)
    grids = (rng.random((400, 3, 5, 5)) < 0.2).astype(np.float32)
    goals = rng.integers(-6, 7, size=(400, 2)).astype(np.float32)
    acts = rng.integers(0, 5, size=400)
    np.savez(path, states_grid=grids, goal_dirs=goals, actions=acts)


class TestReference(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.npz = self.dir / "train.npz"
        synthetic_npz(self.npz)

    def tearDown(self):
        self.tmp.cleanup()

    def test_percentiles_and_determinism(self):
        ref = ic.build_reference(self.npz)
        with np.load(self.npz) as d:
            gd = d["goal_dirs"].astype(np.float64)
            walls = d["states_grid"][:, 0].reshape(400, -1).sum(1)
        self.assertEqual(ref["features"]["goal_row"]["p0.5"], float(np.percentile(gd[:, 0], 0.5)))
        self.assertEqual(ref["features"]["goal_col"]["p99.5"], float(np.percentile(gd[:, 1], 99.5)))
        self.assertEqual(ref["features"]["wall_count"]["max"], float(walls.max()))
        self.assertEqual(ref["source"]["n_samples"], 400)
        self.assertEqual(ic.reference_json(ref), ic.reference_json(ic.build_reference(self.npz)))

    def test_script_refuses_existing_output(self):
        out = self.dir / "ref.json"
        out.write_text("keep")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            rc = bir.main(["--npz", str(self.npz), "--out", str(out)])
        self.assertEqual(rc, 2)
        self.assertEqual(out.read_text(), "keep")

    def test_script_rejects_npz_not_matching_checkpoint(self):
        out = self.dir / "ref.json"
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            rc = bir.main(["--npz", str(self.npz), "--out", str(out)])
        self.assertEqual(rc, 1)                  # synthetic stats != MANIFEST normalization
        self.assertFalse(out.exists())


class TestCommittedReference(unittest.TestCase):
    def test_reference_matches_training_npz_and_checkpoint(self):
        ref = ic.load_reference(REFERENCE)
        self.assertLess(ref["normalization_check"]["max_abs_diff"], bir.MAX_NORMALIZATION_DIFF)
        self.assertEqual(ref["percentiles"], [0.5, 99.5])
        if not bir.DEFAULT_NPZ.is_file():
            self.skipTest("Alpha1_legacy/real_v03_fd.npz not available")
        self.assertEqual(REFERENCE.read_bytes(), ic.reference_json(ic.build_reference(bir.DEFAULT_NPZ, bir.MANIFEST)))


class TestCheckInstance(unittest.TestCase):
    REF = {"source": {"sha256": "x"}, "features": {
        "goal_row": {"p0.5": -5.0, "p99.5": 5.0}, "goal_col": {"p0.5": -5.0, "p99.5": 5.0},
        "wall_count": {"p0.5": 0.0, "p99.5": 17.0}, "robot_count": {"p0.5": 0.0, "p99.5": 4.0},
        "other_goal_count": {"p0.5": 0.0, "p99.5": 4.0}}}

    def test_start_features_use_runtime_inputs(self):
        grid = np.zeros((4, 4), dtype=int)
        grid[1, 1] = 1
        starts, goals = [(0, 0), (3, 3)], [(3, 0), (0, 3)]
        f = ic.start_features(grid, starts, goals)
        hint = make_goal_dir(bfs_dist(grid, goals[0]), starts[0], goals[0], "flowdist")
        self.assertEqual((f[0]["goal_row"], f[0]["goal_col"]), (float(hint[0]), float(hint[1])))
        self.assertEqual(f[0]["robot_count"], 0.0)          # other robot 3 cells away in both axes
        self.assertEqual(f[0]["wall_count"], 17.0)          # 16 outside cells + wall (1, 1)

    def test_in_range_and_ood(self):
        grid = np.zeros((3, 3), dtype=int)                  # every hint within +-4
        res = ic.check_instance(grid, [(1, 1)], [(2, 0)], self.REF)
        self.assertEqual((res["status"], res["warnings"]), (ic.IN_RANGE, []))
        long = np.zeros((1, 12), dtype=int)
        res = ic.check_instance(long, [(0, 0)], [(0, 11)], self.REF)
        self.assertEqual(res["status"], ic.OOD_WARNING)
        kinds = {(w["kind"], w["feature"]) for w in res["warnings"]}
        self.assertIn(("START", "goal_col"), kinds)
        self.assertIn(("START", "wall_count"), kinds)
        self.assertIn(("REACHABLE_RANGE", "goal_col"), kinds)
        self.assertFalse(res["auto_fail"])


if __name__ == "__main__":
    unittest.main()
