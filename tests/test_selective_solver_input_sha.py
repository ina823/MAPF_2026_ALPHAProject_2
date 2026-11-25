"""SELECTIVE_HYBRID_DEV solver-input sha256 provenance (DEV, step 6).

Part 1 (no solver): solver_input.yaml files are written by the production
writer (src.cbs_adapter.create_atb033_input + run_atb033_cbs, with only its
``subprocess.run`` stubbed so cbs.py never runs) into the call directories the
Selective adapters use; attempt records come from
``selective_bridge.solver_input_record``; ``provenance.cbs_request_states``
re-hashes and matches them.

Part 2 (MOCK SOLVER, needs torch): scripts/run_compare_batch end to end on
the Corridor (G attempt, real chain up to ``subprocess.run``), and the five
default methods unchanged.
"""

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.compare import provenance as pv
from src.compare import selective_bridge as sb
from src.compare.common_runner import SELECTIVE_HYBRID_DEV

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

GRID = np.zeros((6, 6), dtype=np.int64)
GRID[2, 2] = 1
S_T = {0: (0, 0), 1: (0, 5), 2: (5, 0), 3: (5, 5)}
GOALS = {0: (0, 1), 1: (1, 5), 2: (4, 0), 3: (5, 4)}
GROUP = {"L0": [0, 2], "L1": [0, 1, 2], "G": [0, 1, 2, 3]}


def write_solver_input(work_dir, group, s_t=S_T):
    """Write solver_input.yaml exactly as the production chain does."""
    from src import cbs_adapter
    replan = Path(work_dir).joinpath(*sb.SOLVER_INPUT_RELPATH[:-1])
    replan.mkdir(parents=True)
    cbs_adapter.create_atb033_input(GRID, [s_t[a] for a in group], [GOALS[a] for a in group],
                                    replan / "input.yaml")
    solver = Path(work_dir).parent / "_solver"
    solver.mkdir(exist_ok=True)
    (solver / "cbs.py").write_text("# never run\n", encoding="utf-8")
    with patch.object(cbs_adapter.subprocess, "run",
                      return_value=subprocess.CompletedProcess([], 1, stdout="", stderr="stub")):
        try:
            cbs_adapter.run_atb033_cbs(solver, replan / "input.yaml", replan / "output.yaml")
        except RuntimeError:
            pass   # the stubbed solver "fails"; solver_input.yaml has been written already
    return replan / "solver_input.yaml"


class Run:
    """A fake run directory + EpisodeRecord-shaped record with Selective attempts."""

    def __init__(self, root):
        self.dir = Path(root) / "run"
        self.dir.mkdir()
        self.counters = {"subset": 0, "global": 0}
        self.cbs_calls, self.attempts = [], []
        self.trajectory = {t: dict(S_T) for t in range(10)}

    def request(self, t=4, s_t=None):
        if s_t is not None:
            self.trajectory[t] = dict(s_t)
        self.cbs_calls.append({"trigger_timestep": t})
        return len(self.cbs_calls)

    def attempt(self, req, level, *, called=True, group=None):
        side = "global" if level == "G" else "subset"
        self.counters[side] += 1
        sub, prefix = pv.SELECTIVE_SIDES[side]
        work_dir = self.dir / pv.SELECTIVE_CBS_DIR / sub / f"{prefix}{self.counters[side]:02d}"
        group = list(GROUP[level] if group is None else group)
        if called:
            t = self.cbs_calls[req - 1]["trigger_timestep"]
            write_solver_input(work_dir, group, self.trajectory[t])
        call = {"called": called, "work_dir": str(work_dir) if called else None}
        n = sum(1 for a in self.attempts if a["request_index"] == req) + 1
        att = {"request_index": req, "attempt_index": n, "level": level, "group": group, "called": called,
               **sb.solver_input_record(call)}
        self.attempts.append(att)
        return att

    def record(self):
        return SimpleNamespace(run_id="run", method=SELECTIVE_HYBRID_DEV, run_dir=str(self.dir),
                               trajectory=self.trajectory, goals=dict(GOALS),
                               cbs_calls=self.cbs_calls, planner_calls=self.attempts)

    def states(self):
        return pv.cbs_request_states([self.record()])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TestSolverInputSha(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.run_ = Run(tmp.name)

    def checks(self, states):
        return [(s["call_index"], s["attempt_index"], s.get("level"), s["solver_input_check"]) for s in states]

    # -- correct records -----------------------------------------------------
    def test_subset_input_sha256_is_raw_file_bytes(self):
        att = self.run_.attempt(self.run_.request(), "L0")
        path = Path(att["solver_input_path"])
        self.assertEqual(path.parent.parent.parent.parent.name, "subset_cbs_call_01")
        self.assertEqual(att["solver_input_sha256"], sha(path))
        (s,) = self.run_.states()
        self.assertEqual((s["solver_input_check"], s["solver_input_sha256"], s["solver_input_file_sha256"]),
                         ("VERIFIED", sha(path), sha(path)))
        self.assertEqual(s["solver_input_path"],
                         "selective_cbs/subset_cbs/subset_cbs_call_01/outputs/paths/replan/solver_input.yaml")
        self.assertEqual(pv.solver_input_problems([s]), [])

    def test_global_input_sha256(self):
        att = self.run_.attempt(self.run_.request(), "G")
        (s,) = self.run_.states()
        self.assertIn("global_cbs/global_cbs_call_01/", s["solver_input_path"])
        self.assertEqual((s["solver_input_check"], s["solver_input_file_sha256"]),
                         ("VERIFIED", sha(att["solver_input_path"])))

    def test_l0_success(self):
        self.run_.attempt(self.run_.request(), "L0")
        self.assertEqual(self.checks(self.run_.states()), [(1, 1, "L0", "VERIFIED")])

    def test_l0_timeout_then_g_success_each_attempt_its_own_input(self):
        req = self.run_.request()
        l0, g = self.run_.attempt(req, "L0"), self.run_.attempt(req, "G")
        states = self.run_.states()
        self.assertEqual(self.checks(states), [(1, 1, "L0", "VERIFIED"), (1, 2, "G", "VERIFIED")])
        self.assertNotEqual(states[0]["solver_input_sha256"], states[1]["solver_input_sha256"])
        self.assertEqual([s["solver_input_sha256"] for s in states],
                         [sha(l0["solver_input_path"]), sha(g["solver_input_path"])])

    def test_l0_failure_then_l1_success(self):
        req = self.run_.request()
        self.run_.attempt(req, "L0")
        self.run_.attempt(req, "L1")
        states = self.run_.states()
        self.assertEqual(self.checks(states), [(1, 1, "L0", "VERIFIED"), (1, 2, "L1", "VERIFIED")])
        self.assertIn("subset_cbs_call_02/", states[1]["solver_input_path"])

    def test_d2_two_requests(self):
        r1 = self.run_.request(t=3)
        self.run_.attempt(r1, "L0")
        r2 = self.run_.request(t=9, s_t={**S_T, 0: (1, 0)})   # agent 0 moved before request 2
        self.run_.attempt(r2, "L0")
        self.run_.attempt(r2, "G")
        states = self.run_.states()
        self.assertEqual(self.checks(states), [(1, 1, "L0", "VERIFIED"), (2, 1, "L0", "VERIFIED"),
                                               (2, 2, "G", "VERIFIED")])
        self.assertEqual([s["request_timestep"] for s in states], [3, 9, 9])
        self.assertEqual(len({s["solver_input_sha256"] for s in states}), 3)

    def test_attempt_without_solver_call_has_no_hash(self):
        req = self.run_.request()
        att = self.run_.attempt(req, "L0", called=False)
        self.run_.attempt(req, "G")
        self.assertEqual((att["solver_input_status"], att["solver_input_sha256"], att["solver_input_path"]),
                         ("SOLVER_NOT_CALLED", None, None))
        states = self.run_.states()
        self.assertEqual(self.checks(states), [(1, 1, "L0", "NOT_CALLED"), (1, 2, "G", "VERIFIED")])
        self.assertIsNone(states[0]["solver_input_file_sha256"])
        self.assertEqual(pv.solver_input_problems(states), [])

    # -- broken records / files ---------------------------------------------
    def test_missing_input_file(self):
        att = self.run_.attempt(self.run_.request(), "L0")
        Path(att["solver_input_path"]).unlink()
        self.assertEqual(sb.solver_input_record({"called": True, "work_dir": str(Path(att["solver_input_path"])
                                                                               .parents[3])}),
                         {"solver_input_path": att["solver_input_path"], "solver_input_sha256": None,
                          "solver_input_status": "FILE_NOT_FOUND"})
        (s,) = self.run_.states()
        self.assertEqual((s["solver_input_check"], s["solver_input_file_sha256"]), ("MISSING", None))
        self.assertTrue(pv.solver_input_problems([s]))

    def test_called_but_unreadable_at_run_end_is_not_hashed(self):
        self.assertEqual(sb.solver_input_record({"called": True, "work_dir": None})["solver_input_status"],
                         "NO_WORK_DIR")
        att = self.run_.attempt(self.run_.request(), "L0")
        att.update(solver_input_sha256=None, solver_input_status="FILE_NOT_FOUND")
        (s,) = self.run_.states()
        self.assertEqual(s["solver_input_check"], "UNRECORDED")

    def test_tampered_input_file(self):
        att = self.run_.attempt(self.run_.request(), "L0")
        path = Path(att["solver_input_path"])
        path.write_bytes(path.read_bytes() + b"\n# edited\n")
        (s,) = self.run_.states()
        self.assertEqual(s["solver_input_check"], "MISMATCH")
        self.assertEqual(s["solver_input_file_sha256"], sha(path))
        self.assertNotEqual(s["solver_input_sha256"], s["solver_input_file_sha256"])

    def test_recorded_sha256_differs(self):
        att = self.run_.attempt(self.run_.request(), "G")
        att["solver_input_sha256"] = "0" * 64
        (s,) = self.run_.states()
        self.assertEqual(s["solver_input_check"], "MISMATCH")
        self.assertIn("MISMATCH", pv.solver_input_problems([s])[0])

    def test_attempt_and_input_file_swapped(self):
        req = self.run_.request()
        a, b = self.run_.attempt(req, "L0"), self.run_.attempt(req, "L1")
        a["solver_input_path"], b["solver_input_path"] = b["solver_input_path"], a["solver_input_path"]
        a["solver_input_sha256"], b["solver_input_sha256"] = b["solver_input_sha256"], a["solver_input_sha256"]
        self.assertEqual([s["solver_input_check"] for s in self.run_.states()],
                         ["MAPPING_MISMATCH", "MAPPING_MISMATCH"])

    def test_attempt_order_swapped_in_the_record(self):
        req = self.run_.request()
        self.run_.attempt(req, "L0", group=[0, 2])
        self.run_.attempt(req, "L1", group=[0, 1, 2])
        self.run_.attempts.reverse()          # L1 record now claims subset_cbs_call_01
        self.assertIn("MAPPING_MISMATCH", [s["solver_input_check"] for s in self.run_.states()])

    def test_input_of_another_group(self):
        att = self.run_.attempt(self.run_.request(), "L0")
        att["group"] = [1, 3]
        (s,) = self.run_.states()
        self.assertEqual((s["solver_input_check"], s["solver_input_reason"]),
                         ("MAPPING_MISMATCH", "input starts/goals are not this attempt's group at S_t"))

    def test_request_index_without_runner_call(self):
        att = self.run_.attempt(self.run_.request(), "L0")
        att["request_index"] = 2
        (s,) = self.run_.states()
        self.assertEqual(s["solver_input_check"], "MAPPING_MISMATCH")

    def test_uncalled_attempt_with_an_input_file(self):
        req = self.run_.request()
        att = self.run_.attempt(req, "L0")
        att.update(called=False, solver_input_path=None, solver_input_sha256=None,
                   solver_input_status="SOLVER_NOT_CALLED")
        (s,) = self.run_.states()
        self.assertEqual(s["solver_input_check"], "MAPPING_MISMATCH")

    def test_input_no_attempt_owns(self):
        self.run_.attempt(self.run_.request(), "L0")
        write_solver_input(self.run_.dir / "selective_cbs" / "global_cbs" / "global_cbs_call_01", GROUP["G"])
        states = self.run_.states()
        self.assertEqual([s["solver_input_check"] for s in states], ["VERIFIED", "UNCLAIMED_FILE"])
        self.assertEqual(len(pv.solver_input_problems(states)), 1)

    def test_attempt_record_without_sha_fields(self):
        att = self.run_.attempt(self.run_.request(), "L0")
        for key in ("solver_input_path", "solver_input_sha256", "solver_input_status"):
            del att[key]
        (s,) = self.run_.states()
        self.assertEqual(s["solver_input_check"], "UNRECORDED")


def _fake_cbs_subprocess(command, *args, **kwargs):
    from tests.test_selective_real_e2e_dev import _recorded_subprocess
    if isinstance(command, (list, tuple)) and len(command) > 3 and str(command[1]).endswith("cbs.py") \
            and "stdout" not in kwargs:
        return _recorded_subprocess(command, *args, **kwargs)
    return _REAL_RUN(command, *args, **kwargs)


_REAL_RUN = subprocess.run
DEFAULT_STATE_KEYS = {"run_id", "method", "call_index", "request_kind", "request_timestep",
                      "start_state_sha256", "solver_input_sha256"}


@unittest.skipIf(torch is None, "torch is not installed")
class TestBatchManifest(unittest.TestCase):

    def cli(self, *methods):
        """run_compare_batch with the real CBS chain; only cbs.py's subprocess is mocked
        (tests.test_run_compare_batch.run_cli stubs replan_with_status, so no solver
        input would be written there)."""
        import contextlib
        import io
        from scripts.run_compare_batch import main
        from tests.test_common_runner import CORRIDOR
        from tests.test_run_compare_batch import CLEAN
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        solver = Path(tmp.name) / "solver"
        solver.mkdir()
        (solver / "cbs.py").write_text("raise SystemExit('fake solver must not run')\n", encoding="utf-8")
        out_dir = Path(tmp.name) / "compare_test"
        argv = ["--solver-root", str(solver), "--scenario", str(CORRIDOR), "--out-dir", str(out_dir)]
        for m in methods:
            argv += ["--method", m]
        buf = io.StringIO()
        with patch("src.compare.common_runner._git", return_value=CLEAN), \
                patch("src.cbs_adapter.subprocess.run", side_effect=_fake_cbs_subprocess), \
                contextlib.redirect_stdout(buf):
            code = main(argv)
        manifest = json.loads((out_dir / "batch_manifest.json").read_text(encoding="utf-8"))
        return code, buf.getvalue(), manifest, out_dir

    def test_default_methods_request_states_unchanged(self):
        from tests.test_run_compare_batch import _Case

        class Batch(_Case):
            def runTest(self):
                pass

        case = Batch()
        case.setUp()
        self.addCleanup(case.tearDown)
        code, out = case.run_cli()          # the five default methods, existing harness
        self.assertEqual(code, 0, out)
        manifest = json.loads((case.out / "batch_manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["cbs_request_states"])
        for s in manifest["cbs_request_states"]:
            self.assertEqual(set(s), DEFAULT_STATE_KEYS)
        self.assertNotIn("solver_inputs_verified", manifest["reproducibility"])
        self.assertNotIn("selective_solver_inputs_note", manifest)

    def test_default_method_hash_is_the_raw_file_bytes(self):
        code, out, manifest, out_dir = self.cli("GLOBAL_HYBRID_DEV_H3")
        self.assertEqual(code, 0, out)
        (s,) = manifest["cbs_request_states"]
        self.assertEqual(set(s), DEFAULT_STATE_KEYS)
        f = out_dir / "runs" / s["run_id"] / "cbs" / "cbs_call_01" / "outputs" / "paths" / "replan" / "solver_input.yaml"
        self.assertEqual(s["solver_input_sha256"], sha(f))
        self.assertNotIn("solver_inputs_verified", manifest["reproducibility"])
        self.assertNotIn("selective_solver_inputs_note", manifest)

    def test_selective_inputs_verified_in_the_manifest(self):
        code, out, manifest, out_dir = self.cli("GLOBAL_HYBRID_DEV_H3", SELECTIVE_HYBRID_DEV)
        self.assertEqual(code, 0, out)
        states = manifest["cbs_request_states"]
        h3 = [s for s in states if s["method"] == "GLOBAL_HYBRID_DEV_H3"]
        self.assertTrue(h3 and all(set(s) == DEFAULT_STATE_KEYS for s in h3))
        sel = [s for s in states if s["method"] == SELECTIVE_HYBRID_DEV]
        (s,) = sel
        self.assertEqual((s["call_index"], s["attempt_index"], s["level"], s["solver_input_check"]),
                         (1, 1, "G", "VERIFIED"))
        run_dir = out_dir / "runs" / s["run_id"]
        self.assertEqual(s["solver_input_sha256"], sha(run_dir / s["solver_input_path"]))
        self.assertTrue(manifest["reproducibility"]["solver_inputs_verified"])
        self.assertIn("selective_solver_inputs_note", manifest)
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["planner_calls"][0]["solver_input_sha256"], s["solver_input_sha256"])

    def test_wrong_recorded_sha_makes_the_batch_not_reproducible(self):
        real = sb.solver_input_record

        def wrong(call):
            return {**real(call), "solver_input_sha256": "f" * 64}

        with patch.object(sb, "solver_input_record", side_effect=wrong):
            code, _, manifest, _ = self.cli("GLOBAL_HYBRID_DEV_H3", SELECTIVE_HYBRID_DEV)
        rep = manifest["reproducibility"]
        self.assertFalse(rep["solver_inputs_verified"])
        self.assertFalse(rep["reproducible_batch"])
        self.assertTrue(any("MISMATCH" in r for r in rep["reasons"]), rep["reasons"])
        self.assertEqual(code, 0)    # fairness is a separate verdict and is left as computed


if __name__ == "__main__":
    unittest.main()
