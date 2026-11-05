"""Hash-seed provenance, batch comparability and output-path checks
(src/compare/provenance.py, scripts/run_compare_batch.py).

PYTHONHASHSEED takes effect when an interpreter starts, so tests that need a
process really started with a seed launch a fresh interpreter with that
seed in its start-up environment. In-process tests pin
provenance.STARTUP_ENV_VALUE explicitly so they do not depend on how the test
runner itself was started. The real solver never runs: CBS is the recorded
atb033 plan injected below CbsRuntimePlanner (tests.test_common_runner).
"""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from src.compare import provenance as pv
from src.compare.common_runner import CBS_METHODS, GLOBAL_HYBRID_DEV_H3, PURE_CBS_DEV_SINGLE_PLAN
from src.common.episode_logger import make_run_id
from tests.test_common_runner import CORRIDOR, recorded_solver

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OLD15_MANIFEST = PROJECT_ROOT / "outputs" / "logs" / "compare_20261103_215010" / "batch_manifest.json"
CLEAN = {"commit": "c0ffee", "dirty": False}


def _env_without_seed():
    return {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}


def _run_child(code, seed):
    """Run `code` in a fresh interpreter started with PYTHONHASHSEED=seed (None = unset)."""
    env = _env_without_seed()
    if seed is not None:
        env["PYTHONHASHSEED"] = seed
    cp = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=PROJECT_ROOT, env=env,
                        capture_output=True, text=True, timeout=300)
    if cp.returncode not in (0, 1, 2):
        raise AssertionError(f"child failed rc={cp.returncode}\n{cp.stderr[-2000:]}")
    return cp


class TestSeedModeAndParentRecord(unittest.TestCase):

    def test_seed_mode(self):
        self.assertEqual(pv.seed_mode(None), pv.UNSET_RANDOM)
        self.assertEqual(pv.seed_mode("random"), pv.UNSET_RANDOM)
        self.assertEqual(pv.seed_mode("0"), pv.FIXED_DISABLED)      # hash randomization disabled
        self.assertEqual(pv.seed_mode("7"), pv.FIXED)
        self.assertEqual(pv.seed_mode("4294967295"), pv.FIXED)
        for bad in ("4294967296", "-1", "abc", ""):
            self.assertEqual(pv.seed_mode(bad), pv.INVALID, bad)

    def test_mode_follows_startup_not_runtime_env(self):
        with patch.object(pv, "STARTUP_ENV_VALUE", None), patch.dict(os.environ, {"PYTHONHASHSEED": "5"}):
            info = pv.hash_seed_info()
        self.assertEqual((info["startup_env_value"], info["current_env_value"], info["mode"]),
                         (None, "5", pv.UNSET_RANDOM))
        self.assertIn("'0' = hash randomization disabled", info["note"])

    def test_environment_info_carries_hash_seed(self):
        from src.compare.common_runner import environment_info
        self.assertEqual(set(environment_info()["hash_seed"]),
                         {"startup_env_value", "current_env_value", "mode", "hash_randomization", "probe_hash", "note"})


class TestSubprocessProbes(unittest.TestCase):

    def test_explicit_seed_child_is_deterministic(self):
        a, b = pv.subprocess_probe(explicit_seed="3"), pv.subprocess_probe(explicit_seed="3")
        self.assertTrue(a["ok"] and b["ok"])
        self.assertEqual((a["env_value"], a["hash_randomization"]), ("3", 1))
        self.assertEqual(a["probe_hash"], b["probe_hash"])
        z = pv.subprocess_probe(explicit_seed="0")
        self.assertEqual((z["env_value"], z["hash_randomization"]), ("0", 0))

    def test_inherited_child_sees_runtime_env_but_parent_does_not_change(self):
        with patch.object(pv, "STARTUP_ENV_VALUE", None), patch.dict(os.environ, {"PYTHONHASHSEED": "4"}):
            parent = pv.hash_seed_info()
            layers = pv.verify_seed_layers(parent)
        self.assertEqual(layers["inherited_probe"]["env_value"], "4")   # child inherits the mutated env
        self.assertEqual(layers["inherited_status"], "MISMATCH")
        self.assertTrue(any("changed after start-up" in r for r in layers["reasons"]))

    def test_unset_seed_layers(self):
        with patch.object(pv, "STARTUP_ENV_VALUE", None), patch.dict(os.environ, _env_without_seed(), clear=True):
            layers = pv.verify_seed_layers(pv.hash_seed_info())
        self.assertEqual((layers["parent_verified"], layers["inherited_status"], layers["reasons"]),
                         ("NOT_APPLICABLE", "OK", []))

    def test_probe_failure_is_reported(self):
        r = pv.subprocess_probe(python=str(PROJECT_ROOT / "no_such_python.exe"))
        self.assertFalse(r["ok"])
        self.assertIn("error", r)

    def test_interpreter_started_with_seed_is_verified(self):
        code = """
            import json
            from src.compare import provenance as p
            print(json.dumps(p.verify_seed_layers(p.hash_seed_info())))
        """
        for seed in ("0", "7"):
            with self.subTest(seed=seed):
                layers = json.loads(_run_child(code, seed).stdout.strip().splitlines()[-1])
                self.assertEqual((layers["parent_verified"], layers["inherited_status"], layers["reasons"]),
                                 ("VERIFIED", "OK", []))

    def test_env_changed_after_start_is_not_verified(self):
        code = """
            import json, os
            from src.compare import provenance as p
            os.environ["PYTHONHASHSEED"] = "8"      # too late: this interpreter started with 7
            print(json.dumps(p.verify_seed_layers(p.hash_seed_info())))
        """
        layers = json.loads(_run_child(code, "7").stdout.strip().splitlines()[-1])
        self.assertEqual(layers["parent_verified"], "VERIFIED")        # the parent really runs with 7 ...
        self.assertEqual(layers["inherited_status"], "MISMATCH")       # ... but cbs.py would get 8
        self.assertTrue(any("changed after start-up" in r for r in layers["reasons"]))


class TestCbsAdapterInheritsEnvironment(unittest.TestCase):
    """Contract: cbs.py inherits this process's environment (no env= passed).
    If this ever changes, the CBS-call seed record no longer describes the child."""

    def test_run_atb033_cbs_passes_no_env(self):
        from src import cbs_adapter
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "cbs.py").write_text("", encoding="utf-8")
            inp = tmp / "input.yaml"
            inp.write_text("map:\n  dimensions: [3, 3]\n  obstacles: []\nagents: []\n", encoding="utf-8")
            out = tmp / "output.yaml"

            def fake_run(cmd, **kwargs):
                out.write_text("schedule: {}\n", encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, "solution found", "")

            with patch.object(cbs_adapter.subprocess, "run", side_effect=fake_run) as run:
                cbs_adapter.run_atb033_cbs(tmp, inp, out)
        self.assertEqual(run.call_count, 1)
        self.assertIsNone(run.call_args.kwargs.get("env"))

    def test_planner_records_env_at_call_time(self):
        from src.hybrid.cbs_runtime_bridge import CbsRuntimePlanner
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "cbs.py").write_text("", encoding="utf-8")
            planner = CbsRuntimePlanner(solver_root=tmp, work_root=Path(tmp) / "work")
            with patch("src.hybrid.cbs_runtime_bridge.run_cbs_runtime", return_value=object()) as rt, \
                    patch.dict(os.environ, {"PYTHONHASHSEED": "11"}):
                rt.return_value = type("R", (), {"status": "success", "latency_sec": 0.0, "error": None, "paths": {}})()
                planner({0: (0, 0)}, {0: (0, 1)}, None)
            with patch("src.hybrid.cbs_runtime_bridge.run_cbs_runtime", return_value=rt.return_value), \
                    patch.dict(os.environ, _env_without_seed(), clear=True):
                planner({0: (0, 0)}, {0: (0, 1)}, None)
        self.assertEqual([c["pythonhashseed_env"] for c in planner.calls], ["11", None])


class _BatchCase(unittest.TestCase):
    METHODS = (PURE_CBS_DEV_SINGLE_PLAN, GLOBAL_HYBRID_DEV_H3)

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.solver = self.tmp / "solver"
        self.solver.mkdir()
        (self.solver / "cbs.py").write_text("raise SystemExit('fake solver must not run')\n", encoding="utf-8")
        self.out = self.tmp / "b"

    def tearDown(self):
        self._tmp.cleanup()

    def argv(self, *extra):
        argv = ["--solver-root", str(self.solver), "--scenario", str(CORRIDOR), "--out-dir", str(self.out)]
        for m in self.METHODS:
            argv += ["--method", m]
        return argv + list(extra)

    def run_in_process(self, *extra, solver=recorded_solver):
        from scripts.run_compare_batch import main
        buf = io.StringIO()
        with patch("src.compare.common_runner._git", return_value=CLEAN), \
                patch("src.cbs_runtime_adapter.replan_with_status", side_effect=solver), \
                contextlib.redirect_stdout(buf):
            code = main(self.argv(*extra))
        return code, buf.getvalue()

    def run_in_child(self, seed, *extra):
        code = f"""
            import sys
            from unittest.mock import patch
            from scripts.run_compare_batch import main
            from tests.test_common_runner import recorded_solver
            with patch("src.compare.common_runner._git", return_value={CLEAN!r}), \\
                    patch("src.cbs_runtime_adapter.replan_with_status", side_effect=recorded_solver):
                sys.exit(main({self.argv(*extra)!r}))
        """
        return _run_child(code, seed)

    def manifest(self):
        return json.loads((self.out / "batch_manifest.json").read_text(encoding="utf-8"))

    def results(self):
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(self.out.glob("runs/*/result.json"))]


class TestBatchSeedRecords(_BatchCase):

    def test_seed_fixed_at_startup_is_recorded_and_verified(self):
        cp = self.run_in_child("0", "--require-hash-seed")
        self.assertEqual(cp.returncode, 0, cp.stdout + cp.stderr)
        rep = self.manifest()["reproducibility"]
        self.assertEqual((rep["seed_mode"], rep["seed_value"]), (pv.FIXED_DISABLED, "0"))
        self.assertEqual(rep["parent"]["hash_randomization"], 0)
        self.assertEqual((rep["parent_verified"], rep["inherited_probe_status"]), ("VERIFIED", "OK"))
        self.assertTrue(rep["cbs_calls_seed_consistent"])
        self.assertTrue(rep["reproducible_batch"])
        self.assertTrue(self.manifest()["fair_batch"])
        self.assertEqual(self.manifest()["environment"]["hash_seed"]["startup_env_value"], "0")
        for res in self.results():
            self.assertEqual(res["environment"]["hash_seed"]["startup_env_value"], "0")
            self.assertEqual([c["pythonhashseed_env"] for c in res["planner_calls"]], ["0"])

    def test_unset_seed_keeps_fair_batch_and_is_not_reproducible(self):
        with patch.object(pv, "STARTUP_ENV_VALUE", None), patch.dict(os.environ, _env_without_seed(), clear=True):
            code, out = self.run_in_process()
        self.assertEqual(code, 0, out)                         # exit code unchanged
        m = self.manifest()
        self.assertEqual((m["fair_batch"], m["fairness_reasons"]), (True, []))
        rep = m["reproducibility"]
        self.assertEqual((rep["seed_mode"], rep["seed_value"], rep["reproducible_batch"]),
                         (pv.UNSET_RANDOM, None, False))
        self.assertEqual(rep["parent_verified"], "NOT_APPLICABLE")
        self.assertTrue(rep["reasons"][0].startswith("hash seed not fixed"))
        for res in self.results():
            self.assertEqual([c["pythonhashseed_env"] for c in res["planner_calls"]], [None])

    def test_require_hash_seed_refuses_unset_before_running(self):
        with patch.object(pv, "STARTUP_ENV_VALUE", None), patch.dict(os.environ, _env_without_seed(), clear=True):
            code, out = self.run_in_process("--require-hash-seed")
        self.assertEqual(code, 2)
        self.assertIn("PYTHONHASHSEED not fixed", out)
        self.assertFalse(self.out.exists())

    def test_require_hash_seed_refuses_when_child_does_not_inherit(self):
        def broken(*, explicit_seed=None, **kw):
            if explicit_seed is None:   # the inherited child sees something else
                return {"ok": True, "env_value": "999", "hash_randomization": 1, "probe_hash": 0}
            return real(explicit_seed=explicit_seed, **kw)

        real = pv.subprocess_probe
        with patch.object(pv, "STARTUP_ENV_VALUE", "0"), patch.dict(os.environ, {"PYTHONHASHSEED": "0"}), \
                patch.object(pv, "subprocess_probe", side_effect=broken):
            code, out = self.run_in_process("--require-hash-seed")
        self.assertEqual(code, 2)
        self.assertIn("inherited child sees PYTHONHASHSEED='999'", out)
        self.assertFalse(self.out.exists())

    def test_seed_changing_between_cbs_calls_makes_batch_unfair(self):
        def solver_then_change_seed(**kw):
            os.environ["PYTHONHASHSEED"] = "5"   # what the NEXT CBS call would inherit
            return recorded_solver(**kw)

        with patch.object(pv, "STARTUP_ENV_VALUE", None), patch.dict(os.environ, _env_without_seed(), clear=True):
            code, out = self.run_in_process(solver=solver_then_change_seed)
        self.assertEqual(code, 1)
        m = self.manifest()
        self.assertFalse(m["fair_batch"])
        self.assertTrue(any("PYTHONHASHSEED differs" in r for r in m["fairness_reasons"]), m["fairness_reasons"])
        self.assertFalse(m["reproducibility"]["cbs_calls_seed_consistent"])
        self.assertFalse(m["reproducibility"]["reproducible_batch"])

    def test_cbs_request_states_mark_different_problems(self):
        with patch.object(pv, "STARTUP_ENV_VALUE", None), patch.dict(os.environ, _env_without_seed(), clear=True):
            self.run_in_process()
        m = self.manifest()
        states = {s["method"]: s for s in m["cbs_request_states"]}
        self.assertEqual((states[PURE_CBS_DEV_SINGLE_PLAN]["request_kind"], states[PURE_CBS_DEV_SINGLE_PLAN]["request_timestep"]),
                         ("INITIAL_STATE", 0))
        self.assertEqual(states[GLOBAL_HYBRID_DEV_H3]["request_kind"], "IL_STALL_TRIGGER")
        self.assertGreaterEqual(states[GLOBAL_HYBRID_DEV_H3]["request_timestep"], 1)
        self.assertNotEqual(states[PURE_CBS_DEV_SINGLE_PLAN]["start_state_sha256"],
                            states[GLOBAL_HYBRID_DEV_H3]["start_state_sha256"])
        self.assertIn("different CBS planning problems", m["cbs_request_states_note"])


class TestCompareBatchProvenance(unittest.TestCase):

    def manifest(self, seed="0", reproducible=True, **cond):
        conditions = {"max_steps": 160, "cbs_timeout_sec": 30, "solver_root": "s", "checkpoint_sha256": "c",
                      "scenarios": {"x.json": "h"}, **cond}
        env = {"solver_git": {"commit": "89f"}, "project_git": {"commit": "729"}, "python": "3.14.6",
               "torch": "t", "numpy": "n", "pyyaml": "y", "platform": "p"}
        m = {"conditions": conditions, "environment": env}
        if seed != "legacy":
            m["reproducibility"] = {"seed_mode": pv.seed_mode(seed), "seed_value": seed if seed is not None else None,
                                    "reproducible_batch": reproducible}
        return m

    def test_verdicts(self):
        cases = {
            "SAME_SEED_COMPARABLE": [self.manifest("0"), self.manifest("0")],
            "SEED_SWEEP": [self.manifest("0"), self.manifest("1")],
            "LEGACY_UNSEEDED": [self.manifest("0"), self.manifest("legacy")],
            "UNSEEDED_OR_UNVERIFIED": [self.manifest("0"), self.manifest(None, reproducible=False)],
            "NOT_COMPARABLE": [self.manifest("0"), self.manifest("0", max_steps=100)],
        }
        for verdict, ms in cases.items():
            with self.subTest(verdict):
                self.assertEqual(pv.compare_batch_provenance(ms)["verdict"], verdict)
        self.assertIn("max_steps differs between batches",
                      pv.compare_batch_provenance(cases["NOT_COMPARABLE"])["reasons"])

    def test_legacy_seed_is_never_guessed(self):
        self.assertEqual(pv.manifest_seed(self.manifest("legacy")),
                         {"seed_mode": pv.NOT_RECORDED, "seed_value": None, "reproducible_batch": None})

    @unittest.skipUnless(OLD15_MANIFEST.exists(), "15-run Development batch not present")
    def test_existing_15_run_batch_is_not_recorded(self):
        before = OLD15_MANIFEST.read_bytes()
        seed = pv.manifest_seed(json.loads(before.decode("utf-8")))
        self.assertEqual(seed["seed_mode"], pv.NOT_RECORDED)
        self.assertEqual(OLD15_MANIFEST.read_bytes(), before)     # read only


class TestOutputPathLength(_BatchCase):

    def test_run_id_format_matches_make_run_id(self):
        paths = pv.output_paths(self.out, [CORRIDOR], [GLOBAL_HYBRID_DEV_H3], cbs_methods=CBS_METHODS)
        rid_len = len(make_run_id(CORRIDOR.name, GLOBAL_HYBRID_DEV_H3))
        self.assertTrue(any(Path(p).name == "solver_input.yaml" for p in paths))
        self.assertIn(rid_len, {len(Path(p).parts[-2]) for p in paths if p.endswith("_summary.csv")})

    def test_limits(self):
        args = (self.out, [CORRIDOR], [GLOBAL_HYBRID_DEV_H3])
        self.assertEqual(pv.path_length_errors(*args, cbs_methods=CBS_METHODS, limit=None), [])
        self.assertEqual(pv.path_length_errors(*args, cbs_methods=CBS_METHODS, limit=10_000), [])
        errs = pv.path_length_errors(*args, cbs_methods=CBS_METHODS, limit=50)
        self.assertEqual(len(errs), 1)
        self.assertIn("Windows MAX_PATH", errs[0])

    def test_batch_refused_before_creating_anything(self):
        longest = max(len(p) for p in pv.output_paths(self.out, [CORRIDOR], list(self.METHODS), cbs_methods=CBS_METHODS))
        with patch.object(pv, "path_limit", return_value=longest - 1):
            code, out = self.run_in_process()
        self.assertEqual(code, 2)
        self.assertIn(f"output path would be {longest} chars", out)
        self.assertFalse(self.out.exists())

    def test_prediction_covers_every_written_file(self):
        with patch.object(pv, "STARTUP_ENV_VALUE", None), patch.dict(os.environ, _env_without_seed(), clear=True):
            self.assertEqual(self.run_in_process()[0], 0)
        predicted = max(len(p) for p in pv.output_paths(self.out, [CORRIDOR], list(self.METHODS), cbs_methods=CBS_METHODS))
        written = [str(p.resolve()) for p in (self.out / "runs").rglob("*") if p.is_file()]
        self.assertTrue(written)
        self.assertLessEqual(max(len(p) for p in written), predicted)


if __name__ == "__main__":
    unittest.main()
