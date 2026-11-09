"""scripts/run_pilot_v2_cbs.py with fake planners. No test here starts the solver."""

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
from unittest import mock

import numpy as np

from scripts import check_pilot_v2_candidates as chk
from scripts import generate_pilot_v2_candidates as genv2
from scripts import run_pilot_v2_cbs as rc
from src import cbs_adapter
from src.cbs_runtime_adapter import CBSRuntimeResult
from src.compare.compare_csv import sum_of_costs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_GIT = {"commit": rc.SOLVER_COMMIT, "dirty": False}
HASH_ENV = {"PYTHONHASHSEED": "0"}
_static_report_text = None


def static_report_text() -> str:
    """The static report rebuilt from committed inputs only (pre-registration,
    scenarios/pilot_v2, registry) by the real checker; outputs/ is not read."""
    global _static_report_text
    if _static_report_text is None:
        _static_report_text = json.dumps(chk.build_report(PROJECT_ROOT, PROJECT_ROOT / "registry"),
                                         indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return _static_report_text


def use_static_report_fixture(test: unittest.TestCase, directory: Path) -> Path:
    """Write the rebuilt report under ``directory`` and point the script's pins at it."""
    path = Path(directory) / "static_checks.json"
    data = static_report_text().encode("utf-8")
    path.write_bytes(data)
    for name, value in (("STATIC_REPORT", path), ("STATIC_REPORT_SHA256", hashlib.sha256(data).hexdigest())):
        patcher = mock.patch.object(rc, name, value)
        patcher.start()
        test.addCleanup(patcher.stop)
    return path


class FakePlanner:
    """Stands in for CbsRuntimePlanner; records calls and returns a fixed outcome."""
    instances = []

    def __init__(self, *, solver_root, work_root, timeout_sec, outcome=None):
        self.timeout_sec = timeout_sec
        self.work_root = work_root
        self.outcome = outcome
        self.calls = 0
        FakePlanner.instances.append(self)

    def __call__(self, starts, goals, grid):
        self.calls += 1
        return self.outcome(starts, goals, grid)


def factory(outcome):
    def make(**kwargs):
        return FakePlanner(outcome=outcome, **kwargs)
    return make


def timeout_outcome(starts, goals, grid):
    return CBSRuntimeResult("timeout", None, 120.05, "timed out")


class TestCheckPaths(unittest.TestCase):
    grid = np.zeros((3, 3), dtype=int)
    starts = {0: (0, 0), 1: (2, 2)}
    goals = {0: (0, 2), 1: (2, 0)}
    good = {0: [(0, 0), (0, 1), (0, 2)], 1: [(2, 2), (2, 1), (2, 0)]}

    def test_valid_paths_pass(self):
        check = rc.check_paths(self.grid, self.starts, self.goals, self.good)
        self.assertTrue(rc.paths_all_ok(check))

    def _bad(self, paths, *fields, starts=None, goals=None, grid=None):
        check = rc.check_paths(self.grid if grid is None else grid, starts or self.starts, goals or self.goals, paths)
        self.assertFalse(rc.paths_all_ok(check))
        self.assertFalse(check["path_guard_ok"])
        for f in fields:
            self.assertFalse(not check[f] if f.endswith(("conflicts", "moves", "cells")) else check[f], f)

    def test_path0_must_be_the_start(self):
        self._bad({0: [(0, 1), (0, 1), (0, 2)], 1: self.good[1]}, "path0_is_start")

    def test_agent_ids_must_be_preserved(self):
        self._bad({0: self.good[0], 7: self.good[1]}, "agent_ids_preserved")

    def test_swapped_ids_are_caught(self):
        self._bad({0: self.good[1], 1: self.good[0]}, "path0_is_start")

    def test_vertex_conflict(self):
        starts, goals = {0: (1, 0), 1: (1, 2)}, {0: (1, 2), 1: (1, 1)}
        check = rc.check_paths(self.grid, starts, goals, {0: [(1, 0), (1, 1), (1, 2)], 1: [(1, 2), (1, 1), (1, 1)]})
        self.assertTrue(check["vertex_conflicts"])
        self.assertFalse(rc.paths_all_ok(check))

    def test_edge_conflict(self):
        starts, goals = {0: (0, 0), 1: (0, 1)}, {0: (0, 1), 1: (0, 0)}
        check = rc.check_paths(self.grid, starts, goals, {0: [(0, 0), (0, 1)], 1: [(0, 1), (0, 0)]})
        self.assertTrue(check["edge_conflicts"])
        self.assertFalse(rc.paths_all_ok(check))

    def test_jump_and_wall(self):
        check = rc.check_paths(self.grid, {0: (0, 0)}, {0: (0, 2)}, {0: [(0, 0), (0, 2)]})
        self.assertTrue(check["illegal_moves"])
        walled = np.array([[0, 1, 0]])
        check = rc.check_paths(walled, {0: (0, 0)}, {0: (0, 2)}, {0: [(0, 0), (0, 1), (0, 2)]})
        self.assertTrue(check["wall_or_outside_cells"])
        self.assertFalse(rc.paths_all_ok(check))


class TestSolutionMetrics(unittest.TestCase):
    goal = (0, 2)

    def test_wait_padding_does_not_inflate_soc(self):
        short = {0: [(0, 0), (0, 1), (0, 2)], 1: [(1, 0), (1, 1)]}
        goals = {0: (0, 2), 1: (1, 1)}
        padded = {0: short[0] + [(0, 2)] * 7, 1: short[1] + [(1, 1)] * 8}
        a, b = rc.path_metrics(short, goals), rc.path_metrics(padded, goals)
        self.assertEqual((a["soc"], a["makespan"]), (3, 2))
        self.assertEqual((b["soc"], b["makespan"], b["settle_times"]), (a["soc"], a["makespan"], a["settle_times"]))
        self.assertEqual(b["padded_steps_total"], 18)

    def test_pv2c002_shape_gives_soc_40_not_44(self):
        lengths = {0: 11, 1: 11, 2: 8, 3: 10}
        goals = {a: (a, 100) for a in lengths}
        paths = {a: [(a, c) for c in range(100 - n, 101)] + [(a, 100)] * (11 - n) for a, n in lengths.items()}
        m = rc.path_metrics(paths, goals)
        self.assertEqual((m["soc"], m["makespan"], m["padded_steps_total"]), (40, 11, 44))

    def test_leaving_the_goal_uses_the_final_settling_time(self):
        path = [(0, 1), (0, 2), (0, 3), (0, 2), (0, 2)]
        self.assertEqual(rc.settle_time(path, self.goal), 3)
        m = rc.path_metrics({0: path}, {0: self.goal})
        self.assertEqual(m["soc"], 3)
        self.assertEqual(m["first_arrival_times"], {"0": 1})
        self.assertEqual(m["agents_left_goal_after_arrival"], [0])

    def test_start_on_goal_is_zero_even_after_padding(self):
        self.assertEqual(rc.settle_time([(0, 2), (0, 2), (0, 2)], self.goal), 0)

    def test_start_on_goal_then_leaving_and_returning_counts_the_return(self):
        self.assertEqual(rc.settle_time([(0, 2), (0, 1), (0, 2)], self.goal), 2)

    def test_goal_not_reached_has_no_soc(self):
        self.assertIsNone(rc.settle_time([(0, 0), (0, 1)], self.goal))
        m = rc.path_metrics({0: [(0, 0), (0, 1)]}, {0: self.goal})
        self.assertIsNone(m["soc"])
        self.assertIsNone(m["makespan"])

    def test_agrees_with_the_comparison_csv_definition(self):
        paths = {0: [(0, 0), (0, 1), (0, 2), (0, 2)], 1: [(2, 2), (2, 1), (2, 2), (2, 1)]}
        goals = {0: (0, 2), 1: (2, 1)}
        trajectory = [{a: paths[a][t] for a in paths} for t in range(4)]
        self.assertEqual(rc.path_metrics(paths, goals)["soc"], sum_of_costs(trajectory, goals))

    def test_solver_cost_is_kept_apart_from_soc(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "output.yaml"
            out.write_text("cost: 5\nschedule:\n  agent0:\n  - {t: 0, x: 0, y: 0}\n  - {t: 1, x: 1, y: 0}\n"
                           "  agent1:\n  - {t: 0, x: 0, y: 1}\n", encoding="utf-8")
            raw = rc.solver_output_metrics(out, {0: (0, 1), 1: (1, 0)})
        self.assertEqual(raw["solver_cost"], 5)
        self.assertEqual(raw["raw_schedule"]["soc"], 1)

    def test_missing_output_file_is_none_not_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(rc.solver_output_metrics(Path(d) / "x.yaml", {}),
                             {"solver_cost": None, "raw_schedule": None})

    def test_consistency_detects_a_mismatch_and_a_value_below_the_bound(self):
        good = rc.path_metrics({0: [(0, 0), (0, 1), (0, 2)]}, {0: self.goal})
        raw = {"solver_cost": 3, "raw_schedule": good}
        self.assertEqual(rc.metrics_consistent(good, raw, 2, 2),
                         {"padded_matches_raw_schedule": True, "at_or_above_lower_bounds": True})
        self.assertFalse(rc.metrics_consistent(good, raw, 3, 2)["at_or_above_lower_bounds"])
        other = rc.path_metrics({0: [(0, 1), (0, 2)]}, {0: self.goal})
        self.assertFalse(rc.metrics_consistent(good, {"solver_cost": 2, "raw_schedule": other}, 1, 1)
                         ["padded_matches_raw_schedule"])


class TestRecording(unittest.TestCase):
    def test_arguments_and_result_pass_through_and_module_is_restored(self):
        original = cbs_adapter.subprocess
        sink = {}
        with rc.record_solver_process(sink):
            r = cbs_adapter.subprocess.run([sys.executable, "-c", "print('hello')"], capture_output=True,
                                           text=True, timeout=30, check=False)
        self.assertIs(cbs_adapter.subprocess, original)
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "hello"))
        self.assertEqual((sink["returncode"], sink["stdout"].strip(), sink["timeout_arg"], sink["timed_out"]),
                         (0, "hello", 30, False))

    def test_timeout_propagates_and_is_recorded(self):
        sink = {}
        with rc.record_solver_process(sink):
            with self.assertRaises(subprocess.TimeoutExpired):
                cbs_adapter.subprocess.run([sys.executable, "-c", "import time; time.sleep(30)"],
                                           capture_output=True, text=True, timeout=0.2, check=False)
        self.assertTrue(sink["timed_out"])

    def test_restored_after_an_exception(self):
        original = cbs_adapter.subprocess
        with self.assertRaises(RuntimeError):
            with rc.record_solver_process({}):
                raise RuntimeError("boom")
        self.assertIs(cbs_adapter.subprocess, original)


FAKE_SOLVERS = {
    "success": "import sys\nprint('solver out')\nsys.stderr.write('solver err')\n"
               "open(sys.argv[2], 'w').write('schedule: {}\\n')\n",
    "crash": "import sys\nprint('solver out')\nsys.exit(3)\n",
    "no_solution": "print('Solution not found')\n",
    "slow": "import time\ntime.sleep(30)\n",
}


class TestWrapperDoesNotChangeTheAdapter(unittest.TestCase):
    """Real src.cbs_adapter.run_atb033_cbs, a fake cbs.py in a temporary directory,
    with and without the recording wrapper. The original solver is not involved."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.input = self.tmp / "input.yaml"
        self.input.write_text("map:\n  dimensions: [2, 2]\n  obstacles: [[1, 1]]\nagents: []\n", encoding="utf-8")
        self.output = self.tmp / "output.yaml"

    def _call(self, timeout):
        try:
            r = cbs_adapter.run_atb033_cbs(self.tmp, self.input, self.output, timeout_sec=timeout)
            return ("returned", r.returncode, r.stdout, r.stderr, list(map(str, r.args)))
        except Exception as exc:
            return ("raised", type(exc), str(exc))

    def _spied(self, wrapped, timeout):
        calls = []
        real = subprocess.run

        def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return real(*args, **kwargs)
        sink = {}
        with mock.patch("subprocess.run", spy):
            if wrapped:
                with rc.record_solver_process(sink):
                    outcome = self._call(timeout)
            else:
                outcome = self._call(timeout)
        output = self.output.read_bytes() if self.output.exists() else None
        return outcome, calls, output, sink

    def _compare(self, behaviour, timeout=20):
        (self.tmp / "cbs.py").write_text(FAKE_SOLVERS[behaviour], encoding="utf-8")
        plain = self._spied(False, timeout)
        wrapped = self._spied(True, timeout)
        self.assertEqual(wrapped[0], plain[0], "result or exception changed")
        self.assertEqual(wrapped[1], plain[1], "call arguments changed")
        self.assertEqual(wrapped[2], plain[2], "solver output file changed")
        self.assertEqual(len(plain[1]), 1)
        kwargs = plain[1][0][1]
        self.assertEqual(Path(kwargs["cwd"]), self.tmp.resolve())
        self.assertEqual(kwargs["timeout"], timeout)
        return plain[0], wrapped[3]

    def test_success_is_unchanged_and_logs_are_recorded(self):
        outcome, sink = self._compare("success")
        self.assertEqual(outcome[:4], ("returned", 0, "solver out\n", "solver err"))
        self.assertEqual((sink["returncode"], sink["stdout"], sink["stderr"], sink["timed_out"]),
                         (0, "solver out\n", "solver err", False))
        self.assertEqual(Path(sink["cwd"]), self.tmp.resolve())
        self.assertEqual(sink["timeout_arg"], 20)

    def test_nonzero_exit_is_unchanged(self):
        outcome, sink = self._compare("crash")
        self.assertIs(outcome[1], RuntimeError)
        self.assertEqual(sink["returncode"], 3)

    def test_no_solution_is_unchanged(self):
        outcome, sink = self._compare("no_solution")
        self.assertIs(outcome[1], cbs_adapter.CBSNoSolutionError)
        self.assertIn("Solution not found", sink["stdout"])

    def test_timeout_is_unchanged_and_stays_timeout(self):
        outcome, sink = self._compare("slow", timeout=1)
        self.assertIs(outcome[1], cbs_adapter.CBSTimeoutError)
        self.assertTrue(sink["timed_out"])
        self.assertIsNone(sink["returncode"])


class TestPreflight(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.solver = self.tmp / "solver"
        self.solver.mkdir()
        (self.solver / "cbs.py").write_text("# fake\n")
        self.out_root = self.tmp / "out"
        self.out_dir = self.out_root / "cbs_x"
        self.static_path = use_static_report_fixture(self, self.tmp)

    def _plan(self, env=HASH_ENV, git=CLEAN_GIT):
        return rc.preflight(self.solver, self.out_root, self.out_dir, env=env, git_info=lambda p: git)

    def test_clean_state_has_no_refusal_and_three_calls_of_120s(self):
        plan = self._plan()
        self.assertEqual(plan["refusals"], [])
        self.assertEqual((plan["expected_cbs_calls"], plan["timeout_sec_per_call"], plan["worst_case_wall_sec"]),
                         (3, 120, 360))
        self.assertEqual([c["candidate_id"] for c in plan["calls"]], ["PV2C-001", "PV2C-002", "PV2C-003"])

    def test_hash_seed_is_required(self):
        for env in ({}, {"PYTHONHASHSEED": "1"}, {"PYTHONHASHSEED": "random"}):
            self.assertTrue(any("PYTHONHASHSEED" in r for r in self._plan(env=env)["refusals"]), env)

    def test_wrong_or_dirty_solver_is_refused(self):
        self.assertTrue(any("commit" in r for r in self._plan(git={"commit": "abc", "dirty": False})["refusals"]))
        self.assertTrue(any("dirty" in r for r in self._plan(git={"commit": rc.SOLVER_COMMIT, "dirty": True})["refusals"]))
        self.assertTrue(any("dirty" in r for r in self._plan(git={"commit": rc.SOLVER_COMMIT, "dirty": None})["refusals"]))

    def test_an_earlier_run_of_the_same_candidate_is_refused_but_not_of_another(self):
        (self.out_root / "cbs_20200101_000000" / "pilot2_loop_w2_7x9").mkdir(parents=True)
        def plan(cid):
            return rc.preflight(self.solver, self.out_root, self.out_dir, env=HASH_ENV,
                                git_info=lambda p: CLEAN_GIT, candidate=cid)
        self.assertTrue(any("already exists" in r for r in plan("PV2C-002")["refusals"]))
        self.assertTrue(any("already exists" in r for r in self._plan()["refusals"]))
        self.assertEqual(plan("PV2C-001")["refusals"], [])

    def test_candidate_selects_exactly_one_call(self):
        plan = rc.preflight(self.solver, self.out_root, self.out_dir, env=HASH_ENV,
                            git_info=lambda p: CLEAN_GIT, candidate="PV2C-002")
        self.assertEqual(plan["refusals"], [])
        self.assertEqual([c["candidate_id"] for c in plan["calls"]], ["PV2C-002"])
        self.assertEqual((plan["expected_cbs_calls"], plan["worst_case_wall_sec"]), (1, 120))

    def test_unknown_candidate_is_refused(self):
        plan = rc.preflight(self.solver, self.out_root, self.out_dir, env=HASH_ENV,
                            git_info=lambda p: CLEAN_GIT, candidate="PV2C-999")
        self.assertTrue(any("unknown candidate" in r for r in plan["refusals"]))
        self.assertEqual(plan["calls"], [])

    def test_changed_or_missing_static_report_is_refused(self):
        self.assertEqual(self._plan()["refusals"], [])
        self.static_path.write_bytes(self.static_path.read_bytes() + b" ")
        self.assertTrue(any("static_checks.json" in r for r in self._plan()["refusals"]))
        self.static_path.unlink()
        self.assertTrue(any("static_checks.json" in r for r in self._plan()["refusals"]))

    def test_missing_solver_is_refused(self):
        (self.solver / "cbs.py").unlink()
        self.assertTrue(any("cbs.py" in r for r in self._plan()["refusals"]))


class TestStaticReportFixture(unittest.TestCase):
    def test_rebuilt_report_has_no_problem_and_keeps_the_prereg_grades(self):
        report = json.loads(static_report_text())
        self.assertEqual(report["problems"], [])
        self.assertEqual(report["prereg_sha256"], genv2.PREREG_SHA256)
        self.assertFalse(report["cbs_run"])
        self.assertEqual([(c["candidate_id"], c["grade"]) for c in report["candidates"]],
                         [(s["candidate_id"], s["grade"]) for s in genv2.load_prereg()["candidates"]])

    def test_pins_are_restored_after_the_test(self):
        pinned = rc.STATIC_REPORT, rc.STATIC_REPORT_SHA256
        with tempfile.TemporaryDirectory() as d:
            inner = unittest.TestCase()
            path = use_static_report_fixture(inner, Path(d))
            self.assertEqual(rc.STATIC_REPORT, path)
            inner.doCleanups()
        self.assertEqual((rc.STATIC_REPORT, rc.STATIC_REPORT_SHA256), pinned)


class TestRunAll(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.out = self.tmp / "cbs_x"
        self.specs = genv2.load_prereg()["candidates"]
        self.static = json.loads(static_report_text())
        FakePlanner.instances = []

    def _run(self, outcome):
        with redirect_stdout(io.StringIO()):
            return rc.run_all({"note": "test"}, self.out, self.tmp / "solver", PROJECT_ROOT, self.static, self.specs,
                              planner_factory=factory(outcome))

    def test_timeout_stays_timeout_for_every_candidate_and_is_not_no_solution(self):
        report = self._run(timeout_outcome)
        for r in report["results"]:
            self.assertEqual(r["status"], "timeout")
            self.assertEqual(r["interpretation"]["cbs_status"], "timeout")
            self.assertEqual(r["interpretation"]["solution_existence"], "UNKNOWN")
            self.assertFalse(r["interpretation"]["cbs_solved"])
            self.assertIsNone(r["path_check"])
        self.assertEqual(rc.exit_code(report), 0)

    def test_each_candidate_is_called_once_in_order_with_120s(self):
        self._run(timeout_outcome)
        self.assertEqual([p.calls for p in FakePlanner.instances], [1, 1, 1])
        self.assertEqual({p.timeout_sec for p in FakePlanner.instances}, {120})
        self.assertEqual([Path(p.work_root).name for p in FakePlanner.instances], [s["slug"] for s in self.specs])
        events = [json.loads(x) for x in (self.out / "attempts.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([e["event"] for e in events], ["started", "finished"] * 3)

    def test_grades_target_and_holds_are_carried_unchanged(self):
        report = self._run(timeout_outcome)
        got = {r["candidate_id"]: (r["grade"], r["target_met_static"], r["ood_class_static"], r["holds_unchanged"])
               for r in report["results"]}
        self.assertEqual(got, {"PV2C-001": ("NORMAL", False, "IN_RANGE", ["PV2R0001"]),
                               "PV2C-002": ("NORMAL", True, "IN_RANGE", ["PV2R0002"]),
                               "PV2C-003": ("STRESS", True, "OOD_TAIL", ["PV2R0003", "PV2R0004"])})
        self.assertFalse(report["registry_written"])
        self.assertTrue(report["protected_unchanged"])

    def test_success_with_bad_paths_fails_the_run_and_is_kept(self):
        def bad(starts, goals, grid):
            return CBSRuntimeResult("success", {a: [(0, 0)] for a in starts}, 0.1)
        report = self._run(bad)
        for r in report["results"]:
            self.assertEqual(r["status"], "success")
            self.assertFalse(r["path_check_all_ok"])
            self.assertFalse(r["interpretation"]["cbs_solved"])
        self.assertEqual(rc.exit_code(report), 1)

    def test_success_records_soc_settle_times_and_cost_separately(self):
        def staggered(starts, goals, grid):
            # one agent per step behind the next, padded to the same length as the adapter does
            paths = {}
            for a in starts:
                paths[a] = [starts[a]] * (a + 1) + [goals[a]]
            return CBSRuntimeResult("success", paths, 0.1)
        report = self._run(staggered)
        for r in report["results"]:
            n = len(r["settle_times"])
            self.assertEqual(r["soc"], sum(a + 1 for a in range(n)))
            self.assertEqual(r["makespan"], n)
            self.assertNotIn("sum_of_path_lengths", r)
            self.assertIsNone(r["solver_cost"])                      # fake planner writes no output.yaml
            self.assertIsNone(r["metrics_check"]["padded_matches_raw_schedule"])

    def test_a_metrics_mismatch_fails_the_run_and_is_kept(self):
        report = {"protected_unchanged": True, "results": [
            {"status": "success", "path_check_all_ok": True,
             "metrics_check": {"padded_matches_raw_schedule": False, "at_or_above_lower_bounds": True}}]}
        self.assertEqual(rc.exit_code(report), 1)
        report["results"][0]["metrics_check"]["padded_matches_raw_schedule"] = None
        self.assertEqual(rc.exit_code(report), 0)

    def test_a_script_failure_is_not_a_cbs_status_and_the_next_candidate_still_runs(self):
        seen = []

        def failing_then_timeout(starts, goals, grid):
            seen.append(1)
            if len(seen) == 1:
                raise KeyError("boom")
            return timeout_outcome(starts, goals, grid)
        report = self._run(failing_then_timeout)
        self.assertEqual([r["status"] for r in report["results"]], [rc.SCRIPT_EXCEPTION, "timeout", "timeout"])
        self.assertIsNone(report["results"][0]["interpretation"])
        self.assertEqual(rc.exit_code(report), 1)

    def test_a_used_output_directory_is_refused(self):
        self._run(timeout_outcome)
        with self.assertRaises(FileExistsError):
            self._run(timeout_outcome)

    def test_files_are_kept_per_candidate(self):
        self._run(timeout_outcome)
        for name in ("run_manifest.json", "attempts.jsonl", "cbs_results.json"):
            self.assertTrue((self.out / name).is_file(), name)
        for s in self.specs:
            for name in ("result.json", "solver_stdout.txt", "solver_stderr.txt"):
                self.assertTrue((self.out / s["slug"] / name).is_file(), name)


class TestMain(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.out_root = self.tmp / "out"
        self.out_root.mkdir()
        (self.out_root / "generation_existing").mkdir()
        use_static_report_fixture(self, self.tmp)
        for patcher in (mock.patch.object(rc, "OUT_ROOT", self.out_root),
                        mock.patch.dict(os.environ, HASH_ENV)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _listing(self):
        return sorted(str(p.relative_to(self.out_root)) for p in self.out_root.rglob("*"))

    def test_without_run_nothing_is_called_or_written(self):
        before = self._listing()
        with mock.patch.object(rc, "run_all") as run_all, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            rc.main([])
        run_all.assert_not_called()
        self.assertEqual(self._listing(), before)

    def test_clean_plan_without_run_returns_zero_and_writes_nothing(self):
        before = self._listing()
        out = io.StringIO()
        with mock.patch.object(rc, "preflight", return_value={"refusals": [], "calls": []}), \
                mock.patch.object(rc, "run_all") as run_all, redirect_stdout(out):
            self.assertEqual(rc.main(["--candidate", "PV2C-002"]), 0)
        run_all.assert_not_called()
        self.assertIn("plan only", out.getvalue())
        self.assertEqual(self._listing(), before)

    def test_run_without_candidate_is_refused_before_anything_happens(self):
        err = io.StringIO()
        with mock.patch.object(rc, "preflight") as preflight, mock.patch.object(rc, "run_all") as run_all, \
                redirect_stderr(err):
            self.assertEqual(rc.main(["--run"]), 2)
        preflight.assert_not_called()
        run_all.assert_not_called()
        self.assertIn("--candidate", err.getvalue())

    def test_timeout_is_fixed_and_not_an_argument(self):
        self.assertEqual(rc.TIMEOUT_SEC, 120)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            rc.main(["--timeout-sec", "600"])


if __name__ == "__main__":
    unittest.main()
