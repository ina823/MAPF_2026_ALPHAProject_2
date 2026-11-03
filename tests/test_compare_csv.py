"""src/compare/compare_csv.py: comparison rows, fairness, written-CSV checks.

Runs come from common_runner.run_method on the real Corridor scenario with
the real NavHint checkpoint; CBS methods use the real CbsRuntimePlanner +
PathGuardValidator with the recorded atb033 output injected (see
tests/test_common_runner.py), so no solver runs.
"""

import csv
import dataclasses
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_il import DEFAULT_CKPT
from src.compare.common_runner import (
    GLOBAL_HYBRID_DEV_FULL,
    GLOBAL_HYBRID_DEV_H3,
    IL_ONLY,
    METHODS,
    PURE_CBS_DEV_SINGLE_PLAN,
    SAFETY_ONLY_DEV_K2,
    RunSpec,
    run_method,
)
from src.compare.compare_csv import (
    CALL_FIELDS,
    EPISODE_FIELDS,
    CompareCsvError,
    build_rows,
    check_against_records,
    check_fairness,
    manifest_checkpoint_sha256,
    sum_of_costs,
    verify_written,
    write_rows,
)
from tests.test_common_runner import (
    CORRIDOR,
    _OverlapAtT1,
    _bad_start_solver,
    _timeout_solver,
    recorded_solver,
)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class _Case(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.solver_root = self.tmp / "solver"
        self.solver_root.mkdir()
        (self.solver_root / "cbs.py").write_text("raise SystemExit('fake solver must not run')\n",
                                                 encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def record(self, method, solver=recorded_solver, **kw):
        spec = RunSpec(method=method, scenario=CORRIDOR, out_dir=self.tmp / "runs",
                       solver_root=self.solver_root, **kw)
        with patch("src.cbs_runtime_adapter.replan_with_status", side_effect=solver):
            return run_method(spec)


class TestSumOfCosts(unittest.TestCase):

    def test_definition(self):
        goals = {0: (0, 2), 1: (1, 0)}
        traj = [{0: (0, 0), 1: (1, 0)}, {0: (0, 1), 1: (1, 1)},   # agent 1 leaves its goal ...
                {0: (0, 2), 1: (1, 0)}, {0: (0, 2), 1: (1, 0)}]   # ... and is back at t=2
        self.assertEqual(sum_of_costs(traj, goals), 2 + 2)
        self.assertEqual(sum_of_costs([{0: (0, 2)}], {0: (0, 2)}), 0)
        self.assertIsNone(sum_of_costs(traj[:2], goals))


@unittest.skipIf(torch is None, "torch is not installed")
class TestRowsPerMethod(_Case):

    def test_il_only(self):
        ep, calls = build_rows(self.record(IL_ONLY), "B")
        self.assertEqual(calls, [])
        self.assertEqual((ep["termination"], ep["success"], ep["steps"]), ("TIMEOUT", False, 160))
        self.assertIsNone(ep["makespan"])
        self.assertIsNone(ep["sum_of_costs"])
        for key in ("cbs_request_kind", "cbs_call_count", "cbs_latency_ms_total", "cbs_exec_steps",
                    "cbs_controlled_agent_steps", "cbs_timeout_sec", "safety_hold_agent_steps"):
            self.assertIsNone(ep[key], key)
        self.assertEqual((ep["il_steps"], ep["is_dev_setting"]), (160, False))
        self.assertEqual((ep["overlap_checked_at_runtime"], ep["integrity_unhandled"]), (False, False))
        self.assertEqual(ep["monitor_trigger_timestep"], 5)

    def test_safety_only(self):
        ep, calls = build_rows(self.record(SAFETY_ONLY_DEV_K2), "B")
        self.assertEqual((calls, ep["termination"], ep["safety_hold_agent_steps"]), ([], "TIMEOUT", 2))
        self.assertTrue(ep["is_dev_setting"])
        self.assertIn('"hold_steps": 2', ep["dev_settings"])

    def test_pure_cbs(self):
        ep, (call,) = build_rows(self.record(PURE_CBS_DEV_SINGLE_PLAN), "B")
        self.assertEqual((ep["termination"], ep["makespan"], ep["sum_of_costs"]), ("SUCCESS", 12, 12 + 11))
        self.assertEqual((ep["cbs_request_kind"], call["cbs_request_kind"], call["request_timestep"]),
                         ("INITIAL_STATE", "INITIAL_STATE", 0))
        self.assertEqual((call["planned_horizon"], call["exec_window"], call["exec_steps"]), (12, 12, 12))
        self.assertEqual((call["controlled_agent_steps"], call["exec_end"]), (24, "GOAL_REACHED"))
        self.assertEqual((call["exec_policy"], call["h_exec_config"]), ("DEV_SINGLE_PLAN_FULL", None))
        self.assertIsNotNone(call["planner_wall_ms"])
        self.assertIsNone(ep["checkpoint_sha256"])
        self.assertIsNone(ep["il_steps"])
        self.assertEqual(ep["cbs_latency_ms_total"], 50.0)

    def test_global_full(self):
        ep, (call,) = build_rows(self.record(GLOBAL_HYBRID_DEV_FULL), "B")
        self.assertEqual((ep["termination"], ep["makespan"], ep["il_steps"]), ("SUCCESS", 13, 5))
        self.assertEqual((call["cbs_request_kind"], call["request_timestep"]), ("IL_STALL_TRIGGER", 5))
        self.assertEqual((call["planned_horizon"], call["exec_window"], call["exec_steps"]), (8, 8, 8))
        self.assertEqual((call["exec_policy"], call["exec_end"]), ("DEV_FULL_PATH", "GOAL_REACHED"))
        self.assertIsNone(call["planner_wall_ms"])          # not recorded by the Global runner
        self.assertIsNone(ep["planner_wall_ms_total"])
        self.assertEqual(ep["cbs_controlled_agent_steps"], 16)

    def test_global_h3(self):
        ep, (call,) = build_rows(self.record(GLOBAL_HYBRID_DEV_H3), "B")
        self.assertEqual((call["h_exec_config"], call["planned_horizon"], call["exec_window"],
                          call["exec_steps"]), (3, 8, 3, 3))
        self.assertEqual((call["exec_policy"], call["exec_end"]), ("DEV_FIXED_H_EXEC", "WINDOW_END"))
        self.assertEqual((ep["il_resume_t"], ep["termination"], ep["steps"]), (8, "SUCCESS", 13))
        self.assertEqual(ep["il_steps"], 5 + 5)


@unittest.skipIf(torch is None, "torch is not installed")
class TestFailureRows(_Case):

    def test_pure_refused_no_step(self):
        ep, (call,) = build_rows(self.record(PURE_CBS_DEV_SINGLE_PLAN, solver=_bad_start_solver), "B")
        self.assertEqual((ep["termination"], ep["stop_reason"], ep["steps"]), ("SAFE_STOP", "CBS_INVALID_PATH", 0))
        self.assertEqual((call["exec_end"], call["exec_steps"], call["controlled_agent_steps"]), ("REFUSED", 0, 0))
        for key in ("planned_horizon", "exec_window", "agents_at_goal_at_exec_end"):
            self.assertIsNone(call[key], key)

    def test_global_timeout_refused(self):
        ep, (call,) = build_rows(self.record(GLOBAL_HYBRID_DEV_FULL, solver=_timeout_solver), "B")
        self.assertEqual((ep["termination"], ep["stop_reason"]), ("SAFE_STOP", "CBS_TIMEOUT"))
        self.assertEqual((call["status"], call["exec_end"], call["request_timestep"]), ("timeout", "REFUSED", 5))
        self.assertEqual(call["solver_latency_ms"], 30000.0)

    def test_pure_plan_longer_than_max_steps(self):
        ep, (call,) = build_rows(self.record(PURE_CBS_DEV_SINGLE_PLAN, max_steps=5), "B")
        self.assertEqual((ep["termination"], ep["timed_out"]), ("TIMEOUT", True))
        self.assertEqual((call["exec_window"], call["exec_steps"], call["exec_end"]), (12, 5, "EPISODE_TIMEOUT"))

    def test_il_only_unhandled_overlap(self):
        with patch("scripts.run_il.MAPFStepSimulator", _OverlapAtT1):
            ep, _ = build_rows(self.record(IL_ONLY, max_steps=4), "B")
        self.assertEqual((ep["overlap_any_step"], ep["first_overlap_t"], ep["integrity_unhandled"]), (True, 1, True))
        self.assertEqual(ep["overlap_checked_at_runtime"], False)


@unittest.skipIf(torch is None, "torch is not installed")
class TestTamperDetected(_Case):

    def test_steps_csv_edit(self):
        rec = self.record(IL_ONLY, max_steps=6)
        path = Path(rec.run_dir) / f"{rec.run_id}_steps.csv"
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fields, rows = reader.fieldnames, list(reader)
        rows[-1]["next_col"] = str(int(rows[-1]["next_col"]) + 1)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(CompareCsvError, "next position differs"):
            build_rows(rec, "B")

    def test_extra_solver_work_dir(self):
        rec = self.record(PURE_CBS_DEV_SINGLE_PLAN)
        (Path(rec.run_dir) / "cbs" / "cbs_call_99").mkdir()
        with self.assertRaisesRegex(CompareCsvError, "one to one"):
            build_rows(rec, "B")

    def test_ambiguous_trace(self):
        rec = self.record(GLOBAL_HYBRID_DEV_H3)
        trace = [dict(e) for e in rec.cbs_exec_trace]
        trace[1]["k"] = 5
        with self.assertRaisesRegex(CompareCsvError, "attributed unambiguously"):
            build_rows(dataclasses.replace(rec, cbs_exec_trace=trace), "B")


@unittest.skipIf(torch is None, "torch is not installed")
class TestFairnessAndWriting(_Case):

    @classmethod
    def setUpClass(cls):
        cls._shared = tempfile.TemporaryDirectory()
        root = Path(cls._shared.name)
        solver = root / "solver"
        solver.mkdir()
        (solver / "cbs.py").write_text("raise SystemExit(1)\n", encoding="utf-8")
        cls.records = []
        with patch("src.cbs_runtime_adapter.replan_with_status", side_effect=recorded_solver):
            for m in METHODS:
                cls.records.append(run_method(RunSpec(method=m, scenario=CORRIDOR, out_dir=root / "runs",
                                                      solver_root=solver)))
        cls.built = [build_rows(r, "B") for r in cls.records]

    @classmethod
    def tearDownClass(cls):
        cls._shared.cleanup()

    def clean_rows(self):
        rows = [dict(ep) for ep, _ in self.built]
        for r in rows:   # stand-ins for a committed tree and a clean solver clone
            r["project_dirty"] = False
            if r["method"] in (PURE_CBS_DEV_SINGLE_PLAN, GLOBAL_HYBRID_DEV_FULL, GLOBAL_HYBRID_DEV_H3):
                r["solver_commit"], r["solver_dirty"] = "c0ffee", False
        return rows

    def fairness(self, rows):
        return check_fairness(rows, scenarios=[CORRIDOR.name], methods=METHODS, max_steps=160,
                              cbs_timeout_sec=30, checkpoint_sha256=manifest_checkpoint_sha256(DEFAULT_CKPT))

    def test_fair_when_everything_matches(self):
        self.assertEqual(self.fairness(self.clean_rows()), (True, []))

    def test_each_mismatch_is_reported(self):
        cases = {
            "max_steps": lambda rows: rows[0].update(max_steps=20),
            "checkpoint_sha256": lambda rows: rows[1].update(checkpoint_sha256="other"),
            "cbs_timeout_sec": lambda rows: rows[2].update(cbs_timeout_sec=10),
            "solver_root": lambda rows: rows[3].update(solver_root="elsewhere"),
            "torch": lambda rows: rows[4].update(torch="0.0"),
            "project repository dirty": lambda rows: rows[0].update(project_dirty=True),
            "solver repository dirty": lambda rows: rows[2].update(solver_dirty=True),
            "scenarios x methods": lambda rows: rows.pop(),
            "grid_sha256": lambda rows: rows[1].update(grid_sha256="other"),
        }
        for label, edit in cases.items():
            with self.subTest(label):
                rows = self.clean_rows()
                edit(rows)
                fair, reasons = self.fairness(rows)
                self.assertFalse(fair)
                self.assertTrue(any(label in r for r in reasons), reasons)

    def test_write_verify_and_compare_with_runner(self):
        out = self.tmp / "csv"
        out.mkdir()
        episodes = [ep for ep, _ in self.built]
        calls = [c for _, cs in self.built for c in cs]
        write_rows(out / "e.csv", EPISODE_FIELDS, episodes)
        write_rows(out / "c.csv", CALL_FIELDS, calls)
        self.assertGreater(verify_written(out / "e.csv", EPISODE_FIELDS, episodes), 0)
        self.assertGreater(verify_written(out / "c.csv", CALL_FIELDS, calls), 0)
        self.assertGreater(check_against_records(out / "e.csv", out / "c.csv", self.records), 0)
        self.assertEqual(len(calls), 3)
        with self.assertRaises(FileExistsError):
            write_rows(out / "e.csv", EPISODE_FIELDS, episodes)
        text = (out / "c.csv").read_text(encoding="utf-8").replace("GOAL_REACHED", "WINDOW_END", 1)
        (out / "c.csv").write_text(text, encoding="utf-8")
        with self.assertRaises(CompareCsvError):
            verify_written(out / "c.csv", CALL_FIELDS, calls)


if __name__ == "__main__":
    unittest.main()
