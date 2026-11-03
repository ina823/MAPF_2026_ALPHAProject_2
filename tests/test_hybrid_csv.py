"""src/hybrid/hybrid_csv.py + scripts/export_hybrid_csv.py: Global-only CSV export.

Run directories are produced the way the two DEV scripts produce them: the
real Global Hybrid runner with MOCK failure injection (test-only planner and
AcceptAllValidator, stub monitor for multi-call), the scripts' own
build_report(), EpisodeLogger steps CSV and result.json -- all inside a
temporary directory. The last class reads the saved real-CBS runs under
outputs/hybrid_e2e/ read-only (skipped when they are not present).
"""

import contextlib
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.export_hybrid_csv import main as export_main
from scripts.run_corridor_hybrid_e2e import (
    MODE as FULL_MODE,
    DevRecordSafeStop,
    _jsonable,
    build_report as build_full_report,
    dev_config,
    evaluate,
)
from scripts.run_hybrid_resume_dev import (
    ActRecordingPolicy,
    DevResumeIL,
    build_report as build_fixed_report,
    dev_resume_config,
    evaluate_resume,
    mode_name,
    step_recording_simulator,
)
from scripts.run_il import PROJECT_ROOT
from src.common.deadlock_monitor import DeadlockMonitor
from src.common.episode_logger import EpisodeLogger
from src.hybrid.global_hybrid import GlobalHybridConfig, run_global_hybrid_episode
from src.hybrid.hybrid_csv import (
    CALL_FIELDS,
    CALLS_CSV,
    EPISODE_FIELDS,
    EPISODES_CSV,
    MANIFEST_JSON,
    HybridCsvError,
    build_rows,
    export_runs,
    format_value,
    load_run,
)
from src.simulator.mapf_step_simulator import MAPFStepSimulator
from tests.hybrid_mocks import (
    GOALS,
    STARTS,
    AcceptAllValidator,
    AllowAllRecalls,
    DenyAllRecalls,
    MockReplanResult,
    ScriptedPolicy,
    open_grid,
    perturbing_simulator,
)


def straight_paths(current, goals):
    """Row-then-column moves, padded to equal length (open grid only)."""
    paths = {}
    for aid, (r, c) in current.items():
        gr, gc = goals[aid]
        path = [(r, c)]
        while c != gc:
            c += 1 if gc > c else -1
            path.append((r, c))
        while r != gr:
            r += 1 if gr > r else -1
            path.append((r, c))
        paths[aid] = path
    length = max(len(p) for p in paths.values())
    return {aid: p + [p[-1]] * (length - len(p)) for aid, p in paths.items()}


class ScriptPlanner:
    """TEST ONLY: agent-id keyed like CbsRuntimePlanner, records calls the same way."""

    test_only = True
    paths_keying = "agent_id"

    def __init__(self, solver_root, result=None):
        self.solver_root = solver_root
        self.result = result
        self.calls = []

    def __call__(self, current_positions, goals, grid):
        result = self.result or MockReplanResult(
            status="success", paths=straight_paths(current_positions, goals), latency_sec=0.25)
        self.calls.append({"artifacts_dir": "", "paths": result.paths if result.status == "success" else None})
        return result


class StubMonitor:
    """STUB: fresh TRIGGER_FULL_REPEAT at each listed t, CLEAN integrity."""

    def __init__(self, trigger_ts):
        self.trigger_ts = trigger_ts

    def reset(self, positions, goals):
        pass

    def observe_step(self, timestep, positions, conflict_types, **kwargs):
        fired = [t for t in self.trigger_ts if t <= timestep]
        return SimpleNamespace(
            trigger_reason_code="TRIGGER_FULL_REPEAT" if fired else "TRIGGER_NONE",
            trigger_timestep=fired[-1] if fired else None,
            overlap_integrity_status="CLEAN",
        )


def make_run(root, run_id, *, kind="full", result=None, h_exec=1, max_steps=8,
             sim_base=MAPFStepSimulator, monitor=None, config=None):
    """One run directory exactly as the DEV scripts write it (result.json + steps CSV)."""
    run_dir = Path(root) / run_id
    run_dir.mkdir(parents=True)
    planner = ScriptPlanner(Path(root), result)
    mode = FULL_MODE if kind == "full" else mode_name(h_exec)
    logger = EpisodeLogger(run_id, scenario="scenario_synthetic.json", mode=mode, out_dir=run_dir)
    policy = ActRecordingPolicy(ScriptedPolicy())
    sim_cls = step_recording_simulator(sim_base)
    if config is None:
        config = dev_config(max_steps) if kind == "full" else dev_resume_config(h_exec)
    episode = run_global_hybrid_episode(
        policy, open_grid(), STARTS, GOALS, max_steps,
        planner=planner, path_validator=AcceptAllValidator(), config=config,
        monitor=monitor or DeadlockMonitor(), logger=logger, simulator_cls=sim_cls,
    )
    logger.write_steps()
    common = dict(scenario="scenarios/synthetic/scenario_synthetic.json", max_steps=max_steps,
                  ckpt="none", reproduce="test")
    if kind == "full":
        report = build_full_report(episode, planner, evaluate(episode, planner.calls), **common)
    else:
        evaluation = evaluate_resume(episode, planner.calls, policy.act_positions, sim_cls.step_ts, h_exec)
        report = build_fixed_report(episode, planner, evaluation, h_exec=h_exec, **common)
    (run_dir / "result.json").write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    return run_dir


def rows_of(run_dir):
    return build_rows(load_run(run_dir))


class _TmpCase(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


# -- Successful runs ------------------------------------------------------------

class TestSuccessRows(_TmpCase):

    def test_full_path_success(self):
        ep, calls = rows_of(make_run(self.tmp, "full_ok"))
        self.assertEqual((ep["termination"], ep["verdict"], ep["all_at_goal"]), ("SUCCESS", "PASS", True))
        self.assertEqual(ep["h_exec_policy"], "DEV_ONLY_FULL_PATH")
        self.assertIsNone(ep["h_exec_config"])          # max_steps is never read as H_exec
        self.assertEqual((ep["trigger_timestep"], ep["first_seen"], ep["period"]), (1, 0, 1))
        self.assertEqual(ep["first_seen_source"], "MONITOR_REPLAY_TRAJECTORY")
        self.assertEqual((ep["cbs_exec_steps"], ep["cbs_controlled_agent_steps"], ep["il_steps"]), (2, 4, 1))
        self.assertIsNone(ep["il_resume_t"])
        (call,) = calls
        self.assertEqual((call["planned_horizon"], call["exec_window"], call["exec_steps"]), (2, 2, 2))
        self.assertEqual((call["exec_end"], call["agents_at_goal_at_exec_end"]), ("GOAL_REACHED", 2))
        self.assertEqual(call["solver_latency_ms"], 250.0)
        self.assertEqual(ep["cbs_latency_ms_total"], 250.0)

    def test_fixed_window_resume_then_timeout_stays_fail(self):
        ep, calls = rows_of(make_run(self.tmp, "fixed_timeout", kind="fixed", h_exec=1, max_steps=6))
        self.assertEqual((ep["termination"], ep["verdict"], ep["all_at_goal"]), ("TIMEOUT", "FAIL", False))
        self.assertEqual((ep["h_exec_policy"], ep["h_exec_config"]), ("DEV_ONLY_FIXED", 1))
        self.assertEqual(ep["il_resume_t"], 2)
        self.assertEqual(ep["resume_outcome"], "RESUME_TIMEOUT_RESTALL")
        self.assertIsNone(ep["stop_reason"])
        (call,) = calls
        # Configured window vs. path length vs. executed steps stay separate.
        self.assertEqual((call["h_exec_config"], call["planned_horizon"], call["exec_window"],
                          call["exec_steps"]), (1, 2, 1, 1))
        self.assertEqual((call["exec_end"], call["agents_at_goal_at_exec_end"]), ("WINDOW_END", 0))

    def test_no_cbs_call(self):
        config = GlobalHybridConfig(h_exec=8, resume_policy=DevResumeIL(),
                                    recall_policy=DenyAllRecalls(), safe_stop_handler=DevRecordSafeStop())
        ep, calls = rows_of(make_run(self.tmp, "no_call", config=config, max_steps=4))
        self.assertEqual(calls, [])
        self.assertEqual((ep["cbs_call_count"], ep["cbs_exec_steps"], ep["cbs_controlled_agent_steps"]), (0, 0, 0))
        self.assertIsNone(ep["trigger_timestep"])
        self.assertIsNone(ep["first_seen"])
        self.assertEqual(ep["first_seen_source"], "NO_CBS_CALL")
        self.assertIsNone(ep["cbs_latency_ms_total"])  # empty, not 0
        self.assertEqual(ep["termination"], "TIMEOUT")


# -- Failure records ------------------------------------------------------------

class TestFailureRows(_TmpCase):

    def _refused(self, run_id, status, error):
        result = MockReplanResult(status=status, latency_sec=30.0, error=error)
        ep, calls = rows_of(make_run(self.tmp, run_id, result=result))
        (call,) = calls
        self.assertEqual((ep["termination"], ep["stop_reason"]), ("SAFE_STOP", f"CBS_{status.upper()}"))
        self.assertEqual((call["status"], call["accepted"], call["reason"], call["error"]),
                         (status, False, f"CBS_{status.upper()}", error))
        self.assertEqual((call["exec_end"], call["exec_steps"], call["controlled_agent_steps"]), ("REFUSED", 0, 0))
        self.assertIsNone(call["exec_window"])
        self.assertIsNone(call["planned_horizon"])
        self.assertIsNone(call["agents_at_goal_at_exec_end"])
        self.assertEqual(call["solver_latency_ms"], 30000.0)
        self.assertEqual(ep["cbs_exec_steps"], 0)

    def test_timeout(self):
        self._refused("cbs_timeout", "timeout", "atb033 CBS timed out")

    def test_invalid_path(self):
        self._refused("cbs_invalid", "invalid_path", "Agent 0, t=0->1: illegal movement")

    def test_exec_mismatch(self):
        sim = perturbing_simulator(inject_t=2, agent_id=0, new_pos=(1, 1))
        ep, calls = rows_of(make_run(self.tmp, "mismatch", sim_base=sim))
        (call,) = calls
        self.assertEqual((ep["termination"], ep["stop_reason"]), ("SAFE_STOP", "EXEC_MISMATCH"))
        self.assertEqual((call["accepted"], call["exec_end"], call["exec_steps"]), (True, "EXEC_MISMATCH", 1))
        self.assertEqual(ep["cbs_controlled_agent_steps"], 2)

    def test_overlap(self):
        sim = perturbing_simulator(inject_t=2, agent_id=1, new_pos=(0, 1))
        ep, calls = rows_of(make_run(self.tmp, "overlap", sim_base=sim))
        self.assertEqual((ep["termination"], ep["overlap_any_step"], ep["stop_reason"]),
                         ("INTEGRITY_ABORT", True, None))
        self.assertEqual(calls[0]["exec_end"], "INTEGRITY_ABORT")


# -- Several CBS calls and attribution ------------------------------------------

class TestMultipleCalls(_TmpCase):

    def _two_call_run(self):
        config = GlobalHybridConfig(h_exec=1, resume_policy=DevResumeIL(),
                                    recall_policy=AllowAllRecalls(), safe_stop_handler=DevRecordSafeStop())
        return make_run(self.tmp, "two_calls", kind="fixed", h_exec=1, monitor=StubMonitor([1, 3]),
                        config=config)

    def test_one_row_per_call(self):
        ep, calls = rows_of(self._two_call_run())
        self.assertEqual([c["call_index"] for c in calls], [1, 2])
        self.assertEqual([c["trigger_timestep"] for c in calls], [1, 3])
        self.assertEqual([c["exec_steps"] for c in calls], [1, 1])
        self.assertEqual([c["exec_end"] for c in calls], ["WINDOW_END", "GOAL_REACHED"])
        self.assertEqual((ep["cbs_call_count"], ep["cbs_exec_steps"], ep["cbs_controlled_agent_steps"]), (2, 2, 4))
        self.assertEqual(ep["cbs_latency_ms_total"], 500.0)
        self.assertEqual(ep["termination"], "SUCCESS")

    def test_ambiguous_attribution_is_refused(self):
        run_dir = self._two_call_run()
        report = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        report["cbs_exec_trace"][1]["k"] = 2   # call 2's step relabelled as a 2nd step of call 1
        (run_dir / "result.json").write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(HybridCsvError, "attributed unambiguously"):
            rows_of(run_dir)


# -- Broken or inconsistent input -----------------------------------------------

class TestInputErrors(_TmpCase):

    def _edit_report(self, run_dir, edit):
        path = run_dir / "result.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        edit(report)
        path.write_text(json.dumps(report), encoding="utf-8")

    def test_missing_field(self):
        run_dir = make_run(self.tmp, "no_term")
        self._edit_report(run_dir, lambda r: r.pop("termination"))
        with self.assertRaisesRegex(HybridCsvError, "missing field.*termination"):
            rows_of(run_dir)

    def test_missing_call_field(self):
        run_dir = make_run(self.tmp, "no_latency")
        self._edit_report(run_dir, lambda r: r["cbs_calls"][0].pop("latency_ms"))
        with self.assertRaisesRegex(HybridCsvError, r"cbs_calls\[1\] missing field.*latency_ms"):
            rows_of(run_dir)

    def test_broken_json(self):
        run_dir = make_run(self.tmp, "broken")
        (run_dir / "result.json").write_text("{", encoding="utf-8")
        with self.assertRaisesRegex(HybridCsvError, "not valid JSON"):
            load_run(run_dir)

    def test_missing_steps_csv(self):
        run_dir = make_run(self.tmp, "no_steps")
        (run_dir / "no_steps_steps.csv").unlink()
        with self.assertRaisesRegex(HybridCsvError, "missing"):
            load_run(run_dir)

    def test_pass_verdict_on_timeout_is_refused(self):
        run_dir = make_run(self.tmp, "fake_pass", kind="fixed", max_steps=6)
        self._edit_report(run_dir, lambda r: r.update(verdict="PASS"))
        with self.assertRaisesRegex(HybridCsvError, "verdict PASS on a TIMEOUT run"):
            rows_of(run_dir)

    def test_unknown_h_exec_policy(self):
        run_dir = make_run(self.tmp, "odd_mode")
        self._edit_report(run_dir, lambda r: r.update(mode="GLOBAL_HYBRID_OTHER"))
        with self.assertRaisesRegex(HybridCsvError, "unknown H_exec policy"):
            rows_of(run_dir)

    def test_steps_csv_row_missing(self):
        run_dir = make_run(self.tmp, "short_csv")
        path = run_dir / "short_csv_steps.csv"
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        path.write_text("".join(lines[:-1]), encoding="utf-8")
        with self.assertRaisesRegex(HybridCsvError, "rows, expected"):
            rows_of(run_dir)

    def test_steps_csv_position_disagrees(self):
        run_dir = make_run(self.tmp, "moved_csv")
        path = run_dir / "moved_csv_steps.csv"
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fields, rows = reader.fieldnames, list(reader)
        rows[-1]["next_col"] = str(int(rows[-1]["next_col"]) + 1)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(HybridCsvError, "next position differs"):
            rows_of(run_dir)


# -- Export: files, JSON<->CSV, no overwrite ------------------------------------

class TestExport(_TmpCase):

    def _snapshot(self, root):
        return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(root).rglob("*")) if p.is_file()}

    def test_export_writes_new_dir_and_matches_json(self):
        runs_root = self.tmp / "runs"
        dirs = [make_run(runs_root, "a_full"),
                make_run(runs_root, "b_fixed", kind="fixed", max_steps=6),
                make_run(runs_root, "c_timeout", result=MockReplanResult(status="timeout", latency_sec=1.5,
                                                                         error="t/o"))]
        before = self._snapshot(runs_root)
        out = export_runs(dirs, self.tmp / "export")
        self.assertEqual(self._snapshot(runs_root), before)       # inputs untouched
        self.assertGreater(out["checks"], 0)
        with (self.tmp / "export" / EPISODES_CSV).open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            self.assertEqual(reader.fieldnames, list(EPISODE_FIELDS))
            episodes = list(reader)
        with (self.tmp / "export" / CALLS_CSV).open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            self.assertEqual(reader.fieldnames, list(CALL_FIELDS))
            calls = list(reader)
        self.assertEqual([e["run_id"] for e in episodes], ["a_full", "b_fixed", "c_timeout"])
        self.assertEqual(len(calls), 3)
        self.assertEqual(episodes[0]["h_exec_config"], "")
        self.assertEqual(calls[2]["solver_latency_ms"], "1500.0")
        self.assertEqual(calls[2]["exec_window"], "")
        manifest = json.loads((self.tmp / "export" / MANIFEST_JSON).read_text(encoding="utf-8"))
        self.assertEqual([i["run_id"] for i in manifest["inputs"]], ["a_full", "b_fixed", "c_timeout"])
        self.assertEqual(manifest["inputs"][0]["result_json_sha256"], before["a_full/result.json"])

    def test_duplicate_run_id_refused_before_writing(self):
        run_dir = make_run(self.tmp / "runs", "dup")
        with self.assertRaisesRegex(HybridCsvError, "duplicate run_id"):
            export_runs([run_dir, run_dir], self.tmp / "export")
        self.assertFalse((self.tmp / "export").exists())

    def test_invalid_run_writes_nothing(self):
        good = make_run(self.tmp / "runs", "good")
        bad = make_run(self.tmp / "runs", "bad")
        (bad / "result.json").write_text("{", encoding="utf-8")
        with self.assertRaises(HybridCsvError):
            export_runs([good, bad], self.tmp / "export")
        self.assertFalse((self.tmp / "export").exists())

    def test_existing_output_dir_never_overwritten(self):
        run_dir = make_run(self.tmp / "runs", "one")
        out = self.tmp / "export"
        out.mkdir()
        (out / EPISODES_CSV).write_text("SENTINEL\n", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            export_runs([run_dir], out)
        with contextlib.redirect_stdout(io.StringIO()):
            code = export_main(["--run-dir", str(run_dir), "--output-dir", str(out)])
        self.assertEqual(code, 2)
        self.assertEqual((out / EPISODES_CSV).read_text(encoding="utf-8"), "SENTINEL\n")

    def test_cli_exports(self):
        run_dir = make_run(self.tmp / "runs", "cli")
        with contextlib.redirect_stdout(io.StringIO()):
            code = export_main(["--run-dir", str(run_dir), "--output-dir", str(self.tmp / "export")])
        self.assertEqual(code, 0)
        self.assertTrue((self.tmp / "export" / CALLS_CSV).is_file())

    def test_format_value(self):
        self.assertEqual([format_value(v) for v in (None, True, False, 0, 0.1 + 0.2, "x")],
                         ["", "True", "False", "0", "0.30000000000000004", "x"])


# -- Saved real-CBS runs (read only) ---------------------------------------------

SAVED_ROOT = PROJECT_ROOT / "outputs" / "hybrid_e2e"
# run_id -> (termination, verdict, h_exec_policy, h_exec_config, cbs_exec_steps, agents, resume_outcome)
SAVED_RUNS = {
    "20261030_224821_corridor_bay_n2_GLOBAL_HYBRID_DEV": ("SUCCESS", "PASS", "DEV_ONLY_FULL_PATH", None, 8, 2, None),
    "20261103_151303_corridor_bay_n2_GLOBAL_HYBRID_DEV": ("SUCCESS", "PASS", "DEV_ONLY_FULL_PATH", None, 8, 2, None),
    "20261103_151308_intersection_n2_GLOBAL_HYBRID_DEV": ("SUCCESS", "PASS", "DEV_ONLY_FULL_PATH", None, 4, 2, None),
    "20261103_151313_bottleneck_n4_GLOBAL_HYBRID_DEV": ("SUCCESS", "PASS", "DEV_ONLY_FULL_PATH", None, 13, 4, None),
    "20261103_152638_corridor_bay_n2_GLOBAL_HYBRID_DEV_H3": ("SUCCESS", "PASS", "DEV_ONLY_FIXED", 3, 3, 2, "RESUME_SUCCESS"),
    "20261103_152700_intersection_n2_GLOBAL_HYBRID_DEV_H3": ("SUCCESS", "PASS", "DEV_ONLY_FIXED", 3, 3, 2, "RESUME_SUCCESS"),
    "20261103_152705_bottleneck_n4_GLOBAL_HYBRID_DEV_H3": ("TIMEOUT", "FAIL", "DEV_ONLY_FIXED", 3, 3, 4, "RESUME_TIMEOUT_RESTALL"),
}
HAVE_SAVED = all((SAVED_ROOT / run_id / "result.json").is_file() for run_id in SAVED_RUNS)


@unittest.skipUnless(HAVE_SAVED, "saved real-CBS runs not present under outputs/hybrid_e2e")
class TestSavedRealRuns(_TmpCase):

    def test_saved_runs_export_read_only(self):
        dirs = [SAVED_ROOT / run_id for run_id in SAVED_RUNS]
        before = {d.name: {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in d.iterdir() if p.is_file()} for d in dirs}
        out = export_runs(dirs, self.tmp / "export")
        after = {d.name: {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in d.iterdir() if p.is_file()} for d in dirs}
        self.assertEqual(after, before)
        episodes = {e["run_id"]: e for e in out["episodes"]}
        for run_id, (term, verdict, policy, h, cbs_steps, agents, outcome) in SAVED_RUNS.items():
            with self.subTest(run_id):
                ep = episodes[run_id]
                self.assertEqual((ep["termination"], ep["verdict"]), (term, verdict))
                self.assertEqual(ep["all_at_goal"], term == "SUCCESS")
                self.assertEqual((ep["h_exec_policy"], ep["h_exec_config"]), (policy, h))
                self.assertEqual(ep["cbs_exec_steps"], cbs_steps)
                self.assertEqual(ep["cbs_controlled_agent_steps"], cbs_steps * agents)
                self.assertEqual(ep["resume_outcome"], outcome)
                self.assertEqual((ep["cbs_call_count"], ep["overlap_any_step"]), (1, False))
                self.assertEqual(ep["first_seen_source"], "MONITOR_REPLAY_TRAJECTORY")


if __name__ == "__main__":
    unittest.main()
