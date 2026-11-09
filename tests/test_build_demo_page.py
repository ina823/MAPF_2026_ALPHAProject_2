"""scripts/build_demo_page.py. Fixtures are written to a temporary directory, so
these tests do not need outputs/. The last class reads the saved Bottleneck batch
read-only and is skipped when that local batch is not present."""

import contextlib
import csv
import hashlib
import io
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import build_demo_page as bd
from src.common.spec import ACTION_DELTA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOALS = {0: (0, 2), 1: (2, 2)}
STARTS = {0: (0, 0), 1: (2, 0)}
DELTA_TO_ACTION = {delta: action for action, delta in ACTION_DELTA.items() if action != 4}
STEPS_FIELDS = ("run_id", "timestep", "agent_id", "current_row", "current_col", "next_row", "next_col")
EPISODE_FIELDS = ("run_id", "method", "scenario_sha256", "termination", "steps", "makespan", "sum_of_costs",
                  "overlap_any_step", "vertex_conflicts", "edge_conflicts", "cbs_call_count", "cbs_latency_ms_total",
                  "monitor_trigger_timestep", "monitor_first_seen", "monitor_period")
CALL_FIELDS = ("run_id", "call_index", "status", "accepted", "planned_horizon", "exec_window", "exec_steps", "exec_end")
MOVED = {0: (0, 1), 1: (2, 1)}
H3_DEV = {"h_exec": 3, "resume": "always IL", "recall": "first Trigger only; latches", "status": "development setting"}
REARM_DEV = {"h_exec": 3, "resume": "always IL", "recall": "limits",
             "rearm_limits": {"max_cbs_calls": 2, "cooldown_steps": 2, "cbs_time_budget_ms": 60000.0},
             "status": "development setting"}
REARM_PATHS_1 = {"0": [[0, 1], [0, 2]], "1": [[2, 1], [2, 2]]}
H3_PATHS_1 = {"0": [[0, 1], [0, 2], [0, 2]], "1": [[2, 1], [2, 1], [2, 2]]}


def state(s):
    return {str(a): list(p) for a, p in s.items()}


class Fixture:
    """A small, consistent batch: 3x3 open map, two agents, the three methods."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.batch = self.root / "batch"
        (self.batch / "runs").mkdir(parents=True)
        scen = self.root / "scen"
        scen.mkdir()
        self.grid = np.zeros((3, 3), dtype=np.int64)
        np.save(scen / "map_t.npy", self.grid)
        self.scenario = scen / "scenario_t.json"
        self.scenario.write_text(json.dumps({
            "scenario_id": "t", "map_id": "t", "map_file": "map_t.npy",
            "agents": [{"agent_id": a, "start": [STARTS[a][1], STARTS[a][0]], "goal": [GOALS[a][1], GOALS[a][0]]}
                       for a in STARTS]}), encoding="utf-8")
        from src.compare.common_runner import _sha256_grid
        self.grid_sha = _sha256_grid(self.grid)
        self.scenario_sha = hashlib.sha256(self.scenario.read_bytes()).hexdigest()
        self.episodes, self.call_rows, self.results = [], [], {}
        self.add_il_only()
        self.add_h3()
        self.add_rearm()
        self.write_tables()

    def add_il_only(self):
        traj = [STARTS, MOVED, MOVED, MOVED, MOVED]
        monitor = {"trigger_type": "FULL", "trigger_reason_code": "TRIGGER_FULL_REPEAT", "trigger_timestep": 2,
                   "first_seen": 1, "period": 1, "overlap_integrity_status": "CLEAN"}
        self.add_run("IL_ONLY", traj, ["IL"] * 4, "TIMEOUT", monitor=monitor, dev={}, vertex=6)

    def add_h3(self):
        at3 = {0: (0, 2), 1: (2, 1)}
        traj = [STARTS, MOVED, MOVED, at3, at3, at3]
        monitor = {"trigger_type": "FULL", "trigger_reason_code": "TRIGGER_FULL_REPEAT", "trigger_timestep": 2,
                   "first_seen": 1, "period": 1, "overlap_integrity_status": "CLEAN"}
        self.add_run("GLOBAL_HYBRID_DEV_H3", traj, ["IL", "IL", "CBS", "IL", "IL"], "TIMEOUT", monitor=monitor,
                     dev=H3_DEV, vertex=3,
                     transitions=[(2, "IL", "TRIGGER", "TRIGGER_FULL_REPEAT"), (2, "TRIGGER", "CBS_REQUEST", ""),
                                  (2, "CBS_REQUEST", "CBS_EXEC", ""), (3, "CBS_EXEC", "IL_RESUME", "EXEC_WINDOW_END"),
                                  (3, "IL_RESUME", "IL", "")],
                     calls=[(2, MOVED, H3_PATHS_1, 200.5, 2, 1)])

    def add_rearm(self):
        at3 = {0: (0, 2), 1: (2, 1)}
        traj = [STARTS, MOVED, MOVED, at3, at3, GOALS]
        monitor = {"trigger_type": "NONE", "trigger_reason_code": "TRIGGER_NONE", "trigger_timestep": None,
                   "first_seen": None, "period": None, "overlap_integrity_status": "CLEAN", "rearm_count": 2,
                   "trigger_history": [
                       {"trigger_type": "FULL", "trigger_reason_code": "TRIGGER_FULL_REPEAT", "trigger_timestep": 2,
                        "first_seen": 1, "period": 1, "blocking_type": "VERTEX", "blocking_agents": [0, 1]},
                       {"trigger_type": "FULL", "trigger_reason_code": "TRIGGER_FULL_REPEAT", "trigger_timestep": 4,
                        "first_seen": 3, "period": 1, "blocking_type": "VERTEX", "blocking_agents": [0, 1]}]}
        self.add_run("GLOBAL_HYBRID_DEV_H3_REARM", traj, ["IL", "IL", "CBS", "IL", "CBS"], "SUCCESS", monitor=monitor,
                     dev=REARM_DEV, vertex=2,
                     transitions=[(2, "IL", "TRIGGER", "TRIGGER_FULL_REPEAT"), (2, "TRIGGER", "CBS_REQUEST", ""),
                                  (2, "CBS_REQUEST", "CBS_EXEC", ""), (3, "CBS_EXEC", "IL_RESUME", "EXEC_WINDOW_END"),
                                  (3, "IL_RESUME", "IL", ""), (4, "IL", "TRIGGER", "TRIGGER_FULL_REPEAT"),
                                  (4, "TRIGGER", "CBS_REQUEST", ""), (4, "CBS_REQUEST", "CBS_EXEC", "")],
                     calls=[(2, MOVED, REARM_PATHS_1, 300.25, 1, 1),
                            (4, at3, {"0": [[0, 2], [0, 2]], "1": [[2, 1], [2, 2]]}, 40.75, 1, 1)])

    def add_run(self, method, traj, sources, termination, *, monitor, dev, vertex, transitions=(), calls=()):
        run_id = f"20260101_000000_t_{method}"
        run_dir = self.batch / "runs" / run_id
        run_dir.mkdir()
        steps = len(traj) - 1
        actions = []
        for t in range(steps):
            step = {}
            for a in traj[t]:
                delta = (traj[t + 1][a][0] - traj[t][a][0], traj[t + 1][a][1] - traj[t][a][1])
                step[str(a)] = 4 if delta == (0, 0) and traj[t][a] == GOALS[a] else DELTA_TO_ACTION.get(delta, 0)
            actions.append(step)
        cbs_calls, planner_calls, trace = [], [], []
        call_list = list(calls)
        for i, (trig, s_t, paths, latency, horizon, window) in enumerate(call_list):
            cbs_calls.append({"trigger_timestep": trig, "agent_order": [0, 1], "s_t": state(s_t), "status": "success",
                              "accepted": True, "reason": "", "error": None, "latency_ms": latency})
            planner_calls.append({"solver_root": "x", "work_dir": "x", "timeout_sec": 30, "status": "success", "paths": paths})
            for k in range(1, window + 1):
                trace.append({"t": trig + k, "k": k, "expected": state(traj[trig + k]), "actual": state(traj[trig + k]),
                              "match": True, "overlap": False})
            self.call_rows.append({"run_id": run_id, "call_index": i + 1, "status": "success", "accepted": "True",
                                   "planned_horizon": horizon, "exec_window": window, "exec_steps": window,
                                   "exec_end": "WINDOW_END"})
        all_at_goal = traj[-1] == GOALS
        result = {
            "method": method, "dev_settings": dev, "run_id": run_id, "run_dir": str(run_dir),
            "scenario": str(self.scenario), "scenario_sha256": self.scenario_sha, "grid_sha256": self.grid_sha,
            "max_steps": max(steps, 5), "agent_ids": [0, 1], "starts": state(STARTS), "goals": state(GOALS),
            "termination": termination, "stop_reason": None, "all_at_goal": all_at_goal, "timed_out": not all_at_goal,
            "steps": steps, "trajectory": [state(s) for s in traj], "actions": actions, "action_sources": sources,
            "overlap_any_step": False, "cbs_calls": cbs_calls, "planner_calls": planner_calls, "cbs_exec_trace": trace,
            "transitions": [{"t": t, "from": a, "to": b, "reason": r} for t, a, b, r in transitions],
            "safety_holds": [], "monitor": monitor,
            "environment": {"project_git": {"commit": "abcdef0123", "dirty": True},
                            "solver_git": {"commit": "89f69e1", "dirty": False} if calls else None},
        }
        (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
        self.results[method] = run_dir / "result.json"
        with (run_dir / f"{run_id}_steps.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, STEPS_FIELDS)
            w.writeheader()
            for t in range(steps):
                for a in traj[t]:
                    w.writerow({"run_id": run_id, "timestep": t + 1, "agent_id": a,
                                "current_row": traj[t][a][0], "current_col": traj[t][a][1],
                                "next_row": traj[t + 1][a][0], "next_col": traj[t + 1][a][1]})
        soc = 3 + 5 if all_at_goal else ""
        latest = monitor.get("trigger_timestep")
        self.episodes.append({
            "run_id": run_id, "method": method, "scenario_sha256": self.scenario_sha, "termination": termination,
            "steps": steps, "makespan": steps if all_at_goal else "", "sum_of_costs": soc, "overlap_any_step": "False",
            "vertex_conflicts": vertex, "edge_conflicts": 0, "cbs_call_count": len(call_list) if call_list else "",
            "cbs_latency_ms_total": sum(c[3] for c in call_list) if call_list else "",
            "monitor_trigger_timestep": "" if latest is None else latest,
            "monitor_first_seen": monitor.get("first_seen") or "", "monitor_period": monitor.get("period") or ""})

    def write_tables(self):
        for name, fields, rows in (("compare_episodes.csv", EPISODE_FIELDS, self.episodes),
                                   ("compare_cbs_calls.csv", CALL_FIELDS, self.call_rows)):
            with (self.batch / name).open("w", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fields)
                w.writeheader()
                w.writerows(rows)

    def edit_result(self, method, fn):
        path = self.results[method]
        doc = json.loads(path.read_text(encoding="utf-8"))
        fn(doc)
        path.write_text(json.dumps(doc), encoding="utf-8")

    def edit_episode(self, method, **fields):
        for row in self.episodes:
            if row["method"] == method:
                row.update(fields)
        self.write_tables()


class _Case(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.fx = Fixture(self.tmp / "f")

    def data(self):
        return bd.build_panel_data(self.fx.batch, "t")

    def run_of(self, method):
        return next(r for r in self.data()["runs"] if r["method"] == method)

    def assertRefused(self, fragment=""):
        with self.assertRaises(bd.BuildError) as ctx:
            self.data()
        self.assertIn(fragment, str(ctx.exception))


class TestDerivedValues(unittest.TestCase):
    def test_segments_merge_steps(self):
        self.assertEqual(bd.segments(["IL", "IL", "CBS", "CBS", "CBS", "IL"]),
                         [{"from": 1, "to": 2, "source": "IL"}, {"from": 3, "to": 5, "source": "CBS"},
                          {"from": 6, "to": 6, "source": "IL"}])

    def test_stalls_need_no_move_and_not_all_at_goal(self):
        kinds = [["MOVE", "MOVE"], ["BLOCKED", "BLOCKED"], ["BLOCKED", "WAIT"], ["MOVE", "WAIT"], ["WAIT", "WAIT"]]
        at_goal = [[False, False]] * 4 + [[True, False], [True, True]]
        self.assertEqual(bd.stall_segments(kinds, at_goal), [{"from": 2, "to": 3}])

    def test_monitor_history_is_preferred_and_latched_is_the_fallback(self):
        history = [{"trigger_timestep": 2, "first_seen": 1, "period": 1}]
        self.assertEqual(bd.monitor_records({"trigger_timestep": None, "trigger_history": history}), history)
        latched = bd.monitor_records({"trigger_timestep": 3, "first_seen": 2, "period": 1})
        self.assertEqual((latched[0]["trigger_timestep"], latched[0]["first_seen"]), (3, 2))
        self.assertEqual(bd.monitor_records({"trigger_timestep": None}), [])

    def test_recheck_passes_fails_or_reports_no_saved_path(self):
        grid = np.zeros((3, 3), dtype=np.int64)
        s_t = state(MOVED)
        self.assertEqual(bd.recheck_path_guard(grid, s_t, GOALS, {"paths": REARM_PATHS_1})["status"], bd.PG_PASS)
        bad = bd.recheck_path_guard(grid, s_t, GOALS, {"paths": {"0": [[0, 0], [0, 2]], "1": [[2, 1], [2, 2]]}})
        self.assertEqual(bad["status"], bd.PG_FAIL)
        self.assertTrue(bad["reason"])
        self.assertEqual(bd.recheck_path_guard(grid, s_t, GOALS, {"status": "timeout", "paths": None})["status"],
                         bd.PG_NOT_RECORDED)
        self.assertEqual(bd.recheck_path_guard(grid, s_t, GOALS, None)["status"], bd.PG_NOT_RECORDED)

    def test_dev_lines_come_only_from_the_record(self):
        self.assertEqual(bd.dev_lines({}), [])
        self.assertEqual(bd.dev_lines(REARM_DEV), ["H_exec=3", "cooldown=2", "CBS 최대 2회"])
        self.assertEqual(bd.dev_lines(H3_DEV), ["H_exec=3", "CBS는 첫 Trigger에서만 호출"])
        self.assertEqual(bd.dev_lines({"resume": "x"}), [])


class TestPanelData(_Case):
    def test_three_runs_in_order(self):
        self.assertEqual([r["method"] for r in self.data()["runs"]], list(bd.METHODS))

    def test_real_soc_uses_settling_times_not_padded_lengths(self):
        rearm = self.run_of("GLOBAL_HYBRID_DEV_H3_REARM")
        self.assertEqual(rearm["soc"], 8)
        self.assertEqual(rearm["makespan"], 5)
        self.assertNotEqual(rearm["soc"], rearm["steps"] * 2)

    def test_timeout_has_no_soc_or_makespan(self):
        h3 = self.run_of("GLOBAL_HYBRID_DEV_H3")
        self.assertEqual((h3["termination"], h3["soc"], h3["makespan"]), ("TIMEOUT", None, None))

    def test_overlap_and_vertex_conflicts_are_separate(self):
        il = self.run_of("IL_ONLY")
        self.assertEqual((il["overlap_steps"], il["vertex_conflicts"], il["edge_conflicts"]), ([], 6, 0))

    def test_call_keeps_accepted_and_recheck_apart(self):
        c = self.run_of("GLOBAL_HYBRID_DEV_H3_REARM")["calls"][0]
        self.assertIs(c["accepted"], True)
        self.assertEqual(c["recheck"]["status"], bd.PG_PASS)
        c = self.run_of("GLOBAL_HYBRID_DEV_H3_REARM")["calls"][1]
        self.assertEqual((c["trigger"], c["latency_ms"], c["first_seen"], c["period"]), (4, 40.75, 3, 1))

    def test_accepted_true_with_a_bad_saved_path_shows_both_facts(self):
        def corrupt(doc):
            doc["planner_calls"][0]["paths"]["0"] = [[0, 0], [0, 2]]
        self.fx.edit_result("GLOBAL_HYBRID_DEV_H3_REARM", corrupt)
        c = self.run_of("GLOBAL_HYBRID_DEV_H3_REARM")["calls"][0]
        self.assertIs(c["accepted"], True)
        self.assertEqual(c["recheck"]["status"], bd.PG_FAIL)

    def test_missing_latency_and_monitor_values_stay_none(self):
        def strip(doc):
            del doc["cbs_calls"][0]["latency_ms"]
            doc["monitor"]["first_seen"] = None
            doc["monitor"]["period"] = None
        self.fx.edit_result("GLOBAL_HYBRID_DEV_H3", strip)
        self.fx.edit_episode("GLOBAL_HYBRID_DEV_H3", cbs_latency_ms_total="", monitor_first_seen="", monitor_period="")
        c = self.run_of("GLOBAL_HYBRID_DEV_H3")["calls"][0]
        self.assertEqual((c["latency_ms"], c["first_seen"], c["period"]), (None, None, None))
        self.assertIsNone(self.run_of("GLOBAL_HYBRID_DEV_H3")["cbs_latency_total_ms"])

    def test_events_timeline_and_exec_steps(self):
        rearm = self.run_of("GLOBAL_HYBRID_DEV_H3_REARM")
        self.assertEqual([(e["t"], e["kind"]) for e in rearm["events"]],
                         [(2, "TRIGGER"), (2, "CBS"), (2, "GUARD"), (3, "RESUME"),
                          (4, "TRIGGER"), (4, "CBS"), (4, "GUARD"), (5, "END")])
        self.assertEqual(rearm["exec_step"], {"3": [1, 1], "5": [2, 1]})
        il = self.run_of("IL_ONLY")
        self.assertEqual([e["kind"] for e in il["events"]], ["MONITOR", "END"])
        self.assertEqual(il["stalls"], [{"from": 2, "to": 4}])

    def test_guard_event_reports_only_the_accepted_record(self):
        guard = next(e for e in self.run_of("GLOBAL_HYBRID_DEV_H3")["events"] if e["kind"] == "GUARD")
        self.assertIn("통과", guard["label"])
        self.assertEqual(guard["detail"], "accepted=True")

    def test_comparison_records_different_first_path_same_state_and_no_seed(self):
        self.assertEqual(self.data()["comparison"],
                         {"same_first_s_t": True, "same_first_cbs_path": False, "hash_seed_recorded": False})

    def test_comparison_records_a_recorded_seed(self):
        for m in ("GLOBAL_HYBRID_DEV_H3", "GLOBAL_HYBRID_DEV_H3_REARM"):
            def add(doc):
                for p in doc["planner_calls"]:
                    p["pythonhashseed_env"] = "0"
            self.fx.edit_result(m, add)
        self.assertTrue(self.data()["comparison"]["hash_seed_recorded"])


class TestCrossChecksRefuse(_Case):
    def test_csv_termination(self):
        self.fx.edit_episode("IL_ONLY", termination="SUCCESS")
        self.assertRefused("termination")

    def test_csv_soc(self):
        self.fx.edit_episode("GLOBAL_HYBRID_DEV_H3_REARM", sum_of_costs=10)
        self.assertRefused("sum_of_costs")

    def test_csv_overlap(self):
        self.fx.edit_episode("IL_ONLY", overlap_any_step="True")
        self.assertRefused("overlap")

    def test_csv_latency_total(self):
        self.fx.edit_episode("GLOBAL_HYBRID_DEV_H3", cbs_latency_ms_total=1.0)
        self.assertRefused("latency")

    def test_csv_call_count(self):
        self.fx.edit_episode("GLOBAL_HYBRID_DEV_H3", cbs_call_count=2)
        self.assertRefused("call count")

    def test_csv_monitor_trigger(self):
        self.fx.edit_episode("IL_ONLY", monitor_trigger_timestep=3)
        self.assertRefused("Monitor")

    def test_steps_csv_position(self):
        path = next((self.fx.batch / "runs" / "20260101_000000_t_IL_ONLY").glob("*_steps.csv"))
        text = path.read_text(encoding="utf-8").splitlines()
        cols = text[1].split(",")
        cols[STEPS_FIELDS.index("next_col")] = "2"
        text[1] = ",".join(cols)
        path.write_text("\n".join(text) + "\n", encoding="utf-8")
        self.assertRefused("steps CSV")

    def test_overlap_flag_against_trajectory(self):
        self.fx.edit_result("IL_ONLY", lambda d: d.update(overlap_any_step=True))
        self.assertRefused("overlap_any_step")

    def test_scenario_file_changed(self):
        self.fx.scenario.write_text(self.fx.scenario.read_text(encoding="utf-8") + " ", encoding="utf-8")
        self.assertRefused("scenario")

    def test_trajectory_failing_the_render_checks(self):
        def jump(doc):
            doc["trajectory"][1]["0"] = [2, 2]
        self.fx.edit_result("IL_ONLY", jump)
        self.assertRefused()

    def test_missing_or_duplicate_run(self):
        shutil.rmtree(self.fx.batch / "runs" / "20260101_000000_t_IL_ONLY")
        self.assertRefused("found 0")
        extra = self.fx.batch / "runs" / "20260101_000001_t_GLOBAL_HYBRID_DEV_H3"
        shutil.copytree(self.fx.batch / "runs" / "20260101_000000_t_GLOBAL_HYBRID_DEV_H3", extra)
        self.assertRefused()

    def test_missing_tables(self):
        (self.fx.batch / "compare_cbs_calls.csv").unlink()
        self.assertRefused("compare_cbs_calls.csv")


class TestOutput(_Case):
    def build(self):
        return bd.build(self.fx.batch, self.tmp / "out", scenario_key="t")

    def embedded(self, page):
        text = page.read_text(encoding="utf-8")
        return text, json.loads(re.search(r'id="panel-data">(.*?)</script>', text, re.S).group(1))

    def test_writes_page_and_manifest_only(self):
        page = self.build()
        self.assertEqual(sorted(p.name for p in page.parent.iterdir()), ["demo_manifest.json", "demo_panel.html"])
        manifest = json.loads((page.parent / "demo_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["output_sha256"]["demo_panel.html"],
                         hashlib.sha256(page.read_bytes()).hexdigest())
        self.assertEqual([r["real_soc"] for r in manifest["runs"]], [None, None, 8])
        self.assertEqual(len(manifest["inputs_sha256"]), 9)

    def test_existing_output_is_refused_and_untouched(self):
        out = self.tmp / "out"
        out.mkdir()
        (out / "keep.txt").write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            bd.build(self.fx.batch, out, scenario_key="t")
        self.assertEqual([p.name for p in out.iterdir()], ["keep.txt"])

    def test_inconsistent_input_writes_nothing(self):
        self.fx.edit_episode("IL_ONLY", termination="SUCCESS")
        with self.assertRaises(bd.BuildError):
            self.build()
        self.assertFalse((self.tmp / "out").exists())

    def test_inputs_are_not_changed(self):
        before = {p: p.read_bytes() for p in self.fx.batch.rglob("*") if p.is_file()}
        self.build()
        after = {p: p.read_bytes() for p in self.fx.batch.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_page_is_self_contained_and_escapes_script_end(self):
        def inject(doc):
            doc["transitions"].append({"t": 3, "from": "IL", "to": "IL_RESUME", "reason": "</script><b>x"})
        self.fx.edit_result("GLOBAL_HYBRID_DEV_H3", inject)
        text, data = self.embedded(self.build())
        self.assertNotRegex(text, r'(?:src|href)="https?:')
        self.assertNotIn("@import", text)
        self.assertEqual(text.count("</script><b>x"), 0)
        self.assertIn("</script><b>x", json.dumps(data, ensure_ascii=False))

    def test_page_states_evidence_rules(self):
        text, _ = self.embedded(self.build())
        for needle in ("기록 없음", "Path Guard 단독 결과 (실행 당시)", "Path Guard 사후 재검증", "수용 기록 (실행 당시)",
                       "통제된 인과 비교", "합의된 운영 정책이 아닙니다", "이동 시도 충돌", "실제 위치 겹침", "실제 SoC",
                       "Monitor 기록 아님", "해당 없음 (미완료)"):
            self.assertIn(needle, text, needle)

    def test_no_absolute_path_in_the_page_or_manifest(self):
        page = self.build()
        manifest = (page.parent / "demo_manifest.json").read_text(encoding="utf-8")
        for text in (page.read_text(encoding="utf-8"), manifest):
            for needle in (str(self.tmp), self.tmp.as_posix(), "C:/Users"):
                self.assertNotIn(needle, text)

    def test_evidence_lists_every_input_with_its_hash(self):
        data = self.data()
        self.assertEqual(sorted(f["sha256"] for f in data["evidence"]["files"]), sorted(data["inputs_sha256"].values()))
        _, embedded = self.embedded(self.build())
        self.assertNotIn("inputs_sha256", embedded)

    def test_presentation_layout_order_and_closed_details(self):
        text, _ = self.embedded(self.build())
        order = [text.index(k) for k in ('id="cards"', 'id="board"', 'id="tripleView"', 'id="play"', 'id="slider"',
                                         'id="detailsBox"', 'id="evidenceBox"', 'id="footnotes"')]
        self.assertEqual(order, sorted(order))
        self.assertNotIn("이 화면을 읽을 때", text)
        self.assertIn('<details class="fold" id="detailsBox">', text)
        self.assertIn('<details class="fold" id="evidenceBox">', text)
        self.assertIn("저장된 Development 사례이며 최종 운영 정책이 아닙니다.", text)
        self.assertIn("H3와 H3_REARM은 첫 CBS 경로가 달라 엄밀한 성능 비교가 아닙니다.", text)
        for needle in ("완료 step", "실제 SoC", "목표 도달", "CBS 호출", "실제 충돌", "이동 시도 충돌", "미완료", "vertex_conflicts"):
            self.assertIn(needle, text, needle)

    def test_soc_label_and_definition_use_the_settling_time(self):
        text, _ = self.embedded(self.build())
        self.assertIn("실제 SoC (총 도착 시간)", text)
        self.assertIn("목표에 도착해 끝까지 머물기 시작한 시점의 합", text)

    def test_real_collision_and_move_conflict_stay_separate(self):
        text, _ = self.embedded(self.build())
        self.assertIn("실제 충돌 (같은 칸)", text)
        self.assertIn("이동 시도 충돌 (이동 취소)", text)
        self.assertIn("실제 충돌이 0이어도 발생할 수 있습니다", text)
        self.assertNotIn("막힌 이동 시도", text)

    def test_stall_wording_names_the_cancelled_contest_not_physical_blocking(self):
        text, _ = self.embedded(self.build())
        self.assertIn("같은 칸 경합으로 전원 이동 취소", text)
        for wrong in ("서로 막혀", "서로 막히면", "막히면 CBS", "다시 막히면"):
            self.assertNotIn(wrong, text, wrong)

    def test_hidden_views_stay_hidden_and_the_page_does_not_overflow(self):
        text, _ = self.embedded(self.build())
        self.assertIn("[hidden]{display:none!important}", text)
        self.assertIn("grid-template-columns:minmax(0,1fr);gap:12px}", text)

    def test_timeline_uses_merged_segments_not_per_step_nodes(self):
        run = self.run_of("IL_ONLY")
        self.assertEqual(run["segments"], [{"from": 1, "to": 4, "source": "IL"}])

    def test_cli_exit_codes(self):
        out = self.tmp / "cli"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(bd.main(["--batch-dir", str(self.fx.batch), "--scenario-key", "t", "--out-dir", str(out)]), 0)
            self.assertEqual(bd.main(["--batch-dir", str(self.fx.batch), "--scenario-key", "t", "--out-dir", str(out)]), 2)
            self.fx.edit_episode("IL_ONLY", steps=1)
            self.assertEqual(bd.main(["--batch-dir", str(self.fx.batch), "--scenario-key", "t",
                                      "--out-dir", str(self.tmp / "other")]), 1)
        self.assertFalse((self.tmp / "other").exists())

    def test_script_cannot_start_a_solver_or_policy(self):
        source = (PROJECT_ROOT / "scripts" / "build_demo_page.py").read_text(encoding="utf-8")
        for word in ("CbsRuntimePlanner", "run_method", "run_cbs_runtime", "NavHintPolicy", "subprocess", "run_atb033"):
            self.assertNotIn(word, source, word)


@unittest.skipUnless(bd.DEFAULT_BATCH.is_dir(), "saved Bottleneck batch not present (outputs/ is not committed)")
class TestSavedBottleneckBatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = bd.build_panel_data(bd.DEFAULT_BATCH)

    def run_of(self, method):
        return next(r for r in self.data["runs"] if r["method"] == method)

    def test_outcomes(self):
        got = [(r["termination"], r["steps"], r["soc"], len(r["overlap_steps"]), r["vertex_conflicts"], len(r["calls"]))
               for r in self.data["runs"]]
        self.assertEqual(got, [("TIMEOUT", 160, None, 0, 318, 0), ("TIMEOUT", 160, None, 0, 310, 1),
                               ("SUCCESS", 16, 52, 0, 4, 2)])

    def test_rearm_flow_and_evidence(self):
        rearm = self.run_of("GLOBAL_HYBRID_DEV_H3_REARM")
        self.assertEqual([(e["t"], e["kind"]) for e in rearm["events"]],
                         [(2, "TRIGGER"), (2, "CBS"), (2, "GUARD"), (5, "RESUME"),
                          (7, "TRIGGER"), (7, "CBS"), (7, "GUARD"), (10, "RESUME"), (16, "END")])
        self.assertEqual(rearm["dev_lines"], ["H_exec=3", "cooldown=2", "CBS 최대 2회"])
        self.assertEqual([(c["first_seen"], c["period"], c["recheck"]["status"], c["accepted"]) for c in rearm["calls"]],
                         [(1, 1, bd.PG_PASS, True), (6, 1, bd.PG_PASS, True)])

    def _steps_by_time(self, method):
        rows = bd.read_csv(next(bd.find_run_dir(bd.DEFAULT_BATCH, bd.DEFAULT_SCENARIO_KEY, method).glob("*_steps.csv")))
        by_time = {}
        for row in rows:
            by_time.setdefault(int(row["timestep"]), []).append(row)
        return by_time

    def test_every_stalled_step_is_an_all_agent_vertex_cancellation(self):
        for run in self.data["runs"]:
            by_time = self._steps_by_time(run["method"])
            for stall in run["stalls"]:
                for t in range(stall["from"], stall["to"] + 1):
                    self.assertEqual({r["conflict_type"] for r in by_time[t]}, {"vertex"}, (run["method"], t))

    def test_il_only_conflicts_are_two_contested_cells_for_159_steps(self):
        contested = {}
        for t, rows in self._steps_by_time("IL_ONLY").items():
            cells = {}
            for r in rows:
                cells.setdefault((int(r["intended_row"]), int(r["intended_col"])), []).append(int(r["agent_id"]))
            for cell, agents in cells.items():
                if len(agents) > 1:
                    contested[(cell, tuple(sorted(agents)))] = contested.get((cell, tuple(sorted(agents))), 0) + 1
        self.assertEqual(contested, {((2, 2), (0, 1)): 159, ((2, 7), (2, 3)): 159})

    def test_h3_restall_is_only_a_computed_stall_not_a_monitor_event(self):
        h3 = self.run_of("GLOBAL_HYBRID_DEV_H3")
        self.assertEqual(h3["stalls"], [{"from": 2, "to": 2}, {"from": 7, "to": 160}])
        self.assertEqual([e["t"] for e in h3["events"] if e["kind"] == "TRIGGER"], [2])

    def test_comparison_is_not_controlled(self):
        self.assertEqual(self.data["comparison"],
                         {"same_first_s_t": True, "same_first_cbs_path": False, "hash_seed_recorded": False})


if __name__ == "__main__":
    unittest.main()
