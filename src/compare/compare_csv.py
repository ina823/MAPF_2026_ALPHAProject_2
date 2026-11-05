"""Comparison CSVs for EpisodeRecords from src/compare/common_runner.run_method().

    compare_episodes.csv   -- one row per run
    compare_cbs_calls.csv  -- one row per CBS call (CBS methods only)

Rows are built from the EpisodeRecord and checked against the run's own files
(steps / summary / monitor CSV) before anything is written; anything that
does not add up raises CompareCsvError instead of being guessed. Reused
unchanged from src/hybrid/hybrid_csv.py: format_value, attribute_trace and
cross_validate_steps.

When CBS is asked (cbs_request_kind):
  * INITIAL_STATE     -- Pure CBS: once, at t=0, on S_0; whole plan executed;
  * IL_STALL_TRIGGER  -- Global Hybrid: after an IL stall Trigger at t >= 1.

Field rules:
  * empty cell = not applicable / not recorded, never a stand-in for 0;
  * success = termination == SUCCESS (must equal all_at_goal);
  * makespan / sum_of_costs only for SUCCESS. sum_of_costs adds, per agent,
    the first timestep from which the agent stays on its goal until the end;
  * solver_latency_ms keeps the recorded definition (replan_with_status
    perf_counter span: YAML I/O, cbs.py subprocess incl. interpreter start-up,
    parsing, replanner Path Guard) -- not pure CBS search time.
    planner_wall_ms is only recorded by the Pure CBS runner;
  * planned_horizon = len(longest accepted CBS path) - 1, from the paths the
    planner returned; empty when the call was not accepted or no paths were
    recorded. exec_window = steps the method intends to execute from that
    plan (Pure CBS: planned_horizon; DEV_FULL_PATH: min(max_steps,
    planned_horizon); DEV_FIXED_H_EXEC: min(h_exec, planned_horizon)).
    exec_steps = steps actually executed;
  * cbs_controlled_agent_steps = steps-CSV rows at CBS-controlled timesteps.

All method settings except IL_ONLY are DEV settings, not agreed policies.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from src.compare.common_runner import (
    CBS_METHODS,
    GLOBAL_HYBRID_DEV_FULL,
    GLOBAL_HYBRID_DEV_H3,
    GLOBAL_HYBRID_DEV_H3_REARM,
    H_EXEC_DEV,
    IL_ONLY,
    PURE_CBS_DEV_SINGLE_PLAN,
    SAFETY_ONLY_DEV_K2,
)
from src.hybrid.hybrid_csv import HybridCsvError, attribute_trace, cross_validate_steps, format_value

EPISODE_FIELDS: Sequence[str] = (
    "batch_id", "run_id", "method", "is_dev_setting", "dev_settings",
    "scenario", "scenario_sha256", "grid_sha256", "num_agents",
    "checkpoint_sha256", "max_steps", "cbs_timeout_sec",
    "termination", "termination_source", "stop_reason", "success", "all_at_goal",
    "timed_out", "agents_at_goal_final", "steps", "makespan", "sum_of_costs",
    "overlap_any_step", "first_overlap_t", "overlap_checked_at_runtime", "integrity_unhandled",
    "vertex_conflicts", "edge_conflicts", "wait_count",
    "monitor_trigger_timestep", "monitor_first_seen", "monitor_period",
    "cbs_request_kind", "cbs_call_count", "cbs_latency_ms_total", "planner_wall_ms_total",
    "cbs_exec_steps", "cbs_controlled_agent_steps",
    "il_steps", "safety_hold_agent_steps", "il_resume_t",
    "project_commit", "project_dirty", "solver_root", "solver_commit", "solver_dirty",
    "python", "torch", "numpy", "pyyaml", "platform", "run_dir",
)

CALL_FIELDS: Sequence[str] = (
    "batch_id", "run_id", "method", "call_index",
    "cbs_request_kind", "request_timestep",
    "status", "accepted", "reason", "error",
    "solver_latency_ms", "planner_wall_ms",
    "exec_policy", "h_exec_config",
    "planned_horizon", "exec_window", "exec_steps",
    "controlled_agent_steps", "agents_at_goal_at_exec_end", "exec_end",
)

EPISODES_CSV = "compare_episodes.csv"
CALLS_CSV = "compare_cbs_calls.csv"

REQUEST_KIND = {
    PURE_CBS_DEV_SINGLE_PLAN: "INITIAL_STATE",
    GLOBAL_HYBRID_DEV_FULL: "IL_STALL_TRIGGER",
    GLOBAL_HYBRID_DEV_H3: "IL_STALL_TRIGGER",
    GLOBAL_HYBRID_DEV_H3_REARM: "IL_STALL_TRIGGER",
}
EXEC_POLICY = {
    PURE_CBS_DEV_SINGLE_PLAN: "DEV_SINGLE_PLAN_FULL",
    GLOBAL_HYBRID_DEV_FULL: "DEV_FULL_PATH",
    GLOBAL_HYBRID_DEV_H3: "DEV_FIXED_H_EXEC",
    GLOBAL_HYBRID_DEV_H3_REARM: "DEV_FIXED_H_EXEC",
}
FIXED_H_EXEC_METHODS = (GLOBAL_HYBRID_DEV_H3, GLOBAL_HYBRID_DEV_H3_REARM)


class CompareCsvError(HybridCsvError):
    """A run cannot be turned into comparison rows without guessing."""


def _fail(run_id, message):
    raise CompareCsvError(f"{run_id}: {message}")


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def sum_of_costs(trajectory, goals):
    """Per agent: first t from which it stays on its goal to the end; summed."""
    total = 0
    for aid, goal in goals.items():
        if trajectory[-1][aid] != goal:
            return None
        t = len(trajectory) - 1
        while t > 0 and trajectory[t - 1][aid] == goal:
            t -= 1
        total += t
    return total


def _request_timestep(method, call):
    return call["request_timestep"] if method == PURE_CBS_DEV_SINGLE_PLAN else call["trigger_timestep"]


def _exec_end(run_id, record, call, entries, horizon):
    if not call["accepted"]:
        return "REFUSED"
    if not entries:
        if record.method == PURE_CBS_DEV_SINGLE_PLAN and horizon == 0:
            return "EMPTY_PLAN"
        if any(tr["from"] == "CBS_REQUEST" and tr["to"] == "IL_RESUME" and tr["reason"] == "EMPTY_PLAN"
               for tr in record.transitions):
            return "EMPTY_PLAN"
        _fail(run_id, f"call {call['index']} accepted but executed no CBS step")
    last = entries[-1]
    if last["overlap"]:
        return "INTEGRITY_ABORT"
    if not last["match"]:
        return "EXEC_MISMATCH"
    if any(tr["t"] == last["t"] and tr["from"] == "CBS_EXEC" and tr["to"] == "IL_RESUME"
           for tr in record.transitions):
        return "WINDOW_END"
    if last["t"] == record.steps and record.termination == "SUCCESS":
        return "GOAL_REACHED"
    if last["t"] == record.steps and record.termination == "TIMEOUT":
        return "EPISODE_TIMEOUT"
    if last["t"] == record.steps and record.stop_reason == "PATH_END_NOT_AT_GOAL":
        return "PATH_END_NOT_AT_GOAL"
    _fail(run_id, f"call {call['index']}: cannot tell how CBS execution ended at t={last['t']}")


def _check_run_files(record):
    """steps / summary / monitor CSV of the run against the record. Returns
    (steps rows by timestep, summary row)."""
    run_id, run_dir = record.run_id, Path(record.run_dir)
    _, steps_rows = _read_csv(run_dir / f"{run_id}_steps.csv")
    if record.steps == 0:
        if steps_rows:
            _fail(run_id, "steps CSV has rows for a run without a simulator step")
        by_t = {}
    else:
        norm = {"goals": record.goals, "trajectory": record.trajectory, "steps": record.steps,
                "all_at_goal": record.all_at_goal, "termination": record.termination}
        try:
            by_t, _ = cross_validate_steps(run_id, {"mode": record.method}, norm, steps_rows)
        except HybridCsvError as exc:
            raise CompareCsvError(str(exc)) from exc
    _, summary = _read_csv(run_dir / f"{run_id}_summary.csv")
    if len(summary) != 1:
        _fail(run_id, "summary CSV must have exactly one row")
    summary = summary[0]
    if (summary["steps"], summary["success"]) != (str(record.steps), format_value(record.all_at_goal)):
        _fail(run_id, "summary CSV steps/success differ from the record")
    _, monitor = _read_csv(run_dir / f"{run_id}_monitor.csv")
    if len(monitor) != 1 or monitor[0]["trigger_timestep"] != format_value(record.monitor["trigger_timestep"]):
        _fail(run_id, "monitor CSV trigger_timestep differs from the record")

    sources = record.action_sources
    if len(sources) != record.steps:
        _fail(run_id, "action_sources length differs from steps")
    cbs_ts = [t for t in range(1, record.steps + 1) if sources[t - 1] == "CBS"]
    if sorted(e["t"] for e in record.cbs_exec_trace) != cbs_ts:
        _fail(run_id, "CBS trace does not cover exactly the CBS-controlled timesteps")
    for e in record.cbs_exec_trace:
        if e["actual"] != record.trajectory[e["t"]] or e["match"] != (e["expected"] == e["actual"]):
            _fail(run_id, f"CBS trace entry t={e['t']} inconsistent with the trajectory")
    if record.method in CBS_METHODS:
        work_dirs = sorted(p.name for p in (run_dir / "cbs").iterdir()) if (run_dir / "cbs").is_dir() else []
        if len(work_dirs) != len(record.cbs_calls) or len(record.planner_calls) != len(record.cbs_calls):
            _fail(run_id, "solver work dirs / planner calls / CBS calls do not match one to one")
    return by_t, summary


def build_rows(record, batch_id):
    """(episode_row, call_rows) for one EpisodeRecord."""
    run_id, method = record.run_id, record.method
    success = record.termination == "SUCCESS"
    if success != record.all_at_goal:
        _fail(run_id, f"termination {record.termination} inconsistent with all_at_goal {record.all_at_goal}")
    by_t, summary = _check_run_files(record)
    is_cbs = method in CBS_METHODS

    call_rows = []
    if is_cbs:
        calls = [{**c, "index": i, "trigger_timestep": _request_timestep(method, c)}
                 for i, c in enumerate(record.cbs_calls, start=1)]
        try:
            owned = attribute_trace(run_id, calls, record.cbs_exec_trace)
        except HybridCsvError as exc:
            raise CompareCsvError(str(exc)) from exc
        for call, pcall in zip(calls, record.planner_calls):
            entries = owned[call["index"]]
            paths = pcall["paths"] if call["accepted"] else None
            if paths:
                if any(paths[a][0] != record.trajectory[call["trigger_timestep"]][a] for a in paths):
                    _fail(run_id, f"call {call['index']}: accepted Path[0] differs from the request state")
                if entries and entries[0]["expected"] != {a: p[min(1, len(p) - 1)] for a, p in paths.items()}:
                    _fail(run_id, f"call {call['index']}: first CBS step is not Path[1]")
                horizon = max(len(p) for p in paths.values()) - 1
            else:
                horizon = None
            if horizon is None:
                window = None
            elif method == PURE_CBS_DEV_SINGLE_PLAN:
                window = horizon
            elif method == GLOBAL_HYBRID_DEV_FULL:
                window = min(record.max_steps, horizon)
            else:
                window = min(H_EXEC_DEV, horizon)
            if window is not None and len(entries) > window:
                _fail(run_id, f"call {call['index']}: executed {len(entries)} steps, window {window}")
            call_rows.append({
                "batch_id": batch_id,
                "run_id": run_id,
                "method": method,
                "call_index": call["index"],
                "cbs_request_kind": REQUEST_KIND[method],
                "request_timestep": call["trigger_timestep"],
                "status": call["status"],
                "accepted": call["accepted"],
                "reason": call["reason"],
                "error": call["error"],
                "solver_latency_ms": call["latency_ms"],
                "planner_wall_ms": call.get("planner_wall_ms"),
                "exec_policy": EXEC_POLICY[method],
                "h_exec_config": H_EXEC_DEV if method in FIXED_H_EXEC_METHODS else None,
                "planned_horizon": horizon,
                "exec_window": window,
                "exec_steps": len(entries),
                "controlled_agent_steps": sum(len(by_t[e["t"]]) for e in entries),
                "agents_at_goal_at_exec_end": (
                    sum(1 for a, g in record.goals.items() if record.trajectory[entries[-1]["t"]][a] == g)
                    if entries else None),
                "exec_end": _exec_end(run_id, record, call, entries, horizon),
            })

    latencies = [c["latency_ms"] for c in record.cbs_calls]
    walls = [c.get("planner_wall_ms") for c in record.cbs_calls]
    controlled = sum(r["controlled_agent_steps"] for r in call_rows)
    if is_cbs and controlled != len(record.cbs_exec_trace) * len(record.goals):
        _fail(run_id, "CBS-controlled agent-steps differ from CBS steps x agents")
    env = record.environment
    solver_git = env.get("solver_git") or {}
    solver_roots = {c["solver_root"] for c in record.planner_calls}
    if len(solver_roots) > 1:
        _fail(run_id, "more than one solver_root in one run")
    episode = {
        "batch_id": batch_id,
        "run_id": run_id,
        "method": method,
        "is_dev_setting": method != IL_ONLY,
        "dev_settings": json.dumps(record.dev_settings, sort_keys=True, ensure_ascii=False),
        "scenario": Path(record.scenario).name,
        "scenario_sha256": record.scenario_sha256,
        "grid_sha256": record.grid_sha256,
        "num_agents": len(record.goals),
        "checkpoint_sha256": record.checkpoint_sha256,
        "max_steps": record.max_steps,
        "cbs_timeout_sec": record.cbs_timeout_sec,
        "termination": record.termination,
        "termination_source": record.termination_source,
        "stop_reason": record.stop_reason,
        "success": success,
        "all_at_goal": record.all_at_goal,
        "timed_out": record.timed_out,
        "agents_at_goal_final": sum(1 for a, g in record.goals.items() if record.trajectory[-1][a] == g),
        "steps": record.steps,
        "makespan": record.steps if success else None,
        "sum_of_costs": sum_of_costs(record.trajectory, record.goals) if success else None,
        "overlap_any_step": record.overlap_any_step,
        "first_overlap_t": record.first_overlap_t,
        "overlap_checked_at_runtime": record.overlap_checked_at_runtime,
        "integrity_unhandled": record.integrity_unhandled,
        "vertex_conflicts": int(summary["vertex_conflicts"]),
        "edge_conflicts": int(summary["edge_conflicts"]),
        "wait_count": int(summary["wait_count"]),
        "monitor_trigger_timestep": record.monitor["trigger_timestep"],
        "monitor_first_seen": record.monitor["first_seen"],
        "monitor_period": record.monitor["period"],
        "cbs_request_kind": REQUEST_KIND.get(method),
        "cbs_call_count": len(record.cbs_calls) if is_cbs else None,
        "cbs_latency_ms_total": sum(latencies) if latencies and None not in latencies else None,
        "planner_wall_ms_total": sum(walls) if walls and None not in walls else None,
        "cbs_exec_steps": len(record.cbs_exec_trace) if is_cbs else None,
        "cbs_controlled_agent_steps": controlled if is_cbs else None,
        "il_steps": (sum(1 for s in record.action_sources if s in ("IL", "SAFETY"))
                     if method != PURE_CBS_DEV_SINGLE_PLAN else None),
        "safety_hold_agent_steps": len(record.safety_holds) if method == SAFETY_ONLY_DEV_K2 else None,
        "il_resume_t": next((tr["t"] for tr in record.transitions if tr["to"] == "IL_RESUME"), None),
        "project_commit": env["project_git"]["commit"],
        "project_dirty": env["project_git"]["dirty"],
        "solver_root": next(iter(solver_roots)) if solver_roots else None,
        "solver_commit": solver_git.get("commit"),
        "solver_dirty": solver_git.get("dirty"),
        "python": env["python"],
        "torch": env["torch"],
        "numpy": env["numpy"],
        "pyyaml": env["pyyaml"],
        "platform": env["platform"],
        "run_dir": record.run_dir,
    }
    return episode, call_rows


# -- Fairness -------------------------------------------------------------------

def manifest_checkpoint_sha256(ckpt_path):
    """sha256 recorded for this checkpoint in checkpoints/MANIFEST.json."""
    ckpt_path = Path(ckpt_path)
    manifest = json.loads((ckpt_path.parent / "MANIFEST.json").read_text(encoding="utf-8"))
    try:
        return manifest["checkpoints"][ckpt_path.name]["sha256"]
    except KeyError as exc:
        raise CompareCsvError(f"{ckpt_path.name} not listed in MANIFEST.json") from exc


def check_fairness(episodes, *, scenarios, methods, max_steps, cbs_timeout_sec, checkpoint_sha256):
    """Returns (fair, reasons). Every condition must hold across the batch."""
    reasons = []

    def same(key, rows, label):
        values = {r[key] for r in rows}
        if len(values) != 1:
            reasons.append(f"{label}: {key} differs ({sorted(map(str, values))})")
        return values

    expected = {(s, m) for s in scenarios for m in methods}
    got = [(r["scenario"], r["method"]) for r in episodes]
    if sorted(got) != sorted(expected) or len(got) != len(set(got)):
        reasons.append(f"runs are not exactly scenarios x methods ({len(got)} rows, {len(expected)} expected)")
    for s in scenarios:
        rows = [r for r in episodes if r["scenario"] == s]
        if rows:
            same("scenario_sha256", rows, s)
            same("grid_sha256", rows, s)
            same("num_agents", rows, s)
    il_rows = [r for r in episodes if r["method"] != PURE_CBS_DEV_SINGLE_PLAN]
    if il_rows and {r["checkpoint_sha256"] for r in il_rows} != {checkpoint_sha256}:
        reasons.append("checkpoint_sha256 differs from checkpoints/MANIFEST.json")
    if {r["max_steps"] for r in episodes} != {max_steps}:
        reasons.append(f"max_steps is not {max_steps} for every run")
    cbs_rows = [r for r in episodes if r["method"] in CBS_METHODS]
    if cbs_rows:
        if {r["cbs_timeout_sec"] for r in cbs_rows} != {cbs_timeout_sec}:
            reasons.append(f"cbs_timeout_sec is not {cbs_timeout_sec} for every CBS run")
        same("solver_root", cbs_rows, "CBS runs")
        same("solver_commit", cbs_rows, "CBS runs")
        if any(r["solver_dirty"] is not False for r in cbs_rows):
            reasons.append("solver repository dirty or unknown")
    for key in ("project_commit", "python", "torch", "numpy", "pyyaml", "platform"):
        same(key, episodes, "batch")
    if any(r["project_dirty"] is not False for r in episodes):
        reasons.append("project repository dirty or unknown")
    return not reasons, reasons


# -- Writing --------------------------------------------------------------------

def write_rows(path, fields, rows):
    with Path(path).open("x", newline="", encoding="utf-8") as fh:  # never overwrite
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: format_value(row[k]) for k in fields})


def verify_written(path, fields, rows):
    """Re-read a written CSV; every cell must equal the formatted row value."""
    header, read = _read_csv(path)
    if header != list(fields) or len(read) != len(rows):
        raise CompareCsvError(f"{path}: header or row count differs from what was written")
    checks = 0
    for got, row in zip(read, rows):
        for k in fields:
            if got[k] != format_value(row[k]):
                raise CompareCsvError(f"{path}: {row.get('run_id')} {k}={got[k]!r} != {format_value(row[k])!r}")
            checks += 1
    return checks


def check_against_records(episodes_path, calls_path, records):
    """Compare the written CSVs with each runner's own result (record.raw and
    its call records), not with the built rows. Returns the number of checks."""
    _, episodes = _read_csv(episodes_path)
    _, calls = _read_csv(calls_path)
    by_run = {r["run_id"]: r for r in episodes}
    calls_by_run = defaultdict(list)
    for r in calls:
        calls_by_run[r["run_id"]].append(r)
    checks = 0
    for rec in records:
        row = by_run.get(rec.run_id)
        if row is None:
            _fail(rec.run_id, "missing from the episodes CSV")
        raw = rec.raw
        pairs = [("steps", raw["steps"]), ("all_at_goal", raw["all_at_goal"]),
                 ("timed_out", raw["timed_out"]), ("termination", rec.termination),
                 ("stop_reason", raw.get("stop_reason")), ("method", rec.method),
                 ("max_steps", rec.max_steps)]
        for key, value in pairs:
            if row[key] != format_value(value):
                _fail(rec.run_id, f"CSV {key}={row[key]!r} != runner {value!r}")
            checks += 1
        rows = sorted(calls_by_run[rec.run_id], key=lambda r: int(r["call_index"]))
        if len(rows) != (len(raw["cbs_calls"]) if rec.method in CBS_METHODS else 0):
            _fail(rec.run_id, "CBS call rows differ from the runner's calls")
        for row_c, call in zip(rows, raw.get("cbs_calls", [])):
            for key, value in (("status", call["status"]), ("accepted", call["accepted"]),
                               ("reason", call["reason"]), ("error", call["error"]),
                               ("solver_latency_ms", call["latency_ms"])):
                if row_c[key] != format_value(value):
                    _fail(rec.run_id, f"call {row_c['call_index']} CSV {key}={row_c[key]!r} != runner {value!r}")
                checks += 1
    return checks
