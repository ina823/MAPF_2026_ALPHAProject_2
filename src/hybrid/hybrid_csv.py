"""Global Hybrid CSV export: saved result.json + steps CSV -> two Global-only CSVs.

Reads a finished Global Hybrid run directory (outputs/hybrid_e2e/<run_id>/,
written by scripts/run_corridor_hybrid_e2e.py or
scripts/run_hybrid_resume_dev.py) and never writes into it. The existing
EpisodeLogger CSVs (steps/summary/monitor) and their schemas are untouched;
these two files are separate:

    hybrid_episodes.csv   -- one row per run
    hybrid_cbs_calls.csv  -- one row per CBS call (any number per run)

Every run is validated before anything is written: result.json internal
consistency, steps CSV <-> trajectory/action_sources, and the attribution of
each CBS execution step to exactly one call. Anything missing or ambiguous
raises HybridCsvError -- values are never guessed.

Field semantics:
  * solver_latency_ms / cbs_latency_ms_total: ReplanResult.latency_sec * 1000
    as recorded by the run, i.e. the perf_counter span of replan_with_status()
    -- YAML writing, the atb033 cbs.py subprocess (interpreter start-up
    included), output parsing and replanner's Path Guard. NOT pure CBS search
    time, and it excludes the runtime adapter's and the boundary's Path Guard.
  * h_exec_policy: DEV_ONLY_FULL_PATH (the whole accepted path; there is no
    numeric H_exec, so h_exec_config is empty) or DEV_ONLY_FIXED (h_exec_config
    = the configured window). Neither is an agreed operating policy.
  * planned_horizon = len(longest CBS path) - 1; exec_window = planned_horizon
    (FULL_PATH) or min(h_exec_config, planned_horizon) (FIXED); exec_steps =
    CBS steps actually executed for that call.
  * cbs_exec_steps counts CBS-controlled timesteps; cbs_controlled_agent_steps
    counts steps-CSV (timestep, agent) rows at those timesteps.
  * trigger_timestep: the Trigger that started the first CBS call. first_seen /
    period: recorded only when replaying the real DeadlockMonitor over the
    saved trajectory reproduces that same trigger_timestep (its repeat test
    depends on positions only); otherwise empty, see first_seen_source.
  * termination (SUCCESS / TIMEOUT / SAFE_STOP / INTEGRITY_ABORT) and the final
    all_at_goal are kept as recorded; verdict is the run script's own verdict.

Empty cell = not applicable / not recorded -- never a stand-in for 0.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Sequence

from src.common.deadlock_monitor import DeadlockMonitor

EPISODE_FIELDS: Sequence[str] = (
    "run_id", "scenario", "mode", "num_agents", "max_steps", "steps",
    "termination", "stop_reason", "all_at_goal", "overlap_any_step",
    "verdict", "first_failed_check",
    "trigger_timestep", "first_seen", "period", "first_seen_source",
    "h_exec_config", "h_exec_policy",
    "cbs_call_count", "cbs_latency_ms_total",
    "cbs_exec_steps", "cbs_controlled_agent_steps",
    "il_steps", "il_resume_t", "resume_outcome",
    "solver_commit", "project_commit", "project_dirty",
    "source_result_json",
)

CALL_FIELDS: Sequence[str] = (
    "run_id", "call_index", "trigger_timestep",
    "status", "accepted", "reason", "error",
    "solver_latency_ms",
    "h_exec_config", "planned_horizon", "exec_window", "exec_steps",
    "controlled_agent_steps", "agents_at_goal_at_exec_end", "exec_end",
)

EPISODES_CSV = "hybrid_episodes.csv"
CALLS_CSV = "hybrid_cbs_calls.csv"
MANIFEST_JSON = "export_manifest.json"

TERMINATIONS = ("SUCCESS", "TIMEOUT", "SAFE_STOP", "INTEGRITY_ABORT")
FULL_PATH = "DEV_ONLY_FULL_PATH"
FIXED = "DEV_ONLY_FIXED"

REQUIRED_KEYS = (
    "mode", "scenario", "max_steps", "steps", "termination", "stop_reason",
    "all_at_goal", "overlap_any_step", "verdict", "goals", "trajectory",
    "action_sources", "transitions", "cbs_calls", "cbs_exec_trace",
)
CALL_KEYS = ("trigger_timestep", "s_t", "status", "accepted", "reason", "error", "latency_ms")
TRACE_KEYS = ("t", "k", "expected", "actual", "match", "overlap")


class HybridCsvError(ValueError):
    """A run cannot be exported without guessing; nothing is written."""


def _fail(run_id, message):
    raise HybridCsvError(f"{run_id}: {message}")


def _pos(p):
    return (int(p[0]), int(p[1]))


def _state(mapping):
    return {int(aid): _pos(p) for aid, p in mapping.items()}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def format_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return repr(value)
    return str(value)


# -- Loading ------------------------------------------------------------------

def load_run(run_dir):
    """Read result.json and <run_id>_steps.csv of one run directory (read only)."""
    run_dir = Path(run_dir)
    run_id = run_dir.name
    result_path = run_dir / "result.json"
    steps_path = run_dir / f"{run_id}_steps.csv"
    if not result_path.is_file():
        _fail(run_id, f"missing {result_path}")
    if not steps_path.is_file():
        _fail(run_id, f"missing {steps_path}")
    try:
        report = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _fail(run_id, f"result.json is not valid JSON ({exc})")
    if not isinstance(report, dict):
        _fail(run_id, "result.json is not a JSON object")
    with steps_path.open(newline="", encoding="utf-8") as fh:
        steps_rows = list(csv.DictReader(fh))
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "report": report,
        "steps_rows": steps_rows,
        "result_path": result_path,
        "steps_path": steps_path,
    }


# -- Validation ---------------------------------------------------------------

def h_exec_policy(run_id, report):
    """(h_exec_policy, h_exec_config or None). Never reads max_steps as H_exec."""
    if "h_exec" in report:
        h = report["h_exec"]
        if not isinstance(h, int) or isinstance(h, bool) or h < 1:
            _fail(run_id, f"h_exec must be an int >= 1, got {h!r}")
        if report["mode"] != f"GLOBAL_HYBRID_DEV_H{h}":
            _fail(run_id, f"mode {report['mode']!r} does not match h_exec={h}")
        return FIXED, h
    dev_h = str(report.get("dev_only_policies", {}).get("h_exec", ""))
    if report["mode"] == "GLOBAL_HYBRID_DEV" and dev_h.startswith("max_steps="):
        return FULL_PATH, None
    _fail(run_id, f"unknown H_exec policy (mode={report['mode']!r}, dev h_exec={dev_h!r})")


def replay_monitor_trigger(trajectory, goals):
    """Replay the real DeadlockMonitor over a saved trajectory.

    Its Full-repeat Trigger depends on positions only, so trigger_timestep /
    first_seen / period are reproducible. Returns (t, first_seen, period) or
    None when no Trigger fires."""
    monitor = DeadlockMonitor()
    monitor.reset(trajectory[0], goals)
    for t in range(1, len(trajectory)):
        r = monitor.observe_step(t, trajectory[t], {})
        if r.trigger_type == "FULL":
            return r.trigger_timestep, r.first_seen, r.period
    return None


def normalize_report(run_id, report):
    """Check result.json's internal consistency; return typed values."""
    missing = [k for k in REQUIRED_KEYS if k not in report]
    if missing:
        _fail(run_id, f"result.json missing field(s): {', '.join(missing)}")

    goals = _state(report["goals"])
    trajectory = [_state(s) for s in report["trajectory"]]
    steps = report["steps"]
    sources = list(report["action_sources"])
    termination = report["termination"]
    all_at_goal = report["all_at_goal"]
    stop_reason = report["stop_reason"]

    if not isinstance(steps, int) or len(trajectory) != steps + 1 or len(sources) != steps:
        _fail(run_id, "steps, trajectory and action_sources lengths disagree")
    if set(sources) - {"IL", "CBS"}:
        _fail(run_id, f"unknown action source(s) {sorted(set(sources) - {'IL', 'CBS'})}")
    if any(set(s) != set(goals) for s in trajectory):
        _fail(run_id, "trajectory agent ids differ from goals")
    if termination not in TERMINATIONS:
        _fail(run_id, f"unknown termination {termination!r}")
    if not isinstance(all_at_goal, bool) or (termination == "SUCCESS") != all_at_goal:
        _fail(run_id, f"termination={termination} inconsistent with all_at_goal={all_at_goal!r}")
    if all_at_goal and trajectory[-1] != goals:
        _fail(run_id, "all_at_goal=True but the last state is not the goal state")
    if termination == "TIMEOUT" and steps != report["max_steps"]:
        _fail(run_id, "TIMEOUT before max_steps")
    if termination == "SAFE_STOP":
        if not isinstance(stop_reason, str) or not stop_reason:
            _fail(run_id, "SAFE_STOP without stop_reason")
    elif stop_reason is not None:
        _fail(run_id, f"stop_reason {stop_reason!r} on a {termination} run")

    overlaps = [len(set(s.values())) != len(s) for s in trajectory]
    if any(overlaps) != report["overlap_any_step"]:
        _fail(run_id, "overlap_any_step disagrees with the trajectory")
    if (termination == "INTEGRITY_ABORT") != overlaps[-1]:
        _fail(run_id, "INTEGRITY_ABORT and a final-state overlap must coincide")

    if report["verdict"] not in ("PASS", "FAIL"):
        _fail(run_id, f"unknown verdict {report['verdict']!r}")
    if report["verdict"] == "PASS" and termination != "SUCCESS":
        _fail(run_id, f"verdict PASS on a {termination} run")

    calls = []
    for i, call in enumerate(report["cbs_calls"], start=1):
        missing = [k for k in CALL_KEYS if k not in call]
        if missing:
            _fail(run_id, f"cbs_calls[{i}] missing field(s): {', '.join(missing)}")
        t = call["trigger_timestep"]
        if not isinstance(t, int) or not 1 <= t <= steps:
            _fail(run_id, f"cbs_calls[{i}] trigger_timestep {t!r} out of range")
        s_t = _state(call["s_t"])
        if s_t != trajectory[t]:
            _fail(run_id, f"cbs_calls[{i}] S_t differs from the trajectory at t={t}")
        paths = call.get("paths")
        paths = {int(a): [_pos(p) for p in path] for a, path in paths.items()} if paths else None
        if call["accepted"] is True and paths is not None:
            if set(paths) != set(goals) or any(paths[a][0] != s_t[a] for a in paths):
                _fail(run_id, f"cbs_calls[{i}] accepted paths do not start at S_t for every agent")
        calls.append({**call, "index": i, "s_t": s_t, "paths": paths})
    if any(b["trigger_timestep"] <= a["trigger_timestep"] for a, b in zip(calls, calls[1:])):
        _fail(run_id, "cbs_calls trigger_timesteps are not strictly increasing")

    trace = []
    for j, e in enumerate(report["cbs_exec_trace"]):
        missing = [k for k in TRACE_KEYS if k not in e]
        if missing:
            _fail(run_id, f"cbs_exec_trace[{j}] missing field(s): {', '.join(missing)}")
        entry = {**e, "expected": _state(e["expected"]), "actual": _state(e["actual"])}
        t = entry["t"]
        if not isinstance(t, int) or not 1 <= t <= steps or sources[t - 1] != "CBS":
            _fail(run_id, f"cbs_exec_trace[{j}] t={t!r} is not a CBS step")
        if entry["actual"] != trajectory[t]:
            _fail(run_id, f"cbs_exec_trace[{j}] actual differs from the trajectory at t={t}")
        if entry["match"] != (entry["expected"] == entry["actual"]):
            _fail(run_id, f"cbs_exec_trace[{j}] match flag is inconsistent")
        if entry["overlap"] != overlaps[t]:
            _fail(run_id, f"cbs_exec_trace[{j}] overlap flag is inconsistent")
        trace.append(entry)
    cbs_ts = [i + 1 for i, s in enumerate(sources) if s == "CBS"]
    if sorted(e["t"] for e in trace) != cbs_ts:
        _fail(run_id, "cbs_exec_trace does not cover exactly the CBS steps in action_sources")

    return {
        "goals": goals,
        "trajectory": trajectory,
        "steps": steps,
        "sources": sources,
        "termination": termination,
        "all_at_goal": all_at_goal,
        "stop_reason": stop_reason,
        "overlap_any_step": report["overlap_any_step"],
        "transitions": list(report["transitions"]),
        "calls": calls,
        "trace": trace,
    }


def attribute_trace(run_id, calls, trace):
    """Assign each CBS step to the call whose Trigger precedes it. A call's
    steps must be k = 1..n at t = trigger+1 .. trigger+n; a refused call owns
    none. Any other layout is ambiguous and raises."""
    owned = {c["index"]: [] for c in calls}
    for e in sorted(trace, key=lambda e: e["t"]):
        owners = [c for c in calls if c["trigger_timestep"] < e["t"]]
        if not owners:
            _fail(run_id, f"CBS step t={e['t']} precedes every CBS call")
        owned[owners[-1]["index"]].append(e)
    for c in calls:
        entries = owned[c["index"]]
        if not c["accepted"] and entries:
            _fail(run_id, f"call {c['index']} was refused but owns CBS steps {[e['t'] for e in entries]}")
        expect_t = [c["trigger_timestep"] + k for k in range(1, len(entries) + 1)]
        if [e["t"] for e in entries] != expect_t or [e["k"] for e in entries] != list(range(1, len(entries) + 1)):
            _fail(run_id, f"call {c['index']}: CBS steps cannot be attributed unambiguously "
                          f"(t={[e['t'] for e in entries]}, k={[e['k'] for e in entries]})")
    return owned


def _exec_end(run_id, call, entries, norm):
    if not call["accepted"]:
        return "REFUSED"
    transitions = norm["transitions"]
    if not entries:
        if any(tr["t"] == call["trigger_timestep"] and tr["from"] == "CBS_REQUEST"
               and tr["to"] == "IL_RESUME" and tr["reason"] == "EMPTY_PLAN" for tr in transitions):
            return "EMPTY_PLAN"
        _fail(run_id, f"call {call['index']} accepted but executed no CBS step")
    last = entries[-1]
    if last["overlap"]:
        return "INTEGRITY_ABORT"
    if not last["match"]:
        return "EXEC_MISMATCH"
    if any(tr["t"] == last["t"] and tr["from"] == "CBS_EXEC" and tr["to"] == "IL_RESUME"
           for tr in transitions):
        return "WINDOW_END"
    if last["t"] == norm["steps"] and norm["termination"] == "SUCCESS":
        return "GOAL_REACHED"
    if last["t"] == norm["steps"] and norm["termination"] == "TIMEOUT":
        return "EPISODE_TIMEOUT"
    _fail(run_id, f"call {call['index']}: cannot tell how CBS execution ended at t={last['t']}")


def cross_validate_steps(run_id, report, norm, steps_rows):
    """steps CSV <-> result.json. Returns the rows grouped by timestep and the
    first timestep whose trigger_reason is TRIGGER_FULL_REPEAT (or None)."""
    goals, trajectory, steps = norm["goals"], norm["trajectory"], norm["steps"]
    ids = set(goals)
    if len(steps_rows) != steps * len(ids):
        _fail(run_id, f"steps CSV has {len(steps_rows)} rows, expected steps*agents = {steps * len(ids)}")
    by_t = defaultdict(list)
    for row in steps_rows:
        by_t[int(row["timestep"])].append(row)
    if sorted(by_t) != list(range(1, steps + 1)):
        _fail(run_id, "steps CSV timesteps are not exactly 1..steps")
    for t, rows in by_t.items():
        if {int(r["agent_id"]) for r in rows} != ids or len(rows) != len(ids):
            _fail(run_id, f"steps CSV t={t}: agent rows differ from the run's agents")
        for r in rows:
            aid = int(r["agent_id"])
            if r["run_id"] != run_id or r["mode"] != report["mode"]:
                _fail(run_id, f"steps CSV t={t}: run_id/mode {r['run_id']!r}/{r['mode']!r} do not match")
            if (int(r["current_row"]), int(r["current_col"])) != trajectory[t - 1][aid]:
                _fail(run_id, f"steps CSV t={t} agent {aid}: current position differs from the trajectory")
            if (int(r["next_row"]), int(r["next_col"])) != trajectory[t][aid]:
                _fail(run_id, f"steps CSV t={t} agent {aid}: next position differs from the trajectory")
    if {r["success"] for r in by_t[steps]} != {format_value(norm["all_at_goal"])}:
        _fail(run_id, "steps CSV final success column differs from all_at_goal")
    reasons = {r["failure_reason"] for r in steps_rows} - {""}
    if ("integrity_overlap" in reasons) != (norm["termination"] == "INTEGRITY_ABORT"):
        _fail(run_id, "steps CSV integrity_overlap rows disagree with termination")
    if ("timeout" in reasons) != (norm["termination"] == "TIMEOUT"):
        _fail(run_id, "steps CSV timeout rows disagree with termination")
    first_trigger = next((t for t in sorted(by_t)
                          if by_t[t][0]["trigger_reason"] == "TRIGGER_FULL_REPEAT"), None)
    return by_t, first_trigger


# -- Row building -------------------------------------------------------------

def _scenario_name(path_str):
    return re.split(r"[\\/]", str(path_str))[-1]


def build_rows(run):
    """(episode_row, call_rows) for one loaded run; validates everything first."""
    run_id, report = run["run_id"], run["report"]
    norm = normalize_report(run_id, report)
    policy, h_exec = h_exec_policy(run_id, report)
    by_t, csv_first_trigger = cross_validate_steps(run_id, report, norm, run["steps_rows"])
    calls, trace, goals, trajectory = norm["calls"], norm["trace"], norm["goals"], norm["trajectory"]
    owned = attribute_trace(run_id, calls, trace)

    call_rows = []
    for c in calls:
        entries = owned[c["index"]]
        horizon = max(len(p) for p in c["paths"].values()) - 1 if c["paths"] else None
        if not c["accepted"] or horizon is None:
            window = None
        elif policy == FULL_PATH:
            window = horizon
        else:
            window = min(h_exec, horizon)
        if c["paths"] and entries and entries[0]["expected"] != {a: p[1] for a, p in c["paths"].items()
                                                                if len(p) > 1}:
            _fail(run_id, f"call {c['index']}: first CBS step is not Path[1]")
        if window is not None and len(entries) > window:
            _fail(run_id, f"call {c['index']}: executed {len(entries)} steps, window {window}")
        exec_end = _exec_end(run_id, c, entries, norm)
        if exec_end == "WINDOW_END" and window is not None and len(entries) != window:
            _fail(run_id, f"call {c['index']}: WINDOW_END after {len(entries)} of {window} steps")
        controlled = sum(len(by_t[e["t"]]) for e in entries)
        at_goal = (sum(1 for a in goals if trajectory[entries[-1]["t"]][a] == goals[a])
                   if entries else None)
        call_rows.append({
            "run_id": run_id,
            "call_index": c["index"],
            "trigger_timestep": c["trigger_timestep"],
            "status": c["status"],
            "accepted": c["accepted"],
            "reason": c["reason"],
            "error": c["error"],
            "solver_latency_ms": c["latency_ms"],
            "h_exec_config": h_exec,
            "planned_horizon": horizon,
            "exec_window": window,
            "exec_steps": len(entries),
            "controlled_agent_steps": controlled,
            "agents_at_goal_at_exec_end": at_goal,
            "exec_end": exec_end,
        })

    if calls:
        trigger_t = calls[0]["trigger_timestep"]
        if csv_first_trigger != trigger_t:
            _fail(run_id, f"steps CSV first TRIGGER_FULL_REPEAT at t={csv_first_trigger}, "
                          f"first CBS call at t={trigger_t}")
        replay = replay_monitor_trigger(trajectory, goals)
        if replay is not None and replay[0] == trigger_t:
            first_seen, period, source = replay[1], replay[2], "MONITOR_REPLAY_TRAJECTORY"
        else:
            first_seen, period = None, None
            source = f"UNAVAILABLE:replay_trigger={replay[0] if replay else None}"
    else:
        trigger_t, first_seen, period, source = None, None, None, "NO_CBS_CALL"

    latencies = [c["latency_ms"] for c in calls]
    latency_total = sum(latencies) if latencies and None not in latencies else None
    resume_t = next((tr["t"] for tr in norm["transitions"] if tr["to"] == "IL_RESUME"), None)
    episode = {
        "run_id": run_id,
        "scenario": _scenario_name(report["scenario"]),
        "mode": report["mode"],
        "num_agents": len(goals),
        "max_steps": report["max_steps"],
        "steps": norm["steps"],
        "termination": norm["termination"],
        "stop_reason": norm["stop_reason"],
        "all_at_goal": norm["all_at_goal"],
        "overlap_any_step": norm["overlap_any_step"],
        "verdict": report["verdict"],
        "first_failed_check": report.get("first_failed_check"),
        "trigger_timestep": trigger_t,
        "first_seen": first_seen,
        "period": period,
        "first_seen_source": source,
        "h_exec_config": h_exec,
        "h_exec_policy": policy,
        "cbs_call_count": len(calls),
        "cbs_latency_ms_total": latency_total,
        "cbs_exec_steps": len(trace),
        "cbs_controlled_agent_steps": sum(r["controlled_agent_steps"] for r in call_rows),
        "il_steps": norm["sources"].count("IL"),
        "il_resume_t": resume_t,
        "resume_outcome": (report.get("resume") or {}).get("outcome"),
        "solver_commit": (report.get("solver_git") or {}).get("commit"),
        "project_commit": (report.get("project_git") or {}).get("commit"),
        "project_dirty": (report.get("project_git") or {}).get("dirty"),
        "source_result_json": str(run["result_path"]),
    }
    if episode["cbs_controlled_agent_steps"] != episode["cbs_exec_steps"] * episode["num_agents"]:
        _fail(run_id, "CBS-controlled agent-steps differ from CBS steps x agents")
    return episode, call_rows


# -- Writing and checking -----------------------------------------------------

def _write_csv(path, fields, rows):
    with Path(path).open("x", newline="", encoding="utf-8") as fh:  # never overwrite
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: format_value(row[k]) for k in fields})


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def _same(csv_value, json_value):
    if json_value is None:
        return csv_value == ""
    if isinstance(json_value, bool):
        return csv_value == format_value(json_value)
    if isinstance(json_value, (int, float)):
        return csv_value != "" and float(csv_value) == float(json_value)
    return csv_value == str(json_value)


def check_csv_against_json(output_dir, runs):
    """Re-read the written CSVs and compare them with the source result.json
    values directly (not with the built rows). Returns the number of checks."""
    episode_fields, episode_rows = _read_csv(Path(output_dir) / EPISODES_CSV)
    call_fields, call_rows = _read_csv(Path(output_dir) / CALLS_CSV)
    if episode_fields != list(EPISODE_FIELDS) or call_fields != list(CALL_FIELDS):
        raise HybridCsvError("written CSV headers differ from the defined fields")
    episodes = {r["run_id"]: r for r in episode_rows}
    calls = defaultdict(list)
    for r in call_rows:
        calls[r["run_id"]].append(r)
    checks = 0
    for run in runs:
        run_id, report = run["run_id"], run["report"]
        ep = episodes.get(run_id)
        if ep is None:
            _fail(run_id, "missing from hybrid_episodes.csv")
        for key in ("mode", "termination", "stop_reason", "all_at_goal", "overlap_any_step",
                    "verdict", "steps", "max_steps"):
            if not _same(ep[key], report[key]):
                _fail(run_id, f"CSV {key}={ep[key]!r} != JSON {report[key]!r}")
            checks += 1
        json_calls = report["cbs_calls"]
        if int(ep["cbs_call_count"]) != len(json_calls) or len(calls[run_id]) != len(json_calls):
            _fail(run_id, "CBS call count differs between CSV and JSON")
        checks += 1
        if json_calls and int(ep["trigger_timestep"]) != json_calls[0]["trigger_timestep"]:
            _fail(run_id, "trigger_timestep differs between CSV and JSON")
        if ep["cbs_exec_steps"] != str(len(report["cbs_exec_trace"])):
            _fail(run_id, "cbs_exec_steps differs from the JSON trace length")
        if ep["il_steps"] != str(report["action_sources"].count("IL")):
            _fail(run_id, "il_steps differs from JSON action_sources")
        checks += 3
        for row, jc in zip(sorted(calls[run_id], key=lambda r: int(r["call_index"])), json_calls):
            for csv_key, json_key in (("trigger_timestep", "trigger_timestep"), ("status", "status"),
                                      ("accepted", "accepted"), ("reason", "reason"),
                                      ("error", "error"), ("solver_latency_ms", "latency_ms")):
                if not _same(row[csv_key], jc[json_key]):
                    _fail(run_id, f"call {row['call_index']} CSV {csv_key}={row[csv_key]!r} "
                                  f"!= JSON {json_key}={jc[json_key]!r}")
                checks += 1
    return checks


def export_runs(run_dirs, output_dir, *, command=None):
    """Validate every run, then write both CSVs and a manifest into a NEW
    output_dir. Nothing is written if any run fails validation."""
    runs = [load_run(d) for d in run_dirs]
    if not runs:
        raise HybridCsvError("no run directories given")
    seen = set()
    for run in runs:
        if run["run_id"] in seen:
            _fail(run["run_id"], "duplicate run_id in the export")
        seen.add(run["run_id"])
    built = [build_rows(run) for run in runs]
    hashes_before = {run["run_id"]: (_sha256(run["result_path"]), _sha256(run["steps_path"]))
                     for run in runs}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)  # never reuse an output directory
    _write_csv(output_dir / EPISODES_CSV, EPISODE_FIELDS, [ep for ep, _ in built])
    _write_csv(output_dir / CALLS_CSV, CALL_FIELDS, [row for _, rows in built for row in rows])
    checks = check_csv_against_json(output_dir, runs)

    for run in runs:
        if (_sha256(run["result_path"]), _sha256(run["steps_path"])) != hashes_before[run["run_id"]]:
            _fail(run["run_id"], "source files changed during the export")
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "command": command,
        "files": [EPISODES_CSV, CALLS_CSV],
        "inputs": [{
            "run_id": run["run_id"],
            "result_json": str(run["result_path"]),
            "result_json_sha256": hashes_before[run["run_id"]][0],
            "steps_csv": str(run["steps_path"]),
            "steps_csv_sha256": hashes_before[run["run_id"]][1],
        } for run in runs],
        "json_csv_value_checks": checks,
        "notes": {
            "solver_latency_ms": "replan_with_status() perf_counter span (YAML I/O, cbs.py "
                                 "subprocess incl. interpreter start-up, parsing, replanner "
                                 "Path Guard); not pure CBS search time",
            "h_exec_policy": "DEV_ONLY_FULL_PATH has no numeric H_exec; neither policy is agreed",
            "first_seen": "set only when a DeadlockMonitor replay of the trajectory reproduces "
                          "the first CBS call's trigger_timestep",
        },
    }
    with (output_dir / MANIFEST_JSON).open("x", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return {"output_dir": output_dir, "episodes": [ep for ep, _ in built],
            "calls": [row for _, rows in built for row in rows], "checks": checks}
