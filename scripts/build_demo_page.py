"""Build one self-contained demo page from saved Bottleneck runs (no solver, no IL).

Reads, read-only, from a saved comparison batch: each run's result.json and
*_steps.csv, compare_episodes.csv, compare_cbs_calls.csv and the scenario file.
Every value shown is copied from those files or recomputed from them, and the
copies are cross-checked first; a disagreement raises BuildError and nothing is
written.

Evidence rules the page keeps visible:
  * A value with no source is shown as "기록 없음", never as a guess or a zero.
  * `accepted` is what the run recorded at the time. No run recorded a separate
    Path Guard result, so that row says "기록 없음". The post-hoc recheck runs the
    real validate_path_guard here, on the saved planner paths, and is labelled so.
  * Real SoC is recomputed from the trajectory (first t from which an agent stays
    on its goal, summed), never from padded path lengths.
  * vertex_conflicts counts conflict events (a cell two or more agents tried to
    enter, so their moves were cancelled); real position overlap is counted
    separately from the trajectory.
  * H_exec, cooldown and the CBS call limit are Development settings.
  * The runs did not record PYTHONHASHSEED and may differ in their first CBS path,
    so H3 and H3_REARM are not presented as a controlled causal comparison.

Output: a NEW directory (refused if it exists), default
outputs/videos/demo_panel_<YYYYmmdd_HHMMSS>/, with demo_panel.html and
demo_manifest.json (inputs and output with sha256).

Exit codes: 0 written, 1 inconsistent input (nothing written), 2 output exists.

Usage (from the repository root):
    python -m scripts.build_demo_page [--batch-dir <batch>] [--scenario-key bottleneck_n4] [--out-dir <new dir>]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.render_trajectory import RenderError, load_run_data
from scripts.run_il import load_scenario
from src.compare.compare_csv import sum_of_costs
from src.path_guard import validate_path_guard

DEFAULT_BATCH = PROJECT_ROOT / "outputs" / "logs" / "compare_rearm_dev_20261105"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs" / "videos"
DEFAULT_SCENARIO_KEY = "bottleneck_n4"
METHODS = ("IL_ONLY", "GLOBAL_HYBRID_DEV_H3", "GLOBAL_HYBRID_DEV_H3_REARM")
LABELS = {"IL_ONLY": "IL_ONLY", "GLOBAL_HYBRID_DEV_H3": "H3", "GLOBAL_HYBRID_DEV_H3_REARM": "H3_REARM"}
DESCRIPTIONS = {"IL_ONLY": "학습된 IL만 사용", "GLOBAL_HYBRID_DEV_H3": "정체 감지 시 CBS 개입, 3 step 후 IL 복귀",
                "GLOBAL_HYBRID_DEV_H3_REARM": "다시 정체되면 CBS 재개입 가능"}
PG_PASS, PG_FAIL, PG_NOT_RECORDED = "PASS", "FAIL", "NOT_RECORDED"


class BuildError(ValueError):
    """The saved inputs disagree or are incomplete; no page is written."""


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project_relative(path) -> str:
    """A path relative to the project; outside it only the last two parts are kept,
    so no absolute local path reaches the page or its manifest."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return f"(프로젝트 외부)/{resolved.parent.name}/{resolved.name}"


def read_csv(path) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def find_run_dir(batch_dir: Path, scenario_key: str, method: str) -> Path:
    matches = sorted((batch_dir / "runs").glob(f"*_{scenario_key}_{method}"))
    if len(matches) != 1:
        raise BuildError(f"expected one run for {scenario_key}/{method}, found {len(matches)}")
    return matches[0]


# -- values derived from the saved run ---------------------------------------------

def segments(sources: Sequence[str]) -> List[Dict]:
    """Merge per-step action sources into [from_step, to_step] segments."""
    out: List[Dict] = []
    for step, source in enumerate(sources, 1):
        if out and out[-1]["source"] == source:
            out[-1]["to"] = step
        else:
            out.append({"from": step, "to": step, "source": source})
    return out


def stall_segments(kinds: Sequence[Sequence[str]], at_goal: Sequence[Sequence[bool]]) -> List[Dict]:
    """Steps where no agent moved and not everyone is at its goal. Computed here from
    the trajectory and actions; this is not a Monitor record."""
    out: List[Dict] = []
    for step, row in enumerate(kinds, 1):
        if "MOVE" in row or all(at_goal[step]):
            continue
        if out and out[-1]["to"] == step - 1:
            out[-1]["to"] = step
        else:
            out.append({"from": step, "to": step})
    return out


def monitor_records(monitor: Dict) -> List[Dict]:
    """Monitor Trigger records: the rearm history if present, else the latched one."""
    history = monitor.get("trigger_history") or []
    if history:
        return [dict(h) for h in history]
    if monitor.get("trigger_timestep") is not None:
        return [{k: monitor.get(k) for k in ("trigger_type", "trigger_reason_code", "trigger_timestep",
                                             "first_seen", "period", "blocking_type", "blocking_agents")}]
    return []


def recheck_path_guard(grid, s_t: Dict, goals: Dict, planner_call: Optional[Dict]) -> Dict:
    """Post-hoc run of the real Path Guard on a saved path; not a record of the run."""
    paths = (planner_call or {}).get("paths")
    if not paths:
        return {"status": PG_NOT_RECORDED, "reason": ""}
    try:
        validate_path_guard(paths={int(a): [tuple(c) for c in p] for a, p in paths.items()},
                            current_positions={int(a): tuple(v) for a, v in s_t.items()},
                            goals=goals, grid_map=grid)
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        return {"status": PG_FAIL, "reason": f"{type(exc).__name__}: {exc}"}
    return {"status": PG_PASS, "reason": ""}


def dev_lines(dev: Dict) -> List[str]:
    """Development settings as recorded in result.json; nothing is filled in."""
    if not dev:
        return []
    lines = []
    if "h_exec" in dev:
        lines.append(f"H_exec={dev['h_exec']}")
    limits = dev.get("rearm_limits")
    if isinstance(limits, dict):
        if "cooldown_steps" in limits:
            lines.append(f"cooldown={limits['cooldown_steps']}")
        if "max_cbs_calls" in limits:
            lines.append(f"CBS 최대 {limits['max_cbs_calls']}회")
    elif "first Trigger only" in str(dev.get("recall", "")):
        lines.append("CBS는 첫 Trigger에서만 호출")
    return lines


def _optional_int(text: str) -> Optional[int]:
    return int(text) if text not in ("", None) else None


def _optional_float(text: str) -> Optional[float]:
    return float(text) if text not in ("", None) else None


def build_run_view(run_dir: Path, episode: Dict, call_rows: List[Dict]) -> Dict:
    """Checked, display-ready data for one run."""
    run_dir = Path(run_dir)
    try:
        view = load_run_data(run_dir)
    except RenderError as exc:
        raise BuildError(str(exc)) from exc
    run_id = view["run_id"]
    raw = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))

    def fail(message):
        raise BuildError(f"{run_id}: {message}")

    if sha256_file(raw["scenario"]) != raw["scenario_sha256"]:
        fail("scenario file differs from the recorded sha256")
    grid, _, goals = load_scenario(raw["scenario"])
    trajectory = [{int(a): tuple(p) for a, p in s.items()} for s in raw["trajectory"]]
    agents = view["agents"]

    overlap_steps = [t for t, s in enumerate(trajectory) if len(set(s.values())) != len(s)]
    if bool(overlap_steps) != raw["overlap_any_step"]:
        fail("overlap_any_step disagrees with the trajectory")
    soc = sum_of_costs(trajectory, goals) if raw["all_at_goal"] else None

    if episode["termination"] != raw["termination"] or int(episode["steps"]) != raw["steps"]:
        fail("compare_episodes.csv termination/steps differ from result.json")
    if episode["scenario_sha256"] != raw["scenario_sha256"]:
        fail("compare_episodes.csv scenario sha differs from result.json")
    if episode["sum_of_costs"] != ("" if soc is None else str(soc)):
        fail(f"sum_of_costs {episode['sum_of_costs']!r} differs from the recomputed {soc!r}")
    if episode["overlap_any_step"] != str(bool(overlap_steps)):
        fail("compare_episodes.csv overlap_any_step differs from the trajectory")
    makespan = _optional_int(episode["makespan"])
    if makespan is not None and makespan != raw["steps"]:
        fail("makespan differs from steps")

    rows = read_csv(next(iter(sorted(run_dir.glob("*_steps.csv"))), run_dir / "missing"))
    if len(rows) != raw["steps"] * len(agents):
        fail("steps CSV row count differs from steps x agents")
    for row in rows:
        t, aid = int(row["timestep"]), int(row["agent_id"])
        current = (int(row["current_row"]), int(row["current_col"]))
        following = (int(row["next_row"]), int(row["next_col"]))
        if trajectory[t - 1][aid] != current or trajectory[t][aid] != following:
            fail(f"steps CSV differs from the trajectory at t={t}, agent {aid}")

    monitor = monitor_records(raw["monitor"])
    by_trigger = {m["trigger_timestep"]: m for m in monitor}
    if not raw["monitor"].get("trigger_history"):
        recorded = _optional_int(episode["monitor_trigger_timestep"])
        if recorded != raw["monitor"].get("trigger_timestep"):
            fail("compare_episodes.csv Monitor Trigger differs from result.json")

    if len(raw["planner_calls"]) != len(raw["cbs_calls"]) or len(call_rows) != len(raw["cbs_calls"]):
        fail("CBS call records differ in number")
    if (_optional_int(episode["cbs_call_count"]) or 0) != len(raw["cbs_calls"]):
        fail("compare_episodes.csv CBS call count differs from result.json")
    latencies = [c.get("latency_ms") for c in raw["cbs_calls"]]
    total = _optional_float(episode["cbs_latency_ms_total"])
    if total is not None and (None in latencies or abs(sum(latencies) - total) > 1e-6):
        fail("compare_episodes.csv CBS latency total differs from result.json")

    trace_by_call: Dict[int, List[Dict]] = {i: [] for i in range(len(raw["cbs_calls"]))}
    exec_step: Dict[int, List[int]] = {}
    triggers = [c["trigger_timestep"] for c in raw["cbs_calls"]]
    for entry in raw["cbs_exec_trace"]:
        call = max(i for i, t in enumerate(triggers) if t < entry["t"])
        trace_by_call[call].append(entry)
        exec_step[entry["t"]] = [call + 1, entry["k"]]

    calls = []
    for i, (call, planner, row) in enumerate(zip(raw["cbs_calls"], raw["planner_calls"], call_rows)):
        if row["status"] != call["status"] or row["accepted"] != str(call["accepted"]):
            fail(f"compare_cbs_calls.csv call {i + 1} differs from result.json")
        record = by_trigger.get(call["trigger_timestep"], {})
        calls.append({
            "index": i + 1,
            "trigger": call["trigger_timestep"],
            "s_t": {str(a): list(call["s_t"][str(a)]) for a in agents},
            "status": call["status"],
            "latency_ms": call.get("latency_ms"),
            "accepted": call["accepted"],
            "reason": call.get("reason") or "",
            "recheck": recheck_path_guard(grid, call["s_t"], goals, planner),
            "first_seen": record.get("first_seen"),
            "period": record.get("period"),
            "planned_horizon": _optional_int(row["planned_horizon"]),
            "exec_window": _optional_int(row["exec_window"]),
            "exec_steps": _optional_int(row["exec_steps"]),
            "exec_end": row["exec_end"] or None,
            "plan_match": sum(1 for e in trace_by_call[i] if e["match"]),
            "plan_total": len(trace_by_call[i]),
        })

    events = []
    for m in monitor:
        if not raw["cbs_calls"]:
            events.append({"t": m["trigger_timestep"], "kind": "MONITOR", "label": "정체 감지 (Monitor)",
                           "detail": f"{m['trigger_reason_code']}, CBS로 연결되지 않음 (IL 단독)"})
    for tr in raw["transitions"]:
        if tr["to"] == "TRIGGER":
            events.append({"t": tr["t"], "kind": "TRIGGER", "label": "정체 감지 (Monitor Trigger)", "detail": tr["reason"]})
        elif tr["to"] == "CBS_REQUEST":
            call = next(c for c in calls if c["trigger"] == tr["t"])
            events.append({"t": tr["t"], "kind": "CBS", "label": f"CBS 개입 (호출 {call['index']})", "detail": call["status"]})
            events.append({"t": tr["t"], "kind": "GUARD",
                           "label": "Path Guard 포함 수용 검사 " + ("통과" if call["accepted"] else "거절"),
                           "detail": f"accepted={call['accepted']}" + (f", {call['reason']}" if call["reason"] else "")})
        elif tr["to"] == "IL_RESUME":
            events.append({"t": tr["t"], "kind": "RESUME", "label": "IL 복귀", "detail": tr["reason"]})
        elif tr["to"] == "SAFE_STOP":
            events.append({"t": tr["t"], "kind": "STOP", "label": "SAFE_STOP", "detail": tr["reason"]})
    events.append({"t": raw["steps"], "kind": "END", "label": raw["termination"], "detail": raw["stop_reason"] or ""})

    stalls = stall_segments(view["kinds"], view["at_goal"])
    return {
        "method": raw["method"],
        "label": LABELS.get(raw["method"], raw["method"]),
        "desc": DESCRIPTIONS.get(raw["method"], ""),
        "run_id": run_id,
        "termination": raw["termination"],
        "stop_reason": raw["stop_reason"],
        "steps": raw["steps"],
        "max_steps": raw["max_steps"],
        "all_at_goal": raw["all_at_goal"],
        "agents_at_goal_final": sum(view["at_goal"][-1]),
        "soc": soc,
        "makespan": makespan,
        "positions": view["positions"],
        "actions": view["actions"],
        "kinds": view["kinds"],
        "sources": view["sources"],
        "at_goal": view["at_goal"],
        "segments": segments(view["sources"]),
        "stalls": stalls,
        "events": events,
        "calls": calls,
        "exec_step": {str(t): v for t, v in exec_step.items()},
        "monitor": [{"t": m["trigger_timestep"], "first_seen": m.get("first_seen"), "period": m.get("period"),
                     "reason": m.get("trigger_reason_code")} for m in monitor],
        "overlap_steps": overlap_steps,
        "vertex_conflicts": _optional_int(episode["vertex_conflicts"]),
        "edge_conflicts": _optional_int(episode["edge_conflicts"]),
        "cbs_latency_total_ms": total,
        "dev_lines": dev_lines(raw["dev_settings"]),
        "is_dev": bool(raw["dev_settings"]),
        "project_commit": (raw["environment"].get("project_git") or {}).get("commit"),
        "project_dirty": (raw["environment"].get("project_git") or {}).get("dirty"),
        "solver_commit": (raw["environment"].get("solver_git") or {}).get("commit"),
        "hash_seed_recorded": any("pythonhashseed_env" in p for p in raw["planner_calls"]),
        "first_paths": (raw["planner_calls"][0].get("paths") if raw["planner_calls"] else None),
        "first_s_t": (raw["cbs_calls"][0]["s_t"] if raw["cbs_calls"] else None),
        "grid": view["grid"],
        "goals": view["goals"],
        "starts": view["starts"],
        "agents": agents,
        "scenario": view["scenario"],
        "scenario_sha256": raw["scenario_sha256"],
        "grid_sha256": raw["grid_sha256"],
        "scenario_path": raw["scenario"],
        "source_result_path": Path(view["source_result_json"]).as_posix(),
        "source_result_json": project_relative(view["source_result_json"]),
        "source_result_sha256": view["source_result_sha256"],
        "steps_csv_path": next(iter(sorted(run_dir.glob("*_steps.csv")))).as_posix(),
    }


def comparison_note(h3: Dict, rearm: Dict) -> Dict:
    """What the saved H3 and H3_REARM runs do and do not share. Never a causal claim."""
    same_state = h3["first_s_t"] == rearm["first_s_t"]
    same_path = h3["first_paths"] == rearm["first_paths"]
    seed_recorded = h3["hash_seed_recorded"] and rearm["hash_seed_recorded"]
    return {"same_first_s_t": same_state, "same_first_cbs_path": same_path, "hash_seed_recorded": seed_recorded}


def build_panel_data(batch_dir: Path, scenario_key: str = DEFAULT_SCENARIO_KEY) -> Dict:
    batch_dir = Path(batch_dir)
    episodes_path = batch_dir / "compare_episodes.csv"
    calls_path = batch_dir / "compare_cbs_calls.csv"
    for path in (episodes_path, calls_path):
        if not path.is_file():
            raise BuildError(f"missing {path}")
    episodes = {r["run_id"]: r for r in read_csv(episodes_path)}
    call_rows = read_csv(calls_path)

    runs = []
    for method in METHODS:
        run_dir = find_run_dir(batch_dir, scenario_key, method)
        if run_dir.name not in episodes:
            raise BuildError(f"{run_dir.name}: not in compare_episodes.csv")
        rows = sorted((r for r in call_rows if r["run_id"] == run_dir.name), key=lambda r: int(r["call_index"]))
        run = build_run_view(run_dir, episodes[run_dir.name], rows)
        if run["method"] != method:
            raise BuildError(f"{run_dir.name}: method {run['method']} is not {method}")
        runs.append(run)

    first = runs[0]
    for run in runs[1:]:
        for field in ("scenario_sha256", "grid_sha256", "starts", "goals"):
            if run[field] != first[field]:
                raise BuildError(f"{run['run_id']}: {field} differs between runs")

    inputs = [episodes_path, calls_path, Path(first["scenario_path"])]
    for run in runs:
        inputs += [Path(run["source_result_path"]), Path(run["steps_csv_path"])]
    inputs_sha = {p.as_posix(): sha256_file(p) for p in inputs}

    public = ("first_paths", "first_s_t", "grid", "goals", "starts", "agents", "scenario", "scenario_sha256",
              "grid_sha256", "scenario_path", "hash_seed_recorded", "steps_csv_path", "source_result_path")
    note = comparison_note(runs[1], runs[2])
    return {
        "batch_id": batch_dir.name,
        "scenario": first["scenario"],
        "grid": first["grid"],
        "goals": first["goals"],
        "starts": first["starts"],
        "agents": first["agents"],
        "comparison": note,
        "runs": [{k: v for k, v in run.items() if k not in public} for run in runs],
        "evidence": {"batch_id": batch_dir.name,
                     "files": [{"path": project_relative(path), "sha256": sha} for path, sha in inputs_sha.items()]},
        "inputs_sha256": inputs_sha,
    }


# -- page ----------------------------------------------------------------------------

TEMPLATE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>로봇 경로 복구 시연</title>
<style>
:root{--bg:#f4f6f4;--panel:#fff;--fg:#1b211e;--muted:#56625c;--line:#d5dbd7;--wall:#3a423e;--free:#edf0ed;--grid:#dfe4e0;
--il:#2f66ad;--cbs:#b85f14;--stall:#7b4fa8;--ok:#22703a;--bad:#a93526;--warn:#8a5f00;--mark:#1b211e;
--font:system-ui,-apple-system,"Segoe UI","Malgun Gothic",sans-serif;--mono:ui-monospace,Consolas,monospace;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121715;--panel:#1b211f;--fg:#e3e8e5;--muted:#9aa6a0;--line:#323b37;
--wall:#c6cfc9;--free:#232a27;--grid:#2d3531;--il:#79a6e6;--cbs:#eba25b;--stall:#b99ae0;--ok:#86cf8f;--bad:#ef8a7c;--warn:#e3b35a;--mark:#e3e8e5;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#121715;--panel:#1b211f;--fg:#e3e8e5;--muted:#9aa6a0;--line:#323b37;
--wall:#c6cfc9;--free:#232a27;--grid:#2d3531;--il:#79a6e6;--cbs:#eba25b;--stall:#b99ae0;--ok:#86cf8f;--bad:#ef8a7c;--warn:#e3b35a;--mark:#e3e8e5;color-scheme:dark}
*{box-sizing:border-box}
[hidden]{display:none!important}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 var(--font);padding:14px 16px 28px}
main{max-width:1240px;margin:0 auto;display:grid;grid-template-columns:minmax(0,1fr);gap:12px}
h1{font-size:1.3rem;margin:0;text-wrap:balance} h2{font-size:.95rem;margin:0 0 6px}
.top{display:flex;flex-wrap:wrap;gap:4px 16px;align-items:center;justify-content:space-between}
.sub{color:var(--muted);font-size:.9rem}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;min-width:0}
.chip{display:inline-block;padding:0 8px;border:1px solid var(--line);border-radius:999px;font-size:.8rem;white-space:nowrap}
.chip.ok{color:var(--ok);border-color:var(--ok)} .chip.bad{color:var(--bad);border-color:var(--bad)}
button,select{font:inherit;color:var(--fg);background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:5px 11px;cursor:pointer}
button[aria-pressed="true"]{background:var(--fg);color:var(--bg);border-color:var(--fg)}
button:focus-visible,select:focus-visible,input:focus-visible,summary:focus-visible{outline:2px solid var(--il);outline-offset:2px}
#play{background:var(--fg);color:var(--bg);border-color:var(--fg);font-weight:700;min-width:96px}
.row{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(240px,100%),1fr));gap:8px}
button.card{display:block;width:100%;text-align:left;padding:8px 10px;background:var(--panel)}
button.card[aria-pressed="true"]{background:var(--panel);color:var(--fg);border:2px solid var(--fg);padding:7px 9px}
.card-head{display:flex;flex-wrap:wrap;gap:4px 8px;align-items:center;justify-content:space-between;font-weight:700}
.card-sub{color:var(--muted);font-size:.8rem}
.card dl{display:grid;grid-template-columns:auto minmax(0,1fr);gap:0 10px;margin:4px 0 0;font-size:.85rem}
.card dt{color:var(--muted)} .card dd{margin:0;font-variant-numeric:tabular-nums}
.conflict-note{margin:6px 0 0;color:var(--muted);font-size:.8rem}
.stage{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:14px;margin-top:10px}
.triple{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-top:10px}
@media (max-width:860px){.stage,.triple{grid-template-columns:minmax(0,1fr)}}
svg.board{width:100%;max-height:54vh;display:block}
.triple svg.board{max-height:42vh}
.cell-free{fill:var(--free);stroke:var(--grid)} .cell-wall{fill:var(--wall)}
.agent{transition:transform .14s linear} .no-anim .agent{transition:none}
@media (prefers-reduced-motion:reduce){.agent{transition:none}}
.banner{font-size:1.05rem;font-weight:700;padding:8px 10px;border-radius:6px;border-left:6px solid var(--line);background:color-mix(in srgb,var(--fg) 5%,transparent)}
.banner.il{border-color:var(--il)} .banner.cbs{border-color:var(--cbs)} .banner.stall{border-color:var(--stall)} .banner.ok{border-color:var(--ok)} .banner.bad{border-color:var(--bad)}
.steps{display:flex;flex-wrap:wrap;gap:4px;font-size:.8rem;margin:8px 0 0;list-style:none;padding:0}
.steps li{padding:1px 8px;border-radius:999px;border:1px solid var(--line);color:var(--muted)}
.steps li.on{background:var(--fg);color:var(--bg);border-color:var(--fg);font-weight:600}
.evlist{list-style:none;margin:0;padding:0;font-size:.88rem}
.evlist button{width:100%;text-align:left;border:0;background:none;padding:2px 4px;border-radius:4px}
.evlist button.now{background:color-mix(in srgb,var(--il) 16%,transparent);font-weight:600}
.controls{margin-top:10px}
.tnum{font-family:var(--mono);font-variant-numeric:tabular-nums}
input[type=range]{flex:1;min-width:140px;width:100%}
.tl-row{display:grid;grid-template-columns:110px minmax(0,1fr);gap:8px;align-items:center;margin-top:4px;font-size:.8rem;color:var(--muted)}
@media (max-width:560px){.tl-row{grid-template-columns:minmax(0,1fr)}}
svg.tl{width:100%;height:26px;display:block;cursor:pointer;background:var(--free);border-radius:3px}
.legend{display:flex;flex-wrap:wrap;gap:2px 14px;font-size:.78rem;color:var(--muted);margin-top:6px}
.sw{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:4px;vertical-align:-1px}
.mini h3{font-size:.95rem;margin:0 0 4px}
.mini p{margin:4px 0 0;font-size:.85rem}
.mini .banner{font-size:.85rem;padding:4px 8px;border-left-width:4px;margin-top:6px}
details.fold{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px 12px}
details.fold>summary{cursor:pointer;font-weight:600;font-size:.9rem}
.fold-body{margin-top:8px;display:grid;grid-template-columns:minmax(0,1fr);gap:10px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.84rem}
th,td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:500}
.figures td{white-space:nowrap}
.evidence td{overflow-wrap:anywhere;font-family:var(--mono);font-size:.76rem}
.note{margin:0;padding-left:1.1em;font-size:.84rem} .note li{margin:2px 0}
.call{border:1px solid var(--line);border-radius:6px;padding:8px;font-size:.84rem}
.call dl{display:grid;grid-template-columns:auto minmax(0,1fr);gap:1px 10px;margin:4px 0 0}
.call dt{color:var(--muted)} .call dd{margin:0;overflow-wrap:anywhere}
.none{color:var(--muted);font-style:italic}
.notes{color:var(--muted);font-size:.84rem;margin:0;padding-left:1.1em}
#theme{font-size:.8rem;padding:3px 8px}
</style></head><body><main>
<header class="top">
  <div><h1>로봇 경로 복구 시연</h1>
  <div class="sub">로봇들이 같은 칸을 두고 경합해 멈추면, 이를 감지하고 안전한 경로를 다시 계산합니다.</div></div>
  <button id="theme" type="button">화면 테마</button>
</header>
<section aria-label="실행 결과 요약">
  <div class="cards" id="cards"></div>
  <p class="conflict-note">실제 충돌은 두 로봇이 같은 칸에 있는 경우입니다. 이동 시도 충돌(vertex_conflicts)은 같은 칸으로 가려다 이동이 취소된 사건 수로, 실제 충돌이 0이어도 발생할 수 있습니다.</p>
</section>
<section class="panel">
  <div class="row" role="group" aria-label="보기 방식">
    <button id="modeSingle" type="button">한 실행 보기</button>
    <button id="modeTriple" type="button">3개 동시 보기</button>
  </div>
  <div id="singleView" class="stage">
    <svg id="board" class="board" role="img" aria-label="로봇 위치 지도"></svg>
    <div>
      <div class="banner" id="status" role="status"></div>
      <ol class="steps" id="steps" aria-label="현재 단계"></ol>
      <h2 style="margin-top:10px">이벤트 (누르면 이동)</h2>
      <ul class="evlist" id="events"></ul>
    </div>
  </div>
  <div id="tripleView" class="triple" hidden></div>
  <div class="row controls">
    <button id="play" type="button">재생</button>
    <button id="first" type="button" aria-label="처음으로">|&lt;</button>
    <button id="prev" type="button" aria-label="이전 step">&lt;</button>
    <button id="next" type="button" aria-label="다음 step">&gt;</button>
    <button id="last" type="button" aria-label="마지막으로">&gt;|</button>
    <button id="prevEv" type="button">이전 이벤트</button>
    <button id="nextEv" type="button">다음 이벤트</button>
    <label for="speed">속도</label>
    <select id="speed"><option value="700">0.5x</option><option value="350" selected>1x</option><option value="150">2x</option><option value="60">5x</option><option value="20">20x</option></select>
    <label><input type="checkbox" id="skip"> 긴 정지 구간 건너뛰기</label>
  </div>
  <div class="row" style="margin-top:6px">
    <span class="tnum" id="tlabel"></span>
    <input id="slider" type="range" min="0" step="1" aria-label="시점 선택">
  </div>
  <div id="timelines"></div>
  <div class="legend">
    <span><span class="sw" style="background:var(--il)"></span>IL 이동</span>
    <span><span class="sw" style="background:var(--cbs)"></span>CBS 경로 이동</span>
    <span><span class="sw" style="background:var(--stall)"></span>전원 정지</span>
    <span>세로선: 정체 감지(굵게), IL 복귀(얇게)</span>
    <span>원 = 로봇, 점선 = 목표</span>
  </div>
</section>
<details class="fold" id="detailsBox">
  <summary>상세 정보 (개발자용)</summary>
  <div class="fold-body">
    <div class="scroll"><table id="summary" class="figures"></table></div>
    <ul class="note" id="caveats"></ul>
    <div class="scroll"><table id="agents"></table></div>
    <div id="settings"></div>
    <div id="calls" class="fold-body" style="margin-top:0"></div>
  </div>
</details>
<details class="fold" id="evidenceBox">
  <summary>실험 근거 (프로젝트 기준 경로와 해시)</summary>
  <div class="fold-body">
    <div class="sub" id="evidenceHead"></div>
    <div class="scroll"><table id="evidence" class="evidence"></table></div>
  </div>
</details>
<footer><ul class="notes" id="footnotes"></ul></footer>
</main>
<script type="application/json" id="panel-data">__DATA__</script>
<script>
(function(){
const D=JSON.parse(document.getElementById('panel-data').textContent);
const NONE='기록 없음';
const palette=['#d1495b','#22857a','#c9851c','#5465ff','#8f5db7','#3c9d4a','#c2569b','#6b7f2a'];
const color=i=>palette[i%palette.length];
const $=id=>document.getElementById(id);
const svgNS='http://www.w3.org/2000/svg';
const el=(n,a)=>{const e=document.createElementNS(svgNS,n);for(const k in a)e.setAttribute(k,a[k]);return e;};
const H=D.grid.length,W=D.grid[0].length,C=48;
const txt=(v,f)=>(v===null||v===undefined)?NONE:(f?f(v):String(v));
const ms=v=>`${Number(v).toFixed(1)} ms`;
const S={mode:'single',run:D.runs.length-1,t:0,timer:null};
const cur=()=>D.runs[S.run];
const maxT=()=>S.mode==='single'?cur().steps:Math.max(...D.runs.map(r=>r.steps));
const active=()=>S.mode==='single'?[cur()]:D.runs;

// short notes under the page, full notes in the details box
const dev=D.runs.find(r=>r.method.endsWith('REARM'));
const C_=D.comparison;
const foot=['저장된 Development 사례이며 최종 운영 정책이 아닙니다.',
 C_.same_first_cbs_path?'H3와 H3_REARM은 실행 조건이 기록되지 않아 엄밀한 성능 비교가 아닙니다.':'H3와 H3_REARM은 첫 CBS 경로가 달라 엄밀한 성능 비교가 아닙니다.'];
foot.forEach(t=>{const li=document.createElement('li');li.textContent=t;$('footnotes').appendChild(li);});
function socNote(){const base='실제 SoC(총 도착 시간)는 각 로봇이 목표에 도착해 끝까지 머물기 시작한 시점의 합입니다.';
 const r=D.runs.find(x=>x.soc!==null);if(!r)return base;
 const first=D.agents.reduce((sum,_,i)=>sum+r.at_goal.findIndex(row=>row[i]),0);
 return first===r.soc?`${base} ${r.label}에서는 목표 최초 도착 시점의 합과도 같습니다.`:base;}
const cav=[
 '저장된 실행을 재생하는 화면입니다. 이 화면은 CBS나 IL을 다시 실행하지 않습니다.',
 `${dev.dev_lines.join(', ')}은(는) Development 설정이며 합의된 운영 정책이 아닙니다.`,
 `PYTHONHASHSEED: 이 실행들의 기록에 ${C_.hash_seed_recorded?'값이 있습니다':'값이 없습니다 (재현성 문서상 미설정)'}. `
  +`H3와 H3_REARM의 첫 Trigger 상태 S_t는 ${C_.same_first_s_t?'같지만':'다르고'}, 첫 CBS 경로는 ${C_.same_first_cbs_path?'같습니다':'다릅니다'}. `
  +(C_.same_first_cbs_path?'그래도 seed가 기록되지 않아 통제된 인과 비교로 보지 마세요.':'따라서 이후 상태도 달라지며, 두 실행은 통제된 인과 비교가 아닙니다. "재Trigger 여부만 다른 실험"으로 해석하지 마세요.'),
 socNote(),
 '이동 시도 충돌은 step마다 경합이 생긴 칸 수의 합이며, 이동 취소로 이어진 연쇄 경합도 더합니다.',
 `프로젝트 commit ${(dev.project_commit||'').slice(0,7)}${dev.project_dirty?' (dirty)':''} 상태의 1회 실행이므로 성능 비교 근거가 아닙니다.`];
cav.forEach(t=>{const li=document.createElement('li');li.textContent=t;$('caveats').appendChild(li);});

// summary cards double as the run selector
const finishStep=r=>r.termination==='SUCCESS'?String(r.steps):`미완료 (${r.steps} step에서 종료)`;
D.runs.forEach((r,i)=>{const b=document.createElement('button');b.type='button';b.className='card';
 const head=document.createElement('div');head.className='card-head';const name=document.createElement('span');name.textContent=r.label;
 const chip=document.createElement('span');chip.className='chip '+(r.termination==='SUCCESS'?'ok':'bad');chip.textContent=r.termination;head.append(name,chip);
 const sub=document.createElement('div');sub.className='card-sub';sub.textContent=r.desc;
 const dl=document.createElement('dl');
 [['완료 step',finishStep(r)],['실제 SoC (총 도착 시간)',r.soc===null?'미완료':String(r.soc)],['목표 도달',`${r.agents_at_goal_final}/${D.agents.length}`],['CBS 호출',`${r.calls.length}회`],
  ['실제 충돌 (같은 칸)',`${r.overlap_steps.length} step`],['이동 시도 충돌 (이동 취소)',r.vertex_conflicts===null?NONE:`${r.vertex_conflicts}건`]]
  .forEach(([k,v])=>{const dt=document.createElement('dt');dt.textContent=k;const dd=document.createElement('dd');dd.textContent=v;dl.append(dt,dd);});
 b.append(head,sub,dl);b.onclick=()=>{if(S.mode!=='single')setMode('single');setRun(i);};$('cards').appendChild(b);});

// developer figures
const ROWS=[['실행',r=>r.label],['종료',r=>r.termination],['Step',r=>`${r.steps} / ${r.max_steps}`],
 ['실제 SoC',r=>r.soc===null?'해당 없음 (미완료)':r.soc],['makespan',r=>r.makespan===null?'해당 없음 (미완료)':r.makespan],
 ['CBS latency 합',r=>r.calls.length?txt(r.cbs_latency_total_ms,ms):'CBS 호출 없음'],
 ['Monitor Trigger',r=>r.monitor.length?r.monitor.map(m=>`t=${m.t}`).join(', '):NONE],
 ['IL 복귀',r=>{const v=r.events.filter(e=>e.kind==='RESUME').map(e=>`t=${e.t}`);return v.length?v.join(', '):'해당 없음';}],
 ['실제 위치 겹침',r=>`${r.overlap_steps.length} step`],
 ['이동 시도 충돌 (vertex / edge)',r=>`${txt(r.vertex_conflicts)} / ${txt(r.edge_conflicts)}`]];
(function(){const t=$('summary');const head=document.createElement('tr');
 ROWS.forEach(([n])=>{const th=document.createElement('th');th.textContent=n;th.scope='col';head.appendChild(th);});t.appendChild(head);
 D.runs.forEach(r=>{const tr=document.createElement('tr');ROWS.forEach(([,f])=>{const td=document.createElement('td');td.textContent=f(r);tr.appendChild(td);});t.appendChild(tr);});})();

// evidence
$('evidenceHead').textContent=`저장 배치 ${D.evidence.batch_id} · 시나리오 ${D.scenario}`;
(function(){const t=$('evidence');const h=document.createElement('tr');['파일','SHA-256'].forEach(n=>{const th=document.createElement('th');th.textContent=n;h.appendChild(th);});t.appendChild(h);
 D.evidence.files.forEach(f=>{const tr=document.createElement('tr');[f.path,f.sha256].forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.appendChild(td);});t.appendChild(tr);});})();

// boards
function makeBoard(svg){svg.setAttribute('viewBox',`0 0 ${W*C} ${H*C}`);
 for(let r=0;r<H;r++)for(let c=0;c<W;c++)svg.appendChild(el('rect',{x:c*C,y:r*C,width:C,height:C,class:D.grid[r][c]?'cell-wall':'cell-free'}));
 D.goals.forEach(([r,c],i)=>{svg.appendChild(el('rect',{x:c*C+4,y:r*C+4,width:C-8,height:C-8,fill:'none',stroke:color(i),'stroke-width':2.5,'stroke-dasharray':'5 4'}));
  const t=el('text',{x:c*C+8,y:r*C+C-8,'font-size':12,fill:color(i)});t.textContent='G'+D.agents[i];svg.appendChild(t);});
 return D.agents.map((id,i)=>{const g=el('g',{class:'agent'});g.appendChild(el('circle',{cx:C/2,cy:C/2,r:C*.34,fill:color(i),stroke:'#000','stroke-opacity':.25}));
  const t=el('text',{x:C/2,y:C/2+6,'text-anchor':'middle','font-size':17,fill:'#fff','font-weight':700});t.textContent=id;g.appendChild(t);svg.appendChild(g);return g;});}
function setBoard(glyphs,run,t){run.positions[Math.min(t,run.steps)].forEach(([r,c],i)=>glyphs[i].setAttribute('transform',`translate(${c*C},${r*C})`));}
const mainGlyphs=makeBoard($('board'));

// timelines: merged segments, so a 160-step run costs no more than a short one
function makeTimeline(run,scale){const svg=el('svg',{class:'tl',viewBox:`0 0 ${scale} 28`,preserveAspectRatio:'none',role:'img','aria-label':`${run.label} 실행 구간`});
 run.segments.forEach(s=>svg.appendChild(el('rect',{x:s.from-1,y:0,width:s.to-s.from+1,height:18,fill:s.source==='CBS'?'var(--cbs)':'var(--il)'})));
 run.stalls.forEach(s=>svg.appendChild(el('rect',{x:s.from-1,y:20,width:s.to-s.from+1,height:8,fill:'var(--stall)'})));
 run.events.filter(e=>e.kind==='TRIGGER'||e.kind==='MONITOR').forEach(e=>svg.appendChild(el('line',{x1:e.t,x2:e.t,y1:0,y2:28,stroke:'var(--mark)','stroke-width':3,'vector-effect':'non-scaling-stroke'})));
 run.events.filter(e=>e.kind==='RESUME').forEach(e=>svg.appendChild(el('line',{x1:e.t,x2:e.t,y1:0,y2:28,stroke:'var(--mark)','stroke-width':1,'vector-effect':'non-scaling-stroke'})));
 const curLine=el('line',{x1:0,x2:0,y1:0,y2:28,stroke:'var(--bad)','stroke-width':2,'vector-effect':'non-scaling-stroke'});svg.appendChild(curLine);
 svg.addEventListener('click',ev=>{const b=svg.getBoundingClientRect();stop();show(Math.round((ev.clientX-b.left)/b.width*scale));});
 return {svg,curLine};}
let timelines=[];
function buildTimelines(){const box=$('timelines');box.textContent='';timelines=[];
 const runs=active();const scale=Math.max(1,...runs.map(r=>r.steps));
 runs.forEach(r=>{const row=document.createElement('div');row.className='tl-row';const l=document.createElement('span');l.textContent=`${r.label} (${r.steps})`;
  const t=makeTimeline(r,scale);row.append(l,t.svg);box.appendChild(row);timelines.push(t);});}

// three boards
const mini=D.runs.map(r=>{const p=document.createElement('div');p.className='mini';const h=document.createElement('h3');h.textContent=r.label;
 const svg=el('svg',{class:'board',role:'img','aria-label':`${r.label} 로봇 위치 지도`});const glyphs=makeBoard(svg);const s=document.createElement('div');s.className='banner';const d=document.createElement('p');d.className='sub';
 p.append(h,svg,s,d);$('tripleView').appendChild(p);return {run:r,glyphs,s,d};});

// describing one moment in plain words
const isStall=(r,t)=>r.stalls.some(s=>t>=s.from&&t<=s.to);
function describe(r,t){const c=Math.min(t,r.steps);const goal=`목표 ${r.at_goal[c].filter(Boolean).length}/${D.agents.length}`;let cls='il',what;
 if(c===0)what='출발 전';
 else if(c===r.steps){cls=r.termination==='SUCCESS'?'ok':'bad';what=r.termination==='SUCCESS'?'완료: 전원 목표 도달':`종료: ${r.termination}, 목표 미도달`;}
 else if(r.exec_step[String(c)]){const x=r.exec_step[String(c)],call=r.calls[x[0]-1];cls='cbs';what=`CBS 복구 경로로 이동 (호출 ${x[0]}, ${x[1]}/${txt(call.exec_window)} step)`;}
 else if(isStall(r,c)){cls='stall';what='같은 칸 경합으로 전원 이동 취소';}
 else what='IL이 제안한 대로 이동';
 if(r.overlap_steps.includes(c))what+=' · 실제 위치 겹침';
 if(t>r.steps)what+=` (t=${r.steps}에서 끝난 상태)`;
 return {cls,text:`t=${c} · ${what} · ${goal}`};}
const STEP_NAMES=['IL 이동','정체 감지','CBS 개입','Path Guard','CBS 경로 이동','IL 복귀','종료'];
function phaseOn(r,t){const on=new Set();const evs=r.events.filter(e=>e.t===t);
 if(evs.some(e=>e.kind==='TRIGGER'||e.kind==='MONITOR'))on.add('정체 감지');
 if(evs.some(e=>e.kind==='CBS'))on.add('CBS 개입');
 if(evs.some(e=>e.kind==='GUARD'))on.add('Path Guard');
 if(evs.some(e=>e.kind==='RESUME'))on.add('IL 복귀');
 if(t===r.steps)on.add('종료');
 if(t>0){if(r.exec_step[String(t)])on.add('CBS 경로 이동');else if(!on.size)on.add('IL 이동');}
 return on;}
STEP_NAMES.forEach(n=>{const li=document.createElement('li');li.textContent=n;li.dataset.n=n;$('steps').appendChild(li);});

// per-agent table: rows are reused
(function(){const t=$('agents');const h=document.createElement('tr');['로봇','(row,col)','행동','결과','목표'].forEach(n=>{const th=document.createElement('th');th.textContent=n;h.appendChild(th);});t.appendChild(h);
 D.agents.forEach((id,i)=>{const tr=document.createElement('tr');for(let k=0;k<5;k++)tr.appendChild(document.createElement('td'));tr.children[0].textContent=id;tr.children[0].style.color=color(i);tr.children[0].style.fontWeight=700;tr.children[1].className='tnum';t.appendChild(tr);});})();
const RESULT={MOVE:'이동',WAIT:'대기',BLOCKED:'이동 취소'};
function fillAgents(r,t){const rows=$('agents').querySelectorAll('tr');D.agents.forEach((id,i)=>{const c=rows[i+1].children;const p=r.positions[t][i];
 c[1].textContent=`(${p[0]},${p[1]})`;c[2].textContent=t?r.actions[t-1][i]:'-';c[3].textContent=t?(RESULT[r.kinds[t-1][i]]||r.kinds[t-1][i]):'-';c[4].textContent=r.at_goal[t][i]?'도달':'아직';});}

// per-run panels
function addDl(parent,pairs){const dl=document.createElement('dl');pairs.forEach(([k,v])=>{const dt=document.createElement('dt');dt.textContent=k;const dd=document.createElement('dd');dd.textContent=v;if(v===NONE)dd.className='none';dl.append(dt,dd);});parent.appendChild(dl);}
const RECHECK={PASS:'통과 (패널 생성 시 저장 경로로 재실행한 값)',FAIL:'실패',NOT_RECORDED:'저장 경로 없음'};
function fillRunPanels(){const r=cur();const ev=$('events');ev.textContent='';
 const items=r.events.map(e=>({t:e.t,text:`t=${e.t}  ${e.label}${e.detail?` (${e.detail})`:''}`,o:e.kind==='END'?2:0}))
  .concat(r.stalls.filter(s=>s.to-s.from>=1).map(s=>({t:s.from,text:`t=${s.from}~${s.to}  전원 정지 (trajectory 계산값, Monitor 기록 아님)`,o:1})));
 items.sort((a,b)=>a.t-b.t||a.o-b.o);
 items.forEach(x=>{const li=document.createElement('li');const b=document.createElement('button');b.type='button';b.dataset.t=x.t;b.textContent=x.text;b.onclick=()=>{stop();show(x.t);};li.appendChild(b);ev.appendChild(li);});
 const st=$('settings');st.textContent='';const box=document.createElement('div');box.className='call';
 const title=document.createElement('strong');title.textContent=`${r.label}: `+(r.is_dev?'Development 설정 (합의 정책 아님)':'Development 설정 없음 (CBS 미사용)');box.appendChild(title);
 if(r.dev_lines.length){const p=document.createElement('div');p.textContent=r.dev_lines.join(' · ');box.appendChild(p);}
 if(!r.calls.length&&r.monitor.length){addDl(box,[['Monitor Trigger',`t=${r.monitor[0].t}`],['first_seen',txt(r.monitor[0].first_seen)],['period',txt(r.monitor[0].period)]]);}
 st.appendChild(box);
 const cs=$('calls');cs.textContent='';r.calls.forEach(c=>{const b=document.createElement('div');b.className='call';const t=document.createElement('strong');t.textContent=`CBS 호출 ${c.index} · Trigger t=${c.trigger}`;b.appendChild(t);
  addDl(b,[['status',c.status],['latency',txt(c.latency_ms,ms)],
   ['Monitor first_seen',txt(c.first_seen)],['Monitor period',txt(c.period)],
   ['수용 기록 (실행 당시)',`accepted=${c.accepted}${c.reason?` (${c.reason})`:''}`],
   ['Path Guard 단독 결과 (실행 당시)',NONE],
   ['Path Guard 사후 재검증',RECHECK[c.recheck.status]+(c.recheck.reason?`: ${c.recheck.reason}`:'')],
   ['계획 길이 / 실행 step',`${txt(c.planned_horizon)} / ${txt(c.exec_steps)}`],
   ['실행 종료 방식',txt(c.exec_end)],
   ['계획 위치 = 실제 위치',`${c.plan_match}/${c.plan_total} step`]]);cs.appendChild(b);});}

// render
function setBanner(node,d){node.className='banner '+d.cls;node.textContent=d.text;}
function show(t){S.t=Math.max(0,Math.min(maxT(),t));$('slider').value=S.t;
 $('tlabel').textContent=`t=${String(S.t).padStart(3,' ')} / ${maxT()}`;
 timelines.forEach(x=>{x.curLine.setAttribute('x1',S.t);x.curLine.setAttribute('x2',S.t);});
 if(S.mode==='single'){const r=cur();setBoard(mainGlyphs,r,S.t);setBanner($('status'),describe(r,S.t));fillAgents(r,S.t);
  const on=phaseOn(r,S.t);$('steps').querySelectorAll('li').forEach(li=>li.classList.toggle('on',on.has(li.dataset.n)));
  $('events').querySelectorAll('button').forEach(b=>b.classList.toggle('now',Number(b.dataset.t)===S.t));}
 else mini.forEach(m=>{setBoard(m.glyphs,m.run,S.t);setBanner(m.s,describe(m.run,S.t));
  const e=m.run.events.filter(x=>x.t===Math.min(S.t,m.run.steps)).map(x=>x.label);m.d.textContent=e.length?`이벤트: ${e.join(', ')}`:' ';});}
function eventTimes(){const set=new Set([0]);active().forEach(r=>{r.events.forEach(e=>set.add(e.t));r.stalls.forEach(s=>set.add(s.from));});return [...set].sort((a,b)=>a-b);}
function stop(){if(S.timer){clearInterval(S.timer);S.timer=null;}$('play').textContent='재생';document.body.classList.remove('no-anim');}
function tick(){if(S.t>=maxT()){stop();return;}let n=S.t+1;
 if($('skip').checked){const runs=active();
  const all=runs.every(r=>S.t>=r.steps||r.stalls.some(s=>S.t>=s.from&&S.t<s.to&&s.to-S.t>5));
  if(all){const later=runs.map(r=>{const s=r.stalls.find(x=>S.t>=x.from&&S.t<x.to);return s?s.to-3:r.steps;});n=Math.max(n,Math.min(...later));}}
 show(n);}
function play(){if(S.t>=maxT())show(0);stop();$('play').textContent='일시정지';document.body.classList.toggle('no-anim',Number($('speed').value)<100);S.timer=setInterval(tick,Number($('speed').value));}
function markCards(){$('cards').querySelectorAll('button.card').forEach((b,k)=>b.setAttribute('aria-pressed',S.mode==='single'&&k===S.run));}
function setMode(m){stop();S.mode=m;$('modeSingle').setAttribute('aria-pressed',m==='single');$('modeTriple').setAttribute('aria-pressed',m==='triple');
 $('singleView').hidden=m!=='single';$('tripleView').hidden=m!=='triple';markCards();
 $('slider').max=maxT();buildTimelines();show(S.t);}
function setRun(i){stop();S.run=i;markCards();fillRunPanels();$('slider').max=maxT();buildTimelines();show(Math.min(S.t,maxT()));}
$('modeSingle').onclick=()=>setMode('single');$('modeTriple').onclick=()=>setMode('triple');
$('play').onclick=()=>S.timer?stop():play();
$('first').onclick=()=>{stop();show(0);};$('last').onclick=()=>{stop();show(maxT());};
$('prev').onclick=()=>{stop();show(S.t-1);};$('next').onclick=()=>{stop();show(S.t+1);};
$('prevEv').onclick=()=>{stop();const v=eventTimes().filter(x=>x<S.t);show(v.length?v[v.length-1]:0);};
$('nextEv').onclick=()=>{stop();const v=eventTimes().filter(x=>x>S.t);show(v.length?v[0]:maxT());};
$('speed').onchange=()=>{if(S.timer)play();};
$('slider').oninput=()=>{stop();show(Number($('slider').value));};
document.addEventListener('keydown',e=>{if(/INPUT|SELECT|BUTTON|SUMMARY/.test(e.target.tagName)&&e.target.id!=='slider')return;
 if(e.key==='ArrowLeft'&&e.target.id!=='slider'){stop();show(S.t-1);}else if(e.key==='ArrowRight'&&e.target.id!=='slider'){stop();show(S.t+1);}});
$('theme').onclick=()=>{const dark=getComputedStyle(document.documentElement).colorScheme==='dark';document.documentElement.setAttribute('data-theme',dark?'light':'dark');};
setRun(S.run);setMode('single');
window.__panel={show,setMode,setRun,state:S};
})();
</script></body></html>
"""


def render_panel_html(data: Dict) -> str:
    public = {k: v for k, v in data.items() if k != "inputs_sha256"}   # absolute paths stay out of the page
    payload = json.dumps(public, ensure_ascii=False).replace("</", "<\\/")
    return TEMPLATE.replace("__DATA__", payload)


def build(batch_dir: Path, out_dir: Path, *, scenario_key: str = DEFAULT_SCENARIO_KEY, command: Optional[str] = None) -> Path:
    """Check every input, then write the page and manifest into a NEW ``out_dir``."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(str(out_dir))
    data = build_panel_data(batch_dir, scenario_key)
    html_text = render_panel_html(data)
    changed = [p for p, sha in data["inputs_sha256"].items() if sha256_file(p) != sha]
    if changed:
        raise BuildError(f"input changed during the build: {changed}")
    out_dir.mkdir(parents=True, exist_ok=False)
    page = out_dir / "demo_panel.html"
    with page.open("x", encoding="utf-8", newline="\n") as fh:
        fh.write(html_text)
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "command": command,
        "note": "Built only from saved run files. No solver or IL was run; settings are Development settings.",
        "batch_id": data["batch_id"],
        "scenario": data["scenario"],
        "comparison": data["comparison"],
        "runs": [{"run_id": r["run_id"], "method": r["method"], "termination": r["termination"], "steps": r["steps"],
                  "real_soc": r["soc"], "overlap_steps": len(r["overlap_steps"]),
                  "vertex_conflicts": r["vertex_conflicts"],
                  "post_hoc_path_guard": [c["recheck"]["status"] for c in r["calls"]]} for r in data["runs"]],
        "inputs_sha256": {f["path"]: f["sha256"] for f in data["evidence"]["files"]},
        "output_sha256": {page.name: sha256_file(page)},
    }
    with (out_dir / "demo_manifest.json").open("x", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    return page


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH)
    ap.add_argument("--scenario-key", default=DEFAULT_SCENARIO_KEY)
    ap.add_argument("--out-dir", type=Path, default=None, help="new directory (default: outputs/videos/demo_panel_<time>)")
    args = ap.parse_args(argv)
    out_dir = args.out_dir or DEFAULT_OUT_ROOT / f"demo_panel_{datetime.now():%Y%m%d_%H%M%S}"
    command = " ".join([Path(sys.executable).name, "-m", "scripts.build_demo_page", *argv])
    try:
        page = build(args.batch_dir, out_dir, scenario_key=args.scenario_key, command=command)
    except FileExistsError:
        print(f"ERROR: output directory already exists, refusing to overwrite: {out_dir}")
        return 2
    except (BuildError, OSError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"wrote {page}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
