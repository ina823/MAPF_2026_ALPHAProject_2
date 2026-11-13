"""DEV: real Selective Hybrid Runner E2E -- real IL + real atb033 CBS + real Simulator.

    NavHintPolicy -> MAPFStepSimulator -> ShadowMonitor(DeadlockMonitor) -> TRIGGER
    -> SelectivePlanner (L0/L1 = real SubsetCbsPlanner, G = real BoundedGlobalCbsPlanner)
    -> SelectivePathValidator -> unchanged Global Hybrid Runner CBS_EXEC (expected/actual
    over ALL agents) -> IL_RESUME / SAFE_STOP.

Nothing is mocked in the scenario runs. The failure-injection runs (F*) inject
exactly one fault each (named in the record) on top of the same real chain.

DEV ONLY -- NOT agreed project policy:
  * TriggerBudget DEV_BUDGET and DEV_RESERVE_MS below (P4 numbers are PENDING);
  * h_exec = 3, DevResumeIL, DevSingleCbsCall (same as scripts/run_hybrid_resume_dev.py);
  * acknowledge_unverified_outside_policy=True (P2/P3 outside WAIT is unverified).
Scenarios are built in memory from the ORIGINAL map/scenario files, which are only read.

Wall-clock: the solver subprocess is killed at its timeout, but the measured call
can exceed timeout_ms (kill + pipe drain, ~0.1 s measured). Overruns are recorded
per call; this is NOT a real-time guarantee.

Every run writes only to a NEW directory:
    <out-root>/<YYYYmmdd_HHMMSS>_selective_real_e2e/<run>/result.json (+ cbs work dirs)
    <out-root>/<YYYYmmdd_HHMMSS>_selective_real_e2e/summary.json

Usage (repository root; CBS calls run sequentially in this one process):
    set PYTHONHASHSEED=0
    python -m scripts.run_selective_real_e2e_dev --solver-root <atb033>/centralized/cbs
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_corridor_hybrid_e2e import _jsonable
from scripts.run_hybrid_resume_dev import dev_resume_config
from scripts.run_il import DEFAULT_CKPT, load_scenario
from src.common import spec
from src.common.deadlock_monitor import DeadlockMonitor
from src.hybrid import subset_cbs_adapter as adapter_mod
from src.hybrid.cbs_runtime_bridge import CbsRuntimePlanner, PathGuardValidator
from src.hybrid.global_hybrid import run_global_hybrid_episode
from src.hybrid.selective_planner import SelectiveConfig, SelectivePathValidator, SelectivePlanner
from src.hybrid.selective_safety import TriggerBudget
from src.hybrid.selective_shadow import ShadowMonitor
from src.hybrid.subset_cbs_adapter import BoundedGlobalCbsPlanner, SubsetCbsPlanner
from src.simulator.mapf_step_simulator import MAPFStepSimulator

EXPECTED_SOLVER_COMMIT = "89f69e1"
MAX_STEPS = 160
H_EXEC = 3                                   # DEV_ONLY (= run_hybrid_resume_dev default)
DEV_RESERVE_MS = 500.0                       # DEV_ONLY (> measured ~0.1 s kill overrun)
DEV_BUDGET = TriggerBudget(total_time_ms=30_000.0, max_attempts=3,
                           level_timeout_ms={"L0": 10_000.0, "L1": 10_000.0, "G": 20_000.0})   # DEV_ONLY
WAIT = spec.WAIT if hasattr(spec, "WAIT") else 4
CONTROLLED = PROJECT_ROOT / "scenarios" / "controlled"
CORRIDOR = CONTROLLED / "scenario_corridor_bay_n2.json"
BOTTLENECK = CONTROLLED / "scenario_bottleneck_n4.json"


# -- scenarios (in memory; originals are only read) ------------------------------------------

def scenario(name):
    """-> (grid, starts, goals, description). Coordinates (row, col)."""
    if name in ("S1", "S2", "S1B"):
        grid, starts, goals = load_scenario(CORRIDOR)
        grid = np.array(grid, dtype=np.int64, copy=True)
        if name == "S1":
            # The original corridor map has NO free cell off the 0/1 route: every free cell
            # is on row 2 (all of it is traversed by 0 and 1) or the single passing bay (1,5)
            # (that case is S2). A corridor cell is therefore the spec-conform choice; (2,8)
            # is used. Agent 2 then physically blocks 0/1 -- the result is reported as is.
            cell = (2, 8)
            desc = "corridor original map + agent 2 start=goal=(2,8) on the corridor"
        elif name == "S2":
            cell = (1, 5)
            desc = "corridor original map + agent 2 start=goal=(1,5) holding the only bay"
        else:
            # SUPPLEMENTAL, deviates from the spec: a COPY of the map with one extra
            # dead-end cell (3,1) below the corridor, so agent 2 can stand off the route.
            grid[3, 1] = 0
            cell = (3, 1)
            desc = "SUPPLEMENTAL map copy: corridor + extra dead-end cell (3,1); agent 2 start=goal=(3,1)"
        starts = {**starts, 2: cell}
        goals = {**goals, 2: cell}
        return grid, starts, goals, desc
    if name == "S3":
        grid, starts, goals = load_scenario(BOTTLENECK)
        return np.array(grid, dtype=np.int64, copy=True), starts, goals, "bottleneck_n4 original (4 agents)"
    if name == "CORRIDOR":
        grid, starts, goals = load_scenario(CORRIDOR)
        return np.array(grid, dtype=np.int64, copy=True), starts, goals, "corridor_bay_n2 original (2 agents)"
    raise ValueError(name)


# -- helpers ----------------------------------------------------------------------------------

def _git(path, *args):
    try:
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def preflight(solver_root: Path):
    problems = []
    if not (solver_root / "cbs.py").is_file():
        problems.append(f"cbs.py not found under {solver_root}")
    head = _git(solver_root, "rev-parse", "HEAD")
    if not head or not head.startswith(EXPECTED_SOLVER_COMMIT):
        problems.append(f"solver commit {head!r} != {EXPECTED_SOLVER_COMMIT}")
    if os.environ.get("PYTHONHASHSEED") != "0":
        problems.append(f"PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')!r}, need '0'")
    return {"solver_root": str(solver_root), "solver_commit": head,
            "solver_dirty": bool(_git(solver_root, "status", "--porcelain")),
            "project_commit": _git(PROJECT_ROOT, "rev-parse", "HEAD"),
            "project_dirty_files": (_git(PROJECT_ROOT, "status", "--porcelain") or "").splitlines(),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"), "python": sys.executable,
            "problems": problems}


def perturbing(inject_t, agent_id, new_pos):
    """FAILURE INJECTION: after the step reaching inject_t, force agent_id onto new_pos."""
    class _Perturb(MAPFStepSimulator):
        def step(self, actions):
            obs, done, info = super().step(actions)
            if info["t"] == inject_t:
                self._positions[agent_id] = tuple(new_pos)
            return obs, done, info
    return _Perturb


def safety_scan(result):
    """Vertex overlap / edge swap / unexpected outside movement over the WHOLE trajectory."""
    traj = result["trajectory"]
    vertex, swap = [], []
    for t, s in enumerate(traj):
        if len(set(s.values())) != len(s):
            vertex.append(t)
        if t:
            p = traj[t - 1]
            for a in s:
                for b in s:
                    if a < b and s[a] == p[b] and s[b] == p[a] and s[a] != p[a]:
                        swap.append((t, a, b))
    return {"vertex_overlap_steps": vertex, "edge_swaps": swap}


def run_selective(name, solver_root, run_dir, *, budget=DEV_BUDGET, ladder=("L0", "L1", "G"),
                  subset_reserve=DEV_RESERVE_MS, global_reserve=DEV_RESERVE_MS, simulator_cls=MAPFStepSimulator,
                  runtime_wrapper=None):
    from src.il.policy import NavHintPolicy
    grid, starts, goals, desc = scenario(name)
    s = SubsetCbsPlanner(solver_root=solver_root, work_root=run_dir / "subset_cbs", overhead_reserve_ms=subset_reserve)
    g = BoundedGlobalCbsPlanner(solver_root=solver_root, work_root=run_dir / "global_cbs",
                                overhead_reserve_ms=global_reserve)
    mon = ShadowMonitor(DeadlockMonitor())
    cfg = SelectiveConfig(ladder=ladder, budget=budget, acknowledge_unverified_outside_policy=True)
    planner = SelectivePlanner(g, mon, cfg, subset_planner=s, allow_real_subset_cbs=True)
    validator = SelectivePathValidator(planner)
    ctx = patch.object(adapter_mod, "run_cbs_runtime", runtime_wrapper) if runtime_wrapper else _null()
    t0 = time.perf_counter()
    with ctx:
        result = run_global_hybrid_episode(NavHintPolicy(DEFAULT_CKPT), grid, starts, goals, MAX_STEPS,
                                           planner=planner, path_validator=validator,
                                           config=dev_resume_config(H_EXEC), monitor=mon,
                                           simulator_cls=simulator_cls)
    wall = time.perf_counter() - t0
    return dict(name=name, desc=desc, grid=grid, starts=starts, goals=goals, result=result, planner=planner,
                validator=validator, subset=s, glob=g, wall_sec=wall)


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def _input_yaml(call):
    wd = call.get("work_dir")
    if not wd:
        return None
    p = Path(wd) / "outputs/paths/replan/input.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.is_file() else None


def record(run, budget=DEV_BUDGET):
    """Everything section 7 asks for, from the run's own objects (no recomputation of results)."""
    res, planner, grid = run["result"], run["planner"], run["grid"]
    n = len(run["starts"])
    recs = []
    for r in planner.records:
        recs.append({
            "call_index": r.call_index, "mode": r.mode, "trigger_t": r.trigger_t,
            "s_t_minus_1": r.s_t_minus_1, "intended": r.intended, "blocked": r.blocked, "s_t": r.s_t,
            "analysis_error": r.analysis_error,
            "wfg_groups": r.wfg.groups if r.wfg else None,
            "wfg_edges": [(e.src, e.dst, e.kind, e.cell) for e in r.wfg.edges] if r.wfg else None,
            "levels": [{"level": lv.level, "group": lv.group, "viable": lv.viable, "skip_reason": lv.skip_reason,
                        "goal_blocked": lv.goal_blocked, "unreachable": lv.unreachable} for lv in r.levels],
            "attempts": [{"level": a.level, "group": a.group, "planner_kind": a.planner_kind,
                          "solver_invoked": a.solver_invoked, "raw_status": a.raw_status, "status": a.status,
                          "timeout_ms": a.timeout_ms, "latency_ms": a.latency_ms,
                          "budget_overrun_ms": a.budget_overrun_ms, "full_check_ok": a.full_check_ok,
                          "full_check_reason": a.full_check_reason} for a in r.attempts],
            "attempt_latency_sum_ms": (sum(a.latency_ms for a in r.attempts)
                                       if r.attempts and all(a.latency_ms is not None for a in r.attempts) else None),
            "outcome": r.outcome, "returned_status": r.returned_status, "executed_group": r.executed_group,
            "outside_held": r.outside_held, "goal_status": r.goal_status,
            "real_subset_cbs_called": r.real_subset_cbs_called, "real_global_cbs_called": r.real_global_cbs_called,
        })
    adapter_calls = []
    for kind, p in (("subset", run["subset"]), ("global", run["glob"])):
        for c in p.calls:
            inp = _input_yaml(c)
            adapter_calls.append({
                "planner": kind, "call_index": c["call_index"], "called": c["called"], "status": c.get("status"),
                "error": (c.get("error") or "")[:300] or None, "timeout_ms": c["timeout_ms"],
                "solver_timeout_sec": c["solver_timeout_sec"], "latency_ms": c.get("latency_sec", 0) * 1000.0,
                "solver_latency_ms": None if c.get("solver_latency_sec") is None else c["solver_latency_sec"] * 1000.0,
                "budget_overrun_ms": c.get("budget_overrun_ms"), "group": c.get("group"),
                "outside": c.get("outside"), "paths": c.get("paths"), "work_dir": c.get("work_dir"),
                "solver_input_agents": [(a["name"], a["start"], a["goal"]) for a in inp["agents"]] if inp else None,
                "solver_input_obstacles_xy": inp["map"]["obstacles"] if inp else None,
                "pythonhashseed_env": c.get("pythonhashseed_env"),
            })
    cbs_idx = [i for i, src in enumerate(res["action_sources"]) if src == "CBS"]
    exec_window = []
    for i in cbs_idx:
        before, after, acts = res["trajectory"][i], res["trajectory"][i + 1], res["actions"][i]
        exec_window.append({"t_after": i + 1, "actions": acts,
                            "moved": sorted(a for a in after if after[a] != before[a])})
    return {
        "scenario": run["name"], "description": run["desc"], "n_agents": n,
        "starts": run["starts"], "goals": run["goals"], "grid_rows": ["".join(str(int(v)) for v in row) for row in grid],
        "budget_dev_only": {"total_time_ms": budget.total_time_ms, "max_attempts": budget.max_attempts,
                            "level_timeout_ms": dict(budget.level_timeout_ms)},
        "selective_records": recs, "adapter_calls": adapter_calls,
        "runner_cbs_calls": res["cbs_calls"], "validator_calls": run["validator"].calls,
        "cbs_step_indices": cbs_idx, "exec_window": exec_window, "cbs_exec_trace": res["cbs_exec_trace"],
        "exec_mismatch": res["exec_mismatch"], "safety_scan": safety_scan(res),
        "transitions": res["transitions"], "termination": res["termination"], "stop_reason": res["stop_reason"],
        "integrity_abort": res["integrity_abort"], "all_at_goal": res["all_at_goal"], "steps": res["steps"],
        "final_positions": res["final_positions"], "action_sources": res["action_sources"],
        "trajectory": res["trajectory"], "actions": res["actions"], "wall_sec": run["wall_sec"],
    }


# -- verdicts ---------------------------------------------------------------------------------

def _intended(prev, acts):
    return {a: (prev[a][0] + spec.ACTION_DELTA[acts[a]][0], prev[a][1] + spec.ACTION_DELTA[acts[a]][1]) for a in prev}


def _first_call_starts_are(rec, attempt, s_t):
    """The first real CBS call's solver input (agentK = K-th sorted group id, start [x, y]) == S_t."""
    if not attempt:
        return False
    kind = "global" if attempt["level"] == "G" else "subset"
    call = next((c for c in rec["adapter_calls"] if c["planner"] == kind and c["call_index"] == 1), None)
    if not call or not call["solver_input_agents"]:
        return False
    ids = sorted(attempt["group"])
    got = [tuple(st) for _, st, _ in call["solver_input_agents"]]
    return len(got) == len(ids) and all(got[i] == (s_t[a][1], s_t[a][0]) for i, a in enumerate(ids))


def partial_checks(rec):
    """S1 items 1-12. Every item must hold on the REAL Runner for S1 PASS."""
    res_tr, res_act = rec["trajectory"], rec["actions"]
    srs = rec["selective_records"]
    sr = srs[0] if srs else {}
    t = sr.get("trigger_t")
    at = sr.get("attempts") or []
    a0 = at[0] if at else {}
    sub = [c for c in rec["adapter_calls"] if c["planner"] == "subset"]
    c0 = sub[0] if sub else {}
    outside = tuple(sr.get("outside_held") or ())
    group = tuple(sr.get("executed_group") or ())
    s_t = res_tr[t] if t is not None else None
    xy_obs = [tuple(o) for o in (c0.get("solver_input_obstacles_xy") or [])]
    win = rec["exec_window"]
    tr = rec["transitions"]
    i_end = next((i for i, x in enumerate(tr) if x["to"] == "IL_RESUME"), None)
    checks = [
        ("1_trigger_in_real_IL", t is not None and all(s == "IL" for s in rec["action_sources"][:t])
         and any(x["to"] == "TRIGGER" and x["t"] == t for x in tr)),
        ("2_wfg_uses_s_t_minus_1_and_intended", t is not None and sr["s_t_minus_1"] == res_tr[t - 1]
         and sr["intended"] == _intended(res_tr[t - 1], res_act[t - 1])),
        ("3_cbs_start_is_s_t", s_t is not None and sr["s_t"] == s_t and rec["runner_cbs_calls"][0]["s_t"] == s_t
         and _first_call_starts_are(rec, a0, s_t)),
        ("4_real_subset_cbs_at_L0", a0.get("level") == "L0" and a0.get("planner_kind") == "REAL"
         and a0.get("solver_invoked") is True and c0.get("called") is True),
        ("5_real_subset_cbs_called_flag", sr.get("real_subset_cbs_called") is True and a0.get("planner_kind") == "REAL"),
        ("6_group_smaller_than_all", 0 < len(group) < rec["n_agents"]
         and sr.get("outcome") in ("EXECUTE:L0", "EXECUTE:L1")),
        ("7_outside_is_static_obstacle", bool(outside) and all((s_t[a][1], s_t[a][0]) in xy_obs for a in outside)
         and c0.get("solver_input_agents") is not None and len(c0["solver_input_agents"]) == len(group)),
        ("8_full_compose_path_guard", a0.get("full_check_ok") is True
         and [(k, tuple(v)) for k, v in rec["validator_calls"]] == [("SELECTIVE", (True, ""))]
         and rec["runner_cbs_calls"][0]["accepted"] is True),
        ("9_outside_wait_every_exec_step", bool(win) and bool(outside)
         and all(w["actions"][a] == WAIT and a not in w["moved"] for w in win for a in outside)),
        ("10_expected_equals_actual_all_agents", bool(rec["cbs_exec_trace"])
         and all(e["match"] and set(e["expected"]) == set(rec["starts"]) for e in rec["cbs_exec_trace"])
         and len(rec["cbs_exec_trace"]) == len(win)),
        ("11_no_collision_no_unplanned_move", not rec["safety_scan"]["vertex_overlap_steps"]
         and not rec["safety_scan"]["edge_swaps"] and not rec["integrity_abort"] and rec["exec_mismatch"] is None),
        ("12_il_resume_then_safe", i_end is not None and tr[i_end]["reason"] == "EXEC_WINDOW_END"
         and i_end + 1 < len(tr) and tr[i_end + 1]["to"] == "IL" and rec["termination"] == "SUCCESS"
         and rec["all_at_goal"]),
    ]
    first = next((n for n, ok in checks if not ok), None)
    return {"verdict": "PASS" if first is None else "FAIL", "first_failed_check": first,
            "checks": [{"name": n, "ok": bool(ok)} for n, ok in checks]}


def generic_checks(rec):
    """Safety invariants that must hold for ANY outcome (fallback / SAFE_STOP included)."""
    srs = rec["selective_records"]
    sr = srs[0] if srs else {}
    call = rec["runner_cbs_calls"][0] if rec["runner_cbs_calls"] else None
    moved_by_cbs = bool(rec["cbs_step_indices"])
    checks = [
        ("no_vertex_overlap", not rec["safety_scan"]["vertex_overlap_steps"]),
        ("no_edge_swap", not rec["safety_scan"]["edge_swaps"]),
        ("cbs_moves_only_if_accepted", (not moved_by_cbs) or (call is not None and call["accepted"] is True)),
        ("not_accepted_means_no_cbs_move", call is None or call["accepted"] or not moved_by_cbs),
        ("exec_trace_matches_or_stopped", all(e["match"] for e in rec["cbs_exec_trace"])
         or rec["stop_reason"] == "EXEC_MISMATCH" or rec["termination"] == "INTEGRITY_ABORT"),
        ("runner_latency_equals_attempt_sum", call is None or sr.get("attempt_latency_sum_ms") is None
         or abs(call["latency_ms"] - sr["attempt_latency_sum_ms"]) < 1e-6),
        ("timeout_or_late_success_never_executed",
         all(not (a["status"] != "success" and sr.get("outcome") == f"EXECUTE:{a['level']}") for a in sr.get("attempts", []))),
        ("mock_never_reported_real", all(a["planner_kind"] == "REAL" for a in sr.get("attempts", []))),
    ]
    first = next((n for n, ok in checks if not ok), None)
    return {"verdict": "PASS" if first is None else "FAIL", "first_failed_check": first,
            "checks": [{"name": n, "ok": bool(ok)} for n, ok in checks]}


def fallback_checks(rec):
    """S2/S3: correct recovery OR correct SAFE_STOP; a partial-group claim only if L0/L1 really executed."""
    sr = rec["selective_records"][0] if rec["selective_records"] else {}
    at = sr.get("attempts") or []
    out = sr.get("outcome") or ""
    vcalls = [(k, tuple(v)) for k, v in rec["validator_calls"]]
    g_tried = any(a["level"] == "G" and a["planner_kind"] == "REAL" and a["solver_invoked"] for a in at)
    if out.startswith("EXECUTE:G"):
        result_ok = vcalls == [("G", (True, ""))] and all(e["match"] for e in rec["cbs_exec_trace"])
    elif out.startswith("EXECUTE:"):
        result_ok = vcalls == [("SELECTIVE", (True, ""))] and all(e["match"] for e in rec["cbs_exec_trace"])
    else:
        result_ok = rec["termination"] == "SAFE_STOP" and rec["cbs_step_indices"] == [] and vcalls == []
    checks = [("generic_safety", rec["generic_checks"]["verdict"] == "PASS"),
              ("skip_reasons_recorded", all(lv["viable"] or lv["skip_reason"] for lv in sr.get("levels", []))),
              ("real_g_reached_when_lower_levels_fail", out.startswith("EXECUTE:L") or g_tried
               or out.startswith("SAFE_STOP:CBS_INVALID") or "UNAGREED" in out or "ATTEMPT_LIMIT" in out
               or "TIME_BUDGET" in out),
              ("recovery_verified_or_clean_safe_stop", result_ok)]
    first = next((n for n, ok in checks if not ok), None)
    partial = out in ("EXECUTE:L0", "EXECUTE:L1")
    return {"verdict": "PASS" if first is None else "FAIL", "first_failed_check": first,
            "partial_group_replan_observed": partial,
            "checks": [{"name": n, "ok": bool(ok)} for n, ok in checks]}


def injection_checks(fname, rec, t, outside_agent):
    """Per-injection verdict: the fault must be caught at once, nothing unsafe executed after it."""
    sr = rec["selective_records"][0] if rec["selective_records"] else {}
    at = sr.get("attempts") or []
    call = rec["runner_cbs_calls"][0] if rec["runner_cbs_calls"] else None
    lat_ok = call is not None and sr.get("attempt_latency_sum_ms") is not None \
        and abs(call["latency_ms"] - sr["attempt_latency_sum_ms"]) < 1e-6
    g_ok = sr.get("outcome") != "EXECUTE:G" or ([(k, tuple(v)) for k, v in rec["validator_calls"]] == [("G", (True, ""))]
                                                and all(e["match"] for e in rec["cbs_exec_trace"]))
    if fname == "F1_OUTSIDE_MOVES":
        mm = rec["exec_mismatch"] or {}
        checks = [("safe_stop_exec_mismatch", (rec["termination"], rec["stop_reason"]) == ("SAFE_STOP", "EXEC_MISMATCH")),
                  ("detected_at_injected_step", mm.get("t") == t + 1 and rec["steps"] == t + 1),
                  ("outside_agent_is_the_mismatch", mm.get("expected", {}).get(outside_agent)
                   != mm.get("actual", {}).get(outside_agent))]
    elif fname == "F2_INVALID_PATH":
        checks = [("safe_stop_invalid_path", (rec["termination"], rec["stop_reason"]) == ("SAFE_STOP", "CBS_INVALID_PATH")),
                  ("no_cbs_move", rec["cbs_step_indices"] == []),
                  ("no_g_after_invalid_path", [a["level"] for a in at] == ["L0"]),
                  ("validator_never_reached", rec["validator_calls"] == [])]
    elif fname == "F3_REAL_L0_TIMEOUT":
        checks = [("l0_real_timeout", bool(at) and at[0]["raw_status"] == "timeout" and at[0]["solver_invoked"] is True),
                  ("l0_not_executed", sr.get("outcome") != "EXECUTE:L0"),
                  ("failed_l0_latency_charged", lat_ok), ("g_execution_checked", g_ok)]
    elif fname == "F4_LATE_SUCCESS":
        checks = [("late_success_downgraded", bool(at) and at[0]["raw_status"] == "success" and at[0]["status"] == "timeout"
                   and (at[0]["budget_overrun_ms"] or 0) > 0),
                  ("late_success_not_executed", sr.get("outcome") != "EXECUTE:L0"),
                  ("overrun_charged", lat_ok), ("g_execution_checked", g_ok)]
    elif fname == "F5_INTEGRITY_ABORT":
        checks = [("integrity_abort", rec["termination"] == "INTEGRITY_ABORT" and rec["integrity_abort"]),
                  ("aborted_at_injected_step", rec["steps"] == t + 1),
                  ("overlap_only_at_injected_step", rec["safety_scan"]["vertex_overlap_steps"] == [t + 1])]
    else:
        raise ValueError(fname)
    first = next((n for n, ok in checks if not ok), None)
    return {"verdict": "PASS" if first is None else "FAIL", "first_failed_check": first,
            "checks": [{"name": n, "ok": bool(ok)} for n, ok in checks]}


# -- G-mode regression (section 5) -------------------------------------------------------------

NONDETERMINISTIC_KEYS = {"latency_ms", "latency_sec", "work_dir", "artifacts_dir"}


def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if k not in NONDETERMINISTIC_KEYS}
    if isinstance(o, (list, tuple)):
        return [_strip(v) for v in o]
    return o


def g_regression(name, solver_root, run_dir):
    from src.il.policy import NavHintPolicy
    grid, starts, goals, _ = scenario(name)
    out = {}
    for mode in ("GLOBAL", "SELECTIVE_DEFAULT_G"):
        inner = CbsRuntimePlanner(solver_root=solver_root, work_root=run_dir / f"{mode.lower()}_cbs")
        mon, planner, validator = DeadlockMonitor(), inner, PathGuardValidator()
        if mode != "GLOBAL":
            mon = ShadowMonitor(mon)
            planner = SelectivePlanner(inner, mon, SelectiveConfig())
            validator = SelectivePathValidator(planner)
        res = run_global_hybrid_episode(NavHintPolicy(DEFAULT_CKPT), grid, starts, goals, MAX_STEPS,
                                        planner=planner, path_validator=validator,
                                        config=dev_resume_config(H_EXEC), monitor=mon)
        out[mode] = {"result": res, "inner_calls": [{k: v for k, v in c.items() if k in ("status", "error", "paths")}
                                                    for c in inner.calls],
                     "validator_calls": getattr(validator, "calls", None),
                     "outcomes": [r.outcome for r in planner.records] if mode != "GLOBAL" else None}
    a, b = out["GLOBAL"], out["SELECTIVE_DEFAULT_G"]
    fields = ("transitions", "trajectory", "actions", "action_sources", "cbs_calls", "cbs_exec_trace",
              "termination", "stop_reason", "all_at_goal", "steps", "integrity_abort", "exec_mismatch")
    diffs = [f for f in fields if _strip(a["result"][f]) != _strip(b["result"][f])]
    whole_equal = _strip(a["result"]) == _strip(b["result"])
    inner_equal = _strip(a["inner_calls"]) == _strip(b["inner_calls"])
    return {"scenario": name, "verdict": "PASS" if not diffs and whole_equal and inner_equal else "FAIL",
            "differing_fields": diffs, "whole_result_equal_excl_latency": whole_equal,
            "cbs_solver_answers_equal": inner_equal, "selective_outcomes": b["outcomes"],
            "excluded_nondeterministic_keys": sorted(NONDETERMINISTIC_KEYS),
            "global": {k: a["result"][k] for k in ("termination", "stop_reason", "steps", "all_at_goal")}
            | {"cbs_calls": _strip(a["result"]["cbs_calls"]), "transitions": a["result"]["transitions"]},
            "safety_scan_global": safety_scan(a["result"]), "safety_scan_selective": safety_scan(b["result"])}


# -- failure injections (section 6) -----------------------------------------------------------

def tamper_wrapper(real):
    """F2: real solver answer, then teleport the first group agent at step 1 (invalid path)."""
    def wrapped(current_positions, goals, grid_map, timeout_sec=30, solver_root=None):
        res = real(current_positions, goals, grid_map, timeout_sec=timeout_sec, solver_root=solver_root)
        if res.status == "success" and res.paths:
            a = sorted(res.paths)[0]
            p = list(res.paths[a])
            far = (p[0][0], p[0][1] + 3)
            paths = {**res.paths, a: [p[0], far] + p[2:]}
            return type(res)(status="success", paths=paths, latency_sec=res.latency_sec, error=None)
        return res
    return wrapped


def late_wrapper(real, extra_sec, only_ids=None):
    """F4: real solver answer delivered late (sleep after the real call)."""
    def wrapped(current_positions, goals, grid_map, timeout_sec=30, solver_root=None):
        res = real(current_positions, goals, grid_map, timeout_sec=timeout_sec, solver_root=solver_root)
        if only_ids is None or set(current_positions) == set(only_ids):
            time.sleep(extra_sec)
        return res
    return wrapped


# -- main -------------------------------------------------------------------------------------

def _write(path: Path, obj):
    with path.open("x", encoding="utf-8") as fh:
        json.dump(_jsonable(obj), fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solver-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "outputs" / "selective_real_e2e_dev")
    ap.add_argument("--only", nargs="*", default=None, help="subset of: S1 S2 S3 S1B G F")
    args = ap.parse_args(argv)
    solver_root = args.solver_root.resolve()
    pre = preflight(solver_root)
    if pre["problems"]:
        print("REFUSED (nothing run or written):")
        for p in pre["problems"]:
            print("  -", p)
        return 2
    root = args.out_root / f"{datetime.now():%Y%m%d_%H%M%S}_selective_real_e2e"
    root.mkdir(parents=True, exist_ok=False)
    want = set(args.only or ["S1", "S2", "S1B", "S3", "G", "F"])
    summary = {"preflight": pre, "dev_only": {"budget": {
        "total_time_ms": DEV_BUDGET.total_time_ms, "max_attempts": DEV_BUDGET.max_attempts,
        "level_timeout_ms": dict(DEV_BUDGET.level_timeout_ms)}, "reserve_ms": DEV_RESERVE_MS, "h_exec": H_EXEC,
        "resume": "DevResumeIL", "recall": "DevSingleCbsCall", "max_steps": MAX_STEPS},
        "wall_clock_note": "solver killed at timeout; measured call time may exceed timeout_ms (not real-time)",
        "runs": {}}

    def scen(name):
        d = root / name
        d.mkdir()
        run = run_selective(name, solver_root, d)
        rec = record(run)
        rec["partial_checks"] = partial_checks(rec)
        rec["generic_checks"] = generic_checks(rec)
        rec["fallback_checks"] = fallback_checks(rec)
        _write(d / "result.json", rec)
        sr = rec["selective_records"][0] if rec["selective_records"] else {}
        summary["runs"][name] = {
            "description": rec["description"], "trigger_t": sr.get("trigger_t"), "wfg_groups": sr.get("wfg_groups"),
            "levels": [(lv["level"], lv["group"], lv["viable"], lv["skip_reason"]) for lv in sr.get("levels", [])],
            "attempts": [(a["level"], a["group"], a["planner_kind"], a["solver_invoked"], a["raw_status"], a["status"],
                          a["timeout_ms"], round(a["latency_ms"], 1) if a["latency_ms"] is not None else None)
                         for a in sr.get("attempts", [])],
            "outcome": sr.get("outcome"), "real_subset_cbs_called": sr.get("real_subset_cbs_called"),
            "real_global_cbs_called": sr.get("real_global_cbs_called"),
            "termination": rec["termination"], "stop_reason": rec["stop_reason"], "all_at_goal": rec["all_at_goal"],
            "steps": rec["steps"], "partial_verdict": rec["partial_checks"]["verdict"],
            "partial_first_failed": rec["partial_checks"]["first_failed_check"],
            "generic_verdict": rec["generic_checks"]["verdict"],
            "generic_first_failed": rec["generic_checks"]["first_failed_check"],
            "fallback_verdict": rec["fallback_checks"]["verdict"],
            "fallback_first_failed": rec["fallback_checks"]["first_failed_check"],
            "partial_group_replan_observed": rec["fallback_checks"]["partial_group_replan_observed"],
        }
        print(json.dumps(_jsonable({name: summary["runs"][name]}), ensure_ascii=False))
        return rec

    recs = {}
    for name in ("S1", "S2", "S1B", "S3"):
        if name in want:
            recs[name] = scen(name)

    if "G" in want:
        gdir = root / "G_regression"
        gdir.mkdir()
        greg = {}
        for name in ("CORRIDOR", "S1", "S2", "S1B", "S3"):
            d = gdir / name
            d.mkdir()
            greg[name] = g_regression(name, solver_root, d)
            print(json.dumps(_jsonable({f"G_regression:{name}": {k: greg[name][k] for k in
                                        ("verdict", "differing_fields", "whole_result_equal_excl_latency",
                                         "cbs_solver_answers_equal", "selective_outcomes", "global")}}),
                             ensure_ascii=False)[:600])
        _write(gdir / "result.json", greg)
        summary["g_regression"] = {k: {"verdict": v["verdict"], "differing_fields": v["differing_fields"]}
                                   for k, v in greg.items()}

    if "F" in want:
        base = next((recs[n] for n in ("S1", "S1B") if n in recs and recs[n]["partial_checks"]["verdict"] == "PASS"),
                    None)
        summary["failure_injection_base"] = base["scenario"] if base else None
        if base is None:
            print("F*: SKIPPED -- no scenario executed a partial group, injections need one")
        else:
            fdir = root / "F_injections"
            fdir.mkdir()
            sr = base["selective_records"][0]
            t, s_t = sr["trigger_t"], base["trajectory"][sr["trigger_t"]]
            out_a = sr["outside_held"][0]
            grp_a = sr["executed_group"][0]
            grid = np.array([[int(c) for c in row] for row in base["grid_rows"]])
            free_nb = next(((s_t[out_a][0] + dr, s_t[out_a][1] + dc) for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                            if 0 <= s_t[out_a][0] + dr < grid.shape[0] and 0 <= s_t[out_a][1] + dc < grid.shape[1]
                            and grid[s_t[out_a][0] + dr, s_t[out_a][1] + dc] == 0
                            and (s_t[out_a][0] + dr, s_t[out_a][1] + dc) not in base["trajectory"][t + 1].values()))
            real = adapter_mod.run_cbs_runtime
            injections = {
                "F1_OUTSIDE_MOVES": dict(simulator_cls=perturbing(t + 1, out_a, free_nb),
                                         expect=("SAFE_STOP", "EXEC_MISMATCH")),
                "F2_INVALID_PATH": dict(runtime_wrapper=tamper_wrapper(real), expect=("SAFE_STOP", "CBS_INVALID_PATH")),
                "F3_REAL_L0_TIMEOUT": dict(budget=TriggerBudget(30_000.0, 3, {"L0": 60.0, "L1": 60.0, "G": 20_000.0}),
                                           subset_reserve=0.0, expect=None),
                "F4_LATE_SUCCESS": dict(budget=TriggerBudget(30_000.0, 3, {"L0": 3_000.0, "L1": 3_000.0, "G": 20_000.0}),
                                        runtime_wrapper=late_wrapper(real, 3.2, only_ids=sr["executed_group"]),
                                        expect=None),
                "F5_INTEGRITY_ABORT": dict(simulator_cls=perturbing(t + 1, out_a, base["trajectory"][t + 1][grp_a]),
                                           expect=("INTEGRITY_ABORT", None)),
            }
            summary["failure_injections"] = {}
            for fname, kw in injections.items():
                expect = kw.pop("expect")
                d = fdir / fname
                d.mkdir()
                run = run_selective(base["scenario"], solver_root, d, **kw)
                rec = record(run, kw.get("budget", DEV_BUDGET))
                rec["injection"] = fname
                rec["generic_checks"] = generic_checks(rec)
                rec["injection_checks"] = injection_checks(fname, rec, t, out_a)
                _write(d / "result.json", rec)
                fsr = rec["selective_records"][0] if rec["selective_records"] else {}
                post_inject_moves = [i for i in rec["cbs_step_indices"] if i >= t + 1]
                row = {
                    "termination": rec["termination"], "stop_reason": rec["stop_reason"],
                    "expected": expect, "outcome": fsr.get("outcome"),
                    "attempts": [(a["level"], a["raw_status"], a["status"], a["timeout_ms"],
                                  None if a["latency_ms"] is None else round(a["latency_ms"], 1),
                                  None if a["budget_overrun_ms"] is None else round(a["budget_overrun_ms"], 1))
                                 for a in fsr.get("attempts", [])],
                    "runner_latency_ms": rec["runner_cbs_calls"][0]["latency_ms"] if rec["runner_cbs_calls"] else None,
                    "attempt_latency_sum_ms": fsr.get("attempt_latency_sum_ms"),
                    "cbs_steps": rec["cbs_step_indices"], "steps": rec["steps"],
                    "exec_mismatch": rec["exec_mismatch"], "generic_verdict": rec["generic_checks"]["verdict"],
                    "generic_first_failed": rec["generic_checks"]["first_failed_check"],
                    "cbs_moves_after_injection_step": post_inject_moves,
                    "injection_verdict": rec["injection_checks"]["verdict"],
                    "injection_first_failed": rec["injection_checks"]["first_failed_check"],
                }
                summary["failure_injections"][fname] = row
                print(json.dumps(_jsonable({fname: row}), ensure_ascii=False)[:900])

    _write(root / "summary.json", summary)
    print(f"wrote {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
