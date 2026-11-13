"""DEV: P3 check -- can an outside agent that was MOVING before the Trigger be held (WAIT, static
obstacle) during the Selective CBS window and resume afterwards? Real IL + real atb033 CBS +
real Simulator + unchanged Runner / SelectivePlanner / WFG / L0-L1 / timeout rules.

Structural note recorded with every run (code + data, see report): the v0 Trigger is
TRIGGER_FULL_REPEAT (src/common/deadlock_monitor.py: the whole joint state equals an earlier
one), so at a natural Trigger every agent -- outside agents included -- stands where it stood at
`first_seen`. An outside agent that is not at its goal must then be stopped or cycling without
being blocked; a BLOCKED off-goal agent always has an occupy/contest WFG edge and lands in L0.

Two candidate scenarios are pre-registered in predictions.json BEFORE running and each is run
ONCE. They are not re-tuned after their results. No Trigger or group is injected.

  P3Q  intersection_n2 7x7 ORIGINAL map; agents 0/1 original + added agent 2 (0,3) -> (6,3)
       (follows agent 1 south). Predicted: agent 2 blocked behind 1 -> occupy edge -> WFG = all
       -> L0 SKIP -> class D.
  P3F  DEV map copy 7x11: intersection_n2 + row 3 extended to col 10 + column 10 opened
       (rows 0..6); agents 0/1 original + added agent 2 (0,10) -> (6,10) on its own column.
       Predicted: no full repeat while agent 2 moves -> Trigger only after agent 2 is at its goal
       -> outside at goal -> class D.

Section-4 abnormal checks run on the first run that executed a partial group (a P3 candidate if
one did, else D1 = parked outside, labelled NOT P3), each with exactly one injected fault.

Results go only to a NEW directory:
    <out-root>/<YYYYmmdd_HHMMSS>_selective_p3/...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_selective_real_e2e_dev as base
from scripts import run_selective_real_e2e_dx_dev as dx
from scripts.run_hybrid_resume_dev import dev_resume_config
from scripts.run_il import DEFAULT_CKPT, load_scenario
from src.common.deadlock_monitor import DeadlockMonitor
from src.hybrid import subset_cbs_adapter as adapter_mod
from src.hybrid.global_hybrid import run_global_hybrid_episode
from src.hybrid.selective_planner import SelectiveConfig, SelectivePathValidator, SelectivePlanner
from src.hybrid.selective_safety import TriggerBudget
from src.hybrid.selective_shadow import ShadowMonitor
from src.hybrid.subset_cbs_adapter import BoundedGlobalCbsPlanner, SubsetCbsPlanner
from src.simulator.mapf_step_simulator import MAPFStepSimulator

INTERSECTION = PROJECT_ROOT / "scenarios" / "controlled" / "scenario_intersection_n2.json"

PREDICTIONS = {
    "P3Q": {"outside_candidate": 2, "expected": "agent 2 queues behind agent 1 -> blocked -> occupy edge 2->1 "
            "-> WFG group {0,1,2} = all -> L0/L1 SKIP -> G", "expected_class": "D"},
    "P3F": {"outside_candidate": 2, "expected": "agent 2 keeps moving on column 10 -> no full repeat until it "
            "parks at (6,10) -> Trigger with agent 2 AT goal", "expected_class": "D"},
}


def scenario(name):
    grid, starts, goals = load_scenario(INTERSECTION)
    grid = np.array(grid, dtype=np.int64, copy=True)
    if name == "P3Q":
        return grid, {**starts, 2: (0, 3)}, {**goals, 2: (6, 3)}, \
            "intersection_n2 7x7 original map + added agent 2 (0,3)->(6,3)"
    if name == "P3F":
        g = np.ones((7, 11), dtype=np.int64)
        g[:, :7] = grid
        g[3, 7:] = 0
        g[:, 10] = 0
        return g, {**starts, 2: (0, 10)}, {**goals, 2: (6, 10)}, \
            "DEV map copy 7x11 (intersection_n2 + row 3 to col 10 + open column 10) + agent 2 (0,10)->(6,10)"
    if name == "D1":
        return dx.scenario("D1")
    raise ValueError(name)


def run(name, solver_root, run_dir, *, budget=base.DEV_BUDGET, subset_reserve=base.DEV_RESERVE_MS,
        global_reserve=base.DEV_RESERVE_MS, simulator_cls=MAPFStepSimulator, runtime_wrapper=None):
    from src.il.policy import NavHintPolicy
    grid, starts, goals, desc = scenario(name)
    s = SubsetCbsPlanner(solver_root=solver_root, work_root=run_dir / "subset_cbs", overhead_reserve_ms=subset_reserve)
    g = BoundedGlobalCbsPlanner(solver_root=solver_root, work_root=run_dir / "global_cbs",
                                overhead_reserve_ms=global_reserve)
    mon = ShadowMonitor(DeadlockMonitor())
    cfg = SelectiveConfig(ladder=("L0", "L1", "G"), budget=budget, acknowledge_unverified_outside_policy=True)
    planner = SelectivePlanner(g, mon, cfg, subset_planner=s, allow_real_subset_cbs=True)
    validator = SelectivePathValidator(planner)
    ctx = patch.object(adapter_mod, "run_cbs_runtime", runtime_wrapper) if runtime_wrapper else base._null()
    t0 = time.perf_counter()
    with ctx:
        result = run_global_hybrid_episode(NavHintPolicy(DEFAULT_CKPT), grid, starts, goals, base.MAX_STEPS,
                                           planner=planner, path_validator=validator,
                                           config=dev_resume_config(base.H_EXEC), monitor=mon,
                                           simulator_cls=simulator_cls)
    rec = base.record(dict(name=name, desc=desc, grid=grid, starts=starts, goals=goals, result=result,
                           planner=planner, validator=validator, subset=s, glob=g,
                           wall_sec=time.perf_counter() - t0), budget)
    rec["partial_checks"] = base.partial_checks(rec)
    rec["generic_checks"] = base.generic_checks(rec)
    rec["monitor_trigger_codes"] = [(st.t, st.trigger_reason_code) for st in mon.steps
                                    if st.trigger_reason_code != "TRIGGER_NONE"]
    return rec


# -- P3 evaluation ------------------------------------------------------------------------------

def _first(transitions, to, after_t=None):
    return next((x for x in transitions if x["to"] == to and (after_t is None or x["t"] >= after_t)), None)


def p3_checks(rec):
    """A/B/C/D/E classification (section 5) with every item recorded."""
    tr, T, A = rec["transitions"], rec["trajectory"], rec["actions"]
    srs = rec["selective_records"]
    sr = srs[0] if srs else {}
    t = sr.get("trigger_t")
    goals = rec["goals"]
    out = {"items": {}, "moving_outside": [], "outside_detail": {}}
    items = out["items"]
    if t is None or not any(x["to"] == "TRIGGER" for x in tr):
        out["class"], out["reason"] = "D", "NO_NATURAL_TRIGGER"
        return out
    trig = rec["runner_cbs_calls"][0] if rec["runner_cbs_calls"] else {}
    outcome = sr.get("outcome") or ""
    outside = tuple(sr.get("outside_held") or ())
    s_t = T[t]
    # every non-group agent, also when no partial group was executed (for the record)
    group = tuple(sr.get("executed_group") or ())
    non_group = [a for a in sorted(s_t) if a not in group] if group else []
    for a in (outside or non_group):
        moves_before = sum(1 for k in range(t) if T[k][a] != T[k + 1][a])
        out["outside_detail"][a] = {
            "s_t": s_t[a], "goal": goals[a], "at_goal_at_trigger": s_t[a] == goals[a],
            "moves_before_trigger": moves_before,
            "last_move_t": max((k + 1 for k in range(t) if T[k][a] != T[k + 1][a]), default=None),
            "positions_before_trigger": [T[k][a] for k in range(t + 1)]}
    if rec["generic_checks"]["verdict"] != "PASS" or rec["integrity_abort"]:
        out["class"], out["reason"] = "E", f"GENERIC:{rec['generic_checks']['first_failed_check']}"
        return out
    if outcome not in ("EXECUTE:L0", "EXECUTE:L1"):
        if rec["termination"] == "SAFE_STOP" and not rec["cbs_step_indices"]:
            out["class"], out["reason"] = "C", f"SAFE_STOP:{rec['stop_reason']}"
        else:
            out["class"], out["reason"] = "D", f"NO_PARTIAL_GROUP:{outcome}"
        return out
    moving = [a for a in outside if s_t[a] != goals[a] and out["outside_detail"][a]["moves_before_trigger"] >= 1]
    out["moving_outside"] = moving
    pc = {c["name"]: c["ok"] for c in rec["partial_checks"]["checks"]}
    win = rec["exec_window"]
    obs = [tuple(o) for o in next((c["solver_input_obstacles_xy"] for c in rec["adapter_calls"]
                                   if c["planner"] == "subset" and c["called"]), []) or []]
    items.update({
        "natural_trigger": pc["1_trigger_in_real_IL"] and pc["2_wfg_uses_s_t_minus_1_and_intended"],
        "l0_smaller_than_all": pc["6_group_smaller_than_all"],
        "path0_is_s_t": pc["3_cbs_start_is_s_t"],
        "obstacle_cells_equal_outside_s_t": bool(outside) and all((s_t[a][1], s_t[a][0]) in obs for a in outside),
        "outside_all_wait": bool(win) and all(w["actions"][a] == base.WAIT for w in win for a in outside),
        "outside_actual_equals_s_t_every_step": all(e["actual"][a] == s_t[a] for e in rec["cbs_exec_trace"]
                                                    for a in outside),
        "full_path_guard": pc["8_full_compose_path_guard"],
        "expected_equals_actual": pc["10_expected_equals_actual_all_agents"],
        "no_vertex_edge_conflict": pc["11_no_collision_no_unplanned_move"],
    })
    if not all(items.values()):
        out["class"], out["reason"] = "E", "WINDOW:" + next(k for k, v in items.items() if not v)
        return out
    if not moving:
        out["class"], out["reason"] = "D", "OUTSIDE_AT_GOAL_OR_NEVER_MOVED (not a P3 case)"
        return out
    res = _first(tr, "IL_RESUME")
    r_t = res["t"] if res else None
    il_back = r_t is not None and any(x["to"] == "IL" and x["t"] == r_t for x in tr)
    after = range(r_t, len(A)) if r_t is not None else range(0)
    resumed = {a: next((k + 1 for k in after if rec["action_sources"][k] == "IL" and T[k + 1][a] != T[k][a]), None)
               for a in moving}
    post_rep = _post_resume_repeats(T, r_t)
    items.update({"il_resume_then_il": il_back,
                  "wait_released_and_outside_moved_again": all(v is not None for v in resumed.values()),
                  "all_at_goal": rec["termination"] == "SUCCESS" and rec["all_at_goal"]})
    out["resume_t"], out["outside_first_move_after_resume"] = r_t, resumed
    out["post_resume_joint_state_repeats"] = post_rep
    out["post_resume_no_progress_runs"] = _stall_runs(T, r_t, goals)
    if items["il_resume_then_il"] and items["wait_released_and_outside_moved_again"] and items["all_at_goal"]:
        out["class"], out["reason"] = "A", "P3_FULL_SUCCESS"
    else:
        out["class"], out["reason"] = "B", "WAIT_VERIFIED_RESUME_OR_GOAL_NOT"
    return out


def _post_resume_repeats(T, r_t):
    if r_t is None:
        return []
    seen, reps = {}, []
    for k in range(r_t, len(T)):
        key = tuple(sorted(T[k].items()))
        if key in seen:
            reps.append((seen[key], k))
        else:
            seen[key] = k
    return reps


def _stall_runs(T, r_t, goals, min_len=2):
    if r_t is None:
        return []
    runs, start = [], None
    for k in range(r_t, len(T) - 1):
        still = T[k] == T[k + 1] and T[k + 1] != goals
        if still and start is None:
            start = k
        if not still and start is not None:
            if k - start >= min_len:
                runs.append((start, k))
            start = None
    return runs


# -- section-4 injections ----------------------------------------------------------------------

def _bfs_path(grid, a, b):
    H, W = grid.shape
    prev, q = {a: None}, deque([a])
    while q:
        c = q.popleft()
        if c == b:
            break
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (c[0] + dr, c[1] + dc)
            if 0 <= n[0] < H and 0 <= n[1] < W and grid[n] == 0 and n not in prev:
                prev[n] = c
                q.append(n)
    if b not in prev:
        return None
    p, c = [], b
    while c is not None:
        p.append(c)
        c = prev[c]
    return p[::-1]


def detour_wrapper(real, grid, outside_cell):
    """Real solver answer, then the group agent whose goal is closest to the held outside cell
    walks from its goal onto that cell and back (moves stay legal; lengths padded)."""
    def wrapped(current_positions, goals, grid_map, timeout_sec=30, solver_root=None):
        res = real(current_positions, goals, grid_map, timeout_sec=timeout_sec, solver_root=solver_root)
        if res.status != "success" or not res.paths:
            return res
        a = min(res.paths, key=lambda k: abs(goals[k][0] - outside_cell[0]) + abs(goals[k][1] - outside_cell[1]))
        go = _bfs_path(grid, tuple(goals[a]), tuple(outside_cell))
        if not go:
            return res
        p = list(res.paths[a]) + go[1:] + go[::-1][1:]
        n = len(p)
        paths = {k: (p if k == a else list(v) + [v[-1]] * (n - len(v))) for k, v in res.paths.items()}
        return type(res)(status="success", paths=paths, latency_sec=res.latency_sec, error=None)
    return wrapped


def goal_blocked_planner_check(solver_root, work):
    """Planner level: an outside agent standing on a group goal is refused before any solver call."""
    p = SubsetCbsPlanner(solver_root=solver_root, work_root=work, overhead_reserve_ms=base.DEV_RESERVE_MS)
    r = p({0: (3, 2), 1: (2, 3)}, {0: (3, 5), 1: (5, 3)}, scenario("D1")[0], outside={2: (3, 5)}, timeout_ms=10_000.0)
    return {"status": r.status, "error": r.error, "solver_invoked": p.calls[0]["called"],
            "verdict": "PASS" if r.status == "input_error" and "GOAL_BLOCKED" in (r.error or "")
            and p.calls[0]["called"] is False else "FAIL"}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solver-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "outputs" / "selective_p3_dev")
    args = ap.parse_args(argv)
    solver_root = args.solver_root.resolve()
    pre = base.preflight(solver_root)
    if pre["problems"]:
        print("REFUSED (nothing run or written):")
        for p in pre["problems"]:
            print("  -", p)
        return 2
    root = args.out_root / f"{datetime.now():%Y%m%d_%H%M%S}_selective_p3"
    root.mkdir(parents=True, exist_ok=False)
    preds = {}
    for n in ("P3Q", "P3F"):
        grid, starts, goals, desc = scenario(n)
        preds[n] = {"description": desc, "starts": starts, "goals": goals,
                    "grid_rows": ["".join(str(int(v)) for v in row) for row in grid], **PREDICTIONS[n]}
    base._write(root / "predictions.json", {"written_before_runs": datetime.now().isoformat(),
                                            "run_once_each_no_retuning": True, "scenarios": preds})
    summary = {"preflight": pre, "runs": {}, "abnormal": {}}

    partial_base = None
    for n in ("P3Q", "P3F"):
        d = root / n
        d.mkdir()
        rec = run(n, solver_root, d)
        rec["p3"] = p3_checks(rec)
        base._write(d / "result.json", rec)
        sr = rec["selective_records"][0] if rec["selective_records"] else {}
        row = {"class": rec["p3"]["class"], "reason": rec["p3"]["reason"], "trigger_t": sr.get("trigger_t"),
               "wfg_edges": sr.get("wfg_edges"), "wfg_groups": sr.get("wfg_groups"),
               "levels": [(lv["level"], lv["group"], lv["skip_reason"]) for lv in sr.get("levels", [])],
               "outcome": sr.get("outcome"), "outside_detail": rec["p3"]["outside_detail"],
               "termination": rec["termination"], "stop_reason": rec["stop_reason"], "steps": rec["steps"],
               "all_at_goal": rec["all_at_goal"], "generic": rec["generic_checks"]["verdict"]}
        summary["runs"][n] = row
        print(json.dumps(base._jsonable({n: row}), ensure_ascii=False))
        if partial_base is None and sr.get("outcome") in ("EXECUTE:L0", "EXECUTE:L1") and rec["p3"]["class"] != "E":
            partial_base = (n, rec)

    if partial_base is None:                          # fall back to D1, explicitly NOT a P3 case
        d = root / "D1_base_not_P3"
        d.mkdir()
        rec = run("D1", solver_root, d)
        rec["p3"] = p3_checks(rec)
        base._write(d / "result.json", rec)
        partial_base = ("D1", rec)
    bname, brec = partial_base
    summary["abnormal_base"] = {"scenario": bname, "is_p3_case": bname != "D1" and brec["p3"]["class"] in "AB"}
    sr = brec["selective_records"][0]
    t = sr["trigger_t"]
    out_a = sr["outside_held"][0]
    s_t = brec["trajectory"][t]
    grid = scenario(bname)[0]
    nb = next(((s_t[out_a][0] + dr, s_t[out_a][1] + dc) for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
               if 0 <= s_t[out_a][0] + dr < grid.shape[0] and 0 <= s_t[out_a][1] + dc < grid.shape[1]
               and grid[s_t[out_a][0] + dr, s_t[out_a][1] + dc] == 0
               and (s_t[out_a][0] + dr, s_t[out_a][1] + dc) not in brec["trajectory"][t + 1].values()), None)
    real = adapter_mod.run_cbs_runtime
    inj = {
        "X1_OUTSIDE_MOVES_DURING_EXEC": dict(simulator_cls=base.perturbing(t + 1, out_a, nb)),
        "X2_GROUP_PATH_ENTERS_HELD_OUTSIDE_CELL": dict(runtime_wrapper=detour_wrapper(real, grid, s_t[out_a])),
        "X3_INVALID_PATH_TELEPORT": dict(runtime_wrapper=base.tamper_wrapper(real)),
        "X4_ALL_LEVELS_REAL_TIMEOUT": dict(budget=TriggerBudget(30_000.0, 3, {"L0": 60.0, "L1": 60.0, "G": 60.0}),
                                           subset_reserve=0.0, global_reserve=0.0),
    }
    for fname, kw in inj.items():
        if fname == "X1_OUTSIDE_MOVES_DURING_EXEC" and nb is None:
            summary["abnormal"][fname] = {"verdict": "SKIP", "reason": "no free neighbour for the outside agent"}
            continue
        d = root / "abnormal" / fname
        d.mkdir(parents=True)
        rec = run(bname, solver_root, d, **kw)
        base._write(d / "result.json", rec)
        fsr = rec["selective_records"][0] if rec["selective_records"] else {}
        moved = rec["cbs_step_indices"]
        if fname == "X1_OUTSIDE_MOVES_DURING_EXEC":
            ok = (rec["termination"], rec["stop_reason"]) == ("SAFE_STOP", "EXEC_MISMATCH") and rec["steps"] == t + 1
        elif fname == "X2_GROUP_PATH_ENTERS_HELD_OUTSIDE_CELL":
            err = next((c["error"] for c in rec["adapter_calls"] if c["planner"] == "subset"), "") or ""
            ok = (rec["stop_reason"] == "CBS_INVALID_PATH" and not moved and "REJECTED" in err
                  and "obstacle" in err.lower())
        elif fname == "X3_INVALID_PATH_TELEPORT":
            ok = rec["stop_reason"] == "CBS_INVALID_PATH" and not moved
        else:
            ok = (rec["termination"] == "SAFE_STOP" and not moved
                  and all(a["status"] != "success" for a in fsr.get("attempts", [])))
        row = {"verdict": "PASS" if ok else "FAIL", "termination": rec["termination"],
               "stop_reason": rec["stop_reason"], "steps": rec["steps"], "cbs_moves": moved,
               "attempts": [(a["level"], a["raw_status"], a["status"], a["timeout_ms"],
                             None if a["latency_ms"] is None else round(a["latency_ms"], 1)) for a in fsr.get("attempts", [])],
               "adapter_errors": [((c["error"] or "")[:160]) for c in rec["adapter_calls"] if c["error"]],
               "exec_mismatch": rec["exec_mismatch"], "safety_scan": rec["safety_scan"]}
        summary["abnormal"][fname] = row
        print(json.dumps(base._jsonable({fname: row}), ensure_ascii=False)[:700])
    gb = goal_blocked_planner_check(solver_root, root / "abnormal" / "X5_goal_blocked_planner_level")
    summary["abnormal"]["X5_OUTSIDE_ON_GROUP_GOAL_PLANNER_LEVEL"] = gb
    print(json.dumps({"X5_OUTSIDE_ON_GROUP_GOAL_PLANNER_LEVEL": gb}, ensure_ascii=False))
    base._write(root / "summary.json", summary)
    print(f"wrote {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
