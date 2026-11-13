"""DEV: D1-D3 real Selective Hybrid E2E + Global Hybrid comparison (exploratory, NOT held-out).

Same real chain and DEV settings as scripts/run_selective_real_e2e_dev.py (that script and
its saved S1-S3 / S1-B results are left untouched; its record/verdict functions are reused):
real NavHint IL + MAPFStepSimulator + ShadowMonitor(DeadlockMonitor) + SelectivePlanner with
the real SubsetCbsPlanner (L0/L1) and BoundedGlobalCbsPlanner (G), unchanged L0/L1/WFG/timeout
rules. No Trigger is ever injected: a scenario without a real Trigger is reported NO_TRIGGER.

Scenarios (built in memory; original map/scenario files are only read):
  D1  controlled/scenario_intersection_n2 (7x7 cross) + ADDED agent 2 start=goal=(0,3)
      (north dead-end arm, behind agent 1's start). Designed after the S1-S3 results.
  D2  pilot_v2/scenario_pilot2_staggered_blocks_9x9, original 4 agents unchanged.
  D3  pilot_v2/scenario_pilot2_loop_w2_7x9, original 4 agents unchanged.
All three are DEV exploration cases, not an independent held-out evaluation.

predictions.json (start/goal, expected conflict cells, candidate WFG groups) is written
BEFORE any episode runs and is never rewritten.

Global comparison: run_global_hybrid_episode with the real CbsRuntimePlanner
(timeout_sec=GLOBAL_TIMEOUT_SEC) + PathGuardValidator + DeadlockMonitor and the same
dev_resume_config(H_EXEC). Its solver limit (20 s, whole seconds) differs from the Selective
G call's limit (min(20 s, remaining) - 0.5 s reserve): recorded, not hidden.

Usage (repository root; CBS calls are sequential in this one process):
    set PYTHONHASHSEED=0
    python -m scripts.run_selective_real_e2e_dx_dev --solver-root <atb033>/centralized/cbs
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_selective_real_e2e_dev as base
from scripts.run_hybrid_resume_dev import dev_resume_config
from scripts.run_il import DEFAULT_CKPT, load_scenario
from src.common.deadlock_monitor import DeadlockMonitor
from src.hybrid.cbs_runtime_bridge import CbsRuntimePlanner, PathGuardValidator
from src.hybrid.global_hybrid import run_global_hybrid_episode
from src.hybrid.selective_planner import SelectiveConfig, SelectivePathValidator, SelectivePlanner
from src.hybrid.selective_shadow import ShadowMonitor
from src.hybrid.subset_cbs_adapter import BoundedGlobalCbsPlanner, SubsetCbsPlanner

GLOBAL_TIMEOUT_SEC = 20          # DEV_ONLY; = Selective G level timeout (20 s)
SCEN = PROJECT_ROOT / "scenarios"

# Written to predictions.json BEFORE running. Coordinates (row, col).
PREDICTIONS = {
    "D1": {"expected_conflict_cells": [[3, 3]],
           "candidate_wfg_groups": [[0, 1]],
           "why_l0_smaller": "agent 2 parks in the north dead-end (0,3), off both routes",
           "risk": "IL may cross the intersection without a FULL_REPEAT -> NO_TRIGGER",
           "basis": "designed after seeing S1-S3 (not independent)"},
    "D2": {"expected_conflict_cells": "unknown (local contests between block gaps expected)",
           "candidate_wfg_groups": "unknown; pairs expected, could be all 4",
           "why_l0_smaller": "agents may not all be blocked at the same step",
           "risk": "NO_TRIGGER, or L0 == all agents",
           "basis": "original pilot2 scenario, no prediction basis from Selective runs"},
    "D3": {"expected_conflict_cells": "unknown (opposite flows on the width-2 loop)",
           "candidate_wfg_groups": "unknown; pairs or all 4",
           "why_l0_smaller": "opposite flows may separate into pairs",
           "risk": "NO_TRIGGER, or L0 == all agents",
           "basis": "original pilot2 scenario, no prediction basis from Selective runs"},
}


def scenario(name):
    if name == "D1":
        grid, starts, goals = load_scenario(SCEN / "controlled" / "scenario_intersection_n2.json")
        grid = np.array(grid, dtype=np.int64, copy=True)
        assert grid[0, 3] == 0
        return grid, {**starts, 2: (0, 3)}, {**goals, 2: (0, 3)}, \
            "intersection_n2 7x7 original map + added agent 2 start=goal=(0,3)"
    if name == "D2":
        grid, starts, goals = load_scenario(SCEN / "pilot_v2" / "scenario_pilot2_staggered_blocks_9x9.json")
        return np.array(grid, dtype=np.int64, copy=True), starts, goals, "pilot2_staggered_blocks_9x9 original"
    if name == "D3":
        grid, starts, goals = load_scenario(SCEN / "pilot_v2" / "scenario_pilot2_loop_w2_7x9.json")
        return np.array(grid, dtype=np.int64, copy=True), starts, goals, "pilot2_loop_w2_7x9 original"
    raise ValueError(name)


def run_selective(name, solver_root, run_dir):
    from src.il.policy import NavHintPolicy
    grid, starts, goals, desc = scenario(name)
    s = SubsetCbsPlanner(solver_root=solver_root, work_root=run_dir / "subset_cbs",
                         overhead_reserve_ms=base.DEV_RESERVE_MS)
    g = BoundedGlobalCbsPlanner(solver_root=solver_root, work_root=run_dir / "global_cbs",
                                overhead_reserve_ms=base.DEV_RESERVE_MS)
    mon = ShadowMonitor(DeadlockMonitor())
    cfg = SelectiveConfig(ladder=("L0", "L1", "G"), budget=base.DEV_BUDGET, acknowledge_unverified_outside_policy=True)
    planner = SelectivePlanner(g, mon, cfg, subset_planner=s, allow_real_subset_cbs=True)
    validator = SelectivePathValidator(planner)
    t0 = time.perf_counter()
    result = run_global_hybrid_episode(NavHintPolicy(DEFAULT_CKPT), grid, starts, goals, base.MAX_STEPS,
                                       planner=planner, path_validator=validator,
                                       config=dev_resume_config(base.H_EXEC), monitor=mon)
    return dict(name=name, desc=desc, grid=grid, starts=starts, goals=goals, result=result, planner=planner,
                validator=validator, subset=s, glob=g, wall_sec=time.perf_counter() - t0)


def run_global(name, solver_root, run_dir):
    from src.il.policy import NavHintPolicy
    grid, starts, goals, _ = scenario(name)
    planner = CbsRuntimePlanner(solver_root=solver_root, work_root=run_dir / "global_cbs",
                                timeout_sec=GLOBAL_TIMEOUT_SEC)
    validator = PathGuardValidator()
    t0 = time.perf_counter()
    res = run_global_hybrid_episode(NavHintPolicy(DEFAULT_CKPT), grid, starts, goals, base.MAX_STEPS,
                                    planner=planner, path_validator=validator,
                                    config=dev_resume_config(base.H_EXEC), monitor=DeadlockMonitor())
    return res, planner, time.perf_counter() - t0


def classify(rec):
    """Section-4 class of one Selective run. Safe stops are never SUCCESS."""
    srs = rec["selective_records"]
    if not any(t["to"] == "TRIGGER" for t in rec["transitions"]) or not srs:
        return "NO_TRIGGER"
    if rec["generic_checks"]["verdict"] != "PASS":
        return "FAIL"
    out = srs[0]["outcome"] or ""
    if out in ("EXECUTE:L0", "EXECUTE:L1"):
        items = {c["name"]: c["ok"] for c in rec["partial_checks"]["checks"]}
        if not all(ok for n, ok in items.items() if not n.startswith("12_")):
            return "FAIL"
        if rec["termination"] == "SUCCESS" and items["12_il_resume_then_safe"]:
            return "SELECTIVE_E2E_SUCCESS"
        if rec["termination"] == "TIMEOUT":
            return "SELECTIVE_EXEC_THEN_EPISODE_TIMEOUT"
        return f"SELECTIVE_EXEC_THEN_{rec['termination']}"
    if out == "EXECUTE:G":
        return f"L0L1_NOT_EXECUTED_THEN_GLOBAL:{rec['termination']}"
    if out.startswith("SAFE_STOP") and rec["termination"] == "SAFE_STOP" and not rec["cbs_step_indices"]:
        return f"SAFE_STOP:{rec['stop_reason']}"
    return "FAIL"


def global_summary(res, planner, wall):
    calls = res["cbs_calls"]
    return {"termination": res["termination"], "stop_reason": res["stop_reason"], "all_at_goal": res["all_at_goal"],
            "steps": res["steps"], "cbs_calls": len(calls),
            "group_size": [len(c["agent_order"]) for c in calls],
            "trigger_t": [c["trigger_timestep"] for c in calls],
            "cbs_status": [c["status"] for c in calls], "accepted": [c["accepted"] for c in calls],
            "cbs_latency_ms": [c["latency_ms"] for c in calls],
            "il_resume": any(t["to"] == "IL_RESUME" for t in res["transitions"]),
            "cbs_exec_all_match": all(e["match"] for e in res["cbs_exec_trace"]),
            "safety_scan": base.safety_scan(res), "integrity_abort": res["integrity_abort"],
            "transitions": res["transitions"], "wall_sec": wall,
            "solver_timeout_sec": GLOBAL_TIMEOUT_SEC}


def selective_summary(rec):
    sr = rec["selective_records"][0] if rec["selective_records"] else {}
    exec_grp = sr.get("executed_group")
    return {"class": rec["class"], "termination": rec["termination"], "stop_reason": rec["stop_reason"],
            "all_at_goal": rec["all_at_goal"], "steps": rec["steps"], "trigger_t": sr.get("trigger_t"),
            "wfg_edges": sr.get("wfg_edges"), "wfg_groups": sr.get("wfg_groups"),
            "levels": [(lv["level"], lv["group"], lv["viable"], lv["skip_reason"]) for lv in sr.get("levels", [])],
            "attempts": [(a["level"], a["group"], a["planner_kind"], a["solver_invoked"], a["raw_status"], a["status"],
                          a["timeout_ms"], None if a["latency_ms"] is None else round(a["latency_ms"], 1))
                         for a in sr.get("attempts", [])],
            "outcome": sr.get("outcome"), "executed_group_size": None if exec_grp is None else len(exec_grp),
            "outside_held": sr.get("outside_held"),
            "real_subset_cbs_called": sr.get("real_subset_cbs_called"),
            "real_global_cbs_called": sr.get("real_global_cbs_called"),
            "runner_cbs_latency_ms": rec["runner_cbs_calls"][0]["latency_ms"] if rec["runner_cbs_calls"] else None,
            "cbs_calls": len(rec["runner_cbs_calls"]),
            "il_resume": any(t["to"] == "IL_RESUME" for t in rec["transitions"]),
            "outside_wait_all_exec_steps": (bool(rec["exec_window"]) and bool(sr.get("outside_held"))
                                            and all(w["actions"][a] == base.WAIT and a not in w["moved"]
                                                    for w in rec["exec_window"] for a in sr["outside_held"])),
            "validator_calls": rec["validator_calls"], "exec_all_match": all(e["match"] for e in rec["cbs_exec_trace"]),
            "safety_scan": rec["safety_scan"], "partial_checks": rec["partial_checks"]["checks"],
            "partial_first_failed": rec["partial_checks"]["first_failed_check"],
            "generic_verdict": rec["generic_checks"]["verdict"], "wall_sec": rec["wall_sec"]}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solver-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "outputs" / "selective_real_e2e_dx_dev")
    ap.add_argument("--only", nargs="*", default=None, help="subset of D1 D2 D3")
    args = ap.parse_args(argv)
    solver_root = args.solver_root.resolve()
    pre = base.preflight(solver_root)
    if pre["problems"]:
        print("REFUSED (nothing run or written):")
        for p in pre["problems"]:
            print("  -", p)
        return 2
    names = args.only or ["D1", "D2", "D3"]
    root = args.out_root / f"{datetime.now():%Y%m%d_%H%M%S}_selective_real_e2e_dx"
    root.mkdir(parents=True, exist_ok=False)

    # 1) predictions BEFORE any run
    preds = {}
    for n in names:
        grid, starts, goals, desc = scenario(n)
        preds[n] = {"description": desc, "starts": starts, "goals": goals,
                    "grid_rows": ["".join(str(int(v)) for v in row) for row in grid], **PREDICTIONS[n],
                    "classification": "DEV exploration, not held-out"}
    base._write(root / "predictions.json", {"written_before_runs": datetime.now().isoformat(), "scenarios": preds})

    env = {"platform": platform.platform(), "processor": platform.processor(), "python": sys.version,
           "sequential_single_process": True, "dev_budget": {"total_time_ms": base.DEV_BUDGET.total_time_ms,
                                                               "max_attempts": base.DEV_BUDGET.max_attempts,
                                                               "level_timeout_ms": dict(base.DEV_BUDGET.level_timeout_ms)},
           "reserve_ms": base.DEV_RESERVE_MS, "h_exec": base.H_EXEC, "max_steps": base.MAX_STEPS,
           "global_solver_timeout_sec": GLOBAL_TIMEOUT_SEC}
    summary = {"preflight": pre, "environment": env, "runs": {}}
    for n in names:
        d = root / n
        d.mkdir()
        (d / "selective").mkdir()
        (d / "global").mkdir()
        try:
            run = run_selective(n, solver_root, d / "selective")
            rec = base.record(run)
            rec["partial_checks"] = base.partial_checks(rec)
            rec["generic_checks"] = base.generic_checks(rec)
            rec["fallback_checks"] = base.fallback_checks(rec)
            rec["class"] = classify(rec)
            base._write(d / "selective" / "result.json", rec)
            sel = selective_summary(rec)
        except Exception as exc:                                   # reported, never hidden
            sel = {"class": "FAIL", "exception": f"{type(exc).__name__}: {exc}"}
        try:
            gres, gplanner, gwall = run_global(n, solver_root, d / "global")
            glob = global_summary(gres, gplanner, gwall)
            base._write(d / "global" / "result.json", {"summary": glob, "result": gres})
        except Exception as exc:
            glob = {"exception": f"{type(exc).__name__}: {exc}"}
        summary["runs"][n] = {"selective": sel, "global": glob}
        brief = {k: sel.get(k) for k in ("class", "trigger_t", "wfg_groups", "levels", "attempts", "outcome",
                                         "termination", "stop_reason", "steps", "all_at_goal", "exception")}
        gbrief = {k: glob.get(k) for k in ("termination", "stop_reason", "steps", "all_at_goal", "cbs_calls",
                                           "group_size", "cbs_status", "cbs_latency_ms", "exception")}
        print(json.dumps(base._jsonable({n: {"selective": brief, "global": gbrief}}), ensure_ascii=False))
    base._write(root / "summary.json", summary)
    print(f"wrote {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
