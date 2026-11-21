"""DEV: real P3 E2E through the Partial Trigger Runner opt-in (stage 2).

Real NavHint IL + real MAPFStepSimulator + real atb033 CBS (SubsetCbsPlanner for L0/L1,
BoundedGlobalCbsPlanner for G) + the Global Hybrid Runner with
``run_global_hybrid_episode(..., extra_trigger=PartialTriggerSource(...))``:

    monitor  = ShadowMonitor(PartialTriggerShadow(DeadlockMonitor(), {"K2": detector}))
    planner  = SelectivePlanner(..., allow_real_subset_cbs=True)          (opt-in ladder)
    source   = PartialTriggerSource(monitor=..., planner=..., detector_name="K2",
                                    allow_partial_trigger=True)
    config   = dev_rearm_config(h_exec=3)   (max 2 CBS calls, cooldown 2, 60 s; DEV values)

DEV ONLY, not research parameters: K_conflict = K_no_progress = 2, the Selective
TriggerBudget / reserve of scripts/run_selective_real_e2e_dev.py, h_exec 3, the rearm values.
No Trigger or group is injected. predictions.json is written before any run.

Verdict (the 10 items of the task, judged on the FIRST call -- the one the Partial Trigger
started; later re-triggers are recorded, not hidden):
  A  P3 full success      items 1-10 all hold
  B  P3 hold-safety only  items 1-8 hold, 9 or 10 does not
  C  SAFE_STOP            stopped safely before executing a partial plan
  D  SKIP                 no Partial Trigger, or no partial group, or no moving outside agent
  E  FAIL                 a safety item failed / unexpected outcome

Wall-clock: solver calls are killed at their limit but may overrun it (measured, recorded);
this is not a real-time guarantee.

Writes only to a NEW directory: <out-root>/<YYYYmmdd_HHMMSS>_partial_trigger_p3/
Usage: set PYTHONHASHSEED=0 & python -m scripts.run_partial_trigger_p3_dev --solver-root <atb033>/centralized/cbs
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_selective_p3_dev as p3
from scripts import run_selective_real_e2e_dev as base
from scripts import run_selective_real_e2e_dx_dev as dx
from scripts.run_hybrid_resume_dev import dev_rearm_config
from scripts.run_il import DEFAULT_CKPT
from src.common.deadlock_monitor import DeadlockMonitor
from src.hybrid.global_hybrid import run_global_hybrid_episode
from src.hybrid.partial_trigger import (
    PARTIAL_TRIGGER_REASON, PartialTriggerConfig, PartialTriggerDetector, PartialTriggerShadow,
    PartialTriggerSource,
)
from src.hybrid.selective_planner import SelectiveConfig, SelectivePathValidator, SelectivePlanner
from src.hybrid.selective_shadow import ShadowMonitor
from src.hybrid.subset_cbs_adapter import BoundedGlobalCbsPlanner, SubsetCbsPlanner

K = 2
H_EXEC = 3
SCENARIOS = ("P3F", "D2")
PREDICTIONS = {
    "P3F": {"partial_trigger_t": 3, "group": [0, 1], "outside_moving": [2],
            "expected": "L0 {0,1} with agent 2 held at its S_3 cell on column 10 (off the cross); 3-step window, "
                        "2 WAITs, IL_RESUME, 2 continues to (6,10); class A if IL then finishes",
            "basis": "shadow run 20261113_223352 (K2 first event t=3); not independent"},
    "D2": {"partial_trigger_t": 5, "group": [0, 2], "outside_moving": [1, 3],
           "expected": "L0 {0,2}; outside 3 stands at (4,4) next to the contested (4,3), so the group may need a "
                       "detour or the level may be skipped (GOAL/UNREACHABLE) -> uncertain; class A/B/C/D all possible",
           "basis": "shadow run 20261113_223352 (K2 first event t=5); not independent"},
}


def scenario(name):
    return p3.scenario(name) if name == "P3F" else dx.scenario(name)


def run(name, solver_root, work):
    from src.il.policy import NavHintPolicy
    grid, starts, goals, desc = scenario(name)
    det = PartialTriggerDetector(PartialTriggerConfig(K, K))
    mon = ShadowMonitor(PartialTriggerShadow(DeadlockMonitor(), {"K2": det}))
    s = SubsetCbsPlanner(solver_root=solver_root, work_root=work / "subset_cbs", overhead_reserve_ms=base.DEV_RESERVE_MS)
    g = BoundedGlobalCbsPlanner(solver_root=solver_root, work_root=work / "global_cbs",
                                overhead_reserve_ms=base.DEV_RESERVE_MS)
    cfg = SelectiveConfig(ladder=("L0", "L1", "G"), budget=base.DEV_BUDGET, acknowledge_unverified_outside_policy=True)
    planner = SelectivePlanner(g, mon, cfg, subset_planner=s, allow_real_subset_cbs=True)
    validator = SelectivePathValidator(planner)
    source = PartialTriggerSource(monitor=mon, planner=planner, detector_name="K2", allow_partial_trigger=True)
    t0 = time.perf_counter()
    result = run_global_hybrid_episode(NavHintPolicy(DEFAULT_CKPT), grid, starts, goals, base.MAX_STEPS,
                                       planner=planner, path_validator=validator,
                                       config=dev_rearm_config(H_EXEC), monitor=mon, extra_trigger=source)
    rec = base.record(dict(name=name, desc=desc, grid=grid, starts=starts, goals=goals, result=result,
                           planner=planner, validator=validator, subset=s, glob=g,
                           wall_sec=time.perf_counter() - t0))
    rec["generic_checks"] = base.generic_checks(rec)
    rec["partial_source"] = {"fired": [source.describe(o) for o in source.fired], "refused": source.refused}
    rec["detector_events"] = [source.describe(e) for e in det.events]
    rec["runner_triggers"] = result.get("triggers")
    rec["monitor_rearms"] = result.get("monitor_rearms")
    rec["record_errors"] = mon.inner.record_errors
    return rec


def p3_verdict(rec):
    tr, T, A, src = rec["transitions"], rec["trajectory"], rec["actions"], rec["action_sources"]
    goals = rec["goals"]
    items, info = {}, {}
    first_trig = next((x for x in tr if x["to"] == "TRIGGER"), None)
    srs = rec["selective_records"]
    if first_trig is None or first_trig["reason"] != PARTIAL_TRIGGER_REASON or not srs:
        return {"class": "D", "reason": "NO_PARTIAL_TRIGGER" if first_trig is None else
                f"FIRST_TRIGGER_WAS_{first_trig['reason']}", "items": items, "info": info}
    t = first_trig["t"]
    sr = srs[0]
    s_t = T[t]
    fired = rec["partial_source"]["fired"]
    items["1_natural_partial_trigger"] = (all(x == "IL" for x in src[:t]) and bool(fired) and fired[0]["t"] == t
                                          and sr["trigger_t"] == t)
    outcome = sr["outcome"] or ""
    outside = tuple(sr["outside_held"] or ())
    info.update(trigger_t=t, candidate_group=fired[0]["group"] if fired else None, wfg_groups=sr["wfg_groups"],
                levels=[(lv["level"], lv["group"], lv["viable"], lv["skip_reason"]) for lv in sr["levels"]],
                attempts=[(a["level"], a["group"], a["planner_kind"], a["solver_invoked"], a["raw_status"],
                           a["status"], a["timeout_ms"], a["latency_ms"], a["budget_overrun_ms"]) for a in sr["attempts"]],
                outcome=outcome, executed_group=sr["executed_group"], outside_held=outside)
    if outcome not in ("EXECUTE:L0", "EXECUTE:L1"):
        if rec["termination"] == "SAFE_STOP" and not rec["cbs_step_indices"]:
            return {"class": "C", "reason": f"SAFE_STOP:{rec['stop_reason']}", "items": items, "info": info}
        return {"class": "D", "reason": f"NO_PARTIAL_GROUP_EXECUTED:{outcome}", "items": items, "info": info}
    moving = [a for a in outside if s_t[a] != goals[a] and any(T[k][a] != T[k + 1][a] for k in range(t))]
    info["moving_outside"] = moving
    info["outside_detail"] = {a: {"s_t": s_t[a], "goal": goals[a],
                                  "moves_before": sum(1 for k in range(t) if T[k][a] != T[k + 1][a]),
                                  "positions_0_to_t": [T[k][a] for k in range(t + 1)]} for a in outside}
    items["2_outside_not_at_goal"] = bool(moving) or any(s_t[a] != goals[a] for a in outside)
    items["3_outside_moved_before_trigger"] = bool(moving)
    a0 = sr["attempts"][-1]
    items["4_real_partial_subset_cbs"] = (a0["level"] in ("L0", "L1") and a0["planner_kind"] == "REAL"
                                          and a0["solver_invoked"] is True and sr["real_subset_cbs_called"] is True
                                          and 0 < len(sr["executed_group"]) < rec["n_agents"])
    resume = next((x for x in tr if x["to"] == "IL_RESUME" and x["t"] > t), None)
    win_end = resume["t"] if resume else len(src)
    window = [i for i in range(t, win_end) if src[i] == "CBS"]
    info["first_window_steps"] = [i + 1 for i in window]
    sub = next((c for c in rec["adapter_calls"] if c["planner"] == "subset" and c["called"]
                and c["status"] == "success"), None)
    obs = [tuple(o) for o in (sub["solver_input_obstacles_xy"] or [])] if sub else []
    items["5_outside_wait_and_static"] = (bool(window) and all(A[i][a] == base.WAIT and T[i + 1][a] == s_t[a]
                                                                for i in window for a in outside)
                                          and all((s_t[a][1], s_t[a][0]) in obs for a in outside))
    v0 = rec["validator_calls"][0] if rec["validator_calls"] else None
    items["6_full_path_guard"] = (v0 is not None and v0[0] == "SELECTIVE" and tuple(v0[1]) == (True, "")
                                  and rec["runner_cbs_calls"][0]["accepted"] is True and a0["full_check_ok"] is True)
    items["7_expected_actual_and_no_collision"] = (bool(rec["cbs_exec_trace"]) and all(e["match"] for e in rec["cbs_exec_trace"])
                                                   and not rec["safety_scan"]["vertex_overlap_steps"]
                                                   and not rec["safety_scan"]["edge_swaps"]
                                                   and not rec["integrity_abort"])
    items["8_il_resume"] = (resume is not None and resume["reason"] == "EXEC_WINDOW_END"
                            and any(x["to"] == "IL" and x["t"] == resume["t"] for x in tr))
    resumed = {a: next((i + 1 for i in range(win_end, len(A)) if src[i] == "IL" and T[i + 1][a] != T[i][a]), None)
               for a in moving}
    info["outside_first_il_move_after_resume"] = resumed
    items["9_outside_moves_again_under_il"] = bool(moving) and all(v is not None for v in resumed.values())
    items["10_all_at_goal"] = rec["termination"] == "SUCCESS" and rec["all_at_goal"]
    info["retriggers"] = [(x["t"], x.get("reason"), x["outcome"]) for x in (rec["runner_triggers"] or [])][1:]
    info["termination"] = (rec["termination"], rec["stop_reason"], rec["steps"])
    if not all(items[k] for k in ("1_natural_partial_trigger", "4_real_partial_subset_cbs", "5_outside_wait_and_static",
                                  "6_full_path_guard", "7_expected_actual_and_no_collision")):
        return {"class": "E", "reason": "SAFETY_OR_WIRING_ITEM_FAILED", "items": items, "info": info}
    if not items["3_outside_moved_before_trigger"]:
        return {"class": "D", "reason": "OUTSIDE_AT_GOAL_OR_NEVER_MOVED", "items": items, "info": info}
    if all(items.values()):
        return {"class": "A", "reason": "P3_FULL_SUCCESS", "items": items, "info": info}
    if items["8_il_resume"]:
        return {"class": "B", "reason": "HOLD_SAFE_RESUME_OR_GOAL_NOT_CONFIRMED", "items": items, "info": info}
    return {"class": "E", "reason": "NO_IL_RESUME_AFTER_EXECUTION", "items": items, "info": info}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solver-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "outputs" / "partial_trigger_p3_dev")
    args = ap.parse_args(argv)
    solver_root = args.solver_root.resolve()
    pre = base.preflight(solver_root)
    if pre["problems"]:
        print("REFUSED (nothing run or written):")
        for p in pre["problems"]:
            print("  -", p)
        return 2
    root = args.out_root / f"{datetime.now():%Y%m%d_%H%M%S}_partial_trigger_p3"
    root.mkdir(parents=True, exist_ok=False)
    base._write(root / "predictions.json", {"written_before_runs": datetime.now().isoformat(), "k": K,
                                            "h_exec": H_EXEC, "rearm": "dev_rearm_config (2 calls, cooldown 2, 60 s)",
                                            "dev_only": True, "predictions": PREDICTIONS})
    summary = {"preflight": pre, "runs": {}}
    for name in SCENARIOS:
        d = root / name
        d.mkdir()
        try:
            rec = run(name, solver_root, d)
            rec["p3"] = p3_verdict(rec)
            base._write(d / "result.json", rec)
            row = {"class": rec["p3"]["class"], "reason": rec["p3"]["reason"], "items": rec["p3"]["items"],
                   **{k: rec["p3"]["info"].get(k) for k in ("trigger_t", "candidate_group", "wfg_groups", "levels",
                                                          "attempts", "outcome", "executed_group", "outside_held",
                                                          "moving_outside", "first_window_steps",
                                                          "outside_first_il_move_after_resume", "retriggers",
                                                          "termination")},
                   "generic": rec["generic_checks"]["verdict"], "record_errors": rec["record_errors"],
                   "source_refused": rec["partial_source"]["refused"]}
        except Exception as exc:                       # reported, never hidden
            row = {"class": "E", "reason": f"EXCEPTION:{type(exc).__name__}:{exc}"}
        summary["runs"][name] = row
        print(json.dumps(base._jsonable({name: row}), ensure_ascii=False))
    base._write(root / "summary.json", summary)
    print(f"wrote {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
