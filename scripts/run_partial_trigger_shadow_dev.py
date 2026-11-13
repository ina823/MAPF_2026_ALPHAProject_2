"""DEV: Partial Trigger candidates in SHADOW on real runs -- no control change.

For each scenario the unchanged Global Hybrid Runner runs twice with the real NavHint IL,
real MAPFStepSimulator and the real atb033 CBS (CbsRuntimePlanner, timeout_sec=20):
  BASE    monitor = DeadlockMonitor()
  SHADOW  monitor = PartialTriggerShadow(DeadlockMonitor(), {K2, K3, K5 detectors})
The two results must be identical (latency / work-dir keys excluded). The detectors only
record PARTIAL_CANDIDATE events; nothing calls CBS, forces WAIT, or changes a transition.
K = 2/3/5 (k_conflict = k_no_progress = K) are DEV test candidates, not validated values.

Scenarios are the existing DEV cases, unchanged: P3Q, P3F (scripts/run_selective_p3_dev),
S1B (scripts/run_selective_real_e2e_dev), D1, D2 (scripts/run_selective_real_e2e_dx_dev).
predictions.json is written before any run.

Per candidate event it records: t, group, conflict key, streaks, parked members, other
conflict-flagged agents, group < all agents, agents outside the group that MOVED at that step,
the WFG groups computed from the same S_{t-1}/intended/blocked (P1-c), whether it came
before the FULL Trigger, the Runner phase at t (IL / CBS window / after resume), and whether
the group progressed without CBS within K steps afterwards ("possible false positive").

Writes only to a NEW directory: <out-root>/<YYYYmmdd_HHMMSS>_partial_trigger_shadow/
Usage: set PYTHONHASHSEED=0 & python -m scripts.run_partial_trigger_shadow_dev --solver-root <atb033>/centralized/cbs
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_selective_p3_dev as p3
from scripts import run_selective_real_e2e_dev as base
from scripts import run_selective_real_e2e_dx_dev as dx
from scripts.run_hybrid_resume_dev import dev_resume_config
from scripts.run_il import DEFAULT_CKPT
from src.common.deadlock_monitor import DeadlockMonitor
from src.hybrid.cbs_runtime_bridge import CbsRuntimePlanner, PathGuardValidator
from src.hybrid.global_hybrid import run_global_hybrid_episode
from src.hybrid.partial_trigger import PartialTriggerConfig, PartialTriggerDetector, PartialTriggerShadow
from src.hybrid.wfg import build_wfg
from src.il.nav_hint import bfs_dist

KS = (2, 3, 5)
GLOBAL_TIMEOUT_SEC = 20
SCENARIOS = ("P3Q", "P3F", "S1B", "D1", "D2")

PREDICTIONS = {
    "P3F": "agent 2 moves t=0..6 while 0/1 contest (3,3) from t=1 -> K2/K3 candidate {0,1} before FULL (t=7), "
           "with agent 2 moving outside the group; K5 at t=5..6 if streaks hold",
    "P3Q": "two live keys at t=2 ({0,1}@(3,3) and {1,2}@(2,3)); only one is exposed (Monitor limit) -> group "
           "uncertain; FULL fires at t=2, so no candidate before FULL",
    "S1B": "all other agents static -> FULL at the first stalled step (t=5) -> no candidate before FULL",
    "D1": "same as S1B (FULL at t=2) -> no candidate before FULL",
    "D2": "outside agents already parked -> FULL at the first stalled step (t=11) -> no candidate before FULL",
}


def scenario(name):
    if name in ("P3Q", "P3F"):
        return p3.scenario(name)
    if name == "S1B":
        return base.scenario(name)
    return dx.scenario(name)


def run_once(name, solver_root, work, shadow):
    from src.il.policy import NavHintPolicy
    grid, starts, goals, desc = scenario(name)
    dets = {f"K{k}": PartialTriggerDetector(PartialTriggerConfig(k, k)) for k in KS} if shadow else {}
    mon = PartialTriggerShadow(DeadlockMonitor(), dets) if shadow else DeadlockMonitor()
    planner = CbsRuntimePlanner(solver_root=solver_root, work_root=work, timeout_sec=GLOBAL_TIMEOUT_SEC)
    res = run_global_hybrid_episode(NavHintPolicy(DEFAULT_CKPT), grid, starts, goals, base.MAX_STEPS,
                                    planner=planner, path_validator=PathGuardValidator(),
                                    config=dev_resume_config(base.H_EXEC), monitor=mon)
    answers = [{k: v for k, v in c.items() if k in ("status", "error", "paths")} for c in planner.calls]
    return res, answers, mon, dets, (grid, starts, goals, desc)


def phase_at(res, t):
    """Runner phase of the step that produced S_t: IL, CBS (window) or IL_AFTER_RESUME."""
    src = res["action_sources"][t - 1] if 0 < t <= len(res["action_sources"]) else None
    resumed = any(x["to"] == "IL_RESUME" and x["t"] < t for x in res["transitions"])
    if src == "CBS":
        return "CBS_WINDOW"
    return "IL_AFTER_RESUME" if resumed else "IL"


def event_row(ev, res, steps, grid, goals, k, full_t):
    t = ev.t
    st = steps[t - 1] if 0 < t <= len(steps) else None
    wfg_groups = None
    if st and st["s_t_minus_1"] is not None and st["intended"] is not None:
        wfg_groups = [list(g) for g in build_wfg(st["s_t_minus_1"], st["intended"], st["blocked"], grid).groups]
    moved_outside = sorted(a for a in st["s_t"] if a not in ev.group and st["s_t_minus_1"][a] != st["s_t"][a]) \
        if st and st["s_t_minus_1"] else []
    outside_not_at_goal = sorted(a for a in st["s_t"] if a not in ev.group and st["s_t"][a] != goals[a]) if st else []
    # progress of group members without CBS within k steps after t (possible false positive)
    T, src = res["trajectory"], res["action_sources"]
    dist = {a: bfs_dist(grid, goals[a]) for a in ev.group}
    self_resolved = None
    for j in range(t, min(t + k, len(T) - 1)):
        if src[j] != "IL":
            break
        if any(int(dist[a][T[j + 1][a]]) < int(dist[a][T[t][a]]) for a in ev.group if T[t][a] != goals[a]):
            self_resolved = j + 1
            break
    in_wfg = None
    if wfg_groups is not None:
        in_wfg = {"equals_a_wfg_group": any(set(g) == set(ev.group) for g in wfg_groups),
                  "subset_of_wfg_union": set(ev.group) <= set().union(*map(set, wfg_groups)) if wfg_groups else False}
    return {"t": t, "group": ev.group, "key": [ev.key_kind, ev.key_cells], "conflict_streak": ev.conflict_streak,
            "no_progress": ev.no_progress, "parked_in_group": ev.parked_in_group,
            "conflicted_outside_key": ev.conflicted_outside_key, "group_smaller_than_all": ev.group_smaller_than_all,
            "n_agents": ev.n_agents, "repeat_since_rearm": ev.repeat_since_rearm,
            "outside_moved_at_t": moved_outside, "outside_not_at_goal_at_t": outside_not_at_goal,
            "wfg_groups_same_inputs": wfg_groups, "vs_wfg": in_wfg,
            "before_full_trigger": full_t is None or t < full_t, "steps_before_full": None if full_t is None else full_t - t,
            "phase": phase_at(res, t), "group_progressed_without_cbs_at": self_resolved}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solver-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "outputs" / "partial_trigger_shadow_dev")
    args = ap.parse_args(argv)
    solver_root = args.solver_root.resolve()
    pre = base.preflight(solver_root)
    if pre["problems"]:
        print("REFUSED (nothing run or written):")
        for p in pre["problems"]:
            print("  -", p)
        return 2
    root = args.out_root / f"{datetime.now():%Y%m%d_%H%M%S}_partial_trigger_shadow"
    root.mkdir(parents=True, exist_ok=False)
    base._write(root / "predictions.json", {"written_before_runs": datetime.now().isoformat(), "ks": KS,
                                            "predictions": PREDICTIONS})
    summary = {"preflight": pre, "ks": KS, "global_solver_timeout_sec": GLOBAL_TIMEOUT_SEC, "runs": {}}
    for name in SCENARIOS:
        d = root / name
        d.mkdir()
        b_res, b_ans, _, _, _ = run_once(name, solver_root, d / "base_cbs", shadow=False)
        s_res, s_ans, mon, dets, (grid, starts, goals, desc) = run_once(name, solver_root, d / "shadow_cbs", shadow=True)
        identical = base._strip(b_res) == base._strip(s_res) and base._strip(b_ans) == base._strip(s_ans)
        full_t = next((x["t"] for x in s_res["transitions"] if x["to"] == "TRIGGER"), None)
        per_k = {}
        for kname, det in dets.items():
            k = det.config.k_conflict
            rows = [event_row(ev, s_res, mon.steps, grid, goals, k, full_t) for ev in det.events]
            status_counts = {}
            for o in det.observations:
                status_counts[o.status] = status_counts.get(o.status, 0) + 1
            per_k[kname] = {"events": rows, "n_events": len(rows), "status_counts": status_counts,
                            "first_event_t": rows[0]["t"] if rows else None,
                            "first_event_before_full": bool(rows) and rows[0]["before_full_trigger"],
                            "repeats": sum(1 for r in rows if r["repeat_since_rearm"]),
                            "possible_false_positives": [r["t"] for r in rows if r["group_progressed_without_cbs_at"]]}
        rec = {"scenario": name, "description": desc, "starts": starts, "goals": goals,
               "global_identical_with_shadow": identical, "full_trigger_t": full_t,
               "termination": s_res["termination"], "stop_reason": s_res["stop_reason"], "steps": s_res["steps"],
               "all_at_goal": s_res["all_at_goal"], "record_errors": mon.record_errors, "per_k": per_k,
               "observations": {kn: [asdict(o) for o in dt.observations] for kn, dt in dets.items()},
               "shadow_steps": mon.steps, "transitions": s_res["transitions"], "trajectory": s_res["trajectory"],
               "action_sources": s_res["action_sources"], "cbs_calls": base._strip(s_res["cbs_calls"])}
        base._write(d / "result.json", rec)
        brief = {"identical": identical, "full_t": full_t, "termination": s_res["termination"],
                 **{kn: {"first_t": v["first_event_t"], "n": v["n_events"], "fp": v["possible_false_positives"],
                         "events": [(r["t"], r["group"], r["key"][1], r["outside_moved_at_t"], r["phase"])
                                    for r in v["events"]]} for kn, v in per_k.items()}}
        summary["runs"][name] = brief
        print(json.dumps(base._jsonable({name: brief}), ensure_ascii=False))
    base._write(root / "summary.json", summary)
    print(f"wrote {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
