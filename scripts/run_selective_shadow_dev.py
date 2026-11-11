"""DEV: Selective SHADOW opt-in on a real-CBS Global Hybrid H3 episode (stage B).

One scenario (default: Corridor DEV), Global Hybrid with the existing DEV H3
config (``dev_resume_config(3)``), run TWICE with identical inputs:

    OFF  run_global_hybrid_episode(planner=CbsRuntimePlanner, monitor=DeadlockMonitor)
    ON   the same, with planner/monitor wrapped by ShadowSelectivePlanner /
         ShadowMonitor (src/hybrid/selective_shadow.py, unchanged)

The shadow only records (WFG, candidate groups, decision reason). It never
runs a Selective/group CBS and never changes what the Global planner gets or
returns. Nothing in the Runner, Simulator, Monitor, CBS adapter or Path Guard
is modified; everything is injected through the Runner's existing arguments.

Refused before any CBS call (exit 2, nothing written):
  * cbs_timeout_sec outside 1..30 (the existing per-call limit is 30 s);
  * wall_limit_sec <= 0 (separate whole-run limit);
  * solver root without cbs.py / a_star.py, or not a clean git checkout;
  * PYTHONHASHSEED not fixed at interpreter start-up, or not verified by
    src/compare/provenance (both OFF and ON must see the same solver hashing);
  * an existing run directory (a new unique folder is always created).

Stop rules: if the OFF episode shows a safety problem (CBS call not accepted,
Path Guard refusal, expected/actual mismatch, overlap, SAFE_STOP,
INTEGRITY_ABORT) the ON episode is NOT run. If the wall limit is exceeded
after OFF, ON is not run. The CLI also arms a hard watchdog that ends the
process at the wall limit (exit 124, ABORTED_WALL_LIMIT.txt written).

Output (only under ``<out_root>/<run_id>/``, every file created with "x"):
    off/<run_id>_steps.csv _summary.csv _monitor.csv  off/cbs/cbs_call_NN/...
    on/ (same layout)
    shadow.json       ShadowRecord list + ShadowMonitor rearms/errors
    comparison.json   verdict, every compared field, latency (reported only),
                      Path Guard re-check, preflight + environment

Latency is reported per call for OFF and ON and is NOT required to match.

Usage (repository root; PYTHONHASHSEED must be set BEFORE python starts):
    $env:PYTHONHASHSEED = "0"
    python -m scripts.run_selective_shadow_dev --solver-root <atb033>/centralized/cbs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.compare import provenance  # noqa: E402  first: captures the start-up PYTHONHASHSEED

DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs" / "selective_shadow_dev"
DEFAULT_SCENARIO = PROJECT_ROOT / "scenarios" / "controlled" / "scenario_corridor_bay_n2.json"
MODE = "GLOBAL_HYBRID_DEV_H3_SHADOW"
H_EXEC = 3                      # DEV_ONLY, same as GLOBAL_HYBRID_DEV_H3
DEFAULT_MAX_STEPS = 160         # common_runner.DEFAULT_MAX_STEPS
MAX_CBS_TIMEOUT_SEC = 30        # existing per-call limit; never raised here
DEFAULT_WALL_LIMIT_SEC = 300    # whole run (OFF + ON), separate from the per-call limit

EXIT_PASS, EXIT_FAIL, EXIT_REFUSED, EXIT_WALL = 0, 1, 2, 124

PASS = "PASS"
DIFF = "GLOBAL_DIFF"                    # Shadow ON changed a Global observable
STOPPED_OFF_SAFETY = "STOPPED_OFF_SAFETY"
STOPPED_WALL_LIMIT = "STOPPED_WALL_LIMIT"
ON_SAFETY = "ON_SAFETY"                 # ON has a safety problem (OFF did not)
SHADOW_RECORD_FAIL = "SHADOW_RECORD_FAIL"

RESULT_KEYS = ("trajectory", "actions", "action_sources", "transitions", "termination",
               "stop_reason", "safe_stop", "cbs_exec_trace", "exec_mismatch", "final_positions",
               "steps", "done", "all_at_goal", "timed_out", "integrity_abort",
               "integrity_abort_timestep", "goals")
REARM_KEYS = ("rearm_config", "monitor_rearms", "triggers", "monitor_trigger_history")
CALL_KEYS = ("trigger_timestep", "agent_order", "s_t", "status", "accepted", "reason", "error")
PLANNER_CALL_KEYS = ("status", "error", "paths", "timeout_sec", "solver_root", "pythonhashseed_env")


# -- preflight --------------------------------------------------------------------

def preflight(*, solver_root, scenario, ckpt, cbs_timeout_sec, wall_limit_sec, max_steps,
              require_solver_git=True, require_fixed_seed=True) -> tuple[Dict[str, Any], List[str]]:
    """Checks that need no CBS call. Returns (record, errors)."""
    from src.compare.common_runner import _git

    errors: List[str] = []
    if isinstance(cbs_timeout_sec, bool) or not isinstance(cbs_timeout_sec, int) \
            or not 1 <= cbs_timeout_sec <= MAX_CBS_TIMEOUT_SEC:
        errors.append(f"cbs_timeout_sec must be an int in 1..{MAX_CBS_TIMEOUT_SEC}, got {cbs_timeout_sec!r}")
    if isinstance(wall_limit_sec, bool) or not isinstance(wall_limit_sec, (int, float)) or not wall_limit_sec > 0:
        errors.append(f"wall_limit_sec must be a number > 0, got {wall_limit_sec!r}")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        errors.append(f"max_steps must be an int >= 1, got {max_steps!r}")

    solver_root = Path(solver_root)
    for name in ("cbs.py", "a_star.py"):
        if not (solver_root / name).is_file():
            errors.append(f"solver {name} not found under {solver_root}")
    solver_git = _git(solver_root) if solver_root.is_dir() else {"commit": None, "dirty": None}
    if require_solver_git:
        if solver_git["commit"] is None:
            errors.append(f"solver root is not a git checkout: {solver_root}")
        elif solver_git["dirty"] is not False:
            errors.append(f"solver checkout is dirty: {solver_root}")
    for label, p in (("scenario", scenario), ("checkpoint", ckpt)):
        if not Path(p).is_file():
            errors.append(f"{label} not found: {p}")

    parent = provenance.hash_seed_info()
    layers = None
    if require_fixed_seed:
        if parent["mode"] not in provenance.FIXED_MODES:
            errors.append(f"PYTHONHASHSEED not fixed at start-up ({parent['mode']}); "
                          "set it before python starts")
        else:
            layers = provenance.verify_seed_layers(parent)
            errors += [f"hash seed: {r}" for r in layers["reasons"]]
    record = {
        "solver_root": str(solver_root.resolve()) if solver_root.exists() else str(solver_root),
        "solver_git": solver_git,
        "project_git": _git(PROJECT_ROOT),
        "hash_seed": parent,
        "hash_seed_layers": layers,
        "cbs_timeout_sec": cbs_timeout_sec,
        "wall_limit_sec": wall_limit_sec,
        "max_steps": max_steps,
        "h_exec": H_EXEC,
        "config": "scripts.run_hybrid_resume_dev.dev_resume_config(3) (DEV_ONLY)",
        "require_solver_git": require_solver_git,
        "require_fixed_seed": require_fixed_seed,
    }
    return record, errors


# -- one episode --------------------------------------------------------------------

def run_episode(*, shadow, grid, starts, goals, solver_root, cbs_timeout_sec, max_steps, ckpt,
                ep_dir: Path, run_id, scenario_name) -> Dict[str, Any]:
    from scripts.run_hybrid_resume_dev import dev_resume_config
    from src.common.deadlock_monitor import DeadlockMonitor
    from src.common.episode_logger import EpisodeLogger
    from src.hybrid.cbs_runtime_bridge import CbsRuntimePlanner, PathGuardValidator
    from src.hybrid.global_hybrid import run_global_hybrid_episode
    from src.hybrid.selective_shadow import ShadowMonitor, ShadowSelectivePlanner
    from src.il.policy import NavHintPolicy

    ep_dir.mkdir(parents=True, exist_ok=False)
    logger = EpisodeLogger(run_id, scenario=scenario_name, mode=MODE, out_dir=ep_dir)
    inner = CbsRuntimePlanner(solver_root=solver_root, work_root=ep_dir / "cbs", timeout_sec=cbs_timeout_sec)
    monitor = DeadlockMonitor()
    planner = inner
    if shadow:
        monitor = ShadowMonitor(monitor)
        planner = ShadowSelectivePlanner(inner, monitor)
    t0 = time.monotonic()
    result = run_global_hybrid_episode(
        NavHintPolicy(ckpt), grid, starts, goals, max_steps,
        planner=planner, path_validator=PathGuardValidator(), config=dev_resume_config(H_EXEC),
        monitor=monitor, logger=logger)
    wall_sec = time.monotonic() - t0
    files = {
        "steps": logger.write_steps(),
        "summary": logger.write_summary(num_agents=len(goals), steps=result["steps"],
                                        success=result["all_at_goal"], timed_out=result["timed_out"],
                                        makespan=result["steps"]),
        "monitor": logger.write_monitor(monitor.result),
    }
    return {"result": result, "inner": inner, "planner": planner, "monitor": monitor,
            "rows": logger.step_rows, "files": files, "dir": ep_dir, "wall_sec": wall_sec}


# -- checks -------------------------------------------------------------------------

def _artifacts(inner) -> List[Dict[str, bytes]]:
    out = []
    for c in inner.calls:
        d = Path(c["artifacts_dir"])
        out.append({p.relative_to(d).as_posix(): p.read_bytes()
                    for p in sorted(d.rglob("*")) if p.is_file()} if d.is_dir() else {})
    return out


def path_guard_recheck(ep, grid, goals) -> List[Dict[str, Any]]:
    """Re-run the real PathGuardValidator on every returned plan (record only)."""
    from src.hybrid.cbs_runtime_bridge import PathGuardValidator

    checks = []
    for i, (call, pcall) in enumerate(zip(ep["result"]["cbs_calls"], ep["inner"].calls), start=1):
        paths = pcall.get("paths")
        if paths is None:
            checks.append({"call": i, "ok": None, "reason": f"no paths (status={pcall.get('status')})"})
            continue
        ok, reason = PathGuardValidator()(grid, call["s_t"], goals, paths)
        checks.append({"call": i, "ok": bool(ok), "reason": reason})
    return checks


def safety_problems(ep, grid, goals) -> List[str]:
    r = ep["result"]
    problems = []
    if r["integrity_abort"]:
        problems.append(f"INTEGRITY_ABORT at t={r['integrity_abort_timestep']}")
    for t, state in enumerate(r["trajectory"]):
        if len(set(state.values())) != len(state):
            problems.append(f"position overlap at t={t}")
    if r["exec_mismatch"] is not None or any(not e["match"] for e in r["cbs_exec_trace"]):
        problems.append(f"expected/actual mismatch: {r['exec_mismatch']}")
    if any(e["overlap"] for e in r["cbs_exec_trace"]):
        problems.append("overlap during CBS_EXEC")
    for i, c in enumerate(r["cbs_calls"], start=1):
        if not c["accepted"]:
            problems.append(f"CBS call {i} not accepted: status={c['status']} reason={c['reason']} "
                            f"error={c['error']}")
    for chk in path_guard_recheck(ep, grid, goals):
        if chk["ok"] is False:
            problems.append(f"Path Guard re-check failed on call {chk['call']}: {chk['reason']}")
    if r["termination"] in ("SAFE_STOP", "INTEGRITY_ABORT"):
        problems.append(f"termination={r['termination']} stop_reason={r['stop_reason']}")
    return problems


def compare_global(off, on) -> List[str]:
    """Every Global observable that differs between OFF and ON (empty = identical)."""
    diffs = []
    ro, rn = off["result"], on["result"]
    for k in RESULT_KEYS:
        if ro.get(k) != rn.get(k):
            diffs.append(f"result.{k}")
    for k in REARM_KEYS:
        if (k in ro) != (k in rn) or ro.get(k) != rn.get(k):
            diffs.append(f"result.{k}")
    if len(ro["cbs_calls"]) != len(rn["cbs_calls"]):
        diffs.append(f"cbs_calls count {len(ro['cbs_calls'])} != {len(rn['cbs_calls'])}")
    for i, (a, b) in enumerate(zip(ro["cbs_calls"], rn["cbs_calls"]), start=1):
        for k in CALL_KEYS:
            if a[k] != b[k]:
                diffs.append(f"cbs_calls[{i}].{k}")
    pi, pn = off["inner"].calls, on["inner"].calls
    if len(pi) != len(pn):
        diffs.append(f"planner call count {len(pi)} != {len(pn)}")
    for i, (a, b) in enumerate(zip(pi, pn), start=1):
        for k in PLANNER_CALL_KEYS:
            if a.get(k) != b.get(k):
                diffs.append(f"planner_calls[{i}].{k}")
    for i, (a, b) in enumerate(zip(_artifacts(off["inner"]), _artifacts(on["inner"])), start=1):
        if sorted(a) != sorted(b):
            diffs.append(f"solver artifacts[{i}] file set")
        for name in sorted(set(a) & set(b)):
            if a[name] != b[name]:
                diffs.append(f"solver artifacts[{i}] {name}")
    mo, mn = off["monitor"], on["monitor"]
    if mo.result != mn.result:
        diffs.append("monitor.result")
    if mo.trigger_history != mn.trigger_history:
        diffs.append("monitor.trigger_history")
    if mo.rearm_count != mn.rearm_count:
        diffs.append("monitor.rearm_count")
    if off["rows"] != on["rows"]:
        diffs.append("logger step rows")
    for name in ("steps", "summary", "monitor"):
        if Path(off["files"][name]).read_bytes() != Path(on["files"][name]).read_bytes():
            diffs.append(f"csv {name}")
    return diffs


def check_shadow_records(on, grid, goals) -> List[str]:
    """The shadow's own records against the ON episode (time axis, WFG, counts)."""
    from src.hybrid.selective_decision import ROUTE_GLOBAL, decide_selective
    from src.hybrid.wfg import build_wfg

    w, res = on["planner"], on["result"]
    calls, traj = res["cbs_calls"], res["trajectory"]
    problems = []
    if not (len(w.records) == len(calls) == len(on["inner"].calls)):
        problems.append(f"record/call/inner counts {len(w.records)}/{len(calls)}/{len(on['inner'].calls)}")
    ids = tuple(sorted(goals))
    for i, (rec, call) in enumerate(zip(w.records, calls), start=1):
        if rec.shadow_error is not None:
            problems.append(f"record {i} shadow_error={rec.shadow_error}")
            continue
        t = rec.trigger_t
        checks = {
            "trigger_t": t == call["trigger_timestep"],
            "s_t == call s_t": rec.s_t == call["s_t"],
            "s_t == trajectory[t]": rec.s_t == traj[t],
            "s_t_minus_1 == trajectory[t-1]": rec.s_t_minus_1 == traj[t - 1],
            "wfg recomputed": rec.wfg == build_wfg(rec.s_t_minus_1, rec.intended, rec.blocked, grid),
            "decision recomputed": rec.decision == decide_selective(ids, rec.s_t, goals, rec.wfg),
            "route GLOBAL": rec.decision.route == ROUTE_GLOBAL,
            "global_status": rec.global_status == call["status"],
        }
        problems += [f"record {i}: {name}" for name, ok in checks.items() if not ok]
    return problems


def shadow_payload(on) -> Dict[str, Any]:
    w, mon = on["planner"], on["monitor"]
    records = []
    for rec in w.records:
        d = asdict(rec)
        d["selective_candidate_note"] = (
            "CANDIDATE ONLY: v0 routes GLOBAL; no Selective/group CBS was run"
            if rec.decision is not None and rec.decision.selective_candidate else None)
        records.append(d)
    return {"rule_version": w.records[0].rule_version if w.records else None,
            "records": records, "monitor_rearms": mon.rearms, "monitor_record_errors": mon.record_errors,
            "episode": mon.episode}


def _latency(ep) -> List[Optional[float]]:
    return [c["latency_ms"] for c in ep["result"]["cbs_calls"]]


# -- whole run ----------------------------------------------------------------------

def run_shadow_dev(*, solver_root, out_root, scenario=DEFAULT_SCENARIO, ckpt=None,
                   cbs_timeout_sec=MAX_CBS_TIMEOUT_SEC, wall_limit_sec=DEFAULT_WALL_LIMIT_SEC,
                   max_steps=DEFAULT_MAX_STEPS, require_solver_git=True, require_fixed_seed=True,
                   run_id=None, clock=time.monotonic, command=None) -> Dict[str, Any]:
    """Preflight, OFF, ON, compare, write. Returns the comparison dict
    (``verdict`` / ``exit_code``). Refusals raise nothing and write nothing."""
    from scripts.run_il import DEFAULT_CKPT, load_scenario
    from src.common.episode_logger import make_run_id
    from src.compare.common_runner import _jsonable, _sha256_file, _sha256_grid, environment_info

    start = clock()
    ckpt = Path(ckpt) if ckpt is not None else DEFAULT_CKPT
    scenario = Path(scenario)
    pre, errors = preflight(solver_root=solver_root, scenario=scenario, ckpt=ckpt,
                            cbs_timeout_sec=cbs_timeout_sec, wall_limit_sec=wall_limit_sec,
                            max_steps=max_steps, require_solver_git=require_solver_git,
                            require_fixed_seed=require_fixed_seed)
    run_id = run_id or make_run_id(scenario.name, MODE)
    run_dir = Path(out_root).resolve() / run_id
    if run_dir.exists():
        errors.append(f"run directory already exists: {run_dir}")
    if errors:
        return {"verdict": "REFUSED", "exit_code": EXIT_REFUSED, "errors": errors, "preflight": pre,
                "run_dir": None}

    grid, starts, goals = load_scenario(scenario)
    run_dir.mkdir(parents=True, exist_ok=False)
    common = dict(grid=grid, starts=starts, goals=goals, solver_root=solver_root,
                  cbs_timeout_sec=cbs_timeout_sec, max_steps=max_steps, ckpt=ckpt, run_id=run_id,
                  scenario_name=scenario.name)

    off = run_episode(shadow=False, ep_dir=run_dir / "off", **common)
    off_safety = safety_problems(off, grid, goals)
    on = None
    verdict, problems, diffs, shadow_problems, on_safety = PASS, [], [], [], []
    if off_safety:
        verdict, problems = STOPPED_OFF_SAFETY, off_safety
    elif clock() - start > wall_limit_sec:
        verdict, problems = STOPPED_WALL_LIMIT, [f"wall limit {wall_limit_sec}s exceeded after OFF"]
    else:
        on = run_episode(shadow=True, ep_dir=run_dir / "on", **common)
        diffs = compare_global(off, on)
        on_safety = safety_problems(on, grid, goals)
        shadow_problems = check_shadow_records(on, grid, goals)
        if diffs:
            verdict = DIFF
        elif on_safety:
            verdict = ON_SAFETY
        elif shadow_problems:
            verdict = SHADOW_RECORD_FAIL
        problems = diffs + on_safety + shadow_problems

    summary = lambda ep: None if ep is None else {  # noqa: E731
        "termination": ep["result"]["termination"], "stop_reason": ep["result"]["stop_reason"],
        "steps": ep["result"]["steps"], "all_at_goal": ep["result"]["all_at_goal"],
        "cbs_calls": [{k: c[k] for k in CALL_KEYS} for c in ep["result"]["cbs_calls"]],
        "planner_call_count": len(ep["inner"].calls),
        "path_guard_recheck": path_guard_recheck(ep, grid, goals),
        "monitor": {"trigger_reason_code": ep["monitor"].result.trigger_reason_code,
                    "trigger_timestep": ep["monitor"].result.trigger_timestep,
                    "trigger_history": [asdict(h) for h in ep["monitor"].trigger_history],
                    "rearm_count": ep["monitor"].rearm_count},
        "episode_wall_sec": ep["wall_sec"],
    }
    comparison = {
        "verdict": verdict,
        "exit_code": EXIT_PASS if verdict == PASS else EXIT_FAIL,
        "problems": problems,
        "global_diffs": diffs,
        "off_safety": off_safety,
        "on_safety": on_safety,
        "shadow_record_problems": shadow_problems,
        "compared": {"result_keys": RESULT_KEYS, "rearm_keys": REARM_KEYS, "cbs_call_keys": CALL_KEYS,
                     "planner_call_keys": PLANNER_CALL_KEYS,
                     "also": ["solver artifact bytes (input.yaml, solver_input.yaml, output.yaml)",
                              "monitor result / trigger_history / rearm_count",
                              "logger step rows", "steps/summary/monitor CSV bytes"]},
        "latency_ms_reported_not_compared": {"off": _latency(off), "on": None if on is None else _latency(on)},
        "off": summary(off),
        "on": summary(on),
        "shadow_records_summary": None if on is None else [
            {"call_index": r.call_index, "trigger_t": r.trigger_t, "shadow_error": r.shadow_error,
             "wfg_groups": None if r.wfg is None else r.wfg.groups,
             "decision_reason": None if r.decision is None else r.decision.reason,
             "selective_candidate": None if r.decision is None else r.decision.selective_candidate,
             "route": None if r.decision is None else r.decision.route,
             "l0_group": r.l0_group, "outside": r.outside, "global_status": r.global_status}
            for r in on["planner"].records],
        "selective_group_cbs_executed": False,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "mode": MODE,
        "scenario": str(scenario),
        "scenario_sha256": _sha256_file(scenario),
        "grid_sha256": _sha256_grid(grid),
        "checkpoint": str(ckpt),
        "checkpoint_sha256": _sha256_file(ckpt),
        "preflight": pre,
        "environment": environment_info(Path(solver_root)),
        "command": command,
        "total_wall_sec": clock() - start,
    }
    if on is not None:
        with (run_dir / "shadow.json").open("x", encoding="utf-8") as fh:
            json.dump(_jsonable(shadow_payload(on)), fh, indent=2, ensure_ascii=False)
            fh.write("\n")
    with (run_dir / "comparison.json").open("x", encoding="utf-8") as fh:
        json.dump(_jsonable(comparison), fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return comparison


def _arm_watchdog(wall_limit_sec, out_root, run_id):
    """Hard stop for the CLI: end the process at the wall limit."""
    def fire():
        try:
            d = Path(out_root).resolve() / run_id
            if d.is_dir():
                with (d / "ABORTED_WALL_LIMIT.txt").open("x", encoding="utf-8") as fh:
                    fh.write(f"wall limit {wall_limit_sec}s reached; process ended (exit {EXIT_WALL})\n")
        finally:
            print(f"ABORTED: wall limit {wall_limit_sec}s reached", flush=True)
            os._exit(EXIT_WALL)
    timer = threading.Timer(wall_limit_sec, fire)
    timer.daemon = True
    timer.start()
    return timer


def main(argv=None):
    from src.common.episode_logger import make_run_id

    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solver-root", type=Path, required=True,
                    help="atb033 .../multi_agent_path_planning/centralized/cbs (explicit, no auto-search)")
    ap.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--cbs-timeout-sec", type=int, default=MAX_CBS_TIMEOUT_SEC,
                    help=f"per CBS call, 1..{MAX_CBS_TIMEOUT_SEC}")
    ap.add_argument("--wall-limit-sec", type=float, default=DEFAULT_WALL_LIMIT_SEC,
                    help="whole run (OFF + ON); hard stop")
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--preflight-only", action="store_true", help="run the checks, call no CBS, write nothing")
    args = ap.parse_args(argv)

    if args.preflight_only:
        from scripts.run_il import DEFAULT_CKPT
        pre, errors = preflight(solver_root=args.solver_root, scenario=args.scenario,
                                ckpt=args.ckpt or DEFAULT_CKPT, cbs_timeout_sec=args.cbs_timeout_sec,
                                wall_limit_sec=args.wall_limit_sec, max_steps=args.max_steps)
        print(json.dumps({"preflight_errors": errors, **{k: pre[k] for k in
                          ("solver_root", "solver_git", "project_git", "cbs_timeout_sec",
                           "wall_limit_sec", "max_steps", "h_exec")},
                          "hash_seed_mode": pre["hash_seed"]["mode"],
                          "hash_seed_parent_verified": (pre["hash_seed_layers"] or {}).get("parent_verified"),
                          "hash_seed_inherited": (pre["hash_seed_layers"] or {}).get("inherited_status")},
                         indent=2, default=str))
        return EXIT_REFUSED if errors else EXIT_PASS

    run_id = make_run_id(args.scenario.name, MODE)
    timer = _arm_watchdog(args.wall_limit_sec, args.out_root, run_id) \
        if isinstance(args.wall_limit_sec, (int, float)) and args.wall_limit_sec > 0 else None
    try:
        comp = run_shadow_dev(solver_root=args.solver_root, out_root=args.out_root, scenario=args.scenario,
                              ckpt=args.ckpt, cbs_timeout_sec=args.cbs_timeout_sec,
                              wall_limit_sec=args.wall_limit_sec, max_steps=args.max_steps, run_id=run_id,
                              command=" ".join([sys.executable, "-m", "scripts.run_selective_shadow_dev", *argv]))
    finally:
        if timer is not None:
            timer.cancel()

    if comp["verdict"] == "REFUSED":
        print("REFUSED (no CBS call, nothing written):")
        for e in comp["errors"]:
            print(f"  - {e}")
        return EXIT_REFUSED
    print(f"run_dir   : {comp['run_dir']}")
    for side in ("off", "on"):
        s = comp[side]
        if s is None:
            print(f"{side.upper():4s}      : not run")
            continue
        print(f"{side.upper():4s}      : termination={s['termination']} stop_reason={s['stop_reason']} "
              f"steps={s['steps']} planner_calls={s['planner_call_count']} "
              f"calls={[(c['trigger_timestep'], c['status'], c['accepted']) for c in s['cbs_calls']]} "
              f"path_guard={[c['ok'] for c in s['path_guard_recheck']]}")
    print(f"latency_ms (reported only): {comp['latency_ms_reported_not_compared']}")
    for r in comp["shadow_records_summary"] or []:
        print(f"shadow    : call={r['call_index']} t={r['trigger_t']} groups={r['wfg_groups']} "
              f"reason={r['decision_reason']} route={r['route']} error={r['shadow_error']}")
    print(f"VERDICT   : {comp['verdict']}")
    for p in comp["problems"]:
        print(f"  - {p}")
    print(f"total_wall_sec={comp['total_wall_sec']:.2f}")
    return comp["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
