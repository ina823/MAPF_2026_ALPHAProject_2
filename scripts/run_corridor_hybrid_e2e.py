"""DEV: first automatic Global Hybrid E2E on Corridor with the real CBS.

Real IL (NavHint checkpoint) + MAPFStepSimulator + DeadlockMonitor ->
first fresh TRIGGER_FULL_REPEAT -> S_t -> run_cbs_runtime() (atb033 CBS +
Path Guard) -> boundary (real Path Guard again) -> execute from Path[1],
comparing expected/actual positions after every step.

DEV ONLY -- the policies below are NOT the agreed project rules (H_exec, IL
resume, CBS recall and SAFE_STOP are still open with the CBS side):
  * h_exec = max_steps, i.e. the whole accepted CBS path is executed;
  * at most one CBS call per episode;
  * if the path ends without all agents at their goals -> SAFE_STOP.

Every run writes only to a new directory (it refuses to reuse one):
    outputs/hybrid_e2e/<run_id>/result.json        -- verdict + full record
    outputs/hybrid_e2e/<run_id>/<run_id>_steps.csv -- EpisodeLogger step rows
    outputs/hybrid_e2e/<run_id>/cbs/cbs_call_NN/   -- solver YAML files

Usage (from the repository root):
    python -m scripts.run_corridor_hybrid_e2e --solver-root <atb033>/centralized/cbs
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_il import DEFAULT_CKPT, load_scenario
from src.common.deadlock_monitor import DeadlockMonitor
from src.hybrid.cbs_runtime_bridge import CbsRuntimePlanner, PathGuardValidator
from src.hybrid.global_hybrid import SAFE_STOP, SUCCESS, GlobalHybridConfig, run_global_hybrid_episode
from src.simulator.mapf_step_simulator import MAPFStepSimulator

DEFAULT_SCENARIO = PROJECT_ROOT / "scenarios" / "controlled" / "scenario_corridor_bay_n2.json"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs" / "hybrid_e2e"
DEFAULT_MAX_STEPS = 160
MODE = "GLOBAL_HYBRID_DEV"


# -- DEV-only policies (temporary, NOT project defaults) ----------------------

class DevStopAfterPath:
    """DEV_ONLY resume policy: never return to IL after the CBS path."""

    def after_exec(self, context):
        return SAFE_STOP


class DevSingleCbsCall:
    """DEV_ONLY recall policy: the first fresh Trigger only."""

    def allow_call(self, call_count, trigger_timestep):
        return call_count == 0


class DevRecordSafeStop:
    """DEV_ONLY SAFE_STOP record: reason, timestep and positions."""

    def on_safe_stop(self, reason, context):
        return {
            "reason": reason,
            "t": context["t"],
            "positions": {aid: tuple(p) for aid, p in context["positions"].items()},
        }


def dev_config(max_steps):
    return GlobalHybridConfig(
        h_exec=max_steps,  # DEV_ONLY: covers any CBS path inside the episode
        resume_policy=DevStopAfterPath(),
        recall_policy=DevSingleCbsCall(),
        safe_stop_handler=DevRecordSafeStop(),
    )


def run_corridor_e2e(planner, *, scenario=DEFAULT_SCENARIO, max_steps=DEFAULT_MAX_STEPS,
                     ckpt=DEFAULT_CKPT, logger=None, simulator_cls=MAPFStepSimulator):
    from src.il.policy import NavHintPolicy

    grid, starts, goals = load_scenario(scenario)
    return run_global_hybrid_episode(
        NavHintPolicy(ckpt), grid, starts, goals, max_steps,
        planner=planner,
        path_validator=PathGuardValidator(),
        config=dev_config(max_steps),
        monitor=DeadlockMonitor(),
        logger=logger,
        simulator_cls=simulator_cls,
    )


# -- Verdict ------------------------------------------------------------------

def evaluate(result, planner_calls):
    """Ordered success checks; the first failing one is the reported cause."""
    calls = result["cbs_calls"]
    call = calls[0] if calls else {}
    paths = (planner_calls[0].get("paths") if planner_calls else None) or {}
    trace = result["cbs_exec_trace"]
    trigger_t = call.get("trigger_timestep")
    path1 = {aid: tuple(p[1]) for aid, p in paths.items() if len(p) > 1}
    horizon = (max(len(p) for p in paths.values()) - 1) if paths else None

    checks = [
        ("cbs_called_once_on_trigger", len(calls) == 1),
        ("s_t_equals_simulator_state",
         trigger_t is not None and call["s_t"] == result["trajectory"][trigger_t]),
        ("cbs_status_success", call.get("status") == "success"),
        ("boundary_path_guard_accepted", call.get("accepted") is True),
        ("path0_equals_s_t",
         bool(paths) and {aid: tuple(p[0]) for aid, p in paths.items()} == call.get("s_t")),
        ("first_cbs_step_reaches_path1",
         bool(trace) and trace[0]["expected"] == path1 and trace[0]["actual"] == path1),
        ("whole_path_executed", horizon is not None and len(trace) == horizon),
        ("expected_equals_actual_every_step", bool(trace) and all(e["match"] for e in trace)),
        ("no_overlap_any_step",
         all(len(set(state.values())) == len(state) for state in result["trajectory"])),
        ("all_at_goal", result["all_at_goal"]),
        ("termination_success", result["termination"] == SUCCESS),
    ]
    first_failure = next((name for name, ok in checks if not ok), None)
    return {
        "verdict": "PASS" if first_failure is None else "FAIL",
        "first_failed_check": first_failure,
        "checks": [{"name": name, "ok": bool(ok)} for name, ok in checks],
        "path1": path1,
    }


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    return obj


def _git_head(path):
    try:
        head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return {"commit": head, "dirty": bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def build_report(result, planner, evaluation, *, scenario, max_steps, ckpt, reproduce):
    transitions = result["transitions"]
    first_trigger = next(({"t": tr["t"], "reason": tr["reason"]}
                          for tr in transitions if tr["to"] == "TRIGGER"), None)
    calls = []
    for call, pcall in zip(result["cbs_calls"], planner.calls):
        calls.append({
            **call,
            "artifacts_dir": pcall["artifacts_dir"],
            "paths": pcall.get("paths"),
        })
    final_t = result["steps"]
    return {
        "mode": MODE,
        "dev_only_policies": {
            "h_exec": f"max_steps={max_steps} (whole CBS path, DEV only)",
            "resume": "SAFE_STOP after the path (no IL resume, DEV only)",
            "recall": "first Trigger only (DEV only)",
        },
        "verdict": evaluation["verdict"],
        "first_failed_check": evaluation["first_failed_check"],
        "checks": evaluation["checks"],
        "reproduce": reproduce,
        "scenario": str(scenario),
        "checkpoint": str(ckpt),
        "max_steps": max_steps,
        "solver_root": str(planner.solver_root),
        "solver_git": _git_head(planner.solver_root),
        "project_git": _git_head(PROJECT_ROOT),
        "first_trigger": first_trigger,
        "s_t": calls[0]["s_t"] if calls else None,
        "cbs_calls": calls,
        "path1": evaluation["path1"],
        "cbs_exec_trace": result["cbs_exec_trace"],
        "exec_mismatch": result["exec_mismatch"],
        "overlap_any_step": not next(c["ok"] for c in evaluation["checks"]
                                     if c["name"] == "no_overlap_any_step"),
        "all_at_goal": result["all_at_goal"],
        "termination": result["termination"],
        "stop_reason": result["stop_reason"],
        "safe_stop": result["safe_stop"],
        "last_state": {"t": final_t, "positions": result["final_positions"]},
        "goals": result["goals"],
        "steps": final_t,
        "action_sources": result["action_sources"],
        "transitions": transitions,
        "trajectory": result["trajectory"],
    }


def main(argv=None):
    from src.common.episode_logger import EpisodeLogger, make_run_id

    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solver-root", type=Path, required=True,
                    help="atb033 .../multi_agent_path_planning/centralized/cbs (explicit, no auto-search)")
    ap.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--timeout-sec", type=int, default=30)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = ap.parse_args(argv)

    run_id = make_run_id(args.scenario.name, MODE)
    run_dir = args.out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)  # never reuse a run directory

    planner = CbsRuntimePlanner(solver_root=args.solver_root, work_root=run_dir / "cbs",
                                timeout_sec=args.timeout_sec)
    logger = EpisodeLogger(run_id, scenario=args.scenario.name, mode=MODE, out_dir=run_dir)
    result = run_corridor_e2e(planner, scenario=args.scenario, max_steps=args.max_steps,
                              ckpt=args.ckpt, logger=logger)
    logger.write_steps()

    evaluation = evaluate(result, planner.calls)
    reproduce = " ".join([sys.executable, "-m", "scripts.run_corridor_hybrid_e2e", *argv])
    report = build_report(result, planner, evaluation, scenario=args.scenario,
                          max_steps=args.max_steps, ckpt=args.ckpt, reproduce=reproduce)
    report_path = run_dir / "result.json"
    with report_path.open("x", encoding="utf-8") as fh:
        json.dump(_jsonable(report), fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print(f"run_dir        : {run_dir}")
    print(f"first trigger  : {report['first_trigger']}  S_t={report['s_t']}")
    for call in report["cbs_calls"]:
        print(f"cbs call       : status={call['status']} accepted={call['accepted']} "
              f"reason={call['reason']!r} latency_ms={call['latency_ms']} error={call['error']!r}")
    print(f"Path[1]        : {report['path1']}")
    for e in report["cbs_exec_trace"]:
        print(f"  t={e['t']:3d} k={e['k']:2d} expected={e['expected']} actual={e['actual']} "
              f"match={e['match']} overlap={e['overlap']}")
    print(f"termination    : {report['termination']}  stop_reason={report['stop_reason']}  "
          f"all_at_goal={report['all_at_goal']}  steps={report['steps']}")
    print(f"VERDICT        : {report['verdict']}"
          + (f"  (first failed check: {report['first_failed_check']})" if report["first_failed_check"] else ""))
    print(f"reproduce      : {reproduce}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
