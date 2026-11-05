"""DEV: Global Hybrid E2E with a partial CBS window and a real IL resume.

Real IL (NavHint checkpoint) + MAPFStepSimulator + DeadlockMonitor ->
first fresh TRIGGER_FULL_REPEAT -> S_t -> run_cbs_runtime() (atb033 CBS +
Path Guard) -> boundary (real Path Guard again) -> execute H_exec CBS steps
from Path[1] (expected/actual compared after every step) -> IL_RESUME ->
real policy.act() + sim.step() until the episode ends.

DEV ONLY -- the policies below are NOT the agreed project rules (H_exec, IL
resume, CBS recall and SAFE_STOP are still open with the CBS side):
  * h_exec = 3 by default (a development assumption, not an operating value);
  * after the CBS window the episode always resumes IL;
  * at most one CBS call per episode. The DeadlockMonitor also latches its
    first TRIGGER_FULL_REPEAT, so a re-stall after the resume is never
    re-triggered -- this script detects it afterwards from the trajectory
    and records it, without changing the Monitor.

"Reached IL_RESUME" and "really ran IL again" are recorded separately: the
first is the CBS_EXEC -> IL_RESUME transition, the second counts the
policy.act() and sim.step() calls made after it.

Every run writes only to a new directory (it refuses to reuse one):
    outputs/hybrid_e2e/<run_id>/result.json        -- verdict + full record
    outputs/hybrid_e2e/<run_id>/<run_id>_steps.csv -- EpisodeLogger step rows
    outputs/hybrid_e2e/<run_id>/cbs/cbs_call_NN/   -- solver YAML files

Usage (from the repository root):
    python -m scripts.run_hybrid_resume_dev --solver-root <atb033>/centralized/cbs \
        --scenario scenarios/controlled/scenario_intersection_n2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_corridor_hybrid_e2e import (
    DEFAULT_MAX_STEPS,
    DEFAULT_OUT_ROOT,
    DEFAULT_SCENARIO,
    DevRecordSafeStop,
    DevSingleCbsCall,
    _git_head,
    _jsonable,
)
from scripts.run_il import DEFAULT_CKPT, load_scenario
from src.common.deadlock_monitor import DeadlockMonitor
from src.hybrid.cbs_runtime_bridge import CbsRuntimePlanner, PathGuardValidator
from src.hybrid.global_hybrid import (
    CBS_EXEC,
    IL,
    IL_RESUME,
    INTEGRITY_ABORT,
    SUCCESS,
    TIMEOUT,
    TRIGGER,
    GlobalHybridConfig,
    MonitorRearmConfig,
    run_global_hybrid_episode,
)
from src.simulator.mapf_step_simulator import MAPFStepSimulator

DEFAULT_H_EXEC = 3  # DEV_ONLY assumption, not the agreed H_exec

# Resume outcomes (DEV record only)
SUCCESS_WITHIN_H_EXEC = "SUCCESS_WITHIN_H_EXEC"  # goal reached inside the CBS window
RESUME_SUCCESS = "RESUME_SUCCESS"
RESUME_TIMEOUT_RESTALL = "RESUME_TIMEOUT_RESTALL"  # joint state repeated after resume, no new Trigger
RESUME_TIMEOUT_NO_REPEAT = "RESUME_TIMEOUT_NO_REPEAT"
RESUME_INTEGRITY_ABORT = "RESUME_INTEGRITY_ABORT"
RESUME_NO_IL_STEP = "RESUME_NO_IL_STEP"  # IL_RESUME reached, but IL never moved again
NO_TRIGGER = "NO_TRIGGER"


def mode_name(h_exec):
    return f"GLOBAL_HYBRID_DEV_H{h_exec}"


# -- DEV-only policies (temporary, NOT project defaults) ----------------------

class DevResumeIL:
    """DEV_ONLY resume policy: always hand control back to IL after the window."""

    def after_exec(self, context):
        return IL


def dev_resume_config(h_exec=DEFAULT_H_EXEC):
    return GlobalHybridConfig(
        h_exec=h_exec,  # DEV_ONLY
        resume_policy=DevResumeIL(),
        recall_policy=DevSingleCbsCall(),
        safe_stop_handler=DevRecordSafeStop(),
    )


# -- DEV_ONLY rearm candidate (GLOBAL_HYBRID_DEV_H3_REARM) ---------------------
# Candidate values for a development trial, NOT agreed project policy.
DEV_REARM_MAX_CBS_CALLS = 2            # first call included
DEV_REARM_COOLDOWN_STEPS = 2           # IL steps since the last resume before a recall
DEV_REARM_CBS_TIME_BUDGET_MS = 60_000.0  # summed recorded latency_ms of earlier calls


class DevLimitedRecall:
    """DEV_ONLY recall policy for the rearm trial: always allows; the call
    limit, budget, cooldown and same-S_t checks are MonitorRearmConfig's."""

    def allow_call(self, call_count, trigger_timestep):
        return True


def dev_rearm_config(h_exec=DEFAULT_H_EXEC, *, max_cbs_calls=DEV_REARM_MAX_CBS_CALLS,
                     cooldown_steps=DEV_REARM_COOLDOWN_STEPS,
                     cbs_time_budget_ms=DEV_REARM_CBS_TIME_BUDGET_MS):
    return GlobalHybridConfig(
        h_exec=h_exec,  # DEV_ONLY
        resume_policy=DevResumeIL(),
        recall_policy=DevLimitedRecall(),
        safe_stop_handler=DevRecordSafeStop(),
        rearm=MonitorRearmConfig(  # DEV_ONLY candidate values
            max_cbs_calls=max_cbs_calls,
            cooldown_steps=cooldown_steps,
            cbs_time_budget_ms=cbs_time_budget_ms,
        ),
    )


# -- Call recorders (observe only, never change behaviour) ---------------------

class ActRecordingPolicy:
    """Wraps the IL policy; records the positions every act() call saw."""

    def __init__(self, inner):
        self._inner = inner
        self.act_positions = []

    def __getattr__(self, name):  # _dist etc. (used by the runner's logging)
        return getattr(self._inner, name)

    def reset(self, grid, goals):
        self.act_positions = []
        return self._inner.reset(grid, goals)

    def act(self, obs, positions):
        self.act_positions.append({aid: tuple(p) for aid, p in positions.items()})
        return self._inner.act(obs, positions)


def step_recording_simulator(base=MAPFStepSimulator):
    """A fresh simulator subclass that records the t of every step() call."""

    class _StepRecordingSimulator(base):
        step_ts = []

        def step(self, actions):
            obs, done, info = super().step(actions)
            type(self).step_ts.append(info["t"])
            return obs, done, info

    _StepRecordingSimulator.step_ts = []
    return _StepRecordingSimulator


def run_resume_e2e(planner, *, scenario=DEFAULT_SCENARIO, h_exec=DEFAULT_H_EXEC,
                   max_steps=DEFAULT_MAX_STEPS, ckpt=DEFAULT_CKPT, logger=None,
                   simulator_cls=MAPFStepSimulator, policy=None):
    """Returns (result, policy_recorder, simulator_cls_used)."""
    if policy is None:
        from src.il.policy import NavHintPolicy
        policy = NavHintPolicy(ckpt)
    recorder = ActRecordingPolicy(policy)
    sim_cls = step_recording_simulator(simulator_cls)
    grid, starts, goals = load_scenario(scenario)
    result = run_global_hybrid_episode(
        recorder, grid, starts, goals, max_steps,
        planner=planner,
        path_validator=PathGuardValidator(),
        config=dev_resume_config(h_exec),
        monitor=DeadlockMonitor(),
        logger=logger,
        simulator_cls=sim_cls,
    )
    return result, recorder, sim_cls


# -- Analysis -----------------------------------------------------------------

def _joint(state):
    return tuple(tuple(state[aid]) for aid in sorted(state))


def find_repeat_after(trajectory, start_t, goals):
    """First joint-state repeat at t > start_t among trajectory[start_t:],
    with the Monitor's rule (sorted-agent joint state, goal state excluded).
    Returns {"t", "first_seen", "period"} or None."""
    goal_state = _joint(goals)
    seen = {}
    for t in range(start_t, len(trajectory)):
        state = _joint(trajectory[t])
        if state == goal_state:
            continue
        if state in seen:
            return {"t": t, "first_seen": seen[state], "period": t - seen[state]}
        seen[state] = t
    return None


def analyze_resume(result, act_positions, sim_step_ts):
    """IL_RESUME record: reaching the state vs. really running IL again."""
    transitions = result["transitions"]
    sources = result["action_sources"]
    resume_tr = next((tr for tr in transitions
                      if tr["from"] == CBS_EXEC and tr["to"] == IL_RESUME), None)
    back_to_il = next((tr for tr in transitions
                       if tr["from"] == IL_RESUME and tr["to"] == IL), None)
    resume_t = resume_tr["t"] if resume_tr else None

    # Step i (0-based) produced trajectory[i + 1], i.e. simulator t = i + 1.
    il_step_ts = [i + 1 for i, s in enumerate(sources) if s == "IL"]
    il_after = [t for t in il_step_ts if resume_t is not None and t > resume_t]
    # act() is called once per IL step, in order, so act i <-> il_step_ts[i].
    acts_after = [pos for t, pos in zip(il_step_ts, act_positions)
                  if resume_t is not None and t > resume_t]
    sim_steps_after = [t for t in sim_step_ts if resume_t is not None and t > resume_t]
    retriggers = [tr["t"] for tr in transitions
                  if tr["to"] == TRIGGER and resume_t is not None and tr["t"] > resume_t]
    restall = (find_repeat_after(result["trajectory"], resume_t, result["goals"])
               if resume_t is not None else None)

    if not result["cbs_calls"]:
        outcome = NO_TRIGGER
    elif resume_t is None:
        if result["termination"] == SUCCESS:
            outcome = SUCCESS_WITHIN_H_EXEC
        else:
            outcome = f"NO_RESUME_{result['stop_reason'] or result['termination']}"
    elif not il_after:
        outcome = RESUME_NO_IL_STEP
    elif result["termination"] == SUCCESS:
        outcome = RESUME_SUCCESS
    elif result["termination"] == INTEGRITY_ABORT:
        outcome = RESUME_INTEGRITY_ABORT
    elif result["termination"] == TIMEOUT:
        outcome = RESUME_TIMEOUT_RESTALL if restall else RESUME_TIMEOUT_NO_REPEAT
    else:
        outcome = f"RESUME_{result['termination']}"

    return {
        "outcome": outcome,
        "il_resume_reached": resume_tr is not None,
        "il_resume_t": resume_t,
        "il_resume_reason": resume_tr["reason"] if resume_tr else None,
        "resume_policy_returned_il": back_to_il is not None,
        "il_actions_executed_after_resume": len(il_after) > 0,
        "il_steps_after_resume": len(il_after),
        "first_il_step_after_resume": il_after[0] if il_after else None,
        "policy_act_calls_after_resume": len(acts_after),
        "sim_step_calls_after_resume": len(sim_steps_after),
        "first_act_positions_after_resume": acts_after[0] if acts_after else None,
        "state_at_resume": result["trajectory"][resume_t] if resume_t is not None else None,
        "monitor_retriggers_after_resume": retriggers,
        "restall_after_resume": restall,
    }


def evaluate_resume(result, planner_calls, act_positions, sim_step_ts, h_exec):
    """Ordered checks; the first failing one is the reported cause. PASS
    requires all agents at goal -- a resume that times out is a FAIL."""
    calls = result["cbs_calls"]
    call = calls[0] if calls else {}
    paths = (planner_calls[0].get("paths") if planner_calls else None) or {}
    trace = result["cbs_exec_trace"]
    trigger_t = call.get("trigger_timestep")
    path1 = {aid: tuple(p[1]) for aid, p in paths.items() if len(p) > 1}
    horizon = (max(len(p) for p in paths.values()) - 1) if paths else None
    window = min(h_exec, horizon) if horizon is not None else None
    agent_ids = set(result["goals"])
    resume = analyze_resume(result, act_positions, sim_step_ts)

    if resume["il_resume_reached"]:
        resume_ok = (resume["resume_policy_returned_il"]
                     and resume["il_actions_executed_after_resume"]
                     and resume["policy_act_calls_after_resume"] == resume["il_steps_after_resume"]
                     and resume["sim_step_calls_after_resume"] == resume["il_steps_after_resume"]
                     and resume["first_act_positions_after_resume"] == resume["state_at_resume"])
    else:
        resume_ok = resume["outcome"] == SUCCESS_WITHIN_H_EXEC

    checks = [
        ("cbs_called_once_on_trigger", len(calls) == 1 and len(planner_calls) == 1),
        ("s_t_equals_simulator_state",
         trigger_t is not None and call["s_t"] == result["trajectory"][trigger_t]),
        ("cbs_status_success", call.get("status") == "success"),
        ("boundary_path_guard_accepted", call.get("accepted") is True),
        ("agent_ids_preserved",
         bool(paths) and set(paths) == agent_ids and set(call.get("agent_order", ())) == agent_ids
         and all(set(e["actual"]) == agent_ids for e in trace)),
        ("path0_equals_s_t",
         bool(paths) and {aid: tuple(p[0]) for aid, p in paths.items()} == call.get("s_t")),
        ("first_cbs_step_reaches_path1",
         bool(trace) and trace[0]["expected"] == path1 and trace[0]["actual"] == path1),
        ("cbs_window_executed", window is not None and len(trace) == window),
        ("expected_equals_actual_every_cbs_step", bool(trace) and all(e["match"] for e in trace)),
        ("no_overlap_any_step",
         all(len(set(state.values())) == len(state) for state in result["trajectory"])),
        ("sim_steps_equal_executed_steps", len(sim_step_ts) == len(result["action_sources"])),
        ("il_resumed_and_really_acted", resume_ok),
        ("all_at_goal", result["all_at_goal"]),
        ("termination_success", result["termination"] == SUCCESS),
    ]
    first_failure = next((name for name, ok in checks if not ok), None)
    return {
        "verdict": "PASS" if first_failure is None else "FAIL",
        "first_failed_check": first_failure,
        "checks": [{"name": name, "ok": bool(ok)} for name, ok in checks],
        "path1": path1,
        "cbs_horizon": horizon,
        "cbs_window": window,
        "resume": resume,
    }


def build_report(result, planner, evaluation, *, scenario, h_exec, max_steps, ckpt, reproduce):
    transitions = result["transitions"]
    first_trigger = next(({"t": tr["t"], "reason": tr["reason"]}
                          for tr in transitions if tr["to"] == TRIGGER), None)
    calls = [{**call, "artifacts_dir": pcall["artifacts_dir"], "paths": pcall.get("paths")}
             for call, pcall in zip(result["cbs_calls"], planner.calls)]
    return {
        "mode": mode_name(h_exec),
        "dev_only_policies": {
            "h_exec": f"{h_exec} (DEV only, not the agreed H_exec)",
            "resume": "always IL after the CBS window (DEV only)",
            "recall": "first Trigger only, at most one CBS call per episode (DEV only)",
            "monitor": "unchanged; latches its first Trigger, so no re-Trigger after resume",
        },
        "verdict": evaluation["verdict"],
        "first_failed_check": evaluation["first_failed_check"],
        "checks": evaluation["checks"],
        "resume": evaluation["resume"],
        "reproduce": reproduce,
        "scenario": str(scenario),
        "checkpoint": str(ckpt),
        "h_exec": h_exec,
        "max_steps": max_steps,
        "solver_root": str(planner.solver_root),
        "solver_git": _git_head(planner.solver_root),
        "project_git": _git_head(PROJECT_ROOT),
        "first_trigger": first_trigger,
        "s_t": calls[0]["s_t"] if calls else None,
        "cbs_calls": calls,
        "cbs_horizon": evaluation["cbs_horizon"],
        "cbs_window": evaluation["cbs_window"],
        "path1": evaluation["path1"],
        "cbs_exec_trace": result["cbs_exec_trace"],
        "exec_mismatch": result["exec_mismatch"],
        "overlap_any_step": not next(c["ok"] for c in evaluation["checks"]
                                     if c["name"] == "no_overlap_any_step"),
        "all_at_goal": result["all_at_goal"],
        "termination": result["termination"],
        "stop_reason": result["stop_reason"],
        "safe_stop": result["safe_stop"],
        "last_state": {"t": result["steps"], "positions": result["final_positions"]},
        "goals": result["goals"],
        "steps": result["steps"],
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
    ap.add_argument("--h-exec", type=int, default=DEFAULT_H_EXEC, help="DEV_ONLY CBS window length")
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--timeout-sec", type=int, default=30)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = ap.parse_args(argv)

    mode = mode_name(args.h_exec)
    run_id = make_run_id(args.scenario.name, mode)
    run_dir = args.out_root.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)  # never reuse a run directory

    planner = CbsRuntimePlanner(solver_root=args.solver_root, work_root=run_dir / "cbs",
                                timeout_sec=args.timeout_sec)
    logger = EpisodeLogger(run_id, scenario=args.scenario.name, mode=mode, out_dir=run_dir)
    result, recorder, sim_cls = run_resume_e2e(
        planner, scenario=args.scenario, h_exec=args.h_exec, max_steps=args.max_steps,
        ckpt=args.ckpt, logger=logger)
    logger.write_steps()

    evaluation = evaluate_resume(result, planner.calls, recorder.act_positions,
                                 sim_cls.step_ts, args.h_exec)
    reproduce = " ".join([sys.executable, "-m", "scripts.run_hybrid_resume_dev", *argv])
    report = build_report(result, planner, evaluation, scenario=args.scenario, h_exec=args.h_exec,
                          max_steps=args.max_steps, ckpt=args.ckpt, reproduce=reproduce)
    report_path = run_dir / "result.json"
    with report_path.open("x", encoding="utf-8") as fh:
        json.dump(_jsonable(report), fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    resume = report["resume"]
    print(f"run_dir        : {run_dir}")
    print(f"first trigger  : {report['first_trigger']}  S_t={report['s_t']}")
    for call in report["cbs_calls"]:
        print(f"cbs call       : status={call['status']} accepted={call['accepted']} "
              f"reason={call['reason']!r} latency_ms={call['latency_ms']} error={call['error']!r}")
    print(f"Path[1]        : {report['path1']}  horizon={report['cbs_horizon']} window={report['cbs_window']}")
    for e in report["cbs_exec_trace"]:
        print(f"  t={e['t']:3d} k={e['k']:2d} expected={e['expected']} actual={e['actual']} "
              f"match={e['match']} overlap={e['overlap']}")
    print(f"IL_RESUME      : reached={resume['il_resume_reached']} t={resume['il_resume_t']} "
          f"-> IL={resume['resume_policy_returned_il']}")
    print(f"IL after resume: steps={resume['il_steps_after_resume']} "
          f"act()={resume['policy_act_calls_after_resume']} sim.step()={resume['sim_step_calls_after_resume']} "
          f"first_t={resume['first_il_step_after_resume']}")
    print(f"re-stall       : {resume['restall_after_resume']}  "
          f"monitor re-triggers={resume['monitor_retriggers_after_resume']}")
    print(f"termination    : {report['termination']}  stop_reason={report['stop_reason']}  "
          f"all_at_goal={report['all_at_goal']}  steps={report['steps']}")
    print(f"OUTCOME        : {resume['outcome']}")
    print(f"VERDICT        : {report['verdict']}"
          + (f"  (first failed check: {report['first_failed_check']})" if report["first_failed_check"] else ""))
    print(f"reproduce      : {reproduce}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
