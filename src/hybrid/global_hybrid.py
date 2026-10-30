"""Global Hybrid Runner v0: IL -> TRIGGER -> CBS_REQUEST -> CBS_EXEC ->
IL_RESUME / SAFE_STOP (docs/global_hybrid_runner_v0.md).

A separate runner -- scripts/run_il.py, MAPFStepSimulator, DeadlockMonitor
and EpisodeLogger are used as-is and never modified.

Safety invariants enforced here regardless of any injected policy:
  * I-1  one CBS request per Trigger event: only a FRESH Trigger
         (trigger_timestep == this step's t) is acted on, and each
         trigger_timestep at most once. DeadlockMonitor keeps reporting
         TRIGGER_FULL_REPEAT after its first repeat, so a stale code must
         never cause another call.
  * I-2  no CBS request when the Monitor reports an overlap integrity issue.
  * I-3  position overlap after any step ends the episode (INTEGRITY_ABORT,
         failure_reason=integrity_overlap -- same meaning as run_il.py).
  * I-4  a plan refused by the boundary, a failed CBS call, or an
         expected/actual mismatch during CBS_EXEC means no further move
         from that plan; SAFE_STOP issues no further move at all.

Not agreed yet with the CBS side, therefore injected and never defaulted:
H_exec (``GlobalHybridConfig.h_exec``), the IL-resume rule
(``resume_policy``), the CBS recall rule (``recall_policy``) and SAFE_STOP's
record contents (``safe_stop_handler``).

``planner(current_positions, goals, grid) -> ReplanResult``: positional
lists in ``sorted(agent_id)`` order, matching the CBS branch's
replan_with_status() contract (paths indexed 0..n-1). The real
replan_with_status() is not wired in v0; tests inject Mock planners.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Protocol

from src.common import spec
from src.hybrid.replan_boundary import accept_replan_result, check_validator_pairing
from src.simulator.mapf_step_simulator import MAPFStepSimulator

# States
IL = "IL"
TRIGGER = "TRIGGER"
CBS_REQUEST = "CBS_REQUEST"
CBS_EXEC = "CBS_EXEC"
IL_RESUME = "IL_RESUME"
SAFE_STOP = "SAFE_STOP"

# Terminations
SUCCESS = "SUCCESS"
TIMEOUT = "TIMEOUT"
INTEGRITY_ABORT = "INTEGRITY_ABORT"


class ResumePolicy(Protocol):
    def after_exec(self, context: Dict[str, Any]) -> str:
        """Return IL or SAFE_STOP once a CBS execution window has ended."""


class RecallPolicy(Protocol):
    def allow_call(self, call_count: int, trigger_timestep: int) -> bool:
        """Whether a fresh Trigger may start a CBS request."""


class SafeStopHandler(Protocol):
    def on_safe_stop(self, reason: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Return the record stored as result["safe_stop"]."""


@dataclass(frozen=True)
class GlobalHybridConfig:
    """All fields required -- none of these rules is agreed yet."""

    h_exec: int
    resume_policy: ResumePolicy
    recall_policy: RecallPolicy
    safe_stop_handler: SafeStopHandler

    def __post_init__(self):
        if not isinstance(self.h_exec, int) or self.h_exec < 1:
            raise ValueError("h_exec must be an int >= 1")


def run_global_hybrid_episode(
    policy,
    grid,
    starts,
    goals,
    max_steps,
    *,
    planner: Callable,
    path_validator: Callable,
    config: GlobalHybridConfig,
    monitor,
    logger=None,
    simulator_cls=MAPFStepSimulator,
):
    """Run one Global Hybrid episode. starts/goals: {agent_id: (row, col)}.

    Returns run_il.run_episode()'s keys plus integrity_abort(_timestep),
    termination, stop_reason, safe_stop, transitions, cbs_calls and
    action_sources ("IL" / "CBS" per executed step).
    """
    if monitor is None:
        raise ValueError("Global Hybrid requires a DeadlockMonitor")
    check_validator_pairing(planner, path_validator)

    sim = simulator_cls(cbs_solver_root=None, max_steps=max_steps)
    obs = sim.reset(grid, starts, goals)
    policy.reset(grid, goals)
    monitor.reset(sim._positions, goals)

    trajectory = [{aid: tuple(pos) for aid, pos in sim._positions.items()}]
    actions_log = []
    action_sources = []
    transitions = []
    cbs_calls = []
    handled_triggers = set()

    info = {"t": 0, "all_at_goal": False, "timed_out": False}
    done = False
    integrity_abort = False
    termination = None
    stop_reason = None
    safe_stop_record = None
    state = IL
    s_t = None
    plan = None
    exec_k = 0
    exec_horizon = 0

    def move_to(new_state, reason=""):
        nonlocal state
        transitions.append({"t": info["t"], "from": state, "to": new_state, "reason": reason})
        state = new_state

    def context():
        return {
            "t": info["t"],
            "positions": dict(sim._positions),
            "cbs_call_count": len(cbs_calls),
            "executed_cbs_steps": exec_k,
        }

    def run_step(actions, source):
        """One simulator step + Monitor/logger bookkeeping, as in run_il.py."""
        nonlocal obs, done, info
        before_positions = {aid: sim._positions[aid] for aid in sim._agent_ids}
        obs, done, info = sim.step(actions)
        overlap = len(set(sim._positions.values())) != len(sim._positions)
        actions_log.append(dict(actions))
        action_sources.append(source)
        trajectory.append({aid: tuple(pos) for aid, pos in sim._positions.items()})

        goal_distance_before = {
            aid: int(policy._dist[aid][before_positions[aid][0], before_positions[aid][1]])
            for aid in sim._agent_ids
        }
        goal_distance_after = {
            aid: int(policy._dist[aid][sim._positions[aid][0], sim._positions[aid][1]])
            for aid in sim._agent_ids
        }
        intended_positions = {}
        for aid in sim._agent_ids:
            drow, dcol = spec.ACTION_DELTA[actions[aid]]
            intended_positions[aid] = (before_positions[aid][0] + drow, before_positions[aid][1] + dcol)
        r = monitor.observe_step(
            info["t"], sim._positions, sim._last_conflict_types,
            current_positions=before_positions,
            intended_positions=intended_positions,
            goal_distance_before=goal_distance_before,
            goal_distance_after=goal_distance_after,
        )
        if logger is not None:
            from scripts.run_il import _log_step
            _log_step(
                logger, sim, policy, before_positions, actions, info, done,
                trigger_reason=r.trigger_reason_code,
                integrity_abort=overlap,
            )
        return r, overlap

    def finish_if_done():
        nonlocal termination
        if done:
            termination = SUCCESS if info["all_at_goal"] else TIMEOUT
        return termination is not None

    while termination is None:
        if state == IL:
            r, overlap = run_step(policy.act(obs, sim._positions), IL)
            if overlap:
                integrity_abort = True
                termination = INTEGRITY_ABORT
                break
            if finish_if_done():
                break
            t = info["t"]
            fresh_trigger = (
                r.trigger_reason_code == "TRIGGER_FULL_REPEAT"
                and r.trigger_timestep == t
                and t not in handled_triggers
            )
            if fresh_trigger:
                handled_triggers.add(t)
                if (
                    r.overlap_integrity_status == "CLEAN"
                    and config.recall_policy.allow_call(len(cbs_calls), t)
                ):
                    move_to(TRIGGER, "TRIGGER_FULL_REPEAT")

        elif state == TRIGGER:
            s_t = {aid: tuple(sim._positions[aid]) for aid in sim._agent_ids}
            move_to(CBS_REQUEST)

        elif state == CBS_REQUEST:
            agent_order = tuple(sim._agent_ids)
            call = {
                "trigger_timestep": info["t"],
                "agent_order": agent_order,
                "s_t": dict(s_t),
                "status": None,
                "accepted": False,
                "reason": "",
                "latency_ms": None,
            }
            cbs_calls.append(call)
            try:
                result = planner(
                    [s_t[aid] for aid in agent_order],
                    [tuple(goals[aid]) for aid in agent_order],
                    grid,
                )
            except Exception:
                call["reason"] = "PLANNER_EXCEPTION"
                stop_reason = "PLANNER_EXCEPTION"
                move_to(SAFE_STOP, stop_reason)
                continue
            call["status"] = getattr(result, "status", None)
            latency_sec = getattr(result, "latency_sec", None)
            call["latency_ms"] = None if latency_sec is None else latency_sec * 1000.0

            outcome = accept_replan_result(
                result,
                agent_order=agent_order,
                current_positions=s_t,
                goals=goals,
                grid=grid,
                path_validator=path_validator,
            )
            call["accepted"] = outcome.accepted
            call["reason"] = outcome.reason
            if not outcome.accepted:
                stop_reason = outcome.reason
                move_to(SAFE_STOP, stop_reason)
                continue
            plan = outcome.plan
            exec_k = 0
            exec_horizon = min(config.h_exec, len(plan.actions))
            if exec_horizon == 0:
                move_to(IL_RESUME, "EMPTY_PLAN")
            else:
                move_to(CBS_EXEC)

        elif state == CBS_EXEC:
            expected = plan.expected[exec_k]
            r, overlap = run_step(dict(plan.actions[exec_k]), "CBS")
            exec_k += 1
            if overlap:
                integrity_abort = True
                termination = INTEGRITY_ABORT
                break
            actual = {aid: tuple(sim._positions[aid]) for aid in sim._agent_ids}
            if actual != expected:
                stop_reason = "EXEC_MISMATCH"
                move_to(SAFE_STOP, stop_reason)
                continue
            if finish_if_done():
                break
            if exec_k >= exec_horizon:
                move_to(IL_RESUME, "EXEC_WINDOW_END")

        elif state == IL_RESUME:
            decision = config.resume_policy.after_exec(context())
            if decision == IL:
                plan = None
                move_to(IL)
            elif decision == SAFE_STOP:
                stop_reason = "RESUME_POLICY_SAFE_STOP"
                move_to(SAFE_STOP, stop_reason)
            else:
                raise ValueError(f"resume_policy returned unknown state {decision!r}")

        elif state == SAFE_STOP:
            plan = None
            safe_stop_record = config.safe_stop_handler.on_safe_stop(stop_reason, context())
            termination = SAFE_STOP

    final = {aid: tuple(pos) for aid, pos in sim._positions.items()}
    return {
        "steps": info["t"],
        "done": bool(done),
        "all_at_goal": bool(info["all_at_goal"]),
        "timed_out": bool(info["timed_out"]),
        "final_positions": final,
        "goals": {aid: tuple(g) for aid, g in goals.items()},
        "trajectory": trajectory,
        "actions": actions_log,
        "integrity_abort": integrity_abort,
        "integrity_abort_timestep": info["t"] if integrity_abort else None,
        "termination": termination,
        "stop_reason": stop_reason,
        "safe_stop": safe_stop_record,
        "transitions": transitions,
        "cbs_calls": cbs_calls,
        "action_sources": action_sources,
    }
