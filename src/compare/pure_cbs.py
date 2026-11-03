"""Pure CBS episode runner: one real CBS plan from S_0, executed to the end.

Pure CBS is a baseline, not the Global Hybrid: the solver is called exactly
once, at t=0 on the simulator's initial state S_0, and its whole accepted plan
is executed. There is no IL policy, no Monitor Trigger, no re-planning and no
IL resume. (Global Hybrid calls CBS only after an IL stall Trigger at t >= 1;
see src/hybrid/global_hybrid.py.)

Reused as-is, never modified: MAPFStepSimulator, the runner-side acceptance
boundary (src/hybrid/replan_boundary.accept_replan_result: status, agent ids,
Path[0] == S_0, adjacent moves only, injected Path Guard), the CBS planner and
Path Guard validator (src/hybrid/cbs_runtime_bridge), and
scripts/run_il._log_step for the existing steps-CSV schema.

Safety contract:
  * the plan is refused before any move unless status == "success", the
    agent ids match, every Path[0] equals S_0, every move is adjacent and the
    validator (the real Path Guard in real runs) accepts it -> SAFE_STOP with
    the boundary's reason code and zero simulator steps;
  * execution step k issues the plan's Path[k-1] -> Path[k] moves (the first
    move goes to Path[1]) and compares every agent's actual position with
    Path[k]. An overlap ends the episode (INTEGRITY_ABORT); a mismatch ends
    it (SAFE_STOP, EXEC_MISMATCH). No move follows either;
  * max_steps (the simulator's episode limit) and the planner's solver
    timeout are separate limits. A plan longer than max_steps is executed up
    to the limit and ends TIMEOUT if not every agent is at its goal.

DEV setting: one CBS call, no re-planning. This is the development baseline
used for comparison, not an agreed project policy.
"""

from __future__ import annotations

from time import perf_counter

from src.common import spec
from src.hybrid.global_hybrid import INTEGRITY_ABORT, SAFE_STOP, SUCCESS, TIMEOUT
from src.hybrid.replan_boundary import accept_replan_result, check_validator_pairing
from src.il.nav_hint import bfs_dist
from src.simulator.mapf_step_simulator import MAPFStepSimulator

# States
PLAN = "PLAN"
CBS_EXEC = "CBS_EXEC"


class _GoalDistances:
    """Only the ``_dist`` attribute scripts/run_il._log_step and the Monitor's
    goal-distance inputs need -- Pure CBS has no IL policy."""

    def __init__(self, grid, goals):
        self._dist = {aid: bfs_dist(grid, goal) for aid, goal in goals.items()}


def run_pure_cbs_episode(
    grid,
    starts,
    goals,
    max_steps,
    *,
    planner,
    path_validator,
    monitor=None,
    logger=None,
    simulator_cls=MAPFStepSimulator,
):
    """Run one Pure CBS episode. starts/goals: {agent_id: (row, col)}.

    ``monitor`` (optional) only observes -- it never changes the run.

    Returns the Global Hybrid runner's result keys (steps, done, all_at_goal,
    timed_out, final_positions, goals, trajectory, actions, integrity_abort,
    integrity_abort_timestep, termination, stop_reason, safe_stop,
    transitions, cbs_calls, action_sources, cbs_exec_trace, exec_mismatch)
    plus plan_horizon and plan_exceeds_max_steps. cbs_calls holds exactly one
    call record with request_timestep=0 and s_0 (not a Trigger).
    """
    check_validator_pairing(planner, path_validator)
    paths_keying = getattr(planner, "paths_keying", "index")

    sim = simulator_cls(cbs_solver_root=None, max_steps=max_steps)
    obs = sim.reset(grid, starts, goals)
    distances = _GoalDistances(grid, goals)
    if monitor is not None:
        monitor.reset(sim._positions, goals)

    agent_order = tuple(sim._agent_ids)
    s_0 = {aid: tuple(sim._positions[aid]) for aid in agent_order}
    trajectory = [dict(s_0)]
    actions_log = []
    action_sources = []
    transitions = []
    cbs_exec_trace = []
    exec_mismatch = None
    info = {"t": 0, "all_at_goal": False, "timed_out": False}
    done = False
    termination = None
    stop_reason = None
    plan_horizon = None

    call = {
        "request_timestep": 0,
        "agent_order": agent_order,
        "s_0": dict(s_0),
        "status": None,
        "accepted": False,
        "reason": "",
        "error": None,
        "latency_ms": None,
        "planner_wall_ms": None,
    }

    def stop(reason, from_state):
        nonlocal termination, stop_reason
        stop_reason = reason
        termination = SAFE_STOP
        transitions.append({"t": info["t"], "from": from_state, "to": SAFE_STOP, "reason": reason})

    # -- PLAN: one CBS call on the real initial state -------------------------
    started = perf_counter()
    try:
        if paths_keying == "agent_id":
            result = planner(dict(s_0), {aid: tuple(goals[aid]) for aid in agent_order}, grid)
        else:
            result = planner([s_0[aid] for aid in agent_order],
                             [tuple(goals[aid]) for aid in agent_order], grid)
    except Exception as exc:
        result = None
        call["error"] = f"{type(exc).__name__}: {exc}"
        call["reason"] = "PLANNER_EXCEPTION"
    call["planner_wall_ms"] = (perf_counter() - started) * 1000.0

    plan = None
    if result is None:
        stop("PLANNER_EXCEPTION", PLAN)
    else:
        call["status"] = getattr(result, "status", None)
        call["error"] = getattr(result, "error", None)
        latency_sec = getattr(result, "latency_sec", None)
        call["latency_ms"] = None if latency_sec is None else latency_sec * 1000.0
        outcome = accept_replan_result(
            result,
            agent_order=agent_order,
            current_positions=s_0,
            goals=goals,
            grid=grid,
            path_validator=path_validator,
            paths_keying=paths_keying,
        )
        call["accepted"] = outcome.accepted
        call["reason"] = outcome.reason
        if outcome.accepted:
            plan = outcome.plan
            plan_horizon = len(plan.actions)
        else:
            stop(outcome.reason, PLAN)

    # -- CBS_EXEC: the whole plan, checked after every step ---------------------
    if plan is not None:
        transitions.append({"t": 0, "from": PLAN, "to": CBS_EXEC, "reason": ""})
        if plan_horizon == 0:
            if s_0 == {aid: tuple(goals[aid]) for aid in agent_order}:
                termination = SUCCESS
            else:
                stop("PATH_END_NOT_AT_GOAL", CBS_EXEC)
        for k in range(1, plan_horizon + 1):
            actions = dict(plan.actions[k - 1])
            expected = dict(plan.expected[k - 1])
            before = {aid: sim._positions[aid] for aid in agent_order}
            obs, done, info = sim.step(actions)
            actual = {aid: tuple(sim._positions[aid]) for aid in agent_order}
            overlap = len(set(actual.values())) != len(actual)
            actions_log.append(actions)
            action_sources.append("CBS")
            trajectory.append(dict(actual))

            trigger_reason = ""
            if monitor is not None:
                intended = {}
                for aid in agent_order:
                    drow, dcol = spec.ACTION_DELTA[actions[aid]]
                    intended[aid] = (before[aid][0] + drow, before[aid][1] + dcol)
                r = monitor.observe_step(
                    info["t"], sim._positions, sim._last_conflict_types,
                    current_positions=before,
                    intended_positions=intended,
                    goal_distance_before={aid: int(distances._dist[aid][before[aid]]) for aid in agent_order},
                    goal_distance_after={aid: int(distances._dist[aid][actual[aid]]) for aid in agent_order},
                )
                trigger_reason = r.trigger_reason_code
            if logger is not None:
                from scripts.run_il import _log_step
                _log_step(logger, sim, distances, before, actions, info, done,
                          trigger_reason=trigger_reason, integrity_abort=overlap)

            entry = {"t": info["t"], "k": k, "expected": expected, "actual": actual,
                     "match": actual == expected, "overlap": overlap}
            cbs_exec_trace.append(entry)
            if overlap:
                termination = INTEGRITY_ABORT
                break
            if actual != expected:
                exec_mismatch = entry
                stop("EXEC_MISMATCH", CBS_EXEC)
                break
            if done:
                termination = SUCCESS if info["all_at_goal"] else TIMEOUT
                break
        if termination is None:
            # The validator requires every path to end at its goal, so this is
            # only reachable with a validator that does not check goals.
            stop("PATH_END_NOT_AT_GOAL", CBS_EXEC)

    final = {aid: tuple(pos) for aid, pos in sim._positions.items()}
    integrity_abort = termination == INTEGRITY_ABORT
    safe_stop = None
    if termination == SAFE_STOP:
        safe_stop = {"reason": stop_reason, "t": info["t"], "positions": dict(final)}
    return {
        "steps": info["t"],
        "done": bool(done),
        "all_at_goal": bool(info["all_at_goal"]) or (info["t"] == 0 and termination == SUCCESS),
        "timed_out": bool(info["timed_out"]),
        "final_positions": final,
        "goals": {aid: tuple(g) for aid, g in goals.items()},
        "trajectory": trajectory,
        "actions": actions_log,
        "integrity_abort": integrity_abort,
        "integrity_abort_timestep": info["t"] if integrity_abort else None,
        "termination": termination,
        "stop_reason": stop_reason,
        "safe_stop": safe_stop,
        "transitions": transitions,
        "cbs_calls": [call],
        "action_sources": action_sources,
        "cbs_exec_trace": cbs_exec_trace,
        "exec_mismatch": exec_mismatch,
        "max_steps": max_steps,
        "plan_horizon": plan_horizon,
        "plan_exceeds_max_steps": plan_horizon is not None and plan_horizon > max_steps,
    }
