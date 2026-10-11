"""Roll out the Alpha 1 NavHint IL policy on one scenario JSON.

Baseline-reproduction runner (no new features). The episode loop mirrors
scripts/eval_flow.py::rollout from Alpha 1:

    sim = MAPFStepSimulator(cbs_solver_root=None, max_steps=max_steps)
    obs = sim.reset(grid, starts, goals)
    loop: actions = policy(obs)  ->  obs, done, info = sim.step(actions)

Usage (from the repository root):
    python -m scripts.run_il
    python -m scripts.run_il --scenario scenarios/smoke/scenario_empty_s11_n3.json --verbose
    python -m scripts.run_il --out outputs/logs/run_il_empty_s11_n3.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.scenario_loader import load_map_generator_scenario
from src.common import spec
from src.il.policy import NavHintPolicy
from src.simulator.mapf_step_simulator import MAPFStepSimulator

DEFAULT_CKPT = PROJECT_ROOT / "checkpoints" / "cnn_navhint.pt"
DEFAULT_SCENARIO = PROJECT_ROOT / "scenarios" / "smoke" / "scenario_empty_s11_n3.json"
DEFAULT_MAX_STEPS = 160  # same default as Alpha 1 scripts/eval_flow.py


def _log_step(logger, sim, policy, before_positions, actions, info, done, trigger_reason=""):
    """Bridge one simulator step into EpisodeLogger rows (IL-specific glue).

    Lives here, not in the simulator/policy/logger, so that MAPFStepSimulator
    and NavHintPolicy keep doing only their own job (movement / action
    prediction) and src.common.episode_logger stays IL-agnostic and reusable
    by future CBS/hybrid runners. This function only reads values that were
    already computed elsewhere (spec.ACTION_DELTA, sim._positions,
    sim._last_conflict_types, policy._dist / nav_hint.bfs_dist) -- it invents
    no new distance/collision logic of its own.

    ``trigger_reason`` (Phase 3B-3, optional, default ""): the Monitor's
    ``DeadlockMonitorResult.trigger_reason_code`` for THIS timestep (e.g.
    "TRIGGER_NONE" / "TRIGGER_FULL_REPEAT"), already computed by the caller
    before this function runs. Written identically into every agent's row
    for this timestep -- same pattern as ``done``/``success``, which are
    also episode/monitor-level values repeated across all agent rows, not
    per-agent ones. This is the existing ``EpisodeLogger.log_step``
    ``trigger_reason`` parameter (present since Common Logging, never
    populated until now) -- no new field, no schema change. It is a
    Trigger signal only (docs/deadlock_definition_v1.md Section 2 concept
    4), never a final Operational Label (concept 5); this function does
    not compute, and must never be read as, a deadlock verdict.
    """
    success = bool(info["all_at_goal"])
    failure_reason = "timeout" if (done and info["timed_out"] and not success) else ""
    for agent_id in sim._agent_ids:
        current = before_positions[agent_id]
        action_id = actions[agent_id]
        drow, dcol = spec.ACTION_DELTA[action_id]
        intended = (current[0] + drow, current[1] + dcol)
        next_pos = sim._positions[agent_id]
        dist_field = policy._dist[agent_id]
        goal_distance_before = int(dist_field[current[0], current[1]])
        goal_distance_after = int(dist_field[next_pos[0], next_pos[1]])
        conflict_type = sim._last_conflict_types.get(agent_id, "none")
        logger.log_step(
            timestep=info["t"],
            agent_id=agent_id,
            current=current,
            action_id=action_id,
            intended=intended,
            next_pos=next_pos,
            goal_distance_before=goal_distance_before,
            goal_distance_after=goal_distance_after,
            conflict_type=conflict_type,
            done=done,
            success=success,
            failure_reason=failure_reason,
            trigger_reason=trigger_reason,
        )
    logger.log_timestep_conflicts(
        vertex_events=sim._last_vertex_conflict_count,
        edge_events=sim._last_edge_conflict_count,
    )


def run_episode(policy, grid, starts, goals, max_steps, logger=None, monitor=None):
    """Run one episode. starts/goals: {agent_id: (row, col)}.

    Returns a dict with steps / done / all_at_goal / timed_out, the final
    positions, and the per-step trajectory and actions (for logging/tests).
    Success is ``info["all_at_goal"]`` (see docs/alpha1_baseline.md, KI-2).

    ``logger``: optional src.common.episode_logger.EpisodeLogger. When None
    (the default), this function is byte-for-byte the pre-logging baseline
    rollout -- no extra bookkeeping is done. When provided, log_step()/
    log_timestep_conflicts() are called once per simulator step; write_steps()
    /write_summary() are the caller's responsibility (see main() below).

    ``monitor``: optional src.common.deadlock_monitor.DeadlockMonitor
    (Phase 3A full joint-state repeat + Phase 3B-1 persistent_conflict +
    Phase 3B-2A no_progress -- see docs/deadlock_definition_v1.md). Same
    additive-only contract as ``logger``: when None (the default) this
    function is unaffected; when provided, it only observes positions/
    conflict_type/goal-distance and never changes actions or stops the
    rollout. Phase 3B-2B wires goal_distance_before/after through to it
    (the same values ``_log_step`` already computes for the logger);
    Phase 3B-3 additionally feeds its per-step ``trigger_reason_code``
    into the logger's existing ``trigger_reason`` CSV column (see
    ``_log_step``). Phase 3B (persistent_conflict wiring) now also
    passes current_positions/intended_positions -- persistent_conflict
    stays a separate, observational candidate signal (never a Trigger,
    never a final Operational Label); it is not written to any CSV
    column yet (left for a future sidecar output, Phase 3B-4)."""
    sim = MAPFStepSimulator(cbs_solver_root=None, max_steps=max_steps)
    obs = sim.reset(grid, starts, goals)
    policy.reset(grid, goals)
    if monitor is not None:
        monitor.reset(sim._positions, goals)

    trajectory = [{aid: tuple(pos) for aid, pos in sim._positions.items()}]
    actions_log = []
    info = {"t": 0, "all_at_goal": False, "timed_out": False}
    done = False
    for _ in range(max_steps):
        actions = policy.act(obs, sim._positions)
        # Phase 3B-2B: monitor's no_progress signal needs the same
        # pre-step snapshot logger's _log_step already needed -- extend
        # the guard, not the computation itself, so a bare baseline run
        # (logger=None, monitor=None) still skips this entirely.
        need_before_positions = logger is not None or monitor is not None
        before_positions = {aid: sim._positions[aid] for aid in sim._agent_ids} if need_before_positions else None
        obs, done, info = sim.step(actions)
        actions_log.append(dict(actions))
        trajectory.append({aid: tuple(pos) for aid, pos in sim._positions.items()})

        # Phase 3B-3: the monitor is now observed BEFORE _log_step() is
        # called, so its trigger_reason_code for THIS timestep is ready in
        # time to be written into the CSV row below. This reordering does
        # not touch policy.act()/sim.step() (already both run above,
        # unchanged) and observe_step() is read-only, so it cannot alter
        # the trajectory/actions/goal_distance values _log_step() uses.
        trigger_reason = ""
        if monitor is not None:
            # Same raw computation _log_step uses for the logger
            # (policy._dist[aid] looked up at the pre-/post-step
            # position) -- duplicated here, not shared, so _log_step and
            # EpisodeLogger's existing behaviour stay untouched.
            goal_distance_before = {
                aid: int(policy._dist[aid][before_positions[aid][0], before_positions[aid][1]])
                for aid in sim._agent_ids
            }
            goal_distance_after = {
                aid: int(policy._dist[aid][sim._positions[aid][0], sim._positions[aid][1]])
                for aid in sim._agent_ids
            }
            # Phase 3B (persistent_conflict wiring): raw, pre-wall-clamp
            # intended cell per agent -- the same formula _log_step uses
            # for the CSV's `intended` column (current + the chosen
            # action's spec.ACTION_DELTA). This is the agent's PROPOSED
            # target before collision resolution, not sim._positions[aid]
            # (the ACTUAL post-resolution position after bounce-back).
            # DeadlockMonitor only ever reads this for agents classified
            # vertex/edge (never wall-hit), for which this raw value is
            # identical to the simulator's own internal pre-clamp snapshot
            # (see docs/deadlock_definition_v1.md Section 3.3/3.4).
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
            trigger_reason = r.trigger_reason_code

        if logger is not None:
            _log_step(logger, sim, policy, before_positions, actions, info, done,
                      trigger_reason=trigger_reason)
        if done:
            break

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
    }


def load_scenario(path):
    """Scenario JSON -> (grid, starts dict, goals dict); (row, col) internally."""
    grid, starts, goals = load_map_generator_scenario(path)
    return grid, {i: starts[i] for i in range(len(starts))}, {i: goals[i] for i in range(len(goals))}


def _jsonable(result):
    out = dict(result)
    for key in ("final_positions", "goals"):
        out[key] = {str(a): list(p) for a, p in result[key].items()}
    out["trajectory"] = [{str(a): list(p) for a, p in f.items()} for f in result["trajectory"]]
    out["actions"] = [{str(a): int(v) for a, v in f.items()} for f in result["actions"]]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--verbose", action="store_true", help="print positions at every step")
    ap.add_argument("--out", type=Path, default=None, help="optional JSON log path")
    ap.add_argument("--log", action="store_true",
                     help="write step/summary CSV logs to outputs/logs/ (default: off, "
                          "identical to pre-logging behaviour)")
    ap.add_argument("--monitor", action="store_true",
                     help="attach a DeadlockMonitor (Phase 3A, full joint-state repeat "
                          "only -- see docs/deadlock_definition_v1.md) and print its "
                          "result; observation only, never changes the rollout "
                          "(default: off, identical to pre-monitor behaviour)")
    args = ap.parse_args(argv)

    policy = NavHintPolicy(args.ckpt)
    grid, starts, goals = load_scenario(args.scenario)

    logger = None
    if args.log:
        from src.common.episode_logger import EpisodeLogger, make_run_id
        run_id = make_run_id(args.scenario.name, "IL")
        logger = EpisodeLogger(run_id, scenario=args.scenario.name, mode="IL")

    monitor = None
    if args.monitor:
        from src.common.deadlock_monitor import DeadlockMonitor
        monitor = DeadlockMonitor()

    result = run_episode(policy, grid, starts, goals, args.max_steps, logger=logger, monitor=monitor)

    print(f"checkpoint : {args.ckpt.name}  (hint_mode={policy.hint_mode}, goal_dim={policy.model['goal_dim']})")
    print(f"scenario   : {args.scenario.name}  map={grid.shape}  agents={len(goals)}")
    if args.verbose:
        for t, frame in enumerate(result["trajectory"]):
            print(f"  t={t:3d}  {frame}")
    print(f"steps={result['steps']}  all_at_goal={result['all_at_goal']}  timed_out={result['timed_out']}")
    print(f"final positions == goals : {result['final_positions'] == result['goals']}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
        print(f"saved {args.out}")

    if logger is not None:
        steps_path = logger.write_steps()
        summary_path = logger.write_summary(
            num_agents=len(goals),
            steps=result["steps"],
            success=result["all_at_goal"],
            timed_out=result["timed_out"],
            makespan=result["steps"],
        )
        print(f"log steps  : {steps_path}")
        print(f"log summary: {summary_path}")

    if monitor is not None:
        r = monitor.result
        print(f"monitor    : trigger_type={r.trigger_type} blocking_type={r.blocking_type} "
              f"first_seen={r.first_seen} trigger_timestep={r.trigger_timestep} period={r.period} "
              f"overlap_integrity_status={r.overlap_integrity_status} "
              f"overlap_timesteps={r.overlap_timesteps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
