"""Offline integration gate: logged Full Repeat -> actual states -> CBS -> replay.

Run from MAPF_2026_ALPHAProject_2 project root. Does NOT switch Git branches,
modify the IL runner, or install a runtime Hybrid controller.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from src.cbs_adapter import CBSNoSolutionError, CBSTimeoutError
from src.common.scenario_loader import load_map_generator_scenario
from src.replanner import replan
from src.simulator.mapf_step_simulator import MAPFStepSimulator, Simulator


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def sha(root: Path) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def input_from_monitor(mon_path: Path, common_root: Path):
    monitor_rows = read_csv(mon_path)
    if len(monitor_rows) != 1:
        raise ValueError(f"Expected one monitor row, got {len(monitor_rows)}")
    mon = monitor_rows[0]
    if mon["trigger_reason_code"] != "TRIGGER_FULL_REPEAT" or mon["trigger_type"] != "FULL":
        raise ValueError("Monitor does not report a FULL_REPEAT trigger")
    if mon["overlap_integrity_status"] != "CLEAN" or int(mon["overlap_timesteps"]) != 0:
        raise ValueError("Position overlap integrity is not CLEAN; do not use as CBS start state")

    if not mon_path.name.endswith("_monitor.csv"):
        raise ValueError("Monitor filename must end in _monitor.csv")
    steps_path = mon_path.with_name(mon_path.name.replace("_monitor.csv", "_steps.csv"))
    if not steps_path.is_file():
        raise FileNotFoundError(f"Missing matching steps CSV: {steps_path}")
    all_rows = read_csv(steps_path)
    t = int(mon["trigger_timestep"])
    trigger_rows = sorted(
        (r for r in all_rows if r["run_id"] == mon["run_id"] and int(r["timestep"]) == t),
        key=lambda r: int(r["agent_id"]),
    )
    if not trigger_rows:
        raise ValueError(f"No step rows for trigger timestep={t}")
    agent_ids = [int(r["agent_id"]) for r in trigger_rows]
    if agent_ids != list(range(len(agent_ids))):
        raise ValueError(f"Agent IDs must be contiguous 0..N-1; got {agent_ids}")
    if any(r["trigger_reason"] != "TRIGGER_FULL_REPEAT" for r in trigger_rows):
        raise ValueError("Step rows do not agree with monitor trigger reason")
    starts = [
        (int(r["next_row"]), int(r["next_col"])) for r in trigger_rows
    ]
    if len(set(starts)) != len(starts):
        raise ValueError("Repeated robot positions at trigger timestep (overlap)")

    scenario_name = Path(mon["scenario"]).name
    scenario_path = common_root / "scenarios" / "controlled" / scenario_name
    if not scenario_path.is_file():
        raise FileNotFoundError(f"Scenario not found: {scenario_path}")
    with scenario_path.open(encoding="utf-8") as fh:
        scenario_data = json.load(fh)
    scenario_ids = [int(a.get("agent_id", i)) for i, a in enumerate(scenario_data["agents"])]
    if scenario_ids != agent_ids:
        raise ValueError(f"Scenario agent order {scenario_ids} != log order {agent_ids}")
    grid, _, goals = load_map_generator_scenario(scenario_path)
    if len(goals) != len(starts):
        raise ValueError("Goal count does not match actual trigger-state robot count")

    first_seen = int(mon["first_seen"])
    prev_rows = sorted(
        (r for r in all_rows if r["run_id"] == mon["run_id"] and int(r["timestep"]) == first_seen),
        key=lambda r: int(r["agent_id"]),
    )
    if first_seen > 0 and len(prev_rows) == len(starts):
        previous = [(int(r["next_row"]), int(r["next_col"])) for r in prev_rows]
        if previous != starts:
            raise ValueError("Monitor first_seen does not match recorded full joint state")

    return mon, steps_path, t, agent_ids, starts, goals, grid, scenario_path


def replay(grid, starts, goals, paths):
    n = len(starts)
    expected_ids = list(range(n))
    if sorted(paths) != expected_ids:
        raise ValueError(f"CBS output agent IDs differ from input: {sorted(paths)}")
    start_match = all(paths[i] and tuple(paths[i][0]) == starts[i] for i in expected_ids)
    goal_match = all(paths[i] and tuple(paths[i][-1]) == goals[i] for i in expected_ids)
    if not start_match or not goal_match:
        raise ValueError(f"CBS endpoints invalid: start_match={start_match} goal_match={goal_match}")

    validator = Simulator()
    action_array, collision_free = validator.validate_and_parse_paths(
        validator.build_grid_3d(grid), paths
    )
    if action_array is None or not collision_free:
        raise ValueError("CBS output failed vertex/edge/wall validation")

    sim = MAPFStepSimulator(cbs_solver_root=None, max_steps=len(action_array) + 5)
    sim.reset(grid, dict(enumerate(starts)), dict(enumerate(goals)))
    exact = True
    overlaps = 0
    trace = []
    for k in range(len(action_array)):
        actions = {aid: int(action_array[k, aid]) for aid in expected_ids}
        _, _, info = sim.step(actions)
        actual = {aid: tuple(sim._positions[aid]) for aid in expected_ids}
        expected = {aid: tuple(paths[aid][k + 1]) for aid in expected_ids}
        match = actual == expected
        overlap = len(set(actual.values())) != n
        exact &= match
        overlaps += int(overlap)
        trace.append({
            "recovery_t": k + 1,
            "actions": actions,
            "actual": {str(i): list(v) for i, v in actual.items()},
            "expected": {str(i): list(v) for i, v in expected.items()},
            "step_match": match,
            "overlap": overlap,
        })
        if info["all_at_goal"]:
            break
    goal_reached = all(tuple(sim._positions[i]) == goals[i] for i in expected_ids)
    return {
        "collision_free": bool(collision_free),
        "start_match": start_match,
        "goal_match": goal_match,
        "path_followed_exactly": exact,
        "overlap_timesteps": overlaps,
        "all_at_goal": goal_reached,
        "execution_success": bool(collision_free and exact and not overlaps and goal_reached),
        "execution_log": trace,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--monitor-file", type=Path, required=True,
                    help="Existing <run_id>_monitor.csv from common-logging")
    ap.add_argument("--common-root", type=Path, default=Path("../MAPF_common_logging"))
    ap.add_argument("--out-dir", type=Path, default=Path("docs/evidence/cbs_trigger_csv_bridge"))
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true", help="Verify CSV state and scenario without calling CBS")
    args = ap.parse_args()
    mon_path = args.monitor_file.resolve()
    common_root = args.common_root.resolve()
    mon, steps_path, t, ids, starts, goals, grid, scenario_path = input_from_monitor(mon_path, common_root)
    if args.dry_run:
        print(f"OK run_id={mon['run_id']}, trigger_t={t}, positions={starts}, goals={goals}")
        print(f"Scenario={scenario_path.name}, map_shape={grid.shape}, integrity=CLEAN")
        return
    out = (args.out_dir / mon["run_id"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(mon_path, out / mon_path.name)
    shutil.copy2(steps_path, out / steps_path.name)
    shutil.copy2(scenario_path, out / scenario_path.name)

    result = {
        "experiment": "offline_logged_trigger_to_cbs_replay",
        "is_automatic_hybrid_runner": False,
        "run_id": mon["run_id"],
        "monitor_trigger_reason": mon["trigger_reason_code"],
        "trigger_timestep": t,
        "first_seen": int(mon["first_seen"]),
        "agent_ids": ids,
        "current_positions": [list(p) for p in starts],
        "goals": [list(p) for p in goals],
        "monitor_source_sha": sha(common_root),
        "cbs_source_sha": sha(Path(__file__).resolve().parents[1]),
        "timeout_sec": args.timeout,
        "status": None,
        "error": None,
    }
    print(f"Monitor trigger: t={t}, actual_positions={starts}")
    print(f"Scenario goals: {goals}")
    print("Replanning with logged actual positions, not original starts...")
    tic = time.perf_counter()
    paths = None
    original_cwd = Path.cwd()
    # replan() writes to a cwd-relative work directory. Isolate it here so
    # previous local outputs/paths/replan files are never overwritten.
    try:
        os.chdir(out)
        paths = replan(starts, goals, grid, timeout_sec=args.timeout)
        result["status"] = "success"
    except CBSTimeoutError as exc:
        result["status"] = "timeout"
        result["error"] = str(exc)
    except CBSNoSolutionError as exc:
        result["status"] = "no_solution"
        result["error"] = str(exc)
    except Exception as exc:
        result["status"] = "error"
        result["error"] = repr(exc)
    finally:
        os.chdir(original_cwd)
    result["latency_ms"] = round((time.perf_counter() - tic) * 1000, 3)

    # replan() stores YAML in a shared work dir; snapshot it immediately.
    work_dir = out / "outputs" / "paths" / "replan"
    artifact_names = ["input.yaml", "solver_input.yaml"]
    if result["status"] == "success":
        artifact_names.append("output.yaml")
    for name in artifact_names:
        source = work_dir / name
        if source.is_file():
            shutil.copy2(source, out / name)

    if paths is not None:
        result["paths"] = {str(i): [list(p) for p in path] for i, path in sorted(paths.items())}
        try:
            result.update(replay(grid, starts, goals, paths))
        except Exception as exc:
            result["execution_success"] = False
            result["replay_error"] = repr(exc)

    result_path = out / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"CBS status: {result['status']}, latency_ms={result['latency_ms']}")
    print(f"Replay execution_success: {result.get('execution_success')}")
    print(f"Evidence saved: {result_path}")
    raise SystemExit(0 if result["status"] == "success" and result.get("execution_success") else 2)


if __name__ == "__main__":
    main()
