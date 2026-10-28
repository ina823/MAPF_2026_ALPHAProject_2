"""Audit CBS on nine controlled variants from ORIGINAL starts (not monitor trigger states).

Place this file at scripts/validate_cbs_variants.py and invoke:
  python -m scripts.validate_cbs_variants --common-root ../MAPF_common_logging --timeout 30

Artifacts are separated by variant; does not change the existing CBS Gate evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from src.cbs_adapter import (
    CBSAdapter, CBSAdapterConfig, CBSTimeoutError, CBSNoSolutionError,
    save_internal_paths_json,
)
from src.simulator.mapf_step_simulator import Simulator, MAPFStepSimulator

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def git_sha(folder: Path) -> str | None:
    try:
        p = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=folder,
            capture_output=True, text=True, check=True,
        )
        return p.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_case(scenario_file: Path, common_root: Path):
    scenario = json.loads(scenario_file.read_text(encoding="utf-8-sig"))
    map_path = common_root / scenario["map_file"]
    if not map_path.is_file():
        raise FileNotFoundError(f"Map file missing: {map_path}")
    grid = np.load(map_path)
    agents = sorted(scenario["agents"], key=lambda a: int(a["agent_id"]))
    ids = [int(a["agent_id"]) for a in agents]
    if ids != list(range(len(agents))):
        raise ValueError(f"Agent IDs must be 0..N-1, got {ids}")
    # JSON=[x,y], internal=(row,col)
    starts = [(int(a["start"][1]), int(a["start"][0])) for a in agents]
    goals = [(int(a["goal"][1]), int(a["goal"][0])) for a in agents]
    return scenario, map_path, grid, starts, goals


def replay(grid, starts, goals, paths):
    agent_count = len(starts)
    start_match = all(paths[i][0] == starts[i] for i in range(agent_count))
    goal_match = all(paths[i][-1] == goals[i] for i in range(agent_count))
    validator = Simulator()
    action_array, collision_free = validator.validate_and_parse_paths(
        validator.build_grid_3d(grid), paths
    )
    report = {
        "start_match": bool(start_match),
        "goal_match": bool(goal_match),
        "collision_free": bool(collision_free),
        "path_followed_exactly": False,
        "overlap_timesteps": None,
        "all_at_goal": False,
        "execution_success": False,
        "action_array_shape": list(action_array.shape),
        "execution_log": [],
    }
    if not (start_match and goal_match and collision_free):
        return report

    sim = MAPFStepSimulator(cbs_solver_root=None, max_steps=len(action_array) + 5)
    sim.reset(grid, dict(enumerate(starts)), dict(enumerate(goals)))
    exact = True
    overlaps = 0
    for t in range(len(action_array)):
        actions = {i: int(action_array[t, i]) for i in range(agent_count)}
        _, _, info = sim.step(actions)
        actual = dict(sim._positions)
        expected = {i: paths[i][t + 1] for i in range(agent_count)}
        step_exact = all(actual[i] == expected[i] for i in range(agent_count))
        overlap = len(set(actual.values())) != len(actual)
        exact = exact and step_exact
        overlaps += int(overlap)
        report["execution_log"].append({
            "t": t + 1,
            "actions": actions,
            "expected": {str(i): list(expected[i]) for i in expected},
            "actual": {str(i): list(actual[i]) for i in actual},
            "step_match": bool(step_exact),
            "overlap": bool(overlap),
        })
        if info["all_at_goal"]:
            break

    final_positions = dict(sim._positions)
    at_goal = all(final_positions[i] == goals[i] for i in range(agent_count))
    report.update({
        "path_followed_exactly": bool(exact),
        "overlap_timesteps": overlaps,
        "all_at_goal": bool(at_goal),
        "execution_success": bool(at_goal and exact and overlaps == 0),
        "final_positions": {str(i): list(p) for i, p in final_positions.items()},
    })
    return report


def run_one(scenario_file, common_root, out_dir, timeout, solver_root, metadata):
    name = scenario_file.stem.removeprefix("scenario_")
    case_dir = out_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "variant": name, "input_type": "original_starts_not_trigger_state",
        "scenario_file": str(scenario_file.resolve()),
        "timeout_sec": timeout, "status": "error", "latency_ms": None,
        "solver_stdout_file": "solver_stdout.txt", "solver_stderr_file": "solver_stderr.txt",
        "source_code_sha": metadata["source_code_sha"],
        "scenario_source_sha": metadata["scenario_source_sha"],
        "solver_sha": metadata["solver_sha"],
        "execution_success": None,
    }
    (case_dir / "solver_stdout.txt").write_text("", encoding="utf-8")
    (case_dir / "solver_stderr.txt").write_text("", encoding="utf-8")
    try:
        scenario, map_path, grid, starts, goals = read_case(scenario_file, common_root)
        result.update({
            "map_file": str(map_path.resolve()), "num_agents": len(starts),
            "starts_row_col": starts, "goals_row_col": goals,
        })
        write_json(case_dir / "scenario_snapshot.json", scenario)
        adapter = CBSAdapter(CBSAdapterConfig(
            solver_root=solver_root,
            work_dir=case_dir,
            timeout_sec=timeout,
            python_executable=sys.executable,
        ))
        t0 = time.perf_counter()
        try:
            paths = adapter.plan(starts, goals, grid)
        finally:
            result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            if adapter.last_run is not None:
                (case_dir / "solver_stdout.txt").write_text(
                    adapter.last_run.stdout or "", encoding="utf-8"
                )
                (case_dir / "solver_stderr.txt").write_text(
                    adapter.last_run.stderr or "", encoding="utf-8"
                )
        result["status"] = "success"
        save_internal_paths_json(paths, case_dir / "paths.json")
        try:
            result["replay"] = replay(grid, starts, goals, paths)
            result["execution_success"] = result["replay"]["execution_success"]
        except Exception as replay_exc:
            # A successful Solver run is not the same as successful replay.
            result["replay_error"] = repr(replay_exc)
            result["execution_success"] = False
    except CBSTimeoutError as exc:
        result.update(status="timeout", error=str(exc))
    except CBSNoSolutionError as exc:
        result.update(status="no_solution", error=str(exc))
    except Exception as exc:
        result.update(status="error", error=repr(exc))
    if result.get("error"):
        (case_dir / "error.txt").write_text(result["error"] + "\n", encoding="utf-8")
    write_json(case_dir / "result.json", result)
    print(f"{name}: status={result['status']}, latency_ms={result['latency_ms']}, "
          f"replay={result['execution_success']}", flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description="CBS raw-evidence audit: 9 variants at original starts")
    p.add_argument("--common-root", type=Path, default=Path("../MAPF_common_logging"))
    p.add_argument("--out", type=Path, default=Path("docs/evidence/cbs_variants_initial"))
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--only", help="Run one variant slug first (without scenario_ prefix)")
    p.add_argument("--dry-run", action="store_true", help="List/validate inputs without CBS")
    args = p.parse_args()
    if args.timeout <= 0:
        p.error("--timeout must be positive")
    common_root = args.common_root.resolve()
    solver_root = (PROJECT_ROOT / "third_party" / "multi_agent_path_planning" / "centralized" / "cbs").resolve()
    out_dir = args.out.resolve()
    variants = sorted((common_root / "scenarios" / "controlled_variants").glob("scenario_*.json"))
    if args.only:
        variants = [v for v in variants if v.stem == "scenario_" + args.only]
        if not variants:
            p.error(f"Cannot find variant: {args.only}")
    elif len(variants) != 9:
        p.error(f"Expected 9 variants, found {len(variants)} at {common_root}")

    meta = {
        "audit_type": "initial_start_CBS_not_trigger_recovery",
        "timeout_sec": args.timeout,
        "common_root": str(common_root),
        "source_code_sha": git_sha(PROJECT_ROOT),
        "scenario_source_sha": git_sha(common_root),
        "solver_sha": git_sha(solver_root),
        "python_executable": sys.executable,
        "solver_root": str(solver_root),
        "variants": [v.name for v in variants],
    }
    if args.dry_run:
        for path in variants:
            _, map_path, _, starts, goals = read_case(path, common_root)
            print(f"OK {path.name}: agents={len(starts)}, map={map_path.name}")
        return
    if not (solver_root / "cbs.py").is_file():
        p.error(f"CBS solver missing: {solver_root / 'cbs.py'}")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "run_info.json", meta)
    results = [run_one(v, common_root, out_dir, args.timeout, solver_root, meta) for v in variants]
    write_json(out_dir / "summary.json", results)
    fields = ["variant", "input_type", "num_agents", "timeout_sec", "status", "latency_ms", "execution_success"]
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved evidence: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
