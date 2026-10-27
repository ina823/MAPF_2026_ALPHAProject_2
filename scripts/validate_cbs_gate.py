import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.replanner import replan
from src.cbs_adapter import CBSTimeoutError, CBSNoSolutionError
from src.simulator.mapf_step_simulator import Simulator, MAPFStepSimulator


CASES = [
    {
        "name": "Corridor",
        "trigger_timestep": 5,
        "map": "scenarios/controlled/map_corridor_bay_n2.npy",
        "current_positions": [(2, 4), (2, 6)],
        "goals": [(2, 10), (2, 0)],
    },
    {
        "name": "Intersection",
        "trigger_timestep": 2,
        "map": "scenarios/controlled/map_intersection_n2.npy",
        "current_positions": [(3, 2), (2, 3)],
        "goals": [(3, 5), (5, 3)],
    },
    {
        "name": "Bottleneck",
        "trigger_timestep": 2,
        "map": "scenarios/controlled/map_bottleneck_n4.npy",
        "current_positions": [(1, 2), (2, 1), (1, 7), (3, 7)],
        "goals": [(1, 7), (3, 7), (0, 1), (4, 1)],
    },
]


def main():
    parser = argparse.ArgumentParser(
        description="Validate CBS recovery from actual DeadlockMonitor trigger states."
    )
    parser.add_argument(
        "--common-root",
        type=Path,
        default=Path("../MAPF_common_logging"),
        help="Path to common-logging worktree containing controlled map files.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="CBS timeout in seconds.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for detailed JSON results.",
    )
    args = parser.parse_args()

    results = []

    print("#" * 100)
    print("CBS GATE VALIDATION")
    print("#" * 100)
    print("solver_root = None")
    print("timeout_sec =", args.timeout)
    print("common_root =", args.common_root.resolve())
    print("coordinate convention = (row, col)")
    print()

    for case in CASES:
        grid_path = args.common_root / case["map"]
        grid = np.load(grid_path)

        starts_list = case["current_positions"]
        goals_list = case["goals"]

        starts = {
            i: starts_list[i]
            for i in range(len(starts_list))
        }

        goals = {
            i: goals_list[i]
            for i in range(len(goals_list))
        }

        print()
        print("=" * 100)
        print("CASE =", case["name"])
        print("trigger_timestep =", case["trigger_timestep"])
        print("current_positions =", starts_list)
        print("goals =", goals_list)
        print("grid_path =", grid_path)
        print("grid_shape =", grid.shape)
        print("=" * 100)

        t0 = time.perf_counter()

        try:
            paths = replan(
                current_positions=starts_list,
                goals=goals_list,
                grid_map=grid,
                solver_root=None,
                timeout_sec=args.timeout,
            )

            latency_ms = (time.perf_counter() - t0) * 1000.0
            status = "success"
            error = None

        except CBSTimeoutError as exc:
            paths = None
            latency_ms = (time.perf_counter() - t0) * 1000.0
            status = "timeout"
            error = str(exc)

        except CBSNoSolutionError as exc:
            paths = None
            latency_ms = (time.perf_counter() - t0) * 1000.0
            status = "no_solution"
            error = str(exc)

        except Exception as exc:
            paths = None
            latency_ms = (time.perf_counter() - t0) * 1000.0
            status = "error"
            error = repr(exc)

        result = {
            "case": case["name"],
            "trigger_timestep": case["trigger_timestep"],
            "current_positions": starts_list,
            "goals": goals_list,
            "status": status,
            "cbs_latency_ms": latency_ms,
            "error": error,
            "paths": None,
            "collision_free": None,
            "path_followed_exactly": None,
            "overlap_timesteps": None,
            "all_at_goal": None,
            "execution_success": None,
        }

        print()
        print("--- CBS RESULT ---")
        print("status =", status)
        print(f"cbs_latency_ms = {latency_ms:.3f}")

        if error is not None:
            print("error =", error)

        if paths is None:
            results.append(result)
            continue

        result["paths"] = {
            str(agent_id): [list(p) for p in path]
            for agent_id, path in paths.items()
        }

        print()
        print("--- FULL CBS PATHS ---")

        for agent_id in sorted(paths):
            print(f"agent{agent_id}: {paths[agent_id]}")

        start_match = all(
            paths[i][0] == starts_list[i]
            for i in range(len(starts_list))
        )

        goal_match = all(
            paths[i][-1] == goals_list[i]
            for i in range(len(goals_list))
        )

        validator = Simulator()
        grid_3d = validator.build_grid_3d(grid)

        action_array, collision_free = validator.validate_and_parse_paths(
            grid_3d,
            paths,
        )

        result["start_match"] = bool(start_match)
        result["goal_match"] = bool(goal_match)
        result["collision_free"] = bool(collision_free)
        result["action_array_shape"] = list(action_array.shape)

        print()
        print("--- PATH VALIDATION ---")
        print("start_match =", start_match)
        print("goal_match =", goal_match)
        print("collision_free =", collision_free)
        print("action_array_shape =", action_array.shape)

        recovery_sim = MAPFStepSimulator(
            cbs_solver_root=None,
            max_steps=len(action_array) + 5,
        )

        recovery_sim.reset(
            grid,
            starts,
            goals,
        )

        path_followed_exactly = True
        overlap_timesteps = 0
        execution_log = []

        print()
        print("--- RECOVERY EXECUTION ---")
        print("recovery_t=0 positions =", dict(recovery_sim._positions))

        for t in range(len(action_array)):
            actions = {
                aid: int(action_array[t, aid])
                for aid in sorted(starts)
            }

            _, _, info = recovery_sim.step(actions)

            actual_positions = dict(recovery_sim._positions)

            expected_positions = {
                aid: paths[aid][t + 1]
                for aid in sorted(starts)
            }

            step_match = all(
                actual_positions[aid] == expected_positions[aid]
                for aid in expected_positions
            )

            if not step_match:
                path_followed_exactly = False

            values = list(actual_positions.values())
            has_overlap = len(set(values)) != len(values)

            if has_overlap:
                overlap_timesteps += 1

            row = {
                "recovery_t": t + 1,
                "actions": actions,
                "expected": expected_positions,
                "actual": actual_positions,
                "step_match": step_match,
                "overlap": has_overlap,
                "all_at_goal": bool(info["all_at_goal"]),
            }

            execution_log.append(row)

            print(
                f"recovery_t={t + 1} "
                f"actions={actions} "
                f"expected={expected_positions} "
                f"actual={actual_positions} "
                f"step_match={step_match} "
                f"overlap={has_overlap} "
                f"all_at_goal={info['all_at_goal']}"
            )

            if info["all_at_goal"]:
                break

        final_positions = dict(recovery_sim._positions)

        all_at_goal = all(
            final_positions[aid] == goals[aid]
            for aid in goals
        )

        execution_success = (
            all_at_goal
            and path_followed_exactly
            and overlap_timesteps == 0
        )

        result["execution_log"] = execution_log
        result["final_positions"] = {
            str(k): list(v)
            for k, v in final_positions.items()
        }
        result["path_followed_exactly"] = path_followed_exactly
        result["overlap_timesteps"] = overlap_timesteps
        result["all_at_goal"] = all_at_goal
        result["execution_success"] = execution_success

        print()
        print("--- RECOVERY RESULT ---")
        print("final_positions =", final_positions)
        print("all_at_goal =", all_at_goal)
        print("path_followed_exactly =", path_followed_exactly)
        print("overlap_timesteps =", overlap_timesteps)
        print("execution_success =", execution_success)

        results.append(result)

    print()
    print("#" * 100)
    print("FINAL SUMMARY")
    print("#" * 100)

    for r in results:
        print(
            f"{r['case']}: "
            f"status={r['status']}, "
            f"latency_ms={r['cbs_latency_ms']:.3f}, "
            f"execution_success={r['execution_success']}, "
            f"all_at_goal={r['all_at_goal']}, "
            f"path_followed_exactly={r['path_followed_exactly']}, "
            f"overlap_timesteps={r['overlap_timesteps']}"
        )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)

        with args.json_out.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                results,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print()
        print("JSON saved to:", args.json_out)


if __name__ == "__main__":
    main()
