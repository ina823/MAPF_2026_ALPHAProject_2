
import argparse
import tempfile
from pathlib import Path

import numpy as np

from src.cbs_adapter import _resolve_solver_root
from src.hybrid.cbs_runtime_bridge import CbsRuntimePlanner
from src.hybrid.selective_compose import compose_selective_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver-root", default=None)
    args = parser.parse_args()

    # 실제 CBS Solver 위치 확인
    solver_root = (
        Path(args.solver_root).resolve()
        if args.solver_root
        else _resolve_solver_root(None)
    )

    print("CBS Solver:", solver_root)

    # 3 x 7 빈 격자
    grid = np.zeros((3, 7), dtype=int)

    # 실제 Agent ID는 7, 10, 20
    positions = {
        7: (1, 5),
        10: (0, 1),
        20: (2, 1),
    }

    goals = {
        7: (1, 6),
        10: (2, 1),
        20: (0, 1),
    }

    # CBS를 실행할 그룹
    group = (10, 20)

    # 그룹 밖 로봇을 정적 장애물로 가정
    planning_grid = grid.copy()
    planning_grid[positions[7]] = 1

    subset_starts = {a: positions[a] for a in group}
    subset_goals = {a: goals[a] for a in group}

    # 임시 작업 폴더에서 실제 CBS 실행
    with tempfile.TemporaryDirectory() as temp_dir:
        planner = CbsRuntimePlanner(
            solver_root=solver_root,
            work_root=Path(temp_dir),
            timeout_sec=15,
        )

        result = planner(
            subset_starts,
            subset_goals,
            planning_grid,
        )

        print("REAL_CBS_STATUS:", result.status)
        print("CBS_LATENCY_SEC:", result.latency_sec)

        if result.status != "success" or result.paths is None:
            raise RuntimeError(f"CBS failed: {result.error}")

        print("REAL_CBS_PATHS:", result.paths)

        if set(result.paths) != set(group):
            raise RuntimeError("Agent ID mismatch")

        # 그룹 경로 + 외부 로봇 WAIT 경로 합성 검증
        checked = compose_selective_plan(
            group_paths=result.paths,
            s_t=positions,
            goals=goals,
            agent_ids=sorted(positions),
            grid=grid,
        )

        print("FULL_PATH_GUARD:", checked.ok)
        print("CHECK_REASON:", checked.reason)

        if not checked.ok:
            raise RuntimeError(checked.reason)

        print("FULL_PATHS:", checked.full_paths)
        print("REAL_SUBSET_CBS_AND_FULL_GUARD_PASS")

        # 실제 Simulator에서 전체 경로 실행
        from src.common.spec import delta_to_action
        from src.simulator.mapf_step_simulator import MAPFStepSimulator
        from src.hybrid.selective_safety import exec_mismatches

        full_paths = checked.full_paths
        sim = MAPFStepSimulator(cbs_solver_root=None, max_steps=50)
        sim.reset(grid, positions, goals)

        horizon = len(next(iter(full_paths.values()))) - 1

        for k in range(1, horizon + 1):
            actions = {}

            for aid, path in full_paths.items():
                prev = path[k - 1]
                nxt = path[k]
                actions[aid] = delta_to_action(
                    nxt[0] - prev[0],
                    nxt[1] - prev[1],
                )

            # 그룹 밖 로봇은 반드시 WAIT
            assert actions[7] == 4, "Outside robot is not WAIT"

            sim.step(actions)
            actual = dict(sim._positions)

            mismatches = exec_mismatches(full_paths, k, actual)

            if mismatches:
                raise AssertionError(
                    f"EXEC_MISMATCH at step {k}: {mismatches}"
                )

            print(f"EXEC_STEP_{k}: PASS, positions={actual}")

        print("SIMULATOR_WAIT_EXECUTION_PASS")

        # 비정상 상황 주입:
        # 7번 로봇을 의도적으로 오른쪽으로 이동시켜
        # expected/actual 검사가 감지하는지 확인
        bad_sim = MAPFStepSimulator(
            cbs_solver_root=None, max_steps=50
        )
        bad_sim.reset(grid, positions, goals)

        bad_actions = {}
        for aid, path in full_paths.items():
            prev, nxt = path[0], path[1]
            bad_actions[aid] = delta_to_action(
                nxt[0] - prev[0], nxt[1] - prev[1]
            )

        bad_actions[7] = 3  # 잘못된 오른쪽 이동
        bad_sim.step(bad_actions)

        detected = exec_mismatches(
            full_paths, 1, dict(bad_sim._positions)
        )

        assert 7 in detected, "Outside movement not detected"
        print("OUTSIDE_UNEXPECTED_MOVE_DETECTED: PASS")

        # 5단계: 실제 CBS 결과의 실행 승인 및 거부 검사
        from src.cbs_runtime_adapter import CBSRuntimeResult
        from src.hybrid.selective_compose import (
            selective_boundary_check,
            SelectiveFullPathValidator,
        )

        # 1. 실제 CBS 경로는 실행 승인되어야 함
        outcome, _ = selective_boundary_check(
            result, s_t=positions, goals=goals, grid=grid
        )
        assert outcome.accepted, outcome.reason
        assert outcome.plan is not None
        assert all(a[7] == 4 for a in outcome.plan.actions)
        print("REAL_CBS_BOUNDARY_PASS")

        # 2. 비정상 경로는 실행 전에 거부되어야 함
        blocked_grid = grid.copy()
        blocked_grid[1, 1] = 1

        cases = [
            (
                "WRONG_START",
                {**result.paths,
                 10: [(0, 0)] + result.paths[10][1:]},
                positions, grid,
            ),
            (
                "ILLEGAL_JUMP",
                {**result.paths,
                 10: [(0, 1), (2, 1)]},
                positions, grid,
            ),
            (
                "WRONG_GOAL",
                {**result.paths,
                 10: [(0, 1), (1, 1)]},
                positions, grid,
            ),
            (
                "OUTSIDE_COLLISION",
                result.paths,
                {**positions, 7: (1, 1)},
                grid,
            ),
            (
                "OBSTACLE_COLLISION",
                result.paths,
                positions, blocked_grid,
            ),
        ]

        for name, paths, starts, test_grid in cases:
            bad_result = CBSRuntimeResult(
                status="success",
                paths=paths,
                latency_sec=0.001,
            )

            verdict, _ = selective_boundary_check(
                bad_result,
                s_t=starts,
                goals=goals,
                grid=test_grid,
            )

            assert not verdict.accepted, name
            assert verdict.plan is None, name
            print(f"{name}_REJECTED: PASS")

        # 3. 검증된 전체 경로를 변경해 외부 WAIT 위반 주입
        tampered = {
            aid: list(path)
            for aid, path in checked.full_paths.items()
        }
        tampered[7][1] = (1, 6)

        ok, reason = SelectiveFullPathValidator(group)(
            grid, positions, goals, tampered
        )

        assert not ok
        assert "OUTSIDE_NOT_HELD" in reason
        print("TAMPERED_OUTSIDE_WAIT_REJECTED: PASS")

        # 4. CBS 실패 결과에는 실행 계획이 없어야 함
        for status in (
            "timeout", "no_solution",
            "invalid_path", "solver_error", "input_error"
        ):
            failure = CBSRuntimeResult(
                status=status,
                paths=None,
                latency_sec=0.001,
                error="Injected failure",
            )

            verdict, _ = selective_boundary_check(
                failure,
                s_t=positions,
                goals=goals,
                grid=grid,
            )

            assert not verdict.accepted
            assert verdict.plan is None
            print(f"{status.upper()}_REFUSED: PASS")

        print("SELECTIVE_PATH_GUARD_NEGATIVE_TESTS_PASS")


if __name__ == "__main__":
    main()
