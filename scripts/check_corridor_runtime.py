
import importlib.util
import sys
from pathlib import Path

import json
import numpy as np

from scripts.run_il import DEFAULT_CKPT
from src.common.scenario_loader import xy_to_rowcol
from src.il.policy import NavHintPolicy
from src.simulator.mapf_step_simulator import MAPFStepSimulator
from src.cbs_runtime_adapter import run_cbs_runtime


# 1. 인아의 Deadlock Monitor 불러오기
root = Path(__file__).resolve().parents[1]

monitor_path = (
    root.parent
    / "MAPF_common_logging"
    / "src"
    / "common"
    / "deadlock_monitor.py"
)

module_name = "common_logging_deadlock_monitor"
spec = importlib.util.spec_from_file_location(
    module_name, monitor_path
)

if spec is None or spec.loader is None:
    raise RuntimeError("Deadlock Monitor 파일을 찾을 수 없습니다.")

module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

DeadlockMonitor = module.DeadlockMonitor


# 2. 실제 IL 환경 준비

# Corridor 시나리오 불러오기
scenario_path = (
    root.parent
    / "MAPF_common_logging"
    / "scenarios"
    / "controlled"
    / "scenario_corridor_bay_n2.json"
)

map_path = scenario_path.parent / "map_corridor_bay_n2.npy"

with scenario_path.open("r", encoding="utf-8") as f:
    scenario_data = json.load(f)

grid = np.load(map_path)

# 프로젝트 내부 좌표계: (row, col)
starts = {
    i: xy_to_rowcol(agent["start"])
    for i, agent in enumerate(scenario_data["agents"])
}

goals = {
    i: xy_to_rowcol(agent["goal"])
    for i, agent in enumerate(scenario_data["agents"])
}

policy = NavHintPolicy(DEFAULT_CKPT)

sim = MAPFStepSimulator(
    cbs_solver_root=None,
    max_steps=160,
)

obs = sim.reset(grid, starts, goals)
policy.reset(grid, goals)

monitor = DeadlockMonitor()
monitor.reset(dict(sim._positions), goals)

print("Scenario:", scenario_path.name)
print("Initial Positions:", starts)


# 3. IL 실행 + Deadlock 감지
trigger_found = False

for _ in range(160):

    # 실제 IL 모델이 Action 생성
    actions = policy.act(obs, sim._positions)

    # 실제 Simulator 이동
    obs, done, info = sim.step(actions)

    # 이동 후 Monitor 검사
    result = monitor.observe_step(
        info["t"],
        dict(sim._positions),
        {},
    )

    reason = result.trigger_reason_code

    
    if reason == "TRIGGER_FULL_REPEAT":
        trigger_found = True

        print("\nDEADLOCK TRIGGER DETECTED")
        print("Timestep:", info["t"])
        print("Positions:", dict(sim._positions))
        print("Reason:", reason)
        print("Integrity:", result.overlap_integrity_status)

        # 실제 Trigger 시점의 로봇 위치
        current = {
            aid: tuple(pos)
            for aid, pos in sim._positions.items()
        }

        # 위치 겹침이 발견되면 CBS 실행 차단
        if (
            result.overlap_integrity_status != "CLEAN"
            or len(set(current.values())) != len(current)
        ):
            print("CBS BLOCKED: Position integrity issue")
            break

        # 실제 Simulator의 Agent ID 확인
        agent_ids = sorted(current)

        # 오늘 만든 CBS Runtime Adapter 호출
        cbs_result = run_cbs_runtime(
            current_positions=current,
            goals=goals,
            grid_map=grid,
            timeout_sec=30,
        )

        print("\n=== LIVE CBS RECOVERY ===")
        print("CBS Status:", cbs_result.status)
        print("CBS Latency:", round(cbs_result.latency_sec, 4))
        print("CBS Paths:", cbs_result.paths)
        print("CBS Error:", cbs_result.error)

        if cbs_result.status == "success":
            assert cbs_result.paths is not None

            # CBS 경로가 Trigger 당시 실제 위치에서 시작하는지 검사
            for aid in agent_ids:
                assert (
                    tuple(cbs_result.paths[aid][0])
                    == current[aid]
                )

            print("LIVE IL TRIGGER TO CBS PASS")

            
            # CBS 경로를 Simulator에서 끝까지 실행
            paths = cbs_result.paths

            action_map = {
                (-1, 0): 0,  # UP
                (1, 0): 1,   # DOWN
                (0, -1): 2,  # LEFT
                (0, 1): 3,   # RIGHT
                (0, 0): 4,   # WAIT
            }

            total_steps = len(paths[agent_ids[0]]) - 1

            print("\n=== LIVE RECOVERY EXECUTION ===")

            for k in range(total_steps):
                recovery_actions = {}
                expected = {}

                for aid in agent_ids:
                    path = paths[aid]

                    current_pos = tuple(path[k])
                    next_pos = tuple(path[k + 1])

                    assert tuple(sim._positions[aid]) == current_pos

                    delta = (
                        next_pos[0] - current_pos[0],
                        next_pos[1] - current_pos[1],
                    )

                    recovery_actions[aid] = action_map[delta]
                    expected[aid] = next_pos

                # 기존 Simulator에서 실제 이동
                obs, done, info = sim.step(recovery_actions)

                actual = {
                    aid: tuple(sim._positions[aid])
                    for aid in agent_ids
                }

                # 계획대로 이동했는지 확인
                assert actual == expected, (
                    f"Recovery mismatch at t={info['t']}"
                )

                # 위치 겹침 검사
                assert len(set(actual.values())) == len(actual)

                print(f"t={info['t']} Positions: {actual}")

                if done and k < total_steps - 1:
                    raise AssertionError("Recovery ended too early")

            # 최종 목표 도착 확인
            final_positions = {
                aid: tuple(sim._positions[aid])
                for aid in agent_ids
            }

            assert final_positions == goals
            assert info["all_at_goal"]

            print("\nAll At Goal:", info["all_at_goal"])
            print("LIVE IL TO CBS RECOVERY FULL PASS")

        else:
            print("CBS RECOVERY FAILED OR BLOCKED")

        break

    if done:
        print("\nIL Episode Finished")
        print("All At Goal:", info["all_at_goal"])
        print("Timed Out:", info["timed_out"])
        break


print("\nTrigger Found:", trigger_found)
print("IL MONITOR TEST COMPLETE")
