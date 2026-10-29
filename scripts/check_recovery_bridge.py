
import numpy as np

from src.replan_status import replan_with_status
from src.simulator.mapf_step_simulator import MAPFStepSimulator


# Action 규칙
ACTION_FROM_DELTA = {
    (-1, 0): 0,  # UP
    (1, 0): 1,   # DOWN
    (0, -1): 2,  # LEFT
    (0, 1): 3,   # RIGHT
    (0, 0): 4,   # WAIT
}


# 1. 지도 및 로봇 정보
grid = np.zeros((3, 3), dtype=int)

starts = {
    0: (1, 0),
    1: (1, 2),
}

goals = {
    0: (1, 2),
    1: (1, 0),
}


# 2. Simulator 초기화
sim = MAPFStepSimulator(
    cbs_solver_root=None,
    max_steps=50,
)

sim.reset(grid, starts, goals)

agent_ids = sorted(starts.keys())

current_positions = [
    tuple(sim._positions[aid]) for aid in agent_ids
]
goal_positions = [
    goals[aid] for aid in agent_ids
]


# 3. CBS 재계획 + Path Guard 검사
result = replan_with_status(
    current_positions,
    goal_positions,
    grid,
    timeout_sec=10,
)

print("CBS Status:", result.status)
print("CBS Latency:", round(result.latency_sec, 4), "sec")

if result.status != "success" or result.paths is None:
    print("RECOVERY BLOCKED:", result.error)
    raise SystemExit(1)

paths = result.paths

print("\nCBS Paths:")
for index, agent_id in enumerate(agent_ids):
    print(f"Agent {agent_id}: {paths[index]}")


# 4. 전체 경로 실행
total_steps = len(paths[0]) - 1

print("\n=== Recovery Execution ===")

for t in range(total_steps):

    actions = {}
    expected_positions = {}

    for index, agent_id in enumerate(agent_ids):
        path = paths[index]

        current = path[t]
        next_pos = path[t + 1]

        # 현재 Simulator 위치와 경로가 일치하는지 확인
        if tuple(sim._positions[agent_id]) != current:
            raise AssertionError(
                f"t={t}: Agent {agent_id} current position mismatch"
            )

        delta = (
            next_pos[0] - current[0],
            next_pos[1] - current[1],
        )

        actions[agent_id] = ACTION_FROM_DELTA[delta]
        expected_positions[agent_id] = next_pos

    # 실제 한 스텝 이동
    obs, done, info = sim.step(actions)

    actual_positions = {
        aid: tuple(sim._positions[aid])
        for aid in agent_ids
    }

    print(f"\nt={t+1}")
    print("Actions :", actions)
    print("Expected:", expected_positions)
    print("Actual  :", actual_positions)

    # CBS가 예상한 위치와 Simulator 실제 위치 비교
    assert actual_positions == expected_positions, (
        f"Recovery path mismatch at t={t+1}"
    )

    # 계획된 경로가 끝나기 전에 Simulator가 종료되면 실패
    if done and t < total_steps - 1:
        raise AssertionError(
            f"Simulator ended before CBS path finished at t={t+1}"
        )


# 5. 최종 목표 도착 확인
final_positions = {
    aid: tuple(sim._positions[aid])
    for aid in agent_ids
}

assert final_positions == goals, "Agents did not reach all goals"

print("\nFinal Positions:", final_positions)
print("All At Goal:", final_positions == goals)
print("RECOVERY FULL PATH PASS")
