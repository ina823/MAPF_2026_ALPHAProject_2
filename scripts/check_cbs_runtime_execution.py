
import numpy as np

from src.cbs_runtime_adapter import run_cbs_runtime
from src.simulator.mapf_step_simulator import MAPFStepSimulator


# 1. Action 변환표
DELTA_TO_ACTION = {
    (-1, 0): 0,  # UP
    (1, 0): 1,   # DOWN
    (0, -1): 2,  # LEFT
    (0, 1): 3,   # RIGHT
    (0, 0): 4,   # WAIT
}


# 2. 테스트 환경
grid = np.zeros((3, 3), dtype=int)

starts = {
    7: (1, 2),
    2: (1, 0),
}

goals = {
    7: (1, 0),
    2: (1, 2),
}

sim = MAPFStepSimulator(
    cbs_solver_root=None,
    max_steps=40,
)

sim.reset(grid, starts, goals)


# 3. CBS Runtime 호출
result = run_cbs_runtime(
    current_positions=dict(sim._positions),
    goals=goals,
    grid_map=grid,
    timeout_sec=10,
)

print("CBS Status:", result.status)

if result.status != "success" or result.paths is None:
    raise RuntimeError(
        f"CBS failed; no movement executed: {result.error}"
    )


# 4. CBS 경로를 Action으로 변환하여 실행
paths = result.paths
path_length = len(next(iter(paths.values())))

for k in range(1, path_length):

    actions = {}

    for aid, path in paths.items():
        before = path[k - 1]
        after = path[k]

        delta = (
            after[0] - before[0],
            after[1] - before[1],
        )

        actions[aid] = DELTA_TO_ACTION[delta]

    # 실제 Simulator 이동
    _, done, info = sim.step(actions)

    # 예상 위치와 실제 위치 비교
    expected = {
        aid: path[k]
        for aid, path in paths.items()
    }

    actual = dict(sim._positions)

    print(
        f"t={info['t']} "
        f"Actions={actions} "
        f"Match={actual == expected}"
    )

    if actual != expected:
        raise AssertionError(
            f"Position mismatch: {actual} != {expected}"
        )


# 5. 최종 목표 도착 확인
assert all(
    sim._positions[aid] == goals[aid]
    for aid in goals
)

print("ALL AT GOAL:", True)
print("CBS RUNTIME EXECUTION PASS")
