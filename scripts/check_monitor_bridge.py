
import importlib.util
import sys
from pathlib import Path

import numpy as np

from src.replan_status import replan_with_status
from src.simulator.mapf_step_simulator import MAPFStepSimulator


# 1. Common Logging의 Deadlock Monitor 불러오기
project_root = Path(__file__).resolve().parents[1]

monitor_path = (
    project_root.parent
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
    raise RuntimeError("Deadlock Monitor 모듈을 찾을 수 없습니다.")

monitor_module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = monitor_module
spec.loader.exec_module(monitor_module)

DeadlockMonitor = monitor_module.DeadlockMonitor


# 2. 테스트 환경 생성
grid = np.zeros((3, 3), dtype=int)

starts = {
    0: (1, 0),
    1: (1, 2),
}

goals = {
    0: (1, 2),
    1: (1, 0),
}

sim = MAPFStepSimulator(
    cbs_solver_root=None,
    max_steps=20,
)

sim.reset(grid, starts, goals)

monitor = DeadlockMonitor()
monitor.reset(dict(sim._positions), goals)


# 3. 반복 상태 발생 및 CBS 자동 호출
trigger_handled = False
cbs_call_count = 0

for _ in range(2):

    # 테스트용: 두 로봇 모두 WAIT
    actions = {0: 4, 1: 4}

    _, done, info = sim.step(actions)

    # WAIT만 했으므로 충돌 이벤트는 없음
    monitor_result = monitor.observe_step(
        info["t"],
        dict(sim._positions),
        {},
    )

    reason = monitor_result.trigger_reason_code

    print(f"t={info['t']} Trigger: {reason}")

    # 최초 Trigger에서만 CBS 실행
    if reason == "TRIGGER_FULL_REPEAT" and not trigger_handled:

        trigger_handled = True

        current = dict(sim._positions)

        # 중복 위치가 있으면 CBS 실행 차단
        if len(set(current.values())) != len(current):
            raise RuntimeError("Position overlap detected")

        agent_ids = sorted(current)

        current_positions = [
            current[aid] for aid in agent_ids
        ]

        goal_positions = [
            goals[aid] for aid in agent_ids
        ]

        result = replan_with_status(
            current_positions,
            goal_positions,
            grid,
            timeout_sec=10,
        )

        cbs_call_count += 1

        print("CBS Status:", result.status)
        print("CBS Paths:", result.paths)

        assert result.status == "success"
        assert result.paths is not None


# 4. 검증
assert trigger_handled
assert cbs_call_count == 1

print("CBS Call Count:", cbs_call_count)
print("MONITOR TO CBS PASS")
