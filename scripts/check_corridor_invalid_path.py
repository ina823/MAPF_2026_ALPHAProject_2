
import runpy
from unittest.mock import patch

from src.replan_status import ReplanResult
from src.simulator.mapf_step_simulator import MAPFStepSimulator


# 1. 테스트 상태 기록
state = {
    "cbs_called": False,
    "steps_after_cbs": 0,
}

original_step = MAPFStepSimulator.step


# 2. 잘못된 CBS 경로 강제 반환
def fake_replan(*args, **kwargs):
    state["cbs_called"] = True

    return ReplanResult(
        status="success",
        paths={
            0: [
                (2, 4),
                (2, 6),  # 잘못된 이동: 한 번에 두 칸
                (2, 5),
                (2, 6),
                (2, 7),
                (2, 8),
                (2, 9),
                (2, 10),
                (2, 10),
            ],
            1: [
                (2, 6),
                (2, 5),
                (1, 5),
                (2, 5),
                (2, 4),
                (2, 3),
                (2, 2),
                (2, 1),
                (2, 0),
            ],
        },
        latency_sec=0.01,
    )


# 3. CBS 호출 이후 Simulator 이동 감시
def guarded_step(self, actions):
    if state["cbs_called"]:
        state["steps_after_cbs"] += 1
        raise AssertionError(
            "Invalid CBS path 이후 이동 발생!"
        )

    return original_step(self, actions)


# 4. 실제 Corridor 실행 중 잘못된 경로 주입
with patch(
    "src.cbs_runtime_adapter.replan_with_status",
    side_effect=fake_replan,
):
    with patch.object(
        MAPFStepSimulator,
        "step",
        guarded_step,
    ):
        context = runpy.run_module(
            "scripts.check_corridor_runtime",
            run_name="__main__",
        )


# 5. 검증
assert state["cbs_called"]
assert context["trigger_found"]
assert context["sim"]._t == 5

assert dict(context["sim"]._positions) == {
    0: (2, 4),
    1: (2, 6),
}

assert state["steps_after_cbs"] == 0

print("\nCBS INVALID PATH NO-MOVE PASS")
