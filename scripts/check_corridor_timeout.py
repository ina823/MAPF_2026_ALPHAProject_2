
import runpy
from unittest.mock import patch

from src.cbs_runtime_adapter import CBSRuntimeResult
from src.simulator.mapf_step_simulator import MAPFStepSimulator


# 1. 테스트 상태 기록
state = {
    "cbs_called": False,
    "steps_after_cbs": 0,
}

original_step = MAPFStepSimulator.step


# 2. CBS Timeout 강제 발생
def fake_cbs(*args, **kwargs):
    state["cbs_called"] = True

    return CBSRuntimeResult(
        status="timeout",
        paths=None,
        latency_sec=30.0,
        error="Injected CBS timeout",
    )


# 3. CBS 호출 이후 이동 금지
def guarded_step(self, actions):

    if state["cbs_called"]:
        state["steps_after_cbs"] += 1

        raise AssertionError(
            "CBS timeout 이후 Simulator 이동 발생!"
        )

    return original_step(self, actions)


# 4. 실제 Corridor 코드에서 CBS 결과만 교체
with patch(
    "src.cbs_runtime_adapter.run_cbs_runtime",
    side_effect=fake_cbs,
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


# 5. 실패 처리 결과 검증
assert state["cbs_called"]

assert context["trigger_found"]

assert context["sim"]._t == 5

assert dict(context["sim"]._positions) == {
    0: (2, 4),
    1: (2, 6),
}

assert state["steps_after_cbs"] == 0

print("\nCBS TIMEOUT NO-MOVE PASS")
