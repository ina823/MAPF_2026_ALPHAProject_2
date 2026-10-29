
import runpy
from unittest.mock import patch

from src.replan_status import ReplanResult
from src.simulator.mapf_step_simulator import MAPFStepSimulator


# CBS 실패 이후 추가 이동 여부 확인
cbs_called = False
original_step = MAPFStepSimulator.step


def fake_replan(*args, **kwargs):
    global cbs_called
    cbs_called = True

    # 테스트용 CBS Timeout 결과
    return ReplanResult(
        status="timeout",
        paths=None,
        latency_sec=30.0,
        error="Simulated CBS timeout",
    )


def guarded_step(self, actions):
    if cbs_called:
        raise AssertionError(
            "CBS 실패 이후 Simulator가 이동했습니다."
        )

    return original_step(self, actions)


# 실제 IL 실행 코드에서 CBS 결과만 가짜 Timeout으로 교체
with patch(
    "src.replan_status.replan_with_status",
    side_effect=fake_replan,
):
    with patch.object(
        MAPFStepSimulator,
        "step",
        guarded_step,
    ):
        result = runpy.run_module(
            "scripts.check_il_monitor",
            run_name="__main__",
        )


# 실패 시 위치와 상태 검증
assert cbs_called
assert result["sim"]._t == 5
assert dict(result["sim"]._positions) == {
    0: (2, 4),
    1: (2, 6),
}

print("\nCBS FAILURE GATE PASS")
