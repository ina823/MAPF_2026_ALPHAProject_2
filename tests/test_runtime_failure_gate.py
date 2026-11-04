import runpy
import unittest
from unittest.mock import patch

from src.replan_status import ReplanResult
from src.simulator.mapf_step_simulator import MAPFStepSimulator


class TestRuntimeFailureGate(unittest.TestCase):

    def check_no_move(self, failure_status):

        state = {
            "cbs_called": False,
            "steps_after_cbs": 0,
        }

        original_step = MAPFStepSimulator.step

        # 1. CBS 실패 결과를 인위적으로 생성
        def fake_replan(*args, **kwargs):
            state["cbs_called"] = True

            return ReplanResult(
                status=failure_status,
                paths=None,
                latency_sec=0.01,
                error=f"Injected {failure_status}",
            )

        # 2. CBS 실패 이후 추가 이동 여부 검사
        def guarded_step(self, actions):

            if state["cbs_called"]:
                state["steps_after_cbs"] += 1

                raise AssertionError(
                    "CBS 실패 이후 Simulator 이동 발생!"
                )

            return original_step(self, actions)

        # 3. 실제 Corridor 실행 코드에 실패 주입
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

        # 4. 실패 후 안전성 확인
        self.assertTrue(state["cbs_called"])
        self.assertTrue(context["trigger_found"])

        self.assertEqual(context["sim"]._t, 5)

        self.assertEqual(
            dict(context["sim"]._positions),
            {
                0: (2, 4),
                1: (2, 6),
            },
        )

        self.assertEqual(state["steps_after_cbs"], 0)

    def test_no_solution_no_move(self):
        self.check_no_move("no_solution")

    def test_solver_error_no_move(self):
        self.check_no_move("solver_error")


if __name__ == "__main__":
    unittest.main()