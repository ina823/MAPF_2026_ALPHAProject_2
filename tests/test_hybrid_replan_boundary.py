"""Unit tests for src.hybrid.replan_boundary (Global Hybrid Runner v0).

The boundary only checks the runner-side contract (status, agent-id mapping,
Path[0] == S_t, every move convertible to a Simulator action). Geometric path
validation is the injected validator's job (the CBS branch's Path Guard) and
is NOT re-implemented here -- these tests use test-only validator stubs.
"""

import unittest

from src.hybrid.replan_boundary import (
    CBS_STATUSES,
    accept_replan_result,
    check_validator_pairing,
)
from tests.hybrid_mocks import (
    DOWN, RIGHT, UP, WAIT,
    AcceptAllValidator,
    MockReplanResult,
    PATHS_TO_GOALS,
    RecordingPlanner,
    RejectAllValidator,
    open_grid,
    success,
)

S_T = {0: (0, 0), 1: (2, 0)}
GOALS = {0: (0, 2), 1: (2, 2)}


def _accept(result, *, agent_order=(0, 1), current=S_T, goals=GOALS, validator=None):
    return accept_replan_result(
        result,
        agent_order=agent_order,
        current_positions=current,
        goals=goals,
        grid=open_grid(),
        path_validator=validator or AcceptAllValidator(),
    )


class TestStatus(unittest.TestCase):

    def test_contract_has_six_statuses(self):
        self.assertEqual(
            set(CBS_STATUSES),
            {"success", "timeout", "no_solution", "invalid_path", "input_error", "solver_error"},
        )

    def test_non_success_statuses_rejected_even_with_valid_paths(self):
        for status in ("timeout", "no_solution", "invalid_path", "input_error", "solver_error"):
            with self.subTest(status=status):
                validator = AcceptAllValidator()
                outcome = _accept(MockReplanResult(status=status, paths=PATHS_TO_GOALS),
                                  validator=validator)
                self.assertFalse(outcome.accepted)
                self.assertEqual(outcome.reason, f"CBS_{status.upper()}")
                self.assertIsNone(outcome.plan)
                self.assertEqual(validator.calls, [])

    def test_unknown_status_rejected(self):
        outcome = _accept(MockReplanResult(status="ok", paths=PATHS_TO_GOALS))
        self.assertFalse(outcome.accepted)
        self.assertEqual(outcome.reason, "CBS_UNKNOWN_STATUS")

    def test_success_without_paths_rejected(self):
        outcome = _accept(MockReplanResult(status="success", paths=None))
        self.assertFalse(outcome.accepted)
        self.assertEqual(outcome.reason, "BOUNDARY_PATHS_TYPE")


class TestAgentMapping(unittest.TestCase):

    def test_list_index_maps_to_sorted_agent_order(self):
        current = {3: (0, 0), 7: (2, 0)}
        goals = {3: (0, 2), 7: (1, 0)}
        paths = [
            [(0, 0), (0, 1), (0, 2)],   # index 0 -> agent 3
            [(2, 0), (1, 0)],           # index 1 -> agent 7
        ]
        outcome = _accept(success(paths), agent_order=(3, 7), current=current, goals=goals)
        self.assertTrue(outcome.accepted, outcome.reason)
        self.assertEqual(outcome.plan.paths[3], paths[0])
        self.assertEqual(outcome.plan.paths[7], paths[1])

    def test_dict_paths_keyed_by_index_also_accepted(self):
        paths = {0: PATHS_TO_GOALS[0], 1: PATHS_TO_GOALS[1]}
        outcome = _accept(success(paths))
        self.assertTrue(outcome.accepted, outcome.reason)
        self.assertEqual(outcome.plan.paths[1], PATHS_TO_GOALS[1])

    def test_dict_paths_keyed_by_agent_id_not_index_rejected(self):
        current = {3: (0, 0), 7: (2, 0)}
        goals = {3: (0, 2), 7: (2, 2)}
        paths = {3: PATHS_TO_GOALS[0], 7: PATHS_TO_GOALS[1]}
        outcome = _accept(success(paths), agent_order=(3, 7), current=current, goals=goals)
        self.assertFalse(outcome.accepted)
        self.assertEqual(outcome.reason, "BOUNDARY_PATH_COUNT")

    def test_path_count_mismatch_rejected(self):
        outcome = _accept(success(PATHS_TO_GOALS[:1]))
        self.assertFalse(outcome.accepted)
        self.assertEqual(outcome.reason, "BOUNDARY_PATH_COUNT")

    def test_swapped_paths_rejected_by_start_check(self):
        outcome = _accept(success(list(reversed(PATHS_TO_GOALS))))
        self.assertFalse(outcome.accepted)
        self.assertEqual(outcome.reason, "BOUNDARY_START_MISMATCH")


class TestPathStartAndFirstMove(unittest.TestCase):

    def test_path0_must_equal_current_position(self):
        paths = [[(1, 0), (0, 0), (0, 1)], PATHS_TO_GOALS[1]]
        outcome = _accept(success(paths))
        self.assertFalse(outcome.accepted)
        self.assertEqual(outcome.reason, "BOUNDARY_START_MISMATCH")

    def test_empty_path_rejected(self):
        outcome = _accept(success([[], PATHS_TO_GOALS[1]]))
        self.assertFalse(outcome.accepted)
        self.assertEqual(outcome.reason, "BOUNDARY_EMPTY_PATH")

    def test_malformed_cell_rejected(self):
        outcome = _accept(success([[(0, 0), (0,)], PATHS_TO_GOALS[1]]))
        self.assertFalse(outcome.accepted)
        self.assertEqual(outcome.reason, "BOUNDARY_MALFORMED_PATH")

    def test_first_action_and_expected_come_from_path1(self):
        paths = [[(0, 0), (1, 0), (1, 1)], [(2, 0), (2, 1), (2, 2)]]
        outcome = _accept(success(paths), goals={0: (1, 1), 1: (2, 2)})
        self.assertTrue(outcome.accepted, outcome.reason)
        self.assertEqual(outcome.plan.actions[0], {0: DOWN, 1: RIGHT})
        self.assertEqual(outcome.plan.expected[0], {0: (1, 0), 1: (2, 1)})

    def test_all_moves_converted_and_short_paths_wait_at_end(self):
        paths = [[(0, 0)], [(2, 0), (1, 0), (0, 0)]]
        outcome = _accept(success(paths), current={0: (0, 0), 1: (2, 0)},
                          goals={0: (0, 0), 1: (0, 0)})
        # (Overlap at (0,0) is the validator's concern, not the boundary's;
        # AcceptAllValidator lets it through so only conversion is checked.)
        self.assertTrue(outcome.accepted, outcome.reason)
        self.assertEqual(outcome.plan.actions, [{0: WAIT, 1: UP}, {0: WAIT, 1: UP}])
        self.assertEqual(outcome.plan.expected[-1], {0: (0, 0), 1: (0, 0)})

    def test_latency_converted_to_ms(self):
        outcome = _accept(success(PATHS_TO_GOALS, latency_sec=0.25))
        self.assertEqual(outcome.plan.latency_ms, 250.0)
        outcome = _accept(success(PATHS_TO_GOALS, latency_sec=None))
        self.assertIsNone(outcome.plan.latency_ms)


class TestIllegalMoveRejectedBeforeExecution(unittest.TestCase):
    """The Simulator would silently turn these into WAIT/stay; the boundary
    must refuse the whole plan instead, before any move."""

    def test_two_cell_jump_rejected(self):
        paths = [[(0, 0), (0, 2)], PATHS_TO_GOALS[1]]
        outcome = _accept(success(paths))
        self.assertFalse(outcome.accepted)
        self.assertEqual(outcome.reason, "BOUNDARY_NON_ADJACENT")
        self.assertIsNone(outcome.plan)

    def test_diagonal_move_rejected(self):
        paths = [[(0, 0), (1, 1), (0, 2)], PATHS_TO_GOALS[1]]
        outcome = _accept(success(paths))
        self.assertFalse(outcome.accepted)
        self.assertEqual(outcome.reason, "BOUNDARY_NON_ADJACENT")

    def test_late_illegal_move_rejects_whole_plan(self):
        paths = [[(0, 0), (0, 1), (0, 2), (2, 2)], PATHS_TO_GOALS[1]]
        outcome = _accept(success(paths))
        self.assertFalse(outcome.accepted)
        self.assertEqual(outcome.reason, "BOUNDARY_NON_ADJACENT")

    def test_illegal_move_rejected_even_if_validator_accepts(self):
        validator = AcceptAllValidator()
        outcome = _accept(success([[(0, 0), (0, 2)], PATHS_TO_GOALS[1]]), validator=validator)
        self.assertFalse(outcome.accepted)


class TestInjectedValidator(unittest.TestCase):

    def test_validator_reject(self):
        outcome = _accept(success(PATHS_TO_GOALS), validator=RejectAllValidator("vertex_conflict"))
        self.assertFalse(outcome.accepted)
        self.assertEqual(outcome.reason, "VALIDATOR_REJECT:vertex_conflict")
        self.assertIsNone(outcome.plan)

    def test_validator_receives_agent_id_keyed_inputs(self):
        validator = AcceptAllValidator()
        current = {3: (0, 0), 7: (2, 0)}
        goals = {3: (0, 2), 7: (2, 2)}
        outcome = _accept(success(PATHS_TO_GOALS), agent_order=(3, 7),
                          current=current, goals=goals, validator=validator)
        self.assertTrue(outcome.accepted, outcome.reason)
        self.assertEqual(validator.calls[0]["starts"], current)
        self.assertEqual(validator.calls[0]["goals"], goals)
        self.assertEqual(validator.calls[0]["paths"][7], PATHS_TO_GOALS[1])

    def test_validator_exception_rejected(self):
        def boom(grid, starts, goals, paths):
            raise RuntimeError("guard crashed")
        outcome = _accept(success(PATHS_TO_GOALS), validator=boom)
        self.assertFalse(outcome.accepted)
        self.assertEqual(outcome.reason, "VALIDATOR_ERROR")

    def test_validator_non_bool_result_rejected(self):
        outcome = _accept(success(PATHS_TO_GOALS), validator=lambda g, s, go, p: ("yes", ""))
        self.assertFalse(outcome.accepted)
        self.assertEqual(outcome.reason, "VALIDATOR_ERROR")


class TestValidatorPairing(unittest.TestCase):

    def test_missing_validator_refused(self):
        with self.assertRaises(ValueError):
            check_validator_pairing(RecordingPlanner(success(PATHS_TO_GOALS)), None)

    def test_test_only_validator_refused_with_real_planner(self):
        def real_like_planner(current_positions, goals, grid):
            raise AssertionError("not called")
        with self.assertRaises(ValueError):
            check_validator_pairing(real_like_planner, AcceptAllValidator())

    def test_test_only_validator_allowed_with_mock_planner(self):
        check_validator_pairing(RecordingPlanner(success(PATHS_TO_GOALS)), AcceptAllValidator())

    def test_real_validator_allowed_with_real_planner(self):
        def real_like_planner(current_positions, goals, grid):
            raise AssertionError("not called")

        def real_like_validator(grid, starts, goals, paths):
            return True, ""
        check_validator_pairing(real_like_planner, real_like_validator)


if __name__ == "__main__":
    unittest.main()
