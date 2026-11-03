"""Characterization tests for MAPFStepSimulator.

Alpha 1의 기존 동작을 기록하고,
Alpha 2에서 수정한 KI-1 연쇄 충돌 방지 동작도 검증한다.

KI-1 수정 후에는 충돌 처리 과정에서 이동이 취소되더라도
최종 위치에 Vertex Conflict가 발생하면 안 된다.
"""

import unittest

import numpy as np

from src.simulator.mapf_step_simulator import MAPFStepSimulator

UP, DOWN, LEFT, RIGHT, WAIT = 0, 1, 2, 3, 4


def make_sim(grid, starts, goals, max_steps=5):
    sim = MAPFStepSimulator(cbs_solver_root=None, max_steps=max_steps)
    sim.reset(np.array(grid), starts, goals)
    return sim


CORRIDOR = [[0, 0, 0, 0, 0]]  # 1 x 5, all free


class TestMovementRules(unittest.TestCase):
    def test_move_into_wall_stays(self):
        sim = make_sim([[0, 1, 0]], {0: (0, 0)}, {0: (0, 2)})
        sim.step({0: RIGHT})
        self.assertEqual(sim._positions, {0: (0, 0)})

    def test_move_out_of_map_stays(self):
        sim = make_sim([[0, 0, 0]], {0: (0, 0)}, {0: (0, 2)})
        sim.step({0: LEFT})
        self.assertEqual(sim._positions, {0: (0, 0)})

    def test_action_convention_row_col(self):
        grid = np.zeros((5, 5), dtype=int)
        expected = {UP: (1, 2), DOWN: (3, 2), LEFT: (2, 1), RIGHT: (2, 3), WAIT: (2, 2)}
        for action, pos in expected.items():
            sim = make_sim(grid, {0: (2, 2)}, {0: (0, 0)})
            sim.step({0: action})
            self.assertEqual(sim._positions[0], pos, f"action {action}")

    def test_vertex_collision_both_stay(self):
        sim = make_sim(CORRIDOR, {0: (0, 0), 1: (0, 2)}, {0: (0, 2), 1: (0, 0)})
        sim.step({0: RIGHT, 1: LEFT})
        self.assertEqual(sim._positions, {0: (0, 0), 1: (0, 2)})

    def test_moving_into_waiting_agent_both_stay(self):
        sim = make_sim(CORRIDOR, {0: (0, 0), 1: (0, 1)}, {0: (0, 1), 1: (0, 0)})
        sim.step({0: RIGHT, 1: WAIT})
        self.assertEqual(sim._positions, {0: (0, 0), 1: (0, 1)})

    def test_edge_swap_both_stay(self):
        sim = make_sim(CORRIDOR, {0: (0, 0), 1: (0, 1)}, {0: (0, 1), 1: (0, 0)})
        sim.step({0: RIGHT, 1: LEFT})
        self.assertEqual(sim._positions, {0: (0, 0), 1: (0, 1)})

    def test_following_into_vacated_cell_is_allowed(self):
        sim = make_sim(CORRIDOR, {0: (0, 0), 1: (0, 1)}, {0: (0, 2), 1: (0, 3)})
        sim.step({0: RIGHT, 1: RIGHT})
        self.assertEqual(sim._positions, {0: (0, 1), 1: (0, 2)})


class TestTermination(unittest.TestCase):
    def test_all_at_goal_terminates(self):
        sim = make_sim(CORRIDOR, {0: (0, 0)}, {0: (0, 1)}, max_steps=5)
        _, done, info = sim.step({0: RIGHT})
        self.assertTrue(done)
        self.assertEqual(info, {"t": 1, "all_at_goal": True, "timed_out": False})

    def test_timeout(self):
        sim = make_sim(CORRIDOR, {0: (0, 0)}, {0: (0, 4)}, max_steps=2)
        sim.step({0: RIGHT})
        _, done, info = sim.step({0: RIGHT})
        self.assertTrue(done)
        self.assertEqual(info, {"t": 2, "all_at_goal": False, "timed_out": True})

    def test_goal_reached_exactly_at_max_steps_reports_both_flags(self):
        # KI-2: timed_out is simply (t >= max_steps), even when the goal was reached on that step.
        sim = make_sim(CORRIDOR, {0: (0, 0)}, {0: (0, 1)}, max_steps=1)
        _, done, info = sim.step({0: RIGHT})
        self.assertTrue(done)
        self.assertTrue(info["all_at_goal"])
        self.assertTrue(info["timed_out"])


class TestObservation(unittest.TestCase):
    def setUp(self):
        self.sim = MAPFStepSimulator(cbs_solver_root=None)
        self.obs = self.sim.reset(np.zeros((6, 6), dtype=int), {0: (0, 0), 1: (0, 1)}, {0: (5, 5), 1: (1, 1)})

    def test_shapes_and_dtypes(self):
        g = self.obs[0]["grid"]
        self.assertEqual(g.shape, (3, 5, 5))
        self.assertEqual(g.dtype, np.float32)
        self.assertEqual(self.obs[0]["goal_dir"].shape, (2,))
        self.assertEqual(self.obs[0]["goal_dir"].dtype, np.float32)

    def test_goal_dir_is_straight_vector_row_col(self):
        np.testing.assert_array_equal(self.obs[0]["goal_dir"], np.array([5.0, 5.0], dtype=np.float32))

    def test_channels(self):
        g = self.obs[0]["grid"]              # agent 0 at (0,0): local centre is (2,2)
        self.assertEqual(int(g[0].sum()), 16)            # 25 - 9 in-map cells are outside the map => wall channel
        self.assertEqual(g[1, 2, 3], 1.0)                # other robot (agent 1 at (0,1)) -> local (2,3)
        self.assertEqual(g[2, 3, 3], 1.0)                # other agent's goal (1,1) -> local (3,3)
        self.assertEqual(int(g[:, 2, 2].sum()), 0)       # own cell/goal is never drawn
        self.assertLessEqual(int(g.sum(axis=0).max()), 1)  # at most one channel per cell

    def test_channel_priority_wall_over_robot_over_goal(self):
        grid = np.zeros((5, 5), dtype=int)
        grid[2, 3] = 1                                    # wall right of the centre agent (also agent 1's goal)
        sim = MAPFStepSimulator(cbs_solver_root=None)
        obs = sim.reset(grid, {0: (2, 2), 1: (1, 2)}, {0: (0, 0), 1: (2, 3)})
        g = obs[0]["grid"]
        self.assertEqual(g[0, 2, 3], 1.0)   # (2,3) is a wall AND agent 1's goal -> wall wins
        self.assertEqual(g[2, 2, 3], 0.0)
        self.assertEqual(g[1, 1, 2], 1.0)   # agent 1 at (1,2) -> local (1,2)


class TestKnownQuirkKI1(unittest.TestCase):
    """KI-1 수정 검증: 충돌 취소 후 연쇄 충돌이 발생하지 않아야 한다."""

    def test_cascade_overlap_1d(self):
        # A(col1)->col2 and C(col3)->col2 collide and both bounce back;
        # B(col0)->col1 enters A's old cell -> A and B share col1.
        sim = make_sim(CORRIDOR, {0: (0, 1), 1: (0, 0), 2: (0, 3)}, {0: (0, 4), 1: (0, 4), 2: (0, 0)})
        sim.step({0: RIGHT, 1: RIGHT, 2: LEFT})
        self.assertEqual(sim._positions, {
            0: (0, 1),
            1: (0, 0),
            2: (0, 3),
        })
        self.assertEqual(len(set(sim._positions.values())), 3)

    def test_swap_broken_by_third_agent_vertex_bounce_2d(self):
        # Pattern seen in the Alpha 1 rollout of maze_s11_n3 (t=23):
        # agent0 (9,2) UP -> (8,2); agent1 (8,3) LEFT -> (8,2); agent2 (8,2) DOWN -> (9,2).
        # agents 0 and 1 collide on (8,2) and bounce; the 0<->2 swap is no longer detected;
        # agent2 lands on (9,2) where agent0 was bounced back to.
        sim = make_sim(np.zeros((11, 11), dtype=int),
                       {0: (9, 2), 1: (8, 3), 2: (8, 2)}, {0: (0, 0), 1: (0, 1), 2: (0, 2)})
        sim.step({0: UP, 1: LEFT, 2: DOWN})
        self.assertEqual(sim._positions, {
            0: (9, 2),
            1: (8, 3),
            2: (8, 2),
        })
        self.assertEqual(len(set(sim._positions.values())), 3)


if __name__ == "__main__":
    unittest.main()
