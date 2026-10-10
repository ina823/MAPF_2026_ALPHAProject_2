"""Tests for src/common/deadlock_monitor.py (Phase 3A: full joint-state
repeat detection only -- see docs/deadlock_definition_v1.md).

Two kinds of test data are used, and they are never mixed:

  * REAL -- actual `action_id` sequences read off real, already-committed
    `outputs/logs/*_steps.csv` from genuine `scripts/run_il.py` + the real
    NavHintPolicy checkpoint (Phase 1/2 evidence). Replaying these fixed
    actions through the unmodified MAPFStepSimulator reproduces the exact
    same trajectory (Phase 1's reproducibility audit: 11/11 identical
    reruns), so this is a legitimate, torch-free way to cross-check the
    Monitor against recorded first_seen/first_repeat/period/blocking_type
    -- it is NOT a synthetic claim about IL behaviour.
  * SYNTHETIC -- hand-fed positions/conflict_types with no claim of being
    real IL behaviour, used only to exercise Monitor logic (period>1,
    S_0-return, no-blocking-evidence) for which no real example exists yet
    (docs/deadlock_definition_v1.md Section 6/8.2 explicitly record this as
    an evidence gap). Each such test says so in its name/docstring.

TestKnownQuirkKI1Separation is kept in its own class on purpose and is
never combined with the "did the Monitor classify this correctly" tests --
per docs/deadlock_definition_v1.md Section 8.1, KI-1 instances get a normal
trigger_type computed, but must never be pooled into detector-accuracy
statistics.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from src.common.deadlock_monitor import DeadlockMonitor
from src.simulator.mapf_step_simulator import MAPFStepSimulator
from src.common.scenario_loader import load_map_generator_scenario

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UP, DOWN, LEFT, RIGHT, WAIT = 0, 1, 2, 3, 4


def load(path):
    grid, starts, goals = load_map_generator_scenario(PROJECT_ROOT / path)
    return grid, dict(enumerate(starts)), dict(enumerate(goals))


def replay(grid, starts, goals, actions_per_agent, max_steps):
    """Replay a FIXED action sequence (not a policy) through the real,
    unmodified MAPFStepSimulator + a fresh DeadlockMonitor. Returns the
    monitor's final result. `actions_per_agent`: {agent_id: [action_id, ...]}."""
    sim = MAPFStepSimulator(cbs_solver_root=None, max_steps=max_steps)
    sim.reset(grid, dict(starts), dict(goals))
    monitor = DeadlockMonitor()
    monitor.reset(sim._positions, goals)
    for t in range(max_steps):
        actions = {aid: (seq[t] if t < len(seq) else WAIT) for aid, seq in actions_per_agent.items()}
        sim.step(actions)
        result = monitor.observe_step(sim._t, sim._positions, sim._last_conflict_types)
        if result.trigger_type == "FULL":
            break
    return monitor.result


class TestRealCrossCheck(unittest.TestCase):
    """REAL data: action_id sequences read from actual outputs/logs/*_steps.csv
    (genuine NavHintPolicy rollouts), cross-checked against the
    first_seen/first_repeat/period/blocking_type already recorded in
    scenarios/controlled_variants/RESULTS.md and docs/deadlock_definition_v1.md.
    """

    def test_corridor_bay_n2_original_vertex(self):
        # outputs/logs/20261010_203403_corridor_bay_n2_IL_steps.csv
        grid, starts, goals = load("scenarios/controlled/scenario_corridor_bay_n2.json")
        actions = {0: [RIGHT] * 6, 1: [LEFT] * 6}
        r = replay(grid, starts, goals, actions, max_steps=10)
        self.assertEqual(r.trigger_type, "FULL")
        self.assertEqual(r.first_seen, 4)
        self.assertEqual(r.trigger_timestep, 5)
        self.assertEqual(r.period, 1)
        self.assertEqual(r.blocking_type, "VERTEX")
        self.assertEqual(r.overlap_integrity_status, "CLEAN")

    def test_intersection_n2_original_vertex(self):
        # outputs/logs/20261010_203644_intersection_n2_IL_steps.csv
        grid, starts, goals = load("scenarios/controlled/scenario_intersection_n2.json")
        actions = {0: [RIGHT] * 4, 1: [DOWN] * 4}
        r = replay(grid, starts, goals, actions, max_steps=10)
        self.assertEqual(r.trigger_type, "FULL")
        self.assertEqual(r.first_seen, 1)
        self.assertEqual(r.trigger_timestep, 2)
        self.assertEqual(r.period, 1)
        self.assertEqual(r.blocking_type, "VERTEX")

    def test_bottleneck_n4_original_vertex(self):
        # outputs/logs/20261010_203842_bottleneck_n4_IL_steps.csv
        grid, starts, goals = load("scenarios/controlled/scenario_bottleneck_n4.json")
        actions = {
            0: [RIGHT, DOWN, DOWN, DOWN],
            1: [UP, RIGHT, RIGHT, RIGHT],
            2: [DOWN, DOWN, DOWN, DOWN],
            3: [UP, UP, UP, UP],
        }
        r = replay(grid, starts, goals, actions, max_steps=10)
        self.assertEqual(r.trigger_type, "FULL")
        self.assertEqual(r.first_seen, 1)
        self.assertEqual(r.trigger_timestep, 2)
        self.assertEqual(r.period, 1)
        self.assertEqual(r.blocking_type, "VERTEX")

    def test_corridor_3rdagent_perturbation_edge(self):
        # outputs/logs/20261011_003253_corridor_bay_n2_3rdagent_perturbation_IL_steps.csv
        grid, starts, goals = load(
            "scenarios/experimental_v1/scenario_corridor_bay_n2_3rdagent_perturbation.json")
        actions = {
            0: [RIGHT] * 6,
            1: [LEFT] * 6,
            2: [LEFT, LEFT, LEFT, LEFT, WAIT, WAIT],
        }
        r = replay(grid, starts, goals, actions, max_steps=10)
        self.assertEqual(r.trigger_type, "FULL")
        self.assertEqual(r.first_seen, 4)
        self.assertEqual(r.trigger_timestep, 5)
        self.assertEqual(r.period, 1)
        self.assertEqual(r.blocking_type, "EDGE")
        self.assertEqual(r.overlap_integrity_status, "CLEAN")

    def test_success_cycle_ring_n4_never_triggers(self):
        # outputs/logs/20261010_213604_cycle_ring_n4_IL_steps.csv -- success in 2 steps
        grid, starts, goals = load("scenarios/controlled/scenario_cycle_ring_n4.json")
        actions = {0: [RIGHT, RIGHT], 1: [DOWN, DOWN], 2: [LEFT, LEFT], 3: [UP, UP]}
        r = replay(grid, starts, goals, actions, max_steps=2)
        self.assertEqual(r.trigger_type, "NONE")
        self.assertIsNone(r.first_seen)
        self.assertIsNone(r.trigger_timestep)
        self.assertEqual(r.blocking_type, "NONE")


class TestKnownQuirkKI1Separation(unittest.TestCase):
    """KI-1 (overlap) instances: kept in their own class, never pooled with
    TestRealCrossCheck's detector-accuracy assertions
    (docs/deadlock_definition_v1.md Section 8.1). This only checks that (a)
    the Monitor's online overlap tracking fires, and (b) it does not
    suppress or distort trigger_type/blocking_type computation."""

    def test_bottleneck_n4_convoy_overlap_does_not_suppress_trigger(self):
        # outputs/logs/20261010_212255_bottleneck_n4_convoy_IL_steps.csv
        # (Category B, overlap_timesteps=160 at 160 steps -- the documented
        # KI-1 single-pass vertex-resolution quirk, not a new bug.)
        grid, starts, goals = load("scenarios/controlled_variants/scenario_bottleneck_n4_convoy.json")
        actions = {0: [DOWN] * 4, 1: [DOWN] * 4, 2: [UP] * 4, 3: [UP] * 4}
        r = replay(grid, starts, goals, actions, max_steps=4)
        self.assertEqual(r.overlap_integrity_status, "INTEGRITY_ISSUE")
        self.assertGreater(r.overlap_timesteps, 0)
        # Trigger/blocking computation must still proceed normally.
        self.assertEqual(r.trigger_type, "FULL")
        self.assertEqual(r.first_seen, 1)
        self.assertEqual(r.trigger_timestep, 2)
        self.assertEqual(r.blocking_type, "VERTEX")


class TestMonitorLogicSynthetic(unittest.TestCase):
    """SYNTHETIC: hand-fed positions/conflict_types, not claimed as real IL
    behaviour. Used only where no real example exists yet
    (docs/deadlock_definition_v1.md Section 6/8.2)."""

    def test_period_1_fixed_point(self):
        m = DeadlockMonitor()
        m.reset({0: (0, 0), 1: (0, 5)}, goals={0: (9, 9), 1: (9, 8)})
        m.observe_step(1, {0: (0, 0), 1: (0, 5)}, {0: "vertex", 1: "vertex"})
        r = m.observe_step(2, {0: (0, 0), 1: (0, 5)}, {0: "vertex", 1: "vertex"})
        self.assertEqual(r.trigger_type, "FULL")
        self.assertEqual(r.first_seen, 0)  # S_0 == S_1 == S_2
        self.assertEqual(r.trigger_timestep, 1)
        self.assertEqual(r.period, 1)
        self.assertEqual(r.blocking_type, "VERTEX")

    def test_period_greater_than_1_genuine_cycle(self):
        """No real period>1 example exists in this repository yet (an open
        evidence gap, docs/deadlock_definition_v1.md Section 6). This is a
        pure logic test of the Monitor's exact-match repeat detection over
        a hand-fed 2-state A/B/A/B cycle -- it does not claim this pattern
        has been observed from a real IL rollout."""
        m = DeadlockMonitor()
        state_a = {0: (0, 0), 1: (0, 9)}
        state_b = {0: (0, 1), 1: (0, 8)}
        m.reset(state_a, goals={0: (9, 9), 1: (9, 0)})
        m.observe_step(1, state_b, {0: "edge", 1: "edge"})
        m.observe_step(2, state_a, {0: "edge", 1: "edge"})
        m.observe_step(3, state_b, {0: "edge", 1: "edge"})
        r = m.observe_step(4, state_a, {0: "edge", 1: "edge"})
        self.assertEqual(r.trigger_type, "FULL")
        self.assertEqual(r.first_seen, 0)
        self.assertEqual(r.trigger_timestep, 2)
        self.assertEqual(r.period, 2)
        self.assertEqual(r.blocking_type, "EDGE")

    def test_repeat_returning_exactly_to_s0(self):
        """SYNTHETIC: positions trace a genuine loop (4 distinct cells, no
        state revisited early) and return exactly to S_0 several steps
        later (period == the number of steps taken, not 1) -- exercises the
        "any period, not just adjacent-step" path."""
        m = DeadlockMonitor()
        s0 = {0: (2, 2)}
        m.reset(s0, goals={0: (9, 9)})
        m.observe_step(1, {0: (2, 3)}, {0: "none"})
        m.observe_step(2, {0: (3, 3)}, {0: "none"})
        m.observe_step(3, {0: (3, 2)}, {0: "none"})
        r = m.observe_step(4, {0: (2, 2)}, {0: "vertex"})
        self.assertEqual(r.trigger_type, "FULL")
        self.assertEqual(r.first_seen, 0)
        self.assertEqual(r.trigger_timestep, 4)
        self.assertEqual(r.period, 4)

    def test_all_agents_at_goal_is_never_a_trigger(self):
        m = DeadlockMonitor()
        m.reset({0: (0, 0)}, goals={0: (0, 1)})
        m.observe_step(1, {0: (0, 1)}, {0: "none"})
        r = m.observe_step(2, {0: (0, 1)}, {0: "none"})  # WAIT at goal, repeats forever
        self.assertEqual(r.trigger_type, "NONE")
        self.assertIsNone(r.first_seen)

    def test_full_repeat_without_blocking_evidence_is_left_as_none(self):
        """SYNTHETIC: a full repeat occurs but every agent's conflict_type
        in the closing cycle is 'none' (e.g. deliberate WAIT, not a bounce-
        back) -- docs/deadlock_definition_v1.md Section 4.4's
        REPEAT_WITHOUT_BLOCKING_EVIDENCE case. The Monitor must not invent
        VERTEX/EDGE evidence that was not observed."""
        m = DeadlockMonitor()
        m.reset({0: (0, 0), 1: (5, 5)}, goals={0: (9, 9), 1: (9, 8)})
        m.observe_step(1, {0: (0, 0), 1: (5, 5)}, {0: "none", 1: "none"})
        r = m.observe_step(2, {0: (0, 0), 1: (5, 5)}, {0: "none", 1: "none"})
        self.assertEqual(r.trigger_type, "FULL")
        self.assertEqual(r.blocking_type, "NONE")
        self.assertEqual(r.blocking_agents, ())

    def test_vertex_and_edge_together_are_mixed(self):
        """SYNTHETIC: two independent contested pairs in a >=4-agent team,
        one vertex-blocked and one edge-blocked, in the same closing cycle.
        No real example exists yet (Section 6/8.2) -- pure logic coverage
        for the MIXED branch."""
        m = DeadlockMonitor()
        state = {0: (0, 0), 1: (0, 1), 2: (5, 5), 3: (5, 6)}
        m.reset(state, goals={0: (9, 9), 1: (9, 8), 2: (9, 7), 3: (9, 6)})
        conflicts = {0: "vertex", 1: "vertex", 2: "edge", 3: "edge"}
        m.observe_step(1, state, conflicts)
        r = m.observe_step(2, state, conflicts)
        self.assertEqual(r.trigger_type, "FULL")
        self.assertEqual(r.blocking_type, "MIXED")
        self.assertEqual(r.blocking_agents, (0, 1, 2, 3))

    def test_agent_id_order_independent(self):
        """Same positions, keys inserted in a different order -> identical
        joint state / identical result (docs/deadlock_definition_v1.md
        Section 3.3: fixed, sorted-by-agent_id order)."""
        m1 = DeadlockMonitor()
        m1.reset({0: (1, 1), 1: (2, 2), 2: (3, 3)}, goals={0: (9, 9), 1: (8, 8), 2: (7, 7)})
        m1.observe_step(1, {0: (1, 1), 1: (2, 2), 2: (3, 3)}, {0: "vertex", 1: "none", 2: "none"})
        r1 = m1.observe_step(2, {0: (1, 1), 1: (2, 2), 2: (3, 3)}, {0: "vertex", 1: "none", 2: "none"})

        m2 = DeadlockMonitor()
        m2.reset({2: (3, 3), 0: (1, 1), 1: (2, 2)}, goals={2: (7, 7), 0: (9, 9), 1: (8, 8)})
        m2.observe_step(1, {2: (3, 3), 1: (2, 2), 0: (1, 1)}, {2: "none", 0: "vertex", 1: "none"})
        r2 = m2.observe_step(2, {1: (2, 2), 2: (3, 3), 0: (1, 1)}, {1: "none", 2: "none", 0: "vertex"})

        self.assertEqual((r1.trigger_type, r1.first_seen, r1.trigger_timestep, r1.period, r1.blocking_type),
                          (r2.trigger_type, r2.first_seen, r2.trigger_timestep, r2.period, r2.blocking_type))
        self.assertEqual(r1.trigger_type, "FULL")


class TestBaselineUnaffectedByMonitor(unittest.TestCase):
    """monitor=None (default) must remain byte-for-byte unaffected, and
    attaching a monitor must not change the simulated trajectory at all --
    only observe it. Uses a tiny local stub policy (no torch/checkpoint
    needed) purely to exercise scripts.run_il.run_episode's control flow."""

    class _StubPolicy:
        """Deterministic, stateless: always RIGHT until off the grid edge,
        otherwise WAIT. Good enough to produce a non-trivial, fully
        deterministic trajectory without needing NavHintPolicy/torch."""

        def reset(self, grid, goals):
            self._grid = grid

        def act(self, obs, positions):
            actions = {}
            h, w = self._grid.shape
            for aid, (row, col) in positions.items():
                actions[aid] = RIGHT if col + 1 < w and self._grid[row, col + 1] == 0 else WAIT
            return actions

    def test_monitor_none_vs_attached_same_trajectory(self):
        import scripts.run_il as run_il

        grid, starts, goals = load("scenarios/controlled/scenario_corridor_bay_n2.json")

        result_no_monitor = run_il.run_episode(
            self._StubPolicy(), grid, starts, goals, max_steps=8, logger=None, monitor=None)

        from src.common.deadlock_monitor import DeadlockMonitor
        monitor = DeadlockMonitor()
        result_with_monitor = run_il.run_episode(
            self._StubPolicy(), grid, starts, goals, max_steps=8, logger=None, monitor=monitor)

        self.assertEqual(result_no_monitor["steps"], result_with_monitor["steps"])
        self.assertEqual(result_no_monitor["trajectory"], result_with_monitor["trajectory"])
        self.assertEqual(result_no_monitor["actions"], result_with_monitor["actions"])
        self.assertEqual(result_no_monitor["all_at_goal"], result_with_monitor["all_at_goal"])
        self.assertEqual(result_no_monitor["timed_out"], result_with_monitor["timed_out"])
        # The monitor itself still did its job (observation-only, non-trivial).
        self.assertIsNotNone(monitor.result)


if __name__ == "__main__":
    unittest.main()
