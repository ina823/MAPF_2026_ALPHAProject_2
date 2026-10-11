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

from src.common import spec
from src.common.deadlock_monitor import DeadlockMonitor
from src.il.nav_hint import bfs_dist
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


def replay_with_persistent_conflict(grid, starts, goals, actions_per_agent, max_steps, *, logger=None):
    """Phase 3B-1 replay helper. Like `replay()`, but:
      (a) ALWAYS runs the full `max_steps` instead of stopping at the first
          Full-repeat trigger -- needed to observe persistent_conflict
          streaks continuing (or stopping) over *multiple* steps, which a
          single-step-past-trigger replay cannot show;
      (b) feeds `current_positions`/`intended_positions` into
          `observe_step()` every step, computing `intended` the exact same
          raw way `scripts/run_il.py::_log_step` does (current + the
          chosen action's `spec.ACTION_DELTA`, not wall-clamped) so the
          reconstructed conflict keys line up with what
          `scripts/analyze_failure_logs.py` would read from a real CSV's
          `intended_row`/`intended_col` columns;
      (c) optionally drives a real `EpisodeLogger` with the same rows, so
          a test can cross-check the online streak against the offline
          analyzer on the exact same trajectory (point 8 of the 3B-1
          approval).
    Returns the list of per-step `DeadlockMonitorResult` snapshots (one per
    `observe_step()` call, in order)."""
    sim = MAPFStepSimulator(cbs_solver_root=None, max_steps=max_steps)
    sim.reset(grid, dict(starts), dict(goals))
    monitor = DeadlockMonitor()
    monitor.reset(sim._positions, goals)
    per_step = []
    for t in range(max_steps):
        current = dict(sim._positions)
        actions = {aid: (seq[t] if t < len(seq) else WAIT) for aid, seq in actions_per_agent.items()}
        intended = {}
        for aid, action_id in actions.items():
            drow, dcol = spec.ACTION_DELTA[action_id]
            intended[aid] = (current[aid][0] + drow, current[aid][1] + dcol)
        obs, done, info = sim.step(actions)
        if logger is not None:
            for aid in sorted(current):
                logger.log_step(
                    timestep=info["t"], agent_id=aid, current=current[aid],
                    action_id=actions[aid], intended=intended[aid],
                    next_pos=sim._positions[aid],
                    goal_distance_before=0, goal_distance_after=0,
                    conflict_type=sim._last_conflict_types.get(aid, "none"),
                    done=done, success=info["all_at_goal"],
                )
        r = monitor.observe_step(
            info["t"], sim._positions, sim._last_conflict_types,
            current_positions=current, intended_positions=intended)
        per_step.append(r)
        if done:
            break
    return per_step


def replay_with_no_progress(grid, starts, goals, actions_per_agent, max_steps):
    """Phase 3B-2A replay helper. Like `replay()`, but runs the full
    `max_steps` (not stopping at the first Full-repeat trigger) and feeds
    `goal_distance_before`/`goal_distance_after` into `observe_step()`
    every step, computed with the real `bfs_dist()` function
    (`src.il.nav_hint.bfs_dist`) -- the exact same function
    `src.il.policy.NavHintPolicy.reset()` uses to build `policy._dist`,
    looked up at the real pre-/post-step positions exactly the way
    `scripts/run_il.py::_log_step` does. This needs no torch/checkpoint
    (pure BFS over the real scenario grid), so it stays REAL data, not
    SYNTHETIC, even without loading the actual NavHintPolicy.
    Returns the list of per-step `DeadlockMonitorResult` snapshots."""
    dist_fields = {aid: bfs_dist(grid, goals[aid]) for aid in goals}
    sim = MAPFStepSimulator(cbs_solver_root=None, max_steps=max_steps)
    sim.reset(grid, dict(starts), dict(goals))
    monitor = DeadlockMonitor()
    monitor.reset(sim._positions, goals)
    per_step = []
    for t in range(max_steps):
        current = dict(sim._positions)
        before = {aid: int(dist_fields[aid][current[aid][0], current[aid][1]]) for aid in current}
        actions = {aid: (seq[t] if t < len(seq) else WAIT) for aid, seq in actions_per_agent.items()}
        obs, done, info = sim.step(actions)
        after = {aid: int(dist_fields[aid][pos[0], pos[1]]) for aid, pos in sim._positions.items()}
        r = monitor.observe_step(
            info["t"], sim._positions, sim._last_conflict_types,
            goal_distance_before=before, goal_distance_after=after)
        per_step.append(r)
        if done:
            break
    return per_step


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
            # Phase 3B-2B: run_il.py::run_episode now looks up
            # policy._dist[aid] whenever monitor is not None (to feed the
            # no_progress signal), the same way the real NavHintPolicy
            # does in its own reset() -- this stub needs the same
            # attribute to exercise that code path.
            self._dist = {aid: bfs_dist(grid, goal) for aid, goal in goals.items()}

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


class TestPersistentConflictCandidateReal(unittest.TestCase):
    """Phase 3B-1: persistent_conflict candidate signal, cross-checked
    against REAL scenario files + the real, unmodified MAPFStepSimulator
    (same REAL-data discipline as TestRealCrossCheck above). Each test
    documents, in its own name/docstring, exactly which point of the 3B-1
    approval it targets."""

    def test_corridor_bay_n2_persistent_conflict_streak_grows_while_blocked(self):
        """Approval point 6 (Corridor) + point 1 (Full-repeat trigger
        preserved unchanged). With a constant-action replay (no fallback
        to WAIT), the same vertex key must recur identically every step
        from the first collision onward -- this is not a guess, it follows
        directly from the already-confirmed Full-repeat + period=1 fixed
        point (Section 4.2): once S_5 = S_4 and the action choice never
        changes, F(S_4) must produce the identical bounce-back every time.
        """
        grid, starts, goals = load("scenarios/controlled/scenario_corridor_bay_n2.json")
        actions = {0: [RIGHT] * 8, 1: [LEFT] * 8}
        per_step = replay_with_persistent_conflict(grid, starts, goals, actions, max_steps=8)
        streaks = [r.current_persistent_conflict_streak for r in per_step]
        # t=1..4: approaching, no conflict yet. t=5..8: identical vertex
        # bounce-back every step -> streak climbs 1,2,3,4.
        self.assertEqual(streaks, [0, 0, 0, 0, 1, 2, 3, 4])
        last = per_step[-1]
        self.assertEqual(last.current_persistent_conflict_key[0], "vertex")
        self.assertEqual(set(last.current_persistent_conflict_key[2]), {0, 1})
        self.assertEqual(last.persistent_conflict_status, "OK")
        self.assertEqual(last.persistent_conflict_incomplete_steps, 0)
        # Phase 3A's own trigger/blocking fields must be byte-for-byte what
        # they were before 3B-1 (approval point 1).
        full_trigger_step = per_step[4]  # t=5
        self.assertEqual(full_trigger_step.trigger_type, "FULL")
        self.assertEqual(full_trigger_step.first_seen, 4)
        self.assertEqual(full_trigger_step.trigger_timestep, 5)
        self.assertEqual(full_trigger_step.period, 1)
        self.assertEqual(full_trigger_step.blocking_type, "VERTEX")

    def test_corridor_bay_n2_streak_resets_when_conflict_pattern_stops(self):
        """Approval point 2: a CURRENT streak, not a cumulative total -- it
        must drop back to 0 the moment the same key stops recurring, even
        though the earlier occurrences are still true facts about the past.
        Uses the exact 6-action list the original Phase 3A test already
        established reaches the trigger at t=5 (`test_corridor_bay_n2_original_vertex`
        above); steps beyond index 5 fall back to WAIT (no attempted move,
        so conflict_type becomes "none" and the vertex key disappears)."""
        grid, starts, goals = load("scenarios/controlled/scenario_corridor_bay_n2.json")
        actions = {0: [RIGHT] * 6, 1: [LEFT] * 6}
        per_step = replay_with_persistent_conflict(grid, starts, goals, actions, max_steps=8)
        streaks = [r.current_persistent_conflict_streak for r in per_step]
        self.assertEqual(streaks, [0, 0, 0, 0, 1, 2, 0, 0])
        self.assertIsNone(per_step[-1].current_persistent_conflict_key)

    def test_intersection_n2_persistent_conflict_vertex(self):
        """Approval point 6 (Intersection)."""
        grid, starts, goals = load("scenarios/controlled/scenario_intersection_n2.json")
        actions = {0: [RIGHT] * 6, 1: [DOWN] * 6}
        per_step = replay_with_persistent_conflict(grid, starts, goals, actions, max_steps=6)
        # original Phase 3A evidence: first_seen=1, trigger_timestep=2, period=1
        self.assertEqual(per_step[1].trigger_type, "FULL")
        self.assertGreaterEqual(per_step[1].current_persistent_conflict_streak, 1)
        self.assertEqual(per_step[1].current_persistent_conflict_key[0], "vertex")
        # streak keeps climbing every further step while the same action choice
        # (and therefore the same bounce-back) continues.
        self.assertEqual(
            per_step[-1].current_persistent_conflict_streak,
            per_step[1].current_persistent_conflict_streak + (len(per_step) - 1 - 1))

    def test_bottleneck_n4_persistent_conflict_vertex(self):
        """Approval point 6 (Bottleneck)."""
        grid, starts, goals = load("scenarios/controlled/scenario_bottleneck_n4.json")
        actions = {
            0: [RIGHT, DOWN, DOWN, DOWN, DOWN, DOWN],
            1: [UP, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT],
            2: [DOWN, DOWN, DOWN, DOWN, DOWN, DOWN],
            3: [UP, UP, UP, UP, UP, UP],
        }
        per_step = replay_with_persistent_conflict(grid, starts, goals, actions, max_steps=6)
        # original Phase 3A evidence: first_seen=1, trigger_timestep=2, period=1
        self.assertEqual(per_step[1].trigger_type, "FULL")
        self.assertEqual(per_step[1].blocking_type, "VERTEX")
        self.assertGreaterEqual(per_step[1].current_persistent_conflict_streak, 1)
        self.assertGreater(per_step[-1].current_persistent_conflict_streak,
                            per_step[1].current_persistent_conflict_streak)

    def test_corridor_3rdagent_perturbation_persistent_conflict_edge(self):
        """Approval point 6 (edge conflict case). The third agent reaches
        its goal and WAITs from t=4 onward (per
        docs/deadlock_definition_v1.md Section 4.3's worked correction),
        so it never contributes a conflict key; only the edge-swapping pair
        should drive the streak."""
        grid, starts, goals = load(
            "scenarios/experimental_v1/scenario_corridor_bay_n2_3rdagent_perturbation.json")
        actions = {
            0: [RIGHT] * 8,
            1: [LEFT] * 8,
            2: [LEFT, LEFT, LEFT, LEFT, WAIT, WAIT, WAIT, WAIT],
        }
        per_step = replay_with_persistent_conflict(grid, starts, goals, actions, max_steps=8)
        streaks = [r.current_persistent_conflict_streak for r in per_step]
        # Unlike the vertex cases above, the edge swap attempt itself IS the
        # first-seen state here (agents 0/1 are already adjacent and trying
        # to swap at t=4), so persistent_conflict picks it up one step
        # BEFORE the Full-repeat trigger confirms it at t=5 -- a concrete
        # illustration of why this signal is a useful *earlier* candidate,
        # not a redundant restatement of trigger_type.
        self.assertEqual(streaks, [0, 0, 0, 1, 2, 3, 4, 5])
        last_key = per_step[-1].current_persistent_conflict_key
        self.assertEqual(last_key[0], "edge")
        self.assertEqual(set(last_key[2]), {0, 1})
        self.assertEqual(per_step[-1].blocking_type, "EDGE")

    def test_success_cycle_ring_n4_never_produces_a_false_candidate(self):
        """Approval point 6: a clean success run must never show a
        persistent_conflict streak -- no conflict ever occurs, so the
        signal must stay 0 throughout, even though the feature is engaged
        (status="OK") on every step."""
        grid, starts, goals = load("scenarios/controlled/scenario_cycle_ring_n4.json")
        actions = {0: [RIGHT, RIGHT], 1: [DOWN, DOWN], 2: [LEFT, LEFT], 3: [UP, UP]}
        per_step = replay_with_persistent_conflict(grid, starts, goals, actions, max_steps=2)
        for r in per_step:
            self.assertEqual(r.current_persistent_conflict_streak, 0)
            self.assertIsNone(r.current_persistent_conflict_key)
            self.assertEqual(r.persistent_conflict_status, "OK")
            self.assertEqual(r.persistent_conflict_incomplete_steps, 0)
        self.assertEqual(per_step[-1].trigger_type, "NONE")  # Phase 3A unaffected


class TestPersistentConflictKI1Separation(unittest.TestCase):
    """Approval point 6: KI-1 (overlap) instances get their own class, kept
    separate from TestPersistentConflictCandidateReal's accuracy-style
    assertions, same discipline as the existing
    TestKnownQuirkKI1Separation class above (docs/deadlock_definition_v1.md
    Section 8.1)."""

    def test_bottleneck_n4_convoy_overlap_does_not_suppress_persistent_conflict(self):
        grid, starts, goals = load("scenarios/controlled_variants/scenario_bottleneck_n4_convoy.json")
        actions = {0: [DOWN] * 4, 1: [DOWN] * 4, 2: [UP] * 4, 3: [UP] * 4}
        per_step = replay_with_persistent_conflict(grid, starts, goals, actions, max_steps=4)
        last = per_step[-1]
        self.assertEqual(last.overlap_integrity_status, "INTEGRITY_ISSUE")
        self.assertGreater(last.overlap_timesteps, 0)
        # persistent_conflict tracking must still run normally alongside the
        # KI-1 overlap, exactly like trigger_type/blocking_type already do.
        self.assertEqual(last.trigger_type, "FULL")
        self.assertEqual(last.persistent_conflict_status, "OK")
        self.assertEqual(last.persistent_conflict_incomplete_steps, 0)


class TestPersistentConflictOnlineVsOfflineTiming(unittest.TestCase):
    """Approval point 8: directly compares the online, mid-episode
    current_persistent_conflict_streak against
    scripts/analyze_failure_logs.py's offline, whole-episode
    repeated_conflict_candidate_max_consecutive on the SAME real trajectory
    -- demonstrating they are the same underlying concept observed at two
    different, non-interchangeable times (Section 7.1(c))."""

    def test_online_streak_at_first_occurrence_is_less_than_offline_final_max(self):
        import tempfile

        import scripts.analyze_failure_logs as analyze_failure_logs
        from src.common.episode_logger import EpisodeLogger

        grid, starts, goals = load("scenarios/controlled/scenario_corridor_bay_n2.json")
        actions = {0: [RIGHT] * 8, 1: [LEFT] * 8}

        with tempfile.TemporaryDirectory() as tmp:
            logger = EpisodeLogger("persistent_conflict_3b1_test", scenario="test", mode="IL", out_dir=tmp)
            per_step = replay_with_persistent_conflict(
                grid, starts, goals, actions, max_steps=8, logger=logger)
            steps_path = logger.write_steps()
            rows = analyze_failure_logs.read_steps(steps_path)
            _, episode = analyze_failure_logs.analyze_episode(rows)

        online_streaks = [r.current_persistent_conflict_streak for r in per_step]
        offline_max_consecutive = episode["repeated_conflict_candidate_max_consecutive"]

        # The online streak, read at the instant it first becomes nonzero
        # (t=5, the earliest the step could possibly be observed), is
        # strictly less than the offline figure, which is only computable
        # once the whole 8-step episode has been read (Section 7.1(c)).
        first_nonzero_idx = next(i for i, s in enumerate(online_streaks) if s > 0)
        self.assertEqual(first_nonzero_idx, 4)  # t=5
        self.assertEqual(online_streaks[first_nonzero_idx], 1)
        self.assertLess(online_streaks[first_nonzero_idx], offline_max_consecutive)
        # By the LAST step of the same (now-finished) episode, the online
        # value and the offline value agree exactly -- same key, same
        # steps, two different computations of "how long has this recurred".
        self.assertEqual(online_streaks[-1], offline_max_consecutive)


class TestPersistentConflictSynthetic(unittest.TestCase):
    """SYNTHETIC (see the module-level docstring's REAL/SYNTHETIC
    distinction): hand-fed inputs used only to exercise Monitor logic paths
    that have no real-CSV example (approval point 4's "insufficient info"
    handling, and the default opt-out path)."""

    def test_not_computed_by_default_when_inputs_never_supplied(self):
        """Approval point 3: calling observe_step() with only the original
        3 positional arguments (as every pre-3B-1 caller does) must leave
        the persistent_conflict fields at their inert defaults."""
        m = DeadlockMonitor()
        m.reset({0: (0, 0), 1: (0, 5)}, goals={0: (9, 9), 1: (9, 8)})
        r = m.observe_step(1, {0: (1, 0), 1: (0, 6)}, {0: "none", 1: "none"})
        self.assertEqual(r.persistent_conflict_status, "NOT_COMPUTED")
        self.assertEqual(r.current_persistent_conflict_streak, 0)
        self.assertIsNone(r.current_persistent_conflict_key)
        self.assertEqual(r.persistent_conflict_incomplete_steps, 0)
        # Phase 3A fields are completely unaffected by the new kwargs' absence.
        self.assertEqual(r.trigger_type, "NONE")

    def test_incomplete_data_does_not_guess_a_key(self):
        """Approval point 4: a conflict-flagged agent missing from
        `intended_positions` must not produce a fabricated key -- the step
        is recorded as incomplete and any in-progress streak ends."""
        m = DeadlockMonitor()
        m.reset({0: (0, 0), 1: (0, 5)}, goals={0: (9, 9), 1: (9, 8)})
        # Step 1: both agents' data present, genuine vertex key -> streak=1.
        r1 = m.observe_step(
            1, {0: (1, 1), 1: (0, 5)}, {0: "vertex", 1: "vertex"},
            current_positions={0: (0, 0), 1: (0, 5)},
            intended_positions={0: (1, 1), 1: (1, 1)})
        self.assertEqual(r1.persistent_conflict_status, "OK")
        self.assertEqual(r1.current_persistent_conflict_streak, 1)
        self.assertEqual(r1.persistent_conflict_incomplete_steps, 0)
        # Step 2: agent 1 is flagged "vertex" again but missing from
        # intended_positions -- must NOT be treated as "no conflict"
        # (which would wrongly look identical to a resolved episode) nor
        # guessed as a continuing streak; it must be flagged incomplete.
        r2 = m.observe_step(
            2, {0: (1, 1), 1: (0, 5)}, {0: "vertex", 1: "vertex"},
            current_positions={0: (1, 1), 1: (0, 5)},
            intended_positions={0: (2, 1)})
        self.assertEqual(r2.persistent_conflict_status, "OK")
        self.assertEqual(r2.persistent_conflict_incomplete_steps, 1)
        self.assertEqual(r2.current_persistent_conflict_streak, 0)
        self.assertIsNone(r2.current_persistent_conflict_key)

    def test_multiple_concurrent_keys_only_longest_streak_is_surfaced(self):
        """Documents the known 3B-1 limitation (see
        DeadlockMonitorResult.current_persistent_conflict_streak's
        docstring): when more than one distinct conflict key is live in
        the same step, the public result exposes only the single
        longest-streak key. Internally BOTH keys are still tracked (this
        is a reporting simplification, not a detection gap) -- and,
        crucially, this signal never feeds into trigger_type/blocking_type.
        In this trace, the (0,1) vertex pair's bounce-back already makes
        S_1 == S_0 (Full-repeat fires at step 1, blocking_type="VERTEX",
        settled from that one step's evidence); the (2,3) edge conflict
        only starts at step 2, *after* the trigger already settled.
        persistent_conflict still tracks and reports it (proving it is not
        gated by an already-fired trigger, same as overlap tracking), but
        -- just as importantly -- blocking_type stays "VERTEX" and does
        NOT retroactively become "MIXED": the confirmed classification is
        frozen at the moment of the first repeat and is never revised by
        later persistent_conflict evidence.
        """
        m = DeadlockMonitor()
        m.reset({0: (0, 0), 1: (0, 1), 2: (5, 5), 3: (5, 6)},
                 goals={0: (9, 9), 1: (9, 8), 2: (9, 7), 3: (9, 6)})
        # Step 1: the (0,1) vertex pair bounces back to exactly S_0 ->
        # Full-repeat fires immediately (first_seen=0, trigger_timestep=1).
        r1 = m.observe_step(
            1, {0: (0, 0), 1: (0, 1), 2: (5, 5), 3: (5, 6)},
            {0: "vertex", 1: "vertex", 2: "none", 3: "none"},
            current_positions={0: (0, 0), 1: (0, 1), 2: (5, 5), 3: (5, 6)},
            intended_positions={0: (0, 2), 1: (0, 2), 2: (5, 5), 3: (5, 6)})
        self.assertEqual(r1.trigger_type, "FULL")
        self.assertEqual(r1.first_seen, 0)
        self.assertEqual(r1.trigger_timestep, 1)
        self.assertEqual(r1.blocking_type, "VERTEX")
        # Step 2: the (0,1) vertex key recurs (streak -> 2); a second,
        # independent (2,3) edge key appears for the first time (streak=1)
        # -- after the trigger has already settled.
        r2 = m.observe_step(
            2, {0: (0, 0), 1: (0, 1), 2: (5, 5), 3: (5, 6)},
            {0: "vertex", 1: "vertex", 2: "edge", 3: "edge"},
            current_positions={0: (0, 0), 1: (0, 1), 2: (5, 5), 3: (5, 6)},
            intended_positions={0: (0, 2), 1: (0, 2), 2: (5, 6), 3: (5, 5)})

        self.assertEqual(r2.current_persistent_conflict_streak, 2)
        self.assertEqual(r2.current_persistent_conflict_key,
                          ("vertex", (0, 2), frozenset({0, 1})))
        # Whitebox: the shorter (2,3) edge streak is still tracked
        # internally even though it is not the one surfaced above.
        self.assertEqual(len(m._persistent_conflict_streaks), 2)
        self.assertEqual(
            m._persistent_conflict_streaks[("edge", frozenset({(5, 5), (5, 6)}), frozenset({2, 3}))], 1)
        # The already-settled Trigger/Label fields are untouched by any of
        # this -- never revised, never merged with persistent_conflict data.
        self.assertEqual(r2.trigger_type, "FULL")
        self.assertEqual(r2.first_seen, 0)
        self.assertEqual(r2.trigger_timestep, 1)
        self.assertEqual(r2.blocking_type, "VERTEX")

    def test_only_one_of_the_pair_supplied_is_also_incomplete(self):
        """Approval point 4: current_positions and intended_positions must
        be supplied together; supplying only one is insufficient info, not
        a silent no-op and not a guess."""
        m = DeadlockMonitor()
        m.reset({0: (0, 0)}, goals={0: (9, 9)})
        r = m.observe_step(
            1, {0: (0, 1)}, {0: "vertex"},
            current_positions={0: (0, 0)})  # intended_positions omitted
        self.assertEqual(r.persistent_conflict_status, "OK")
        self.assertEqual(r.persistent_conflict_incomplete_steps, 1)
        self.assertEqual(r.current_persistent_conflict_streak, 0)


class TestNoProgressCandidateReal(unittest.TestCase):
    """Phase 3B-2A: no_progress candidate signal, cross-checked against
    REAL scenario files + the real, unmodified MAPFStepSimulator + the
    real `bfs_dist()` function (same REAL-data discipline as
    TestRealCrossCheck/TestPersistentConflictCandidateReal above)."""

    def test_corridor_bay_n2_no_progress_streak_grows_for_both_agents(self):
        """Approval point 3 (정상 전진 vs 정체) + point 2 (naming/semantics
        separated from offline). Before the freeze (t=1..4) both agents
        make genuine BFS-distance progress every step -> streak stays
        empty. From t=5 (the confirmed vertex bounce-back) onward, both
        agents' positions -- and therefore their BFS distance -- stop
        changing every step -> both streaks climb together."""
        grid, starts, goals = load("scenarios/controlled/scenario_corridor_bay_n2.json")
        actions = {0: [RIGHT] * 8, 1: [LEFT] * 8}
        per_step = replay_with_no_progress(grid, starts, goals, actions, max_steps=8)
        streaks = [r.current_no_progress_streaks for r in per_step]
        self.assertEqual(streaks, [
            (), (), (), (),
            ((0, 1), (1, 1)),
            ((0, 2), (1, 2)),
            ((0, 3), (1, 3)),
            ((0, 4), (1, 4)),
        ])
        for r in per_step:
            self.assertEqual(r.no_progress_status, "OK")
            self.assertEqual(r.no_progress_incomplete_steps, 0)
        # Phase 3A/3B-1 fields are completely unaffected (approval point 5).
        self.assertEqual(per_step[4].trigger_type, "FULL")
        self.assertEqual(per_step[4].first_seen, 4)
        self.assertEqual(per_step[4].blocking_type, "VERTEX")

    def test_intersection_n2_no_progress_streak(self):
        """Approval point 8 (Intersection)."""
        grid, starts, goals = load("scenarios/controlled/scenario_intersection_n2.json")
        actions = {0: [RIGHT] * 6, 1: [DOWN] * 6}
        per_step = replay_with_no_progress(grid, starts, goals, actions, max_steps=6)
        streaks = [r.current_no_progress_streaks for r in per_step]
        self.assertEqual(streaks, [
            (),
            ((0, 1), (1, 1)),
            ((0, 2), (1, 2)),
            ((0, 3), (1, 3)),
            ((0, 4), (1, 4)),
            ((0, 5), (1, 5)),
        ])

    def test_bottleneck_n4_no_progress_streak_all_four_agents(self):
        """Approval point 8 (Bottleneck) -- all 4 agents stall together."""
        grid, starts, goals = load("scenarios/controlled/scenario_bottleneck_n4.json")
        actions = {
            0: [RIGHT, DOWN, DOWN, DOWN, DOWN, DOWN],
            1: [UP, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT],
            2: [DOWN, DOWN, DOWN, DOWN, DOWN, DOWN],
            3: [UP, UP, UP, UP, UP, UP],
        }
        per_step = replay_with_no_progress(grid, starts, goals, actions, max_steps=6)
        streaks = [r.current_no_progress_streaks for r in per_step]
        self.assertEqual(streaks, [
            (),
            ((0, 1), (1, 1), (2, 1), (3, 1)),
            ((0, 2), (1, 2), (2, 2), (3, 2)),
            ((0, 3), (1, 3), (2, 3), (3, 3)),
            ((0, 4), (1, 4), (2, 4), (3, 4)),
            ((0, 5), (1, 5), (2, 5), (3, 5)),
        ])

    def test_corridor_3rdagent_perturbation_excludes_agent_already_at_goal(self):
        """Approval point 3 (목표 도달 agent 처리) + point 8 (edge 사례).
        Agent 2 reaches its own goal at t=4 and WAITs from then on; it must
        NEVER appear in current_no_progress_streaks (goal_distance_before
        == 0 from that point on -- not "active"), even though it is
        stationary exactly like the two stalled agents. Only agents 0/1
        (still active, edge-blocked from t=4) accumulate a streak."""
        grid, starts, goals = load(
            "scenarios/experimental_v1/scenario_corridor_bay_n2_3rdagent_perturbation.json")
        actions = {
            0: [RIGHT] * 8,
            1: [LEFT] * 8,
            2: [LEFT, LEFT, LEFT, LEFT, WAIT, WAIT, WAIT, WAIT],
        }
        per_step = replay_with_no_progress(grid, starts, goals, actions, max_steps=8)
        streaks = [r.current_no_progress_streaks for r in per_step]
        self.assertEqual(streaks, [
            (), (), (),
            ((0, 1), (1, 1)),
            ((0, 2), (1, 2)),
            ((0, 3), (1, 3)),
            ((0, 4), (1, 4)),
            ((0, 5), (1, 5)),
        ])
        for s in streaks:
            self.assertNotIn(2, dict(s))  # agent 2 never appears

    def test_success_cycle_ring_n4_never_produces_a_false_candidate(self):
        """Approval point 8: a clean success run (every step is genuine
        BFS-distance progress for every active agent) must never show a
        no_progress streak."""
        grid, starts, goals = load("scenarios/controlled/scenario_cycle_ring_n4.json")
        actions = {0: [RIGHT, RIGHT], 1: [DOWN, DOWN], 2: [LEFT, LEFT], 3: [UP, UP]}
        per_step = replay_with_no_progress(grid, starts, goals, actions, max_steps=2)
        for r in per_step:
            self.assertEqual(r.current_no_progress_streaks, ())
            self.assertEqual(r.no_progress_status, "OK")
            self.assertEqual(r.no_progress_incomplete_steps, 0)
        self.assertEqual(per_step[-1].trigger_type, "NONE")  # Phase 3A unaffected


class TestNoProgressKI1Separation(unittest.TestCase):
    """Approval point 8: KI-1 (overlap) instances get their own class, kept
    separate from TestNoProgressCandidateReal's accuracy-style assertions,
    same discipline as TestKnownQuirkKI1Separation /
    TestPersistentConflictKI1Separation above
    (docs/deadlock_definition_v1.md Section 8.1)."""

    def test_bottleneck_n4_convoy_overlap_does_not_suppress_no_progress(self):
        grid, starts, goals = load("scenarios/controlled_variants/scenario_bottleneck_n4_convoy.json")
        actions = {0: [DOWN] * 4, 1: [DOWN] * 4, 2: [UP] * 4, 3: [UP] * 4}
        per_step = replay_with_no_progress(grid, starts, goals, actions, max_steps=4)
        last = per_step[-1]
        self.assertEqual(last.overlap_integrity_status, "INTEGRITY_ISSUE")
        self.assertGreater(last.overlap_timesteps, 0)
        # no_progress tracking must still run normally alongside the KI-1
        # overlap, exactly like trigger_type/blocking_type already do.
        self.assertEqual(last.trigger_type, "FULL")
        self.assertEqual(last.no_progress_status, "OK")
        self.assertEqual(last.no_progress_incomplete_steps, 0)
        self.assertEqual(len(dict(last.current_no_progress_streaks)), 4)


class TestNoProgressSynthetic(unittest.TestCase):
    """SYNTHETIC (see the module-level docstring's REAL/SYNTHETIC
    distinction): hand-fed inputs used to exercise Monitor logic paths
    that have no real-CSV example -- in particular, a transient detour
    followed by recovery, which docs/deadlock_definition_v1.md Section 6
    already records as a real-data evidentiary gap ("zero examples ...
    of a successful episode that experienced an active block/WAIT/
    no-progress window and then recovered") -- so this case can only be
    exercised synthetically, exactly as flagged in the 3B-2 approval."""

    def test_not_computed_by_default_when_inputs_never_supplied(self):
        """Approval point 7: calling observe_step() with only the original
        3 positional arguments (as every pre-3B-2A caller does, including
        every 3B-1 caller that does not pass goal_distance_*) must leave
        the no_progress fields at their inert defaults."""
        m = DeadlockMonitor()
        m.reset({0: (0, 0)}, goals={0: (9, 9)})
        r = m.observe_step(1, {0: (1, 0)}, {0: "none"})
        self.assertEqual(r.no_progress_status, "NOT_COMPUTED")
        self.assertEqual(r.current_no_progress_streaks, ())
        self.assertEqual(r.no_progress_incomplete_steps, 0)
        self.assertEqual(r.trigger_type, "NONE")

    def test_transient_detour_then_recovery_is_not_a_false_candidate(self):
        """SYNTHETIC transient-recovery case (approval point 8, explicit
        requirement). An agent's BFS distance INCREASES for two steps (a
        detour away from the goal -- counted as no-progress, matching
        analyze_failure_logs.py's cause-agnostic `progress <= 0` exactly),
        then decreases again (genuine advance) -> the streak must climb
        during the detour and drop back to 0 (not merely plateau) the
        instant real progress resumes."""
        m = DeadlockMonitor()
        m.reset({0: (0, 0)}, goals={0: (9, 9)})
        # t=1: distance 5 -> 6 (detour, got worse) -> no-progress streak=1.
        r1 = m.observe_step(1, {0: (0, 0)}, {0: "none"},
                             goal_distance_before={0: 5}, goal_distance_after={0: 6})
        self.assertEqual(r1.current_no_progress_streaks, ((0, 1),))
        # t=2: distance 6 -> 6 (still stuck, unchanged) -> streak=2.
        r2 = m.observe_step(2, {0: (0, 0)}, {0: "none"},
                             goal_distance_before={0: 6}, goal_distance_after={0: 6})
        self.assertEqual(r2.current_no_progress_streaks, ((0, 2),))
        # t=3: distance 6 -> 3 (real recovery) -> streak resets to 0 (dropped).
        r3 = m.observe_step(3, {0: (0, 0)}, {0: "none"},
                             goal_distance_before={0: 6}, goal_distance_after={0: 3})
        self.assertEqual(r3.current_no_progress_streaks, ())
        self.assertEqual(r3.no_progress_status, "OK")
        self.assertEqual(r3.no_progress_incomplete_steps, 0)

    def test_wait_and_every_conflict_type_count_identically(self):
        """Approval point 3: WAIT, and a wall/vertex/edge conflict
        bounce-back, must all be treated identically by no_progress (it is
        cause-agnostic by design -- distinguishing *why* is
        persistent_conflict's job, Phase 3B-1). Four independent agents,
        one per cause, all with before==after (no distance change) in the
        same step, must all accumulate streak=1 identically regardless of
        `conflict_type`."""
        m = DeadlockMonitor()
        m.reset({0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3, 0)},
                 goals={0: (9, 0), 1: (9, 1), 2: (9, 2), 3: (9, 3)})
        r = m.observe_step(
            1, {0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3, 0)},
            {0: "none", 1: "wall", 2: "vertex", 3: "edge"},
            goal_distance_before={0: 9, 1: 9, 2: 9, 3: 9},
            goal_distance_after={0: 9, 1: 9, 2: 9, 3: 9})
        self.assertEqual(r.current_no_progress_streaks, ((0, 1), (1, 1), (2, 1), (3, 1)))

    def test_agent_at_or_past_goal_is_excluded(self):
        """Approval point 3 (목표 도달): goal_distance_before <= 0 (already
        at goal) must never produce a streak, even though the agent is
        stationary -- same gate as offline `_active()`."""
        m = DeadlockMonitor()
        m.reset({0: (0, 0), 1: (5, 5)}, goals={0: (0, 1), 1: (9, 9)})
        r = m.observe_step(
            1, {0: (0, 0), 1: (5, 5)}, {0: "none", 1: "none"},
            goal_distance_before={0: 0, 1: 9}, goal_distance_after={0: 0, 1: 9})
        self.assertEqual(r.current_no_progress_streaks, ((1, 1),))  # only agent 1

    def test_before_unreachable_is_excluded_not_incomplete(self):
        """Approval point 4: `before < 0` (BFS-unreachable from this cell)
        is treated the same as `before == 0` -- dropped as "not active",
        NOT counted as an incomplete-data step. Distinguishing this from
        genuinely missing/insufficient data matters: an unreachable cell
        is a normal (if unusual) observation, not a data problem."""
        m = DeadlockMonitor()
        m.reset({0: (0, 0)}, goals={0: (9, 9)})
        r = m.observe_step(1, {0: (0, 0)}, {0: "none"},
                            goal_distance_before={0: -1}, goal_distance_after={0: -1})
        self.assertEqual(r.current_no_progress_streaks, ())
        self.assertEqual(r.no_progress_status, "OK")
        self.assertEqual(r.no_progress_incomplete_steps, 0)

    def test_after_unreachable_is_incomplete_not_a_guessed_advance(self):
        """Approval point 4: `before > 0` but `after < 0` must NOT be
        naively computed as `before - after` (which would look like a
        large, spurious advance) -- it must be flagged incomplete instead.
        This is a deliberate online safety check beyond what the offline
        analyzer's own `progress` field guards against."""
        m = DeadlockMonitor()
        m.reset({0: (0, 0)}, goals={0: (9, 9)})
        r = m.observe_step(1, {0: (1, 0)}, {0: "none"},
                            goal_distance_before={0: 5}, goal_distance_after={0: -1})
        self.assertEqual(r.current_no_progress_streaks, ())
        self.assertEqual(r.no_progress_status, "OK")
        self.assertEqual(r.no_progress_incomplete_steps, 1)

    def test_one_agent_missing_data_does_not_reset_another_agents_streak(self):
        """Approval point 4 (core requirement): agent 1's missing entry
        must NOT reset agent 0's already-building streak."""
        m = DeadlockMonitor()
        m.reset({0: (0, 0), 1: (5, 5)}, goals={0: (9, 9), 1: (9, 8)})
        r1 = m.observe_step(
            1, {0: (0, 0), 1: (5, 6)}, {0: "none", 1: "none"},
            goal_distance_before={0: 5, 1: 9}, goal_distance_after={0: 5, 1: 8})
        self.assertEqual(r1.current_no_progress_streaks, ((0, 1),))  # agent0 stuck, agent1 advanced
        self.assertEqual(r1.no_progress_incomplete_steps, 0)
        # Step 2: agent 1 is missing from goal_distance_after entirely;
        # agent 0's data is present and still shows no progress.
        r2 = m.observe_step(
            2, {0: (0, 0), 1: (5, 6)}, {0: "none", 1: "none"},
            goal_distance_before={0: 5, 1: 8}, goal_distance_after={0: 5})
        self.assertEqual(r2.no_progress_incomplete_steps, 1)
        self.assertEqual(r2.current_no_progress_streaks, ((0, 2),))  # agent0 streak kept growing
        self.assertNotIn(1, dict(r2.current_no_progress_streaks))  # agent1 dropped, not guessed

    def test_only_one_of_the_pair_supplied_is_also_incomplete(self):
        """Approval point 4: goal_distance_before/after omitted entirely
        for an agent counts as missing, same as a partially-missing dict."""
        m = DeadlockMonitor()
        m.reset({0: (0, 0)}, goals={0: (9, 9)})
        r = m.observe_step(1, {0: (1, 0)}, {0: "none"},
                            goal_distance_before={0: 5})  # goal_distance_after omitted
        self.assertEqual(r.no_progress_status, "OK")
        self.assertEqual(r.no_progress_incomplete_steps, 1)
        self.assertEqual(r.current_no_progress_streaks, ())

    def test_result_snapshot_is_not_aliased_across_steps(self):
        """Approval point 6: confirms the mutable-field aliasing fix
        extends correctly to the new tuple-valued
        current_no_progress_streaks field -- each returned snapshot must
        keep ITS OWN value even as later calls change the live state."""
        m = DeadlockMonitor()
        m.reset({0: (0, 0)}, goals={0: (9, 9)})
        r1 = m.observe_step(1, {0: (0, 0)}, {0: "none"},
                             goal_distance_before={0: 5}, goal_distance_after={0: 5})
        r2 = m.observe_step(2, {0: (0, 0)}, {0: "none"},
                             goal_distance_before={0: 5}, goal_distance_after={0: 5})
        self.assertEqual(r1.current_no_progress_streaks, ((0, 1),))
        self.assertEqual(r2.current_no_progress_streaks, ((0, 2),))
        # r1 must NOT have silently become equal to r2's later value.
        self.assertNotEqual(r1.current_no_progress_streaks, r2.current_no_progress_streaks)


class _FixedActionPolicy:
    """Deterministic, stateless, driven by a pre-set per-agent action list
    (falls back to WAIT once exhausted). Unlike TestBaselineUnaffectedByMonitor's
    `_StubPolicy` (which just hugs a wall), this reuses the exact fixed
    action sequences already established and cross-checked elsewhere in
    this file, so it can drive `scripts.run_il.run_episode()`'s REAL code
    path (Phase 3B-2B's actual wiring, not DeadlockMonitor's API called
    directly) while landing on already-known first_seen/trigger_timestep/
    blocking_type values. Computes `self._dist` via the real `bfs_dist()`,
    exactly like `NavHintPolicy.reset()` does, since
    `scripts.run_il.run_episode()` now reads `policy._dist` whenever a
    monitor is attached (Phase 3B-2B)."""

    def __init__(self, actions_per_agent):
        self._actions_per_agent = actions_per_agent
        self._t = 0

    def reset(self, grid, goals):
        self._grid = grid
        self._dist = {aid: bfs_dist(grid, goal) for aid, goal in goals.items()}
        self._t = 0

    def act(self, obs, positions):
        actions = {
            aid: (seq[self._t] if self._t < len(seq) else WAIT)
            for aid, seq in self._actions_per_agent.items()
        }
        self._t += 1
        return actions


class TestRunIlNoProgressWiring(unittest.TestCase):
    """Phase 3B-2B: validates the ACTUAL `scripts/run_il.py::run_episode()`
    wiring -- not DeadlockMonitor's API in isolation. Confirms
    goal_distance_before/after now really reach `observe_step()` through
    the real function (not a hand-built call), that doing so changes
    nothing about the simulated rollout, and that it stays fully
    independent of persistent_conflict (which this phase deliberately does
    NOT wire -- condition 6) and of Logger's existing CSV output."""

    def test_corridor_bay_n2_no_progress_reaches_monitor_through_run_episode(self):
        import scripts.run_il as run_il
        from src.common.deadlock_monitor import DeadlockMonitor

        grid, starts, goals = load("scenarios/controlled/scenario_corridor_bay_n2.json")
        monitor = DeadlockMonitor()
        policy = _FixedActionPolicy({0: [RIGHT] * 8, 1: [LEFT] * 8})
        run_il.run_episode(policy, grid, starts, goals, max_steps=8, logger=None, monitor=monitor)
        r = monitor.result
        self.assertEqual(r.no_progress_status, "OK")
        self.assertEqual(r.current_no_progress_streaks, ((0, 4), (1, 4)))
        self.assertEqual(r.no_progress_incomplete_steps, 0)
        # Phase 3A/3B-1 are unaffected by this wiring (approval condition 5):
        # trigger/blocking fields identical to every prior-phase check of
        # this same scenario, and persistent_conflict stays NOT_COMPUTED
        # because condition 6 explicitly excludes wiring it this round.
        self.assertEqual(r.trigger_type, "FULL")
        self.assertEqual(r.first_seen, 4)
        self.assertEqual(r.trigger_timestep, 5)
        self.assertEqual(r.blocking_type, "VERTEX")
        self.assertEqual(r.persistent_conflict_status, "NOT_COMPUTED")

    def test_monitor_on_off_trajectory_identical_on_a_stalling_scenario(self):
        """Approval condition 9. Unlike TestBaselineUnaffectedByMonitor's
        wall-hugging stub (which never stalls), this exercises the ON/OFF
        comparison on a scenario that actually drives the new
        goal_distance computation through a Full-repeat freeze, which is
        exactly the path most likely to perturb something if the wiring
        were wrong."""
        import scripts.run_il as run_il
        from src.common.deadlock_monitor import DeadlockMonitor

        grid, starts, goals = load("scenarios/controlled/scenario_corridor_bay_n2.json")
        result_no_monitor = run_il.run_episode(
            _FixedActionPolicy({0: [RIGHT] * 8, 1: [LEFT] * 8}), grid, starts, goals,
            max_steps=8, logger=None, monitor=None)
        result_with_monitor = run_il.run_episode(
            _FixedActionPolicy({0: [RIGHT] * 8, 1: [LEFT] * 8}), grid, starts, goals,
            max_steps=8, logger=None, monitor=DeadlockMonitor())

        self.assertEqual(result_no_monitor["steps"], result_with_monitor["steps"])
        self.assertEqual(result_no_monitor["trajectory"], result_with_monitor["trajectory"])
        self.assertEqual(result_no_monitor["actions"], result_with_monitor["actions"])
        self.assertEqual(result_no_monitor["all_at_goal"], result_with_monitor["all_at_goal"])
        self.assertEqual(result_no_monitor["timed_out"], result_with_monitor["timed_out"])
        self.assertEqual(result_no_monitor["final_positions"], result_with_monitor["final_positions"])

    def test_csv_differs_only_in_trigger_reason_when_monitor_is_attached(self):
        """Phase 3B-3 REPLACES the old 3B-2B invariant here (which asserted
        byte-identical CSVs regardless of monitor). That assertion is now
        deliberately false: Phase 3B-3's entire purpose is to make
        attaching a monitor alongside a logger populate the existing
        `trigger_reason` column, which previously stayed "" always
        (approval condition 6: replace, with the reason recorded here).
        The corrected, precise invariant: EVERY column except
        `trigger_reason` is identical with/without a monitor, the
        `monitor=None` path's `trigger_reason` is unconditionally "" (no
        regression there), and the `monitor` path's `trigger_reason`
        reflects the real Trigger state (checked in detail by the
        dedicated sticky-timeline test below)."""
        import csv
        import io
        import tempfile

        import scripts.run_il as run_il
        from src.common.deadlock_monitor import DeadlockMonitor
        from src.common.episode_logger import EpisodeLogger

        grid, starts, goals = load("scenarios/controlled_variants/scenario_bottleneck_n4_convoy.json")
        actions = {0: [DOWN] * 4, 1: [DOWN] * 4, 2: [UP] * 4, 3: [UP] * 4}

        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            logger_alone = EpisodeLogger("x", scenario="s", mode="IL", out_dir=tmp1)
            run_il.run_episode(
                _FixedActionPolicy(actions), grid, starts, goals,
                max_steps=4, logger=logger_alone, monitor=None)
            path_alone = logger_alone.write_steps()

            logger_with_monitor = EpisodeLogger("x", scenario="s", mode="IL", out_dir=tmp2)
            run_il.run_episode(
                _FixedActionPolicy(actions), grid, starts, goals,
                max_steps=4, logger=logger_with_monitor, monitor=DeadlockMonitor())
            path_with_monitor = logger_with_monitor.write_steps()

            rows_alone = list(csv.DictReader(io.StringIO(path_alone.read_text(encoding="utf-8"))))
            rows_with_monitor = list(csv.DictReader(io.StringIO(path_with_monitor.read_text(encoding="utf-8"))))

        self.assertEqual(len(rows_alone), len(rows_with_monitor))
        self.assertEqual(list(rows_alone[0].keys()), list(rows_with_monitor[0].keys()))  # schema/order unchanged
        for r1, r2 in zip(rows_alone, rows_with_monitor):
            self.assertEqual(r1["trigger_reason"], "")  # monitor=None path: unchanged regression check
            for key in r1:
                if key == "trigger_reason":
                    continue
                self.assertEqual(r1[key], r2[key], f"column {key!r} differs: {r1[key]!r} vs {r2[key]!r}")
        # The monitor path must have ACTUALLY populated the column with
        # something other than the blank default, or this test would pass
        # vacuously without proving anything.
        self.assertTrue(any(r["trigger_reason"] for r in rows_with_monitor))

    def test_trigger_reason_csv_timeline_is_sticky_and_consistent_across_agents(self):
        """Approval condition 5 (dedicated test, real IL run + real CSV
        read-back, not DeadlockMonitor's API in isolation): before the
        confirmed Full-repeat (first_seen=1, trigger_timestep=2 for this
        scenario, matching every prior phase's evidence for it), every row
        reads "TRIGGER_NONE"; from trigger_timestep onward, every row
        reads "TRIGGER_FULL_REPEAT" (sticky -- Phase 3A's trigger_type
        never reverts once settled). Both of this scenario's two agents'
        rows at a given timestep must carry the IDENTICAL value (it is an
        episode/monitor-level signal, not a per-agent one -- same pattern
        as the existing `done`/`success` columns)."""
        import csv
        import io
        import tempfile

        import scripts.run_il as run_il
        from src.common.deadlock_monitor import DeadlockMonitor
        from src.common.episode_logger import EpisodeLogger

        grid, starts, goals = load("scenarios/controlled/scenario_intersection_n2.json")
        actions = {0: [RIGHT] * 6, 1: [DOWN] * 6}

        with tempfile.TemporaryDirectory() as tmp:
            logger = EpisodeLogger("x", scenario="s", mode="IL", out_dir=tmp)
            monitor = DeadlockMonitor()
            run_il.run_episode(
                _FixedActionPolicy(actions), grid, starts, goals,
                max_steps=6, logger=logger, monitor=monitor)
            path = logger.write_steps()
            csv_text = path.read_text(encoding="utf-8")

        rows = list(csv.DictReader(io.StringIO(csv_text)))
        by_timestep = {}
        for row in rows:
            by_timestep.setdefault(int(row["timestep"]), []).append(row["trigger_reason"])

        for t in range(1, 7):
            values_at_t = set(by_timestep[t])
            self.assertEqual(len(values_at_t), 1,
                              f"t={t}: agent rows disagree on trigger_reason: {by_timestep[t]}")
            expected = "TRIGGER_NONE" if t < 2 else "TRIGGER_FULL_REPEAT"
            self.assertEqual(values_at_t.pop(), expected, f"t={t}")

        # Cross-check against the monitor's own final state for this run.
        r = monitor.result
        self.assertEqual(r.trigger_reason_code, "TRIGGER_FULL_REPEAT")
        self.assertEqual(r.trigger_timestep, 2)

    def test_ki1_bottleneck_convoy_no_progress_not_suppressed_by_overlap(self):
        """Approval point 8 (KI-1), through the real run_episode() path."""
        import scripts.run_il as run_il
        from src.common.deadlock_monitor import DeadlockMonitor

        grid, starts, goals = load("scenarios/controlled_variants/scenario_bottleneck_n4_convoy.json")
        actions = {0: [DOWN] * 4, 1: [DOWN] * 4, 2: [UP] * 4, 3: [UP] * 4}
        monitor = DeadlockMonitor()
        run_il.run_episode(
            _FixedActionPolicy(actions), grid, starts, goals,
            max_steps=4, logger=None, monitor=monitor)
        r = monitor.result
        self.assertEqual(r.overlap_integrity_status, "INTEGRITY_ISSUE")
        self.assertGreater(r.overlap_timesteps, 0)
        self.assertEqual(r.trigger_type, "FULL")
        self.assertEqual(r.no_progress_status, "OK")
        self.assertEqual(r.no_progress_incomplete_steps, 0)
        self.assertEqual(dict(r.current_no_progress_streaks), {0: 3, 1: 4, 2: 4, 3: 3})

    def test_success_cycle_ring_n4_no_false_candidate_through_run_episode(self):
        """Approval point 8 (성공 Cycle)."""
        import scripts.run_il as run_il
        from src.common.deadlock_monitor import DeadlockMonitor

        grid, starts, goals = load("scenarios/controlled/scenario_cycle_ring_n4.json")
        actions = {0: [RIGHT, RIGHT], 1: [DOWN, DOWN], 2: [LEFT, LEFT], 3: [UP, UP]}
        monitor = DeadlockMonitor()
        result = run_il.run_episode(
            _FixedActionPolicy(actions), grid, starts, goals,
            max_steps=2, logger=None, monitor=monitor)
        r = monitor.result
        self.assertTrue(result["all_at_goal"])
        self.assertEqual(r.no_progress_status, "OK")
        self.assertEqual(r.current_no_progress_streaks, ())
        self.assertEqual(r.trigger_type, "NONE")

    def test_real_navhint_checkpoint_end_to_end_smoke_scenario(self):
        """Approval point 8 (성공 Smoke) + point 9: the ACTUAL trained
        checkpoint and NavHintPolicy, not a fixed-action stand-in, driven
        through the real run_episode() path with a monitor attached. This
        is the strictest form of "실제 실행 결과" -- genuine IL inference."""
        import scripts.run_il as run_il
        from src.common.deadlock_monitor import DeadlockMonitor
        from src.il.policy import NavHintPolicy

        ckpt = PROJECT_ROOT / "checkpoints" / "cnn_navhint.pt"
        grid, starts, goals = run_il.load_scenario(
            PROJECT_ROOT / "scenarios" / "smoke" / "scenario_empty_s11_n3.json")
        policy = NavHintPolicy(ckpt)
        monitor = DeadlockMonitor()
        result = run_il.run_episode(policy, grid, starts, goals, max_steps=160, logger=None, monitor=monitor)
        r = monitor.result
        # Same reference behaviour scripts/smoke_test_il.py's strict
        # baseline already certifies (steps=9, all_at_goal=True).
        self.assertEqual(result["steps"], 9)
        self.assertTrue(result["all_at_goal"])
        self.assertEqual(r.no_progress_status, "OK")
        self.assertEqual(r.current_no_progress_streaks, ())  # never stalls
        self.assertEqual(r.trigger_type, "NONE")

    def test_real_navhint_checkpoint_end_to_end_corridor_bay_n2(self):
        """Approval point 8 (원본 Corridor) + point 9, with the real
        checkpoint: confirms the real IL policy's own decisions (not a
        fixed action list) reproduce the documented first_seen=4/
        trigger_timestep=5/VERTEX stall (docs/deadlock_definition_v1.md
        Section 4.3's worked example), and that no_progress grows to
        exactly steps-first_seen by the time the episode times out."""
        import scripts.run_il as run_il
        from src.common.deadlock_monitor import DeadlockMonitor
        from src.il.policy import NavHintPolicy

        ckpt = PROJECT_ROOT / "checkpoints" / "cnn_navhint.pt"
        grid, starts, goals = run_il.load_scenario(
            PROJECT_ROOT / "scenarios" / "controlled" / "scenario_corridor_bay_n2.json")
        policy = NavHintPolicy(ckpt)
        monitor = DeadlockMonitor()
        result = run_il.run_episode(policy, grid, starts, goals, max_steps=160, logger=None, monitor=monitor)
        r = monitor.result
        self.assertEqual(result["steps"], 160)
        self.assertFalse(result["all_at_goal"])
        self.assertTrue(result["timed_out"])
        self.assertEqual(r.trigger_type, "FULL")
        self.assertEqual(r.first_seen, 4)
        self.assertEqual(r.trigger_timestep, 5)
        self.assertEqual(r.blocking_type, "VERTEX")
        self.assertEqual(r.no_progress_status, "OK")
        self.assertEqual(r.current_no_progress_streaks, ((0, 156), (1, 156)))  # 160 - 4
        self.assertEqual(r.no_progress_incomplete_steps, 0)
        self.assertEqual(r.persistent_conflict_status, "NOT_COMPUTED")  # condition 6: not wired this round


class TestTriggerReasonCsvRealScenarios(unittest.TestCase):
    """Phase 3B-3, approval condition 7: the real CSV `trigger_reason`
    timeline, read back from disk, for the original Corridor/Intersection/
    Bottleneck stalls, a successful Cycle/Smoke run (with the REAL
    NavHintPolicy checkpoint), and KI-1. These read the written CSV
    directly (not just DeadlockMonitorResult) so the actual Logger output
    is what gets checked, matching this phase's stated goal."""

    @staticmethod
    def _csv_rows(path):
        import csv
        import io
        return list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))

    def _trigger_reason_by_timestep(self, rows):
        by_t = {}
        for row in rows:
            by_t.setdefault(int(row["timestep"]), set()).add(row["trigger_reason"])
        return by_t

    def test_corridor_bay_n2_real_checkpoint_trigger_reason_timeline(self):
        import tempfile

        import scripts.run_il as run_il
        from src.common.deadlock_monitor import DeadlockMonitor
        from src.common.episode_logger import EpisodeLogger
        from src.il.policy import NavHintPolicy

        ckpt = PROJECT_ROOT / "checkpoints" / "cnn_navhint.pt"
        grid, starts, goals = run_il.load_scenario(
            PROJECT_ROOT / "scenarios" / "controlled" / "scenario_corridor_bay_n2.json")
        with tempfile.TemporaryDirectory() as tmp:
            logger = EpisodeLogger("x", scenario="s", mode="IL", out_dir=tmp)
            monitor = DeadlockMonitor()
            run_il.run_episode(NavHintPolicy(ckpt), grid, starts, goals, max_steps=20,
                                logger=logger, monitor=monitor)
            rows = self._csv_rows(logger.write_steps())

        by_t = self._trigger_reason_by_timestep(rows)
        for t in range(1, 21):
            values = by_t[t]
            self.assertEqual(len(values), 1, f"t={t}: {values}")
            expected = "TRIGGER_NONE" if t < 5 else "TRIGGER_FULL_REPEAT"  # doc Section 4.3 worked example
            self.assertEqual(values.pop(), expected, f"t={t}")

    def test_intersection_n2_real_checkpoint_trigger_reason_timeline(self):
        import tempfile

        import scripts.run_il as run_il
        from src.common.deadlock_monitor import DeadlockMonitor
        from src.common.episode_logger import EpisodeLogger
        from src.il.policy import NavHintPolicy

        ckpt = PROJECT_ROOT / "checkpoints" / "cnn_navhint.pt"
        grid, starts, goals = run_il.load_scenario(
            PROJECT_ROOT / "scenarios" / "controlled" / "scenario_intersection_n2.json")
        with tempfile.TemporaryDirectory() as tmp:
            logger = EpisodeLogger("x", scenario="s", mode="IL", out_dir=tmp)
            monitor = DeadlockMonitor()
            run_il.run_episode(NavHintPolicy(ckpt), grid, starts, goals, max_steps=10,
                                logger=logger, monitor=monitor)
            rows = self._csv_rows(logger.write_steps())

        by_t = self._trigger_reason_by_timestep(rows)
        for t in range(1, 11):
            values = by_t[t]
            self.assertEqual(len(values), 1, f"t={t}: {values}")
            expected = "TRIGGER_NONE" if t < 2 else "TRIGGER_FULL_REPEAT"
            self.assertEqual(values.pop(), expected, f"t={t}")

    def test_bottleneck_n4_real_checkpoint_trigger_reason_timeline(self):
        import tempfile

        import scripts.run_il as run_il
        from src.common.deadlock_monitor import DeadlockMonitor
        from src.common.episode_logger import EpisodeLogger
        from src.il.policy import NavHintPolicy

        ckpt = PROJECT_ROOT / "checkpoints" / "cnn_navhint.pt"
        grid, starts, goals = run_il.load_scenario(
            PROJECT_ROOT / "scenarios" / "controlled" / "scenario_bottleneck_n4.json")
        with tempfile.TemporaryDirectory() as tmp:
            logger = EpisodeLogger("x", scenario="s", mode="IL", out_dir=tmp)
            monitor = DeadlockMonitor()
            run_il.run_episode(NavHintPolicy(ckpt), grid, starts, goals, max_steps=10,
                                logger=logger, monitor=monitor)
            rows = self._csv_rows(logger.write_steps())

        by_t = self._trigger_reason_by_timestep(rows)
        for t in range(1, 11):
            values = by_t[t]
            self.assertEqual(len(values), 1, f"t={t}: {values}")
            expected = "TRIGGER_NONE" if t < 2 else "TRIGGER_FULL_REPEAT"
            self.assertEqual(values.pop(), expected, f"t={t}")

    def test_success_smoke_scenario_real_checkpoint_trigger_reason_always_none(self):
        """Approval condition 7 (성공 Smoke), real checkpoint: a clean
        success run must show "TRIGGER_NONE" for every row, every
        timestep -- never a false trigger."""
        import tempfile

        import scripts.run_il as run_il
        from src.common.deadlock_monitor import DeadlockMonitor
        from src.common.episode_logger import EpisodeLogger
        from src.il.policy import NavHintPolicy

        ckpt = PROJECT_ROOT / "checkpoints" / "cnn_navhint.pt"
        grid, starts, goals = run_il.load_scenario(
            PROJECT_ROOT / "scenarios" / "smoke" / "scenario_empty_s11_n3.json")
        with tempfile.TemporaryDirectory() as tmp:
            logger = EpisodeLogger("x", scenario="s", mode="IL", out_dir=tmp)
            monitor = DeadlockMonitor()
            result = run_il.run_episode(NavHintPolicy(ckpt), grid, starts, goals, max_steps=160,
                                         logger=logger, monitor=monitor)
            rows = self._csv_rows(logger.write_steps())

        self.assertTrue(result["all_at_goal"])
        self.assertEqual(result["steps"], 9)
        self.assertTrue(all(row["trigger_reason"] == "TRIGGER_NONE" for row in rows))

    def test_success_cycle_ring_n4_trigger_reason_always_none(self):
        """Approval condition 7 (성공 Cycle)."""
        import tempfile

        import scripts.run_il as run_il
        from src.common.deadlock_monitor import DeadlockMonitor
        from src.common.episode_logger import EpisodeLogger

        grid, starts, goals = load("scenarios/controlled/scenario_cycle_ring_n4.json")
        actions = {0: [RIGHT, RIGHT], 1: [DOWN, DOWN], 2: [LEFT, LEFT], 3: [UP, UP]}
        with tempfile.TemporaryDirectory() as tmp:
            logger = EpisodeLogger("x", scenario="s", mode="IL", out_dir=tmp)
            monitor = DeadlockMonitor()
            result = run_il.run_episode(_FixedActionPolicy(actions), grid, starts, goals,
                                         max_steps=2, logger=logger, monitor=monitor)
            rows = self._csv_rows(logger.write_steps())

        self.assertTrue(result["all_at_goal"])
        self.assertTrue(all(row["trigger_reason"] == "TRIGGER_NONE" for row in rows))

    def test_ki1_bottleneck_convoy_trigger_reason_not_suppressed_by_overlap(self):
        """Approval condition 7 (KI-1): the documented single-pass
        vertex-resolution overlap quirk must not prevent trigger_reason
        from correctly reflecting the Full-repeat, kept in its own test
        per this file's established KI-1-separation discipline."""
        import tempfile

        import scripts.run_il as run_il
        from src.common.deadlock_monitor import DeadlockMonitor
        from src.common.episode_logger import EpisodeLogger

        grid, starts, goals = load("scenarios/controlled_variants/scenario_bottleneck_n4_convoy.json")
        actions = {0: [DOWN] * 4, 1: [DOWN] * 4, 2: [UP] * 4, 3: [UP] * 4}
        with tempfile.TemporaryDirectory() as tmp:
            logger = EpisodeLogger("x", scenario="s", mode="IL", out_dir=tmp)
            monitor = DeadlockMonitor()
            run_il.run_episode(_FixedActionPolicy(actions), grid, starts, goals,
                                max_steps=4, logger=logger, monitor=monitor)
            rows = self._csv_rows(logger.write_steps())

        self.assertEqual(monitor.result.overlap_integrity_status, "INTEGRITY_ISSUE")
        by_t = self._trigger_reason_by_timestep(rows)
        for t in range(1, 5):
            values = by_t[t]
            self.assertEqual(len(values), 1, f"t={t}: {values}")
            expected = "TRIGGER_NONE" if t < 2 else "TRIGGER_FULL_REPEAT"
            self.assertEqual(values.pop(), expected, f"t={t}")


if __name__ == "__main__":
    unittest.main()
