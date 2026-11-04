"""Tests for DeadlockMonitor.rearm() (Monitor rearm, DEV step 1).

Two kinds of test data, never mixed:

  * REAL -- the Bottleneck GLOBAL_HYBRID_DEV_H3 run
    20261103_152705_bottleneck_n4_GLOBAL_HYBRID_DEV_H3 (real NavHint IL +
    real CBS, H_exec=3): t=2 Trigger, CBS t=3..5, IL_RESUME at t=5, re-stall
    S_7 == S_6 held to t=160. Its first ten joint states and per-step
    conflict_type values are copied below (H3_TRAJECTORY / H3_VERTEX_TS) so
    the core tests do not depend on the gitignored outputs/ folder; one test
    additionally replays the full saved result.json when it is present, and
    is skipped otherwise. The saved file is only read, never written.
  * SYNTHETIC -- hand-fed positions with no claim of being real IL
    behaviour, used to exercise rearm()'s bookkeeping rules.

rearm() is only ever called explicitly here; no Runner is involved.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.common.deadlock_monitor import DeadlockMonitor, DeadlockMonitorResult, TriggerRecord
from src.common.episode_logger import MONITOR_FIELDS, EpisodeLogger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
H3_RUN_ID = "20261103_152705_bottleneck_n4_GLOBAL_HYBRID_DEV_H3"
H3_RESULT = PROJECT_ROOT / "outputs" / "hybrid_e2e" / H3_RUN_ID / "result.json"

# REAL: trajectory[0..9] of the run above (agent_id -> (row, col)).
H3_GOALS = {0: (1, 7), 1: (3, 7), 2: (0, 1), 3: (4, 1)}
_S6 = {0: (2, 5), 1: (2, 4), 2: (1, 5), 3: (3, 5)}
H3_TRAJECTORY = [
    {0: (1, 1), 1: (3, 1), 2: (0, 7), 3: (4, 7)},   # t=0
    {0: (1, 2), 1: (2, 1), 2: (1, 7), 3: (3, 7)},   # t=1
    {0: (1, 2), 1: (2, 1), 2: (1, 7), 3: (3, 7)},   # t=2  Trigger (first_seen 1)
    {0: (1, 3), 1: (2, 2), 2: (1, 6), 3: (4, 7)},   # t=3  CBS k=1
    {0: (2, 3), 1: (2, 2), 2: (1, 5), 3: (4, 6)},   # t=4  CBS k=2
    {0: (2, 4), 1: (2, 3), 2: (0, 5), 3: (4, 5)},   # t=5  CBS k=3 -> IL_RESUME
    dict(_S6),                                      # t=6  first IL step after resume
    dict(_S6),                                      # t=7  re-stall
    dict(_S6),                                      # t=8
    dict(_S6),                                      # t=9
]
H3_RESUME_T = 5
H3_VERTEX_TS = {2, 7, 8, 9}  # every agent conflict_type == "vertex" at these t; "none" otherwise


def _h3_conflicts(t):
    return {aid: ("vertex" if t in H3_VERTEX_TS else "none") for aid in H3_GOALS}


def _replay(monitor, trajectory, conflicts_fn, *, until, start=1, rearm_at=None):
    """observe_step for t in [start, until]; rearm right after observing rearm_at.
    Returns {t: snapshot}."""
    out = {}
    for t in range(start, until + 1):
        out[t] = monitor.observe_step(t, trajectory[t], conflicts_fn(t))
        if t == rearm_at:
            monitor.rearm(trajectory[t], t)
    return out


# SYNTHETIC helpers: two agents on an open line, goal far away.
GOALS = {0: (0, 9), 1: (5, 9)}
A = {0: (0, 0), 1: (5, 0)}
B = {0: (0, 1), 1: (5, 1)}
C = {0: (0, 2), 1: (5, 2)}
D = {0: (0, 3), 1: (5, 3)}
NONE = {0: "none", 1: "none"}
VERTEX = {0: "vertex", 1: "vertex"}


def _fresh(start=A, goals=GOALS):
    m = DeadlockMonitor()
    m.reset(start, goals)
    return m


class TestRearmBottleneckH3(unittest.TestCase):
    """REAL: the H3 re-stall the latched Monitor missed."""

    def test_baseline_without_rearm_latches_first_trigger(self):
        m = DeadlockMonitor()
        m.reset(H3_TRAJECTORY[0], H3_GOALS)
        snaps = _replay(m, H3_TRAJECTORY, _h3_conflicts, until=9)
        self.assertEqual(snaps[1].trigger_type, "NONE")
        for t in range(2, 10):
            self.assertEqual((snaps[t].trigger_timestep, snaps[t].first_seen, snaps[t].period), (2, 1, 1))
        self.assertEqual(m.rearm_count, 0)
        self.assertEqual(m.trigger_history, ())

    def test_rearm_at_resume_detects_restall_at_t7(self):
        m = DeadlockMonitor()
        m.reset(H3_TRAJECTORY[0], H3_GOALS)
        snaps = _replay(m, H3_TRAJECTORY, _h3_conflicts, until=9, rearm_at=H3_RESUME_T)
        self.assertEqual(m.armed_since, 5)
        # t=6: a new state, observed by the rearmed detector -- no Trigger, stale code gone.
        self.assertEqual(snaps[6].trigger_type, "NONE")
        self.assertEqual(snaps[6].trigger_reason_code, "TRIGGER_NONE")
        self.assertIsNone(snaps[6].trigger_timestep)
        # t=7: Full repeat of S_6, on the global clock.
        r7 = snaps[7]
        self.assertEqual(r7.trigger_reason_code, "TRIGGER_FULL_REPEAT")
        self.assertEqual((r7.trigger_timestep, r7.first_seen, r7.period), (7, 6, 1))
        self.assertEqual(r7.blocking_type, "VERTEX")
        self.assertEqual(r7.blocking_agents, (0, 1, 2, 3))
        # latched again afterwards (no rearm in between)
        self.assertEqual((snaps[9].trigger_timestep, snaps[9].first_seen), (7, 6))

    def test_first_trigger_preserved_in_history(self):
        m = DeadlockMonitor()
        m.reset(H3_TRAJECTORY[0], H3_GOALS)
        _replay(m, H3_TRAJECTORY, _h3_conflicts, until=9, rearm_at=H3_RESUME_T)
        self.assertEqual(m.trigger_history, (TriggerRecord(
            trigger_type="FULL", trigger_reason_code="TRIGGER_FULL_REPEAT",
            trigger_timestep=2, first_seen=1, period=1,
            blocking_type="VERTEX", blocking_agents=(0, 1, 2, 3),
            armed_since=0, rearmed_at=5,
        ),))
        self.assertEqual(m.rearm_count, 1)

    @unittest.skipUnless(H3_RESULT.exists(), f"saved run {H3_RUN_ID} not present")
    def test_saved_run_full_replay_matches_offline_restall(self):
        from scripts.run_hybrid_resume_dev import find_repeat_after

        raw = json.loads(H3_RESULT.read_text(encoding="utf-8"))
        traj = [{int(a): tuple(p) for a, p in s.items()} for s in raw["trajectory"]]
        goals = {int(a): tuple(p) for a, p in raw["goals"].items()}
        resume_t = next(tr["t"] for tr in raw["transitions"]
                        if tr["from"] == "CBS_EXEC" and tr["to"] == "IL_RESUME")
        self.assertEqual(resume_t, H3_RESUME_T)
        self.assertEqual(traj[:10], H3_TRAJECTORY)  # fixture above is a faithful copy

        m = DeadlockMonitor()
        m.reset(traj[0], goals)
        new_triggers = []
        for t in range(1, len(traj)):
            r = m.observe_step(t, traj[t], {})
            if r.trigger_timestep == t:
                new_triggers.append((t, r.first_seen, r.period))
            if t == resume_t:
                m.rearm(traj[t], t)
        offline = find_repeat_after(traj, resume_t, goals)
        self.assertEqual(new_triggers, [(2, 1, 1), (offline["t"], offline["first_seen"], offline["period"])])
        self.assertEqual(new_triggers[1], (7, 6, 1))
        self.assertEqual(raw["resume"]["restall_after_resume"], {"t": 7, "first_seen": 6, "period": 1})


class TestRearmNoPastLeak(unittest.TestCase):
    """SYNTHETIC: no state observed before rearm() can be a new first_seen."""

    def test_pre_rearm_state_is_forgotten(self):
        # Control: without rearm, returning to A (t=0) at t=3 triggers with first_seen 0.
        ctrl = _fresh()
        for t, s in ((1, B), (2, C)):
            ctrl.observe_step(t, s, NONE)
        self.assertEqual(ctrl.observe_step(3, A, NONE).first_seen, 0)

        m = _fresh()
        for t, s in ((1, B), (2, C)):
            m.observe_step(t, s, NONE)
        m.rearm(C, 2)
        r3 = m.observe_step(3, A, NONE)
        self.assertEqual(r3.trigger_type, "NONE")
        r4 = m.observe_step(4, B, NONE)
        self.assertEqual(r4.trigger_type, "NONE")
        r5 = m.observe_step(5, A, NONE)
        self.assertEqual((r5.trigger_timestep, r5.first_seen, r5.period), (5, 3, 2))
        self.assertGreaterEqual(r5.first_seen, m.armed_since)

    def test_rearm_state_registered_at_global_timestep(self):
        m = _fresh()
        for t, s in ((1, B), (2, C), (3, D)):
            m.observe_step(t, s, NONE)
        m.rearm(D, 3)
        r = m.observe_step(4, D, NONE)  # immediate repeat of the rearm state
        self.assertEqual((r.trigger_timestep, r.first_seen, r.period), (4, 3, 1))

    def test_pre_rearm_conflict_evidence_not_used(self):
        m = _fresh()
        m.observe_step(1, B, VERTEX)
        m.observe_step(2, B, VERTEX)            # Trigger, VERTEX
        self.assertEqual(m.result.blocking_type, "VERTEX")
        m.rearm(B, 2)
        r = m.observe_step(3, B, NONE)          # repeat with no conflict evidence in (2, 3]
        self.assertEqual((r.trigger_timestep, r.first_seen), (3, 2))
        self.assertEqual(r.blocking_type, "NONE")
        self.assertEqual(r.blocking_agents, ())

    def test_rearm_clears_stale_trigger_code(self):
        m = _fresh()
        m.observe_step(1, B, NONE)
        m.observe_step(2, B, NONE)
        self.assertEqual(m.result.trigger_reason_code, "TRIGGER_FULL_REPEAT")
        m.rearm(B, 2)
        after = m.result
        for name in ("trigger_type", "trigger_reason_code", "trigger_timestep",
                     "first_seen", "period", "blocking_type", "blocking_agents"):
            self.assertEqual(getattr(after, name), getattr(DeadlockMonitorResult(), name), name)
        self.assertEqual(m.observe_step(3, C, NONE).trigger_reason_code, "TRIGGER_NONE")


class TestRearmPreservesCumulativeRecords(unittest.TestCase):
    """SYNTHETIC: integrity and observational counters survive rearm()."""

    def test_overlap_integrity_preserved(self):
        m = _fresh()
        overlap = {0: (0, 1), 1: (0, 1)}
        m.observe_step(1, overlap, NONE)
        m.observe_step(2, C, NONE)
        m.rearm(C, 2)
        r = m.result
        self.assertEqual((r.overlap_timesteps, r.overlap_integrity_status), (1, "INTEGRITY_ISSUE"))
        m.observe_step(3, overlap, NONE)
        self.assertEqual(m.result.overlap_timesteps, 2)

    def test_persistent_conflict_and_no_progress_preserved(self):
        # Agents 0 and 1 both try to enter (0,1) from (0,0)/(0,2) and stay put.
        goals = {0: (0, 5), 1: (0, 0)}
        start = {0: (0, 0), 1: (0, 2)}
        intended = {0: (0, 1), 1: (0, 1)}
        conflicts = VERTEX
        m = DeadlockMonitor()
        m.reset(start, goals)
        kw = dict(current_positions=start, intended_positions=intended,
                  goal_distance_before={0: 5, 1: 2}, goal_distance_after={0: 5, 1: 2})
        m.observe_step(1, start, conflicts, **kw)
        m.observe_step(2, start, conflicts, current_positions=start)  # incomplete pc step
        m.observe_step(3, start, conflicts, **kw)
        m.observe_step(4, start, conflicts, goal_distance_before={0: 5}, goal_distance_after={0: 5},
                       current_positions=start, intended_positions=intended)  # agent 1 np-incomplete
        before = m.result
        m.rearm(start, 4)
        after = m.result
        for name in ("persistent_conflict_status", "persistent_conflict_incomplete_steps",
                     "current_persistent_conflict_streak", "current_persistent_conflict_key",
                     "no_progress_status", "no_progress_incomplete_steps",
                     "current_no_progress_streaks", "overlap_timesteps", "overlap_integrity_status"):
            self.assertEqual(getattr(after, name), getattr(before, name), name)
        self.assertEqual(before.persistent_conflict_incomplete_steps, 1)
        self.assertEqual(before.no_progress_incomplete_steps, 1)
        # live streaks continue across the rearm, not restarted
        r5 = m.observe_step(5, start, conflicts, **kw)
        self.assertEqual(r5.current_persistent_conflict_streak, before.current_persistent_conflict_streak + 1)
        self.assertEqual(dict(r5.current_no_progress_streaks)[0],
                         dict(before.current_no_progress_streaks)[0] + 1)


class TestRearmGuards(unittest.TestCase):
    """SYNTHETIC: invalid calls raise and leave the Monitor untouched."""

    def _state(self, m):
        return (m.result, m.trigger_history, m.rearm_count, m.armed_since, dict(m._seen))

    def _monitor_after_trigger(self):
        m = _fresh()
        m.observe_step(1, B, NONE)
        m.observe_step(2, B, NONE)
        return m

    def test_before_reset(self):
        with self.assertRaises(RuntimeError):
            DeadlockMonitor().rearm(A, 0)

    def test_wrong_timesteps(self):
        m = self._monitor_after_trigger()
        snap = self._state(m)
        for bad in (1, 0, -1, 3, 99):          # backwards and ahead of the last observation
            with self.assertRaises(ValueError):
                m.rearm(B, bad)
        for bad in (2.0, True, None, "2"):
            with self.assertRaises(TypeError):
                m.rearm(B, bad)
        self.assertEqual(self._state(m), snap)

    def test_wrong_positions_or_agents(self):
        m = self._monitor_after_trigger()
        snap = self._state(m)
        with self.assertRaises(ValueError):
            m.rearm(A, 2)                        # not the state observed at t=2
        with self.assertRaises(ValueError):
            m.rearm({0: (0, 1)}, 2)              # agent missing
        with self.assertRaises(ValueError):
            m.rearm({0: (0, 1), 1: (5, 1), 2: (9, 9)}, 2)
        self.assertEqual(self._state(m), snap)

    def test_list_positions_accepted(self):
        m = self._monitor_after_trigger()
        m.rearm({0: [0, 1], 1: [5, 1]}, 2)
        self.assertEqual(m.observe_step(3, B, NONE).first_seen, 2)

    def test_observe_timestep_must_advance_after_rearm(self):
        m = self._monitor_after_trigger()
        m.rearm(B, 2)
        snap = self._state(m)
        for bad in (2, 1):
            with self.assertRaises(ValueError):
                m.observe_step(bad, C, NONE)
        self.assertEqual(self._state(m), snap)
        self.assertEqual(m.observe_step(3, C, NONE).trigger_type, "NONE")

    def test_rearm_right_after_reset(self):
        m = _fresh()
        m.rearm(A, 0)
        self.assertEqual((m.armed_since, m.rearm_count, m.trigger_history), (0, 1, ()))
        r = m.observe_step(1, A, NONE)
        self.assertEqual((r.trigger_timestep, r.first_seen, r.period), (1, 0, 1))


class TestRearmGoalAndRepeats(unittest.TestCase):

    def test_rearm_at_goal_state_not_registered(self):
        goal = {0: (0, 9), 1: (5, 9)}
        m = _fresh()
        m.observe_step(1, goal, NONE)
        m.rearm(goal, 1)
        self.assertEqual(m._seen, {})
        for t in (2, 3):
            self.assertEqual(m.observe_step(t, goal, NONE).trigger_type, "NONE")
        m.observe_step(4, B, NONE)
        r = m.observe_step(5, B, NONE)
        self.assertEqual((r.trigger_timestep, r.first_seen), (5, 4))

    def test_repeated_rearm_without_trigger_keeps_history(self):
        m = _fresh()
        m.observe_step(1, B, NONE)
        m.rearm(B, 1)
        m.rearm(B, 1)                              # same t, same state, nothing triggered
        self.assertEqual((m.rearm_count, m.trigger_history, m.armed_since), (2, (), 1))
        self.assertEqual(m._seen, {((0, 1), (5, 1)): 1})

    def test_history_accumulates_in_order(self):
        m = _fresh()
        m.observe_step(1, B, NONE)
        m.observe_step(2, B, NONE)                 # trigger #1 at 2
        m.rearm(B, 2)
        m.observe_step(3, C, NONE)
        m.observe_step(4, D, NONE)
        m.observe_step(5, C, NONE)                 # trigger #2 at 5, period 2
        m.observe_step(6, D, NONE)                 # still latched
        m.rearm(D, 6)
        hist = m.trigger_history
        self.assertEqual([(h.trigger_timestep, h.first_seen, h.period, h.armed_since, h.rearmed_at)
                          for h in hist], [(2, 1, 1, 0, 2), (5, 3, 2, 2, 6)])
        self.assertEqual(m.result.trigger_type, "NONE")

    def test_reset_clears_rearm_state(self):
        m = _fresh()
        m.observe_step(1, B, NONE)
        m.observe_step(2, B, NONE)
        m.rearm(B, 2)
        m.reset(A, GOALS)
        self.assertEqual((m.rearm_count, m.trigger_history, m.armed_since), (0, (), 0))
        m.observe_step(1, B, NONE)                 # timestep restarts are legal again after reset
        self.assertEqual(m.observe_step(2, B, NONE).trigger_timestep, 2)


class TestNoRearmBehaviourUnchanged(unittest.TestCase):

    def test_normal_motion_snapshots_identical(self):
        path = [A, B, C, D]
        plain, rearmed = _fresh(), _fresh()
        for t in range(1, 4):
            p = plain.observe_step(t, path[t], NONE)
            q = rearmed.observe_step(t, path[t], NONE)
            self.assertEqual(p, q)
            if t == 2:
                rearmed.rearm(path[t], t)        # rearm while nothing has triggered
        self.assertEqual(plain.result, rearmed.result)
        self.assertEqual(plain.result.trigger_type, "NONE")
        self.assertEqual((plain.rearm_count, plain.trigger_history, plain.armed_since), (0, (), 0))

    def test_without_rearm_timesteps_are_not_policed(self):
        # Pre-existing contract: observe_step never validated timestep order.
        m = _fresh()
        m.observe_step(3, B, NONE)
        m.observe_step(1, C, NONE)
        self.assertEqual(m.result.trigger_type, "NONE")

    def test_monitor_csv_schema_unchanged(self):
        m = DeadlockMonitor()
        m.reset(H3_TRAJECTORY[0], H3_GOALS)
        _replay(m, H3_TRAJECTORY, _h3_conflicts, until=9, rearm_at=H3_RESUME_T)
        with tempfile.TemporaryDirectory() as tmp:
            path = EpisodeLogger("rearm_csv", scenario="synthetic", mode="TEST", out_dir=tmp).write_monitor(m.result)
            with Path(path).open(encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
                header = rows and list(rows[0].keys())
        self.assertEqual(header, list(MONITOR_FIELDS))
        self.assertEqual((rows[0]["trigger_timestep"], rows[0]["first_seen"], rows[0]["period"]), ("7", "6", "1"))

    def test_result_snapshot_is_independent(self):
        m = _fresh()
        m.observe_step(1, B, NONE)
        r2 = m.observe_step(2, B, NONE)
        kept = replace(r2)
        m.rearm(B, 2)
        self.assertEqual(r2, kept)                 # an earlier snapshot is not mutated by rearm()


if __name__ == "__main__":
    unittest.main()
