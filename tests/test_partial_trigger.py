"""Partial Trigger candidate detector (src/hybrid/partial_trigger.py) -- SHADOW ONLY.

Synthetic Monitor results test the rule itself; the Runner tests use the REAL, unchanged
Global Hybrid Runner + MAPFStepSimulator + DeadlockMonitor with a scripted policy and a
Mock planner (no CBS, no IL). Nothing here shows a real-IL / real-solver result."""

import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from scripts.run_selective_mock_dev import ScriptedGlobalPlanner, hybrid_config
from src.common.deadlock_monitor import DeadlockMonitor
from src.hybrid.cbs_runtime_bridge import PathGuardValidator
from src.hybrid.global_hybrid import run_global_hybrid_episode
from src.hybrid.partial_trigger import (
    FULL_REPEAT, INSUFFICIENT_DATA, NONE, PARTIAL_CANDIDATE,
    PartialTriggerConfig, PartialTriggerDetector, PartialTriggerShadow,
)
from src.hybrid.selective_shadow import ShadowMonitor
from tests.hybrid_mocks import DOWN, RIGHT, UP, WAIT, MockReplanResult, ScriptedPolicy

K2 = PartialTriggerConfig(2, 2)
GOALS = {0: (2, 1), 1: (0, 1), 2: (1, 6)}
POS = {0: (0, 1), 1: (2, 1), 2: (1, 4)}
KEY01 = ("vertex", (1, 1), frozenset({0, 1}))


def res(key=KEY01, streak=2, np_=((0, 2), (1, 2)), pc="OK", npst="OK", pc_inc=0, np_inc=0,
        code="TRIGGER_NONE", tt=None):
    return SimpleNamespace(trigger_reason_code=code, trigger_timestep=tt,
                           persistent_conflict_status=pc, no_progress_status=npst,
                           persistent_conflict_incomplete_steps=pc_inc, no_progress_incomplete_steps=np_inc,
                           current_persistent_conflict_key=key, current_persistent_conflict_streak=streak,
                           current_no_progress_streaks=tuple(np_))


CT = {0: "vertex", 1: "vertex", 2: "none"}


def det(cfg=K2):
    d = PartialTriggerDetector(cfg)
    d.reset(GOALS)
    return d


class TestConfig(unittest.TestCase):
    def test_thresholds_are_explicit(self):
        for bad in ((0, 2), (2, 0), (True, 2), (2.0, 2), (None, 2)):
            with self.subTest(bad), self.assertRaises(ValueError):
                PartialTriggerConfig(*bad)
        with self.assertRaises(ValueError):
            PartialTriggerDetector(object())


class TestRule(unittest.TestCase):
    def test_candidate_and_fields(self):
        o = det().observe(5, res(), POS, CT)
        self.assertEqual((o.status, o.group, o.key_kind, o.key_cells), (PARTIAL_CANDIDATE, (0, 1), "vertex", ((1, 1),)))
        self.assertTrue(o.new_event and o.group_smaller_than_all)
        self.assertEqual(o.no_progress, ((0, 2), (1, 2)))

    def test_full_repeat_has_priority(self):
        o = det().observe(5, res(code="TRIGGER_FULL_REPEAT", tt=5), POS, CT)
        self.assertEqual(o.status, FULL_REPEAT)
        # a settled (old) FULL does not mask the partial evaluation
        self.assertEqual(det().observe(6, res(code="TRIGGER_FULL_REPEAT", tt=5), POS, CT).status, PARTIAL_CANDIDATE)

    def test_insufficient_data_holds(self):
        self.assertEqual(det().observe(5, res(pc="NOT_COMPUTED"), POS, CT).status, INSUFFICIENT_DATA)
        self.assertEqual(det().observe(5, res(npst="NOT_COMPUTED"), POS, CT).status, INSUFFICIENT_DATA)
        d = det()
        self.assertEqual(d.observe(5, res(np_inc=1), POS, CT).status, INSUFFICIENT_DATA)
        self.assertEqual(d.observe(6, res(np_inc=1), POS, CT).status, PARTIAL_CANDIDATE)  # counter did not grow
        self.assertEqual(det().observe(5, res(pc_inc=1), POS, CT).reason, "SIGNAL_INCOMPLETE_THIS_STEP")
        d = PartialTriggerDetector(K2)                                      # no goals (no reset)
        self.assertEqual(d.observe(5, res(), POS, CT).status, INSUFFICIENT_DATA)

    def test_no_false_alarm_on_wait_transient_progress_or_goal(self):
        self.assertEqual(det().observe(5, res(key=None, streak=0), POS, CT).reason, "NO_CONFLICT_KEY")   # plain WAIT
        self.assertEqual(det().observe(5, res(streak=1), POS, CT).status, NONE)                         # transient
        o = det().observe(5, res(np_=((0, 2), (1, 1))), POS, CT)                                       # 1 progressing
        self.assertEqual((o.status, o.reason), (NONE, "KEY_AGENTS_PROGRESSING:[1]"))
        at_goal = {0: (2, 1), 1: (0, 1), 2: (1, 4)}
        self.assertEqual(det().observe(5, res(np_=()), at_goal, CT).reason, "ALL_KEY_AGENTS_AT_GOAL")
        one = ("vertex", (1, 1), frozenset({0}))
        self.assertEqual(det().observe(5, res(key=one), POS, CT).reason, "KEY_HAS_FEWER_THAN_2_AGENTS")

    def test_parked_occupant_in_key_is_not_required_to_stall(self):
        key = ("vertex", (1, 6), frozenset({1, 2}))
        pos = {0: (0, 0), 1: (1, 5), 2: (1, 6)}                             # 2 parked at its goal
        o = det().observe(5, res(key=key, np_=((1, 3),)), pos, {0: "none", 1: "vertex", 2: "none"})
        self.assertEqual((o.status, o.parked_in_group), (PARTIAL_CANDIDATE, (2,)))

    def test_unrelated_agents_ignored_and_other_conflicts_only_listed(self):
        o = det().observe(5, res(), POS, {0: "vertex", 1: "vertex", 2: "edge"})
        self.assertEqual((o.status, o.group, o.conflicted_outside_key), (PARTIAL_CANDIDATE, (0, 1), (2,)))

    def test_dedup_and_repeat_and_rearm(self):
        d = det()
        seq = [d.observe(t, res(streak=t, np_=((0, t), (1, t))), POS, CT) for t in (2, 3, 4)]
        self.assertEqual([o.new_event for o in seq], [True, False, False])
        d.observe(5, res(key=None, streak=0), POS, CT)                       # episode ends
        again = d.observe(7, res(streak=2), POS, CT)
        self.assertTrue(again.new_event and again.repeat_since_rearm)
        self.assertEqual(len(d.events), 2)
        d.on_rearm(10)
        old = d.observe(11, res(streak=5, np_=((0, 5), (1, 5))), POS, CT)    # evidence older than rearm
        self.assertEqual((old.status, old.reason), (NONE, "EVIDENCE_BEFORE_REARM_AT_10"))
        fresh = d.observe(12, res(streak=2), POS, CT)
        self.assertTrue(fresh.new_event and not fresh.repeat_since_rearm)

    def test_edge_key_cells(self):
        key = ("edge", frozenset({(0, 1), (0, 2)}), frozenset({0, 1}))
        o = det().observe(3, res(key=key), POS, {0: "edge", 1: "edge", 2: "none"})
        self.assertEqual((o.key_kind, o.key_cells), ("edge", ((0, 1), (0, 2))))


# -- real Runner / Simulator / DeadlockMonitor, scripted policy, Mock planner -------------------
GRID = np.zeros((3, 7), dtype=np.int64)
STARTS = {0: (0, 1), 1: (2, 1), 2: (1, 3)}
# 0 and 1 contest (1,1) forever; 2 walks right to its goal (1,6) and stops.
ACTIONS = {0: [DOWN] * 30, 1: [UP] * 30, 2: [RIGHT, RIGHT, RIGHT] + [WAIT] * 30}


def episode(monitor, planner=None):
    planner = planner or ScriptedGlobalPlanner(MockReplanResult("no_solution", None, 0.1, "mock"))
    return run_global_hybrid_episode(ScriptedPolicy({a: list(v) for a, v in ACTIONS.items()}), GRID, STARTS, GOALS,
                                     20, planner=planner, path_validator=PathGuardValidator(),
                                     config=hybrid_config(4), monitor=monitor)


class TestShadowOnRealRunner(unittest.TestCase):
    def test_runner_result_identical_and_candidate_before_full(self):
        base = episode(DeadlockMonitor())
        dets = {f"K{k}": PartialTriggerDetector(PartialTriggerConfig(k, k)) for k in (2, 3, 5)}
        sh = PartialTriggerShadow(DeadlockMonitor(), dets)
        self.assertEqual(episode(sh), base)                                  # Runner result unchanged
        self.assertEqual(sh.record_errors, [])
        full_t = next(x["t"] for x in base["transitions"] if x["to"] == "TRIGGER")
        self.assertEqual(full_t, 4)                                         # FULL waits for agent 2 to stop
        first = {k: d.events[0].t for k, d in dets.items() if d.events}
        self.assertEqual(first, {"K2": 2, "K3": 3})                         # K5 never reached before stop
        e = dets["K2"].events[0]
        self.assertEqual((e.group, e.group_smaller_than_all), ((0, 1), True))
        moving = sh.steps[e.t - 1]
        self.assertNotEqual(moving["s_t_minus_1"][2], moving["s_t"][2])      # agent 2 moving at candidate time
        self.assertEqual(len(dets["K2"].events), 1)                         # one event, no duplicates

    def test_wrapped_inside_shadow_monitor_is_transparent(self):
        base = episode(ShadowMonitor(DeadlockMonitor()))
        sh = PartialTriggerShadow(DeadlockMonitor(), {"K2": PartialTriggerDetector(K2)})
        self.assertEqual(episode(ShadowMonitor(sh)), base)

    def test_recording_failure_never_changes_monitor_path(self):
        class Broken(PartialTriggerDetector):
            def observe(self, *a, **k):
                raise RuntimeError("boom")
        base = episode(DeadlockMonitor())
        sh = PartialTriggerShadow(DeadlockMonitor(), {"X": Broken(K2)})
        self.assertEqual(episode(sh), base)
        self.assertTrue(sh.record_errors)

    def test_rearm_is_forwarded_to_inner_monitor_and_detectors(self):
        plain = DeadlockMonitor()
        d = PartialTriggerDetector(K2)
        sh = PartialTriggerShadow(DeadlockMonitor(), {"K2": d})
        pos = {0: (0, 1), 1: (2, 1)}
        goals = {0: (2, 1), 1: (0, 1)}
        for m in (plain, sh):
            m.reset(dict(pos), goals)
            for t in (1, 2):
                m.observe_step(t, dict(pos), {0: "vertex", 1: "vertex"}, current_positions=dict(pos),
                               intended_positions={0: (1, 1), 1: (1, 1)},
                               goal_distance_before={0: 2, 1: 2}, goal_distance_after={0: 2, 1: 2})
            m.rearm(dict(pos), 2)
        self.assertEqual(len(sh.inner.trigger_history), 1)                    # archived by the INNER monitor
        self.assertEqual(sh.inner.trigger_history, plain.trigger_history)
        self.assertEqual(sh.inner.result, plain.result)
        self.assertEqual(d._rearm_t, 2)

    def test_default_runtime_does_not_import_partial_trigger(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        for rel in ("src/hybrid/global_hybrid.py", "src/common/deadlock_monitor.py", "src/hybrid/selective_planner.py",
                    "src/compare/common_runner.py"):
            self.assertNotIn("partial_trigger", (root / rel).read_text(encoding="utf-8"), rel)


if __name__ == "__main__":
    unittest.main()
