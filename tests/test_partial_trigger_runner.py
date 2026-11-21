"""Partial Trigger Runner opt-in (stage 2) on the REAL, unchanged-by-default Global Hybrid Runner,
MAPFStepSimulator, DeadlockMonitor, ShadowMonitor, SelectivePlanner and validators.

MOCK here: the policy (scripted actions) and every CBS answer (ScriptedSubsetPlanner /
ScriptedGlobalPlanner, ``test_only``). Nothing here shows a real-IL or real-solver result.
"""

import unittest
from dataclasses import replace

import numpy as np

from scripts.run_selective_mock_dev import (
    ScriptedGlobalPlanner, ScriptedSubsetPlanner, TEST_BUDGET, hybrid_config, success,
)
from src.common.deadlock_monitor import DeadlockMonitor
from src.hybrid.cbs_runtime_bridge import PathGuardValidator
from src.hybrid.global_hybrid import (
    GlobalHybridConfig, MonitorRearmConfig, run_global_hybrid_episode,
)
from src.hybrid.partial_trigger import (
    PARTIAL_CANDIDATE, PARTIAL_TRIGGER_REASON, PartialObservation, PartialTriggerConfig,
    PartialTriggerDetector, PartialTriggerShadow, PartialTriggerSource,
)
from src.hybrid.selective_planner import SelectiveConfig, SelectivePathValidator, SelectivePlanner
from src.hybrid.selective_shadow import ShadowMonitor
from tests.hybrid_mocks import DOWN, RIGHT, UP, WAIT, MockReplanResult, ScriptedPolicy

# 0 / 1 contest (1,1) forever; 2 walks right from (1,3) towards (1,6), 3 moves, then waits.
GRID = np.zeros((3, 7), dtype=np.int64)
STARTS = {0: (0, 1), 1: (2, 1), 2: (1, 3)}
GOALS = {0: (2, 1), 1: (0, 1), 2: (1, 6)}
ACTIONS = {0: [DOWN] * 40, 1: [UP] * 40, 2: [RIGHT, RIGHT, RIGHT] + [WAIT] * 40}
# a valid group plan for {0, 1} from S_2 (2 is held at (1,5)); 4 moves
GROUP_PATHS = {0: [(0, 1), (0, 2), (1, 2), (2, 2), (2, 1)],
               1: [(2, 1), (1, 1), (0, 1), (0, 1), (0, 1)]}
OPT_IN = SelectiveConfig(ladder=("L0", "L1", "G"), budget=TEST_BUDGET, acknowledge_unverified_outside_policy=True)
REARM = MonitorRearmConfig(max_cbs_calls=2, cooldown_steps=0, cbs_time_budget_ms=10_000.0)


def rig(subset_results, *, k=2, rearm=REARM, h_exec=4, glob=None, allow=True, config=OPT_IN):
    det = PartialTriggerDetector(PartialTriggerConfig(k, k))
    mon = ShadowMonitor(PartialTriggerShadow(DeadlockMonitor(), {"K": det}))
    g = glob or ScriptedGlobalPlanner(MockReplanResult("no_solution", None, 0.1, "mock"))
    s = ScriptedSubsetPlanner(*subset_results)
    planner = SelectivePlanner(g, mon, config, subset_planner=s)
    src = PartialTriggerSource(monitor=mon, planner=planner, detector_name="K", allow_partial_trigger=allow)
    cfg = hybrid_config(h_exec, rearm)
    return dict(det=det, mon=mon, planner=planner, src=src, cfg=cfg, s=s, g=g,
                validator=SelectivePathValidator(planner))


def run(r, actions=ACTIONS, max_steps=30, grid=GRID, starts=STARTS, goals=GOALS, extra="src"):
    return run_global_hybrid_episode(ScriptedPolicy({a: list(v) for a, v in actions.items()}), grid, starts, goals,
                                     max_steps, planner=r["planner"], path_validator=r["validator"],
                                     config=r["cfg"], monitor=r["mon"],
                                     extra_trigger=r["src"] if extra == "src" else extra)


def triggers(res):
    return [(x["t"], x["reason"]) for x in res["transitions"] if x["to"] == "TRIGGER"]


class TestDefaultUnchanged(unittest.TestCase):
    def test_no_extra_trigger_means_full_only(self):
        mon = ShadowMonitor(DeadlockMonitor())
        planner = SelectivePlanner(ScriptedGlobalPlanner(MockReplanResult("no_solution", None, 0.1, "m")), mon,
                                   OPT_IN, subset_planner=ScriptedSubsetPlanner(success(GROUP_PATHS, 0.1)))
        res = run_global_hybrid_episode(ScriptedPolicy({a: list(v) for a, v in ACTIONS.items()}), GRID, STARTS,
                                        GOALS, 30, planner=planner, path_validator=SelectivePathValidator(planner),
                                        config=hybrid_config(4, REARM), monitor=mon)
        self.assertEqual(triggers(res)[0], (4, "TRIGGER_FULL_REPEAT"))      # FULL waits for agent 2
        self.assertTrue(all("reason" not in x for x in res["triggers"]))    # record shape unchanged
        import dataclasses
        self.assertNotIn("extra_trigger", [f.name for f in dataclasses.fields(GlobalHybridConfig)])


class TestPartialDrivesRunner(unittest.TestCase):
    def test_partial_trigger_executes_selective_plan_and_outside_resumes(self):
        r = rig([success(GROUP_PATHS, 0.1)])
        res = run(r)
        self.assertEqual(triggers(res)[0], (2, PARTIAL_TRIGGER_REASON))      # before FULL (t=4)
        (call,) = r["s"].calls
        self.assertEqual(call["current_positions"], {0: (0, 1), 1: (2, 1)})  # S_t, original ids
        self.assertEqual(call["outside"], {2: (1, 5)})                       # moving agent held where it is
        rec = r["planner"].records[0]
        self.assertEqual((rec.trigger_t, rec.wfg.groups, rec.outcome), (2, ((0, 1),), "EXECUTE:L0"))
        self.assertEqual(rec.s_t_minus_1[2], (1, 4))                         # agent 2 moved into S_t
        self.assertEqual(res["triggers"][0]["extra_event"]["group"], (0, 1))
        cbs = [i for i, s in enumerate(res["action_sources"]) if s == "CBS"]
        self.assertEqual(cbs, [2, 3, 4, 5])
        self.assertTrue(all(res["actions"][i][2] == WAIT for i in cbs))
        self.assertTrue(all(e["match"] and e["actual"][2] == (1, 5) for e in res["cbs_exec_trace"]))
        self.assertEqual(res["monitor_rearms"][0]["t"], 6)                   # rearm once at IL_RESUME
        self.assertEqual(res["trajectory"][7][2], (1, 6))                    # 2 moves again under IL
        self.assertEqual(res["termination"], "SUCCESS")
        self.assertEqual(len(r["src"].fired), 1)

    def test_one_event_one_call_no_stale_reuse(self):
        r = rig([MockReplanResult("no_solution", None, 0.1, "m")])          # L0 fails -> G no_solution -> SAFE_STOP
        res = run(r)
        self.assertEqual(triggers(res), [(2, PARTIAL_TRIGGER_REASON)])
        self.assertEqual((len(r["s"].calls), len(r["g"].calls)), (1, 1))
        self.assertEqual(res["stop_reason"], "CBS_NO_SOLUTION")
        self.assertEqual([i for i, s in enumerate(res["action_sources"]) if s == "CBS"], [])

    def test_full_has_priority_on_the_same_step(self):
        # 0/1 only (no moving third agent): FULL at t=1, and a K=1 detector would also fire at t=1
        r = rig([success(GROUP_PATHS, 0.1)], k=1)
        acts = {0: [DOWN] * 40, 1: [UP] * 40, 2: [WAIT] * 40}
        res = run(r, actions=acts)
        self.assertEqual(triggers(res)[0], (1, "TRIGGER_FULL_REPEAT"))
        self.assertEqual(r["src"].fired, [])

    def test_l0_success_without_paths_is_refused_no_move(self):
        r = rig([MockReplanResult("success", None, 0.1)])
        res = run(r)
        self.assertEqual(triggers(res), [(2, PARTIAL_TRIGGER_REASON)])
        self.assertEqual(r["planner"].records[0].outcome, "SAFE_STOP:FULL_CHECK_FAILED")
        self.assertEqual(r["g"].calls, [])                                   # invalid result -> no G, SAFE_STOP
        self.assertEqual((res["termination"], res["stop_reason"]), ("SAFE_STOP", "CBS_INVALID_PATH"))
        self.assertEqual([i for i, s in enumerate(res["action_sources"]) if s == "CBS"], [])

    def test_second_partial_after_resume_needs_k_fresh_steps(self):
        # long corridor: agent 2 keeps moving after the resume, so the re-stall is again PARTIAL (not FULL)
        grid = np.zeros((3, 14), dtype=np.int64)
        goals = {0: (2, 1), 1: (0, 1), 2: (1, 13)}
        acts = {0: [DOWN] * 40, 1: [UP] * 40, 2: [RIGHT] * 10 + [WAIT] * 30}
        hold = {0: [(0, 1), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1)],         # step 1 = both WAIT
                1: [(2, 1), (2, 1), (2, 1), (1, 1), (0, 1), (0, 1)]}
        r = rig([success(hold, 0.1), success(hold, 0.1)], h_exec=1,
                rearm=MonitorRearmConfig(max_cbs_calls=3, cooldown_steps=0, cbs_time_budget_ms=10_000.0))
        res = run(r, actions=acts, grid=grid, goals=goals)
        trig = triggers(res)
        self.assertEqual([reason for _, reason in trig[:2]], [PARTIAL_TRIGGER_REASON] * 2)
        rearms = [x["t"] for x in res["monitor_rearms"]]
        records = [(x["t"], x["reason"], x["outcome"]) for x in res["triggers"]]
        # every handed-out event = exactly one Runner Trigger record at the same step
        self.assertEqual([e.t for e in r["src"].fired], [t for t, _, _ in records])
        self.assertTrue(all(reason == PARTIAL_TRIGGER_REASON for _, reason, _ in records))
        # each re-trigger needs exactly K=2 fresh steps after its rearm
        self.assertEqual([t for t, _, _ in records[1:]], [rt + 2 for rt in rearms[:len(records) - 1]])
        # one CBS call per CBS_REQUEST; the limit (3) refuses the 4th -> SAFE_STOP, no extra call
        self.assertEqual(len(res["cbs_calls"]), sum(1 for *_, o in records if o == "CBS_REQUEST"))
        self.assertEqual(len(res["cbs_calls"]), 3)
        self.assertEqual(records[-1][2], "RECALL_LIMIT_EXCEEDED")
        self.assertEqual((res["termination"], res["stop_reason"]), ("SAFE_STOP", "RECALL_LIMIT_EXCEEDED"))

    def test_call_limit_and_budget_apply_to_partial(self):
        # limit 1: the first (partial) call uses it; the next Trigger -> SAFE_STOP RECALL_LIMIT_EXCEEDED
        good = {0: [(0, 1), (0, 2), (1, 2), (2, 2), (2, 1)], 1: [(2, 1), (1, 1), (0, 1), (0, 1), (0, 1)]}
        r = rig([success(good, 0.1)], h_exec=1,
                rearm=MonitorRearmConfig(max_cbs_calls=1, cooldown_steps=0, cbs_time_budget_ms=10_000.0))
        res = run(r)
        self.assertEqual(len(res["cbs_calls"]), 1)
        self.assertEqual((res["termination"], res["stop_reason"]), ("SAFE_STOP", "RECALL_LIMIT_EXCEEDED"))
        r = rig([success(good, 0.3)], h_exec=1,
                rearm=MonitorRearmConfig(max_cbs_calls=3, cooldown_steps=0, cbs_time_budget_ms=100.0))
        res = run(r)
        self.assertEqual((res["termination"], res["stop_reason"]), ("SAFE_STOP", "CBS_TIME_BUDGET_EXCEEDED"))


class AlwaysEventSource:
    """TEST DOUBLE that misbehaves on purpose: claims an event at every poll."""
    reason = PARTIAL_TRIGGER_REASON

    def __init__(self, planner, monitor):
        self.bound_planner, self.bound_monitor, self.polled = planner, monitor, []

    def reset(self):
        self.polled = []

    def poll(self, t):
        self.polled.append(t)
        return PartialObservation(t=t, status=PARTIAL_CANDIDATE, reason="stub", group=(0, 1), new_event=True)

    @staticmethod
    def describe(obs):
        return {"t": obs.t}


class TestRunnerLevelRules(unittest.TestCase):
    def _run_stub(self, acts, rearm=REARM, max_steps=6):
        mon = ShadowMonitor(DeadlockMonitor())
        planner = SelectivePlanner(ScriptedGlobalPlanner(MockReplanResult("no_solution", None, 0.1, "m")), mon,
                                   OPT_IN, subset_planner=ScriptedSubsetPlanner(
                                       MockReplanResult("no_solution", None, 0.1, "m")))
        stub = AlwaysEventSource(planner, mon)
        res = run_global_hybrid_episode(ScriptedPolicy({a: list(v) for a, v in acts.items()}), GRID, STARTS, GOALS,
                                        max_steps, planner=planner, path_validator=SelectivePathValidator(planner),
                                        config=hybrid_config(4, rearm), monitor=mon, extra_trigger=stub)
        return res, stub

    def test_full_step_never_polls_and_keeps_full_reason(self):
        acts = {0: [DOWN] * 40, 1: [UP] * 40, 2: [WAIT] * 40}               # FULL at t=1
        res, stub = self._run_stub(acts)
        self.assertEqual(triggers(res), [(1, "TRIGGER_FULL_REPEAT")])
        self.assertNotIn(1, stub.polled)

    def test_overlap_integrity_still_blocks_a_partial_event(self):
        from dataclasses import replace as dc_replace

        class IntegrityIssueMonitor:
            """TEST DOUBLE inner monitor: real DeadlockMonitor, overlap status forced to INTEGRITY_ISSUE."""
            def __init__(self):
                self.m = DeadlockMonitor()

            def reset(self, *a):
                return self.m.reset(*a)

            def rearm(self, *a):
                return self.m.rearm(*a)

            def observe_step(self, *a, **k):
                return dc_replace(self.m.observe_step(*a, **k), overlap_integrity_status="INTEGRITY_ISSUE")

            def __getattr__(self, n):
                return getattr(self.m, n)

        mon = ShadowMonitor(IntegrityIssueMonitor())
        sub = ScriptedSubsetPlanner(success(GROUP_PATHS, 0.1))
        planner = SelectivePlanner(ScriptedGlobalPlanner(), mon, OPT_IN, subset_planner=sub)
        stub = AlwaysEventSource(planner, mon)
        res = run_global_hybrid_episode(ScriptedPolicy({a: list(v) for a, v in ACTIONS.items()}), GRID, STARTS,
                                        GOALS, 3, planner=planner, path_validator=SelectivePathValidator(planner),
                                        config=hybrid_config(4, REARM), monitor=mon, extra_trigger=stub)
        self.assertTrue(res["triggers"])
        self.assertTrue(all(x["outcome"] == "OVERLAP_NOT_CLEAN" for x in res["triggers"]))
        self.assertEqual((res["cbs_calls"], sub.calls), ([], []))

    def test_poll_only_in_il_and_once_per_step(self):
        res, stub = self._run_stub(ACTIONS)                                  # stub fires at t=1 -> CBS fails -> stop
        self.assertEqual(triggers(res), [(1, PARTIAL_TRIGGER_REASON)])
        self.assertEqual(stub.polled, [1])
        self.assertEqual(res["stop_reason"], "CBS_NO_SOLUTION")


class TestRefusals(unittest.TestCase):
    def test_contract_refusals(self):
        with self.assertRaisesRegex(ValueError, "allow_partial_trigger"):
            rig([], allow=False)
        with self.assertRaisesRegex(ValueError, "opt-in"):
            rig([], config=SelectiveConfig())
        r = rig([success(GROUP_PATHS, 0.1)], rearm=None)
        with self.assertRaisesRegex(ValueError, "rearm"):
            run(r)
        self.assertEqual(r["s"].calls, [])
        r = rig([success(GROUP_PATHS, 0.1)])
        for bad in (object(), AlwaysEventSource(r["planner"], r["mon"])):
            if isinstance(bad, AlwaysEventSource):
                bad.reason = "TRIGGER_FULL_REPEAT"                           # must not impersonate FULL
            with self.assertRaises(ValueError):
                run(r, extra=bad)
        det = PartialTriggerDetector(PartialTriggerConfig(2, 2))
        plain = ShadowMonitor(DeadlockMonitor())
        p = SelectivePlanner(ScriptedGlobalPlanner(), plain, OPT_IN, subset_planner=ScriptedSubsetPlanner())
        with self.assertRaisesRegex(ValueError, "PartialTriggerShadow"):
            PartialTriggerSource(monitor=plain, planner=p, detector_name="K", allow_partial_trigger=True)
        other = ShadowMonitor(PartialTriggerShadow(DeadlockMonitor(), {"K": det}))
        with self.assertRaisesRegex(ValueError, "own ShadowMonitor"):
            PartialTriggerSource(monitor=other, planner=p, detector_name="K", allow_partial_trigger=True)
        with self.assertRaisesRegex(ValueError, "no detector"):
            PartialTriggerSource(monitor=r["mon"], planner=r["planner"], detector_name="X", allow_partial_trigger=True)

    def test_runner_refuses_source_bound_to_another_run(self):
        a, b = rig([success(GROUP_PATHS, 0.1)]), rig([success(GROUP_PATHS, 0.1)])
        with self.assertRaisesRegex(ValueError, "bound to this run"):
            run(a, extra=b["src"])

    def test_poll_rejects_stale_reused_insufficient(self):
        r = rig([success(GROUP_PATHS, 0.1)])
        src, det, mon = r["src"], r["det"], r["mon"]
        mon.reset(dict(STARTS), dict(GOALS))
        ev = PartialObservation(t=3, status=PARTIAL_CANDIDATE, reason="CANDIDATE", group=(0, 1), new_event=True)
        det.observations.append(ev)
        self.assertIsNone(src.poll(3))                                       # monitor has no step 3 -> mismatch
        self.assertEqual(src.refused[-1], (3, "MONITOR_STEP_MISMATCH"))
        mon.observe_step(1, dict(STARTS), {0: "none", 1: "none", 2: "none"}, current_positions=dict(STARTS),
                         intended_positions=dict(STARTS), goal_distance_before={0: 2, 1: 2, 2: 3},
                         goal_distance_after={0: 2, 1: 2, 2: 3})
        self.assertIsNone(src.poll(2))                                       # stale: last obs is t=1
        det.observations.append(replace(ev, t=1))
        self.assertIs(src.poll(1), det.observations[-1])
        self.assertIsNone(src.poll(1))                                       # never handed out twice
        self.assertEqual(src.refused[-1], (1, "ALREADY_HANDED_OUT"))
        # detector missed step 2 (e.g. a recording error) while the Monitor did observe it:
        # the older t=1 event must not be handed out at t=2
        detectors, mon.inner.detectors = mon.inner.detectors, {}
        mon.observe_step(2, dict(STARTS), {0: "none", 1: "none", 2: "none"}, current_positions=dict(STARTS),
                         intended_positions=dict(STARTS), goal_distance_before={0: 2, 1: 2, 2: 3},
                         goal_distance_after={0: 2, 1: 2, 2: 3})
        mon.inner.detectors = detectors
        det.observations.append(replace(ev, t=1))
        self.assertEqual(mon.last_step.t, 2)
        self.assertIsNone(src.poll(2))
        self.assertEqual(src.refused[-1], (2, "NO_OBSERVATION_AT_T"))
        det.observations.append(replace(ev, t=2, new_event=False))
        src.reset()
        self.assertIsNone(src.poll(2))                                       # not a new event
        det.observations.append(PartialObservation(t=2, status="INSUFFICIENT_DATA", reason="x"))
        self.assertIsNone(src.poll(2))                                       # incomplete data never triggers


if __name__ == "__main__":
    unittest.main()
