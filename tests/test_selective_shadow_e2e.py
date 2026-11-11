"""Selective SHADOW opt-in E2E on the real Global Hybrid simulation (stage A).

Real pieces: ``run_global_hybrid_episode``, MAPFStepSimulator, DeadlockMonitor,
NavHintPolicy (DEFAULT_CKPT), the DEV scenarios (Corridor n2, WH8-A), the
DEV configs (``dev_config`` / ``dev_resume_config`` / ``dev_rearm_config``),
CbsRuntimePlanner, run_cbs_runtime's own Path Guard re-check, the runner's
boundary + PathGuardValidator, EpisodeLogger and ``common_runner.run_method``.

Mock: ONLY ``src.cbs_runtime_adapter.replan_with_status`` (the atb033 call),
replaced exactly as tests/test_common_runner.py does (recorded atb033 plans or
a fixed ReplanResult). The fake solver root's cbs.py exits if it is ever run.
No real CBS solver runs. Every file goes to a TemporaryDirectory.

Each check runs a Shadow OFF episode and a Shadow ON episode on fresh objects
with identical inputs and asserts the Global outcome is identical: result
dict, solver inputs, planner return objects, recorded latency, Monitor state,
step/summary/monitor CSV bytes. Shadow wall-clock time is never compared.

Values marked TEST-ONLY (h_exec=1 rearm variants) are not project policy.
A partial-group shadow record is a CANDIDATE ONLY: v0 always routes GLOBAL,
Selective is never executed, and the Global planner still gets every agent.
"""

import tempfile
import unittest
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import numpy as np

from scripts.run_corridor_hybrid_e2e import dev_config
from scripts.run_hybrid_resume_dev import dev_rearm_config, dev_resume_config
from scripts.run_il import DEFAULT_CKPT, PROJECT_ROOT, load_scenario
from src.common import spec
from src.common.deadlock_monitor import DeadlockMonitor
from src.common.episode_logger import EpisodeLogger
from src.compare.common_runner import (
    GLOBAL_HYBRID_DEV_FULL,
    GLOBAL_HYBRID_DEV_H3,
    GLOBAL_HYBRID_DEV_H3_REARM,
    RunSpec,
    _planner_calls,
    run_method,
)
from src.hybrid import cbs_runtime_bridge
from src.hybrid.cbs_runtime_bridge import CbsRuntimePlanner, PathGuardValidator
from src.hybrid.global_hybrid import run_global_hybrid_episode
from src.hybrid.replan_boundary import accept_replan_result, check_validator_pairing
from src.hybrid.selective_decision import (
    PENDING_BLOCKERS,
    ROUTE_GLOBAL,
    SELECTIVE_CANDIDATE_BLOCKED,
    decide_selective,
)
from src.hybrid.selective_safety import l0_union_group
from src.hybrid.selective_shadow import ShadowMonitor, ShadowSelectivePlanner
from src.hybrid.wfg import build_wfg
from src.replan_status import ReplanResult
from tests.test_common_runner import CORRIDOR, _bad_start_solver, _timeout_solver, recorded_solver

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

WH8A = PROJECT_ROOT / "scenarios" / "warehouse_v3_dev" / "scenario_wh8a_9x8.json"
MAX_STEPS = 160          # common_runner.DEFAULT_MAX_STEPS
RUN_ID = "shadow_e2e"    # same run_id/mode for OFF and ON so CSV bytes are comparable
_REAL_RUN_CBS_RUNTIME = cbs_runtime_bridge.run_cbs_runtime


def _status_solver(status, latency_sec):
    """Stand-in for replan_with_status returning a fixed non-success status."""
    def solver(**kwargs):
        return ReplanResult(status=status, paths=None, latency_sec=latency_sec, error=f"mock {status}")
    return solver


def _rearm_two_calls():
    """TEST-ONLY: h_exec=1, no cooldown -> real IL re-stalls, second CBS call."""
    return dev_rearm_config(1, max_cbs_calls=3, cooldown_steps=0)


def _rearm_cooldown():
    """TEST-ONLY: h_exec=1, cooldown 2 -> the re-Trigger is refused, no second call."""
    return dev_rearm_config(1, max_cbs_calls=3, cooldown_steps=2)


@dataclass
class Run:
    result: Dict[str, Any]
    inner: CbsRuntimePlanner
    planner: Any                 # what the runner got (inner, or the shadow wrapper)
    monitor: Any
    solver_calls: List[Dict[str, Any]]
    returned: List[Any]          # objects run_cbs_runtime returned (= what inner returned)
    received: List[Any]          # result objects the runner handed to its boundary
    rows: List[Dict[str, Any]]
    csv: Dict[str, bytes]
    grid: Any = None
    goals: Optional[Dict[int, tuple]] = None
    inner_invocations: List[int] = field(default_factory=list)


class _Env(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.solver_root = self.tmp / "solver"
        self.solver_root.mkdir()
        (self.solver_root / "cbs.py").write_text("raise SystemExit('fake solver must not run')\n",
                                                 encoding="utf-8")
        self._n = 0

    def tearDown(self):
        self._tmp.cleanup()

    def run_ep(self, scenario, config, *, shadow, solver=recorded_solver, bridge_raises=None,
               wrapper=None, mode="GLOBAL_HYBRID_SHADOW_E2E", run_id=RUN_ID, out_dir=None):
        """One real episode. shadow=True wraps fresh objects unless ``wrapper``
        (an existing ShadowSelectivePlanner, for multi-episode tests) is given."""
        self._n += 1
        grid, starts, goals = load_scenario(scenario)
        run_dir = out_dir if out_dir is not None else self.tmp / f"run_{self._n:02d}"
        logger = EpisodeLogger(run_id, scenario=scenario.name, mode=mode, out_dir=run_dir)
        if wrapper is not None:
            inner, mon, planner = wrapper.inner, wrapper.monitor, wrapper
        else:
            inner = CbsRuntimePlanner(solver_root=self.solver_root, work_root=run_dir / "cbs")
            mon = DeadlockMonitor()
            planner = inner
            if shadow:
                mon = ShadowMonitor(mon)
                planner = ShadowSelectivePlanner(inner, mon)

        solver_calls, returned, received, invocations = [], [], [], []

        def solver_spy(**kwargs):
            solver_calls.append({
                "current_positions": [tuple(p) for p in kwargs["current_positions"]],
                "goals": [tuple(g) for g in kwargs["goals"]],
                "grid_map": np.array(kwargs["grid_map"], copy=True),
                "solver_root": kwargs["solver_root"],
                "timeout_sec": kwargs["timeout_sec"],
            })
            return solver(**kwargs)

        def bridge(*args, **kwargs):
            invocations.append(1)
            if bridge_raises is not None:
                raise bridge_raises
            out = _REAL_RUN_CBS_RUNTIME(*args, **kwargs)
            returned.append(out)
            return out

        def boundary(result, **kwargs):
            received.append(result)
            return accept_replan_result(result, **kwargs)

        from src.il.policy import NavHintPolicy
        with patch("src.cbs_runtime_adapter.replan_with_status", side_effect=solver_spy), \
                patch("src.hybrid.cbs_runtime_bridge.run_cbs_runtime", side_effect=bridge), \
                patch("src.hybrid.global_hybrid.accept_replan_result", side_effect=boundary):
            result = run_global_hybrid_episode(
                NavHintPolicy(DEFAULT_CKPT), grid, starts, goals, MAX_STEPS,
                planner=planner, path_validator=PathGuardValidator(), config=config,
                monitor=mon, logger=logger)

        # Same three writes as common_runner.run_method, into the temp run dir.
        paths = {
            "steps": logger.write_steps(),
            "summary": logger.write_summary(num_agents=len(goals), steps=result["steps"],
                                            success=result["all_at_goal"],
                                            timed_out=result["timed_out"], makespan=result["steps"]),
            "monitor": logger.write_monitor(mon.result),
        }
        for p in paths.values():
            self.assertTrue(Path(p).resolve().is_relative_to(self.tmp.resolve()))
        return Run(result=result, inner=inner, planner=planner, monitor=mon,
                   solver_calls=solver_calls, returned=returned, received=received,
                   rows=logger.step_rows, csv={k: Path(p).read_bytes() for k, p in paths.items()},
                   grid=grid, goals=goals, inner_invocations=invocations)

    # -- assertions --------------------------------------------------------------

    def assert_same_global(self, off: Run, on: Run):
        """Shadow ON must not change a single Global observable."""
        for key in ("trajectory", "actions", "action_sources", "transitions", "cbs_calls",
                    "cbs_exec_trace", "exec_mismatch", "termination", "stop_reason", "safe_stop",
                    "final_positions", "steps", "all_at_goal", "timed_out", "integrity_abort"):
            self.assertEqual(on.result[key], off.result[key], key)
        for key in ("rearm_config", "monitor_rearms", "triggers", "monitor_trigger_history"):
            self.assertEqual(key in on.result, key in off.result, key)
            if key in off.result:
                self.assertEqual(on.result[key], off.result[key], key)
        self.assertEqual(on.result, off.result)                      # whole dict

        # Planner: same number of inner invocations, solver inputs, returns.
        self.assertEqual(len(on.inner_invocations), len(off.inner_invocations))
        self.assertEqual(len(on.solver_calls), len(off.solver_calls))
        for a, b in zip(on.solver_calls, off.solver_calls):
            for k in ("current_positions", "goals", "solver_root", "timeout_sec"):
                self.assertEqual(a[k], b[k], k)
            self.assertTrue(np.array_equal(a["grid_map"], b["grid_map"]))
        self.assertEqual(on.returned, off.returned)                  # status/paths/latency/error
        strip = lambda calls: [{k: v for k, v in c.items() if k not in ("work_dir", "artifacts_dir")}  # noqa: E731
                               for c in calls]
        self.assertEqual(strip(on.inner.calls), strip(off.inner.calls))
        self.assertEqual([{k: v for k, v in c.items() if k != "work_dir"} for c in _planner_calls(on.inner)],
                         [{k: v for k, v in c.items() if k != "work_dir"} for c in _planner_calls(off.inner)])

        # The runner received exactly the object the inner planner returned.
        for run in (off, on):
            self.assertEqual(len(run.received), len(run.returned))
            for got, ret in zip(run.received, run.returned):
                self.assertIs(got, ret)
            # Recorded latency is the planner's own measurement (shadow time excluded).
            self.assertEqual([c["latency_ms"] for c in run.result["cbs_calls"]][:len(run.returned)],
                             [r.latency_sec * 1000.0 for r in run.returned])

        # Monitor state and every CSV byte.
        self.assertEqual(on.monitor.result, off.monitor.result)
        self.assertEqual(on.monitor.trigger_history, off.monitor.trigger_history)
        self.assertEqual(on.monitor.rearm_count, off.monitor.rearm_count)
        self.assertEqual(on.rows, off.rows)
        for name in ("steps", "summary", "monitor"):
            self.assertEqual(on.csv[name], off.csv[name], name)

    def assert_records_match_episode(self, on: Run, *, episode=1):
        """Time axis + WFG + count checks of every shadow record of ``episode``."""
        w, mon, res = on.planner, on.monitor, on.result
        recs = [r for r in w.records if r.episode == episode]
        calls = res["cbs_calls"]
        # One record per Global planner call; the inner planner ran once per call.
        self.assertEqual(len(recs), len(calls))
        if mon.episode == episode:          # inner ran exactly once per Global call
            self.assertEqual(len(on.inner_invocations), len(calls))
        self.assertEqual([r.call_index for r in recs], list(range(1, len(recs) + 1)))
        ids = tuple(sorted(on.goals))
        traj, actions = res["trajectory"], res["actions"]
        solver_inputs_by_call = len(on.solver_calls) == len(calls)
        for k, (rec, call) in enumerate(zip(recs, calls)):
            self.assertIsNone(rec.shadow_error)
            t = rec.trigger_t
            self.assertEqual(t, call["trigger_timestep"])
            # S_{t-1}: the state before step t's move; S_t: after it = CBS start.
            self.assertEqual(rec.s_t_minus_1, traj[t - 1])
            self.assertEqual(rec.s_t, traj[t])
            self.assertEqual(rec.s_t, call["s_t"])
            # (S_{t-1} may equal S_t at a full stall; the indices above are what pin the axis.)
            # intended: step t's proposed move (action t-1 applied to S_{t-1}).
            expected_intent = {}
            for a in ids:
                dr, dc = spec.ACTION_DELTA[actions[t - 1][a]]
                expected_intent[a] = (traj[t - 1][a][0] + dr, traj[t - 1][a][1] + dc)
            self.assertEqual(rec.intended, expected_intent)
            # Independent source: the logger rows of step t.
            rows_t = [r for r in on.rows if r["timestep"] == t]
            self.assertEqual(sorted(r["agent_id"] for r in rows_t), list(ids))
            self.assertEqual({r["agent_id"]: (r["intended_row"], r["intended_col"]) for r in rows_t},
                             rec.intended)
            self.assertEqual({r["agent_id"]: (r["current_row"], r["current_col"]) for r in rows_t},
                             rec.s_t_minus_1)
            self.assertEqual({r["agent_id"]: (r["next_row"], r["next_col"]) for r in rows_t}, rec.s_t)
            self.assertEqual(rec.blocked, tuple(sorted(r["agent_id"] for r in rows_t
                                                       if r["conflict_type"] != "none")))
            # The CBS input actually handed to the solver was S_t (agent order).
            if solver_inputs_by_call:
                self.assertEqual(on.solver_calls[k]["current_positions"], [rec.s_t[a] for a in ids])
                self.assertEqual(on.solver_calls[k]["goals"], [tuple(on.goals[a]) for a in ids])
            self.assertEqual(call["agent_order"], ids)               # Global request = ALL agents
            # WFG recomputed from the recorded inputs; agent ids are runtime ids.
            self.assertEqual(rec.wfg, build_wfg(rec.s_t_minus_1, rec.intended, rec.blocked, on.grid))
            for e in rec.wfg.edges:
                self.assertIn(e.src, ids)
                self.assertIn(e.dst, ids)
            self.assertEqual(rec.decision, decide_selective(ids, rec.s_t, on.goals, rec.wfg))
            self.assertEqual(rec.l0_group, l0_union_group(rec.wfg.groups))
            self.assertEqual(set(rec.l0_group) | set(rec.outside), set(ids))
            self.assertEqual(rec.decision.route, ROUTE_GLOBAL)      # v0 never executes Selective
            # global_status is the GLOBAL planner's status, never a Selective outcome.
            self.assertEqual(rec.global_status, call["status"])
        # Every ShadowMonitor step belongs to this episode and mirrors the trajectory.
        if mon.episode == episode:
            self.assertEqual([s.t for s in mon.steps], list(range(1, res["steps"] + 1)))
            for s in mon.steps:
                self.assertEqual(s.s_t, traj[s.t])
                self.assertEqual(s.s_t_minus_1, traj[s.t - 1])
        return recs


# -- A/B. Shadow ON == OFF on the DEV configurations, records exact -------------------

@unittest.skipIf(torch is None, "torch is not installed")
class TestCorridorDevConfigs(_Env):

    def _pair(self, make_config, **kw):
        off = self.run_ep(CORRIDOR, make_config(), shadow=False, **kw)
        on = self.run_ep(CORRIDOR, make_config(), shadow=True, **kw)
        self.assert_same_global(off, on)
        return off, on

    def test_global_full(self):
        off, on = self._pair(lambda: dev_config(MAX_STEPS))
        self.assertEqual((off.result["termination"], off.result["steps"]), ("SUCCESS", 13))
        (rec,) = self.assert_records_match_episode(on)
        self.assertEqual((rec.trigger_t, rec.decision.reason, rec.wfg.groups), (5, "FULL_GROUP", ((0, 1),)))
        self.assertEqual(rec.global_status, "success")
        self.assertEqual(off.result["cbs_calls"][0]["latency_ms"], 50.0)   # recorded_solver 0.05 s

    def test_global_h3(self):
        off, on = self._pair(lambda: dev_resume_config(3))
        self.assertEqual(off.result["termination"], "SUCCESS")
        self.assertIn(("CBS_EXEC", "IL_RESUME"), [(t["from"], t["to"]) for t in off.result["transitions"]])
        (rec,) = self.assert_records_match_episode(on)
        self.assertEqual(rec.decision.reason, "FULL_GROUP")

    def test_h3_rearm_dev_values(self):
        off, on = self._pair(lambda: dev_rearm_config(3))
        self.assertEqual([x["t"] for x in off.result["monitor_rearms"]], [8])
        (rec,) = self.assert_records_match_episode(on)
        # rearm forwarded exactly once per IL_RESUME -> IL, recorded with its t.
        self.assertEqual(on.monitor.rearms, [(1, x["t"]) for x in off.result["monitor_rearms"]])
        self.assertEqual(on.monitor.inner.rearm_count, off.monitor.rearm_count)
        self.assertEqual(off.monitor.rearm_count, 1)

    def test_rearm_second_call_after_restall_testonly(self):
        off, on = self._pair(_rearm_two_calls)
        calls = off.result["cbs_calls"]
        self.assertEqual([(c["trigger_timestep"], c["status"]) for c in calls],
                         [(5, "success"), (7, "solver_error")])     # no recorded plan for the 2nd S_t
        self.assertEqual(off.result["stop_reason"], "CBS_SOLVER_ERROR")
        recs = self.assert_records_match_episode(on)
        self.assertEqual([r.trigger_t for r in recs], [5, 7])
        self.assertEqual(on.monitor.rearms, [(1, 6)])
        # The second record is built from post-rearm steps, not the first Trigger's.
        self.assertNotEqual(recs[1].s_t, recs[0].s_t)
        self.assertEqual(len(on.inner_invocations), 2)

    def test_rearm_cooldown_refusal_makes_no_call_testonly(self):
        off, on = self._pair(_rearm_cooldown)
        self.assertEqual([(x["t"], x["outcome"]) for x in off.result["triggers"]],
                         [(5, "CBS_REQUEST"), (7, "RECALL_COOLDOWN_ACTIVE")])
        recs = self.assert_records_match_episode(on)
        self.assertEqual(len(recs), 1)                     # refused Trigger -> no planner call, no record
        self.assertEqual(len(on.inner_invocations), 1)


@unittest.skipIf(torch is None, "torch is not installed")
class TestWarehousePartialCandidate(_Env):
    """WH8-A (8 robots): the real IL stalls at t=10 into three WFG groups with
    one outside agent. The mock solver answers no_solution (no recorded atb033
    plan exists for this state), so Global ends in SAFE_STOP."""

    def test_partial_group_is_candidate_only_all_dev_configs(self):
        solver = _status_solver("no_solution", 0.2)
        for name, make in (("FULL", lambda: dev_config(MAX_STEPS)), ("H3", lambda: dev_resume_config(3)),
                           ("H3_REARM", lambda: dev_rearm_config(3))):
            with self.subTest(name):
                off = self.run_ep(WH8A, make(), shadow=False, solver=solver)
                on = self.run_ep(WH8A, make(), shadow=True, solver=solver)
                self.assert_same_global(off, on)
                self.assertEqual((off.result["termination"], off.result["stop_reason"]),
                                 ("SAFE_STOP", "CBS_NO_SOLUTION"))
                (rec,) = self.assert_records_match_episode(on)
                d = rec.decision
                self.assertEqual(rec.trigger_t, 10)
                self.assertEqual(rec.wfg.groups, ((0, 3), (1, 2, 5), (4, 6)))
                self.assertEqual((rec.l0_group, rec.outside), ((0, 1, 2, 3, 4, 5, 6), (7,)))
                # Marked as a partial-group CANDIDATE, explicitly not executed.
                self.assertEqual(d.reason, SELECTIVE_CANDIDATE_BLOCKED)
                self.assertTrue(d.selective_candidate)
                self.assertEqual(d.route, ROUTE_GLOBAL)
                self.assertEqual(d.blockers, PENDING_BLOCKERS)
                self.assertIn(("MULTIPLE_GROUPS", 3), d.findings)
                self.assertIsNotNone(rec.precheck)
                # Global still asked for all 8 agents and its own status is recorded.
                self.assertEqual(off.result["cbs_calls"][0]["agent_order"], tuple(range(8)))
                self.assertEqual(len(on.solver_calls[0]["current_positions"]), 8)
                self.assertEqual(rec.global_status, "no_solution")
                self.assertNotIn("CBS", off.result["action_sources"])


# -- Same record / CSV as the existing common_runner output ---------------------------

@unittest.skipIf(torch is None, "torch is not installed")
class TestAgainstCommonRunner(_Env):
    """run_method (Shadow OFF, untouched) vs a direct Shadow ON episode written
    with the same run_id/mode: raw result and all three CSVs byte-identical."""

    def test_raw_result_and_csvs_identical(self):
        out = self.tmp / "runner_out"
        for method, make in ((GLOBAL_HYBRID_DEV_FULL, lambda: dev_config(MAX_STEPS)),
                             (GLOBAL_HYBRID_DEV_H3, lambda: dev_resume_config(3)),
                             (GLOBAL_HYBRID_DEV_H3_REARM, lambda: dev_rearm_config(3))):
            with self.subTest(method):
                with patch("src.cbs_runtime_adapter.replan_with_status", side_effect=recorded_solver):
                    record = run_method(RunSpec(method=method, scenario=CORRIDOR, out_dir=out,
                                                solver_root=self.solver_root))
                on = self.run_ep(CORRIDOR, make(), shadow=True, mode=method, run_id=record.run_id,
                                 out_dir=self.tmp / f"shadow_{method}")
                self.assertEqual(on.result, record.raw)
                run_dir = Path(record.run_dir)
                for name in ("steps", "summary", "monitor"):
                    self.assertEqual(on.csv[name], (run_dir / f"{record.run_id}_{name}.csv").read_bytes(), name)
                strip = lambda calls: [{k: v for k, v in c.items() if k != "work_dir"} for c in calls]  # noqa: E731
                self.assertEqual(strip(_planner_calls(on.inner)), strip(record.planner_calls))
                mr = on.monitor.result
                for k in ("trigger_type", "trigger_reason_code", "trigger_timestep", "first_seen",
                          "period", "overlap_integrity_status"):
                    self.assertEqual(getattr(mr, k), record.monitor[k], k)
                if method == GLOBAL_HYBRID_DEV_H3_REARM:
                    self.assertEqual(on.monitor.rearm_count, record.monitor["rearm_count"])
                    self.assertEqual([asdict(h) for h in on.monitor.trigger_history],
                                     record.monitor["trigger_history"])
                self.assert_records_match_episode(on)


# -- C. planner failures: Global failure path unchanged ------------------------------

@unittest.skipIf(torch is None, "torch is not installed")
class TestPlannerFailures(_Env):

    def _pair(self, **kw):
        off = self.run_ep(CORRIDOR, dev_resume_config(3), shadow=False, **kw)
        on = self.run_ep(CORRIDOR, dev_resume_config(3), shadow=True, **kw)
        self.assert_same_global(off, on)
        return off, on

    def test_solver_statuses(self):
        cases = (("timeout", _timeout_solver, "CBS_TIMEOUT", 30000.0),
                 ("no_solution", _status_solver("no_solution", 0.4), "CBS_NO_SOLUTION", 400.0),
                 ("invalid_path", _status_solver("invalid_path", 0.3), "CBS_INVALID_PATH", 300.0),
                 ("solver_error", _status_solver("solver_error", 0.1), "CBS_SOLVER_ERROR", 100.0))
        for status, solver, stop, latency_ms in cases:
            with self.subTest(status):
                off, on = self._pair(solver=solver)
                self.assertEqual((off.result["termination"], off.result["stop_reason"], off.result["steps"]),
                                 ("SAFE_STOP", stop, 5))
                self.assertEqual(off.result["cbs_calls"][0]["status"], status)
                self.assertEqual(on.result["cbs_calls"][0]["latency_ms"], latency_ms)
                self.assertNotIn("CBS", off.result["action_sources"])
                (rec,) = self.assert_records_match_episode(on)
                self.assertEqual(rec.global_status, status)

    def test_success_with_bad_start_rejected_by_path_guard(self):
        off, on = self._pair(solver=_bad_start_solver)
        self.assertEqual((off.result["stop_reason"], off.result["cbs_calls"][0]["status"]),
                         ("CBS_INVALID_PATH", "invalid_path"))
        (rec,) = self.assert_records_match_episode(on)
        self.assertEqual(rec.global_status, "invalid_path")

    def test_planner_exception_propagates_unchanged(self):
        off = self.run_ep(CORRIDOR, dev_resume_config(3), shadow=False,
                          bridge_raises=RuntimeError("mock planner crash"))
        on = self.run_ep(CORRIDOR, dev_resume_config(3), shadow=True,
                         bridge_raises=RuntimeError("mock planner crash"))
        self.assert_same_global(off, on)
        self.assertEqual(off.result["stop_reason"], "PLANNER_EXCEPTION")
        self.assertEqual(off.result["cbs_calls"][0]["error"], "RuntimeError: mock planner crash")
        self.assertEqual(len(on.inner_invocations), 1)      # inner called once, not retried
        (rec,) = on.planner.records
        self.assertIsNone(rec.shadow_error)                 # analysis done before the call
        self.assertIsNone(rec.global_status)                # no result object came back
        self.assertEqual(rec.s_t, off.result["cbs_calls"][0]["s_t"])


# -- C. failures inside the shadow only --------------------------------------------

@unittest.skipIf(torch is None, "torch is not installed")
class TestShadowInternalFailure(_Env):

    def test_wfg_failure_keeps_global_and_planner_result(self):
        off = self.run_ep(CORRIDOR, dev_resume_config(3), shadow=False)
        with patch("src.hybrid.selective_shadow.build_wfg", side_effect=RuntimeError("wfg boom")):
            on = self.run_ep(CORRIDOR, dev_resume_config(3), shadow=True)
        self.assert_same_global(off, on)
        (rec,) = on.planner.records
        self.assertEqual(rec.shadow_error, "EXCEPTION:RuntimeError:wfg boom")
        self.assertIsNone(rec.decision)
        self.assertEqual(rec.global_status, "success")
        self.assertEqual(len(on.inner_invocations), 1)

    def test_precheck_failure_on_partial_group_keeps_global(self):
        solver = _status_solver("no_solution", 0.2)
        off = self.run_ep(WH8A, dev_config(MAX_STEPS), shadow=False, solver=solver)
        with patch("src.hybrid.selective_shadow.bfs_dist", side_effect=ValueError("bfs boom")):
            on = self.run_ep(WH8A, dev_config(MAX_STEPS), shadow=True, solver=solver)
        self.assert_same_global(off, on)
        (rec,) = on.planner.records
        self.assertEqual(rec.shadow_error, "EXCEPTION:ValueError:bfs boom")
        self.assertEqual(rec.global_status, "no_solution")

    def test_monitor_recording_failure_keeps_global(self):
        off = self.run_ep(CORRIDOR, dev_rearm_config(3), shadow=False)
        with patch("src.hybrid.selective_shadow.ShadowStep", side_effect=RuntimeError("rec boom")):
            on = self.run_ep(CORRIDOR, dev_rearm_config(3), shadow=True)
        self.assert_same_global(off, on)
        self.assertTrue(on.monitor.record_errors)
        self.assertEqual(on.monitor.steps, [])
        (rec,) = on.planner.records
        self.assertEqual(rec.shadow_error, "NO_STEP_OBSERVED")
        self.assertEqual(rec.global_status, "success")


# -- C. episode boundaries: reset / rearm / two consecutive episodes ------------------

@unittest.skipIf(torch is None, "torch is not installed")
class TestEpisodeBoundaries(_Env):

    def test_off_runs_are_deterministic(self):
        """Precondition of every ON/OFF comparison."""
        a = self.run_ep(CORRIDOR, _rearm_two_calls(), shadow=False)
        b = self.run_ep(CORRIDOR, _rearm_two_calls(), shadow=False)
        self.assert_same_global(a, b)

    def test_two_consecutive_episodes_on_same_wrappers(self):
        solver_wh = _status_solver("no_solution", 0.2)
        off1 = self.run_ep(CORRIDOR, _rearm_two_calls(), shadow=False)
        off2 = self.run_ep(WH8A, dev_resume_config(3), shadow=False, solver=solver_wh)

        first = self.run_ep(CORRIDOR, _rearm_two_calls(), shadow=True)
        w = first.planner
        self.assert_same_global(off1, first)
        self.assert_records_match_episode(first, episode=1)
        ep1_records = list(w.records)

        second = self.run_ep(WH8A, dev_resume_config(3), shadow=True, solver=solver_wh, wrapper=w)
        # Global outcome of episode 2 equals an independent OFF run (no carry-over).
        for key in ("trajectory", "actions", "transitions", "cbs_calls", "termination", "stop_reason"):
            self.assertEqual(second.result[key], off2.result[key], key)
        self.assertEqual(second.monitor.result, off2.monitor.result)
        self.assertEqual(second.csv, off2.csv)
        # Shadow state: episode 2 only, call index restarted, episode-1 records untouched.
        self.assertEqual(w.monitor.episode, 2)
        self.assertTrue(all(s.episode == 2 for s in w.monitor.steps))
        self.assertEqual(w.monitor.rearms, [])               # episode-1 rearm (t=6) not carried
        self.assertEqual(w.records[:len(ep1_records)], ep1_records)
        recs2 = self.assert_records_match_episode(second, episode=2)
        self.assertEqual([(r.episode, r.call_index) for r in recs2], [(2, 1)])
        self.assertEqual(sorted(recs2[0].s_t), list(range(8)))
        self.assertEqual(w.monitor.goals, {a: tuple(g) for a, g in second.goals.items()})
        self.assertEqual(len(second.inner_invocations), 1)

    def test_fresh_wrappers_per_run_do_not_share_state(self):
        a = self.run_ep(CORRIDOR, _rearm_two_calls(), shadow=True)
        b = self.run_ep(CORRIDOR, _rearm_two_calls(), shadow=True)
        self.assertIsNot(a.planner, b.planner)
        self.assertEqual((a.monitor.episode, b.monitor.episode), (1, 1))
        self.assertEqual([r.trigger_t for r in b.planner.records], [5, 7])
        self.assertEqual(b.monitor.rearms, [(1, 6)])

    def test_wrapper_contract_with_real_planner(self):
        mon = ShadowMonitor(DeadlockMonitor())
        inner = CbsRuntimePlanner(solver_root=self.solver_root, work_root=self.tmp / "contract")
        w = ShadowSelectivePlanner(inner, mon)
        self.assertEqual((w.paths_keying, w.test_only), ("agent_id", False))
        check_validator_pairing(w, PathGuardValidator())        # real pairing accepted
        self.assertFalse(hasattr(w, "calls"))                    # read inner.calls, not wrapper.calls
        self.assertIs(w.inner.calls, inner.calls)


if __name__ == "__main__":
    unittest.main()
