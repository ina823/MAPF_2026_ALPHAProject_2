"""scripts/run_hybrid_resume_dev.py: DEV_ONLY H_exec window + IL resume.

  * DEV policies and recorders (no solver needed);
  * resume analysis / verdict on hand-built episode records -- "reached
    IL_RESUME" must stay separate from "IL really acted again", and a resume
    that times out is never a PASS;
  * real IL + a stand-in cbs.py that exits 3 (real subprocess, real Path
    Guard): SAFE_STOP with zero moves after the Trigger, no resume;
  * real atb033 solver on Corridor with H_exec=3 (skipped when not found).

All solver files go to a temporary directory.
"""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts.run_hybrid_resume_dev import (
    NO_TRIGGER,
    RESUME_NO_IL_STEP,
    RESUME_SUCCESS,
    RESUME_TIMEOUT_NO_REPEAT,
    RESUME_TIMEOUT_RESTALL,
    SUCCESS_WITHIN_H_EXEC,
    ActRecordingPolicy,
    DevResumeIL,
    dev_resume_config,
    evaluate_resume,
    find_repeat_after,
    main,
    mode_name,
    run_resume_e2e,
)
from scripts.run_il import PROJECT_ROOT
from src.hybrid.cbs_runtime_bridge import CbsRuntimePlanner
from src.hybrid.global_hybrid import IL

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

REAL_SOLVER_ROOT = Path(os.environ.get(
    "ALPHA2_CBS_SOLVER_ROOT",
    PROJECT_ROOT.parent / "MAPF_ALPHA2_CBS_CHECK/third_party/multi_agent_path_planning/centralized/cbs",
))
HAVE_REAL_SOLVER = (REAL_SOLVER_ROOT / "cbs.py").is_file()


def row(c0, c1):
    return {0: (0, c0), 1: (1, c1)}


def synthetic(*, il_before, paths, window, il_after, goals, termination,
              stop_reason=None, resume=True, back_to_il=True):
    """A run_global_hybrid_episode()-shaped record: IL steps, Trigger on the
    last IL state, `window` CBS steps along `paths`, then `il_after` states."""
    trajectory = list(il_before)
    trigger_t = len(trajectory) - 1
    s_t = dict(trajectory[-1])
    sources = ["IL"] * trigger_t
    transitions = [
        {"t": trigger_t, "from": "IL", "to": "TRIGGER", "reason": "TRIGGER_FULL_REPEAT"},
        {"t": trigger_t, "from": "TRIGGER", "to": "CBS_REQUEST", "reason": ""},
        {"t": trigger_t, "from": "CBS_REQUEST", "to": "CBS_EXEC", "reason": ""},
    ]
    trace = []
    for k in range(1, window + 1):
        state = {aid: p[min(k, len(p) - 1)] for aid, p in paths.items()}
        trajectory.append(state)
        sources.append("CBS")
        trace.append({"t": len(trajectory) - 1, "k": k, "expected": dict(state),
                      "actual": dict(state), "match": True, "overlap": False})
    resume_t = len(trajectory) - 1
    if resume:
        transitions.append({"t": resume_t, "from": "CBS_EXEC", "to": "IL_RESUME",
                            "reason": "EXEC_WINDOW_END"})
        if back_to_il:
            transitions.append({"t": resume_t, "from": "IL_RESUME", "to": "IL", "reason": ""})
    for state in il_after:
        trajectory.append(state)
        sources.append("IL")
    result = {
        "steps": len(trajectory) - 1,
        "all_at_goal": trajectory[-1] == goals,
        "final_positions": trajectory[-1],
        "goals": goals,
        "trajectory": trajectory,
        "termination": termination,
        "stop_reason": stop_reason,
        "transitions": transitions,
        "cbs_calls": [{"trigger_timestep": trigger_t, "agent_order": (0, 1), "s_t": s_t,
                       "status": "success", "accepted": True, "reason": "", "error": None}],
        "action_sources": sources,
        "cbs_exec_trace": trace,
        "exec_mismatch": None,
    }
    planner_calls = [{"paths": paths}]
    # act() sees the state right before every IL step.
    act_positions = [dict(trajectory[i]) for i, s in enumerate(sources) if s == "IL"]
    sim_step_ts = list(range(1, len(sources) + 1))
    return result, planner_calls, act_positions, sim_step_ts


LONG_PATHS = {0: [(0, c) for c in range(1, 7)], 1: [(1, c) for c in range(1, 7)]}  # horizon 5
GOALS = row(6, 6)
IL_BEFORE = [row(0, 0), row(1, 1), row(1, 1)]  # repeat -> Trigger at t=2


class TestDevPolicies(unittest.TestCase):

    def test_resume_policy_always_returns_il(self):
        self.assertEqual(DevResumeIL().after_exec({"t": 5}), IL)

    def test_dev_config(self):
        config = dev_resume_config()
        self.assertEqual(config.h_exec, 3)
        self.assertIsInstance(config.resume_policy, DevResumeIL)
        self.assertTrue(config.recall_policy.allow_call(0, 5))
        self.assertFalse(config.recall_policy.allow_call(1, 9))
        self.assertEqual(dev_resume_config(2).h_exec, 2)
        self.assertEqual(mode_name(3), "GLOBAL_HYBRID_DEV_H3")

    def test_act_recorder_forwards_and_copies(self):
        class _Inner:
            _dist = {0: "field"}

            def reset(self, grid, goals):
                self.was_reset = True

            def act(self, obs, positions):
                return {aid: 4 for aid in positions}

        inner = _Inner()
        rec = ActRecordingPolicy(inner)
        rec.reset(None, {})
        self.assertTrue(inner.was_reset)
        self.assertEqual(rec._dist, {0: "field"})
        live = {0: (1, 2)}
        self.assertEqual(rec.act(None, live), {0: 4})
        live[0] = (9, 9)
        self.assertEqual(rec.act_positions, [{0: (1, 2)}])


class TestFindRepeatAfter(unittest.TestCase):

    def test_repeat_after_start(self):
        traj = [row(0, 0), row(1, 1), row(2, 2), row(1, 1)]
        self.assertEqual(find_repeat_after(traj, 1, GOALS), {"t": 3, "first_seen": 1, "period": 2})

    def test_states_before_start_are_ignored(self):
        traj = [row(1, 1), row(2, 2), row(1, 1)]
        self.assertIsNone(find_repeat_after(traj, 1, GOALS))

    def test_goal_state_is_never_a_repeat(self):
        traj = [row(5, 5), GOALS, GOALS]
        self.assertIsNone(find_repeat_after(traj, 0, GOALS))


class TestEvaluateResume(unittest.TestCase):

    def _eval(self, *args, h_exec=3):
        return evaluate_resume(*args, h_exec)

    def test_resume_then_goal_is_pass(self):
        rec = synthetic(il_before=IL_BEFORE, paths=LONG_PATHS, window=3,
                        il_after=[row(5, 5), row(6, 6)], goals=GOALS, termination="SUCCESS")
        ev = self._eval(*rec)
        self.assertEqual(ev["verdict"], "PASS", ev["first_failed_check"])
        res = ev["resume"]
        self.assertEqual(res["outcome"], RESUME_SUCCESS)
        self.assertEqual(res["il_resume_t"], 5)
        self.assertEqual(res["first_il_step_after_resume"], 6)
        self.assertEqual(res["il_steps_after_resume"], 2)
        self.assertEqual(res["policy_act_calls_after_resume"], 2)
        self.assertEqual(res["sim_step_calls_after_resume"], 2)
        self.assertEqual(res["first_act_positions_after_resume"], row(4, 4))
        self.assertEqual(ev["cbs_window"], 3)

    def test_restall_after_resume_is_fail_and_recorded(self):
        rec = synthetic(il_before=IL_BEFORE, paths=LONG_PATHS, window=3,
                        il_after=[row(5, 5), row(4, 4), row(5, 5)], goals=GOALS,
                        termination="TIMEOUT")
        ev = self._eval(*rec)
        self.assertEqual(ev["verdict"], "FAIL")
        self.assertEqual(ev["first_failed_check"], "all_at_goal")
        res = ev["resume"]
        self.assertEqual(res["outcome"], RESUME_TIMEOUT_RESTALL)
        self.assertEqual(res["restall_after_resume"], {"t": 7, "first_seen": 5, "period": 2})
        self.assertEqual(res["monitor_retriggers_after_resume"], [])
        # Resume itself happened and IL really acted -- only the goal is missing.
        self.assertTrue(next(c["ok"] for c in ev["checks"] if c["name"] == "il_resumed_and_really_acted"))

    def test_timeout_without_repeat(self):
        rec = synthetic(il_before=IL_BEFORE, paths=LONG_PATHS, window=3,
                        il_after=[row(5, 5), row(5, 4)], goals=GOALS, termination="TIMEOUT")
        ev = self._eval(*rec)
        self.assertEqual(ev["verdict"], "FAIL")
        self.assertEqual(ev["resume"]["outcome"], RESUME_TIMEOUT_NO_REPEAT)

    def test_reaching_il_resume_without_il_step_is_not_a_resume(self):
        rec = synthetic(il_before=IL_BEFORE, paths=LONG_PATHS, window=3, il_after=[],
                        goals=GOALS, termination="SAFE_STOP", stop_reason="RESUME_POLICY_SAFE_STOP",
                        back_to_il=False)
        ev = self._eval(*rec)
        res = ev["resume"]
        self.assertTrue(res["il_resume_reached"])
        self.assertFalse(res["il_actions_executed_after_resume"])
        self.assertEqual(res["outcome"], RESUME_NO_IL_STEP)
        self.assertEqual(ev["first_failed_check"], "il_resumed_and_really_acted")

    def test_act_count_mismatch_fails_resume_check(self):
        result, calls, acts, ts = synthetic(il_before=IL_BEFORE, paths=LONG_PATHS, window=3,
                                            il_after=[row(5, 5), row(6, 6)], goals=GOALS,
                                            termination="SUCCESS")
        ev = self._eval(result, calls, acts[:-1], ts)
        self.assertEqual(ev["first_failed_check"], "il_resumed_and_really_acted")

    def test_goal_within_window_is_success_without_resume(self):
        short = {0: [(0, 1), (0, 2), (0, 3)], 1: [(1, 1), (1, 2), (1, 3)]}  # horizon 2
        rec = synthetic(il_before=IL_BEFORE, paths=short, window=2, il_after=[],
                        goals=row(3, 3), termination="SUCCESS", resume=False)
        ev = self._eval(*rec)
        self.assertEqual(ev["verdict"], "PASS", ev["first_failed_check"])
        self.assertEqual(ev["resume"]["outcome"], SUCCESS_WITHIN_H_EXEC)
        self.assertFalse(ev["resume"]["il_resume_reached"])
        self.assertEqual(ev["cbs_window"], 2)

    def test_exec_mismatch_is_not_a_resume(self):
        result, calls, acts, ts = synthetic(il_before=IL_BEFORE, paths=LONG_PATHS, window=1,
                                            il_after=[], goals=GOALS, termination="SAFE_STOP",
                                            stop_reason="EXEC_MISMATCH", resume=False)
        result["cbs_exec_trace"][0]["actual"] = row(3, 2)
        result["cbs_exec_trace"][0]["match"] = False
        ev = self._eval(result, calls, acts, ts)
        self.assertEqual(ev["verdict"], "FAIL")
        self.assertEqual(ev["first_failed_check"], "first_cbs_step_reaches_path1")
        self.assertEqual(ev["resume"]["outcome"], "NO_RESUME_EXEC_MISMATCH")

    def test_no_trigger(self):
        result, _, acts, ts = synthetic(il_before=IL_BEFORE, paths=LONG_PATHS, window=0,
                                        il_after=[], goals=GOALS, termination="TIMEOUT", resume=False)
        result["cbs_calls"] = []
        ev = self._eval(result, [], acts, ts)
        self.assertEqual(ev["resume"]["outcome"], NO_TRIGGER)
        self.assertEqual(ev["first_failed_check"], "cbs_called_once_on_trigger")

    def test_agent_id_mismatch_fails(self):
        result, calls, acts, ts = synthetic(il_before=IL_BEFORE, paths=LONG_PATHS, window=3,
                                            il_after=[row(5, 5), row(6, 6)], goals=GOALS,
                                            termination="SUCCESS")
        calls[0]["paths"] = {0: LONG_PATHS[0], 7: LONG_PATHS[1]}
        ev = self._eval(result, calls, acts, ts)
        self.assertEqual(ev["first_failed_check"], "agent_ids_preserved")


@unittest.skipIf(torch is None, "torch is not installed")
class _RealIlCase(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


class TestSolverFailureNoResume(_RealIlCase):

    def test_solver_error_safe_stop_without_moves(self):
        root = self.tmp / "solver_crash"
        root.mkdir()
        (root / "cbs.py").write_text("import sys\nsys.exit(3)\n", encoding="utf-8")
        planner = CbsRuntimePlanner(solver_root=root, work_root=self.tmp / "run", timeout_sec=30)
        result, rec, sim_cls = run_resume_e2e(planner)
        trigger_t = result["cbs_calls"][0]["trigger_timestep"]
        self.assertEqual(result["termination"], "SAFE_STOP")
        self.assertEqual(result["stop_reason"], "CBS_SOLVER_ERROR")
        self.assertEqual(sim_cls.step_ts[-1], trigger_t)     # no move after the Trigger
        self.assertEqual(len(rec.act_positions), trigger_t)  # no IL act after the Trigger
        ev = evaluate_resume(result, planner.calls, rec.act_positions, sim_cls.step_ts, 3)
        self.assertEqual(ev["verdict"], "FAIL")
        self.assertEqual(ev["first_failed_check"], "cbs_status_success")
        self.assertEqual(ev["resume"]["outcome"], "NO_RESUME_CBS_SOLVER_ERROR")
        self.assertFalse(ev["resume"]["il_resume_reached"])


@unittest.skipUnless(HAVE_REAL_SOLVER, f"atb033 solver not found at {REAL_SOLVER_ROOT}")
class TestCorridorResumeWithRealSolver(_RealIlCase):

    def test_h3_window_then_real_il(self):
        planner = CbsRuntimePlanner(solver_root=REAL_SOLVER_ROOT, work_root=self.tmp / "run")
        result, rec, sim_cls = run_resume_e2e(planner, h_exec=3)
        ev = evaluate_resume(result, planner.calls, rec.act_positions, sim_cls.step_ts, 3)
        res = ev["resume"]
        self.assertEqual(len(planner.calls), 1)
        self.assertEqual(len(result["cbs_exec_trace"]), 3)
        self.assertTrue(all(e["match"] and not e["overlap"] for e in result["cbs_exec_trace"]))
        self.assertTrue(res["il_resume_reached"])
        self.assertGreater(res["policy_act_calls_after_resume"], 0)
        self.assertEqual(res["first_act_positions_after_resume"], res["state_at_resume"])
        self.assertEqual(res["monitor_retriggers_after_resume"], [])

    def test_main_writes_a_new_run_dir(self):
        out_root = self.tmp / "out"
        with contextlib.redirect_stdout(io.StringIO()):
            main(["--solver-root", str(REAL_SOLVER_ROOT), "--out-root", str(out_root)])
        (run_dir,) = list(out_root.iterdir())
        self.assertTrue(run_dir.name.endswith("GLOBAL_HYBRID_DEV_H3"))
        report = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(report["h_exec"], 3)
        self.assertIn("verdict", report)
        self.assertTrue((run_dir / "cbs" / "cbs_call_01").is_dir())


if __name__ == "__main__":
    unittest.main()
