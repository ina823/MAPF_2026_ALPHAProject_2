"""DEV-only Selective Hybrid run on a synthetic 3-agent fixture with MOCK planners.

Runs the real, unchanged ``run_global_hybrid_episode`` + MAPFStepSimulator +
DeadlockMonitor three times on the same fixture and writes one summary:

    GLOBAL                  plain Global Hybrid (PathGuardValidator)
    SELECTIVE_DEFAULT_G     SelectivePlanner, ladder ("G",)  -> must equal GLOBAL
    SELECTIVE_OPT_IN_MOCK   SelectivePlanner, ladder ("L0", "L1", "G"), MOCK subset planner

No CBS solver is called and no IL checkpoint is loaded (ScriptedPolicy).
Every planner here is ``test_only``. The OPT_IN run's group plan is a scripted
MOCK answer: it shows the Runner wiring and the safety checks, NOT that a real
subset CBS call would succeed (PENDING:REAL_SUBSET_CBS). ``--real`` is refused.
Budget numbers below are TEST values, not agreed policy (PENDING P4).

Output: a NEW directory (refused if it exists), default
outputs/selective_mock_dev/<YYYYmmdd_HHMMSS>_selective_mock_dev/summary.json.

Usage (from the repository root):
    python -m scripts.run_selective_mock_dev [--out-root DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.deadlock_monitor import DeadlockMonitor
from src.hybrid.cbs_runtime_bridge import PathGuardValidator
from src.hybrid.global_hybrid import GlobalHybridConfig, run_global_hybrid_episode
from src.hybrid.selective_planner import (
    PENDING,
    SelectiveConfig,
    SelectivePathValidator,
    SelectivePlanner,
)
from src.hybrid.selective_safety import TriggerBudget
from src.hybrid.selective_shadow import ShadowMonitor
from tests.hybrid_mocks import (
    DOWN, LEFT, RIGHT, UP, WAIT,
    AllowAllRecalls, AlwaysResumeIL, MockReplanResult, RecordSafeStop, ScriptedPolicy,
)

DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs" / "selective_mock_dev"

# -- fixture A: 3 agents, partial group {10, 20}, outside agent 7 -----------------------
#   3x7 open grid. 10 (DOWN) and 20 (UP) contest (1,1) forever; 7 oscillates
#   (1,6) <-> (1,5). S_2 == S_0 -> Trigger at t=2 with S_1 != S_2 (7 moved).
GRID_A = np.zeros((3, 7), dtype=np.int64)
STARTS_A = {7: (1, 6), 10: (0, 1), 20: (2, 1)}
GOALS_A = {7: (1, 5), 10: (2, 1), 20: (0, 1)}
ACTIONS_A = {10: [DOWN] * 40, 20: [UP] * 40, 7: [LEFT, RIGHT, LEFT] + [WAIT] * 40}
GROUP_PATHS_A = {10: [(0, 1), (0, 1), (1, 1), (2, 1), (2, 1)],
                 20: [(2, 1), (2, 2), (1, 2), (0, 2), (0, 1)]}
GLOBAL_PATHS_A = {**GROUP_PATHS_A, 7: [(1, 6), (1, 5), (1, 5), (1, 5), (1, 5)]}

TEST_BUDGET = TriggerBudget(total_time_ms=1000.0, max_attempts=3,
                            level_timeout_ms={"L0": 400.0, "L1": 400.0, "G": 800.0})


class ScriptedGlobalPlanner:
    """TEST ONLY. agent_id keyed like CbsRuntimePlanner; answers from a queue
    (the last answer repeats). Accepts ``timeout_ms`` for opt-in ladders."""
    test_only = True
    paths_keying = "agent_id"
    accepts_timeout_ms = True

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, current_positions, goals, grid, timeout_ms=None):
        self.calls.append({"current_positions": dict(current_positions), "goals": dict(goals),
                           "timeout_ms": timeout_ms})
        r = self.results[min(len(self.calls), len(self.results)) - 1]
        if isinstance(r, BaseException):
            raise r
        return r


class ScriptedSubsetPlanner:
    """TEST ONLY stand-in for a group CBS call (no real subset CBS exists)."""
    test_only = True

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, current_positions, goals, grid, *, outside, timeout_ms):
        self.calls.append({"current_positions": dict(current_positions), "goals": dict(goals),
                           "outside": dict(outside), "timeout_ms": timeout_ms})
        r = self.results[min(len(self.calls), len(self.results)) - 1]
        if isinstance(r, BaseException):
            raise r
        return r


def success(paths, latency_sec):
    return MockReplanResult(status="success", paths=paths, latency_sec=latency_sec)


def hybrid_config(h_exec=4, rearm=None):
    return GlobalHybridConfig(h_exec=h_exec, resume_policy=AlwaysResumeIL(),
                              recall_policy=AllowAllRecalls(), safe_stop_handler=RecordSafeStop(),
                              rearm=rearm)


def run_global(global_planner, *, grid=GRID_A, starts=STARTS_A, goals=GOALS_A, actions=ACTIONS_A,
               max_steps=20, h_exec=4, rearm=None, simulator_cls=None):
    kw = {} if simulator_cls is None else {"simulator_cls": simulator_cls}
    return run_global_hybrid_episode(
        ScriptedPolicy({a: list(v) for a, v in actions.items()}), grid, starts, goals, max_steps,
        planner=global_planner, path_validator=PathGuardValidator(),
        config=hybrid_config(h_exec, rearm), monitor=DeadlockMonitor(), **kw)


def run_selective(global_planner, config=None, subset_planner=None, *, grid=GRID_A, starts=STARTS_A,
                  goals=GOALS_A, actions=ACTIONS_A, max_steps=20, h_exec=4, rearm=None, simulator_cls=None):
    mon = ShadowMonitor(DeadlockMonitor())
    planner = SelectivePlanner(global_planner, mon, config or SelectiveConfig(), subset_planner=subset_planner)
    validator = SelectivePathValidator(planner)
    kw = {} if simulator_cls is None else {"simulator_cls": simulator_cls}
    result = run_global_hybrid_episode(
        ScriptedPolicy({a: list(v) for a, v in actions.items()}), grid, starts, goals, max_steps,
        planner=planner, path_validator=validator, config=hybrid_config(h_exec, rearm), monitor=mon, **kw)
    return result, planner, validator


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if hasattr(o, "item"):
        return o.item()
    return o


def _episode_summary(result):
    return {k: result[k] for k in ("termination", "stop_reason", "steps", "all_at_goal",
                                   "action_sources", "transitions", "final_positions")} | {
        "cbs_calls": [{k: c[k] for k in ("trigger_timestep", "s_t", "status", "accepted", "reason", "latency_ms")}
                      for c in result["cbs_calls"]],
        "exec_trace_all_match": all(e["match"] for e in result["cbs_exec_trace"]),
    }


def _record_summary(rec):
    return {
        "mode": rec.mode, "trigger_t": rec.trigger_t, "s_t_minus_1": rec.s_t_minus_1, "s_t": rec.s_t,
        "wfg_groups": None if rec.wfg is None else rec.wfg.groups,
        "decision_route": None if rec.decision is None else rec.decision.route,
        "decision_reason": None if rec.decision is None else rec.decision.reason,
        "levels": [asdict(lv) for lv in rec.levels],
        "attempts": [asdict(a) for a in rec.attempts],
        "outcome": rec.outcome, "executed_group": rec.executed_group, "outside_held": rec.outside_held,
        "goal_status": rec.goal_status, "real_subset_cbs_called": rec.real_subset_cbs_called,
    }


def run_all():
    base = run_global(ScriptedGlobalPlanner(success(GLOBAL_PATHS_A, 0.2)))
    dflt, dflt_planner, _ = run_selective(ScriptedGlobalPlanner(success(GLOBAL_PATHS_A, 0.2)))
    opt, opt_planner, _ = run_selective(
        ScriptedGlobalPlanner(success(GLOBAL_PATHS_A, 0.2)),
        SelectiveConfig(ladder=("L0", "L1", "G"), budget=TEST_BUDGET,
                        acknowledge_unverified_outside_policy=True),
        subset_planner=ScriptedSubsetPlanner(success(GROUP_PATHS_A, 0.1)))
    return {
        "note": ("MOCK ONLY: no CBS solver, no IL checkpoint. The opt-in group plan is a scripted "
                 "answer; it is NOT evidence that a real subset CBS succeeds."),
        "mock_only": True,
        "real_cbs_called": False,
        "pending": list(PENDING),
        "test_budget_not_policy": asdict(TEST_BUDGET),
        "fixture": {"grid_shape": list(GRID_A.shape), "starts": STARTS_A, "goals": GOALS_A},
        "default_g_equals_global": dflt == base,
        "runs": {
            "GLOBAL": _episode_summary(base),
            "SELECTIVE_DEFAULT_G": _episode_summary(dflt) | {"records": [_record_summary(r) for r in dflt_planner.records]},
            "SELECTIVE_OPT_IN_MOCK": _episode_summary(opt) | {"records": [_record_summary(r) for r in opt_planner.records]},
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--real", action="store_true", help="refused: real subset CBS is PENDING")
    args = ap.parse_args(argv)
    if args.real:
        print("REFUSED: PENDING:REAL_SUBSET_CBS -- no real subset CBS planner exists; nothing run or written.")
        return 2
    summary = run_all()
    out_dir = Path(args.out_root) / f"{datetime.now():%Y%m%d_%H%M%S}_selective_mock_dev"
    out_dir.mkdir(parents=True, exist_ok=False)            # never reuse a directory
    with (out_dir / "summary.json").open("x", encoding="utf-8") as fh:
        json.dump(_jsonable(summary), fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"default_g_equals_global={summary['default_g_equals_global']}  (MOCK ONLY)")
    print(f"wrote {out_dir / 'summary.json'}")
    return 0 if summary["default_g_equals_global"] else 1


if __name__ == "__main__":
    sys.exit(main())
