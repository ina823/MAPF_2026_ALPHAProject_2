"""Test-only Mock CBS planners, validator stubs and policy stubs for the
Global Hybrid Runner v0 tests (docs/global_hybrid_runner_v0.md).

Nothing in this module may be used for a real CBS run:
  * every Mock planner and every validator stub carries ``test_only = True``;
  * src.hybrid.replan_boundary.check_validator_pairing() refuses a
    ``test_only`` validator unless the planner is ``test_only`` too, so
    AcceptAllValidator can never sit behind the real replan_with_status().

The policy stubs (AlwaysResumeIL, AllowAllRecalls, ...) stand in for the
H_exec / IL-resume / CBS-recall / SAFE_STOP rules that are NOT agreed yet --
they are temporary values for these tests only, not project defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.il.nav_hint import bfs_dist
from src.simulator.mapf_step_simulator import MAPFStepSimulator

UP, DOWN, LEFT, RIGHT, WAIT = 0, 1, 2, 3, 4


# -- Mock CBS -----------------------------------------------------------------

@dataclass
class MockReplanResult:
    """Same four attributes as the CBS branch's ReplanResult contract
    (status, paths, latency_sec, error) -- paths are indexed 0..n-1 in the
    order of the planner's input lists."""

    status: str
    paths: Any = None
    latency_sec: float | None = None
    error: Any = None


class RecordingPlanner:
    """Returns a fixed result and records every call's inputs."""

    test_only = True

    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, current_positions, goals, grid):
        self.calls.append({
            "current_positions": list(current_positions),
            "goals": list(goals),
        })
        return self.result


class ReversedOrderPlanner(RecordingPlanner):
    """Returns the given paths in reversed index order -- an agent-mapping bug."""

    def __call__(self, current_positions, goals, grid):
        result = super().__call__(current_positions, goals, grid)
        return MockReplanResult(
            status=result.status,
            paths=list(reversed(result.paths)),
            latency_sec=result.latency_sec,
            error=result.error,
        )


class RaisingPlanner:
    test_only = True

    def __init__(self):
        self.calls = 0

    def __call__(self, current_positions, goals, grid):
        self.calls += 1
        raise RuntimeError("mock solver crashed")


class FailIfCalledPlanner:
    test_only = True

    def __call__(self, current_positions, goals, grid):
        raise AssertionError("planner must not be called")


def success(paths, latency_sec=0.25):
    return MockReplanResult(status="success", paths=paths, latency_sec=latency_sec)


# -- Validator stubs (stand-ins for the CBS branch's src/path_guard.py) --------

class AcceptAllValidator:
    """TEST ONLY: accepts every path. Refused with a non-test planner."""

    test_only = True

    def __init__(self):
        self.calls = []

    def __call__(self, grid, starts, goals, paths):
        self.calls.append({"starts": dict(starts), "goals": dict(goals), "paths": dict(paths)})
        return True, ""


class RejectAllValidator:
    test_only = True

    def __init__(self, reason="mock_reject"):
        self.reason = reason
        self.calls = 0

    def __call__(self, grid, starts, goals, paths):
        self.calls += 1
        return False, self.reason


# -- Policy stubs (temporary, test-only values for undecided rules) -----------

class AlwaysResumeIL:
    def after_exec(self, context):
        return "IL"


class AlwaysSafeStopAfterExec:
    def after_exec(self, context):
        return "SAFE_STOP"


class AllowAllRecalls:
    def __init__(self):
        self.asked = []

    def allow_call(self, call_count, trigger_timestep):
        self.asked.append((call_count, trigger_timestep))
        return True


class DenyAllRecalls:
    def allow_call(self, call_count, trigger_timestep):
        return False


class RecordSafeStop:
    def __init__(self):
        self.calls = []

    def on_safe_stop(self, reason, context):
        self.calls.append((reason, context))
        return {"reason": reason, "t": context["t"]}


# -- IL policy stubs (torch-free) ---------------------------------------------

class ScriptedPolicy:
    """Replays fixed per-agent action lists, then WAITs. Builds `_dist` with
    bfs_dist() because the runner feeds goal distances to the Monitor."""

    def __init__(self, actions_per_agent=None):
        self._actions_per_agent = actions_per_agent or {}
        self._t = 0

    def reset(self, grid, goals):
        self._dist = {aid: bfs_dist(grid, goal) for aid, goal in goals.items()}
        self._t = 0

    def act(self, obs, positions):
        actions = {}
        for aid in positions:
            seq = self._actions_per_agent.get(aid, [])
            actions[aid] = seq[self._t] if self._t < len(seq) else WAIT
        self._t += 1
        return actions


# -- Fault-injecting simulators -----------------------------------------------

def perturbing_simulator(inject_t, agent_id, new_pos):
    """After the real step() reaching timestep `inject_t`, force `agent_id`
    onto `new_pos` (an expected/actual mismatch, or an overlap when
    `new_pos` is another agent's cell)."""

    class _PerturbingSimulator(MAPFStepSimulator):
        def step(self, actions):
            obs, done, info = super().step(actions)
            if info["t"] == inject_t:
                self._positions[agent_id] = new_pos
            return obs, done, info

    return _PerturbingSimulator


# -- Tiny synthetic map -------------------------------------------------------
#
#   3x4 open grid. With an all-WAIT policy the joint state at t=1 equals
#   S_0, so the real DeadlockMonitor fires TRIGGER_FULL_REPEAT at t=1.

def open_grid():
    return np.zeros((3, 4), dtype=np.int64)


STARTS = {0: (0, 0), 1: (2, 0)}
GOALS = {0: (0, 2), 1: (2, 2)}
PATHS_TO_GOALS = [
    [(0, 0), (0, 1), (0, 2)],
    [(2, 0), (2, 1), (2, 2)],
]
