"""Runner-side acceptance boundary for CBS replan results (Global Hybrid v0).

Every CBS result passes through ``accept_replan_result()`` before a single
move reaches the simulator. The boundary checks only the runner's own
contract (docs/global_hybrid_runner_v0.md):

  1. status == "success" (the CBS branch's six-status contract; anything
     else, including an unknown string, is refused)
  2. paths_keying="index" (default): paths are indexed 0..n-1 in the
     planner's input order and map back onto ``agent_order`` (index i ->
     agent_order[i]). paths_keying="agent_id": paths are a mapping keyed by
     exactly the ids in ``agent_order`` and are used as-is (no re-mapping --
     the planner already converted, e.g. src/cbs_runtime_adapter.py)
  3. every path is non-empty and Path[0] == S_t for its agent
  4. every move of every path converts to a Simulator action via
     spec.delta_to_action() -- a non-adjacent move refuses the WHOLE plan,
     so nothing ever relies on MAPFStepSimulator's stay-in-place handling
  5. the injected ``path_validator`` accepts the paths

Geometric path validation (walls, bounds, vertex/edge conflicts, goals) is
deliberately NOT implemented here: it belongs to the injected validator --
in the real CBS integration, the CBS branch's src/path_guard.py.

``path_validator(grid, starts, goals, paths) -> (ok: bool, reason: str)``
receives agent-id-keyed dicts. A validator marked ``test_only = True``
(e.g. tests/hybrid_mocks.py::AcceptAllValidator) is refused unless the
planner is ``test_only`` too -- see ``check_validator_pairing()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from src.common import spec

Position = Tuple[int, int]

# ReplanResult.status values (CBS branch, docs/cbs_path_guard_handoff.md).
CBS_STATUSES = ("success", "timeout", "no_solution", "invalid_path", "input_error", "solver_error")

# How ReplanResult.paths is keyed: "index" (replan_with_status, Mock
# planners) or "agent_id" (run_cbs_runtime).
PATHS_KEYINGS = ("index", "agent_id")


@dataclass(frozen=True)
class AcceptedPlan:
    """A CBS plan that passed the boundary, ready to execute.

    ``actions[k]`` / ``expected[k]``: per-agent action for execution step
    k+1 and the position every agent must occupy right after it (i.e.
    Path[k+1], or the path's last cell once a shorter path has ended).
    """

    agent_order: Tuple[int, ...]
    paths: Dict[int, List[Position]]
    actions: List[Dict[int, int]]
    expected: List[Dict[int, Position]]
    latency_ms: Optional[float]


@dataclass(frozen=True)
class BoundaryOutcome:
    accepted: bool
    reason: str = ""
    plan: Optional[AcceptedPlan] = None


def _reject(reason: str) -> BoundaryOutcome:
    return BoundaryOutcome(accepted=False, reason=reason)


def check_validator_pairing(planner: Callable, path_validator: Optional[Callable]) -> None:
    """Refuse a missing validator, and a test-only validator behind a
    non-test planner (so an accept-all stub can never guard a real CBS run)."""
    if path_validator is None:
        raise ValueError("path_validator is required")
    if getattr(path_validator, "test_only", False) and not getattr(planner, "test_only", False):
        raise ValueError("a test_only path_validator cannot be used with a non-test planner")


def _paths_by_index(paths: Any, n: int) -> Tuple[Optional[List[Any]], str]:
    if isinstance(paths, Mapping):
        if set(paths.keys()) != set(range(n)):
            return None, "BOUNDARY_PATH_COUNT"
        return [paths[i] for i in range(n)], ""
    if isinstance(paths, (list, tuple)):
        if len(paths) != n:
            return None, "BOUNDARY_PATH_COUNT"
        return list(paths), ""
    return None, "BOUNDARY_PATHS_TYPE"


def _paths_by_agent_id(paths: Any, agent_order: Tuple[int, ...]) -> Tuple[Optional[List[Any]], str]:
    if not isinstance(paths, Mapping):
        return None, "BOUNDARY_PATHS_TYPE"
    if set(paths.keys()) != set(agent_order):
        return None, "BOUNDARY_PATH_COUNT"
    return [paths[aid] for aid in agent_order], ""


def _normalize_path(path: Any) -> Optional[List[Position]]:
    try:
        cells = []
        for cell in path:
            if len(cell) != 2:
                return None
            cells.append((int(cell[0]), int(cell[1])))
        return cells
    except (TypeError, ValueError):
        return None


def accept_replan_result(
    result: Any,
    *,
    agent_order: Sequence[int],
    current_positions: Mapping[int, Position],
    goals: Mapping[int, Position],
    grid: Any,
    path_validator: Callable,
    paths_keying: str = "index",
) -> BoundaryOutcome:
    """Accept or refuse one ReplanResult(status, paths, latency_sec, error).

    ``current_positions`` must be S_t, the actual simulator positions at
    the Trigger. A refusal carries a reason code and never a plan.
    """
    if paths_keying not in PATHS_KEYINGS:
        raise ValueError(f"unknown paths_keying {paths_keying!r}")
    agent_order = tuple(agent_order)

    status = getattr(result, "status", None)
    if status not in CBS_STATUSES:
        return _reject("CBS_UNKNOWN_STATUS")
    if status != "success":
        return _reject(f"CBS_{status.upper()}")

    raw = getattr(result, "paths", None)
    if paths_keying == "agent_id":
        raw_paths, reason = _paths_by_agent_id(raw, agent_order)
    else:
        raw_paths, reason = _paths_by_index(raw, len(agent_order))
    if raw_paths is None:
        return _reject(reason)

    paths: Dict[int, List[Position]] = {}
    for index, agent_id in enumerate(agent_order):
        path = _normalize_path(raw_paths[index])
        if path is None:
            return _reject("BOUNDARY_MALFORMED_PATH")
        if not path:
            return _reject("BOUNDARY_EMPTY_PATH")
        if path[0] != tuple(current_positions[agent_id]):
            return _reject("BOUNDARY_START_MISMATCH")
        paths[agent_id] = path

    # Convert every move now, before execution starts.
    horizon = max(len(p) for p in paths.values()) - 1
    actions: List[Dict[int, int]] = []
    expected: List[Dict[int, Position]] = []
    for k in range(1, horizon + 1):
        step_actions, step_expected = {}, {}
        for agent_id, path in paths.items():
            prev = path[min(k - 1, len(path) - 1)]
            cur = path[min(k, len(path) - 1)]
            try:
                step_actions[agent_id] = spec.delta_to_action(cur[0] - prev[0], cur[1] - prev[1])
            except ValueError:
                return _reject("BOUNDARY_NON_ADJACENT")
            step_expected[agent_id] = cur
        actions.append(step_actions)
        expected.append(step_expected)

    try:
        verdict = path_validator(
            grid,
            {aid: tuple(current_positions[aid]) for aid in agent_order},
            {aid: tuple(goals[aid]) for aid in agent_order},
            {aid: list(p) for aid, p in paths.items()},
        )
        ok, why = verdict
    except Exception:
        return _reject("VALIDATOR_ERROR")
    if not isinstance(ok, bool):
        return _reject("VALIDATOR_ERROR")
    if not ok:
        return _reject(f"VALIDATOR_REJECT:{why}")

    latency_sec = getattr(result, "latency_sec", None)
    return BoundaryOutcome(
        accepted=True,
        plan=AcceptedPlan(
            agent_order=agent_order,
            paths=paths,
            actions=actions,
            expected=expected,
            latency_ms=None if latency_sec is None else latency_sec * 1000.0,
        ),
    )
