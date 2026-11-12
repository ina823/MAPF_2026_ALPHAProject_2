"""Selective full-agent plan composition and pre-execution check — DRAFT, Mock-verified.

Status: rule ``selective-compose-v0``. NOT wired into any runtime: Global
Hybrid, ``common_runner`` and every script are unchanged and do not import it.
It does not make Selective safe or enabled: ``decide_selective`` still routes
GLOBAL in v0 and P1-P5 stay open (the outside-agent WAIT policy itself is the
unverified P2/P3 candidate). No CBS is called here.

What it does, for ONE group CBS result:

    group paths (group agents only, agent_id keyed, Path[0] = S_t)
      + outside agents held at S_t (WAIT)
      -> one equal-length path per agent of the episode
      -> checked BEFORE any move:
           * agent set: every runtime id kept, none added, none renumbered
           * group agents: Path Guard's own ``validate_path_endpoints``
             (Path[0] == S_t AND Path[-1] == goal) and
             ``validate_path_movements`` -- the existing goal rule, unchanged
           * outside agents: Path[k] == S_t for every k. Their goal is NOT
             evaluated and they are never reported as having reached it
             (``OUTSIDE_HELD`` in ``goal_status``).
           * everyone: Path Guard's ``validate_path_map_and_conflicts``
             (bounds, static obstacles, vertex, edge, equal length), unchanged

``src/path_guard.validate_path_guard`` is not called on the composed plan and
is not modified: it requires every agent to end at its goal, which a held
outside agent does not, so it correctly keeps rejecting such a plan.

``selective_boundary_check`` hands the composed plan to the existing
``replan_boundary.accept_replan_result`` (unchanged) with
``SelectiveFullPathValidator`` as the injected validator, so actions and
expected positions come from the same boundary code Global uses. Every check
failure is a refusal (``accepted=False``, no plan). Nothing is ever repaired.

Full-check horizon: the whole padded plan (conservative). Whether only the
execution window should be checked is PENDING (contract OPEN_DECISION).
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Any, Collection, Mapping, Optional, Sequence

from src.hybrid.replan_boundary import (
    CBS_STATUSES,
    AcceptedPlan,
    BoundaryOutcome,
    accept_replan_result,
)
from src.hybrid.selective_safety import validate_full_robot
from src.path_guard import InvalidPathError, validate_path_endpoints, validate_path_movements

COMPOSE_RULE_VERSION = "selective-compose-v0"

Cell = tuple[int, int]

# goal_status values
GROUP_REACHES_GOAL = "GROUP_REACHES_GOAL"   # Path[-1] == goal, checked by Path Guard
OUTSIDE_HELD = "OUTSIDE_HELD"               # held at S_t; goal NOT evaluated

# refusal reason prefixes
R_AGENT_SET = "SELECTIVE_AGENT_SET"
R_S_T_OVERLAP = "SELECTIVE_S_T_OVERLAP"
R_GROUP_PATH = "SELECTIVE_GROUP_PATH"
R_FULL_CHECK = "SELECTIVE_FULL_CHECK"
R_OUTSIDE_NOT_HELD = "SELECTIVE_OUTSIDE_NOT_HELD"
R_PATHS_MISMATCH = "SELECTIVE_PATHS_MISMATCH"
R_GROUP_RESULT = "SELECTIVE_GROUP_RESULT"


@dataclass(frozen=True)
class ComposedPlan:
    ok: bool
    reason: str                                           # "" when ok
    group: tuple[int, ...] = ()
    outside: tuple[int, ...] = ()
    full_paths: Optional[dict[int, list[Cell]]] = None    # equal length, every agent
    goal_status: dict[int, str] = field(default_factory=dict)
    # True only if every agent's LAST cell equals its goal -- for a held
    # outside agent that means it already stood on its goal at S_t.
    ends_all_at_goal: bool = False
    rule_version: str = COMPOSE_RULE_VERSION


def _refuse(reason: str, group=(), outside=()) -> ComposedPlan:
    return ComposedPlan(ok=False, reason=reason, group=tuple(group), outside=tuple(outside))


def _id(v) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"agent id must be int, got {v!r}")
    return v


def _cell(v) -> Cell:
    if isinstance(v, (str, bytes)) or not isinstance(v, (tuple, list)) or len(v) != 2 \
            or any(isinstance(x, bool) for x in v):
        raise ValueError(f"cell must be (row, col), got {v!r}")
    return (operator.index(v[0]), operator.index(v[1]))


def _cell_map(d, label) -> dict[int, Cell]:
    if not isinstance(d, Mapping):
        raise ValueError(label)
    return {_id(a): _cell(p) for a, p in d.items()}


def compose_selective_plan(
    group_paths: Mapping[int, Sequence[Sequence[int]]],
    s_t: Mapping[int, Sequence[int]],
    goals: Mapping[int, Sequence[int]],
    agent_ids: Collection[int],
    grid,
) -> ComposedPlan:
    """Compose group paths with outside agents held at S_t and check the
    result for every agent. Never raises; a failure is ``ok=False`` with a
    reason and no paths."""
    # -- agent set ----------------------------------------------------------
    try:
        if isinstance(agent_ids, (str, bytes, Mapping)):
            raise ValueError
        ids = [_id(a) for a in agent_ids]
    except (TypeError, ValueError):
        return _refuse(f"{R_AGENT_SET}:agent_ids")
    if not ids or len(set(ids)) != len(ids):
        return _refuse(f"{R_AGENT_SET}:agent_ids")
    all_ids = set(ids)
    try:
        start = _cell_map(s_t, "s_t")
    except (TypeError, ValueError):
        return _refuse(f"{R_AGENT_SET}:s_t")
    if set(start) != all_ids:
        return _refuse(f"{R_AGENT_SET}:s_t")
    try:
        goal = _cell_map(goals, "goals")
    except (TypeError, ValueError):
        return _refuse(f"{R_AGENT_SET}:goals")
    if set(goal) != all_ids:
        return _refuse(f"{R_AGENT_SET}:goals")
    if not isinstance(group_paths, Mapping) or not group_paths:
        return _refuse(f"{R_AGENT_SET}:group_paths")
    try:
        group_ids = {_id(a) for a in group_paths}
    except ValueError:
        return _refuse(f"{R_AGENT_SET}:group_paths")
    if not group_ids <= all_ids:
        return _refuse(f"{R_AGENT_SET}:group_paths")
    group = tuple(sorted(group_ids))
    outside = tuple(sorted(all_ids - group_ids))
    if len(set(start.values())) != len(start):
        return _refuse(R_S_T_OVERLAP, group, outside)

    # -- group agents: the existing Path Guard goal rule, on the group only --
    try:
        norm = {}
        for a in group:
            p = group_paths[a]
            if isinstance(p, (str, bytes)) or not isinstance(p, Sequence):
                raise InvalidPathError(f"Agent {a}: malformed path")
            norm[a] = [_cell(c) for c in p]
        validate_path_endpoints(norm, {a: start[a] for a in group}, {a: goal[a] for a in group})
        validate_path_movements(norm)
    except (InvalidPathError, ValueError, TypeError) as exc:
        return _refuse(f"{R_GROUP_PATH}:{exc}", group, outside)

    # -- everyone: compose (outside held at S_t) + map / vertex / edge / length
    full = validate_full_robot(norm, start, sorted(all_ids), grid)
    if not full.ok:
        return _refuse(f"{R_FULL_CHECK}:{full.reason}", group, outside)
    paths = full.full_paths
    for a in outside:                         # composition invariant, never repaired
        if any(c != start[a] for c in paths[a]):
            return _refuse(f"{R_OUTSIDE_NOT_HELD}:{a}", group, outside)

    status = {a: GROUP_REACHES_GOAL for a in group}
    status.update({a: OUTSIDE_HELD for a in outside})
    return ComposedPlan(
        ok=True, reason="", group=group, outside=outside, full_paths=paths,
        goal_status=status,
        ends_all_at_goal=all(paths[a][-1] == goal[a] for a in all_ids),
    )


class SelectiveFullPathValidator:
    """``path_validator(grid, starts, goals, paths) -> (ok, reason)`` for the
    existing boundary. Accepts ``paths`` only if they are exactly what
    ``compose_selective_plan`` builds for ``group``: group agents per the
    Path Guard goal rule, every other agent held at its start. Not test_only."""

    def __init__(self, group: Collection[int]):
        self.group = tuple(sorted(group))

    def __call__(self, grid, starts, goals, paths):
        try:
            if not isinstance(paths, Mapping) or set(paths) != set(starts):
                return False, f"{R_AGENT_SET}:paths"
            if not set(self.group) <= set(paths):
                return False, f"{R_AGENT_SET}:group"
            for a in sorted(set(paths) - set(self.group)):
                if any(tuple(c) != tuple(starts[a]) for c in paths[a]):
                    return False, f"{R_OUTSIDE_NOT_HELD}:{a}"
            plan = compose_selective_plan({a: paths[a] for a in self.group}, starts, goals,
                                          sorted(starts), grid)
            if not plan.ok:
                return False, plan.reason
            given = {a: [tuple(c) for c in p] for a, p in paths.items()}
            if given != plan.full_paths:
                return False, R_PATHS_MISMATCH
        except Exception as exc:  # a validator must answer, never raise
            return False, f"{R_FULL_CHECK}:{type(exc).__name__}"
        return True, ""


@dataclass(frozen=True)
class _ComposedResult:
    """ReplanResult-shaped carrier for the boundary (status is the GROUP call's)."""
    status: str
    paths: Any
    latency_sec: Optional[float]
    error: Any = None


def selective_boundary_check(
    group_result: Any,
    *,
    s_t: Mapping[int, Sequence[int]],
    goals: Mapping[int, Sequence[int]],
    grid,
) -> tuple[BoundaryOutcome, Optional[ComposedPlan]]:
    """Group CBS result -> the existing boundary's verdict for ALL agents.

    ``group_result`` is the group call's ReplanResult (status, paths keyed by
    agent_id for the group agents only, latency_sec). A non-success status is
    refused with the boundary's own ``CBS_*`` reason; a composition failure
    is refused with its reason. On success the boundary (unchanged) converts
    every agent's moves -- outside agents get WAIT at every step."""
    agent_order = tuple(sorted(s_t))
    status = getattr(group_result, "status", None)
    if status not in CBS_STATUSES:
        return BoundaryOutcome(accepted=False, reason="CBS_UNKNOWN_STATUS"), None
    if status != "success":
        return BoundaryOutcome(accepted=False, reason=f"CBS_{status.upper()}"), None
    raw = getattr(group_result, "paths", None)
    if not isinstance(raw, Mapping):
        return BoundaryOutcome(accepted=False, reason=f"{R_GROUP_RESULT}:paths_not_agent_id_keyed"), None
    plan = compose_selective_plan(raw, s_t, goals, agent_order, grid)
    if not plan.ok:
        return BoundaryOutcome(accepted=False, reason=plan.reason), plan
    outcome = accept_replan_result(
        _ComposedResult("success", plan.full_paths, getattr(group_result, "latency_sec", None)),
        agent_order=agent_order,
        current_positions={a: tuple(s_t[a]) for a in agent_order},
        goals=goals,
        grid=grid,
        path_validator=SelectiveFullPathValidator(plan.group),
        paths_keying="agent_id",
    )
    return outcome, plan


def exec_step_mismatches(plan: AcceptedPlan, k: int, actual: Mapping[int, Sequence[int]]) -> tuple[int, ...]:
    """Agents whose ``actual`` position after executed step ``k`` (1-based)
    differs from ``plan.expected[k-1]`` -- outside agents included. Missing or
    extra agents count as mismatches. ``k`` outside 1..len(expected) raises."""
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= len(plan.expected):
        raise ValueError(f"k must be in 1..{len(plan.expected)}, got {k!r}")
    expected = plan.expected[k - 1]
    out = set(expected) ^ set(actual)
    for a in set(expected) & set(actual):
        if tuple(actual[a]) != tuple(expected[a]):
            out.add(a)
    return tuple(sorted(out))
