"""Selective-vs-Global routing decision for one Trigger — pure function, DRAFT.

Status: DRAFT, rule ``selective-decision-v0``. Nothing calls this module: there
is no Selective Runner and no ``group_selector``. Global Hybrid is unchanged.
See ``docs/selective_hybrid_contract_v0.md`` §3.3-3.6 and §4 (P1-P5).

What it decides
---------------
Given S_t, goals, the full agent set and a candidate WFG result
(``src.hybrid.wfg.build_wfg``), it decides whether a Selective attempt could
even be considered. It also reports, as data, why a partial group cannot
be executed yet.

**v0 never routes to Selective execution.** Every outcome has ``route ==
"GLOBAL"``, meaning that the existing Global Hybrid CBS_REQUEST for all agents
runs exactly as today. The Global Hybrid's own gates (overlap integrity I-2/I-3,
recall limits, boundary, Path Guard, SAFE_STOP) are not touched or bypassed.
A partial group is only *marked* as a Selective candidate.

Why execution is always blocked in v0: the following are unverified, so outside-
agent safety cannot be established (contract §3.4-3.5). They are reported as
fixed blockers, not parameters, so that no caller can switch them on before
the policy is agreed:

- P1 ``P1_TEMPORAL_RESERVATION_UNVERIFIED``  atb033 time-indexed reservations
- P2 ``P2_OUTSIDE_AGENT_CONTROL_UNVERIFIED`` safe behaviour of outside agents
- P3 ``P3_FULL_ROBOT_VALIDATION_UNAVAILABLE`` group+outside joint validation
- P4 ``P4_BUDGET_UNDEFINED``                 CBS call/time budget per level
- P5 ``P5_GROUP_RULE_UNAGREED``              WFG format / expansion rule

Reasons (``SelectiveDecision.reason``)
-------------------------------------
- ``FULL_GROUP``             one candidate group == all agents -> Global is the
                             natural path (not a fallback).
- ``NO_GROUP``               WFG produced no group -> Global fallback.
- ``SELECTIVE_CANDIDATE_BLOCKED`` a partial group, or several groups ->
                             candidate only; Global fallback with ``blockers``.
- ``INSUFFICIENT_INPUT:<cause>`` input missing; Selective is not evaluated.
- ``INVALID_INPUT:<cause>``  input inconsistent; Selective is not evaluated.
- ``INVALID_CANDIDATE_PATHS:<cause>`` the optional mock group paths are
                             malformed; Selective is not evaluated.

Facts computed for a candidate (``findings``), without running CBS:

- ``OUTSIDE_OCCUPIES_GROUP_GOAL``  an outside agent stands on a group goal
- ``GROUP_OCCUPIES_OUTSIDE_GOAL``  a group agent stands on an outside goal
- ``MULTIPLE_GROUPS``              more than one group; union vs per-group
                                   CBS calls are undecided (P4/P5)
- ``GROUPS_COVER_ALL_AGENTS``      the union of groups is every agent
- ``GROUP_DEPENDS_ON_PENDING_T7``  a group is connected only through an
                                   ``occupy`` edge whose occupant moves (T7)
- ``OUTSIDE_VERTEX_CONFLICT`` / ``OUTSIDE_EDGE_CONFLICT`` (only with
  ``candidate_paths``) the group paths collide with outside agents under the
  hypothesis that outside agents hold still. A clean result proves nothing:
  the hold-still behaviour itself is unverified (P2). Under that hypothesis an
  outside agent never moves, so ``OUTSIDE_EDGE_CONFLICT`` cannot fire; the
  code exists so that a later outside-motion model reuses the same labels.
  Walls are not checked here (Path Guard's job; no grid input).
- ``GROUP_INTERNAL_*_CONFLICT`` / ``CROSS_GROUP_*_CONFLICT`` the mock paths
  collide inside one group / between two independent groups.

Every output is a sorted tuple, so equal inputs give equal outputs.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Collection, Mapping, Optional, Sequence

from src.hybrid.wfg import RULE_VERSION as WFG_RULE_VERSION
from src.hybrid.wfg import WfgResult

GridLocation = tuple[int, int]

DECISION_RULE_VERSION = "selective-decision-v0"
ROUTE_GLOBAL = "GLOBAL"

FULL_GROUP = "FULL_GROUP"
NO_GROUP = "NO_GROUP"
SELECTIVE_CANDIDATE_BLOCKED = "SELECTIVE_CANDIDATE_BLOCKED"
INSUFFICIENT_INPUT = "INSUFFICIENT_INPUT"
INVALID_INPUT = "INVALID_INPUT"
INVALID_CANDIDATE_PATHS = "INVALID_CANDIDATE_PATHS"

PENDING_BLOCKERS = (
    "P1_TEMPORAL_RESERVATION_UNVERIFIED",
    "P2_OUTSIDE_AGENT_CONTROL_UNVERIFIED",
    "P3_FULL_ROBOT_VALIDATION_UNAVAILABLE",
    "P4_BUDGET_UNDEFINED",
    "P5_GROUP_RULE_UNAGREED",
)


@dataclass(frozen=True)
class SelectiveDecision:
    route: str                                   # always "GLOBAL" in v0
    reason: str                                  # see module docstring
    fallback: bool                               # False only for FULL_GROUP
    selective_candidate: bool                    # partial group(s) found
    candidate_groups: tuple[tuple[int, ...], ...]
    outside_agents: tuple[int, ...]
    findings: tuple[tuple, ...]                  # (code, *details), sorted
    blockers: tuple[str, ...]                    # why Selective cannot run
    rule_version: str = DECISION_RULE_VERSION


class _Reject(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _id(v) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise _Reject(f"{INVALID_INPUT}:AGENT_ID_TYPE")
    return v


def _cell(v, reason: str) -> GridLocation:
    if not isinstance(v, (tuple, list)) or len(v) != 2 or any(isinstance(x, bool) for x in v):
        raise _Reject(reason)
    try:
        return (operator.index(v[0]), operator.index(v[1]))
    except TypeError:
        raise _Reject(reason) from None


def _result(reason, *, groups=(), outside=(), findings=(), candidate=False, fallback=True,
            blockers=()) -> SelectiveDecision:
    return SelectiveDecision(
        route=ROUTE_GLOBAL,
        reason=reason,
        fallback=fallback,
        selective_candidate=candidate,
        candidate_groups=tuple(groups),
        outside_agents=tuple(outside),
        findings=tuple(sorted(set(findings))),
        blockers=tuple(blockers),
    )


def _check_wfg(wfg, ids: set[int]) -> tuple[tuple[int, ...], ...]:
    if not isinstance(wfg, WfgResult):
        raise _Reject(f"{INVALID_INPUT}:WFG_TYPE")
    if wfg.rule_version != WFG_RULE_VERSION:
        raise _Reject(f"{INVALID_INPUT}:WFG_RULE_VERSION")
    try:
        groups = tuple(tuple(g) for g in wfg.groups)
        edges = tuple(wfg.edges)
    except TypeError:
        raise _Reject(f"{INVALID_INPUT}:WFG_GROUP_SHAPE") from None
    seen: set[int] = set()
    for g in groups:
        for a in g:
            _id(a)
        if len(g) < 2 or list(g) != sorted(set(g)):
            raise _Reject(f"{INVALID_INPUT}:WFG_GROUP_SHAPE")
        for a in g:
            if a not in ids:
                raise _Reject(f"{INVALID_INPUT}:WFG_UNKNOWN_AGENT")
            if a in seen:
                raise _Reject(f"{INVALID_INPUT}:WFG_GROUPS_OVERLAP")
            seen.add(a)
    if list(groups) != sorted(groups, key=lambda g: g[0]):
        raise _Reject(f"{INVALID_INPUT}:WFG_GROUP_ORDER")
    member = {a: i for i, g in enumerate(groups) for a in g}
    for e in edges:
        if _id(e.src) not in ids or _id(e.dst) not in ids:
            raise _Reject(f"{INVALID_INPUT}:WFG_UNKNOWN_AGENT")
        # An edge never crosses a group boundary (groups are components).
        if member.get(e.src) != member.get(e.dst):
            raise _Reject(f"{INVALID_INPUT}:WFG_EDGE_CROSSES_GROUP")
    return groups


def _depends_on_t7(wfg: WfgResult, group: tuple[int, ...]) -> bool:
    """True if the group is not connected without moving-occupant occupy edges."""
    parent = {a: a for a in group}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in wfg.edges:
        if e.src in parent and not (e.kind == "occupy" and e.occupant_moves):
            parent[find(e.src)] = find(e.dst)
    return len({find(a) for a in group}) > 1


def _check_paths(paths, positions, group_ids: set[int]):
    if not isinstance(paths, Mapping):
        raise _Reject(f"{INVALID_CANDIDATE_PATHS}:TYPE")
    norm: dict[int, list[GridLocation]] = {}
    for a, p in paths.items():
        a = _id(a)
        if isinstance(p, (str, bytes)) or not isinstance(p, Sequence) or len(p) == 0:
            raise _Reject(f"{INVALID_CANDIDATE_PATHS}:EMPTY_OR_MALFORMED")
        norm[a] = [_cell(c, f"{INVALID_CANDIDATE_PATHS}:EMPTY_OR_MALFORMED") for c in p]
    if set(norm) != group_ids:
        raise _Reject(f"{INVALID_CANDIDATE_PATHS}:AGENT_SET")
    for a, p in norm.items():
        if p[0] != positions[a]:
            raise _Reject(f"{INVALID_CANDIDATE_PATHS}:START_MISMATCH")
        for x, y in zip(p, p[1:]):
            if abs(x[0] - y[0]) + abs(x[1] - y[1]) > 1:
                raise _Reject(f"{INVALID_CANDIDATE_PATHS}:NON_ADJACENT")
    return norm


def _path_findings(norm, positions, groups, outside: tuple[int, ...]) -> list[tuple]:
    """Collisions under the HYPOTHESIS that outside agents hold still (P2).

    Shorter paths are padded with their last cell (same convention as
    ``cbs_adapter.pad_paths``). Pair labels: same group -> GROUP_INTERNAL,
    different groups -> CROSS_GROUP, one side outside -> OUTSIDE.
    """
    horizon = max(len(p) for p in norm.values())
    group_of = {a: i for i, g in enumerate(groups) for a in g}

    def at(a, k):
        if a in norm:
            p = norm[a]
            return p[min(k, len(p) - 1)]
        return positions[a]

    everyone = sorted(set(norm) | set(outside))
    out: list[tuple] = []
    for k in range(horizon):
        for i, a in enumerate(everyone):
            for b in everyone[i + 1:]:
                if a not in norm and b not in norm:
                    continue
                if a in norm and b in norm:
                    kind = "GROUP_INTERNAL" if group_of[a] == group_of[b] else "CROSS_GROUP"
                else:
                    kind = "OUTSIDE"
                if at(a, k) == at(b, k):
                    out.append((f"{kind}_VERTEX_CONFLICT", a, b, k))
                if (k + 1 < horizon and at(a, k) == at(b, k + 1) and at(b, k) == at(a, k + 1)
                        and at(a, k) != at(a, k + 1)):
                    out.append((f"{kind}_EDGE_CONFLICT", a, b, k))
    return out


def decide_selective(
    agent_ids: Collection[int],
    positions: Mapping[int, Sequence[int]],
    goals: Optional[Mapping[int, Sequence[int]]],
    wfg: Optional[WfgResult],
    candidate_paths: Optional[Mapping[int, Sequence[Sequence[int]]]] = None,
) -> SelectiveDecision:
    """Return the routing decision for one Trigger. Never raises on bad input:
    bad input yields ``route="GLOBAL"`` with an ``INVALID_*``/``INSUFFICIENT_*``
    reason, so the caller's existing Global path and its safety gates apply.
    This is a provisional choice: when invalid input should instead lead to
    SAFE_STOP is not agreed yet (contract §3.3.1, PENDING).

    ``candidate_paths`` is an optional MOCK of a group CBS result
    (``{agent_id: [(row, col), ...]}``, path[0] = S_t). It is used only to
    report collision findings; no CBS is run here.
    """
    try:
        if agent_ids is None:
            raise _Reject(f"{INSUFFICIENT_INPUT}:AGENT_IDS_MISSING")
        if isinstance(agent_ids, (str, bytes, Mapping)):
            raise _Reject(f"{INVALID_INPUT}:AGENT_IDS_TYPE")
        try:
            ids_list = [_id(a) for a in agent_ids]
        except TypeError:
            raise _Reject(f"{INVALID_INPUT}:AGENT_IDS_TYPE") from None
        if not ids_list:
            raise _Reject(f"{INSUFFICIENT_INPUT}:AGENT_IDS_MISSING")
        if len(set(ids_list)) != len(ids_list):
            raise _Reject(f"{INVALID_INPUT}:DUPLICATE_AGENT_ID")
        ids = set(ids_list)
        if positions is None:
            raise _Reject(f"{INSUFFICIENT_INPUT}:POSITIONS_MISSING")
        if not isinstance(positions, Mapping):
            raise _Reject(f"{INVALID_INPUT}:POSITIONS_TYPE")
        pos = {_id(a): _cell(p, f"{INVALID_INPUT}:POSITION_SHAPE") for a, p in positions.items()}
        if set(pos) != ids:
            raise _Reject(f"{INVALID_INPUT}:POSITIONS_AGENT_SET")
        if len(set(pos.values())) != len(pos):
            raise _Reject(f"{INVALID_INPUT}:POSITIONS_OVERLAP")
        if wfg is None:
            raise _Reject(f"{INSUFFICIENT_INPUT}:WFG_MISSING")
        groups = _check_wfg(wfg, ids)
    except _Reject as r:
        return _result(r.reason)

    all_ids = tuple(sorted(ids))
    if not groups:
        return _result(NO_GROUP)
    if len(groups) == 1 and groups[0] == all_ids:
        return _result(FULL_GROUP, groups=groups, fallback=False)

    in_group = {a for g in groups for a in g}
    outside = tuple(a for a in all_ids if a not in in_group)
    try:
        if goals is None:
            raise _Reject(f"{INSUFFICIENT_INPUT}:GOALS_MISSING")
        if not isinstance(goals, Mapping):
            raise _Reject(f"{INVALID_INPUT}:GOALS_TYPE")
        gl = {_id(a): _cell(p, f"{INVALID_INPUT}:GOAL_SHAPE") for a, p in goals.items()}
        if set(gl) != ids:
            raise _Reject(f"{INVALID_INPUT}:GOALS_AGENT_SET")
        norm = None if candidate_paths is None else _check_paths(candidate_paths, pos, in_group)
    except _Reject as r:
        return _result(r.reason, groups=groups, outside=outside, candidate=True,
                       blockers=PENDING_BLOCKERS)

    findings: list[tuple] = []
    if len(groups) > 1:
        findings.append(("MULTIPLE_GROUPS", len(groups)))
    if not outside:
        findings.append(("GROUPS_COVER_ALL_AGENTS",))
    for g in groups:
        if _depends_on_t7(wfg, g):
            findings.append(("GROUP_DEPENDS_ON_PENDING_T7", g))
        # "Outside" relative to this group: every agent not in it (outside
        # agents and members of other groups alike).
        others = [b for b in all_ids if b not in g]
        for a in g:
            for b in others:
                if pos[b] == gl[a]:
                    findings.append(("OUTSIDE_OCCUPIES_GROUP_GOAL", a, b))
                if pos[a] == gl[b]:
                    findings.append(("GROUP_OCCUPIES_OUTSIDE_GOAL", a, b))
    if norm is not None:
        findings.extend(_path_findings(norm, pos, groups, outside))

    return _result(SELECTIVE_CANDIDATE_BLOCKED, groups=groups, outside=outside,
                   findings=findings, candidate=True, blockers=PENDING_BLOCKERS)
