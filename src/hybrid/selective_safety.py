"""Selective Hybrid v0 safety checks and per-Trigger ladder decision — pure, DRAFT.

Status: DRAFT, rule ``selective-safety-v0``. Nothing in the runtime calls this
module: there is no Selective Runner and no ``group_selector``. Global Hybrid
(``src/hybrid/global_hybrid.py``) is unchanged and does not import it.
See ``docs/selective_hybrid_contract_v0.md`` (AGREED_DESIGN P1-P5).

What is here (only what the agreed design needs and existing code lacks):

1. ``l0_union_group``         conservative v0 L0: the union of all WFG groups
                              (agreed P2: independent groups -> union).
2. ``expansion_candidates``   outside agents that block a group's goal or a
                              group path cell at S_t (agreed P2 expansion
                              trigger). One level only; closure is OPEN.
3. ``validate_full_robot``    full-agent check of group paths + outside agents
                              held at S_t (agreed P4). Reuses Path Guard's
                              ``validate_path_movements`` and
                              ``validate_path_map_and_conflicts`` unchanged.
4. ``exec_mismatches``        execution-time expected/actual over ALL agents.
5. ``next_ladder_step``       one Trigger's L0/L1/G attempt + time ledger
                              (agreed P5): no_solution/timeout -> next level
                              while budget remains; invalid_path, full-check
                              failure -> SAFE_STOP at once; failed-call
                              latency counts; next timeout <= remaining time.

Holding outside agents at S_t (WAIT) is the CANDIDATE outside policy (P3). It
is NOT verified and the runtime cannot enforce it today. ``validate_full_robot``
checks the paths *under that hypothesis*; passing it does not make the policy
safe. Until it is verified, the only ladder to use at runtime is
``LADDER_UNTIL_OUTSIDE_POLICY_VERIFIED == ("G",)`` (Global first).

Coordinates are ``(row, col)`` runtime cells and agent IDs are runtime IDs.
This module never converts coordinates; ``src/cbs_adapter.py`` is the only
place that does. No number (timeout, budget, attempt count) has a default
here: the per-level allocation is an OPEN decision and callers must pass it.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Collection, Mapping, Optional, Sequence

from src.hybrid.replan_boundary import CBS_STATUSES
from src.path_guard import (
    InvalidPathError,
    validate_path_map_and_conflicts,
    validate_path_movements,
)

GridLocation = tuple[int, int]

SAFETY_RULE_VERSION = "selective-safety-v0"

LEVEL_GLOBAL = "G"
# Agreed P3: Global first until the outside-agent policy is verified safe.
LADDER_UNTIL_OUTSIDE_POLICY_VERIFIED = (LEVEL_GLOBAL,)

# next_ladder_step actions
ACTION_CALL = "CALL"
ACTION_EXECUTE = "EXECUTE"
ACTION_SAFE_STOP = "SAFE_STOP"

# Status classes (agreed P5 for the first two; the rest is OPEN)
RETRYABLE_STATUSES = ("no_solution", "timeout")
IMMEDIATE_STOP_STATUSES = ("invalid_path",)
# input_error, solver_error, unknown: no agreed rule. Handled like the current
# Global Hybrid (SAFE_STOP), reason prefixed UNAGREED_STATUS. Provisional.


# -- 1/2. group selection (WFG result -> candidate groups) ---------------------

def l0_union_group(groups: Sequence[Sequence[int]]) -> tuple[int, ...]:
    """Sorted union of all WFG groups (conservative v0 L0).

    This is a *selection* only. It says nothing about whether a CBS call for
    the union will succeed or be safe to execute.
    """
    out: set[int] = set()
    for g in groups:
        for a in g:
            if isinstance(a, bool) or not isinstance(a, int):
                raise ValueError(f"agent id must be int, got {a!r}")
            out.add(a)
    return tuple(sorted(out))


def _cell(value, label: str) -> GridLocation:
    if not isinstance(value, (tuple, list)) or len(value) != 2 or any(isinstance(v, bool) for v in value):
        raise ValueError(f"{label}: cell must be (row, col), got {value!r}")
    try:
        return (operator.index(value[0]), operator.index(value[1]))
    except TypeError:
        raise ValueError(f"{label}: cell must hold ints, got {value!r}") from None


@dataclass(frozen=True, order=True)
class ExpansionHit:
    """``kind`` is GOAL_BLOCKED (outside agent stands on a group goal at S_t)
    or PATH_BLOCKED (outside agent stands on a cell of a group path at S_t)."""

    kind: str
    group_agent: int
    outside_agent: int
    cell: GridLocation


def expansion_candidates(
    group: Collection[int],
    s_t: Mapping[int, Sequence[int]],
    goals: Mapping[int, Sequence[int]],
    group_paths: Optional[Mapping[int, Sequence[Sequence[int]]]] = None,
) -> tuple[tuple[ExpansionHit, ...], tuple[int, ...]]:
    """Return (hits, sorted outside agents to add) for ONE expansion level.

    Positions are S_t (after the step's moves), because these are the cells
    the planner must work around. Whether to repeat until no new hit appears
    (closure) is OPEN; callers decide.
    """
    members = set(group)
    pos = {a: _cell(p, f"s_t[{a}]") for a, p in s_t.items()}
    if not members <= set(pos):
        raise ValueError("group has agents missing from s_t")
    on_cell: dict[GridLocation, int] = {}
    for a in sorted(pos):
        if a not in members:
            on_cell[pos[a]] = a
    hits: set[ExpansionHit] = set()
    for a in sorted(members):
        goal = _cell(goals[a], f"goals[{a}]")
        if goal in on_cell:
            hits.add(ExpansionHit("GOAL_BLOCKED", a, on_cell[goal], goal))
        if group_paths is not None and a in group_paths:
            for c in group_paths[a]:
                c = _cell(c, f"group_paths[{a}]")
                if c in on_cell:
                    hits.add(ExpansionHit("PATH_BLOCKED", a, on_cell[c], c))
    ordered = tuple(sorted(hits))
    return ordered, tuple(sorted({h.outside_agent for h in ordered}))


# -- 3/4. full-robot validation -------------------------------------------------

@dataclass(frozen=True)
class FullCheckResult:
    ok: bool
    reason: str                                   # "" when ok
    full_paths: Optional[dict[int, list[GridLocation]]] = None


def _fail(reason: str) -> FullCheckResult:
    return FullCheckResult(ok=False, reason=reason)


def validate_full_robot(
    group_paths: Mapping[int, Sequence[Sequence[int]]],
    s_t: Mapping[int, Sequence[int]],
    agent_ids: Collection[int],
    grid,
) -> FullCheckResult:
    """Check a group plan against EVERY agent before execution.

    Outside agents (``agent_ids`` minus the group) are held at S_t for the
    whole horizon (candidate WAIT policy, P3, unverified). Group paths are
    padded with their last cell (same convention as ``cbs_adapter.pad_paths``).
    The composed plan then goes through Path Guard's movement and
    map/vertex/edge checks, unchanged. Never raises; failures are reasons:

    - ``FULL_CHECK_AGENT_SET:<detail>``  ID loss/extra (group, S_t, agent_ids)
    - ``FULL_CHECK_MALFORMED:<agent>``   empty or non-(row, col) path
    - ``FULL_CHECK_START_MISMATCH:<agent>`` Path[0] != S_t
    - ``FULL_CHECK_REJECT:<Path Guard message>`` move, bounds, obstacle,
      vertex or edge conflict (message names the agents)
    """
    try:
        ids = [a for a in agent_ids]
    except TypeError:
        return _fail("FULL_CHECK_AGENT_SET:agent_ids")
    if not ids or any(isinstance(a, bool) or not isinstance(a, int) for a in ids) or len(set(ids)) != len(ids):
        return _fail("FULL_CHECK_AGENT_SET:agent_ids")
    all_ids = set(ids)
    if not isinstance(s_t, Mapping) or set(s_t) != all_ids:
        return _fail("FULL_CHECK_AGENT_SET:s_t")
    if not isinstance(group_paths, Mapping) or not group_paths:
        return _fail("FULL_CHECK_AGENT_SET:group_paths")
    if not set(group_paths) <= all_ids:
        return _fail("FULL_CHECK_AGENT_SET:group_paths")
    try:
        start = {a: _cell(p, "s_t") for a, p in s_t.items()}
    except ValueError:
        return _fail("FULL_CHECK_AGENT_SET:s_t")

    norm: dict[int, list[GridLocation]] = {}
    for a in sorted(group_paths):
        p = group_paths[a]
        if isinstance(p, (str, bytes)) or not isinstance(p, Sequence) or len(p) == 0:
            return _fail(f"FULL_CHECK_MALFORMED:{a}")
        try:
            norm[a] = [_cell(c, "path") for c in p]
        except ValueError:
            return _fail(f"FULL_CHECK_MALFORMED:{a}")
        if norm[a][0] != start[a]:
            return _fail(f"FULL_CHECK_START_MISMATCH:{a}")

    length = max(len(p) for p in norm.values())
    full: dict[int, list[GridLocation]] = {}
    for a in sorted(all_ids):
        if a in norm:
            p = norm[a]
            full[a] = p + [p[-1]] * (length - len(p))
        else:
            full[a] = [start[a]] * length
    try:
        validate_path_movements(full)
        validate_path_map_and_conflicts(full, grid)
    except (InvalidPathError, ValueError, TypeError, IndexError) as exc:
        return _fail(f"FULL_CHECK_REJECT:{exc}")
    return FullCheckResult(ok=True, reason="", full_paths=full)


def exec_mismatches(
    full_paths: Mapping[int, Sequence[GridLocation]],
    k: int,
    actual: Mapping[int, Sequence[int]],
) -> tuple[int, ...]:
    """Agents whose actual position after execution step k differs from the
    full plan's Path[k] (outside agents included). Missing or extra agents in
    ``actual`` count as mismatches."""
    out = set(full_paths) ^ set(actual)
    for a in set(full_paths) & set(actual):
        p = full_paths[a]
        if tuple(actual[a]) != tuple(p[min(k, len(p) - 1)]):
            out.add(a)
    return tuple(sorted(out))


# -- 5. per-Trigger ladder + budget -------------------------------------------

@dataclass(frozen=True)
class TriggerBudget:
    """One Trigger's shared budget for ALL levels (L0, L1, ..., G).

    No defaults: the values are an OPEN decision (contract OPEN_DECISION).
    ``level_timeout_ms`` is the timeout each level would ask for; the call
    actually gets ``min(requested, remaining)``.
    """

    total_time_ms: float
    max_attempts: int
    level_timeout_ms: Mapping[str, float]

    def __post_init__(self):
        if isinstance(self.total_time_ms, bool) or not isinstance(self.total_time_ms, (int, float)) \
                or not math.isfinite(self.total_time_ms) or self.total_time_ms <= 0:
            raise ValueError("total_time_ms must be a finite number > 0")
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int) or self.max_attempts < 1:
            raise ValueError("max_attempts must be an int >= 1")
        for level, v in self.level_timeout_ms.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                raise ValueError(f"level_timeout_ms[{level!r}] must be a finite number > 0")


@dataclass(frozen=True)
class Attempt:
    """One finished CBS call. ``latency_ms`` is the measured wall time, also
    for failed calls; None means unknown. ``full_check_ok`` is required for a
    ``success`` (the result of ``validate_full_robot``) and ignored otherwise.
    Selection (``group``) and call outcome (``status``) are kept apart."""

    level: str
    group: tuple[int, ...]
    status: str
    latency_ms: Optional[float]
    full_check_ok: Optional[bool] = None


@dataclass(frozen=True)
class LadderStep:
    action: str                      # CALL / EXECUTE / SAFE_STOP
    level: Optional[str]             # level to call or execute
    timeout_ms: Optional[float]      # only for CALL
    reason: str
    remaining_ms: float
    attempts_used: int


def remaining_time_ms(budget: TriggerBudget, attempts: Sequence[Attempt]) -> float:
    """Total minus the summed latency of every attempt, failed ones included.
    An unknown latency makes the remainder unknown -> 0.0 (same stance as the
    Global rearm guard, which refuses a call when a latency is missing)."""
    spent = 0.0
    for at in attempts:
        if at.latency_ms is None:
            return 0.0
        spent += float(at.latency_ms)
    return max(0.0, float(budget.total_time_ms) - spent)


def _check_ladder(ladder: Sequence[str], budget: TriggerBudget) -> tuple[str, ...]:
    lad = tuple(ladder)
    if not lad or len(set(lad)) != len(lad) or lad[-1] != LEVEL_GLOBAL:
        raise ValueError("ladder must be non-empty, unique and end with 'G'")
    missing = [lv for lv in lad if lv not in budget.level_timeout_ms]
    if missing:
        raise ValueError(f"no requested timeout for levels {missing}")
    return lad


def next_ladder_step(
    ladder: Sequence[str],
    attempts: Sequence[Attempt],
    budget: TriggerBudget,
) -> LadderStep:
    """Decide what one Trigger does next, given the attempts made so far.

    Rules (agreed P5 unless marked):
    - success + full check ok        -> EXECUTE that level's plan
    - success + full check failed    -> SAFE_STOP (FULL_CHECK_FAILED)
    - invalid_path                   -> SAFE_STOP at once, budget irrelevant
    - no_solution / timeout          -> next level if one is left, attempts
                                        remain and time remains; else SAFE_STOP
    - input_error / solver_error / unknown -> SAFE_STOP (UNAGREED_STATUS,
                                        provisional = current Global behaviour)
    The next level is simply the next ladder entry. Skipping straight to G
    when time is short is an OPEN decision and is not done here.
    """
    lad = _check_ladder(ladder, budget)
    used = len(attempts)
    remaining = remaining_time_ms(budget, attempts)

    def stop(reason):
        return LadderStep(ACTION_SAFE_STOP, None, None, reason, remaining, used)

    def call(level, reason):
        if used >= budget.max_attempts:
            return stop("ATTEMPT_LIMIT")
        if remaining <= 0:
            return stop("TIME_BUDGET_EXHAUSTED")
        timeout = min(float(budget.level_timeout_ms[level]), remaining)
        return LadderStep(ACTION_CALL, level, timeout, reason, remaining, used)

    if not attempts:
        return call(lad[0], "FIRST_LEVEL")

    last = attempts[-1]
    if last.level not in lad:
        raise ValueError(f"attempt level {last.level!r} not in ladder")
    if [at.level for at in attempts] != list(lad[:used]):
        raise ValueError("attempts must follow the ladder order, one per level")

    if last.status == "success":
        if last.full_check_ok is None:
            raise ValueError("a success attempt needs full_check_ok")
        if last.full_check_ok:
            return LadderStep(ACTION_EXECUTE, last.level, None, "SUCCESS_FULL_CHECK_OK", remaining, used)
        return stop("FULL_CHECK_FAILED")
    if last.status in IMMEDIATE_STOP_STATUSES:
        return stop(f"CBS_{last.status.upper()}")
    if last.status in RETRYABLE_STATUSES:
        idx = lad.index(last.level)
        if idx + 1 >= len(lad):
            return stop(f"LADDER_EXHAUSTED:CBS_{last.status.upper()}")
        return call(lad[idx + 1], f"AFTER_CBS_{last.status.upper()}")
    known = last.status if last.status in CBS_STATUSES else "unknown"
    return stop(f"UNAGREED_STATUS:{known}")
