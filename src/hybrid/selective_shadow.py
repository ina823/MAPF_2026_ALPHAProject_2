"""Selective Hybrid SHADOW mode — observe only, never steer. DRAFT, Mock-verified.

Status: rule ``selective-shadow-v0``. Not wired into any runtime: no Runner,
``common_runner`` or script constructs these classes. They are meant to be
*injected* into ``run_global_hybrid_episode(planner=..., monitor=...)``.

While Global Hybrid controls every robot, the shadow records what the
Selective pipeline WOULD have selected at each CBS_REQUEST:

    build_wfg -> decide_selective -> L0 union group -> expansion candidates
    -> precheck (goal blocked / goal unreachable with outside agents held)

and nothing else. It never runs a Selective CBS, never injects WAIT, never
changes the grid passed to the planner, and never changes what the planner
receives or returns.

``ShadowMonitor(inner)``
    Forwards ``observe_step`` to ``inner`` exactly once, with the same
    arguments, and returns its result object unchanged. Afterwards it copies
    S_{t-1} (``current_positions``), ``intended_positions``, the conflict
    types, blocked agents and S_t (``positions``) into ``steps``. ``reset`` and
    ``rearm`` are forwarded once each; ``reset`` starts a new shadow episode
    (records of the previous episode are never reused). Everything else
    (``result``, ``trigger_history``, ``rearm_count`` ...) is delegated.

``ShadowSelectivePlanner(inner, monitor)``
    ``paths_keying`` / ``test_only`` are read from ``inner`` with the Runner's
    own defaults, so ``check_validator_pairing`` and the Runner's keying see
    exactly what they would see for ``inner``. On each call it first builds a
    shadow record inside a broad ``try`` (any failure -> ``shadow_error``),
    then calls ``inner`` once with the original argument objects and returns
    its result object as-is. An exception raised by ``inner`` propagates
    untouched (the Runner's PLANNER_EXCEPTION path stays intact).

Time axis (contract P1-c/d): the WFG uses the Trigger step's S_{t-1} and
intended; decision, group and precheck use S_t (the CBS start state).

Precheck uses a PRIVATE numpy copy of the grid with outside agents' S_t cells
marked, only to run BFS. The grid object handed to the planner is untouched.
Nothing here means Selective is safe or enabled: ``decide_selective`` still
routes GLOBAL in v0 and P1-P5 remain as recorded in the contract.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Optional

import numpy as np

from src.hybrid.selective_decision import SelectiveDecision, decide_selective
from src.hybrid.selective_safety import ExpansionHit, expansion_candidates, l0_union_group
from src.hybrid.wfg import WfgResult, build_wfg
from src.il.nav_hint import bfs_dist

SHADOW_RULE_VERSION = "selective-shadow-v0"

Cell = tuple[int, int]

# shadow_error codes (the shadow failed; Global behaviour is unaffected)
ERR_NO_STEP = "NO_STEP_OBSERVED"
ERR_STALE_STEP = "STALE_STEP"              # no new step since the previous record
ERR_NO_GOALS = "GOALS_UNKNOWN"
ERR_NO_INTENT = "INTENT_NOT_OBSERVED"      # observe_step got no current/intended kwargs
ERR_S_T_MISMATCH = "S_T_MISMATCH"          # planner args != last observed S_t
ERR_EXCEPTION = "EXCEPTION"                # "EXCEPTION:<Type>:<msg>"


def _cells(d: Optional[Mapping]) -> Optional[dict[int, Cell]]:
    if d is None:
        return None
    return {int(a): (int(p[0]), int(p[1])) for a, p in d.items()}


@dataclass(frozen=True)
class ShadowStep:
    episode: int
    t: int
    s_t_minus_1: Optional[dict[int, Cell]]
    intended: Optional[dict[int, Cell]]
    conflict_types: dict[int, str]
    blocked: tuple[int, ...]
    s_t: dict[int, Cell]
    trigger_reason_code: Optional[str]
    trigger_timestep: Optional[int]

    @property
    def fresh_trigger(self) -> bool:
        return self.trigger_reason_code == "TRIGGER_FULL_REPEAT" and self.trigger_timestep == self.t


@dataclass(frozen=True)
class ShadowPrecheck:
    goal_blocked: tuple[int, ...]          # L0 agents whose goal an outside agent occupies
    unreachable: tuple[int, ...]           # L0 agents with BFS goal distance -1 (outside held)


@dataclass(frozen=True)
class ShadowRecord:
    episode: int
    call_index: int                        # 1-based within the episode
    trigger_t: Optional[int]
    s_t_minus_1: Optional[dict[int, Cell]]
    intended: Optional[dict[int, Cell]]
    blocked: tuple[int, ...]
    s_t: Optional[dict[int, Cell]]
    wfg: Optional[WfgResult]
    decision: Optional[SelectiveDecision]
    l0_group: tuple[int, ...]
    outside: tuple[int, ...]
    expansion_hits: tuple[ExpansionHit, ...]
    expansion_add: tuple[int, ...]
    precheck: Optional[ShadowPrecheck]
    shadow_error: Optional[str]
    global_status: Optional[str] = None    # filled after the planner returns
    rule_version: str = SHADOW_RULE_VERSION


class ShadowMonitor:
    """Transparent DeadlockMonitor wrapper that records each observed step."""

    def __init__(self, inner):
        self.inner = inner
        self.episode = 0
        self.goals: Optional[dict[int, Cell]] = None
        self.steps: list[ShadowStep] = []
        self.rearms: list[tuple[int, int]] = []     # (episode, t)
        self.record_errors: list[str] = []

    def reset(self, initial_positions, goals):
        out = self.inner.reset(initial_positions, goals)
        self.episode += 1
        self.steps = []
        self.rearms = []
        try:
            self.goals = _cells(goals)
        except Exception as exc:  # recording only
            self.goals = None
            self.record_errors.append(f"{ERR_EXCEPTION}:{type(exc).__name__}:{exc}")
        return out

    def rearm(self, current_positions, current_timestep):
        out = self.inner.rearm(current_positions, current_timestep)
        self.rearms.append((self.episode, current_timestep))
        return out

    def observe_step(self, timestep, positions, conflict_types, **kwargs):
        result = self.inner.observe_step(timestep, positions, conflict_types, **kwargs)
        try:
            ct = {int(a): str(c) for a, c in conflict_types.items()}
            self.steps.append(ShadowStep(
                episode=self.episode,
                t=int(timestep),
                s_t_minus_1=_cells(kwargs.get("current_positions")),
                intended=_cells(kwargs.get("intended_positions")),
                conflict_types=ct,
                blocked=tuple(sorted(a for a, c in ct.items() if c != "none")),
                s_t=_cells(positions),
                trigger_reason_code=getattr(result, "trigger_reason_code", None),
                trigger_timestep=getattr(result, "trigger_timestep", None),
            ))
        except Exception as exc:  # recording only; never alter the Monitor path
            self.record_errors.append(f"{ERR_EXCEPTION}:{type(exc).__name__}:{exc}")
        return result

    @property
    def last_step(self) -> Optional[ShadowStep]:
        return self.steps[-1] if self.steps else None

    def __getattr__(self, name):
        # Only reached for attributes not defined on the wrapper.
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)


class ShadowSelectivePlanner:
    """Global planner wrapper that records a Selective shadow analysis."""

    def __init__(self, inner, monitor: ShadowMonitor):
        self.inner = inner
        self.monitor = monitor
        self.records: list[ShadowRecord] = []

    # Same values the Runner would read from ``inner`` (same defaults).
    @property
    def paths_keying(self):
        return getattr(self.inner, "paths_keying", "index")

    @property
    def test_only(self):
        return getattr(self.inner, "test_only", False)

    def __call__(self, current_positions, goals, grid):
        try:
            record = self._analyse(current_positions, grid)
        except Exception as exc:
            record = self._empty(f"{ERR_EXCEPTION}:{type(exc).__name__}:{exc}")
        self.records.append(record)
        result = self.inner(current_positions, goals, grid)   # exceptions propagate
        try:
            status = getattr(result, "status", None)
            self.records[-1] = replace(record, global_status=None if status is None else str(status))
        except Exception:
            pass
        return result

    # -- internals ---------------------------------------------------------

    def _episode_records(self):
        return [r for r in self.records if r.episode == self.monitor.episode]

    def _empty(self, error, step: Optional[ShadowStep] = None) -> ShadowRecord:
        return ShadowRecord(
            episode=self.monitor.episode,
            call_index=len(self._episode_records()) + 1,
            trigger_t=None if step is None else step.t,
            s_t_minus_1=None if step is None else step.s_t_minus_1,
            intended=None if step is None else step.intended,
            blocked=() if step is None else step.blocked,
            s_t=None if step is None else step.s_t,
            wfg=None, decision=None, l0_group=(), outside=(),
            expansion_hits=(), expansion_add=(), precheck=None,
            shadow_error=error,
        )

    def _analyse(self, current_positions, grid) -> ShadowRecord:
        step = self.monitor.last_step
        if step is None or step.episode != self.monitor.episode:
            return self._empty(ERR_NO_STEP)
        if any(r.trigger_t == step.t for r in self._episode_records()):
            return self._empty(ERR_STALE_STEP, step)
        goals = self.monitor.goals
        if goals is None:
            return self._empty(ERR_NO_GOALS, step)
        if step.s_t_minus_1 is None or step.intended is None:
            return self._empty(ERR_NO_INTENT, step)

        ids = sorted(step.s_t)
        if self.paths_keying == "agent_id":
            given = _cells(current_positions)
        else:
            seq = list(current_positions)
            given = {a: (int(seq[i][0]), int(seq[i][1])) for i, a in enumerate(ids)} \
                if len(seq) == len(ids) else None
        if given != step.s_t:
            return self._empty(ERR_S_T_MISMATCH, step)

        grid_arr = np.array(grid, dtype=np.int64, copy=True)   # private copy
        wfg = build_wfg(step.s_t_minus_1, step.intended, step.blocked, grid_arr)
        decision = decide_selective(ids, step.s_t, goals, wfg)
        l0 = l0_union_group(wfg.groups)
        outside = tuple(a for a in ids if a not in set(l0))
        hits: tuple[ExpansionHit, ...] = ()
        add: tuple[int, ...] = ()
        precheck = None
        if l0 and outside:
            hits, add = expansion_candidates(l0, step.s_t, goals)
            held = grid_arr.copy()
            for a in outside:
                held[step.s_t[a]] = 1
            unreachable = tuple(
                a for a in l0
                if held[goals[a]] == 1 or int(bfs_dist(held, goals[a])[step.s_t[a]]) < 0
            )
            precheck = ShadowPrecheck(
                goal_blocked=tuple(sorted({h.group_agent for h in hits if h.kind == "GOAL_BLOCKED"})),
                unreachable=unreachable,
            )
        return ShadowRecord(
            episode=self.monitor.episode,
            call_index=len(self._episode_records()) + 1,
            trigger_t=step.t,
            s_t_minus_1=step.s_t_minus_1,
            intended=step.intended,
            blocked=step.blocked,
            s_t=step.s_t,
            wfg=wfg,
            decision=decision,
            l0_group=l0,
            outside=outside,
            expansion_hits=hits,
            expansion_add=add,
            precheck=precheck,
            shadow_error=None,
        )
