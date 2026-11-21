"""Partial Trigger CANDIDATE detector -- DRAFT (rule ``partial-trigger-candidate-v0``).

``PartialTriggerDetector`` / ``PartialTriggerShadow`` are SHADOW ONLY: no CBS call, no WAIT,
no state transition, no Simulator action. ``PartialTriggerShadow`` is a transparent
``DeadlockMonitor`` wrapper that returns the inner Monitor's result object untouched and only
RECORDS what the detector concludes. The default Global Hybrid path does not import this module.

``PartialTriggerSource`` (stage 2, 2026-11-13) is the ONLY way a candidate can reach the
Runner: an explicit DEV opt-in passed as ``run_global_hybrid_episode(extra_trigger=...)`` with a
Selective opt-in planner and ``rearm``. Without it nothing in this module affects a run.

Why: the v0 Trigger is TRIGGER_FULL_REPEAT (the whole joint state repeats). Any agent that
keeps moving -- even one unrelated to a stall -- postpones it (P3F, 2026-11-13: 5 steps).

Inputs reused (observational fields the Monitor already computes, never re-derived here):
  * ``current_persistent_conflict_key`` / ``_streak`` -- the single longest live conflict key
    ``("vertex", cell, agents)`` or ``("edge", {cells}, agents)``; agents = movers + occupants.
  * ``current_no_progress_streaks`` -- per agent, consecutive steps whose goal distance did not
    improve while not at the goal.
  * ``persistent_conflict_status`` / ``no_progress_status`` and their ``*_incomplete_steps``.
  * the step's positions, conflict types, and the goals given to ``reset``.

Per step ``t`` the detector returns ONE status:
  FULL_REPEAT        the inner Monitor reported a fresh full repeat at t (unchanged v0 Trigger)
  INSUFFICIENT_DATA  a signal is NOT_COMPUTED, or its incomplete counter grew this step -> hold
  PARTIAL_CANDIDATE  all of:
                       (1) a key is exposed and its streak >= k_conflict
                       (2) the key names >= 2 agents (group G = the key's agents)
                       (3) at least one agent of G is not at its goal ("active")
                       (4) every active agent of G has a no_progress streak >= k_no_progress
                       (5) after a rearm, (1) and (4) also hold with both streaks clamped
                           to the steps after the rearm (only fresh evidence counts)
  NONE               otherwise (no key / transient contest / members progressing / all at goal)

A candidate is NOT a deadlock verdict. Agents outside G are never consulted, so their moves
cannot cancel a candidate. KNOWN LIMIT: the Monitor exposes one key per step; other concurrent
conflicts are not reconstructed or guessed. Conflict-flagged agents outside G are only LISTED
(``conflicted_outside_key``) as evidence that more than one conflict may be live.

De-duplication: an event is identified by ``(kind, cell(s), agents)``. While the candidate
condition holds continuously for the same identity, only the first step emits ``new_event``.
If it ends and later holds again, a new event is emitted with ``repeat_since_rearm=True``.
``on_rearm(t)`` (Monitor rearm after IL_RESUME) clears the active set and the per-rearm
history and requires fresh evidence: only the part of each streak after t counts
(proposal). Fixed 2026-11-13: the first version rejected any streak that began before the
rearm, which -- because no_progress is not reset by a CBS window -- suppressed every later
candidate after a resume.

Thresholds have NO defaults: ``PartialTriggerConfig`` must be given explicitly (DEV values;
K = 2/3/5 are test candidates, not validated operating values).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

RULE_VERSION = "partial-trigger-candidate-v0"

FULL_REPEAT = "FULL_REPEAT"
PARTIAL_CANDIDATE = "PARTIAL_CANDIDATE"
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
NONE = "NONE"

Cell = tuple[int, int]


def _pos_int(name, v):
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        raise ValueError(f"{name} must be an int >= 1 (DEV value, no default), got {v!r}")
    return v


@dataclass(frozen=True)
class PartialTriggerConfig:
    k_conflict: int
    k_no_progress: int

    def __post_init__(self):
        _pos_int("k_conflict", self.k_conflict)
        _pos_int("k_no_progress", self.k_no_progress)


@dataclass(frozen=True)
class PartialObservation:
    t: int
    status: str
    reason: str
    group: tuple[int, ...] = ()
    key_kind: Optional[str] = None
    key_cells: tuple = ()
    conflict_streak: int = 0
    no_progress: tuple[tuple[int, int], ...] = ()       # (agent, streak) for G members
    parked_in_group: tuple[int, ...] = ()               # G members at their goal
    conflicted_outside_key: tuple[int, ...] = ()        # conflict-flagged agents not in G
    group_smaller_than_all: Optional[bool] = None
    n_agents: int = 0
    new_event: bool = False
    repeat_since_rearm: bool = False


def _cells_of(key) -> tuple:
    kind, where = key[0], key[1]
    if kind == "vertex":
        return (tuple(where),)
    return tuple(sorted(tuple(c) for c in where))


class PartialTriggerDetector:
    """Pure per-step evaluation over the Monitor's result snapshot."""

    def __init__(self, config: PartialTriggerConfig):
        if not isinstance(config, PartialTriggerConfig):
            raise ValueError("config must be a PartialTriggerConfig (explicit DEV thresholds)")
        self.config = config
        self.reset(None)

    def reset(self, goals: Optional[Mapping[int, Cell]]):
        self.goals = None if goals is None else {int(a): (int(g[0]), int(g[1])) for a, g in goals.items()}
        self._active: set = set()
        self._fired_since_rearm: set = set()
        self._rearm_t: Optional[int] = None
        self._prev_pc_incomplete = 0
        self._prev_np_incomplete = 0
        self.observations: list[PartialObservation] = []
        self.events: list[PartialObservation] = []

    def on_rearm(self, t: int):
        self._rearm_t = int(t)
        self._active = set()
        self._fired_since_rearm = set()

    def observe(self, t: int, result: Any, positions: Mapping[int, Cell],
                conflict_types: Mapping[int, str]) -> PartialObservation:
        obs = self._evaluate(int(t), result, positions, conflict_types)
        self.observations.append(obs)
        if obs.new_event:
            self.events.append(obs)
        return obs

    # ------------------------------------------------------------------------------
    def _evaluate(self, t, r, positions, conflict_types):
        pos = {int(a): (int(p[0]), int(p[1])) for a, p in positions.items()}
        n = len(pos)
        pc_inc = int(getattr(r, "persistent_conflict_incomplete_steps", 0))
        np_inc = int(getattr(r, "no_progress_incomplete_steps", 0))
        grew = pc_inc > self._prev_pc_incomplete or np_inc > self._prev_np_incomplete
        self._prev_pc_incomplete, self._prev_np_incomplete = pc_inc, np_inc

        def done(status, reason, **kw):
            if status != PARTIAL_CANDIDATE:
                self._active = set()          # a continuous candidate episode ends here
            return PartialObservation(t=t, status=status, reason=reason, n_agents=n, **kw)

        if getattr(r, "trigger_reason_code", None) == "TRIGGER_FULL_REPEAT" \
                and getattr(r, "trigger_timestep", None) == t:
            return done(FULL_REPEAT, "MONITOR_FULL_REPEAT")
        if self.goals is None or set(self.goals) != set(pos):
            return done(INSUFFICIENT_DATA, "GOALS_UNKNOWN_OR_AGENT_SET_MISMATCH")
        if getattr(r, "persistent_conflict_status", None) != "OK" or getattr(r, "no_progress_status", None) != "OK":
            return done(INSUFFICIENT_DATA, "SIGNAL_NOT_COMPUTED")
        if grew:
            return done(INSUFFICIENT_DATA, "SIGNAL_INCOMPLETE_THIS_STEP")

        key = getattr(r, "current_persistent_conflict_key", None)
        streak = int(getattr(r, "current_persistent_conflict_streak", 0) or 0)
        if key is None:
            return done(NONE, "NO_CONFLICT_KEY")
        group = tuple(sorted(int(a) for a in key[2]))
        kind, cells = key[0], _cells_of(key)
        np_all = {int(a): int(s) for a, s in getattr(r, "current_no_progress_streaks", ())}
        parked = tuple(a for a in group if pos.get(a) == self.goals.get(a))
        active = [a for a in group if a not in parked]
        others = tuple(sorted(a for a, c in conflict_types.items()
                              if c in ("vertex", "edge") and int(a) not in group))
        info = dict(group=group, key_kind=kind, key_cells=cells, conflict_streak=streak,
                    no_progress=tuple((a, np_all.get(a, 0)) for a in group), parked_in_group=parked,
                    conflicted_outside_key=others, group_smaller_than_all=len(group) < n)
        cfg = self.config
        if len(group) < 2:
            return done(NONE, "KEY_HAS_FEWER_THAN_2_AGENTS", **info)
        if streak < cfg.k_conflict:
            return done(NONE, f"CONFLICT_STREAK_{streak}<{cfg.k_conflict}", **info)
        if not active:
            return done(NONE, "ALL_KEY_AGENTS_AT_GOAL", **info)
        slow = [a for a in active if np_all.get(a, 0) < cfg.k_no_progress]
        if slow:
            return done(NONE, f"KEY_AGENTS_PROGRESSING:{slow}", **info)
        if self._rearm_t is not None:
            # Only evidence gathered AFTER the rearm counts. The Monitor keeps both streaks
            # through a rearm, and a CBS window resets the conflict streak but not no_progress
            # (a planned WAIT is "no progress"), so streaks are clamped to the post-rearm window
            # instead of being rejected (rejecting would suppress every later candidate).
            window = t - self._rearm_t
            fresh_c = min(streak, window)
            fresh_np = {a: min(np_all.get(a, 0), window) for a in active}
            if fresh_c < cfg.k_conflict or any(v < cfg.k_no_progress for v in fresh_np.values()):
                return done(NONE, f"FRESH_EVIDENCE_SINCE_REARM_AT_{self._rearm_t}:"
                                  f"conflict={fresh_c},no_progress={min(fresh_np.values())}", **info)

        ident = (kind, cells, group)
        new = ident not in self._active
        repeat = new and ident in self._fired_since_rearm
        self._active = {ident}
        if new:
            self._fired_since_rearm.add(ident)
        return PartialObservation(t=t, status=PARTIAL_CANDIDATE, reason="CANDIDATE", n_agents=n,
                                  new_event=new, repeat_since_rearm=repeat, **info)


PARTIAL_TRIGGER_REASON = "TRIGGER_PARTIAL_CANDIDATE"


class PartialTriggerSource:
    """``run_global_hybrid_episode(..., extra_trigger=...)`` for an explicit, DEV-only Partial Trigger opt-in.

    Construction refuses unless ALL hold: ``allow_partial_trigger=True``; ``planner`` is a
    ``SelectivePlanner`` with an opt-in (L0/L1) ladder; ``monitor`` is that planner's own
    ``ShadowMonitor`` and its inner monitor is a ``PartialTriggerShadow`` holding the named
    detector. So the Runner, the planner's ``last_step`` (S_{t-1}, intended, S_t) and the
    detector all see the SAME single ``observe_step`` per step.

    ``poll(t)`` returns the detector's observation only if it was made at exactly step ``t``,
    is ``PARTIAL_CANDIDATE`` with ``new_event=True``, the ShadowMonitor's last step is also
    ``t``, and ``t`` was not handed out before (no stale or reused event). Everything else
    -- FULL priority, integrity, recall limits, budget, cooldown, rearm -- is the Runner's.
    """

    reason = PARTIAL_TRIGGER_REASON

    def __init__(self, *, monitor, planner, detector_name: str, allow_partial_trigger: bool = False):
        from src.hybrid.selective_planner import SelectivePlanner
        from src.hybrid.selective_shadow import ShadowMonitor
        if allow_partial_trigger is not True:
            raise ValueError("Partial Trigger is DEV opt-in only: pass allow_partial_trigger=True")
        if not isinstance(planner, SelectivePlanner) or not planner.config.opt_in:
            raise ValueError("Partial Trigger needs a SelectivePlanner with an opt-in L0/L1 ladder")
        if not isinstance(monitor, ShadowMonitor) or planner.monitor is not monitor:
            raise ValueError("monitor must be the SelectivePlanner's own ShadowMonitor")
        inner = monitor.inner
        if not isinstance(inner, PartialTriggerShadow):
            raise ValueError("ShadowMonitor must wrap a PartialTriggerShadow (one observe_step per step)")
        if detector_name not in inner.detectors:
            raise ValueError(f"no detector {detector_name!r} in the PartialTriggerShadow")
        self.bound_planner = planner
        self.bound_monitor = monitor
        self.detector = inner.detectors[detector_name]
        self.detector_name = detector_name
        self.reset()

    def reset(self):
        self._handed_out: set[int] = set()
        self.fired: list[PartialObservation] = []
        self.refused: list[tuple[int, str]] = []

    def poll(self, t: int) -> Optional[PartialObservation]:
        obs = self.detector.observations[-1] if self.detector.observations else None
        last = self.bound_monitor.last_step
        why = None
        if obs is None or obs.t != t:
            why = "NO_OBSERVATION_AT_T"
        elif last is None or last.t != t or last.episode != self.bound_monitor.episode:
            why = "MONITOR_STEP_MISMATCH"
        elif obs.status != PARTIAL_CANDIDATE or not obs.new_event:
            return None
        elif t in self._handed_out:
            why = "ALREADY_HANDED_OUT"
        if why is not None:
            self.refused.append((t, why))
            return None
        self._handed_out.add(t)
        self.fired.append(obs)
        return obs

    @staticmethod
    def describe(obs: PartialObservation) -> dict:
        return {"t": obs.t, "group": obs.group, "key_kind": obs.key_kind, "key_cells": obs.key_cells,
                "conflict_streak": obs.conflict_streak, "no_progress": obs.no_progress,
                "parked_in_group": obs.parked_in_group, "conflicted_outside_key": obs.conflicted_outside_key,
                "group_smaller_than_all": obs.group_smaller_than_all, "rule_version": RULE_VERSION}


class PartialTriggerShadow:
    """Transparent DeadlockMonitor wrapper: forwards every call, returns the inner result
    unchanged, and feeds one or more detectors (e.g. several DEV K values) for the record.

    It also keeps, per step, S_{t-1} / intended / blocked so a later offline WFG comparison
    can use the same inputs the Selective path uses (P1-c). Recording failures never alter
    the Monitor path."""

    def __init__(self, inner, detectors: Mapping[str, PartialTriggerDetector]):
        if not detectors or not all(isinstance(d, PartialTriggerDetector) for d in detectors.values()):
            raise ValueError("at least one PartialTriggerDetector is required")
        self.inner = inner
        self.detectors = dict(detectors)
        self.steps: list[dict] = []
        self.record_errors: list[str] = []

    def reset(self, initial_positions, goals):
        out = self.inner.reset(initial_positions, goals)
        self.steps = []
        for d in self.detectors.values():
            d.reset(goals)
        return out

    def rearm(self, current_positions, current_timestep):
        out = self.inner.rearm(current_positions, current_timestep)
        for d in self.detectors.values():
            d.on_rearm(current_timestep)
        return out

    def observe_step(self, timestep, positions, conflict_types, **kwargs):
        result = self.inner.observe_step(timestep, positions, conflict_types, **kwargs)
        try:
            cur, want = kwargs.get("current_positions"), kwargs.get("intended_positions")
            self.steps.append({
                "t": int(timestep),
                "s_t_minus_1": None if cur is None else {int(a): tuple(p) for a, p in cur.items()},
                "intended": None if want is None else {int(a): tuple(p) for a, p in want.items()},
                "s_t": {int(a): tuple(p) for a, p in positions.items()},
                "blocked": tuple(sorted(int(a) for a, c in conflict_types.items() if c != "none")),
            })
            for d in self.detectors.values():
                d.observe(timestep, result, positions, conflict_types)
        except Exception as exc:   # recording only; never alter the Monitor path
            self.record_errors.append(f"{type(exc).__name__}:{exc}")
        return result

    def __getattr__(self, name):
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)
