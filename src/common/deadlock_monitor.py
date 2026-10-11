"""Deadlock Monitor v1 -- Phase 3A (full joint-state repeat) + Phase 3B-1
(persistent_conflict candidate signal).

Scope of this module (see docs/deadlock_definition_v1.md for the full
vocabulary this implements a slice of):

  * Detects a **full** joint-state repeat (Section 3.3/4.3) -- every agent's
    position matches an earlier snapshot simultaneously. A local/subset
    repeat (Trigger-Partial, Section 5.1/5.3) is explicitly NOT implemented
    here; that is left to a future phase.
  * On a confirmed repeat, classifies the closing-cycle evidence
    (`timestep` in `(first_seen, first_repeat]`, Section 3.3's indexing
    rule) as VERTEX / EDGE / MIXED / NONE persistent blocking
    (Section 3.4/4.3 condition 4). "NONE" means a full repeat was confirmed
    but no inter-agent conflict_type evidence was found in that window --
    this module reports that honestly rather than guessing; it does **not**
    decide a final Operational Label (UNKNOWN vs. COORDINATION_STALL_*) --
    that decision needs the determinism-precondition and solvability checks
    in Section 5.2, which are out of scope here (Section 2 concept 5 vs.
    concept 4: this module only ever produces a Trigger, never a Label).
  * Tracks `overlap_timesteps` (position overlap, Section 3.6) as a
    SEPARATE, always-reported field, independent of trigger_type --
    an integrity issue is recorded but never suppresses or short-circuits
    the repeat-detection logic (a KI-1-affected episode still gets a normal
    trigger_type computed; see docs/deadlock_definition_v1.md Section 8.1's
    self-consistency table). `invalid_distance_rows` is explicitly NOT
    computed here (it needs the policy's BFS goal-distance field, which
    this module never touches) and `overlap_integrity_status` must never be
    read as a full integrity clearance.
  * (Phase 3B-1) Tracks a `persistent_conflict` candidate signal: the
    current, *live* consecutive-step streak of the same reconstructed
    conflict key (vertex: contested cell + involved agents; edge: swapping
    cell pair + agent pair -- reconstructed with the exact same key scheme
    as `scripts/analyze_failure_logs.py`'s offline `repeated_conflict_candidate_*`,
    Section 3.4). This is a **third, independent observational field**,
    never a Trigger and never a Label: it does not feed into, gate, or get
    gated by `trigger_type`/`blocking_type` (which stay exactly Phase 3A's
    unmodified full-joint-state-repeat logic -- see `_update_persistent_conflict()`,
    called unconditionally, in parallel with overlap tracking, same as
    overlap_timesteps). It is a strictly *online, current-streak* quantity
    and must never be read as the offline, whole-episode
    `repeated_conflict_candidate_max`/`_max_consecutive` -- the two answer
    different questions at different times (Section 7.1(c)): "how long has
    this exact conflict been recurring *so far*" (this module, mid-episode)
    vs. "what was the longest such run, *in the whole finished episode*"
    (`analyze_failure_logs.py`, after the fact). No numeric threshold is
    introduced for "candidate" status -- the raw streak length and the key
    itself are always reported; deciding a cutoff is deferred pending
    validation data (docs/deadlock_definition_v1.md Section 5.4/8.2).
    Computing this signal requires each agent's position *before* this
    step (`current_positions`) and its attempted, pre-collision-resolution
    target cell (`intended_positions`) -- information the Phase 3A inputs
    (`positions`/`conflict_types` alone) do not carry. Both are optional,
    keyword-only inputs to `observe_step()`; the three original positional
    arguments are completely unaffected, so every existing caller keeps
    working unchanged. If they are not supplied, the signal is simply not
    computed (`persistent_conflict_status="NOT_COMPUTED"`) -- same
    additive-only, opt-in contract as `monitor=None` elsewhere in this
    pipeline. If they are supplied but a conflicted agent's data is
    missing, this module never guesses a key: it records the step as
    incomplete (`persistent_conflict_incomplete_steps`) and lets any
    in-progress streak end, rather than fabricating continuity.
  * Does not implement: Trigger-Partial, K_label continuation, CBS
    recovery, Hybrid runner integration, or a `no_progress` candidate
    signal (deferred to a later phase -- see `DeadlockMonitorConfig`).

This module only *observes* -- it never changes an action, never stops a
rollout, and never touches `src/simulator/mapf_step_simulator.py`,
`src/il/policy.py`, or `src/common/episode_logger.py`'s existing field
semantics. `scripts/run_il.py`'s behaviour with `monitor=None` (the
default) is byte-for-byte unaffected. The persistent_conflict signal keeps
this neutrality too: it only ever reads the `current_positions`/
`intended_positions` dicts a caller chooses to pass in, and never reaches
into a policy object to compute them itself.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Tuple

Position = Tuple[int, int]
JointState = Tuple[Position, ...]
# ("vertex", cell, frozenset(agent_ids)) or
# ("edge", frozenset({cell_a, cell_b}), frozenset({agent_a, agent_b})) --
# deliberately the same shape as scripts/analyze_failure_logs.py's
# conflict_keys dict keys, for direct cross-validation (Section 3.4).
ConflictKey = Tuple[str, Any, "frozenset[int]"]


@dataclass
class DeadlockMonitorConfig:
    """Placeholder for future, validation-calibrated settings.

    Phase 3A's full joint-state repeat check is an exact-match test, not a
    threshold-based one, so it reads neither field below. They exist only so
    a future phase (no-progress windows, K_label, Trigger-Partial --
    docs/deadlock_definition_v1.md Section 5.4/8.2) can be wired in through
    config instead of a hardcoded constant, per that document's explicit
    requirement that no such number be guessed in advance.
    """

    no_progress_window: int | None = None
    k_label_window: int | None = None


@dataclass
class DeadlockMonitorResult:
    """A Trigger-level report. This is NOT an Operational Label.

    `trigger_type`/`trigger_reason_code` only ever say "a full joint-state
    repeat did or did not occur" -- never "this episode is a confirmed
    deadlock". See the module docstring and
    docs/deadlock_definition_v1.md Section 2 (concepts 3-5) for why the two
    must stay separate.
    """

    trigger_type: str = "NONE"  # "NONE" | "FULL"
    trigger_reason_code: str = "TRIGGER_NONE"  # "TRIGGER_NONE" | "TRIGGER_FULL_REPEAT"
    trigger_timestep: int | None = None  # == first_repeat_timestep, once triggered
    first_seen: int | None = None
    period: int | None = None
    blocking_type: str = "NONE"  # "NONE" | "VERTEX" | "EDGE" | "MIXED"
    blocking_agents: Tuple[int, ...] = field(default_factory=tuple)
    overlap_timesteps: int = 0
    overlap_integrity_status: str = "CLEAN"  # "CLEAN" | "INTEGRITY_ISSUE"

    # -- Phase 3B-1: persistent_conflict candidate signal (observational
    # only -- see the module docstring; never a Trigger, never a Label,
    # never feeds back into any field above). --
    # "NOT_COMPUTED" until observe_step() is ever called with both
    # current_positions and intended_positions; "OK" from then on (even on
    # a step with zero active conflict, since the signal *was* evaluated).
    persistent_conflict_status: str = "NOT_COMPUTED"  # "NOT_COMPUTED" | "OK"
    # Steps where current_positions/intended_positions were supplied but a
    # conflicted agent's entry was missing from one of them -- the step was
    # skipped (no key guessed) and any in-progress streak was ended.
    persistent_conflict_incomplete_steps: int = 0
    # The longest *currently live* (still-recurring-as-of-this-call)
    # consecutive-step streak across all conflict keys active this step;
    # 0 if none. This is the online, current-streak counterpart of
    # analyze_failure_logs.py's offline, whole-episode
    # repeated_conflict_candidate_max_consecutive -- the two are NOT the
    # same quantity and must never be substituted for one another.
    #
    # KNOWN 3B-1 LIMITATION: if more than one distinct conflict key is
    # live in the same step (e.g. a MIXED-style episode with one
    # vertex-blocked pair and a different edge-blocked pair active at
    # once), only the single longest-streak key is exposed here -- the
    # other key(s)' streaks are still tracked internally
    # (DeadlockMonitor._persistent_conflict_streaks) but not surfaced in
    # this result. This field is therefore a lower bound on "how much
    # persistent-conflict evidence exists this step", not an exhaustive
    # report. It is observational only: nothing here feeds into, or may
    # be used to derive, `blocking_type`/`trigger_type` or any future
    # Operational Label -- the confirmed MIXED/VERTEX/EDGE classification
    # stays exclusively `_classify_blocking()`'s, computed only after a
    # Full-repeat is confirmed (Section 4.3/4.4). A high
    # current_persistent_conflict_streak must never be read as, or used
    # to assemble, a deadlock verdict.
    current_persistent_conflict_streak: int = 0
    current_persistent_conflict_key: ConflictKey | None = None


class DeadlockMonitor:
    """Observes a rollout step by step. Never alters actions or stops it.

    Usage (mirrors `src.common.episode_logger.EpisodeLogger`'s optional,
    additive-only pattern):

        monitor = DeadlockMonitor()
        monitor.reset(sim._positions, goals)          # registers S_0 at t=0
        ...
        obs, done, info = sim.step(actions)
        monitor.observe_step(info["t"], sim._positions, sim._last_conflict_types)
        ...
        result = monitor.result
    """

    def __init__(self, config: DeadlockMonitorConfig | None = None):
        self.config = config or DeadlockMonitorConfig()
        self._seen: Dict[JointState, int] = {}
        self._goal_state: JointState | None = None
        self._conflict_log: Dict[int, Dict[int, str]] = {}
        self._result = DeadlockMonitorResult()
        # Phase 3B-1: key -> current live consecutive-step streak. Rebuilt
        # from scratch each observe_step() call (keys absent this step are
        # dropped -- a streak that does not recur is, by definition, over).
        self._persistent_conflict_streaks: Dict[ConflictKey, int] = {}

    @staticmethod
    def _joint_state(positions: Dict[int, Position]) -> JointState:
        """Fixed, sorted-by-agent_id order -- independent of dict iteration
        order, so two callers passing the same positions in different key
        order always produce the same joint state (docs/deadlock_definition_v1.md
        Section 3.3: "full joint state ... every agent")."""
        return tuple(positions[aid] for aid in sorted(positions))

    def reset(self, initial_positions: Dict[int, Position], goals: Dict[int, Position]) -> None:
        """Register S_0 at t=0, before the first `step()` call.

        This must be called once, right after the simulator's own
        `reset()`, so the seen-states table starts at t=0 exactly like
        `scripts/analyze_failure_logs.py::analyze_episode` scans from
        `t_first - 1` (docs/deadlock_definition_v1.md Section 7.1(b)) --
        otherwise an instance whose freeze starts at `first_seen=0`
        (e.g. `intersection_n2_tight`) would be missed or mis-indexed.
        """
        self._seen = {}
        self._conflict_log = {}
        self._result = DeadlockMonitorResult()
        self._persistent_conflict_streaks = {}
        self._goal_state = self._joint_state(goals)
        s0 = self._joint_state(initial_positions)
        if s0 != self._goal_state:
            self._seen[s0] = 0

    def observe_step(
        self,
        timestep: int,
        positions: Dict[int, Position],
        conflict_types: Dict[int, str],
        *,
        current_positions: Dict[int, Position] | None = None,
        intended_positions: Dict[int, Position] | None = None,
    ) -> DeadlockMonitorResult:
        """Call once per simulator step, after `sim.step()` returns.

        `timestep` must be the simulator's own post-increment `t` (the same
        value `EpisodeLogger` logs as `timestep`); `positions` is `S_t`;
        `conflict_types` describes the `S_{t-1} -> S_t` transition that just
        produced it (docs/deadlock_definition_v1.md Section 3.3's indexing
        rule -- conflict evidence is evidence about *arriving at* `S_t`).
        These three positional arguments are exactly Phase 3A's original
        signature and behaviour -- unchanged.

        `current_positions`/`intended_positions` (Phase 3B-1, optional,
        keyword-only): each agent's position *before* this step and its
        attempted, pre-collision-resolution target cell, in the same raw
        (unclamped-by-walls for non-wall-hit agents) form
        `scripts/run_il.py::_log_step` computes and
        `episode_logger.py`/`analyze_failure_logs.py` call `intended`. Both
        must be supplied together to compute the persistent_conflict
        signal for this step; if either is omitted, the signal is left at
        its previous value for this call (see `_update_persistent_conflict`).
        """
        state = self._joint_state(positions)
        self._conflict_log[timestep] = dict(conflict_types)

        # Integrity (position overlap) is tracked unconditionally, in
        # parallel with -- and never gating -- the repeat/trigger logic
        # below (docs/deadlock_definition_v1.md Section 8.1: a KI-1
        # episode still gets a normal trigger_type computed).
        if len(set(positions.values())) != len(positions):
            self._result.overlap_timesteps += 1
            self._result.overlap_integrity_status = "INTEGRITY_ISSUE"

        # Phase 3B-1: likewise tracked unconditionally, in parallel with --
        # and never gating or gated by -- trigger_type/blocking_type below.
        # Continues to update even after trigger_type has settled to
        # "FULL", same as overlap tracking above.
        self._update_persistent_conflict(positions, conflict_types, current_positions, intended_positions)

        if self._result.trigger_type == "FULL":
            return self._snapshot()  # first repeat only; already settled

        if state == self._goal_state:
            # Normal, successful "everyone is at their goal" state is never
            # a stall (Section 4.3 condition 3) -- do not register or
            # compare it, so it can never be mistaken for a repeat.
            return self._snapshot()

        if state in self._seen:
            first_seen = self._seen[state]
            self._result.trigger_type = "FULL"
            self._result.trigger_reason_code = "TRIGGER_FULL_REPEAT"
            self._result.trigger_timestep = timestep
            self._result.first_seen = first_seen
            self._result.period = timestep - first_seen
            self._classify_blocking(first_seen, timestep)
        else:
            self._seen[state] = timestep

        return self._snapshot()

    def _snapshot(self) -> DeadlockMonitorResult:
        """Returns an independent copy of the current result.

        `observe_step()` mutates one internal `DeadlockMonitorResult` in
        place over the whole episode (so `self._result` always reflects the
        latest state cheaply); callers that keep each call's return value
        around to build a per-step history (e.g. to compare the
        persistent_conflict streak at different timesteps, Phase 3B-1)
        would otherwise find every entry silently aliasing the same,
        still-mutating object and ending up equal to the *final* step's
        values. Returning a fresh `dataclasses.replace()` copy each call
        fixes this without changing any field's computed value -- existing
        callers that only ever read fields off the most recent return (or
        off the `result` property, once, at the end) are unaffected."""
        return replace(self._result)

    def _classify_blocking(self, first_seen: int, first_repeat: int) -> None:
        """Evidence window is `timestep in (first_seen, first_repeat]`
        (Section 4.3 condition 4) -- NOT `timestep == first_seen`, which
        describes arriving at the state that later repeats, not the blocked
        attempt to leave it. Checking just this one, already-confirmed
        cycle is sufficient (not merely a sample): the determinism argument
        (Section 4.2) guarantees this exact pattern recurs identically
        forever once the full joint state is confirmed to repeat, so there
        is nothing to gain from re-scanning subsequent cycles.
        """
        vertex_agents: set[int] = set()
        edge_agents: set[int] = set()
        for t in range(first_seen + 1, first_repeat + 1):
            for agent_id, conflict_type in self._conflict_log.get(t, {}).items():
                if conflict_type == "vertex":
                    vertex_agents.add(agent_id)
                elif conflict_type == "edge":
                    edge_agents.add(agent_id)

        if vertex_agents and edge_agents:
            self._result.blocking_type = "MIXED"
            self._result.blocking_agents = tuple(sorted(vertex_agents | edge_agents))
        elif vertex_agents:
            self._result.blocking_type = "VERTEX"
            self._result.blocking_agents = tuple(sorted(vertex_agents))
        elif edge_agents:
            self._result.blocking_type = "EDGE"
            self._result.blocking_agents = tuple(sorted(edge_agents))
        else:
            # A full repeat with no vertex/edge evidence in the closing
            # cycle (e.g. every agent is WAITing with conflict_type=none).
            # Left as "NONE" on purpose -- the caller must not read this as
            # a confirmed stall; it is exactly the
            # REPEAT_WITHOUT_BLOCKING_EVIDENCE case that
            # docs/deadlock_definition_v1.md Section 4.4 says must fall to
            # UNKNOWN rather than being forced into a coordination label.
            self._result.blocking_type = "NONE"
            self._result.blocking_agents = ()

    @staticmethod
    def _reconstruct_conflict_keys_for_step(
        agents,
        current_positions: Dict[int, Position],
        intended_positions: Dict[int, Position],
        positions: Dict[int, Position],
        conflict_types: Dict[int, str],
    ) -> tuple[set, bool]:
        """Reconstruct this single step's vertex/edge conflict keys, using
        the exact same key scheme as
        `scripts/analyze_failure_logs.py::analyze_episode`'s offline
        `conflict_keys` reconstruction (Section 3.4), so the two can be
        cross-validated directly:
          vertex key = ("vertex", contested_cell, frozenset(movers | occupants))
          edge key   = ("edge", frozenset({cell_a, cell_b}), frozenset({agent_a, agent_b}))
        where `movers` are agents whose `conflict_type == "vertex"` and
        shared `intended` cell, and `occupants` are agents who did not
        move (`current == next == cell`) and are not already a mover --
        offline reads this straight from CSV columns; here it is the same
        computation over this step's current/intended/next/conflict_type.

        Returns `(keys, incomplete)`. `incomplete=True` means a
        conflict-flagged agent's `current`/`intended` entry was missing --
        the caller must not use `keys` (it is always empty in that case)
        and must treat this step as unknown rather than "no conflict"
        (docs/deadlock_definition_v1.md's "do not guess" principle).
        """
        conflicted = [aid for aid in agents if conflict_types.get(aid) in ("vertex", "edge")]
        if any(aid not in current_positions for aid in agents) or \
                any(aid not in intended_positions for aid in conflicted):
            return set(), True

        keys: set = set()

        # vertex: group agents proposing the same intended cell.
        cells: Dict[Position, list] = defaultdict(list)
        for aid in agents:
            if conflict_types.get(aid) == "vertex":
                cells[intended_positions[aid]].append(aid)
        for cell, movers in cells.items():
            movers_set = set(movers)
            occupants = [
                aid for aid in agents
                if current_positions[aid] == cell and positions[aid] == cell
                and aid not in movers_set
            ]
            keys.add(("vertex", cell, frozenset(movers_set | set(occupants))))

        # edge: pairs of agents whose intended cells are each other's current cell.
        edge_agents = [aid for aid in agents if conflict_types.get(aid) == "edge"]
        for i, a in enumerate(edge_agents):
            for b in edge_agents[i + 1:]:
                if intended_positions[a] == current_positions[b] and \
                        intended_positions[b] == current_positions[a]:
                    keys.add((
                        "edge",
                        frozenset({current_positions[a], current_positions[b]}),
                        frozenset({a, b}),
                    ))

        return keys, False

    def _update_persistent_conflict(
        self,
        positions: Dict[int, Position],
        conflict_types: Dict[int, str],
        current_positions: Dict[int, Position] | None,
        intended_positions: Dict[int, Position] | None,
    ) -> None:
        """Phase 3B-1. Advances the live, current-streak-per-key state.

        Not computed at all (status stays/becomes "NOT_COMPUTED") if the
        caller never supplies either input on any call. Once engaged
        ("OK"), a step whose conflict-flagged agents cannot be fully
        reconstructed is counted in `persistent_conflict_incomplete_steps`
        and ends every in-progress streak -- never fabricates a guess.

        Streaks are rebuilt from scratch each call: only keys observed
        *this* step keep (and increment) their running count; any key not
        seen this step is dropped. This is what makes the reported value a
        live "current streak", never a cumulative whole-episode total
        (that quantity belongs to `scripts/analyze_failure_logs.py`,
        offline, after the episode ends -- Section 7.1(c)).
        """
        if current_positions is None and intended_positions is None:
            return  # feature not engaged on this call -- leave status as-is

        self._result.persistent_conflict_status = "OK"

        if current_positions is None or intended_positions is None:
            # Engaged, but only one of the two was supplied this call --
            # cannot safely reconstruct anything. Do not guess.
            self._result.persistent_conflict_incomplete_steps += 1
            self._persistent_conflict_streaks = {}
            self._result.current_persistent_conflict_streak = 0
            self._result.current_persistent_conflict_key = None
            return

        agents = sorted(positions)
        keys, incomplete = self._reconstruct_conflict_keys_for_step(
            agents, current_positions, intended_positions, positions, conflict_types)
        if incomplete:
            self._result.persistent_conflict_incomplete_steps += 1
            keys = set()

        new_streaks: Dict[ConflictKey, int] = {}
        for key in keys:
            new_streaks[key] = self._persistent_conflict_streaks.get(key, 0) + 1
        self._persistent_conflict_streaks = new_streaks

        if new_streaks:
            best_key = max(new_streaks, key=lambda k: (new_streaks[k], str(k)))
            self._result.current_persistent_conflict_streak = new_streaks[best_key]
            self._result.current_persistent_conflict_key = best_key
        else:
            self._result.current_persistent_conflict_streak = 0
            self._result.current_persistent_conflict_key = None

    @property
    def result(self) -> DeadlockMonitorResult:
        return self._snapshot()
