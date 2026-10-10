"""Deadlock Monitor v1 -- Phase 3A: full joint-state repeat detection only.

Scope of this module (see docs/deadlock_definition_v1.md for the full
vocabulary this implements a small slice of):

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
  * Does not implement: Trigger-Partial, K_label continuation, CBS
    recovery, Hybrid runner integration, or any no-progress/persistent-
    conflict trigger beyond the single joint-state-repeat check above
    (deferred to a later phase -- see `DeadlockMonitorConfig`).

This module only *observes* -- it never changes an action, never stops a
rollout, and never touches `src/simulator/mapf_step_simulator.py`,
`src/il/policy.py`, or `src/common/episode_logger.py`'s existing field
semantics. `scripts/run_il.py`'s behaviour with `monitor=None` (the
default) is byte-for-byte unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

Position = Tuple[int, int]
JointState = Tuple[Position, ...]


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
        self._goal_state = self._joint_state(goals)
        s0 = self._joint_state(initial_positions)
        if s0 != self._goal_state:
            self._seen[s0] = 0

    def observe_step(
        self,
        timestep: int,
        positions: Dict[int, Position],
        conflict_types: Dict[int, str],
    ) -> DeadlockMonitorResult:
        """Call once per simulator step, after `sim.step()` returns.

        `timestep` must be the simulator's own post-increment `t` (the same
        value `EpisodeLogger` logs as `timestep`); `positions` is `S_t`;
        `conflict_types` describes the `S_{t-1} -> S_t` transition that just
        produced it (docs/deadlock_definition_v1.md Section 3.3's indexing
        rule -- conflict evidence is evidence about *arriving at* `S_t`).
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

        if self._result.trigger_type == "FULL":
            return self._result  # first repeat only; already settled

        if state == self._goal_state:
            # Normal, successful "everyone is at their goal" state is never
            # a stall (Section 4.3 condition 3) -- do not register or
            # compare it, so it can never be mistaken for a repeat.
            return self._result

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

        return self._result

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

    @property
    def result(self) -> DeadlockMonitorResult:
        return self._result
