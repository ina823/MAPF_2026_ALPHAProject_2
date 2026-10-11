"""Reusable step/episode CSV logging for MAPF rollouts.

Generic on purpose: this module knows nothing about the IL policy, the CBS
adapter, or MAPFStepSimulator. A caller (currently scripts/run_il.py; later
CBS / Global Hybrid / Selective Hybrid runners) extracts the per-step values
from its own simulator/policy objects and hands them to ``EpisodeLogger``.
Logging is opt-in: a caller that never constructs an ``EpisodeLogger`` (or
passes ``logger=None`` down its call chain) does no extra work and produces
no CSV output, so the non-logging code path is unaffected byte-for-byte.

Two CSVs per episode, sharing one run_id:
    outputs/logs/<run_id>_steps.csv    -- one row per (timestep, agent)
    outputs/logs/<run_id>_summary.csv  -- one row for the whole episode

``trigger_reason`` / ``cbs_latency_ms`` are placeholders for the future
Deadlock Monitor / CBS hybrid work (not implemented here); they are always
written but are empty/None for IL runs.

Summary metric semantics (read this before adding a caller):
    wait_count / blocked_count
        Agent-step counts, summed over the whole episode: e.g. wait_count=12
        means WAIT was the chosen action in 12 (agent, timestep) rows total
        across however many agents, not "N agents used WAIT at least once".
        These are unambiguous because is_wait/blocked are per-agent-per-step
        properties by definition -- there is no separate "event" they could
        be miscounting. Derived directly from the collected step rows.
    vertex_conflicts / edge_conflicts
        TRUE conflict-EVENT counts, NOT a count of step rows whose
        conflict_type == "vertex"/"edge". A single vertex conflict where 3
        agents converge on one cell produces 3 per-agent rows tagged
        "vertex" but is exactly 1 event; a single edge swap produces 2 rows
        tagged "edge" but is exactly 1 event. Counting rows would therefore
        over-count. These totals are instead accumulated by
        ``log_timestep_conflicts()``, called once per timestep (not once per
        agent) with the simulator's own per-timestep event counts
        (MAPFStepSimulator._last_vertex_conflict_count /
        _last_edge_conflict_count), which are exact because they read the
        simulator's existing collision-resolution bookkeeping (landing-cell
        groups / swapping pairs) that already exists for the unrelated
        purpose of deciding who bounces back -- no new collision logic, no
        change to collision behaviour, see mapf_step_simulator.py::step().
        If a caller never calls log_timestep_conflicts() (e.g. a future
        caller without that instrumentation available), both totals stay 0;
        this is a known limitation, not a silent wrong answer -- such a
        caller must not rely on these two summary fields until it wires up
        the equivalent per-timestep event counts.
"""

from __future__ import annotations

import csv
from pathlib import Path
from datetime import datetime
from typing import Any, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "logs"

# Action convention (0=UP 1=DOWN 2=LEFT 3=RIGHT 4=WAIT) is defined once, and
# only, in src/common/spec.py (ACTION_DELTA). This module never redefines or
# reinterprets it -- these are just English display labels for the CSV.
ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "WAIT"}
WAIT_ACTION_ID = 4

STEP_FIELDS: Sequence[str] = (
    "run_id", "scenario", "mode", "timestep", "agent_id",
    "current_row", "current_col",
    "action_id", "action_name", "is_wait",
    "intended_row", "intended_col",
    "next_row", "next_col",
    "goal_distance_before", "goal_distance_after", "progress",
    "conflict_type", "blocked",
    "done", "success", "failure_reason",
    "trigger_reason", "cbs_latency_ms",
)

SUMMARY_FIELDS: Sequence[str] = (
    "run_id", "scenario", "mode", "num_agents", "steps", "success",
    "timed_out", "makespan", "wait_count", "blocked_count",
    "vertex_conflicts", "edge_conflicts",
)

# Phase 3B-4: a THIRD, separate sidecar CSV -- one row per episode,
# written by write_monitor() (below), never read or written by
# write_steps()/write_summary(). Holds src.common.deadlock_monitor
# .DeadlockMonitorResult's fields, accepted duck-typed (this module
# imports nothing from deadlock_monitor.py, so no new cross-module
# dependency is introduced either direction).
#
# This is a Trigger-level + observational-candidate-signal record, NOT a
# final Operational Label (docs/deadlock_definition_v1.md Section 2
# concepts 4-5): trigger_type only ever says a Full joint-state repeat
# did or did not occur, never "this episode is a confirmed deadlock".
#
# Column order may grow by APPENDING new fields in a future phase (e.g.
# CBS-backed joint_plan_exists/candidate_state_recoverable/
# CBS_recovery_result, docs/deadlock_definition_v1.md Section 5.2/7.2) --
# no unimplemented placeholder columns are added here now.
MONITOR_FIELDS: Sequence[str] = (
    "run_id", "scenario", "mode",
    # Phase 3A -- Full joint-state repeat Trigger (never a final Label).
    "trigger_type", "trigger_reason_code", "trigger_timestep", "first_seen", "period",
    "blocking_type", "blocking_agents",
    # Phase 3A -- position-overlap integrity ONLY. NOT a full integrity
    # verdict: invalid_distance_rows (the offline analyzer's other
    # integrity signal) is not computed by this Monitor at all.
    "overlap_integrity_status", "overlap_timesteps",
    # Phase 3B-1 -- persistent_conflict: the Monitor's CURRENT, online
    # value as of the FINAL step of this episode -- NOT
    # scripts/analyze_failure_logs.py's offline, whole-episode
    # repeated_conflict_candidate_max/_max_consecutive (a different
    # quantity, computed at a different time; never substitute one for
    # the other). persistent_conflict_status=="NOT_COMPUTED" means the
    # Monitor was never given current_positions/intended_positions this
    # run; every field below it is then written "" (empty), NEVER "0" --
    # a blank means "not computed", not "computed and found none".
    "persistent_conflict_status", "persistent_conflict_incomplete_steps",
    "current_persistent_conflict_streak",
    "current_persistent_conflict_key_type",      # "vertex" | "edge" | ""
    "current_persistent_conflict_key_location",  # "row,col" or "row1,col1|row2,col2"
    "current_persistent_conflict_key_agents",     # ";"-joined, sorted agent ids
    # Phase 3B-2A -- no_progress: likewise the CURRENT, online,
    # final-step value -- NOT analyze_failure_logs.py's offline
    # no_progress_steps (a total count) or max_no_progress_span (the
    # episode's single largest streak). Same "" (not "0") rule applies
    # when no_progress_status=="NOT_COMPUTED".
    "no_progress_status", "no_progress_incomplete_steps",
    "current_no_progress_streaks",                # "agent_id=streak;agent_id=streak;..."
)


def _scenario_slug(scenario_name: str) -> str:
    """'scenario_empty_s11_n3.json' -> 'empty_s11_n3' (strip dir/ext/prefix)."""
    stem = Path(scenario_name).stem
    if stem.startswith("scenario_"):
        stem = stem[len("scenario_"):]
    return stem


def make_run_id(scenario_name: str, mode: str, timestamp: Optional[datetime] = None) -> str:
    """'<YYYYmmdd_HHMMSS>_<scenario_slug>_<mode>', e.g. 20260922_142500_empty_s11_n3_IL."""
    ts = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{_scenario_slug(scenario_name)}_{mode}"


def _agent_list(agents) -> str:
    """Sorted agent ids, ';'-joined -- a stable, order-independent serialization
    (used for MONITOR_FIELDS' blocking_agents/*_key_agents columns)."""
    return ";".join(str(a) for a in sorted(agents))


def _persistent_conflict_key_parts(key) -> tuple[str, str, str]:
    """DeadlockMonitorResult.current_persistent_conflict_key ->
    (key_type, key_location, key_agents) for MONITOR_FIELDS' three
    dedicated columns (Section 3.4):
      None                                                      -> ("", "", "")
      ("vertex", cell, frozenset(agents))                       -> ("vertex", "row,col", "a;b")
      ("edge", frozenset({cell_a, cell_b}), frozenset(agents))  -> ("edge", "r1,c1|r2,c2", "a;b")
    Cells inside an edge key are sorted so the two-cell location string is
    deterministic regardless of frozenset iteration order.
    """
    if key is None:
        return "", "", ""
    kind, location, agents = key
    if kind == "vertex":
        location_str = f"{location[0]},{location[1]}"
    else:  # "edge": `location` is a frozenset of the two cells
        location_str = "|".join(f"{cell[0]},{cell[1]}" for cell in sorted(location))
    return kind, location_str, _agent_list(agents)


def _no_progress_streaks_str(streaks) -> str:
    """DeadlockMonitorResult.current_no_progress_streaks (a tuple of
    (agent_id, streak) pairs, already sorted by agent_id) ->
    'agent_id=streak;agent_id=streak;...', "" if empty."""
    return ";".join(f"{aid}={streak}" for aid, streak in streaks)


class EpisodeLogger:
    """Collects step rows for one episode, then writes the two CSVs.

    See the module docstring for what wait_count/blocked_count vs.
    vertex_conflicts/edge_conflicts actually mean.
    """

    def __init__(self, run_id: str, scenario: str, mode: str, out_dir: Path | str = DEFAULT_OUT_DIR):
        self.run_id = run_id
        self.scenario = scenario
        self.mode = mode
        self.out_dir = Path(out_dir)
        self._step_rows: list[dict[str, Any]] = []
        self._vertex_conflict_events = 0
        self._edge_conflict_events = 0

    def log_step(
        self,
        *,
        timestep: int,
        agent_id: int,
        current: tuple[int, int],
        action_id: int,
        intended: tuple[int, int],
        next_pos: tuple[int, int],
        goal_distance_before: int,
        goal_distance_after: int,
        conflict_type: str,
        done: bool,
        success: bool,
        failure_reason: str = "",
        trigger_reason: str = "",
        cbs_latency_ms: Optional[float] = None,
    ) -> dict[str, Any]:
        """Append one step-level row (one call per agent per timestep).

        Returns the row dict (mainly for tests).

        is_wait / blocked are derived here (single source of truth, shared by
        the step CSV and the wait_count/blocked_count summary totals):
            is_wait = action_id == WAIT
            blocked = action_id != WAIT and next_pos == current
        conflict_type is the agent's OWN classification for this step (one
        of "none"/"wall"/"vertex"/"edge") -- see log_timestep_conflicts() for
        the separate, event-level vertex_conflicts/edge_conflicts totals.
        """
        is_wait = action_id == WAIT_ACTION_ID
        blocked = (not is_wait) and (tuple(next_pos) == tuple(current))
        row = {
            "run_id": self.run_id,
            "scenario": self.scenario,
            "mode": self.mode,
            "timestep": timestep,
            "agent_id": agent_id,
            "current_row": current[0],
            "current_col": current[1],
            "action_id": action_id,
            "action_name": ACTION_NAMES[action_id],
            "is_wait": is_wait,
            "intended_row": intended[0],
            "intended_col": intended[1],
            "next_row": next_pos[0],
            "next_col": next_pos[1],
            "goal_distance_before": goal_distance_before,
            "goal_distance_after": goal_distance_after,
            "progress": goal_distance_before - goal_distance_after,
            "conflict_type": conflict_type,
            "blocked": blocked,
            "done": bool(done),
            "success": bool(success),
            "failure_reason": failure_reason,
            "trigger_reason": trigger_reason,
            "cbs_latency_ms": cbs_latency_ms,
        }
        self._step_rows.append(row)
        return row

    def log_timestep_conflicts(self, vertex_events: int, edge_events: int) -> None:
        """Accumulate TRUE conflict-event counts for one timestep.

        Call this exactly once per timestep (not once per agent), with the
        simulator's own per-timestep event counts -- for
        MAPFStepSimulator this is
        (sim._last_vertex_conflict_count, sim._last_edge_conflict_count).
        See the module docstring for why row-counting by conflict_type would
        over-count and why these per-timestep totals are exact instead.
        """
        self._vertex_conflict_events += vertex_events
        self._edge_conflict_events += edge_events

    @property
    def step_rows(self) -> list[dict[str, Any]]:
        return list(self._step_rows)

    def write_steps(self) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{self.run_id}_steps.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=STEP_FIELDS)
            writer.writeheader()
            writer.writerows(self._step_rows)
        return path

    def write_summary(
        self,
        *,
        num_agents: int,
        steps: int,
        success: bool,
        timed_out: bool,
        makespan: int,
    ) -> Path:
        """wait_count/blocked_count: summed per-agent-step counts from the
        collected step rows (unambiguous, see module docstring).
        vertex_conflicts/edge_conflicts: accumulated event totals from
        log_timestep_conflicts(), NOT derived from step rows (see module
        docstring) -- 0 if log_timestep_conflicts() was never called.
        """
        wait_count = sum(1 for r in self._step_rows if r["is_wait"])
        blocked_count = sum(1 for r in self._step_rows if r["blocked"])

        row = {
            "run_id": self.run_id,
            "scenario": self.scenario,
            "mode": self.mode,
            "num_agents": num_agents,
            "steps": steps,
            "success": bool(success),
            "timed_out": bool(timed_out),
            "makespan": makespan,
            "wait_count": wait_count,
            "blocked_count": blocked_count,
            "vertex_conflicts": self._vertex_conflict_events,
            "edge_conflicts": self._edge_conflict_events,
        }
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{self.run_id}_summary.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerow(row)
        return path

    def existing_output_paths(self, *, monitor: bool = False) -> list[Path]:
        """Which of this run's target CSV paths already exist on disk --
        <run_id>_steps.csv/_summary.csv always, plus <run_id>_monitor.csv
        too when ``monitor=True``. A caller should check this BEFORE
        calling write_steps()/write_summary()/write_monitor() for this
        run_id, so a collision is caught before any of the files for
        this run -- not just the last one checked -- gets silently
        overwritten (all three share the same run_id, so a collision on
        one likely means a collision on all of them)."""
        paths = [
            self.out_dir / f"{self.run_id}_steps.csv",
            self.out_dir / f"{self.run_id}_summary.csv",
        ]
        if monitor:
            paths.append(self.out_dir / f"{self.run_id}_monitor.csv")
        return [p for p in paths if p.exists()]

    def write_monitor(self, result) -> Path:
        """Write the <run_id>_monitor.csv sidecar -- exactly one row, the
        Monitor's FINAL per-episode snapshot. ``result`` is a
        src.common.deadlock_monitor.DeadlockMonitorResult, accepted
        duck-typed (only named attributes are read; this module imports
        nothing from deadlock_monitor.py).

        Never touches *_steps.csv/*_summary.csv, their schema, or
        anything already written by write_steps()/write_summary() --
        this is a wholly separate file. See MONITOR_FIELDS' module-level
        comment for the full field semantics (Trigger vs. Label,
        overlap-only integrity, online-current vs. offline-aggregate,
        and the "" vs. "0" NOT_COMPUTED rule enforced below).
        """
        pc_ok = result.persistent_conflict_status == "OK"
        np_ok = result.no_progress_status == "OK"
        key_type, key_location, key_agents = (
            _persistent_conflict_key_parts(result.current_persistent_conflict_key) if pc_ok else ("", "", "")
        )
        row = {
            "run_id": self.run_id,
            "scenario": self.scenario,
            "mode": self.mode,
            "trigger_type": result.trigger_type,
            "trigger_reason_code": result.trigger_reason_code,
            "trigger_timestep": result.trigger_timestep if result.trigger_timestep is not None else "",
            "first_seen": result.first_seen if result.first_seen is not None else "",
            "period": result.period if result.period is not None else "",
            "blocking_type": result.blocking_type,
            "blocking_agents": _agent_list(result.blocking_agents) if result.blocking_agents else "",
            "overlap_integrity_status": result.overlap_integrity_status,
            "overlap_timesteps": result.overlap_timesteps,
            "persistent_conflict_status": result.persistent_conflict_status,
            "persistent_conflict_incomplete_steps": (
                result.persistent_conflict_incomplete_steps if pc_ok else ""
            ),
            "current_persistent_conflict_streak": (
                result.current_persistent_conflict_streak if pc_ok else ""
            ),
            "current_persistent_conflict_key_type": key_type,
            "current_persistent_conflict_key_location": key_location,
            "current_persistent_conflict_key_agents": key_agents,
            "no_progress_status": result.no_progress_status,
            "no_progress_incomplete_steps": result.no_progress_incomplete_steps if np_ok else "",
            "current_no_progress_streaks": (
                _no_progress_streaks_str(result.current_no_progress_streaks) if np_ok else ""
            ),
        }
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{self.run_id}_monitor.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=MONITOR_FIELDS)
            writer.writeheader()
            writer.writerow(row)
        return path
