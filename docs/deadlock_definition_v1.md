# Deadlock Definition v1 + Operational Label Rule v1

Status: **design document, v1, draft for review**. No code implements this yet.
Nothing in this document has been applied to change `src/simulator/`, `src/il/`,
`src/common/episode_logger.py`, any `scenarios/` file, or any existing CSV.
It formalizes, and checks for self-consistency against, the evidence
collected in Alpha 2 Phase 1 (`scenarios/controlled_variants/RESULTS.md`) and
Phase 2 evidence-audit / negative-control work (`scenarios/experimental_v1/`,
this session's chat record -- there is no separate phase-2 results file yet).

---

## 1. Research problem and scope

**Problem**: in the IL-navhint baseline, detect when a multi-agent rollout has
entered a state from which it will **not** reach all goals, and distinguish
that from (a) an agent that is simply slow but still progressing, (b) an
agent that has already reached its goal and is correctly idling, and (c)
a data-integrity artifact that makes the question unanswerable for that
episode.

**Scope -- what this document covers**:
- The existing deterministic IL-navhint baseline (`src/il/policy.py::NavHintPolicy`
  with a fixed, `eval()`-mode checkpoint) rolled out in the existing
  `src/simulator/mapf_step_simulator.py::MAPFStepSimulator`, via
  `scripts/run_il.py`.
- Finite episodes with a fixed `max_steps` and fixed, static `starts`/`goals`
  (no re-planning, no goal changes mid-episode).
- Offline, CSV-based evidence (`src/common/episode_logger.py` +
  `scripts/analyze_failure_logs.py`) as the primary data source, plus the
  small number of controlled, hand-/search-verified scenario instances built
  in Phase 1/2 (`scenarios/controlled/`, `scenarios/controlled_variants/`,
  `scenarios/experimental_v1/`).

**Explicitly out of scope (do not generalize this document to)**:
- Any future CBS, Hybrid, or mixed-policy runner. The determinism argument in
  Section 4 depends on the *specific* current policy+simulator pairing being
  stateless/deterministic; it does not transfer automatically to a system
  with internal planner state, re-planning, or stochastic exploration.
- Physical-robot deadlock (sensing noise, actuation delay, asynchronous
  execution). This document is about a discrete-time, fully-observed,
  synchronous simulator only.
- Any claim about the *scenario/map* being inherently unsolvable. Phase 1
  already proved, for all 9 controlled variants, that a conflict-free joint
  plan exists (replayed against the unmodified simulator). A label produced
  under this document describes **this policy's behavior**, never the
  instance's solvability.

---

## 2. Concept definitions

These six terms are used precisely and are **not interchangeable**. Each one
is a different stage of the same pipeline (raw data -> cheap heuristic ->
strict structural claim -> operational decision procedure -> final record ->
an unrelated, not-yet-available kind of evidence).

| # | Concept | What it is | Produced by | Already implemented? |
|---|---|---|---|---|
| 1 | **Failure Evidence** | Raw, uninterpreted per-step/per-episode facts: positions, actions, `conflict_type`, `success`/`timed_out`, etc. No judgement attached. | `src/common/episode_logger.py` (CSV rows) | Yes (unchanged) |
| 2 | **Stagnation Candidate** | A cheap, threshold-light first-pass heuristic computed purely from Failure Evidence: `stagnation_candidate` in {`yes`,`no`,`undetermined`} -- `yes` means "the episode failed AND at least one raw signal (`candidate_signals`) fired". Says nothing about *persistence* or *structure* of the failure. | `scripts/analyze_failure_logs.py` (unchanged) | Yes (unchanged) |
| 3 | **Coordination-stall / Deadlock Candidate** | The new, *strictly narrower* structural claim defined in Section 4: determinism + integrity preconditions hold, a **full** joint-state repeat is confirmed (not a local/subset repeat, Section 3.3), and the repeat is specifically a persistent vertex-, edge-, or mixed-type block (not just "some signal fired"). Every Coordination-stall is also `stagnation_candidate=yes`; the converse is false (Section 4.4). | This document (Section 4) | **No** -- defined here, not yet coded |
| 4 | **Trigger** | A cheap, real-time-computable condition that decides *when* to spend further evaluation budget (Section 5) on a candidate. Allowed to have false positives; it only decides "look closer", not "label this". | This document (Section 5) | **No** |
| 5 | **Operational Label** | The final categorical output for one episode after Trigger -> (optional) continuation -> integrity/solvability checks (Section 5): one of `SUCCESS`, `COORDINATION_STALL_VERTEX`, `COORDINATION_STALL_EDGE`, `COORDINATION_STALL_MIXED`, `NAV_CANDIDATE`, `UNKNOWN`, `INTEGRITY_ISSUE`. | This document (Section 5) | **No** |
| 6 | **CBS Recovery Result** | A *separate, not-yet-available* piece of evidence: "if an actual CBS solver were invoked, does it find a continuation to the goals?" -- and, critically, invoked **from where**: Section 5.2 distinguishes solvability of the *original start* (`joint_plan_exists`) from recoverability of the *specific stalled state* (`candidate_state_recoverable`); `CBS_recovery_result` is a third, separate field (enum `{SUCCESS, NO_SOLUTION, TIMEOUT, ERROR, NOT_RUN}`) that records what a real CBS run reports, for whichever of the two it was run against. Requires real CBS integration, which does not exist in this repository (`CBSAdapter` is referenced only as a placeholder in `scenario_loader.py`/`mapf_step_simulator.py`, never implemented). **None of these three fields may be merged into one another.** | Not available | **No** (no CBS solver exists to produce it) |

---

## 3. Signals and conditions

All of the following are read from existing, unmodified
`EpisodeLogger`/`analyze_failure_logs.py` fields unless noted. No new metric
is introduced; this section only says which existing fields mean what, and
how they combine.

### 3.1 Active agent vs. normal at-goal WAIT

An agent-step is **active** iff `goal_distance_before > 0` (the existing
`_active()` helper in `analyze_failure_logs.py`). `wait_count_active` (WAIT
while not yet at goal) and `wait_count_at_goal` (WAIT after arrival) are
already separate fields and must stay separate: a WAIT at goal is expected,
correct behavior, never a stagnation signal by itself.

### 3.2 No-progress and goal-distance stagnation

Two different granularities, both already implemented, both kept because
they catch different things:
- **No-progress** (`no_progress_steps`, `max_no_progress_span`): per-agent,
  "this one agent's own BFS goal-distance did not improve this step while
  active". Can fire for one stuck agent even if the team as a whole is still
  making progress elsewhere.
- **Goal-distance stagnation** (`goal_distance_stagnation_max_span`,
  `goal_distance_stagnation_tail_span`): system-level, "the summed BFS
  goal-distance across *all* agents has not reached a new minimum". Catches
  team-wide stalls even when individual per-agent spans look choppy.

### 3.3 Joint-state repeat and period; `first_seen` vs. `first_repeat`

A **full joint state** at time t, `S_t`, is the tuple of **all** agents'
positions at t (every agent, not a subset). A **full joint-state repeat** is
detected when `S_t` equals some earlier `S_s`. `analyze_failure_logs.py`
exposes `joint_state_first_repeat_timestep` (the *later* time t the state
recurs) and `joint_state_period` (`t - s`), but **not** `first_seen` (the
*earlier* time `s` that state was first visited) as its own field -- it is
only used internally to compute `period`. For this document:

> `first_seen = joint_state_first_repeat_timestep - joint_state_period`

`first_seen` is the moment the system actually *entered* the eventually-stuck
state; `first_repeat_timestep` is the later moment the repeat is *detected*.
These are different instants and must not be conflated (see the worked
numbers in Section 6).

**Full repeat vs. local/subset repeat (must not be confused).** A **local
(subset) repeat** is when a proper subset of agents' positions matches an
earlier configuration for that same subset, while at least one agent outside
the subset is at a *different* position than it was at that earlier time.
A local repeat is **not** a full joint-state repeat, and Section 4.2's
determinism guarantee does **not** apply to it: the deterministic transition
function `F` (Section 4.1) takes the *entire* joint position as input (every
agent's position feeds every nearby agent's 5x5 FOV), so as long as even one
agent elsewhere is still moving, the full state `S_t` is still changing
step-to-step even though a subset of it looks frozen. Only once **every**
agent's position matches the earlier snapshot simultaneously does `S_t = S_s`
hold and the guarantee in Section 4.2 take effect. Section 5.3 depends on
this distinction.

**Timestep indexing of a CSV row (verified against code, not assumed).**
In `scripts/run_il.py::run_episode`, one loop iteration does:
`before_positions = sim._positions` (this is `S_{t-1}`) ->
`actions = policy.act(...)` -> `obs, done, info = sim.step(actions)` (which
internally does `self._t += 1` *before* returning, so `info["t"]` is the
*new*, post-increment t) -> `_log_step(...)` is called with
`timestep=info["t"]`, `current=before_positions[agent_id]` (= `S_{t-1}`),
`next_pos=sim._positions[agent_id]` (= `S_t`), and
`conflict_type=sim._last_conflict_types[agent_id]` (computed *inside*
`step()` from the `S_{t-1} -> S_t` resolution). **Therefore: the CSV row
with `timestep = t` records the transition `S_{t-1} -> S_t`, and its
`conflict_type` explains why `S_t` looks the way it does -- it is evidence
*about arriving at* `S_t`, not evidence about `S_{t-1}`.** A row's
`conflict_type` must never be read as describing the state at its own
`timestep` value's *departure*; it describes the *arrival*.

One consequence: the row at `timestep = first_seen` describes the transition
`S_{first_seen-1} -> S_{first_seen}` -- i.e. **the step that produced the
state that later turns out to repeat**, which (per the worked example below)
is typically a perfectly ordinary, unblocked move. It is **not** the row that
explains the freeze. The row(s) that explain the freeze are the ones in
`timestep in (first_seen, first_repeat]` (Section 4.3 corrects this exact
point).

### 3.4 Persistent vertex blocking / persistent edge-swap blocking

- **Persistent vertex blocking**: in the CSV rows with
  `timestep in (first_seen, first_repeat]` (one full closing cycle -- see
  the indexing rule in 3.3), at least one agent's `conflict_type` is
  `vertex` at the *same* contested cell. Given a confirmed **full**
  joint-state repeat (Section 4.2), this same pattern is then guaranteed to
  recur identically every subsequent cycle -- it does not need to be
  re-checked empirically cycle after cycle.
- **Persistent edge-swap blocking**: same window, but `conflict_type` is
  `edge` (a mutual-swap attempt) for the same agent pair/cell pair.

Both are read directly from the per-step `conflict_type` column plus the
reconstructed `*_conflict_candidate` events; no new conflict logic is
introduced (the simulator's own classification, unchanged, is the source).

### 3.5 Oscillation and active WAIT -- kept as auxiliary, observational signals only

`oscillation_count` (A-B-A movement cycles) and `wait_count_active` are
**not** used as primary triggers in v1, because **zero** occurrences of
either have been observed across all evidence collected so far (Section 6) --
there is no calibration data for them yet. They remain in the schema
specifically so that if a future scenario produces them, they are captured
and can be folded into a later version rather than silently dropped.

### 3.6 Integrity failures (hard gate, not a stagnation signal)

`overlap_timesteps > 0` or `invalid_distance_rows > 0` on an episode means
the episode's own position/data history is not trustworthy for *any*
downstream judgement (see KI-1, Section 6). This is a **hard gate**: such an
episode is labeled `INTEGRITY_ISSUE` and excluded from deadlock-label
statistics entirely -- it is never pooled with clean episodes, regardless of
what its other signals say.

---

## 4. Deadlock Definition v1 (Coordination-Stall / Deadlock Candidate)

### 4.1 Why determinism and state-completeness matter

`NavHintPolicy.act(obs, positions)` (`src/il/policy.py`) is a pure function
of the *current* observation and positions; the only state it carries across
an episode is `_dist`/`_goals`, both fixed once in `reset()`. The checkpoint
is loaded with `model.eval()` (`src/il/model_navhint.py`) and inference runs
under `@torch.no_grad()` -- no dropout/batchnorm randomness, no sampling.
`MAPFStepSimulator.step()` (`src/simulator/mapf_step_simulator.py`) is a pure
function of the current `_positions` and the chosen `actions` -- no RNG, no
hidden history. The 5x5 FOV observation (`_build_obs`/`_extract_local_grid`)
is itself a pure function of current positions and the static grid/goals.

Composed together, "joint position at t" -> "joint position at t+1" is a
single, **time-invariant deterministic function F of the joint position
alone** (nothing else from the episode's history leaks in). This is why
state-completeness matters: if *any* of the above gained hidden state (a
recurrent policy, a stateful adapter, mid-episode re-planning), the argument
in 4.2 would no longer hold and this definition would not transfer without
re-derivation -- which is exactly why Section 1 excludes any future
Hybrid/CBS runner from this document's scope.

### 4.2 The general period-p fixed-cycle argument (period=1 is not a special case)

If the **full** joint position at time t equals the full joint position at
an earlier time s (`first_seen`), i.e. `S_t = S_s` with every agent included
(Section 3.3), then because the transition is the single deterministic
function F described above, `F` applied to `S_t` must produce the exact same
successor as `F` applied to `S_s` did -- namely `S_{s+1}`. Inductively, the
entire trajectory from t onward is forced to replay the trajectory from s
onward, **forever**, regardless of the value of `period = t - s`. This holds
for **any** `period >= 1` by the same single argument: once one full cycle
`S_s, S_{s+1}, ..., S_{s+period-1}, S_s` is confirmed to close (i.e. the state
truly returns to `S_s`), the whole cycle repeats verbatim forever after. This
is not a statistical tendency observed across samples; given 4.1's
preconditions, it is a direct consequence of `F` being well-defined and
time-invariant, with no dependence on how large `period` is.

`period = 1` is simply the smallest possible case (`F(S_s) = S_s`, a true
fixed point) and is the **only** value observed in evidence collected so far
(Section 6) -- every repeat found to date is a full freeze, not a longer
back-and-forth cycle. This is recorded as an open empirical gap (no
`period > 1` case has been exercised yet), **not** as a difference in the
strength of the mathematical guarantee: a confirmed `period = 7` cycle would
be exactly as permanent as a confirmed `period = 1` cycle, by the same
argument above. Section 5.3 relies on this general-`period` statement, not
on a period-1-only special case.

### 4.3 The decision rule

An episode is a **Coordination-Stall / Deadlock Candidate** at a type
(`VERTEX` or `EDGE`) iff **all** of the following hold:

1. **Determinism precondition** (checked once per run configuration, not
   per episode): the checkpoint, policy, and simulator code match what
   Section 4.1 describes (fixed `eval()`-mode checkpoint, no RNG). This
   precondition has three possible outcomes, not two: **holds**, **does not
   hold**, or **cannot be verified** (e.g. no checkpoint hash was recorded
   for this run). Only the first outcome allows this definition to proceed.
   The other two -- "does not hold" *and* "unverifiable" -- both route to
   `UNKNOWN` (reason `PRECONDITION_UNVERIFIED`), never silently treated as
   "probably fine" (Section 4.4).
2. **Integrity precondition**: `overlap_timesteps == 0` and
   `invalid_distance_rows == 0` for this episode (Section 3.6). If violated,
   label is `INTEGRITY_ISSUE` instead, and the episode is excluded here.
3. **Full joint-state repeat confirmed** (Section 3.3 -- *full*, not a
   local/subset repeat): there exist `first_seen < first_repeat` with
   `S_{first_seen} = S_{first_repeat}` across **every** agent, and the
   repeated state is **not** the trivial "all agents already at their goal"
   state (this excludes normal successful termination). A local/subset
   repeat (some agents match, at least one other agent elsewhere still
   differs) does **not** satisfy this condition -- see Section 5.3.
4. **A persistent blocking type is evidenced in the closing cycle**: in the
   CSV rows with `timestep in (first_seen, first_repeat]` (one full period;
   see the indexing rule in Section 3.3 -- **not** the row at
   `timestep = first_seen`, which describes arriving *at* the state, not the
   blocked attempt to leave it), at least one agent shows
   `conflict_type = vertex` (-> type `VERTEX`) and/or at least one agent pair
   shows `conflict_type = edge` (-> type `EDGE`), at a fixed location/pair.
   If **both** types are present and persistent in that same window (e.g.
   one pair vertex-blocked while a different pair is simultaneously
   edge-blocked, possible with >=4 agents), the type is `MIXED`
   (`COORDINATION_STALL_MIXED`, Section 4.4) -- v1 does not force a single
   type pick when both are genuinely present. If the window shows **no**
   agent with `conflict_type` in `{vertex, edge}` at all (e.g. every agent
   is actively `WAIT`ing rather than being bounced back), condition 4 fails
   and the episode falls to `UNKNOWN` (reason `REPEAT_WITHOUT_BLOCKING_EVIDENCE`)
   -- a confirmed repeat alone does not automatically mean *inter-agent*
   blocking caused it.

**Worked example (`corridor_bay_n2`, run `20261010_203403`, verified against
the actual step CSV):** `first_repeat_timestep = 5`, `period = 1`, so
`first_seen = 4`. The row at `timestep = 4` shows both agents moving
normally (`conflict_type = none`, arriving at `S_4 = {(2,4), (2,6)}`) -- it
is *not* blocked. The row at `timestep = 5` (the window `(4, 5]`) shows both
agents attempting to continue toward each other and bouncing back to `S_4`
with `conflict_type = vertex` for both -- **this** is the evidence, and it is
at `first_repeat` (= 5), not at `first_seen` (= 4). This corrects an error in
an earlier draft of this document, which looked up the evidence at
`first_seen`.

**A single collision, or `timed_out=True` alone, is explicitly NOT
sufficient** to satisfy this definition: condition 4 requires the blocking to
be shown, via condition 3, to recur identically -- a one-off bump that
resolves on the next step never produces a joint-state repeat and therefore
never qualifies. Likewise a slow-but-still-progressing episode that simply
runs out of `max_steps` without ever repeating a joint state is **not** a
Coordination-Stall under this definition -- see `UNKNOWN` below.

### 4.4 Categories for unclear cases

| Operational Label | Condition |
|---|---|
| `SUCCESS` | `success = True` |
| `COORDINATION_STALL_VERTEX` | 4.3 holds, blocking type is vertex only |
| `COORDINATION_STALL_EDGE` | 4.3 holds, blocking type is edge only |
| `COORDINATION_STALL_MIXED` | 4.3 holds, **both** persistent vertex and persistent edge blocking are evidenced in the same closing-cycle window (different agents/pairs) -- **zero instances observed to date**; category defined so a future mixed case is not forced into an arbitrary single type. |
| `NAV_CANDIDATE` | Failed episode where the dominant blocking is **not** inter-agent (`blocked_by_wall` dominates `blocked_count`, or a single-agent episode fails to progress despite a reachable goal) -- **zero instances observed to date**; category defined, not yet calibrated (Section 6/8). |
| `UNKNOWN` | Failed episode (`success=False`), integrity clean, and **any** of: (a) no full joint-state repeat observed by end of episode/horizon (still progressing, just slowly, or ran out of steps first); (b) a full repeat *is* confirmed but condition 4 fails -- no `vertex`/`edge` evidence in the closing-cycle window (reason `REPEAT_WITHOUT_BLOCKING_EVIDENCE`); (c) evidence does not clearly favor `NAV_CANDIDATE` over `COORDINATION_STALL_*` (reason `INSUFFICIENT_NAV_COORD_EVIDENCE`); (d) the determinism precondition does not hold or cannot be verified (reason `PRECONDITION_UNVERIFIED`, Section 4.3 condition 1). |
| `INTEGRITY_ISSUE` | `overlap_timesteps > 0` or `invalid_distance_rows > 0`. Excluded from all of the above. |

**Label priority when `SUCCESS` and `INTEGRITY_ISSUE` conditions both hold:
`INTEGRITY_ISSUE` wins, unconditionally.** `overlap_timesteps`/
`invalid_distance_rows` are computed over the whole episode (Section 3.6)
and nothing in the simulator prevents an episode from accumulating an
integrity violation at some point and *still* reaching `success=True` by the
end -- these two conditions are not mutually exclusive in the data model,
only unobserved together so far (Section 6's 4 KI-1 instances all happen to
be failures). The table above must be read as **checked top-to-bottom, first
match wins, with the integrity check logically evaluated before the success
check** (matching Section 4.3 condition 2's position ahead of conditions
3-4, and Section 5.2 step 1's position ahead of step 2): a `success=True`
episode that also has an integrity violation is labeled `INTEGRITY_ISSUE`,
**not** `SUCCESS`, and must not be used as evidence for anything (including,
critically, as a candidate "successful episode with a transient block"
example for `K_label` calibration, Section 5.4 -- any future candidate for
that purpose must itself pass the integrity gate first).

**`UNKNOWN` is a superset of, not identical to,
`analyze_failure_logs.py`'s existing `stagnation_candidate = undetermined`.**
Checked against that module's actual logic: `stagnation_candidate` is `"no"`
if `success=True`, `"undetermined"` if the episode failed **and**
`candidate_signals` is empty, and `"yes"` if the episode failed **and** at
least one raw signal fired (and `joint_state_repeat`, when present, is
*always* included threshold-free -- see that module's own tests). This means
`stagnation_candidate = undetermined` only ever covers sub-case (a) above (no
repeat, no other signal). It does **not** cover (b)/(c)/(d): a case can be
`stagnation_candidate = yes` (some signal fired, e.g. an oscillation-window
threshold) while still being `UNKNOWN` under this document (no confirmed
*full* repeat with blocking evidence), and a case can have
`stagnation_candidate = undetermined` while a future, richer analysis still
resolves it to something other than `UNKNOWN`. **The two fields must be
reported side by side, never substituted for one another.**

---

## 5. Operational Label Rule v1

### 5.1 Trigger vs. Label (explicitly separated)

Two distinct trigger conditions exist, and they lead to **different**
downstream handling (Section 5.3) -- they must not be collapsed into one:

- **Trigger-Full** = a **full** joint-state repeat is detected
  (`S_{first_repeat} = S_{first_seen}` across every agent, Section 3.3/4.3).
  Under the determinism precondition (Section 4.1), this is the strong case:
  Section 4.2's guarantee already applies, so by the time this trigger fires,
  the outcome is mathematically settled (see 5.3).
- **Trigger-Partial** = a **local/subset repeat** is detected (some agents'
  positions match an earlier snapshot while at least one other agent is
  still at a different position than it was then, Section 3.3). This is a
  genuinely weaker, "worth watching" signal: the determinism guarantee does
  **not** yet apply, because the full joint state has not actually repeated.
  **False-positive guard (definitional, not a numeric threshold):** the
  matching subset must include at least one **active** agent
  (`goal_distance_before > 0`, Section 3.1). A subset consisting *only* of
  agent(s) already at their goal must **not** fire Trigger-Partial: such an
  agent's position trivially "repeats" itself on every correct, expected
  at-goal `WAIT` step (Section 3.1), so without this guard, ordinary
  goal-arrival idling elsewhere in the team would spuriously fire the
  trigger on every single step after that arrival, regardless of whether
  the still-active agents are actually stuck. This closes an obvious,
  structurally-predictable false-positive hole using a concept (`active`)
  v1 already defines in Section 3.1 -- it does **not** introduce a new
  numeric threshold, and Trigger-Partial otherwise remains an unvalidated,
  future-extension concept (Section 5.3/8).
- **Label** = the final `Operational Label` (Section 4.4), assigned only
  after the candidate evaluation below. Neither trigger by itself is
  sufficient to produce a `COORDINATION_STALL_*` label -- a Trigger only
  decides whether to spend the evaluation budget in Section 5.2.

### 5.2 Candidate evaluation: integrity, solvability, and recovery

Once triggered, a candidate must pass, in order:

1. **Integrity check** (Section 3.6) -- fail -> `INTEGRITY_ISSUE`, stop.
2. **Solvability context** (recorded, not used to override the label) --
   **three different questions, three different fields, never merged**:
   1. `joint_plan_exists: {YES, NO, NOT_CHECKED}` -- does a conflict-free
      joint plan exist from the instance's **original start state**? (Phase
      1's hand-written/search-found replay method, independent of CBS.) This
      says the *instance* is not inherently unsolvable; it says **nothing**
      about whether the *specific stalled mid-episode joint state* can still
      reach the goals.
   2. `candidate_state_recoverable: {YES, NO, NOT_CHECKED}` -- does a
      conflict-free plan exist **from the stalled joint state at
      `first_seen`/`first_repeat` onward** (i.e. re-solving from where the
      policy actually got stuck, not from the original start)? This is a
      **different computation** (it would need to be re-run from the
      candidate's current positions) and has **never been performed for any
      instance in this repository** -- it is `NOT_CHECKED` for all evidence
      in Section 6. A solvable original start does **not** imply a solvable
      stalled state (an IL rollout could, in principle, walk itself into a
      sub-configuration that is *actually* unrecoverable even though the
      original instance was solvable from t=0 by a different, coordinated
      plan) -- this document takes no position on whether that has happened
      here, only that it has not been checked.
   3. `CBS_recovery_result: {SUCCESS, NO_SOLUTION, TIMEOUT, ERROR, NOT_RUN}`
      -- if an actual CBS solver were invoked (from either the original
      start or the candidate state -- the field must record *which*), what
      does it report? Always `NOT_RUN` today (no CBS solver exists in this
      repository). The five-way enum exists so that, once CBS is integrated,
      "it timed out" and "it searched and proved no solution" and "it
      crashed" are never collapsed into a single "no" the way a boolean
      would.
3. **Recovery evaluation via IL-only continuation -- only exercised for
   Trigger-Partial candidates** (Section 5.1/5.3). For a **Trigger-Full**
   candidate, this step is **skipped**: conditions 3-4 of Section 4.3 and
   the Section 4.2 argument already confirm
   `COORDINATION_STALL_VERTEX`/`EDGE`/`MIXED` directly, with
   `continuation_used=False`. For a **Trigger-Partial** candidate, take the
   *same*, unmodified IL policy and simulator, and continue the rollout (no
   intervention, no other policy) for a further, **pre-registered fixed
   horizon `K_label`** steps beyond the trigger point. Observe whether a
   *full* joint-state repeat ever locks in within that window.
   - If a full repeat locks in and persists through the whole `K_label`
     window -> confirm `COORDINATION_STALL_VERTEX`/`EDGE`/`MIXED` (per
     Section 4.3's conditions, now satisfied).
   - If no full repeat ever locks in, or the team's joint position keeps
     changing -> label is **not** a stall; re-classify per whatever the
     episode does afterward (`SUCCESS` if it then reaches goals, `UNKNOWN`
     otherwise).

### 5.3 Why `K_label` continuation is unnecessary for Trigger-Full, and necessary for everything else

**This is the current, confirmed scope boundary of v1 -- stated precisely so
it is not read more broadly than the evidence supports.**

For **Trigger-Full** (Section 5.1: a *full* joint-state repeat, across every
agent, is already confirmed), Section 4.2's argument **proves** -- for any
`period >= 1`, not just `period = 1` -- that the trajectory is locked into
repeating the same length-`period` cycle of states forever, *given* Section
4.1's determinism precondition holds. **This is not the same claim for every
`period`**: for `period = 1` the joint position is a true fixed point and
literally stops changing from `first_seen` onward (every subsequent step is
identical). For `period > 1`, the joint position keeps changing step to
step -- the team visits `S_s, S_{s+1}, ..., S_{s+period-1}` in a genuine
repeating loop -- but it is permanently confined to that same finite set of
`period` states and never escapes to reach the goals. Either way, no new,
previously-unvisited joint state (in particular no goal-reaching state) can
ever occur again once the cycle is confirmed, which is the property that
makes continuation redundant for Trigger-Full: running a `K_label`
continuation on a Trigger-Full candidate would be a redundant,
zero-information check -- whether the specific pattern is "frozen" (p=1) or
"looping" (p>1), the answer to "can this still reach the goals" is already
known analytically.
**v1 can and does finalize a `COORDINATION_STALL_*` label on a Trigger-Full
candidate without spending any continuation budget.** Every one of the 9
Coordination-Stall instances in Section 6 (8 vertex + 1 edge) is a
Trigger-Full case confirmed this way.

**Correction to an earlier draft of this document**: an earlier version of
this section described Experiment 1 (Section 6) as a case where "a
still-moving third agent could later perturb an already-confirmed repeat" --
this is **not actually what Experiment 1 showed**, and would in fact
contradict Section 4.2 if it were a full repeat. Re-examined precisely: in
that run, the third agent reaches its own goal and stops moving in the
*same* step (t=4) that the other two agents enter their blocked state. From
t=4 onward, **all three** agents' positions are constant, so the *full*
3-agent joint state is itself already a period-1 fixed point from t=4 --
this is a Trigger-Full case like all the others, not a counterexample, and
it required no continuation either. What Experiment 1 did **not** test (and
what no evidence in this repository tests) is the genuinely open question
below.

**`K_label` continuation earns its keep only for cases that are *not*
Trigger-Full**, i.e. for **Trigger-Partial** (Section 5.1) and related
not-yet-settled situations:
- A **local/subset repeat**: two (or more) agents' positions match an
  earlier snapshot while some *other* agent is still moving. The full joint
  state has **not** repeated yet, so Section 4.2 does not apply -- whether
  the still-moving agent's eventual position change perturbs the apparently-
  stuck pair (by altering their 5x5 FOV input) before a full repeat ever
  locks in is an open, empirically-testable question that this repository
  has not yet exercised. (Designing and running such a case was the explicit
  goal of the Section 6 perturbation experiment; it did not end up
  producing a Trigger-Partial situation, because the third agent stopped at
  the same timestep the pair froze, rather than continuing to move
  afterward.)
- General, weaker stagnation signals that have **not** yet produced a
  confirmed full repeat at all (e.g. `max_no_progress_span` climbing, or
  `stagnation_candidate=yes` from a non-repeat signal) -- these say "this
  looks bad" without proving permanence; only continued observation (or a
  future, successfully-recovering counterexample) can settle them.
- Any system with **stateful** policy/adapter components, or any
  **non-deterministic** component (sampling, async timing) -- Section 4.1's
  precondition does not hold, so even a full repeat carries no permanence
  guarantee, and empirical continuation becomes the *only* available
  evidence.

**v1's confirmed scope is therefore exactly**: Trigger-Full candidates can be
labeled without continuation, today, on existing evidence. Trigger-Partial
and non-deterministic/stateful cases are explicitly **future extension**
work -- v1 defines the concept (`continuation_used`, `K_label_window`,
`continuation_outcome` in Section 7.2) but does not fix a numeric horizon or
claim any case of this kind has been evaluated.

### 5.4 `K_label`: purpose, and why its value is not set here

**Purpose**: give the policy a fair, fixed chance to escape a *suspected*
stall before the stall is finalized into a label, specifically to avoid
mislabeling a transient slow patch (brief block/WAIT followed by recovery)
as a permanent stall.

**Why no number is proposed in this document**: Section 6's evidence audit
found **zero** examples, anywhere in this repository's existing or newly
collected IL-only runs, of a *successful* episode that experienced an active
block/WAIT/no-progress window and then recovered. Without at least a small
set of such examples (what span lengths occur in episodes that still
succeed), any numeric `K_label` would be an unvalidated guess, not a
calibrated threshold. **`K_label`'s value is explicitly left TODO pending
that validation evidence** (Section 8).

---

## 6. Evidence and limitations

Every bullet below is tagged **[OBSERVED]** (a specific number/fact read
from an actual CSV produced by a real `scripts/run_il.py` execution) or
**[DESIGN ASSUMPTION]** (a choice this document makes that is not yet
empirically validated). Source run_ids are given so each claim can be
re-checked against the files still on disk.

- **[OBSERVED]** 8 "clean" failing instances (3 original controlled scenarios
  -- `corridor_bay_n2`, `intersection_n2`, `bottleneck_n4` -- plus 5 Category-A
  variants -- `corridor_bay_n2_offset`, `intersection_n2_4way`,
  `intersection_n2_lshape`, `intersection_n2_tight`, `bottleneck_n4_tightdoor`)
  all show: `overlap_timesteps=0`, `invalid_distance_rows=0`,
  `blocked_by_vertex` = 100% of `blocked_count`, `joint_state_period=1`,
  `oscillation_count=0`, `wait_count_active=0`, and
  `goal_distance_stagnation_tail_span` equal (or within 1) of
  `max_no_progress_span` -- i.e. persistent **vertex** stagnation, never
  recovered before `max_steps=160`. (run_ids: `20261010_203403`,
  `20261010_203644`, `20261010_203842` for the originals;
  `20261010_212310/212314/212317/212320/212301` for the 5 variants.)
- **[OBSERVED]** New perturbation experiment
  (`scenarios/experimental_v1/scenario_corridor_bay_n2_3rdagent_perturbation.json`,
  run_id `20261011_003253_...`): `edge_conflicts=157`, `vertex_conflicts=0`,
  `blocked_by_vertex=0`, `overlap_timesteps=0`, `invalid_distance_rows=0`,
  `success=False`, `timed_out=True`, `agents_at_goal=1/3`. First persistent
  edge-swap block at t=4, `joint_state_period=1` from t=4 onward. This is
  the **first and only edge-type** stall observed in this project to date --
  every other instance so far is vertex-type. **Solvability-evidence
  provenance (do not conflate the two):** `joint_plan_exists=YES` for this
  instance was established the **same way as Phase 1** (an exhaustive
  joint-state BFS over the real `MAPFStepSimulator` resolution rules, then
  replayed and verified against the real simulator, horizon=9) but was run
  as an **ad hoc, one-off scratch check for this single experiment**, not as
  part of the committed `tests/test_controlled_variants.py` suite that
  covers the Phase 1 original-4 and 9-variant instances. Both are
  independent of, and neither is, a CBS solver run (Section 5.2) -- no CBS
  solver exists in this repository.
- **[OBSERVED]** Single-agent NAV control
  (`scenarios/experimental_v1/scenario_bottleneck_n4_tightdoor_1agent_control.json`,
  run_id `20261011_003255_...`): `success=True`, 4 steps (= BFS lower bound),
  zero wall-blocks, zero conflicts, `progress_ratio=1.0`, navigating a genuine
  detour (the only agent/goal pair in this repository whose straight-line
  path is wall-blocked, forcing a door detour). **[DESIGN ASSUMPTION]**: this
  shows this *specific* agent/goal pair is not a navigation problem in
  isolation; it does **not** prove the other 3 agents in that scenario are
  equally innocent, nor that the 4-agent failure's root cause is fully
  explained -- only 1 of 4 agents from that scenario has been isolated this
  way.
- **[OBSERVED]** 2 success controls (`cycle_ring_n4`, 2 steps;
  `scenario_empty_s11_n3`, 9 steps) show **zero** `blocked`/`wait_count_active`/
  `oscillation`/`joint_state_repeat` events of any kind -- both are
  low-contention instances with no agent ever contesting a cell.
- **[OBSERVED]** Across **every** `success=True` summary CSV in
  `outputs/logs/` (full directory scan, Phase 2 evidence audit), there is
  **no** episode showing an active block/WAIT/no-progress window followed by
  recovery to success. **This is the central evidentiary gap for `K_label`
  calibration (Section 5.4)** -- not fabricated or worked around.
- **[OBSERVED]** 4 Category-B instances (`corridor_bay_n2_3agent`,
  `corridor_bay_n2_4agent`, `bottleneck_n4_convoy`, `bottleneck_n4_imbalanced`)
  have `overlap_timesteps` of 1, 157, 160, 160 respectively -- the documented,
  intentionally-preserved Alpha 1 simulator quirk KI-1
  (`tests/test_simulator_characterization.py::TestKnownQuirkKI1`: vertex
  resolution is single-pass, so a bounce-back target is not re-checked
  against an already-resolved move into that same cell). **[DESIGN
  ASSUMPTION]**: per Section 3.6/4.3, these 4 are excluded from all label
  statistics (`INTEGRITY_ISSUE`), not merged with the 8 clean failures, and
  this document does not attempt to fix KI-1 (out of scope).
- **[DESIGN ASSUMPTION]** `NAV_CANDIDATE` (Section 4.4) is a defined but
  **empty** category: no evidence collected so far populates it. This is
  stated as a limitation, not evidence that individual navigation failure is
  impossible under this policy.
- **[DESIGN ASSUMPTION]** No `period > 1` joint-state cycle has been
  observed. Section 4.2's argument is written to cover the general case, but
  only the `period = 1` case has empirical support today.
- **[OBSERVED]** For all 9 Coordination-Stall instances, `joint_plan_exists`
  (solvability **from the original start**) is `YES` -- for the 8
  vertex-type instances via Phase 1's committed
  `tests/test_controlled_variants.py` replay suite, and for the 1 edge-type
  instance via the separate, ad hoc Phase 2 scratch check described above
  (same method, not the same test file). **[DESIGN ASSUMPTION]** None of these 9
  have had `candidate_state_recoverable` checked (solvability **from the
  specific stalled mid-episode state**, Section 5.2) -- that computation has
  not been run for any instance. The two must not be conflated: a solvable
  original start is not evidence about the recoverability of the particular
  configuration the policy stalled in.
- **[DESIGN ASSUMPTION]** **No online Trigger-Partial detector exists, and no
  systematic search for Trigger-Partial cases has been performed.** The one
  perturbation experiment (Section 6) that was specifically designed to try
  to produce a local/subset repeat did not land on one (the third agent
  stopped at the same timestep the pair froze) -- but that is a single ad
  hoc scratch-script observation from one offline rerun of one instance, not
  the result of running a real-time Monitor over the full evidence set. It
  must **not** be read as "we checked and found zero Trigger-Partial cases
  among existing evidence" -- no such check has been built or run. `K_label`
  continuation therefore remains entirely unexercised and unevaluated in
  this repository; only the continuation-free, Trigger-Full path (Section
  5.3) has real evidence behind it.

---

## 7. Implementation contract for a future Deadlock Monitor v1 (interface only -- not implemented)

This section specifies the **shape** of the input/output a future Monitor
would need. No Monitor code is written in this change. A key correction from
an earlier draft: **the Monitor must not assume it already has the
offline analyzer's final, whole-episode aggregates** (e.g.
`stagnation_candidate`, `progress_ratio`) while an episode is still running.
The three subsections below separate what is genuinely available step by
step, what the Monitor must build up itself, and what can only be known
after the episode ends.

### 7.1 (a) Per-step online primitives (available the instant each step happens)

| Primitive | Source | Notes |
|---|---|---|
| Current full joint position `S_t` | `MAPFStepSimulator._positions` after `step()` returns | One tuple per agent, this step only |
| Per-agent `conflict_type` for *this* step | `MAPFStepSimulator._last_conflict_types` | Describes the `S_{t-1} -> S_t` transition just taken (Section 3.3's indexing rule) |
| Action taken per agent | the `actions` dict passed into `step()` | |
| `done` / `timed_out` flags, current `t` | `MAPFStepSimulator.step()`'s return `info` | |
| Episode/scenario metadata | scenario file (`scenario_id`, map, agent count) | Static, known before the episode starts |
| Checkpoint identity | sha256 of the `.pt` file (as already checked in `scripts/smoke_test_il.py`) | Known before the episode starts; needed for the Section 4.1 determinism precondition |

### 7.1 (b) History / integrity state the Monitor must maintain itself

None of the following are handed to the Monitor for free -- they must be
accumulated incrementally, one step at a time, as the episode runs (the
*computation* is the same one `analyze_failure_logs.py` already does
offline; only *when* it runs differs):

| State | How it is built online | Notes |
|---|---|---|
| Seen-states table (`{S -> first-seen timestep}`) | **Must be initialized with `S_0` (the initial joint position, before any action is taken) registered at `t=0` before the first `step()` call** -- then, for each subsequent step, look up `S_t` before inserting it, to detect a repeat. This matches `analyze_failure_logs.py::analyze_episode`'s own loop, which scans `t` starting at `t_first - 1` (i.e. the pre-step initial state, not `t=1`) when building `pos[aid][...]` for the joint-state-repeat check -- an online Monitor that started its table at `t=1` instead of `t=0` would silently disagree with the offline analyzer on `first_seen`/`period` for any instance (such as `intersection_n2_tight`, Section 6) whose freeze starts at `first_seen=0`. | This is exactly what makes `first_seen`/`first_repeat`/`period`/Trigger-Full-vs-Partial (Section 5.1) computable **in real time**, not just offline |
| Running `overlap_timesteps` count | Check, each step, whether two agents' `next` position coincide (Section 3.6) | Incremental; does not need the full trajectory |
| Running `invalid_distance_rows` count | Check, each step, whether any `goal_distance_before/after < 0` | Incremental |
| Per-pair/per-cell persistent-blocking tally (for `blocking_type`, Section 3.4/4.3) | Track which `(agent, cell)` or `(agent-pair, cell-pair)` combinations recur with `conflict_type != none` across the closing cycle once a repeat is found | Only needs the one cycle's worth of rows once Trigger-Full fires, not the whole episode |

### 7.1 (c) Episode-end-only (offline) aggregates -- must not be assumed mid-episode

These require the **whole, finished** trajectory and are computed only after
`done` is reached (by `scripts/analyze_failure_logs.py`, unchanged):
`progress_ratio`, `max_no_progress_span` (the true maximum, as opposed to a
running current span), `goal_distance_stagnation_tail_span` (needs
`t_last`), `oscillation_count` totals, `repeated_conflict_candidate_*`
aggregates, and `stagnation_candidate` itself. A live Monitor may track
*running* versions of some of these (e.g. current no-progress streak length)
but must not report the *final* aggregate value before the episode ends.

### 7.2 Outputs (per episode, or per candidate once triggered)

| Field | Type | Meaning |
|---|---|---|
| `episode_id` / `run_id` | string | Links back to the source CSVs |
| `trigger_type` | enum `{NONE, PARTIAL, FULL}` | Section 5.1 -- which trigger condition fired, if any |
| `trigger_reason_code` | enum | See 7.3 |
| `first_seen` | int or null | Section 3.3 |
| `first_repeat_timestep` | int or null | Section 3.3 |
| `period` | int or null | Section 3.3 |
| `blocking_type` | enum `{VERTEX, EDGE, MIXED, NONE}` | Section 3.4 / 4.3 |
| `integrity_status` | enum `{CLEAN, INTEGRITY_ISSUE}` | Section 3.6 |
| `determinism_precondition` | enum `{HOLDS, VIOLATED, UNVERIFIABLE}` | Section 4.3 condition 1 -- three-way, not boolean |
| `continuation_used` | bool | Whether Section 5.3's empirical continuation was actually run (only applicable/expected for `trigger_type=PARTIAL`; always `False` for `trigger_type=FULL`, since 5.3 proves it is redundant there) |
| `K_label_window` | int or null | The horizon actually used, if continuation was run |
| `continuation_outcome` | enum `{NOT_RUN, CONFIRMED_STALL, RECOVERED}` | Section 5.2 step 3 |
| `joint_plan_exists` | enum `{YES, NO, NOT_CHECKED}` | Section 5.2 step 2.1 -- solvability from the **original start** |
| `candidate_state_recoverable` | enum `{YES, NO, NOT_CHECKED}` | Section 5.2 step 2.2 -- recoverability from the **stalled state itself**; distinct from the above |
| `CBS_recovery_result` | enum `{SUCCESS, NO_SOLUTION, TIMEOUT, ERROR, NOT_RUN}` | Section 5.2 step 2.3 -- always `NOT_RUN` until a real CBS solver exists |
| `label` | enum (Section 4.4) | Final Operational Label |
| `label_reason_code` | enum | See 7.3 |
| `timestamp_wallclock` | datetime | When the Monitor produced this record (distinct from simulation timestep) |
| `timestep` | int | The simulation timestep the record refers to |

### 7.3 Reason codes (proposed enum values, not yet wired to code)

- Trigger: `TRIGGER_FULL_REPEAT`, `TRIGGER_PARTIAL_REPEAT`, `TRIGGER_NONE`
- Label: `LABEL_SUCCESS`, `LABEL_COORD_STALL_VERTEX`, `LABEL_COORD_STALL_EDGE`,
  `LABEL_COORD_STALL_MIXED`, `LABEL_NAV_CANDIDATE`,
  `LABEL_UNKNOWN_NO_REPEAT`, `LABEL_UNKNOWN_REPEAT_WITHOUT_BLOCKING_EVIDENCE`,
  `LABEL_UNKNOWN_INSUFFICIENT_NAV_COORD_EVIDENCE`,
  `LABEL_UNKNOWN_PRECONDITION_UNVERIFIED`, `LABEL_INTEGRITY_ISSUE`

---

## 8. Validation conditions

### 8.1 Self-consistency check against existing evidence

Applying Section 4.4's categories (and the corrected 4.2/4.3 rules) to every
episode referenced in Section 6 produces no contradiction. All runs below
used the same fixed, `eval()`-mode checkpoint with no RNG, so
`determinism_precondition = HOLDS` for every row (none are `UNVERIFIABLE`):

| Instance | `trigger_type` | `integrity_status` | `blocking_type` | Expected label | Matches evidence? |
|---|---|---|---|---|---|
| 8 clean failures (Section 6, bullet 1) | FULL | CLEAN | VERTEX | `COORDINATION_STALL_VERTEX` | Yes (`continuation_used=False` in all 8, per 5.3) |
| 4 KI-1 instances | FULL (each has its own confirmed `joint_state_repeat`, e.g. period=1 -- Trigger fires normally) | INTEGRITY_ISSUE | n/a (never evaluated -- 5.2 step 1 gates before `blocking_type` is assessed) | `INTEGRITY_ISSUE` | Yes -- the integrity gate (5.2 step 1) overrides a fired Trigger; `trigger_type=FULL` is still recorded for provenance, it just never reaches a `COORDINATION_STALL_*` label |
| `cycle_ring_n4`, `scenario_empty_s11_n3` | NONE | CLEAN | NONE | `SUCCESS` | Yes |
| `corridor_bay_n2_3rdagent_perturbation` (new) | FULL (confirmed at t=5, all 3 agents static from t=4) | CLEAN | EDGE | `COORDINATION_STALL_EDGE` | Yes (`continuation_used=False`; see the 5.3 correction) |
| `bottleneck_n4_tightdoor_1agent_control` (new) | NONE | CLEAN | NONE | `SUCCESS` (trivial, no other agent to coordinate with) | Yes |

For all 9 Coordination-Stall rows, `joint_plan_exists = YES` and
`candidate_state_recoverable = NOT_CHECKED` (Section 6) -- consistent with
Section 5.2's requirement that the two never be conflated.

No instance *collected and analyzed so far* falls into `NAV_CANDIDATE`,
`UNKNOWN`, or `COORDINATION_STALL_MIXED` -- all three are defined but
currently uncalibrated. `trigger_type = PARTIAL` is a separate, stronger gap:
**no systematic, Monitor-based check for it exists at all** (Section 6), so
"zero instances" here must be read as "never produced or evaluated", not as
"searched for and not found".

### 8.2 TODO / explicit limitations

- [ ] `K_label` numeric value -- blocked on collecting >=1 real,
      non-fabricated successful episode with a transient block/WAIT window
      (Section 5.4), **and** on first producing at least one genuine
      `trigger_type=PARTIAL` case (Section 5.3), since that is the only
      situation the window is meant to be used for.
- [ ] `trigger_type=PARTIAL` / `K_label` continuation -- never exercised in
      this repository (Section 6); v1's continuation-free path only covers
      `trigger_type=FULL`.
- [ ] `candidate_state_recoverable` -- never computed for any instance
      (Section 5.2/6); would require re-running a solver from a stalled
      mid-episode state, not just from the original start.
- [ ] `COORDINATION_STALL_MIXED` calibration -- no episode with both
      persistent vertex and persistent edge blocking observed yet.
- [ ] `NAV_CANDIDATE` calibration -- no instance observed yet; needs a
      scenario where wall-blocking or single-agent failure actually occurs.
- [ ] `UNKNOWN` sub-case calibration -- `REPEAT_WITHOUT_BLOCKING_EVIDENCE`
      and `INSUFFICIENT_NAV_COORD_EVIDENCE` (Section 4.4) have no example
      yet; only the plain "no repeat" sub-case has evidence.
- [ ] `determinism_precondition = UNVERIFIABLE` path -- never exercised
      (every run so far had a known, matching checkpoint hash).
- [ ] `period > 1` calibration -- no multi-state cycle observed yet.
- [ ] CBS integration -- `CBS_recovery_result` cannot be populated until a
      real solver exists in this repository.
- [ ] Only 1 of 4 agents has been isolated for the NAV-control method
      (Section 6); the other 3 are untested.

### 8.3 Validation set vs. held-out set

Once enough instances exist to calibrate anything in this document (in
particular `K_label`), the calibration must use a **validation set**
disjoint from whatever **held-out set** is later used to report this
pipeline's final operational numbers. At present there are not enough
distinct, non-degenerate instances in this repository to split meaningfully
-- this split is a requirement for whenever calibration work starts, not
something already done.

### 8.4 Repeated runs of the same instance are not independent samples

Phase 1's reproducibility audit reran all 9 Phase-1 variants plus
`cycle_ring_n4` and the smoke scenario (11 reruns total) and found them
**byte-identical** to the originals in every step/summary/episode/event
field (ignoring only `run_id`/timestamp). This is expected under Section
4.1's determinism argument and is recorded here as the standing rule for any
future validation work: **N reruns of the same scenario with the same
checkpoint produce 1 data point, not N**. Any future `K_label`/threshold
calibration must count *distinct instances*, not reruns, when reporting how
much evidence backs a number.

---

## Summary of what this document decides vs. leaves open

**Decided**: the six-concept vocabulary (Section 2); that integrity issues
are a hard pre-filter, never blended with clean data (Section 3.6); that a
single collision or timeout is never sufficient by itself (Section 4.3);
that the determinism/permanence guarantee holds for **any** confirmed
**full** joint-state repeat period, not just period=1, and specifically does
**not** hold for a local/subset repeat (Section 3.3/4.2); that vertex-,
edge-, and mixed-type stalls are all in-scope and recorded separately, never
merged (Section 4.4); that `UNKNOWN` has four distinct sub-reasons, and is
not the same thing as the existing `stagnation_candidate=undetermined`
(Section 4.4); that Trigger-Full and Trigger-Partial are different objects
with different downstream handling, and that `K_label` continuation is
provably unnecessary for Trigger-Full but is the only available evidence for
Trigger-Partial (Section 5.1/5.3); that `joint_plan_exists` (original-start
solvability), `candidate_state_recoverable` (stalled-state recoverability),
and `CBS_recovery_result` (a five-way, not boolean, actual-solver outcome)
are three different fields that must never be merged (Section 5.2); and the
Monitor's online-primitive / self-maintained-history / offline-only-aggregate
three-way split (Section 7).

**Left open**: `K_label`'s numeric value; any genuine `trigger_type=PARTIAL`
case (never produced yet); `candidate_state_recoverable` for any instance
(never computed); `COORDINATION_STALL_MIXED` and `NAV_CANDIDATE` calibration;
the `UNKNOWN` sub-reasons beyond plain "no repeat"; the
`determinism_precondition=UNVERIFIABLE` path; `period>1` calibration; and any
CBS-backed evidence (Section 8.2) -- all pending additional, real
(non-synthetic) evidence that does not yet exist.
