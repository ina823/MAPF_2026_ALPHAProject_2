# Deadlock Monitor -- Trigger Timeline Evidence v1

Status: **evidence record, read-only**. This document does not change, and is
not itself, the Deadlock Definition (`docs/deadlock_definition_v1.md`) or any
Operational Label rule. It records, for three representative scenarios, the
actual recorded Full-repeat Trigger timeline produced by the Deadlock Monitor
implementation already committed on `feature/common-logging` through Phase
3B-4 (commit `13d76db`), so the timing of `first_seen` vs. `trigger_timestep`
and the distinction between the Trigger and the two observational candidate
signals (`persistent_conflict`, `no_progress`) can be inspected from real,
reproducible data instead of re-derived from memory each time.

**What this document is NOT**:
- It is **not** an Operational Label determination. `trigger_type` /
  `trigger_reason_code` only ever mean "a full joint-state repeat did or did
  not occur" (`docs/deadlock_definition_v1.md` Section 2, concept 4). None of
  the three episodes below have been passed through a Label rule (concept 5)
  -- no such rule is implemented anywhere in this repository yet.
- It is **not** a `K_label` or threshold calibration. No numeric window or
  cutoff is proposed, confirmed, or implied by anything in this document.
- It is **not** a CBS solvability or recovery result. `joint_plan_exists`,
  `candidate_state_recoverable`, and `CBS_recovery_result`
  (`docs/deadlock_definition_v1.md` Section 5.2) remain entirely unverified;
  no CBS solver is invoked anywhere in the code this document reports on.

---

## 1. Reproducibility record

| Item | Value |
|---|---|
| Branch | `feature/common-logging` |
| Commit used | `13d76db` ("Add Monitor sidecar CSV, decoupled from existing logs (Phase 3B-4)") |
| Checkpoint | `checkpoints/cnn_navhint.pt` |
| Checkpoint sha256 | `eb92f2f7487342d958fd8c6b23dc951cafe36f1109a2293c2e37907313cf3471` |
| Policy | `src.il.policy.NavHintPolicy` (`hint_mode=flowdist`), `eval()`-mode, no RNG |
| Simulator | `src.simulator.mapf_step_simulator.MAPFStepSimulator`, `max_steps=160` |

### 1.1 Commands that reproduce the FINAL snapshot and the full `trigger_reason` timeline exactly as shipped

These three commands use only the already-committed CLI (`scripts/run_il.py`
with `--log --monitor`) -- no extra script, no modification to any tracked
file:

```bash
python -m scripts.run_il --scenario scenarios/controlled/scenario_corridor_bay_n2.json --log --monitor
python -m scripts.run_il --scenario scenarios/controlled/scenario_intersection_n2.json --log --monitor
python -m scripts.run_il --scenario scenarios/controlled/scenario_bottleneck_n4.json --log --monitor
```

Each run writes `outputs/logs/<run_id>_steps.csv`, `<run_id>_summary.csv`, and
`<run_id>_monitor.csv` (the Phase 3B-4 sidecar). The sidecar's single row is
the FINAL per-episode snapshot; the steps CSV's `trigger_reason` column gives
the per-timestep value (sticky once it flips, Phase 3A's own semantics).
`<run_id>` is timestamp-based, so re-running produces new files rather than
overwriting the ones used for this document (which were generated, read, and
then deleted -- they are not committed; `outputs/logs/` is gitignored).

**Section 2**'s final-snapshot values were (re-)verified this way,
immediately before this document was finalized, in the course of this exact
review. **Section 3**'s per-step values were produced by the separate
instrumentation path in Section 1.2 below, not by these three CLI commands
alone -- see Section 1.2 for exactly what that path can and cannot
reproduce. Both paths were (re-)run during this same review, and their
overlap was cross-checked: the Section 1.2 run's own FINAL step for each
scenario reproduces the exact same `trigger_type`/`trigger_reason_code`/
`first_seen`/`trigger_timestep`/`blocking_type`/
`current_persistent_conflict_streak`/`current_persistent_conflict_key`/
`current_no_progress_streaks` values as this section's CLI-produced sidecar,
for all three scenarios, with zero discrepancies.

### 1.2 Reproducing the per-step `persistent_conflict`/`no_progress` streak values (Section 3)

The committed CLI only ever reports the **final** per-episode snapshot
(`<run_id>_monitor.csv`) and the per-timestep `trigger_reason` column (which
only carries the Trigger, not the two candidate-signal streaks). The
per-step `current_persistent_conflict_streak` / `current_no_progress_streaks`
values in Section 3's timelines were obtained by instrumenting the exact
same, already-committed call sequence `scripts/run_il.py::run_episode()` and
`_log_step()` already use (same `policy._dist` lookups, same
`spec.ACTION_DELTA` intended-position formula, same
`monitor.observe_step(...)` call shape), capturing the returned
`DeadlockMonitorResult` snapshot after every step instead of only reading it
once at the end. No new logic was introduced; this is a read-instrumentation
difference only. The snippet below reproduces it for any of the three
scenarios (just change `SCENARIO`):

```python
import sys
sys.path.insert(0, ".")
import scripts.run_il as run_il
from src.common import spec
from src.common.deadlock_monitor import DeadlockMonitor
from src.il.policy import NavHintPolicy
from src.simulator.mapf_step_simulator import MAPFStepSimulator
from pathlib import Path

SCENARIO = "scenarios/controlled/scenario_corridor_bay_n2.json"  # or intersection_n2 / bottleneck_n4
CKPT = Path("checkpoints/cnn_navhint.pt")

grid, starts, goals = run_il.load_scenario(Path(SCENARIO))
policy = NavHintPolicy(CKPT)
sim = MAPFStepSimulator(cbs_solver_root=None, max_steps=160)
obs = sim.reset(grid, starts, goals)
policy.reset(grid, goals)
monitor = DeadlockMonitor()
monitor.reset(sim._positions, goals)

for _ in range(160):
    actions = policy.act(obs, sim._positions)
    before = {aid: sim._positions[aid] for aid in sim._agent_ids}
    obs, done, info = sim.step(actions)
    gdb = {aid: int(policy._dist[aid][before[aid][0], before[aid][1]]) for aid in sim._agent_ids}
    gda = {aid: int(policy._dist[aid][sim._positions[aid][0], sim._positions[aid][1]]) for aid in sim._agent_ids}
    intended = {}
    for aid in sim._agent_ids:
        dr, dc = spec.ACTION_DELTA[actions[aid]]
        intended[aid] = (before[aid][0] + dr, before[aid][1] + dc)
    r = monitor.observe_step(info["t"], sim._positions, sim._last_conflict_types,
                              current_positions=before, intended_positions=intended,
                              goal_distance_before=gdb, goal_distance_after=gda)
    print(info["t"], r.trigger_type, r.current_persistent_conflict_streak,
          r.current_no_progress_streaks)
    if done:
        break
```

This was run per scenario, twice across this document's drafting and its
final review pass, for all three scenarios; both runs produced identical
output (deterministic policy + simulator, Phase 1's reproducibility audit).
The resulting values are reported verbatim in Section 3, not estimated or
interpolated for any timestep not actually observed. This instrumentation
reproduces per-step values; it is **not** the same code path as the
Section 1.1 CLI commands, which only expose the final snapshot and the
`trigger_reason` column -- see the limitation recorded in Section 6.

---

## 2. Comparison table (final snapshot, 160-step run, re-verified via the commands in 1.1)

| Field | Corridor (`scenario_corridor_bay_n2.json`) | Intersection (`scenario_intersection_n2.json`) | Bottleneck (`scenario_bottleneck_n4.json`) |
|---|---|---|---|
| Agents | 2 | 2 | 4 |
| `first_seen` | 4 | 1 | 1 |
| `trigger_timestep` | 5 | 2 | 2 |
| `period` | 1 | 1 | 1 |
| `trigger_type` | FULL | FULL | FULL |
| `trigger_reason_code` | `TRIGGER_FULL_REPEAT` | `TRIGGER_FULL_REPEAT` | `TRIGGER_FULL_REPEAT` |
| `blocking_type` | VERTEX | VERTEX | VERTEX |
| `blocking_agents` | 0;1 | 0;1 | 0;1;2;3 |
| `overlap_integrity_status` | CLEAN | CLEAN | CLEAN |
| `current_persistent_conflict_streak` (at step 160) | 156 | 159 | 159 |
| `current_persistent_conflict_key` | vertex, (2,5), {0,1} | vertex, (3,3), {0,1} | vertex, (2,7), {2,3} |
| `current_no_progress_streaks` (at step 160) | 0=156;1=156 | 0=159;1=159 | 0=159;1=159;2=159;3=159 |
| Episode outcome | `all_at_goal=False`, `timed_out=True`, `steps=160` | same | same |

All nine numeric/enum values above were read directly from the
`<run_id>_monitor.csv` sidecar produced by the Section 1.1 commands during
this review -- none were carried over unverified from an earlier session.

---

## 3. Per-step timeline (first 8-10 steps, via the Section 1.2 instrumentation)

### 3.1 Corridor (`first_seen=4`, `trigger_timestep=5`)

| t | `conflict_type` (agent0;agent1) | `trigger_reason` (steps CSV) | `current_persistent_conflict_streak` | `current_no_progress_streaks` |
|---|---|---|---|---|
| 1-4 | none;none | `TRIGGER_NONE` | 0 | (empty) |
| 5 | vertex;vertex | **`TRIGGER_FULL_REPEAT`** (first flip) | 1 | 0=1;1=1 |
| 6 | vertex;vertex | `TRIGGER_FULL_REPEAT` | 2 | 0=2;1=2 |
| 7 | vertex;vertex | `TRIGGER_FULL_REPEAT` | 3 | 0=3;1=3 |
| 8 | vertex;vertex | `TRIGGER_FULL_REPEAT` | 4 | 0=4;1=4 |
| 9 | vertex;vertex | `TRIGGER_FULL_REPEAT` | 5 | 0=5;1=5 |
| 10 | vertex;vertex | `TRIGGER_FULL_REPEAT` | 6 | 0=6;1=6 |

### 3.2 Intersection (`first_seen=1`, `trigger_timestep=2`)

| t | `conflict_type` | `trigger_reason` | `current_persistent_conflict_streak` | `current_no_progress_streaks` |
|---|---|---|---|---|
| 1 | none;none | `TRIGGER_NONE` | 0 | (empty) |
| 2 | vertex;vertex | **`TRIGGER_FULL_REPEAT`** (first flip) | 1 | 0=1;1=1 |
| 3-8 | vertex;vertex | `TRIGGER_FULL_REPEAT` | 2 -> 7 | 0=2;1=2 -> 0=7;1=7 |

### 3.3 Bottleneck (`first_seen=1`, `trigger_timestep=2`, 4 agents)

| t | `conflict_type` (a0;a1;a2;a3) | `trigger_reason` | `current_persistent_conflict_streak` | `current_no_progress_streaks` |
|---|---|---|---|---|
| 1 | none×4 | `TRIGGER_NONE` | 0 | (empty) |
| 2 | vertex×4 | **`TRIGGER_FULL_REPEAT`** (first flip) | 1 | 0=1;1=1;2=1;3=1 |
| 3-8 | vertex×4 | `TRIGGER_FULL_REPEAT` | 2 -> 7 | 0=2.. -> 0=7.. |

---

## 4. Interpretation

1. **`first_seen` vs. `trigger_timestep` are never the same instant.** In all
   three scenarios `trigger_timestep = first_seen + period` (`period=1`
   here). The CSV's `trigger_reason` column flips from `TRIGGER_NONE` to
   `TRIGGER_FULL_REPEAT` exactly at `trigger_timestep`, never at
   `first_seen` -- confirmed by direct inspection of the steps CSV, not
   inferred (`docs/deadlock_definition_v1.md` Section 3.3's indexing rule).
2. **The Trigger is a one-time, sticky determination; the two candidate
   signals keep changing after it.** `trigger_type`/`trigger_reason_code`/
   `blocking_type` are set once, at `trigger_timestep`, and never revisited.
   `current_persistent_conflict_streak` and `current_no_progress_streaks`
   keep incrementing every subsequent step (e.g. Corridor: 1 at t=5, 6 at
   t=10). These are deliberately different kinds of fields and must not be
   conflated: the Trigger says an event occurred; the two candidate signals
   describe how long a condition has persisted *so far*, as an ongoing,
   online measurement -- not a final, whole-episode aggregate (that belongs
   to `scripts/analyze_failure_logs.py`, computed offline, after the episode
   ends).
3. **All three cases are `conflict_type=vertex`**, with no `edge` conflict
   observed in any of them (the one known edge-type case in this repository,
   `scenario_corridor_bay_n2_3rdagent_perturbation.json`, is out of scope for
   this document's three Corridor/Intersection/Bottleneck originals).
4. **Bottleneck's `current_persistent_conflict_key` names only agents
   `{2,3}`** even though all 4 agents show an identical, growing
   `no_progress` streak. This is the already-documented Phase 3B-1
   limitation (only the single longest-streak conflict key is surfaced in
   the public result; other concurrently active keys are tracked internally
   but not exposed) -- it is not evidence that agents 0/1 are conflict-free,
   only that their key is not the one reported.
5. **KI-1 is out of scope here.** All three scenarios report
   `overlap_integrity_status=CLEAN`; the known KI-1 overlap variant
   (`scenario_bottleneck_n4_convoy.json`) is a deliberately separate case
   (per `docs/deadlock_definition_v1.md` Section 8.1's KI-1 separation rule)
   and is not included in this comparison.

---

## 5. Explicit scope disclaimer

- **Trigger is an observational signal, not a final Operational Deadlock
  Label.** Everything in this document is Phase 3A/3B-1/3B-2A territory
  (`docs/deadlock_definition_v1.md` Section 2, concepts 3-4). None of the
  three episodes has been passed through the Operational Label Rule
  (concept 5): no determinism-precondition check, no
  `joint_plan_exists`/`candidate_state_recoverable` solvability check, and no
  `INTEGRITY_ISSUE`/`SUCCESS`/`COORDINATION_STALL_*`/`UNKNOWN` classification
  has been computed or is implied by any value in this document.
- **`K_label` remains undefined and unvalidated.** No continuation window,
  no numeric threshold of any kind is proposed, used, or implied here.
- **CBS recovery/solvability verification is not part of this document.**
  `joint_plan_exists`, `candidate_state_recoverable`, and
  `CBS_recovery_result` are not evaluated by anything reported above; that
  work depends on 윤지's CBS-side verification, which has not yet produced
  results for these three instances in this repository.

---

## 6. Known limitations of this record

- The per-step timelines in Section 3 only cover the first 8-10 steps of
  each 160-step run. The remaining steps were not individually re-inspected
  for this document; the claim that the pattern continues unchanged is based
  on the already-proven determinism argument
  (`docs/deadlock_definition_v1.md` Section 4.2: a confirmed `period=1`
  full-repeat is a literal fixed point under the documented determinism
  precondition) plus the FINAL snapshot (Section 2), not on inspecting every
  intermediate timestep.
- Section 1.2's instrumentation snippet is **not** a committed script -- it
  is not part of `scripts/`, `tests/`, or any tracked file, per this
  review's "documentation only" scope. Anyone reproducing Section 3 must
  copy it into their own throwaway script.
- `scripts/run_il.py`'s shipped CLI alone can reproduce every value in
  Section 2 (the final snapshot) and the full `trigger_reason` timeline, but
  **cannot** by itself reproduce the per-step `persistent_conflict`/
  `no_progress` streak columns in Section 3 -- those are not persisted
  anywhere except the final sidecar row. This is a real, current gap in what
  the shipped tooling exposes, not an error in this document.
- `bottleneck_n4`'s single-key limitation (point 4 above) means the
  comparison table under-reports how many distinct conflict pairs may be
  simultaneously active in a >=3-agent stall; this document does not attempt
  to work around that known limitation.
