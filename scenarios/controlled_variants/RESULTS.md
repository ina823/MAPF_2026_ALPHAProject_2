# Controlled variants -- verification + IL-only execution results

Tracks the 9 new instances in `scenarios/controlled_variants/` **separately**
from the original 4 instances in `scenarios/controlled/` (see
`outputs/logs/` and `outputs/failure_analysis/` from the earlier session for
those -- run_ids there are prefixed `*_corridor_bay_n2_IL`,
`*_intersection_n2_IL`, `*_bottleneck_n4_IL`, `*_cycle_ring_n4_IL`, none of
which collide with the slugs below). No file under `scenarios/controlled/`,
`src/simulator/`, `src/il/`, or `src/common/episode_logger.py` was touched to
produce any of this.

## 1. Solvability verification (`tests/test_controlled_variants.py`)

All 9 pass `TestControlledVariantsWellFormed` (free-cell starts/goals, no
duplicate start, no duplicate goal, map reused from `scenarios/controlled/`,
no scenario_id collision with the original 4) and
`TestControlledVariantsSolvable` (conflict-free joint-plan replay in the real,
unmodified `MAPFStepSimulator`).

| slug | agents | plan horizon | plan method | verified_via_joint_plan_replay | CBS_verified |
|---|---|---|---|---|---|
| corridor_bay_n2_3agent | 3 | 17 | hand-written (bay shuttle, extended from original) | yes | no |
| corridor_bay_n2_offset | 2 | 11 | prioritized time-expanded planner | yes | no |
| corridor_bay_n2_4agent | 4 | 17 | exhaustive joint-state BFS | yes | no |
| intersection_n2_4way | 4 | 11 | exhaustive joint-state BFS | yes | no |
| intersection_n2_lshape | 2 | 7 | prioritized time-expanded planner | yes | no |
| intersection_n2_tight | 2 | 3 | prioritized time-expanded planner | yes | no |
| bottleneck_n4_convoy | 4 | 12 | prioritized time-expanded planner | yes | no |
| bottleneck_n4_imbalanced | 4 | 14 | prioritized time-expanded planner | yes | no |
| bottleneck_n4_tightdoor | 4 | 10 | prioritized time-expanded planner | yes | no |

`CBS_verified` is "no" for every row on purpose: this repository has no
integrated CBS solver to run (`CBSAdapter` is a referenced/placeholder hook
only, in `scenario_loader.py` / `mapf_step_simulator.py`). A passing replay
is solvability evidence, not a CBS proof -- do not read "yes" into the
`verified_via_joint_plan_replay` column as CBS confirmation.

No variant needed its originally-proposed coordinates changed -- all 9 match
the layout approved before implementation (see "Design-change log" in
`README.md`).

## 2. IL-only execution (`scripts/run_il.py`, unmodified checkpoint/policy/simulator/logger, `--max-steps 160`)

Every one of the 9 variants **fails to reach all goals within 160 steps**
(same `max_steps` as the original failure-analysis runs). This matches the
original 4 scenarios' pattern and is consistent with the project's
established finding that the IL-navhint baseline has no explicit
coordination mechanism.

| slug | run_id | success | timed_out | agents_at_goal | progress_ratio | vertex_conflicts | edge_conflicts | joint_state_repeat_count | first_repeat_t | stagnation_candidate |
|---|---|---|---|---|---|---|---|---|---|---|
| corridor_bay_n2_3agent | 20261010_212304_corridor_bay_n2_3agent_IL | False | True | 1/3 | 0.6071 | 156 | 0 | 151 | 10 | yes |
| corridor_bay_n2_offset | 20261010_212310_corridor_bay_n2_offset_IL | False | True | 0/2 | 0.25 | 158 | 0 | 158 | 3 | yes |
| corridor_bay_n2_4agent | 20261010_212307_corridor_bay_n2_4agent_IL | False | True | 0/4 | 0.3889 | 157 | 0 | 156 | 5 | yes |
| intersection_n2_4way | 20261010_212314_intersection_n2_4way_IL | False | True | 0/4 | 0.3333 | 158 | 0 | 158 | 3 | yes |
| intersection_n2_lshape | 20261010_212317_intersection_n2_lshape_IL | False | True | 0/2 | 0.3333 | 158 | 0 | 158 | 3 | yes |
| intersection_n2_tight | 20261010_212320_intersection_n2_tight_IL | False | True | 0/2 | 0.0 | 160 | 0 | 160 | 1 | yes |
| bottleneck_n4_convoy | 20261010_212255_bottleneck_n4_convoy_IL | False | True | 0/4 | 0.0556 | 160 | 0 | 159 | 2 | yes |
| bottleneck_n4_imbalanced | 20261010_212258_bottleneck_n4_imbalanced_IL | False | True | 2/4 | 0.5278 | 160 | 0 | 146 | 15 | yes |
| bottleneck_n4_tightdoor | 20261010_212301_bottleneck_n4_tightdoor_IL | False | True | 0/4 | 0.1 | 319 | 0 | 159 | 2 | yes |

All 9 raise the `joint_state_repeat` signal (`stagnation_candidate: yes`),
meaning the deterministic IL policy revisits a joint position it was already
in -- for a stateless deterministic policy this means the rollout cycles
forever from that point on. **This is a stagnation *candidate*, not a
deadlock label**: `stagnation_candidate` only means "the failed episode has
>= 1 raw signal"; it is produced by `scripts/analyze_failure_logs.py`
unchanged and is explicitly documented there as not a deadlock determination
(see that script's module docstring). `edge_conflicts` is 0 in every run;
all conflicts observed are vertex conflicts.

Raw CSVs (unmodified logger/analyzer output):
`outputs/logs/<run_id>_steps.csv`, `outputs/logs/<run_id>_summary.csv`,
`outputs/failure_analysis/<run_id>_events.csv`,
`outputs/failure_analysis/<run_id>_episode.csv`.

## 3. Integrity flag -- position overlap (NOT treated as deadlock evidence)

**Correction (verification round 2, see Section 6):** this section originally
(first-pass report) listed only 3 of the 4 runs that actually have
`overlap_timesteps > 0` -- `corridor_bay_n2_3agent` (`overlap_timesteps = 1`)
was missed. The table below is the corrected, audited set; Section 6 records
how the omission was caught (full re-dump of every episode CSV field against
this file).

4 of the 9 runs report `overlap_timesteps > 0` (two agents logged at the same
cell after a step); `invalid_distance_rows` is 0 for all 9:

| slug | overlap_timesteps | invalid_distance_rows |
|---|---|---|
| corridor_bay_n2_3agent | 1 | 0 |
| corridor_bay_n2_4agent | 157 | 0 |
| bottleneck_n4_convoy | 160 | 0 |
| bottleneck_n4_imbalanced | 160 | 0 |

This is the simulator's pre-existing, **already documented and intentionally
preserved** Alpha 1 quirk **KI-1** (`tests/test_simulator_characterization.py
::TestKnownQuirkKI1`): vertex-collision resolution is a single pass, not
iterated, so an agent bounced back into its previous cell is not re-checked
against a different agent that is simultaneously moving into that same
previous cell -- two agents can end up sharing a cell. It only shows up here
because these are the first *controlled* scenarios dense enough (3-4 agents in
a narrow corridor/bottleneck) to trigger it; all 2-agent variants show
`overlap_timesteps = 0`. It is a known simulator-resolution-order artifact,
**not** evidence of a coordination deadlock, and per this task's scope it was
left as-is (`src/simulator/mapf_step_simulator.py` was not modified). See
Section 6 for the exact step-level trace confirming the KI-1 mechanism in
each of the 4 cases.

## 4. Regression check (round 1)

- `python -m unittest discover -s tests`: **76/76 tests pass** (70 existing +
  6 new in `tests/test_controlled_variants.py`), including the untouched
  `tests/test_failure_analysis.py::TestControlledScenarios` for the original
  4 scenarios.
- `python -m scripts.smoke_test_il`: **9/9 checks PASS**, identical baseline
  trajectory (steps=9, all_at_goal=True, deterministic), confirming the
  IL-navhint baseline is unchanged.

## 5. What was explicitly not built in this round

No Deadlock Monitor, no Hybrid Runner, no IL retraining, no CBS integration.
`stagnation_candidate` / `candidate_signals` are read-only outputs of the
existing `scripts/analyze_failure_logs.py`; nothing here infers or labels a
deadlock.

---

# Verification round 2: data integrity + reproducibility audit

No new variants, no code changes (`src/simulator/`, `src/il/`,
`src/common/episode_logger.py`, `scenarios/controlled/` all untouched in this
round). This section only reads existing files and reruns `scripts/run_il.py`
/ `scripts/analyze_failure_logs.py` exactly as before.

## 6. Data integrity audit (Sections 1-3 vs. the actual files on disk)

For all 9 variants, cross-checked: `scenario_id`, `map_file` (resolves into
`scenarios/controlled/`, no new `.npy`), agent count, and that all four
expected files (`*_steps.csv`, `*_summary.csv`, `*_episode.csv`,
`*_events.csv`) exist -- all 9 pass. For every episode CSV, confirmed
`summary_source=summary_csv` and that `success`/`timed_out`/`steps`/
`vertex_conflicts`/`edge_conflicts` in `*_summary.csv` match the same fields
in `*_episode.csv` exactly (the episode analyzer is built on top of the
summary CSV, not a separate re-derivation) -- all 9 linked correctly.

Full re-dump of every episode-level field
(`success, timed_out, steps, agents_at_goal, progress_ratio, vertex_conflicts,
edge_conflicts, vertex_conflicts_reconstructed, edge_conflicts_reconstructed,
conflict_reconstruction_matches_summary, no_progress_steps,
max_no_progress_span, goal_distance_stagnation_max_span,
joint_state_repeat_count, joint_state_first_repeat_timestep,
joint_state_period, overlap_timesteps, invalid_distance_rows,
stagnation_candidate, candidate_signals`) against Section 2/3 of this file
confirmed every number in Section 2 is correct, and surfaced the one error:
**Section 3 (first pass) omitted `corridor_bay_n2_3agent`'s
`overlap_timesteps = 1`.** It has been corrected in place above. No other
discrepancy was found. `invalid_distance_rows = 0` and
`conflict_reconstruction_matches_summary = True` for all 9 (i.e. the
analyzer's own row-by-row conflict reconstruction agrees with the
simulator-reported summary totals in every run -- no silent mismatch).

## 7. KI-1 step-level trace (all 4 integrity-flagged runs)

For each flagged run, the first `position_overlap` event was located in
`*_events.csv`, then cross-referenced against the corresponding rows of
`*_steps.csv` at that timestep (`current`, `action`, `intended`, `next`,
`conflict_type`) and the colliding agents' positions at `t-1`. All 4 show the
**exact same mechanism**: one agent's move into a cell is uncontested at
resolution time (`conflict_type=none`) and succeeds, while a *different*
agent loses a vertex conflict and bounces back into *its own* previous cell
(`conflict_type=vertex`) -- which happens to be the exact cell the first
agent just moved into. The single-pass resolution in
`mapf_step_simulator.py::step()` does not re-check bounce-back targets
against already-resolved moves, so both agents end up sharing that cell. This
is precisely KI-1 as documented in
`tests/test_simulator_characterization.py::TestKnownQuirkKI1`, not a new bug
and not something introduced by these scenarios -- it is simply the first time
a *controlled* scenario has been dense enough to trigger it.

| slug | first overlap t | agents | cell | agent that succeeded in (conflict_type=none) | agent that bounced back into its own prior cell (conflict_type=vertex) |
|---|---|---|---|---|---|
| `corridor_bay_n2_3agent` | 5 | 1, 2 | (2,6) | agent1: (2,5)->intended(2,6)->next(2,6) | agent2: current(2,6), intended(2,5) contested by agent0, bounces back to (2,6) |
| `corridor_bay_n2_4agent` | 4 | 0, 1 | (2,4) | agent0: (2,3)->intended(2,4)->next(2,4) | agent1: current(2,4), intended(2,5) contested by agent2, bounces back to (2,4) |
| `bottleneck_n4_convoy` | 1 | 0, 1 | (1,1) | agent0: (0,1)->intended(1,1)->next(1,1) | agent1: current(1,1), intended(2,1) contested by agent2, bounces back to (1,1) |
| `bottleneck_n4_imbalanced` | 1 | 0, 1 | (1,1) | agent0: (0,1)->intended(1,1)->next(1,1) | agent1: current(1,1), intended(2,1) contested by agent2, bounces back to (1,1) |

`bottleneck_n4_convoy` and `bottleneck_n4_imbalanced` share the exact same
first-overlap event (same cell, same timestep, same agents 0/1/2) because
both scenarios give agents 0-2 identical starts -- only agent 3 differs
between the two, which is why the two scenarios' total `position_overlap`
event counts differ later in the episode (320 vs. a smaller but still large
count) even though they start identically.

Once triggered, the overlap is **persistent**: the deterministic IL policy
repeats the same joint action from an already-repeated joint state, so the
overlapping agents never separate again for the rest of the 160-step episode
(this is also exactly what `joint_state_repeat` is reporting for these runs
-- the two signals are describing the same underlying stall, not two
independent problems).

### Integrity classification for use in any future deadlock labelling

The 4 flagged runs above are marked **`integrity_issue: simulator_KI-1`** and
must **not** be pooled with the 5 clean runs when computing any future
deadlock/stagnation statistic, because their `vertex_conflicts`,
`blocked_count`, and position-based metrics are computed over a step history
that includes cells with 2 agents physically overlapping -- a state the
simulator's own collision model is not supposed to allow. The 5 clean runs'
metrics have no such caveat.

## 8. Reproducibility (independent rerun, same checkpoint/scenario/`max_steps=160`)

Every one of the 9 variants, plus the existing `cycle_ring_n4` (success
control group, from `scenarios/controlled/`) and the smoke-test scenario
`scenario_empty_s11_n3` (from `scenarios/smoke/`), was rerun from scratch
with `python -m scripts.run_il --scenario <path> --max-steps 160 --log`
(identical checkpoint, identical `max_steps`, no code change), producing new
`outputs/logs/<new_run_id>_{steps,summary}.csv`. **None of the original CSVs
were deleted or overwritten** -- `make_run_id()` embeds a fresh timestamp, so
every rerun lands in a new file automatically.

Each new `*_steps.csv` was also re-analyzed with the unmodified
`scripts/analyze_failure_logs.py` to produce new `*_episode.csv` /
`*_events.csv`. Old vs. new were then diffed field-by-field, ignoring only
`run_id` (which embeds the rerun's timestamp by construction) and
`analysis_params` (echoes CLI flags, identical here by construction, not a
measurement):

| scenario | steps.csv identical | summary.csv identical | episode.csv identical | events.csv identical |
|---|---|---|---|---|
| corridor_bay_n2_3agent | yes | yes | yes | yes |
| corridor_bay_n2_offset | yes | yes | yes | yes |
| corridor_bay_n2_4agent | yes | yes | yes | yes |
| intersection_n2_4way | yes | yes | yes | yes |
| intersection_n2_lshape | yes | yes | yes | yes |
| intersection_n2_tight | yes | yes | yes | yes |
| bottleneck_n4_convoy | yes | yes | yes | yes |
| bottleneck_n4_imbalanced | yes | yes | yes | yes |
| bottleneck_n4_tightdoor | yes | yes | yes | yes |
| `cycle_ring_n4` (control, `scenarios/controlled/`) | yes | yes | yes | yes |
| `scenario_empty_s11_n3` (smoke, `scenarios/smoke/`) | yes | yes | yes | yes |

**11/11 identical**, including per-timestep per-agent position
(`current_row/col`, `next_row/col`), the full action sequence (`action_id`),
every `conflict_type`/`blocked`/`is_wait` flag, `success`/`timed_out`, every
episode-level metric (`progress_ratio`, `vertex_conflicts`, `edge_conflicts`,
`joint_state_repeat_count`, `overlap_timesteps`, `invalid_distance_rows`,
`stagnation_candidate`, ...), and every logged event
(`position_overlap`/`blocked`/`wait_run`/... rows). This confirms the
IL-navhint baseline is fully deterministic end-to-end for these 11 scenarios,
including the 4 KI-1-affected runs (the overlap pattern itself reproduces
identically, not intermittently).

`scripts/smoke_test_il.py` was also rerun standalone and reproduced its own
internal two-episode determinism check (check 8/9) with the identical
trajectory as the first-pass report.

**Note on sample counting:** these 11 reruns are repeated executions of the
*same* 11 inputs (same checkpoint, same scenario, same `max_steps`) used only
to confirm determinism/reproducibility. They are **not** 11 additional,
independent experimental trials and must not be added to any future count of
"number of episodes observed" for these scenarios -- the underlying
IL-navhint policy and `MAPFStepSimulator` are deterministic given a fixed
scenario, so a rerun with identical inputs can only reproduce or fail to
reproduce the original result, never provide new statistical signal.

## 9. Final classification

Classification uses only (a) whether `tests/test_controlled_variants.py`'s
joint-plan replay passed, and (b) whether the IL execution's own logs are
integrity-clean (`overlap_timesteps == 0` and `invalid_distance_rows == 0`).
**IL timeout and `joint_state_repeat` are explicitly excluded as
classification inputs** -- they describe the IL policy's behaviour, not data
quality, and per this task's constraint must not be used to assign a
deadlock label.

| category | definition | variants |
|---|---|---|
| **A** | joint-plan verified + IL log integrity clean | `corridor_bay_n2_offset`, `intersection_n2_4way`, `intersection_n2_lshape`, `intersection_n2_tight`, `bottleneck_n4_tightdoor` |
| **B** | joint-plan verified + IL log integrity issue (KI-1 overlap) found | `corridor_bay_n2_3agent`, `corridor_bay_n2_4agent`, `bottleneck_n4_convoy`, `bottleneck_n4_imbalanced` |
| **C** | joint-plan verification failed or other problem | *(none)* |

All 9 variants are category A or B; none are category C. `CBS_verified`
remains **"no" for all 9** (and for the original 4) pending an actual CBS
solver run, which this round -- like the first -- did not perform (none
exists in this repository; see Section 1).

## 10. Regression check (round 2) + repository state

- `python -m unittest discover -s tests`: **76/76 tests pass** (unchanged
  from round 1 -- no test or source file was modified in this round).
- `python -m scripts.smoke_test_il`: **9/9 checks PASS**, identical baseline
  trajectory.
- `git status` / `git diff --stat`: no tracked file was modified; only new
  files (this file's edits plus the new rerun CSVs, which fall under the
  existing `outputs/*/.gitignore` rule and stay untracked). No commit/push
  was made.
