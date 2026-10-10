# Controlled scenario variants (Alpha 2 failure-analysis expansion)

Nine additional instances (3 per category: Corridor / Intersection / Bottleneck)
built on top of the four frozen instances in `scenarios/controlled/`. None of
the original scenario JSON or `.npy` map files were modified; every variant
here **reuses an existing map** (no new `.npy` files were created) and lives
in this separate directory so the original `tests/test_failure_analysis.py`
glob (`scenarios/controlled/scenario_*.json`) is unaffected.

Format is the same as `scenarios/controlled/`: agent `start`/`goal` are
`[x, y]` = `[col, row]`; `scenario_loader` converts to internal `(row, col)`.
`map_file` is a project-root-relative path into `scenarios/controlled/`.

## Verification status legend

- `verified_via_joint_plan_replay`: a hand-written (or offline-search-found)
  joint plan was replayed step-by-step through the **unmodified**
  `MAPFStepSimulator` (`tests/test_controlled_variants.py`) and produced zero
  vertex/edge/wall conflicts, no position overlap at any timestep, and ended
  with every agent exactly on its goal.
- `CBS_verified`: **always "no"** in this round. This repository has no
  integrated CBS solver (`CBSAdapter` exists only as a referenced/placeholder
  hook in `scenario_loader.py` / `mapf_step_simulator.py`, not an
  implementation) -- see RESULTS.md for the explicit per-variant table. A
  passing joint-plan replay is evidence of solvability but is **not** a CBS
  proof and must not be reported as one.

## Variants

| slug | map (reused) | agents | coordination problem | relation to the original 4 |
|---|---|---|---|---|
| `corridor_bay_n2_3agent` | `map_corridor_bay_n2.npy` | 3 (2 east + 1 west) | single bay cell must serve 3 agents in sequence, not just 1 swap | denser than `corridor_bay_n2` (2 agents) |
| `corridor_bay_n2_offset` | `map_corridor_bay_n2.npy` | 2 | asymmetric path lengths; the west-bound agent starts 1 cell from the bay (less slack before the crossing) | same head-on idea, tighter margin |
| `corridor_bay_n2_4agent` | `map_corridor_bay_n2.npy` | 4 (2 east + 2 west) | two full east/west pairs must cross through one bay cell -- requires multiple bay visits in sequence, not one | densest corridor variant |
| `intersection_n2_4way` | `map_intersection_n2.npy` | 4 (N/S/E/W) | all four arms converge on the centre cell simultaneously (true 4-way stop), vs. the original's 2-way cross | denser than `intersection_n2` (2 agents) |
| `intersection_n2_lshape` | `map_intersection_n2.npy` | 2 | each path bends at the centre (west-arm->south-arm, north-arm->east-arm): paths share the centre cell as a turn, not just a straight-through crossing | different conflict geometry, same map |
| `intersection_n2_tight` | `map_intersection_n2.npy` | 2 | same perpendicular crossing as the original, but agents start only 1 cell from the centre instead of 2 (half the yield room) | tighter version of `intersection_n2` |
| `bottleneck_n4_convoy` | `map_bottleneck_n4.npy` | 4, all eastbound | no opposing flow -- all 4 agents queue through the single door in the same direction (ordering/queueing problem) | different dynamic than `bottleneck_n4` (2 vs 2 opposing) |
| `bottleneck_n4_imbalanced` | `map_bottleneck_n4.npy` | 4 (3 east + 1 west) | asymmetric flow; the lone west-bound agent must thread through a 3-agent majority flow | imbalanced version of `bottleneck_n4` |
| `bottleneck_n4_tightdoor` | `map_bottleneck_n4.npy` | 4 (2 east + 2 west) | all agents start immediately adjacent to the door (col3/col5) instead of at the far room edges -- near-zero slack before contention | tight-margin version of `bottleneck_n4` |

## Design-change log

All 9 variants matched their first-proposed coordinates from the approval
round without needing to change start/goal placements -- every instance
listed above is the originally proposed layout. (No "originally proposed
but unsolvable, changed to X" cases occurred in this round.)

## How solvability was established

1. A **prioritized time-expanded planner** (per-agent space-time search,
   avoiding higher-priority agents' reserved cells/edges, with optional
   per-agent initial-wait search) found a conflict-free plan directly for 6
   of the 9 variants: `corridor_bay_n2_offset`, `intersection_n2_lshape`,
   `intersection_n2_tight`, `bottleneck_n4_convoy`, `bottleneck_n4_imbalanced`,
   `bottleneck_n4_tightdoor`.
2. For `corridor_bay_n2_3agent`, the shuttle-via-bay idea from the original
   `corridor_bay_n2` plan was extended by hand (one agent parks in the bay
   while the other two pass, then returns) and confirmed to need no priority
   re-ordering.
3. `corridor_bay_n2_4agent` and `intersection_n2_4way` need the single bay /
   centre cell to be used by multiple agents in an order that no *fixed*
   priority ever produces (a higher-priority agent would have to
   deliberately delay itself for a lower-priority agent later in the same
   crossing). These two were solved with an **exhaustive joint-state BFS**
   (the free-cell count is small enough -- 12 cells / 4 agents for the
   corridor, 13 cells / 4 agents for the intersection -- that the full joint
   configuration graph fits in memory) over the exact action-resolution rule
   the real simulator uses, so the result is a minimum-length conflict-free
   plan by construction.
4. Every plan from steps 1-3 was then independently replayed through the
   real `MAPFStepSimulator` (not the search's own internal bookkeeping) via
   `tests/test_controlled_variants.py::TestControlledVariantsSolvable`,
   which is the actual gate `verified_via_joint_plan_replay` refers to.

None of `src/simulator/mapf_step_simulator.py`, `src/il/`, or
`src/common/episode_logger.py` were modified to make any of this work.
