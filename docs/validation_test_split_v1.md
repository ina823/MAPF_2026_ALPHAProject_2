# Validation / Test Split Rules v1

Status: **rule document, v1, draft for review**. This document fixes the
*rules* for separating Development, Validation, and Test data. It is
explicitly **not** a held-out instance list -- no Validation or Test
instance exists in this repository yet (Section 3). Nothing in this
document changes `src/`, `scripts/`, `tests/`, any `scenarios/` file, any
checkpoint, or any other `docs/*.md` file.

This document extends, and must be read together with,
`docs/deadlock_definition_v1.md` Section 8.3 ("Validation set vs. held-out
set") and Section 8.4 ("Repeated runs of the same instance are not
independent samples") -- it does not restate their reasoning, it makes it
operational (concrete tiers, concrete allowed/forbidden actions, a concrete
classification of every scenario file that exists today).

---

## 1. Scope and non-goals

**What this document does:**
- Defines three data roles (Development / Validation / Test) and what is
  allowed and forbidden in each.
- Classifies every scenario file that exists in this repository today into
  one of those roles, with an exact, independently-recounted file count.
- States the rule for when a repeated run (including a seed-only rerun) does
  and does not count as a new, independent instance.
- States the leakage-prevention criteria (scenario family, map layout,
  start/goal configuration, obstacle/bottleneck structure) that must be
  applied once real Validation/Test instances exist.
- States the freeze principle for threshold / `K_label` / `H_exec` /
  Selective-Hybrid parameters.
- States the pre-registration vs. post-hoc-retry distinction for the
  eventual Test protocol.
- Separates out, as an explicit TODO list, everything that cannot be done
  without 윤지's CBS verification or without benchmark instances that do not
  exist in this repository.

**What this document does NOT do:**
- It does not invent a target instance count (e.g. "~25 per condition") as a
  confirmed number. Any such figure appearing in the Notion master plan is
  cited there as a *target to document before starting*, not a count this
  repository has reached -- see Section 7.
- It does not fix `K_label`, any numeric threshold, `H_exec`, or any
  Selective-Hybrid parameter. These remain exactly as undetermined as
  `docs/deadlock_definition_v1.md` Section 5.4/8.2 already recorded them.
- It does not create, rename, or move any scenario file.
- It does not decide Trigger-Partial, CBS Recovery, or Hybrid Runner
  behaviour -- out of scope, as in every prior phase of this work.

---

## 2. Three data roles

| Role | Purpose | Allowed | Forbidden |
|---|---|---|---|
| **Development** | Mechanism demonstration, bug reproduction, Monitor/signal exploration, unit-test fixtures. | Repeated inspection; used to *derive* detector logic and write tests; iterated on freely. | Reporting as a final performance number; being relabeled "held-out Test" at any later point. |
| **Validation** | The *only* place threshold / `K_label` / `H_exec` / Safety-only parameters / Selective-Hybrid parameters may be chosen or adjusted. | Repeated runs, parameter sweeps, inspecting results and revising a parameter in response. | Being reused, under the same instances, to report the pipeline's final operational numbers. |
| **Test (held-out)** | Final, one-time evaluation after every relevant freeze. | Exactly one evaluation pass per frozen configuration (pre-registered repeated measurements for variance are a narrow, declared exception -- Section 6). | Any adjustment of a rule/threshold/parameter in response to a Test result; any re-run used to "fix" an unfavourable number; reuse of an instance already used in Development or Validation. |

**Why a Development instance can never become Validation or Test for *this*
detector**: every trigger/blocking-type rule in
`docs/deadlock_definition_v1.md` was derived by observing the exact
Development scenarios classified in Section 3 below (see that document's own
Section 6 evidence list). Relabeling one of them "held-out" later would be
circular -- the detector was shaped to fit it.

---

## 3. Current scenario inventory (exact recount, this review)

Recounted directly from the filesystem during this review (not carried over
from an earlier, miscounted report):

```
scenarios/controlled/            4 files
scenarios/controlled_variants/   9 files
scenarios/experimental_v1/       2 files
scenarios/smoke/                 1 file
--------------------------------------
Total scenario files in repo:   16
Research scenarios (excl. smoke): 15   (controlled + controlled_variants + experimental_v1)
```

An earlier verbal report in this review session stated "14" for the
research-scenario count and, separately, "8" for `controlled_variants`; both
were arithmetic errors, corrected here against a direct filesystem listing.
`docs/deadlock_definition_v1.md` independently corroborates the 9-variant
count ("5 Category-A variants" + "4 Category-B [KI-1] instances" = 9; "all 9
Phase-1 variants" in its Section 8.4).

### 3.1 Classification -- every file is Development; none is Validation or Test

| File(s) | Role | Reason |
|---|---|---|
| `controlled/scenario_corridor_bay_n2.json`, `scenario_intersection_n2.json`, `scenario_bottleneck_n4.json` | Development -- mechanism demonstration | Directly used to derive the Full-repeat Trigger logic itself (`deadlock_definition_v1.md` Section 6's own evidence list); repeatedly re-observed across Phase 3A-3B-4 and `docs/monitor_trigger_timeline_v1.md`. |
| `controlled/scenario_cycle_ring_n4.json` | Development -- negative control | Used throughout as the "no false trigger" regression check, never as a performance instance. |
| `smoke/scenario_empty_s11_n3.json` | Development -- pipeline regression anchor, not a research scenario | Sole purpose is `scripts/smoke_test_il.py`'s strict-baseline check ("did the baseline pipeline silently change"); excluded from the 15-scenario research count above precisely because it is not a MAPF-difficulty research instance. |
| `controlled_variants/*` (9 files: `corridor_bay_n2_3agent`, `corridor_bay_n2_offset`, `corridor_bay_n2_4agent`, `intersection_n2_4way`, `intersection_n2_lshape`, `intersection_n2_tight`, `bottleneck_n4_convoy`, `bottleneck_n4_imbalanced`, `bottleneck_n4_tightdoor`) | Development -- same base families, parameter sweep | Same base map/topology as the 3 originals; only agent count / door width / offset changed. Not independent families (Section 4.1). `CBS_verified="no"` for every one in `tests/test_controlled_variants.py` -- verified only via hand-written joint-plan replay, never by an actual CBS solver. |
| KI-1 subset of the above (`corridor_bay_n2_3agent`, `corridor_bay_n2_4agent`, `bottleneck_n4_convoy`, `bottleneck_n4_imbalanced`) | Development -- simulator-quirk demonstration, excluded from label statistics entirely (already true under `deadlock_definition_v1.md` Section 3.6/4.3's hard `INTEGRITY_ISSUE` gate, independent of this document) | KI-1 is a documented single-pass vertex-resolution artifact of `MAPFStepSimulator`, not a deadlock-candidate signal. |
| `experimental_v1/scenario_corridor_bay_n2_3rdagent_perturbation.json`, `scenario_bottleneck_n4_tightdoor_1agent_control.json` | Development -- ad hoc scratch experiments | `deadlock_definition_v1.md` Section 6 itself calls these "ad hoc, one-off scratch check[s]," outside the committed `test_controlled_variants.py` solvability suite. |

### 3.2 Validation / Test instance counts

```
Validation instances available today: 0
Test (held-out) instances available today: 0
```

No MovingAI or other benchmark `.map`/`.scen` file, and no benchmark loader,
exists on this branch (`feature/common-logging`) as of this review. 윤지's
`src/movingai_loader.py` (per the Notion checklist, committed as `32fe8f6`)
exists only on her separate branch (`feature/cbs-replanning`) and has not
been merged here. **No placeholder, example, or target-count instance list
is written below** -- inventing one would misrepresent data this project
does not yet have.

---

## 4. Leakage-prevention criteria for future Validation/Test selection

### 4.1 Scenario family -- definition

A **family** is the set of scenario files that share:
1. the same base map/grid layout (free/wall cell pattern), **and**
2. the same qualitative obstacle/bottleneck structure (corridor, 4-way
   intersection, single-door bottleneck, open cycle, etc.), **and**
3. start/goal configurations that are a direct parametric variation of one
   another (different agent count, a shifted offset, a widened/narrowed
   door) rather than a structurally different map.

Under this definition, the current repository has exactly **3 families**:
`corridor_bay`, `intersection`, `bottleneck` -- each family's "original"
(in `controlled/`) and all of that family's entries in `controlled_variants/`
and `experimental_v1/` belong to the same family. `cycle_ring_n4` and the
smoke scenario are each their own singleton family (no variant exists for
either in this repository).

### 4.2 Held-out selection rule (for when Validation/Test instances exist)

- **Instance-level separation is not sufficient.** A future Test instance
  must not share a family (Section 4.1) with any instance already used in
  Development or Validation -- not merely avoid being byte-identical to one.
- **A seed-only rerun of the same map/start/goal is never a new instance.**
  This extends `docs/deadlock_definition_v1.md` Section 8.4's existing rule
  ("N reruns of the same scenario with the same checkpoint produce 1 data
  point, not N") explicitly to benchmark instances with a seed parameter:
  changing only the RNG seed on an otherwise-identical map/start/goal
  produces the same deterministic trajectory under this project's
  deterministic IL policy (Section 4.1 of that document), so it is not an
  independent sample for performance reporting. Same-instance reruns are
  legitimate only for wall-clock runtime variance or system-stability
  checks, never for growing a success/deadlock sample count.
- **Map layout, obstacle structure, and start/goal configuration are all
  checked, not just one.** Two instances with a different start/goal pair
  on the *same* map and the *same* obstacle structure are still the same
  family under Section 4.1 and must not be split across Development and
  Test.
- **This section states the criteria only.** It does not apply them to
  produce an actual Validation or Test list -- there is nothing to apply
  them to yet (Section 3.2).

---

## 5. Threshold / `K_label` / `H_exec` / Selective-parameter freeze principle

- Every numeric threshold, `K_label` (the fixed label-continuation window,
  `docs/deadlock_definition_v1.md` Section 5.4), `H_exec` (the Notion master
  plan's CBS-execution-horizon parameter), and every future Selective-Hybrid
  parameter **may be selected or adjusted only on Validation instances**.
- Once a parameter is frozen for a Test run, it is **not** revised in
  response to that Test run's results, under any circumstances.
- **None of these values are determined by this document.** This document
  does not propose a `K_label` number, a threshold number, an `H_exec`
  number, or any Selective-Hybrid parameter value -- all remain exactly as
  undetermined as `docs/deadlock_definition_v1.md` Section 5.4/8.2 already
  recorded, because no Validation data exists yet to calibrate them against
  (Section 3.2).

---

## 6. Test protocol: pre-registration vs. post-hoc retry/tuning

A future Test run must follow a **freeze-then-run** protocol:

1. **Freeze**, before any Test instance is touched: the detector rule, every
   threshold/`K_label`/`H_exec`/Safety-only/Selective parameter (Section 5),
   the exact Test instance list (Section 4.2, once it exists), and the exact
   evaluation code/commit SHA to be used.
2. **Pre-registered repeated measurements** (if any) must be declared
   *before* the Test run as part of the frozen protocol -- for example, "run
   each Test instance exactly `N` times to report median/tail wall-clock
   CBS latency" is a legitimate, declared repeated measurement if `N` and
   the purpose (runtime-variance reporting, not sample-count inflation) are
   fixed in advance. This is explicitly distinct from:
3. **Post-hoc, undeclared re-runs or tuning**, which are forbidden after a
   Test run: re-running a failed instance "to see if it does better," tuning
   a threshold because the frozen one performed poorly on Test, or adding/
   removing Test instances after seeing results. Any of these invalidates
   the Test result and requires a documented, dated protocol amendment
   before a new (and now itself frozen) Test pass -- it does not let the
   original numbers stand in alongside the new ones.
4. If a schedule slips, the freeze deadline slips with it -- Test execution
   is never started ahead of its freeze, per the Notion master plan's own
   rule ("held-out test는 모든 관련 freeze 이후에만 실행한다. 일정이 밀리면
   test 실행도 같이 늦춘다").

---

## 7. Open items requiring 윤지's CBS verification or benchmark instances (separate, incomplete)

None of the following are implemented or decided by this document:

- [ ] Acquiring any actual Validation or Test benchmark instance (MovingAI
      or otherwise) on this branch -- the loader exists only on 윤지's
      `feature/cbs-replanning` branch, not here.
- [ ] Any concrete "instances per condition" target count. The Notion
      master plan's "~25 per condition" figure is that document's own stated
      *target to document before starting*, not a count this repository has
      reached -- repeating it here as a decided number would be exactly the
      "근거 없는 목표 개수" this document was asked not to write.
- [ ] CBS/joint-plan solvability confirmation for the 9 `controlled_variants`
      instances and the 2 `experimental_v1` instances -- all currently
      `CBS_verified="no"` (hand-written-plan replay only, per
      `tests/test_controlled_variants.py`'s own module docstring).
- [ ] Excluding any instance found to be CBS-unsolvable from
      detector-performance statistics (per the Notion master plan: "unsolvable
      instance는 detector 성능평가용 deadlock positive로 섞지 않고 별도
      표시") -- depends on the above solvability confirmation existing at all.
- [ ] `H_exec` candidate comparison and freeze (10/18 per the master plan) --
      depends on CBS recovery existing, which it does not yet
      (`CBS_recovery_result` is `NOT_RUN` for every instance in this
      repository, per `docs/deadlock_definition_v1.md`).
- [ ] Any Selective-Hybrid parameter -- Selective Hybrid itself is not
      implemented.

---

## 8. Consistency check against existing documents (performed for this review)

- **`docs/deadlock_definition_v1.md` Section 8.3**: this document's
  three-tier role split (Section 2) and zero-Validation/zero-Test count
  (Section 3.2) are a direct operationalization of that section's statement
  that "there are not enough distinct, non-degenerate instances in this
  repository to split meaningfully" -- no contradiction; this document does
  not claim the split has been performed, only that the *rule* for
  performing it is now written down.
- **`docs/deadlock_definition_v1.md` Section 8.4**: this document's Section
  4.2 seed-rerun rule explicitly extends that section's "N reruns = 1 data
  point" principle to a seed parameter; no contradiction, no redefinition.
- **Notion master plan ("Validation / Test 분리 및 동결 원칙")**: the three
  data roles (Controlled handmade / Benchmark validation / Benchmark test),
  the deterministic-IL-repeat principle, and the freeze-schedule rule
  ("held-out test는 모든 관련 freeze 이후에만 실행... 일정이 밀리면 test
  실행도 같이 늦춘다... threshold/rule/Selective parameter를 다시 조정하지
  않는다") are all reproduced here without alteration (Sections 2, 4.2, 6).
  One honest gap, stated plainly rather than papered over: the master plan's
  10/11 target was a "val/test split 파일/목록 확정"; this document fixes
  the *rules* only, because zero benchmark instances exist to put into a
  list (Section 3.2) -- it does not fabricate a list to appear to meet that
  target.
- **Internal consistency**: the file counts in Section 3 (4 + 9 + 2 + 1 = 16
  total, 15 excluding smoke) were produced by direct, repeated filesystem
  listing during this review and match on every recount; no other section of
  this document states a different count for the same directories.
