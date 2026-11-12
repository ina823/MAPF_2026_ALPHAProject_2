"""Selective Hybrid planner wrapper for the EXISTING Global Hybrid Runner — DRAFT, Mock-verified.

Status: rule ``selective-planner-v0``. Opt-in only. Nothing in the default
runtime constructs it: ``run_global_hybrid_episode``, ``common_runner``,
the baselines and every existing script are unchanged. It plugs into the
Runner's existing injection points:

    monitor        = ShadowMonitor(DeadlockMonitor())          # records S_{t-1}, intended
    planner        = SelectivePlanner(global_planner, monitor, SelectiveConfig(...))
    path_validator = SelectivePathValidator(planner)

DEFAULT (``ladder == ("G",)``, the v0 rule LADDER_UNTIL_OUTSIDE_POLICY_VERIFIED)
    The Global planner is called exactly once with the Runner's own argument
    objects and its result object is returned as-is. The validator delegates
    to the real ``PathGuardValidator`` unchanged. The Selective analysis is
    recorded only (like Shadow). Episode results equal plain Global.

OPT-IN (``ladder`` containing "L0" / "L1", e.g. ("L0", "L1", "G"))
    Requires an explicit ``TriggerBudget`` (no default numbers: PENDING),
    ``acknowledge_unverified_outside_policy=True`` (P1-P5 are NOT resolved;
    ``decide_selective`` v0 still says GLOBAL and that decision is recorded,
    not changed), a ``subset_planner`` that is ``test_only`` (a real subset
    CBS does not exist yet -- PENDING, refused) and a Global planner that
    accepts ``timeout_ms``. Per CBS_REQUEST:

      1. WFG from the Trigger step's S_{t-1} + intended (ShadowMonitor);
         decision / groups / prechecks on S_t (= the planner's arguments,
         cross-checked against the Monitor's S_t).
      2. L0 = union of WFG groups; L1 = L0 + outside agents standing on an
         L0 goal (one expansion level; closure is PENDING). A level whose
         group is empty, equals all agents, equals the previous level, has a
         goal occupied by an outside agent or a goal unreachable with outside
         agents held (BFS) is SKIPPED before any CBS call.
      3. The remaining levels (always ending in G) run through
         ``selective_safety.next_ladder_step`` (agreed P5): no_solution /
         timeout -> next level while attempts and time remain; invalid_path,
         full-check failure -> SAFE_STOP; input_error / solver_error /
         unknown -> SAFE_STOP (UNAGREED_STATUS). Every attempt's latency,
         failed ones included, is charged; each call gets
         ``min(level timeout, remaining)``.
      4. A group success is composed with outside agents held at S_t by
         ``selective_compose.compose_selective_plan`` (group goal rule via
         Path Guard's own functions, all-agent map/vertex/edge/length).
         A G success must pass the real Path Guard.
      5. The returned result carries the SUMMED latency of all attempts,
         so the Runner's episode-level budget sees failed attempts too.
         A SAFE_STOP returns a non-success status, which the Runner's
         boundary refuses -> SAFE_STOP with no move.

Execution, expected/actual over ALL agents (outside WAIT included) and
IL_RESUME stay in the unchanged Runner (CBS_EXEC / IL_RESUME).

Mock vs real: every attempt records ``planner_kind`` ("MOCK" for a
``test_only`` planner). ``real_subset_cbs_called`` is always False in this
version. A Mock success is never evidence that a real subset CBS succeeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from src.hybrid.cbs_runtime_bridge import PathGuardValidator
from src.hybrid.replan_boundary import CBS_STATUSES
from src.hybrid.selective_compose import SelectiveFullPathValidator, compose_selective_plan
from src.hybrid.selective_decision import SelectiveDecision, decide_selective
from src.hybrid.selective_safety import (
    ACTION_CALL,
    ACTION_EXECUTE,
    LADDER_UNTIL_OUTSIDE_POLICY_VERIFIED,
    LEVEL_GLOBAL,
    Attempt,
    TriggerBudget,
    expansion_candidates,
    l0_union_group,
    next_ladder_step,
)
from src.hybrid.selective_shadow import ShadowMonitor
from src.hybrid.wfg import WfgResult, build_wfg
from src.il.nav_hint import bfs_dist

PLANNER_RULE_VERSION = "selective-planner-v0"
LEVEL_L0, LEVEL_L1 = "L0", "L1"
KNOWN_LEVELS = (LEVEL_L0, LEVEL_L1, LEVEL_GLOBAL)

# level skip reasons (decided before any CBS call)
SKIP_NOT_IN_LADDER = "NOT_IN_LADDER"
SKIP_NO_GROUP = "NO_GROUP"
SKIP_EQUALS_ALL = "EQUALS_ALL_AGENTS"
SKIP_NO_EXPANSION = "SAME_AS_PREVIOUS_LEVEL"
SKIP_GOAL_BLOCKED = "GOAL_BLOCKED"
SKIP_UNREACHABLE = "GOAL_UNREACHABLE_WITH_OUTSIDE_HELD"
SKIP_ANALYSIS = "ANALYSIS_UNAVAILABLE"

PENDING = (
    "PENDING:REAL_SUBSET_CBS",            # no real group CBS call exists (Yunji)
    "PENDING:OUTSIDE_WAIT_POLICY_P2_P3",  # holding outside agents is unverified
    "PENDING:BUDGET_NUMBERS_P4",          # total/per-level ms, max_attempts
    "PENDING:EXPANSION_CLOSURE",          # L1 is one level only
    "PENDING:SKIP_TO_GLOBAL_RULE",        # always the next ladder entry
    "PENDING:FULL_CHECK_WINDOW",          # whole padded plan is checked
    "PENDING:T7_OCCUPY_FOLLOW",           # WFG occupy edge with a moving occupant
)

Cell = tuple[int, int]


def _cells(d) -> dict[int, Cell]:
    return {int(a): (int(p[0]), int(p[1])) for a, p in d.items()}


@dataclass(frozen=True)
class SelectiveConfig:
    ladder: tuple[str, ...] = LADDER_UNTIL_OUTSIDE_POLICY_VERIFIED
    budget: Optional[TriggerBudget] = None
    acknowledge_unverified_outside_policy: bool = False

    @property
    def opt_in(self) -> bool:
        return tuple(self.ladder) != LADDER_UNTIL_OUTSIDE_POLICY_VERIFIED

    def __post_init__(self):
        lad = tuple(self.ladder)
        if not lad or lad[-1] != LEVEL_GLOBAL or len(set(lad)) != len(lad) \
                or any(lv not in KNOWN_LEVELS for lv in lad) \
                or list(lad) != sorted(lad, key=KNOWN_LEVELS.index):
            raise ValueError("ladder must be an ordered subset of ('L0', 'L1', 'G') ending with 'G'")
        if self.opt_in:
            if not isinstance(self.budget, TriggerBudget):
                raise ValueError("an L0/L1 ladder needs an explicit TriggerBudget (numbers are PENDING)")
            missing = [lv for lv in lad if lv not in self.budget.level_timeout_ms]
            if missing:
                raise ValueError(f"budget has no level_timeout_ms for {missing}")
            if self.acknowledge_unverified_outside_policy is not True:
                raise ValueError("an L0/L1 ladder needs acknowledge_unverified_outside_policy=True "
                                 "(P1-P5 unresolved; outside WAIT policy unverified)")


@dataclass(frozen=True)
class LevelPlan:
    level: str
    group: tuple[int, ...]
    viable: bool
    skip_reason: str = ""
    goal_blocked: tuple[int, ...] = ()
    unreachable: tuple[int, ...] = ()


@dataclass(frozen=True)
class AttemptRecord:
    level: str
    group: tuple[int, ...]
    timeout_ms: float
    status: Optional[str]
    latency_ms: Optional[float]
    full_check_ok: Optional[bool]
    full_check_reason: str
    planner_kind: str                      # "MOCK" | "REAL"


@dataclass
class SelectiveCallRecord:
    call_index: int
    mode: str                              # "DEFAULT_G" | "OPT_IN"
    trigger_t: Optional[int] = None
    s_t_minus_1: Optional[dict] = None
    intended: Optional[dict] = None
    blocked: tuple = ()
    s_t: Optional[dict] = None
    wfg: Optional[WfgResult] = None
    decision: Optional[SelectiveDecision] = None
    analysis_error: Optional[str] = None
    levels: list = field(default_factory=list)
    attempts: list = field(default_factory=list)
    outcome: str = ""                      # EXECUTE:<level> | SAFE_STOP:<reason> | PASS_THROUGH
    returned_status: Optional[str] = None
    executed_group: Optional[tuple] = None
    outside_held: tuple = ()
    goal_status: Optional[dict] = None
    real_subset_cbs_called: bool = False
    pending: tuple = PENDING
    rule_version: str = PLANNER_RULE_VERSION


@dataclass(frozen=True)
class SelectiveReplanResult:
    """ReplanResult-shaped answer handed back to the Runner (opt-in mode)."""
    status: str
    paths: Any
    latency_sec: Optional[float]
    error: Any = None
    level: Optional[str] = None
    group: Optional[tuple] = None


class SelectivePlanner:
    """``planner(current_positions, goals, grid)`` for run_global_hybrid_episode."""

    paths_keying = "agent_id"

    def __init__(self, global_planner: Callable, monitor: ShadowMonitor, config: SelectiveConfig,
                 subset_planner: Optional[Callable] = None):
        if not isinstance(monitor, ShadowMonitor):
            raise ValueError("SelectivePlanner needs the ShadowMonitor injected into the Runner")
        if not isinstance(config, SelectiveConfig):
            raise ValueError("config must be a SelectiveConfig")
        if getattr(global_planner, "paths_keying", "index") != "agent_id":
            raise ValueError("global_planner must be agent_id keyed (like CbsRuntimePlanner)")
        if config.opt_in:
            if subset_planner is None:
                raise ValueError("an L0/L1 ladder needs a subset_planner")
            if not getattr(subset_planner, "test_only", False):
                raise ValueError("PENDING:REAL_SUBSET_CBS -- a real subset CBS planner is not available/verified; "
                                 "only test_only subset planners are accepted")
            if not getattr(global_planner, "accepts_timeout_ms", False):
                raise ValueError("an L0/L1 ladder needs a global planner that accepts timeout_ms "
                                 "(per-call timeout <= remaining budget)")
        self.global_planner = global_planner
        self.subset_planner = subset_planner
        self.monitor = monitor
        self.config = config
        self.records: list[SelectiveCallRecord] = []
        self._pending: Optional[tuple] = None    # consumed by SelectivePathValidator

    @property
    def test_only(self):
        return bool(getattr(self.global_planner, "test_only", False))

    # -- Runner entry point --------------------------------------------------
    def __call__(self, current_positions, goals, grid):
        self._pending = None
        rec = SelectiveCallRecord(call_index=len(self.records) + 1,
                                  mode="OPT_IN" if self.config.opt_in else "DEFAULT_G")
        self.records.append(rec)
        try:
            self._analyse(rec, current_positions, goals, grid)
        except Exception as exc:  # analysis must never change the Global path
            rec.analysis_error = f"EXCEPTION:{type(exc).__name__}:{exc}"
        if not self.config.opt_in:
            rec.outcome = "PASS_THROUGH"
            try:
                result = self.global_planner(current_positions, goals, grid)
            except BaseException as exc:                                   # recorded, then propagated
                rec.returned_status = f"EXCEPTION:{type(exc).__name__}"
                raise
            rec.returned_status = getattr(result, "status", None)
            self._pending = ("G",)
            return result
        return self._run_ladder(rec, current_positions, goals, grid)

    # -- analysis (S_{t-1}/intended for WFG, S_t for the rest) ---------------
    def _analyse(self, rec, current_positions, goals, grid):
        step = self.monitor.last_step
        if step is None or step.episode != self.monitor.episode:
            rec.analysis_error = "NO_STEP_OBSERVED"
            return
        rec.trigger_t, rec.s_t_minus_1, rec.intended = step.t, step.s_t_minus_1, step.intended
        rec.blocked, rec.s_t = step.blocked, step.s_t
        if step.s_t_minus_1 is None or step.intended is None:
            rec.analysis_error = "INTENT_NOT_OBSERVED"
            return
        if _cells(current_positions) != step.s_t:
            rec.analysis_error = "S_T_MISMATCH"
            return
        grid_arr = np.array(grid, dtype=np.int64, copy=True)
        rec.wfg = build_wfg(step.s_t_minus_1, step.intended, step.blocked, grid_arr)
        rec.decision = decide_selective(sorted(step.s_t), step.s_t, _cells(goals), rec.wfg)

    def _level_plans(self, rec, s_t, goals, grid_arr) -> dict[str, LevelPlan]:
        ids = tuple(sorted(s_t))
        lad = self.config.ladder
        plans = {LEVEL_GLOBAL: LevelPlan(LEVEL_GLOBAL, ids, True)}
        if rec.wfg is None:
            for lv in (LEVEL_L0, LEVEL_L1):
                plans[lv] = LevelPlan(lv, (), False, SKIP_ANALYSIS)
            return plans

        def check(level, group, previous=None):
            if level not in lad:
                return LevelPlan(level, group, False, SKIP_NOT_IN_LADDER)
            if not group:
                return LevelPlan(level, group, False, SKIP_NO_GROUP)
            if set(group) == set(ids):
                return LevelPlan(level, group, False, SKIP_EQUALS_ALL)
            if previous is not None and group == previous:
                return LevelPlan(level, group, False, SKIP_NO_EXPANSION)
            hits, _ = expansion_candidates(group, s_t, goals)
            gb = tuple(sorted({h.group_agent for h in hits if h.kind == "GOAL_BLOCKED"}))
            held = grid_arr.copy()
            for a in ids:
                if a not in group:
                    held[s_t[a]] = 1
            unreach = tuple(a for a in group
                            if held[goals[a]] == 1 or int(bfs_dist(held, goals[a])[s_t[a]]) < 0)
            if gb:
                return LevelPlan(level, group, False, SKIP_GOAL_BLOCKED, gb, unreach)
            if unreach:
                return LevelPlan(level, group, False, SKIP_UNREACHABLE, gb, unreach)
            return LevelPlan(level, group, True, "", gb, unreach)

        l0 = l0_union_group(rec.wfg.groups)
        plans[LEVEL_L0] = check(LEVEL_L0, l0)
        _, add = expansion_candidates(l0, s_t, goals) if l0 else ((), ())
        l1 = tuple(sorted(set(l0) | set(add)))
        plans[LEVEL_L1] = check(LEVEL_L1, l1, previous=l0)
        return plans

    # -- opt-in ladder --------------------------------------------------------
    def _run_ladder(self, rec, current_positions, goals, grid):
        s_t = _cells(current_positions)
        gl = _cells(goals)
        ids = tuple(sorted(s_t))
        grid_arr = np.array(grid, dtype=np.int64, copy=True)
        if rec.analysis_error is not None:
            rec.wfg = None
        plans = self._level_plans(rec, s_t, gl, grid_arr)
        rec.levels = [plans[lv] for lv in KNOWN_LEVELS]
        ladder = tuple(lv for lv in self.config.ladder if plans[lv].viable)   # always ends in G
        budget = self.config.budget
        attempts: list[Attempt] = []
        results = []
        while True:
            step = next_ladder_step(ladder, attempts, budget)
            if step.action == ACTION_CALL:
                plan = plans[step.level]
                res, att, extra = self._call_level(plan, current_positions, goals, grid, s_t, gl, ids,
                                                   step.timeout_ms)
                rec.attempts.append(att)
                attempts.append(Attempt(level=plan.level, group=plan.group, status=att.status,
                                        latency_ms=att.latency_ms, full_check_ok=att.full_check_ok))
                results.append((res, extra))
                continue
            latencies = [a.latency_ms for a in attempts]
            total_sec = None if any(v is None for v in latencies) else sum(latencies) / 1000.0
            if step.action == ACTION_EXECUTE:
                res, extra = results[-1]
                rec.outcome = f"EXECUTE:{step.level}"
                if step.level == LEVEL_GLOBAL:
                    rec.executed_group = ids
                    self._pending = ("G",)
                    out = SelectiveReplanResult("success", res.paths, total_sec, None, LEVEL_GLOBAL, ids)
                else:
                    composed = extra
                    rec.executed_group = composed.group
                    rec.outside_held = composed.outside
                    rec.goal_status = dict(composed.goal_status)
                    self._pending = ("SELECTIVE", composed.group, composed.full_paths)
                    out = SelectiveReplanResult("success", composed.full_paths, total_sec, None,
                                                step.level, composed.group)
                rec.returned_status = out.status
                return out
            # SAFE_STOP: a non-success status -> the Runner's boundary refuses, no move.
            last = attempts[-1].status if attempts else None
            if step.reason == "FULL_CHECK_FAILED" or last in (None, "success"):
                status = "invalid_path"
            else:
                status = last if last in CBS_STATUSES else str(last)
            rec.outcome = f"SAFE_STOP:{step.reason}"
            rec.returned_status = status
            return SelectiveReplanResult(status, None, total_sec, f"SELECTIVE_SAFE_STOP:{step.reason}")

    def _call_level(self, plan, current_positions, goals, grid, s_t, gl, ids, timeout_ms):
        if plan.level == LEVEL_GLOBAL:
            planner = self.global_planner
            res = planner(current_positions, goals, grid, timeout_ms=timeout_ms)
        else:
            planner = self.subset_planner
            res = planner({a: s_t[a] for a in plan.group}, {a: gl[a] for a in plan.group},
                          np.array(grid, copy=True),
                          outside={a: s_t[a] for a in ids if a not in plan.group},
                          timeout_ms=timeout_ms)
        status = getattr(res, "status", None)
        lat = getattr(res, "latency_sec", None)
        lat_ms = None if lat is None else float(lat) * 1000.0
        ok, why, extra = None, "", None
        if status == "success":
            if plan.level == LEVEL_GLOBAL:
                try:
                    ok, why = PathGuardValidator()(grid, s_t, gl, res.paths)
                except Exception as exc:
                    ok, why = False, f"{type(exc).__name__}"
            else:
                raw = getattr(res, "paths", None)
                if not isinstance(raw, dict) or set(raw) != set(plan.group):
                    ok, why = False, "SELECTIVE_GROUP_RESULT:agent_set"
                else:
                    composed = compose_selective_plan(raw, s_t, gl, ids, grid)
                    ok, why, extra = composed.ok, composed.reason, composed
        kind = "MOCK" if getattr(planner, "test_only", False) else "REAL"
        return res, AttemptRecord(plan.level, plan.group, float(timeout_ms), status, lat_ms, ok, why, kind), extra


class SelectivePathValidator:
    """``path_validator`` paired with one SelectivePlanner.

    G plan          -> the real PathGuardValidator, unchanged (all-agent goal check kept).
    Selective plan  -> SelectiveFullPathValidator(group) AND the paths must be
                       exactly the plan the planner composed and checked.
    Nothing pending -> rejected. Each verdict consumes the pending plan."""

    def __init__(self, planner: SelectivePlanner):
        if not isinstance(planner, SelectivePlanner):
            raise ValueError("SelectivePathValidator needs its SelectivePlanner")
        self.planner = planner
        self.calls: list[tuple] = []

    def __call__(self, grid, starts, goals, paths):
        pending, self.planner._pending = self.planner._pending, None
        if pending is None:
            verdict = (False, "SELECTIVE_NO_PENDING_PLAN")
        elif pending[0] == "G":
            verdict = PathGuardValidator()(grid, starts, goals, paths)
        else:
            _, group, full = pending
            verdict = SelectiveFullPathValidator(group)(grid, starts, goals, paths)
            if verdict[0]:
                given = {int(a): [tuple(c) for c in p] for a, p in paths.items()}
                if given != full:
                    verdict = (False, "SELECTIVE_PLAN_NOT_THE_COMPOSED_ONE")
        self.calls.append(("G" if pending and pending[0] == "G" else
                           "SELECTIVE" if pending else None, verdict))
        return verdict
