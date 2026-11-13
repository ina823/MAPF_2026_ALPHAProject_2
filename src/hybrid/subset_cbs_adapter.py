"""Real subset-CBS adapter (and timeout-bounded Global CBS) for SelectivePlanner -- DRAFT.

Status: rule ``subset-cbs-adapter-v0``. Wired OPT-IN only (2026-11-13, 1st
connection): ``SelectivePlanner`` accepts this exact class as its subset
planner only with ``allow_real_subset_cbs=True``; every other non-``test_only``
subset planner is still refused (PENDING:REAL_SUBSET_CBS). This class is
deliberately NOT ``test_only``. Nothing in the default runtime constructs it.
Yunji's standalone check (``devtools/check_selective_subset_cbs.py``) and the
planner-level real-solver tests show that the real atb033 CBS can plan one
subset. Real Runner E2E on DEV scenarios (not held-out) is recorded in
docs/selective_real_e2e_checkpoint_20261113.md; it does not cover an outside
agent that was still moving before the Trigger.

``BoundedGlobalCbsPlanner`` (below) is the ladder's G level: the same bounded
call on ALL agents (no outside agents), ``accepts_timeout_ms = True``. It
exists because ``CbsRuntimePlanner`` has only a whole-object ``timeout_sec``
and cannot honour the per-call ``min(level timeout, remaining)``.

Timeout enforcement measured with the real solver on Windows (2026-11-13):
``subprocess.run(timeout=...)`` does terminate ``cbs.py`` (no process left
after the call), but the measured wall time of a timed-out call exceeded
``timeout_ms`` by ~55-105 ms (kill + pipe drain + YAML I/O). The limit is
therefore a hard STOP, not a hard wall-clock BOUND; ``overhead_reserve_ms``
is the only margin (value PENDING P4) and overruns are recorded, charged to
the Trigger budget, and never executed by ``SelectivePlanner``.

Call contract (the signature ``SelectivePlanner._call_level`` already uses)::

    planner(current_positions, goals, grid, *, outside, timeout_ms) -> CBSRuntimeResult

    current_positions / goals : {agent_id: (row, col)} for the GROUP only
    outside                   : {agent_id: (row, col)} S_t of every agent NOT in the group
    grid                      : 2-D, 0 = free, 1 = obstacle; never modified
    timeout_ms                : this call's budget (the ladder's min(level, remaining))

    result.status     : one of replan_boundary.CBS_STATUSES
    result.paths      : {group agent_id: [(row, col), ...]} on success, else None
    result.latency_sec: wall time of the WHOLE adapter call (>= solver latency)
    result.error      : reason on failure, None on success

What it does:
  * Only the group goes to CBS. Agent ids and (row, col) stay as given; the
    only index <-> id and (row, col) <-> [x, y] conversions are the existing
    ones inside ``run_cbs_runtime`` / ``cbs_adapter``.
  * Outside agents are static obstacles on a COPY of the grid (candidate
    policy P3-b, unverified). The caller's grid is never written.
  * Each call runs with cwd = a fresh ``<work_root>/subset_cbs_call_NN``
    (``exist_ok=False``), so ``src/replanner.py``'s fixed relative
    ``outputs/paths/replan`` never touches an existing YAML or outputs.
    The cwd switch is process-wide: not thread-safe.
  * Timeout: the existing chain passes ``timeout_sec`` unchanged down to
    ``subprocess.run(timeout=...)`` in ``cbs_adapter.run_atb033_cbs``; it is
    the only place a limit applies and it accepts a float. This adapter
    passes ``(timeout_ms - overhead_reserve_ms) / 1000`` as a float -- never
    rounded up, so the solver's limit is <= ``timeout_ms``. Work outside the
    subprocess (YAML I/O, interpreter start inside the limit, Path Guard) is
    NOT bounded by it; the measured total is returned and any overrun is
    recorded (``budget_overrun_ms``). If no positive solver limit is left
    the call is BLOCKED (not run).
  * Before any call (BLOCKED -> ``input_error``, CBS not run): malformed
    ids/cells, group/outside overlap, out-of-bounds or obstacle cells,
    overlapping starts, a group goal occupied by an outside agent, bad
    timeout, existing work dir.
  * After a success: paths must be keyed by exactly the group ids, satisfy
    Path Guard's endpoint/movement rules, avoid outside cells, and compose
    with the outside agents held at S_t without any vertex/edge conflict on
    the ORIGINAL grid (``validate_full_robot``) -> otherwise ``invalid_path``
    with no paths. Nothing is repaired.
"""

from __future__ import annotations

import contextlib
import math
import operator
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from src.cbs_runtime_adapter import CBSRuntimeResult, run_cbs_runtime
from src.hybrid.replan_boundary import CBS_STATUSES
from src.hybrid.selective_safety import validate_full_robot
from src.path_guard import (
    InvalidPathError,
    validate_path_endpoints,
    validate_path_map_and_conflicts,
    validate_path_movements,
)

ADAPTER_RULE_VERSION = "subset-cbs-adapter-v0"
CALL_DIR_PREFIX = "subset_cbs_call_"
GLOBAL_CALL_DIR_PREFIX = "global_cbs_call_"

BLOCKED = "SUBSET_CBS_BLOCKED"          # refused before CBS; status input_error
REJECTED = "SUBSET_CBS_REJECTED"        # CBS ran, result refused; status invalid_path

Cell = tuple[int, int]


class _Block(ValueError):
    pass


def _id(v) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
        raise _Block(f"AGENT_ID:{v!r}")
    return int(v)


def _cell(v, label) -> Cell:
    if isinstance(v, (str, bytes)) or not isinstance(v, (tuple, list, np.ndarray)) or len(v) != 2 \
            or any(isinstance(x, bool) for x in v):
        raise _Block(f"CELL:{label}:{v!r}")
    try:
        return (operator.index(v[0]), operator.index(v[1]))
    except TypeError:
        raise _Block(f"CELL:{label}:{v!r}") from None


def _cell_map(d, label) -> dict[int, Cell]:
    if not isinstance(d, Mapping):
        raise _Block(f"NOT_A_MAPPING:{label}")
    return {_id(a): _cell(p, f"{label}[{a!r}]") for a, p in d.items()}


def _solver_timeout_sec(timeout_ms, reserve_ms: float) -> float:
    if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, (int, float)) \
            or not math.isfinite(timeout_ms) or timeout_ms <= 0:
        raise _Block(f"TIMEOUT_MS:{timeout_ms!r}")
    sec = (float(timeout_ms) - reserve_ms) / 1000.0
    if not sec > 0:
        raise _Block(f"TIMEOUT_BUDGET:timeout_ms={timeout_ms} <= overhead_reserve_ms={reserve_ms}")
    return sec


class SubsetCbsPlanner:
    """Real subset CBS for one Selective group. Not ``test_only``."""

    paths_keying = "agent_id"
    test_only = False
    rule_version = ADAPTER_RULE_VERSION
    call_dir_prefix = CALL_DIR_PREFIX

    def __init__(self, *, solver_root, work_root, overhead_reserve_ms: float):
        self.solver_root = Path(solver_root).resolve()
        if not (self.solver_root / "cbs.py").is_file():
            raise FileNotFoundError(f"atb033 cbs.py not found under {self.solver_root}")
        if isinstance(overhead_reserve_ms, bool) or not isinstance(overhead_reserve_ms, (int, float)) \
                or not math.isfinite(overhead_reserve_ms) or overhead_reserve_ms < 0:
            raise ValueError("overhead_reserve_ms must be a finite number >= 0 (value is PENDING P4)")
        self.overhead_reserve_ms = float(overhead_reserve_ms)
        self.work_root = Path(work_root).resolve()
        self.calls: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------------
    def __call__(self, current_positions, goals, grid, *, outside, timeout_ms) -> CBSRuntimeResult:
        t0 = perf_counter()
        rec: Dict[str, Any] = {
            "call_index": len(self.calls) + 1,
            "timeout_ms": timeout_ms,
            "overhead_reserve_ms": self.overhead_reserve_ms,
            "solver_timeout_sec": None,
            "called": False,
            "work_dir": None,
            "pythonhashseed_env": os.environ.get("PYTHONHASHSEED"),
        }
        self.calls.append(rec)

        def finish(status, paths, error, inner=None):
            total = perf_counter() - t0
            rec.update(status=status, error=error, paths=paths, latency_sec=total,
                       solver_latency_sec=inner)
            if rec["called"] and isinstance(timeout_ms, (int, float)) and not isinstance(timeout_ms, bool):
                over = total * 1000.0 - float(timeout_ms)
                rec["budget_overrun_ms"] = over if over > 0 else 0.0
            return CBSRuntimeResult(status=status, paths=paths, latency_sec=total, error=error)

        # -- 1. checks before any CBS call ----------------------------------
        try:
            prep = self._prepare(current_positions, goals, grid, outside, timeout_ms)
        except _Block as exc:
            return finish("input_error", None, f"{BLOCKED}:{exc}")
        start, goal, out, base, planning, solver_sec = prep
        rec.update(group=tuple(sorted(start)), outside=tuple(sorted(out)), solver_timeout_sec=solver_sec)

        work_dir = self.work_root / f"{self.call_dir_prefix}{rec['call_index']:02d}"
        try:
            work_dir.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            return finish("input_error", None, f"{BLOCKED}:WORK_DIR:{type(exc).__name__}:{work_dir}")
        rec["work_dir"] = work_dir

        # -- 2. the real CBS chain, group only, copied grid -------------------
        rec["called"] = True
        try:
            with contextlib.chdir(work_dir):
                res = run_cbs_runtime(start, goal, planning, timeout_sec=solver_sec,
                                      solver_root=self.solver_root)
        except Exception as exc:  # the chain should not raise; never let it escape
            return finish("solver_error", None, f"{type(exc).__name__}:{exc}")
        inner = getattr(res, "latency_sec", None)
        status = getattr(res, "status", None)
        if status not in CBS_STATUSES:
            return finish("solver_error", None, f"UNKNOWN_STATUS:{status!r}", inner)
        if status != "success":
            return finish(status, None, getattr(res, "error", None), inner)

        # -- 3. checks after a success ---------------------------------------
        try:
            why, paths = self._check_success(getattr(res, "paths", None), start, goal, out, base, planning)
        except Exception as exc:  # a malformed result must be refused, never raise or pass
            why, paths = f"CHECK_EXCEPTION:{type(exc).__name__}", None
        if why:
            return finish("invalid_path", None, f"{REJECTED}:{why}", inner)
        return finish("success", paths, None, inner)

    # ------------------------------------------------------------------------
    def _prepare(self, current_positions, goals, grid, outside, timeout_ms):
        solver_sec = _solver_timeout_sec(timeout_ms, self.overhead_reserve_ms)
        start = _cell_map(current_positions, "current_positions")
        goal = _cell_map(goals, "goals")
        out = _cell_map(outside, "outside")
        if not start:
            raise _Block("EMPTY_GROUP")
        if set(start) != set(goal):
            raise _Block("GROUP_ID_MISMATCH")
        if set(start) & set(out):
            raise _Block(f"GROUP_OUTSIDE_OVERLAP:{sorted(set(start) & set(out))}")
        try:
            raw = np.array(grid, copy=True)                        # check values BEFORE any int cast
            if raw.ndim != 2 or raw.size == 0 or raw.dtype == object or not np.isin(raw, (0, 1)).all():
                raise _Block("GRID")
            base = raw.astype(np.int64)
        except (TypeError, ValueError):
            raise _Block("GRID") from None
        h, w = base.shape
        for label, cells in (("start", start), ("goal", goal), ("outside", out)):
            for a, (r, c) in cells.items():
                if not (0 <= r < h and 0 <= c < w):
                    raise _Block(f"OUT_OF_BOUNDS:{label}:{a}:{(r, c)}")
                if base[r, c] == 1:
                    raise _Block(f"ON_OBSTACLE:{label}:{a}:{(r, c)}")
        occupied = list(start.values()) + list(out.values())
        if len(set(occupied)) != len(occupied):
            raise _Block("OVERLAPPING_STARTS")
        if len(set(goal.values())) != len(goal):
            raise _Block("DUPLICATE_GROUP_GOALS")
        held = set(out.values())
        blocked = sorted(a for a, g in goal.items() if g in held)
        if blocked:
            raise _Block(f"GOAL_BLOCKED:{blocked}")
        planning = base.copy()
        for cell in held:
            planning[cell] = 1
        return start, goal, out, base, planning, solver_sec

    @staticmethod
    def _check_success(raw, start, goal, out, base, planning):
        if not isinstance(raw, Mapping):
            return "PATHS_NOT_A_MAPPING", None
        try:
            keys = {_id(a) for a in raw}
        except _Block:
            return "AGENT_SET", None
        if keys != set(start) or len(keys) != len(raw):
            return "AGENT_SET", None
        try:
            norm = {}
            for a, p in raw.items():
                if isinstance(p, (str, bytes)) or not hasattr(p, "__len__") or len(p) == 0:
                    return f"MALFORMED:{a}", None
                norm[int(a)] = [_cell(c, f"path[{a}]") for c in p]
            if len({len(p) for p in norm.values()}) != 1:
                return "UNEQUAL_LENGTH", None
            validate_path_endpoints(norm, start, goal)
            validate_path_movements(norm)
            validate_path_map_and_conflicts(norm, planning)        # outside cells = obstacles
        except (_Block, InvalidPathError, ValueError, TypeError, IndexError) as exc:
            return f"GROUP_PATH:{exc}", None
        everyone = {**start, **out}
        full = validate_full_robot(norm, everyone, sorted(everyone), base)
        if not full.ok:
            return f"FULL:{full.reason}", None
        for a in out:
            if any(c != out[a] for c in full.full_paths[a]):
                return f"OUTSIDE_NOT_HELD:{a}", None
        return "", {a: list(p) for a, p in sorted(norm.items())}


class BoundedGlobalCbsPlanner:
    """Real Global (all-agent) CBS whose every call honours ``timeout_ms``.

    ``planner(current_positions, goals, grid, *, timeout_ms)`` -> CBSRuntimeResult,
    agent-id keyed like ``CbsRuntimePlanner``. It is the same bounded call as
    ``SubsetCbsPlanner`` with no outside agents: solver limit
    ``(timeout_ms - overhead_reserve_ms) / 1000`` as a float (never rounded
    up), fresh ``<work_root>/global_cbs_call_NN`` cwd, the same pre-call input
    checks and post-success checks (endpoints = every agent reaches its goal,
    movements, map/vertex/edge). ``latency_sec`` is the whole call's wall time.
    A call without ``timeout_ms`` is refused (``input_error``, CBS not run):
    an unbounded Global call cannot be charged to a Trigger budget.
    """

    paths_keying = "agent_id"
    test_only = False
    accepts_timeout_ms = True
    rule_version = ADAPTER_RULE_VERSION

    def __init__(self, *, solver_root, work_root, overhead_reserve_ms: float):
        core = SubsetCbsPlanner(solver_root=solver_root, work_root=work_root,
                                overhead_reserve_ms=overhead_reserve_ms)
        core.call_dir_prefix = GLOBAL_CALL_DIR_PREFIX
        self._core = core
        self.solver_root = core.solver_root
        self.work_root = core.work_root
        self.overhead_reserve_ms = core.overhead_reserve_ms

    @property
    def calls(self) -> List[Dict[str, Any]]:
        return self._core.calls

    def __call__(self, current_positions, goals, grid, *, timeout_ms=None) -> CBSRuntimeResult:
        return self._core(current_positions, goals, grid, outside={}, timeout_ms=timeout_ms)
