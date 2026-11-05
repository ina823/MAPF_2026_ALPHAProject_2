"""Common run interface for comparing MAPF methods on identical inputs.

``run_method(RunSpec) -> EpisodeRecord`` runs one scenario with one method by
calling the existing runner for that method, unchanged:

    IL_ONLY                   scripts/run_il.run_episode (no safety)
    SAFETY_ONLY_DEV_K2        scripts/run_il.run_episode + SafetyOnlyV0(hold_steps=2)
    PURE_CBS_DEV_SINGLE_PLAN  src/compare/pure_cbs.run_pure_cbs_episode
    GLOBAL_HYBRID_DEV_FULL    run_global_hybrid_episode + dev_config(max_steps)
    GLOBAL_HYBRID_DEV_H3      run_global_hybrid_episode + dev_resume_config(3)
    GLOBAL_HYBRID_DEV_H3_REARM  run_global_hybrid_episode + dev_rearm_config(3)
                              (opt-in only: not in METHODS, so not in the
                              default batch; Monitor rearm + bounded recall)

Every method gets the same scenario loader ((row, col), agent id = list
index, t = 0 initial state), the same checkpoint, max_steps (default 160),
an observe-only DeadlockMonitor and an EpisodeLogger. CBS methods use the
real CbsRuntimePlanner (solver timeout default 30 s -- a different limit from
max_steps) and the real PathGuardValidator; their solver work dirs live under
the run directory.

The DEV settings are named in the method and in EpisodeRecord.dev_settings;
none of them is an agreed project policy (H_exec, IL resume, CBS recall,
SAFE_STOP and the Safety-only parameter are still open). The DEV policy
objects come from the existing DEV scripts so they are not redefined here.
Selective / WFG is not implemented and is refused.

EpisodeRecord keeps the runner's own result in ``raw`` untouched. Fields the
runner does not report are derived without changing it and labelled:
run_episode has no termination field, so IL/Safety termination is derived
from its flags (termination_source); run_episode checks overlaps only with a
Safety controller, so IL-only reports overlap_checked_at_runtime=False and,
when the trajectory does contain an overlap, integrity_unhandled=True.

Each run writes only into a new ``<out_dir>/<run_id>/`` (refused if it
exists): result.json, <run_id>_steps.csv, _summary.csv, _monitor.csv and,
for CBS methods, cbs/cbs_call_NN/.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Imported at module load so provenance.STARTUP_ENV_VALUE is captured early.
from src.compare.provenance import hash_seed_info

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IL_ONLY = "IL_ONLY"
SAFETY_ONLY_DEV_K2 = "SAFETY_ONLY_DEV_K2"
PURE_CBS_DEV_SINGLE_PLAN = "PURE_CBS_DEV_SINGLE_PLAN"
GLOBAL_HYBRID_DEV_FULL = "GLOBAL_HYBRID_DEV_FULL"
GLOBAL_HYBRID_DEV_H3 = "GLOBAL_HYBRID_DEV_H3"
GLOBAL_HYBRID_DEV_H3_REARM = "GLOBAL_HYBRID_DEV_H3_REARM"
METHODS = (IL_ONLY, SAFETY_ONLY_DEV_K2, PURE_CBS_DEV_SINGLE_PLAN, GLOBAL_HYBRID_DEV_FULL, GLOBAL_HYBRID_DEV_H3)
OPT_IN_METHODS = (GLOBAL_HYBRID_DEV_H3_REARM,)   # run only when named explicitly
ALL_METHODS = METHODS + OPT_IN_METHODS
CBS_METHODS = (PURE_CBS_DEV_SINGLE_PLAN, GLOBAL_HYBRID_DEV_FULL, GLOBAL_HYBRID_DEV_H3,
               GLOBAL_HYBRID_DEV_H3_REARM)
IL_METHODS = (IL_ONLY, SAFETY_ONLY_DEV_K2, GLOBAL_HYBRID_DEV_FULL, GLOBAL_HYBRID_DEV_H3,
              GLOBAL_HYBRID_DEV_H3_REARM)
NOT_IMPLEMENTED = ("SELECTIVE_HYBRID", "SELECTIVE_WFG", "WFG")

DEFAULT_MAX_STEPS = 160
DEFAULT_CBS_TIMEOUT_SEC = 30
SAFETY_HOLD_STEPS = 2   # DEV trial value of SAFETY_ONLY_DEV_K2
H_EXEC_DEV = 3          # DEV trial value of GLOBAL_HYBRID_DEV_H3

NOT_AGREED = "development setting, not an agreed project policy"
DEV_SETTINGS = {
    IL_ONLY: {},
    SAFETY_ONLY_DEV_K2: {
        "hold_steps": SAFETY_HOLD_STEPS,
        "rule": "after the first Full-repeat Trigger, the lowest-id active agent WAITs hold_steps steps",
        "status": NOT_AGREED,
    },
    PURE_CBS_DEV_SINGLE_PLAN: {
        "cbs_calls": "exactly one, at t=0 on S_0; no re-planning",
        "execution": "whole accepted plan, up to max_steps",
        "status": NOT_AGREED,
    },
    GLOBAL_HYBRID_DEV_FULL: {
        "h_exec": "whole accepted CBS path (GlobalHybridConfig.h_exec = max_steps)",
        "resume": "SAFE_STOP when the path ends off-goal (no IL resume)",
        "recall": "first Trigger only",
        "status": NOT_AGREED,
    },
    GLOBAL_HYBRID_DEV_H3: {
        "h_exec": H_EXEC_DEV,
        "resume": "always IL after the CBS window",
        "recall": "first Trigger only; the Monitor latches its first Trigger",
        "status": NOT_AGREED,
    },
    GLOBAL_HYBRID_DEV_H3_REARM: {
        "h_exec": H_EXEC_DEV,
        "resume": "always IL after the CBS window; Monitor rearmed at each IL_RESUME -> IL",
        "recall": "DEV_ONLY candidate limits (MonitorRearmConfig)",
        "rearm_limits": "filled per run from dev_rearm_config (max_cbs_calls includes the first call)",
        "on_refusal": "SAFE_STOP: RECALL_LIMIT_EXCEEDED / CBS_TIME_BUDGET_EXCEEDED / "
                      "RECALL_COOLDOWN_ACTIVE / RECALL_SAME_STATE",
        "status": NOT_AGREED,
    },
}


@dataclass(frozen=True)
class RunSpec:
    method: str
    scenario: Path
    out_dir: Path
    max_steps: int = DEFAULT_MAX_STEPS
    ckpt: Optional[Path] = None              # None -> scripts/run_il.DEFAULT_CKPT
    solver_root: Optional[Path] = None       # required by CBS methods
    cbs_timeout_sec: int = DEFAULT_CBS_TIMEOUT_SEC


@dataclass
class EpisodeRecord:
    method: str
    dev_settings: Dict[str, Any]
    run_id: str
    run_dir: str
    scenario: str
    scenario_sha256: str
    grid_sha256: str
    checkpoint: Optional[str]
    checkpoint_sha256: Optional[str]
    max_steps: int
    cbs_timeout_sec: Optional[int]
    agent_ids: Tuple[int, ...]
    starts: Dict[int, Tuple[int, int]]
    goals: Dict[int, Tuple[int, int]]
    termination: str
    termination_source: str                  # "RUNNER" | "DERIVED_FROM_FLAGS"
    stop_reason: Optional[str]
    all_at_goal: bool
    timed_out: bool
    steps: int
    trajectory: List[Dict[int, Tuple[int, int]]]
    actions: List[Dict[int, int]]
    action_sources: List[str]
    overlap_any_step: bool
    first_overlap_t: Optional[int]
    overlap_checked_at_runtime: bool
    integrity_unhandled: bool
    cbs_calls: List[Dict[str, Any]]
    planner_calls: List[Dict[str, Any]]      # CbsRuntimePlanner.calls copy (paths, work_dir, solver_root)
    cbs_exec_trace: List[Dict[str, Any]]
    transitions: List[Dict[str, Any]]
    safety_holds: List[Tuple[int, int]]      # (t, agent_id) steps Safety-only forced WAIT
    safety_armed_t: Optional[int]
    monitor: Dict[str, Any]
    environment: Dict[str, Any]
    raw: Dict[str, Any] = field(repr=False)


class _SafetyRecorder:
    """Delegates to SafetyOnlyV0 unchanged; records where it forced a WAIT."""

    def __init__(self, inner):
        self._inner = inner
        self.holds = []
        self.armed_t = None
        self._t = 0

    def reset(self):
        self.holds, self.armed_t, self._t = [], None, 0
        return self._inner.reset()

    def apply(self, actions):
        self._t += 1                          # apply() runs once per step, before sim.step()
        if self._inner.remaining > 0:
            self.holds.append((self._t, self._inner.selected_agent))
        return self._inner.apply(actions)

    def arm(self, positions, goals):
        armed = self._inner.arm(positions, goals)
        if armed:
            self.armed_t = self._t
        return armed


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha256_grid(grid):
    h = hashlib.sha256()
    h.update(f"{grid.dtype.str}|{grid.shape}|".encode())
    h.update(grid.tobytes())
    return h.hexdigest()


def _git(path):
    try:
        commit = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return {"commit": commit, "dirty": bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _version(dist):
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def environment_info(solver_root=None):
    return {
        "hash_seed": hash_seed_info(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": _version("torch"),
        "numpy": _version("numpy"),
        "pyyaml": _version("PyYAML"),
        "project_git": _git(PROJECT_ROOT),
        "solver_git": _git(solver_root) if solver_root is not None else None,
        "command": " ".join(sys.argv),
    }


def _first_overlap(trajectory):
    return next((t for t, state in enumerate(trajectory) if len(set(state.values())) != len(state)), None)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    return obj


def _planner_calls(planner):
    """Record-only copy of CbsRuntimePlanner.calls (the paths the solver returned)."""
    if planner is None:
        return []
    calls = []
    for c in planner.calls:
        paths = c.get("paths")
        calls.append({
            "solver_root": str(c["solver_root"]),
            "work_dir": str(c["work_dir"]),
            "timeout_sec": c["timeout_sec"],
            "status": c.get("status"),
            **({"pythonhashseed_env": c["pythonhashseed_env"]} if "pythonhashseed_env" in c else {}),
            "paths": ({aid: [tuple(p) for p in path] for aid, path in paths.items()}
                      if paths is not None else None),
        })
    return calls


def _check_spec(spec):
    if spec.method in NOT_IMPLEMENTED:
        raise NotImplementedError(f"{spec.method} is not implemented")
    if spec.method not in ALL_METHODS:
        raise ValueError(f"unknown method {spec.method!r}; known: {', '.join(ALL_METHODS)}")
    if not isinstance(spec.max_steps, int) or spec.max_steps < 1:
        raise ValueError("max_steps must be an int >= 1")
    if spec.method in CBS_METHODS:
        if spec.solver_root is None:
            raise ValueError(f"{spec.method} needs solver_root")
        if not isinstance(spec.cbs_timeout_sec, int) or spec.cbs_timeout_sec < 1:
            raise ValueError("cbs_timeout_sec must be an int >= 1")


def run_method(spec: RunSpec) -> EpisodeRecord:
    """Run one scenario with one method into a new run directory."""
    _check_spec(spec)

    from scripts.run_corridor_hybrid_e2e import dev_config
    from scripts.run_hybrid_resume_dev import dev_rearm_config, dev_resume_config
    from scripts.run_il import DEFAULT_CKPT, load_scenario, run_episode
    from src.common.deadlock_monitor import DeadlockMonitor
    from src.common.episode_logger import EpisodeLogger, make_run_id
    from src.common.safety_only import SafetyOnlyV0
    from src.compare.pure_cbs import run_pure_cbs_episode
    from src.hybrid.cbs_runtime_bridge import CbsRuntimePlanner, PathGuardValidator
    from src.hybrid.global_hybrid import run_global_hybrid_episode

    scenario = Path(spec.scenario)
    ckpt = Path(spec.ckpt) if spec.ckpt is not None else DEFAULT_CKPT
    grid, starts, goals = load_scenario(scenario)

    run_id = make_run_id(scenario.name, spec.method)
    run_dir = Path(spec.out_dir).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)  # never reuse a run directory
    logger = EpisodeLogger(run_id, scenario=scenario.name, mode=spec.method, out_dir=run_dir)
    monitor = DeadlockMonitor()

    def il_policy():
        from src.il.policy import NavHintPolicy
        return NavHintPolicy(ckpt)

    def cbs_planner():
        return CbsRuntimePlanner(solver_root=spec.solver_root, work_root=run_dir / "cbs",
                                 timeout_sec=spec.cbs_timeout_sec)

    safety = None
    planner = cbs_planner() if spec.method in CBS_METHODS else None
    if spec.method == IL_ONLY:
        raw = run_episode(il_policy(), grid, starts, goals, spec.max_steps, logger=logger, monitor=monitor)
    elif spec.method == SAFETY_ONLY_DEV_K2:
        safety = _SafetyRecorder(SafetyOnlyV0(hold_steps=SAFETY_HOLD_STEPS))
        raw = run_episode(il_policy(), grid, starts, goals, spec.max_steps,
                          logger=logger, monitor=monitor, safety=safety)
    elif spec.method == PURE_CBS_DEV_SINGLE_PLAN:
        raw = run_pure_cbs_episode(grid, starts, goals, spec.max_steps, planner=planner,
                                   path_validator=PathGuardValidator(), monitor=monitor, logger=logger)
    else:
        if spec.method == GLOBAL_HYBRID_DEV_FULL:
            config = dev_config(spec.max_steps)
        elif spec.method == GLOBAL_HYBRID_DEV_H3:
            config = dev_resume_config(H_EXEC_DEV)
        else:
            config = dev_rearm_config(H_EXEC_DEV)
        raw = run_global_hybrid_episode(il_policy(), grid, starts, goals, spec.max_steps,
                                        planner=planner, path_validator=PathGuardValidator(),
                                        config=config, monitor=monitor, logger=logger)

    if "termination" in raw:
        termination, source = raw["termination"], "RUNNER"
    elif raw.get("integrity_abort"):
        termination, source = "INTEGRITY_ABORT", "DERIVED_FROM_FLAGS"
    elif raw["all_at_goal"]:
        termination, source = "SUCCESS", "DERIVED_FROM_FLAGS"
    elif raw["timed_out"]:
        termination, source = "TIMEOUT", "DERIVED_FROM_FLAGS"
    else:
        raise RuntimeError(f"{run_id}: run_episode ended without a recognised termination flag")

    trajectory = raw["trajectory"]
    first_overlap = _first_overlap(trajectory)
    checked = spec.method != IL_ONLY
    if "action_sources" in raw:
        sources = list(raw["action_sources"])
    elif safety is not None:
        held = {t for t, _ in safety.holds}
        sources = ["SAFETY" if t in held else "IL" for t in range(1, raw["steps"] + 1)]
    else:
        sources = ["IL"] * raw["steps"]
    mr = monitor.result
    monitor_record = {
        "trigger_type": mr.trigger_type,
        "trigger_reason_code": mr.trigger_reason_code,
        "trigger_timestep": mr.trigger_timestep,
        "first_seen": mr.first_seen,
        "period": mr.period,
        "overlap_integrity_status": mr.overlap_integrity_status,
    }
    if spec.method == GLOBAL_HYBRID_DEV_H3_REARM:
        # Rearm method only: the fields above are the LAST armed detector's
        # Trigger; earlier ones are in trigger_history (other methods' record
        # keeps exactly the six keys above).
        monitor_record.update({
            "rearm_count": monitor.rearm_count,
            "trigger_history": raw["monitor_trigger_history"],
            "rearms": raw["monitor_rearms"],
            "triggers": raw["triggers"],
            "rearm_config": raw["rearm_config"],
        })

    dev_settings = dict(DEV_SETTINGS[spec.method])
    if spec.method == GLOBAL_HYBRID_DEV_H3_REARM:
        dev_settings["rearm_limits"] = dict(raw["rearm_config"])
    record = EpisodeRecord(
        method=spec.method,
        dev_settings=dev_settings,
        run_id=run_id,
        run_dir=str(run_dir),
        scenario=str(scenario),
        scenario_sha256=_sha256_file(scenario),
        grid_sha256=_sha256_grid(grid),
        checkpoint=str(ckpt) if spec.method in IL_METHODS else None,
        checkpoint_sha256=_sha256_file(ckpt) if spec.method in IL_METHODS else None,
        max_steps=spec.max_steps,
        cbs_timeout_sec=spec.cbs_timeout_sec if spec.method in CBS_METHODS else None,
        agent_ids=tuple(sorted(starts)),
        starts=dict(starts),
        goals=dict(goals),
        termination=termination,
        termination_source=source,
        stop_reason=raw.get("stop_reason"),
        all_at_goal=raw["all_at_goal"],
        timed_out=raw["timed_out"],
        steps=raw["steps"],
        trajectory=trajectory,
        actions=raw["actions"],
        action_sources=sources,
        overlap_any_step=first_overlap is not None,
        first_overlap_t=first_overlap,
        overlap_checked_at_runtime=checked,
        integrity_unhandled=(not checked) and first_overlap is not None,
        cbs_calls=list(raw.get("cbs_calls", [])),
        planner_calls=_planner_calls(planner),
        cbs_exec_trace=list(raw.get("cbs_exec_trace", [])),
        transitions=list(raw.get("transitions", [])),
        safety_holds=list(safety.holds) if safety is not None else [],
        safety_armed_t=safety.armed_t if safety is not None else None,
        monitor=monitor_record,
        environment=environment_info(spec.solver_root if spec.method in CBS_METHODS else None),
        raw=raw,
    )

    logger.write_steps()
    logger.write_summary(num_agents=len(goals), steps=raw["steps"], success=raw["all_at_goal"],
                         timed_out=raw["timed_out"], makespan=raw["steps"])
    logger.write_monitor(mr)
    payload = {k: v for k, v in asdict(record).items() if k != "raw"}
    with (run_dir / "result.json").open("x", encoding="utf-8") as fh:
        json.dump(_jsonable(payload), fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return record
