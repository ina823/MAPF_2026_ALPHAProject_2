"""Real CBS planner and Path Guard validator for the Global Hybrid Runner.

Wraps the CBS branch's src/cbs_runtime_adapter.run_cbs_runtime() and
src/path_guard.validate_path_guard() without modifying them.

Agent-id contract (docs/global_hybrid_runner_v0.md):
  * CbsRuntimePlanner.paths_keying == "agent_id": the runner hands it
    {agent_id: S_t} / {agent_id: goal} dicts and gets paths keyed by the same
    agent ids back. The only index <-> agent_id conversion is the one inside
    run_cbs_runtime(); neither this module nor the runner/boundary converts.
  * PathGuardValidator runs the real Path Guard once more at the boundary, on
    exactly the paths that will be executed. It is not ``test_only``.

Work directory: src/replanner.py writes its solver YAML files to the fixed
relative path ``outputs/paths/replan`` (resolved against the cwd) and
overwrites them on every call. Each call therefore runs with the cwd set to a
fresh ``<work_root>/cbs_call_NN`` directory; an existing directory is never
reused. solver_root is resolved to an absolute path first. The cwd switch is
process-wide, so planners must not be called from several threads at once.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Dict, List, Mapping

from src.cbs_runtime_adapter import CBSRuntimeResult, run_cbs_runtime
from src.path_guard import validate_path_guard

# Relative work_dir hard-coded in src/replanner.py.
REPLANNER_WORK_DIR = Path("outputs/paths/replan")


class CbsRuntimePlanner:
    """``planner(current_positions, goals, grid)`` with agent-id-keyed dicts."""

    paths_keying = "agent_id"

    def __init__(self, *, solver_root, work_root, timeout_sec: int = 30):
        self.solver_root = Path(solver_root).resolve()
        if not (self.solver_root / "cbs.py").is_file():
            raise FileNotFoundError(f"atb033 cbs.py not found under {self.solver_root}")
        self.work_root = Path(work_root).resolve()
        self.timeout_sec = timeout_sec
        self.calls: List[Dict[str, Any]] = []

    def __call__(
        self,
        current_positions: Mapping[int, tuple],
        goals: Mapping[int, tuple],
        grid,
    ) -> CBSRuntimeResult:
        work_dir = self.work_root / f"cbs_call_{len(self.calls) + 1:02d}"
        work_dir.mkdir(parents=True, exist_ok=False)
        record = {
            "work_dir": work_dir,
            "artifacts_dir": work_dir / REPLANNER_WORK_DIR,
            "solver_root": self.solver_root,
            "timeout_sec": self.timeout_sec,
        }
        self.calls.append(record)
        with contextlib.chdir(work_dir):
            result = run_cbs_runtime(
                current_positions,
                goals,
                grid,
                timeout_sec=self.timeout_sec,
                solver_root=self.solver_root,
            )
        record.update(
            status=result.status,
            latency_sec=result.latency_sec,
            error=result.error,
            paths=result.paths,
        )
        return result


class PathGuardValidator:
    """``path_validator(grid, starts, goals, paths) -> (ok, reason)`` backed by
    the real src/path_guard.validate_path_guard()."""

    def __call__(self, grid, starts, goals, paths):
        try:
            validate_path_guard(
                paths=paths,
                current_positions=starts,
                goals=goals,
                grid_map=grid,
            )
        except (ValueError, TypeError, KeyError, IndexError) as exc:  # InvalidPathError is a ValueError
            return False, f"{type(exc).__name__}: {exc}"
        return True, ""
