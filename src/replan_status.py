
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from src.replanner import replan
from src.cbs_adapter import (
    AgentPath,
    GridLocation,
    CBSTimeoutError,
    CBSNoSolutionError,
)
from src.path_guard import InvalidPathError


@dataclass
class ReplanResult:
    status: str
    paths: dict[int, AgentPath] | None
    latency_sec: float
    error: str | None = None


def replan_with_status(
    current_positions: Sequence[GridLocation],
    goals: Sequence[GridLocation],
    grid_map: Any,
    solver_root: str | Path | None = None,
    timeout_sec: int = 30,
) -> ReplanResult:

    start_time = perf_counter()

    try:
        paths = replan(
            current_positions=current_positions,
            goals=goals,
            grid_map=grid_map,
            solver_root=solver_root,
            timeout_sec=timeout_sec,
        )

        return ReplanResult(
            status="success",
            paths=paths,
            latency_sec=perf_counter() - start_time,
        )

    except (
        CBSTimeoutError,
        CBSNoSolutionError,
        InvalidPathError,
        ValueError,
        RuntimeError,
        OSError,
    ) as exc:

        if isinstance(exc, CBSTimeoutError):
            status = "timeout"

        elif isinstance(exc, CBSNoSolutionError):
            status = "no_solution"

        elif isinstance(exc, InvalidPathError):
            status = "invalid_path"

        elif isinstance(exc, ValueError):
            status = "input_error"

        else:
            status = "solver_error"

        return ReplanResult(
            status=status,
            paths=None,
            latency_sec=perf_counter() - start_time,
            error=str(exc),
        )
