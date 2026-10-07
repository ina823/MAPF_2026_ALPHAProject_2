"""Adapter for running the atb033 CBS solver via subprocess.

The project uses ``(row, col)`` coordinates internally. The atb033 solver uses
``[x, y]`` coordinates, so this module converts coordinates only at the solver
boundary.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised by environment setup
    raise ImportError("CBSAdapter requires PyYAML. Install it before running CBS.") from exc

GridLocation = tuple[int, int]
AgentPath = list[GridLocation]


@dataclass(frozen=True)
class CBSAdapterConfig:
    """Configuration for the subprocess-based atb033 CBS adapter."""

    solver_root: Path | None = None
    work_dir: Path = Path("outputs/paths/atb033_example")
    timeout_sec: int = 30
    python_executable: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CBSRunArtifacts:
    """Paths and logs from the most recent CBS subprocess run."""

    input_yaml_path: Path
    output_yaml_path: Path
    stdout: str
    stderr: str

class CBSTimeoutError(RuntimeError):
    """CBS solver exceeded the configured time limit."""
    pass


class CBSNoSolutionError(RuntimeError):
    """CBS solver finished but could not find a solution."""
    pass

def internal_to_atb033(location: GridLocation) -> list[int]:
    """Convert ``(row, col)`` to atb033 ``[x, y]``."""
    row, col = location
    return [int(col), int(row)]


def atb033_to_internal(step: Mapping[str, Any]) -> GridLocation:
    """Convert an atb033 schedule step to internal ``(row, col)``."""
    return (int(step["y"]), int(step["x"]))


def create_atb033_input(
    grid_map: Any,
    starts: Sequence[GridLocation],
    goals: Sequence[GridLocation],
    save_path: str | Path,
) -> Path:
    """Write an atb033-compatible input YAML file and return its path."""
    height, width = _grid_shape(grid_map)
    _validate_inputs(grid_map, starts, goals)

    obstacles = [
        internal_to_atb033((row, col))
        for row in range(height)
        for col in range(width)
        if _grid_value(grid_map, row, col) == 1
    ]
    agents = [
        {
            "start": internal_to_atb033(start),
            "goal": internal_to_atb033(goal),
            "name": f"agent{agent_id}",
        }
        for agent_id, (start, goal) in enumerate(zip(starts, goals))
    ]
    payload = {
        "agents": agents,
        "map": {
            "dimensions": [width, height],
            "obstacles": obstacles,
        },
    }

    path = Path(save_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.dump(payload, handle, sort_keys=False)
    return path


def run_atb033_cbs(
    solver_root: str | Path,
    input_yaml_path: str | Path,
    output_yaml_path: str | Path,
    timeout_sec: int = 30,
    python_executable: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run atb033 ``cbs.py`` as a subprocess."""
    solver_root_path = Path(solver_root).resolve()
    input_path = Path(input_yaml_path).resolve()
    output_path = Path(output_yaml_path).resolve()
    cbs_script = solver_root_path / "cbs.py"

    if not cbs_script.is_file():
        raise FileNotFoundError(f"atb033 cbs.py not found: {cbs_script}")
    if not input_path.is_file():
        raise FileNotFoundError(f"CBS input YAML not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    # atb033 CBS는 obstacle을 (x, y) tuple과 비교하므로
    # solver에 전달할 YAML에서는 obstacle을 tuple로 변환한다.
    with input_path.open("r", encoding="utf-8") as handle:
        solver_payload = yaml.safe_load(handle) or {}

    solver_payload["map"]["obstacles"] = [
        tuple(obstacle)
        for obstacle in solver_payload["map"]["obstacles"]
    ]

    solver_input_path = input_path.with_name("solver_input.yaml")

    with solver_input_path.open("w", encoding="utf-8") as handle:   
        yaml.dump(solver_payload, handle, sort_keys=False)

    command = [
        python_executable or sys.executable,
        "cbs.py",
        str(solver_input_path),
        str(output_path),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=solver_root_path,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise CBSTimeoutError(
            "atb033 CBS timed out after "
            f"{timeout_sec}s\nstdout:\n{exc.stdout or ''}\nstderr:\n{exc.stderr or ''}"
        ) from exc

    if result.returncode != 0:
        raise RuntimeError(
            "atb033 CBS subprocess failed\n"
            f"command: {' '.join(command)}\n"
            f"cwd: {solver_root_path}\n"
            f"returncode: {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    if not output_path.is_file():
        if "solution not found" in (result.stdout or "").lower():
            raise CBSNoSolutionError(
                "atb033 CBS finished but no solution was found."
            )
        raise RuntimeError(
            "atb033 CBS did not create output YAML.\n "
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def load_atb033_output(output_yaml_path: str | Path) -> dict[str, Any]:
    """Load atb033 output YAML."""
    path = Path(output_yaml_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if "schedule" not in data:
        raise ValueError(f"Missing 'schedule' in atb033 output: {path}")
    return data


def convert_schedule_to_internal_paths(
    schedule: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[int, AgentPath]:
    """Convert atb033 schedule output to internal ``{agent_id: [(row, col)]}``."""
    paths: dict[int, AgentPath] = {}
    for agent_name, steps in schedule.items():
        match = re.fullmatch(r"agent(\d+)", agent_name)
        if not match:
            raise ValueError(f"Unexpected agent name in schedule: {agent_name}")
        agent_id = int(match.group(1))
        sorted_steps = sorted(steps, key=lambda step: int(step["t"]))
        paths[agent_id] = [atb033_to_internal(step) for step in sorted_steps]
    return dict(sorted(paths.items()))


def pad_paths(paths: Mapping[int, AgentPath]) -> dict[int, AgentPath]:
    """Pad all paths to the same length by repeating each final location."""
    if not paths:
        return {}
    makespan = max(len(path) for path in paths.values())
    padded: dict[int, AgentPath] = {}
    for agent_id, path in paths.items():
        if not path:
            raise ValueError(f"Path for agent {agent_id} is empty.")
        padded[agent_id] = list(path) + [path[-1]] * (makespan - len(path))
    return padded


def save_internal_paths_json(
    paths: Mapping[int, Sequence[GridLocation]],
    save_path: str | Path,
) -> Path:
    """Save internal paths in a JSON format that simulator code can consume."""
    payload = {
        "coordinate_format": "row_col",
        "paths": {
            str(agent_id): [[int(row), int(col)] for row, col in path]
            for agent_id, path in sorted(paths.items())
        },
    }
    path = Path(save_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return path


class CBSAdapter:
    """Run atb033 CBS through YAML files and convert paths back to project format."""

    def __init__(self, config: CBSAdapterConfig | None = None) -> None:
        self.config = config or CBSAdapterConfig()
        self.last_run: CBSRunArtifacts | None = None

    def plan(
        self,
        starts: Sequence[GridLocation],
        goals: Sequence[GridLocation],
        grid: Any,
    ) -> dict[int, AgentPath]:
        """Run CBS and return padded internal paths."""
        _validate_inputs(grid, starts, goals)

        solver_root = _resolve_solver_root(self.config.solver_root)
        work_dir = Path(self.config.work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        input_yaml_path = work_dir / "input.yaml"
        output_yaml_path = work_dir / "output.yaml"

        create_atb033_input(grid, starts, goals, input_yaml_path)
        result = run_atb033_cbs(
            solver_root=solver_root,
            input_yaml_path=input_yaml_path,
            output_yaml_path=output_yaml_path,
            timeout_sec=self.config.timeout_sec,
            python_executable=self.config.python_executable,
        )
        output = load_atb033_output(output_yaml_path)
        paths = convert_schedule_to_internal_paths(output["schedule"])
        padded_paths = pad_paths(paths)

        self.last_run = CBSRunArtifacts(
            input_yaml_path=input_yaml_path,
            output_yaml_path=output_yaml_path,
            stdout=result.stdout,
            stderr=result.stderr,
        )
        return padded_paths


def _resolve_solver_root(configured_solver_root: Path | None) -> Path:
    if configured_solver_root is not None:
        return Path(configured_solver_root).resolve()

    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        project_root / "third_party/multi_agent_path_planning/centralized/cbs",
        project_root.parent / "third_party/multi_agent_path_planning/centralized/cbs",
    ]
    for candidate in candidates:
        if (candidate / "cbs.py").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find atb033 CBS solver. Expected cbs.py under one of: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _grid_shape(grid_map: Any) -> tuple[int, int]:
    if hasattr(grid_map, "shape"):
        shape = grid_map.shape
        if len(shape) != 2:
            raise ValueError(f"grid_map must be 2-D, got shape {shape}.")
        return int(shape[0]), int(shape[1])

    height = len(grid_map)
    if height == 0:
        raise ValueError("grid_map must not be empty.")
    width = len(grid_map[0])
    if width == 0:
        raise ValueError("grid_map rows must not be empty.")
    if any(len(row) != width for row in grid_map):
        raise ValueError("grid_map rows must all have the same width.")
    return height, width


def _grid_value(grid_map: Any, row: int, col: int) -> int:
    return int(grid_map[row][col])


def _validate_inputs(
    grid_map: Any,
    starts: Sequence[GridLocation],
    goals: Sequence[GridLocation],
) -> None:
    height, width = _grid_shape(grid_map)
    if len(starts) != len(goals):
        raise ValueError(
            f"starts and goals must have the same length: {len(starts)} != {len(goals)}"
        )
    if not starts:
        raise ValueError("At least one agent is required.")

    for label, locations in (("start", starts), ("goal", goals)):
        for agent_id, (row, col) in enumerate(locations):
            if not (0 <= row < height and 0 <= col < width):
                raise ValueError(
                    f"Agent {agent_id} {label} {(row, col)} is outside grid "
                    f"with shape {(height, width)}."
                )
            if _grid_value(grid_map, row, col) == 1:
                raise ValueError(f"Agent {agent_id} {label} {(row, col)} is blocked.")
