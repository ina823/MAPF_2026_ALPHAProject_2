"""Load MovingAI MAPF benchmark files into Alpha2 internal format.

MovingAI scenario coordinates use (x, y):
    x = column
    y = row

Alpha2 uses (row, col) internally, so coordinate conversion happens here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


GridLocation = tuple[int, int]


def parse_map(path: str | Path) -> np.ndarray:
    """Read a MovingAI .map file and return a 2-D grid.

    Alpha2 grid convention:
        0 = free cell
        1 = obstacle
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()

    height = int(
        next(line for line in lines if line.startswith("height")).split()[1]
    )
    width = int(
        next(line for line in lines if line.startswith("width")).split()[1]
    )

    map_start = lines.index("map") + 1

    grid = np.zeros((height, width), dtype=int)

    for row, line in enumerate(lines[map_start : map_start + height]):
        for col, char in enumerate(line[:width]):
            grid[row, col] = 0 if char == "." else 1

    return grid


def parse_scen(
    path: str | Path,
) -> list[tuple[GridLocation, GridLocation]]:
    """Read a MovingAI .scen file.

    Returns:
        [
            ((start_row, start_col), (goal_row, goal_col)),
            ...
        ]
    """
    path = Path(path)
    agents: list[tuple[GridLocation, GridLocation]] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("version") or not line.strip():
            continue

        parts = line.split("\t")

        start_x = int(parts[4])
        start_y = int(parts[5])
        goal_x = int(parts[6])
        goal_y = int(parts[7])

        start = (start_y, start_x)
        goal = (goal_y, goal_x)

        agents.append((start, goal))

    return agents


def select_agents(
    agents: list[tuple[GridLocation, GridLocation]],
    num_agents: int,
) -> list[tuple[GridLocation, GridLocation]]:
    """Select agents with unique starts and unique goals."""
    selected = []
    seen_starts = set()
    seen_goals = set()

    for start, goal in agents:
        if start in seen_starts:
            continue

        if goal in seen_goals:
            continue

        if start == goal:
            continue

        selected.append((start, goal))
        seen_starts.add(start)
        seen_goals.add(goal)

        if len(selected) == num_agents:
            break

    if len(selected) != num_agents:
        raise ValueError(
            f"Could only select {len(selected)} valid agents; "
            f"requested {num_agents}."
        )

    return selected


def load_movingai_instance(
    map_path: str | Path,
    scen_path: str | Path,
    num_agents: int,
) -> tuple[np.ndarray, list[GridLocation], list[GridLocation]]:
    """Return MovingAI instance as Alpha2 grid, starts, goals."""

    grid = parse_map(map_path)

    all_agents = parse_scen(scen_path)
    selected = select_agents(all_agents, num_agents)

    starts = [start for start, _ in selected]
    goals = [goal for _, goal in selected]

    return grid, starts, goals