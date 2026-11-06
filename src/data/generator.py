"""Deterministic structured map/instance generator (GENERATOR_VERSION 1).

Pure functions of (parameters, seed): the same call always returns the same
grid, starts, goals and descriptor, and ``scenario_files`` the same bytes.
``seed`` only chooses agent cells inside fixed zones via
``numpy.random.default_rng(seed)``; the layout is fully given by parameters.

Coordinates: internal (row, col). Scenario JSON is written as [x, y] =
[col, row] and read back only through src.common.scenario_loader.

Grid: 0 = free, 1 = wall, int64 (same as the existing controlled maps).
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.il.nav_hint import bfs_dist

GENERATOR_NAME = "alpha2_structured"
GENERATOR_VERSION = 1

Cell = Tuple[int, int]


class GeneratorError(ValueError):
    """Parameters cannot produce a valid instance."""


@dataclass(frozen=True)
class GeneratedInstance:
    grid: np.ndarray
    starts: Tuple[Cell, ...]
    goals: Tuple[Cell, ...]
    structure_type: str
    descriptor: Dict
    seed: int


def _pick(rng: np.random.Generator, cells: Sequence[Cell], n: int) -> List[Cell]:
    if n > len(cells):
        raise GeneratorError(f"zone has {len(cells)} cells, {n} needed")
    idx = rng.choice(len(cells), size=n, replace=False)
    return [cells[int(i)] for i in sorted(idx.tolist())]


def validate_instance(grid: np.ndarray, starts: Sequence[Cell], goals: Sequence[Cell]) -> None:
    """Walls, bounds, distinct starts/goals, and every goal reachable from its start."""
    h, w = grid.shape
    if not set(np.unique(grid).tolist()) <= {0, 1}:
        raise GeneratorError("grid must be 0/1")
    if len(starts) != len(goals) or not starts:
        raise GeneratorError("starts/goals empty or of different length")
    for kind, cells in (("start", starts), ("goal", goals)):
        if len(set(cells)) != len(cells):
            raise GeneratorError(f"duplicate {kind}")
        for r, c in cells:
            if not (0 <= r < h and 0 <= c < w) or grid[r, c] != 0:
                raise GeneratorError(f"{kind} {(r, c)} outside the grid or on a wall")
    for i, (s, g) in enumerate(zip(starts, goals)):
        if bfs_dist(grid, g)[s] < 0:
            raise GeneratorError(f"agent {i}: goal {g} unreachable from start {s}")


def _finish(grid, starts, goals, structure_type, descriptor, seed) -> GeneratedInstance:
    grid = np.ascontiguousarray(grid.astype(np.int64))
    validate_instance(grid, starts, goals)
    grid.setflags(write=False)
    return GeneratedInstance(grid, tuple(starts), tuple(goals), structure_type,
                             dict(descriptor, generator=GENERATOR_NAME,
                                  generator_version=GENERATOR_VERSION), seed)


# -- structures ---------------------------------------------------------------

def corridor_bays(*, length: int, bays_up: Sequence[int], bays_down: Sequence[int],
                  n_east: int, n_west: int, seed: int) -> GeneratedInstance:
    """1-wide horizontal corridor with single-cell bays above/below it.

    East-bound agents start in the west end zone and end in the east end zone
    (and vice versa); within each direction the order along the corridor is
    kept (sorted starts paired with sorted goals), so only opposing traffic
    has to use the bays."""
    rows = 1 + bool(bays_up) + bool(bays_down)
    cr = 1 if bays_up else 0
    grid = np.ones((rows, length), dtype=np.int64)
    grid[cr, :] = 0
    for c in bays_up:
        grid[cr - 1, c] = 0
    for c in bays_down:
        grid[cr + 1, c] = 0
    k = n_east + n_west
    west = [(cr, c) for c in range(k)]
    east = [(cr, c) for c in range(length - k, length)]
    rng = np.random.default_rng(seed)
    e_starts, e_goals = _pick(rng, west, n_east), _pick(rng, east, n_east)
    w_starts, w_goals = _pick(rng, east, n_west), _pick(rng, west, n_west)
    starts = e_starts + w_starts
    goals = e_goals + w_goals
    desc = {"structure": "corridor_bays", "length": length, "bays_up": list(bays_up),
            "bays_down": list(bays_down), "n_east": n_east, "n_west": n_west}
    return _finish(grid, starts, goals, "corridor", desc, seed)


def rooms_door(*, room_h: int, room_w: int, door_len: int, door_row: int,
               n_east: int, n_west: int, seed: int) -> GeneratedInstance:
    """Two open rooms joined by a 1-wide door passage of ``door_len`` cells."""
    width = 2 * room_w + door_len
    grid = np.zeros((room_h, width), dtype=np.int64)
    grid[:, room_w:room_w + door_len] = 1
    grid[door_row, room_w:room_w + door_len] = 0
    left = [(r, c) for c in (0, 1) for r in range(room_h)]
    right = [(r, c) for c in (width - 2, width - 1) for r in range(room_h)]
    rng = np.random.default_rng(seed)
    e_starts, w_goals = _pick(rng, left, n_east), _pick(rng, left, n_west)
    w_starts, e_goals = _pick(rng, right, n_west), _pick(rng, right, n_east)
    starts = e_starts + w_starts
    goals = e_goals + w_goals
    desc = {"structure": "rooms_door", "room_h": room_h, "room_w": room_w, "door_len": door_len,
            "door_row": door_row, "n_east": n_east, "n_west": n_west}
    return _finish(grid, starts, goals, "bottleneck", desc, seed)


def crossing(*, height: int, width: int, corridor_rows: Sequence[int], corridor_cols: Sequence[int],
             seed: int) -> GeneratedInstance:
    """Full-length 1-wide corridors on the given rows and columns.

    Agents: one south-bound per column corridor, then alternately east-/west-
    bound per row corridor. Each starts in the first two cells of its corridor
    and ends in the last two (seed picks which)."""
    grid = np.ones((height, width), dtype=np.int64)
    for r in corridor_rows:
        grid[r, :] = 0
    for c in corridor_cols:
        grid[:, c] = 0
    rng = np.random.default_rng(seed)
    starts, goals, flows = [], [], []
    for c in corridor_cols:
        starts += _pick(rng, [(0, c), (1, c)], 1)
        goals += _pick(rng, [(height - 2, c), (height - 1, c)], 1)
        flows.append(f"S@col{c}")
    for i, r in enumerate(corridor_rows):
        a, b = [(r, 0), (r, 1)], [(r, width - 2), (r, width - 1)]
        if i % 2:
            a, b = b, a
        starts += _pick(rng, a, 1)
        goals += _pick(rng, b, 1)
        flows.append(("W" if i % 2 else "E") + f"@row{r}")
    desc = {"structure": "crossing", "height": height, "width": width,
            "corridor_rows": list(corridor_rows), "corridor_cols": list(corridor_cols), "flows": flows}
    return _finish(grid, starts, goals, "intersection", desc, seed)


# -- files --------------------------------------------------------------------

def rowcol_to_xy(cell: Cell) -> List[int]:
    """Internal (row, col) -> scenario JSON [x, y] (inverse of scenario_loader.xy_to_rowcol)."""
    return [int(cell[1]), int(cell[0])]


def npy_bytes(grid: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(grid.astype(np.int64)), allow_pickle=False)
    return buf.getvalue()


def scenario_files(inst: GeneratedInstance, slug: str,
                   generator_code_sha256: str) -> Tuple[bytes, bytes]:
    """(map .npy bytes, scenario JSON bytes) in the scenarios/controlled format.

    ``map_file`` is ``map_<slug>.npy`` next to the scenario JSON (resolved by
    scenario_loader against the scenario's own directory)."""
    scen = {
        "scenario_id": f"scen_{slug}",
        "map_id": slug,
        "map_file": f"map_{slug}.npy",
        "agents": [{"agent_id": i, "start": rowcol_to_xy(s), "goal": rowcol_to_xy(g)}
                   for i, (s, g) in enumerate(zip(inst.starts, inst.goals))],
        "generator": {"name": GENERATOR_NAME, "version": GENERATOR_VERSION, "seed": inst.seed,
                      "code_sha256": generator_code_sha256, "descriptor": inst.descriptor},
    }
    return npy_bytes(inst.grid), (json.dumps(scen, indent=2, sort_keys=True) + "\n").encode("utf-8")
