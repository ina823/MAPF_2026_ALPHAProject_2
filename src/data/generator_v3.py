"""Deterministic warehouse-aisle generator v3 (DEV pilot, docs/warehouse_generator_v3_plan.md).

Separate from src/data/generator.py (v1) and generator_v2.py, which are imported
and never changed. Pure functions of (parameters, seed): parameters fix the
layout and the start/goal zones; ``seed`` only picks agent cells inside those
zones and the start-to-goal pairing, through ``numpy.random.default_rng``.

Coordinates are internal (row, col); scenario JSON stores [x, y] = [col, row]
(rowcol_to_xy from v1, read back by src.common.scenario_loader). Grid: 0 free,
1 wall (rack), int64.

Layout (docs/warehouse_generator_v3_plan.md §2.1)::

    rows 0 .. cross_w-1                top cross aisle   (row 0 = top goal row,
                                                           row cross_w-1 = top start row)
    rows cross_w .. cross_w+rack_len-1 racks (vertical wall columns) and aisles
    next cross_w rows                  bottom cross aisle
    last station_rows rows             picking/shipping station band
                                       (first station row = station start row,
                                        last row = station goal row)
    columns: aisle, rack, aisle, ..., rack, aisle
             W = (n_racks + 1) * aisle_w + n_racks * rack_w

Agents, in order: ``n_down`` from the top start row to the station goal row,
then ``n_up`` from the station start row to the top goal row. Goals sit on the
outer edge rows and starts on the inner rows, so an agent that arrives does not
park inside an aisle or in front of an agent that still has to leave.
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np

from src.data.generator import (
    Cell,
    GeneratedInstance,
    GeneratorError,
    _pick,
    npy_bytes,
    rowcol_to_xy,
    validate_instance,
)

GENERATOR_NAME = "alpha2_warehouse"
GENERATOR_VERSION = 3
STRUCTURE_TYPE = "warehouse"


def _positive_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise GeneratorError(f"{name} must be an int >= 1, got {value!r}")
    return value


def warehouse_layout(*, n_racks: int, rack_len: int, rack_w: int, aisle_w: int,
                     cross_w: int, station_rows: int) -> np.ndarray:
    """The grid only (no agents). Raises GeneratorError on unusable parameters."""
    for name, v in (("n_racks", n_racks), ("rack_len", rack_len), ("rack_w", rack_w),
                    ("aisle_w", aisle_w), ("cross_w", cross_w), ("station_rows", station_rows)):
        _positive_int(name, v)
    if cross_w < 2:
        raise GeneratorError("cross_w must be >= 2 (separate top start and top goal rows)")
    if station_rows < 2:
        raise GeneratorError("station_rows must be >= 2 (separate station start and goal rows)")
    width = (n_racks + 1) * aisle_w + n_racks * rack_w
    height = cross_w + rack_len + cross_w + station_rows
    grid = np.zeros((height, width), dtype=np.int64)
    col = aisle_w
    for _ in range(n_racks):
        grid[cross_w:cross_w + rack_len, col:col + rack_w] = 1
        col += rack_w + aisle_w
    return grid


def zone_rows(*, height: int, cross_w: int, station_rows: int) -> Dict[str, int]:
    return {"top_goal": 0, "top_start": cross_w - 1,
            "station_start": height - station_rows, "station_goal": height - 1}


def _row_cells(grid: np.ndarray, row: int) -> List[Cell]:
    return [(row, c) for c in range(grid.shape[1]) if grid[row, c] == 0]


def warehouse_aisles(*, n_racks: int, rack_len: int, rack_w: int, aisle_w: int, cross_w: int,
                     station_rows: int, n_down: int, n_up: int, seed: int) -> GeneratedInstance:
    """Warehouse with vertical racks; ``n_down`` top->station and ``n_up`` station->top agents.

    RNG call order is fixed (part of the reproducibility contract):
    down starts, down goals, up starts, up goals, down pairing, up pairing.
    """
    for name, v in (("n_down", n_down), ("n_up", n_up)):
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            raise GeneratorError(f"{name} must be an int >= 0, got {v!r}")
    if n_down + n_up == 0:
        raise GeneratorError("at least one agent is required")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise GeneratorError(f"seed must be an int, got {seed!r}")
    grid = warehouse_layout(n_racks=n_racks, rack_len=rack_len, rack_w=rack_w, aisle_w=aisle_w,
                            cross_w=cross_w, station_rows=station_rows)
    rows = zone_rows(height=grid.shape[0], cross_w=cross_w, station_rows=station_rows)
    rng = np.random.default_rng(seed)
    d_starts = _pick(rng, _row_cells(grid, rows["top_start"]), n_down)
    d_goals = _pick(rng, _row_cells(grid, rows["station_goal"]), n_down)
    u_starts = _pick(rng, _row_cells(grid, rows["station_start"]), n_up)
    u_goals = _pick(rng, _row_cells(grid, rows["top_goal"]), n_up)
    d_goals = [d_goals[int(i)] for i in rng.permutation(n_down)] if n_down else []
    u_goals = [u_goals[int(i)] for i in rng.permutation(n_up)] if n_up else []
    starts = d_starts + u_starts
    goals = d_goals + u_goals
    desc = {"structure": "warehouse_aisles", "n_racks": n_racks, "rack_len": rack_len,
            "rack_w": rack_w, "aisle_w": aisle_w, "cross_w": cross_w, "station_rows": station_rows,
            "n_down": n_down, "n_up": n_up, "zone_rows": rows,
            "agent_order": ["top->station"] * n_down + ["station->top"] * n_up}
    grid = np.ascontiguousarray(grid)
    validate_instance(grid, starts, goals)
    grid.setflags(write=False)
    return GeneratedInstance(grid, tuple(starts), tuple(goals), STRUCTURE_TYPE,
                             dict(desc, generator=GENERATOR_NAME, generator_version=GENERATOR_VERSION),
                             seed)


BUILDERS = {"warehouse_aisles": warehouse_aisles}


def build(builder: str, params: Dict, seed: int) -> GeneratedInstance:
    if builder not in BUILDERS:
        raise GeneratorError(f"unknown builder {builder!r}")
    return BUILDERS[builder](seed=seed, **params)


def scenario_files(inst: GeneratedInstance, slug: str, *, code_sha256: str, base_generator_sha256: str,
                   prereg_sha256: str, candidate_id: str) -> Tuple[bytes, bytes]:
    """(map .npy bytes, scenario JSON bytes) in the scenarios/pilot_v2 file format,
    with generator-v3 provenance in the ``generator`` block."""
    scen = {
        "scenario_id": f"scen_{slug}",
        "map_id": slug,
        "map_file": f"map_{slug}.npy",
        "agents": [{"agent_id": i, "start": rowcol_to_xy(s), "goal": rowcol_to_xy(g)}
                   for i, (s, g) in enumerate(zip(inst.starts, inst.goals))],
        "generator": {"name": GENERATOR_NAME, "version": GENERATOR_VERSION, "seed": inst.seed,
                      "code_sha256": code_sha256, "base_generator_sha256": base_generator_sha256,
                      "prereg_sha256": prereg_sha256, "candidate_id": candidate_id,
                      "descriptor": inst.descriptor},
    }
    return npy_bytes(inst.grid), (json.dumps(scen, indent=2, sort_keys=True) + "\n").encode("utf-8")
