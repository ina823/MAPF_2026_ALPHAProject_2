"""Deterministic structured generator v2: three map structures for PILOT v2 candidates.

Separate from src/data/generator.py, whose code sha256 is written into the
PILOT v1 scenario files; that module is imported here and never changed.

Pure functions of (parameters, seed), as in v1: parameters fix the layout and
``seed`` only picks agent cells inside fixed zones through
``numpy.random.default_rng``. Coordinates are internal (row, col); scenario
JSON stores [x, y] = [col, row]. Grid: 0 = free, 1 = wall, int64.
"""

from __future__ import annotations

import json
from typing import Dict, List, Mapping, Sequence, Tuple

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

GENERATOR_NAME = "alpha2_structured_v2"
GENERATOR_VERSION = 2


def _finish(grid, starts, goals, structure_type, descriptor, seed) -> GeneratedInstance:
    grid = np.ascontiguousarray(grid.astype(np.int64))
    validate_instance(grid, starts, goals)
    grid.setflags(write=False)
    return GeneratedInstance(grid, tuple(starts), tuple(goals), structure_type,
                             dict(descriptor, generator=GENERATOR_NAME,
                                  generator_version=GENERATOR_VERSION), seed)


def _zone(grid: np.ndarray, spec: Mapping) -> List[Cell]:
    """Cells of an inclusive {"rows": [r0, r1], "cols": [c0, c1]} range, row by row."""
    (r0, r1), (c0, c1) = spec["rows"], spec["cols"]
    cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    h, w = grid.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w) or grid[r, c] != 0:
            raise GeneratorError(f"zone cell {(r, c)} is outside the grid or on a wall")
    return cells


def staggered_blocks(*, height: int, width: int, block_h: int, block_w: int,
                     block_tops: Sequence[Sequence[int]], n_east: int, zones: Mapping,
                     seed: int) -> GeneratedInstance:
    """Open room with staggered rectangular blocks (no single-cell pillars).

    Agents: ``n_east`` from the west zone to the east zone, then one north to
    south and one south to north."""
    if min(block_h, block_w) < 2:
        raise GeneratorError("blocks must be at least 2x2")
    grid = np.zeros((height, width), dtype=np.int64)
    for r, c in block_tops:
        if r < 0 or c < 0 or r + block_h > height or c + block_w > width:
            raise GeneratorError(f"block at {(r, c)} does not fit the grid")
        grid[r:r + block_h, c:c + block_w] = 1
    rng = np.random.default_rng(seed)
    west = _pick(rng, _zone(grid, zones["west"]), n_east)
    east = _pick(rng, _zone(grid, zones["east"]), n_east)
    north = _pick(rng, _zone(grid, zones["north"]), 2)
    south = _pick(rng, _zone(grid, zones["south"]), 2)
    starts = west + [north[0], south[1]]
    goals = east + [south[0], north[1]]
    desc = {"structure": "staggered_blocks", "height": height, "width": width,
            "block_h": block_h, "block_w": block_w, "block_tops": [list(b) for b in block_tops],
            "n_east": n_east, "zones": zones}
    return _finish(grid, starts, goals, "open_obstacles", desc, seed)


def loop_corridor(*, height: int, width: int, band: int, zones: Mapping, seed: int) -> GeneratedInstance:
    """Free loop ``band`` cells wide around one central wall block.

    Agents, in order: north to south, south to north, west to east, east to west."""
    if band < 1 or 2 * band >= min(height, width):
        raise GeneratorError(f"band {band} leaves no central block in {height}x{width}")
    grid = np.zeros((height, width), dtype=np.int64)
    grid[band:height - band, band:width - band] = 1
    rng = np.random.default_rng(seed)
    north = _pick(rng, _zone(grid, zones["north"]), 2)
    south = _pick(rng, _zone(grid, zones["south"]), 2)
    west = _pick(rng, _zone(grid, zones["west"]), 2)
    east = _pick(rng, _zone(grid, zones["east"]), 2)
    starts = [north[0], south[1], west[0], east[1]]
    goals = [south[0], north[1], east[0], west[1]]
    desc = {"structure": "loop_corridor", "height": height, "width": width, "band": band, "zones": zones}
    return _finish(grid, starts, goals, "loop", desc, seed)


def hub_spokes(*, hub_size: int, arm_len: int, chamber_depth: int, chamber_width: int,
               n_south: int, n_east: int, seed: int) -> GeneratedInstance:
    """Square central hub with four width-1 arms, each ending in a chamber.

    Agents, in order: ``n_south`` north chamber to south chamber, ``n_east`` west
    chamber to east chamber, one from the middle of the south arm to the north
    chamber, one from the middle of the east arm to the west chamber. Chamber
    goals of the last two avoid the chamber starts of the first groups."""
    if hub_size < 1 or hub_size % 2 == 0 or chamber_width % 2 == 0:
        raise GeneratorError("hub_size and chamber_width must be odd")
    half_hub, half_ch = hub_size // 2, chamber_width // 2
    centre = chamber_depth + arm_len + half_hub
    size = 2 * centre + 1
    last = size - 1
    grid = np.ones((size, size), dtype=np.int64)
    grid[centre - half_hub:centre + half_hub + 1, centre - half_hub:centre + half_hub + 1] = 0
    grid[:, centre] = 0
    grid[centre, :] = 0
    ch_lo, ch_hi = centre - half_ch, centre + half_ch
    north_ch = {"rows": [0, chamber_depth - 1], "cols": [ch_lo, ch_hi]}
    south_ch = {"rows": [last - chamber_depth + 1, last], "cols": [ch_lo, ch_hi]}
    west_ch = {"rows": [ch_lo, ch_hi], "cols": [0, chamber_depth - 1]}
    east_ch = {"rows": [ch_lo, ch_hi], "cols": [last - chamber_depth + 1, last]}
    for z in (north_ch, south_ch, west_ch, east_ch):
        (r0, r1), (c0, c1) = z["rows"], z["cols"]
        grid[r0:r1 + 1, c0:c1 + 1] = 0

    arm_middle = centre + half_hub + 1 + arm_len // 2
    rng = np.random.default_rng(seed)
    north_cells, west_cells = _zone(grid, north_ch), _zone(grid, west_ch)
    s_starts = _pick(rng, north_cells, n_south)
    s_goals = _pick(rng, _zone(grid, south_ch), n_south)
    e_starts = _pick(rng, west_cells, n_east)
    e_goals = _pick(rng, _zone(grid, east_ch), n_east)
    n_goal = _pick(rng, [c for c in north_cells if c not in s_starts], 1)
    w_goal = _pick(rng, [c for c in west_cells if c not in e_starts], 1)
    starts = s_starts + e_starts + [(arm_middle, centre), (centre, arm_middle)]
    goals = s_goals + e_goals + n_goal + w_goal
    desc = {"structure": "hub_spokes", "hub_size": hub_size, "arm_len": arm_len,
            "chamber_depth": chamber_depth, "chamber_width": chamber_width,
            "n_south": n_south, "n_east": n_east}
    return _finish(grid, starts, goals, "hub_spokes", desc, seed)


BUILDERS = {"staggered_blocks": staggered_blocks, "loop_corridor": loop_corridor, "hub_spokes": hub_spokes}


def build(builder: str, params: Mapping, seed: int) -> GeneratedInstance:
    if builder not in BUILDERS:
        raise GeneratorError(f"unknown builder {builder!r}")
    return BUILDERS[builder](seed=seed, **params)


def scenario_files(inst: GeneratedInstance, slug: str, *, code_sha256: str, base_generator_sha256: str,
                   prereg_sha256: str, candidate_id: str) -> Tuple[bytes, bytes]:
    """(map .npy bytes, scenario JSON bytes) in the scenarios/pilot_v1 file format,
    with the generator-v2 provenance in the ``generator`` block."""
    scen: Dict = {
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
