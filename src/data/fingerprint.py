"""Scenario fingerprints (schema FP_SCHEMA = 1). Pure functions, numpy only.

Coordinates are the project-internal ``(row, col)`` everywhere in this module.
Scenario JSON ``[x, y]`` is converted only by ``src.common.scenario_loader``
(``xy_to_rowcol``); this module never converts coordinates itself.

Normalisation levels (docs/fingerprint_registry_v1.md, section 2):

  N1 grid      the 0/1 grid as given (any integer/bool dtype, .npy header
               ignored) -> ``grid_fp``
  N2 trimmed   outer rows/columns that are entirely wall removed. The
               simulator, the IL local view and the BFS hint all treat
               "outside the map" like a wall, so this is the same problem
               for them -- but coordinates shift, so CBS tie-breaking (which
               hashes coordinates) may differ -> ``grid_trim_fp``
  N3 canonical the trimmed grid under the 8 rotations/reflections; the
               smallest key wins -> ``map_canon_fp``. Used for FAMILY
               relations only: the IL policy is not rotation-equivariant
               (fixed up/down/left/right actions, anisotropic goal_std), so a
               rotated instance is never treated as the same instance.

Agents are fingerprinted twice at every level: in agent-id order
(``*_ordered_fp``, id = list index, as the scenario loader assigns it) and as
an id-free set of (start, goal) pairs (``*_set_fp``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.common.scenario_loader import (
    PROJECT_ROOT,
    _resolve_map_file,
    load_map_generator_scenario,
)

FP_SCHEMA = 1
HASH = "sha256"

Cell = Tuple[int, int]
# (k quarter turns counter-clockwise as np.rot90, then optional left-right flip)
TRANSFORMS: Tuple[Tuple[int, bool], ...] = tuple((k, f) for f in (False, True) for k in range(4))


class FingerprintError(ValueError):
    """The scenario cannot be fingerprinted (malformed grid or agents)."""


# -- hashing helpers ----------------------------------------------------------

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# -- grid ---------------------------------------------------------------------

def normalize_grid(grid) -> np.ndarray:
    """Return a read-only uint8 2-D copy with values in {0, 1} (0 free, 1 wall)."""
    arr = np.asarray(grid)
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise FingerprintError(f"grid must be a non-empty 2-D array, got shape {arr.shape}")
    if not (np.issubdtype(arr.dtype, np.integer) or arr.dtype == np.bool_):
        raise FingerprintError(f"grid dtype must be integer or bool, got {arr.dtype}")
    values = set(np.unique(arr).tolist())
    if not values <= {0, 1}:
        raise FingerprintError(f"grid values must be 0/1, got {sorted(values)}")
    out = np.ascontiguousarray(arr.astype(np.uint8))
    out.setflags(write=False)
    return out


def grid_rows(grid: np.ndarray) -> List[str]:
    return ["".join("1" if v else "0" for v in row) for row in grid.tolist()]


def grid_from_rows(rows: Sequence[str]) -> np.ndarray:
    if not rows or len({len(r) for r in rows}) != 1:
        raise FingerprintError("grid rows must be non-empty and of equal length")
    return normalize_grid(np.array([[int(ch) for ch in r] for r in rows], dtype=np.uint8))


def grid_key(grid: np.ndarray) -> str:
    """Canonical text of a grid: ``"H,W:"`` + rows joined by ``/``."""
    h, w = grid.shape
    return f"{h},{w}:" + "/".join(grid_rows(grid))


def trim_walls(grid: np.ndarray) -> Tuple[np.ndarray, Cell]:
    """Drop outer rows/cols that are entirely wall. Returns (trimmed, (row0, col0))."""
    free_rows = np.where((grid == 0).any(axis=1))[0]
    free_cols = np.where((grid == 0).any(axis=0))[0]
    if free_rows.size == 0:
        raise FingerprintError("grid has no free cell")
    r0, r1 = int(free_rows[0]), int(free_rows[-1])
    c0, c1 = int(free_cols[0]), int(free_cols[-1])
    out = np.ascontiguousarray(grid[r0:r1 + 1, c0:c1 + 1])
    out.setflags(write=False)
    return out, (r0, c0)


# -- rotations / reflections (grid and cells transformed together) ------------

def transform_name(k: int, flip: bool) -> str:
    return f"r{90 * k}" + ("_flip" if flip else "")


def transform_grid(grid: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(grid, k)
    if flip:
        out = np.fliplr(out)
    out = np.ascontiguousarray(out)
    out.setflags(write=False)
    return out


def transform_cell(cell: Cell, shape: Tuple[int, int], k: int, flip: bool) -> Cell:
    """Where ``cell`` of a grid of ``shape`` lands under ``transform_grid(.., k, flip)``."""
    r, c = cell
    h, w = shape
    for _ in range(k):          # np.rot90 (counter-clockwise): new[w-1-c, r] = old[r, c]
        r, c, h, w = w - 1 - c, r, w, h
    if flip:                    # np.fliplr: new[r, w-1-c] = old[r, c]
        c = w - 1 - c
    return (r, c)


# -- agents -------------------------------------------------------------------

def _pairs(starts: Sequence[Cell], goals: Sequence[Cell]) -> List[List[int]]:
    return [[int(s[0]), int(s[1]), int(g[0]), int(g[1])] for s, g in zip(starts, goals)]


def agents_ordered_text(starts, goals) -> str:
    return json.dumps(_pairs(starts, goals), separators=(",", ":"))


def agents_set_text(starts, goals) -> str:
    return json.dumps(sorted(_pairs(starts, goals)), separators=(",", ":"))


def validate_agents(grid: np.ndarray, starts: Sequence[Cell], goals: Sequence[Cell]) -> None:
    if len(starts) != len(goals) or not starts:
        raise FingerprintError("starts and goals must be non-empty and of equal length")
    h, w = grid.shape
    for kind, cells in (("start", starts), ("goal", goals)):
        for i, (r, c) in enumerate(cells):
            if not (0 <= r < h and 0 <= c < w):
                raise FingerprintError(f"agent {i} {kind} {(r, c)} is outside the {h}x{w} grid")
            if grid[r, c] != 0:
                raise FingerprintError(f"agent {i} {kind} {(r, c)} is on a wall")
        if len(set(map(tuple, cells))) != len(cells):
            raise FingerprintError(f"two agents share a {kind} cell")


# -- fingerprints ---------------------------------------------------------------

def compute_fingerprints(grid, starts: Sequence[Cell], goals: Sequence[Cell]) -> Dict:
    """All fingerprint fields of one instance (starts/goals in (row, col))."""
    g = normalize_grid(grid)
    starts = [tuple(int(v) for v in s) for s in starts]
    goals = [tuple(int(v) for v in s) for s in goals]
    validate_agents(g, starts, goals)

    trimmed, (r0, c0) = trim_walls(g)
    t_starts = [(r - r0, c - c0) for r, c in starts]
    t_goals = [(r - r0, c - c0) for r, c in goals]

    keyed = []
    for k, flip in TRANSFORMS:
        keyed.append((grid_key(transform_grid(trimmed, k, flip)), k, flip))
    canon_key = min(key for key, _, _ in keyed)
    canon_ordered, canon_set, canon_names = [], [], []
    for key, k, flip in keyed:
        if key != canon_key:
            continue
        cs = [transform_cell(p, trimmed.shape, k, flip) for p in t_starts]
        cg = [transform_cell(p, trimmed.shape, k, flip) for p in t_goals]
        canon_ordered.append(agents_ordered_text(cs, cg))
        canon_set.append(agents_set_text(cs, cg))
        canon_names.append(transform_name(k, flip))

    return {
        "fp_schema": FP_SCHEMA,
        "grid_shape": [int(g.shape[0]), int(g.shape[1])],
        "free_cells": int((g == 0).sum()),
        "num_agents": len(starts),
        "starts_rc": [list(s) for s in starts],
        "goals_rc": [list(s) for s in goals],
        "grid_fp": sha256_text(grid_key(g)),
        "grid_trim_fp": sha256_text(grid_key(trimmed)),
        "grid_trim_offset": [r0, c0],
        "grid_trim_shape": [int(trimmed.shape[0]), int(trimmed.shape[1])],
        "grid_trim_rows": grid_rows(trimmed),
        "map_canon_fp": sha256_text(canon_key),
        "map_canon_transforms": canon_names,
        "agents_ordered_fp": sha256_text(agents_ordered_text(starts, goals)),
        "agents_set_fp": sha256_text(agents_set_text(starts, goals)),
        "agents_trim_ordered_fp": sha256_text(agents_ordered_text(t_starts, t_goals)),
        "agents_trim_set_fp": sha256_text(agents_set_text(t_starts, t_goals)),
        "agents_canon_ordered_fp": sha256_text(min(canon_ordered)),
        "agents_canon_set_fp": sha256_text(min(canon_set)),
    }


def _repo_relative(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def fingerprint_scenario(scenario_path) -> Dict:
    """Read a scenario JSON (read-only) and return its source info + fingerprints.

    Start/goal come from ``load_map_generator_scenario`` (``[x, y]`` ->
    ``(row, col)``); agent id = list index, exactly as the loader assigns it.
    The JSON ``agent_id`` fields are only compared with that index.
    """
    scenario_path = Path(scenario_path).resolve()
    raw = json.loads(scenario_path.read_text(encoding="utf-8"))
    map_path = _resolve_map_file(raw["map_file"], scenario_path)
    grid, starts, goals = load_map_generator_scenario(scenario_path)
    ids_in_file: List[Optional[int]] = [a.get("agent_id") for a in raw["agents"]]
    record = {
        "source_path": _repo_relative(scenario_path),
        "scenario_json_sha256": sha256_file(scenario_path),
        "map_file_path": _repo_relative(map_path),
        "map_file_sha256": sha256_file(map_path),
        "scenario_id": raw.get("scenario_id"),
        "map_id": raw.get("map_id"),
        "agent_ids_in_file": ids_in_file,
        "agent_id_matches_index": ids_in_file == list(range(len(ids_in_file))),
    }
    record.update(compute_fingerprints(grid, starts, goals))
    return record
