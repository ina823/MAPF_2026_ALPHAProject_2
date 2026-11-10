"""Candidate wait-for graph (WFG) from one simulator step — pure function.

Status: DRAFT, rule ``wfg-candidate-v0``. This is the candidate rule proposed in
``docs/selective_hybrid_contract_v0.md`` §3.2. It has NOT been agreed with the
CBS side (PENDING P5) and nothing in the runtime calls it: there is no
Selective Runner and no ``group_selector``. Global Hybrid is unchanged.

Input is one step's per-agent ``current`` and ``intended`` cells (the step CSV
``current_*`` and pre-resolution ``intended_*`` columns), the set of agents
whose move was cancelled, and the grid. Coordinates are ``(row, col)``; agent
IDs are the runtime IDs and are never renumbered.

The function deliberately does not read the simulator's ``conflict_type`` or
the Monitor's ``blocking_agents``. ``conflict_type`` can hide a swap: in
Bottleneck t=7 the 1<->3 swap is logged as ``vertex`` because the vertex pass
cancels agent 3 first. Edges are therefore recomputed from current/intended.

Edge kinds (a "mover" is an agent whose intended cell is in bounds, free, and
different from its current cell; a wall or out-of-bounds intent is "no move"):

- ``swap``    a and b are movers, a -> current[b] and b -> current[a].
              Undirected, ``src < dst``; ``cell`` is current[dst]
              (= src's target).
- ``occupy``  mover a -> current[b] and the pair is not a swap. Directed a -> b.
              ``occupant_moves`` records whether b is itself a mover.
- ``contest`` movers a and b target the same cell. Undirected, ``src < dst``;
              one edge per pair.

PENDING (T7): an ``occupy`` edge whose occupant moves out (the simulator lets
an agent follow into a vacated cell) may not be a real wait. v0 still emits the
edge and connects it in ``groups``. This is a conservative choice that can only
enlarge groups, and it is NOT a verified rule. ``occupant_moves=True`` marks
these edges so a later rule can drop them without recomputing.

Groups are the connected components of the undirected edge graph that have at
least two agents and at least one agent in ``blocked``.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Collection, Mapping, Optional, Sequence

GridLocation = tuple[int, int]

RULE_VERSION = "wfg-candidate-v0"
EDGE_KINDS = ("contest", "occupy", "swap")


@dataclass(frozen=True, order=True)
class WfgEdge:
    """One candidate edge. Ordering is (src, dst, kind, cell)."""

    src: int
    dst: int
    kind: str
    cell: GridLocation
    # Only set for kind == "occupy". Excluded from ordering/equality: it is
    # derived from the other fields' inputs, and None vs bool is not orderable.
    occupant_moves: Optional[bool] = field(default=None, compare=False)


@dataclass(frozen=True)
class WfgResult:
    edges: tuple[WfgEdge, ...]
    groups: tuple[tuple[int, ...], ...]
    rule_version: str = RULE_VERSION


def _agent_id(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label}: agent id must be int, got {value!r}")
    return value


def _cell(value, label: str) -> GridLocation:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{label}: cell must be (row, col), got {value!r}")
    row, col = value
    out = []
    for v in (row, col):
        if isinstance(v, bool):
            raise ValueError(f"{label}: cell must hold ints, got {value!r}")
        try:
            out.append(operator.index(v))  # int or numpy integer; rejects float/str
        except TypeError:
            raise ValueError(f"{label}: cell must hold ints, got {value!r}") from None
    return (out[0], out[1])


def _grid_rows(grid) -> list[list[int]]:
    try:
        rows = [[int(v) for v in row] for row in grid]
    except TypeError:
        raise ValueError("grid must be a 2D sequence") from None
    if not rows or not rows[0]:
        raise ValueError("grid must be non-empty")
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError("grid must be rectangular")
    return rows


def build_wfg(
    current: Mapping[int, Sequence[int]],
    intended: Mapping[int, Sequence[int]],
    blocked: Collection[int],
    grid,
) -> WfgResult:
    """Return the candidate WFG for one step. Raises ValueError on bad input."""
    rows = _grid_rows(grid)
    height, width = len(rows), len(rows[0])

    def free(cell: GridLocation) -> bool:
        r, c = cell
        return 0 <= r < height and 0 <= c < width and rows[r][c] == 0

    cur = {_agent_id(a, "current"): _cell(p, f"current[{a}]") for a, p in current.items()}
    want = {_agent_id(a, "intended"): _cell(p, f"intended[{a}]") for a, p in intended.items()}
    if not cur:
        raise ValueError("no agents")
    if set(cur) != set(want):
        raise ValueError(f"agent ids differ: current={sorted(cur)} intended={sorted(want)}")
    blocked_ids = {_agent_id(a, "blocked") for a in blocked}
    if not blocked_ids <= set(cur):
        raise ValueError(f"blocked has unknown agents: {sorted(blocked_ids - set(cur))}")

    occupant: dict[GridLocation, int] = {}
    for a in sorted(cur):
        if not free(cur[a]):
            raise ValueError(f"current[{a}]={cur[a]} is out of bounds or an obstacle")
        if cur[a] in occupant:
            raise ValueError(f"agents {occupant[cur[a]]} and {a} share cell {cur[a]}")
        occupant[cur[a]] = a
        dr, dc = want[a][0] - cur[a][0], want[a][1] - cur[a][1]
        if abs(dr) + abs(dc) > 1:
            raise ValueError(f"intended[{a}]={want[a]} is not adjacent to current {cur[a]}")

    # Movers: in-bounds, free, actual displacement. Wall/out-of-bounds = no move.
    target = {a: want[a] for a in cur if want[a] != cur[a] and free(want[a])}

    edges: set[WfgEdge] = set()
    for a in sorted(target):
        b = occupant.get(target[a])
        if b is None:
            continue
        if target.get(b) == cur[a]:
            lo, hi = min(a, b), max(a, b)
            edges.add(WfgEdge(lo, hi, "swap", cur[hi]))
        else:
            edges.add(WfgEdge(a, b, "occupy", target[a], occupant_moves=b in target))

    by_cell: dict[GridLocation, list[int]] = {}
    for a in sorted(target):
        by_cell.setdefault(target[a], []).append(a)
    for cell, movers in by_cell.items():
        for i, a in enumerate(movers):
            for b in movers[i + 1:]:
                edges.add(WfgEdge(a, b, "contest", cell))

    parent = {a: a for a in cur}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in edges:
        ra, rb = find(e.src), find(e.dst)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    comps: dict[int, list[int]] = {}
    for a in sorted(cur):
        comps.setdefault(find(a), []).append(a)
    groups = tuple(
        sorted(
            (tuple(m) for m in comps.values() if len(m) >= 2 and blocked_ids & set(m)),
            key=lambda g: g[0],
        )
    )
    return WfgResult(edges=tuple(sorted(edges)), groups=groups)
