"""IL input-distribution reference and per-instance OOD check (read-only).

Features are the exact inputs the NavHint CNN sees for one agent:
  goal_row, goal_col   flowdist goal hint = make_goal_dir(bfs_dist(grid, goal),
                       cell, goal, "flowdist") (src/il/nav_hint.py, as
                       src/il/policy.NavHintPolicy computes it)
  wall_count, robot_count, other_goal_count
                       number of 1-cells per channel of the 3x5x5 local view
                       (wall/outside, other robot, other agent's goal) built by
                       MAPFStepSimulator.reset() -- the runtime code path

The reference holds empirical percentiles (0.5 %, 99.5 %, numpy "linear") of
each feature over the Alpha 1 training set real_v03_fd.npz. A new instance
gets OOD_WARNING when a start-state feature, or the goal hint over any cell
an agent can reach, falls outside [p0.5, p99.5]. A warning is a record, not
a failure: ``auto_fail`` stays False until a rule is approved
(docs/data_split_plan_v1.md section 7.3).

Nothing here loads or changes the IL model, the checkpoint or the npz.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from src.il.nav_hint import bfs_dist, make_goal_dir
from src.simulator.mapf_step_simulator import MAPFStepSimulator

REFERENCE_SCHEMA = 1
PERCENTILES = (0.5, 99.5)
FEATURES = ("goal_row", "goal_col", "wall_count", "robot_count", "other_goal_count")
HINT_MODE = "flowdist"
IN_RANGE, OOD_WARNING = "IN_RANGE", "OOD_WARNING"
AUTO_FAIL_POLICY = "not decided; OOD_WARNING is recorded only (data_split_plan_v1 7.3)"


def _pkey(p: float) -> str:
    return f"p{p:g}"


def feature_matrix(states_grid: np.ndarray, goal_dirs: np.ndarray) -> Dict[str, np.ndarray]:
    """(N,3,5,5) local views + (N,2) goal hints -> {feature: (N,) float64}."""
    sg = np.asarray(states_grid)
    gd = np.asarray(goal_dirs, dtype=np.float64)
    counts = sg.reshape(sg.shape[0], sg.shape[1], -1).sum(axis=2).astype(np.float64)
    return {"goal_row": gd[:, 0], "goal_col": gd[:, 1], "wall_count": counts[:, 0],
            "robot_count": counts[:, 1], "other_goal_count": counts[:, 2]}


def build_reference(npz_path, manifest_path=None) -> Dict:
    """Percentile reference from the training npz (opened read-only, no pickle)."""
    npz_path = Path(npz_path)
    raw = npz_path.read_bytes()
    with np.load(npz_path, allow_pickle=False) as data:
        sg, gd = data["states_grid"], data["goal_dirs"]
        feats = feature_matrix(sg, gd)
    ref = {
        "reference_schema": REFERENCE_SCHEMA,
        "source": {"file": npz_path.name, "sha256": hashlib.sha256(raw).hexdigest(),
                   "n_samples": int(gd.shape[0])},
        "percentile_method": "numpy.percentile(method='linear')",
        "numpy_version": np.__version__,
        "hint_mode": HINT_MODE,
        "percentiles": list(PERCENTILES),
        "features": {},
        "note": ("flow_step breaks distance ties in the order up, down, left, right, so "
                 "vertical hints dominate the training set; goal_row and goal_col have "
                 "different ranges by construction."),
    }
    for name in FEATURES:
        v = feats[name]
        entry = {_pkey(p): float(np.percentile(v, p, method="linear")) for p in PERCENTILES}
        entry.update(min=float(v.min()), max=float(v.max()), mean=float(v.mean()))
        ref["features"][name] = entry
    gmean = gd.astype(np.float64).mean(0)
    gstd = gd.astype(np.float64).std(0, ddof=1)
    ref["normalization_check"] = {"goal_mean": gmean.tolist(), "goal_std_ddof1": gstd.tolist()}
    if manifest_path is not None:
        m = json.loads(Path(manifest_path).read_text(encoding="utf-8"))["checkpoints"]["cnn_navhint.pt"]
        mm = np.asarray(m["normalization"]["goal_mean"], dtype=np.float64).reshape(-1)
        ms = np.asarray(m["normalization"]["goal_std"], dtype=np.float64).reshape(-1)
        ref["normalization_check"].update(
            manifest_goal_mean=mm.tolist(), manifest_goal_std=ms.tolist(),
            max_abs_diff=float(max(np.abs(gmean - mm).max(), np.abs(gstd - ms).max())))
    return ref


def reference_json(ref: Dict) -> bytes:
    return (json.dumps(ref, indent=2, sort_keys=True) + "\n").encode("utf-8")


def load_reference(path) -> Dict:
    ref = json.loads(Path(path).read_text(encoding="utf-8"))
    if ref.get("reference_schema") != REFERENCE_SCHEMA:
        raise ValueError(f"{path}: reference_schema {ref.get('reference_schema')} != {REFERENCE_SCHEMA}")
    return ref


def start_features(grid, starts: Sequence, goals: Sequence) -> List[Dict[str, float]]:
    """Per-agent features at t=0, through the runtime code path."""
    grid = np.asarray(grid)
    sim = MAPFStepSimulator(cbs_solver_root=None, max_steps=1)
    ids = list(range(len(starts)))
    obs = sim.reset(grid, {i: tuple(starts[i]) for i in ids}, {i: tuple(goals[i]) for i in ids})
    out = []
    for i in ids:
        hint = make_goal_dir(bfs_dist(grid, tuple(goals[i])), tuple(starts[i]), tuple(goals[i]), HINT_MODE)
        local = np.asarray(obs[i]["grid"])
        out.append({"goal_row": float(hint[0]), "goal_col": float(hint[1]),
                    "wall_count": float(local[0].sum()), "robot_count": float(local[1].sum()),
                    "other_goal_count": float(local[2].sum())})
    return out


def reachable_hint_range(grid, goal) -> Dict[str, List[float]]:
    """Min/max of the goal hint over every cell that can reach ``goal``."""
    grid = np.asarray(grid)
    dist = bfs_dist(grid, tuple(goal))
    rows, cols = [], []
    for r, c in zip(*np.where(dist >= 0)):
        h = make_goal_dir(dist, (int(r), int(c)), tuple(goal), HINT_MODE)
        rows.append(float(h[0]))
        cols.append(float(h[1]))
    return {"goal_row": [min(rows), max(rows)], "goal_col": [min(cols), max(cols)]}


def check_instance(grid, starts, goals, reference: Dict) -> Dict:
    lo_k, hi_k = _pkey(PERCENTILES[0]), _pkey(PERCENTILES[1])
    feats = reference["features"]
    warnings = []
    starts_f = start_features(grid, starts, goals)
    for i, f in enumerate(starts_f):
        for name in FEATURES:
            lo, hi = feats[name][lo_k], feats[name][hi_k]
            if not lo <= f[name] <= hi:
                warnings.append({"kind": "START", "agent": i, "feature": name, "value": f[name],
                                 "bounds": [lo, hi]})
    ranges = []
    for i, g in enumerate(goals):
        rng = reachable_hint_range(grid, g)
        ranges.append(rng)
        for name in ("goal_row", "goal_col"):
            lo, hi = feats[name][lo_k], feats[name][hi_k]
            vmin, vmax = rng[name]
            if vmin < lo or vmax > hi:
                warnings.append({"kind": "REACHABLE_RANGE", "agent": i, "feature": name,
                                 "value": [vmin, vmax], "bounds": [lo, hi]})
    return {
        "status": OOD_WARNING if warnings else IN_RANGE,
        "warnings": warnings,
        "start_features": starts_f,
        "reachable_hint_range": ranges,
        "auto_fail": False,
        "auto_fail_policy": AUTO_FAIL_POLICY,
        "reference_sha256": reference["source"]["sha256"],
    }
