"""Static checks of the PILOT v2 candidates. Runs no solver and no IL policy.

Checks each pre-registered candidate: regeneration bytes, cell validity, BFS
reachability, agent ids, coordinate round trip, IL input OOD, and relations to
all registry records and to the other candidates. Results are recorded, never
used to change a grade or drop a candidate.

Independence is never approved here. Every automatic relation and every manual
similarity is a HOLD with final_decision null in
registry/pilot_v2_candidate_review_v0.jsonl (created once, then only compared).
The registry itself is opened read-only and its sha256 is checked before and after.

The full report goes to a NEW directory outputs/pilot_v2_checks/static_<YYYYmmdd_HHMMSS>/.

Exit codes: 0 no integrity problem (OOD, relations and missed targets are
reported, not judged), 1 integrity problems, 2 refused before running.

Usage (from the repository root):
    python -m scripts.check_pilot_v2_candidates
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.scenario_loader import load_map_generator_scenario
from src.data.fingerprint import fingerprint_scenario, sha256_file
from src.data.generator import GeneratorError, validate_instance
from src.data.il_compat import check_instance, load_reference
from src.data.quality_gate import classify_ood, submap_evidence
from src.data.registry import EXACT, SUBMAP, Registry, RegistryError, instance_fp, relate
from src.il.nav_hint import bfs_dist
from scripts.generate_pilot_v2_candidates import (
    GENERATOR_V1_FILE, GENERATOR_V2_FILE, OUT_ROOT, PREREG_SHA256,
    candidate_paths, generate_candidate, load_prereg, spec_problems,
)
from scripts.register_dev_scenarios import INSTANCES

REFERENCE = "il_input_reference_v1.json"
REVIEW_FILE = "pilot_v2_candidate_review_v0.jsonl"
MANUAL_SIMILARITY = "MANUAL_STRUCTURE_SIMILARITY"
HOLD = "HOLD"
NO_RELATION_NOT_APPROVED = "NO_RELATION_FOUND_NOT_APPROVED"
NORMAL, STRESS = "NORMAL", "STRESS"


# -- static map metrics ---------------------------------------------------------

def _free(grid, r, c) -> bool:
    h, w = grid.shape
    return 0 <= r < h and 0 <= c < w and grid[r, c] == 0


def width1_cells(grid: np.ndarray) -> List[List[int]]:
    """Free cells blocked on both vertical sides or both horizontal sides."""
    out = []
    for r, c in zip(*np.where(grid == 0)):
        r, c = int(r), int(c)
        vertical = not _free(grid, r - 1, c) and not _free(grid, r + 1, c)
        horizontal = not _free(grid, r, c - 1) and not _free(grid, r, c + 1)
        if vertical or horizontal:
            out.append([r, c])
    return out


def _component_size(grid: np.ndarray, start, blocked=None) -> int:
    seen, queue = {start}, deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if _free(grid, nr, nc) and (nr, nc) != blocked and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen)


def articulation_cells(grid: np.ndarray) -> List[List[int]]:
    """Free cells whose removal disconnects the remaining free cells."""
    free = [(int(r), int(c)) for r, c in zip(*np.where(grid == 0))]
    out = []
    for cell in free:
        rest = [x for x in free if x != cell]
        if rest and _component_size(grid, rest[0], blocked=cell) != len(rest):
            out.append(list(cell))
    return out


def agent_metrics(grid, starts, goals) -> Dict:
    dists = [int(bfs_dist(grid, tuple(g))[tuple(s)]) for s, g in zip(starts, goals)]
    goal_set = {tuple(g) for g in goals}
    on_goal = [i for i, (s, g) in enumerate(zip(starts, goals)) if tuple(s) in goal_set and tuple(s) != tuple(g)]
    reachable = all(d >= 0 for d in dists)
    return {"bfs_distances": dists, "all_goals_reachable": reachable,
            "soc_lower_bound": sum(dists) if reachable else None,
            "makespan_lower_bound": max(dists) if reachable else None,
            "agents_starting_on_another_goal": on_goal}


def max_abs_goal_col(il_check: Dict) -> float:
    values = [abs(f["goal_col"]) for f in il_check["start_features"]]
    for rng in il_check["reachable_hint_range"]:
        values += [abs(v) for v in rng["goal_col"]]
    return max(values)


def evaluate_target(grade: str, ood_class: str, n_width1: int, num_agents: int, abs_goal_col: float) -> Dict:
    """Pre-registered target of a grade. The grade itself is an input and never changes."""
    if grade == NORMAL:
        conditions = {"ood_class_in_range": ood_class == "IN_RANGE", "no_width1_cells": n_width1 == 0,
                      "at_most_4_agents": num_agents <= 4}
    elif grade == STRESS:
        conditions = {"has_width1_cells": n_width1 > 0, "goal_col_hint_at_least_13": abs_goal_col >= 13,
                      "exactly_6_agents": num_agents == 6}
    else:
        raise ValueError(f"unknown grade {grade!r}")
    return {"grade": grade, "target_met": all(conditions.values()), "conditions": conditions}


# -- relations and holds --------------------------------------------------------

def independence_summary(existing: List[Dict], candidate_pairs: List[Dict], manual: List[Dict]) -> Dict:
    """``independence_approved`` is False always: only a separate human step can approve it."""
    auto = len(existing) + len(candidate_pairs)
    return {"automatic_relations_existing": len(existing), "automatic_relations_candidates": len(candidate_pairs),
            "manual_holds": len(manual), "status": HOLD if auto or manual else NO_RELATION_NOT_APPROVED,
            "independence_approved": False}


def pairwise_candidate_relations(records: Dict[str, Dict]) -> List[Dict]:
    """Every relation between two candidates, identity (instance_fp) checked as well."""
    out = []
    for (ida, a), (idb, b) in itertools.combinations(sorted(records.items()), 2):
        rels = relate(a, b)
        same_instance = instance_fp(a) == instance_fp(b)
        if same_instance and not any(r["relation"] == EXACT for r in rels):
            rels = rels + [{"relation": EXACT, "status": "CONFIRMED", "details": {"via": "instance_fp"}}]
        for rel in rels:
            out.append({"a": ida, "b": idb, "relation": rel["relation"], "relation_status": rel["status"],
                        "details": rel["details"]})
    return out


def _relation_evidence(rel: Dict, a: Dict, b: Dict) -> Dict:
    evidence = dict(rel.get("details") or {})
    if rel["relation"] == SUBMAP:
        evidence.update(submap_evidence(a, b))
    return evidence


def review_entries(prereg: Dict, candidates: Dict[str, Dict], registry: Registry,
                   existing: Dict[str, List[Dict]], pairs: List[Dict]) -> List[Dict]:
    """HOLD entries, deterministic: automatic-to-registry, automatic-to-candidate, manual."""
    by_id = registry.by_id()
    rows = []

    def add(cid, scope, other, other_fps, relation, relation_status, source, evidence, basis):
        rows.append({"candidate_id": cid, "candidate_instance_fp": instance_fp(candidates[cid]),
                     "other_scope": scope, "other": other, "other_instance_fps": other_fps,
                     "relation": relation, "relation_status": relation_status, "source": source,
                     "status": HOLD, "final_decision": None, "decision_owner": None, "evidence": evidence,
                     "basis": basis, "recorded_at": prereg["recorded_at"], "recorded_by": prereg["recorded_by"]})

    for cid in sorted(existing):
        for rel in existing[cid]:
            other = by_id[rel["other"]]
            add(cid, "instance", [rel["other"]], [instance_fp(other)], rel["relation"], rel["status"], "AUTOMATIC",
                _relation_evidence(rel, candidates[cid], other),
                "Recomputed by registry.relate; not decided here.")
    for pair in pairs:
        rel = {"relation": pair["relation"], "details": pair["details"]}
        add(pair["a"], "candidate", [pair["b"]], [instance_fp(candidates[pair["b"]])], pair["relation"],
            pair["relation_status"], "AUTOMATIC", _relation_evidence(rel, candidates[pair["a"]], candidates[pair["b"]]),
            "Recomputed between candidates by registry.relate; not decided here.")
    for hold in prereg["manual_similarity_holds"]:
        if hold["candidate_id"] not in candidates:
            continue
        if hold["other_scope"] == "family":
            members = sorted(r["instance_id"] for r in registry.records if r["family_id"] in hold["other"])
        else:
            members = list(hold["other"])
        add(hold["candidate_id"], hold["other_scope"], list(hold["other"]),
            [instance_fp(by_id[m]) for m in members], MANUAL_SIMILARITY, "REVIEW", "MANUAL",
            {"members": members}, hold["basis"])
    for n, row in enumerate(rows, 1):
        row["review_id"] = f"PV2R{n:04d}"
    return rows


def review_bytes(entries: List[Dict]) -> bytes:
    return "".join(json.dumps(e, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
                   for e in entries).encode("utf-8")


# -- per-candidate check --------------------------------------------------------

def check_candidate(spec: Dict, root: Path, reference: Dict) -> Dict:
    """Everything about one candidate. Problems are collected; nothing is raised or dropped."""
    out = {"candidate_id": spec["candidate_id"], "slug": spec["slug"], "grade": spec["grade"],
           "seed": spec["seed"], "predicted": spec["predicted"], "problems": [], "fingerprint": None}
    problems = out["problems"]
    map_path, scen_path = candidate_paths(spec, root)
    if not (map_path.is_file() and scen_path.is_file()):
        problems.append(f"{spec['slug']}: scenario or map file is missing")
        return out

    try:
        inst, npy, js = generate_candidate(spec)
    except (GeneratorError, KeyError, TypeError, ValueError) as exc:
        problems.append(f"{spec['slug']}: regeneration failed ({exc})")
        return out
    out["seed_reproducible"] = map_path.read_bytes() == npy and scen_path.read_bytes() == js
    if not out["seed_reproducible"]:
        problems.append(f"{spec['slug']}: files differ from a same-seed regeneration")
    problems += [f"{spec['slug']}: {p}" for p in spec_problems(spec, inst)]

    grid, starts, goals = load_map_generator_scenario(scen_path)
    grid = np.asarray(grid)
    raw = json.loads(scen_path.read_text(encoding="utf-8"))
    out["grid_shape"] = [int(v) for v in grid.shape]
    out["free_cells"] = int((grid == 0).sum())
    out["num_agents"] = len(starts)
    out["starts_rc"] = [list(map(int, s)) for s in starts]
    out["goals_rc"] = [list(map(int, g)) for g in goals]

    try:
        validate_instance(grid, starts, goals)
        out["cells_valid"] = True
    except GeneratorError as exc:
        out["cells_valid"] = False
        problems.append(f"{spec['slug']}: invalid cells ({exc})")

    out["agent_id_matches_index"] = [a["agent_id"] for a in raw["agents"]] == list(range(len(raw["agents"])))
    json_as_rc = [((a["start"][1], a["start"][0]), (a["goal"][1], a["goal"][0])) for a in raw["agents"]]
    out["coordinates_ok"] = (json_as_rc == [(tuple(s), tuple(g)) for s, g in zip(starts, goals)]
                             and tuple(map(tuple, starts)) == tuple(inst.starts)
                             and tuple(map(tuple, goals)) == tuple(inst.goals)
                             and raw["generator"]["seed"] == spec["seed"])
    if not out["agent_id_matches_index"]:
        problems.append(f"{spec['slug']}: agent_id differs from the list index")
    if not out["coordinates_ok"]:
        problems.append(f"{spec['slug']}: coordinate round trip differs")

    metrics = agent_metrics(grid, starts, goals)
    out.update(metrics)
    if not metrics["all_goals_reachable"]:
        problems.append(f"{spec['slug']}: a goal is unreachable by BFS")
    wide1 = width1_cells(grid)
    out["width1_cells"] = wide1
    out["articulation_cells"] = articulation_cells(grid)

    il = check_instance(grid, starts, goals, reference)
    ood = classify_ood(il, reference)
    out["il_check"] = il
    out["ood"] = ood
    out["max_abs_goal_col"] = max_abs_goal_col(il)
    out["target"] = evaluate_target(spec["grade"], ood["ood_class"], len(wide1), len(starts), out["max_abs_goal_col"])
    out["prediction_vs_result"] = {
        "ood_class": {"predicted": spec["predicted"]["ood_class"], "result": ood["ood_class"]},
        "target_met": {"predicted": spec["predicted"]["target_met"], "result": out["target"]["target_met"]},
        "automatic_relations": {"predicted": spec["predicted"]["automatic_relations"]},
    }

    fp = fingerprint_scenario(scen_path)
    fp.update(structure_type=inst.structure_type, structure_descriptor=inst.descriptor)
    out["fingerprint"] = fp
    return out


def build_report(root: Path, registry_dir: Path) -> Dict:
    prereg = load_prereg()
    registry = Registry.load(registry_dir / INSTANCES)
    reference = load_reference(registry_dir / REFERENCE)
    results = [check_candidate(spec, root, reference) for spec in prereg["candidates"]]
    fingerprints = {r["candidate_id"]: r["fingerprint"] for r in results if r["fingerprint"]}

    by_id = registry.by_id()
    existing: Dict[str, List[Dict]] = {}
    for result in results:
        cid = result["candidate_id"]
        fp = result["fingerprint"]
        if fp is None:
            continue
        rels = [dict(rel, other=rec["instance_id"], other_family=rec["family_id"], other_split=rec["split"])
                for rec in registry.records for rel in relate(rec, fp)]
        existing[cid] = rels
        decision = registry.evaluate(fp, "PILOT")
        result["registry_evaluate_in_memory"] = {
            "status": decision.status, "proposed_family_id_unregistered": decision.family_id,
            "reasons": decision.reasons}
        result["relations_existing"] = rels
        result["instance_fp"] = instance_fp(fp)
        result["map_canon_fp"] = fp["map_canon_fp"]
    pairs = pairwise_candidate_relations(fingerprints)
    entries = review_entries(prereg, fingerprints, registry, existing, pairs)
    for result in results:
        cid = result["candidate_id"]
        if result["fingerprint"] is None:
            result["independence"] = {"status": "NOT_CHECKED", "independence_approved": False}
            continue
        mine = [p for p in pairs if cid in (p["a"], p["b"])]
        manual = [h for h in prereg["manual_similarity_holds"] if h["candidate_id"] == cid]
        result["relations_candidates"] = mine
        result["independence"] = independence_summary(existing[cid], mine, manual)
        result["holds"] = [e["review_id"] for e in entries
                           if e["candidate_id"] == cid or (e["other_scope"] == "candidate" and cid in e["other"])]
    return {
        "prereg_sha256": PREREG_SHA256,
        "generator_v2_sha256": sha256_file(GENERATOR_V2_FILE),
        "generator_v1_sha256": sha256_file(GENERATOR_V1_FILE),
        "registry_instances_sha256": registry.source_sha256,
        "registry_records": len(registry.records),
        "reference_sha256": reference["source"]["sha256"],
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "cbs_run": False,
        "candidates": results,
        "candidate_pairs": pairs,
        "review_entries": entries,
        "problems": [p for r in results for p in r["problems"]],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--registry-dir", default=str(PROJECT_ROOT / "registry"))
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    args = ap.parse_args(argv)
    root, registry_dir = Path(args.root), Path(args.registry_dir)
    out_dir = Path(args.out_root) / f"static_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        before = sha256_file(registry_dir / INSTANCES)
        report = build_report(root, registry_dir)
    except (OSError, ValueError, RegistryError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    if out_dir.exists():
        print(f"REFUSED: output directory exists: {out_dir}", file=sys.stderr)
        return 2

    problems = report["problems"]
    data = review_bytes(report["review_entries"])
    review_path = registry_dir / REVIEW_FILE
    if not review_path.exists():
        with open(review_path, "xb") as fh:
            fh.write(data)
    elif review_path.read_bytes() != data:
        problems.append(f"{review_path} differs from the recomputed entries; not overwritten")
    if sha256_file(registry_dir / INSTANCES) != before:
        problems.append("registry instances file changed during the check")

    out_dir.mkdir(parents=True, exist_ok=False)
    with open(out_dir / "static_checks.json", "x", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")

    for r in report["candidates"]:
        ood = r.get("ood", {}).get("ood_class")
        target = r.get("target", {}).get("target_met")
        print(f"{r['candidate_id']} {r['slug']:30s} {r['grade']:6s} reachable={r.get('all_goals_reachable')} "
              f"ood={ood} target_met={target} independence={r['independence']['status']} holds={r.get('holds')}")
    print(f"wrote {out_dir / 'static_checks.json'}")
    for p in problems:
        print(f"PROBLEM: {p}", file=sys.stderr)
    print("no integrity problems" if not problems else f"{len(problems)} problem(s)")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
