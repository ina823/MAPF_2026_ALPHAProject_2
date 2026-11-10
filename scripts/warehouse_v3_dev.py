"""Warehouse generator v3 DEV pilot: generate WH8-A and run static checks.

Runs no CBS solver and no IL episode. ``check`` builds only the t=0 simulator
observation (MAPFStepSimulator.reset) for the IL input check.

    python -m scripts.warehouse_v3_dev generate   # writes scenarios/warehouse_v3_dev/*.{npy,json}
    python -m scripts.warehouse_v3_dev check      # writes outputs/warehouse_v3_checks/static_<ts>/

``generate`` creates the map/scenario files exclusively (never overwrites);
if they already exist it only verifies that a regeneration is byte-identical.
``check`` opens registry/instances_v1.jsonl read-only, compares its sha256
before and after, and writes nothing into registry/. Every automatic relation
is reported as a HOLD with final_decision null; independence is never approved.

Exit codes: 0 no integrity problem (OOD and relations are reported, not judged),
1 integrity problem, 2 refused before running.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import spec as il_spec
from src.common.scenario_loader import load_map_generator_scenario
from src.data import generator_v3 as g3
from src.data.fingerprint import fingerprint_scenario, grid_rows, normalize_grid, sha256_file
from src.data.generator import GeneratorError, validate_instance
from src.data.il_compat import check_instance, load_reference
from src.data.quality_gate import classify_ood
from src.data.registry import Registry, RegistryError, create_files_exclusive, instance_fp, relate
from src.il.nav_hint import bfs_dist, flow_step
from src.simulator.mapf_step_simulator import _STEP_ACTION_DELTA, MAPFStepSimulator
from scripts.check_pilot_v2_candidates import agent_metrics, articulation_cells, width1_cells
from scripts.register_dev_scenarios import INSTANCES

DEV_DIR = "scenarios/warehouse_v3_dev"
PREREG_PATH = PROJECT_ROOT / DEV_DIR / "prereg_wh8a_v0.json"
PREREG_SHA256 = "211b46baf575a823685254c5fa8b09feb84cb510428aaa3a3c6db0a4cd9839f9"
GENERATOR_V3_FILE = PROJECT_ROOT / "src" / "data" / "generator_v3.py"
GENERATOR_V1_FILE = PROJECT_ROOT / "src" / "data" / "generator.py"
OUT_ROOT = PROJECT_ROOT / "outputs" / "warehouse_v3_checks"
REFERENCE = "il_input_reference_v1.json"
SPLIT = "DEV"
HOLD = "HOLD"


# -- prereg and generation --------------------------------------------------------

def load_prereg(path: Path = PREREG_PATH, pinned: str = PREREG_SHA256) -> dict:
    raw = Path(path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != pinned:
        raise RegistryError(f"{path}: sha256 {actual} differs from the pinned {pinned}")
    return json.loads(raw)


def candidate_paths(spec: dict, root: Path = PROJECT_ROOT):
    out = Path(root) / DEV_DIR
    return out / f"map_{spec['slug']}.npy", out / f"scenario_{spec['slug']}.json"


def spec_problems(spec: dict, inst) -> List[str]:
    problems = []
    if grid_rows(normalize_grid(inst.grid)) != spec["grid_rows"]:
        problems.append("grid differs from the pre-registered grid_rows")
    if len(inst.starts) != spec["num_agents"]:
        problems.append(f"{len(inst.starts)} agents, pre-registered {spec['num_agents']}")
    if inst.seed != spec["seed"]:
        problems.append(f"seed {inst.seed}, pre-registered {spec['seed']}")
    if inst.structure_type != spec["structure_type"]:
        problems.append(f"structure_type {inst.structure_type}, pre-registered {spec['structure_type']}")
    if inst.descriptor.get("zone_rows") != spec["zone_rows"]:
        problems.append("zone_rows differ from the pre-registration")
    return problems


def generate_candidate(spec: dict, prereg_sha: str = PREREG_SHA256):
    """(instance, map bytes, scenario bytes); generates twice and compares bytes."""
    code_sha, base_sha = sha256_file(GENERATOR_V3_FILE), sha256_file(GENERATOR_V1_FILE)

    def once():
        inst = g3.build(spec["builder"], spec["params"], spec["seed"])
        return inst, g3.scenario_files(inst, spec["slug"], code_sha256=code_sha, base_generator_sha256=base_sha,
                                       prereg_sha256=prereg_sha, candidate_id=spec["candidate_id"])
    inst, first = once()
    _, second = once()
    if first != second:
        raise GeneratorError(f"{spec['slug']}: same-seed regeneration is not byte-identical")
    return inst, first[0], first[1]


def generate(root: Path = PROJECT_ROOT) -> List[Dict]:
    prereg = load_prereg()
    records = []
    for spec in prereg["candidates"]:
        rec = {"candidate_id": spec["candidate_id"], "slug": spec["slug"], "seed": spec["seed"],
               "generated": False, "already_present": False, "error": None}
        records.append(rec)
        try:
            inst, npy, js = generate_candidate(spec)
            problems = spec_problems(spec, inst)
            if problems:
                raise GeneratorError("; ".join(problems))
            rec.update(map_sha256=hashlib.sha256(npy).hexdigest(), scenario_sha256=hashlib.sha256(js).hexdigest())
            map_path, scen_path = candidate_paths(spec, root)
            if map_path.exists() or scen_path.exists():
                if not (map_path.is_file() and scen_path.is_file()
                        and map_path.read_bytes() == npy and scen_path.read_bytes() == js):
                    raise GeneratorError("existing files differ from a same-seed regeneration; not overwritten")
                rec["already_present"] = True
            else:
                create_files_exclusive([(map_path, npy), (scen_path, js)])
            rec["generated"] = True
        except (GeneratorError, RegistryError, OSError, KeyError, TypeError, ValueError) as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
    return records


# -- static checks ----------------------------------------------------------------

def action_checks(grid: np.ndarray, starts, goals) -> Dict:
    """Action ids 0..4 agree between spec and simulator; one BFS shortest path per
    agent converts to valid action ids (moves only, 0..3)."""
    table_ok = (dict(il_spec.ACTION_DELTA) == dict(_STEP_ACTION_DELTA)
                and sorted(il_spec.ACTION_DELTA) == [0, 1, 2, 3, 4]
                and il_spec.ACTION_DELTA[4] == (0, 0))
    per_agent = []
    for s, g in zip(starts, goals):
        dist = bfs_dist(grid, tuple(g))
        cell, actions = tuple(s), []
        for _ in range(int(dist[cell]) if dist[cell] >= 0 else 0):
            dr, dc = flow_step(dist, cell)
            actions.append(il_spec.delta_to_action(dr, dc))
            cell = (cell[0] + dr, cell[1] + dc)
        per_agent.append({"bfs_path_actions": actions, "reaches_goal": cell == tuple(g),
                          "action_ids_ok": all(a in (0, 1, 2, 3) for a in actions)})
    return {"action_table_matches_simulator": table_ok, "agents": per_agent,
            "ok": table_ok and all(a["reaches_goal"] and a["action_ids_ok"] for a in per_agent)}


def loader_checks(scen_path: Path, inst) -> Dict:
    from scripts.run_il import load_scenario  # runtime path used by every runner
    grid, starts, goals = load_scenario(scen_path)
    raw = json.loads(scen_path.read_text(encoding="utf-8"))
    ids_ok = [a["agent_id"] for a in raw["agents"]] == list(range(len(raw["agents"])))
    runtime_ok = (list(starts) == list(range(len(inst.starts)))
                  and [tuple(starts[i]) for i in starts] == list(inst.starts)
                  and [tuple(goals[i]) for i in goals] == list(inst.goals)
                  and np.array_equal(np.asarray(grid), inst.grid))
    sim = MAPFStepSimulator(cbs_solver_root=None, max_steps=1)
    obs = sim.reset(np.asarray(grid), dict(starts), dict(goals))
    obs_ok = all(o["grid"].shape == (il_spec.GRID_C, il_spec.GRID_H, il_spec.GRID_W)
                 and o["goal_dir"].shape == (il_spec.GOAL_DIR_DIM,) for o in obs.values())
    return {"agent_id_matches_index": ids_ok, "run_il_load_scenario_matches_generator": runtime_ok,
            "t0_observation_shapes_ok": obs_ok, "grid_dtype": str(np.asarray(grid).dtype),
            "grid_values": sorted(int(v) for v in np.unique(grid))}


def check_candidate(spec: dict, root: Path, reference: Dict) -> Dict:
    out = {"candidate_id": spec["candidate_id"], "slug": spec["slug"], "seed": spec["seed"],
           "predicted": spec["predicted"], "problems": [], "fingerprint": None}
    problems = out["problems"]
    map_path, scen_path = candidate_paths(spec, root)
    if not (map_path.is_file() and scen_path.is_file()):
        problems.append(f"{spec['slug']}: scenario or map file is missing (run generate first)")
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
    out.update(grid_shape=[int(v) for v in grid.shape], grid_rows=grid_rows(normalize_grid(grid)),
               free_cells=int((grid == 0).sum()), wall_cells=int((grid == 1).sum()),
               num_agents=len(starts), density=round(len(starts) / int((grid == 0).sum()), 4),
               starts_rc=[list(map(int, s)) for s in starts], goals_rc=[list(map(int, g)) for g in goals],
               agent_order=inst.descriptor["agent_order"])
    try:
        validate_instance(grid, starts, goals)
        out["cells_valid"] = True
    except GeneratorError as exc:
        out["cells_valid"] = False
        problems.append(f"{spec['slug']}: invalid cells ({exc})")
    out["starts_distinct"] = len({tuple(s) for s in starts}) == len(starts)
    out["goals_distinct"] = len({tuple(g) for g in goals}) == len(goals)

    json_as_rc = [((a["start"][1], a["start"][0]), (a["goal"][1], a["goal"][0])) for a in raw["agents"]]
    out["coordinates_ok"] = (json_as_rc == [(tuple(s), tuple(g)) for s, g in zip(starts, goals)]
                             and tuple(map(tuple, starts)) == tuple(inst.starts)
                             and tuple(map(tuple, goals)) == tuple(inst.goals)
                             and raw["generator"]["seed"] == spec["seed"])
    if not out["coordinates_ok"]:
        problems.append(f"{spec['slug']}: coordinate round trip differs")
    out["loader"] = loader_checks(scen_path, inst)
    if not all(out["loader"][k] for k in ("agent_id_matches_index", "run_il_load_scenario_matches_generator",
                                          "t0_observation_shapes_ok")):
        problems.append(f"{spec['slug']}: loader/agent-id/observation check failed")
    out["actions"] = action_checks(grid, starts, goals)
    if not out["actions"]["ok"]:
        problems.append(f"{spec['slug']}: action convention check failed")

    metrics = agent_metrics(grid, starts, goals)
    out.update(metrics)
    if not metrics["all_goals_reachable"]:
        problems.append(f"{spec['slug']}: a goal is unreachable by BFS")
    out["width1_cells"] = width1_cells(grid)
    out["articulation_cells"] = articulation_cells(grid)

    il = check_instance(grid, starts, goals, reference)
    ood = classify_ood(il, reference)
    out["il_check"] = il
    out["ood"] = ood
    feats = reference["features"]
    out["il_reference_bounds"] = {k: {"p0.5": feats[k]["p0.5"], "p99.5": feats[k]["p99.5"],
                                      "min": feats[k]["min"], "max": feats[k]["max"]} for k in feats}
    out["prediction_vs_result"] = {
        "ood_class": {"predicted": spec["predicted"]["ood_class"], "result": ood["ood_class"]},
        "width1_cells": {"predicted": spec["predicted"]["width1_cells"], "result": len(out["width1_cells"])},
        "all_goals_reachable": {"predicted": spec["predicted"]["all_goals_reachable"],
                                "result": metrics["all_goals_reachable"]},
        "free_cells": {"predicted": spec["free_cells"], "result": out["free_cells"]},
    }

    fp = fingerprint_scenario(scen_path)
    fp.update(structure_type=inst.structure_type, structure_descriptor=inst.descriptor)
    out["fingerprint"] = fp
    return out


def registry_checks(result: Dict, registry: Registry, prereg: Dict) -> Dict:
    fp = result["fingerprint"]
    rels = [dict(rel, other=rec["instance_id"], other_family=rec["family_id"], other_split=rec["split"])
            for rec in registry.records for rel in relate(rec, fp)]
    decision = registry.evaluate(fp, SPLIT)
    splits = sorted({rec["split"] for rec in registry.records})
    holds = [{"candidate_id": result["candidate_id"], "other": r["other"], "other_split": r["other_split"],
              "other_family": r["other_family"], "relation": r["relation"], "relation_status": r["status"],
              "status": HOLD, "final_decision": None, "decision_owner": None,
              "basis": "Recomputed by registry.relate; not decided here."} for r in rels]
    test_side = [h for h in holds if h["other_split"] in ("VAL", "TEST")]
    struct = [h for h in holds if h["relation"] == "STRUCT_MATCH"]
    return {
        "split_evaluated_in_memory": SPLIT,
        "instance_fp": instance_fp(fp),
        "map_canon_fp": fp["map_canon_fp"],
        "duplicate_instance_fp": any(instance_fp(rec) == instance_fp(fp) for rec in registry.records),
        "relations_existing": rels,
        "registry_evaluate_in_memory": {"status": decision.status,
                                        "proposed_family_id_unregistered": decision.family_id,
                                        "reasons": decision.reasons},
        "registry_splits_present": splits,
        "holds": holds,
        "test_conflict_risk": {
            "status": HOLD if (test_side or struct) else "NONE_FOUND_NOT_APPROVED",
            "relations_to_val_or_test": test_side,
            "struct_match_relations": struct,
            "note": ("No VAL/TEST record exists yet; a STRUCT_MATCH to another domain would be forbidden "
                     "for TEST later (docs/fingerprint_registry_v1.md 4). Nothing is approved here."),
        },
        "independence_approved": False,
        "predicted_automatic_relations": prereg["candidates"][0]["predicted"]["automatic_relations"],
    }


def build_report(root: Path, registry_dir: Path) -> Dict:
    prereg = load_prereg()
    registry = Registry.load(registry_dir / INSTANCES)
    reference = load_reference(registry_dir / REFERENCE)
    results = [check_candidate(spec, root, reference) for spec in prereg["candidates"]]
    for r in results:
        if r["fingerprint"] is not None:
            r["registry"] = registry_checks(r, registry, prereg)
    return {
        "prereg_sha256": PREREG_SHA256,
        "generator_v3_sha256": sha256_file(GENERATOR_V3_FILE),
        "generator_v1_sha256": sha256_file(GENERATOR_V1_FILE),
        "registry_instances_sha256": registry.source_sha256,
        "registry_records": len(registry.records),
        "reference_sha256": reference["source"]["sha256"],
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "cbs_run": False,
        "il_episode_run": False,
        "candidates": results,
        "problems": [p for r in results for p in r["problems"]],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("generate", "check"))
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--registry-dir", default=str(PROJECT_ROOT / "registry"))
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    args = ap.parse_args(argv)
    root, registry_dir = Path(args.root), Path(args.registry_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.command == "generate":
        try:
            records = generate(root)
        except (OSError, ValueError, RegistryError) as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 2
        for rec in records:
            print(f"{rec['candidate_id']} {rec['slug']} generated={rec['generated']} "
                  f"already_present={rec['already_present']} error={rec['error']}")
        return 0 if all(r["generated"] for r in records) else 1

    out_dir = Path(args.out_root) / f"static_{stamp}"
    if out_dir.exists():
        print(f"REFUSED: output directory exists: {out_dir}", file=sys.stderr)
        return 2
    try:
        before = sha256_file(registry_dir / INSTANCES)
        report = build_report(root, registry_dir)
    except (OSError, ValueError, RegistryError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    problems = report["problems"]
    if sha256_file(registry_dir / INSTANCES) != before:
        problems.append("registry instances file changed during the check")
    out_dir.mkdir(parents=True, exist_ok=False)
    with open(out_dir / "static_checks.json", "x", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    for r in report["candidates"]:
        reg = r.get("registry", {})
        print(f"{r['candidate_id']} {r['slug']} reachable={r.get('all_goals_reachable')} "
              f"il={r.get('il_check', {}).get('status')} ood={r.get('ood', {}).get('ood_class')} "
              f"relations={len(reg.get('relations_existing', []))} "
              f"test_risk={reg.get('test_conflict_risk', {}).get('status')}")
    print(f"wrote {out_dir / 'static_checks.json'}")
    for p in problems:
        print(f"PROBLEM: {p}", file=sys.stderr)
    print("no integrity problems" if not problems else f"{len(problems)} problem(s)")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
