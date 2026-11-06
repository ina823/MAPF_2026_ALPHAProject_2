"""Register the 16 existing Development scenarios in a NEW fingerprint registry.

Read-only towards scenarios/: every scenario JSON and map .npy is hashed
before and after, and the run fails if any byte changed. Writes, only if
neither exists yet (never overwrites, never leaves a partial file):

    <registry-dir>/instances_v1.jsonl   one record per instance
    <registry-dir>/families_v1.json     family counts, relations, REVIEW items

Exit codes: 0 registered, 1 a scenario was not ACCEPTed or a source changed,
2 refused before writing (registry exists / scenario set differs).

Usage (from the repository root):
    python -m scripts.register_dev_scenarios [--registry-dir registry] [--registered-at YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fingerprint import fingerprint_scenario, sha256_file
from src.data.registry import ACCEPT, Registry, RegistryError, create_files_exclusive

SCENARIO_DIRS = ("controlled", "controlled_variants", "experimental_v1", "smoke")

# role / usage status per docs/data_split_plan_v1.md section 1 (file stem -> (dev_role, usage_status))
DEV_SCENARIOS = {
    "controlled/scenario_corridor_bay_n2": ("mechanism_demo", "DEV_TUNED"),
    "controlled/scenario_intersection_n2": ("mechanism_demo", "DEV_TUNED"),
    "controlled/scenario_bottleneck_n4": ("mechanism_demo", "DEV_TUNED"),
    "controlled/scenario_cycle_ring_n4": ("negative_control", "REGRESSION"),
    "controlled_variants/scenario_corridor_bay_n2_offset": ("variant", "DEV"),
    "controlled_variants/scenario_intersection_n2_4way": ("variant", "DEV"),
    "controlled_variants/scenario_intersection_n2_lshape": ("variant", "DEV"),
    "controlled_variants/scenario_intersection_n2_tight": ("variant", "DEV"),
    "controlled_variants/scenario_bottleneck_n4_tightdoor": ("variant", "DEV"),
    "controlled_variants/scenario_corridor_bay_n2_3agent": ("variant_ki1", "HOLD"),
    "controlled_variants/scenario_corridor_bay_n2_4agent": ("variant_ki1", "HOLD"),
    "controlled_variants/scenario_bottleneck_n4_convoy": ("variant_ki1", "HOLD"),
    "controlled_variants/scenario_bottleneck_n4_imbalanced": ("variant_ki1", "HOLD"),
    "experimental_v1/scenario_corridor_bay_n2_3rdagent_perturbation": ("adhoc_experiment", "DEV"),
    "experimental_v1/scenario_bottleneck_n4_tightdoor_1agent_control": ("adhoc_experiment", "DEV"),
    "smoke/scenario_empty_s11_n3": ("regression_anchor", "REGRESSION"),
}

STRUCTURE_BY_MAP_ID = {
    "corridor_bay_n2": "corridor",
    "intersection_n2": "intersection",
    "bottleneck_n4": "bottleneck",
    "cycle_ring_n4": "cycle_ring",
    "empty_s11_n3": "empty",
}

INSTANCES = "instances_v1.jsonl"
FAMILIES = "families_v1.json"


def scenario_paths(root: Path = PROJECT_ROOT) -> list:
    """The 16 Development scenario files, in a fixed order. Raises if the set differs."""
    found = sorted(p for d in SCENARIO_DIRS for p in (root / "scenarios" / d).glob("scenario_*.json"))
    found_keys = {p.relative_to(root / "scenarios").with_suffix("").as_posix() for p in found}
    if found_keys != set(DEV_SCENARIOS):
        raise RegistryError(f"scenario set differs from the 16 Development files: "
                            f"extra={sorted(found_keys - set(DEV_SCENARIOS))} "
                            f"missing={sorted(set(DEV_SCENARIOS) - found_keys)}")
    return [root / "scenarios" / f"{k}.json" for k in DEV_SCENARIOS]


def source_hashes(paths) -> dict:
    out = {}
    for p in paths:
        rec = fingerprint_scenario(p)
        out[rec["source_path"]] = rec["scenario_json_sha256"]
        out[rec["map_file_path"]] = sha256_file(PROJECT_ROOT / rec["map_file_path"])
    return out


def build_dev_registry(registered_at: str, root: Path = PROJECT_ROOT) -> Registry:
    reg = Registry()
    for path in scenario_paths(root):
        key = path.relative_to(root / "scenarios").with_suffix("").as_posix()
        dev_role, usage_status = DEV_SCENARIOS[key]
        cand = fingerprint_scenario(path)
        cand["structure_type"] = STRUCTURE_BY_MAP_ID[cand["map_id"]]
        cand["structure_descriptor"] = None
        cand["generator"] = None
        # legacy import: the family cap (registry.FAMILY_CAP) is not applied retroactively
        decision = reg.evaluate(cand, "DEV", apply_family_cap=False)
        if decision.status != ACCEPT:
            raise RegistryError(f"{key}: {decision.status} {decision.reasons}")
        reg.register(cand, "DEV", registered_at=registered_at, apply_family_cap=False,
                     extra={"dev_role": dev_role, "usage_status": usage_status,
                            "role_source": "docs/data_split_plan_v1.md section 1"})
    return reg


def print_summary(summary: dict) -> None:
    print(f"instances={summary['instance_count']} splits={summary['split_counts']} "
          f"families={summary['family_count']}")
    for fid, fam in summary["families"].items():
        print(f"  {fid:8s} count={fam['count']:2d} {fam['structure_types']} {fam['splits']}")
    print(f"relations={summary['relation_counts']}")
    print(f"REVIEW items={len(summary['review_items'])} "
          f"unresolved={summary['unresolved_review_count']}")
    for item in summary["review_items"]:
        print(f"  {item['relation']}: {item['a']} / {item['b']} {item['details']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--registry-dir", default=str(PROJECT_ROOT / "registry"))
    ap.add_argument("--registered-at", default=date.today().isoformat())
    args = ap.parse_args(argv)
    out_dir = Path(args.registry_dir)
    targets = [out_dir / INSTANCES, out_dir / FAMILIES]
    existing = [str(t) for t in targets if t.exists()]
    if existing:
        print(f"REFUSED: registry file(s) already exist: {existing}", file=sys.stderr)
        return 2
    try:
        paths = scenario_paths()
    except RegistryError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    before = source_hashes(paths)
    try:
        reg = build_dev_registry(args.registered_at)
    except RegistryError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    after = source_hashes(paths)
    if before != after:
        print("FAILED: a scenario or map file changed during registration", file=sys.stderr)
        return 1

    create_files_exclusive([(targets[0], reg.to_jsonl()), (targets[1], reg.summary_json())])
    print(f"sources unchanged: {len(before)} files (16 scenario JSON + 5 maps)")
    print_summary(reg.family_summary())
    print(f"wrote {targets[0]} and {targets[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
