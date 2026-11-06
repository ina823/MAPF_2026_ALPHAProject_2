"""Generate the 3 PILOT instances and append them to the registry as split=PILOT.

PILOT = Development-side pilot data (docs/data_split_plan_v1.md 6.4): it is
never moved to Validation/Test, and its families are not used there.

Steps (all-or-nothing):
  1. generate every pilot twice and require byte-identical files;
  2. validate cells (bounds, walls, distinct starts/goals, goal reachable);
  3. evaluate against the registry as PILOT: must be ACCEPT with no confirmed
     relation (each pilot must be a NEW family); REVIEW relations are recorded;
  4. write map .npy + scenario JSON into a NEW directory (never overwrites);
  5. re-read the written files through scenario_loader and require the same
     fingerprints as in memory (coordinate round trip);
  6. append the records to instances_v1.jsonl (existing lines byte-identical)
     and rewrite families_v1.json. Any failure after step 4 removes the new
     files and restores the registry.

Exit codes: 0 done (or dry run ok), 1 a check failed, 2 refused before writing.

Usage (from the repository root):
    python -m scripts.generate_pilot_instances [--dry-run] [--registered-at YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import generator as gen
from src.data.fingerprint import (
    _repo_relative, compute_fingerprints, fingerprint_scenario, sha256_file,
)
from src.data.registry import (
    ACCEPT, CONFIRMED, Registry, RegistryError, create_files_exclusive, replace_file_checked,
)
from scripts.check_registry import RECOMPUTED_FIELDS
from scripts.register_dev_scenarios import FAMILIES, INSTANCES, print_summary

PILOT_DIR = "scenarios/pilot_v1"
GENERATOR_FILE = PROJECT_ROOT / "src" / "data" / "generator.py"

# slug -> (builder, kwargs). Seeds are in the PILOT range 0-99.
PILOT_SPECS = {
    "pilot_corridor_bays_l15": (gen.corridor_bays, dict(
        length=15, bays_up=(5, 9), bays_down=(), n_east=2, n_west=1, seed=11)),
    "pilot_rooms_door_6x14": (gen.rooms_door, dict(
        room_h=6, room_w=6, door_len=2, door_row=2, n_east=2, n_west=2, seed=12)),
    "pilot_crossing_9x11": (gen.crossing, dict(
        height=9, width=11, corridor_rows=(2, 6), corridor_cols=(5,), seed=13)),
}


def generate(slug: str):
    builder, kwargs = PILOT_SPECS[slug]
    return builder(**kwargs)


def pilot_files(slug: str, root: Path = PROJECT_ROOT):
    """(instance, map path, map bytes, scenario path, scenario bytes) for one pilot."""
    inst = generate(slug)
    code_sha = sha256_file(GENERATOR_FILE)
    npy, js = gen.scenario_files(inst, slug, code_sha)
    again_npy, again_js = gen.scenario_files(generate(slug), slug, code_sha)
    if (npy, js) != (again_npy, again_js):
        raise RegistryError(f"{slug}: regeneration with the same seed is not byte-identical")
    out = Path(root) / PILOT_DIR
    return inst, out / f"map_{slug}.npy", npy, out / f"scenario_{slug}.json", js


def candidate_record(slug: str, inst, map_path: Path, npy: bytes, scen_path: Path, js: bytes) -> dict:
    rec = compute_fingerprints(inst.grid, inst.starts, inst.goals)
    rel = _repo_relative          # same path form as fingerprint_scenario()
    rec.update({
        "source_path": rel(scen_path),
        "scenario_json_sha256": hashlib.sha256(js).hexdigest(),
        "map_file_path": rel(map_path),
        "map_file_sha256": hashlib.sha256(npy).hexdigest(),
        "scenario_id": f"scen_{slug}",
        "map_id": slug,
        "agent_ids_in_file": list(range(len(inst.starts))),
        "agent_id_matches_index": True,
        "structure_type": inst.structure_type,
        "structure_descriptor": inst.descriptor,
        "generator": {"name": gen.GENERATOR_NAME, "version": gen.GENERATOR_VERSION,
                      "seed": inst.seed, "code_sha256": sha256_file(GENERATOR_FILE)},
        "dev_role": "pilot",
        "usage_status": "PILOT",
        "role_source": "docs/data_split_plan_v1.md section 6.4",
    })
    return rec


def plan(reg: Registry, registered_at: str, root: Path = PROJECT_ROOT):
    """Generate + evaluate every pilot in memory. Returns [(slug, record, files)]."""
    planned = []
    for slug in PILOT_SPECS:
        inst, map_path, npy, scen_path, js = pilot_files(slug, root)
        cand = candidate_record(slug, inst, map_path, npy, scen_path, js)
        decision = reg.evaluate(cand, "PILOT")
        if decision.status != ACCEPT:
            raise RegistryError(f"{slug}: {decision.status} {decision.reasons}")
        confirmed = [r for r in decision.relations if r["status"] == CONFIRMED]
        if confirmed:
            raise RegistryError(f"{slug}: must be a new family, but has {confirmed}")
        rec = reg.register(cand, "PILOT", registered_at=registered_at)
        planned.append((slug, rec, [(map_path, npy), (scen_path, js)]))
    return planned


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--registry-dir", default=str(PROJECT_ROOT / "registry"))
    ap.add_argument("--registered-at", default=date.today().isoformat())
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--root", default=str(PROJECT_ROOT),
                    help="directory that receives scenarios/pilot_v1/ (tests use a temporary one)")
    args = ap.parse_args(argv)
    reg_dir = Path(args.registry_dir)
    root = Path(args.root)
    inst_path, fam_path = reg_dir / INSTANCES, reg_dir / FAMILIES

    try:
        reg = Registry.load(inst_path)
    except (OSError, RegistryError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    if any(r["split"] == "PILOT" for r in reg.records):
        print("REFUSED: the registry already has PILOT records", file=sys.stderr)
        return 2
    n_existing = len(reg.records)
    old_inst = inst_path.read_bytes()
    old_fam = fam_path.read_bytes()

    try:
        planned = plan(reg, args.registered_at, root)
    except (RegistryError, gen.GeneratorError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    for slug, rec, files in planned:
        rels = [(r["relation"], r["status"], r["other"]) for r in rec["relations"]]
        print(f"{rec['instance_id']:24s} {slug:26s} grid={rec['grid_shape']} agents={rec['num_agents']} "
              f"seed={rec['generator']['seed']} relations={rels or 'none'}")
    if args.dry_run:
        print("dry run: nothing written")
        return 0

    files = [f for _, _, fs in planned for f in fs]
    existing = [str(p) for p, _ in files if p.exists()]
    if existing:
        print(f"REFUSED: pilot file(s) already exist: {existing}", file=sys.stderr)
        return 2
    create_files_exclusive(files)
    try:
        for slug, rec, fs in planned:
            fresh = fingerprint_scenario(fs[1][0])
            diff = [f for f in RECOMPUTED_FIELDS if fresh.get(f) != rec.get(f)]
            if diff:
                raise RegistryError(f"{slug}: written files re-read differently in {diff}")
        reg.append_to_file(inst_path, n_existing)
        try:
            replace_file_checked(fam_path, reg.summary_json(), hashlib.sha256(old_fam).hexdigest())
        except BaseException:
            replace_file_checked(inst_path, old_inst, hashlib.sha256(reg.to_jsonl()).hexdigest())
            raise
    except BaseException as exc:
        for p, _ in files:
            if p.exists():
                p.unlink()
        pilot_dir = root / PILOT_DIR
        if pilot_dir.is_dir() and not any(pilot_dir.iterdir()):
            pilot_dir.rmdir()
        print(f"FAILED (rolled back): {exc}", file=sys.stderr)
        return 1

    if inst_path.read_bytes()[:len(old_inst)] != old_inst:
        print("FAILED: existing registry lines changed", file=sys.stderr)
        return 1
    print_summary(reg.family_summary())
    print(f"wrote {len(files)} files under {PILOT_DIR}/ and appended {len(planned)} PILOT records")
    return 0


if __name__ == "__main__":
    sys.exit(main())
