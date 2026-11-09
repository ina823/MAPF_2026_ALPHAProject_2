"""Generate the pre-registered PILOT v2 candidates. Never touches the registry.

Candidates come only from registry/pilot_v2_prereg_v0.json, whose sha256 is
pinned below: a changed pre-registration refuses generation. Each candidate is
generated twice and must be byte-identical and match the pre-registered grid,
seed and agent count. Files are created exclusively under scenarios/pilot_v2/
(existing files are refused, nothing is overwritten) and read back through
scenario_loader. A failed candidate is recorded, not dropped; the others are
still written. The record goes to a NEW directory
outputs/pilot_v2_checks/generation_<YYYYmmdd_HHMMSS>/.

Exit codes: 0 all generated, 1 a candidate failed, 2 refused before writing.

Usage (from the repository root):
    python -m scripts.generate_pilot_v2_candidates [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import generator_v2 as g2
from src.data.fingerprint import fingerprint_scenario, grid_rows, normalize_grid, sha256_file
from src.data.generator import GeneratorError
from src.data.registry import RegistryError, create_files_exclusive

PREREG_PATH = PROJECT_ROOT / "registry" / "pilot_v2_prereg_v0.json"
PREREG_SHA256 = "67a748efad4d435acb3a69bf61ed8d6a6e018de5f6e61fdf235b29059ec778af"
GENERATOR_V2_FILE = PROJECT_ROOT / "src" / "data" / "generator_v2.py"
GENERATOR_V1_FILE = PROJECT_ROOT / "src" / "data" / "generator.py"
PILOT_V2_DIR = "scenarios/pilot_v2"
OUT_ROOT = PROJECT_ROOT / "outputs" / "pilot_v2_checks"


def load_prereg(path: Path = PREREG_PATH, pinned: str = PREREG_SHA256) -> dict:
    """The pre-registration, only if it still has the pinned sha256."""
    raw = Path(path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != pinned:
        raise RegistryError(f"{path}: sha256 {actual} differs from the pinned {pinned}")
    return json.loads(raw)


def candidate_paths(spec: dict, root: Path = PROJECT_ROOT):
    out = Path(root) / PILOT_V2_DIR
    return out / f"map_{spec['slug']}.npy", out / f"scenario_{spec['slug']}.json"


def spec_problems(spec: dict, inst) -> list:
    """Differences between a generated instance and its pre-registration."""
    problems = []
    if grid_rows(normalize_grid(inst.grid)) != spec["grid_rows"]:
        problems.append("grid differs from the pre-registered grid_rows")
    if len(inst.starts) != spec["num_agents"]:
        problems.append(f"{len(inst.starts)} agents, pre-registered {spec['num_agents']}")
    if inst.seed != spec["seed"]:
        problems.append(f"seed {inst.seed}, pre-registered {spec['seed']}")
    if inst.structure_type != spec["structure_type"]:
        problems.append(f"structure_type {inst.structure_type}, pre-registered {spec['structure_type']}")
    for start in spec.get("fixed_starts", {}).values():
        if tuple(start) not in inst.starts:
            problems.append(f"fixed start {start} is not among the starts")
    return problems


def generate_candidate(spec: dict, prereg_sha: str = PREREG_SHA256):
    """(instance, map bytes, scenario bytes); regenerates and compares bytes."""
    code_sha, base_sha = sha256_file(GENERATOR_V2_FILE), sha256_file(GENERATOR_V1_FILE)

    def once():
        inst = g2.build(spec["builder"], spec["params"], spec["seed"])
        return inst, g2.scenario_files(inst, spec["slug"], code_sha256=code_sha, base_generator_sha256=base_sha,
                                       prereg_sha256=prereg_sha, candidate_id=spec["candidate_id"])
    inst, first = once()
    _, second = once()
    if first != second:
        raise GeneratorError(f"{spec['slug']}: same-seed regeneration is not byte-identical")
    return inst, first[0], first[1]


def generate_all(specs, root: Path = PROJECT_ROOT, *, write: bool = True, prereg_sha: str = PREREG_SHA256):
    """One record per candidate, failed ones included. Writes the good ones if ``write``."""
    records = []
    for spec in specs:
        rec = {"candidate_id": spec["candidate_id"], "slug": spec["slug"], "grade": spec["grade"],
               "seed": spec["seed"], "generated": False, "error": None}
        records.append(rec)
        try:
            inst, npy, js = generate_candidate(spec, prereg_sha)
            problems = spec_problems(spec, inst)
            if problems:
                raise GeneratorError("; ".join(problems))
            rec.update(map_sha256=hashlib.sha256(npy).hexdigest(), scenario_sha256=hashlib.sha256(js).hexdigest())
            if write:
                map_path, scen_path = candidate_paths(spec, root)
                create_files_exclusive([(map_path, npy), (scen_path, js)])
                fresh = fingerprint_scenario(scen_path)
                if fresh["scenario_json_sha256"] != rec["scenario_sha256"] or fresh["map_file_sha256"] != rec["map_sha256"]:
                    raise GeneratorError("written files read back differently")
                rec.update(source_path=fresh["source_path"], map_file_path=fresh["map_file_path"])
            rec["generated"] = True
        except (GeneratorError, RegistryError, OSError, KeyError, TypeError, ValueError) as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
    return records


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="print file hashes only; nothing is written")
    ap.add_argument("--root", default=str(PROJECT_ROOT), help="directory that receives scenarios/pilot_v2/")
    ap.add_argument("--out-root", default=str(OUT_ROOT), help="directory that receives generation_<time>/")
    args = ap.parse_args(argv)
    root = Path(args.root)
    try:
        prereg = load_prereg()
    except (OSError, ValueError, RegistryError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    specs = prereg["candidates"]

    if args.dry_run:
        records = generate_all(specs, root, write=False)
        print(json.dumps([{k: r.get(k) for k in ("candidate_id", "generated", "error", "map_sha256", "scenario_sha256")}
                          for r in records], indent=2, sort_keys=True))
        return 0 if all(r["generated"] for r in records) else 1

    existing = [str(p) for spec in specs for p in candidate_paths(spec, root) if p.exists()]
    out_dir = Path(args.out_root) / f"generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if existing or out_dir.exists():
        print(f"REFUSED: already exists: {existing or out_dir}", file=sys.stderr)
        return 2

    records = generate_all(specs, root)
    out_dir.mkdir(parents=True, exist_ok=False)
    report = {"prereg_sha256": PREREG_SHA256, "generator_v2_sha256": sha256_file(GENERATOR_V2_FILE),
              "generator_v1_sha256": sha256_file(GENERATOR_V1_FILE), "generator": g2.GENERATOR_NAME,
              "generator_version": g2.GENERATOR_VERSION, "records": records}
    with open(out_dir / "generation.json", "x", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    for r in records:
        print(f"{r['candidate_id']} {r['slug']:30s} {r['grade']:6s} seed={r['seed']} "
              f"generated={r['generated']} error={r['error']}")
    print(f"wrote {out_dir / 'generation.json'}")
    return 0 if all(r["generated"] for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())
