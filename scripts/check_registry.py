"""Verify a fingerprint registry against the files it describes. Read-only.

Checks: every record loads and has the current fp_schema; each source file
still exists with the recorded sha256; fingerprints recomputed from the file
equal the stored ones; relations recomputed from fingerprints equal the
stored ones; confirmed relations stay inside one family and one domain
(development / validation / test); no unresolved REVIEW touches VAL/TEST;
families_v1.json equals the summary regenerated from instances_v1.jsonl.

Exit codes: 0 consistent, 1 problems found (listed on stderr).

Usage (from the repository root):
    python -m scripts.check_registry [--registry-dir registry]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fingerprint import FingerprintError, fingerprint_scenario
from src.data.registry import Registry, RegistryError
from scripts.register_dev_scenarios import FAMILIES, INSTANCES, print_summary

# fields recomputed from the source file and compared with the stored record
RECOMPUTED_FIELDS = (
    "scenario_json_sha256", "map_file_path", "map_file_sha256", "scenario_id", "map_id",
    "agent_ids_in_file", "agent_id_matches_index", "fp_schema", "grid_shape", "free_cells",
    "num_agents", "starts_rc", "goals_rc", "grid_fp", "grid_trim_fp", "grid_trim_offset",
    "grid_trim_shape", "grid_trim_rows", "map_canon_fp", "map_canon_transforms",
    "agents_ordered_fp", "agents_set_fp", "agents_trim_ordered_fp", "agents_trim_set_fp",
    "agents_canon_ordered_fp", "agents_canon_set_fp",
)


def check(registry_dir: Path, root: Path = PROJECT_ROOT):
    """Return (registry or None, list of problems)."""
    problems = []
    try:
        reg = Registry.load(registry_dir / INSTANCES)
    except (OSError, RegistryError) as exc:
        return None, [f"cannot load {registry_dir / INSTANCES}: {exc}"]

    for rec in reg.records:
        src = root / rec["source_path"]
        if not src.is_file():
            problems.append(f"{rec['instance_id']}: source missing {rec['source_path']}")
            continue
        try:
            fresh = fingerprint_scenario(src)
        except (OSError, FingerprintError, KeyError, ValueError) as exc:
            problems.append(f"{rec['instance_id']}: cannot fingerprint source ({exc})")
            continue
        for f in RECOMPUTED_FIELDS:
            if fresh.get(f) != rec.get(f):
                problems.append(f"{rec['instance_id']}: {f} differs from the source file")

    problems += reg.consistency_errors()

    fam_path = registry_dir / FAMILIES
    if not fam_path.is_file():
        problems.append(f"missing {fam_path}")
    elif fam_path.read_bytes() != reg.summary_json():
        problems.append(f"{fam_path} differs from the summary regenerated from {INSTANCES}")
    return reg, problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--registry-dir", default=str(PROJECT_ROOT / "registry"))
    args = ap.parse_args(argv)
    reg, problems = check(Path(args.registry_dir))
    if reg is not None:
        print_summary(reg.family_summary())
    for p in problems:
        print(f"PROBLEM: {p}", file=sys.stderr)
    print("registry consistent" if not problems else f"{len(problems)} problem(s)")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
