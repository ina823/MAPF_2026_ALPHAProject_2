"""PILOT data-quality gate v0: print per-PILOT flags. Read-only.

Loads the registry, the review ledger, the solvability record and the IL
input reference; recomputes the OOD check for each PILOT scenario; checks
that the ledger and solvability record agree with the registry and with the
raw CBS check output they cite. Runs no solver and writes no file.

Flags (gate_status FLAGGED = needs a human look; the gate never rejects):
  REVIEW_HOLD            a REVIEW relation is recorded as HOLD in the ledger
  OOD_TAIL / OOD_BEYOND_OBSERVED   IL input outside [p0.5, p99.5] / training [min, max]
  CBS_<STATUS>           the CBS adapter did not return success
  WITNESS_UNARCHIVED     solution existence rests on a non-CBS plan with no saved file

Exit codes: 0 records consistent (flags are reported, not judged), 1 problems.

Usage (from the repository root):
    python -m scripts.check_pilot_gate [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.scenario_loader import load_map_generator_scenario
from src.data.il_compat import check_instance, load_reference
from src.data.quality_gate import GateError, classify_ood, gate_report, load_jsonl
from src.data.registry import Registry, RegistryError
from scripts.register_dev_scenarios import INSTANCES

LEDGER = "review_ledger_v0.jsonl"
SOLVABILITY = "pilot_solvability_v0.json"
REFERENCE = "il_input_reference_v1.json"


def build_report(registry_dir: Path, root: Path = PROJECT_ROOT) -> dict:
    try:
        reg = Registry.load(registry_dir / INSTANCES)
        ledger = load_jsonl(registry_dir / LEDGER)
        solv = json.loads((registry_dir / SOLVABILITY).read_text(encoding="utf-8"))
        reference = load_reference(registry_dir / REFERENCE)
    except (OSError, ValueError, RegistryError, GateError) as exc:
        return {"problems": [f"cannot load gate inputs: {exc}"], "notes": [], "instances": []}
    ood = {}
    for r in reg.records:
        if r["split"] == "PILOT":
            grid, starts, goals = load_map_generator_scenario(root / r["source_path"])
            ood[r["instance_id"]] = classify_ood(check_instance(grid, starts, goals, reference), reference)
    return gate_report(reg, ledger, solv, ood, root=root)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--registry-dir", default=str(PROJECT_ROOT / "registry"))
    ap.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = ap.parse_args(argv)
    rep = build_report(Path(args.registry_dir))
    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        for i in rep["instances"]:
            s = i["solvability"] or {}
            print(f"{i['instance_id']:18s} {i['gate_status']:7s} flags={i['flags']}")
            print(f"    review_holds={i['review_holds']}")
            if i["ood"]:
                print(f"    ood={i['ood']['ood_class']} causes={i['ood']['causes']} auto_fail={i['ood']['auto_fail']}")
            print(f"    cbs_status={s.get('cbs_status')} cbs_solved={s.get('cbs_solved')} "
                  f"solution_existence={s.get('solution_existence')} witness={s.get('witness_evidence')}")
    for n in rep["notes"]:
        print(f"NOTE: {n}")
    for p in rep["problems"]:
        print(f"PROBLEM: {p}", file=sys.stderr)
    print("gate records consistent" if not rep["problems"] else f"{len(rep['problems'])} problem(s)")
    return 0 if not rep["problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
