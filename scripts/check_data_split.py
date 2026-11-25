"""Data split status and leakage check (read-only over scenarios/, registry/, outputs/, docs/).

Lists every known instance (scenario files + in-memory DEV instances of the
DEV scripts) with its data role and evidence, compares every pair with the
registry rules (EXACT / RELABEL / TRIM / SYMMETRY / SAME_MAP confirmed,
SUBMAP / NEAR_MAP / STRUCT_MATCH for review), groups seed-only re-runs, and
reports how many Validation / Test instances are still missing against the
PROPOSED targets of docs/data_split_plan_v1.md 6.1 (not agreed numbers).

Nothing is moved, re-labelled or registered. The report goes to a NEW file
    <out-root>/<YYYYmmdd_HHMMSS>/data_split_report.json
and is never overwritten.

Exit codes: 0 no error and no cross-role leak, 1 problems listed.

Usage (repository root):
    python -m scripts.check_data_split [--out-root outputs/data_split_checks]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.inventory import TEST, VALIDATION, build_report
from src.data.registry import create_files_exclusive
from src.data.split_policy import held_out_shortfall

PROPOSED_TARGETS = {VALIDATION: 45, TEST: 75}   # docs/data_split_plan_v1.md 6.1: proposal, not agreed
REPORT = "data_split_report.json"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "outputs" / "data_split_checks")
    ap.add_argument("--root", type=Path, default=PROJECT_ROOT, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    report = build_report(args.root)
    report["created"] = datetime.now().isoformat(timespec="seconds")
    report["held_out_shortfall"] = held_out_shortfall(report["role_counts"], PROPOSED_TARGETS)
    report["held_out_targets_note"] = "targets are the PROPOSED numbers of docs/data_split_plan_v1.md 6.1"
    out = args.out_root / f"{datetime.now():%Y%m%d_%H%M%S}" / REPORT
    create_files_exclusive([(out, (json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n")
                             .encode("utf-8"))])

    print(f"records     : {report['record_count']} {report['kind_counts']}")
    print(f"data roles  : {report['role_counts']}")
    print(f"pairs       : {report['pair_counts']}")
    print(f"role pairs  : {report['role_pair_counts']}")
    unresolved = sum(1 for r in report["manual_reviews"] if r["unresolved"])
    print(f"reviews     : {len(report['manual_reviews'])} recorded, {unresolved} unresolved")
    print(f"seed groups : {sum(1 for s in report['seed_sweeps'] if s['seed_sweep'])} seed sweeps over "
          f"{len(report['seed_sweeps'])} re-run instance(s)")
    for role, s in report["held_out_shortfall"].items():
        print(f"{role:11s} : have {s['have']}, proposed {s['target']}, missing {s['missing']}")
    for e in report["errors"]:
        print(f"ERROR: {e}")
    for leak in report["leaks"]:
        print(f"LEAK: {leak['role_pair']} {leak['relation']} {leak['a']} / {leak['b']}")
    print(f"report      : {out}")
    return 1 if report["errors"] or report["leaks"] else 0


if __name__ == "__main__":
    sys.exit(main())
