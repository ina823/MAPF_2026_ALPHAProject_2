"""Export saved Global Hybrid runs to hybrid_episodes.csv + hybrid_cbs_calls.csv.

Reads outputs/hybrid_e2e/<run_id>/result.json and <run_id>_steps.csv (never
writes to them), validates every run, then writes into a NEW directory:
    <output-dir>/hybrid_episodes.csv
    <output-dir>/hybrid_cbs_calls.csv
    <output-dir>/export_manifest.json   -- inputs + sha256, value-check count
Field definitions: src/hybrid/hybrid_csv.py. Refuses an existing output dir;
writes nothing if any run fails validation.

Usage (from the repository root):
    python -m scripts.export_hybrid_csv \
        --run-dir outputs/hybrid_e2e/<run_id_1> --run-dir outputs/hybrid_e2e/<run_id_2>
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid.hybrid_csv import HybridCsvError, export_runs

DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs" / "csv"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, action="append", required=True,
                    help="a saved Global Hybrid run directory (repeat for several runs)")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="new directory for the CSVs (default: outputs/csv/hybrid_export_<timestamp>)")
    args = ap.parse_args(argv)

    output_dir = args.output_dir or DEFAULT_OUT_ROOT / f"hybrid_export_{datetime.now():%Y%m%d_%H%M%S}"
    if output_dir.exists():
        print(f"ERROR: output directory already exists, refusing to overwrite: {output_dir}")
        return 2
    command = " ".join([sys.executable, "-m", "scripts.export_hybrid_csv", *argv])
    try:
        out = export_runs(args.run_dir, output_dir, command=command)
    except HybridCsvError as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"output_dir : {out['output_dir']}")
    for ep in out["episodes"]:
        print(f"  {ep['run_id']}: {ep['termination']} verdict={ep['verdict']} "
              f"policy={ep['h_exec_policy']} h_exec={ep['h_exec_config']} "
              f"trigger={ep['trigger_timestep']} cbs_steps={ep['cbs_exec_steps']} "
              f"agent_steps={ep['cbs_controlled_agent_steps']}")
    print(f"cbs calls  : {len(out['calls'])} rows")
    print(f"JSON<->CSV value checks passed: {out['checks']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
