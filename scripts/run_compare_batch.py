"""DEV: Development comparison batch -- scenarios x methods through one runner.

Runs every (scenario, method) pair with src/compare/common_runner.run_method,
sequentially in one process (the CBS bridge switches the cwd), then writes

    <out-dir>/runs/<run_id>/            one run directory per pair
    <out-dir>/compare_episodes.csv      one row per run
    <out-dir>/compare_cbs_calls.csv     one row per CBS call
    <out-dir>/batch_manifest.json       conditions, order, sha256, fairness

Refused before any run: an unknown or unimplemented method (Selective / WFG),
a duplicate scenario or method, a missing scenario or solver, a checkpoint
whose sha256 differs from checkpoints/MANIFEST.json, a dirty solver clone,
a dirty project tree (unless --allow-dirty, which makes the batch unfair) and
an existing output directory. After the runs, the batch is fair only if every
run used the same scenario files, checkpoint, max_steps=160, solver_root,
CBS timeout=30 s, project/solver commits and environment versions.

CBS is asked at different times: Pure CBS once at t=0 on S_0
(INITIAL_STATE); Global Hybrid after an IL stall Trigger (IL_STALL_TRIGGER).
All method settings except IL_ONLY are DEV settings, not agreed policies, and
every scenario here is a Development scenario (no Validation / Test data).

Exit codes: 0 fair batch, 1 batch written but not fair, 2 refused before running.

Usage (from the repository root):
    python -m scripts.run_compare_batch --solver-root <atb033>/centralized/cbs
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

from scripts.run_il import DEFAULT_CKPT
from src.compare import common_runner
from src.compare.common_runner import (
    DEFAULT_CBS_TIMEOUT_SEC,
    DEFAULT_MAX_STEPS,
    DEV_SETTINGS,
    METHODS,
    NOT_IMPLEMENTED,
    RunSpec,
    environment_info,
    run_method,
)
from src.compare.compare_csv import (
    CALL_FIELDS,
    CALLS_CSV,
    EPISODE_FIELDS,
    EPISODES_CSV,
    REQUEST_KIND,
    CompareCsvError,
    build_rows,
    check_against_records,
    check_fairness,
    manifest_checkpoint_sha256,
    verify_written,
    write_rows,
)

CONTROLLED = PROJECT_ROOT / "scenarios" / "controlled"
DEFAULT_SCENARIOS = (
    CONTROLLED / "scenario_corridor_bay_n2.json",
    CONTROLLED / "scenario_intersection_n2.json",
    CONTROLLED / "scenario_bottleneck_n4.json",
)
DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs" / "logs"
MANIFEST = "batch_manifest.json"


def preflight(args, out_dir):
    """Reasons to refuse the batch before any run (empty list = go)."""
    errors = []
    for m in args.methods:
        if m in NOT_IMPLEMENTED:
            errors.append(f"method {m} is not implemented")
        elif m not in METHODS:
            errors.append(f"unknown method {m!r}")
    if len(set(args.methods)) != len(args.methods):
        errors.append("duplicate method")
    names = [Path(s).name for s in args.scenarios]
    if len(set(names)) != len(names):
        errors.append("duplicate scenario")
    errors += [f"scenario not found: {s}" for s in args.scenarios if not Path(s).is_file()]
    if not (Path(args.solver_root) / "cbs.py").is_file():
        errors.append(f"atb033 cbs.py not found under {args.solver_root}")
    else:
        solver = common_runner._git(args.solver_root)
        if solver["dirty"] is not False:
            errors.append(f"solver repository dirty or unknown: {solver}")
    try:
        if common_runner._sha256_file(args.ckpt) != manifest_checkpoint_sha256(args.ckpt):
            errors.append("checkpoint sha256 differs from MANIFEST.json")
    except (OSError, CompareCsvError, ValueError) as exc:
        errors.append(f"checkpoint check failed: {exc}")
    project = common_runner._git(PROJECT_ROOT)
    if project["dirty"] is not False and not args.allow_dirty:
        errors.append("project tree is dirty (commit first, or --allow-dirty for an unfair batch)")
    if out_dir.exists():
        errors.append(f"output directory already exists: {out_dir}")
    return errors


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solver-root", type=Path, required=True)
    ap.add_argument("--scenario", dest="scenarios", type=Path, action="append", default=None)
    ap.add_argument("--method", dest="methods", action="append", default=None)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--cbs-timeout-sec", type=int, default=DEFAULT_CBS_TIMEOUT_SEC)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="new batch directory (default: outputs/logs/compare_<timestamp>)")
    ap.add_argument("--allow-dirty", action="store_true", help="run on a dirty tree; the batch is then unfair")
    args = ap.parse_args(argv)
    args.scenarios = list(args.scenarios or DEFAULT_SCENARIOS)
    args.methods = list(args.methods or METHODS)

    out_dir = (args.out_dir or DEFAULT_OUT_ROOT / f"compare_{datetime.now():%Y%m%d_%H%M%S}").resolve()
    errors = preflight(args, out_dir)
    if errors:
        for e in errors:
            print(f"REFUSED: {e}")
        return 2

    batch_id = out_dir.name
    out_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for scenario in args.scenarios:
        for method in args.methods:
            rec = run_method(RunSpec(method=method, scenario=Path(scenario), out_dir=out_dir / "runs",
                                     max_steps=args.max_steps, ckpt=args.ckpt, solver_root=args.solver_root,
                                     cbs_timeout_sec=args.cbs_timeout_sec))
            records.append(rec)
            print(f"  {Path(scenario).name:32s} {method:26s} {rec.termination:15s} "
                  f"steps={rec.steps:3d} stop_reason={rec.stop_reason}")

    built = [build_rows(r, batch_id) for r in records]
    episodes = [ep for ep, _ in built]
    calls = [c for _, cs in built for c in cs]
    write_rows(out_dir / EPISODES_CSV, EPISODE_FIELDS, episodes)
    write_rows(out_dir / CALLS_CSV, CALL_FIELDS, calls)
    checks = verify_written(out_dir / EPISODES_CSV, EPISODE_FIELDS, episodes)
    checks += verify_written(out_dir / CALLS_CSV, CALL_FIELDS, calls)
    runner_checks = check_against_records(out_dir / EPISODES_CSV, out_dir / CALLS_CSV, records)
    ckpt_sha = manifest_checkpoint_sha256(args.ckpt)
    fair, reasons = check_fairness(
        episodes, scenarios=[Path(s).name for s in args.scenarios], methods=args.methods,
        max_steps=DEFAULT_MAX_STEPS, cbs_timeout_sec=DEFAULT_CBS_TIMEOUT_SEC, checkpoint_sha256=ckpt_sha)

    manifest = {
        "batch_id": batch_id,
        "created": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join([sys.executable, "-m", "scripts.run_compare_batch", *argv]),
        "data_role": "Development scenarios only (no Validation / Test instances); one run each",
        "conditions": {
            "max_steps": args.max_steps,
            "cbs_timeout_sec": args.cbs_timeout_sec,
            "note": "max_steps (episode limit) and cbs_timeout_sec (solver limit) are separate limits",
            "solver_root": str(Path(args.solver_root).resolve()),
            "checkpoint": str(args.ckpt),
            "checkpoint_sha256": ckpt_sha,
            "scenarios": {Path(s).name: common_runner._sha256_file(s) for s in args.scenarios},
        },
        "methods": {m: {"dev_settings": DEV_SETTINGS[m], "cbs_request_kind": REQUEST_KIND.get(m)}
                    for m in args.methods},
        "cbs_request_timing": {
            "INITIAL_STATE": "Pure CBS: one call at t=0 on S_0, whole plan executed",
            "IL_STALL_TRIGGER": "Global Hybrid: call after an IL stall Trigger at t >= 1",
        },
        "solver_latency_ms": "replan_with_status perf_counter span; not pure CBS search time",
        "run_order": [r.run_id for r in records],
        "environment": environment_info(args.solver_root),
        "fair_batch": fair,
        "fairness_reasons": reasons,
        "csv_cell_checks": checks,
        "csv_vs_runner_checks": runner_checks,
    }
    with (out_dir / MANIFEST).open("x", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print(f"batch_dir   : {out_dir}")
    print(f"rows        : {len(episodes)} episodes, {len(calls)} CBS calls; "
          f"cell checks {checks}, runner checks {runner_checks}")
    print(f"fair_batch  : {fair}" + (f"  reasons: {reasons}" if reasons else ""))
    return 0 if fair else 1


if __name__ == "__main__":
    sys.exit(main())
