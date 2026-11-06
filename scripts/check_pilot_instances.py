"""Check the PILOT instances: regeneration, IL input range, CBS solvability.

Read-only towards scenarios/, registry/, checkpoints/ and the Solver. Two modes:

  (default)      plan only: regenerate every pilot and compare bytes with the
                 files on disk and fingerprints with the registry; IL input
                 OOD check; print the CBS plan (calls, timeout, output paths,
                 solver commit, PYTHONHASHSEED) without running the solver.
  --run-solver   additionally run atb033 CBS once per pilot on S_0 through the
                 existing CbsRuntimePlanner (no change to the adapter/solver)
                 and re-check accepted paths with PathGuardValidator. Results
                 and solver files go to a NEW directory
                 outputs/pilot_checks_v1/<YYYYmmdd_HHMMSS>/ (refused if it exists).

CBS outcomes are reported as returned by the adapter: success / timeout /
no_solution / invalid_path / input_error / solver_error -- a timeout is
"unknown", never counted as no_solution.

Exit codes: 0 all checks passed (solver statuses are reported, not judged),
1 a check failed, 2 refused before running.

Usage (from the repository root):
    PYTHONHASHSEED=0 python -m scripts.check_pilot_instances [--run-solver] [--timeout-sec 120]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.scenario_loader import load_map_generator_scenario
from src.data.fingerprint import fingerprint_scenario
from src.data.il_compat import check_instance, load_reference
from src.data.registry import Registry
from scripts.check_registry import RECOMPUTED_FIELDS
from scripts.generate_pilot_instances import PILOT_SPECS, pilot_files
from scripts.register_dev_scenarios import INSTANCES

DEFAULT_SOLVER_ROOT = (PROJECT_ROOT.parent / "MAPF_ALPHA2_CBS_CHECK" / "third_party"
                       / "multi_agent_path_planning" / "centralized" / "cbs")
REFERENCE = PROJECT_ROOT / "registry" / "il_input_reference_v1.json"
OUT_ROOT = PROJECT_ROOT / "outputs" / "pilot_checks_v1"
SOLVER_FILES = ("input.yaml", "solver_input.yaml", "output.yaml")
MAX_PATH = 259


def _git(path: Path) -> dict:
    import subprocess
    def run(*a):
        r = subprocess.run(["git", "-C", str(path), *a], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}


def regeneration_checks(reg: Registry) -> list:
    """Problems found when regenerating each pilot and comparing with disk + registry."""
    problems = []
    by_source = {r["source_path"]: r for r in reg.records}
    for slug in PILOT_SPECS:
        _, map_path, npy, scen_path, js = pilot_files(slug)
        for path, data in ((map_path, npy), (scen_path, js)):
            if not path.is_file():
                problems.append(f"{slug}: missing {path}")
            elif path.read_bytes() != data:
                problems.append(f"{slug}: {path.name} differs from a same-seed regeneration")
        fresh = fingerprint_scenario(scen_path) if scen_path.is_file() else None
        rec = by_source.get(fresh["source_path"]) if fresh else None
        if rec is None:
            problems.append(f"{slug}: not in the registry")
        elif rec["split"] != "PILOT":
            problems.append(f"{slug}: registered as {rec['split']}, expected PILOT")
        else:
            problems += [f"{slug}: {f} differs from the registry" for f in RECOMPUTED_FIELDS
                         if fresh.get(f) != rec.get(f)]
    return problems


def solver_plan(out_dir: Path, solver_root: Path, timeout_sec: int) -> dict:
    calls = []
    for slug in PILOT_SPECS:
        work = out_dir / slug / "cbs_call_01" / "outputs" / "paths" / "replan"
        longest = max(len(str(work / f)) for f in SOLVER_FILES)
        calls.append({"pilot": slug, "request": "INITIAL_STATE (S_0, t=0)",
                      "work_root": str(out_dir / slug), "longest_path_chars": longest})
    return {"expected_cbs_calls": len(calls), "timeout_sec_per_call": timeout_sec,
            "worst_case_wall_sec": len(calls) * timeout_sec, "out_dir": str(out_dir),
            "solver_root": str(solver_root), "solver_git": _git(solver_root),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"), "calls": calls,
            "max_path_ok": all(c["longest_path_chars"] <= MAX_PATH for c in calls)}


def run_solver(out_dir: Path, solver_root: Path, timeout_sec: int) -> list:
    from src.hybrid.cbs_runtime_bridge import CbsRuntimePlanner, PathGuardValidator
    results = []
    for slug in PILOT_SPECS:
        _, _, _, scen_path, _ = pilot_files(slug)
        grid, starts, goals = load_map_generator_scenario(scen_path)
        planner = CbsRuntimePlanner(solver_root=solver_root, work_root=out_dir / slug,
                                    timeout_sec=timeout_sec)
        s = {i: tuple(p) for i, p in enumerate(starts)}
        g = {i: tuple(p) for i, p in enumerate(goals)}
        res = planner(s, g, grid)
        entry = {"pilot": slug, "status": res.status, "latency_sec": res.latency_sec,
                 "error": res.error, "pythonhashseed_env": planner.calls[0]["pythonhashseed_env"]}
        if res.status == "success":
            ok, reason = PathGuardValidator()(grid, s, g, res.paths)
            lengths = {a: len(p) - 1 for a, p in res.paths.items()}
            entry.update(path_guard_ok=ok, path_guard_reason=reason,
                         sum_of_path_lengths=sum(lengths.values()), makespan=max(lengths.values()),
                         paths={str(a): [list(c) for c in p] for a, p in res.paths.items()})
        results.append(entry)
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-solver", action="store_true")
    ap.add_argument("--timeout-sec", type=int, default=120)
    ap.add_argument("--solver-root", default=str(DEFAULT_SOLVER_ROOT))
    ap.add_argument("--registry-dir", default=str(PROJECT_ROOT / "registry"))
    args = ap.parse_args(argv)
    solver_root = Path(args.solver_root).resolve()
    out_dir = OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")

    reg = Registry.load(Path(args.registry_dir) / INSTANCES)
    problems = regeneration_checks(reg)
    reference = load_reference(REFERENCE)
    il = {}
    for slug in PILOT_SPECS:
        _, _, _, scen_path, _ = pilot_files(slug)
        grid, starts, goals = load_map_generator_scenario(scen_path)
        il[slug] = check_instance(grid, starts, goals, reference)
        w = il[slug]["warnings"]
        print(f"IL  {slug:26s} {il[slug]['status']:11s} warnings={len(w)} auto_fail={il[slug]['auto_fail']}")
        for x in w:
            print(f"      {x['kind']:15s} agent={x['agent']} {x['feature']}={x['value']} bounds={x['bounds']}")
    print(f"regeneration/registry problems: {problems or 'none'}")

    plan = solver_plan(out_dir, solver_root, args.timeout_sec)
    print(f"CBS plan: calls={plan['expected_cbs_calls']} timeout={plan['timeout_sec_per_call']}s each "
          f"(worst case {plan['worst_case_wall_sec']}s) out={plan['out_dir']}")
    print(f"          solver={plan['solver_root']} git={plan['solver_git']} "
          f"PYTHONHASHSEED={plan['pythonhashseed']!r} max_path_ok={plan['max_path_ok']}")
    if problems:
        return 1
    if not args.run_solver:
        print("plan only: the solver was not run (pass --run-solver)")
        return 0

    refuse = []
    if not (solver_root / "cbs.py").is_file():
        refuse.append(f"cbs.py not found under {solver_root}")
    if plan["solver_git"]["dirty"] is not False:
        refuse.append(f"solver repository dirty or unknown: {plan['solver_git']}")
    if plan["pythonhashseed"] in (None, "random"):
        refuse.append("PYTHONHASHSEED must be fixed at start-up (docs/reproducibility_v1.md)")
    if not plan["max_path_ok"]:
        refuse.append("an output path would exceed MAX_PATH")
    if out_dir.exists():
        refuse.append(f"output directory exists: {out_dir}")
    if refuse:
        for r in refuse:
            print(f"REFUSED: {r}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=False)
    results = run_solver(out_dir, solver_root, args.timeout_sec)
    for r in results:
        extra = (f" path_guard={r['path_guard_ok']} SoC={r['sum_of_path_lengths']} makespan={r['makespan']}"
                 if r["status"] == "success" else f" error={r['error']!r}")
        print(f"CBS {r['pilot']:26s} {r['status']:12s} {r['latency_sec']:.3f}s{extra}")
    report = {"plan": plan, "il_compat": il, "cbs": results, "registry_sha256": reg.source_sha256}
    with open(out_dir / "pilot_checks.json", "x", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    print(f"wrote {out_dir / 'pilot_checks.json'}")
    bad_guard = [r["pilot"] for r in results if r["status"] == "success" and not r["path_guard_ok"]]
    return 1 if bad_guard else 0


if __name__ == "__main__":
    sys.exit(main())
