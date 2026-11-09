"""CBS pre-check of the PILOT v2 candidates: S_0, once per candidate, one at a time.

Default: print the plan and call nothing. ``--run --candidate <id>`` calls the
existing CbsRuntimePlanner (no change to the adapter or solver) for that one
candidate with a fixed 120 s timeout; ``--run`` alone is refused.

Refused before any call unless: PYTHONHASHSEED=0; solver at commit 89f69e1 and
clean; pre-registration, static report, review file and candidate files have
their recorded contents; and no outputs/pilot_v2_checks/cbs_*/<slug> exists for
the candidate, so a second run of it needs a deliberate human decision.

CBS status is passed through as the adapter returned it: timeout stays timeout
and is never reported as no_solution. A success is re-checked by the Path Guard
and by check_paths() below. Grade, target_met, OOD class and HOLD ids are copied
from the static report and never changed; the registry is never written.

Everything goes to a new outputs/pilot_v2_checks/cbs_<YYYYmmdd_HHMMSS>/:
  run_log.txt, run_manifest.json, attempts.jsonl, cbs_results.json and, per candidate,
  <slug>/result.json, solver_stdout.txt, solver_stderr.txt, cbs_call_01/... (solver YAML).

Exit codes: 0 finished with consistent records (CBS statuses are reported, not
judged), 1 a path check failed, a protected file changed or the script failed on
a candidate, 2 refused before any solver call.

Usage (from the repository root):
    PYTHONHASHSEED=0 python -m scripts.run_pilot_v2_cbs [--candidate PV2C-002] [--run]
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import types
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import cbs_adapter
from src.common.scenario_loader import load_map_generator_scenario
from src.data.fingerprint import sha256_file
from src.data.generator import GeneratorError
from src.data.quality_gate import derive_solvability
from src.data.registry import RegistryError
from src.hybrid.cbs_runtime_bridge import REPLANNER_WORK_DIR, CbsRuntimePlanner, PathGuardValidator
from scripts.generate_pilot_v2_candidates import (
    GENERATOR_V1_FILE, GENERATOR_V2_FILE, OUT_ROOT, PREREG_PATH, PREREG_SHA256,
    candidate_paths, generate_candidate, load_prereg,
)

TIMEOUT_SEC = 120
REQUIRED_HASH_SEED = "0"
SOLVER_COMMIT = "89f69e11944b172e11c3bcbbecc57344eaa4be9c"
DEFAULT_SOLVER_ROOT = (PROJECT_ROOT.parent / "MAPF_ALPHA2_CBS_CHECK" / "third_party"
                       / "multi_agent_path_planning" / "centralized" / "cbs")
STATIC_REPORT = OUT_ROOT / "static_20261108_222153" / "static_checks.json"
STATIC_REPORT_SHA256 = "d11bd5758f5e65bc59630b7082011bbd92a758dd8abc6fb4498dd104460d83de"
REVIEW_PATH = PROJECT_ROOT / "registry" / "pilot_v2_candidate_review_v0.jsonl"
REVIEW_SHA256 = "c54050b83abf0cf96517b0f66b8643b9772ee9fd0d21caca7381a6e43a73939e"
SOLVER_FILES = ("input.yaml", "solver_input.yaml", "output.yaml")
MAX_PATH = 259
SCRIPT_EXCEPTION = "script_exception"      # our failure, not a CBS adapter status
PROTECTED = (PREREG_PATH, REVIEW_PATH, STATIC_REPORT, GENERATOR_V1_FILE, GENERATOR_V2_FILE,
             PROJECT_ROOT / "registry" / "instances_v1.jsonl", PROJECT_ROOT / "registry" / "families_v1.json",
             PROJECT_ROOT / "registry" / "review_ledger_v0.jsonl",
             PROJECT_ROOT / "registry" / "pilot_solvability_v0.json")


def solver_git(path: Path) -> Dict:
    def run(*args):
        r = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}


def protected_hashes() -> Dict[str, Optional[str]]:
    return {p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p) if p.is_file() else None for p in PROTECTED}


# -- independent path checks ------------------------------------------------------

def check_paths(grid, starts: Dict[int, tuple], goals: Dict[int, tuple], paths: Dict[int, List]) -> Dict:
    """Path Guard plus an independent recount, so each required property is visible."""
    grid = np.asarray(grid)
    h, w = grid.shape
    guard_ok, guard_reason = PathGuardValidator()(grid, starts, goals, paths)
    norm = {a: [tuple(int(v) for v in cell) for cell in p] for a, p in paths.items()}
    ids_ok = sorted(norm) == sorted(starts)
    ids = sorted(set(norm) & set(starts))
    lengths = {a: len(norm[a]) for a in ids}
    horizon = max(lengths.values(), default=0)

    def at(a, t):
        p = norm[a]
        return p[min(t, len(p) - 1)]

    vertex, edge, bad_moves, bad_cells = [], [], [], []
    for a in ids:
        for t, (r, c) in enumerate(norm[a]):
            if not (0 <= r < h and 0 <= c < w) or grid[r, c] != 0:
                bad_cells.append([a, t, r, c])
        for t in range(len(norm[a]) - 1):
            (r0, c0), (r1, c1) = norm[a][t], norm[a][t + 1]
            if abs(r0 - r1) + abs(c0 - c1) > 1:
                bad_moves.append([a, t])
    for t in range(horizon):
        seen: Dict[tuple, int] = {}
        for a in ids:
            cell = at(a, t)
            if cell in seen:
                vertex.append([t, seen[cell], a, list(cell)])
            seen[cell] = a
        if t + 1 < horizon:
            for i, a in enumerate(ids):
                for b in ids[i + 1:]:
                    if at(a, t) == at(b, t + 1) and at(b, t) == at(a, t + 1) and at(a, t) != at(a, t + 1):
                        edge.append([t, a, b])
    return {
        "path_guard_ok": guard_ok, "path_guard_reason": guard_reason,
        "agent_ids_preserved": ids_ok,
        "path0_is_start": ids_ok and all(norm[a] and norm[a][0] == tuple(starts[a]) for a in ids),
        "last_is_goal": ids_ok and all(norm[a] and norm[a][-1] == tuple(goals[a]) for a in ids),
        "equal_lengths": len(set(lengths.values())) <= 1,
        "vertex_conflicts": vertex, "edge_conflicts": edge,
        "illegal_moves": bad_moves, "wall_or_outside_cells": bad_cells,
    }


# -- solution metrics -------------------------------------------------------------

def settle_time(path: List, goal: tuple) -> Optional[int]:
    """First t from which the agent stays on its goal to the end of ``path``; None if
    it does not end there. A visit to the goal that it later leaves does not count,
    and WAIT padding after settling does not change the value."""
    cells = [tuple(c) for c in path]
    goal = tuple(goal)
    if not cells or cells[-1] != goal:
        return None
    t = len(cells) - 1
    while t > 0 and cells[t - 1] == goal:
        t -= 1
    return t


def path_metrics(paths: Dict[int, List], goals: Dict[int, tuple]) -> Dict:
    """Per-agent settle and first-arrival times, real SoC, makespan. ``padded_steps_total``
    is the old padded sum (makespan x agents when padded), kept only to show the difference."""
    settle = {a: settle_time(paths[a], goals[a]) for a in sorted(paths)}
    first = {}
    for a in sorted(paths):
        cells = [tuple(c) for c in paths[a]]
        first[a] = cells.index(tuple(goals[a])) if tuple(goals[a]) in cells else None
    reached = all(v is not None for v in settle.values())
    return {"settle_times": {str(a): v for a, v in settle.items()},
            "first_arrival_times": {str(a): v for a, v in first.items()},
            "agents_left_goal_after_arrival": [a for a in settle if settle[a] is not None and first[a] != settle[a]],
            "soc": sum(settle.values()) if reached else None,
            "makespan": max(settle.values()) if reached else None,
            "padded_steps_total": sum(len(p) - 1 for p in paths.values())}


def solver_output_metrics(output_yaml: Path, goals: Dict[int, tuple]) -> Dict:
    """The solver's own ``cost`` as written (it counts t=0 for every agent, so it is not
    SoC), and the metrics recomputed from its unpadded schedule."""
    if not output_yaml.is_file():
        return {"solver_cost": None, "raw_schedule": None}
    data = cbs_adapter.load_atb033_output(output_yaml)
    raw = cbs_adapter.convert_schedule_to_internal_paths(data["schedule"])
    return {"solver_cost": data.get("cost"), "raw_schedule": path_metrics(raw, goals)}


def metrics_consistent(metrics: Dict, raw: Dict, soc_bound, makespan_bound) -> Dict:
    """Padded and raw values must agree, and neither may be below its lower bound."""
    raw_schedule = raw["raw_schedule"]
    same = None
    if raw_schedule is not None:
        same = all(metrics[k] == raw_schedule[k] for k in ("soc", "makespan", "settle_times"))
    above = None
    if metrics["soc"] is not None and soc_bound is not None:
        above = metrics["soc"] >= soc_bound and metrics["makespan"] >= makespan_bound
    return {"padded_matches_raw_schedule": same, "at_or_above_lower_bounds": above}


def paths_all_ok(check: Dict) -> bool:
    return (check["path_guard_ok"] is True and check["agent_ids_preserved"] and check["path0_is_start"]
            and check["last_is_goal"] and check["equal_lengths"] and not check["vertex_conflicts"]
            and not check["edge_conflicts"] and not check["illegal_moves"] and not check["wall_or_outside_cells"])


# -- solver log capture -----------------------------------------------------------

def _text(value) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


@contextlib.contextmanager
def record_solver_process(sink: Dict):
    """While active, src.cbs_adapter's subprocess.run is wrapped to record what it
    returns. Arguments and results pass through unchanged; restored on exit."""
    real = subprocess.run

    def recording_run(command, **kwargs):
        sink.update(command=[str(c) for c in command], cwd=str(kwargs.get("cwd")),
                    timeout_arg=kwargs.get("timeout"), timed_out=False)
        began = time.perf_counter()
        try:
            result = real(command, **kwargs)
        except subprocess.TimeoutExpired as exc:
            sink.update(timed_out=True, returncode=None, stdout=_text(exc.stdout), stderr=_text(exc.stderr),
                        process_sec=time.perf_counter() - began)
            raise
        sink.update(returncode=result.returncode, stdout=_text(result.stdout), stderr=_text(result.stderr),
                    process_sec=time.perf_counter() - began)
        return result

    original = cbs_adapter.subprocess
    cbs_adapter.subprocess = types.SimpleNamespace(run=recording_run, TimeoutExpired=subprocess.TimeoutExpired,
                                                   CompletedProcess=subprocess.CompletedProcess)
    try:
        yield sink
    finally:
        cbs_adapter.subprocess = original


# -- preflight / plan -------------------------------------------------------------

def preflight(solver_root: Path, out_root: Path, out_dir: Path, *, root: Path = PROJECT_ROOT,
              env: Optional[Dict] = None, git_info: Callable = solver_git,
              candidate: Optional[str] = None) -> Dict:
    env = os.environ if env is None else env
    refusals = []
    selected: List[Dict] = []
    try:
        selected = [s for s in load_prereg()["candidates"] if candidate in (None, s["candidate_id"])]
        if not selected:
            refusals.append(f"unknown candidate {candidate!r}")
    except (OSError, ValueError, RegistryError) as exc:
        refusals.append(str(exc))
    if env.get("PYTHONHASHSEED") != REQUIRED_HASH_SEED:
        refusals.append(f"PYTHONHASHSEED must be {REQUIRED_HASH_SEED}, got {env.get('PYTHONHASHSEED')!r}")
    git = git_info(solver_root)
    if not (solver_root / "cbs.py").is_file():
        refusals.append(f"cbs.py not found under {solver_root}")
    if git.get("commit") != SOLVER_COMMIT:
        refusals.append(f"solver commit {git.get('commit')} is not {SOLVER_COMMIT}")
    if git.get("dirty") is not False:
        refusals.append(f"solver repository dirty or unknown: {git}")
    earlier = sorted(f"{run.name}/{s['slug']}" for run in out_root.glob("cbs_*") for s in selected
                     if (run / s["slug"]).exists()) if out_root.is_dir() else []
    if earlier:
        refusals.append(f"a CBS run already exists ({earlier}); each candidate runs once")
    if out_dir.exists():
        refusals.append(f"output directory exists: {out_dir}")
    for path, sha in ((STATIC_REPORT, STATIC_REPORT_SHA256), (REVIEW_PATH, REVIEW_SHA256)):
        actual = sha256_file(path) if path.is_file() else None
        if actual != sha:
            refusals.append(f"{path}: sha256 {actual} is not the recorded {sha}")

    calls = []
    for spec in selected:
        map_path, scen_path = candidate_paths(spec, root)
        try:
            _, npy, js = generate_candidate(spec)
            if map_path.read_bytes() != npy or scen_path.read_bytes() != js:
                refusals.append(f"{spec['slug']}: files differ from a same-seed regeneration")
        except (OSError, GeneratorError) as exc:
            refusals.append(f"{spec['slug']}: cannot verify files ({exc})")
        work = out_dir / spec["slug"] / "cbs_call_01" / REPLANNER_WORK_DIR
        longest = max(len(str(work / f)) for f in SOLVER_FILES)
        if longest > MAX_PATH:
            refusals.append(f"{spec['slug']}: output path of {longest} chars exceeds MAX_PATH")
        calls.append({"candidate_id": spec["candidate_id"], "slug": spec["slug"], "grade": spec["grade"],
                      "seed": spec["seed"], "num_agents": spec["num_agents"], "scenario": str(scen_path),
                      "request": "INITIAL_STATE (S_0, t=0)", "solver_cwd": str(solver_root),
                      "solver_command": [sys.executable, "cbs.py", str(work / "solver_input.yaml"),
                                         str(work / "output.yaml")],
                      "longest_path_chars": longest})
    return {"refusals": refusals, "calls": calls, "expected_cbs_calls": len(calls), "timeout_sec_per_call": TIMEOUT_SEC,
            "worst_case_wall_sec": len(calls) * TIMEOUT_SEC, "out_dir": str(out_dir), "solver_root": str(solver_root),
            "solver_git": git, "required_solver_commit": SOLVER_COMMIT,
            "pythonhashseed": env.get("PYTHONHASHSEED"), "prereg_sha256": PREREG_SHA256,
            "script_sha256": sha256_file(__file__),
            "python_executable": sys.executable, "python_version": platform.python_version(),
            "numpy_version": np.__version__}


# -- run --------------------------------------------------------------------------

def _static_by_id(static_report: Dict) -> Dict[str, Dict]:
    return {c["candidate_id"]: c for c in static_report["candidates"]}


def run_candidate(spec: Dict, static: Dict, out_dir: Path, solver_root: Path, root: Path,
                  planner_factory: Callable, attempts) -> Dict:
    slug_dir = out_dir / spec["slug"]
    slug_dir.mkdir(parents=True, exist_ok=False)
    map_path, scen_path = candidate_paths(spec, root)
    result = {
        "candidate_id": spec["candidate_id"], "slug": spec["slug"], "grade": spec["grade"],
        "split_status": "PILOT candidate (Development side), not registered",
        "target_met_static": static["target"]["target_met"], "ood_class_static": static["ood"]["ood_class"],
        "holds_unchanged": static["holds"], "generator_seed": spec["seed"],
        "pythonhashseed_env": os.environ.get("PYTHONHASHSEED"), "timeout_sec": TIMEOUT_SEC,
        "map_sha256": sha256_file(map_path), "scenario_sha256": sha256_file(scen_path),
        "path_check": None, "path_check_all_ok": None,
    }
    process: Dict = {}
    attempts.write(json.dumps({"candidate_id": spec["candidate_id"], "event": "started",
                               "at": datetime.now().isoformat(timespec="seconds")}) + "\n")
    attempts.flush()
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {spec['candidate_id']} {spec['slug']}: calling CBS "
          f"(timeout {TIMEOUT_SEC}s)", flush=True)
    began = time.perf_counter()
    try:
        grid, starts, goals = load_map_generator_scenario(scen_path)
        s = {i: tuple(p) for i, p in enumerate(starts)}
        g = {i: tuple(p) for i, p in enumerate(goals)}
        planner = planner_factory(solver_root=solver_root, work_root=slug_dir, timeout_sec=TIMEOUT_SEC)
        with record_solver_process(process):
            res = planner(s, g, grid)
        result.update(status=res.status, latency_sec=res.latency_sec, error=res.error)
        if res.status == "success":
            check = check_paths(grid, s, g, res.paths)
            metrics = path_metrics(res.paths, g)
            raw = solver_output_metrics(slug_dir / "cbs_call_01" / REPLANNER_WORK_DIR / "output.yaml", g)
            bounds = (static.get("soc_lower_bound"), static.get("makespan_lower_bound"))
            result.update(path_check=check, path_check_all_ok=paths_all_ok(check),
                          **metrics, solver_cost=raw["solver_cost"], metrics_from_raw_schedule=raw["raw_schedule"],
                          metrics_check=metrics_consistent(metrics, raw, *bounds),
                          soc_lower_bound=bounds[0], makespan_lower_bound=bounds[1],
                          paths={str(a): [list(map(int, c)) for c in p] for a, p in res.paths.items()})
    except Exception as exc:  # a script failure must not stop the record of the others
        result.update(status=SCRIPT_EXCEPTION, latency_sec=None, error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
    result["wall_sec"] = time.perf_counter() - began
    result["interpretation"] = (None if result["status"] == SCRIPT_EXCEPTION else derive_solvability(
        {"cbs": {"status": result["status"], "path_guard_ok": result["path_check_all_ok"]}}))

    (slug_dir / "solver_stdout.txt").write_text(process.get("stdout", ""), encoding="utf-8")
    (slug_dir / "solver_stderr.txt").write_text(process.get("stderr", ""), encoding="utf-8")
    result["solver_process"] = {k: process.get(k) for k in ("command", "cwd", "timeout_arg", "timed_out",
                                                            "returncode", "process_sec")}
    work = slug_dir / "cbs_call_01" / REPLANNER_WORK_DIR
    result["solver_file_sha256"] = {f: sha256_file(work / f) if (work / f).is_file() else None for f in SOLVER_FILES}
    with open(slug_dir / "result.json", "x", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    attempts.write(json.dumps({"candidate_id": spec["candidate_id"], "event": "finished", "status": result["status"],
                               "at": datetime.now().isoformat(timespec="seconds")}) + "\n")
    attempts.flush()
    print(f"    status={result['status']} latency={result['latency_sec']} wall={result['wall_sec']:.2f}s "
          f"path_check_all_ok={result['path_check_all_ok']}", flush=True)
    return result


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)
        return len(text)

    def flush(self):
        for s in self.streams:
            s.flush()


def run_all(plan: Dict, out_dir: Path, solver_root: Path, root: Path, static_report: Dict, specs: List[Dict],
            planner_factory: Callable = CbsRuntimePlanner) -> Dict:
    """Sequential, one call per candidate. Writes everything under ``out_dir``; the
    exclusively created manifest refuses a directory that was already used."""
    out_dir.mkdir(parents=True, exist_ok=True)
    before = protected_hashes()
    with open(out_dir / "run_manifest.json", "x", encoding="utf-8") as fh:
        json.dump({"plan": plan, "protected_sha256_before": before,
                   "started_at": datetime.now().isoformat(timespec="seconds")}, fh, indent=2, sort_keys=True)
        fh.write("\n")
    static = _static_by_id(static_report)
    results = []
    with open(out_dir / "attempts.jsonl", "x", encoding="utf-8") as attempts:
        for spec in specs:
            results.append(run_candidate(spec, static[spec["candidate_id"]], out_dir, solver_root, root,
                                         planner_factory, attempts))
    after = protected_hashes()
    report = {"plan": plan, "results": results, "protected_sha256_before": before, "protected_sha256_after": after,
              "protected_unchanged": before == after, "registry_written": False,
              "finished_at": datetime.now().isoformat(timespec="seconds")}
    with open(out_dir / "cbs_results.json", "x", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    return report


def exit_code(report: Dict) -> int:
    bad = [r for r in report["results"]
           if r["status"] == SCRIPT_EXCEPTION or (r["status"] == "success" and r["path_check_all_ok"] is not True)
           or False in (r.get("metrics_check") or {}).values()]
    return 1 if bad or not report["protected_unchanged"] else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="store_true", help="call the solver; without it only the plan is printed")
    ap.add_argument("--candidate", help="candidate_id to plan or run; required with --run")
    ap.add_argument("--solver-root", default=str(DEFAULT_SOLVER_ROOT))
    args = ap.parse_args(argv)
    if args.run and not args.candidate:
        print("REFUSED: --run needs --candidate (one candidate per run)", file=sys.stderr)
        return 2
    solver_root = Path(args.solver_root).resolve()
    out_dir = OUT_ROOT / f"cbs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    plan = preflight(solver_root, OUT_ROOT, out_dir, candidate=args.candidate)
    print(json.dumps({k: v for k, v in plan.items() if k != "refusals"}, indent=2, sort_keys=True))
    for r in plan["refusals"]:
        print(f"REFUSED: {r}", file=sys.stderr)
    if plan["refusals"]:
        return 2
    if not args.run:
        print("plan only: the solver was not called (pass --run)")
        return 0

    specs = [s for s in load_prereg()["candidates"] if s["candidate_id"] == args.candidate]
    static_report = json.loads(STATIC_REPORT.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=False)
    with open(out_dir / "run_log.txt", "x", encoding="utf-8", buffering=1) as log, \
            contextlib.redirect_stdout(_Tee(sys.stdout, log)), contextlib.redirect_stderr(_Tee(sys.stderr, log)):
        print(f"solver={solver_root} git={plan['solver_git']} PYTHONHASHSEED={plan['pythonhashseed']}")
        try:
            report = run_all(plan, out_dir, solver_root, PROJECT_ROOT, static_report, specs)
        except BaseException:
            traceback.print_exc()
            raise
        print(f"protected_unchanged={report['protected_unchanged']}")
    code = exit_code(report)
    with open(out_dir / "run_log.txt", "a", encoding="utf-8") as log:
        log.write(f"exit_code={code}\n")
    print(f"wrote {out_dir / 'cbs_results.json'}")
    return code


if __name__ == "__main__":
    sys.exit(main())
