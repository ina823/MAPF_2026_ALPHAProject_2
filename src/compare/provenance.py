"""Reproducibility provenance for comparison batches: Python hash seed,
batch-to-batch comparability, CBS request states, output-path length.

PYTHONHASHSEED is read by the Python interpreter once, at start-up. Changing
os.environ afterwards does not change this process's str hashing; it only
changes what child processes inherit. So three things are recorded and
checked separately, never inferred from one another:

  1. parent process  -- the value present when this module was first
     imported (``STARTUP_ENV_VALUE``), ``sys.flags.hash_randomization``
     (fixed at start-up) and ``hash(PROBE_TEXT)`` computed in this process.
     ``verify_parent_seed`` starts a child with that seed set EXPLICITLY;
     equal probe hashes mean this interpreter really runs with that seed.
  2. each CBS call   -- ``os.environ["PYTHONHASHSEED"]`` at call time, which
     is what the cbs.py subprocess inherits: src/cbs_adapter.py calls
     subprocess.run() without ``env=`` (a contract test guards this).
  3. inherited probe -- a child started exactly like cbs.py (sys.executable,
     no ``env=``) reports what it actually sees.

Modes: PYTHONHASHSEED unset or "random" -> UNSET_RANDOM; "0" ->
FIXED_DISABLED (hash randomization disabled: a fixed, deterministic
setting, sys.flags.hash_randomization == 0); 1..4294967295 -> FIXED.

A batch's ``fair_batch`` (same conditions across methods) and
``reproducible_batch`` (a fixed, verified hash seed) are different claims.
A seed-unset batch keeps its fair_batch verdict and is simply not
reproducible. Batches whose manifest has no ``reproducibility`` record
(everything written before this module existed) are NOT_RECORDED -- their
seed is never guessed.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

PROBE_TEXT = "mapf-alpha2-hashseed-probe"
NOT_RECORDED = "NOT_RECORDED"
UNSET_RANDOM = "UNSET_RANDOM"
FIXED = "FIXED"
FIXED_DISABLED = "FIXED_DISABLED"
INVALID = "INVALID"
FIXED_MODES = (FIXED, FIXED_DISABLED)

# Captured once, as early as this module is imported (run_compare_batch and
# common_runner import it at module load).
STARTUP_ENV_VALUE: Optional[str] = os.environ.get("PYTHONHASHSEED")

_PROBE_CODE = (
    "import json, os, sys; print(json.dumps({"
    "'env_value': os.environ.get('PYTHONHASHSEED'), "
    "'hash_randomization': sys.flags.hash_randomization, "
    f"'probe_hash': hash({PROBE_TEXT!r})}}))"
)


def seed_mode(value: Optional[str]) -> str:
    if value is None or value == "random":
        return UNSET_RANDOM
    if value.isdigit() and int(value) <= 4294967295:
        return FIXED_DISABLED if int(value) == 0 else FIXED
    return INVALID


def hash_seed_info() -> Dict[str, Any]:
    """Parent-process hash-seed record (layer 1, plus the current env value)."""
    return {
        "startup_env_value": STARTUP_ENV_VALUE,
        "current_env_value": os.environ.get("PYTHONHASHSEED"),
        "mode": seed_mode(STARTUP_ENV_VALUE),
        "hash_randomization": sys.flags.hash_randomization,
        "probe_hash": hash(PROBE_TEXT),
        "note": "PYTHONHASHSEED applies at interpreter start-up; '0' = hash randomization disabled (fixed)",
    }


def subprocess_probe(*, explicit_seed: Optional[str] = None, python: str = sys.executable,
                     timeout_sec: int = 30) -> Dict[str, Any]:
    """Start a fresh interpreter and report its hash-seed view.

    ``explicit_seed=None`` inherits this process's environment exactly like
    src/cbs_adapter.py's cbs.py subprocess (no ``env=``). Otherwise the child
    gets a copy of the environment with PYTHONHASHSEED=explicit_seed."""
    kwargs = {}
    if explicit_seed is not None:
        kwargs["env"] = {**os.environ, "PYTHONHASHSEED": explicit_seed}
    try:
        cp = subprocess.run([python, "-c", _PROBE_CODE], capture_output=True, text=True,
                            timeout=timeout_sec, check=False, **kwargs)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if cp.returncode != 0:
        return {"ok": False, "error": f"returncode {cp.returncode}: {cp.stderr[-300:]}"}
    try:
        return {"ok": True, **json.loads(cp.stdout.strip().splitlines()[-1])}
    except (ValueError, IndexError) as exc:
        return {"ok": False, "error": f"unparsable probe output: {exc}"}


def verify_seed_layers(parent: Dict[str, Any]) -> Dict[str, Any]:
    """Layers 1 and 3 for a batch, before any run.

    parent_verified: VERIFIED / MISMATCH / NOT_APPLICABLE (seed not fixed) / PROBE_FAILED.
    inherited:       OK / MISMATCH / PROBE_FAILED -- what a cbs.py-like child sees."""
    reasons: List[str] = []
    mode = parent["mode"]
    if parent["current_env_value"] != parent["startup_env_value"]:
        reasons.append(f"PYTHONHASHSEED changed after start-up ({parent['startup_env_value']!r} -> "
                       f"{parent['current_env_value']!r}); children would not match this process")
    if mode == FIXED_DISABLED and parent["hash_randomization"] != 0:
        reasons.append("PYTHONHASHSEED=0 recorded but hash randomization is on")
    if mode == FIXED and parent["hash_randomization"] != 1:
        reasons.append("fixed PYTHONHASHSEED recorded but sys.flags.hash_randomization != 1")

    if mode in FIXED_MODES:
        explicit = subprocess_probe(explicit_seed=parent["startup_env_value"])
        if not explicit["ok"]:
            parent_verified = "PROBE_FAILED"
        elif explicit["probe_hash"] == parent["probe_hash"]:
            parent_verified = "VERIFIED"
        else:
            parent_verified = "MISMATCH"
            reasons.append("this interpreter's str hash differs from a child started with the recorded seed")
    else:
        explicit, parent_verified = None, "NOT_APPLICABLE"

    inherited = subprocess_probe()
    if not inherited["ok"]:
        inherited_status = "PROBE_FAILED"
    else:
        same_env = inherited["env_value"] == parent["startup_env_value"]
        same_hash = (inherited["probe_hash"] == parent["probe_hash"]) if mode in FIXED_MODES else True
        inherited_status = "OK" if same_env and same_hash else "MISMATCH"
        if not same_env:
            reasons.append(f"inherited child sees PYTHONHASHSEED={inherited['env_value']!r}, "
                           f"parent started with {parent['startup_env_value']!r}")
        elif not same_hash:
            reasons.append("inherited child's str hash differs from the parent's")
    if "PROBE_FAILED" in (parent_verified, inherited_status):
        reasons.append("hash-seed probe subprocess failed")
    return {"parent_verified": parent_verified, "explicit_probe": explicit,
            "inherited_status": inherited_status, "inherited_probe": inherited, "reasons": reasons}


def check_seed_consistency(parent: Dict[str, Any], records: Iterable[Any]) -> List[str]:
    """Layer 2 across a batch: every run's environment record and every CBS
    call's inherited value must equal the parent's start-up value. Returns
    reasons (empty = consistent). A consistently unset seed is consistent."""
    want = parent["startup_env_value"]
    reasons = []
    for rec in records:
        env = (rec.environment or {}).get("hash_seed") or {}
        if env.get("current_env_value", want) != want:
            reasons.append(f"PYTHONHASHSEED differs: run {rec.run_id} recorded "
                           f"{env.get('current_env_value')!r}, batch started with {want!r}")
        for i, call in enumerate(rec.planner_calls, start=1):
            if "pythonhashseed_env" not in call:
                reasons.append(f"PYTHONHASHSEED not recorded for {rec.run_id} CBS call {i}")
            elif call["pythonhashseed_env"] != want:
                reasons.append(f"PYTHONHASHSEED differs: {rec.run_id} CBS call {i} inherited "
                               f"{call['pythonhashseed_env']!r}, batch started with {want!r}")
    return reasons


def reproducibility_record(parent, layers, consistency_reasons) -> Dict[str, Any]:
    reasons = list(layers["reasons"]) + list(consistency_reasons)
    mode = parent["mode"]
    if mode not in FIXED_MODES:
        reasons.insert(0, f"hash seed not fixed ({mode})")
    reproducible = (mode in FIXED_MODES and not reasons
                    and layers["parent_verified"] == "VERIFIED" and layers["inherited_status"] == "OK")
    return {
        "seed_mode": mode,
        "seed_value": parent["startup_env_value"] if mode in FIXED_MODES else None,
        "parent": parent,
        "parent_verified": layers["parent_verified"],
        "explicit_probe": layers["explicit_probe"],
        "inherited_probe_status": layers["inherited_status"],
        "inherited_probe": layers["inherited_probe"],
        "cbs_calls_seed_consistent": not consistency_reasons,
        "reproducible_batch": reproducible,
        "reasons": reasons,
    }


def manifest_seed(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Seed record of a manifest; NOT_RECORDED (never guessed) for old ones."""
    rec = manifest.get("reproducibility")
    if not rec:
        return {"seed_mode": NOT_RECORDED, "seed_value": None, "reproducible_batch": None}
    return {k: rec.get(k) for k in ("seed_mode", "seed_value", "reproducible_batch")}


def compare_batch_provenance(manifests: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Can these batches be read as one same-seed reproducibility comparison?

    NOT_COMPARABLE         run conditions differ (reasons say which);
    LEGACY_UNSEEDED        some manifest has no seed record (NOT_RECORDED);
    UNSEEDED_OR_UNVERIFIED some batch has an unset or unverified seed;
    SEED_SWEEP             all verified, but seeds differ -- a sensitivity
                           study, never a same-seed reproducibility check;
    SAME_SEED_COMPARABLE   all verified with one seed value.
    """
    reasons: List[str] = []

    def same(label, values):
        vals = [json.dumps(v, sort_keys=True, default=str) for v in values]
        if len(set(vals)) > 1:
            reasons.append(f"{label} differs between batches")

    conds = [m.get("conditions", {}) for m in manifests]
    envs = [m.get("environment", {}) for m in manifests]
    for key in ("max_steps", "cbs_timeout_sec", "solver_root", "checkpoint_sha256", "scenarios"):
        same(key, [c.get(key) for c in conds])
    same("solver commit", [(e.get("solver_git") or {}).get("commit") for e in envs])
    same("project commit", [(e.get("project_git") or {}).get("commit") for e in envs])
    for key in ("python", "torch", "numpy", "pyyaml", "platform"):
        same(key, [e.get(key) for e in envs])
    seeds = [manifest_seed(m) for m in manifests]
    if reasons:
        verdict = "NOT_COMPARABLE"
    elif any(s["seed_mode"] == NOT_RECORDED for s in seeds):
        verdict = "LEGACY_UNSEEDED"
    elif not all(s["reproducible_batch"] for s in seeds):
        verdict = "UNSEEDED_OR_UNVERIFIED"
    elif len({s["seed_value"] for s in seeds}) > 1:
        verdict = "SEED_SWEEP"
    else:
        verdict = "SAME_SEED_COMPARABLE"
    return {"verdict": verdict, "seeds": seeds, "reasons": reasons}


# -- CBS request states ----------------------------------------------------------

REQUEST_STATES_NOTE = (
    "INITIAL_STATE calls (Pure CBS: t=0, S_0) and IL_STALL_TRIGGER calls (Global "
    "Hybrid: t>=1, the stalled S_t) give the solver different start states: they "
    "are different CBS planning problems and must not be pooled as the same problem."
)


def _canon_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def cbs_request_states(records) -> List[Dict[str, Any]]:
    """One entry per CBS call: kind, request timestep, start-state and
    solver-input sha256 (from the run's own files)."""
    from src.compare.compare_csv import REQUEST_KIND, _request_timestep  # lazy: avoids an import cycle

    out = []
    for rec in records:
        kind = REQUEST_KIND.get(rec.method)
        if kind is None:
            continue
        for i, call in enumerate(rec.cbs_calls, start=1):
            t = _request_timestep(rec.method, call)
            state = {str(a): list(p) for a, p in sorted(rec.trajectory[t].items())}
            solver_input = (Path(rec.run_dir) / "cbs" / f"cbs_call_{i:02d}" / "outputs" / "paths"
                            / "replan" / "solver_input.yaml")
            out.append({
                "run_id": rec.run_id, "method": rec.method, "call_index": i,
                "request_kind": kind, "request_timestep": t,
                "start_state_sha256": _canon_sha(state),
                "solver_input_sha256": (hashlib.sha256(solver_input.read_bytes()).hexdigest()
                                        if solver_input.is_file() else None),
            })
    return out


# -- Output path length (Windows MAX_PATH) ----------------------------------------

RUN_FILE_SUFFIXES = ("_steps.csv", "_summary.csv", "_monitor.csv")
RUN_FILES = ("result.json",)
CBS_FILES = ("input.yaml", "solver_input.yaml", "output.yaml")
CBS_SUBDIR = ("outputs", "paths", "replan")   # src/replanner.py's fixed work_dir


def path_limit() -> Optional[int]:
    """Longest usable path in characters, or None when there is no limit.
    Windows without LongPathsEnabled: MAX_PATH 260 including the NUL -> 259."""
    if os.name != "nt":
        return None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as key:
            if winreg.QueryValueEx(key, "LongPathsEnabled")[0] == 1:
                return None
    except OSError:
        pass
    return 259


def output_paths(out_dir, scenarios, methods, *, cbs_methods, max_cbs_calls: int = 99) -> List[str]:
    """Every file path a batch run would write, with run ids of the real
    format (make_run_id) and the longest call index."""
    from datetime import datetime

    from src.common.episode_logger import make_run_id

    stamp = datetime(2000, 1, 1, 0, 0, 0)   # fixed-width timestamp, same length as any real one
    runs = Path(out_dir).resolve() / "runs"
    paths = []
    for sc in scenarios:
        for m in methods:
            rid = make_run_id(Path(sc).name, m, timestamp=stamp)
            rd = runs / rid
            paths += [str(rd / (rid + s)) for s in RUN_FILE_SUFFIXES] + [str(rd / f) for f in RUN_FILES]
            if m in cbs_methods:
                cd = rd.joinpath("cbs", f"cbs_call_{max_cbs_calls:02d}", *CBS_SUBDIR)
                paths += [str(cd / f) for f in CBS_FILES]
    return paths


_AUTO = object()


def path_length_errors(out_dir, scenarios, methods, *, cbs_methods, limit=_AUTO) -> List[str]:
    """[] if every output path fits; ``limit=None`` disables the check."""
    limit = path_limit() if limit is _AUTO else limit
    if limit is None:
        return []
    longest = max(output_paths(out_dir, scenarios, methods, cbs_methods=cbs_methods), key=len)
    if len(longest) > limit:
        return [f"output path would be {len(longest)} chars (> {limit}, Windows MAX_PATH): {longest}"]
    return []
