"""Read-only data inventory and DEV / Validation / Test leakage report.

Builds one record per known instance -- every ``scenarios/**/scenario_*.json``
plus the DEV instances that development scripts build in memory (map copies,
added agents) -- and compares all pairs with ``registry.relate`` (the same
rules as registry/instances_v1.jsonl, docs/fingerprint_registry_v1.md).
Nothing here writes, moves or re-labels a file; the registry stays the only
place a split is assigned.

data_role (what a record may be used for, docs/data_split_plan_v1.md):
    DEV         development side: registry split DEV / PILOT / EXTERNAL_REF,
                or a pre-registration whose split_intent names DEV / PILOT,
                or an in-memory instance built by a DEV-only script
    VALIDATION  registry split VAL
    TEST        registry split TEST
    UNKNOWN     no recorded evidence -- never guessed

Pair findings are split into
    CONFIRMED   exactly decidable: EXACT (same instance), RELABEL_EQUIV (agent
                ids permuted), TRIM_EQUIV, SYMMETRY_EQUIV (rotated/reflected
                together), SAME_MAP, and byte-identical scenario files
    REVIEW      needs a person: SUBMAP, NEAR_MAP, STRUCT_MATCH
and every pair is tagged with the two roles, so DEV-VALIDATION, DEV-TEST,
VALIDATION-TEST and UNKNOWN pairs are counted separately.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from src.data.fingerprint import compute_fingerprints, fingerprint_scenario
from src.data.registry import (CONFIRMED, DOMAIN, EXACT, REVIEW, Registry, instance_fp, relate)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEV, VALIDATION, TEST, UNKNOWN = "DEV", "VALIDATION", "TEST", "UNKNOWN"
DATA_ROLES = (DEV, VALIDATION, TEST, UNKNOWN)
SPLIT_TO_ROLE = {"DEV": DEV, "PILOT": DEV, "EXTERNAL_REF": DEV, "VAL": VALIDATION, "TEST": TEST}

# Pre-registrations that state a split for files not (yet) in the registry.
# Each is read from disk; the role is taken from the FIRST word of its
# ``split_intent`` and only if that word is a registry split.
PREREGISTRATIONS = (
    ("scenarios/pilot_v2/", "registry/pilot_v2_prereg_v0.json"),
    ("scenarios/warehouse_v3_dev/", "scenarios/warehouse_v3_dev/prereg_wh8a_v0.json"),
)

# DEV instances that exist only in memory, built by DEV-only scripts
# (module, builder, name). Their outputs live under outputs/*_dev.
IN_MEMORY_DEV = (
    ("scripts.run_selective_real_e2e_dev", "scenario", "S1"),
    ("scripts.run_selective_real_e2e_dev", "scenario", "S2"),
    ("scripts.run_selective_real_e2e_dev", "scenario", "S1B"),
    ("scripts.run_selective_real_e2e_dev", "scenario", "S3"),
    ("scripts.run_selective_real_e2e_dev", "scenario", "CORRIDOR"),
    ("scripts.run_selective_real_e2e_dx_dev", "scenario", "D1"),
    ("scripts.run_selective_real_e2e_dx_dev", "scenario", "D2"),
    ("scripts.run_selective_real_e2e_dx_dev", "scenario", "D3"),
    ("scripts.run_selective_p3_dev", "scenario", "P3Q"),
    ("scripts.run_selective_p3_dev", "scenario", "P3F"),
)

USAGE_SCAN_DIRS = ("outputs", "docs")
USAGE_SCAN_SUFFIXES = (".json", ".jsonl", ".csv", ".md")
USAGE_SCAN_MAX_BYTES = 2_000_000


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rel(path: Path, root: Path) -> str:
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


# -- roles ----------------------------------------------------------------------

def prereg_roles(root: Path = PROJECT_ROOT) -> Dict[str, Dict]:
    """{directory prefix: {role, split, evidence, split_intent}} from pre-registrations on disk."""
    out = {}
    for prefix, doc in PREREGISTRATIONS:
        path = Path(root) / doc
        if not path.is_file():
            continue
        intent = json.loads(path.read_text(encoding="utf-8")).get("split_intent")
        word = str(intent or "").split(" ", 1)[0].strip("(),.;:")
        if word in SPLIT_TO_ROLE:
            out[prefix] = {"role": SPLIT_TO_ROLE[word], "split": word, "evidence": doc,
                           "split_intent": intent}
    return out


def _generation_method(fp: Dict, scenario: Dict) -> str:
    gen = scenario.get("generator")
    if isinstance(gen, dict):
        name = gen.get("name") or gen.get("generator") or "generator"
        seed = gen.get("seed", gen.get("generation_seed"))
        return f"generator:{name}" + (f" seed={seed}" if seed is not None else "")
    return "manual (no generator record in the file)"


# -- records --------------------------------------------------------------------

def file_records(root: Path = PROJECT_ROOT, registry: Optional[Registry] = None) -> List[Dict]:
    """One record per scenarios/**/scenario_*.json (sorted), role from registry or prereg."""
    root = Path(root)
    by_source = {r["source_path"]: r for r in (registry.records if registry else [])}
    prereg = prereg_roles(root)
    out = []
    for path in sorted((root / "scenarios").rglob("scenario_*.json")):
        source = _rel(path, root)
        fp = fingerprint_scenario(path)
        scenario = json.loads(path.read_text(encoding="utf-8"))
        rec = dict(fp, kind="FILE", source_path=source, generation_method=_generation_method(fp, scenario))
        reg = by_source.get(source)
        if reg is not None:
            rec.update(data_role=SPLIT_TO_ROLE[reg["split"]], split=reg["split"],
                       registry_instance_id=reg["instance_id"], family_id=reg["family_id"],
                       structure_type=reg.get("structure_type"),
                       structure_descriptor=reg.get("structure_descriptor"),
                       role_evidence=f"registry/instances_v1.jsonl {reg['instance_id']}")
            if reg["scenario_json_sha256"] != fp["scenario_json_sha256"]:
                rec["registry_sha_mismatch"] = True
        else:
            hit = next((v for p, v in prereg.items() if source.startswith(p)), None)
            rec.update(registry_instance_id=None, family_id=None, structure_type=None,
                       structure_descriptor=None)
            if hit is not None:
                rec.update(data_role=hit["role"], split=hit["split"],
                           role_evidence=f"{hit['evidence']} split_intent: {hit['split_intent']}")
            else:
                rec.update(data_role=UNKNOWN, split=None, role_evidence=None)
        out.append(rec)
    return out


def in_memory_records(builders: Sequence = IN_MEMORY_DEV) -> List[Dict]:
    """Fingerprints of the in-memory DEV instances (built by the scripts' own code)."""
    out = []
    for module, func, name in builders:
        sid = f"{module.rsplit('.', 1)[-1]}:{name}"
        try:
            grid, starts, goals, desc = getattr(importlib.import_module(module), func)(name)
        except Exception as exc:   # reported, never skipped silently
            out.append({"kind": "IN_MEMORY", "scenario_id": sid, "data_role": DEV, "split": "DEV",
                        "role_evidence": f"built only by DEV script {module}",
                        "build_error": f"{type(exc).__name__}: {exc}"})
            continue
        ids = sorted(starts)
        fp = compute_fingerprints(grid, [starts[a] for a in ids], [goals[a] for a in ids])
        out.append(dict(fp, kind="IN_MEMORY", scenario_id=sid, map_id=None, source_path=None,
                        scenario_json_sha256=None, map_file_sha256=None, agent_ids_in_file=ids,
                        agent_id_matches_index=ids == list(range(len(ids))),
                        data_role=DEV, split="DEV", registry_instance_id=None, family_id=None,
                        structure_type=None, structure_descriptor=None, description=desc,
                        generation_method=f"in-memory: {module}.{func}({name!r})",
                        role_evidence=f"built only by DEV script {module}"))
    return out


def scan_usage(records: List[Dict], root: Path = PROJECT_ROOT,
               dirs: Sequence[str] = USAGE_SCAN_DIRS) -> None:
    """Add ``used_in_development`` (USED / NO_RECORD_FOUND) and the top-level folders
    whose text files name the record (file name, scenario_id or map_id).
    NO_RECORD_FOUND is not proof of non-use."""
    keys = {}
    for i, rec in enumerate(records):
        names = set()
        if rec.get("source_path"):
            names.add(Path(rec["source_path"]).stem.replace("scenario_", "", 1))
        for k in ("scenario_id", "map_id"):
            if rec.get(k) and rec["kind"] == "FILE":
                names.add(str(rec[k]))
        keys[i] = names
    hits = defaultdict(set)
    for d in dirs:
        base = Path(root) / d
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            rel = _rel(path, root)
            if not path.is_file() or path.suffix not in USAGE_SCAN_SUFFIXES \
                    or path.stat().st_size > USAGE_SCAN_MAX_BYTES:
                continue
            text = path.name + "\n" + path.read_text(encoding="utf-8", errors="replace")
            folder = "/".join(rel.split("/")[:2])
            for i, names in keys.items():
                if any(n in text for n in names):
                    hits[i].add(folder)
    for i, rec in enumerate(records):
        if rec["kind"] == "IN_MEMORY":
            rec["used_in_development"] = "USED"
            rec["usage_evidence"] = [rec["role_evidence"]]
        else:
            rec["used_in_development"] = "USED" if hits[i] else "NO_RECORD_FOUND"
            rec["usage_evidence"] = sorted(hits[i])


# -- pairs ----------------------------------------------------------------------

LEAK_PAIRS = {frozenset((DEV, VALIDATION)): "DEV-VALIDATION", frozenset((DEV, TEST)): "DEV-TEST",
              frozenset((VALIDATION, TEST)): "VALIDATION-TEST"}


def _label(rec: Dict) -> str:
    return rec.get("registry_instance_id") or rec.get("source_path") or rec["scenario_id"]


def pair_findings(records: List[Dict]) -> List[Dict]:
    """Every related pair: relation, CONFIRMED / REVIEW, roles and whether it is a leak."""
    usable = [r for r in records if "grid_fp" in r]
    out = []
    for i, a in enumerate(usable):
        for b in usable[i + 1:]:
            rels = relate(a, b)
            if (a.get("scenario_json_sha256") and a["scenario_json_sha256"] == b.get("scenario_json_sha256")):
                rels = [{"relation": "SAME_FILE_BYTES", "status": CONFIRMED, "details": {}}] + rels
            for rel in rels:
                roles = frozenset((a["data_role"], b["data_role"]))
                out.append({
                    "a": _label(a), "b": _label(b), "relation": rel["relation"], "status": rel["status"],
                    "details": rel["details"], "roles": sorted(roles),
                    "role_pair": LEAK_PAIRS.get(roles, "UNKNOWN" if UNKNOWN in roles else
                                                "SAME_ROLE" if len(roles) == 1 else "-".join(sorted(roles))),
                })
    return out


def seed_sweeps(runs: Iterable[Dict]) -> List[Dict]:
    """Group run entries {instance_fp, seed, source} by instance: >1 seed = one instance re-run,
    never several instances. A run whose seed was not recorded is kept as NOT_RECORDED."""
    by_inst = defaultdict(list)
    for r in runs:
        by_inst[r["instance_fp"]].append(r)
    out = []
    for key, rs in sorted(by_inst.items()):
        seeds = sorted({str(r.get("seed", "NOT_RECORDED")) for r in rs})
        out.append({"instance_fp": key, "runs": len(rs), "seeds": seeds, "counts_as_instances": 1,
                    "seed_sweep": len(seeds) > 1, "sources": sorted({r["source"] for r in rs})})
    return out


def manifest_runs(root: Path, records: List[Dict]) -> List[Dict]:
    """Run entries from every batch_manifest.json under outputs/ (scenario sha -> instance)."""
    by_sha = {r["scenario_json_sha256"]: r for r in records if r.get("scenario_json_sha256")}
    runs = []
    for path in sorted((Path(root) / "outputs").rglob("batch_manifest.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        rep = m.get("reproducibility") or {}
        seed = rep.get("seed_value") if rep else "NOT_RECORDED"
        for name, sha in (m.get("conditions", {}).get("scenarios") or {}).items():
            rec = by_sha.get(sha)
            runs.append({"instance_fp": instance_fp(rec) if rec else f"UNMATCHED:{name}:{sha}",
                         "seed": seed, "source": _rel(path, root)})
    return runs


REVIEW_LEDGERS = ("registry/review_ledger_v0.jsonl", "registry/pilot_v2_candidate_review_v0.jsonl")


def manual_reviews(root: Path = PROJECT_ROOT, ledgers: Sequence[str] = REVIEW_LEDGERS) -> List[Dict]:
    """Similarity reviews recorded by people (read as is); unresolved = no final_decision."""
    out = []
    for doc in ledgers:
        path = Path(root) / doc
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out.append({"ledger": doc, "review_id": r.get("review_id"),
                            "subject": r.get("a") or r.get("candidate_id"),
                            "other": r.get("b") or r.get("other"), "relation": r.get("relation"),
                            "status": r.get("status"), "final_decision": r.get("final_decision"),
                            "unresolved": r.get("final_decision") is None, "basis": r.get("basis")})
    return out


# -- report ---------------------------------------------------------------------

INVENTORY_FIELDS = ("scenario_id", "kind", "data_role", "split", "role_evidence", "registry_instance_id",
                    "family_id", "map_id", "source_path", "map_file_sha256", "grid_fp",
                    "scenario_json_sha256", "num_agents", "starts_rc", "goals_rc", "agent_ids_in_file",
                    "structure_type", "generation_method", "used_in_development", "usage_evidence",
                    "build_error", "registry_sha_mismatch")


def build_report(root: Path = PROJECT_ROOT, *, registry_path: Optional[Path] = None,
                 builders: Sequence = IN_MEMORY_DEV, scan: bool = True) -> Dict:
    root = Path(root)
    reg_path = Path(registry_path) if registry_path else root / "registry" / "instances_v1.jsonl"
    registry = Registry.load(reg_path) if reg_path.is_file() else Registry([])
    records = file_records(root, registry) + in_memory_records(builders)
    if scan:
        scan_usage(records, root)
    pairs = pair_findings(records)
    runs = manifest_runs(root, records)
    roles = Counter(r["data_role"] for r in records)
    leaks = [p for p in pairs if p["role_pair"] in LEAK_PAIRS.values()]
    errors = []
    for r in records:
        if r.get("build_error"):
            errors.append(f"{r['scenario_id']}: cannot build ({r['build_error']})")
        if r.get("registry_sha_mismatch"):
            errors.append(f"{r['source_path']}: file sha256 differs from the registry record")
    for p in pairs:
        if p["relation"] in (EXACT, "SAME_FILE_BYTES") and len(set(p["roles"])) > 1:
            errors.append(f"same instance in two roles: {p['a']} / {p['b']} {p['roles']}")
    return {
        "registry_path": _rel(reg_path, root) if reg_path.is_file() else None,
        "registry_sha256": _sha256(reg_path.read_bytes()) if reg_path.is_file() else None,
        "record_count": len(records),
        "role_counts": {k: roles.get(k, 0) for k in DATA_ROLES},
        "kind_counts": dict(Counter(r["kind"] for r in records)),
        "records": [{k: r.get(k) for k in INVENTORY_FIELDS if k in r} for r in records],
        "pairs": pairs,
        "pair_counts": dict(Counter(f"{p['status']}:{p['relation']}" for p in pairs)),
        "role_pair_counts": {name: sum(1 for p in pairs if p["role_pair"] == name)
                             for name in (*LEAK_PAIRS.values(), "UNKNOWN")},
        "leaks": leaks,
        "seed_sweeps": seed_sweeps(runs),
        "manual_reviews": manual_reviews(root),
        "errors": errors,
        "domains": {k: DOMAIN[k] for k in DOMAIN},
    }
