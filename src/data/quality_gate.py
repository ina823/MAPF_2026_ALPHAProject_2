"""PILOT data-quality gate v0 (read-only): review ledger, OOD classes, solvability.

Records and checks; never decides. Nothing here changes the registry, merges
families, resolves a REVIEW, fails an instance on OOD, or turns another
method's plan into a CBS result (docs/pilot_quality_gate_v0.md).

  Review ledger (registry/review_ledger_v0.jsonl)
    One audit line per REVIEW relation. v0 allows only status HOLD with
    final_decision null; RELATED/UNRELATED need a separate approved step.
    The ledger is NOT a ReviewBook: Registry.evaluate() never sees it, so an
    entry here leaves every relation unresolved.

  OOD classification (on top of il_compat.check_instance)
    TAIL             outside [p0.5, p99.5], inside the training [min, max]
    BEYOND_OBSERVED  outside the training [min, max]
    Descriptive only; auto_fail stays False and neither bound is the
    Validation exclusion policy.

  Solvability (registry/pilot_solvability_v0.json)
    ``cbs`` holds exactly what the CBS adapter returned; ``other_methods``
    holds evidence from anything else. Derived values (cbs_solved,
    solution_existence, witness_evidence) are computed here and may not be
    stored, so a non-CBS plan can never be read as a CBS success.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.data.fingerprint import TRANSFORMS, grid_from_rows, transform_grid
from src.data.il_compat import IN_RANGE
from src.data.registry import REVIEW, REVIEW_RELATIONS, SUBMAP, Registry, instance_fp, relate

GATE_VERSION = 0

# -- review ledger ------------------------------------------------------------

HOLD = "HOLD"
LEDGER_STATUSES = (HOLD,)            # v0: nothing else may be written
LEDGER_FIELDS = ("review_id", "a", "a_instance_fp", "b", "b_instance_fp", "relation", "status",
                 "final_decision", "decision_owner", "evidence", "basis", "recorded_at", "recorded_by")
REVIEW_ID = re.compile(r"^RV\d{4}$")


class GateError(ValueError):
    """A gate input file cannot be read."""


def load_jsonl(path) -> List[Dict]:
    """Strict JSONL: every line terminated, none empty."""
    data = Path(path).read_bytes()
    if data and not data.endswith(b"\n"):
        raise GateError(f"{path}: last line is not terminated (partial write?)")
    out = []
    for n, line in enumerate(data.decode("utf-8").split("\n")[:-1] if data else [], 1):
        if not line.strip():
            raise GateError(f"{path}:{n}: empty line")
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise GateError(f"{path}:{n}: invalid JSON ({exc})") from exc
    return out


def submap_evidence(a: Dict, b: Dict) -> Dict:
    """Shapes and the number of distinct placements (transformed grid, offset) of
    the smaller trimmed map inside the larger one."""
    ga, gb = grid_from_rows(a["grid_trim_rows"]), grid_from_rows(b["grid_trim_rows"])
    small, big = (ga, gb) if ga.size <= gb.size else (gb, ga)
    seen, placements = set(), set()
    for k, flip in TRANSFORMS:
        t = transform_grid(small, k, flip)
        key = (t.shape, t.tobytes())
        if key in seen:
            continue
        seen.add(key)
        h, w = t.shape
        if h > big.shape[0] or w > big.shape[1]:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(big, (h, w))
        for r, c in np.argwhere((windows == t).all(axis=(2, 3))):
            placements.add((key, int(r), int(c)))
    return {"smaller_shape": list(small.shape), "larger_shape": list(big.shape),
            "submap_placements": len(placements)}


def validate_review_ledger(entries: List[Dict], registry: Registry) -> List[str]:
    """Problems with the ledger; [] = every entry matches the registry and every
    unresolved REVIEW relation has exactly one HOLD entry."""
    problems = []
    ids = registry.by_id()
    seen_ids, seen_keys = set(), set()
    for n, e in enumerate(entries, 1):
        tag = f"ledger line {n} ({e.get('review_id')})"
        missing = [f for f in LEDGER_FIELDS if f not in e]
        if missing:
            problems.append(f"{tag}: missing fields {missing}")
            continue
        if not REVIEW_ID.match(str(e["review_id"])):
            problems.append(f"{tag}: review_id must look like RV0001")
        if e["review_id"] in seen_ids:
            problems.append(f"{tag}: duplicate review_id")
        seen_ids.add(e["review_id"])
        if e["status"] not in LEDGER_STATUSES:
            problems.append(f"{tag}: status {e['status']!r} not allowed in gate v0 (only {HOLD})")
        if e["final_decision"] is not None or e["decision_owner"] is not None:
            problems.append(f"{tag}: final_decision/decision_owner must stay null in gate v0; "
                            f"a RELATED/UNRELATED decision needs a separate approved step")
        if e["relation"] not in REVIEW_RELATIONS:
            problems.append(f"{tag}: {e['relation']!r} is not a REVIEW relation")
            continue
        if not str(e["basis"]).strip():
            problems.append(f"{tag}: empty basis")
        ra, rb = ids.get(e["a"]), ids.get(e["b"])
        if ra is None or rb is None:
            problems.append(f"{tag}: unknown instance id(s) {e['a']!r} / {e['b']!r}")
            continue
        for side, rec in (("a", ra), ("b", rb)):
            if e[f"{side}_instance_fp"] != instance_fp(rec):
                problems.append(f"{tag}: {side}_instance_fp does not match {rec['instance_id']}")
        key = tuple(sorted((e["a"], e["b"]))) + (e["relation"],)
        if key in seen_keys:
            problems.append(f"{tag}: duplicate entry for {key}")
        seen_keys.add(key)
        recomputed = {r["relation"]: r for r in relate(ra, rb) if r["status"] == REVIEW}
        if e["relation"] not in recomputed:
            problems.append(f"{tag}: {e['relation']} is not recomputed between {e['a']} and {e['b']}")
            continue
        if e["relation"] == SUBMAP:
            ev = submap_evidence(ra, rb)
            for k, v in ev.items():
                if e["evidence"].get(k) != v:
                    problems.append(f"{tag}: evidence {k}={e['evidence'].get(k)!r}, recomputed {v!r}")
    for rel in registry.all_relations():
        if rel["status"] != REVIEW or rel.get("review_decision") is not None:
            continue
        key = tuple(sorted((rel["a"], rel["b"]))) + (rel["relation"],)
        if key not in seen_keys:
            problems.append(f"unresolved REVIEW {rel['relation']} {rel['a']} / {rel['b']} has no ledger entry")
    return problems


def holds_by_instance(entries: List[Dict]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for e in entries:
        if e.get("status") == HOLD:
            for side in ("a", "b"):
                out.setdefault(e[side], []).append(e["review_id"])
    return out


# -- OOD classification -------------------------------------------------------

TAIL, BEYOND_OBSERVED = "TAIL", "BEYOND_OBSERVED"
OOD_TAIL, OOD_BEYOND_OBSERVED = "OOD_TAIL", "OOD_BEYOND_OBSERVED"
OOD_CAUSES = {"wall_count": "DENSE_WALL_VIEW", "goal_row": "LONG_VERTICAL_GOAL_HINT",
              "goal_col": "LONG_HORIZONTAL_GOAL_HINT", "robot_count": "CROWDED_VIEW",
              "other_goal_count": "CROWDED_VIEW"}
THRESHOLD_STATUS = ("PROVISIONAL (development record only): [p0.5, p99.5] and the training "
                    "[min, max] describe the IL training distribution; they are not the "
                    "Validation exclusion policy")


def classify_ood(check: Dict, reference: Dict) -> Dict:
    """Grade each il_compat.check_instance() warning against the training [min, max]."""
    items = []
    for w in check["warnings"]:
        f = reference["features"][w["feature"]]
        lo, hi = f["min"], f["max"]
        values = w["value"] if isinstance(w["value"], list) else [w["value"]]
        beyond = any(v < lo or v > hi for v in values)
        items.append(dict(w, training_observed=[lo, hi], severity=BEYOND_OBSERVED if beyond else TAIL,
                          cause=OOD_CAUSES[w["feature"]]))
    if not items:
        cls = IN_RANGE
    else:
        cls = OOD_BEYOND_OBSERVED if any(i["severity"] == BEYOND_OBSERVED for i in items) else OOD_TAIL
    return {"ood_class": cls, "causes": sorted({i["cause"] for i in items}), "items": items,
            "auto_fail": False, "threshold_status": THRESHOLD_STATUS,
            "reference_sha256": reference["source"]["sha256"]}


# -- solvability --------------------------------------------------------------

CBS_STATUSES = ("success", "timeout", "no_solution", "invalid_path", "input_error", "solver_error")
OTHER_RESULTS = ("FOUND_VALID_PLAN", "NOT_FOUND", "NOT_RUN")
DERIVED_FIELDS = ("cbs_solved", "solution_existence", "witness_evidence")
CBS_FIELDS = ("status", "timeout_sec", "latency_sec", "solver_commit", "pythonhashseed",
              "source_file", "source_sha256", "source_entry")
OTHER_FIELDS = ("method", "result", "path_guard_ok", "archived", "source")
CONFIRMED_BY_CBS = "CONFIRMED_BY_CBS"
CONFIRMED_BY_OTHER_METHOD = "CONFIRMED_BY_OTHER_METHOD"
NO_SOLUTION_REPORTED_BY_CBS = "NO_SOLUTION_REPORTED_BY_CBS"
UNKNOWN = "UNKNOWN"


def derive_solvability(rec: Dict) -> Dict:
    """Derived view of one record. CBS status is passed through unchanged."""
    cbs = rec["cbs"]
    cbs_solved = cbs["status"] == "success" and cbs.get("path_guard_ok") is True
    witnesses = [m for m in rec.get("other_methods", [])
                 if m["result"] == "FOUND_VALID_PLAN" and m.get("path_guard_ok") is True]
    witness = None
    if cbs_solved:
        existence = CONFIRMED_BY_CBS
    elif witnesses:
        existence = CONFIRMED_BY_OTHER_METHOD
        witness = "ARCHIVED" if any(m["archived"] for m in witnesses) else "UNARCHIVED"
    elif cbs["status"] == "no_solution":
        existence = NO_SOLUTION_REPORTED_BY_CBS
    else:
        existence = UNKNOWN
    return {"cbs_status": cbs["status"], "cbs_solved": cbs_solved, "solution_existence": existence,
            "witness_evidence": witness}


def validate_solvability(doc: Dict, registry: Registry) -> List[str]:
    problems = []
    if doc.get("gate_version") != GATE_VERSION:
        problems.append(f"solvability gate_version {doc.get('gate_version')!r} != {GATE_VERSION}")
    ids = registry.by_id()
    seen = set()
    for rec in doc.get("records", []):
        iid = rec.get("instance_id")
        tag = f"solvability {iid}"
        if iid in seen:
            problems.append(f"{tag}: duplicate record")
        seen.add(iid)
        if iid not in ids:
            problems.append(f"{tag}: not in the registry")
        elif rec.get("instance_fp") != instance_fp(ids[iid]):
            problems.append(f"{tag}: instance_fp does not match the registry")
        stored = [f for f in DERIVED_FIELDS if f in rec or f in rec.get("cbs", {})]
        if stored:
            problems.append(f"{tag}: derived field(s) {stored} must not be stored")
        cbs = rec.get("cbs")
        if not isinstance(cbs, dict) or [f for f in CBS_FIELDS if f not in cbs]:
            problems.append(f"{tag}: cbs block missing or incomplete (needs {list(CBS_FIELDS)})")
            continue
        if cbs["status"] not in CBS_STATUSES:
            problems.append(f"{tag}: cbs.status {cbs['status']!r} is not an adapter status")
        if cbs["status"] == "success" and not isinstance(cbs.get("path_guard_ok"), bool):
            problems.append(f"{tag}: a CBS success needs path_guard_ok")
        for m in rec.get("other_methods", []):
            if [f for f in OTHER_FIELDS if f not in m]:
                problems.append(f"{tag}: other_methods entry needs {list(OTHER_FIELDS)}")
                continue
            if "cbs" in str(m["method"]).lower():
                problems.append(f"{tag}: other_methods may not hold a CBS result ({m['method']!r}); "
                                f"CBS results go in the cbs block only")
            if m["result"] not in OTHER_RESULTS:
                problems.append(f"{tag}: other method result {m['result']!r} not in {OTHER_RESULTS}")
            if m["result"] == "FOUND_VALID_PLAN" and m["path_guard_ok"] is not True:
                problems.append(f"{tag}: FOUND_VALID_PLAN without a Path Guard pass")
            if not isinstance(m["archived"], bool):
                problems.append(f"{tag}: archived must be true/false")
    for r in registry.records:
        if r["split"] == "PILOT" and r["instance_id"] not in seen:
            problems.append(f"{r['instance_id']}: PILOT without a solvability record")
    return problems


def check_cbs_source(rec: Dict, root: Path) -> Tuple[List[str], List[str]]:
    """(problems, notes): compare the cbs block with the raw check output it cites.

    A missing source file is a note (outputs/ is not tracked), a mismatch a problem."""
    cbs = rec["cbs"]
    path = Path(root) / cbs["source_file"]
    if not path.is_file():
        return [], [f"{rec['instance_id']}: CBS source {cbs['source_file']} not present here; not re-checked"]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != cbs["source_sha256"]:
        return [f"{rec['instance_id']}: {cbs['source_file']} sha256 differs from the record"], []
    entries = [e for e in json.loads(raw)["cbs"] if e["pilot"] == cbs["source_entry"]]
    if len(entries) != 1:
        return [f"{rec['instance_id']}: entry {cbs['source_entry']!r} not found once in the source"], []
    e = entries[0]
    problems = []
    if e["status"] != cbs["status"]:
        problems.append(f"{rec['instance_id']}: source status {e['status']!r} != record {cbs['status']!r}")
    if abs(e["latency_sec"] - cbs["latency_sec"]) > 0.01:
        problems.append(f"{rec['instance_id']}: source latency {e['latency_sec']} != record {cbs['latency_sec']}")
    for f in ("path_guard_ok", "sum_of_path_lengths", "makespan"):
        if e.get(f) != cbs.get(f):
            problems.append(f"{rec['instance_id']}: source {f} {e.get(f)!r} != record {cbs.get(f)!r}")
    return problems, []


# -- independence (for new PILOT candidates) -----------------------------------

def independence_check(registry: Registry, candidate: Dict) -> Dict:
    """Every relation (confirmed or REVIEW) between ``candidate`` and the registry.
    Stricter than Registry.evaluate(): a PILOT candidate counts as independent
    only with no relation at all."""
    rels = [dict(r, other=rec["instance_id"]) for rec in registry.records for r in relate(rec, candidate)]
    return {"independent": not rels, "relations": rels}


# -- gate report --------------------------------------------------------------

def gate_report(registry: Registry, ledger: List[Dict], solvability: Dict,
                ood: Dict[str, Dict], root: Optional[Path] = None) -> Dict:
    """Per-PILOT gate flags. ``ood`` maps instance_id -> classify_ood() result.

    gate_status is CLEAR or FLAGGED (needs a human look); the gate never rejects."""
    problems = validate_review_ledger(ledger, registry) + validate_solvability(solvability, registry)
    notes = []
    solv = {r["instance_id"]: r for r in solvability.get("records", [])}
    if root is not None:
        for rec in solv.values():
            if isinstance(rec.get("cbs"), dict) and "source_file" in rec["cbs"]:
                p, n = check_cbs_source(rec, root)
                problems += p
                notes += n
    holds = holds_by_instance(ledger)
    instances = []
    for r in registry.records:
        if r["split"] != "PILOT":
            continue
        iid = r["instance_id"]
        flags = []
        if holds.get(iid):
            flags.append("REVIEW_HOLD")
        o = ood.get(iid)
        if o is None:
            problems.append(f"{iid}: no OOD classification")
        elif o["ood_class"] != IN_RANGE:
            flags.append(o["ood_class"])
        s = derive_solvability(solv[iid]) if iid in solv and "cbs" in solv[iid] else None
        if s is not None:
            if not s["cbs_solved"]:
                flags.append(f"CBS_{s['cbs_status'].upper()}")
            if s["witness_evidence"] == "UNARCHIVED":
                flags.append("WITNESS_UNARCHIVED")
        instances.append({"instance_id": iid, "review_holds": holds.get(iid, []),
                          "ood": o, "solvability": s, "flags": flags,
                          "gate_status": "FLAGGED" if flags else "CLEAR", "auto_reject": False})
    return {"gate_version": GATE_VERSION, "problems": problems, "notes": notes, "instances": instances}
