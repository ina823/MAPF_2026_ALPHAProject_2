"""Instance registry: relations, families, registration decisions, safe writes.

Rules (docs/fingerprint_registry_v1.md, sections 3-5):

  CONFIRMED relations (strict -> loose)
    EXACT           same grid_fp and same (start, goal) in agent-id order
    RELABEL_EQUIV   same grid_fp, same id-free (start, goal) set, different order
    TRIM_EQUIV      different grid_fp, same wall-trimmed grid and trimmed set
    SYMMETRY_EQUIV  same canonical map and canonical (start, goal) set after
                    rotating/reflecting map AND agents together
    SAME_MAP        same canonical map, different start/goal
  Only EXACT is "the same instance". RELABEL/TRIM/SYMMETRY/SAME_MAP are
  related instances: each keeps its own instance_id, nothing is merged, and
  identical execution is never assumed.

  REVIEW relations (never merge families automatically)
    SUBMAP          one trimmed map equals a window of the other (any D4)
    NEAR_MAP        same trimmed shape (any D4), 1..NEAR_MAP_MAX_DIFF cells differ
    STRUCT_MATCH    equal, non-null generator structure_descriptor

  Instance identity vs family (RULES_VERSION 2)
    instance_fp(rec) = sha256(grid_fp ":" agents_ordered_fp) is the ONLY
    instance identity (EXACT). Rotated/reflected, relabelled, trimmed fields
    (agents_canon_*, agents_set_fp, grid_trim_fp, map_canon_fp) are used for
    relations/families only, never as an instance identity.

  Decisions for a candidate in split S (domain(DEV)=domain(PILOT)=
  domain(EXTERNAL_REF)="development"; VAL and TEST are their own domains)
    REJECT  invalid instance, same file already registered (same or other
            split), EXACT with any record, a confirmed relation (or a review
            resolved RELATED) with a record of another domain, confirmed
            relations to two different families, VAL/TEST with agent_id !=
            list index, S = TEST with a STRUCT_MATCH to a record of another
            domain (whatever the review decision), joining a family that
            already has FAMILY_CAP records in S (new candidates only; the
            legacy Development import is exempt, see build_dev_registry)
    HOLD    an unresolved REVIEW relation, when S is VAL/TEST or the other
            record is in another domain
    ACCEPT  otherwise; joins the family of its confirmed relations, or a new one
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.data.fingerprint import (
    FP_SCHEMA,
    TRANSFORMS,
    grid_from_rows,
    transform_grid,
)

# -- vocabulary ---------------------------------------------------------------

SPLITS = ("DEV", "PILOT", "EXTERNAL_REF", "VAL", "TEST")
DOMAIN = {"DEV": "development", "PILOT": "development", "EXTERNAL_REF": "development",
          "VAL": "validation", "TEST": "test"}
HELD_OUT = ("VAL", "TEST")

EXACT = "EXACT"
RELABEL_EQUIV = "RELABEL_EQUIV"
TRIM_EQUIV = "TRIM_EQUIV"
SYMMETRY_EQUIV = "SYMMETRY_EQUIV"
SAME_MAP = "SAME_MAP"
SUBMAP = "SUBMAP"
NEAR_MAP = "NEAR_MAP"
STRUCT_MATCH = "STRUCT_MATCH"
CONFIRMED_RELATIONS = (EXACT, RELABEL_EQUIV, TRIM_EQUIV, SYMMETRY_EQUIV, SAME_MAP)
REVIEW_RELATIONS = (SUBMAP, NEAR_MAP, STRUCT_MATCH)
CONFIRMED = "CONFIRMED"
REVIEW = "REVIEW"

RULES_VERSION = 2          # decision rules (fingerprints themselves: fp_schema)

# REVIEW thresholds (fp_schema 1; pending approval, see docs section 3.3)
NEAR_MAP_MAX_DIFF = 2      # differing cells, same trimmed shape
SUBMAP_MIN_AREA = 9        # the smaller trimmed map must have >= this many cells
FAMILY_CAP = 3             # max records of one family per split, new candidates only

ACCEPT, HOLD, REJECT = "ACCEPT", "HOLD", "REJECT"

STRUCT_ABBR = {"corridor": "corr", "intersection": "int", "bottleneck": "bot",
               "cycle_ring": "cyc", "empty": "emp"}

REQUIRED_FIELDS = (
    "fp_schema", "instance_id", "family_id", "split", "source_path", "scenario_json_sha256",
    "grid_fp", "grid_trim_fp", "grid_trim_rows", "map_canon_fp", "agents_ordered_fp",
    "agents_set_fp", "agents_trim_ordered_fp", "agents_trim_set_fp", "agents_canon_ordered_fp",
    "agents_canon_set_fp", "structure_type", "relations",
)


class RegistryError(RuntimeError):
    """The registry file is malformed, inconsistent, or a write is unsafe."""


# -- relations ----------------------------------------------------------------

def _rel(relation: str, status: str, **details) -> Dict:
    return {"relation": relation, "status": status, "details": details}


def _d4(grid: np.ndarray) -> List[np.ndarray]:
    return [transform_grid(grid, k, f) for k, f in TRANSFORMS]


def _is_submap(small: np.ndarray, big: np.ndarray) -> bool:
    if small.size < SUBMAP_MIN_AREA or small.size >= big.size:
        return False
    for t in _d4(small):
        h, w = t.shape
        if h > big.shape[0] or w > big.shape[1]:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(big, (h, w))
        if (windows == t).all(axis=(2, 3)).any():
            return True
    return False


def _min_cell_diff(a: np.ndarray, b: np.ndarray) -> Optional[int]:
    diffs = [int((a != t).sum()) for t in _d4(b) if t.shape == a.shape]
    return min(diffs) if diffs else None


def relate(a: Dict, b: Dict) -> List[Dict]:
    """Relations between two fingerprint records (symmetric; [] = unrelated).

    Returns exactly one CONFIRMED relation, or zero or more REVIEW relations.
    """
    same_grid = a["grid_fp"] == b["grid_fp"]
    same_trim = a["grid_trim_fp"] == b["grid_trim_fp"]
    if same_grid and a["agents_ordered_fp"] == b["agents_ordered_fp"]:
        return [_rel(EXACT, CONFIRMED)]
    if same_grid and a["agents_set_fp"] == b["agents_set_fp"]:
        return [_rel(RELABEL_EQUIV, CONFIRMED, id_order_equal=False)]
    if same_trim and not same_grid and a["agents_trim_set_fp"] == b["agents_trim_set_fp"]:
        return [_rel(TRIM_EQUIV, CONFIRMED,
                     id_order_equal=a["agents_trim_ordered_fp"] == b["agents_trim_ordered_fp"])]
    if a["map_canon_fp"] == b["map_canon_fp"]:
        if a["agents_canon_set_fp"] == b["agents_canon_set_fp"]:
            return [_rel(SYMMETRY_EQUIV, CONFIRMED,
                         id_order_equal=a["agents_canon_ordered_fp"] == b["agents_canon_ordered_fp"])]
        details = {"grid_equal": same_grid, "trimmed_grid_equal": same_trim,
                   "num_agents": sorted([a["num_agents"], b["num_agents"]])}
        if same_trim:
            pa = {tuple(p) for p in _trim_pairs(a)}
            pb = {tuple(p) for p in _trim_pairs(b)}
            details["shared_start_goal_pairs"] = len(pa & pb)
        return [_rel(SAME_MAP, CONFIRMED, **details)]

    out = []
    ga, gb = grid_from_rows(a["grid_trim_rows"]), grid_from_rows(b["grid_trim_rows"])
    small, big = (ga, gb) if ga.size <= gb.size else (gb, ga)
    if _is_submap(small, big):
        out.append(_rel(SUBMAP, REVIEW, smaller_shape=list(small.shape), larger_shape=list(big.shape)))
    diff = _min_cell_diff(ga, gb)
    if diff is not None and 0 < diff <= NEAR_MAP_MAX_DIFF:
        out.append(_rel(NEAR_MAP, REVIEW, differing_cells=diff))
    da, db = a.get("structure_descriptor"), b.get("structure_descriptor")
    if da is not None and da == db:
        out.append(_rel(STRUCT_MATCH, REVIEW, descriptor=da))
    return out


def _trim_pairs(rec: Dict) -> List[List[int]]:
    r0, c0 = rec["grid_trim_offset"]
    return [[s[0] - r0, s[1] - c0, g[0] - r0, g[1] - c0]
            for s, g in zip(rec["starts_rc"], rec["goals_rc"])]


# -- reviews ------------------------------------------------------------------

RELATED, UNRELATED = "RELATED", "UNRELATED"


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def instance_fp(rec: Dict) -> str:
    """The instance identity: sha256 of ``grid_fp + ":" + agents_ordered_fp``.

    Two records have the same instance_fp exactly when they are EXACT. Unique
    within a registry because EXACT duplicates are refused; also keys review
    decisions so a decision can be made before the candidate has an id.
    """
    return _sha256_bytes(f"{rec['grid_fp']}:{rec['agents_ordered_fp']}".encode("utf-8"))


instance_key = instance_fp   # name used by ReviewBook callers


@dataclass
class ReviewBook:
    """Human decisions on REVIEW relations, keyed by (instance_key a, instance_key b,
    relation) -> RELATED / UNRELATED."""
    decisions: Dict[Tuple[str, str, str], Dict] = field(default_factory=dict)

    def add(self, a: str, b: str, relation: str, decision: str, reviewer: str, reason: str) -> None:
        if relation not in REVIEW_RELATIONS:
            raise RegistryError(f"{relation} is not a REVIEW relation")
        if decision not in (RELATED, UNRELATED):
            raise RegistryError(f"decision must be {RELATED} or {UNRELATED}")
        if not reviewer or not reason:
            raise RegistryError("a review needs a reviewer and a reason")
        self.decisions[_pair_key(a, b) + (relation,)] = {
            "decision": decision, "reviewer": reviewer, "reason": reason}

    def get(self, a: str, b: str, relation: str) -> Optional[str]:
        d = self.decisions.get(_pair_key(a, b) + (relation,))
        return d["decision"] if d else None


# -- decisions ----------------------------------------------------------------

@dataclass
class Decision:
    status: str
    split: str
    family_id: Optional[str]
    relations: List[Dict]
    reasons: List[str]


def _canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Registry:
    """In-memory registry. Records are append-only; split and ids never change."""

    def __init__(self, records: Iterable[Dict] = (), reviews: Optional[ReviewBook] = None,
                 source_sha256: Optional[str] = None):
        self.records: List[Dict] = []
        self.reviews = reviews or ReviewBook()
        self.source_sha256 = source_sha256   # sha of the file this was loaded from
        for rec in records:
            self._check_record(rec)
            self.records.append(rec)

    # -- loading / validation --
    def _check_record(self, rec: Dict) -> None:
        missing = [f for f in REQUIRED_FIELDS if f not in rec]
        if missing:
            raise RegistryError(f"record {rec.get('instance_id')!r} misses fields {missing}")
        if rec["fp_schema"] != FP_SCHEMA:
            raise RegistryError(f"record {rec['instance_id']} has fp_schema {rec['fp_schema']}, "
                                f"expected {FP_SCHEMA}")
        if rec["split"] not in SPLITS:
            raise RegistryError(f"record {rec['instance_id']} has unknown split {rec['split']!r}")
        if not rec["instance_id"].startswith(f"{rec['split']}_{rec['family_id']}_"):
            raise RegistryError(f"instance_id {rec['instance_id']} does not match its split/family")
        for other in self.records:
            if other["instance_id"] == rec["instance_id"]:
                raise RegistryError(f"duplicate instance_id {rec['instance_id']}")
            if other["source_path"] == rec["source_path"]:
                raise RegistryError(f"source {rec['source_path']} registered twice")

    @classmethod
    def load(cls, path, reviews: Optional[ReviewBook] = None) -> "Registry":
        data = Path(path).read_bytes()
        if data and not data.endswith(b"\n"):
            raise RegistryError(f"{path}: last line is not terminated (partial write?)")
        records = []
        for n, line in enumerate(data.decode("utf-8").split("\n")[:-1] if data else [], 1):
            if not line.strip():
                raise RegistryError(f"{path}:{n}: empty line")
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RegistryError(f"{path}:{n}: invalid JSON ({exc})") from exc
        return cls(records, reviews=reviews, source_sha256=_sha256_bytes(data))

    def by_id(self) -> Dict[str, Dict]:
        return {r["instance_id"]: r for r in self.records}

    # -- decisions --
    def evaluate(self, candidate: Dict, split: str, *, apply_family_cap: bool = True) -> Decision:
        """Decide whether ``candidate`` (a fingerprint_scenario()/compute_fingerprints()
        dict plus source info and structure_type) may be registered in ``split``.

        ``apply_family_cap=False`` is only for re-importing the legacy
        Development set (scripts/register_dev_scenarios.py), which predates
        the cap and is not re-judged by it."""
        if split not in SPLITS:
            raise RegistryError(f"unknown split {split!r}")
        reasons, holds, relations = [], [], []
        families = set()

        if split in HELD_OUT and candidate.get("agent_id_matches_index") is False:
            reasons.append("AGENT_ID_NOT_INDEX: held-out instances need agent_id == list index")

        for rec in self.records:
            same_file = (rec["source_path"] == candidate.get("source_path")
                         or (candidate.get("scenario_json_sha256") is not None
                             and rec["scenario_json_sha256"] == candidate["scenario_json_sha256"]))
            if same_file:
                code = "ALREADY_REGISTERED" if rec["split"] == split else "SPLIT_CHANGE"
                reasons.append(f"{code}: same scenario file as {rec['instance_id']} ({rec['split']})")
            cross = DOMAIN[rec["split"]] != DOMAIN[split]
            for rel in relate(rec, candidate):
                entry = dict(rel, other=rec["instance_id"])
                if rel["status"] == REVIEW:
                    entry["review_decision"] = self.reviews.get(instance_key(rec), instance_key(candidate),
                                                                rel["relation"])
                relations.append(entry)
                if rel["relation"] == EXACT:
                    reasons.append(f"DUPLICATE_INSTANCE: EXACT with {rec['instance_id']}")
                    continue
                related = rel["status"] == CONFIRMED or entry.get("review_decision") == RELATED
                unresolved = rel["status"] == REVIEW and entry.get("review_decision") is None
                if rel["relation"] == STRUCT_MATCH and split == "TEST" and cross:
                    reasons.append(f"STRUCT_MATCH_TEST_FORBIDDEN: with {rec['instance_id']} "
                                   f"({rec['split']}); a review decision does not lift this")
                    continue
                if related:
                    families.add(rec["family_id"])
                    if cross:
                        reasons.append(f"FAMILY_LEAK: {rel['relation']} with {rec['instance_id']} "
                                       f"({rec['split']})")
                elif unresolved and (split in HELD_OUT or cross):
                    holds.append(f"UNRESOLVED_REVIEW: {rel['relation']} with {rec['instance_id']}")

        if len(families) > 1:
            reasons.append(f"FAMILY_CONFLICT: related to families {sorted(families)}")
        family_id = next(iter(families)) if len(families) == 1 else None
        if family_id is not None and apply_family_cap:
            members = sum(1 for r in self.records if r["split"] == split and r["family_id"] == family_id)
            if members >= FAMILY_CAP:
                reasons.append(f"FAMILY_CAP: family {family_id} already has {members} record(s) "
                               f"in {split} (cap {FAMILY_CAP})")
        if reasons:
            return Decision(REJECT, split, family_id, relations, reasons + holds)
        if holds:
            return Decision(HOLD, split, family_id, relations, holds)
        if family_id is None:
            family_id = self._new_family_id(candidate.get("structure_type"))
        return Decision(ACCEPT, split, family_id, relations, [])

    def _new_family_id(self, structure_type: Optional[str]) -> str:
        abbr = STRUCT_ABBR.get(structure_type or "")
        if abbr is None:
            abbr = "".join(ch for ch in (structure_type or "unk") if ch.isalnum())[:4].lower() or "unk"
        used = [int(r["family_id"][len("f") + len(abbr):]) for r in self.records
                if r["family_id"].startswith("f" + abbr) and r["family_id"][1 + len(abbr):].isdigit()]
        return f"f{abbr}{(max(used) + 1) if used else 1:02d}"

    def register(self, candidate: Dict, split: str, *, registered_at: str,
                 extra: Optional[Dict] = None, apply_family_cap: bool = True) -> Dict:
        """Evaluate and, only on ACCEPT, append a new record. Raises otherwise."""
        decision = self.evaluate(candidate, split, apply_family_cap=apply_family_cap)
        if decision.status != ACCEPT:
            raise RegistryError(f"{decision.status}: " + "; ".join(decision.reasons))
        n = 1 + sum(1 for r in self.records
                    if r["split"] == split and r["family_id"] == decision.family_id)
        rec = dict(candidate)
        rec.update(extra or {})
        rec.update({
            "instance_id": f"{split}_{decision.family_id}_{n:03d}",
            "family_id": decision.family_id,
            "split": split,
            "registered_at": registered_at,
            "relations": [{"other": r["other"], "relation": r["relation"], "status": r["status"],
                           "details": r["details"]} for r in decision.relations],
        })
        rec.setdefault("structure_descriptor", None)
        self._check_record(rec)
        self.records.append(rec)
        return rec

    # -- serialisation --
    def to_jsonl(self) -> bytes:
        return "".join(_canonical_json(r) + "\n" for r in self.records).encode("utf-8")

    def all_relations(self) -> List[Dict]:
        """Every related pair, recomputed from fingerprints (not from stored fields)."""
        out = []
        for i, a in enumerate(self.records):
            for b in self.records[i + 1:]:
                for rel in relate(a, b):
                    entry = dict(rel, a=a["instance_id"], b=b["instance_id"])
                    if rel["status"] == REVIEW:
                        entry["review_decision"] = self.reviews.get(instance_key(a), instance_key(b),
                                                                    rel["relation"])
                    out.append(entry)
        return out

    def consistency_errors(self) -> List[str]:
        """Stored relations/families vs. relations recomputed from fingerprints."""
        errors = []
        ids = self.by_id()
        stored = set()
        for rec in self.records:
            for rel in rec["relations"]:
                if rel["other"] not in ids:
                    errors.append(f"{rec['instance_id']}: relation to unknown {rel['other']}")
                stored.add(_pair_key(rec["instance_id"], rel["other"]) + (rel["relation"],))
        recomputed = set()
        for rel in self.all_relations():
            recomputed.add(_pair_key(rel["a"], rel["b"]) + (rel["relation"],))
            a, b = ids[rel["a"]], ids[rel["b"]]
            related = rel["status"] == CONFIRMED or rel.get("review_decision") == RELATED
            if rel["relation"] == EXACT:
                errors.append(f"EXACT duplicate {rel['a']} / {rel['b']}")
            if related and a["family_id"] != b["family_id"]:
                errors.append(f"{rel['relation']} {rel['a']} / {rel['b']} in different families")
            if related and DOMAIN[a["split"]] != DOMAIN[b["split"]]:
                errors.append(f"FAMILY_LEAK {rel['relation']} {rel['a']} ({a['split']}) / "
                              f"{rel['b']} ({b['split']})")
            if (rel["status"] == REVIEW and rel.get("review_decision") is None
                    and (a["split"] in HELD_OUT or b["split"] in HELD_OUT)):
                errors.append(f"UNRESOLVED_REVIEW {rel['relation']} {rel['a']} / {rel['b']} "
                              f"touches a held-out split")
            if (rel["relation"] == STRUCT_MATCH and "TEST" in (a["split"], b["split"])
                    and DOMAIN[a["split"]] != DOMAIN[b["split"]]):
                errors.append(f"STRUCT_MATCH {rel['a']} / {rel['b']} crosses into TEST")
        for key in sorted(stored ^ recomputed):
            errors.append(f"stored/recomputed relation mismatch: {key}")
        groups = Counter((r["split"], r["family_id"]) for r in self.records)
        for (split, fid), n in sorted(groups.items()):
            if n > FAMILY_CAP and split != "DEV":   # DEV: legacy import, cap not retroactive
                errors.append(f"FAMILY_CAP exceeded: {fid} has {n} records in {split}")
        return errors

    def append_to_file(self, path, n_existing: int) -> None:
        """Rewrite ``path`` with all records, refusing unless the first
        ``n_existing`` records' bytes are unchanged (append-only) and the file
        still has the sha256 it was loaded with."""
        path = Path(path)
        old = path.read_bytes()
        head = Registry(self.records[:n_existing]).to_jsonl()
        if old != head:
            raise RegistryError(f"{path}: existing records changed (append-only violated)")
        new = self.to_jsonl()
        if not new.startswith(old):
            raise RegistryError(f"{path}: new content does not extend the old content")
        replace_file_checked(path, new, self.source_sha256)

    def family_summary(self) -> Dict:
        fams: Dict[str, Dict] = defaultdict(lambda: {"members": [], "splits": Counter(),
                                                     "structure_types": set()})
        for rec in self.records:
            f = fams[rec["family_id"]]
            f["members"].append(rec["instance_id"])
            f["splits"][rec["split"]] += 1
            f["structure_types"].add(rec["structure_type"])
        rels = self.all_relations()
        review_items = [{"a": r["a"], "b": r["b"], "relation": r["relation"], "details": r["details"],
                         "review_decision": r.get("review_decision")}
                        for r in rels if r["status"] == REVIEW]
        return {
            "fp_schema": FP_SCHEMA,
            "instances_sha256": _sha256_bytes(self.to_jsonl()),
            "instance_count": len(self.records),
            "split_counts": dict(sorted(Counter(r["split"] for r in self.records).items())),
            "family_count": len(fams),
            "families": {
                fid: {"count": len(f["members"]), "members": sorted(f["members"]),
                      "splits": dict(sorted(f["splits"].items())),
                      "structure_types": sorted(f["structure_types"])}
                for fid, f in sorted(fams.items())
            },
            "relation_counts": dict(sorted(Counter(r["relation"] for r in rels).items())),
            "review_items": review_items,
            "unresolved_review_count": sum(1 for r in review_items if r["review_decision"] is None),
            "rules_version": RULES_VERSION,
            "thresholds": {"NEAR_MAP_MAX_DIFF": NEAR_MAP_MAX_DIFF, "SUBMAP_MIN_AREA": SUBMAP_MIN_AREA,
                           "FAMILY_CAP": FAMILY_CAP},
        }

    def summary_json(self) -> bytes:
        return (json.dumps(self.family_summary(), indent=2, sort_keys=True, ensure_ascii=False)
                + "\n").encode("utf-8")


# -- safe file writes ---------------------------------------------------------

def _write_temp(directory: Path, data: bytes) -> Path:
    fd, tmp = tempfile.mkstemp(prefix=".tmp_registry_", dir=str(directory))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        os.unlink(tmp)
        raise
    return Path(tmp)


def create_files_exclusive(files: Sequence[Tuple[Path, bytes]]) -> None:
    """Create every file, or none: refuse if any exists; never leave a partial file.

    Each file is fully written and fsync'ed under a temporary name, then
    published with os.link (fails if the target exists, no overwrite). If a
    later file cannot be published, the earlier ones are removed again.
    """
    files = [(Path(p), d) for p, d in files]
    existing = [str(p) for p, _ in files if p.exists()]
    if existing:
        raise RegistryError(f"refusing to overwrite existing file(s): {existing}")
    temps, published = [], []
    try:
        for path, data in files:
            path.parent.mkdir(parents=True, exist_ok=True)
            temps.append((_write_temp(path.parent, data), path))
        for tmp, path in temps:
            os.link(tmp, path)
            published.append(path)
    except BaseException:
        for path in published:
            path.unlink()
        raise
    finally:
        for tmp, _ in temps:
            if tmp.exists():
                tmp.unlink()


def replace_file_checked(path, data: bytes, expected_sha256: Optional[str]) -> None:
    """Atomically replace ``path`` with ``data`` only if it still has ``expected_sha256``."""
    path = Path(path)
    current = _sha256_bytes(path.read_bytes()) if path.exists() else None
    if current != expected_sha256:
        raise RegistryError(f"{path} changed since it was loaded (sha256 {current} != {expected_sha256})")
    tmp = _write_temp(path.parent, data)
    try:
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
