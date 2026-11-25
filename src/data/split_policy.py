"""Usage rules per data role, result labels, and the held-out (Validation / Test) freeze.

Rules (docs/data_split_plan_v1.md sections 2, 6.3, 9; docs/validation_test_split_v1.md):
    DEV         development and debugging only. A DEV result is never a
                Validation or Test result.
    VALIDATION  parameter selection / tuning only.
    TEST        final evaluation only, and only on a frozen set whose freeze
                record still verifies (same registry records, same file bytes).
    UNKNOWN     nothing until a role is recorded.

A freeze record pins one split's membership and content: every instance's
scenario / map file sha256, its instance_fp, the registry file sha256 and a
commit. ``verify_freeze`` re-hashes the files and re-reads the registry; any
change (a file edited, a record added to or missing from the split, a split
changed) makes the set unusable -- a new set must be frozen instead.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from src.data.inventory import DEV, TEST, UNKNOWN, VALIDATION, SPLIT_TO_ROLE
from src.data.registry import HELD_OUT, Registry, instance_fp

DEVELOPMENT, DEBUGGING, PARAMETER_SELECTION, FINAL_EVALUATION = (
    "development", "debugging", "parameter_selection", "final_evaluation")
PURPOSES = (DEVELOPMENT, DEBUGGING, PARAMETER_SELECTION, FINAL_EVALUATION)
ALLOWED_USE = {
    DEV: (DEVELOPMENT, DEBUGGING),
    VALIDATION: (PARAMETER_SELECTION,),
    TEST: (FINAL_EVALUATION,),
    UNKNOWN: (),
}
FREEZE_SCHEMA = 1


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_use(role: str, purpose: str, *, freeze: Optional[Dict] = None, freeze_problems=None) -> List[str]:
    """Reasons ``role`` data may not be used for ``purpose`` ([] = allowed).

    TEST also needs ``freeze`` and an empty ``freeze_problems`` (from verify_freeze).
    """
    if purpose not in PURPOSES:
        return [f"unknown purpose {purpose!r}"]
    if role not in ALLOWED_USE:
        return [f"unknown data role {role!r}"]
    reasons = []
    if purpose not in ALLOWED_USE[role]:
        reasons.append(f"{role} data may not be used for {purpose} (allowed: {list(ALLOWED_USE[role]) or 'nothing'})")
    if role == TEST and purpose == FINAL_EVALUATION:
        if freeze is None:
            reasons.append("TEST needs a freeze record before final evaluation")
        elif freeze_problems is None:
            reasons.append("TEST freeze record not verified")
        elif freeze_problems:
            reasons.append("TEST freeze record does not verify: " + "; ".join(freeze_problems))
    return reasons


def result_label(roles: Iterable[str]) -> str:
    """The only label a result over instances of these roles may carry.

    One role only -> that role; mixed or UNKNOWN -> UNLABELLED (never TEST/VALIDATION)."""
    roles = set(roles)
    if len(roles) == 1 and UNKNOWN not in roles:
        return next(iter(roles))
    return "UNLABELLED"


def check_result_claim(roles: Iterable[str], claimed: str) -> List[str]:
    """Refuse calling a result ``claimed`` (e.g. TEST) unless every instance has that role."""
    label = result_label(list(roles))
    return [] if label == claimed else [f"result over roles {sorted(set(roles))} cannot be labelled {claimed} "
                                         f"(allowed label: {label})"]


# -- freeze ---------------------------------------------------------------------

def _file_sha(root: Path, rel: Optional[str]) -> Optional[str]:
    if not rel:
        return None
    path = Path(root) / rel
    return _sha256(path.read_bytes()) if path.is_file() else None


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def freeze_record(registry: Registry, registry_path, split: str, *, commit: str, frozen_at: str,
                  root: Path) -> Dict:
    """Freeze record for every registry record of ``split`` (VAL or TEST). Raises if unusable."""
    if split not in HELD_OUT:
        raise ValueError(f"only held-out splits are frozen ({HELD_OUT}), not {split!r}")
    if not commit:
        raise ValueError("a freeze needs the commit it was made at")
    members = sorted((r for r in registry.records if r["split"] == split), key=lambda r: r["instance_id"])
    if not members:
        raise ValueError(f"split {split} has no registered instance to freeze")
    instances = []
    for r in members:
        scen, mp = _file_sha(root, r["source_path"]), _file_sha(root, r.get("map_file_path"))
        if scen is None or scen != r["scenario_json_sha256"]:
            raise ValueError(f"{r['instance_id']}: scenario file missing or differs from the registry")
        if r.get("map_file_path") and (mp is None or mp != r.get("map_file_sha256")):
            raise ValueError(f"{r['instance_id']}: map file missing or differs from the registry")
        instances.append({"instance_id": r["instance_id"], "source_path": r["source_path"],
                          "scenario_json_sha256": scen, "map_file_path": r.get("map_file_path"),
                          "map_file_sha256": mp, "instance_fp": instance_fp(r)})
    return {
        "freeze_schema": FREEZE_SCHEMA, "split": split, "data_role": SPLIT_TO_ROLE[split],
        "frozen_at": frozen_at, "commit": commit,
        "registry_path": str(registry_path), "registry_sha256": _sha256(Path(registry_path).read_bytes()),
        "instance_count": len(instances), "instances": instances,
        "set_sha256": _sha256(_canonical(instances)),
    }


def verify_freeze(freeze: Dict, root: Path, registry: Registry) -> List[str]:
    """Problems with a freeze record now ([] = the frozen set is unchanged)."""
    problems = []
    if freeze.get("freeze_schema") != FREEZE_SCHEMA:
        problems.append(f"freeze_schema {freeze.get('freeze_schema')!r} != {FREEZE_SCHEMA}")
    instances = freeze.get("instances") or []
    if _sha256(_canonical(instances)) != freeze.get("set_sha256"):
        problems.append("set_sha256 does not match the listed instances (record edited)")
    split = freeze.get("split")
    by_id = registry.by_id()
    listed = {i["instance_id"] for i in instances}
    current = {r["instance_id"] for r in registry.records if r["split"] == split}
    for iid in sorted(current - listed):
        problems.append(f"{iid}: registered in {split} after the freeze")
    for inst in instances:
        iid = inst["instance_id"]
        rec = by_id.get(iid)
        if rec is None:
            problems.append(f"{iid}: no longer in the registry")
            continue
        if rec["split"] != split:
            problems.append(f"{iid}: split changed to {rec['split']}")
        if instance_fp(rec) != inst["instance_fp"]:
            problems.append(f"{iid}: registry fingerprint changed")
        if _file_sha(root, inst["source_path"]) != inst["scenario_json_sha256"]:
            problems.append(f"{iid}: scenario file changed or missing")
        if inst.get("map_file_path") and _file_sha(root, inst["map_file_path"]) != inst["map_file_sha256"]:
            problems.append(f"{iid}: map file changed or missing")
    return problems


def held_out_shortfall(role_counts: Dict[str, int], targets: Dict[str, int]) -> Dict[str, Dict]:
    """{role: {have, target, missing}}; targets are PROPOSALS (docs section 6.1), not agreed."""
    return {role: {"have": role_counts.get(role, 0), "target": t,
                   "missing": max(0, t - role_counts.get(role, 0))} for role, t in targets.items()}
