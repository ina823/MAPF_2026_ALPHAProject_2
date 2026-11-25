"""DEV-only Selective Hybrid comparison adapter."""

import hashlib
from pathlib import Path

from src.hybrid.selective_planner import (
    SelectiveConfig,
    SelectivePathValidator,
    SelectivePlanner,
)
from src.hybrid.selective_shadow import ShadowMonitor
from src.hybrid.subset_cbs_adapter import (
    BoundedGlobalCbsPlanner,
    SubsetCbsPlanner,
)


# Selective-only DEV settings owned by this bridge (not agreed project policy).
# common_runner keeps only the method-agnostic settings (h_exec, ladder, budget)
# and merges these into the episode's recorded dev_settings after a run.
PARTIAL_TRIGGER_K = 2
LEVEL_TIMEOUT_MS = {"L0": 10000.0, "L1": 10000.0, "G": 20000.0}
RESERVE_MS = 500.0
LADDER = ("L0", "L1", "G")   # the ladder build_selective_dev_components applies

# settings a Selective run needs; none of them may be defaulted
REQUIRED_SETTINGS = ("h_exec", "partial_trigger_k", "ladder", "total_budget_ms",
                     "max_attempts", "level_timeout_ms", "reserve_ms")


def selective_extra_settings():
    """Bridge-owned settings to record alongside common_runner's DEV_SETTINGS."""
    return {
        "partial_trigger_k": PARTIAL_TRIGGER_K,
        "level_timeout_ms": dict(LEVEL_TIMEOUT_MS),
        "reserve_ms": RESERVE_MS,
    }


def selective_planned_settings(*, h_exec, total_budget_ms, max_attempts):
    """What ``run_selective_compare_episode`` will apply, before any solver runs.

    Builds the same config objects the run builds (GlobalHybridConfig via
    ``dev_rearm_config``, TriggerBudget, SelectiveConfig, PartialTriggerConfig),
    so their own value checks run, and returns them in the shape of
    ``selective_applied_settings``. ``overhead_reserve_ms`` is checked by the
    CBS planners themselves at run time (they need a solver root).
    """
    from dataclasses import asdict

    from scripts.run_hybrid_resume_dev import dev_rearm_config
    from src.hybrid.partial_trigger import PartialTriggerConfig
    from src.hybrid.selective_safety import TriggerBudget

    hybrid = dev_rearm_config(h_exec)
    budget = TriggerBudget(total_time_ms=total_budget_ms, max_attempts=max_attempts,
                           level_timeout_ms=dict(LEVEL_TIMEOUT_MS))
    config = SelectiveConfig(ladder=LADDER, budget=budget, acknowledge_unverified_outside_policy=True)
    detector = PartialTriggerConfig(PARTIAL_TRIGGER_K, PARTIAL_TRIGGER_K)
    return {
        "h_exec": hybrid.h_exec,
        "ladder": list(config.ladder),
        "total_budget_ms": budget.total_time_ms,
        "max_attempts": budget.max_attempts,
        "level_timeout_ms": dict(budget.level_timeout_ms),
        "reserve_ms": {"subset": RESERVE_MS, "global": RESERVE_MS},
        "acknowledge_unverified_outside_policy": config.acknowledge_unverified_outside_policy,
        "partial_trigger_detectors": {f"K{PARTIAL_TRIGGER_K}": asdict(detector)},
        "rearm": asdict(hybrid.rearm),
    }


def validate_selective_dev_settings(settings):
    """Refuse recorded Selective settings that the bridge would not apply.

    Missing (or None) required keys are an error, never defaulted. The
    comparison itself is ``check_selective_settings`` (also used after the run).
    """
    missing = [k for k in REQUIRED_SETTINGS if settings.get(k) is None]
    if missing:
        raise ValueError(f"Selective DEV settings missing {missing}; no default is filled in")
    planned = selective_planned_settings(h_exec=settings["h_exec"],
                                         total_budget_ms=settings["total_budget_ms"],
                                         max_attempts=settings["max_attempts"])
    reasons = check_selective_settings(settings, planned)
    if reasons:
        raise ValueError("recorded Selective DEV settings differ from what the bridge applies: "
                         + "; ".join(reasons))
    return planned


def run_selective_compare_episode(
    *,
    grid,
    starts,
    goals,
    solver_root,
    work_dir,
    checkpoint,
    max_steps,
    h_exec,
    total_budget_ms,
    max_attempts,
    monitor,
    logger=None,
):
    """Common Runner entry point: build the DEV budget and run Selective Hybrid."""
    from src.hybrid.selective_safety import TriggerBudget

    budget = TriggerBudget(
        total_time_ms=total_budget_ms,
        max_attempts=max_attempts,
        level_timeout_ms=dict(LEVEL_TIMEOUT_MS),
    )
    return run_selective_dev_episode(
        grid=grid,
        starts=starts,
        goals=goals,
        solver_root=solver_root,
        work_dir=work_dir,
        checkpoint=checkpoint,
        max_steps=max_steps,
        h_exec=h_exec,
        budget=budget,
        reserve_ms=RESERVE_MS,
        monitor=monitor,
        partial_k=PARTIAL_TRIGGER_K,
        logger=logger,
    )


def build_selective_dev_components(
    *,
    solver_root,
    work_dir,
    monitor,
    budget,
    reserve_ms,
):
    """Build the existing Selective components for a DEV comparison."""

    work_dir = Path(work_dir)

    subset = SubsetCbsPlanner(
        solver_root=solver_root,
        work_root=work_dir / "subset_cbs",
        overhead_reserve_ms=reserve_ms,
    )

    global_cbs = BoundedGlobalCbsPlanner(
        solver_root=solver_root,
        work_root=work_dir / "global_cbs",
        overhead_reserve_ms=reserve_ms,
    )

    shadow = ShadowMonitor(monitor)

    config = SelectiveConfig(
        ladder=LADDER,
        budget=budget,
        acknowledge_unverified_outside_policy=True,
    )

    planner = SelectivePlanner(
        global_cbs,
        shadow,
        config,
        subset_planner=subset,
        allow_real_subset_cbs=True,
    )

    return shadow, planner, SelectivePathValidator(planner)

def run_selective_dev_episode(
    *,
    grid,
    starts,
    goals,
    solver_root,
    work_dir,
    checkpoint,
    max_steps,
    h_exec,
    budget,
    reserve_ms,
    monitor,
    partial_k=None,
    logger=None,
):
    """Run Selective Hybrid on a preloaded scenario (DEV only)."""

    from scripts.run_hybrid_resume_dev import dev_rearm_config
    from src.hybrid.global_hybrid import run_global_hybrid_episode
    from src.il.policy import NavHintPolicy

    from src.hybrid.partial_trigger import (
        PartialTriggerConfig,
        PartialTriggerDetector,
        PartialTriggerShadow,
        PartialTriggerSource,
    )

    detector_name = None

    if partial_k is not None:
        detector_name = f"K{partial_k}"

        detector = PartialTriggerDetector(
            PartialTriggerConfig(partial_k, partial_k)
        )

        monitor = PartialTriggerShadow(
            monitor,
            {detector_name: detector},
        )

    shadow, planner, validator = build_selective_dev_components(
        solver_root=solver_root,
        work_dir=work_dir,
        monitor=monitor,
        budget=budget,
        reserve_ms=reserve_ms,
    )

    source = None

    if detector_name is not None:
        source = PartialTriggerSource(
            monitor=shadow,
            planner=planner,
            detector_name=detector_name,
            allow_partial_trigger=True,
        )

    config = dev_rearm_config(h_exec)
    raw = run_global_hybrid_episode(
        NavHintPolicy(checkpoint),
        grid,
        starts,
        goals,
        max_steps,
        planner=planner,
        path_validator=validator,
        config=config,
        monitor=shadow,
        logger=logger,
        extra_trigger=source,
    )
    # the Runner's result has no h_exec; keep the value of the config object
    # it actually ran with (read by selective_applied_settings)
    raw["h_exec_applied"] = config.h_exec

    return raw, shadow, planner


def selective_applied_settings(planner, raw):
    """Settings read back from the objects a Selective run actually used.

    Sources: ``planner.config`` (ladder, TriggerBudget, outside-policy flag),
    the subset / global CBS planners (``overhead_reserve_ms``), the
    ShadowMonitor chain (PartialTriggerDetector configs, DeadlockMonitor
    config) and the Runner result (``rearm_config``, ``h_exec_applied``).
    A value that cannot be read is listed in ``unobserved`` and never filled
    in from a constant.
    """
    from dataclasses import asdict

    from src.hybrid.partial_trigger import PartialTriggerShadow

    applied, unobserved = {}, []

    def put(key, read):
        try:
            value = read()
        except (AttributeError, KeyError, TypeError):
            value = None
        if value is None:
            unobserved.append(key)
        else:
            applied[key] = value

    config = getattr(planner, "config", None)
    budget = getattr(config, "budget", None)
    shadow_inner = getattr(getattr(planner, "monitor", None), "inner", None)
    partial = shadow_inner if isinstance(shadow_inner, PartialTriggerShadow) else None
    base_monitor = partial.inner if partial is not None else shadow_inner

    put("h_exec", lambda: raw["h_exec_applied"])
    put("ladder", lambda: list(config.ladder))
    put("total_budget_ms", lambda: budget.total_time_ms)
    put("max_attempts", lambda: budget.max_attempts)
    put("level_timeout_ms", lambda: dict(budget.level_timeout_ms))
    put("reserve_ms", lambda: {"subset": planner.subset_planner.overhead_reserve_ms,
                               "global": planner.global_planner.overhead_reserve_ms})
    put("acknowledge_unverified_outside_policy",
        lambda: config.acknowledge_unverified_outside_policy)
    put("partial_trigger_detectors",
        lambda: {name: asdict(d.config) for name, d in partial.detectors.items()})
    put("deadlock_monitor", lambda: {"class": type(base_monitor).__name__,
                                     "config": asdict(base_monitor.config)})
    put("rearm", lambda: dict(raw["rearm_config"]))

    applied["unobserved"] = unobserved
    return applied


def check_selective_settings(recorded, applied):
    """Reasons the recorded DEV settings differ from the applied ones ([] = equal).

    Only keys that were observed are compared; an unobserved key is reported
    by ``selective_applied_settings`` and is not a mismatch here.
    """
    reasons = []

    def differ(name, rec, app):
        if rec != app:
            reasons.append(f"{name}: recorded {rec!r} != applied {app!r}")

    for key in ("h_exec", "total_budget_ms", "max_attempts", "level_timeout_ms"):
        if key in applied:
            differ(key, recorded.get(key), applied[key])
    if "ladder" in applied:
        differ("ladder", list(recorded.get("ladder", [])), applied["ladder"])
    if "reserve_ms" in applied:
        for side, value in applied["reserve_ms"].items():
            differ(f"reserve_ms[{side}]", recorded.get("reserve_ms"), value)
    if "partial_trigger_detectors" in applied:
        k = recorded.get("partial_trigger_k")
        expected = {} if k is None else {f"K{k}": {"k_conflict": k, "k_no_progress": k}}
        differ("partial_trigger_k", expected, applied["partial_trigger_detectors"])
    if "acknowledge_unverified_outside_policy" in applied:
        differ("outside_wait_policy", recorded.get("outside_wait_policy") == "DEV_UNVERIFIED",
               applied["acknowledge_unverified_outside_policy"])
    return reasons


def _check_against_planned(applied, planned):
    """Reasons an applied value differs from ``selective_planned_settings``.

    Covers the keys the recorded DEV settings do not name (rearm limits /
    cooldown, the detector config, both reserve sides). Unobserved keys are
    skipped (they are listed in ``applied["unobserved"]``).
    """
    return [f"{key}: planned {planned[key]!r} != applied {applied[key]!r}"
            for key in planned if key in applied and applied[key] != planned[key]]


def selective_manifest_settings(recorded, applied):
    """One run's Selective settings in the batch-manifest shape.

    Every value comes from ``applied`` (read back from the objects the run
    used); ``partial_trigger_k`` and ``outside_wait_policy`` are derived from
    the applied detector config / outside-policy flag. A value that cannot be
    read or derived is None and named in ``unobserved`` -- never defaulted.
    """
    unobserved = list(applied.get("unobserved", []))

    def take(key):
        return applied.get(key)

    detectors = take("partial_trigger_detectors")
    k = None
    if isinstance(detectors, dict) and len(detectors) == 1:
        (cfg,) = detectors.values()
        if cfg.get("k_conflict") == cfg.get("k_no_progress"):
            k = cfg.get("k_conflict")
    ack = take("acknowledge_unverified_outside_policy")
    policy = recorded.get("outside_wait_policy") if ack is True else None
    rearm = take("rearm")

    view = {
        "h_exec": take("h_exec"),
        "partial_trigger_k": k,
        "ladder": take("ladder"),
        "total_budget_ms": take("total_budget_ms"),
        "max_attempts": take("max_attempts"),
        "level_timeout_ms": take("level_timeout_ms"),
        "reserve_ms": take("reserve_ms"),
        "detectors": {"partial_trigger": detectors, "deadlock_monitor": take("deadlock_monitor")},
        "rearm_limits": None if rearm is None else {
            "max_cbs_calls": rearm.get("max_cbs_calls"),
            "cbs_time_budget_ms": rearm.get("cbs_time_budget_ms"),
        },
        "cooldown_steps": None if rearm is None else rearm.get("cooldown_steps"),
        "outside_wait_policy": policy,
        "acknowledge_unverified_outside_policy": ack,
    }
    for key, value in view.items():
        if value is None and key not in unobserved:
            unobserved.append(key)
    view["unobserved"] = unobserved
    return view


SELECTIVE_TIMEOUT_NOTE = (
    "conditions.cbs_timeout_sec is the per-call solver limit of the default CBS methods; "
    "SELECTIVE_HYBRID_DEV passes it to no solver. Its limits are level_timeout_ms (per ladder "
    "level), total_budget_ms (all attempts of one Selective request) and reserve_ms (overhead "
    "subtracted before the solver limit); each call's own limit is timeout_ms / timeout_sec in the "
    "run's planner_calls. The cbs_timeout_sec column of compare_episodes.csv carries the batch "
    "condition for this method, not an applied limit."
)


def selective_manifest_entry(recorded, records, *, batch_cbs_timeout_sec):
    """batch_manifest.json ``methods[SELECTIVE_HYBRID_DEV]`` additions.

    ``recorded`` is the method's validated DEV settings and ``records`` its
    EpisodeRecords. Every run's applied settings are checked against
    ``recorded`` and against ``selective_planned_settings``; the result is
    one ``applied_settings`` view when all runs agree (otherwise None and the
    per-run views), plus the list of mismatches (``[]`` = all consistent).
    """
    planned = selective_planned_settings(h_exec=recorded["h_exec"],
                                         total_budget_ms=recorded["total_budget_ms"],
                                         max_attempts=recorded["max_attempts"])
    mismatches, views = [], {}
    for rec in records:
        applied = rec.dev_settings.get("applied")
        if applied is None:
            mismatches.append(f"{rec.run_id}: no applied Selective settings recorded")
            continue
        mismatches += [f"{rec.run_id}: {r}" for r in check_selective_settings(recorded, applied)]
        mismatches += [f"{rec.run_id}: {r}" for r in _check_against_planned(applied, planned)]
        views[rec.run_id] = selective_manifest_settings(recorded, applied)
    first = next(iter(views.values()), None)
    identical = first is not None and all(v == first for v in views.values())
    if views and not identical:
        mismatches.append("applied Selective settings differ between runs")
    entry = {
        "applied_settings": first if identical else None,
        "applied_settings_runs": list(views),
        "applied_settings_mismatches": mismatches,
        "timeouts": {
            "batch_cbs_timeout_sec": batch_cbs_timeout_sec,
            "batch_cbs_timeout_sec_applied": False,
            "note": SELECTIVE_TIMEOUT_NOTE,
        },
    }
    if views and not identical:
        entry["applied_settings_by_run"] = views
    return entry


# the file atb033 cbs.py is given (src/cbs_adapter.run_atb033_cbs writes it next to
# input.yaml and passes it as argv[1]); relative to each adapter call's work_dir
SOLVER_INPUT_RELPATH = ("outputs", "paths", "replan", "solver_input.yaml")


def solver_input_record(call):
    """Path and sha256 (raw bytes) of the solver input one adapter call used.

    ``solver_input_status``: RECORDED; SOLVER_NOT_CALLED (refused before the
    CBS chain -- no file is expected); NO_WORK_DIR / FILE_NOT_FOUND (the chain
    was entered but the file cannot be read). No hash is ever made up.
    """
    if call.get("called") is not True:
        return {"solver_input_path": None, "solver_input_sha256": None,
                "solver_input_status": "SOLVER_NOT_CALLED"}
    if not call.get("work_dir"):
        return {"solver_input_path": None, "solver_input_sha256": None, "solver_input_status": "NO_WORK_DIR"}
    path = Path(call["work_dir"]).joinpath(*SOLVER_INPUT_RELPATH)
    if not path.is_file():
        return {"solver_input_path": str(path), "solver_input_sha256": None,
                "solver_input_status": "FILE_NOT_FOUND"}
    return {"solver_input_path": str(path), "solver_input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "solver_input_status": "RECORDED"}


def selective_cbs_calls(planner):
    """Convert real Selective CBS calls (subset + global) in attempt order.

    Every ladder attempt must map to exactly one recorded solver call, and no
    recorded call may be left over; otherwise the run cannot be audited.

    ``request_index`` is the planner's own 1-based Selective request number
    (``SelectiveCallRecord.call_index``) and ``attempt_index`` the 1-based
    position of the attempt within that request, so every success / timeout /
    failure keeps its place. ``pythonhashseed_env`` is copied from the solver
    call record only when that record has it (same rule as the default
    methods' ``_planner_calls``); a missing value is never filled in.
    """
    subset = iter(planner.subset_planner.calls)
    global_calls = iter(planner.global_planner.calls)
    result = []

    for decision in planner.records:
        outcome = getattr(decision, "outcome", None)
        n_attempts = len(decision.attempts)
        for attempt_index, attempt in enumerate(decision.attempts, start=1):
            is_global = attempt.level == "G"
            source = planner.global_planner if is_global else planner.subset_planner
            calls = global_calls if is_global else subset

            call = next(calls, None)
            if call is None:
                raise RuntimeError("Selective CBS call record missing")

            paths = call.get("paths")
            result.append({
                "request_index": decision.call_index,
                "attempt_index": attempt_index,
                "solver_root": str(source.solver_root),
                "work_dir": str(call["work_dir"]) if call.get("work_dir") else None,
                "timeout_sec": call.get("solver_timeout_sec"),
                "status": call.get("status"),
                "ladder_status": attempt.status,
                "level": attempt.level,
                "group": list(attempt.group),
                "called": call.get("called"),
                # the planner's own view of the attempt (None when it kept none)
                "solver_invoked": getattr(attempt, "solver_invoked", None),
                # request-level facts, repeated on each attempt so the CSV builder
                # can tell a dropped / extra attempt from a short ladder
                "request_trigger_t": getattr(decision, "trigger_t", None),
                "request_outcome": outcome,
                "request_returned_status": getattr(decision, "returned_status", None),
                "request_attempt_count": n_attempts,
                # the plan the planner returned to the Runner: only the last
                # attempt of an EXECUTE:<level> request (accepted or not by the Runner)
                "returned_plan": (attempt_index == n_attempts and outcome == f"EXECUTE:{attempt.level}"),
                "timeout_ms": call.get("timeout_ms"),
                "latency_ms": attempt.latency_ms,
                **({"pythonhashseed_env": call["pythonhashseed_env"]}
                   if "pythonhashseed_env" in call else {}),
                **solver_input_record(call),
                "paths": (
                    {aid: [tuple(p) for p in path]
                     for aid, path in paths.items()}
                    if paths is not None else None
                ),
            })

    if next(subset, None) is not None or next(global_calls, None) is not None:
        raise RuntimeError("Unmatched Selective CBS calls")

    return result
