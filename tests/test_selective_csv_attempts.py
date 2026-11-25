"""SELECTIVE_HYBRID_DEV: Runner CBS requests <-> ladder attempts in compare CSV (DEV, step 5).

Part 1 (no solver, no torch): ``selective_returned_attempts`` on fake planner
records shaped like ``selective_cbs_calls`` -- L0/L1/G successes, timeouts,
failures, attempts that never reached the solver, and broken records.

Part 2 (MOCK SOLVER, needs torch): one real Corridor Selective run through
``run_method`` (only ``subprocess.run`` of the CBS chain answers with the
recorded atb033 plan, as in tests/test_selective_real_e2e_dev), then its
attempt records are rewritten in memory and passed to ``build_rows``.
This is not evidence about the real solver.
"""

import copy
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.compare.common_runner import RunSpec, SELECTIVE_HYBRID_DEV, run_method
from src.compare.compare_csv import CompareCsvError, build_rows, selective_returned_attempts

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

PATHS = {0: [(1, 1), (1, 2)], 2: [(3, 3), (3, 4)]}
ALL = {**PATHS, 1: [(5, 5), (5, 6)], 3: [(7, 7), (7, 7)]}
GROUP = {"L0": [0, 2], "L1": [0, 1, 2], "G": [0, 1, 2, 3]}


def attempt(level, status, *, called=True, ladder=None, paths="auto", latency=10.0):
    """One ladder attempt; ``paths`` defaults to a plan only for a success."""
    if paths == "auto":
        paths = ({a: p for a, p in ALL.items() if a in GROUP[level]} if status == "success" else None)
    return {"level": level, "group": list(GROUP[level]), "status": status,
            "ladder_status": status if ladder is None else ladder, "called": called,
            "solver_invoked": called, "latency_ms": latency, "paths": paths,
            "work_dir": "w" if called else None, "timeout_ms": 10000.0, "solver_root": "/s"}


def request(index, t, *attempts):
    """(Runner call, planner attempts) of one consistent Selective request."""
    last = attempts[-1]
    executes = last["ladder_status"] == "success"
    outcome = f"EXECUTE:{last['level']}" if executes else "SAFE_STOP:LADDER_EXHAUSTED"
    returned_status = "success" if executes else last["ladder_status"]
    rows = []
    for k, a in enumerate(attempts, start=1):
        rows.append({**a, "request_index": index, "attempt_index": k, "request_attempt_count": len(attempts),
                     "request_trigger_t": t, "request_outcome": outcome,
                     "request_returned_status": returned_status,
                     "returned_plan": executes and k == len(attempts)})
    call = {"index": index, "trigger_timestep": t, "status": returned_status, "accepted": executes,
            "latency_ms": sum(a["latency_ms"] for a in attempts), "reason": "", "error": None}
    return call, rows


def flatten(*requests):
    return [c for c, _ in requests], [a for _, rows in requests for a in rows]


class TestAttemptMatching(unittest.TestCase):

    def ok(self, *requests):
        calls, atts = flatten(*requests)
        return selective_returned_attempts("run", calls, atts), atts

    def refused(self, calls, atts, text):
        with self.assertRaises(CompareCsvError) as cm:
            selective_returned_attempts("run", calls, atts)
        self.assertIn(text, str(cm.exception))

    # -- legitimate ladders --------------------------------------------------
    def test_l0_success(self):
        got, _ = self.ok(request(1, 5, attempt("L0", "success")))
        self.assertEqual((got[1]["level"], got[1]["paths"]), ("L0", PATHS))

    def test_l0_timeout_then_g_success(self):
        got, atts = self.ok(request(1, 5, attempt("L0", "timeout"), attempt("G", "success")))
        self.assertEqual(got[1]["level"], "G")
        self.assertEqual([(a["level"], a["status"], a["returned_plan"]) for a in atts],
                         [("L0", "timeout", False), ("G", "success", True)])   # original status kept

    def test_l0_failure_then_l1_success(self):
        for failure in ("no_solution", "solver_error", "invalid_path"):
            with self.subTest(failure=failure):
                got, atts = self.ok(request(1, 5, attempt("L0", failure), attempt("L1", "success")))
                self.assertEqual(got[1]["level"], "L1")
                self.assertEqual(atts[0]["status"], failure)
                self.assertIsNone(atts[0]["paths"])

    def test_attempt_without_solver_call_kept_apart_from_a_failed_call(self):
        got, atts = self.ok(request(1, 5, attempt("L0", "input_error", called=False),
                                    attempt("L1", "timeout"), attempt("G", "success")))
        self.assertEqual(got[1]["level"], "G")
        self.assertEqual([(a["called"], a["status"]) for a in atts],
                         [(False, "input_error"), (True, "timeout"), (True, "success")])

    def test_late_success_counted_as_timeout_is_not_returned(self):
        late = attempt("L0", "success", ladder="timeout")   # solver answered after the deadline
        got, _ = self.ok(request(1, 5, late, attempt("G", "success")))
        self.assertEqual(got[1]["level"], "G")

    def test_every_attempt_failed_returns_no_plan(self):
        got, atts = self.ok(request(1, 5, attempt("L0", "timeout"), attempt("G", "no_solution")))
        self.assertIsNone(got[1])
        self.assertEqual([a["status"] for a in atts], ["timeout", "no_solution"])

    def test_returned_plan_refused_by_the_runner_is_allowed_but_not_accepted(self):
        call, rows = request(1, 5, attempt("L0", "success"))
        call["accepted"] = False            # e.g. the path validator refused it
        got = selective_returned_attempts("run", [call], rows)
        self.assertEqual(got[1]["level"], "L0")

    def test_attempt_count_differing_from_request_count_is_not_an_error(self):
        # D2 shape: request 1 L0 success; request 2 L0 timeout -> G success (2 Runner calls, 3 attempts)
        got, atts = self.ok(request(1, 3, attempt("L0", "success")),
                            request(2, 9, attempt("L0", "timeout"), attempt("G", "success")))
        self.assertEqual(len(atts), 3)
        self.assertEqual((got[1]["level"], got[2]["level"]), ("L0", "G"))

    # -- broken records ------------------------------------------------------
    def test_call_order_mismatch_refused(self):
        calls, atts = flatten(request(1, 5, attempt("L0", "timeout"), attempt("G", "success")))
        self.refused(calls, [atts[1], atts[0]], "out of request / attempt order")
        calls, atts = flatten(request(1, 3, attempt("L0", "success")), request(2, 9, attempt("G", "success")))
        self.refused(calls, [atts[1], atts[0]], "out of request / attempt order")

    def test_duplicate_request_index_refused(self):
        calls, atts = flatten(request(1, 3, attempt("L0", "success")), request(2, 9, attempt("G", "success")))
        atts[1] = {**atts[1], "request_index": 1}
        self.refused(calls, atts, "recorded twice")

    def test_missing_cbs_call_record_refused(self):
        calls, atts = flatten(request(1, 3, attempt("L0", "success")), request(2, 9, attempt("G", "success")))
        self.refused(calls, atts[:1], "request 2: CBS call record missing")
        calls, atts = flatten(request(1, 5, attempt("L0", "timeout"), attempt("L1", "timeout"),
                                      attempt("G", "success")))
        self.refused(calls, [atts[0], atts[2]], "planner made 3")    # middle attempt dropped

    def test_attempt_for_unknown_request_refused(self):
        calls, atts = flatten(request(1, 3, attempt("L0", "success")))
        extra = {**atts[0], "request_index": 2}
        self.refused(calls, atts + [extra], "requests the Runner never made")

    def test_no_solver_call_with_a_solver_result_refused(self):
        calls, atts = flatten(request(1, 5, attempt("L0", "success")))
        atts[0]["called"] = atts[0]["solver_invoked"] = False
        self.refused(calls, atts, "no solver call, yet a solver result")

    def test_planner_and_adapter_disagree_on_solver_call_refused(self):
        calls, atts = flatten(request(1, 5, attempt("L0", "timeout"), attempt("G", "success")))
        atts[0]["solver_invoked"] = False
        self.refused(calls, atts, "disagree on whether the solver ran")

    def test_failed_call_with_paths_refused(self):
        calls, atts = flatten(request(1, 5, attempt("L0", "timeout", paths=PATHS), attempt("G", "success")))
        self.refused(calls, atts, "status 'timeout' but paths recorded")

    def test_failed_attempt_marked_as_executed_refused(self):
        calls, atts = flatten(request(1, 5, attempt("L0", "timeout"), attempt("G", "success")))
        atts[0]["returned_plan"] = True
        self.refused(calls, atts, "must return exactly its last attempt")
        calls, atts = flatten(request(1, 5, attempt("L0", "timeout"), attempt("G", "no_solution")))
        atts[1]["returned_plan"] = True
        self.refused(calls, atts, "but an attempt is marked as returned")

    def test_runner_accepting_a_plan_never_returned_refused(self):
        calls, atts = flatten(request(1, 5, attempt("G", "timeout")))
        calls[0]["accepted"] = True
        self.refused(calls, atts, "never returned")

    def test_runner_status_latency_or_time_differing_refused(self):
        for key, value, text in (("status", "timeout", "Runner status"),
                                 ("latency_ms", 999.0, "summed attempts"),
                                 ("trigger_timestep", 6, "planner trigger t=5")):
            with self.subTest(key=key):
                calls, atts = flatten(request(1, 5, attempt("L0", "timeout"), attempt("G", "success")))
                calls[0][key] = value
                self.refused(calls, atts, text)

    def test_incomplete_attempt_record_refused(self):
        calls, atts = flatten(request(1, 5, attempt("L0", "success")))
        del atts[0]["returned_plan"]
        self.refused(calls, atts, "incomplete")


def _fake_cbs_subprocess(command, *args, **kwargs):
    """MOCK SOLVER: only the atb033 cbs.py call answers with the recorded plan."""
    from tests.test_selective_real_e2e_dev import _recorded_subprocess
    if isinstance(command, (list, tuple)) and len(command) > 3 and str(command[1]).endswith("cbs.py") \
            and "stdout" not in kwargs:
        return _recorded_subprocess(command, *args, **kwargs)
    return _REAL_RUN(command, *args, **kwargs)


_REAL_RUN = subprocess.run


@unittest.skipIf(torch is None, "torch is not installed")
class TestBuildRowsWithRetries(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        root = tmp / "solver"
        root.mkdir()
        (root / "cbs.py").write_text("# stand-in, never executed\n", encoding="utf-8")
        from tests.test_common_runner import CORRIDOR
        with patch("src.compare.common_runner._git", return_value={"commit": "c0ffee", "dirty": False}), \
                patch("src.cbs_adapter.subprocess.run", side_effect=_fake_cbs_subprocess):
            cls.record = run_method(RunSpec(method=SELECTIVE_HYBRID_DEV, scenario=CORRIDOR,
                                            out_dir=tmp / "runs", solver_root=root))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def with_l0_timeout_first(self):
        """The run's request with a failed L0 attempt before its returned G attempt."""
        rec = copy.deepcopy(self.record)
        (g,) = rec.planner_calls
        a0 = sorted(g["paths"])[0]
        l0 = {**g, "level": "L0", "group": [a0], "status": "timeout", "ladder_status": "timeout",
              "paths": None, "returned_plan": False, "attempt_index": 1, "latency_ms": 9500.0,
              "request_attempt_count": 2}
        rec.planner_calls = [l0, {**g, "attempt_index": 2, "request_attempt_count": 2}]
        rec.cbs_calls[0]["latency_ms"] = 9500.0 + g["latency_ms"]
        return rec

    def test_real_record_carries_matching_fields(self):
        (call,), (att,) = self.record.cbs_calls, self.record.planner_calls
        self.assertEqual((att["request_index"], att["attempt_index"], att["request_attempt_count"]), (1, 1, 1))
        self.assertEqual((att["request_outcome"], att["returned_plan"], att["called"], att["solver_invoked"]),
                         ("EXECUTE:G", True, True, True))
        self.assertEqual(att["request_trigger_t"], call["trigger_timestep"])
        _, rows = build_rows(self.record, "b")
        self.assertEqual([(r["status"], r["accepted"], r["h_exec_config"], r["exec_steps"]) for r in rows],
                         [("success", True, 3, 3)])

    def test_l0_timeout_then_g_success_checks_the_g_path(self):
        _, base = build_rows(self.record, "b")
        rec = self.with_l0_timeout_first()
        _, rows = build_rows(rec, "b")
        # one Runner request -> one row, horizon / window from the G attempt; latency = both attempts
        self.assertEqual(rows[0]["solver_latency_ms"], rec.cbs_calls[0]["latency_ms"])
        drop = lambda r: {k: v for k, v in r.items() if k != "solver_latency_ms"}
        self.assertEqual([drop(r) for r in rows], [drop(r) for r in base])
        self.assertIsNotNone(rows[0]["planned_horizon"])

    def test_success_path_differing_from_the_trajectory_refused(self):
        for rec in (copy.deepcopy(self.record), self.with_l0_timeout_first()):
            plan = rec.planner_calls[-1]["paths"]
            a = sorted(plan)[0]
            plan[a] = [plan[a][0]] + [(0, 0)] * (len(plan[a]) - 1)
            with self.assertRaises(CompareCsvError):
                build_rows(rec, "b")

    def test_failed_attempt_never_used_as_the_execution_path(self):
        rec = self.with_l0_timeout_first()
        rec.planner_calls[0]["returned_plan"] = True        # claim the timed-out L0 was executed
        with self.assertRaisesRegex(CompareCsvError, "must return exactly its last attempt"):
            build_rows(rec, "b")

    def test_missing_attempt_record_refused(self):
        rec = copy.deepcopy(self.record)
        rec.planner_calls = []
        with self.assertRaisesRegex(CompareCsvError, "CBS call record missing"):
            build_rows(rec, "b")


@unittest.skipIf(torch is None, "torch is not installed")
class TestDefaultMethodsUnaffected(unittest.TestCase):

    def test_five_methods_never_use_the_selective_matcher(self):
        from tests.test_run_compare_batch import _Case

        class Batch(_Case):
            def runTest(self):
                pass

        case = Batch()
        case.setUp()
        try:
            with patch("src.compare.compare_csv.selective_returned_attempts",
                       side_effect=AssertionError("used for a default method")):
                code, out = case.run_cli()
            self.assertEqual(code, 0, out)
        finally:
            case.tearDown()


if __name__ == "__main__":
    unittest.main()
