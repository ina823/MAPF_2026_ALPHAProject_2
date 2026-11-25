"""Selective Hybrid provenance records (DEV): CBS call conversion for the
Common Runner. No solver is run; planners are stand-ins with ``calls`` lists
shaped like SubsetCbsPlanner / BoundedGlobalCbsPlanner records."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compare import provenance as pv
from src.compare.common_runner import _planner_calls
from src.compare.selective_bridge import selective_cbs_calls


def _call(status, seed="__absent__", *, work_dir="w", paths=None):
    c = {"status": status, "called": status != "input_error", "work_dir": work_dir,
         "solver_timeout_sec": 9.5, "timeout_ms": 10000.0, "paths": paths}
    if seed != "__absent__":
        c["pythonhashseed_env"] = seed
    return c


def _attempt(level, group, status, latency=50.0):
    return SimpleNamespace(level=level, group=tuple(group), status=status, latency_ms=latency)


def _planner(records, subset_calls, global_calls):
    return SimpleNamespace(
        records=records,
        subset_planner=SimpleNamespace(solver_root=Path("/s"), calls=subset_calls),
        global_planner=SimpleNamespace(solver_root=Path("/s"), calls=global_calls),
    )


def _request(call_index, *attempts):
    return SimpleNamespace(call_index=call_index, attempts=list(attempts))


PATH = {0: [(1, 1), (1, 2)], 2: [(3, 3), (3, 4)]}


class TestSelectiveCbsCalls(unittest.TestCase):

    def d2_fault_planner(self, s1="0", s2="0", g1="0"):
        """Request 1: L0 success. Request 2: L0 timeout -> G success (c63233f shape)."""
        return _planner(
            [_request(1, _attempt("L0", (0, 2), "success")),
             _request(2, _attempt("L0", (0, 3), "timeout"), _attempt("G", (0, 1, 2, 3), "success"))],
            [_call("success", s1, paths=PATH), _call("timeout", s2)],
            [_call("success", g1, paths=PATH)],
        )

    def test_seed_copied_from_each_subset_and_global_call(self):
        calls = selective_cbs_calls(self.d2_fault_planner(s1="0", s2=None, g1="7"))
        self.assertEqual([c["pythonhashseed_env"] for c in calls], ["0", None, "7"])

    def test_missing_seed_is_not_filled_in(self):
        calls = selective_cbs_calls(self.d2_fault_planner(s2="__absent__"))
        self.assertEqual(calls[0]["pythonhashseed_env"], "0")
        self.assertNotIn("pythonhashseed_env", calls[1])
        self.assertEqual(calls[2]["pythonhashseed_env"], "0")

    def test_request_and_attempt_order_and_status_kept(self):
        calls = selective_cbs_calls(self.d2_fault_planner())
        self.assertEqual([(c["request_index"], c["attempt_index"], c["level"], c["status"], c["ladder_status"])
                          for c in calls],
                         [(1, 1, "L0", "success", "success"),
                          (2, 1, "L0", "timeout", "timeout"),
                          (2, 2, "G", "success", "success")])
        self.assertIsNone(calls[1]["paths"])
        self.assertEqual(calls[2]["group"], [0, 1, 2, 3])

    def test_failed_call_without_solver_kept(self):
        p = _planner([_request(1, _attempt("L0", (0, 1), "input_error"), _attempt("G", (0, 1, 2), "success"))],
                     [_call("input_error", "0", work_dir=None)], [_call("success", "0", paths=PATH)])
        calls = selective_cbs_calls(p)
        self.assertEqual([(c["request_index"], c["attempt_index"], c["status"], c["called"], c["work_dir"])
                          for c in calls],
                         [(1, 1, "input_error", False, None), (1, 2, "success", True, "w")])

    def test_request_index_is_the_planners_own_number(self):
        p = _planner([_request(1), _request(2, _attempt("L0", (0, 1), "success"))],   # request 1: no attempt
                     [_call("success", "0", paths=PATH)], [])
        self.assertEqual([(c["request_index"], c["attempt_index"]) for c in selective_cbs_calls(p)], [(2, 1)])

    def test_missing_or_leftover_call_still_refused(self):
        p = self.d2_fault_planner()
        p.global_planner.calls = []
        with self.assertRaisesRegex(RuntimeError, "missing"):
            selective_cbs_calls(p)
        p = self.d2_fault_planner()
        p.subset_planner.calls.append(_call("success", "0"))
        with self.assertRaisesRegex(RuntimeError, "Unmatched"):
            selective_cbs_calls(p)


class TestSeedConsistencyOnSelectiveRecords(unittest.TestCase):
    """The unchanged batch check, applied to Selective planner_calls."""

    def record(self, planner):
        return SimpleNamespace(run_id="r", environment={"hash_seed": {"current_env_value": "0"}},
                               planner_calls=selective_cbs_calls(planner))

    def test_all_recorded_and_equal_is_consistent(self):
        rec = self.record(TestSelectiveCbsCalls().d2_fault_planner())
        self.assertEqual(pv.check_seed_consistency({"startup_env_value": "0"}, [rec]), [])

    def test_missing_or_different_seed_is_still_reported(self):
        rec = self.record(TestSelectiveCbsCalls().d2_fault_planner(s2="__absent__", g1="5"))
        reasons = pv.check_seed_consistency({"startup_env_value": "0"}, [rec])
        self.assertEqual(len(reasons), 2, reasons)
        self.assertIn("not recorded for r CBS call 2", reasons[0])
        self.assertIn("CBS call 3 inherited '5'", reasons[1])


class TestDefaultMethodsUnchanged(unittest.TestCase):

    def test_default_planner_call_keys(self):
        planner = SimpleNamespace(calls=[{"solver_root": Path("/s"), "work_dir": Path("/w"), "timeout_sec": 30,
                                          "status": "success", "pythonhashseed_env": "0", "paths": PATH}])
        (call,) = _planner_calls(planner)
        self.assertEqual(set(call), {"solver_root", "work_dir", "timeout_sec", "status",
                                     "pythonhashseed_env", "paths"})
        self.assertNotIn("request_index", call)


if __name__ == "__main__":
    unittest.main()
