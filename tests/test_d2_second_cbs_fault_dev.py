import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_partial_trigger_p3_dev as dev
from src.hybrid import subset_cbs_adapter as adapter


class TestD2SecondCbsFailure(unittest.TestCase):

    def test_second_subset_timeout(self):
        solver = Path.home() / "mapf_work/cbs_solver_atb033/centralized/cbs"
        self.assertTrue((solver / "cbs.py").is_file())

        original = adapter.run_cbs_runtime
        count = {"subset": 0}

        def inject(*args, **kwargs):
            if "subset_cbs" in Path.cwd().parts:
                count["subset"] += 1

                if count["subset"] == 2:
                    return adapter.CBSRuntimeResult(
                        status="timeout",
                        paths=None,
                        latency_sec=0.0,
                        error="DEV_INJECTED_SECOND_SUBSET_TIMEOUT",
                    )

            return original(*args, **kwargs)

        with tempfile.TemporaryDirectory() as temp:
            with patch.object(adapter, "run_cbs_runtime", side_effect=inject):
                rec = dev.run("D2", solver, Path(temp))

        subset_calls = [
            x for x in rec["adapter_calls"]
            if x["planner"] == "subset"
        ]

        print("\nSubset runtime calls:", count["subset"])
        print("Subset adapter calls:", subset_calls)
        print("Termination:", rec["termination"])
        print("Stop reason:", rec["stop_reason"])
        print("Safety scan:", rec["safety_scan"])

        print("Transitions:", [
            (x.get("t"), x.get("to"), x.get("reason"))
            for x in rec.get("transitions", [])
        ])

        print("Selective outcomes:", [
            (
                x.get("trigger_t"),
                x.get("outcome"),
                [
                    (a.get("level"), a.get("raw_status"), a.get("status"))
                    for a in x.get("attempts", [])
                ],
            )
            for x in rec.get("selective_records", [])
        ])

        print("CBS step indices:", rec.get("cbs_step_indices"))


        self.assertGreaterEqual(count["subset"], 2)
        self.assertGreaterEqual(len(subset_calls), 2)
        self.assertEqual(subset_calls[0]["status"], "success")
        self.assertEqual(subset_calls[1]["status"], "timeout")

        self.assertFalse(rec["safety_scan"]["vertex_overlap_steps"])
        self.assertFalse(rec["safety_scan"]["edge_swaps"])

        events = rec["selective_records"]

        self.assertEqual(events[0]["trigger_t"], 5)
        self.assertEqual(events[0]["outcome"], "EXECUTE:L0")

        self.assertEqual(events[1]["trigger_t"], 10)
        self.assertEqual(events[1]["outcome"], "EXECUTE:G")

        attempts = events[1]["attempts"]
        self.assertEqual(attempts[0]["level"], "L0")
        self.assertEqual(attempts[0]["status"], "timeout")
        self.assertEqual(attempts[1]["level"], "G")
        self.assertEqual(attempts[1]["status"], "success")

        self.assertEqual(rec["termination"], "SUCCESS")

if __name__ == "__main__":
    unittest.main()
