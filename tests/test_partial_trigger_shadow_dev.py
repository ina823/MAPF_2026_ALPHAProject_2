"""scripts/run_partial_trigger_shadow_dev.py: refusal without a solver and phase labelling
(no solver, no IL). The real shadow evidence is the script's own run output."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_partial_trigger_shadow_dev as dev


class TestShadowDevScript(unittest.TestCase):
    def test_refuses_without_solver_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.dict(os.environ, {"PYTHONHASHSEED": "0"}):
                self.assertEqual(dev.main(["--solver-root", str(Path(d) / "none"), "--out-root", d]), 2)
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_phase_labels(self):
        res = {"action_sources": ["IL", "IL", "CBS", "CBS", "IL"],
               "transitions": [{"t": 2, "to": "TRIGGER"}, {"t": 4, "to": "IL_RESUME"}, {"t": 4, "to": "IL"}]}
        self.assertEqual([dev.phase_at(res, t) for t in (1, 2, 3, 4, 5)],
                         ["IL", "IL", "CBS_WINDOW", "CBS_WINDOW", "IL_AFTER_RESUME"])

    def test_scenarios_are_the_existing_dev_cases(self):
        self.assertEqual(dev.SCENARIOS, ("P3Q", "P3F", "S1B", "D1", "D2"))
        for name in dev.SCENARIOS:
            grid, starts, goals, _ = dev.scenario(name)
            self.assertEqual(set(starts), set(goals))
        self.assertEqual(dev.KS, (2, 3, 5))


if __name__ == "__main__":
    unittest.main()
