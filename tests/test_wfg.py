"""Unit tests for the candidate WFG pure function (src/hybrid/wfg.py).

No CBS, no simulator run, no file writes. Fixtures below are copied from the
saved DEV logs (GLOBAL_HYBRID_DEV_H3_REARM runs in
outputs/logs/compare_rearm_dev_20261105, step CSV row ``timestep == t``):
``current_*`` -> current, ``intended_*`` -> intended, ``blocked`` -> blocked.
Grids are the controlled scenario maps (0 = '.', 1 = '#').

What is FACT vs ASSUMPTION:
- Fact (log): positions, intents, blocked flags, s_t, sha256 of the files.
- Assumption (contract v0, PENDING P5): the edge/group rule itself, i.e. the
  expected ``edges`` and ``groups`` below are the outputs of the proposed rule,
  not values recorded by any run. No Selective run exists.

Test IDs follow the WFG analysis plan: T1-T6, T8-T10. T7 (following into a
vacated cell) is PENDING and has no expected value here.
"""

import csv
import hashlib
import inspect
import itertools
import json
import unittest
from pathlib import Path

import numpy as np

from src.hybrid.wfg import RULE_VERSION, WfgEdge, build_wfg

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs" / "logs" / "compare_rearm_dev_20261105" / "runs"


def _grid(text):
    return [[1 if ch == "#" else 0 for ch in line] for line in text.strip().splitlines()]


CORRIDOR_GRID = _grid("""
###########
#####.#####
...........
###########
###########
""")
INTERSECTION_GRID = _grid("""
###.###
###.###
###.###
.......
###.###
###.###
###.###
""")
BOTTLENECK_GRID = _grid("""
....#....
....#....
.........
....#....
....#....
""")

# Saved log provenance. Logs are git-ignored, so T10 skips if they are absent.
CASES = {
    "corridor_t5": dict(
        scenario="corridor_bay_n2",
        run="20261105_012739_corridor_bay_n2_GLOBAL_HYBRID_DEV_H3_REARM",
        steps_sha256="f0d0badd7dca6cdc92c11d2f64fec49e561d780454184ebcd72ffafc4d33356e",
        result_sha256="71326bbcad7e6626385ec58aab4e9a5a085e13c952b72f1141ddc3f2692be7c8",
        t=5, grid=CORRIDOR_GRID,
        current={0: (2, 4), 1: (2, 6)},
        intended={0: (2, 5), 1: (2, 5)},
        blocked={0, 1},
    ),
    "intersection_t2": dict(
        scenario="intersection_n2",
        run="20261105_012742_intersection_n2_GLOBAL_HYBRID_DEV_H3_REARM",
        steps_sha256="a44c43bdb35edc699b703ffab7a35f463a57c3c38f7fcc59dba518a5fa302f7a",
        result_sha256="95cf660ec1b25024f6a74694d2d8868ec68c3f975623792666d9be402762cb27",
        t=2, grid=INTERSECTION_GRID,
        current={0: (3, 2), 1: (2, 3)},
        intended={0: (3, 3), 1: (3, 3)},
        blocked={0, 1},
    ),
    "bottleneck_t2": dict(
        scenario="bottleneck_n4",
        run="20261105_012754_bottleneck_n4_GLOBAL_HYBRID_DEV_H3_REARM",
        steps_sha256="e8b6b89098db1b1b963c9fd091cacde8731eff58fdcc349a7311a3392e3924e5",
        result_sha256="1e6c5c6c9ef1ac1685a092dde546d5bf92500db380d54512bf4a7e6ab71de5bc",
        t=2, grid=BOTTLENECK_GRID,
        current={0: (1, 2), 1: (2, 1), 2: (1, 7), 3: (3, 7)},
        intended={0: (2, 2), 1: (2, 2), 2: (2, 7), 3: (2, 7)},
        blocked={0, 1, 2, 3},
        monitor_blocking_agents=[0, 1, 2, 3],  # logged; NOT the WFG answer
    ),
    "bottleneck_t7": dict(
        scenario="bottleneck_n4",
        run="20261105_012754_bottleneck_n4_GLOBAL_HYBRID_DEV_H3_REARM",
        steps_sha256="e8b6b89098db1b1b963c9fd091cacde8731eff58fdcc349a7311a3392e3924e5",
        result_sha256="1e6c5c6c9ef1ac1685a092dde546d5bf92500db380d54512bf4a7e6ab71de5bc",
        t=7, grid=BOTTLENECK_GRID,
        current={0: (2, 4), 1: (2, 5), 2: (2, 6), 3: (3, 5)},
        intended={0: (2, 5), 1: (3, 5), 2: (2, 5), 3: (2, 5)},
        blocked={0, 1, 2, 3},
        logged_conflict_types={0: "vertex", 1: "vertex", 2: "vertex", 3: "vertex"},
    ),
}


def _run(name):
    c = CASES[name]
    return build_wfg(c["current"], c["intended"], c["blocked"], c["grid"])


def _edge_keys(result):
    return [(e.src, e.dst, e.kind, e.cell) for e in result.edges]


class TestWfgSavedLogCases(unittest.TestCase):
    """T1-T4: candidate edges/groups for the four saved Trigger states."""

    def test_t1_corridor_t5_single_contest_group(self):
        r = _run("corridor_t5")
        self.assertEqual(_edge_keys(r), [(0, 1, "contest", (2, 5))])
        self.assertEqual(r.groups, ((0, 1),))
        self.assertEqual(r.rule_version, RULE_VERSION)

    def test_t2_intersection_t2_single_contest_group(self):
        r = _run("intersection_t2")
        self.assertEqual(_edge_keys(r), [(0, 1, "contest", (3, 3))])
        self.assertEqual(r.groups, ((0, 1),))

    def test_t3_bottleneck_t2_two_independent_groups(self):
        r = _run("bottleneck_t2")
        self.assertEqual(
            _edge_keys(r),
            [(0, 1, "contest", (2, 2)), (2, 3, "contest", (2, 7))],
        )
        self.assertEqual(r.groups, ((0, 1), (2, 3)))
        # No edge crosses the two groups.
        for e in r.edges:
            self.assertTrue({e.src, e.dst} in ({0, 1}, {2, 3}), e)

    def test_t4_bottleneck_t7_swap_recomputed_and_single_group(self):
        r = _run("bottleneck_t7")
        self.assertEqual(
            _edge_keys(r),
            [
                (0, 1, "occupy", (2, 5)),
                (0, 2, "contest", (2, 5)),
                (0, 3, "contest", (2, 5)),
                (1, 3, "swap", (3, 5)),
                (2, 1, "occupy", (2, 5)),
                (2, 3, "contest", (2, 5)),
            ],
        )
        self.assertEqual(r.groups, ((0, 1, 2, 3),))
        # The log records all four as "vertex" and edge_events=0; the swap
        # exists only in current/intended. The function must not depend on it.
        self.assertTrue(all(v == "vertex" for v in CASES["bottleneck_t7"]["logged_conflict_types"].values()))
        self.assertIn("swap", {e.kind for e in r.edges})

    def test_t4b_bottleneck_t7_group_does_not_hinge_on_pending_t7_edges(self):
        # 0->1 and 2->1 are occupy edges whose occupant (1) intends to move
        # out: exactly the PENDING T7 situation. They are flagged, and the
        # group stays connected through contest/swap edges alone, so this
        # case's group does not depend on how T7 is resolved.
        r = _run("bottleneck_t7")
        occupy = [e for e in r.edges if e.kind == "occupy"]
        self.assertEqual([(e.src, e.dst, e.occupant_moves) for e in occupy], [(0, 1, True), (2, 1, True)])
        parent = {a: a for a in range(4)}

        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x

        for e in r.edges:
            if e.kind != "occupy":
                parent[find(e.src)] = find(e.dst)
        self.assertEqual(len({find(a) for a in range(4)}), 1)


class TestWfgDeterminismAndIsolation(unittest.TestCase):
    def test_t5_input_order_does_not_change_result(self):
        for name, c in CASES.items():
            expected = _run(name)
            ids = list(c["current"])
            for perm in itertools.permutations(ids):
                with self.subTest(case=name, order=perm):
                    cur = {a: c["current"][a] for a in perm}
                    want = {a: list(c["intended"][a]) for a in reversed(perm)}
                    blocked = [a for a in reversed(perm) if a in c["blocked"]]
                    got = build_wfg(cur, want, blocked, np.asarray(c["grid"]))
                    self.assertEqual(got, expected)

    def test_t6_monitor_and_conflict_type_are_not_inputs(self):
        params = list(inspect.signature(build_wfg).parameters)
        self.assertEqual(params, ["current", "intended", "blocked", "grid"])
        # Monitor said all four agents block at Bottleneck t=2; the WFG
        # candidate is two independent groups, not one group of four.
        c = CASES["bottleneck_t2"]
        r = _run("bottleneck_t2")
        self.assertEqual(sorted(c["monitor_blocking_agents"]), [0, 1, 2, 3])
        self.assertNotIn((0, 1, 2, 3), r.groups)
        self.assertEqual(r.groups, ((0, 1), (2, 3)))

    def test_t6b_component_without_blocked_agent_is_not_a_group(self):
        # Synthetic variant of Bottleneck t=2 (NOT a logged state): only 0 and
        # 1 are passed as blocked. Contract v0 rule: a component needs >= 1
        # blocked agent to be a group; edges are unchanged.
        c = CASES["bottleneck_t2"]
        r = build_wfg(c["current"], c["intended"], {0, 1}, c["grid"])
        self.assertEqual(r.edges, _run("bottleneck_t2").edges)
        self.assertEqual(r.groups, ((0, 1),))
        self.assertEqual(build_wfg(c["current"], c["intended"], set(), c["grid"]).groups, ())


class TestWfgWallsAndBounds(unittest.TestCase):
    def test_t8_wall_and_out_of_bounds_intent_is_no_move(self):
        grid = _grid("""
.#.
...
""")
        # 0 at (0,0) intends wall (0,1); 1 at (0,2) intends out of bounds (-1,2).
        # Neither creates an edge even though 0's wall cell is adjacent to 1.
        r = build_wfg({0: (0, 0), 1: (0, 2)}, {0: (0, 1), 1: (-1, 2)}, {0, 1}, grid)
        self.assertEqual(r.edges, ())
        self.assertEqual(r.groups, ())

    def test_t8b_wall_intent_does_not_contest_a_real_mover(self):
        grid = _grid("""
...
...
""")
        # 0 at (0,0) intends to leave the map; 1 at (1,1) moves to free (1,0).
        # 0 is "no move" and so creates no edge; 1's target is empty.
        r = build_wfg({0: (0, 0), 1: (1, 1)}, {0: (0, -1), 1: (1, 0)}, {0}, grid)
        self.assertEqual(r.edges, ())
        self.assertEqual(r.groups, ())

    def test_t8d_two_agents_aiming_at_same_wall_do_not_contest(self):
        # Both bump into wall (0,1) and stay put: no shared landing cell.
        r = build_wfg({0: (0, 0), 1: (0, 2)}, {0: (0, 1), 1: (0, 1)}, {0, 1}, _grid(".#."))
        self.assertEqual(r.edges, ())
        self.assertEqual(r.groups, ())

    def test_t8c_blocked_singleton_is_not_a_group(self):
        grid = _grid("""
.#
..
""")
        r = build_wfg({0: (0, 0), 1: (1, 1)}, {0: (0, 1), 1: (1, 1)}, {0}, grid)
        self.assertEqual(r.groups, ())


class TestWfgInvalidInput(unittest.TestCase):
    GRID = _grid("""
...
...
""")

    def _bad(self, current, intended, blocked=(), grid=None):
        with self.assertRaises(ValueError):
            build_wfg(current, intended, blocked, self.GRID if grid is None else grid)

    def test_t9_invalid_inputs_raise_value_error(self):
        ok_cur, ok_want = {0: (0, 0), 1: (1, 2)}, {0: (0, 1), 1: (1, 1)}
        build_wfg(ok_cur, ok_want, {0}, self.GRID)  # sanity: valid input passes
        cases = {
            "no agents": ({}, {}),
            "id mismatch": ({0: (0, 0)}, {1: (0, 0)}),
            "shared current cell": ({0: (0, 0), 1: (0, 0)}, {0: (0, 0), 1: (0, 0)}),
            "non-adjacent intent": ({0: (0, 0)}, {0: (1, 1)}),
            "jump intent": ({0: (0, 0)}, {0: (0, 2)}),
            "current out of bounds": ({0: (5, 5)}, {0: (5, 5)}),
            "bool id": ({True: (0, 0)}, {True: (0, 0)}),
            "str id": ({"0": (0, 0)}, {"0": (0, 0)}),
            "float cell": ({0: (0.0, 0)}, {0: (0, 0)}),
            "3-tuple cell": ({0: (0, 0, 0)}, {0: (0, 0)}),
            "bool coord": ({0: (True, 0)}, {0: (1, 0)}),
        }
        for label, (cur, want) in cases.items():
            with self.subTest(label):
                self._bad(cur, want)
        with self.subTest("current on obstacle"):
            self._bad({0: (0, 1)}, {0: (0, 1)}, grid=_grid(".#."))
        with self.subTest("unknown blocked id"):
            self._bad(ok_cur, ok_want, blocked={7})
        with self.subTest("empty grid"):
            self._bad(ok_cur, ok_want, grid=[])
        with self.subTest("ragged grid"):
            self._bad(ok_cur, ok_want, grid=[[0, 0, 0], [0, 0]])


class TestWfgFixturesMatchSavedLogs(unittest.TestCase):
    """T10: fixtures are the saved log values (logs are git-ignored)."""

    def test_t10a_grids_match_controlled_scenario_maps(self):
        for name, c in CASES.items():
            with self.subTest(case=name):
                m = np.load(ROOT / "scenarios" / "controlled" / f"map_{c['scenario']}.npy")
                self.assertEqual(m.astype(int).tolist(), c["grid"])

    def test_t10b_fixtures_match_saved_step_csv_and_result_json(self):
        for name, c in CASES.items():
            run_dir = RUNS / c["run"]
            steps = run_dir / f"{c['run']}_steps.csv"
            result = run_dir / "result.json"
            if not (steps.is_file() and result.is_file()):
                self.skipTest(f"saved DEV logs not present: {run_dir}")
            with self.subTest(case=name):
                self.assertEqual(hashlib.sha256(steps.read_bytes()).hexdigest(), c["steps_sha256"])
                self.assertEqual(hashlib.sha256(result.read_bytes()).hexdigest(), c["result_sha256"])
                with steps.open(encoding="utf-8", newline="") as fh:
                    rows = [r for r in csv.DictReader(fh) if int(r["timestep"]) == c["t"]]
                cell = lambda r, k: (int(r[f"{k}_row"]), int(r[f"{k}_col"]))  # noqa: E731
                self.assertEqual({int(r["agent_id"]): cell(r, "current") for r in rows}, c["current"])
                self.assertEqual({int(r["agent_id"]): cell(r, "intended") for r in rows}, c["intended"])
                self.assertEqual({int(r["agent_id"]) for r in rows if r["blocked"] == "True"}, c["blocked"])
                if "logged_conflict_types" in c:
                    self.assertEqual({int(r["agent_id"]): r["conflict_type"] for r in rows},
                                     c["logged_conflict_types"])
                res = json.loads(result.read_text(encoding="utf-8"))
                call = [x for x in res["cbs_calls"] if x["trigger_timestep"] == c["t"]]
                self.assertEqual(len(call), 1)
                # period-1 repeat: S_t (= s_t) equals the step's current cells.
                s_t = {int(k): tuple(v) for k, v in call[0]["s_t"].items()}
                self.assertEqual(s_t, c["current"])
                self.assertEqual({int(r["agent_id"]): cell(r, "next") for r in rows}, c["current"])
                hist = [h for h in res["monitor"]["trigger_history"] if h["trigger_timestep"] == c["t"]]
                self.assertEqual(len(hist), 1)
                self.assertEqual(hist[0]["period"], 1)
                if "monitor_blocking_agents" in c:
                    self.assertEqual(hist[0]["blocking_agents"], c["monitor_blocking_agents"])


class TestWfgPendingRules(unittest.TestCase):
    @unittest.skip("PENDING T7: rule for following into a vacated cell is not agreed (contract P5)")
    def test_t7_follow_into_vacated_cell(self):
        """No expected value on purpose. v0 emits an occupy edge with
        occupant_moves=True; whether that is a real wait is undecided."""


if __name__ == "__main__":
    unittest.main()
