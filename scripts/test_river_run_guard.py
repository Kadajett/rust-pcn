"""Unit tests for the pure decision logic in river_run_guard.py (no systemd, no filesystem side effects)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import river_run_guard as guard  # noqa: E402

REAL_EXECSTART = (
    "ExecStart={ path=/home/kadajett/Dev/rust-pcn/target/live/river-pcn-train-universal ; "
    "argv[]=/home/kadajett/Dev/rust-pcn/target/live/river-pcn-train-universal --dual-expert --replays "
    "/bulk-storage/connectome-merc/marty-continuous-20260916 --registry "
    "/home/kadajett/Dev/rust-pcn/datasets/training-registry-prose.json --relax-steps 100 --max-relax-steps 200 "
    "--inherited-layer-alphas 0.1,0.1,0.1 --request-layer-alphas 0.00005,0.07,0.00001 --byte-target-encoding zero "
    "--inherited-eta 0.003 --task-examples-per-dataset 512 --task-rehearsal-examples-per-dataset 64 "
    "--checkpoint-every-batches 32 --focus-lane-records-per-stage 512 --focus-lane-maintenance "
    "code,structured,choice,score,noul --output "
    "/bulk-storage/connectome-merc/river-universal-checkpoints/river-v8-fresh-seed20261004 --telemetry-dir "
    "/bulk-storage/connectome-merc/river-multimodal-runs/current --run-name River Song fresh start 2 (Oct 4 2026) "
    "--noul-examples-per-stage 8 --replay-examples-per-stage 8 --skip-request-expert-training --corpus-batch-size 128 "
    "--mask-rate 0 --byte-head-reference-batch-size 192 --positive-phase-start fresh --inherited-idle-outputs free ; "
    "ignore_errors=no ; start_time=[n/a] ; stop_time=[n/a] ; pid=0 ; code=(null) ; status=0/0 }"
)
OOM_JOURNAL = "Started river-song-training.service - River Song.\nsome log\nCUDA_ERROR_OUT_OF_MEMORY at layer 2\n"
HEALTHY = {"schema": "river-run-health-v1",
           "last_healthy": {"generation": "generation-e1-b640-100", "epoch": 1, "cumulative_batches": 640},
           "rollbacks": [], "blocked": None}
BLOCKED_HEALTH = {**HEALTHY, "blocked": {"unix_millis": 1, "reason": "3 rollbacks to the same generation",
                                         "last_healthy_generation": "generation-e1-b640-100"}}


class ParseOutputRootTest(unittest.TestCase):
    def test_real_execstart_with_unquoted_run_name(self):
        self.assertEqual(guard.parse_output_root(REAL_EXECSTART),
                         Path("/bulk-storage/connectome-merc/river-universal-checkpoints/river-v8-fresh-seed20261004"))

    def test_missing_output_raises(self):
        with self.assertRaises(ValueError):
            guard.parse_output_root("ExecStart={ path=/x ; argv[]=/x --dual-expert }")


class ClassifyExitTest(unittest.TestCase):
    def test_status_78_is_blocked_even_with_oom_and_healthy_ledger(self):
        self.assertEqual(guard.classify_exit(78, HEALTHY, OOM_JOURNAL), guard.BLOCKED)

    def test_health_blocked_wins_over_status_zero(self):
        self.assertEqual(guard.classify_exit(0, BLOCKED_HEALTH, ""), guard.BLOCKED)

    def test_oom_in_journal_restarts(self):
        self.assertEqual(guard.classify_exit(1, HEALTHY, OOM_JOURNAL), guard.OOM_RESTART)

    def test_oom_before_last_start_is_ignored(self):
        journal = "CUDA_ERROR_OUT_OF_MEMORY\nStarted river-song-training.service - River Song.\nSafetyStop energy\n"
        self.assertEqual(guard.classify_exit(1, HEALTHY, journal), guard.RESTART)

    def test_oom_without_start_marker_uses_whole_tail(self):
        self.assertEqual(guard.classify_exit(1, {}, "x\nCUDA_ERROR_OUT_OF_MEMORY\n"), guard.OOM_RESTART)

    def test_exit_zero_is_finished(self):
        self.assertEqual(guard.classify_exit(0, HEALTHY, "Started x\n"), guard.FINISHED)

    def test_other_nonzero_is_restart(self):
        self.assertEqual(guard.classify_exit(101, {}, "Started x\nthread panicked\n"), guard.RESTART)

    def test_unknown_status_waits(self):
        self.assertEqual(guard.classify_exit(None, {}, ""), guard.WAIT)


class RestartBudgetTest(unittest.TestCase):
    def test_oom_budget_blocks_fourth_restart_within_hour_and_recovers_after(self):
        clock = [0.0]
        budget = guard.RestartBudget(guard.OOM_LIMIT, guard.OOM_WINDOW, clock=lambda: clock[0])
        for minute in (0, 10, 20):
            clock[0] = minute * 60.0
            self.assertTrue(budget.allow())
        clock[0] = 30 * 60.0
        self.assertFalse(budget.allow())
        clock[0] = 61 * 60.0  # first event expired
        self.assertTrue(budget.allow())


class HealthyTagTest(unittest.TestCase):
    def test_matches_last_healthy(self):
        self.assertIs(guard.healthy_tag(HEALTHY, "generation-e1-b640-100"), True)
        self.assertIs(guard.healthy_tag(HEALTHY, "generation-e1-b672-101"), False)

    def test_no_ledger_is_unknown(self):
        self.assertIsNone(guard.healthy_tag({}, "generation-e0-b0-1"))
        self.assertIsNone(guard.healthy_tag({"last_healthy": None}, "generation-e0-b0-1"))


class PrunePlanTest(unittest.TestCase):
    rows = [{"generation": f"generation-e1-b{32 * i}-{i}"} for i in range(12)]

    def test_keeps_newest_keep_and_protected(self):
        protected = guard.protected_generations({**HEALTHY, "last_healthy": {"generation": self.rows[1]["generation"]},
                                                 "rollbacks": [{"to_generation": self.rows[0]["generation"]}]})
        plan = guard.prune_plan(self.rows, 8, protected)
        self.assertEqual(plan, [self.rows[2]["generation"], self.rows[3]["generation"]])
        self.assertNotIn(self.rows[0]["generation"], plan)
        self.assertNotIn(self.rows[1]["generation"], plan)
        for row in self.rows[-8:]:
            self.assertNotIn(row["generation"], plan)

    def test_blocked_last_healthy_is_protected(self):
        protected = guard.protected_generations({"blocked": {"last_healthy_generation": self.rows[0]["generation"]}})
        self.assertNotIn(self.rows[0]["generation"], guard.prune_plan(self.rows, 2, protected))

    def test_nothing_to_prune_under_keep(self):
        self.assertEqual(guard.prune_plan(self.rows[:3], 8, set()), [])


class StalePartialTest(unittest.TestCase):
    def test_only_old_partials(self):
        entries = [("generation-e1-b1-1.partial", 1000.0), ("generation-e1-b2-2.partial", 5000.0),
                   ("generation-e1-b3-3", 0.0)]
        self.assertEqual(guard.stale_partials(entries, now_epoch=5600.0), ["generation-e1-b1-1.partial"])


class GenerationNameTest(unittest.TestCase):
    def test_parse_and_sort_order(self):
        names = ["generation-e1-b605-1791147290795941742", "generation-e0-b0-1791143520580524251", "replay-cache"]
        parsed = [n for n in names if guard.parse_generation_name(n)]
        self.assertEqual(sorted(parsed, key=guard.parse_generation_name),
                         ["generation-e0-b0-1791143520580524251", "generation-e1-b605-1791147290795941742"])


if __name__ == "__main__":
    unittest.main()
