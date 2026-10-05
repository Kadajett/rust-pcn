import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from generate_jev_synthetic import all_real_sources_exhausted, main as generate_synthetic
from prepare_river_datasets import corpus_replay_exposure, read_checkpoint_metadata
from river_dataset_scheduler import incomplete_active_dataset, scheduler_step


class SchedulerReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.active = {
            "id": "accepted-source",
            "status": "active",
            "prepared_training_windows": 4,
        }
        self.registry = {"datasets": [self.active]}
        self.metadata = {
            "epoch": 15,
            "cumulative_batches": 100,
            "corpora": {"accepted-source": {"examples_seen": 1200}},
            "data_replay": {"corpus_baselines": {"accepted-source": 1200}},
        }

    def write_json(self, relative: str, value: dict) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def write_expert_set(self) -> None:
        self.write_json("generation-committed/inherited/checkpoint.json", self.metadata)
        self.write_json(
            "experts.json",
            {
                "schema": "river-universal-expert-set-v1",
                "epoch": self.metadata["epoch"],
                "cumulative_batches": self.metadata["cumulative_batches"],
                "experts": [
                    {
                        "role": "request_conditioned",
                        "checkpoint": "generation-committed/request-conditioned",
                    },
                    {
                        "role": "inherited",
                        "checkpoint": "generation-committed/inherited",
                    },
                ],
            },
        )

    def test_single_expert_metadata_remains_supported(self) -> None:
        self.write_json("checkpoint.json", self.metadata)
        self.assertEqual(
            incomplete_active_dataset(self.registry, read_checkpoint_metadata(self.root)),
            ("accepted-source", 0, 8),
        )

    def test_committed_manifest_overrides_stale_root_and_uncommitted_generation(self) -> None:
        stale = {**self.metadata, "data_replay": None}
        self.write_json("checkpoint.json", stale)
        self.write_json("generation-uncommitted/inherited/checkpoint.json", stale)
        self.write_expert_set()
        self.write_json("registry.json", self.registry)
        args = argparse.Namespace(
            registry=self.root / "registry.json",
            checkpoint=self.root,
            dataset_root=self.root / "datasets",
            env_file=self.root / "absent.env",
        )
        with patch(
            "river_dataset_scheduler.subprocess.run",
            side_effect=AssertionError("admission attempted before replay completed"),
        ):
            self.assertEqual(
                scheduler_step(args),
                "waiting for accepted-source: 0/8 forward+reverse examples",
            )

    def test_invalid_or_incomplete_manifest_never_falls_back_to_stale_root(self) -> None:
        self.write_json("checkpoint.json", {**self.metadata, "data_replay": None})
        for failure in ("missing-committed-metadata", "counter-mismatch", "invalid-json"):
            with self.subTest(failure=failure):
                self.write_expert_set()
                if failure == "missing-committed-metadata":
                    (self.root / "generation-committed/inherited/checkpoint.json").unlink()
                    expected = FileNotFoundError
                elif failure == "counter-mismatch":
                    self.write_json(
                        "generation-committed/inherited/checkpoint.json",
                        {**self.metadata, "cumulative_batches": 99},
                    )
                    expected = ValueError
                else:
                    (self.root / "experts.json").write_text("{", encoding="utf-8")
                    expected = ValueError
                with self.assertRaises(expected):
                    read_checkpoint_metadata(self.root)

    def test_replay_requires_both_traversals_without_resetting_lifetime_counts(self) -> None:
        for lifetime, progress, incomplete in (
            (1199, 0, ("accepted-source", 0, 8)),
            (1200, 0, ("accepted-source", 0, 8)),
            (1204, 4, ("accepted-source", 4, 8)),
            (1207, 7, ("accepted-source", 7, 8)),
            (1208, 8, None),
            (1220, 20, None),
        ):
            with self.subTest(lifetime=lifetime):
                self.metadata["corpora"]["accepted-source"]["examples_seen"] = lifetime
                self.assertEqual(corpus_replay_exposure(self.metadata, "accepted-source"), progress)
                self.assertEqual(incomplete_active_dataset(self.registry, self.metadata), incomplete)
                self.assertEqual(all_real_sources_exhausted(self.registry, self.metadata), incomplete is None)
                self.assertEqual(self.metadata["corpora"]["accepted-source"]["examples_seen"], lifetime)
        self.metadata["data_replay"] = None
        self.assertIsNone(incomplete_active_dataset(self.registry, self.metadata))
        self.metadata["data_replay"] = {"corpus_baselines": {}}
        self.assertEqual(corpus_replay_exposure(self.metadata, "accepted-source"), 1220)

    def test_queued_and_evaluation_only_sources_do_not_become_training_progress(self) -> None:
        self.metadata["corpora"]["accepted-source"]["examples_seen"] = 1208
        self.registry["datasets"].extend(
            [
                {"id": "holdout", "status": "evaluation-only", "prepared_training_windows": 500},
                {"id": "databricks-dolly-15k-v1", "status": "queued-download", "prepared_training_windows": 500},
            ]
        )
        self.assertIsNone(incomplete_active_dataset(self.registry, self.metadata))
        self.assertFalse(all_real_sources_exhausted(self.registry, self.metadata))

    def test_synthetic_consumer_reads_expert_root_and_refuses_unfinished_replay(self) -> None:
        self.write_json("checkpoint.json", {**self.metadata, "data_replay": None})
        self.write_expert_set()
        self.write_json("registry.json", self.registry)
        arguments = [
            "generate_jev_synthetic.py",
            "--registry", str(self.root / "registry.json"),
            "--checkpoint", str(self.root),
            "--dataset-root", str(self.root / "datasets"),
            "--env-file", str(self.root / "absent.env"),
        ]
        with patch("sys.argv", arguments), patch(
            "generate_jev_synthetic.call_jev",
            side_effect=AssertionError("Jev called before replay completed"),
        ), patch(
            "generate_jev_synthetic.read_env_key",
            side_effect=AssertionError("API credentials read before replay completed"),
        ):
            with self.assertRaisesRegex(SystemExit, "real dataset queue or traversal is not exhausted"):
                generate_synthetic()
        self.assertFalse((self.root / "datasets").exists())

    def test_synthetic_wave_gate_uses_replay_progress(self) -> None:
        identifier = "jev-synthetic-wave-000001"
        self.registry = {
            "datasets": [
                {"id": identifier, "status": "active", "prepared_training_windows": 4},
            ]
        }
        self.metadata["corpora"] = {identifier: {"examples_seen": 1207}}
        self.metadata["data_replay"] = {"corpus_baselines": {identifier: 1200}}
        self.write_expert_set()
        self.write_json("registry.json", self.registry)
        arguments = [
            "generate_jev_synthetic.py",
            "--registry", str(self.root / "registry.json"),
            "--checkpoint", str(self.root),
            "--dataset-root", str(self.root / "datasets"),
            "--env-file", str(self.root / "absent.env"),
        ]
        with patch("sys.argv", arguments), patch(
            "generate_jev_synthetic.read_env_key",
            side_effect=AssertionError("API credentials read before prior wave replay completed"),
        ):
            with self.assertRaisesRegex(SystemExit, f"prior synthetic wave is not exhausted: {identifier}"):
                generate_synthetic()


if __name__ == "__main__":
    unittest.main()
