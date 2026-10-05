#!/usr/bin/env python3
"""Admit the next real River dataset after the current source finishes both traversals."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path
from typing import Any

from prepare_river_datasets import (
    corpus_replay_exposure,
    eligible,
    read_checkpoint_metadata,
    read_json,
)


def incomplete_active_dataset(
    registry: dict[str, Any], checkpoint: dict[str, Any]
) -> tuple[str, int, int] | None:
    prepared = [
        dataset
        for dataset in registry.get("datasets", [])
        if dataset.get("status") == "active"
        and isinstance(dataset.get("prepared_training_windows"), int)
        and dataset["prepared_training_windows"] > 0
    ]
    prepared.sort(key=lambda dataset: str(dataset.get("introduced_stage") or ""))
    for dataset in prepared:
        identifier = str(dataset["id"])
        required = 2 * int(dataset["prepared_training_windows"])
        seen = corpus_replay_exposure(checkpoint, identifier)
        if seen < required:
            return identifier, seen, required
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    return args


def scheduler_step(args: argparse.Namespace) -> str:
    registry = read_json(args.registry)
    checkpoint = read_checkpoint_metadata(args.checkpoint)
    current = incomplete_active_dataset(registry, checkpoint)
    if current is not None:
        identifier, seen, required = current
        return f"waiting for {identifier}: {seen}/{required} forward+reverse examples"
    candidates = eligible(registry)
    if candidates:
        command = [
            str(Path(__file__).with_name("prepare_river_datasets.py")),
            "--registry",
            str(args.registry),
            "--dataset-root",
            str(args.dataset_root),
        ]
        subprocess.run(command, check=True)
        return f"admitted next real dataset; {len(candidates) - 1} queued real sources remain"
    command = [
        str(Path(__file__).with_name("generate_jev_synthetic.py")),
        "--registry",
        str(args.registry),
        "--checkpoint",
        str(args.checkpoint),
        "--dataset-root",
        str(args.dataset_root),
        "--env-file",
        str(args.env_file),
    ]
    subprocess.run(command, check=True)
    return "real training queue exhausted; admitted the next Jev synthetic wave"


def main() -> None:
    args = parse_args()
    last = ""
    while True:
        try:
            status = scheduler_step(args)
        except Exception as error:
            status = f"scheduler error: {error}"
        if status != last:
            print(status, flush=True)
            last = status
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
