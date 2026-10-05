#!/usr/bin/env python3
"""Prepare one B200 instruction source with the existing River adapters.

Run beside prepare_river_datasets.py in ~/hf-cache/b200-fetch. The supplied
registry is read-only; each invocation owns a private registry and result file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

from prepare_river_datasets import atomic_json, prepare, read_json, repo_id


DATASET_IDS = (
    "openassistant-oasst2-v1",
    "rajpurkar-squad-v2-v1",
    "hotpotqa-distractor-v1",
    "openai-gsm8k-main-v1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True, choices=DATASET_IDS)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path.home() / "hf-cache/b200-fetch/results",
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("/fast-storage/river-datasets")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = read_json(args.registry)
    dataset = next(item for item in registry["datasets"] if item["id"] == args.id)
    # Resolve even an already-pinned revision through the Hub, so provenance
    # records the commit actually requested, not a moving branch or guessed SHA.
    info = HfApi().dataset_info(
        repo_id(dataset["source"]), revision=dataset["revision"]
    )
    if not info.sha:
        raise RuntimeError(f"Hub returned no revision for {args.id}")
    dataset["revision"] = info.sha
    private_registry = args.results_dir / "work" / f"{args.id}.registry.json"
    atomic_json(private_registry, {"datasets": [dataset]})
    args.dataset_root.mkdir(parents=True, exist_ok=True)

    # All four pinned sources contain repository-native parquet train splits.
    # Keep train-only selection, OASST filtering, serialization, shard sizing,
    # and held-out window accounting in the established preparer unchanged.
    prepare(private_registry, args.dataset_root, dataset)
    prepared = read_json(private_registry)["datasets"][0]
    prepared_path = Path(prepared["prepared_path"])
    attribution_path = prepared_path / "attribution.json"
    attribution = read_json(attribution_path)
    card = info.card_data.to_dict() if info.card_data is not None else {}
    result = {
        **attribution,
        "id": args.id,
        "status": "complete",
        "prepared_path": str(prepared_path),
        "attribution_path": str(attribution_path),
        "hub_license": card.get("license"),
    }
    atomic_json(args.results_dir / f"{args.id}.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
