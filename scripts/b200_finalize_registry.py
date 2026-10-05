#!/usr/bin/env python3
"""Activate B200 datasets only after their helpers report real completed output."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from b200_fetch_text import TARGETS

RAW_IDS = (
    "tinystories-v1",
    "fineweb-edu-sample-10bt-v1",
    "wikipedia-20231101-en-v1",
)
TASK_IDS = (
    "openassistant-oasst2-v1",
    "rajpurkar-squad-v2-v1",
    "hotpotqa-distractor-v1",
    "openai-gsm8k-main-v1",
)


def finalize(base: Path, results_dir: Path, output: Path, require_complete: bool = False) -> dict:
    registry = json.loads(base.read_text(encoding="utf-8"))
    entries = {item["id"]: item for item in registry["datasets"]}
    missing = []
    completed = []
    for identifier in (*TASK_IDS, *RAW_IDS):
        path = results_dir / f"{identifier}.json"
        result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if result.get("status") != "complete":
            missing.append(identifier)
            if identifier in RAW_IDS:
                repo_id, config, license_name, _ = TARGETS[identifier]
                registry["datasets"].append({
                    "id": identifier,
                    "source": f"/bulk-storage/datasets/hf-text/{identifier}",
                    "kind": "text",
                    "status": "queued-download",
                    "hf_source": f"https://huggingface.co/datasets/{repo_id}",
                    "config": config,
                    "license": license_name,
                    "introduced_stage": None,
                })
            continue
        if result.get("id") != identifier or not result.get("revision"):
            raise ValueError(f"invalid completed provenance: {path}")
        if identifier in TASK_IDS:
            item = entries[identifier]
            for key in ("prepared_path", "prepared_records", "prepared_bytes", "prepared_training_windows"):
                item[key] = result[key]
            if min(item["prepared_records"], item["prepared_bytes"], item["prepared_training_windows"]) <= 0:
                raise ValueError(f"empty prepared dataset: {identifier}")
        else:
            if result["bytes"] <= 0 or result["documents"] <= 0:
                raise ValueError(f"empty raw dataset: {identifier}")
            item = {
                "id": identifier,
                "source": result["source"],
                "kind": "text",
                "hf_source": f"https://huggingface.co/datasets/{result['repo_id']}",
                "downloaded_text_bytes": result["bytes"],
                "downloaded_documents": result["documents"],
                "provenance_path": str(path),
            }
            if result.get("config"):
                item["config"] = result["config"]
            # Append rather than reshuffle existing source order: cursor identity is per dataset.
            registry["datasets"].append(item)
        item.update(
            status="active",
            revision=result["revision"],
            license=result["license"],
            introduced_stage="river-b200-hf-20261005",
        )
        completed.append(identifier)
    if require_complete and missing:
        raise RuntimeError(f"datasets not complete: {', '.join(missing)}")
    registry["notes"] = (
        registry.get("notes", "")
        + " B200 copy: only completed Hugging Face acquisitions are additionally active; unfinished raw sources remain queued without invented counts. Base source order and all other entries are preserved."
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(registry, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output)
    print(json.dumps({"registry": str(output), "completed": completed, "incomplete": missing}), flush=True)
    return registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    finalize(args.base, args.results_dir, args.output, args.require_complete)


if __name__ == "__main__":
    main()
