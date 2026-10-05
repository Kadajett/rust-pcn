#!/usr/bin/env python3
"""Generate a bounded typed-supervision wave with Jev after real data is exhausted."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.request
from pathlib import Path
from typing import Any

from prepare_river_datasets import (
    atomic_json,
    corpus_replay_exposure,
    eligible,
    read_checkpoint_metadata,
    record,
    write_shards,
)

DEFAULT_URL = "https://api.typesafe.ai/v1/systemone"
SCORE_LEVELS = [
    "unsupported by the visible evidence",
    "weakly supported",
    "mixed or ambiguous",
    "well supported",
    "directly established by the visible evidence",
]


def read_env_key(path: Path) -> str:
    direct = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if direct:
        return direct
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        if name.strip() == "TYPESAFE_API_KEY":
            return value.strip().strip("\"").strip("'")
    raise RuntimeError("TYPESAFE_API_KEY is unavailable")


def scenario(index: int, rng: random.Random) -> dict[str, Any]:
    temperature = rng.randint(45, 125)
    pressure = rng.randint(10, 95)
    vibration = rng.randint(0, 12)
    coolant = rng.choice(["normal", "low", "critical"])
    door = rng.choice(["closed", "open"])
    mode = rng.choice(["idle", "running", "maintenance"])
    return {
        "scenario_id": index,
        "machine": {
            "temperature_c": temperature,
            "pressure_percent": pressure,
            "vibration_mm_s": vibration,
            "coolant": coolant,
            "access_door": door,
            "mode": mode,
        },
        "operator_goal": rng.choice(
            [
                "continue safely",
                "diagnose the dominant warning",
                "choose the least destructive intervention",
                "preserve evidence before acting",
            ]
        ),
    }


def questions_for(index: int) -> dict[str, Any]:
    return {
        f"unsafe_{index}": {
            "type": "noul",
            "instructions": "Does the visible machine state require stopping normal operation now?",
            "criteria": {
                "true": "The visible measurements or access state make continued normal operation unsafe.",
                "false": "The visible measurements support continued normal operation.",
            },
        },
        f"action_{index}": {
            "type": "choice",
            "instructions": "Choose the best next action using only the visible state and operator goal.",
            "options": [
                "continue and monitor",
                "reduce load and inspect",
                "stop and isolate the machine",
                "enter maintenance diagnostics",
            ],
        },
        f"confidence_{index}": {
            "type": "score",
            "instructions": "Rate how strongly the visible evidence supports the chosen action.",
            "criteria": SCORE_LEVELS,
        },
    }


def call_jev(api_key: str, states: list[dict[str, Any]], timeout: float) -> dict[str, Any]:
    questions: dict[str, Any] = {}
    for item in states:
        questions.update(questions_for(int(item["scenario_id"])))
    payload = json.dumps(
        {
            "model": "jev-latest",
            "state": {
                "task": "Provide typed supervision for each independent machine scenario. Use only visible fields; do not assume hidden facts.",
                "scenarios": states,
            },
            "questions": questions,
        },
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    request = urllib.request.Request(
        os.environ.get("TYPESAFE_API_URL", DEFAULT_URL),
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "river-song-jev-synthetic/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read())
    answers = result.get("answers") if isinstance(result, dict) else None
    if not isinstance(answers, dict):
        raise RuntimeError("Jev response omitted typed answers")
    return answers


def all_real_sources_exhausted(registry: dict[str, Any], checkpoint: dict[str, Any]) -> bool:
    if eligible(registry):
        return False
    corpora = checkpoint.get("corpora") if isinstance(checkpoint, dict) else {}
    if not isinstance(corpora, dict):
        return False
    for dataset in registry.get("datasets", []):
        windows = dataset.get("prepared_training_windows")
        if dataset.get("status") != "active" or not isinstance(windows, int) or windows <= 0:
            continue
        if str(dataset.get("id", "")).startswith("jev-synthetic-wave-"):
            continue
        seen = corpus_replay_exposure(checkpoint, str(dataset.get("id")))
        if seen < 2 * windows:
            return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--scenarios", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.scenarios <= 0 or not 1 <= args.batch_size <= 20 or args.timeout <= 0:
        parser.error("scenarios, batch-size, and timeout must be positive; batch-size must be <= 20")
    return args


def main() -> None:
    args = parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    checkpoint = read_checkpoint_metadata(args.checkpoint)
    if not all_real_sources_exhausted(registry, checkpoint):
        raise SystemExit("real dataset queue or traversal is not exhausted; refusing Jev synthesis")
    prior_waves = [
        dataset
        for dataset in registry.get("datasets", [])
        if str(dataset.get("id", "")).startswith("jev-synthetic-wave-")
    ]
    for dataset in prior_waves:
        windows = int(dataset.get("prepared_training_windows") or 0)
        if corpus_replay_exposure(checkpoint, str(dataset.get("id"))) < 2 * windows:
            raise SystemExit(f"prior synthetic wave is not exhausted: {dataset.get('id')}")
    wave = len(prior_waves) + 1
    identifier = f"jev-synthetic-wave-{wave:06d}"
    target = args.dataset_root / identifier
    if target.exists():
        raise FileExistsError(target)
    api_key = read_env_key(args.env_file)
    rng = random.Random(0x5249564552 ^ wave)
    records: list[str] = []
    for start in range(0, args.scenarios, args.batch_size):
        states = [scenario(index, rng) for index in range(start, min(start + args.batch_size, args.scenarios))]
        answers = call_jev(api_key, states, args.timeout)
        for state in states:
            index = int(state["scenario_id"])
            labels = {
                "unsafe": answers.get(f"unsafe_{index}"),
                "action": answers.get(f"action_{index}"),
                "confidence": answers.get(f"confidence_{index}"),
            }
            records.append(
                record(
                    "jev-generated-typed-decision",
                    [("state", state), ("typed-targets", labels)],
                )
            )
    count, total_bytes, shards = write_shards(target, records)
    training_windows = sum(
        max(0, (int(shard["bytes"]) - 64 + 63) // 64) for shard in shards
    )
    attribution = {
        "dataset_id": identifier,
        "source": "typesafe:jev-latest",
        "generator": "scripts/generate_jev_synthetic.py",
        "prepared_records": count,
        "prepared_bytes": total_bytes,
        "prepared_training_windows": training_windows,
        "prepared_unix_millis": int(time.time() * 1000),
        "shards": shards,
    }
    atomic_json(target / "attribution.json", attribution)
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    registry["datasets"].append(
        {
            "id": identifier,
            "source": "typesafe:jev-latest",
            "kind": "typed-decision-synthetic",
            "status": "active",
            "license": "generated-for-river",
            "introduced_stage": f"river-v5-jev-synthetic-{wave:06d}",
            "prepared_path": str(target),
            "prepared_records": count,
            "prepared_bytes": total_bytes,
            "prepared_training_windows": training_windows,
            "notes": "Generated only after every accessible real training source completed forward and reverse traversal.",
        }
    )
    atomic_json(args.registry, registry)
    print(
        f"activated {identifier}: records={count} windows={training_windows} bytes={total_bytes}",
        flush=True,
    )


if __name__ == "__main__":
    main()
