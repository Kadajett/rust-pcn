#!/usr/bin/env python3
"""Acquire one approved real dataset and compile it into bounded Song text shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from datasets import load_dataset

ELIGIBLE_STATUSES = {
    "queued-download",
    "queued-adapter",
    "queued-attribution",
    "queued-attribution-review",
    "queued-gated-access",
}
PRIORITY = [
    "zefancai-open-jev-release-v2-v1",
    "databricks-dolly-15k-v1",
    "openassistant-oasst2-v1",
    "openai-gsm8k-main-v1",
    "google-mbpp-full-v1",
    "rajpurkar-squad-v2-v1",
    "hotpotqa-distractor-v1",
    "allenai-ai2-arc-v1",
    "vagmi-jevlite-synthetic-v1",
    "salesforce-xlam-function-calling-60k-v1",
]
SHARD_BYTES = 8 * 1024 * 1024

TASK_AWARE_KINDS = {
    "typed-decision",
    "typed-decision-soft-label",
    "instruction-response",
    "conversation-ranked",
    "grounded-question-answering",
    "multi-hop-grounded-question-answering",
    "reasoning-question-answering",
    "code-instruction",
    "structured-function-calling",
}
TASK_HOLDOUT_DIVISOR = 20


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_checkpoint_metadata(root: Path) -> dict[str, Any]:
    """Read single-expert metadata or the committed inherited expert, never a stale root."""
    manifest_path = root / "experts.json"
    if not manifest_path.exists() and not manifest_path.is_symlink():
        return read_json(root / "checkpoint.json")
    manifest = read_json(manifest_path)
    experts = manifest.get("experts")
    if manifest.get("schema") != "river-universal-expert-set-v1" or not isinstance(experts, list):
        raise ValueError(f"invalid expert-set manifest: {manifest_path}")
    inherited = [
        expert
        for expert in experts
        if isinstance(expert, dict) and expert.get("role") == "inherited"
    ]
    if len(inherited) != 1:
        raise ValueError(f"expected one inherited expert: {manifest_path}")
    directory = inherited[0].get("checkpoint")
    if not isinstance(directory, str) or not directory:
        raise ValueError(f"missing inherited checkpoint directory: {manifest_path}")
    relative = Path(directory)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"unsafe inherited checkpoint directory: {directory}")
    checkpoint = read_json(root / relative / "checkpoint.json")
    if any(
        checkpoint.get(key) != manifest.get(key)
        for key in ("epoch", "cumulative_batches")
    ):
        raise ValueError(f"inherited checkpoint does not match committed manifest: {manifest_path}")
    return checkpoint


def corpus_replay_exposure(checkpoint: dict[str, Any], identifier: str) -> int:
    """Return forward/reverse exposure since the latest data-only rewind."""
    corpora = checkpoint.get("corpora")
    state = corpora.get(identifier) if isinstance(corpora, dict) else None
    seen = int(state.get("examples_seen") or 0) if isinstance(state, dict) else 0
    replay = checkpoint.get("data_replay")
    baselines = replay.get("corpus_baselines") if isinstance(replay, dict) else None
    baseline = int(baselines.get(identifier) or 0) if isinstance(baselines, dict) else 0
    return max(0, seen - baseline)


def compact(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def record(kind: str, fields: Iterable[tuple[str, Any]]) -> str:
    parts = [f"<river-example kind={json.dumps(kind)}>\n"]
    for name, value in fields:
        text = compact(value)
        if text:
            parts.append(f"<{name}>\n{text}\n</{name}>\n")
    parts.append("</river-example>\n")
    return "".join(parts)


def repo_id(source: str) -> str:
    prefix = "https://huggingface.co/datasets/"
    if not source.startswith(prefix):
        raise ValueError(f"unsupported remote source: {source}")
    return source.removeprefix(prefix).strip("/")


def dataset_rows(dataset: dict[str, Any]) -> Iterator[dict[str, Any]]:
    identifier = dataset["id"]
    path = repo_id(dataset["source"])
    revision = dataset.get("revision")
    config = dataset.get("config")
    configs = config if isinstance(config, list) else [config]
    for name in configs:
        loaded = load_dataset(path, name, split="train", revision=revision)
        if identifier == "openassistant-oasst2-v1":
            rows = [dict(row) for row in loaded]
            by_id = {row.get("message_id"): row for row in rows}
            for row in rows:
                if row.get("role") != "assistant" or row.get("deleted") is True:
                    continue
                if row.get("review_result") is False or row.get("rank") not in (None, 0):
                    continue
                parent = by_id.get(row.get("parent_id"))
                if not parent or parent.get("role") != "prompter" or parent.get("deleted") is True:
                    continue
                yield {
                    "prompt": parent.get("text"),
                    "response": row.get("text"),
                    "language": row.get("lang"),
                    "rank": row.get("rank"),
                }
        else:
            yield from (dict(row) for row in loaded)


def serialize(identifier: str, row: dict[str, Any]) -> str | None:
    if identifier == "zefancai-open-jev-release-v2-v1":
        return record(
            "typed-decision",
            [
                ("state", row.get("state_json")),
                ("question", row.get("question")),
                ("answer-kind", row.get("kind")),
                ("options", row.get("options")),
                ("target", row.get("target")),
            ],
        )
    if identifier == "vagmi-jevlite-synthetic-v1":
        return record(
            "typed-decision-soft-label",
            [
                ("state", row.get("state")),
                ("question", row.get("question")),
                ("answer-kind", row.get("type")),
                ("options", row.get("options")),
                ("criteria", row.get("criteria")),
                ("target-distribution", row.get("label")),
            ],
        )
    if identifier == "databricks-dolly-15k-v1":
        return record(
            "instruction-response",
            [
                ("instruction", row.get("instruction")),
                ("context", row.get("context")),
                ("response", row.get("response")),
            ],
        )
    if identifier == "openassistant-oasst2-v1":
        return record(
            "conversation-response",
            [("prompt", row.get("prompt")), ("response", row.get("response"))],
        )
    if identifier == "rajpurkar-squad-v2-v1":
        answers = row.get("answers") or {}
        texts = answers.get("text") if isinstance(answers, dict) else []
        answer = texts[0] if texts else "The supplied context does not answer this question."
        return record(
            "grounded-question-answering",
            [
                ("title", row.get("title")),
                ("context", row.get("context")),
                ("question", row.get("question")),
                ("response", answer),
            ],
        )
    if identifier == "hotpotqa-distractor-v1":
        context = row.get("context") or {}
        if isinstance(context, dict):
            titles = context.get("title") or []
            sentences = context.get("sentences") or []
            context = [
                {"title": title, "sentences": text}
                for title, text in zip(titles, sentences, strict=False)
            ]
        return record(
            "multi-hop-grounded-question-answering",
            [
                ("context", context),
                ("question", row.get("question")),
                ("response", row.get("answer")),
                ("supporting-facts", row.get("supporting_facts")),
            ],
        )
    if identifier == "openai-gsm8k-main-v1":
        return record(
            "reasoning-question-answering",
            [("question", row.get("question")), ("response", row.get("answer"))],
        )
    if identifier == "allenai-ai2-arc-v1":
        return record(
            "multiple-choice-science",
            [
                ("question", row.get("question")),
                ("options", row.get("choices")),
                ("target", row.get("answerKey")),
            ],
        )
    if identifier == "google-mbpp-full-v1":
        return record(
            "code-instruction",
            [
                ("instruction", row.get("text")),
                ("response-language", "python"),
                ("response", row.get("code")),
            ],
        )
    if identifier == "salesforce-xlam-function-calling-60k-v1":
        return record(
            "structured-function-calling",
            [
                ("instruction", row.get("query") or row.get("instruction")),
                ("tools", row.get("tools") or row.get("functions")),
                ("response", row.get("answers") or row.get("response")),
            ],
        )
    raise ValueError(f"no adapter for {identifier}")


def write_shards(target: Path, records: Iterable[str]) -> tuple[int, int, list[dict[str, Any]]]:
    target.mkdir(parents=True, exist_ok=False)
    shard_index = 0
    stream = None
    shard_size = 0
    count = 0
    total_bytes = 0
    manifests: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    try:
        for text in records:
            encoded = text.encode("utf-8", errors="strict")
            if not encoded.strip():
                continue
            if stream is None or (shard_size and shard_size + len(encoded) > SHARD_BYTES):
                if stream is not None:
                    stream.flush()
                    os.fsync(stream.fileno())
                    stream.close()
                path = target / f"train-{shard_index:05d}.txt"
                stream = path.open("wb")
                manifests.append({"file": path.name, "bytes": 0, "sha256": ""})
                shard_index += 1
                shard_size = 0
                digest = hashlib.sha256()
            stream.write(encoded)
            shard_size += len(encoded)
            total_bytes += len(encoded)
            count += 1
            digest.update(encoded)
            manifests[-1]["bytes"] = shard_size
            manifests[-1]["sha256"] = digest.hexdigest()
    finally:
        if stream is not None:
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
    if count == 0:
        raise RuntimeError("adapter produced no training records")
    return count, total_bytes, manifests


def eligible(registry: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [
        dataset
        for dataset in registry.get("datasets", [])
        if dataset.get("status") in ELIGIBLE_STATUSES and dataset.get("id") in PRIORITY
    ]
    order = {identifier: index for index, identifier in enumerate(PRIORITY)}
    return sorted(candidates, key=lambda dataset: order[dataset["id"]])


def prepare(registry_path: Path, dataset_root: Path, dataset: dict[str, Any]) -> None:
    identifier = dataset["id"]
    final = dataset_root / identifier
    temporary = dataset_root / f".{identifier}.preparing"
    if temporary.exists():
        shutil.rmtree(temporary)
    if final.exists():
        raise FileExistsError(f"prepared target already exists: {final}")
    print(f"preparing {identifier}", flush=True)
    try:
        rows = dataset_rows(dataset)
        serialized = (text for row in rows if (text := serialize(identifier, row)))
        count, total_bytes, shards = write_shards(temporary, serialized)
        if dataset.get("kind") in TASK_AWARE_KINDS:
            heldout_records = (count + TASK_HOLDOUT_DIVISOR - 1) // TASK_HOLDOUT_DIVISOR
            training_windows = count - heldout_records
        else:
            training_windows = sum(
                max(0, (int(shard["bytes"]) - 64 + 63) // 64) for shard in shards
            )
        attribution = {
            "dataset_id": identifier,
            "source": dataset["source"],
            "revision": dataset.get("revision"),
            "config": dataset.get("config"),
            "license": dataset.get("license"),
            "prepared_records": count,
            "prepared_bytes": total_bytes,
            "prepared_training_windows": training_windows,
            "prepared_unix_millis": int(time.time() * 1000),
            "shards": shards,
        }
        atomic_json(temporary / "attribution.json", attribution)
        temporary.replace(final)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    current = next(item for item in registry["datasets"] if item.get("id") == identifier)
    current["status"] = "active"
    current["prepared_path"] = str(final)
    current["prepared_records"] = count
    current["prepared_bytes"] = total_bytes
    current["prepared_training_windows"] = training_windows
    current["introduced_stage"] = f"river-v5-auto-{int(time.time())}"
    atomic_json(registry_path, registry)
    print(
        f"activated {identifier}: records={count} windows={training_windows} bytes={total_bytes}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--all", action="store_true", help="prepare every currently eligible source")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    candidates = eligible(registry)
    if not candidates:
        print("no eligible real training dataset remains", flush=True)
        return
    available = shutil.disk_usage(args.dataset_root.parent).free
    minimum = int(registry["policy"]["storage_budget"]["minimum_free_bytes"])
    if available <= minimum:
        raise SystemExit(f"storage safety stop: free={available} minimum={minimum}")
    for dataset in candidates if args.all else candidates[:1]:
        try:
            prepare(args.registry, args.dataset_root, dataset)
        except Exception as error:
            if dataset.get("status") == "queued-gated-access":
                print(f"gated dataset remains queued: {dataset['id']}: {error}", flush=True)
                continue
            raise


if __name__ == "__main__":
    main()
