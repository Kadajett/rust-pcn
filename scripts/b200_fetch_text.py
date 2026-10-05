#!/usr/bin/env python3
"""Fetch pinned B200 prose sources with hf; publish resumable UTF-8 text shards.

Run from ~/hf-cache/b200-fetch with huggingface_hub and pyarrow installed.
Results and commit journals live outside the byte loader's recursive source tree.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

TARGETS = {
    "tinystories-v1": ("roneneldan/TinyStories", "default", "CDLA-Sharing-1.0", None),
    "fineweb-edu-sample-10bt-v1": (
        "HuggingFaceFW/fineweb-edu", "sample-10BT", "ODC-By", 20_000_000_000
    ),
    "wikipedia-20231101-en-v1": (
        "wikimedia/wikipedia", "20231101.en", "CC-BY-SA", 10_000_000_000
    ),
}
SHARD_BYTES = 128 * 1024 * 1024
MAX_SHARD_BYTES = 256 * 1024 * 1024
MIN_SHARD_BYTES = 64 * 1024 * 1024
TEXT_DELIMITER = "<|endoftext|>"


def progress(identifier: str, message: str) -> None:
    print(f"[{identifier}] {message}", flush=True)


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    sync_directory(path.parent)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_files(identifier: str, siblings: list[Any]) -> list[Any]:
    """Select only the requested subset, never a whole repository download."""
    names = {entry.rfilename: entry for entry in siblings}
    if identifier == "tinystories-v1":
        selected = [name for name in names if name.startswith("data/") and name.endswith(".parquet")]
        if not selected:
            # These are the original dataset's official train/validation splits,
            # not the alternate GPT4 release or the overlapping archive superset.
            selected = ["TinyStories-train.txt", "TinyStories-valid.txt"]
            if not all(name in names for name in selected):
                raise ValueError("TinyStories has neither canonical parquet nor both official text splits")
    else:
        prefix = "sample/10BT/" if identifier == "fineweb-edu-sample-10bt-v1" else "20231101.en/"
        selected = [name for name in names if name.startswith(prefix) and name.endswith(".parquet")]
    if not selected:
        raise ValueError(f"no matching Hub source files for {identifier}")
    return [names[name] for name in sorted(selected)]


def source_metadata(entry: Any, repo_id: str, revision: str) -> dict[str, Any]:
    lfs = entry.lfs
    return {
        "file": entry.rfilename,
        "url": f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{entry.rfilename}",
        "hub_bytes": entry.size,
        "git_blob_id": entry.blob_id,
        "lfs_sha256": getattr(lfs, "sha256", None) if lfs else None,
    }


def initial_result(identifier: str, output: Path) -> dict[str, Any]:
    from huggingface_hub import HfApi

    repo_id, config, license_name, cap = TARGETS[identifier]
    progress(identifier, f"resolving Hub revision for {repo_id} ({config})")
    info = HfApi().dataset_info(repo_id, files_metadata=True)
    if not info.sha or not re.fullmatch(r"[0-9a-f]{40}", info.sha):
        raise ValueError(f"Hub did not provide a commit SHA for {repo_id}")
    card = info.card_data.to_dict() if info.card_data else {}
    files = select_files(identifier, info.siblings or [])
    return {
        "schema": "b200-raw-text-v1",
        "id": identifier,
        "status": "partial",
        "source": str(output),
        "repo_id": repo_id,
        "repo_url": f"https://huggingface.co/datasets/{repo_id}",
        "config": config,
        "revision": info.sha,
        "license": license_name,
        "hub_license": card.get("license"),
        "attribution_url": f"https://huggingface.co/datasets/{repo_id}/blob/{info.sha}/README.md",
        "bytes": 0,
        "documents": 0,
        "files": [],
        "text_cap_bytes": cap,
        "shard_target_bytes": SHARD_BYTES,
        "source_files": [source_metadata(entry, repo_id, info.sha) for entry in files],
        "cursor": {"file_index": 0, "row": 0},
        "serialization": "text only; surrounding whitespace stripped; UTF-8; blank line after each document",
        "source_order": "lexicographic Hub filename, then original row order; zero-based inclusive source ranges",
    }


def download_source(result: dict[str, Any], source: dict[str, Any], cache: Path, repo_option: str) -> Path:
    from huggingface_hub import try_to_load_from_cache

    progress(result["id"], f"hf download {source['file']} at {result['revision']}")
    subprocess.run(
        ["hf", "download", result["repo_id"], source["file"], repo_option, "dataset",
         "--revision", result["revision"], "--cache-dir", str(cache)],
        check=True,
    )
    cached = try_to_load_from_cache(
        result["repo_id"], source["file"], cache_dir=str(cache),
        revision=result["revision"], repo_type="dataset",
    )
    if not isinstance(cached, str) or not Path(cached).is_file():
        raise RuntimeError(f"hf download did not populate the pinned cache entry: {source['file']}")
    path = Path(cached)
    digest = sha256_file(path)
    if source.get("lfs_sha256") and digest != source["lfs_sha256"]:
        raise ValueError(f"source LFS checksum mismatch: {source['file']}")
    source.update(cache_path=str(path), downloaded_bytes=path.stat().st_size, sha256=digest)
    return path


def text_documents(path: Path) -> Iterator[str]:
    """Parse official TinyStories text without reading a whole file or long line."""
    pending = ""
    with path.open("r", encoding="utf-8", newline=None) as stream:
        while chunk := stream.read(1024 * 1024):
            pending += chunk
            while True:
                position = pending.find(TEXT_DELIMITER)
                if position < 0:
                    break
                yield pending[:position]
                pending = pending[position + len(TEXT_DELIMITER):]
            if len(pending) > MAX_SHARD_BYTES:
                raise ValueError(f"document exceeds 256 Mi characters in {path.name}")
    if pending.strip():
        yield pending


def documents(path: Path, skip: int) -> Iterator[tuple[int, str | None]]:
    if path.suffix != ".parquet":
        for row, text in enumerate(text_documents(path)):
            if row >= skip:
                yield row, text
        return
    import pyarrow.parquet as pq

    with pq.ParquetFile(path) as parquet:
        if "text" not in parquet.schema_arrow.names:
            raise ValueError(f"missing text column: {path.name}")
        row = 0
        # Skip completed row groups without decoding them when resuming.
        groups = []
        for group in range(parquet.num_row_groups):
            count = parquet.metadata.row_group(group).num_rows
            if row + count <= skip:
                row += count
            else:
                groups = list(range(group, parquet.num_row_groups))
                break
        if not groups:
            return
        for batch in parquet.iter_batches(batch_size=32, row_groups=groups, columns=["text"], use_threads=False):
            for value in batch.column(0):
                if row >= skip:
                    yield row, value.as_py()
                row += 1


class ShardWriter:
    def __init__(self, result: dict[str, Any], result_path: Path, staging: Path):
        self.result = result
        self.result_path = result_path
        self.staging = staging
        self.journal = result_path.with_suffix(".pending.json")
        self.stream: Any = None
        self.size = 0
        self.count = 0
        self.digest = hashlib.sha256()
        self.ranges: list[dict[str, Any]] = []
        self.cursor = dict(result["cursor"])

    def add(self, text: str, file_index: int, row: int) -> None:
        if len(text) > MAX_SHARD_BYTES:
            raise ValueError("single source document exceeds the 256 MiB shard limit")
        encoded = text.strip().encode("utf-8")
        self.cursor = {"file_index": file_index, "row": row + 1}
        if not encoded:
            return
        if len(encoded) + 2 > MAX_SHARD_BYTES:
            raise ValueError("single UTF-8 document exceeds the 256 MiB shard limit")
        # Keep normal shards close to 128 MiB without splitting documents.
        if self.size and (self.size + len(encoded) + 2 > MAX_SHARD_BYTES or
                          self.size >= MIN_SHARD_BYTES and self.size + len(encoded) + 2 > SHARD_BYTES):
            # This row belongs to the next shard, not the shard being committed.
            self.cursor = {"file_index": file_index, "row": row}
            self.commit()
            self.cursor = {"file_index": file_index, "row": row + 1}
        if self.stream is None:
            self.stream = self.staging.open("wb")
        self.stream.write(encoded)
        self.stream.write(b"\n\n")
        self.digest.update(encoded)
        self.digest.update(b"\n\n")
        self.size += len(encoded) + 2
        self.count += 1
        filename = self.result["source_files"][file_index]["file"]
        if not self.ranges or self.ranges[-1]["file"] != filename:
            self.ranges.append({"file": filename, "first_row": row, "last_row": row})
        else:
            self.ranges[-1]["last_row"] = row
        if self.size >= SHARD_BYTES:
            self.commit()

    def commit(self) -> None:
        if not self.size:
            return
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        self.stream = None
        output = Path(self.result["source"]) / f"part-{len(self.result['files']):06d}.txt"
        if output.exists():
            raise ValueError(f"refusing to overwrite an untracked output shard: {output}")
        updated = copy.deepcopy(self.result)
        updated["bytes"] += self.size
        updated["documents"] += self.count
        updated["cursor"] = dict(self.cursor)
        updated["files"].append({
            "path": str(output), "bytes": self.size, "documents": self.count,
            "sha256": self.digest.hexdigest(), "source_ranges": self.ranges,
        })
        # The journal closes the crash window between publishing a shard and
        # checkpointing its cursor; the public result lists only finished files.
        atomic_json(self.journal, updated)
        self.staging.replace(output)
        sync_directory(output.parent)
        atomic_json(self.result_path, updated)
        self.journal.unlink()
        sync_directory(self.journal.parent)
        self.result.clear()
        self.result.update(updated)
        progress(updated["id"], f"published {output.name}: {self.size} bytes; total {updated['bytes']} bytes / {updated['documents']} documents")
        self.size = self.count = 0
        self.digest = hashlib.sha256()
        self.ranges = []

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()


def recover_pending(result_path: Path, staging: Path) -> None:
    journal = result_path.with_suffix(".pending.json")
    if not journal.exists():
        return
    pending = read_json(journal)
    shard = pending["files"][-1]
    output = Path(shard["path"])
    available = output if output.exists() else staging
    if not available.is_file() or available.stat().st_size != shard["bytes"] or sha256_file(available) != shard["sha256"]:
        raise ValueError(f"incomplete or corrupted shard transaction: {journal}")
    if available == staging:
        staging.replace(output)
        sync_directory(output.parent)
    atomic_json(result_path, pending)
    journal.unlink()
    sync_directory(journal.parent)


def verify_result(result: dict[str, Any], identifier: str, output: Path) -> None:
    repo_id, config, license_name, cap = TARGETS[identifier]
    expected = {"schema": "b200-raw-text-v1", "id": identifier, "source": str(output),
                "repo_id": repo_id, "config": config, "license": license_name,
                "text_cap_bytes": cap, "shard_target_bytes": SHARD_BYTES}
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("existing result does not match this acquisition; refusing to mix corpora")
    tracked = set()
    for shard in result["files"]:
        path = Path(shard["path"])
        if path.parent != output or not re.fullmatch(r"part-\d{6}\.txt", path.name):
            raise ValueError(f"invalid output shard path: {path}")
        if not path.is_file() or path.stat().st_size != shard["bytes"] or sha256_file(path) != shard["sha256"]:
            raise ValueError(f"missing or corrupted completed shard: {path}")
        tracked.add(path.name)
    if {path.name for path in output.iterdir()} != tracked:
        raise ValueError(f"untracked content in raw source directory: {output}")
    if sum(shard["bytes"] for shard in result["files"]) != result["bytes"] or sum(shard["documents"] for shard in result["files"]) != result["documents"]:
        raise ValueError("completed shard totals do not match result metadata")


def acquire(identifier: str, results_dir: Path, output_root: Path, cache: Path) -> None:
    output = output_root / identifier
    output.mkdir(parents=True, exist_ok=True)
    # Staging is a sibling, never a child of the trainer's raw source directory.
    staging_dir = output_root / ".b200-staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    staging = staging_dir / f"{identifier}.part"
    result_path = results_dir / f"{identifier}.json"
    with (results_dir / f"{identifier}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        recover_pending(result_path, staging)
        if result_path.exists():
            result = read_json(result_path)
            progress(identifier, "verifying completed shard hashes before resume")
            verify_result(result, identifier, output)
        else:
            if any(output.iterdir()):
                raise ValueError(f"refusing nonempty output directory without a result: {output}")
            result = initial_result(identifier, output)
            atomic_json(result_path, result)
        if result["status"] == "complete":
            progress(identifier, f"already complete: {result['bytes']} bytes / {result['documents']} documents")
            return
        result["status"] = "partial"
        result.pop("last_error", None)
        atomic_json(result_path, result)
        help_text = subprocess.run(["hf", "download", "--help"], check=True, capture_output=True, text=True).stdout
        repo_option = "--repo-type" if "--repo-type" in help_text else "--type"
        writer = ShardWriter(result, result_path, staging)
        cap = result["text_cap_bytes"]
        try:
            cap_reached = cap is not None and result["bytes"] >= cap
            first_file = result["cursor"]["file_index"]
            for index in range(first_file, len(result["source_files"])):
                if cap_reached:
                    break
                source = result["source_files"][index]
                path = download_source(result, source, cache, repo_option)
                atomic_json(result_path, result)
                skip = result["cursor"]["row"] if index == first_file else 0
                for row, text in documents(path, skip):
                    if text is not None and not isinstance(text, str):
                        raise ValueError(f"non-string text in {source['file']} row {row}")
                    writer.add(text or "", index, row)
                    if cap is not None and result["bytes"] + writer.size >= cap:
                        cap_reached = True
                        break
                if cap_reached:
                    break
                writer.cursor = {"file_index": index + 1, "row": 0}
            writer.commit()
            if not result["documents"]:
                raise ValueError("source selection produced no nonempty documents")
            result["cursor"] = dict(writer.cursor)
            result["status"] = "complete"
            result["completion_reason"] = "text_cap_reached" if cap_reached else "source_exhausted"
            atomic_json(result_path, result)
            progress(identifier, f"complete ({result['completion_reason']}): {result['bytes']} bytes / {result['documents']} documents")
        except (Exception, KeyboardInterrupt) as error:
            # Only committed shards/cursors are advertised on errors. A pending
            # transaction is recovered on restart, never overwritten here.
            if not writer.journal.exists():
                result["status"] = "partial" if result["files"] else "error"
                result["last_error"] = safe_error(error)
                atomic_json(result_path, result)
            raise
        finally:
            writer.close()


def safe_error(error: BaseException) -> str:
    message = f"{type(error).__name__}: {error}"
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.environ.get(key)
        if token:
            message = message.replace(token, "[redacted]")
    return re.sub(r"hf_[A-Za-z0-9]{12,}", "[redacted]", message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", choices=TARGETS, required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("~/hf-cache/b200-fetch/results"))
    parser.add_argument("--output-root", type=Path, default=Path("/bulk-storage/datasets/hf-text"))
    args = parser.parse_args()
    home = Path(os.environ.setdefault("HF_HOME", str(Path("~/hf-cache").expanduser()))).expanduser().resolve()
    os.environ["HF_HOME"] = str(home)
    os.environ.setdefault("TMPDIR", str(home / "tmp"))
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    cache = home / "hub"
    cache.mkdir(parents=True, exist_ok=True)
    results_dir = args.results_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if results_dir == output_root / args.id or output_root / args.id in results_dir.parents:
        parser.error("results-dir must be outside the raw source directory")
    results_dir.mkdir(parents=True, exist_ok=True)
    try:
        acquire(args.id, results_dir, output_root, cache)
    except (Exception, KeyboardInterrupt) as error:
        progress(args.id, f"ERROR: {safe_error(error)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
