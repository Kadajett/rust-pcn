#!/usr/bin/env python3
"""Show whichever River Song run is on the GPU in the dashboard.

The portal reads one telemetry directory (the main run's). When the main service is stopped and a GPU trial
unit (`river-trial-<name>`) is running instead, mirror that trial's telemetry into the portal directory:
state/manifest/promotion are replaced atomically, event/sample/audit logs are appended, and public probe
requests (`request.json`) are forwarded to the trial trainer with its matching `response.json` copied back.
When the main service runs it writes the portal directory itself, so this does nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

SNAPSHOTS = ("state.json", "manifest.json", "promotion.json")
LOGS = ("events.jsonl", "samples.jsonl", "audit-metrics.jsonl")
FORWARD_TIMEOUT_SECONDS = 300


def active(unit: str) -> bool:
    return subprocess.run(["systemctl", "--user", "is-active", "--quiet", unit]).returncode == 0


def running_trial() -> str | None:
    listing = subprocess.run(
        ["systemctl", "--user", "list-units", "river-trial-*", "--state=active", "--no-legend", "--plain"],
        capture_output=True, text=True,
    ).stdout.split()
    units = [word for word in listing if word.startswith("river-trial-") and word.endswith(".service")]
    return units[0][len("river-trial-"):-len(".service")] if units else None


def replace_atomic(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.mirror")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def mirror_snapshots(source: Path, portal: Path, mtimes: dict[str, float]) -> None:
    for name in SNAPSHOTS:
        path = source / name
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            continue
        if mtimes.get(name) != mtime:
            replace_atomic(path, portal / name)
            mtimes[name] = mtime


def mirror_logs(source: Path, portal: Path, offsets: dict[str, int]) -> None:
    for name in LOGS:
        try:
            with (source / name).open("rb") as stream:
                stream.seek(offsets.get(name, 0))
                chunk = stream.read()
        except FileNotFoundError:
            continue
        complete = chunk[: chunk.rfind(b"\n") + 1]  # never copy a half-written line
        if complete:
            with (portal / name).open("ab") as out:
                out.write(complete)
            offsets[name] = offsets.get(name, 0) + len(complete)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portal-telemetry", type=Path,
                        default=Path("/bulk-storage/connectome-merc/river-multimodal-runs/current"))
    parser.add_argument("--trials", type=Path, default=Path("/bulk-storage/connectome-merc/river-gpu-trials"))
    parser.add_argument("--main-unit", default="river-song-training.service")
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()
    portal = args.portal_telemetry
    source_name: str | None = None
    offsets: dict[str, int] = {}
    mtimes: dict[str, float] = {}
    forwarded: str | None = None
    forwarded_at = 0.0
    while True:
        trial = None if active(args.main_unit) else running_trial()
        if trial != source_name:
            source_name, offsets, mtimes, forwarded = trial, {}, {}, None
            print(f"mirroring {trial or 'nothing (main run writes the dashboard itself)'}", flush=True)
        if trial is not None:
            source = args.trials / trial / "telemetry"
            mirror_snapshots(source, portal, mtimes)
            mirror_logs(source, portal, offsets)
            # Forward one public probe at a time; the trial's own probes share the slot, so wait for it.
            request = portal / "request.json"
            if forwarded is None and request.exists() and not (source / "request.json").exists() \
                    and not (source / "request.processing.json").exists():
                forwarded = request.read_text()
                forwarded_at = time.monotonic()
                (source / "response.json").unlink(missing_ok=True)
                temporary = source / "request.json.tmp"
                temporary.write_text(forwarded)
                os.replace(temporary, source / "request.json")
                os.replace(request, portal / "request.processing.json")
            if forwarded is not None:
                try:
                    wanted = json.loads(forwarded).get("id")
                    response = json.loads((source / "response.json").read_text())
                except (FileNotFoundError, json.JSONDecodeError, AttributeError):
                    response = None
                if isinstance(response, dict) and response.get("id") == wanted:
                    replace_atomic(source / "response.json", portal / "response.json")
                    (portal / "request.processing.json").unlink(missing_ok=True)
                    forwarded = None
                elif time.monotonic() - forwarded_at > FORWARD_TIMEOUT_SECONDS:
                    # The portal gave up long ago; free the public probe slot.
                    (portal / "request.processing.json").unlink(missing_ok=True)
                    forwarded = None
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
