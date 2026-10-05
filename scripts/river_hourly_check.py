#!/usr/bin/env python3
"""Hourly River Song progress check.

1. Training must be on the GPU: the main service or a GPU trial unit. If neither runs and no trial/promotion
   controller is running, resume the main service from its latest saved checkpoint, unless the trainer's
   `<root>/health.json` says the run is blocked (then a human must repair it first).
2. Record progress since the previous check (batch, checkpoint, rate, energy, newest probe answers).
3. Append one line to the progress log and print a summary (all times US Pacific).
"""

from __future__ import annotations

import datetime
import json
import re
import subprocess
from pathlib import Path
from zoneinfo import ZoneInfo

TELEMETRY = Path("/bulk-storage/connectome-merc/river-multimodal-runs/current")
LOG = Path("/bulk-storage/connectome-merc/river-gpu-trials/hourly-progress.jsonl")
MAIN = "river-song-training.service"
CONTROLLERS = ("river_gpu_trial.py", "river_promote_trial.py", "river_stop_for_gpu_trials.py", "river_t5_t6_chain.sh")


def active(unit: str) -> bool:
    return subprocess.run(["systemctl", "--user", "is-active", "--quiet", unit]).returncode == 0


def trial_unit() -> str | None:
    words = subprocess.run(["systemctl", "--user", "list-units", "river-trial-*", "--state=active", "--no-legend", "--plain"],
                           capture_output=True, text=True).stdout.split()
    units = [w for w in words if w.startswith("river-trial-") and w.endswith(".service")]
    return units[0] if units else None


def controller_running() -> bool:
    return any(subprocess.run(["pgrep", "-f", name], capture_output=True).returncode == 0 for name in CONTROLLERS)


def health_block_reason() -> str | None:
    """Reason string from `<checkpoint root>/health.json.blocked`, or None when the main run is not blocked."""
    execstart = subprocess.run(["systemctl", "--user", "show", MAIN, "-p", "ExecStart"],
                               capture_output=True, text=True).stdout
    match = re.search(r"--output\s+(\S+)", execstart)
    if not match:
        return None
    try:
        health = json.loads((Path(match.group(1)) / "health.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    blocked = health.get("blocked") if isinstance(health, dict) else None
    if not blocked:
        return None
    return blocked.get("reason", "unspecified") if isinstance(blocked, dict) else str(blocked)


def main() -> None:
    now = datetime.datetime.now(ZoneInfo("America/Los_Angeles"))
    action = None
    running = MAIN if active(MAIN) else trial_unit()
    if running is None and not controller_running():
        reason = health_block_reason()
        if reason is not None:
            action = f"blocked; not restarted: {reason}"
        else:
            subprocess.run(["systemctl", "--user", "reset-failed", MAIN])
            subprocess.run(["systemctl", "--user", "start", MAIN])
            running, action = MAIN, "GPU was idle with nothing scheduled; resumed the main run from its latest checkpoint"
    try:
        state = json.loads((TELEMETRY / "state.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        state = {}
    previous = None
    if LOG.exists():
        lines = LOG.read_text().splitlines()
        previous = json.loads(lines[-1]) if lines else None
    record = {
        "at": now.isoformat(), "running": running, "action": action, "controller_running": controller_running(),
        "run": state.get("run"), "status": state.get("status"), "batch": state.get("batch"),
        "checkpoint_batch": state.get("checkpoint_batch"), "samples_per_second": state.get("samples_per_second"),
        "free_energy": state.get("free_energy"),
    }
    if previous and isinstance(previous.get("batch"), int) and isinstance(record["batch"], int):
        record["batches_since_last_check"] = record["batch"] - previous["batch"]
    with LOG.open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
